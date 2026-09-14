//
// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "config.h"

#include <optix.h>

#include "launch_parameters.h"
#include "per_ray_data.h"
#include "function_indices.h"
#include "random.h"
#include "shader_common.h"
#include "vector_math.h"
#include "cache/nrc_features.h"
#include "surface_reference.h"

extern "C" {
  __constant__ LaunchParameters theLaunchParameters;
}

#if 0 // FIXME Volume scattering is not implemented. No glTF extension for that.
__forceinline__ __device__ float3 safe_div(const float3& a, const float3& b)
{
  const float x = (b.x != 0.0f) ? a.x / b.x : 0.0f;
  const float y = (b.y != 0.0f) ? a.y / b.y : 0.0f;
  const float z = (b.z != 0.0f) ? a.z / b.z : 0.0f;

  return make_float3(x, y, z);
}


__forceinline__ __device__ float sampleDensity(const float3& albedo,
                                               const float3& throughput,
                                               const float3& sigma_t,
                                               const float   u,
                                               float3&       pdf)
{
  const float3 weights = throughput * albedo;

  const float sum = weights.x + weights.y + weights.z;
  
  pdf = (0.0f < sum) ? weights / sum : make_float3(1.0f / 3.0f);

  if (u < pdf.x)
  {
    return sigma_t.x;
  }
  if (u < pdf.x + pdf.y)
  {
    return sigma_t.y;
  }
  return sigma_t.z;
}


// Determine Henyey-Greenstein phase function cos(theta) of scattering direction
__forceinline__ __device__ float sampleHenyeyGreensteinCos(const float xi, const float g)
{
  // PBRT v3: Chapter 15.2.3
  if (fabsf(g) < 1e-3f) // Isotropic.
  {
    return 1.0f - 2.0f * xi;
  }

  const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi);
  return (1.0f + g * g - s * s) / (2.0f * g);
}


// Determine scatter reflection direction with Henyey-Greenstein phase function.
__forceinline__ __device__ void sampleVolumeScattering(const float2 xi, const float g, float3& dir)
{
  const float cost = sampleHenyeyGreensteinCos(xi.x, g);
  
  float sint = 1.0f - cost * cost;
  sint = (0.0f < sint) ? sqrtf(sint) : 0.0f;
 
  const float phi = 2.0f * M_PIf * xi.y;

  // This vector is oriented in its own local coordinate system:
  const float3 d = make_float3(cosf(phi) * sint, sinf(phi) * sint, cost); 

  // Align the vector with the incoming direction.
  const TBN tbn(dir); // Just some ortho-normal basis along dir as z-axis.
  
  dir = tbn.transformToWorld(d);
}
#endif // FIXME Not implementing volume scattering.


//
// Formula: p_e_sa   = |W|² / (4 * |U| * |V| * cos³(θ_camera))
//
__device__ __forceinline__ float bdrcPerspectivePeSa(
    const float3 surfacePos,
    const float3 surfaceN,
    const float3 cameraP,
    const float3 cameraU,
    const float3 cameraV,
    const float3 cameraW,
    float        fallbackPeArea,
    int          cameraType)
{
    if (cameraType != 1)
        return fallbackPeArea;  // orthographic: use flat approximation

    // Vector from surface to camera
    const float3 r_to_cam = cameraP - surfacePos;
    const float  distSq   = dot(r_to_cam, r_to_cam);
    if (distSq < 1e-12f) return 0.f;
    const float  dist      = sqrtf(distSq);
    const float3 d_to_cam  = r_to_cam / dist;  // unit direction toward camera

    // Surface must face the camera
    const float cos_surface = dot(d_to_cam, surfaceN);
    if (cos_surface <= 0.f) return 0.f;

    // Camera forward (normalized) and cos(θ) at camera
    const float  f_sq    = dot(cameraW, cameraW);
    const float  f       = sqrtf(f_sq);
    const float3 cam_fwd = cameraW / f;
    const float  cos_cam = dot(-d_to_cam, cam_fwd);  // angle between surface dir and forward
    if (cos_cam <= 1e-4f) return 0.f;   // behind or near-perpendicular

    // NDC frustum check: recover (u, v) ∈ [-1, 1]² for the direction to surface
    // From ray generation: ray_dir = normalize(U*u + V*v + W)
    // Given d_to_surface = -d_to_cam, the NDC coords are:
    //   u = (f / cos_cam) * dot(d_to_surface, U) / |U|²
    //   v = (f / cos_cam) * dot(d_to_surface, V) / |V|²
    const float3 d_to_surface = -d_to_cam;
    const float  hx_sq        = dot(cameraU, cameraU);
    const float  hy_sq        = dot(cameraV, cameraV);
    const float  scale        = f / cos_cam;
    const float  u_ndc        = scale * dot(d_to_surface, cameraU) / hx_sq;
    const float  v_ndc        = scale * dot(d_to_surface, cameraV) / hy_sq;
    if (fabsf(u_ndc) > 1.f || fabsf(v_ndc) > 1.f) return 0.f;  // outside frustum

    // Solid-angle PDF for perspective camera:
    //   p_e_sa = f² / (4 * |U| * |V| * cos³(θ_camera))
    const float film_area = 4.f * sqrtf(hx_sq * hy_sq);
    const float p_e_sa    = f_sq / (film_area * cos_cam * cos_cam * cos_cam);

    return p_e_sa;
}

struct NrcSpreadState
{
  float3 x0;
  float  a0;
  float  spread;
  float3 posPrev;
  float  pdfPrev;
};

__forceinline__ __device__ void updateNrcSpreadState(const PerRayData& prd, const int depth, NrcSpreadState& state)
{
  if (depth == 0)
  {
    // Reference spread a0 = ||x0 - x1||^2 / (4pi |cos theta1|).
    const float3 d01    = prd.pos - state.x0;
    const float  distSq = dot(d01, d01);
    const float  cosT1  = fmaxf(dot(prd.nrcN, prd.wo), 1e-6f);
    state.a0 = distSq / (4.0f * M_PIf * cosT1);
  }
  else
  {
    const float3 d      = prd.pos - state.posPrev;
    const float  distSq = dot(d, d);
    const float  cosT   = fmaxf(fabsf(dot(prd.nrcN, prd.wo)), 1e-6f);
    state.spread += sqrtf(distSq / (fmaxf(state.pdfPrev, 1e-6f) * cosT));
  }

  // Track previous hit for the next segment (including specular events).
  state.posPrev = prd.pos;
  state.pdfPrev = prd.pdf;
}

__forceinline__ __device__ bool handleBdrcCacheLogic(
    PerRayData& prd,
    const uint2 theLaunchIndex,
    uint2 payload,
    const float3 radianceBeforeD)
{

  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  const float3 tpK1 = prd.throughputPreBsdf;

  // In eye-cache debug mode the render buffers were already captured at the first-hit
  // vertex (depth=0) by handleBdrcDebugFirstHit, so don't overwrite them here.
  if (theLaunchParameters.nrc.nrcDebugMode != 1)
  {
    nrc.renderThroughput[3 * pixelIdx + 0] = tpK1.x;
    nrc.renderThroughput[3 * pixelIdx + 1] = tpK1.y;
    nrc.renderThroughput[3 * pixelIdx + 2] = tpK1.z;

    nrc.renderPrefix[3 * pixelIdx + 0] = radianceBeforeD.x;
    nrc.renderPrefix[3 * pixelIdx + 1] = radianceBeforeD.y;
    nrc.renderPrefix[3 * pixelIdx + 2] = radianceBeforeD.z;

    nrc.renderValid[pixelIdx] = 1u;
    bdrcPackFeatures(nrc.renderFeatures, pixelIdx, prd.posLocal, prd.localAabbMin, prd.localAabbMax);
  }

  // ---- Training suffix: continue path from D with full NEE + Russian Roulette ----
  // Every diffuse-terminated pixel contributes a training slot (no fraction gate).
  // The suffix runs the same loop as the main integrator — NEE is handled inside
  // the hit shader — until natural absorption or Russian Roulette termination.
  // trainNextValid=0 so k_nrc_build_targets uses the pure MC formula:
  //   target = suffixDelta / trainThroughputAtK

    const unsigned int slot = atomicAdd(nrc.trainCount, 1u);
    if (slot >= static_cast<unsigned int>(theLaunchParameters.resolution.x *
                                          theLaunchParameters.resolution.y))
    {
      prd.nrcTerminated = true;
      return true;
    }

    nrc.trainPixelIndices[slot] = pixelIdx;

    bdrcPackFeatures(nrc.trainFeatures, slot, prd.posLocal, prd.localAabbMin, prd.localAabbMax);

    nrc.trainThroughputAtK[3 * slot + 0] = tpK1.x;
    nrc.trainThroughputAtK[3 * slot + 1] = tpK1.y;
    nrc.trainThroughputAtK[3 * slot + 2] = tpK1.z;
  
    const float3 radianceBefore = radianceBeforeD;

    for (int s = 0; s < theLaunchParameters.pathLengths.y; ++s)
    {
      prd.wo       = -prd.wi;
      prd.distance = RT_DEFAULT_MAX;
      prd.flags    = 0;

      optixTrace(theLaunchParameters.handle,
                 prd.pos, prd.wi,
                 theLaunchParameters.sceneEpsilon, prd.distance, 0.0f,
                 OptixVisibilityMask(0xFF),
                 OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                 TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                 payload.x, payload.y);

      if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
        break;

      if (theLaunchParameters.pathLengths.x <= s)
      {
        const float probability = fmaxf(prd.throughput);
        if (probability < rng(prd.seed))
          break;
        prd.throughput /= probability;
      }
    }

    const float3 suffDelta = prd.radiance - radianceBefore;
    nrc.trainSuffixScaled[3 * slot + 0] = suffDelta.x;
    nrc.trainSuffixScaled[3 * slot + 1] = suffDelta.y;
    nrc.trainSuffixScaled[3 * slot + 2] = suffDelta.z;
    nrc.trainNextValid[slot] = 0u;

  prd.nrcTerminated = true;
  return true;
}

__forceinline__ __device__ bool handleBdrcDebugFirstHit(
    PerRayData& prd,
    const uint2 theLaunchIndex,
    const float3 lastSpecPos)
{
  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  // Eye cache query: pack the first-hit position as the query feature.
  bdrcPackFeatures(nrc.renderFeatures, pixelIdx, prd.posLocal, prd.localAabbMin, prd.localAabbMax);
  nrc.renderValid[pixelIdx] = 1u;

  // S position: last specular before this hit, or cameraP if path started diffuse.
  nrc.eyeVertexPos[3*pixelIdx+0] = lastSpecPos.x;
  nrc.eyeVertexPos[3*pixelIdx+1] = lastSpecPos.y;
  nrc.eyeVertexPos[3*pixelIdx+2] = lastSpecPos.z;

  // Suppress light cache contribution in composite: debug shows eye cache only.
  nrc.eyeVertexThroughput[3*pixelIdx+0] = 0.0f;
  nrc.eyeVertexThroughput[3*pixelIdx+1] = 0.0f;
  nrc.eyeVertexThroughput[3*pixelIdx+2] = 0.0f;

  // No radiance accumulated before the first hit; throughput is identity.
  nrc.renderPrefix[3*pixelIdx+0] = 0.0f;
  nrc.renderPrefix[3*pixelIdx+1] = 0.0f;
  nrc.renderPrefix[3*pixelIdx+2] = 0.0f;
  nrc.renderThroughput[3*pixelIdx+0] = 1.0f;
  nrc.renderThroughput[3*pixelIdx+1] = 1.0f;
  nrc.renderThroughput[3*pixelIdx+2] = 1.0f;

  // R = first-hit position (placeholder; overwritten by the diffuse-termination probe when
  // the path continues, and p_e_area=0 makes it a zero-gradient slot if it is not overwritten).
  bdrcPackFeatures(nrc.resampledEndpoints, pixelIdx, prd.posLocal, prd.localAabbMin, prd.localAabbMax);
  nrc.resampledValid[pixelIdx] = 1u;
  nrc.resampled_p_e_area[pixelIdx] = 0.0f;

  // Modes 1, 5, 6: let the path continue so the diffuse-termination block can write
  // renderValid, resampledEndpoints (hemisphere-probed R), and eyeVertexPos correctly.
  // Mode 3 (mask): also continues so the diffuse-termination probe writes a valid p_e_area
  // and eyeVertexThroughput, which are required for non-zero mask training gradients.
  // Modes 2, 4 terminate here since they only need the first-hit cache outputs.
  const int dbg = theLaunchParameters.nrc.nrcDebugMode;
  if (dbg == 1 || dbg == 3 || dbg == 5 || dbg == 6)
    return false;

  prd.nrcTerminated = true;
  return true;
}

__forceinline__ __device__ bool handleNrcDebugCacheOnly(
    PerRayData& prd,
    const uint2 theLaunchIndex)
{
  // Preconditions are checked by the caller (integrator).
  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;
  nrcPackFeatures(nrc.renderFeatures, pixelIdx,
                  prd.pos, prd.nrcN, prd.wo,
                  prd.nrcRoughness, prd.nrcBaseColor, prd.nrcSpecColor,
                  nrc.aabbMin, nrc.aabbMax);
  nrc.renderValid[pixelIdx] = 1u;
  prd.nrcTerminated = true;
  return true;
}

__forceinline__ __device__ bool handleNrcCacheLogic(
    PerRayData& prd,
    const uint2 theLaunchIndex,
    uint2 payload,
    const float3 radianceBeforeTrace)
{

  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  // ---- Render data (all NRC-terminated pixels) ----
  // Use pre-BSDF throughput and the radiance snapshot taken before the hit-shader ran
  // so the cache learns full outgoing radiance at K: (L_direct_at_K + L_indirect) / T(cam→K)_pre,
  // matching the BDRC eye-colour cache formulation.
  const float3 tpK1 = prd.throughputPreBsdf;
  nrc.renderThroughput[3 * pixelIdx + 0] = tpK1.x;
  nrc.renderThroughput[3 * pixelIdx + 1] = tpK1.y;
  nrc.renderThroughput[3 * pixelIdx + 2] = tpK1.z;

  nrc.renderPrefix[3 * pixelIdx + 0] = radianceBeforeTrace.x;
  nrc.renderPrefix[3 * pixelIdx + 1] = radianceBeforeTrace.y;
  nrc.renderPrefix[3 * pixelIdx + 2] = radianceBeforeTrace.z;

  nrc.renderValid[pixelIdx] = 1u;

  // Reflectance factorization: clamp(α+β, 0.01) per channel.
  nrc.renderReflectance[3 * pixelIdx + 0] = fmaxf(prd.nrcBaseColor.x + prd.nrcSpecColor.x, 0.01f);
  nrc.renderReflectance[3 * pixelIdx + 1] = fmaxf(prd.nrcBaseColor.y + prd.nrcSpecColor.y, 0.01f);
  nrc.renderReflectance[3 * pixelIdx + 2] = fmaxf(prd.nrcBaseColor.z + prd.nrcSpecColor.z, 0.01f);

  nrcPackFeatures(nrc.renderFeatures, pixelIdx,
                  prd.pos, prd.nrcN, prd.wo,
                  prd.nrcRoughness, prd.nrcBaseColor, prd.nrcSpecColor,
                  nrc.aabbMin, nrc.aabbMax);

  // ---- Training suffix (~3% of NRC-terminated pixels) ----
  if (rng(prd.seed) < NRC_TRAIN_FRACTION)
  {
    const bool isUnbiased = (rng(prd.seed) * NRC_UNBIASED_RATE < 1.0f);

    const unsigned int slot = atomicAdd(nrc.trainCount, 1u);
    if (slot >= static_cast<unsigned int>(theLaunchParameters.resolution.x *
                                          theLaunchParameters.resolution.y))
    {
      prd.nrcTerminated = true;
      return true;
    }

    nrc.trainPixelIndices[slot] = pixelIdx;

    nrc.trainReflectance[3 * slot + 0] = fmaxf(prd.nrcBaseColor.x + prd.nrcSpecColor.x, 0.01f);
    nrc.trainReflectance[3 * slot + 1] = fmaxf(prd.nrcBaseColor.y + prd.nrcSpecColor.y, 0.01f);
    nrc.trainReflectance[3 * slot + 2] = fmaxf(prd.nrcBaseColor.z + prd.nrcSpecColor.z, 0.01f);

    nrcPackFeatures(nrc.trainFeatures, slot,
                    prd.pos, prd.nrcN, prd.wo,
                    prd.nrcRoughness, prd.nrcBaseColor, prd.nrcSpecColor,
                    nrc.aabbMin, nrc.aabbMax);

    nrc.trainThroughputAtK[3 * slot + 0] = tpK1.x;
    nrc.trainThroughputAtK[3 * slot + 1] = tpK1.y;
    nrc.trainThroughputAtK[3 * slot + 2] = tpK1.z;

    // Baseline is radianceBeforeTrace — the suffix delta then includes NEE at K
    // plus all subsequent bounces, consistent with dividing by T(cam→K)_pre.
    const float3 radianceBefore = radianceBeforeTrace;
    bool finalSuffixHit = false;

    for (int s = 0; s < NRC_SUFFIX_STEPS; ++s)
    {
      prd.wo       = -prd.wi;
      prd.distance = RT_DEFAULT_MAX;
      prd.flags    = 0;

      optixTrace(theLaunchParameters.handle,
                 prd.pos, prd.wi,
                 theLaunchParameters.sceneEpsilon, prd.distance, 0.0f,
                 OptixVisibilityMask(0xFF),
                 OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                 TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                 payload.x, payload.y);

      if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      {
        break;
      }

      finalSuffixHit = (prd.flags & FLAG_HIT) &&
                       (prd.typeEvent & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY));
    }

    const float3 suffixDelta = prd.radiance - radianceBefore;
    nrc.trainSuffixScaled[3 * slot + 0] = suffixDelta.x;
    nrc.trainSuffixScaled[3 * slot + 1] = suffixDelta.y;
    nrc.trainSuffixScaled[3 * slot + 2] = suffixDelta.z;

    if (finalSuffixHit && !isUnbiased)
    {
      nrcPackFeatures(nrc.trainNextFeatures, slot,
                      prd.pos, prd.nrcN, prd.wo,
                      prd.nrcRoughness, prd.nrcBaseColor, prd.nrcSpecColor,
                      nrc.aabbMin, nrc.aabbMax);
      const float3 tpFinal = prd.throughput;
      nrc.trainSuffixThroughput[3 * slot + 0] = tpFinal.x;
      nrc.trainSuffixThroughput[3 * slot + 1] = tpFinal.y;
      nrc.trainSuffixThroughput[3 * slot + 2] = tpFinal.z;
      nrc.trainNextValid[slot] = 1u;
    }
    else
    {
      nrc.trainSuffixThroughput[3 * slot + 0] = 0.0f;
      nrc.trainSuffixThroughput[3 * slot + 1] = 0.0f;
      nrc.trainSuffixThroughput[3 * slot + 2] = 0.0f;
      nrc.trainNextValid[slot] = 0u;
    }
  }

  prd.nrcTerminated = true;
  return true;
}

// Called by raygen.
// @return radiance
__forceinline__ __device__ float3 integrator(PerRayData& prd, const uint2 theLaunchIndex)
{
  // The integrator starts with black radiance and full path throughput.
  prd.radiance      = make_float3(0.0f);
  prd.pdf           = 0.0f;
  prd.throughput    = make_float3(1.0f);
  prd.occlusion     = 1.0f;
  prd.sigma_t       = make_float3(0.0f); // Extinction coefficient: sigma_a + sigma_s.
  //prd.walk          = 0;               // Number of random walk steps taken through volume scattering.
  prd.typeEvent     = BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
  prd.nrcTerminated = false;
  prd.eyePathFoundDiffuse    = false;
  prd.eyePathInSpecularChain = false;

  // NRC area-spread tracking (Mueller et al. 2021, Section 3.4).
  // All local to the integrator; no need to store in PRD.
  NrcSpreadState nrcSpreadState = {prd.pos, 0.0f, 0.0f, prd.pos, 0.0f};

  // Nested material handling.
  prd.idxStack   = 0;
  // Small stack of four entries of which the first is vacuum.
  prd.stack[0].absorption_ior  = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // No volume absorption, no effective IOR.
  //prd.stack[0].scattering_bias = make_float4(0.0f);  // No volume scattering, isotropic volume scattering.

  // Put payload pointer into two unsigned integers. Actually const, but that's not what optixTrace() expects.
  uint2 payload = splitPointer(&prd);

  // Russian Roulette path termination after a specified number of bounces needs the current depth.
  int depth = 0; // Path segment index. Primary ray is depth == 0.

  // BDRC: track the last specular vertex S before the first diffuse hit.
  // Defaults: S = camera (used when path starts diffuse — matches Python fallback).
  float3 lastSpecPos    = theLaunchParameters.cameraP;
  float3 lastSpecNormal = prd.wi;  // primary ray dir used as hemisphere axis for fallback
  bool   hasSpecular    = false;

  while (depth < theLaunchParameters.pathLengths.y)
  {
    // Self-intersection avoidance:
    // Offset the ray t_min value by sysData.sceneEpsilon when a geometric primitive was hit by the previous ray.
    // Primary rays and volume scattering miss events will not offset the ray t_min.
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;        // Direction to observer.
    prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
    prd.flags    = 0;

    const float3 radianceBeforeTrace = prd.radiance;

    {
      // FIXME Not implementing volume scattering.
      // Special cases for volume scattering!
      //if (0 < prd.idxStack) // Inside a volume?
      //{
      //  // Note that this only supports homogeneous volumes so far!
      //  // No change in sigma_s along the random walk here.
      //  const float3 sigma_s = make_float3(prd.stack[prd.idxStack].scattering_bias);

      //  if (isNotNull(sigma_s)) // We're inside a volume and it has volume scattering?
      //  {
      //    // Indicate that we're inside a random walk. This changes the behavior of the radiance ray miss programs.
      //    prd.flags |= FLAG_VOLUME_SCATTERING;

      //    // Random walk through scattering volume, sampling the distance.
      //    // Note that the entry and exit of the volume is done according to the BSDF sampling.
      //    // Means glass with volume scattering will still do the proper refractions.
      //    // When the number of random walk steps has been exceeded, the next ray is shot with distance RT_DEFAULT_MAX
      //    // to hit something. If that results in a transmission the scattering volume is left.
      //    // If not, this continues until the maximum path length has been exceeded.
      //    if (prd.walk < sysData.walkLength)
      //    {
      //      const float3 albedo = safe_div(sigma_s, prd.sigma_t);
      //      const float2 xi     = rng2(prd.seed);
      //
      //      const float s = sampleDensity(albedo, prd.throughput, prd.sigma_t, xi.x, prd.pdfVolume);

      //      // Prevent logf(0.0f) by sampling the inverse range (0.0f, 1.0f].
      //      prd.distance = -logf(1.0f - xi.y) / s;
      //    }
      //  }
      //}
    }

    // Note that the primary rays and volume scattering miss cases do not offset the ray t_min by sysData.sceneEpsilon.
    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi, // origin, direction
               epsilon, prd.distance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFF),
               // GLTF default is backface culling enabled, but the GAS OptixGeometryFlags disable
               // culling depending on the material doubleSided condition.
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    // NRC area-spread path termination (Mueller et al. 2021, Section 3.4).
    // Update the area spread for this segment, then check whether to cache-terminate the path.
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == NRC && (prd.flags & FLAG_HIT))
    {
      updateNrcSpreadState(prd, depth, nrcSpreadState);
    }

    // BDRC: record the last specular vertex before the first diffuse hit.
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == BDRC &&
        (prd.flags & FLAG_HIT) &&
        (prd.typeEvent & (BSDF_EVENT_SPECULAR | BSDF_EVENT_GLOSSY)))
    {
      lastSpecPos    = prd.pos;
      lastSpecNormal = prd.nrcN;
      hasSpecular    = true;
    }

    // BDRC debug first-hit: capture first-hit render features for all debug modes.
    // Modes >= 2 terminate the path here; mode 1 (eye cache) returns false and the path
    // continues so diffuse-termination training still runs — see handleBdrcDebugFirstHit.
    if (theLaunchParameters.nrc.nrcDebugMode != 0 && theLaunchParameters.nrc.nrcType == BDRC &&
        depth == 0 && (prd.flags & FLAG_HIT))
    {
      if (handleBdrcDebugFirstHit(prd, theLaunchIndex, lastSpecPos))
        break;
    }

    // BDRC: terminate at first diffuse vertex (specular-skip)
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == BDRC && (prd.flags & FLAG_HIT) && (prd.typeEvent & (BSDF_EVENT_DIFFUSE)))
    {
      const unsigned int px = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
      const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

      // S position (last specular before D, or camera default).
      nrc.eyeVertexPos[3*px+0] = lastSpecPos.x;
      nrc.eyeVertexPos[3*px+1] = lastSpecPos.y;
      nrc.eyeVertexPos[3*px+2] = lastSpecPos.z;

      // Pre-BSDF throughput at D: T(camera→D) not including BSDF at D.
      nrc.eyeVertexThroughput[3*px+0] = prd.throughputPreBsdf.x;
      nrc.eyeVertexThroughput[3*px+1] = prd.throughputPreBsdf.y;
      nrc.eyeVertexThroughput[3*px+2] = prd.throughputPreBsdf.z;

      // Resampled position R for mask label-0 (reference/anchor stream).
      // Exact area-uniform sample over ALL diffuse-capable scene surfaces —
      // camera-independent, no probe ray, no visibility test. This is the
      // pure-uniform case of the Python project's surface_mixture reference
      // distribution: density is the single known scalar invTotalArea
      // everywhere, so no per-point density lookup is needed.
      {
        float3 refPos, refNormal;
        sampleSurfaceUniform(prd.seed, theLaunchParameters.surfaceRef, refPos, refNormal);

        // Reuse posLocal/localAabbMin/localAabbMax packing: the reference
        // sample has no GAS-local frame of its own (it isn't a ray hit), so
        // pack it directly in world space against the same global AABB used
        // to normalize all other cache queries (nrc.aabbMin/aabbMax).
        bdrcPackFeatures(nrc.resampledEndpoints, px, refPos,
                         theLaunchParameters.nrc.aabbMin,
                         theLaunchParameters.nrc.aabbMax);
        nrc.resampledValid[px]     = 1u;
        nrc.resampled_p_sa[px]     = 0.0f;  // no longer a solid-angle PDF; unused downstream
        nrc.resampled_p_e_area[px] = theLaunchParameters.surfaceRef.invTotalArea;

        nrc.eyeVertexNormals[3*px+0] = refNormal.x;
        nrc.eyeVertexNormals[3*px+1] = refNormal.y;
        nrc.eyeVertexNormals[3*px+2] = refNormal.z;
      }

      if (handleBdrcCacheLogic(prd, theLaunchIndex, payload, radianceBeforeTrace))
      {
        break;
      }
    }

    // NRC debug-only mode: query at first hit
    if (theLaunchParameters.nrc.nrcDebugMode == 1 && theLaunchParameters.nrc.nrcType == NRC && depth == 0 && (prd.flags & FLAG_HIT))
    {
      if (handleNrcDebugCacheOnly(prd, theLaunchIndex))
      {
        break;
      }
    }

    // NRC termination: area-spread or fallback depth reached at diffuse/glossy vertex
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == NRC &&
        depth >= 0 && nrcSpreadState.a0 > 0.0f &&
        ((nrcSpreadState.spread * nrcSpreadState.spread > theLaunchParameters.nrc.spreadC * nrcSpreadState.a0) ||
         depth >= theLaunchParameters.nrc.terminationDepth) &&
        (prd.flags & FLAG_HIT) && (prd.typeEvent & (BSDF_EVENT_DIFFUSE | BSDF_EVENT_GLOSSY)))
    {
      if (handleNrcCacheLogic(prd, theLaunchIndex, payload, radianceBeforeTrace))
      {
        break; // NRC handles the rest; skip further path tracing.
      }
    }

    // Path termination by miss shader or sample() routines.
    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
    {
      if (theLaunchParameters.showEnvironment == 0 && // If the environment should not be shown in primary rays and
          depth == 0 &&                               // it's a primary ray and
          (prd.flags & FLAG_HIT) == 0)                // it did not hit anything then
      {
        prd.radiance = make_float3(0.0f);             // return black instead of the miss program result.
      }
      break; // Path termination.
    }

    // Unbiased Russian Roulette path termination.
    if (theLaunchParameters.pathLengths.x <= depth) // Start termination after a minimum number of bounces.
    {
      const float probability = fmaxf(prd.throughput);

      if (probability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
      {
        break; // Path termination.
      }

      prd.throughput /= probability; // Path isn't terminated. Adjust the throughput so that the average is right again.
    }

    // We're inside a volume and the scatter ray missed.
    //if (prd.flags & FLAG_VOLUME_SCATTERING_MISS) // This implies FLAG_VOLUME_SCATTERING.
    //{
    //  // Random walk through scattering volume, sampling the direction according to the phase function.
    //  sampleVolumeScattering(rng2(prd.seed), prd.stack[prd.idxStack].scattering_bias.w, prd.wi);
    //}

    ++depth; // Next path segment.
  }
  return prd.radiance;
}


// Accumulates radiance into launch parameters' buffer.
extern "C" __global__ void __raygen__path_tracer()
{
  //const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions());
  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;
  
  float2 fragment;

  if (theLaunchParameters.picking.x < 0.0f) // Normal rendering, not picking.
  {
    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    // PERF This template really generates a lot of instructions.
    prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x, theLaunchParameters.iteration); 

    // Calculate primary ray direction from perspective camera position into the scene, in world space.
    fragment = make_float2(theLaunchIndex) + rng2(prd.seed); // Jitter sub-pixel location.
  }
  else // Picking ray.
  {
    prd.indexMaterial = -1; // Default result when not hitting anything.

    fragment = theLaunchParameters.picking;
  }

  const float2 ndc = (fragment / make_float2(theLaunchParameters.resolution)) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

  if (theLaunchParameters.cameraType) // perspective camera
  {
    // All primary rays start at the camera position and go through the camera plane at the end of the W vector.
    prd.pos = theLaunchParameters.cameraP;
    prd.wi  = normalize(theLaunchParameters.cameraU * ndc.x +
                        theLaunchParameters.cameraV * ndc.y + 
                        theLaunchParameters.cameraW);
  }
  else // if orthographic camera
  {
    // Primary rays start on a camera plane at the camera position and all directions are parallel.
    prd.pos = theLaunchParameters.cameraP + 
              theLaunchParameters.cameraU * ndc.x +
              theLaunchParameters.cameraV * ndc.y;
    prd.wi  = theLaunchParameters.cameraW; // W is normalized for orthographic cameras.
  }

  float3 radiance = integrator(prd, theLaunchIndex);

#if USE_DEBUG_EXCEPTIONS // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
    //printf("NAN at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
    //printf("INF at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
    //printf("NEG at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
#endif

  if (theLaunchParameters.picking.x < 0.0f) // Rendering.
  {
    // NRC-terminated pixels are composited by the host-side accumulate kernel; skip here.
    // In NRC debug modes the debug kernel owns bufferAccum; non-terminated pixels must not
    // write path-traced color, or they corrupt the visualization.
    if (!prd.nrcTerminated && theLaunchParameters.nrc.nrcDebugMode == 0)
    {
      const unsigned int index = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;

      if (0 < theLaunchParameters.iteration)
      {
        const float t = 1.0f / static_cast<float>(theLaunchParameters.iteration + 1);

        const float3 colorAccum = make_float3(theLaunchParameters.bufferAccum[index]);

        radiance = lerp(colorAccum, radiance, t);
      }

      theLaunchParameters.bufferAccum[index] = make_float4(radiance, 1.0f);
    }
  }
  else // Picking.
  {
    theLaunchParameters.bufferPicking[0] = prd.indexMaterial;
  }
}




// BDPT: Samples an emitter and initializes the light path PRD.
// Handles point/spot lights (fixed position) and env sphere lights (bounding-sphere origin).
// Returns true if a valid emitter was selected.
__forceinline__ __device__ bool sampleEmitterRay(PerRayData& prd, float3& origin, float3& direction)
{
  const int numLights = theLaunchParameters.numLights;
  if (numLights <= 0)
  {
    return false;
  }

  // Pick a random light.
  const int indexLight = clamp(static_cast<int>(floorf(rng(prd.seed) * numLights)), 0, numLights - 1);
  const LightDefinition& light = theLaunchParameters.lightDefinitions[indexLight];

  float dirPdf;

  if (light.typeLight == TYPE_LIGHT_ENV_SPHERE || light.typeLight == TYPE_LIGHT_ENV_CONST)
  {
    // Environment light: sample a random point on the scene's bounding sphere and shoot inward.
    // Sample a random direction outward, then the ray origin is on the sphere surface in that direction,
    // and the ray direction points toward the sphere center (inward).
    float3 outward;
    unitSquareToSphere(rng(prd.seed), rng(prd.seed), outward, dirPdf);

    // Origin on the bounding sphere surface, direction pointing inward toward the scene.
    const float r = theLaunchParameters.sceneSphereRadius;
    origin    = theLaunchParameters.sceneSphereCenter + outward * r;
    direction = -outward; // Shoot toward scene center.

    // Throughput: emission * sphere_area * (numLights) (probability terms cancel with 1/4pi)
    // emission is the average env radiance; scale by 4*pi*r^2 to get total power, then /selectionPdf.
    prd.throughput = light.emission * float(numLights);
  }
  else if (light.typeLight == TYPE_LIGHT_MESH)
  {
    const float3* positions = reinterpret_cast<const float3*>(light.cdfU);
    const float*  cdf       = reinterpret_cast<const float*> (light.cdfV);
    const int     N         = static_cast<int>(light.width);

    if (N <= 0 || light.area <= 0.0f)
      return false;

    // Sample a triangle by area CDF.
    const unsigned int triIdx = binarySearchCDF(cdf, static_cast<unsigned int>(N), rng(prd.seed));

    const float3 v0 = positions[3 * triIdx + 0];
    const float3 v1 = positions[3 * triIdx + 1];
    const float3 v2 = positions[3 * triIdx + 2];

    // Uniform barycentric sample (Turk 1990).
    const float su = sqrtf(rng(prd.seed));
    const float b0 = 1.0f - su;
    const float b1 = su * (1.0f - rng(prd.seed));
    const float b2 = su - b1;
    origin = b0 * v0 + b1 * v1 + b2 * v2;

    // Face normal as hemisphere axis.
    const float3 normal = normalize(cross(v1 - v0, v2 - v0));

    // Cosine-weighted direction from front face; cosTheta cancels with pdf_dir = cosTheta/pi.
    unitSquareToCosineHemisphere(make_float2(rng(prd.seed), rng(prd.seed)), normal, direction, dirPdf);

    // Throughput: L_e / (p_pos * p_dir) / selectionPdf = L_e * area * pi * numLights
    prd.throughput = light.emission * (light.area * M_PIf * float(numLights));
  }
  else if (light.typeLight >= TYPE_LIGHT_POINT) // Point or spot light.
  {
    // Light position is stored as the translation column of the light matrix.
    origin = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);

    // Sample a uniform direction on the sphere.
    unitSquareToSphere(rng(prd.seed), rng(prd.seed), direction, dirPdf);

    // Initial throughput: emission * numLights / dirPdf
    // (1/selectionPdf = numLights, 1/dirPdf = 4*pi for uniform sphere).
    prd.throughput = light.emission * (float(numLights) / dirPdf);
  }
  else // TYPE_LIGHT_DIRECTIONAL: no finite origin, skip.
  {
    return false;
  }

  return true;
}


// BDPT: Traces a path starting from a light source.
// Connections to the camera are performed inside chRadiance() when FLAG_LIGHT_PATH is set.
__forceinline__ __device__ void lightPathIntegrator(PerRayData& prd, const uint2 theLaunchIndex)
{
  prd.radiance   = make_float3(0.0f);
  prd.pdf        = 0.0f;
  prd.occlusion  = 1.0f;
  prd.sigma_t    = make_float3(0.0f);
  prd.typeEvent  = BSDF_EVENT_ABSORB;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  prd.lightSpecularCount        = 0;
  prd.lightDiffuseAfterSpecular = 0;
  prd.flags = FLAG_LIGHT_PATH;

  float3 origin;
  float3 direction;

  if (!sampleEmitterRay(prd, origin, direction))
  {
    return;
  }

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  int depth = 0;

  while (depth < theLaunchParameters.lightPathMaxDepth)
  {
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = FLAG_LIGHT_PATH; // Preserve FLAG_LIGHT_PATH; clear FLAG_HIT for next segment.

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    // Path termination: absorb event (including after camera connection) or null throughput.
    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
    {
      break;
    }

    // Russian Roulette after minimum path length.
    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput);

      if (probability < rng(prd.seed))
      {
        break;
      }

      prd.throughput /= probability;
    }

    ++depth;
  }
}


// BDRC light path integrator.
// Traces a path from a light source through specular bounces, connecting to the eye vertex
// at every diffuse vertex encountered after at least one specular bounce (S*D+S*E paths).
// Writes each valid connection directly to the compact atomic slot buffers in NrcLaunchBuffers.
__forceinline__ __device__ void bdrcLightIntegrator(PerRayData& prd, const uint2 theLaunchIndex)
{
  prd.radiance  = make_float3(0.0f);
  prd.pdf       = 0.0f;
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent = BSDF_EVENT_ABSORB;
  prd.flags     = 0; // No FLAG_LIGHT_PATH: disables camera connections in hit.cu
  prd.eyePathFoundDiffuse    = false;
  prd.eyePathInSpecularChain = false;

  // Initialize IOR material stack to vacuum — required for correct glass refraction.
  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;
  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y
                                + theLaunchIndex.x;

  // Requires a valid eye vertex written by the eye-path integrator this frame.
  if (!nrc.renderValid[pixelIdx])
    return;

  float3 origin;
  float3 direction;

  if (!sampleEmitterRay(prd, origin, direction))
    return;

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  int  depth        = 0;
  int  diffuseDepth = 0;    // counts accepted diffuse-after-specular connections (depth index into light buffer)
  bool hasSpecular  = false; // require at least one specular bounce before accepting diffuse

  while (depth < theLaunchParameters.lightPathMaxDepth)
  {
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = 0;

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;

    if (!(prd.flags & FLAG_HIT))
      break;

    if (prd.typeEvent & BSDF_EVENT_DIFFUSE)
    {
      if (hasSpecular)
      {
        // Read S (last specular vertex on the eye path, or camera default).
        const float3 S = make_float3(
            nrc.eyeVertexPos[3*pixelIdx+0],
            nrc.eyeVertexPos[3*pixelIdx+1],
            nrc.eyeVertexPos[3*pixelIdx+2]);

        const float3 vecYS  = S - prd.pos;
        const float  distYS = length(vecYS);

        if (distYS > theLaunchParameters.sceneEpsilon)
        {
          const float3 dir_Y_to_S = vecYS / distYS;

          // Shadow ray: Y → S (full OptiX visibility test).
          unsigned int isVisible  = 0;
          unsigned int shadowSeed = prd.seed;
          optixTrace(theLaunchParameters.handle,
                     prd.pos, dir_Y_to_S,
                     theLaunchParameters.sceneEpsilon,
                     distYS - theLaunchParameters.sceneEpsilon,
                     0.0f,
                     OptixVisibilityMask(0xFF),
                     OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                     TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
                     isVisible, shadowSeed);
          prd.seed = shadowSeed;

          if (isVisible && diffuseDepth < BDRC_MAX_LIGHT_DEPTH)  // shadow miss == Y can see S
          {
            // target = tp_Y_pre × f_Y_toward_S
            // tp_Y_pre: light path throughput at Y BEFORE bsdf_over_pdf (view-independent)
            // f_Y_toward_S: Lambertian BSDF at Y toward S = albedo × (cosθ / π)
            const float3 tp_Y_pre     = prd.throughputPreBsdf;
            const float  cos_Y        = fmaxf(dot(prd.nrcN, dir_Y_to_S), 0.0f);
            const float3 f_Y_toward_S = prd.nrcBaseColor * (cos_Y * M_1_PIf);
            const float3 target       = tp_Y_pre * f_Y_toward_S;

            const float3 eyeVertexNormal = make_float3(
              nrc.eyeVertexNormals[3*pixelIdx+0],
              nrc.eyeVertexNormals[3*pixelIdx+1],
              nrc.eyeVertexNormals[3*pixelIdx+2]);

            // p_e_sa = cos(angle between S-direction and hemisphere axis) / pi
            const float p_e_sa = fmaxf(dot(dir_Y_to_S, eyeVertexNormal), 0.0f) * M_1_PIf;
            // p_e_area(Y) = p_e_sa * cos_Y / dist_YS²  (solid-angle → area conversion at Y)
            const float p_e_area = p_e_sa * cos_Y / fmaxf(distYS * distYS, 1e-8f);

            // Depth-strided slot: d * numPixels + pixelIdx.  No atomics needed.
            const unsigned int numPix = static_cast<unsigned int>(
                theLaunchParameters.resolution.x * theLaunchParameters.resolution.y);
            const unsigned int slot = static_cast<unsigned int>(diffuseDepth) * numPix + pixelIdx;
            bdrcPackFeatures(nrc.lightEndpoints, slot, prd.posLocal, prd.localAabbMin, prd.localAabbMax);
            nrc.lightTargets[3*slot+0] = target.x;
            nrc.lightTargets[3*slot+1] = target.y;
            nrc.lightTargets[3*slot+2] = target.z;
            nrc.light_p_e_area[slot]   = p_e_area;
            const float3 eyeTp = make_float3(
                nrc.eyeVertexThroughput[3*pixelIdx+0],
                nrc.eyeVertexThroughput[3*pixelIdx+1],
                nrc.eyeVertexThroughput[3*pixelIdx+2]);
            nrc.lightEyeTp[3*slot+0] = eyeTp.x;
            nrc.lightEyeTp[3*slot+1] = eyeTp.y;
            nrc.lightEyeTp[3*slot+2] = eyeTp.z;
            nrc.lightValid[slot]  = 1u;
            ++diffuseDepth;
          }
        }
      }
      // Continue tracing from this diffuse vertex — captures S*D+S*E paths.
    }
    else
    {
      // Specular/transmissive bounce: mark and continue.
      hasSpecular = true;
    }

    // Russian Roulette applies to all bounces (diffuse and specular).
    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
      if (probability < rng(prd.seed))
        break;
      prd.throughput /= probability;
    }

    ++depth;
  }
  // No stale-data zeroing needed: all writes go to compact atomic slots.
}


// BDPT: Traces a path from a light source and atomically scatters contributions into bufferAccumLightPath.
// One thread per pixel, using a different seed than the eye path raygen to avoid correlation.
extern "C" __global__ void __raygen__light_path_tracer()
{
  // Skip if the BD accumulation buffer is not allocated (and BDRC is not active) or during picking.
  if ((theLaunchParameters.bufferAccumLightPath == nullptr && theLaunchParameters.nrc.nrcType != BDRC) ||
      0.0f <= theLaunchParameters.picking.x)
  {
    return;
  }

  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;

  // Use a seed offset to decorrelate light paths from eye paths.
  prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x,
                    theLaunchParameters.iteration + 0x12345678u);

  if (theLaunchParameters.nrc.nrcType == BDRC)
  {
    bdrcLightIntegrator(prd, theLaunchIndex);
  }
  else
  {
    lightPathIntegrator(prd, theLaunchIndex);
  }
}


// This raygen program is used with m_interop == INTEROP_IMG and directly accumulates inside the registered OpenGL display texture.
extern "C" __global__ void __raygen__path_tracer_surface()
{
  //const uint2 theLaunchDim   = make_uint2(optixGetLaunchDimensions());
  const uint2 theLaunchIndex = make_uint2(optixGetLaunchIndex());

  PerRayData prd;
  
  float2 fragment;

  if (theLaunchParameters.picking.x < 0.0f) // Normal rendering, not picking.
  {
    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    // PERF This template really generates a lot of instructions.
    prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x, theLaunchParameters.iteration); 

    // Calculate primary ray direction from perspective camera position into the scene, in world space.
    fragment = make_float2(theLaunchIndex) + rng2(prd.seed); // Jitter sub-pixel location.
  }
  else // Picking ray.
  {
    prd.indexMaterial = -1; // Default result when not hitting anything.

    fragment = theLaunchParameters.picking;
  }

  const float2 ndc = (fragment / make_float2(theLaunchParameters.resolution)) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

  if (theLaunchParameters.cameraType) // perspective camera
  {
    // All primary rays start at the camera position and go through the camera plane at the end of the W vector.
    prd.pos = theLaunchParameters.cameraP;
    prd.wi  = normalize(theLaunchParameters.cameraU * ndc.x +
                        theLaunchParameters.cameraV * ndc.y + 
                        theLaunchParameters.cameraW);
  }
  else // if orthographic camera
  {
    // Primary rays start on a camera plane at the camera postion and all directions are parallel.
    prd.pos = theLaunchParameters.cameraP + 
              theLaunchParameters.cameraU * ndc.x +
              theLaunchParameters.cameraV * ndc.y;
    prd.wi  = theLaunchParameters.cameraW; // W is normalized for orthographic cameras.
  }

  float3 radiance = integrator(prd, theLaunchIndex);

#if USE_DEBUG_EXCEPTIONS // DEBUG Highlight numerical errors.
  if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
  {
    radiance = make_float3(1000000.0f, 0.0f, 0.0f); // super red
    //printf("NAN at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z))
  {
    radiance = make_float3(0.0f, 1000000.0f, 0.0f); // super green
    //printf("INF at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
  else if (radiance.x < 0.0f || radiance.y < 0.0f || radiance.z < 0.0f)
  {
    radiance = make_float3(0.0f, 0.0f, 1000000.0f); // super blue
    //printf("NEG at (%d, %d)\n", theLaunchIndex.x, theLaunchIndex.y);
  }
#endif

  if (theLaunchParameters.picking.x < 0.0f) // Rendering.
  {
    // NRC/BDRC-terminated pixels are composited by the host dispatch; skip here.
    // Debug modes also skip — the dispatch kernels own bufferAccum in that case.
    if (!prd.nrcTerminated && theLaunchParameters.nrc.nrcDebugMode == 0)
    {
      if (0 < theLaunchParameters.iteration)
      {
        const float t = 1.0f / static_cast<float>(theLaunchParameters.iteration + 1);

        if (theLaunchParameters.bufferAccum != nullptr) // BDPT mode: progressive average in flat buffer.
        {
          const unsigned int index = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
          const float3 colorAccum = make_float3(theLaunchParameters.bufferAccum[index]);
          radiance = lerp(colorAccum, radiance, t);
        }
        else // Standard surface mode: read previous average directly from the display surface.
        {
          float4 colorAccum;
          surf2Dread(&colorAccum, theLaunchParameters.surface, theLaunchIndex.x * sizeof(float4), theLaunchIndex.y, cudaBoundaryModeZero);
          radiance = lerp(make_float3(colorAccum), radiance, t);
        }
      }

      if (theLaunchParameters.bufferAccum != nullptr) // BDPT mode: write to flat buffer; surface is written by device_blend_to_surface.
      {
        const unsigned int index = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
        theLaunchParameters.bufferAccum[index] = make_float4(radiance, 1.0f);
      }
      else
      {
        surf2Dwrite(make_float4(radiance, 1.0f), theLaunchParameters.surface, theLaunchIndex.x * sizeof(float4), theLaunchIndex.y);
      }
    }
  }
  else // Picking.
  {
    theLaunchParameters.bufferPicking[0] = prd.indexMaterial;
  }
}

