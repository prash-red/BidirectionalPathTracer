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
#include "transform.h"
#include "cache/nrc_features.h"
#include "cache/bdrc/unified_features.h"
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

    // BDRC camera stream: stop at the first diffuse vertex D and record one
    // query there. The pixel reads prefix + T(cam->D) * f_s(D) * U(D); the
    // receiver BSDF is applied here, in the renderer, never in the field.
    //
    // No suppression and no continuation: the path simply ends, so it can never
    // traverse a D S+ chain and never double-counts the photon stream's caustic
    // family.
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == BDRC &&
        (prd.flags & FLAG_HIT) && (prd.typeEvent & BSDF_EVENT_DIFFUSE) && prd.bdrcReceiver)
    {
      const unsigned int px = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
      const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

      bdrcPackQuery5(nrc.camQuery, px, prd.pos, prd.bdrcNg,
                     bdrcSideLabel(prd.bdrcFrontFace),
                     nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

      // The prefix is everything chRadiance accumulated along the way: NEE and
      // emission at the S vertices ahead of D, plus D's own NEE and emission
      // added during this segment's trace.
      nrc.camPrefix[3*px+0] = prd.radiance.x;
      nrc.camPrefix[3*px+1] = prd.radiance.y;
      nrc.camPrefix[3*px+2] = prd.radiance.z;

      // f_s(D) for the Lambertian lobe is baseColor / pi. throughputPreBsdf is
      // T(cam->D), before D's own BSDF was applied.
      const float3 f = prd.nrcBaseColor * M_1_PIf;
      nrc.camFactor[3*px+0] = prd.throughputPreBsdf.x * f.x;
      nrc.camFactor[3*px+1] = prd.throughputPreBsdf.y * f.y;
      nrc.camFactor[3*px+2] = prd.throughputPreBsdf.z * f.z;

      nrc.camValid[px]  = 1u;
      prd.nrcTerminated = true;
      break;
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
      // Clamped to 1: throughput can exceed 1 (e.g. refraction's radiance
      // scaling), and dividing by a "probability" above 1 would shrink the
      // path instead of keeping it unbiased.
      const float probability = fminf(fmaxf(prd.throughput), 1.0f);

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
    // Environment light, as in PBRT's InfiniteLight::SampleLe and the sibling
    // demo's emit(). Pick the direction the light comes FROM, w, uniformly on
    // the sphere, then a point uniformly on the disc of radius R perpendicular
    // to w on the w side of the scene; the photon travels along -w. The rays for
    // one w are parallel and the disc spans the scene's whole cross-section, so
    // the scene is lit evenly from every direction.
    //
    // The ray is perpendicular to its disc, so its power is
    //   L(w) / (pdf_dir * pdf_disc) = L(w) * pi R^2 / pdf_dir
    // (= L * pi R^2 * 4 pi for uniform directions), then / light-selection pmf.
    //
    // This used to aim every ray at the scene centre from a point on the bounding
    // sphere and carry only L * numLights: the rays all crossed one point instead
    // of covering the scene, and without pi R^2 * 4 pi the photons were far too
    // dim. Every caller was affected -- the BDPT light path in mode 0, the unified
    // cache's photon stream (environment caustics) and SPPM, where the
    // environment reaches visible points through photons alone.
    float3 w;
    unitSquareToSphere(rng(prd.seed), rng(prd.seed), w, dirPdf);

    const float R     = theLaunchParameters.sceneSphereRadius;
    const TBN   frame(w);
    const float rDisc = R * sqrtf(rng(prd.seed));
    const float phi   = 2.0f * M_PIf * rng(prd.seed);
    origin    = theLaunchParameters.sceneSphereCenter + w * R
              + frame.tangent   * (rDisc * cosf(phi))
              + frame.bitangent * (rDisc * sinf(phi));
    direction = -w;

    // Radiance arriving along -w is what a ray looking along w sees: the same
    // lookup as the miss programs, so photons and eye paths see one environment.
    float3 L = light.emission;
    if (light.typeLight == TYPE_LIGHT_ENV_SPHERE)
    {
      const float3 Rl = transformVector(light.matrixInv, w);
      const float  u  = (atan2f(-Rl.x, Rl.z) + M_PIf) * 0.5f * M_1_PIf;
      const float  v  = acosf(-Rl.y) * M_1_PIf;
      L *= make_float3(tex2D<float4>(light.textureEmission, u, v));
    }

    prd.throughput = L * (M_PIf * R * R / dirPdf) * float(numLights);
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
  else if (light.typeLight == TYPE_LIGHT_POINT)
  {
    // Light position is stored as the translation column of the light matrix.
    origin = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);

    // Isotropic: uniform direction on the sphere.
    unitSquareToSphere(rng(prd.seed), rng(prd.seed), direction, dirPdf);

    // Initial throughput: emission * numLights / dirPdf
    // (1/selectionPdf = numLights, 1/dirPdf = 4*pi for uniform sphere).
    prd.throughput = light.emission * (float(numLights) / dirPdf);
  }
  else if (light.typeLight == TYPE_LIGHT_SPOT)
  {
    // A spot emits only inside its outer cone, with the same linear-in-cosine
    // falloff NEE applies in __direct_callable__light_spot. This used to share
    // the point light's uniform sphere at full intensity: half the photons went
    // backwards out of the cone and every off-axis one was 1/falloff too bright
    // (on cornell_caustic_glass_sphere.glb, a 90-degree spot, the walls read
    // 1.5-2x the path tracer). Sample the cone uniformly and weight by falloff.
    origin = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);

    const float3 axis     = normalize(-make_float3(light.matrixInv[2])); // glTF lights shine down -z
    const float  cosOuter = fmaxf(fminf(light.cosOuter, 1.0f), -1.0f);
    const float  cosT     = 1.0f - rng(prd.seed) * (1.0f - cosOuter);
    const float  sinT     = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
    const float  phi      = 2.0f * M_PIf * rng(prd.seed);
    const TBN    frame(axis);
    direction = frame.tangent   * (sinT * cosf(phi))
              + frame.bitangent * (sinT * sinf(phi))
              + axis * cosT;
    dirPdf = 1.0f / fmaxf(2.0f * M_PIf * (1.0f - cosOuter), DENOMINATOR_EPSILON);

    const float cosRange = fmaxf(DENOMINATOR_EPSILON, light.cosInner - light.cosOuter);
    const float falloff  = fminf(fmaxf((cosT - light.cosOuter) / cosRange, 0.0f), 1.0f);

    prd.throughput = light.emission * (falloff * float(numLights) / dirPdf);
  }
  else if (light.typeLight == TYPE_LIGHT_DIRECTIONAL)
  {
    // A parallel beam: the environment's disc construction with one fixed
    // direction. Directional lights used to fall into the point-light branch
    // (TYPE_LIGHT_DIRECTIONAL > TYPE_LIGHT_POINT) and were emitted isotropically
    // from their matrix translation, which made the "skip" branch unreachable.
    //
    // The power reproduces what NEE delivers, emission * area per unit
    // perpendicular area (__direct_callable__light_directional), so photons and
    // eye paths agree. NOTE: that callable multiplies illuminance by the scene's
    // disc area, so a directional light's brightness grows with scene size.
    // That convention is the renderer's; it is matched here, not endorsed.
    // Not validated against the path tracer: no test scene has one.
    const float3 d     = -normalize(make_float3(light.matrixInv[2])); // direction of travel
    const float  R     = theLaunchParameters.sceneSphereRadius;
    const TBN    frame(d);
    const float  rDisc = R * sqrtf(rng(prd.seed));
    const float  phi   = 2.0f * M_PIf * rng(prd.seed);
    origin    = theLaunchParameters.sceneSphereCenter - d * R
              + frame.tangent   * (rDisc * cosf(phi))
              + frame.bitangent * (rDisc * sinf(phi));
    direction = d;
    dirPdf    = 1.0f;

    prd.throughput = light.emission * (light.area * M_PIf * R * R * float(numLights));
  }
  else
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

  // Russian roulette is judged relative to the emitted power, see below.
  const float emitted = fmaxf(prd.throughput);

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
    //
    // A light path carries emitter-scale power (x numLights / pdf), far above
    // 1. The survival probability must be the throughput RELATIVE to what was
    // emitted, clamped to 1. The raw max component used to be taken as the
    // probability: it never ended a path, and dividing by it (~90 on
    // cornell_caustic_glass_sphere.glb) cut every connection past
    // pathLengths.x to ~1%. Caustics connect at depth 2 and survived; the
    // light they bounce onto other diffuse surfaces (L S+ D D+ E, which eye
    // paths cannot reach through a delta light) was lost: walls 6% dark there.
    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fminf(fmaxf(prd.throughput) / emitted, 1.0f);

      if (!(probability > 0.0f) || probability < rng(prd.seed))
      {
        break;
      }

      prd.throughput /= probability;
    }

    ++depth;
  }
}


// Light-side visibility rule for the unified cache's two training walks, the
// photon stream and the gather chain. Both estimate flux INCIDENT on a
// receiver, so they must see the scene the way light does, which is the way
// shadow rays do: unculled, with the back of a single-sided surface opaque.
//
// With back-face culling (as eye rays use), a walk passes straight through a
// single-sided wall seen from behind. Light from outside a closed single-sided
// room then leaked in through its walls, where a shadow ray from inside is
// blocked. That is the reciprocity break the SPPM photon pass had (2.16x the
// path tracer on cornell_box.glb). The back face must not reflect either: no
// eye ray ever hits it, so light bounced off a wall's outer face would reach
// receivers along paths no eye path has. So the walk ENDS there.
__forceinline__ __device__ bool bdrcBlockedByBackFace(const PerRayData& prd)
{
  return prd.faceCulled && !prd.bdrcFrontFace;
}


// BDRC photon stream — the ungated L S+ D grammar. Traces from an emitter
// through non-diffuse vertices and stores exactly ONE deposit at the first
// diffuse vertex reached after at least one non-diffuse bounce, then stops.
//
// No shadow ray, no connection to any eye vertex, no dependence on the camera.
// The stored value is RAW clamped photon power: under the incident convention
// the receiver BSDF belongs at the query, not in the field.
//
// This stream has no anchor pairing available and works as a measure or not at
// all — its endpoint density is unknown, but the deposit measure IS the flux
// measure by construction of the transport estimator.
__forceinline__ __device__ void bdrcPhotonIntegrator(PerRayData& prd, const unsigned int lane)
{
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  // Disable the lane up front. A path that reaches no deposit leaves it off
  // while its launch still counts in the 1/photonCount normalizer.
  nrc.recDensity[lane]   = 0.0f;
  nrc.recTailValid[lane] = 0.0f;

  prd.radiance  = make_float3(0.0f);
  prd.pdf       = 0.0f;
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent   = BSDF_EVENT_ABSORB;
  prd.flags       = FLAG_BDRC_PHOTON;
  prd.bdrcSegment = 0;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  float3 origin;
  float3 direction;
  if (!sampleEmitterRay(prd, origin, direction))
    return;

  // Russian roulette is judged relative to the emitted power, see below.
  const float emitted = fmaxf(prd.throughput);

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  int  depth       = 0;
  bool hasSpecular = false;

  while (depth < theLaunchParameters.lightPathMaxDepth)
  {
    // t_min = sceneEpsilon on EVERY segment, the first included. A mesh-light
    // photon starts exactly on its emitter triangle; at t_min = 0 it can re-hit
    // that triangle at t ~ 0 and scatter off it instead of leaving (the SPPM
    // photon pass lost over half of cornell_box.glb's panel light this way).
    const float epsilon = theLaunchParameters.sceneEpsilon;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = FLAG_BDRC_PHOTON;

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               // No back-face culling: see bdrcBlockedByBackFace. Same rule as
               // the SPPM photon pass, which is what makes the two agree.
               OPTIX_RAY_FLAG_NONE,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    if (!(prd.flags & FLAG_HIT))
      break;
    if (bdrcBlockedByBackFace(prd))
      break;

    // Only a receiver ends the walk (see bdrcIsCacheReceiver). A diffuse event
    // on a non-receiver counts as S, like the gather chain and the camera do.
    //
    // The stop is decided by the diffuse lobe being SELECTED, which happens
    // with probability p_d = bdrcDiffuseProb, and the deposit is power / p_d.
    // The field holds the full incident flux at the receiver, and every reader
    // applies the diffuse lobe's own weight on its side (the camera and the
    // gather tail stop with that same selection probability). Depositing the
    // raw power only on a diffuse selection recorded p_d * flux: caustics on
    // any receiver with a glossy, sheen or coat layer came out low by the
    // diffuse weight, most at grazing incidence where Fresnel is largest. The
    // photon that selects another lobe continues as S and may deposit later,
    // which is the L S+ family through this vertex's non-diffuse lobes.
    //
    // This test precedes the ABSORB check below: a diffuse selection whose
    // sampled direction falls under the geometric normal ends the walk, but
    // the flux had already arrived and still counts.
    if (prd.bdrcDiffuseSelected && prd.bdrcReceiver)
    {
      if (hasSpecular && prd.bdrcDiffuseProb > 0.0f)
      {
        // throughputPreBsdf is the power arriving at this vertex, before the
        // local BSDF response. That response is the query's job, not the
        // field's. The clamp applies after the 1/p_d weight.
        const float  c = theLaunchParameters.nrc.photonClamp;
        const float3 p = prd.throughputPreBsdf * (1.0f / prd.bdrcDiffuseProb);
        const float3 power = make_float3(fminf(fmaxf(p.x, 0.0f), c),
                                         fminf(fmaxf(p.y, 0.0f), c),
                                         fminf(fmaxf(p.z, 0.0f), c));

        bdrcPackQuery5(nrc.recQuery, lane, prd.pos, prd.bdrcNg,
                       bdrcSideLabel(prd.bdrcFrontFace),
                       nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

        nrc.recBase[3*lane+0] = power.x;
        nrc.recBase[3*lane+1] = power.y;
        nrc.recBase[3*lane+2] = power.z;
        nrc.recDensity[lane]  = nrc.referenceDensity;
      }
      // Terminate regardless: a diffuse hit with no prior specular bounce is a
      // dead end for this grammar.
      break;
    }

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;

    hasSpecular = true;

    // Relative to the emitted power and clamped to 1, as in
    // lightPathIntegrator: the raw max component of photon power is never a
    // probability, and dividing by it shrank every deposit past pathLengths.x.
    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fminf(fmaxf(prd.throughput) / emitted, 1.0f);
      if (!(probability > 0.0f) || probability < rng(prd.seed))
        break;
      prd.throughput /= probability;
    }

    ++depth;
  }
}


// Direct illumination at a point with a UNIT BSDF (f = 1), i.e. an estimate of
// incident flux per unit area rather than outgoing radiance. The demo achieves
// this by forcing the receiver's kd to pi before calling estimate_direct; here
// the anchor is not a ray hit and has no material state at all, so f = 1 is
// written out directly and the BSDF factor is just cos.
//
// Light sampling only. Its MIS partner is the cosine-hemisphere sample the
// gather continues with, whose emitter and environment hits chRadiance and the
// miss program already weight against prd.pdf.
__forceinline__ __device__ float3 bdrcUnitBsdfDirect(
    PerRayData& prd, const float3 pos, const float3 n)
{
  const int numLights = theLaunchParameters.numLights;
  if (!theLaunchParameters.directLighting || numLights <= 0)
    return make_float3(0.0f);

  const int indexLight = (1 < numLights)
      ? clamp(static_cast<int>(floorf(rng(prd.seed) * numLights)), 0, numLights - 1) : 0;
  const LightDefinition& light = theLaunchParameters.lightDefinitions[indexLight];

  prd.pos       = pos;   // the light callables sample relative to prd.pos
  prd.occlusion = 1.0f;

  LightSample lightSample =
      optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(light.typeLight, light, &prd);

  if (lightSample.pdf <= 0.0f)
    return make_float3(0.0f);

  const float cosTheta = dot(lightSample.direction, n);
  if (cosTheta <= 0.0f)
    return make_float3(0.0f);

  unsigned int isVisible = 0;
  unsigned int seed      = prd.seed;

  optixTrace(theLaunchParameters.handle,
             pos, lightSample.direction,
             theLaunchParameters.sceneEpsilon,
             lightSample.distance - theLaunchParameters.sceneEpsilon, 0.0f,
             OptixVisibilityMask(0xFF),
             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
             isVisible, seed);

  prd.seed = seed;

  if (!isVisible)
    return make_float3(0.0f);

  // Singular lights have no BSDF-sampling partner, so they take no MIS weight.
  // For area and environment lights the partner is the cosine hemisphere, whose
  // solid-angle pdf is cos/pi.
  const float weightMIS = (light.typeLight <= TYPE_LIGHT_ENV_SPHERE ||
                           light.typeLight == TYPE_LIGHT_MESH)
                          ? balanceHeuristic(lightSample.pdf / float(numLights), cosTheta * M_1_PIf)
                          : 1.0f;

  return lightSample.radiance_over_pdf * (cosTheta * float(numLights) * weightMIS);
}


// BDRC gather stream. Per lane:
//   1. draw an anchor from the global area-uniform surface table and pick a
//      side with probability 1/2;
//   2. estimate direct illumination there with a unit BSDF, so the estimate is
//      incident flux rather than outgoing radiance;
//   3. sample a cosine hemisphere, continue through any non-diffuse chain to
//      the first diffuse vertex y, and store the tail query and
//      factor = throughput * f_s(y);
//   4. the anchor is both the reference anchor and the gather start, so one
//      query serves the lane's reference term and its positive term, which
//      makes the gathered half of the fit a regression at known points.
//
// Nothing here reads the viewing camera.
__forceinline__ __device__ void bdrcGatherIntegrator(PerRayData& prd, const unsigned int lane)
{
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  nrc.recDensity[lane]   = 0.0f;
  nrc.recTailValid[lane] = 0.0f;

  float3 anchorPos;
  float3 anchorNg;
  if (!sampleSurfaceUniform(prd.seed, theLaunchParameters.surfaceRef, anchorPos, anchorNg))
    return;  // empty or degenerate table: the lane stays disabled

  const float  side     = (rng(prd.seed) < 0.5f) ? 1.0f : -1.0f;
  const float3 oriented = anchorNg * side;

  // The anchor query is written before anything can fail below: it is the
  // lane's reference sample whether or not its gather segment finds anything.
  bdrcPackQuery5(nrc.recQuery, lane, anchorPos, anchorNg, side,
                 nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);
  nrc.recDensity[lane] = nrc.referenceDensity;
  nrc.recBase[3*lane+0] = 0.0f;
  nrc.recBase[3*lane+1] = 0.0f;
  nrc.recBase[3*lane+2] = 0.0f;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent   = BSDF_EVENT_ABSORB;
  prd.flags       = FLAG_BDRC_GATHER;
  prd.bdrcSegment = 0;
  prd.throughput  = make_float3(1.0f);

  const float3 origin = anchorPos + oriented * theLaunchParameters.sceneEpsilon;

  // E_dir at the anchor, seeded into radiance so the continuation's own NEE and
  // emission accumulate onto it exactly as a camera prefix does.
  prd.radiance = bdrcUnitBsdfDirect(prd, origin, oriented);

  // Cosine hemisphere. With f = 1 the weight is f*cos/pdf = cos/(cos/pi) = pi.
  float3 dir;
  float  dirPdf;
  unitSquareToCosineHemisphere(make_float2(rng(prd.seed), rng(prd.seed)), oriented, dir, dirPdf);

  if (dirPdf > 0.0f)
  {
    prd.pos        = origin;
    prd.wi         = dir;
    prd.throughput = make_float3(M_PIf);
    prd.pdf        = dirPdf;   // the MIS partner of the anchor's own light sample

    uint2 payload = splitPointer(&prd);

    int depth = 0;
    while (depth < theLaunchParameters.pathLengths.y)
    {
      // sceneEpsilon on every segment: with culling off (below), the first
      // segment could otherwise re-hit the anchor's own triangle.
      const float epsilon = theLaunchParameters.sceneEpsilon;

      prd.wo          = -prd.wi;
      prd.distance    = RT_DEFAULT_MAX;
      prd.flags       = FLAG_BDRC_GATHER;
      prd.bdrcSegment = depth + 1;   // 1 on the first segment

      optixTrace(theLaunchParameters.handle,
                 prd.pos, prd.wi,
                 epsilon, prd.distance, 0.0f,
                 OptixVisibilityMask(0xFF),
                 // No culling: the gather measures the flux INCIDENT at the
                 // anchor, the same quantity the photon stream deposits, so it
                 // must see the scene the way light does. See
                 // bdrcBlockedByBackFace.
                 OPTIX_RAY_FLAG_NONE,
                 TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                 payload.x, payload.y);

      if (!(prd.flags & FLAG_HIT))
        break;   // the miss program added the environment iff this was segment 1
      if (bdrcBlockedByBackFace(prd))
        break;   // opaque from behind: nothing arrives through it

      if ((prd.typeEvent & BSDF_EVENT_DIFFUSE) && prd.bdrcReceiver)
      {
        // The tail. Its value holds both its own direct light and everything
        // beyond it, so the segment stops here and reads the field instead of
        // continuing — that recursion is what carries every longer path.
        bdrcPackQuery5(nrc.recTailQuery, lane, prd.pos, prd.bdrcNg,
                       bdrcSideLabel(prd.bdrcFrontFace),
                       nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

        const float3 f = prd.nrcBaseColor * M_1_PIf;
        nrc.recFactor[3*lane+0] = prd.throughputPreBsdf.x * f.x;
        nrc.recFactor[3*lane+1] = prd.throughputPreBsdf.y * f.y;
        nrc.recFactor[3*lane+2] = prd.throughputPreBsdf.z * f.z;
        nrc.recTailValid[lane]  = 1.0f;
        break;
      }

      if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
        break;

      // The chain starts at throughput = pi (cosine-sampled hemisphere of a
      // unit BSDF), so survival is judged relative to pi and clamped to 1;
      // dividing by the raw max component darkened every long chain by ~pi.
      if (theLaunchParameters.pathLengths.x <= depth)
      {
        const float probability = fminf(fmaxf(prd.throughput) * M_1_PIf, 1.0f);
        if (!(probability > 0.0f) || probability < rng(prd.seed))
          break;
        prd.throughput /= probability;
      }

      ++depth;
    }
  }

  nrc.recBase[3*lane+0] = fmaxf(prd.radiance.x, 0.0f);
  nrc.recBase[3*lane+1] = fmaxf(prd.radiance.y, 0.0f);
  nrc.recBase[3*lane+2] = fmaxf(prd.radiance.z, 0.0f);
}


// One lane per gather start. The launch width is gatherCount, which is
// independent of the render resolution and of camera framing — the spec's
// stream-count gate checks exactly that.
extern "C" __global__ void __raygen__bdrc_gather()
{
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  if (!nrc.enabled || nrc.nrcType != BDRC || 0.0f <= theLaunchParameters.picking.x)
    return;

  const unsigned int lane = optixGetLaunchIndex().x;
  if (lane >= nrc.gatherCount)
    return;

  // Gather lanes live above the photon lanes in one records array.
  const unsigned int slot = nrc.photonCount + lane;

  PerRayData prd;
  // Seeded from the persistent training counter, never from `iteration`:
  // `iteration` resets on every camera move, which would make an orbiting
  // camera redraw the identical anchor set every frame and overfit one batch.
  prd.seed = tea<4>(lane, theLaunchParameters.nrc.trainingIteration + 0x9e3779b9u);

  bdrcGatherIntegrator(prd, slot);
}


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
  // BDRC photons seed from the persistent training counter (see the gather
  // raygen); the BDPT light path keeps the per-frame `iteration` because it
  // contributes to the progressively-averaged image, not to a fit.
  const unsigned int lightSeedBase =
      (theLaunchParameters.nrc.nrcType == BDRC && theLaunchParameters.nrc.enabled)
      ? theLaunchParameters.nrc.trainingIteration
      : theLaunchParameters.iteration;

  prd.seed = tea<4>(theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x,
                    lightSeedBase + 0x12345678u);

  if (theLaunchParameters.nrc.nrcType == BDRC)
  {
    // BDRC launches this raygen 1-D at photonCount lanes, independent of the
    // render resolution (see Application::render). A 2-D launch at the window
    // size would tie the photon budget, and with it the caustic noise, to the
    // window. BDPT mode still launches 2-D and takes the branch below.
    const unsigned int lane = theLaunchIndex.x;
    if (lane >= theLaunchParameters.nrc.photonCount)
      return;
    bdrcPhotonIntegrator(prd, lane);
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



// ============================================================================
// SPPM reference renderer
//
// Classic global-photon stochastic progressive photon mapping (Hachisuka and
// Jensen 2009), ported from the sibling project's SPPMRenderer with
// global_photon_transport = True. See SppmLaunchBuffers in launch_parameters.h
// for why this variant and not the hybrid default.
// ============================================================================

// Camera pass. Traces from the camera through any vertex that does not sample
// a diffuse lobe, and places the pixel's visible point at the first one that
// does. Emission met along the way is collected at full weight (FLAG_SPPM_EYE
// disables MIS): with no NEE on either pass, every other path reaches the pixel
// through a photon gathered at the visible point.
//
// The visible point's factor is throughput / pi AFTER hit.cu sampled the
// diffuse lobe. hit.cu picks a lobe with probability equal to its layer weight
// and returns bsdf_over_pdf = tint without dividing that weight back out, so
// the stochastic stop already carries the diffuse lobe's weight and
// throughput / pi = T(cam -> vp) * tint / pi is exactly the renderer's own
// diffuse BRDF. Nothing about the material is re-implemented here, which is
// what makes this reference agree with the renderer's path tracer.
__forceinline__ __device__ void sppmEyeIntegrator(PerRayData& prd, const unsigned int px)
{
  const SppmLaunchBuffers& s = theLaunchParameters.sppm;

  s.vpActive[px] = 0u;

  prd.radiance   = make_float3(0.0f);
  prd.pdf        = 0.0f;
  prd.throughput = make_float3(1.0f);
  prd.occlusion  = 1.0f;
  prd.sigma_t    = make_float3(0.0f);
  prd.typeEvent  = BSDF_EVENT_ABSORB;
  prd.flags      = 0;
  prd.idxStack   = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  uint2 payload = splitPointer(&prd);

  // Same limit as the cache's camera and gather walks (pathLengths.y).
  for (int depth = 0; depth < theLaunchParameters.pathLengths.y; ++depth)
  {
    // Primary rays start at the camera and take no t_min offset.
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = FLAG_SPPM_EYE;

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    if (!(prd.flags & FLAG_HIT))
    {
      // The miss program added the environment. Match the path tracer, which
      // hides a directly-seen environment when showEnvironment is off.
      if (depth == 0 && theLaunchParameters.showEnvironment == 0)
        prd.radiance = make_float3(0.0f);
      break;
    }

    if (prd.typeEvent & BSDF_EVENT_DIFFUSE)
    {
      s.vpPos[px]    = prd.pos;
      s.vpNg[px]     = prd.bdrcNg;
      s.vpFront[px]  = prd.bdrcFrontFace ? 1u : 0u;
      s.vpFactor[px] = prd.throughput * M_1_PIf;
      s.vpActive[px] = 1u;
      break;
    }

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;
  }

  s.eyeRadiance[px] = prd.radiance;
}


extern "C" __global__ void __raygen__sppm_eye()
{
  const SppmLaunchBuffers& s = theLaunchParameters.sppm;
  if (!s.enabled || 0.0f <= theLaunchParameters.picking.x)
    return;

  const uint2        idx = make_uint2(optixGetLaunchIndex());
  const unsigned int px  = theLaunchParameters.resolution.x * idx.y + idx.x;

  PerRayData prd;
  // Salted apart from the photon pass so the two draws never correlate.
  prd.seed = tea<4>(px, s.iteration + 0x5eed0000u);

  // Stochastic SPPM: a fresh jittered sample every iteration, so the visible
  // points move and the per-pixel statistics integrate over the pixel footprint.
  const float2 fragment = make_float2(idx) + rng2(prd.seed);
  const float2 ndc      = (fragment / make_float2(theLaunchParameters.resolution)) * 2.0f - 1.0f;

  if (theLaunchParameters.cameraType) // perspective
  {
    prd.pos = theLaunchParameters.cameraP;
    prd.wi  = normalize(theLaunchParameters.cameraU * ndc.x +
                        theLaunchParameters.cameraV * ndc.y +
                        theLaunchParameters.cameraW);
  }
  else // orthographic
  {
    prd.pos = theLaunchParameters.cameraP +
              theLaunchParameters.cameraU * ndc.x +
              theLaunchParameters.cameraV * ndc.y;
    prd.wi  = theLaunchParameters.cameraW;
  }

  sppmEyeIntegrator(prd, px);
}


// Photon pass. Global photons: every light path deposits at EVERY surface with
// a diffuse lobe it reaches, including the first (that is the direct light;
// the camera pass takes none), then continues by sampling the BSDF.
//
// A deposit records the incident power before the local BSDF. It is gated on
// the material HAVING a diffuse lobe, not on this sample picking it: the
// deposit is the flux arriving at the surface, and the visible point applies
// the diffuse lobe later with its own lobe choice.
__forceinline__ __device__ void sppmPhotonIntegrator(PerRayData& prd)
{
  const SppmLaunchBuffers& s = theLaunchParameters.sppm;

  prd.radiance  = make_float3(0.0f);
  prd.pdf       = 0.0f;
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent = BSDF_EVENT_ABSORB;
  prd.flags     = FLAG_SPPM_PHOTON;
  prd.idxStack  = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  float3 origin;
  float3 direction;
  if (!sampleEmitterRay(prd, origin, direction))
    return;   // contributes nothing, but still counts in the 1/N denominator

  // sampleEmitterRay leaves the emitted power in throughput. Russian roulette
  // is judged against it: the absolute value is emitter-scale flux, usually far
  // above 1, so the raw max component would essentially never end a path.
  const float emitted = fmaxf(prd.throughput);
  if (!(emitted > 0.0f))
    return;

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  // Same limit as the cache's photon stream and the BDPT light tracer.
  for (int depth = 0; depth < theLaunchParameters.lightPathMaxDepth; ++depth)
  {
    // The t_min offset applies to the FIRST segment too. A mesh-light photon
    // starts exactly on its emitter triangle; with t_min = 0 and culling off
    // (see below) it re-hits that triangle at t ~ 0, hit.cu takes it as a back
    // face, flips the normal to the ray side and bounces the photon upward --
    // into the ceiling, losing it. On cornell_box.glb that removed over half
    // the panel's light. Eye paths keep t_min = 0 for primary rays: they start
    // at the camera, not on a surface.
    const float epsilon = theLaunchParameters.sceneEpsilon;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = FLAG_SPPM_PHOTON;

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               // NO back-face culling, unlike eye paths. Culling breaks
               // reciprocity: a photon reaching a single-sided wall from
               // outside hits its back face and passes into the room, while a
               // path tracer's shadow ray (which does not cull) from inside
               // toward the same sky is blocked by the front face. Unculled,
               // the photon stops on the outer face, and the gather's
               // frontFace test keeps that deposit away from interior visible
               // points. Measured on cornell_box.glb (all single-sided): with
               // culling SPPM read 2.16x the path tracer, the back wall 5.3x.
               OPTIX_RAY_FLAG_NONE,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    if (!(prd.flags & FLAG_HIT))
      break;

    // A culled surface seen from behind is an opaque ABSORBER for photons. It
    // must block -- shadow rays do not cull, so the path tracer cannot see
    // the sky through a wall from inside. It must not reflect either -- eye
    // rays do cull, so no eye path ever receives light bounced off a wall's
    // outer face. Letting such photons bounce put light reflected off the
    // outside of cornell_box.glb in through its open front: +6% overall there.
    if (prd.faceCulled && !prd.bdrcFrontFace)
      break;

    if (prd.hasDiffuseLobe)
    {
      const unsigned int slot = atomicAdd(s.photonCount, 1u);
      if (slot < s.photonCapacity)
      {
        const float  c = s.photonClamp;
        const float3 p = prd.throughputPreBsdf;
        s.phPos[slot]   = prd.pos;
        s.phNg[slot]    = prd.bdrcNg;
        s.phFront[slot] = prd.bdrcFrontFace ? 1u : 0u;
        s.phPower[slot] = make_float3(fminf(fmaxf(p.x, 0.0f), c),
                                      fminf(fmaxf(p.y, 0.0f), c),
                                      fminf(fmaxf(p.z, 0.0f), c));
      }
    }

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;

    // Russian roulette: the same rule as the cache's photon stream (after
    // pathLengths.x bounces, survival = throughput relative to the emitted
    // power, clamped to 1). The depth limit bounds the loop, so no cap below 1
    // is needed for termination.
    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float q = fminf(fmaxf(prd.throughput) / emitted, 1.0f);
      if (!(q > 0.0f) || rng(prd.seed) >= q)
        break;
      prd.throughput /= q;
    }
  }
}


extern "C" __global__ void __raygen__sppm_photon()
{
  const SppmLaunchBuffers& s = theLaunchParameters.sppm;
  if (!s.enabled || 0.0f <= theLaunchParameters.picking.x)
    return;

  const unsigned int lane = optixGetLaunchIndex().x;
  if (lane >= s.photonsPerIter)
    return;

  PerRayData prd;
  prd.seed = tea<4>(lane, s.iteration + 0x9407e000u);

  sppmPhotonIntegrator(prd);
}
