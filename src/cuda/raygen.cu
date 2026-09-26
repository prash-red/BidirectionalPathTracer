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
        (prd.flags & FLAG_HIT) && (prd.typeEvent & BSDF_EVENT_DIFFUSE))
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

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  int  depth       = 0;
  bool hasSpecular = false;

  while (depth < theLaunchParameters.lightPathMaxDepth)
  {
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = FLAG_BDRC_PHOTON;

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
        // throughputPreBsdf is the power arriving at this vertex, before the
        // local BSDF response. That response is the query's job, not the
        // field's.
        const float  c = theLaunchParameters.nrc.photonClamp;
        const float3 power = make_float3(fminf(fmaxf(prd.throughputPreBsdf.x, 0.0f), c),
                                         fminf(fmaxf(prd.throughputPreBsdf.y, 0.0f), c),
                                         fminf(fmaxf(prd.throughputPreBsdf.z, 0.0f), c));

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

    hasSpecular = true;

    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
      if (probability < rng(prd.seed))
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
      const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

      prd.wo          = -prd.wi;
      prd.distance    = RT_DEFAULT_MAX;
      prd.flags       = FLAG_BDRC_GATHER;
      prd.bdrcSegment = depth + 1;   // 1 on the first segment

      optixTrace(theLaunchParameters.handle,
                 prd.pos, prd.wi,
                 epsilon, prd.distance, 0.0f,
                 OptixVisibilityMask(0xFF),
                 OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                 TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
                 payload.x, payload.y);

      if (!(prd.flags & FLAG_HIT))
        break;   // the miss program added the environment iff this was segment 1

      if (prd.typeEvent & BSDF_EVENT_DIFFUSE)
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

      if (theLaunchParameters.pathLengths.x <= depth)
      {
        const float probability = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
        if (probability < rng(prd.seed))
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

