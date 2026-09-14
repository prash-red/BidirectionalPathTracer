//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
//

#pragma once

#ifndef LAUNCH_PARAMETERS_H
#define LAUNCH_PARAMETERS_H

#include <cuda.h>

#include <optix.h>

#include "light_definition.h"

enum NrcType {
    NRC,
    BDRC
};

// Maximum number of diffuse-after-specular connections stored per light path.
// Controls the size of the depth-strided light buffer: BDRC_MAX_LIGHT_DEPTH × numPixels.
constexpr int BDRC_MAX_LIGHT_DEPTH = 6;

// Global diffuse-surface reference table for CPM/NCE mask-cache training.
// Density at ANY point on this surface set is the single scalar invTotalArea
// (pure area-uniform reference — the a_eff=0 case of the Python project's
// surface_mixture reference distribution). Built once at scene load; never
// depends on the camera or the current eye path.
struct SurfaceReferenceTable
{
  float3*      positions;    // [3 * triCount] world-space triangle verts (v0,v1,v2 per tri)
  float*       cdf;          // [triCount + 1] normalized cumulative-area CDF
  unsigned int triCount;
  float        invTotalArea; // 1 / total surface area — the known reference density
};

// Device buffers used by the NRC path-termination and self-training loop.
// All float* and unsigned int* are device pointers.
// When enabled == 0, all pointers may be nullptr and no NRC code executes.
struct NrcLaunchBuffers
{
  // ---- Shared render-query buffers -------------------------------------------
  // Written by raygen at the NRC-termination vertex; composited on the host.
  float*        renderThroughput;      // [3 × numPixels]  throughput at vertex K
  float*        renderPrefix;          // [3 × numPixels]  accumulated direct radiance up to K
  float*        renderOutput;          // [3 × numPixels]  filled by nrc.query() on host
  float*        renderReflectance;     // [3 × numPixels]  clamp(α+β, 0.01) at K — reflectance factorization
  unsigned int* renderValid;           // [numPixels] 0 / 1

  // ---- Forward NRC buffers -------------------------------------------------
  float*        renderFeatures;        // [NRC_INPUT_DIM × numPixels] pixel-indexed (14-dim)
  float*        trainNextFeatures;     // [NRC_INPUT_DIM × batchSize] slot-indexed: suffix terminal features
  float*        trainSuffixScaled;     // [3 × batchSize]  slot-indexed: sum of T_{K+j}*Le_{K+j} over suffix
  float*        trainSuffixThroughput; // [3 × batchSize]  slot-indexed: throughput at last suffix vertex
  float*        trainThroughputAtK;    // [3 × batchSize]  slot-indexed: T_{K+1} at termination vertex
  unsigned int* trainNextValid;        // [batchSize] 0 / 1  (slot-indexed)
  float*        trainReflectance;      // [3 × batchSize]  clamp(α+β, 0.01) per training slot

  // ---- Compact training batch (Option B: stream compaction via atomicAdd) ----
  // raygen writes training pixels into a compact slot array; only slot-indexed
  // buffers (trainFeatures, trainThroughputAtK, trainSuffixScaled,
  // trainSuffixThroughput, trainNextFeatures, trainNextValid) use slot addressing.
  unsigned int* trainCount;        // [1] atomic counter: number of training slots used this frame
  unsigned int* trainPixelIndices; // [batchSize] pixel index for each training slot (unused by kernels, diagnostic)
  float*        trainFeatures;     // [NRC_INPUT_DIM × batchSize] compact slot-indexed train features

  // ---- BDRC light-path buffers — written by bdrcLightIntegrator, one slot per pixel.
  // The light-path grammar stores AT MOST ONE sample per path (the first diffuse
  // vertex reached after >=1 specular bounce), so no depth-striding is needed.
  float*        lightEndpoints;        // [3 × numPixels]  photon-hit position (AABB-normalized)
  float*        lightTargets;          // [3 × numPixels]  photon power * BSDF-eval toward query dir
  float*        light_p_e_area;        // [numPixels]      known reference density at the hit (== surfaceRef.invTotalArea)
  unsigned int* lightValid;            // [numPixels]      1 if a photon landed this frame
  float*        lightOutput;           // [3 × numPixels]  inference output (unchanged)

  // ---- BDRC eye-path supervision buffers (written by eye integrator) ---------
  float*        eyeVertexPos;          // [3 × numPixels]  S pos (last specular before D, or camera)
  float*        eyeVertexNormals;       // [3 × numPixels]  S normal (last specular before D, or camera)
  float*        eyeVertexThroughput;   // [3 × numPixels]  T(cam→D) pre-BSDF (Python ev.throughput)
  float*        resampledEndpoints;     // [3 × numPixels]  R pos normalized (hemisphere from S, or D)
  unsigned int* resampledValid;        // [numPixels]  1 if resampledFeatures is valid
  float*        resampled_p_sa;        // [numPixels]  Hemisphere pdf at eye vertex positions - analytic
  float*        resampled_p_e_area;    // [numPixels]  Eye-path area PDF at R: p_e_sa * cos_R / dist_SR²

  // ---- BDRC training phase ---------------------------------------------------
  int   bdrcPhase;    // 0 = Phase 1 (mask only), 1 = Phase 2 (color caches + frozen mask)

  // ---- Scene AABB ------------------------------------------------------------
  float3 aabbMin;
  float3 aabbMax;

  // ---- Area-spread termination (Mueller et al. 2021, Section 3.4) -----------
  float spreadC;              // Default 0.01. Trades variance vs. bias/speed.
  int   terminationDepth;     // Fallback max depth if spread threshold not met.

  // ---- Mode flags -----------------------------------------------------------
  NrcType nrcType;
  int enabled;       // 0 = NRC off, 1 = NRC active.
  int nrcDebugMode;  // 0=normal, 1=cache-only, 2=mask, 3=indirect (matches NrcDebugMode enum).
};

struct LaunchParameters
{
  // 8 byte alignment
  OptixTraversableHandle handle;

  LightDefinition* lightDefinitions;

  cudaTextureObject_t textureSheenLUT; // Sheen sampling weight lookup texture. Addressed with [cosTheta, roughness]
  
  CUsurfObject surface; // HDR display texture image surface object.
                        // Only valid when m_interop == INTEROP_IMG and then bufferAccum == nullptr.

  float4* bufferAccum;   // Output buffer for the accumulated linear radiance
  float4* bufferAccumLightPath; // Output buffer for the accumulated linear radiance of the light path (used in BDPT)
  int*    bufferPicking; // Output buffer for the picked material index (single integer).


  float3 sceneSphereCenter; // World-space center of the scene bounding sphere (for env-light path sampling).
  float  sceneSphereRadius; // Radius of the scene bounding sphere.

  SurfaceReferenceTable surfaceRef;

  int2   resolution; // Rendering resolution. Usually matches optixGetLaunchDimensions, except when picking!
  float2 picking;    // Screen space coordinate of the picking ray inside the resolution
  
  int2 pathLengths;       // .x == minimum path length before Russian Roulette starts, .y = maximum path length (eye paths).
  int  lightPathMaxDepth; // Maximum bounce depth for light paths (both BDPT and BDRC integrators).

  // 4 byte alignment
  //unsigned int width;
  //unsigned int height;

  unsigned int iteration;    // Sub-frame iteration index.
  float        sceneEpsilon; // Scene-dependent epsilon value on ray tmin to avoid self-intersections.
  int          numLights;    // Number of entries inside the lightDefinitions array.
  
  // Overrides:
  int directLighting;   // 0 == off (singular light types won't work then), 1 == on (default)
  int ambientOcclusion; // 0 == off, ignore all occlusionTexture values, 1 == use occlusionTexture when present (default). Modulates diffuse and metal reflections.
  int showEnvironment;  // 0 == primary rays hitting the miss program will return black in raygen, 1 == standard radiance calculation.
  int forceUnlit;       // 0 == use the material unlit state (default), 1 == force unlit rendering for all materials.

  // Derived parameters for the currently active GLTF camera.
  int    cameraType; // 0 == orthographic, 1 == perspective (default)
  float3 cameraP;
  float3 cameraU;
  float3 cameraV;
  float3 cameraW;

  NrcLaunchBuffers nrc; // Neural Radiance Cache buffers and configuration.
};

#endif // LAUNCH_PARAMETERS_H
