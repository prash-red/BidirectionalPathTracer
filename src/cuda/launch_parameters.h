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
  unsigned int* renderValid;           // [numPixels] 0 / 1

  // ---- Forward NRC buffers -------------------------------------------------
  float*        renderFeatures;        // [NRC_INPUT_DIM × numPixels] column-major (14-dim)
  float*        trainNextFeatures;     // [NRC_INPUT_DIM × numPixels] suffix vertex features
  float*        trainSuffixScaled;     // [3 × numPixels]  = throughputAtK * (emit+direct at K+1)
  float*        trainSuffixThroughput; // [3 × numPixels]  = throughput at K+1
  float*        trainThroughputAtK;    // [3 × numPixels]  throughput at vertex K
  unsigned int* trainNextValid;        // [numPixels] 0 / 1

  // ---- Scene AABB ------------------------------------------------------------
  float3 aabbMin;
  float3 aabbMax;

  // ---- Area-spread termination (Mueller et al. 2021, Section 3.4) -----------
  float spreadC;              // Default 0.01. Trades variance vs. bias/speed.
  int   terminationDepth;     // Fallback max depth if spread threshold not met.

  // ---- Mode flags -----------------------------------------------------------
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

  int2   resolution; // Rendering resolution. Usually matches optixGetLaunchDimensions, except when picking!
  float2 picking;    // Screen space coordinate of the picking ray inside the resolution
  
  int2 pathLengths;  // .x == minimum path length before Russian Roulette starts, .y = maximum path length.

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
