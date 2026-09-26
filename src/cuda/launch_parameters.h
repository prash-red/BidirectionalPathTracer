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

// Global diffuse-surface reference table for the density-ratio reference
// measure. Density at ANY point on this surface set is the single scalar
// invTotalArea (pure area-uniform reference), so no per-point density lookup
// is ever needed. Built once at scene load; never depends on the camera or on
// any path.
//
// Positions are WORLD space. The gather stream traces shadow rays and
// continuation rays from these anchors, which is only meaningful in world
// space, and the field's own coordinate is world-normalized too (see
// bdrcPackFeatures). World verts go stale under rigid-body node animation; the
// table is rebuilt on scene reload, and animated-instance support would need a
// per-frame rebuild.
struct SurfaceReferenceTable
{
  float3*      positions;    // [3 * triCount] world-space triangle verts (v0,v1,v2 per tri)
  float*       cdf;          // [triCount + 1] normalized cumulative-area CDF
  unsigned int triCount;
  float        invTotalArea; // 1 / total surface area — the reference density
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

  // ---- Unified transport cache: training lanes ------------------------------
  //
  // One fixed-slot records array: lanes [0, photonCount) are photons, lanes
  // [photonCount, photonCount + gatherCount) are gathers. There is no
  // compaction. An invalid lane writes recDensity = 0 and contributes nothing,
  // while its launch still stands in the normalizer — each stream is normalized
  // by its own draw count, not by the live count of valid samples, so a frame
  // with fewer valid samples must not renormalize its gradient up.
  // Persistent training-frame counter. NEVER reset on camera move, unlike
  // LaunchParameters::iteration (which resets for the progressive-accumulation
  // blend). The training streams seed their RNG from this: seeding from
  // `iteration` made every frame of a camera orbit draw the SAME anchors and
  // the SAME emitter samples, so the fit took hundreds of gradient steps on one
  // frozen batch and overfit it. Moving the camera must change rays and film
  // accumulation only.
  unsigned int trainingIteration;

  unsigned int photonCount;   // its own launch dimension, independent of resolution (like gatherCount)
  unsigned int gatherCount;   // its own launch dimension, independent of resolution and framing

  float*       recQuery;      // [5 x lanes]  start/deposit query (position + 2 labels)
  float*       recBase;       // [3 x lanes]  photon: clamped power. gather: E_dir + chain radiance
  float*       recDensity;    // [lanes]      p_r at this lane; 0 disables the lane
  float*       recTailQuery;  // [5 x lanes]  gather tail query
  float*       recTailValid;  // [lanes]      1 when the tail query is meaningful
  float*       recFactor;     // [3 x lanes]  gather: throughput * f_s at the tail

  // ---- Unified transport cache: camera stream (pixel-indexed) --------------
  float*        camQuery;     // [5 x numPixels] query at the first diffuse vertex D
  float*        camPrefix;    // [3 x numPixels] radiance accumulated before and at D
  float*        camFactor;    // [3 x numPixels] T(cam->D) * f_s(D) — the BSDF is applied HERE,
                              //                 because the field stores incident flux, not
                              //                 outgoing radiance
  unsigned int* camValid;     // [numPixels]

  // ---- Unified transport cache: reference measure --------------------------
  // Halved because the field holds a value per (point, side): an anchor takes
  // either side with probability 1/2, so its density per (point, side) is half
  // the area density. Deposits and gathers use the same halved value.
  float        referenceDensity;

  // Stamped once when the cache is constructed; never updated on camera motion.
  float3       referenceOrigin;

  float        photonClamp;   // upper bound on stored photon power

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

// Stochastic progressive photon mapping (SPPM) reference renderer.
//
// Classic global-photon SPPM (Hachisuka & Jensen 2009), ported from the
// sibling project's SPPMRenderer in its global_photon_transport mode: every
// light path deposits a photon at every surface with a diffuse lobe, and each
// pixel's visible point sits at the first vertex where the camera path samples
// a diffuse lobe. There is no next-event estimation anywhere, so direct light
// reaches visible points only through photons, and the estimator is
// independent of the unified cache's path decomposition. That independence is
// the point: a reference that shares the cache's decomposition can share its
// double-counting bugs.
//
// All device pointers; nullptr when SPPM is off.
struct SppmLaunchBuffers
{
  int           enabled;          // 1 when the SPPM reference mode is active

  // ---- Per-pixel visible points, rewritten every iteration ----
  float3*       vpPos;            // [numPixels] world position
  float3*       vpNg;             // [numPixels] raw geometric normal
  unsigned int* vpFront;          // [numPixels] 1 if the camera path arrived on the front face
  float3*       vpFactor;         // [numPixels] T(cam -> vp) * tint / pi  (the diffuse lobe's weight)
  unsigned int* vpActive;         // [numPixels] 1 if this pixel found a visible point this iteration
  float3*       eyeRadiance;      // [numPixels] emission collected along the camera prefix, this iteration

  // ---- Photon deposits, appended every iteration ----
  float3*       phPos;            // [photonCapacity]
  float3*       phNg;             // [photonCapacity] raw geometric normal at the deposit
  unsigned int* phFront;          // [photonCapacity] 1 if the photon arrived on the front face
  float3*       phPower;          // [photonCapacity] incident power, before the receiver BSDF
  unsigned int* photonCount;      // [1] atomic append counter
  unsigned int  photonCapacity;

  unsigned int  photonsPerIter;   // photon paths launched per iteration (the 1/N denominator)
  unsigned int  iteration;        // SPPM iteration, seeds both passes
  // Depths come from LaunchParameters::pathLengths / lightPathMaxDepth, shared
  // with every render mode (see SppmConfig).
  float         photonClamp;      // upper bound on a single deposit's power
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

  SppmLaunchBuffers sppm; // SPPM reference renderer buffers and configuration.
};

#endif // LAUNCH_PARAMETERS_H
