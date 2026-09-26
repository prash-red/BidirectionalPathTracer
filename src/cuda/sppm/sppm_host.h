//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// sppm_host.h — host interface for the SPPM reference renderer.
//
// Classic global-photon stochastic progressive photon mapping (Hachisuka and
// Jensen 2009), ported from the sibling project's SPPMRenderer with
// global_photon_transport = True. It exists to be a ground truth for the
// unified transport cache, so it deliberately shares none of the cache's path
// decomposition: see SppmLaunchBuffers in launch_parameters.h.
//
// Per iteration:
//   __raygen__sppm_eye     one visible point per pixel + prefix emission
//   __raygen__sppm_photon  photonsPerIter global photon paths
//   sppmEndIteration       hash-grid gather, progressive update, reconstruct
//
// Plain C++-safe: no tiny-cuda-nn, no thrust in this header.
//
#pragma once
#ifndef SPPM_HOST_H
#define SPPM_HOST_H

#include "../launch_parameters.h"

#include <cuda_runtime.h>
#include <cstdint>

struct SppmSystem; // opaque, defined in sppm_host.cu

// Defaults are the sibling project's density-reference-sppm-v2, described
// there as "the new density/indirect ground truth".
struct SppmConfig
{
    uint32_t photonsPerIter = 500000;
    float    initialRadius  = 0.001f;   // fraction of the scene diagonal
    float    minRadius      = 1.0e-5f;  // fraction of the scene diagonal
    float    alpha          = 0.67f;    // fraction of new photons retained per pass
    float    photonClamp    = 10000.0f;
    // No depth limits here. The eye pass stops at pathLengths.y bounces, the
    // photon pass at lightPathMaxDepth, and photon Russian roulette starts after
    // pathLengths.x: the renderer's own limits, shared with every other mode.
    // A reference with its own limits truncates different paths than the method
    // it scores (the unified cache's recursion through U has no bounce limit).
};

struct SppmStats
{
    uint32_t iterations     = 0;
    double   photonsEmitted = 0.0;  // the 1/N denominator: every launched path
    uint32_t lastDeposits   = 0;    // photons stored in the last iteration
    uint32_t lastActiveVps  = 0;    // pixels that found a visible point
    uint32_t discarded      = 0;    // iterations dropped because the photon store overflowed
    float    minRadius      = 0.0f; // world units
    float    maxRadius      = 0.0f;
};

SppmSystem* sppmCreate();
void        sppmDestroy(SppmSystem* sys);

// (Re)allocate for a resolution and configuration, and reset. sceneDiagonal is
// the world-space length of the scene AABB's diagonal; the radii scale by it.
void sppmAllocBuffers(SppmSystem* sys, uint32_t numPixels, const SppmConfig& cfg,
                      float sceneDiagonal);

// Discard all accumulated statistics: tau, N and the eye accumulator to zero,
// every radius back to r0. Call whenever the view or the scene changes.
void sppmReset(SppmSystem* sys, cudaStream_t stream);

// Point the device-side launch buffers at this system's storage.
void sppmFillLaunchBuffers(const SppmSystem* sys, SppmLaunchBuffers& b);

// Before the two launches: clear the photon append counter.
void sppmBeginIteration(SppmSystem* sys, cudaStream_t stream);

// After both launches have completed: build the photon grid, gather at every
// visible point, apply the progressive update, and write the reconstructed
// image  eye_acc / iters + tau / (pi r^2 N_emitted)  into bufferAccum.
void sppmEndIteration(SppmSystem* sys, cudaStream_t stream, float4* bufferAccum,
                      float3 gridOrigin);

SppmStats sppmGetStats(const SppmSystem* sys);

#endif // SPPM_HOST_H
