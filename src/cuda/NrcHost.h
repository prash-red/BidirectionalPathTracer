//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// NrcHost.h — Plain C++ host interface for the Neural Radiance Cache system.
// Implemented in NrcHost.cu (CUDA compilation unit) to keep tcnn headers out
// of the main Application.cpp translation unit.
//

#pragma once

#ifndef NRC_HOST_H
#define NRC_HOST_H

#include <cuda_runtime.h>
#include <cstdint>

// Opaque handle. Full definition is in NrcHost.cu.
struct NrcSystem;

// Debug visualization modes for nrcDispatch.
enum NrcDebugMode
{
  NRC_DEBUG_OFF                = 0, // Normal composited output (prefix + throughput * nrcOutput).
  NRC_DEBUG_CACHE_ONLY         = 1, // Show raw NRC network output (what the cache has learned).
  NRC_DEBUG_MASK               = 2, // Binary mask: white = NRC-terminated pixel, black = regular.
  NRC_DEBUG_INDIRECT           = 3, // Show only throughput * nrcOutput (indirect contribution).
};

// ============================================================================
// Lifecycle
// ============================================================================

// Create/destroy the NRC system (network + EMA + device buffers).
NrcSystem* nrcCreate(cudaStream_t stream);
void       nrcDestroy(NrcSystem* sys);

// ============================================================================
// Buffer management
// ============================================================================

// Allocate (or reallocate on resolution change) all NRC device buffers.
// batchSize must be a multiple of 256 (tcnn BATCH_SIZE_GRANULARITY).
void nrcAllocBuffers(NrcSystem* sys, uint32_t batchSize);

// ---- Shared pointer accessors ---------------------------------------------
float*        nrcRenderThroughput   (const NrcSystem* sys);
float*        nrcRenderPrefix       (const NrcSystem* sys);
float*        nrcRenderOutput       (const NrcSystem* sys);
unsigned int* nrcRenderValid        (const NrcSystem* sys);

// ---- Forward NRC accessors ------------------------------------------------
float*        nrcRenderFeatures     (const NrcSystem* sys);
float*        nrcTrainNextFeatures  (const NrcSystem* sys);
float*        nrcTrainSuffixScaled  (const NrcSystem* sys);
float*        nrcTrainSuffThroughput(const NrcSystem* sys);
float*        nrcTrainThroughputAtK (const NrcSystem* sys);
unsigned int* nrcTrainNextValid     (const NrcSystem* sys);

// ---- Compact training batch accessors ------------------------------------
unsigned int* nrcTrainCount        (const NrcSystem* sys);
unsigned int* nrcTrainPixelIndices (const NrcSystem* sys);
float*        nrcTrainFeatures     (const NrcSystem* sys);

// ---- Per-frame diagnostics -----------------------------------------------
// Call after nrcDispatch() to read the loss and training sample count
// from the most recently completed training step.
float        nrcGetLastLoss       (const NrcSystem* sys);
unsigned int nrcGetLastTrainCount (const NrcSystem* sys);

// ============================================================================
// Per-frame dispatch
// ============================================================================

// Call after optixLaunch for the eye path:
//
//   Forward NRC:
//     1. EMA render inference  → fills renderOutput (14-dim features)
//     2. Accumulate / visualize → writes into bufferAccum per debugMode
//     3. EMA suffix inference  → fills nrcNextOutput
//     4. Build training targets
//     5. Training step
//     6. EMA update
//     7. Zero valid flags
void nrcDispatch(NrcSystem*   sys,
                 cudaStream_t stream,
                 float4*      bufferAccum,
                 int          numPixels,
                 uint32_t     batchSize,
                 unsigned int iteration,
                 NrcDebugMode debugMode = NRC_DEBUG_OFF);

#endif // NRC_HOST_H
