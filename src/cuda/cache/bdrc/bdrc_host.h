// BdrcHost.h — Host-side interface for the Bidirectional Radiance Cache (BDRC) system.
// Three-cache system: eye color cache + light color cache + mask cache (p_l_area estimation).
// Fully separate from NrcHost — NrcHost.h/.cu are NOT modified.
#pragma once
#ifndef BDRC_HOST_H
#define BDRC_HOST_H

#include <cuda_runtime.h>
#include <cstdint>

struct BdrcSystem; // opaque, defined in BdrcHost.cu

// Lifecycle
BdrcSystem* bdrcCreate(cudaStream_t stream);
void        bdrcDestroy(BdrcSystem* sys);

// Allocate (or reallocate) all BDRC device buffers.
// batchSize must be a multiple of 256.
void bdrcAllocBuffers(BdrcSystem* sys, uint32_t batchSize);

// Buffer accessors — for wiring into NrcLaunchBuffers in Application.cpp
float*        bdrcLightEndpoints    (const BdrcSystem* sys);
float*        bdrcLightTargets      (const BdrcSystem* sys);
unsigned int* bdrcLightValid        (const BdrcSystem* sys);
float*        bdrcLight_p_sa        (const BdrcSystem* sys);
float*        bdrcLight_p_e_area    (const BdrcSystem* sys);
unsigned int* bdrcLightCount        (const BdrcSystem* sys);
float*        bdrcLightOutput       (const BdrcSystem* sys);
float*        bdrcMaskOutput        (const BdrcSystem* sys);
float*        bdrcRenderFeatures    (const BdrcSystem* sys);
float*        bdrcRenderPrefix      (const BdrcSystem* sys);
float*        bdrcRenderThroughput  (const BdrcSystem* sys);
unsigned int* bdrcRenderValid       (const BdrcSystem* sys);
float*        bdrcTrainNextFeatures (const BdrcSystem* sys);
float*        bdrcTrainSuffixScaled (const BdrcSystem* sys);
float*        bdrcTrainSuffThroughput(const BdrcSystem* sys);
float*        bdrcTrainThroughputAtK(const BdrcSystem* sys);
unsigned int* bdrcTrainNextValid    (const BdrcSystem* sys);
float*        bdrcAlbedo             (const BdrcSystem* sys);
float*        bdrcEyeVertexPos      (const BdrcSystem* sys);
float*        bdrcEyeVertexNormals  (const BdrcSystem* sys);
float*        bdrcEyeVertexThroughput(const BdrcSystem* sys);
float*        bdrcResampledEndpoints(const BdrcSystem* sys);
unsigned int* bdrcResampledValid    (const BdrcSystem* sys);
float*        bdrcResampled_p_sa         (const BdrcSystem* sys);
float*        bdrcResampled_p_e_area     (const BdrcSystem* sys);
float*        bdrcCompactLightEndpoints  (const BdrcSystem* sys);
float*        bdrcCompactLightTargets    (const BdrcSystem* sys);
unsigned int* bdrcCompactLightCount      (const BdrcSystem* sys);
float*        bdrcCompactLight_p_e_area  (const BdrcSystem* sys);
float*        bdrcCompactResampledEndpoints(const BdrcSystem* sys);
unsigned int* bdrcCompactResampledCount  (const BdrcSystem* sys);
float*        bdrcCompactResampled_p_e_area(const BdrcSystem* sys);
float         bdrcGetLastEyeLoss  (const BdrcSystem* sys);
float         bdrcGetLastLightLoss(const BdrcSystem* sys);
float         bdrcGetLastMaskLoss (const BdrcSystem* sys);
float*        bdrcRenderOutput    (const BdrcSystem* sys);

// Per-frame dispatch. Call after both eye and light raygen launches complete.
// All eye-path and light-path buffers are owned by BdrcSystem.
//
// Phase 1 (bdrcPhase==0): trains mask cache only. Eye cache shown but not trained.
// Phase 2 (bdrcPhase==1): mask frozen (EMA-only); trains eye + light color caches.
//
// d_trainCount / d_trainFeatures come from the NRC system (nrcTrainCount /
// nrcTrainFeatures) and hold the compact slot-indexed training batch written by
// handleBdrcCacheLogic.
void bdrcDispatch(
    BdrcSystem*   sys,
    cudaStream_t  stream,
    float4*       bufferAccum,
    int           numPixels,
    uint32_t      batchSize,
    unsigned int  iteration,
    int           bdrcPhase,
    int           debugMode,
    // Compact training batch (from NRC system)
    unsigned int* d_trainCount,
    float*        d_trainFeatures,
    bool          freezeTraining,
    // Camera and scene AABB — required for debug modes 5 and 6 (endpoint splatting)
    float3        aabbMin,
    float3        aabbMax,
    float3        cameraP,
    float3        cameraU,
    float3        cameraV,
    float3        cameraW,
    int           resolutionX,
    int           resolutionY);

#endif // BDRC_HOST_H