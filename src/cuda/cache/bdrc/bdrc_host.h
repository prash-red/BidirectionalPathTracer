//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// bdrc_host.h — host interface for the unified transport cache.
//
// One neural field holding incident flux per unit world area, supervised by a
// photon stream and a gather stream through one weighted density-ratio
// objective. Replaces the three-network eye/light/mask system and its two-phase
// schedule; there is no warmup and no phase, the single fit runs from frame 0.
//
// Deliberately opaque: BdrcSystem is defined in bdrc_host.cu so Application.cpp
// (a plain C++ translation unit) never sees a tiny-cuda-nn header.
//
#pragma once
#ifndef BDRC_HOST_H
#define BDRC_HOST_H

#include "dre_config.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

struct BdrcSystem; // opaque, defined in bdrc_host.cu

// Analytic fixed-point check of the unified field's weighted density-ratio
// objective, over every generator and two conditioning regimes. Takes thousands
// of optimizer steps, so initBDRC() runs it only when GLTF_BDRC_SELFTEST=1 is
// set in the environment. Lives here rather than in unified_ngp_cache.h so a
// plain C++ translation unit can call it without pulling tiny-cuda-nn's device
// headers into a host compile.
bool unifiedSelfTestAll(cudaStream_t stream);

// ---- Lifecycle --------------------------------------------------------------

BdrcSystem* bdrcCreate(cudaStream_t stream, const DreConfig& dre);
void        bdrcDestroy(BdrcSystem* sys);

// numPixels sizes the camera and photon lanes; gatherCount sizes the gather
// lanes and is independent of the render resolution.
void bdrcAllocBuffers(BdrcSystem* sys, uint32_t numPixels, uint32_t gatherCount);

// ---- Training records -------------------------------------------------------
// Lanes [0, photonCount) are photons, [photonCount, lanes) are gathers.

float* bdrcRecQuery     (const BdrcSystem* sys);  // [5 x lanes]
float* bdrcRecBase      (const BdrcSystem* sys);  // [3 x lanes]
float* bdrcRecDensity   (const BdrcSystem* sys);  // [lanes]
float* bdrcRecTailQuery (const BdrcSystem* sys);  // [5 x lanes]
float* bdrcRecTailValid (const BdrcSystem* sys);  // [lanes]
float* bdrcRecFactor    (const BdrcSystem* sys);  // [3 x lanes]

// ---- Camera stream (pixel-indexed) ------------------------------------------

float*        bdrcCamQuery (const BdrcSystem* sys);  // [5 x numPixels]
float*        bdrcCamPrefix(const BdrcSystem* sys);  // [3 x numPixels]
float*        bdrcCamFactor(const BdrcSystem* sys);  // [3 x numPixels]
unsigned int* bdrcCamValid (const BdrcSystem* sys);  // [numPixels]

// ---- Diagnostics for the design note's verification gates -------------------

float    bdrcGetLastLoss   (const BdrcSystem* sys);
uint32_t bdrcGetPhotonLive (const BdrcSystem* sys);  // lanes with density > 0 last frame
uint32_t bdrcGetGatherLive (const BdrcSystem* sys);
float    bdrcGetLastDensity(const BdrcSystem* sys);  // p_r read back from a live lane

// Field statistics over the camera queries. A field that is uniformly zero or
// uniformly saturated is the failure these distinguish from a plausible
// spatial pattern.
float    bdrcGetFieldMean   (const BdrcSystem* sys);
float    bdrcGetFieldMax    (const BdrcSystem* sys);
uint32_t bdrcGetFieldNonZero(const BdrcSystem* sys);
uint32_t bdrcGetFieldCount  (const BdrcSystem* sys);

void bdrcSetDreConfig(BdrcSystem* sys, const DreConfig& dre);

void bdrcSaveWeights(const BdrcSystem* sys, const std::string& path);
bool bdrcLoadWeights(BdrcSystem* sys, cudaStream_t stream, const std::string& path);

// ---- Per-frame dispatch -----------------------------------------------------
//
// Call once after the camera, photon and gather launches have completed.
//
//   tail queries -> EMA inference -> assemble targets -> one DRE loss ->
//   one backward -> one optimizer step -> EMA update -> camera inference ->
//   composite
//
// The tail read uses the EMA teacher while the fit updates the live student: a
// bootstrapped estimator that ends by reading the field's own EMA is unbiased
// for the measure the current field induces, and the true solution is the fixed
// point of the iteration.
void bdrcDispatch(
    BdrcSystem*  sys,
    cudaStream_t stream,
    float4*      bufferAccum,
    int          numPixels,
    unsigned int iteration,          // resets on camera move; drives the film blend
    unsigned int trainingIteration,  // never resets; drives the LR schedule
    unsigned int lrAnnealIters,
    int          debugMode,
    bool         freezeTraining);

#endif // BDRC_HOST_H
