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

#pragma once

#ifndef HOST_KERNELS_H
#define HOST_KERNELS_H

#include <cuda_runtime.h>

#include "Mesh.h"

extern __host__ void device_morphing(cudaStream_t cudaStream,
                                     const float* d_weights,
                                     dev::DevicePrimitive& devicePrim);

extern __host__ void device_skinning(cudaStream_t cudaStream,
                                     const float4* d_skinMatrices,
                                     const unsigned int numSkinMatrices,
                                     dev::DevicePrimitive& devicePrim);

// BDPT GPU blend (INTEROP_OFF): composite eye path average + scaled light path sum into dst.
extern __host__ void device_blend_buffers(cudaStream_t  cudaStream,
                                          float4*       dst,
                                          const float4* eye,
                                          const float4* lightPath,
                                          float         scaleLightPath,
                                          int           numPixels);

// BDPT GPU blend (INTEROP_IMG): read eye path from surface, add scaled light path, write back.
extern __host__ void device_blend_surface(cudaStream_t        cudaStream,
                                          cudaSurfaceObject_t surface,
                                          const float4*       lightPath,
                                          float               scaleLightPath,
                                          int                 width,
                                          int                 height);

// BDPT GPU composite (INTEROP_IMG): read eye path from flat bufferAccum, add scaled light path,
// write final result to surface (write-only — no feedback into the progressive eye path average).
// Pass scaleLightPath == 0.0f to copy eye to surface without light path contribution.
extern __host__ void device_blend_to_surface(cudaStream_t        cudaStream,
                                             cudaSurfaceObject_t surface,
                                             const float4*       eye,
                                             const float4*       lightPath,
                                             float               scaleLightPath,
                                             int                 width,
                                             int                 height);

// NRC: composite NRC-terminated pixels into bufferAccum (pixel-indexed, numPixels entries).
// Applies reflectance factorization: bufferAccum[i] = progressive_avg(prefix + throughput * renderOutput * renderReflectance).
extern __host__ void device_nrc_accumulate(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        renderOutput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration);

// NRC: same as device_nrc_accumulate but with debug visualization instead of composite.
extern __host__ void device_nrc_debug_accumulate(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode);

// NRC: build training targets from compact slot-indexed suffix data (numTrainSamples entries).
// target[s] = (suffixScaled + suffixThroughput*nrcNext) / (trainThroughputAtK * trainReflectance)
// trainReflectance[s] = clamp(α+β, 0.01) at vertex K — reflectance factorization denominator.
extern __host__ void device_nrc_build_targets(
    cudaStream_t        cudaStream,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const float*        trainReflectance,
    const unsigned int* trainNextValid,
    int                 numTrainSamples);

// BDRC unified cache: composite the field into bufferAccum.
// pixel = camPrefix + camFactor * fieldOutput, where camFactor = T(cam->D) * f_s(D).
// The receiver BSDF is applied here because the field stores incident flux.
extern __host__ void device_bdrc_composite(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        camPrefix,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration);

// BDRC unified cache debug view.
// debugMode 1 = camFactor * fieldOutput (the cache's own contribution).
// debugMode 2 = fieldOutput (raw incident flux per area; structure only).
extern __host__ void device_bdrc_debug(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode);

#endif // HOST_KERNELS_H