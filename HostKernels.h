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


#endif // HOST_KERNELS_H