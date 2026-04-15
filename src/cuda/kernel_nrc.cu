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

#include <cuda_runtime.h>

// Fused NRC accumulate + build targets kernel:
// 1. For render pixels (renderValid[i]==1): blend NRC-composited radiance into bufferAccum
// 2. For all pixels: build training targets from suffix data
// This fusion eliminates one kernel launch and one sync point.
static __global__ void k_nrc_accumulate_and_build_targets(
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        renderOutput,
    const unsigned int* renderValid,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const unsigned int* trainNextValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
  {
    return;
  }

  const int base3 = 3 * i;

  // ---- Part 1: Accumulate render output (if valid) ----
  if (renderValid[i] != 0u)
  {
    // radiance = prefix + throughput * nrcOutput  (element-wise)
    float r = renderPrefix[base3 + 0] + renderThroughput[base3 + 0] * renderOutput[base3 + 0];
    float g = renderPrefix[base3 + 1] + renderThroughput[base3 + 1] * renderOutput[base3 + 1];
    float b = renderPrefix[base3 + 2] + renderThroughput[base3 + 2] * renderOutput[base3 + 2];

    if (0 < iteration)
    {
      const float t = 1.0f / static_cast<float>(iteration + 1);
      const float4 prev = bufferAccum[i];
      r = prev.x + (r - prev.x) * t;
      g = prev.y + (g - prev.y) * t;
      b = prev.z + (b - prev.z) * t;
    }

    bufferAccum[i] = make_float4(r, g, b, 1.0f);
  }

  // ---- Part 2: Build training targets ----
  const float eps = 1e-6f;
  const bool valid = (trainNextValid[i] != 0u);

  #pragma unroll
  for (int c = 0; c < 3; ++c)
  {
    const float tpK = trainThroughputAtK[base3 + c];
    if (tpK < eps)
    {
      trainTargets[base3 + c] = 0.0f;
      continue;
    }
    float numer = trainSuffixScaled[base3 + c];
    if (valid)
    {
      numer += trainSuffixThroughput[base3 + c] * nrcNext[base3 + c];
    }
    trainTargets[base3 + c] = numer / tpK;
  }
}


// Fused NRC debug + build targets kernel.
// debugMode: 1=cache-only (raw nrcOutput), 2=termination mask, 3=indirect-only (throughput*nrcOutput).
// Like the fused accumulate kernel, this also builds training targets in one pass.
static __global__ void k_nrc_debug_and_build_targets(
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const unsigned int* renderValid,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const unsigned int* trainNextValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
  {
    return;
  }

  const int base3 = 3 * i;

  // ---- Part 1: Debug visualization (if valid) ----
  if (renderValid[i] != 0u)
  {
    float r, g, b;

    if (debugMode == 1) // cache-only: raw NRC output
    {
      r = renderOutput[base3 + 0];
      g = renderOutput[base3 + 1];
      b = renderOutput[base3 + 2];
    }
    else if (debugMode == 2) // termination mask: white for NRC-terminated pixels
    {
      r = g = b = 1.0f;
    }
    else // debugMode == 3: indirect contribution = throughput * nrcOutput
    {
      r = renderThroughput[base3 + 0] * renderOutput[base3 + 0];
      g = renderThroughput[base3 + 1] * renderOutput[base3 + 1];
      b = renderThroughput[base3 + 2] * renderOutput[base3 + 2];
    }

    if (0 < iteration)
    {
      const float t = 1.0f / static_cast<float>(iteration + 1);
      const float4 prev = bufferAccum[i];
      r = prev.x + (r - prev.x) * t;
      g = prev.y + (g - prev.y) * t;
      b = prev.z + (b - prev.z) * t;
    }

    bufferAccum[i] = make_float4(r, g, b, 1.0f);
  }

  // ---- Part 2: Build training targets ----
  const float eps = 1e-6f;
  const bool valid = (trainNextValid[i] != 0u);

  #pragma unroll
  for (int c = 0; c < 3; ++c)
  {
    const float tpK = trainThroughputAtK[base3 + c];
    if (tpK < eps)
    {
      trainTargets[base3 + c] = 0.0f;
      continue;
    }
    float numer = trainSuffixScaled[base3 + c];
    if (valid)
    {
      numer += trainSuffixThroughput[base3 + c] * nrcNext[base3 + c];
    }
    trainTargets[base3 + c] = numer / tpK;
  }
}


// Host-callable wrappers (declared in HostKernels.h).

__host__ void device_nrc_accumulate_and_build_targets(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        renderOutput,
    const unsigned int* renderValid,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const unsigned int* trainNextValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_accumulate_and_build_targets<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderPrefix, renderThroughput, renderOutput, renderValid,
      trainTargets, trainSuffixScaled, trainSuffixThroughput, nrcNext,
      trainThroughputAtK, trainNextValid, numPixels, iteration);
}

__host__ void device_nrc_debug_and_build_targets(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const unsigned int* renderValid,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const unsigned int* trainNextValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_debug_and_build_targets<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderOutput, renderThroughput, renderValid,
      trainTargets, trainSuffixScaled, trainSuffixThroughput, nrcNext,
      trainThroughputAtK, trainNextValid, numPixels, iteration, debugMode);
}