#include <cuda_runtime.h>

#include "cuda/vector_math.h"

// ============================================================================
// k_nrc_accumulate
// Pixel-indexed (numPixels entries). For each pixel where renderValid[i]==1,
// composite: bufferAccum[i] = progressive_avg(renderPrefix + renderThroughput * renderOutput).
// ============================================================================
static __global__ void k_nrc_accumulate(
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        renderOutput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels || renderValid[i] == 0u)
    return;

  const int pixelIndex = 3 * i;
  // Reflectance factorization: reconstruct L = f*(α+β). nullptr = no factorization (BDRC path).
  const float r0 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 0] : 1.0f;
  const float r1 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 1] : 1.0f;
  const float r2 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 2] : 1.0f;
  float r = renderPrefix[pixelIndex + 0] + renderThroughput[pixelIndex + 0] * renderOutput[pixelIndex + 0] * r0;
  float g = renderPrefix[pixelIndex + 1] + renderThroughput[pixelIndex + 1] * renderOutput[pixelIndex + 1] * r1;
  float b = renderPrefix[pixelIndex + 2] + renderThroughput[pixelIndex + 2] * renderOutput[pixelIndex + 2] * r2;

  if (0 < iteration)
  {
    const float  t    = 1.0f / static_cast<float>(iteration + 1);
    const float4 prev = bufferAccum[i];
    r  = prev.x + (r  - prev.x) * t;
    g  = prev.y + (g  - prev.y) * t;
    b = prev.z + (b - prev.z) * t;
  }

  bufferAccum[i] = make_float4(r, g, b, 1.0f);
}


// ============================================================================
// k_nrc_debug_accumulate
// Pixel-indexed (numPixels entries). Debug visualization instead of composite.
// debugMode: 1=raw cache output, 2=termination mask, 3=indirect contribution.
// ============================================================================
static __global__ void k_nrc_debug_accumulate(
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    int                 debugMode)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
    return;

  // Non-terminated pixels write black so previous-frame accumulation doesn't bleed through.
  if (renderValid[i] == 0u)
  {
    bufferAccum[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  const int pixelIndex = 3 * i;
  float r, g, b;

  const float q0 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 0] : 1.0f;
  const float q1 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 1] : 1.0f;
  const float q2 = (renderReflectance != nullptr) ? renderReflectance[pixelIndex + 2] : 1.0f;
  if (debugMode == 1) // cache-only: reconstructed radiance = network_out * reflectance
  {
    r = renderOutput[pixelIndex + 0] * q0;
    g = renderOutput[pixelIndex + 1] * q1;
    b = renderOutput[pixelIndex + 2] * q2;
  }
  else if (debugMode == 2) // termination mask
  {
    r = g = b = 1.0f;
  }
  else // debugMode == 3: indirect = throughput * network_out * reflectance
  {
    r = renderThroughput[pixelIndex + 0] * renderOutput[pixelIndex + 0] * q0;
    g = renderThroughput[pixelIndex + 1] * renderOutput[pixelIndex + 1] * q1;
    b = renderThroughput[pixelIndex + 2] * renderOutput[pixelIndex + 2] * q2;
  }

  bufferAccum[i] = make_float4(r, g, b, 1.0f);
}


// ============================================================================
// k_nrc_build_targets
// Slot-indexed (numTrainSamples entries, compact). Builds training targets:
//   target[s] = (trainSuffixScaled[s] + trainSuffixThroughput[s] * nrcNext[s])
//               / trainThroughputAtK[s]
// When trainNextValid[s]==0 (unbiased injection or suffix missed), the nrcNext
// bootstrap term is omitted, injecting a purely MC-estimated training signal.
// ============================================================================
static __global__ void k_nrc_build_targets(
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const float*        trainReflectance,
    const unsigned int* trainNextValid,
    int                 numTrainSamples)
{
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= numTrainSamples)
    return;

  const int    pixelIndex = 3 * s;
  const bool   valid      = (trainNextValid[s] != 0u);
  const float  eps        = 1e-6f;

  #pragma unroll
  for (int c = 0; c < 3; ++c)
  {
    const float tpK = trainThroughputAtK[pixelIndex + c];
    if (tpK < eps)
    {
      trainTargets[pixelIndex + c] = 0.0f;
      continue;
    }
    float numer = trainSuffixScaled[pixelIndex + c];
    if (valid)
      numer += trainSuffixThroughput[pixelIndex + c] * nrcNext[pixelIndex + c];
    // Reflectance factorization: divide by clamped (α+β) so the network learns
    // f = L/(α+β) instead of L directly. nullptr = no factorization (BDRC eye cache).
    const float refl = (trainReflectance != nullptr)
                       ? fmaxf(trainReflectance[pixelIndex + c], 0.01f)
                       : 1.0f;
    trainTargets[pixelIndex + c] = (numer / tpK) / refl;
  }
}


// ============================================================================
// Host-callable wrappers (declared in HostKernels.h).
// ============================================================================

__host__ void device_nrc_accumulate(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        renderOutput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_accumulate<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderPrefix, renderThroughput, renderOutput, renderReflectance,
      renderValid, numPixels, iteration);
}

__host__ void device_nrc_debug_accumulate(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const float*        renderReflectance,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        /*iteration*/,
    int                 debugMode)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_debug_accumulate<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderOutput, renderThroughput, renderReflectance, renderValid,
      numPixels, debugMode);
}

__host__ void device_nrc_build_targets(
    cudaStream_t        cudaStream,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const float*        trainReflectance,
    const unsigned int* trainNextValid,
    int                 numTrainSamples)
{
  if (numTrainSamples <= 0)
    return;
  const int block = 256;
  const int grid  = (numTrainSamples + block - 1) / block;
  k_nrc_build_targets<<<grid, block, 0, cudaStream>>>(
      trainTargets, trainSuffixScaled, trainSuffixThroughput, nrcNext,
      trainThroughputAtK, trainReflectance, trainNextValid, numTrainSamples);
}


// ============================================================================
// BDRC kernels
// ============================================================================

// Composite eye NRC output + light cache output into bufferAccum:
//   radiance = prefix + throughput * eyeOutput + maskOutput * lightOutput
// maskOutput is p_l_area(D) — the scalar mask cache output at the eye vertex D.
// Only writes into bufferAccum for pixels where renderValid[i] != 0.
// pixel = prefix + T(cam->D) * f_s(D) * U(D).
//
// camFactor already carries T * f_s: the receiver BSDF is applied HERE, in the
// renderer, because the field stores incident flux per unit area, not outgoing
// radiance. Every path that reads the field applies it the same way — camera
// composite, gather tail and debug view alike.
static __global__ void k_bdrc_composite(
    float4*             bufferAccum,
    const float*        camPrefix,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPixels || !camValid[i]) return;

    const float3 prefix = make_float3(camPrefix[3*i+0], camPrefix[3*i+1], camPrefix[3*i+2]);
    const float3 factor = make_float3(camFactor[3*i+0], camFactor[3*i+1], camFactor[3*i+2]);
    const float3 u      = make_float3(fieldOutput[3*i+0], fieldOutput[3*i+1], fieldOutput[3*i+2]);

    float3 radiance = prefix + factor * u;

    if (0u < iteration)
    {
        const float  t   = 1.0f / static_cast<float>(iteration + 1u);
        const float3 old = make_float3(bufferAccum[i]);
        radiance = old + (radiance - old) * t;
    }
    bufferAccum[i] = make_float4(radiance, 1.0f);
}

__host__ void device_bdrc_composite(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        camPrefix,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration)
{
    constexpr int THREADS = 256;
    const int blocks = (numPixels + THREADS - 1) / THREADS;
    k_bdrc_composite<<<blocks, THREADS, 0, cudaStream>>>(
        bufferAccum, camPrefix, camFactor, fieldOutput, camValid, numPixels, iteration);
}

// debugMode 1 = the field's own contribution at D (factor * U), i.e. the image
//               the cache is responsible for, with the prefix removed.
// debugMode 2 = the raw field value U, with no BSDF applied. This is incident
//               flux per unit area, so it is not a radiance and is only meant
//               to be read for spatial structure.
static __global__ void k_bdrc_debug(
    float4*             bufferAccum,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPixels) return;

    float3 radiance = make_float3(0.0f);
    if (camValid[i])
    {
        const float3 u = make_float3(fieldOutput[3*i+0], fieldOutput[3*i+1], fieldOutput[3*i+2]);
        if (debugMode == 1)
        {
            const float3 factor = make_float3(camFactor[3*i+0], camFactor[3*i+1], camFactor[3*i+2]);
            radiance = factor * u;
        }
        else
        {
            radiance = u;
        }
    }

    if (0u < iteration)
    {
        const float  t   = 1.0f / static_cast<float>(iteration + 1u);
        const float3 old = make_float3(bufferAccum[i]);
        radiance = old + (radiance - old) * t;
    }
    bufferAccum[i] = make_float4(radiance, 1.0f);
}

__host__ void device_bdrc_debug(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        camFactor,
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
    constexpr int THREADS = 256;
    const int blocks = (numPixels + THREADS - 1) / THREADS;
    k_bdrc_debug<<<blocks, THREADS, 0, cudaStream>>>(
        bufferAccum, camFactor, fieldOutput, camValid, numPixels, iteration, debugMode);
}
