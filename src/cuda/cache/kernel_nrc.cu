#include <cuda_runtime.h>

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
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels || renderValid[i] == 0u)
    return;

  const int pixelIndex = 3 * i;
  float r = renderPrefix[pixelIndex + 0] + renderThroughput[pixelIndex + 0] * renderOutput[pixelIndex + 0];
  float g = renderPrefix[pixelIndex + 1] + renderThroughput[pixelIndex + 1] * renderOutput[pixelIndex + 1];
  float b = renderPrefix[pixelIndex + 2] + renderThroughput[pixelIndex + 2] * renderOutput[pixelIndex + 2];

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
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels || renderValid[i] == 0u)
    return;

  const int pixelIndex = 3 * i;
  float r, g, b;

  if (debugMode == 1) // cache-only: raw NRC output
  {
    r  = renderOutput[pixelIndex + 0];
    g  = renderOutput[pixelIndex + 1];
    b = renderOutput[pixelIndex + 2];
  }
  else if (debugMode == 2) // termination mask
  {
    r = g = b = 1.0f;
  }
  else // debugMode == 3: indirect = throughput * nrcOutput
  {
    r  = renderThroughput[pixelIndex + 0] * renderOutput[pixelIndex + 0];
    g  = renderThroughput[pixelIndex + 1] * renderOutput[pixelIndex + 1];
    b = renderThroughput[pixelIndex + 2] * renderOutput[pixelIndex + 2];
  }

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
    const unsigned int* trainNextValid,
    int                 numTrainSamples)
{
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= numTrainSamples)
    return;

  const int    pixelIndex     = 3 * s;
  const bool   valid = (trainNextValid[s] != 0u);
  const float  eps   = 1e-6f;

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
    trainTargets[pixelIndex + c] = numer / tpK;
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
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_accumulate<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderPrefix, renderThroughput, renderOutput, renderValid,
      numPixels, iteration);
}

__host__ void device_nrc_debug_accumulate(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderOutput,
    const float*        renderThroughput,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_nrc_debug_accumulate<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderOutput, renderThroughput, renderValid,
      numPixels, iteration, debugMode);
}

__host__ void device_nrc_build_targets(
    cudaStream_t        cudaStream,
    float*              trainTargets,
    const float*        trainSuffixScaled,
    const float*        trainSuffixThroughput,
    const float*        nrcNext,
    const float*        trainThroughputAtK,
    const unsigned int* trainNextValid,
    int                 numTrainSamples)
{
  if (numTrainSamples <= 0)
    return;
  const int block = 256;
  const int grid  = (numTrainSamples + block - 1) / block;
  k_nrc_build_targets<<<grid, block, 0, cudaStream>>>(
      trainTargets, trainSuffixScaled, trainSuffixThroughput, nrcNext,
      trainThroughputAtK, trainNextValid, numTrainSamples);
}


// ============================================================================
// BDRC kernels
// ============================================================================

// Composite eye NRC output + light cache output into bufferAccum:
//   radiance = prefix + throughput * eyeOutput + maskOutput * lightOutput
// maskOutput is p_l_area(D) — the scalar mask cache output at the eye vertex D.
// Only writes into bufferAccum for pixels where renderValid[i] != 0.
static __global__ void k_bdrc_composite(
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        eyeOutput,
    const float*        lightOutput,
    const float*        maskOutput,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
    return;

  if (renderValid[i] == 0u)
    return;

  const int   base3 = 3 * i;
  const float mask  = maskOutput[i];  // p_l_area(D): scalar weight for light cache
  float r = renderPrefix[base3+0] + renderThroughput[base3+0]*eyeOutput[base3+0] + mask*lightOutput[base3+0];
  float g = renderPrefix[base3+1] + renderThroughput[base3+1]*eyeOutput[base3+1] + mask*lightOutput[base3+1];
  float b = renderPrefix[base3+2] + renderThroughput[base3+2]*eyeOutput[base3+2] + mask*lightOutput[base3+2];

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

// Host-callable wrappers for BDRC kernels.

__host__ void device_bdrc_composite(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        renderPrefix,
    const float*        renderThroughput,
    const float*        eyeOutput,
    const float*        lightOutput,
    const float*        maskOutput,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_bdrc_composite<<<grid, block, 0, cudaStream>>>(
      bufferAccum, renderPrefix, renderThroughput, eyeOutput, lightOutput,
      maskOutput, renderValid, numPixels, iteration);
}

// BDRC debug visualization kernel.
// debugMode:
//   1 = eye color cache output (ĉ_e)
//   2 = light color cache output (ĉ_l)
//   3 = mask cache output (α) as grayscale
//   4 = combined ĉ_e + ĉ_l
// Only valid pixels (renderValid[i] != 0) are written.
static __global__ void k_bdrc_debug(
    float4*             bufferAccum,
    const float*        eyeOutput,
    const float*        lightOutput,
    const float*        maskOutput,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels || renderValid[i] == 0u)
    return;

  const int base3 = 3 * i;
  float r = 0.f, g = 0.f, b = 0.f;

  if (debugMode == 1)
  {
    r = eyeOutput[base3 + 0];
    g = eyeOutput[base3 + 1];
    b = eyeOutput[base3 + 2];
  }
  else if (debugMode == 2)
  {
    r = lightOutput[base3 + 0];
    g = lightOutput[base3 + 1];
    b = lightOutput[base3 + 2];
  }
  else if (debugMode == 3)
  {
    const float alpha = maskOutput[i];
    r = alpha; g = alpha; b = alpha;
  }
  else if (debugMode == 4)
  {
    r = eyeOutput[base3 + 0] + lightOutput[base3 + 0];
    g = eyeOutput[base3 + 1] + lightOutput[base3 + 1];
    b = eyeOutput[base3 + 2] + lightOutput[base3 + 2];
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

__host__ void device_bdrc_debug(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        eyeOutput,
    const float*        lightOutput,
    const float*        maskOutput,
    const unsigned int* renderValid,
    int                 numPixels,
    unsigned int        iteration,
    int                 debugMode)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_bdrc_debug<<<grid, block, 0, cudaStream>>>(
      bufferAccum, eyeOutput, lightOutput, maskOutput,
      renderValid, numPixels, iteration, debugMode);
}