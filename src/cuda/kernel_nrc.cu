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

  const int    b     = 3 * s;
  const bool   valid = (trainNextValid[s] != 0u);
  const float  eps   = 1e-6f;

  #pragma unroll
  for (int c = 0; c < 3; ++c)
  {
    const float tpK = trainThroughputAtK[b + c];
    if (tpK < eps)
    {
      trainTargets[b + c] = 0.0f;
      continue;
    }
    float numer = trainSuffixScaled[b + c];
    if (valid)
      numer += trainSuffixThroughput[b + c] * nrcNext[b + c];
    trainTargets[b + c] = numer / tpK;
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