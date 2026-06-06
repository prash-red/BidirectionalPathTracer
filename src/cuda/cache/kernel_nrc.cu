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
// Valid pixels write the debug value; invalid pixels write black to avoid prior-frame bleed-through.
static __global__ void k_bdrc_debug(
    float4*             bufferAccum,
    const float*        eyeOutput,
    const float*        lightOutput,
    const float*        maskOutput,
    const unsigned int* renderValid,
    int                 numPixels,
    int                 debugMode)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
    return;

  if (renderValid[i] == 0u)
  {
    bufferAccum[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

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
      renderValid, numPixels, debugMode);
}


// ============================================================================
// k_bdrc_splat_endpoints
// Debug modes 5 (resampled eye endpoints) and 6 (light endpoints).
// For each valid pixel, unpacks the 3D world position from packed BDRC features,
// projects it through the perspective camera, and atomically adds the per-pixel
// value (light cache output or lightTargets) to bufferAccum at the projected pixel.
// bufferAccum must be zero-cleared before this kernel runs.
// ============================================================================
static __global__ void k_bdrc_splat_endpoints(
    float4*             bufferAccum,
    const float*        featureBuffer,    // [3 × numPixels] packed BDRC features (col-major, BDRC_INPUT_DIM=3)
    const float*        valueBuffer,      // [3 × numPixels] RGB contribution to splat
    const unsigned int* validBuffer,      // [numPixels]
    int                 numPixels,
    float               aabbMinX,   float aabbMinY,   float aabbMinZ,
    float               aabbRangeX, float aabbRangeY, float aabbRangeZ,
    float               camPx, float camPy, float camPz,
    float               camUx, float camUy, float camUz,
    float               camVx, float camVy, float camVz,
    float               camWx, float camWy, float camWz,
    int                 resX,
    int                 resY,
    int                 iteration)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels || (validBuffer != nullptr && validBuffer[i] == 0u))
    return;

  // Denormalize packed feature position → world space.
  const float* feat = featureBuffer + i * 3;  // BDRC_INPUT_DIM = 3, features stride by 3
  const float  px   = feat[0] * aabbRangeX + aabbMinX;
  const float  py   = feat[1] * aabbRangeY + aabbMinY;
  const float  pz   = feat[2] * aabbRangeZ + aabbMinZ;

  // Vector from world point toward camera origin (d_to_cam).
  const float dx = camPx - px;
  const float dy = camPy - py;
  const float dz = camPz - pz;

  // dW = dot(d_to_cam, W); negative means point is in front of the camera.
  const float dW = dx*camWx + dy*camWy + dz*camWz;
  if (dW >= 0.0f) return;

  const float W2 = camWx*camWx + camWy*camWy + camWz*camWz;
  const float U2 = camUx*camUx + camUy*camUy + camUz*camUz;
  const float V2 = camVx*camVx + camVy*camVy + camVz*camVz;
  if (U2 < 1e-12f || V2 < 1e-12f) return;

  // Perspective NDC projection — same formula as the BDPT camera connection in hit.cu.
  const float ndcX = (dx*camUx + dy*camUy + dz*camUz) * W2 / (dW * U2);
  const float ndcY = (dx*camVx + dy*camVy + dz*camVz) * W2 / (dW * V2);
  if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f) return;

  const int ix = (int)((ndcX + 1.0f) * 0.5f * (float)resX);
  const int iy = (int)((ndcY + 1.0f) * 0.5f * (float)resY);
  if (ix < 0 || ix >= resX || iy < 0 || iy >= resY) return;

  const unsigned int dstPx  = (unsigned int)(resX * iy + ix);
  float* dst = reinterpret_cast<float*>(bufferAccum) + dstPx * 4;

  float3 prev = make_float3(bufferAccum[dstPx]);
  const float3 colorAccum = make_float3(valueBuffer[i * 3]);
  const float t = 1.0f / static_cast<float>(iteration + 1);

  float3 new_radiance = lerp(prev, colorAccum, t);

  bufferAccum[dstPx] = make_float4(new_radiance.x, new_radiance.y, new_radiance.z, 1.0f);

  // atomicAdd(dst + 0, valueBuffer[i * 3 + 0]);
  // atomicAdd(dst + 1, valueBuffer[i * 3 + 1]);
  // atomicAdd(dst + 2, valueBuffer[i * 3 + 2]);
}

__host__ void device_bdrc_splat_endpoints(
    cudaStream_t        cudaStream,
    float4*             bufferAccum,
    const float*        featureBuffer,
    const float*        valueBuffer,
    const unsigned int* validBuffer,
    int                 numPixels,
    float3              aabbMin,
    float3              aabbRange,
    float3              cameraP,
    float3              cameraU,
    float3              cameraV,
    float3              cameraW,
    int                 resX,
    int                 resY,
    int                 iteration)
{
  const int block = 256;
  const int grid  = (numPixels + block - 1) / block;
  k_bdrc_splat_endpoints<<<grid, block, 0, cudaStream>>>(
      bufferAccum, featureBuffer, valueBuffer, validBuffer, numPixels,
      aabbMin.x,   aabbMin.y,   aabbMin.z,
      aabbRange.x, aabbRange.y, aabbRange.z,
      cameraP.x, cameraP.y, cameraP.z,
      cameraU.x, cameraU.y, cameraU.z,
      cameraV.x, cameraV.y, cameraV.z,
      cameraW.x, cameraW.y, cameraW.z,
      resX, resY, iteration);
}