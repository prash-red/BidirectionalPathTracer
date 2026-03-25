#include <cuda_runtime.h>
// #include <surface_functions.h>
#include <vector_types.h>


// Composite the eye path progressive average (bufferAccum) and the scaled light path
// running sum (bufferAccumLightPath) into a single output buffer entirely on the GPU.
// dst[i] = eye[i] + lightPath[i] * scaleLightPath
static __global__ void kernel_blend_buffers(
    float4*       dst,
    const float4* eye,
    const float4* lightPath,
    const float   scaleLightPath,
    const int     numPixels)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPixels)
  {
    return;
  }

  const float4 e = eye[i];
  const float4 l = lightPath[i];

  dst[i] = make_float4(e.x + l.x * scaleLightPath,
                       e.y + l.y * scaleLightPath,
                       e.z + l.z * scaleLightPath,
                       1.0f);
}


// Host-side launcher for the buffer variant (INTEROP_OFF).
__host__ void device_blend_buffers(
    cudaStream_t  cudaStream,
    float4*       dst,
    const float4* eye,
    const float4* lightPath,
    const float   scaleLightPath,
    const int     numPixels)
{
  constexpr int blockSize = 256;
  const int     gridSize  = (numPixels + blockSize - 1) / blockSize;

  kernel_blend_buffers<<<gridSize, blockSize, 0, cudaStream>>>(dst, eye, lightPath, scaleLightPath, numPixels);
}


// Composite for INTEROP_IMG: the eye path average is already in the mapped surface.
// Reads each pixel from the surface, adds the scaled light path contribution, and writes back.
// surface[x,y] = surface[x,y] + lightPath[y*width+x] * scaleLightPath
static __global__ void kernel_blend_surface(
    cudaSurfaceObject_t surface,
    const float4*       lightPath,
    const float         scaleLightPath,
    const int           width,
    const int           height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
  {
    return;
  }

  float4 e;
  surf2Dread(&e, surface, x * (int)sizeof(float4), y);

  const float4 l = lightPath[y * width + x];

  surf2Dwrite(make_float4(e.x + l.x * scaleLightPath,
                          e.y + l.y * scaleLightPath,
                          e.z + l.z * scaleLightPath,
                          1.0f),
              surface, x * (int)sizeof(float4), y);
}


// Host-side launcher for the surface variant (INTEROP_IMG).
__host__ void device_blend_surface(
    cudaStream_t        cudaStream,
    cudaSurfaceObject_t surface,
    const float4*       lightPath,
    const float         scaleLightPath,
    const int           width,
    const int           height)
{
  const dim3 blockDim(16, 16);
  const dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

  kernel_blend_surface<<<gridDim, blockDim, 0, cudaStream>>>(surface, lightPath, scaleLightPath, width, height);
}


// Composite flat eye path buffer + scaled flat light path buffer into the display surface (write-only).
// Used for INTEROP_IMG BDPT: eye path accumulates in bufferAccum (not the surface), so the
// progressive average is never contaminated by the light path contribution.
// Pass scaleLightPath == 0.0f to copy eye to surface without any light path addition.
// surface[x,y] = eye[y*width+x] + lightPath[y*width+x] * scaleLightPath
static __global__ void kernel_blend_to_surface(
    cudaSurfaceObject_t surface,
    const float4*       eye,
    const float4*       lightPath,
    const float         scaleLightPath,
    const int           width,
    const int           height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
  {
    return;
  }

  const int    i = y * width + x;
  const float4 e = eye[i];
  const float4 l = lightPath[i];

  surf2Dwrite(make_float4(e.x + l.x * scaleLightPath,
                          e.y + l.y * scaleLightPath,
                          e.z + l.z * scaleLightPath,
                          1.0f),
              surface, x * (int)sizeof(float4), y);
}


// Host-side launcher: composite bufferAccum (eye) + bufferAccumLightPath (light) into the display surface.
// For INTEROP_IMG + BDPT: keeps eye path progressive average isolated from the display surface.
__host__ void device_blend_to_surface(
    cudaStream_t        cudaStream,
    cudaSurfaceObject_t surface,
    const float4*       eye,
    const float4*       lightPath,
    const float         scaleLightPath,
    const int           width,
    const int           height)
{
  const dim3 blockDim(16, 16);
  const dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);

  kernel_blend_to_surface<<<gridDim, blockDim, 0, cudaStream>>>(surface, eye, lightPath, scaleLightPath, width, height);
}