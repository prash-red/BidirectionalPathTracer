#ifndef GLTF_RENDERER_SURFACE_REFERENCE_H
#define GLTF_RENDERER_SURFACE_REFERENCE_H

#include "launch_parameters.h"
#include "random.h"
#include "vector_math.h"

// Binary search over a normalized cumulative CDF of length n+1 (cdf[0]=0, cdf[n]=1).
// Mirrors the existing binarySearchCDF used for mesh-light sampling in light_sample.cu.
__forceinline__ __device__ unsigned int surfaceRefBinarySearchCDF(
    const float* cdf, unsigned int n, float u)
{
  unsigned int lo = 0u;
  unsigned int hi = n;
  while (lo < hi)
  {
    const unsigned int mid = (lo + hi) >> 1u;
    if (cdf[mid + 1u] < u)
      lo = mid + 1u;
    else
      hi = mid;
  }
  return lo < n ? lo : (n - 1u);
}

// Exact area-uniform sample of the global diffuse-surface reference table.
// The returned point's per-unit-area density is tbl.invTotalArea everywhere,
// so no per-point density lookup is needed (this is the pure-uniform case;
// there is no shape/NDC mixture in this port).
//
// World space: the gather stream shoots a shadow ray and a continuation ray
// from this anchor, neither of which is meaningful in a per-mesh local frame.
//
// Returns false (nothing written) when the table is empty or the sampled
// triangle is degenerate — callers must check.
__forceinline__ __device__ bool sampleSurfaceUniform(
    unsigned int& seed, const SurfaceReferenceTable& tbl,
    float3& outWorldPos, float3& outWorldNormal)
{
  if (tbl.triCount == 0u || tbl.cdf == nullptr)
    return false;

  const float u0 = rng(seed);
  const unsigned int triIdx = surfaceRefBinarySearchCDF(tbl.cdf, tbl.triCount, u0);

  const float3 v0 = tbl.positions[3u * triIdx + 0u];
  const float3 v1 = tbl.positions[3u * triIdx + 1u];
  const float3 v2 = tbl.positions[3u * triIdx + 2u];

  // Uniform barycentric sample (Turk 1990) — same formula as the mesh-light
  // triangle sampler in light_sample.cu.
  const float u1 = rng(seed);
  const float u2 = rng(seed);
  const float su = sqrtf(u1);
  const float b0 = 1.0f - su;
  const float b1 = su * (1.0f - u2);
  const float b2 = su * u2;

  outWorldPos = b0 * v0 + b1 * v1 + b2 * v2;

  const float3 cr  = cross(v1 - v0, v2 - v0);
  const float  len = sqrtf(dot(cr, cr));
  if (len <= 0.0f)
    return false;   // degenerate triangle: no usable face normal
  outWorldNormal = cr / len;
  return true;
}

#endif // GLTF_RENDERER_SURFACE_REFERENCE_H
