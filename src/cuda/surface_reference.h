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
// The returned position's exact per-unit-area density is tbl.invTotalArea —
// the SAME scalar for every point in the scene, so no per-point density lookup
// is ever needed (this is the pure-uniform case; there is no shape/NDC mixture
// in this port). outNormal is the (unnormalized-safe, already-normalized) face
// normal of the sampled triangle.
__forceinline__ __device__ void sampleSurfaceUniform(
    unsigned int& seed, const SurfaceReferenceTable& tbl, float3& outPos, float3& outNormal)
{
  const float u0 = rng(seed);
  const unsigned int triIdx = surfaceRefBinarySearchCDF(tbl.cdf, tbl.triCount, u0);

  const float3 v0 = tbl.positions[3u * triIdx + 0u];
  const float3 v1 = tbl.positions[3u * triIdx + 1u];
  const float3 v2 = tbl.positions[3u * triIdx + 2u];

  // Uniform barycentric sample (Turk 1990) — same formula as the existing
  // mesh-light triangle sampler in light_sample.cu.
  const float u1 = rng(seed);
  const float u2 = rng(seed);
  const float su = sqrtf(u1);
  const float b0 = 1.0f - su;
  const float b1 = su * (1.0f - u2);
  const float b2 = su * u2;

  outPos    = b0 * v0 + b1 * v1 + b2 * v2;
  const float3 cr = cross(v1 - v0, v2 - v0);
  outNormal = normalize(cr);
}

#endif // GLTF_RENDERER_SURFACE_REFERENCE_H
