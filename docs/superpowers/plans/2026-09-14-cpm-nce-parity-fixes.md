
# CPM/NCE Parity Fixes for BDRC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing three-cache BDRC system (`eyeNrc`/`lightNrc`/`maskNrc` in `src/cuda/cache/bdrc/`) into parity with the current production design of the sibling Python/Mitsuba3 project (`bidirectional-radiance-caching`, docs at `docs/concepts/density-estimation.md` and top-level `CLAUDE.md` there), fixing five identified gaps: (1) the light-path grammar is the project's own deprecated shadow-ray-gated design, not the current ungated `L S⁺ D` design; (2) the mask cache's reference/anchor distribution is a camera-entangled hemisphere probe instead of an exact, camera-independent area-uniform surface sampler; (3) the mask loss is Brier instead of logistic NCE; (4) the light/reference class-balance normalizer is a hard-coded `/100000` instead of the true sample-count ratio; (5) the eye cache has no mechanism to learn the `L S⁺ D [any]⁺` transport family (a caustic reflecting off a second diffuse surface), because the code that suppresses double-counted emission at those vertices never compensates for what it suppresses.

**Architecture:** All five fixes land inside the existing BDRC pipeline (`raygen.cu` device integrators, `hit.cu` closest-hit shader, `mask_ngp_cache.cu`/`bdrc_host.cu` training dispatch) — no new cache, no new pipeline stage. The centerpiece is a new host-built, GPU-resident **global diffuse-surface CDF table** (one flat triangle list + cumulative-area CDF over every diffuse-capable mesh instance in the scene, built once at scene load exactly like the existing per-emitter mesh-light CDF in `Application::initMeshLights()`), whose density is a single precomputed scalar `1/totalArea` — this is the pure area-uniform case of the Python project's `surface_mixture` reference distribution, and it replaces the camera-dependent cosine-hemisphere probe used today. The light-path integrator is simplified from a shadow-ray-gated multi-connection design to a single terminate-and-store step per path. The mask loss swaps its analytic Brier gradient for a logistic-NCE gradient with the same fixed point but better peak conditioning. Finally, the eye path's existing `D S⁺ L`-suppression gate in `hit.cu` gains a companion buffer that records what it suppressed, so a new host-side step can query the frozen (EMA) light cache at those vertices and add the missing radiance into the eye cache's training target — never into the immediate rendered frame, since the eye cache is only ever queried once, at its own first-diffuse vertex, at render time.

**Tech Stack:** C++17, CUDA 12+, OptiX 7.7+, tiny-cuda-nn (vendored, `3rdparty/tiny-cuda-nn`), nlohmann/json. No unit-test framework exists in this repository (no CTest/gtest) — verification is `cmake --build` plus running the renderer against `data/cornell_caustic_glass_sphere.glb` / `data/cornell_caustic_glass_sphere_sds.glb` and reading the existing debug-mode visualizations and console `printf`s.

**Spec:** This plan's spec is the gap analysis produced by directly reading the current source of both projects: `/home/prashanth/projects/BidirectionalPathTracer` (`src/cuda/raygen.cu`, `src/cuda/hit.cu`, `src/cuda/cache/bdrc/*`) against `/home/prashanth/projects/bidirectional-radiance-caching` (`cpm/losses/density_ratio.py`, `cpm/scene/steady/surface_tables.py`, `cpm/rendering/path/cpm/{supervision.py,walks.py}`, and top-level `CLAUDE.md`). There is no separate spec document; the "Global Constraints" section below and each task's own description carry the requirements forward.

## Global Constraints

- No unit-test framework exists in this repo. Every task's verification step is a CMake build plus a runtime smoke test — do not add gtest/CTest as part of this plan.
- Build command: `export CUDACXX=/usr/local/cuda/bin/nvcc && cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)`. If `CUDACXX` is already exported in the shell, the export is a no-op.
- Smoke-test scenes (already present in `data/`): `cornell_caustic_glass_sphere.glb` (simple caustic) and `cornell_caustic_glass_sphere_sds.glb` (specular-diffuse-specular, harder case). Run via `./cmake-build-release-system/bin/GLTF_renderer -f <path-to-glb>`.
- The mask/density cache's NCE loss is **unweighted** by eye-vertex throughput — this matches the Python production `surface_mixture` density objective (`cpm/scene/steady/density_training.py`), where throughput-weighting only applies to the separate *weighted-color-cache* architecture, which this C++ system does not use (it is the factorized architecture: separate mask/density cache + separate color caches). Every task that touches mask-cache training must pass `nullptr` for `d_throughputWeights`/`compactTpLum`-style arguments, not compute a real weight.
- Do not remove or rename existing public function signatures in `bdrc_host.h` unless a task explicitly says to — other call sites in `Application.cpp` depend on them and are not being modified except where a task says so.
- Never query `maskNrc`/`lightNrc`/`eyeNrc` with live (non-EMA) weights from a training-target-construction context — only from the training step itself. This preserves the existing EMA-teacher/live-student separation already implemented in `base_ngp_cache`.

---

## File Map

| File | Change |
|------|--------|
| `src/Application.h` | New members: surface reference table device buffers + count/area scalar |
| `src/Application.cpp` | New `initSurfaceReferenceTable()` / `freeSurfaceReferenceTable()`, called next to `initMeshLights()` |
| `src/cuda/launch_parameters.h` | New top-level `LaunchParameters` fields for the surface table; shrink light-path buffers from depth-strided to per-pixel; new eye-side virtual-emission buffers |
| `src/cuda/surface_reference.h` | New file: device-side `sampleSurfaceUniform()` (binary-search CDF + barycentric sample) |
| `src/cuda/per_ray_data.h` | New `PerRayData` field: virtual-emission slot counter |
| `src/cuda/raygen.cu` | Replace hemisphere-probe reference generation; rewrite `bdrcLightIntegrator()` grammar |
| `src/cuda/hit.cu` | Record suppressed `D S⁺ L` vertices into the new virtual-emission buffer |
| `src/cuda/cache/bdrc/mask_ngp_cache.cu` | Replace Brier gradient kernel with logistic-NCE gradient kernel |
| `src/cuda/cache/bdrc/bdrc_host.cu` | Shrink light-buffer allocation/compaction to per-pixel; fix class-balance normalizer; compact + query + scatter-add virtual emission into eye training target |
| `src/cuda/cache/bdrc/bdrc_host.h` | New buffer accessors for the virtual-emission buffers |

---

## Part A: Exact area-uniform surface reference sampler

### Task 1: Build the global diffuse-surface CDF table at scene load

**Files:**
- Modify: `src/Application.h` (add member declarations near `m_meshLightBuffers`, line 575)
- Modify: `src/Application.cpp` (add `initSurfaceReferenceTable()`/`freeSurfaceReferenceTable()` near `initMeshLights()`/`freeMeshLights()`, line ~4549-4560; call sites near line 395 and 1444)

**Interfaces:**
- Produces: `Application::m_surfaceRefPositions` (`CUdeviceptr`, `float3[3*numTriangles]`, world-space triangle verts), `Application::m_surfaceRefCdf` (`CUdeviceptr`, `float[numTriangles+1]`), `Application::m_surfaceRefTriCount` (`uint32_t`), `Application::m_surfaceRefInvTotalArea` (`float`, `= 1/totalArea`). These four are read by Task 3.

- [ ] **Step 1: Add member fields to `Application.h`**

Immediately after `std::vector<CUdeviceptr> m_meshLightBuffers;` (line 575):
```cpp
// ---- Global diffuse-surface reference table (CPM/NCE mask-cache reference) ----
// Flat world-space triangle list + cumulative-area CDF over every diffuse-capable
// mesh instance in the scene (NOT just emitters). Built once at scene load.
// Density at any point on this surface set is the single scalar 1/m_surfaceRefTotalArea.
CUdeviceptr  m_surfaceRefPositions   = 0;  // float3[3 * m_surfaceRefTriCount], world space
CUdeviceptr  m_surfaceRefCdf         = 0;  // float[m_surfaceRefTriCount + 1], normalized to [0,1]
uint32_t     m_surfaceRefTriCount    = 0;
float        m_surfaceRefTotalArea   = 0.0f;
float        m_surfaceRefInvTotalArea = 0.0f;
```

Add the two method declarations next to `initMeshLights()`/`freeMeshLights()` (line 322-323):
```cpp
void initSurfaceReferenceTable();
void freeSurfaceReferenceTable();
```

- [ ] **Step 2: Implement `freeSurfaceReferenceTable()` in `Application.cpp`**

Add immediately before `Application::initMeshLights()` (line 4560):
```cpp
void Application::freeSurfaceReferenceTable()
{
  if (m_surfaceRefPositions) { cudaFree(reinterpret_cast<void*>(m_surfaceRefPositions)); m_surfaceRefPositions = 0; }
  if (m_surfaceRefCdf)       { cudaFree(reinterpret_cast<void*>(m_surfaceRefCdf));       m_surfaceRefCdf       = 0; }
  m_surfaceRefTriCount     = 0;
  m_surfaceRefTotalArea    = 0.0f;
  m_surfaceRefInvTotalArea = 0.0f;
}
```

- [ ] **Step 3: Implement `initSurfaceReferenceTable()` in `Application.cpp`**

Add immediately after `freeSurfaceReferenceTable()`. This mirrors `initMeshLights()`'s existing loop structure (`m_instances` → `m_deviceMeshes` → `m_hostMeshes` → `hostMesh.primitives`) exactly, but includes every diffuse-capable primitive rather than only emissive ones, and every material (a delta/glass BSDF this renderer treats via `roughness ~= 0`, still counted here since the point set only needs to be "wherever a photon could plausibly deposit," which for this renderer's continuous-roughness BSDF model is any surface):
```cpp
void Application::initSurfaceReferenceTable()
{
  freeSurfaceReferenceTable();

  std::vector<float3> worldPositions;
  std::vector<float>  triAreas;

  for (size_t instanceIdx = 0; instanceIdx < m_instances.size(); ++instanceIdx)
  {
    const dev::Instance&   instance   = m_instances[instanceIdx];
    const dev::DeviceMesh& deviceMesh = m_deviceMeshes[instance.indexDeviceMesh];
    const dev::HostMesh&   hostMesh   = m_hostMeshes[deviceMesh.key.idxHostMesh];

    for (size_t primIdx = 0; primIdx < hostMesh.primitives.size(); ++primIdx)
    {
      const dev::HostPrimitive& hostPrim = hostMesh.primitives[primIdx];
      if (hostPrim.indices.count == 0)
        continue;

      const unsigned int* indexData    = reinterpret_cast<const unsigned int*>(hostPrim.indices.h_ptr);
      const float3*       posData      = reinterpret_cast<const float3*>(hostPrim.positions.h_ptr);
      const size_t        numTriangles = hostPrim.indices.count / 3;
      const glm::mat4x4&  xform        = instance.transform;

      for (size_t t = 0; t < numTriangles; ++t)
      {
        const unsigned int i0 = indexData[3 * t + 0];
        const unsigned int i1 = indexData[3 * t + 1];
        const unsigned int i2 = indexData[3 * t + 2];

        const glm::vec4 p0w = xform * glm::vec4(posData[i0].x, posData[i0].y, posData[i0].z, 1.0f);
        const glm::vec4 p1w = xform * glm::vec4(posData[i1].x, posData[i1].y, posData[i1].z, 1.0f);
        const glm::vec4 p2w = xform * glm::vec4(posData[i2].x, posData[i2].y, posData[i2].z, 1.0f);

        const float3 w0 = make_float3(p0w.x, p0w.y, p0w.z);
        const float3 w1 = make_float3(p1w.x, p1w.y, p1w.z);
        const float3 w2 = make_float3(p2w.x, p2w.y, p2w.z);

        const float3 cr      = cross(w1 - w0, w2 - w0);
        const float  triArea = 0.5f * sqrtf(dot(cr, cr));
        if (triArea <= 0.0f)
          continue;

        worldPositions.push_back(w0);
        worldPositions.push_back(w1);
        worldPositions.push_back(w2);
        triAreas.push_back(triArea);
      }
    }
  }

  const size_t numTriangles = triAreas.size();
  if (numTriangles == 0)
    return;

  std::vector<float> cdf(numTriangles + 1, 0.0f);
  for (size_t t = 0; t < numTriangles; ++t)
    cdf[t + 1] = cdf[t] + triAreas[t];
  const float totalArea = cdf[numTriangles];
  if (totalArea <= 0.0f)
    return;
  const float invTotal = 1.0f / totalArea;
  for (size_t t = 1; t <= numTriangles; ++t)
    cdf[t] *= invTotal;
  cdf[numTriangles] = 1.0f;

  const size_t posBytes = worldPositions.size() * sizeof(float3);
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_surfaceRefPositions), posBytes) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_surfaceRefPositions), worldPositions.data(), posBytes, cudaMemcpyHostToDevice) );

  const size_t cdfBytes = cdf.size() * sizeof(float);
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&m_surfaceRefCdf), cdfBytes) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(m_surfaceRefCdf), cdf.data(), cdfBytes, cudaMemcpyHostToDevice) );

  m_surfaceRefTriCount     = static_cast<uint32_t>(numTriangles);
  m_surfaceRefTotalArea    = totalArea;
  m_surfaceRefInvTotalArea = invTotal;
}
```

- [ ] **Step 4: Call the new function next to every existing `initMeshLights()` call site**

At `src/Application.cpp:395` and `src/Application.cpp:1444`, immediately after each `initMeshLights();` line, add:
```cpp
  initSurfaceReferenceTable();
```

- [ ] **Step 5: Build to verify no compile errors**

```bash
cd /home/prashanth/projects/BidirectionalPathTracer
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -30
```
Expected: build succeeds. (These new members are unused by device code so far — that's expected and fine at this step.)

- [ ] **Step 6: Commit**

```bash
git add src/Application.h src/Application.cpp
git commit -m "feat: build global diffuse-surface CDF table for CPM/NCE reference sampling"
```

---

### Task 2: Device-side area-uniform sampler + wire buffers into `LaunchParameters`

**Files:**
- Create: `src/cuda/surface_reference.h`
- Modify: `src/cuda/launch_parameters.h`
- Modify: `src/Application.cpp` (wire the new members into `m_launchParameters` each frame/on rebuild — same place `m_lightDefinitions`/AABB are wired)

**Interfaces:**
- Consumes: `Application::m_surfaceRefPositions/m_surfaceRefCdf/m_surfaceRefTriCount/m_surfaceRefInvTotalArea` (Task 1)
- Produces: `theLaunchParameters.surfaceRef.{positions,cdf,triCount,invTotalArea}`; device function `sampleSurfaceUniform(unsigned int& seed, const SurfaceReferenceTable& tbl, float3& outPos, float3& outNormal)` — read by Task 3.

- [ ] **Step 1: Add the `SurfaceReferenceTable` struct and top-level field to `LaunchParameters`**

In `src/cuda/launch_parameters.h`, add before `struct NrcLaunchBuffers` (line 52):
```cpp
// Global diffuse-surface reference table for CPM/NCE mask-cache training.
// Density at ANY point on this surface set is the single scalar invTotalArea
// (pure area-uniform reference — the a_eff=0 case of the Python project's
// surface_mixture reference distribution). Built once at scene load; never
// depends on the camera or the current eye path.
struct SurfaceReferenceTable
{
  float3*      positions;    // [3 * triCount] world-space triangle verts (v0,v1,v2 per tri)
  float*       cdf;          // [triCount + 1] normalized cumulative-area CDF
  unsigned int triCount;
  float        invTotalArea; // 1 / total surface area — the known reference density
};
```

Add a field to `LaunchParameters` (the struct that follows `NrcLaunchBuffers` in this file, after `sceneSphereCenter`, i.e. near line 131-132 where you find `struct LaunchParameters`):
```cpp
  SurfaceReferenceTable surfaceRef;
```

- [ ] **Step 2: Create `src/cuda/surface_reference.h` with the device sampler**

```cpp
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
```

- [ ] **Step 3: Wire the table into `m_launchParameters` in `Application.cpp`**

Find where `m_launchParameters.lightDefinitions` is assigned (this is set every time lights/scene are (re)built — search for `m_launchParameters.lightDefinitions =` in `Application.cpp`) and add immediately after it:
```cpp
  m_launchParameters.surfaceRef.positions    = reinterpret_cast<float3*>(m_surfaceRefPositions);
  m_launchParameters.surfaceRef.cdf          = reinterpret_cast<float*>(m_surfaceRefCdf);
  m_launchParameters.surfaceRef.triCount     = m_surfaceRefTriCount;
  m_launchParameters.surfaceRef.invTotalArea = m_surfaceRefInvTotalArea;
```

- [ ] **Step 4: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -20
```
Expected: build succeeds. `surface_reference.h` is not yet `#include`d by any `.cu` file, so it is only compile-checked implicitly via the header syntax; that's fine — Task 3 includes it.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/surface_reference.h src/cuda/launch_parameters.h src/Application.cpp
git commit -m "feat: add device-side area-uniform surface sampler and wire into LaunchParameters"
```

---

### Task 3: Replace the hemisphere-probe reference stream with the surface-table sampler

**Files:**
- Modify: `src/cuda/raygen.cu` (the block at lines 599-736, inside `integrator()`)

**Interfaces:**
- Consumes: `theLaunchParameters.surfaceRef` (Task 2), `sampleSurfaceUniform()` (Task 2)
- Produces: unchanged external contract for `nrc.resampledEndpoints`/`nrc.resampledValid`/`nrc.resampled_p_e_area` (still written per diffuse-terminated eye path this frame — buffer shapes and downstream compaction in `bdrc_host.cu` are untouched by this task), but their *meaning* changes: `resampled_p_e_area` is now the constant `surfaceRef.invTotalArea` for every sample, and `resampledEndpoints` is an exact area-uniform surface sample, independent of the eye path's `S`/`lastSpecPos`.

- [ ] **Step 1: Add the include**

At the top of `src/cuda/raygen.cu`, add:
```cpp
#include "surface_reference.h"
```

- [ ] **Step 2: Replace the resampled-R generation block**

Replace the entire block from `// Resampled position R for mask label-0 and light cache zero supervision.` (`raygen.cu:615`) through the closing `}` that ends the `else` branch (`raygen.cu:730`) — i.e. everything between `if (hasSpecular) { ... } else { ... }` inclusive — with:
```cpp
      // Resampled position R for mask label-0 (reference/anchor stream).
      // Exact area-uniform sample over ALL diffuse-capable scene surfaces —
      // camera-independent, no probe ray, no visibility test. This is the
      // pure-uniform case of the Python project's surface_mixture reference
      // distribution: density is the single known scalar invTotalArea
      // everywhere, so no per-point density lookup is needed.
      {
        float3 refPos, refNormal;
        sampleSurfaceUniform(prd.seed, theLaunchParameters.surfaceRef, refPos, refNormal);

        // Reuse posLocal/localAabbMin/localAabbMax packing: the reference
        // sample has no GAS-local frame of its own (it isn't a ray hit), so
        // pack it directly in world space against the same global AABB used
        // to normalize all other cache queries (nrc.aabbMin/aabbMax).
        bdrcPackFeatures(nrc.resampledEndpoints, px, refPos,
                         theLaunchParameters.nrc.aabbMin,
                         theLaunchParameters.nrc.aabbMax);
        nrc.resampledValid[px]     = 1u;
        nrc.resampled_p_sa[px]     = 0.0f;  // no longer a solid-angle PDF; unused downstream
        nrc.resampled_p_e_area[px] = theLaunchParameters.surfaceRef.invTotalArea;

        nrc.eyeVertexNormals[3*px+0] = refNormal.x;
        nrc.eyeVertexNormals[3*px+1] = refNormal.y;
        nrc.eyeVertexNormals[3*px+2] = refNormal.z;
      }
```

This removes the `hasSpecular`/`lastSpecPos`/cosine-hemisphere-probe/`optixTrace` probe ray and the perspective-sensor-PDF fallback entirely — the reference stream no longer traces any ray or depends on where the eye path's last specular vertex was.

- [ ] **Step 3: Remove now-dead debug-mode-5 NEE-at-R capture code**

The removed block contained two `nrc.renderOutput[...] = 1` placeholder writes gated on `nrcDebugMode == 5` (previously at `raygen.cu:672-682` and `719-729`). Since there is no longer a probe ray whose `chRadiance` NEE result could be captured, delete those two `if (theLaunchParameters.nrc.nrcDebugMode == 5) { ... }` blocks along with the code they were part of (they are already inside the block replaced in Step 2, so if Step 2 was applied verbatim, this is already done — this step is a verification check, not new work).

- [ ] **Step 4: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -30
```
Expected: build succeeds. You may see now-unused-variable warnings for `lastSpecPos`/`hasSpecular` if they are still declared earlier in `integrator()` for the BDRC S-tracking block (`raygen.cu:579-587`) — leave that block alone for now, it is still needed by `eyeVertexPos` (used elsewhere) and by Part D later in this plan.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/raygen.cu
git commit -m "fix: replace camera-entangled hemisphere-probe reference with exact area-uniform surface sampler"
```

---

### Task 4: Smoke-test Part A in isolation

**Files:** none (verification-only task)

- [ ] **Step 1: Run with debug mode 3 (mask visualization) and confirm no crash**

```bash
cd /home/prashanth/projects/BidirectionalPathTracer
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb -i 300
```
In the GUI, enable BDRC and set the NRC debug mode to 3 (mask). Expected: renderer runs without crashing or producing NaN/black output; the mask view should still show a plausible, non-degenerate spatial pattern (it will look different from before this task — that's expected, since the reference distribution changed — but it must not be uniformly zero or uniformly saturated).

- [ ] **Step 2: Check the existing `[mask-dbg] resampled p_e_area` printf**

`bdrc_host.cu:510-518` prints the first few `resampled_p_e_area` values every 100 iterations. Confirm the console shows a **constant** value across all printed samples (this is expected now — `invTotalArea` is the same scalar everywhere), and that the value is a small positive float consistent with `1/totalSceneSurfaceArea` for the loaded scene (not zero, not NaN, not enormous).

- [ ] **Step 3: No commit** — this task only verifies Task 1-3; nothing new to commit.

---

## Part B: Fix the light-path grammar (remove shadow-ray gating, terminate-and-store)

### Task 5: Rewrite `bdrcLightIntegrator` to the ungated `L S⁺ D` grammar

**Files:**
- Modify: `src/cuda/raygen.cu` (`bdrcLightIntegrator()`, lines 1052-1208)
- Modify: `src/cuda/launch_parameters.h` (shrink `lightEndpoints`/`lightTargets`/`light_p_e_area`/`lightValid` from depth-strided to per-pixel; remove `lightEyeTp`)

**Interfaces:**
- Produces: `nrc.lightEndpoints`/`nrc.lightTargets`/`nrc.light_p_e_area`/`nrc.lightValid` now sized `[numPixels]` (not `[BDRC_MAX_LIGHT_DEPTH * numPixels]`) — Task 6 updates `bdrc_host.cu` to match.

- [ ] **Step 1: Shrink the depth-strided buffer declarations**

In `src/cuda/launch_parameters.h`, replace the block at lines 79-87 (`// ---- BDRC light-path depth-strided buffers ...` through `float* lightOutput;`) with:
```cpp
  // ---- BDRC light-path buffers — written by bdrcLightIntegrator, one slot per pixel.
  // The light-path grammar stores AT MOST ONE sample per path (the first diffuse
  // vertex reached after >=1 specular bounce), so no depth-striding is needed.
  float*        lightEndpoints;        // [3 × numPixels]  photon-hit position (AABB-normalized)
  float*        lightTargets;          // [3 × numPixels]  photon power * BSDF-eval toward query dir
  float*        light_p_e_area;        // [numPixels]      known reference density at the hit (== surfaceRef.invTotalArea)
  unsigned int* lightValid;            // [numPixels]      1 if a photon landed this frame
  float*        lightOutput;           // [3 × numPixels]  inference output (unchanged)
```
Note `lightEyeTp` is deleted — the new grammar never connects to an eye vertex, so there is no eye-vertex throughput to weight by (see Global Constraints: mask-cache NCE loss is unweighted).

- [ ] **Step 2: Rewrite `bdrcLightIntegrator()`**

Replace the entire function body (`raygen.cu:1052-1208`, from the function's doc comment through its closing `}`) with:
```cpp
// BDRC light path integrator — the L S⁺ D grammar (ungated, single store-and-terminate).
// Traces a path from a light source through specular bounces; stores exactly ONE
// sample at the FIRST diffuse vertex reached after >=1 specular bounce, then
// terminates. No shadow ray to any eye vertex, no dependence on the current
// camera view, no continuation past the stored vertex — this matches the current
// production Python design (trace_photons(caustic_only=True)), replacing the
// project's own earlier shadow-ray-gated, multi-connection design which measurably
// dropped a large fraction of caustic vertices/flux by requiring visibility to a
// specific eye-path vertex.
__forceinline__ __device__ void bdrcLightIntegrator(PerRayData& prd, const uint2 theLaunchIndex)
{
  prd.radiance  = make_float3(0.0f);
  prd.pdf       = 0.0f;
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent = BSDF_EVENT_ABSORB;
  prd.flags     = 0; // No FLAG_LIGHT_PATH: disables camera connections in hit.cu
  prd.eyePathFoundDiffuse    = false;
  prd.eyePathInSpecularChain = false;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;
  const unsigned int pixelIdx = theLaunchParameters.resolution.x * theLaunchIndex.y
                                + theLaunchIndex.x;

  nrc.lightValid[pixelIdx] = 0u;

  float3 origin;
  float3 direction;
  if (!sampleEmitterRay(prd, origin, direction))
    return;

  prd.pos = origin;
  prd.wi  = direction;

  uint2 payload = splitPointer(&prd);

  int  depth       = 0;
  bool hasSpecular = false;

  while (depth < theLaunchParameters.lightPathMaxDepth)
  {
    const float epsilon = (prd.flags & FLAG_HIT) ? theLaunchParameters.sceneEpsilon : 0.0f;

    prd.wo       = -prd.wi;
    prd.distance = RT_DEFAULT_MAX;
    prd.flags    = 0;

    optixTrace(theLaunchParameters.handle,
               prd.pos, prd.wi,
               epsilon, prd.distance, 0.0f,
               OptixVisibilityMask(0xFF),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               TYPE_RAY_RADIANCE, NUM_RAY_TYPES, TYPE_RAY_RADIANCE,
               payload.x, payload.y);

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;

    if (!(prd.flags & FLAG_HIT))
      break;

    if (prd.typeEvent & BSDF_EVENT_DIFFUSE)
    {
      if (hasSpecular)
      {
        // First diffuse-after-specular hit: store and terminate. tp_Y_pre is the
        // light-path throughput arriving at this vertex BEFORE the local BSDF
        // response (matches the Python "photons" stream's stored quantity:
        // photon_power, with the BSDF applied toward the query direction later).
        // Here we bake the BSDF response toward the surface normal's cosine-weighted
        // outgoing hemisphere approximation is NOT needed: store raw pre-BSDF
        // throughput * Lambertian-albedo/pi response using the vertex's own
        // base color, matching the existing renderer's Lambertian diffuse term.
        const float3 tp_Y_pre = prd.throughputPreBsdf;
        const float3 target   = tp_Y_pre * prd.nrcBaseColor * M_1_PIf;

        bdrcPackFeatures(nrc.lightEndpoints, pixelIdx, prd.posLocal, prd.localAabbMin, prd.localAabbMax);
        nrc.lightTargets[3*pixelIdx+0] = target.x;
        nrc.lightTargets[3*pixelIdx+1] = target.y;
        nrc.lightTargets[3*pixelIdx+2] = target.z;
        // Known reference density at this photon's landing position: the SAME
        // constant scalar used by the reference/anchor stream (Task 3) — both
        // streams share one area-uniform measure over the same surface set.
        nrc.light_p_e_area[pixelIdx] = theLaunchParameters.surfaceRef.invTotalArea;
        nrc.lightValid[pixelIdx]     = 1u;
      }
      break; // Terminate here regardless of hasSpecular — a diffuse hit with no
             // prior specular bounce is a dead end for this grammar (matches
             // the Python trace_photons termination clause 3).
    }

    hasSpecular = true;

    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput.x, fmaxf(prd.throughput.y, prd.throughput.z));
      if (probability < rng(prd.seed))
        break;
      prd.throughput /= probability;
    }

    ++depth;
  }
}
```

- [ ] **Step 3: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -40
```
Expected: build FAILS at this point with errors in `bdrc_host.cu` (it still references `BDRC_MAX_LIGHT_DEPTH`-sized allocations, `d_lightEyeTp`, and depth-strided compaction against the old buffer shapes). This is expected — Task 6 fixes `bdrc_host.cu` to match. Do not attempt to fix `bdrc_host.cu` in this task.

- [ ] **Step 4: Commit**

```bash
git add src/cuda/raygen.cu src/cuda/launch_parameters.h
git commit -m "fix: rewrite bdrcLightIntegrator to the ungated single-store L S+ D grammar"
```

---

### Task 6: Update `bdrc_host.cu` for the simplified per-pixel light buffers

**Files:**
- Modify: `src/cuda/cache/bdrc/bdrc_host.cu`

**Interfaces:**
- Consumes: `nrc.lightEndpoints/lightTargets/light_p_e_area/lightValid` now `[numPixels]`-shaped (Task 5)

- [ ] **Step 1: Remove `d_lightEyeTp` and shrink light-buffer sizes in the `BdrcSystem` struct**

In the `BdrcSystem` struct definition (near line 40-44), delete the line:
```cpp
    float*        d_lightEyeTp      = nullptr; // [3 × BDRC_MAX_LIGHT_DEPTH × numPixels]  RGB (compacted to lum by kernel)
```
and change the comment above the block to note the buffers are now per-pixel (no depth stride). Also delete the corresponding `cudaFree(d_lightEyeTp); d_lightEyeTp = nullptr;` line in `freeAllBuffers()`.

Also remove `d_compactLight_throughput` (it existed only to carry the eye-vertex-throughput weight into the mask loss, which per Global Constraints is no longer used) and its `cudaFree` line.

- [ ] **Step 2: Fix `bdrcAllocBuffers()` sizes**

In `bdrcAllocBuffers()` (line 179), remove the `lightSlots` variable and every buffer sized by it — light buffers are now sized by `batchSize` directly, matching `numPixels` padded to 256:
```cpp
    // Light-path buffers — one slot per pixel (no depth striding; see Task 5).
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightEndpoints),        batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightTargets),          batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_light_p_e_area),        batchSize            * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightValid),            batchSize            * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightEndpoints), batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightTargets),   batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightCount),               1u         * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLight_p_e_area), batchSize            * sizeof(float));
```
Remove the `d_oneTargets1` allocation's use of `lightSlots` (it is unused after Task 7/8 remove the sigmoid/MSE label-based training path entirely — if it is still referenced elsewhere at this point, size it to `batchSize` instead of `lightSlots`). Update the zero-init block (`cudaMemset(sys->d_lightValid, 0, lightSlots * ...)` etc.) to use `batchSize` instead of `lightSlots`.

- [ ] **Step 3: Update the compaction call sites to pass `batchSize` instead of `lightSrc = BDRC_MAX_LIGHT_DEPTH * batchSize`**

`device_bdrc_compact_light()` and `device_bdrc_compact_resampled()` (defined at lines 359-397) already treat their sparse input as a flat array of length `batchSize` with no knowledge of depth-striding — no change needed inside those functions. In `bdrcDispatch()`, every call site that currently computes:
```cpp
const uint32_t lightSrc = static_cast<uint32_t>(BDRC_MAX_LIGHT_DEPTH) * batchSize;
```
(there are three: the main dispatch path and the two debug-mode-6 paths) — replace `lightSrc` with `batchSize` directly (delete the `lightSrc`/`lightSlotsDebug` local variables and use `batchSize` in the `device_bdrc_compact_light(...)` calls and the `cudaMemsetAsync(sys->d_lightValid, 0, ... * sizeof(unsigned int), stream)` calls).

Also update `device_bdrc_compact_light(...)` call arguments: the 4th positional buffer argument was `sparseThroughput` (previously `sys->d_lightEyeTp`) used to compute `compactTpLum`. Since that buffer no longer exists, pass `nullptr` for `sparseThroughput`/`compactTpLum` is not possible because `bdrc_compact_light_kernel`'s signature requires those pointers — instead, delete the `compactTpLum`/`sparseThroughput` parameters from `bdrc_compact_light_kernel`, `device_bdrc_compact_light`, and remove `sys->d_compactLight_throughput` from the call. Do the same for `bdrc_compact_resampled_kernel`/`device_bdrc_compact_resampled` and `sys->d_compactResampled_throughput`.

- [ ] **Step 4: Update the Phase-0 (mask training) call site**

In the `bdrcPhase == 0` branch (around line 556-585), the calls to `sys->maskNrc->dreLossAccumGrad(...)` currently pass `sys->d_compactResampled_throughput` / `sys->d_compactLight_throughput` as the `d_throughputWeights` argument — replace both with `nullptr` (per Global Constraints, the mask/density NCE loss is unweighted). Leave the `normCount`/`normCount / 100000` arguments alone — Task 8 fixes those.

- [ ] **Step 5: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -40
```
Expected: build succeeds.

- [ ] **Step 6: Run smoke test**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb -i 300
```
Enable BDRC, debug mode 2 or 4 (light cache view). Expected: no crash; the console `[bdrc] iter=... resampled=... light=... train=...` printf (already present at `bdrc_host.cu:545-547`) should show `light=` counts on the same order as `resampled=` counts (previously, with shadow-ray gating, `light` counts were typically much smaller and highly view-dependent; after this fix they should track the number of eye-terminated pixels less tightly and instead reflect how many light-emitted paths found a diffuse-after-specular hit at all, independent of camera framing).

- [ ] **Step 7: Commit**

```bash
git add src/cuda/cache/bdrc/bdrc_host.cu
git commit -m "fix: adapt bdrc_host.cu compaction and buffer sizing to per-pixel light buffers"
```

---

## Part C: Logistic NCE loss + correct class-balance normalization

### Task 7: Replace the Brier gradient kernel with logistic NCE

**Files:**
- Modify: `src/cuda/cache/bdrc/mask_ngp_cache.cu`

**Interfaces:**
- Produces: same `dreLossAccumGrad`/`dreLossTrainStep` signatures as before (unchanged) — only the analytic gradient formula inside `k_dre_brier_grad` changes. Rename the kernel to `k_dre_logistic_grad` for clarity; update the one call site inside `dreLossAccumGrad`.

- [ ] **Step 1: Replace the kernel**

Replace `k_dre_brier_grad` (`mask_ngp_cache.cu:42-90`) with:
```cpp
// Logistic-NCE gradient kernel — same physical fixed point as the Brier form
// it replaces (u* = p_l_area), but matches the current Python production
// density_loss="logistic": -log(sigma) for positives, -log(1-sigma) for
// references, where sigma = p_l / (p_l + p_e). Logistic damps the gradient
// less aggressively near the saturated caustic peak than Brier does, which
// the Python project measured as a +0.1-0.33 dB gain.
//
// label=0 (reference/eye anchor): loss = -log(p_e / (p_l + p_e))
//   dL/d(p_l) = 1 / (p_l + p_e)
// label=1 (light/positive):       loss = -log(p_l / (p_l + p_e))
//   dL/d(p_l) = -p_e / (p_l * (p_l + p_e))
//
// gradScale = lossScale / (paddedWidth * normBatch), same convention as before.
__global__ void k_dre_logistic_grad(
    const tcnn::network_precision_t* __restrict__ networkOut,
    const float*                     __restrict__ p_e_area,
    const float*                     __restrict__ throughputWeights,
    tcnn::network_precision_t*       __restrict__ dLdOut,
    float*                           __restrict__ lossOut,
    uint32_t batchSize,
    uint32_t paddedWidth,
    int      label,
    float    gradScale)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;

    const uint32_t base = i * paddedWidth;

    const float p_l = networkOut[base];
    const float p_e = p_e_area[i];
    const float tw  = throughputWeights ? throughputWeights[i] : 1.0f;

    if (p_e < 1e-10f) {
        for (uint32_t r = 0u; r < paddedWidth; ++r)
            dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);
        lossOut[i] = 0.0f;
        return;
    }

    const float p_l_safe = fmaxf(p_l, 1e-20f);
    const float denom    = fmaxf(p_l + p_e, 1e-10f);

    float grad, lv;
    if (label == 0) {
        grad = 1.0f / denom;
        lv   = -logf(p_e / denom);
    } else {
        grad = -p_e / (p_l_safe * denom);
        lv   = -logf(p_l_safe / denom);
    }

    grad *= gradScale * tw;

    lossOut[i] = tw * lv;

    dLdOut[base] = static_cast<tcnn::network_precision_t>(grad);
    for (uint32_t r = 1u; r < paddedWidth; ++r)
        dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);
}
```

- [ ] **Step 2: Update the call site in `dreLossAccumGrad`**

Change the kernel launch (`mask_ngp_cache.cu:124`) from `k_dre_brier_grad<<<...>>>` to `k_dre_logistic_grad<<<...>>>`. No other lines in `dreLossAccumGrad`/`dreLossTrainStep`/`optimizerStep` change.

- [ ] **Step 3: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -20
```
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/cuda/cache/bdrc/mask_ngp_cache.cu
git commit -m "fix: replace Brier mask loss with logistic NCE (matches Python production density_loss)"
```

---

### Task 8: Fix the hard-coded `/100000` class-balance normalizer

**Files:**
- Modify: `src/cuda/cache/bdrc/bdrc_host.cu` (Phase-0 mask training block, lines ~537-591)

**Interfaces:**
- Consumes: `resampledCount`, `lightCount` (already computed as host `unsigned int`s at lines 502-505, just above this block)

- [ ] **Step 1: Replace the hard-coded divisor with the true reference-count normalizer**

The Python objective combines the two streams as `alpha * mean(ref_terms over n_eye anchors) + beta * (n_light/n_eye) * mean(pos_terms over n_light positives)`, i.e. **both terms share the same normalizer, `n_eye` (the anchor/reference count)** — the `(n_light/n_eye)` factor is applied as an explicit multiplier on the light term's mean, not folded into a different normalizer per stream. Replace the two `dreLossAccumGrad` calls' `normBatch` arguments so both pass `resampledCount` (the true anchor/reference count this frame, already computed), and apply the `n_light/n_eye` ratio as `gradScale`'s existing `normBatch` denominator does the averaging — since `dreLossAccumGrad`'s `gradScale = lossScale / (paddedWidth * normBatch)` already divides by `normBatch`, passing `resampledCount` to BOTH calls makes each call average over the anchor-stream size, which is exactly `alpha * mean(ref)` and `beta * (n_light/n_eye) * mean(pos)` combined into one shared-denominator sum — this is algebraically the same decomposition Python uses, just computed as one running sum instead of two separately-scaled means:

```cpp
            if (paddedResampledCount > 0u)
            {
                if (paddedResampledCount > resampledCount)
                    cudaMemsetAsync(sys->d_compactResampled_p_e_area + resampledCount, 0,
                                    (paddedResampledCount - resampledCount) * sizeof(float), stream);
                tcnn::GPUMatrixDynamic<float> inputMat(sys->d_compactResampledEndpoints, 3u, paddedResampledCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                eyeLossSum = sys->maskNrc->dreLossAccumGrad(stream, inputMat,
                                               sys->d_compactResampled_p_e_area, 0,
                                               resampledCount > 0u ? resampledCount : 1u,
                                               nullptr,
                                               tcnn::GradientMode::Overwrite);
                didAccumGrad = true;
            }

            if (paddedLightCount > 0u)
            {
                if (paddedLightCount > lightCount)
                    cudaMemsetAsync(sys->d_compactLight_p_e_area + lightCount, 0,
                                    (paddedLightCount - lightCount) * sizeof(float), stream);
                tcnn::GPUMatrixDynamic<float> inputMat(sys->d_compactLightEndpoints, 3u, paddedLightCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                // normBatch = resampledCount (NOT lightCount, NOT a hard-coded constant):
                // the composite loss total = (sum_ref_terms + sum_pos_terms) / n_eye_anchors,
                // matching Python's alpha*mean(ref) + beta*(n_light/n_eye)*mean(pos) decomposition.
                lightLossSum = sys->maskNrc->dreLossAccumGrad(stream, inputMat,
                                               sys->d_compactLight_p_e_area, 1,
                                               resampledCount > 0u ? resampledCount : 1u,
                                               nullptr,
                                               didAccumGrad ? tcnn::GradientMode::Accumulate
                                                            : tcnn::GradientMode::Overwrite);
                didAccumGrad = true;
            }

            if (didAccumGrad)
            {
                sys->maskNrc->optimizerStep(stream);
                sys->m_lastMaskLoss = (eyeLossSum + lightLossSum) /
                                      (float)(resampledCount > 0u ? resampledCount : 1u);
            }
```
Note `normCount` (`resolutionX * resolutionY`) is no longer used in this block — leave its declaration in place only if other code in the same function still reads it; otherwise remove the now-unused local.

- [ ] **Step 2: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -20
```
Expected: build succeeds (watch for an unused-variable warning on `normCount` — remove the declaration if it is otherwise unused).

- [ ] **Step 3: Run smoke test and compare mask loss magnitude to before the fix**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb -i 2000
```
Watch the `[bdrc] iter=... phase=... ` printf's implied loss trend (or add a temporary `printf("mask loss=%f\n", sys->m_lastMaskLoss)` if not already visible) over the first ~2000 iterations. Expected: the loss should decrease and stabilize (not diverge, not immediately saturate at a constant) — a grossly wrong normalizer typically shows up as either an exploding loss or a loss that never moves from its initial value.

- [ ] **Step 4: Commit**

```bash
git add src/cuda/cache/bdrc/bdrc_host.cu
git commit -m "fix: replace hard-coded /100000 mask-loss normalizer with the true anchor-count ratio"
```

---

## Part D: Virtual light emitter — teach the eye cache the `L S⁺ D [any]⁺` family

### Task 9: Add virtual-emission recording buffers

**Files:**
- Modify: `src/cuda/per_ray_data.h`
- Modify: `src/cuda/launch_parameters.h`
- Modify: `src/cuda/cache/bdrc/bdrc_host.h` and `bdrc_host.cu` (new buffer allocation + accessors)

**Interfaces:**
- Produces: `PerRayData::virtEmitDepth` (int); `nrc.virtEmitPos`/`nrc.virtEmitThroughput`/`nrc.virtEmitValid` (`[VIRT_EMIT_MAX_DEPTH * numPixels]`-shaped, depth-strided — a training-time-only path can visit the suppression gate multiple times per suffix, unlike the now-single-store light integrator); `bdrcVirtEmitPos()`/`bdrcVirtEmitThroughput()`/`bdrcVirtEmitValid()` accessors — read by Task 10 (writer) and Task 11 (consumer).

- [ ] **Step 1: Add the per-ray counter**

In `src/cuda/per_ray_data.h`, immediately after the existing `bool eyePathInSpecularChain;` field (line 159):
```cpp
  int  virtEmitDepth;          // Count of virtual-emission vertices recorded this path (caps at VIRT_EMIT_MAX_DEPTH)
```

- [ ] **Step 2: Add the depth constant and buffers to `launch_parameters.h`**

Immediately after `constexpr int BDRC_MAX_LIGHT_DEPTH = 6;` (line 47):
```cpp
// Max virtual-light-emitter recordings per eye path's training suffix. A suffix
// can re-enter a specular chain after its first diffuse vertex multiple times
// (e.g. diffuse -> mirror -> diffuse -> mirror -> diffuse); this caps how many
// of those re-entries contribute a virtual-emission correction.
constexpr int VIRT_EMIT_MAX_DEPTH = 4;
```
In `NrcLaunchBuffers`, add after the `lightOutput` field (from Task 5's Step 1):
```cpp
  // ---- Virtual light emitter buffers — written by hit.cu at every diffuse
  // vertex reached after the eye path's first diffuse vertex, when that vertex
  // is in a specular chain (i.e. exactly where validForEyeCache suppresses
  // direct emission). Depth-strided: slot for depth d, training slot s is
  // d * batchSize + s. Consumed by bdrc_host.cu Phase 2 to compensate the eye
  // cache's training target for the transport family L S+ D [any]+.
  float*        virtEmitPos;          // [3 × VIRT_EMIT_MAX_DEPTH × batchSize] world-space position (AABB-normalized)
  float*        virtEmitThroughput;   // [3 × VIRT_EMIT_MAX_DEPTH × batchSize] throughput * retro-BSDF-eval at that vertex
  unsigned int* virtEmitValid;        // [VIRT_EMIT_MAX_DEPTH × batchSize]     1 if slot is valid
```

- [ ] **Step 3: Declare buffer accessors in `bdrc_host.h`**

After `float* bdrcRenderOutput(const BdrcSystem* sys);` (line 53):
```cpp
float*        bdrcVirtEmitPos          (const BdrcSystem* sys);
float*        bdrcVirtEmitThroughput   (const BdrcSystem* sys);
unsigned int* bdrcVirtEmitValid        (const BdrcSystem* sys);
```

- [ ] **Step 4: Allocate and implement accessors in `bdrc_host.cu`**

In the `BdrcSystem` struct, add:
```cpp
    float*        d_virtEmitPos        = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_virtEmitThroughput = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    unsigned int* d_virtEmitValid      = nullptr; // [VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_compactVirtEmitPos        = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_compactVirtEmitThroughput = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    unsigned int* d_compactVirtEmitCount       = nullptr; // [1]
    float*        d_virtEmitOutput            = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize] lightNrc EMA query result
    float*        d_virtEmitContribSum        = nullptr; // [3 × batchSize] per-training-slot summed compensation
```
Add `cudaFree` lines for all eight in `freeAllBuffers()`. In `bdrcAllocBuffers()`, allocate all eight sized by `VIRT_EMIT_MAX_DEPTH * batchSize` (the `*Sum` buffer is `batchSize`-sized, 3 channels), and zero-init `d_virtEmitValid`/`d_compactVirtEmitCount` the same way `d_lightValid` is zero-inited. Implement the three accessors next to the existing `bdrcRenderOutput` accessor:
```cpp
float*        bdrcVirtEmitPos        (const BdrcSystem* sys) { return sys->d_virtEmitPos;        }
float*        bdrcVirtEmitThroughput (const BdrcSystem* sys) { return sys->d_virtEmitThroughput; }
unsigned int* bdrcVirtEmitValid      (const BdrcSystem* sys) { return sys->d_virtEmitValid;       }
```

- [ ] **Step 5: Wire the three raw buffers into `NrcLaunchBuffers` in `Application.cpp`**

Find where `nrcBuf.lightEndpoints = bdrcLightEndpoints(m_bdrcSystem);`-style wiring happens (search `Application.cpp` for `bdrcLightEndpoints` or `bdrcResampledEndpoints`) and add, in the same place:
```cpp
  nrcBuf.virtEmitPos        = bdrcVirtEmitPos(m_bdrcSystem);
  nrcBuf.virtEmitThroughput = bdrcVirtEmitThroughput(m_bdrcSystem);
  nrcBuf.virtEmitValid      = bdrcVirtEmitValid(m_bdrcSystem);
```

- [ ] **Step 6: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -30
```
Expected: build succeeds. `PerRayData::virtEmitDepth` and the new buffers are declared/allocated but not yet read or written by device code (unused-field warnings are fine here).

- [ ] **Step 7: Commit**

```bash
git add src/cuda/per_ray_data.h src/cuda/launch_parameters.h src/cuda/cache/bdrc/bdrc_host.h src/cuda/cache/bdrc/bdrc_host.cu src/Application.cpp
git commit -m "feat: add virtual-light-emitter recording buffers (unwired)"
```

---

### Task 10: Record suppressed `D S⁺ L` vertices in `hit.cu`

**Files:**
- Modify: `src/cuda/hit.cu` (the `validForEyeCache` gate, lines 1527-1545)
- Modify: `src/cuda/raygen.cu` (initialize `prd.virtEmitDepth = 0` alongside the other eye-path PRD init fields)

**Interfaces:**
- Consumes: `PerRayData::virtEmitDepth` (Task 9), `nrc.virtEmitPos/virtEmitThroughput/virtEmitValid` (Task 9)

- [ ] **Step 1: Initialize the counter**

In `src/cuda/raygen.cu`, find the eye-path PRD initialization that sets `prd.eyePathFoundDiffuse = false; prd.eyePathInSpecularChain = false;` (line 489-490, inside `integrator()`) and add immediately after:
```cpp
  prd.virtEmitDepth = 0;
```

- [ ] **Step 2: Record at the suppression gate**

In `src/cuda/hit.cu`, immediately after the `validForEyeCache` computation (lines 1527-1532), add:
```cpp
  // Virtual light emitter: when direct emission is suppressed here to avoid
  // double-counting the caustic already captured by the light cache (see
  // validForEyeCache above), record this vertex so the host can later query
  // the frozen (EMA) light-color cache here and add the missing radiance into
  // the EYE CACHE'S TRAINING TARGET ONLY (never into the live rendered frame —
  // the eye cache is queried once, at its own first-diffuse vertex, at render
  // time, so compensating its training target is sufficient and correct).
  if (!validForEyeCache && theLaunchParameters.nrc.enabled
      && theLaunchParameters.nrc.nrcType == BDRC
      && thePrd->virtEmitDepth < VIRT_EMIT_MAX_DEPTH)
  {
    const unsigned int pixelIdx = theLaunchParameters.resolution.x * optixGetLaunchIndex().y
                                 + optixGetLaunchIndex().x;
    const unsigned int numPix   = theLaunchParameters.resolution.x * theLaunchParameters.resolution.y;
    const unsigned int slot     = static_cast<unsigned int>(thePrd->virtEmitDepth) * numPix + pixelIdx;
    const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

    bdrcPackFeatures(nrc.virtEmitPos, slot, thePrd->posLocal, thePrd->localAabbMin, thePrd->localAabbMax);

    // Retro-BSDF response toward the incoming eye-path direction, times the
    // throughput arriving at this vertex (pre-local-BSDF), so multiplying by
    // the light cache's stored radiance at this point gives this vertex's
    // contribution to the pixel's accumulated radiance in the same units as
    // every other term already summed into prd.radiance.
    const float3 f_retro = state.baseColor * (fmaxf(dot(state.N, thePrd->wo), 0.0f) * M_1_PIf);
    const float3 w       = thePrd->throughputPreBsdf * f_retro;
    nrc.virtEmitThroughput[3*slot+0] = w.x;
    nrc.virtEmitThroughput[3*slot+1] = w.y;
    nrc.virtEmitThroughput[3*slot+2] = w.z;
    nrc.virtEmitValid[slot] = 1u;

    thePrd->virtEmitDepth += 1;
  }
```
Place this block textually right after the existing `const bool validForEyeCache = ...;` statement and before the `if (validForEyeCache) { ... }` block that follows it (so it does not change any existing control flow — it only adds a recording side-effect in the complementary case). `state.baseColor`/`state.N` are already in scope at this point in `hit.cu` (used a few lines below for `nrcBaseColor`/`nrcN` capture).

- [ ] **Step 3: Add the `#include` for `VIRT_EMIT_MAX_DEPTH` if needed**

`VIRT_EMIT_MAX_DEPTH` is defined in `launch_parameters.h`, which `hit.cu` already includes (it uses `NrcLaunchBuffers`/`LaunchParameters` throughout) — no new include should be needed. Verify by building.

- [ ] **Step 4: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail-30
```
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/hit.cu src/cuda/raygen.cu
git commit -m "feat: record suppressed D S+ L vertices for virtual-light-emitter compensation"
```

---

### Task 11: Compact, query EMA light cache, and scatter-add into the eye training target

**Files:**
- Modify: `src/cuda/cache/bdrc/bdrc_host.cu` (Phase-2 dispatch, and a new small compaction+scatter kernel)

**Interfaces:**
- Consumes: `nrc.virtEmitPos/virtEmitThroughput/virtEmitValid` (Task 9/10), `sys->lightNrc` (existing), `sys->d_trainSuffixScaled` / `sys->d_trainPixelIndices` (existing eye-training-slot buffers from `NrcHost`-era buffers now owned by `BdrcSystem` — confirm the exact member name in `bdrc_host.cu`'s `BdrcSystem` struct before editing; it is the buffer written by `handleBdrcCacheLogic`'s `nrc.trainSuffixScaled`/`trainPixelIndices`, exposed by the existing `bdrcTrainSuffixScaled`/accessor pattern)

- [ ] **Step 1: Add a compaction + host-side query + scatter-add helper**

Add a new local kernel near the existing `bdrc_compact_light_kernel` (after it, before `device_bdrc_compact_light`):
```cpp
// Compact the depth-strided virtual-emission buffer into a dense array,
// remembering which (pixel, training-slot) each compacted entry maps back to
// so the scatter-add step can find the right eye-training-target slot.
static __global__ void bdrc_compact_virt_emit_kernel(
    float*              compactPos,          // [3 × cap] AoS output
    float*              compactWeight,       // [3 × cap] AoS output
    unsigned int*        compactPixelIdx,    // [cap] output — pixel index this entry belongs to
    unsigned int*       compactCount,        // [1] must be 0 before launch
    const float*        sparsePos,           // [3 × VIRT_EMIT_MAX_DEPTH × numPixels] input
    const float*        sparseWeight,        // [3 × VIRT_EMIT_MAX_DEPTH × numPixels] input
    const unsigned int* sparseValid,         // [VIRT_EMIT_MAX_DEPTH × numPixels] input
    uint32_t             numPixels,
    uint32_t             totalSlots)         // VIRT_EMIT_MAX_DEPTH * numPixels
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalSlots || !sparseValid[i]) return;
    const uint32_t slot = atomicAdd(compactCount, 1u);
    compactPos[3u*slot+0u]    = sparsePos[3u*i+0u];
    compactPos[3u*slot+1u]    = sparsePos[3u*i+1u];
    compactPos[3u*slot+2u]    = sparsePos[3u*i+2u];
    compactWeight[3u*slot+0u] = sparseWeight[3u*i+0u];
    compactWeight[3u*slot+1u] = sparseWeight[3u*i+1u];
    compactWeight[3u*slot+2u] = sparseWeight[3u*i+2u];
    compactPixelIdx[slot]     = i % numPixels; // depth-strided: slot d*numPixels+p -> pixel p
}

// Scatter compensation = lightOutput * weight into the eye training target for
// the pixel each compacted entry belongs to. Multiple virtual-emission entries
// can belong to the same pixel (one per re-entered specular chain along the
// suffix), so this uses atomicAdd.
static __global__ void bdrc_add_virt_emission_kernel(
    float*              trainSuffixScaled,     // [3 × batchSize] eye training target, slot-indexed
    const unsigned int* trainPixelIndices,     // [batchSize] pixel index for each training slot
    const unsigned int* pixelToTrainSlot,      // [numPixels] inverse map, 0xFFFFFFFF if none
    const float*        compactWeight,         // [3 × compactCount]
    const float*        lightOutput,           // [3 × compactCount] EMA lightNrc query result
    const unsigned int* compactPixelIdx,       // [compactCount]
    unsigned int         compactCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= compactCount) return;
    const unsigned int px = compactPixelIdx[i];
    const unsigned int trainSlot = pixelToTrainSlot[px];
    if (trainSlot == 0xFFFFFFFFu) return;

    for (int c = 0; c < 3; ++c)
    {
        const float contrib = compactWeight[3u*i + c] * lightOutput[3u*i + c];
        atomicAdd(&trainSuffixScaled[3u*trainSlot + c], contrib);
    }
}
```

- [ ] **Step 2: Build the pixel-to-training-slot inverse map**

The existing `nrc.trainPixelIndices`/`trainCount` (written by `handleBdrcCacheLogic`) map training slot → pixel, but the scatter-add kernel needs the inverse (pixel → training slot). Add one more small kernel next to the two above:
```cpp
static __global__ void bdrc_build_pixel_to_train_slot_kernel(
    unsigned int*       pixelToTrainSlot,   // [numPixels] output, init to 0xFFFFFFFF before launch
    const unsigned int* trainPixelIndices,  // [trainCount] input
    unsigned int        trainCount)
{
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= trainCount) return;
    pixelToTrainSlot[trainPixelIndices[s]] = s;
}
```
Add a device buffer `unsigned int* d_pixelToTrainSlot = nullptr;` to `BdrcSystem` (allocated at `numPixels` size in `bdrcAllocBuffers`), freed in `freeAllBuffers()`.

- [ ] **Step 3: Call the three kernels in Phase 2, before the eye-cache `trainStep`**

In `bdrcDispatch()`'s `else` branch (Phase 2, after Step 1 "Query maskNrc" and before Step 6/7 "Build eye training targets" / "Train eye cache" — insert this immediately before the block that computes `d_trainTargets`/calls `eyeNrc->trainStep`):
```cpp
        // ---- Virtual light emitter: compensate the eye training target for
        // ---- the L S+ D [any]+ family that validForEyeCache suppressed.
        {
            const uint32_t totalVirtSlots = static_cast<uint32_t>(VIRT_EMIT_MAX_DEPTH) * (uint32_t)numPixels;
            cudaMemsetAsync(sys->d_compactVirtEmitCount, 0, sizeof(unsigned int), stream);
            constexpr uint32_t THREADS = 256u;
            uint32_t blocks = (totalVirtSlots + THREADS - 1u) / THREADS;
            bdrc_compact_virt_emit_kernel<<<blocks, THREADS, 0, stream>>>(
                sys->d_compactVirtEmitPos, sys->d_compactVirtEmitThroughput,
                reinterpret_cast<unsigned int*>(sys->d_virtEmitOutput), // reuse as scratch pixel-index buffer? NO:
                sys->d_compactVirtEmitCount,
                sys->d_virtEmitPos, sys->d_virtEmitThroughput, sys->d_virtEmitValid,
                (uint32_t)numPixels, totalVirtSlots);

            cudaStreamSynchronize(stream);
            unsigned int virtCount = 0;
            cudaMemcpy(&virtCount, sys->d_compactVirtEmitCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            if (virtCount > 0u)
            {
                tcnn::GPUMatrixDynamic<float> inp(sys->d_compactVirtEmitPos, 3u, virtCount,
                                                  tcnn::MatrixLayout::ColumnMajor);
                tcnn::GPUMatrix<float> out(sys->d_virtEmitOutput, 3u, virtCount);
                sys->lightNrc->query(stream, inp, out, /*useEmaWeights=*/true);

                cudaMemsetAsync(sys->d_pixelToTrainSlot, 0xFF, (uint32_t)numPixels * sizeof(unsigned int), stream);
                blocks = (trainCount + THREADS - 1u) / THREADS;
                if (trainCount > 0u)
                    bdrc_build_pixel_to_train_slot_kernel<<<blocks, THREADS, 0, stream>>>(
                        sys->d_pixelToTrainSlot, d_trainFeatures ? nullptr : nullptr, 0); // see note below

                blocks = (virtCount + THREADS - 1u) / THREADS;
                bdrc_add_virt_emission_kernel<<<blocks, THREADS, 0, stream>>>(
                    sys->d_trainSuffixScaled, /*trainPixelIndices*/ nullptr, sys->d_pixelToTrainSlot,
                    sys->d_compactVirtEmitThroughput, sys->d_virtEmitOutput,
                    /*compactPixelIdx*/ nullptr, virtCount);
            }
        }
```
**Before implementing Step 3's kernel-launch arguments verbatim**, locate the actual device buffer in `BdrcSystem`/`NrcLaunchBuffers` that holds `trainPixelIndices` (it is written by `handleBdrcCacheLogic` in `raygen.cu` as `nrc.trainPixelIndices[slot] = pixelIdx;`, per `raygen.cu:244`) and wire its true pointer (likely `sys->d_trainPixelIndices` or an equivalent already-existing member — check `bdrc_host.cu`'s `BdrcSystem` struct near where `d_trainTargets` is declared) into both the `bdrc_build_pixel_to_train_slot_kernel` call (as `trainPixelIndices`) and note it in the `bdrc_add_virt_emission_kernel` call's now-unused second parameter (that parameter can be removed from the kernel signature since the inverse map makes it redundant — simplify `bdrc_add_virt_emission_kernel` to drop its unused `trainPixelIndices` parameter). Also fix the placeholder `reinterpret_cast<unsigned int*>(sys->d_virtEmitOutput)` used above for `compactPixelIdx` in the compaction call — it must be a real `unsigned int*` scratch buffer sized `[VIRT_EMIT_MAX_DEPTH * numPixels]`; add `unsigned int* d_virtEmitPixelIdx = nullptr;` to `BdrcSystem`, allocate it in `bdrcAllocBuffers` at that size, and pass it there instead.

- [ ] **Step 4: Build to verify**

```bash
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | tail -50
```
Expected: build succeeds once Step 3's buffer wiring is corrected against the real `BdrcSystem` field names (this is the one step in this plan where the exact identifier must be confirmed against the live file rather than assumed, since `trainPixelIndices`'s owning buffer name was not directly read during planning — grep `bdrc_host.cu`/`bdrc_host.h` for `trainPixelIndices` or `TrainPixelIndices` before writing this step's final code).

- [ ] **Step 5: Run smoke test on the harder SDS scene**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere_sds.glb -i 3000
```
Let it run past the mask-warmup phase into Phase 2. Expected: no crash, no NaN in the accumulated image (visually check for black/white speckle that wasn't there before this task), and the rendered image in regions with a caustic reflected off a second diffuse surface should look brighter/more physically plausible than before this task (this is the transport family being added) rather than darker — if it goes darker or noisier, the retro-BSDF or throughput weighting in Task 10 Step 2 is very likely miscomputed and should be re-derived rather than shipped.

- [ ] **Step 6: Commit**

```bash
git add src/cuda/cache/bdrc/bdrc_host.cu
git commit -m "feat: fold EMA light-cache virtual emission into eye-cache training target"
```

---

## Self-Review Checklist

### Spec coverage

| Gap identified in the analysis | Task(s) |
|---|---|
| 1. Light-path grammar is the project's own deprecated shadow-ray-gated design | Tasks 5, 6 |
| 2. Reference distribution is a camera-entangled hemisphere probe | Tasks 1, 2, 3, 4 |
| 3. Mask loss is Brier instead of logistic NCE | Task 7 |
| 4. Hard-coded `/100000` class-balance normalizer | Task 8 |
| 5. No virtual-light-emitter fold-in for the eye cache | Tasks 9, 10, 11 |

### Placeholder scan

Task 11 Step 3 intentionally contains one explicit "confirm against the live file" instruction rather than a fabricated identifier, because the exact `BdrcSystem` member backing `trainPixelIndices` was not read during planning (everything else in this plan was verified directly against the current source). This is flagged, not silently guessed, and Step 4 makes the build gate on it being resolved correctly — it is the one piece of this plan an executor must look up rather than transcribe.

### Type consistency

- `SurfaceReferenceTable` (Task 2) is used identically in `launch_parameters.h` and `surface_reference.h`.
- `sampleSurfaceUniform(unsigned int&, const SurfaceReferenceTable&, float3&, float3&)` — declared and defined together in Task 2, called in Task 3 with matching argument order.
- `bdrcVirtEmitPos`/`bdrcVirtEmitThroughput`/`bdrcVirtEmitValid` — declared Task 9 Step 3, implemented Task 9 Step 4, wired Task 9 Step 5, written Task 10, consumed Task 11.
- `VIRT_EMIT_MAX_DEPTH` — defined once in `launch_parameters.h` (Task 9), used in `hit.cu` (Task 10) and `bdrc_host.cu` (Task 11).
- `k_dre_logistic_grad` replaces `k_dre_brier_grad` with the same call signature (Task 7) — no other file references the old kernel name directly (it was `static`/file-local).

### Known simplifications (deliberate, not gaps)

- The reference/anchor distribution ported here is the **pure area-uniform** case only (`a_eff = 0` in the Python project's terms) — no shape-balance or NDC-projected-area mixture. This matches Python's own documented default and is the correct starting point; the mixture refinements are a separate, later enhancement if profiling shows a specific object is starved of anchor density.
- The light-path integrator's stored target uses a Lambertian `albedo/π` retro-BSDF term (Task 5) rather than querying the true material response toward an arbitrary later query direction — this mirrors the renderer's existing Lambertian-only assumption in the pre-fix shadow-ray target computation (`f_Y_toward_S` in the removed code) and is not a regression.
- Task 11's virtual-emission compensation affects only the eye cache's **training target**, never the immediately rendered frame — this is intentional (see Task 10 Step 2's comment) and matches how the Python project's equivalent mechanism works: the eye cache is queried once per pixel at render time, so whatever it was trained to predict is what render time uses.
