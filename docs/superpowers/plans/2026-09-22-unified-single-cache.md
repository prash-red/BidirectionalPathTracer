# Unified Single Radiance Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BDRC's three-network factorized system (`eyeNrc`, `lightNrc`, `maskNrc`) with one neural field holding incident flux per unit world area, trained by a single weighted density-ratio objective over a photon stream and a gather stream.

**Architecture:** One `unified_ngp_cache` (5-dim input: 3 AABB-normalized world position dims through a hash grid + 2 label dims through Identity; 3-channel softplus output). Two supervision streams write into one fixed-slot records array (lanes `[0,photonCount)` photons, `[photonCount, photonCount+gatherCount)` gathers). Per frame: trace photons, trace gathers, pack tail queries, EMA-inference the tails, assemble per-lane `(weight, p_r, 1/N_pos, 1/N_ref)` records, run one DRE loss kernel over the whole batch, one `backward`, one optimizer step, one EMA update. The camera stream is separate: it traces a prefix, stops at the first diffuse vertex `D`, and the host composites `prefix + T(cam→D)·f_s(D)·U(D)`.

**Tech Stack:** CUDA 12 / OptiX 9 / tiny-cuda-nn (FullyFusedMLP + HashGrid + Composite encoding), C++17, Dear ImGui.

**Spec:** `docs/superpowers/specs/2026-09-22-unified-single-cache-design.md`

**Reference implementation (read-only, another repo):**
- `/home/prashanth/projects/bidirectional-radiance-caching/third_party/neural-cache-demo/src/renderer/shaders/transport.wgsl`
- `/home/prashanth/projects/bidirectional-radiance-caching/third_party/neural-cache-demo/src/cache/shaders/dre_loss.wgsl`
- `/home/prashanth/projects/bidirectional-radiance-caching/third_party/neural-cache-demo/src/cache/shaders/activation.wgsl`
- `/home/prashanth/projects/bidirectional-radiance-caching/third_party/neural-cache-demo/src/renderer/cache_render.ts` (`trainStep` dispatch order)

## Global Constraints

- **No unit-test framework may be added.** Every gate is a build plus a runtime observable (console print or GUI readout). This is the spec's explicit instruction and it overrides the TDD default in the executing skill.
- **Build:** `export CUDACXX=/usr/local/cuda/bin/nvcc && cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)`
- **Smoke scenes:** `data/cornell_caustic_glass_sphere.glb`, `data/cornell_caustic_glass_sphere_sds.glb`
- **The objective stays plain.** No self-normalized weights, no learned global scale, no proxy mixture, no detached reweighting (spec contract 1).
- **Raw-ratio members require `alpha == beta`** — `lsif` and `bregman` must throw if `alpha != beta` (spec contract 2).
- **Each stream is normalized by its own launch count**, not the live count of valid samples (spec contract 4). Invalid lanes keep their denominator and contribute zero weight.
- **Reference density is per unit world surface area, halved for the two-sided field**: `p_r = 0.5 * surfaceRef.invTotalArea` at anchors and deposits alike (spec contract 5).
- **EMA teacher, live student.** Tail reads and all render/debug queries use EMA weights; only the fit touches live weights (spec contract 6).
- **The stored quantity is incident flux per area.** The receiver BSDF is applied at the query, never baked into the stored field (spec contract 7).
- **The cache lives on diffuse lobes only** (spec contract 8).
- **Training is camera-independent.** Moving the viewing camera changes rays and film accumulation only (spec contract 9).
- Copyright header style and `dev::` namespace conventions from `CLAUDE.md` apply to all new files.

## Decisions this plan takes that the spec left open

1. **Coordinates move from GAS-local to world space.** The spec's Decisions table says "Keep AABB-normalized **world** position", but the tree currently packs BDRC features in per-mesh **GAS-local** space normalized by each mesh's **own** local AABB (commit `c9a4483`). That is wrong for a single shared spatial field: two different meshes collapse onto the same unit cube, and two instances of one mesh land on identical field coordinates. Task 1 moves every BDRC position to world space normalized by the scene AABB (`nrc.aabbMin/aabbMax`, already wired). This is what the spec asks for and it is required for the gather stream (whose anchors must be shadow-ray-traceable, i.e. world space) to share a domain with the camera and photon streams.
2. **The fixed reference origin is the scene bounding-sphere centre**, snapshotted when the BDRC system is created and stored in `LaunchParameters.nrc.referenceOrigin`. It is never updated on camera motion. The demo uses its authored camera; this port has none, and a deterministic geometric point satisfies the spec's requirement ("captured once when the cache is constructed and stamped with it; it is not the live viewing camera") without introducing camera coupling at all.
3. **Photon lanes = `numPixels`; gather lanes are independently configurable.** The existing `__raygen__light_path_tracer` already launches at render resolution, so photons keep that launch shape. The gather raygen gets its own SBT and launches at `(gatherCount, 1, 1)`, which makes the gather count independent of window size — required by the spec's stream-count gate ("the gather count is independent of camera framing").
4. **No compaction.** The old design's atomic scatter-to-compact machinery is removed. Every lane owns a fixed slot; invalid lanes write `density = 0` and the loss kernel skips them. This matches the demo (`assemble_targets` returns early on `!supported(point) || base.w <= 0`) and satisfies contract 4 for free.
5. **Debug modes 5 and 6 (endpoint splatting) are removed** along with the rest of the old system. The spec's verification needs a cache-only view (kept as debug mode 1) and numeric stream/density readouts (added to the GUI), not endpoint splats.
6. **`tcnn` output activation is `None`; softplus is applied by our own code.** The loss kernel differentiates through softplus analytically (as the demo's `field_log_density` does), so the network must hand us raw logits. `unified_ngp_cache::query()` applies softplus after inference so every consumer reads `u` directly.

## File Structure

**Created**
| File | Responsibility |
|---|---|
| `src/cuda/cache/bdrc/unified_ngp_cache.h` | Public interface of the one field: config, query (softplus applied), weighted-DRE accumulate-gradient, optimizer step, self-test. |
| `src/cuda/cache/bdrc/unified_ngp_cache.cu` | Network config (composite encoding), the generator-family DRE loss kernel, the softplus post-query kernel, the analytic self-test. |
| `src/cuda/cache/bdrc/unified_features.h` | `bdrcPackQuery5()` — writes `[pos.x, pos.y, pos.z, signCam, side]` per lane; `bdrcSignCamLabel()`; `bdrcSideLabel()`. |

**Deleted**
| File | Reason |
|---|---|
| `src/cuda/cache/bdrc/mask_ngp_cache.h` | Superseded by `unified_ngp_cache`. |
| `src/cuda/cache/bdrc/mask_ngp_cache.cu` | Superseded by `unified_ngp_cache`. |

**Modified**
| File | Change |
|---|---|
| `src/cuda/launch_parameters.h` | `SurfaceReferenceTable` gains world positions; `NrcLaunchBuffers` BDRC section replaced by unified records/camera buffers + generator config + `referenceOrigin`. |
| `src/cuda/surface_reference.h` | `sampleSurfaceUniform` returns world position and world face normal. |
| `src/cuda/cache/nrc_features.h` | `bdrcPackFeatures` normalizes world position against the scene AABB. |
| `src/cuda/cache/bdrc/ngp_config.h` | `labelDims` field; `toEncodingJson()` emits a `Composite` when `labelDims > 0`; `weightClip` carried over. |
| `src/cuda/per_ray_data.h` | Remove `bsdfEvalTowardCamera`, `virtEmitDepth`, `eyePathFoundDiffuse`, `eyePathInSpecularChain`, `FLAG_BDRC_LIGHT_PATH`; add `bdrcFrontFace`, `bdrcNg`. |
| `src/cuda/hit.cu` | Remove the virtual-emission block, the `validForEyeCache` suppression, and the `bsdfEvalTowardCamera` block; stash `isFrontFace` and the raw `state.Ng`. |
| `src/cuda/raygen.cu` | Camera stream rewrite; photon stream rewrite; new gather raygen. |
| `src/cuda/cache/bdrc/bdrc_host.h` / `.cu` | One cache, one dispatch, no phases, no compaction, no virtual emission. |
| `src/cuda/cache/kernel_nrc.cu`, `src/HostKernels.h` | `device_bdrc_composite` takes `(prefix, factor, fieldOutput, valid)`; `device_bdrc_debug` shows the single field; splat kernels deleted. |
| `src/Application.h` / `.cpp` | Gather SBT + program group + launch; remove phase/warmup; unified GUI; `referenceOrigin` stamping; world-space surface table. |
| `CMakeLists.txt` | Swap `mask_ngp_cache` for `unified_ngp_cache`. |

## Review Focus

Input classes the spec implies that no task's happy path exercises. Each has a gate in the task that owns the code.

1. **Scene with zero lights or an empty surface-reference table.** `invTotalArea` is 0 and `sampleSurfaceUniform` returns false; every gather lane and photon lane must write `density = 0` and the fit must take a step that changes nothing rather than emitting NaN. — Task 5, Task 6.
2. **Degenerate scene AABB (a planar scene, zero extent on one axis).** World normalization divides by the axis range; a zero range must fall back to 1.0 rather than produce inf/NaN positions that poison every hash-grid cell. — Task 1.
3. **`u → 0` in the loss (an untrained cell whose logit underflows).** `log u` must stay finite; the demo's `field_log_density` low branch (`x < -20`) exists for exactly this. Without it the first frames emit `-inf` gradients. — Task 3.
4. **`alpha != beta` with a raw-ratio generator.** `lsif`/`bregman` have a fixed point that moves with `beta/alpha`; the spec forbids the combination. The constructor must reject it loudly instead of silently fitting the wrong scale. — Task 3.
5. **Camera motion mid-fit.** Every training coordinate, label, reference density and weight must be byte-identical before and after an orbit. The one live-camera read left in the old code (`bsdfEvalTowardCamera` used `theLaunchParameters.cameraP`) is removed in Task 4; Task 8 gates it at runtime. — Task 4, Task 8.

---

## Task 1: World-space BDRC coordinates and a clean tree

Stash the in-flight work and move every BDRC position from per-mesh GAS-local space to world space normalized by the scene AABB. The three-cache system keeps running throughout — this task changes only the coordinate domain.

**Files:**
- Modify: `src/cuda/cache/nrc_features.h` (`bdrcPackFeatures`)
- Modify: `src/cuda/launch_parameters.h` (`SurfaceReferenceTable`)
- Modify: `src/cuda/surface_reference.h` (`sampleSurfaceUniform`)
- Modify: `src/Application.cpp` (`initSurfaceReferenceTable`, `freeSurfaceReferenceTable`, launch-parameter wiring)
- Modify: `src/Application.h` (surface-table members)
- Modify: `src/cuda/raygen.cu` (`sampleSurfaceUniform` call site, `bdrcPackFeatures` call sites)
- Modify: `src/cuda/hit.cu` (`bdrcPackFeatures` call site in the virtual-emission block)

**Interfaces:**
- Produces: `bdrcPackFeatures(float* buf, uint32_t lane, float3 worldPos, float3 aabbMin, float3 aabbMax)` — same name, world position, scene AABB.
- Produces: `bool sampleSurfaceUniform(uint& seed, const SurfaceReferenceTable& tbl, float3& outWorldPos, float3& outWorldNormal)`.
- Produces: `SurfaceReferenceTable { float3* positions; float* cdf; unsigned int triCount; float invTotalArea; }` — world-space verts, no local-AABB arrays.

- [ ] **Step 1: Stash the in-flight work**

```bash
cd /home/prashanth/projects/BidirectionalPathTracer
git status
git stash push -u -m "pre-unified-cache: bsdfEvalTowardCamera + ngp tuning + diagnostics"
git status   # expect: clean apart from .idea/ .worktrees/ docs/
```

The stash is recoverable (`git stash list`). The `bsdfEvalTowardCamera` work in it is dropped deliberately (spec: "The uncommitted `bsdfEvalTowardCamera` work … is dropped"). The `weightClip` guard from that stash is re-added with full rationale in Task 3.

- [ ] **Step 2: Make `bdrcPackFeatures` world-space and division-safe**

Replace the body of `bdrcPackFeatures` in `src/cuda/cache/nrc_features.h`:

```c
// Packs one 3-dim BDRC position query: world position normalized into the
// SCENE AABB, clamped to [0,1].
//
// World space, not GAS-local: one neural field spans the whole scene, so two
// meshes must not share a coordinate. Normalizing each mesh by its own local
// AABB (the pre-2026-09 convention) collapsed every mesh onto the same unit
// cube and made two instances of one mesh indistinguishable to the field.
//
// Layout: element (feature f, lane i) at buf[i * BDRC_INPUT_DIM + f], i.e. a
// column-major [BDRC_INPUT_DIM x N] tcnn matrix.
__device__ __forceinline__ void bdrcPackFeatures(
    float*       buf,
    uint32_t     lane,
    const float3 worldPos,
    const float3 aabbMin,
    const float3 aabbMax)
{
    float* d = buf + lane * BDRC_INPUT_DIM;

    // A planar scene has zero extent on one axis; fall back to 1.0 so the
    // coordinate pins to 0 instead of producing inf/NaN.
    const float rx = aabbMax.x - aabbMin.x;
    const float ry = aabbMax.y - aabbMin.y;
    const float rz = aabbMax.z - aabbMin.z;
    d[0] = fminf(fmaxf((worldPos.x - aabbMin.x) / (rx > 0.f ? rx : 1.f), 0.f), 1.f);
    d[1] = fminf(fmaxf((worldPos.y - aabbMin.y) / (ry > 0.f ? ry : 1.f), 0.f), 1.f);
    d[2] = fminf(fmaxf((worldPos.z - aabbMin.z) / (rz > 0.f ? rz : 1.f), 0.f), 1.f);
}
```

- [ ] **Step 3: Make the surface reference table world-space**

In `src/cuda/launch_parameters.h` replace the `SurfaceReferenceTable` struct:

```c
// Global diffuse-surface reference table for the density-ratio reference
// measure. Density at ANY point on this surface set is the single scalar
// invTotalArea (pure area-uniform reference). Built once at scene load; never
// depends on the camera or on any path.
//
// Positions are WORLD space. The gather stream traces shadow rays and
// continuation rays from these anchors, which is only meaningful in world
// space, and the field's own coordinate is world-normalized too (see
// bdrcPackFeatures). World verts go stale under rigid-body node animation;
// the table is rebuilt on scene reload, and animated-instance support would
// need a per-frame rebuild.
struct SurfaceReferenceTable
{
  float3*      positions;    // [3 * triCount] world-space triangle verts (v0,v1,v2 per tri)
  float*       cdf;          // [triCount + 1] normalized cumulative-area CDF
  unsigned int triCount;
  float        invTotalArea; // 1 / total surface area — the reference density
};
```

Delete `BDRC_MAX_LIGHT_DEPTH` and `VIRT_EMIT_MAX_DEPTH` from the same header — both belong to the removed design. (`VIRT_EMIT_MAX_DEPTH` is still referenced by `bdrc_host.cu` and `hit.cu` at this point; those references go away in Tasks 4 and 6. To keep the build green *now*, leave `VIRT_EMIT_MAX_DEPTH` in place and delete it in Task 6.)

- [ ] **Step 4: Update `sampleSurfaceUniform`**

Replace the function in `src/cuda/surface_reference.h`:

```c
// Exact area-uniform sample of the global diffuse-surface reference table.
// The returned point's per-unit-area density is tbl.invTotalArea everywhere,
// so no per-point density lookup is needed. Returns false (nothing written)
// when the table is empty — callers must check.
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

  // Uniform barycentric sample (Turk 1990).
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
    return false;                 // degenerate triangle: no usable normal
  outWorldNormal = cr / len;
  return true;
}
```

- [ ] **Step 5: Build the table in world space**

In `src/Application.cpp::initSurfaceReferenceTable()`, drop the local-AABB vectors and push the **world** verts that are already computed for the area:

- Delete `std::vector<float3> triLocalAabbMin, triLocalAabbMax;` and both `push_back` calls.
- Replace the three `localPositions.push_back(posData[iN]);` calls with `worldPositions.push_back(wN);` (rename the vector to `worldPositions`).
- Delete the `m_surfaceRefLocalAabbMin` / `m_surfaceRefLocalAabbMax` `cudaMalloc`/`cudaMemcpy` block.
- In `freeSurfaceReferenceTable()` delete the two corresponding `cudaFree` calls.
- In `src/Application.h` delete the `m_surfaceRefLocalAabbMin` and `m_surfaceRefLocalAabbMax` members.
- In both launch-parameter wiring sites (`initLaunchParameters`-side, ~line 7302, and `updateLaunchParameters`, ~line 7332) delete the two `surfaceRef.localAabbMin/Max` assignments.

- [ ] **Step 6: Update every call site**

In `src/cuda/raygen.cu`, the reference-anchor block inside `integrator()`:

```c
      {
        float3 refPos, refNormal;
        if (sampleSurfaceUniform(prd.seed, theLaunchParameters.surfaceRef, refPos, refNormal))
        {
          bdrcPackFeatures(nrc.resampledEndpoints, px, refPos,
                           theLaunchParameters.nrc.aabbMin,
                           theLaunchParameters.nrc.aabbMax);
          nrc.resampledValid[px]     = 1u;
          nrc.resampled_p_sa[px]     = 0.0f;
          nrc.resampled_p_e_area[px] = theLaunchParameters.surfaceRef.invTotalArea;

          nrc.eyeVertexNormals[3*px+0] = refNormal.x;
          nrc.eyeVertexNormals[3*px+1] = refNormal.y;
          nrc.eyeVertexNormals[3*px+2] = refNormal.z;
        }
      }
```

Every other `bdrcPackFeatures(buf, idx, prd.posLocal, prd.localAabbMin, prd.localAabbMax)` becomes `bdrcPackFeatures(buf, idx, prd.pos, theLaunchParameters.nrc.aabbMin, theLaunchParameters.nrc.aabbMax)`. Find them with:

```bash
grep -rn "bdrcPackFeatures" src/cuda/
```

Expect call sites in `raygen.cu` (`handleBdrcCacheLogic` ×2, `handleBdrcDebugFirstHit`, the reference-anchor block, `bdrcLightIntegrator`) and `hit.cu` (the virtual-emission block).

- [ ] **Step 7: Build**

```bash
cd /home/prashanth/projects/BidirectionalPathTracer
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)
```

Expected: links cleanly, no warnings about the deleted members.

- [ ] **Step 8: Runtime gate — the old system still renders**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb
```

In the GUI set the cache mode to BDRC and un-freeze training. Expected: the image is not black and the caustic under the sphere is still visible. The three-cache system's quality may shift (the coordinate domain changed) but it must still converge to a recognizable render. This is the last frame in which the old design is expected to work.

- [ ] **Step 9: Commit**

```bash
git add -A src/ 
git commit -m "refactor: move BDRC cache coordinates from GAS-local to world space

One neural field spans the whole scene, so per-mesh local-AABB
normalization collapsed distinct meshes onto the same unit cube and made
instances of one mesh indistinguishable. The surface reference table now
stores world triangles, which the gather stream (Task 4) also needs for
shadow and continuation rays."
```

---

## Task 2: Composite encoding support in `ngp_config`

Let a `base_ngp_cache` take an input vector whose leading dims go through the hash grid and whose trailing dims bypass it into the MLP. Nothing consumes it yet.

**Files:**
- Modify: `src/cuda/cache/bdrc/ngp_config.h`

**Interfaces:**
- Produces: `ngp_config::labelDims` (default 0) and `ngp_config::weightClip` (default 0).
- Produces: `toEncodingJson()` emits a `Composite` with nested `HashGrid` (`n_dims_to_encode = inputDim - labelDims`) and `Identity` (`n_dims_to_encode = labelDims`) when `labelDims > 0`, and the plain `HashGrid` otherwise.

- [ ] **Step 1: Add the two fields**

In `src/cuda/cache/bdrc/ngp_config.h`, after `int inputDim = 3;`:

```c
    // Trailing input dims that bypass the hash grid and enter the MLP
    // directly (an Identity nested encoding inside a Composite). 0 = the
    // plain hash-grid-only encoding. The unified field uses 2: a fixed-origin
    // face-orientation label and an arrival-side label. Labels must not be
    // hashed — the demo's field.wgsl appends them after the grid features for
    // the same reason: a label is a discrete condition on the function, not a
    // position whose neighbourhood should be interpolated.
    int labelDims = 0;
```

After `bool lamb = false;`:

```c
    // Hard cap on |weight| applied by tcnn's Adam kernel BEFORE the
    // half-precision cast that stores the parameter. 0 = disabled (tcnn's
    // default). Unlike gradient clipping, this is unconditional: no weight
    // this optimizer writes can overflow to Inf on the half cast. That is the
    // exact mechanism behind the reproducible NaN at iteration ~400 seen with
    // gradClip alone — a huge-but-finite fp32 Adam update, divided by a
    // near-zero second-moment estimate, overflowing on the cast without ever
    // producing a non-finite gradient for gradScaler's overflow check to
    // catch.
    float weightClip = 0.0f;
```

- [ ] **Step 2: Emit the composite encoding**

Replace `toEncodingJson()`:

```c
    nlohmann::json toEncodingJson() const
    {
        if (inputDim <= 0) {
            throw std::invalid_argument("NGPConfig.inputDim must be > 0");
        }
        if (labelDims < 0 || labelDims >= inputDim) {
            throw std::invalid_argument("NGPConfig.labelDims must be in [0, inputDim)");
        }
        if (nLevels <= 0 || nFeaturesPerLevel <= 0 || baseResolution <= 0 || log2HashmapSize <= 0) {
            throw std::invalid_argument("NGPConfig hash-grid parameters must be > 0");
        }
        if (perLevelScale <= 1.0f) {
            throw std::invalid_argument("NGPConfig.perLevelScale must be > 1.0");
        }

        const nlohmann::json grid = {
            {"otype", "HashGrid"},
            {"n_levels", nLevels},
            {"n_features_per_level", nFeaturesPerLevel},
            {"log2_hashmap_size", log2HashmapSize},
            {"base_resolution", baseResolution},
            {"per_level_scale", perLevelScale},
        };

        if (labelDims == 0) {
            return grid;
        }

        nlohmann::json gridEntry = grid;
        gridEntry["n_dims_to_encode"]      = inputDim - labelDims;
        gridEntry["dims_to_encode_begin"]  = 0;

        return {
            {"otype", "Composite"},
            {"reduction", "Concatenation"},
            {"nested", nlohmann::json::array({
                gridEntry,
                {
                    {"otype", "Identity"},
                    {"n_dims_to_encode", labelDims},
                    {"dims_to_encode_begin", inputDim - labelDims},
                },
            })},
        };
    }
```

- [ ] **Step 3: Pass `weightClip` to Adam**

In `toOptimizerJson()`'s Adam branch, add after `{"l2_reg", 0.0f},`:

```c
            {"clipping_magnitude", weightClip},
```

- [ ] **Step 4: Build**

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)
```

Expected: clean. Behaviour is unchanged because no config sets `labelDims` or `weightClip` yet.

- [ ] **Step 5: Commit**

```bash
git add src/cuda/cache/bdrc/ngp_config.h
git commit -m "feat: ngp_config composite encoding (hash grid + identity labels) and weightClip"
```

---

## Task 3: `unified_ngp_cache` — the one field and its weighted DRE loss

The highest-risk piece, gated by an analytic self-test before any renderer coupling.

**Files:**
- Create: `src/cuda/cache/bdrc/unified_ngp_cache.h`
- Create: `src/cuda/cache/bdrc/unified_ngp_cache.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces:
  ```cpp
  enum class DreObjective { Logistic = 0, Brier = 1, Lsif = 2, Bregman = 3 };
  struct DreConfig { DreObjective objective = DreObjective::Logistic;
                     float alpha = 1.0f, beta = 1.0f, power = 2.0f; };
  class unified_ngp_cache : public base_ngp_cache {
    explicit unified_ngp_cache(const DreConfig& dre, cudaStream_t stream = nullptr);
    void  setDreConfig(const DreConfig& dre);          // validates alpha==beta for raw-ratio
    void  query(cudaStream_t, const tcnn::GPUMatrixDynamic<float>& input,
                tcnn::GPUMatrix<float>& output, bool useEmaWeights);  // softplus applied
    float dreAccumGrad(cudaStream_t, const tcnn::GPUMatrixDynamic<float>& input,
                       const float* d_weight,      // [3 x batch] RGB positive weight
                       const float* d_density,     // [batch]     p_r; <=0 disables the lane
                       const float* d_scalePos,    // [batch]     1/N_pos or 0
                       const float* d_scaleRef,    // [batch]     1/N_ref or 0
                       uint32_t batchSize);        // returns summed loss
    void  optimizerStep(cudaStream_t);
    bool  selfTest(cudaStream_t);                  // analytic fixed-point check
  };
  ```
- Consumes: `ngp_config::labelDims` and `weightClip` from Task 2.

- [ ] **Step 1: Write the header**

`src/cuda/cache/bdrc/unified_ngp_cache.h`:

```cpp
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_ngp_cache — the single neural field of the unified transport cache.
//
// Stores incident flux per unit world area on diffuse receivers. Not radiance,
// not outgoing, no BSDF baked in: the receiver BSDF is applied at the query.
//
// Input : 5 dims. [0..2] world position normalized into the scene AABB, fed to
//         a hash grid. [3] sign_cam — the face's orientation relative to a
//         FIXED reference origin (never the live camera). [4] arrival side.
//         Both labels bypass the grid via an Identity nested encoding.
// Output: 3 channels, raw from the MLP. softplus is applied by this class
//         (query()) and differentiated analytically by the loss kernel, which
//         is why the network itself carries no output activation.
//
// Supervision: one weighted density-ratio objective over two positive streams
// (photon deposits and gather starts) against one surface reference measure.
// For any Bregman generator the fitted field satisfies U* = sum_s p_s w_bar_s,
// so disjoint path families supervising one network fit their sum.
//
#pragma once
#ifndef GLTF_RENDERER_UNIFIEDNGPCACHE_H
#define GLTF_RENDERER_UNIFIEDNGPCACHE_H

#include "base_ngp_cache.h"

// Density-ratio generator family. Identical membership and numbering to the
// demo's dre_loss.wgsl OBJECTIVE override.
enum class DreObjective
{
    Logistic = 0,  // posterior form; fixed point independent of alpha
    Brier    = 1,  // posterior form; fixed point independent of alpha
    Lsif     = 2,  // raw ratio; requires alpha == beta
    Bregman  = 3,  // raw ratio at arbitrary power; requires alpha == beta
};

struct DreConfig
{
    DreObjective objective = DreObjective::Logistic;

    // alpha is NCE's noise ratio and it matters: the field runs 1e2-1e4 times
    // the reference density, so at alpha = 1 the posterior saturates and the
    // shared head weighs the brightest regions least. The posterior members'
    // fixed point does not move with alpha; only the loss curvature does.
    float alpha = 1000.0f;
    float beta  = 1.0f;

    // Bregman power. |power - 2| < 1e-9 degenerates to Lsif, matching the demo.
    float power = 2.0f;
};

class unified_ngp_cache : public base_ngp_cache
{
public:
    explicit unified_ngp_cache(const DreConfig& dre, cudaStream_t stream = nullptr);

    // Rejects alpha != beta for the raw-ratio members: those read
    // u* = (beta/alpha) * sum_s p_s w_bar_s, so unequal constants silently
    // rescale the whole field.
    void setDreConfig(const DreConfig& dre);
    [[nodiscard]] const DreConfig& dreConfig() const { return m_dre; }

    // Inference with softplus applied in place, so callers read u directly.
    // useEmaWeights selects the teacher; every tail read, render query and
    // debug view must pass true.
    void query(cudaStream_t                         stream,
               const tcnn::GPUMatrixDynamic<float>& input,
               tcnn::GPUMatrix<float>&              output,
               bool                                 useEmaWeights);

    // One forward + analytic dL/dOutput + backward over the WHOLE batch
    // (photon lanes and gather lanes together). Because dL/dOutput is
    // per-sample there is no aliasing between lanes, so both streams reach one
    // backward and one optimizer step without any Accumulate dance.
    //
    // d_weight   [3 x batchSize] RGB positive weight w
    // d_density  [batchSize]     reference density q; <= 0 disables the lane
    // d_scalePos [batchSize]     positive normalizer (1/N_pos), 0 if none
    // d_scaleRef [batchSize]     reference normalizer (1/N_ref), 0 if none
    //
    // Returns the summed per-lane loss (one device->host sync).
    float dreAccumGrad(cudaStream_t                         stream,
                       const tcnn::GPUMatrixDynamic<float>& input,
                       const float*                         d_weight,
                       const float*                         d_density,
                       const float*                         d_scalePos,
                       const float*                         d_scaleRef,
                       uint32_t                             batchSize);

    void optimizerStep(cudaStream_t stream);

    // Analytic fixed-point check, run once at construction.
    //
    // Both streams are placed at one input point. Photons carry weight 2 with
    // scalePos = 1/N; gathers carry weight 3 with scalePos = scaleRef = 1/N.
    // Then sum_s p_s w_bar_s = 5q, and every member of the family has that
    // same minimizer: the posterior members' fixed point does not move with
    // alpha, and the raw-ratio members read (beta/alpha) * 5q = 5q when
    // alpha == beta. Prints the recovered u against 5q and returns whether it
    // lands within 15%.
    bool selfTest(cudaStream_t stream);

private:
    DreConfig m_dre;
};

#endif // GLTF_RENDERER_UNIFIEDNGPCACHE_H
```

- [ ] **Step 2: Write the implementation**

`src/cuda/cache/bdrc/unified_ngp_cache.cu`:

```cpp
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_ngp_cache — see the header for the field's contract.
//
// The loss is a direct port of the demo's dre_loss.wgsl (which in turn mirrors
// weighted_terms in the sibling project's density-ratio module): sum channels
// rather than averaging them, normalize each stream by its own draw count, and
// evaluate in log space so an untrained cell whose logit underflows still
// produces a finite gradient.
//

#include "unified_ngp_cache.h"

#include <tiny-cuda-nn/reduce_sum.h>

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{

constexpr int UNIFIED_INPUT_DIM  = 5;
constexpr int UNIFIED_LABEL_DIMS = 2;
constexpr int UNIFIED_OUTPUT_DIM = 3;

ngp_config makeUnifiedConfig()
{
    ngp_config cfg;

    cfg.inputDim  = UNIFIED_INPUT_DIM;
    cfg.labelDims = UNIFIED_LABEL_DIMS;

    // Hash grid: 8 levels x 2 features from 16 to ~273, matching the demo's
    // validated field. per_level_scale = exp((ln 273 - ln 16)/7) ~= 1.5.
    cfg.nLevels           = 8;
    cfg.nFeaturesPerLevel = 2;
    cfg.baseResolution    = 16;
    cfg.perLevelScale     = 1.5f;
    cfg.log2HashmapSize   = 20;

    cfg.hiddenDim        = 64;
    cfg.numHiddenLayers  = 2;
    cfg.hiddenActivation = "LeakyReLU";

    // No output activation: the loss kernel applies softplus itself and
    // differentiates through it analytically. Letting tcnn apply softplus too
    // would square the activation.
    cfg.outputActivation = "None";

    // m_loss is never used — dreAccumGrad writes dL/dOutput directly. The
    // base class always constructs one, so name a valid member.
    cfg.loss = "L2";

    cfg.lr       = 0.005f;
    cfg.gradClip = 0.05f;

    // Half-precision guards, carried over unchanged from the mask cache, which
    // documented a reproducible NaN by iteration ~400 without all three.
    cfg.gradScaler       = true;
    cfg.gradScalerGrowth = true;
    cfg.weightClip       = 1.0f;

    cfg.ema.enabled   = true;
    cfg.ema.decay     = 0.99f;
    cfg.ema.startIter = 100;

    return cfg;
}

// softplus, and its log with the derivative wrt the raw output.
//
// The low branch matters: an untrained cell can sit far enough below zero that
// softplus underflows to 0 in fp32 and log(0) = -inf poisons the gradient. For
// x < -20, softplus(x) ~= exp(x) to within fp32, so log u = x and
// d(log u)/d(raw) = 1.
__device__ __forceinline__ float uni_softplus(float x)
{
    const float e = expf(-fabsf(x));
    // log1p loses small e in fp32; the 4th-order series below 0.01 has
    // relative truncation error under 2e-9.
    const float small = e * (1.0f + e * (-0.5f + e * (1.0f / 3.0f - 0.25f * e)));
    return fmaxf(x, 0.0f) + ((e < 0.01f) ? small : logf(1.0f + e));
}

__device__ __forceinline__ float uni_sigmoid(float x)
{
    const float e = expf(-fabsf(x));
    return (x >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// Returns (log u, d(log u)/d(raw)).
__device__ __forceinline__ float2 uni_log_density(float raw)
{
    if (raw < -20.0f)
        return make_float2(raw, 1.0f);
    const float u = uni_softplus(raw);
    return make_float2(logf(u), uni_sigmoid(raw) / u);
}

// (positive term, reference term, d positive / d log u, d reference / d log u)
__device__ __forceinline__ float4 uni_terms(
    float logU, float logQ, int objective, float alpha, float beta, float power)
{
    if (objective <= 1)
    {
        const float logit = logU + logf(beta) - logQ - logf(alpha);
        const float s = uni_sigmoid(logit);
        const float t = uni_sigmoid(-logit);
        if (objective == 0)  // logistic
            return make_float4(uni_softplus(-logit), uni_softplus(logit), -t, s);
        // brier
        return make_float4(t * t, s * s, -2.0f * s * t * t, 2.0f * s * s * t);
    }

    const float logR = logU - logQ;

    if (objective == 2)  // lsif
    {
        const float r = expf(logR);
        return make_float4(-r, 0.5f * r * r, -r, r * r);
    }

    // bregman at arbitrary power, with the two limits handled separately.
    if (fabsf(power) < 1e-9f)
    {
        const float invR = expf(-logR);
        return make_float4(invR - 1.0f, logR, -invR, 1.0f);
    }
    if (fabsf(power - 1.0f) < 1e-9f)
    {
        const float r = expf(logR);
        return make_float4(-logR, r - 1.0f, -1.0f, r);
    }
    const float rp = expf((power - 1.0f) * logR);
    const float rr = expf(power * logR);
    return make_float4(-(rp - 1.0f) / (power - 1.0f), (rr - 1.0f) / power, -rp, rr);
}

__global__ void k_unified_dre_grad(
    const tcnn::network_precision_t* __restrict__ networkOut,
    const float*                     __restrict__ weight,     // [3 x batch]
    const float*                     __restrict__ density,    // [batch]
    const float*                     __restrict__ scalePos,   // [batch]
    const float*                     __restrict__ scaleRef,   // [batch]
    tcnn::network_precision_t*       __restrict__ dLdOut,
    float*                           __restrict__ lossOut,
    uint32_t batchSize,
    uint32_t paddedWidth,
    int      objective,
    float    alpha,
    float    beta,
    float    power,
    float    gradScale)
{
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batchSize) return;

    const uint32_t base = lane * paddedWidth;

    for (uint32_t r = 0u; r < paddedWidth; ++r)
        dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);

    const float q  = density[lane];
    const float sp = scalePos[lane];
    const float sr = scaleRef[lane];

    // A lane with no reference density (empty surface table, degenerate
    // triangle, a miss) or no stream membership contributes nothing. Its
    // denominator still stands in the normalizers, which is the point of
    // contract 4: a frame with fewer valid samples must not renormalize its
    // gradient up.
    if (!(q > 0.0f) || (sp == 0.0f && sr == 0.0f))
    {
        lossOut[lane] = 0.0f;
        return;
    }

    const float logQ = logf(q);
    float loss = 0.0f;

    for (uint32_t c = 0u; c < 3u; ++c)
    {
        const float2 head = uni_log_density(static_cast<float>(networkOut[base + c]));
        const float4 t    = uni_terms(head.x, logQ, objective, alpha, beta, power);

        const float positive  = beta  * weight[3u * lane + c] * sp;
        const float reference = alpha * sr;

        float g = 0.0f;
        // Do not multiply a zero stream weight by a possibly overflowing term.
        if (positive != 0.0f)
        {
            loss += positive * t.x;
            g    += positive * t.z * head.y;
        }
        if (reference != 0.0f)
        {
            loss += reference * t.y;
            g    += reference * t.w * head.y;
        }

        g *= gradScale;
        g  = fmaxf(fminf(g, 1.0e4f), -1.0e4f);  // half-precision overflow guard
        dLdOut[base + c] = static_cast<tcnn::network_precision_t>(g);
    }

    lossOut[lane] = loss;
}

__global__ void k_softplus_inplace(float* __restrict__ v, uint32_t n)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = uni_softplus(v[i]);
}

} // namespace

unified_ngp_cache::unified_ngp_cache(const DreConfig& dre, cudaStream_t stream)
    : base_ngp_cache(makeUnifiedConfig(), UNIFIED_OUTPUT_DIM, stream)
{
    setDreConfig(dre);
}

void unified_ngp_cache::setDreConfig(const DreConfig& dre)
{
    if (!(dre.alpha > 0.0f) || !(dre.beta > 0.0f) || !isfinite(dre.alpha) || !isfinite(dre.beta))
        throw std::invalid_argument("DreConfig: alpha and beta must be finite and positive");

    const bool rawRatio = (dre.objective == DreObjective::Lsif ||
                           dre.objective == DreObjective::Bregman);
    if (rawRatio && dre.alpha != dre.beta)
    {
        // u* = (beta/alpha) * sum_s p_s w_bar_s for these members, so unequal
        // constants rescale the entire field off the stream sum.
        throw std::invalid_argument(
            "DreConfig: raw-ratio generators (lsif, bregman) require alpha == beta");
    }
    if (dre.objective == DreObjective::Bregman && !isfinite(dre.power))
        throw std::invalid_argument("DreConfig: Bregman power must be finite");

    m_dre = dre;
}

void unified_ngp_cache::query(
    cudaStream_t                         stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    tcnn::GPUMatrix<float>&              output,
    bool                                 useEmaWeights)
{
    base_ngp_cache::query(stream, input, output, useEmaWeights);

    const uint32_t n = static_cast<uint32_t>(UNIFIED_OUTPUT_DIM) * output.n();
    k_softplus_inplace<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(
        output.data(), n);
    CUDA_CHECK_THROW(cudaGetLastError());
}

float unified_ngp_cache::dreAccumGrad(
    cudaStream_t                         stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    const float*                         d_weight,
    const float*                         d_density,
    const float*                         d_scalePos,
    const float*                         d_scaleRef,
    uint32_t                             batchSize)
{
    if (static_cast<int>(input.m()) != m_config.inputDim)
        throw std::runtime_error("unified_ngp_cache::dreAccumGrad: input row count != inputDim");
    if (input.n() != batchSize)
        throw std::runtime_error("unified_ngp_cache::dreAccumGrad: input batch mismatch");
    if (batchSize == 0u)
        return 0.0f;

    const uint32_t paddedWidth = m_network->padded_output_width();

    tcnn::GPUMatrix<tcnn::network_precision_t> networkOut(paddedWidth, batchSize, stream);
    tcnn::GPUMatrix<tcnn::network_precision_t> dLdOutput(paddedWidth, batchSize, stream);
    tcnn::GPUMemory<float>                     lossVals(batchSize);

    auto ctx = m_network->forward(stream, input, &networkOut, false, false);

    const float lossScale = m_gradScaler.currentScale();
    const float gradScale = lossScale / static_cast<float>(paddedWidth);

    const uint32_t blocks = (batchSize + tcnn::N_THREADS_LINEAR - 1u) / tcnn::N_THREADS_LINEAR;
    k_unified_dre_grad<<<blocks, tcnn::N_THREADS_LINEAR, 0, stream>>>(
        networkOut.data(), d_weight, d_density, d_scalePos, d_scaleRef,
        dLdOutput.data(), lossVals.data(),
        batchSize, paddedWidth,
        static_cast<int>(m_dre.objective), m_dre.alpha, m_dre.beta, m_dre.power,
        gradScale);
    CUDA_CHECK_THROW(cudaGetLastError());

    m_network->backward(stream, *ctx, input, networkOut, dLdOutput,
                        nullptr, false, tcnn::GradientMode::Overwrite);

    return tcnn::reduce_sum(lossVals.data(), lossVals.n(), stream);
}

void unified_ngp_cache::optimizerStep(cudaStream_t stream)
{
    const float lossScale = m_gradScaler.currentScale();
    if (m_config.gradScaler)
    {
        if (checkGradOverflow(stream))
        {
            m_gradScaler.onOverflow();
            return;
        }
        m_gradScaler.onSuccess();
    }
    if (m_config.gradClip > 0.0f)
        clipGradNorm(stream, m_config.gradClip * lossScale);
    m_optimizer->step(stream, lossScale, m_paramsFP, m_params, m_paramGrads);
}

bool unified_ngp_cache::selfTest(cudaStream_t stream)
{
    constexpr uint32_t N     = 256u;      // lanes per stream
    constexpr uint32_t BATCH = 2u * N;    // already a multiple of 256
    constexpr float    Q     = 1.0e-4f;   // reference density
    constexpr float    W_PH  = 2.0f;      // photon weight
    constexpr float    W_GA  = 3.0f;      // gather weight
    const     float    target = (W_PH + W_GA) * Q;

    std::vector<float> h_input(UNIFIED_INPUT_DIM * BATCH);
    std::vector<float> h_weight(3u * BATCH);
    std::vector<float> h_density(BATCH, Q);
    std::vector<float> h_scalePos(BATCH, 1.0f / static_cast<float>(N));
    std::vector<float> h_scaleRef(BATCH, 0.0f);

    for (uint32_t lane = 0u; lane < BATCH; ++lane)
    {
        float* in = h_input.data() + lane * UNIFIED_INPUT_DIM;
        in[0] = 0.5f; in[1] = 0.5f; in[2] = 0.5f;  // one point
        in[3] = 1.0f; in[4] = 1.0f;                // one (layer, side) bucket

        const bool photon = (lane < N);
        const float w = photon ? W_PH : W_GA;
        h_weight[3u * lane + 0u] = w;
        h_weight[3u * lane + 1u] = w;
        h_weight[3u * lane + 2u] = w;
        // Photons carry no anchor pairing; gathers are their own anchors.
        h_scaleRef[lane] = photon ? 0.0f : (1.0f / static_cast<float>(N));
    }

    tcnn::GPUMemory<float> d_input(h_input.size());     d_input.copy_from_host(h_input);
    tcnn::GPUMemory<float> d_weight(h_weight.size());   d_weight.copy_from_host(h_weight);
    tcnn::GPUMemory<float> d_density(h_density.size()); d_density.copy_from_host(h_density);
    tcnn::GPUMemory<float> d_scalePos(h_scalePos.size()); d_scalePos.copy_from_host(h_scalePos);
    tcnn::GPUMemory<float> d_scaleRef(h_scaleRef.size()); d_scaleRef.copy_from_host(h_scaleRef);
    tcnn::GPUMemory<float> d_output(UNIFIED_OUTPUT_DIM * BATCH);

    tcnn::GPUMatrixDynamic<float> inputMat(d_input.data(), UNIFIED_INPUT_DIM, BATCH,
                                           tcnn::MatrixLayout::ColumnMajor);

    for (int step = 0; step < 3000; ++step)
    {
        dreAccumGrad(stream, inputMat, d_weight.data(), d_density.data(),
                     d_scalePos.data(), d_scaleRef.data(), BATCH);
        optimizerStep(stream);
        updateEma(stream);
    }

    tcnn::GPUMatrix<float> outputMat(d_output.data(), UNIFIED_OUTPUT_DIM, BATCH);
    query(stream, inputMat, outputMat, /*useEmaWeights=*/false);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    std::vector<float> h_output(UNIFIED_OUTPUT_DIM * BATCH);
    d_output.copy_to_host(h_output);

    const float recovered = h_output[0];
    const float ratio     = recovered / target;
    const bool  ok        = (ratio > 0.85f && ratio < 1.15f);

    printf("[unified-selftest] objective=%d alpha=%g beta=%g  u=%.6g  expected=%.6g  ratio=%.3f  %s\n",
           static_cast<int>(m_dre.objective), m_dre.alpha, m_dre.beta,
           recovered, target, ratio, ok ? "PASS" : "FAIL");

    return ok;
}
```

- [ ] **Step 3: Update CMake**

In `CMakeLists.txt` replace `src/cuda/cache/bdrc/mask_ngp_cache.h` with `src/cuda/cache/bdrc/unified_ngp_cache.h` (line ~329) and `${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cache/bdrc/mask_ngp_cache.cu` with `${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cache/bdrc/unified_ngp_cache.cu` (line ~344).

`mask_ngp_cache.h/.cu` still exist and are still referenced by `bdrc_host.cu`; they are deleted in Task 6. Keep them listed as an extra source alongside the new one until then, or the build breaks:

```cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cache/bdrc/mask_ngp_cache.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/cache/bdrc/unified_ngp_cache.cu
```

- [ ] **Step 4: Run the self-test at startup**

Temporarily call it from `Application::initBDRC()` right after the system is created, so it runs on the next launch. Add to the end of `initBDRC()`:

```cpp
  // Temporary construction gate — removed in Task 7 once the field is wired.
  {
    for (DreObjective obj : {DreObjective::Logistic, DreObjective::Brier,
                             DreObjective::Lsif, DreObjective::Bregman})
    {
      const bool raw = (obj == DreObjective::Lsif || obj == DreObjective::Bregman);
      DreConfig  dre;
      dre.objective = obj;
      dre.alpha     = raw ? 1.0f : 1000.0f;
      dre.beta      = raw ? 1.0f : 1.0f;
      dre.power     = 3.0f;
      unified_ngp_cache probe(dre, m_cudaStream);
      probe.selfTest(m_cudaStream);
    }
  }
```

- [ ] **Step 5: Build and run the gate**

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb
```

Expected console output — four lines, every one `PASS`, every `u` near `5e-4`:

```
[unified-selftest] objective=0 alpha=1000 beta=1  u=0.000500...  expected=0.0005  ratio=1.00...  PASS
[unified-selftest] objective=1 alpha=1000 beta=1  u=0.000500...  expected=0.0005  ratio=1.00...  PASS
[unified-selftest] objective=2 alpha=1 beta=1     u=0.000500...  expected=0.0005  ratio=1.00...  PASS
[unified-selftest] objective=3 alpha=1 beta=1     u=0.000500...  expected=0.0005  ratio=1.00...  PASS
```

That all four generators land on the same `5q` is the check that matters: it confirms the posterior members' fixed point does not move with `alpha` (contract 3) and that the raw-ratio members read `(beta/alpha)·5q` correctly (contract 2). If a member lands on a different multiple of `q`, its `uni_terms` branch is wrong; if every member lands on the wrong multiple, the normalizer wiring is wrong.

- [ ] **Step 6: Gate the `alpha != beta` rejection**

Add one more probe before the loop and confirm the process prints the throw message and does not fit:

```cpp
    try {
      DreConfig bad; bad.objective = DreObjective::Lsif; bad.alpha = 1000.0f; bad.beta = 1.0f;
      unified_ngp_cache probe(bad, m_cudaStream);
      printf("[unified-selftest] alpha!=beta rejection: FAIL (constructed)\n");
    } catch (const std::invalid_argument& e) {
      printf("[unified-selftest] alpha!=beta rejection: PASS (%s)\n", e.what());
    }
```

Expected: `PASS (DreConfig: raw-ratio generators (lsif, bregman) require alpha == beta)`.

- [ ] **Step 7: Commit**

```bash
git add src/cuda/cache/bdrc/unified_ngp_cache.h src/cuda/cache/bdrc/unified_ngp_cache.cu CMakeLists.txt src/Application.cpp
git commit -m "feat: unified_ngp_cache with the weighted density-ratio generator family

One 5-dim field (3 hashed position dims + 2 identity label dims), 3-channel
softplus output applied outside tcnn so the loss can differentiate through it
analytically. Startup self-test confirms all four generators reach the same
sum-of-streams fixed point and that raw-ratio members reject alpha != beta."
```

---

## Task 4: Strip the old BDRC machinery out of the path tracer

Remove the caustic-suppression state, the virtual-emission recording and the camera-facing BSDF evaluation. Everything the spec's "Removed" section lists that lives in `hit.cu` and `per_ray_data.h` goes here. This also removes the last live-camera read from the training path (Review Focus 5).

**Files:**
- Modify: `src/cuda/per_ray_data.h`
- Modify: `src/cuda/hit.cu`
- Modify: `src/cuda/raygen.cu` (initializers of the removed fields)

**Interfaces:**
- Produces: `PerRayData::bdrcFrontFace` (`bool`) — true when the ray struck the triangle's front face, captured before the normal is flipped to the ray side.
- Produces: `PerRayData::bdrcNg` (`float3`) — the raw, unflipped world-space geometric normal at the hit.
- Removes: `bsdfEvalTowardCamera`, `virtEmitDepth`, `eyePathFoundDiffuse`, `eyePathInSpecularChain`, `FLAG_BDRC_LIGHT_PATH`.

- [ ] **Step 1: Trim `PerRayData`**

In `src/cuda/per_ray_data.h`, delete the `FLAG_BDRC_LIGHT_PATH` define (if the stash re-applied it, it will not be present after Task 1's stash) and replace the BDRC block at the end of the struct:

```c
  // BDRC: geometric state at the hit, captured BEFORE the shading and
  // geometric normals are flipped to the ray side.
  //
  // bdrcNg is the raw outward face normal as authored. bdrcFrontFace records
  // which side the ray arrived on. Together they give the field's two labels:
  // sign_cam (bdrcNg against a FIXED reference origin) and arrival side
  // (bdrcFrontFace). The ray-flipped normal cannot serve: dot(flippedN, wo) is
  // non-negative by construction, so it can never tell the two sides apart.
  float3 bdrcNg;
  bool   bdrcFrontFace;
```

Delete `eyePathFoundDiffuse`, `eyePathInSpecularChain`, `virtEmitDepth`, and `bsdfEvalTowardCamera` along with their comments.

- [ ] **Step 2: Capture the geometric state in `hit.cu`**

In `chRadiance()`, immediately after the existing `isFrontFace` computation (~line 1473) and **before** the flip:

```c
  const bool isFrontFace = (0.0f <= dot(thePrd->wo, state.Ng));

  // BDRC labels are taken here, before the flip below: state.Ng is still the
  // authored outward face normal.
  thePrd->bdrcNg        = state.Ng;
  thePrd->bdrcFrontFace = isFrontFace;
```

- [ ] **Step 3: Delete the virtual-emission block**

Remove the whole `if (!validForEyeCache && theLaunchParameters.nrc.enabled && ... VIRT_EMIT_MAX_DEPTH)` block (~lines 1536–1570) including its comment.

- [ ] **Step 4: Delete the caustic suppression**

Replace

```c
  const bool validForEyeCache = !(theLaunchParameters.nrc.enabled
                                  && theLaunchParameters.nrc.nrcType == BDRC
                                  && thePrd->eyePathFoundDiffuse
                                  && thePrd->eyePathInSpecularChain);
```

with nothing, and drop every use of `validForEyeCache`:
- `if (validForEyeCache) { ...emission... }` becomes the emission block unguarded.
- `if (!theLaunchParameters.directLighting || numLights <= 0 || !validForEyeCache)` becomes `if (!theLaunchParameters.directLighting || numLights <= 0)`.

Also delete the suppression-state update after BSDF sampling:

```c
  // Update BDRC caustic-suppression state (mirrors Python bdpt.py lines 691/696).
  if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == BDRC)
  {
    thePrd->eyePathFoundDiffuse    = ...;
    thePrd->eyePathInSpecularChain = ...;
  }
```

The unified design needs no suppression: the camera path stops at its first diffuse vertex and never traverses a `D S+` chain, and the photon and gather families are disjoint by grammar rather than by gating.

- [ ] **Step 5: Delete the `bsdfEvalTowardCamera` block**

If Task 1's stash left it out, confirm with `grep -n bsdfEvalTowardCamera src/cuda/hit.cu` returning nothing. If it is present, delete the whole `if ((thePrd->flags & FLAG_BDRC_LIGHT_PATH) && ...)` block. This is the last read of `theLaunchParameters.cameraP` on any training path.

- [ ] **Step 6: Drop the initializers in `raygen.cu`**

In `integrator()` remove:

```c
  prd.eyePathFoundDiffuse    = false;
  prd.eyePathInSpecularChain = false;
  prd.virtEmitDepth = 0;
```

and the same two lines in `bdrcLightIntegrator()`.

- [ ] **Step 7: Build**

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)
```

`bdrc_host.cu` still writes `virtEmit*` buffers and `launch_parameters.h` still declares them; they are untouched here, so the build stays green. Expected: clean.

- [ ] **Step 8: Runtime gate — camera-independent shading state**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb
```

Expected: BDRC mode still renders (quality shifts, since suppression is gone and the light cache now double-counts — this is the "worse image at an intermediate task" the spec warns about, not a bug). The gate is that nothing in the hit shader reads `cameraP` any more:

```bash
grep -n "cameraP" src/cuda/hit.cu
```

Expected: only the BDPT light-path camera-connection block (guarded by `FLAG_LIGHT_PATH`, used by BDPT mode, not by BDRC).

- [ ] **Step 9: Commit**

```bash
git add src/cuda/per_ray_data.h src/cuda/hit.cu src/cuda/raygen.cu
git commit -m "refactor: remove BDRC caustic suppression, virtual emission and camera-facing BSDF eval

The unified design needs none of them: the camera path stops at its first
diffuse vertex, and the photon and gather families are disjoint by grammar.
Capture the pre-flip face normal and front-face flag instead, which the
field's two labels are built from."
```

---

## Task 5: The three streams in `raygen.cu`

Rewrite the camera and photon streams and add the gather stream. The gather raygen is the spec's largest flagged risk; it is reached without any hit-record shading state because the anchor's own BSDF is replaced by a unit BSDF.

**Files:**
- Create: `src/cuda/cache/bdrc/unified_features.h`
- Modify: `src/cuda/launch_parameters.h` (the `NrcLaunchBuffers` BDRC section)
- Modify: `src/cuda/raygen.cu`

**Interfaces:**
- Produces: `bdrcPackQuery5(float* buf, uint32_t lane, float3 worldPos, float3 ng, float side, float3 refOrigin, float3 aabbMin, float3 aabbMax)`.
- Produces: `extern "C" __global__ void __raygen__bdrc_gather()`.
- Consumes: `NrcLaunchBuffers` fields added below; `sampleSurfaceUniform` from Task 1.

- [ ] **Step 1: Write the query packer**

`src/cuda/cache/bdrc/unified_features.h`:

```c
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_features.h — device-side query packing for the unified field.
//
// One query is 5 floats: [0..2] world position normalized into the scene AABB
// (hash grid), [3] sign_cam, [4] arrival side (both Identity).
//
#pragma once
#ifndef UNIFIED_FEATURES_H
#define UNIFIED_FEATURES_H

#include "../nrc_features.h"

#define UNIFIED_INPUT_DIM 5

// Face orientation against a FIXED reference origin. Never the live camera: a
// label taken against an interactive camera flips for every point the user
// orbits past, silently relabelling the whole learned function mid-fit.
__device__ __forceinline__ float bdrcSignCamLabel(
    const float3 ng, const float3 worldPos, const float3 referenceOrigin)
{
    return (dot(ng, referenceOrigin - worldPos) >= 0.0f) ? 1.0f : -1.0f;
}

// Which of the two sides of the field this sample belongs to. For a ray this
// is the side it arrived on; for an anchor it is the coin flip.
__device__ __forceinline__ float bdrcSideLabel(const bool frontFace)
{
    return frontFace ? 1.0f : -1.0f;
}

__device__ __forceinline__ void bdrcPackQuery5(
    float*       buf,
    uint32_t     lane,
    const float3 worldPos,
    const float3 ng,
    const float  side,
    const float3 referenceOrigin,
    const float3 aabbMin,
    const float3 aabbMax)
{
    float* d = buf + lane * UNIFIED_INPUT_DIM;

    const float rx = aabbMax.x - aabbMin.x;
    const float ry = aabbMax.y - aabbMin.y;
    const float rz = aabbMax.z - aabbMin.z;
    d[0] = fminf(fmaxf((worldPos.x - aabbMin.x) / (rx > 0.f ? rx : 1.f), 0.f), 1.f);
    d[1] = fminf(fmaxf((worldPos.y - aabbMin.y) / (ry > 0.f ? ry : 1.f), 0.f), 1.f);
    d[2] = fminf(fmaxf((worldPos.z - aabbMin.z) / (rz > 0.f ? rz : 1.f), 0.f), 1.f);
    d[3] = bdrcSignCamLabel(ng, worldPos, referenceOrigin);
    d[4] = side;
}

#endif // UNIFIED_FEATURES_H
```

- [ ] **Step 2: Replace the BDRC section of `NrcLaunchBuffers`**

In `src/cuda/launch_parameters.h`, delete every BDRC field (`lightEndpoints`, `lightTargets`, `light_p_e_area`, `lightValid`, `lightOutput`, `virtEmit*`, `eyeVertex*`, `resampled*`, `bdrcPhase`) and the `BDRC_MAX_LIGHT_DEPTH` / `VIRT_EMIT_MAX_DEPTH` constants, and add:

```c
  // ---- Unified transport cache ---------------------------------------------
  //
  // Training lanes are a single fixed-slot array: [0, photonCount) photons,
  // [photonCount, photonCount + gatherCount) gathers. There is no compaction.
  // An invalid lane writes recDensity = 0 and contributes nothing, while its
  // launch count stays in the normalizer — a frame with fewer valid samples
  // must not renormalize its gradient up.
  unsigned int photonCount;   // == numPixels: the photon raygen launches at render resolution
  unsigned int gatherCount;   // independent of resolution and of camera framing

  float*       recQuery;      // [5 x lanes]  start/deposit query (position + 2 labels)
  float*       recBase;       // [3 x lanes]  photon: clamped power. gather: E_dir + chain radiance
  float*       recDensity;    // [lanes]      p_r at this lane; 0 = lane disabled
  float*       recTailQuery;  // [5 x lanes]  gather tail query; density 0 when absent
  float*       recTailValid;  // [lanes]      1 when the tail query is meaningful
  float*       recFactor;     // [3 x lanes]  gather: throughput * f_s at the tail

  // ---- Camera stream (pixel-indexed) ---------------------------------------
  float*       camQuery;      // [5 x numPixels] query at the first diffuse vertex D
  float*       camPrefix;     // [3 x numPixels] radiance accumulated before D
  float*       camFactor;     // [3 x numPixels] T(cam->D) * f_s(D) — the BSDF is applied HERE
  unsigned int* camValid;     // [numPixels]

  // ---- Reference measure ---------------------------------------------------
  // Halved because the field holds a value per (point, side): an anchor takes
  // either side with probability 1/2, so its density per (point, side) is half
  // the area density. Deposits and gathers use the same halved value.
  float        referenceDensity;

  // Fixed at construction, stamped, never updated on camera motion.
  float3       referenceOrigin;

  float        photonClamp;   // upper bound on stored photon power
```

Keep `aabbMin`, `aabbMax`, `spreadC`, `terminationDepth`, `nrcType`, `enabled`, `nrcDebugMode` and the whole forward-NRC section (NRC mode is not modified).

- [ ] **Step 3: Rewrite the camera stream**

In `integrator()`, replace the entire "BDRC: terminate at first diffuse vertex" block (the one that writes `eyeVertexPos`, samples the reference anchor and calls `handleBdrcCacheLogic`) with:

```c
    // BDRC camera stream: stop at the first diffuse vertex D and record one
    // query there. The pixel reads prefix + T(cam->D) * f_s(D) * U(D); the
    // receiver BSDF is applied here, in the renderer, never in the field.
    if (theLaunchParameters.nrc.enabled && theLaunchParameters.nrc.nrcType == BDRC &&
        (prd.flags & FLAG_HIT) && (prd.typeEvent & BSDF_EVENT_DIFFUSE))
    {
      const unsigned int px = theLaunchParameters.resolution.x * theLaunchIndex.y + theLaunchIndex.x;
      const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

      bdrcPackQuery5(nrc.camQuery, px, prd.pos, prd.bdrcNg,
                     bdrcSideLabel(prd.bdrcFrontFace),
                     nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

      // radianceBeforeTrace is the prefix: everything accumulated before this
      // segment's trace, which is NEE and emission at the S vertices ahead of D
      // plus D's own NEE and emission added by chRadiance during this trace.
      nrc.camPrefix[3*px+0] = prd.radiance.x;
      nrc.camPrefix[3*px+1] = prd.radiance.y;
      nrc.camPrefix[3*px+2] = prd.radiance.z;

      // f_s(D) for the Lambertian lobe is baseColor / pi. throughputPreBsdf is
      // T(cam->D) before D's own BSDF was applied.
      const float3 f = prd.nrcBaseColor * M_1_PIf;
      nrc.camFactor[3*px+0] = prd.throughputPreBsdf.x * f.x;
      nrc.camFactor[3*px+1] = prd.throughputPreBsdf.y * f.y;
      nrc.camFactor[3*px+2] = prd.throughputPreBsdf.z * f.z;

      nrc.camValid[px]   = 1u;
      prd.nrcTerminated  = true;
      break;
    }
```

Delete `handleBdrcCacheLogic()` and `handleBdrcDebugFirstHit()` entirely, along with the call site of the latter and the `lastSpecPos` / `lastSpecNormal` / `hasSpecular` tracking in `integrator()`.

- [ ] **Step 4: Rewrite the photon stream**

Replace `bdrcLightIntegrator()`:

```c
// BDRC photon stream — the ungated L S+ D grammar. Traces from an emitter
// through non-diffuse vertices and stores exactly ONE deposit at the first
// diffuse vertex reached after at least one non-diffuse bounce, then stops.
//
// No shadow ray, no connection to any eye vertex, no dependence on the camera.
// The stored value is RAW clamped photon power: under the incident convention
// the receiver BSDF belongs at the query, not in the field.
__forceinline__ __device__ void bdrcPhotonIntegrator(PerRayData& prd, const unsigned int lane)
{
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  // Disable the lane up front; a path that reaches no deposit leaves it off
  // while its launch still counts in the 1/photonCount normalizer.
  nrc.recDensity[lane] = 0.0f;

  prd.radiance  = make_float3(0.0f);
  prd.pdf       = 0.0f;
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent = BSDF_EVENT_ABSORB;
  prd.flags     = 0;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

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
        // throughputPreBsdf is the power arriving at this vertex, before the
        // local BSDF response. That response is the query's job.
        const float c = theLaunchParameters.nrc.photonClamp;
        const float3 power = make_float3(fminf(prd.throughputPreBsdf.x, c),
                                         fminf(prd.throughputPreBsdf.y, c),
                                         fminf(prd.throughputPreBsdf.z, c));

        bdrcPackQuery5(nrc.recQuery, lane, prd.pos, prd.bdrcNg,
                       bdrcSideLabel(prd.bdrcFrontFace),
                       nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

        nrc.recBase[3*lane+0] = fmaxf(power.x, 0.0f);
        nrc.recBase[3*lane+1] = fmaxf(power.y, 0.0f);
        nrc.recBase[3*lane+2] = fmaxf(power.z, 0.0f);
        nrc.recDensity[lane]  = nrc.referenceDensity;
        nrc.recTailValid[lane] = 0.0f;
      }
      break;  // a diffuse hit with no prior specular bounce is a dead end here
    }

    hasSpecular = true;

    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput);
      if (probability < rng(prd.seed))
        break;
      prd.throughput /= probability;
    }

    ++depth;
  }
}
```

Change `__raygen__light_path_tracer` to call `bdrcPhotonIntegrator(prd, pixelIdx)` in the BDRC branch.

- [ ] **Step 5: Add the unit-BSDF direct estimate**

Add above the gather integrator in `raygen.cu`:

```c
// Direct illumination at a point with a UNIT BSDF (f = 1), i.e. an estimate of
// incident flux per unit area rather than outgoing radiance. The demo achieves
// this by forcing the receiver's kd to pi before calling estimate_direct; here
// the anchor has no material at all, so f = 1 is written out directly and
// bxdf = f * cos = cos.
//
// Light sampling only. Its MIS partner is the cosine-hemisphere sample the
// gather continues with, whose emitter and environment hits chRadiance and the
// miss program already weight against prd.pdf.
__forceinline__ __device__ float3 bdrcUnitBsdfDirect(
    PerRayData& prd, const float3 pos, const float3 n)
{
  const int numLights = theLaunchParameters.numLights;
  if (!theLaunchParameters.directLighting || numLights <= 0)
    return make_float3(0.0f);

  const int indexLight = (1 < numLights)
      ? clamp(static_cast<int>(floorf(rng(prd.seed) * numLights)), 0, numLights - 1) : 0;
  const LightDefinition& light = theLaunchParameters.lightDefinitions[indexLight];

  prd.pos       = pos;   // the light callables sample relative to prd.pos
  prd.occlusion = 1.0f;

  LightSample lightSample =
      optixDirectCall<LightSample, const LightDefinition&, PerRayData*>(light.typeLight, light, &prd);

  if (lightSample.pdf <= 0.0f)
    return make_float3(0.0f);

  const float cosTheta = dot(lightSample.direction, n);
  if (cosTheta <= 0.0f)
    return make_float3(0.0f);

  unsigned int isVisible = 0;
  unsigned int seed      = prd.seed;

  optixTrace(theLaunchParameters.handle,
             pos, lightSample.direction,
             theLaunchParameters.sceneEpsilon,
             lightSample.distance - theLaunchParameters.sceneEpsilon, 0.0f,
             OptixVisibilityMask(0xFF),
             OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
             TYPE_RAY_SHADOW, NUM_RAY_TYPES, TYPE_RAY_SHADOW,
             isVisible, seed);

  prd.seed = seed;

  if (!isVisible)
    return make_float3(0.0f);

  // Singular lights have no BSDF-sampling partner, so they take no MIS weight.
  // For area and environment lights the partner is the cosine hemisphere,
  // whose solid-angle pdf is cos/pi.
  const float weightMIS = (light.typeLight <= TYPE_LIGHT_ENV_SPHERE ||
                           light.typeLight == TYPE_LIGHT_MESH)
                          ? balanceHeuristic(lightSample.pdf / float(numLights), cosTheta * M_1_PIf)
                          : 1.0f;

  return lightSample.radiance_over_pdf * (cosTheta * float(numLights) * weightMIS);
}
```

- [ ] **Step 6: Write the gather integrator and its raygen**

```c
// BDRC gather stream. Per lane:
//   1. draw an anchor from the global area-uniform surface table and pick a
//      side with probability 1/2;
//   2. estimate direct illumination there with a unit BSDF, so the estimate is
//      incident flux rather than outgoing radiance;
//   3. sample a cosine hemisphere, continue through any non-diffuse chain to
//      the first diffuse vertex y, and store the tail query and
//      factor = throughput * f_s(y);
//   4. the anchor is both the reference anchor and the gather start, so one
//      query serves the lane's reference term and its positive term, and the
//      gathered half of the fit is a regression at known points.
//
// Nothing here reads the viewing camera.
__forceinline__ __device__ void bdrcGatherIntegrator(PerRayData& prd, const unsigned int lane)
{
  const NrcLaunchBuffers& nrc = theLaunchParameters.nrc;

  nrc.recDensity[lane]   = 0.0f;
  nrc.recTailValid[lane] = 0.0f;

  float3 anchorPos;
  float3 anchorNg;
  if (!sampleSurfaceUniform(prd.seed, theLaunchParameters.surfaceRef, anchorPos, anchorNg))
    return;  // empty or degenerate table: the lane stays disabled

  const float  side     = (rng(prd.seed) < 0.5f) ? 1.0f : -1.0f;
  const float3 oriented = anchorNg * side;

  bdrcPackQuery5(nrc.recQuery, lane, anchorPos, anchorNg, side,
                 nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);
  nrc.recDensity[lane] = nrc.referenceDensity;

  prd.idxStack = 0;
  prd.stack[0].absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  prd.occlusion = 1.0f;
  prd.sigma_t   = make_float3(0.0f);
  prd.typeEvent = BSDF_EVENT_ABSORB;
  prd.flags     = 0;

  const float3 origin = anchorPos + oriented * theLaunchParameters.sceneEpsilon;

  // E_dir at the anchor, seeded into radiance so the continuation's own NEE and
  // emission accumulate onto it exactly as a prefix does.
  prd.radiance = bdrcUnitBsdfDirect(prd, origin, oriented);

  // Cosine hemisphere. With f = 1 the weight is f*cos/pdf = cos/(cos/pi) = pi.
  float3 dir;
  float  dirPdf;
  unitSquareToCosineHemisphere(make_float2(rng(prd.seed), rng(prd.seed)), oriented, dir, dirPdf);
  if (dirPdf <= 0.0f)
  {
    nrc.recBase[3*lane+0] = prd.radiance.x;
    nrc.recBase[3*lane+1] = prd.radiance.y;
    nrc.recBase[3*lane+2] = prd.radiance.z;
    return;
  }

  prd.pos        = origin;
  prd.wi         = dir;
  prd.throughput = make_float3(M_PIf);
  prd.pdf        = dirPdf;   // the MIS partner of the anchor's own light sample

  uint2 payload = splitPointer(&prd);

  int depth = 0;
  while (depth < theLaunchParameters.pathLengths.y)
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

    if (!(prd.flags & FLAG_HIT))
      break;                                   // env radiance already added by the miss program

    if (prd.typeEvent & BSDF_EVENT_DIFFUSE)
    {
      // The tail. Its value holds both its own direct light and everything
      // beyond, so the segment stops here and reads the field instead.
      bdrcPackQuery5(nrc.recTailQuery, lane, prd.pos, prd.bdrcNg,
                     bdrcSideLabel(prd.bdrcFrontFace),
                     nrc.referenceOrigin, nrc.aabbMin, nrc.aabbMax);

      const float3 f = prd.nrcBaseColor * M_1_PIf;
      nrc.recFactor[3*lane+0] = prd.throughputPreBsdf.x * f.x;
      nrc.recFactor[3*lane+1] = prd.throughputPreBsdf.y * f.y;
      nrc.recFactor[3*lane+2] = prd.throughputPreBsdf.z * f.z;
      nrc.recTailValid[lane]  = 1.0f;
      break;
    }

    if ((prd.typeEvent == BSDF_EVENT_ABSORB) || isNull(prd.throughput))
      break;

    if (theLaunchParameters.pathLengths.x <= depth)
    {
      const float probability = fmaxf(prd.throughput);
      if (probability < rng(prd.seed))
        break;
      prd.throughput /= probability;
    }

    ++depth;
  }

  nrc.recBase[3*lane+0] = fmaxf(prd.radiance.x, 0.0f);
  nrc.recBase[3*lane+1] = fmaxf(prd.radiance.y, 0.0f);
  nrc.recBase[3*lane+2] = fmaxf(prd.radiance.z, 0.0f);
}


// One lane per gather start. The launch width is gatherCount, which is
// independent of the render resolution and of camera framing.
extern "C" __global__ void __raygen__bdrc_gather()
{
  if (theLaunchParameters.nrc.nrcType != BDRC || !theLaunchParameters.nrc.enabled ||
      0.0f <= theLaunchParameters.picking.x)
    return;

  const unsigned int lane = optixGetLaunchIndex().x;
  if (lane >= theLaunchParameters.nrc.gatherCount)
    return;

  // Gather lanes live above the photon lanes in one records array.
  const unsigned int slot = theLaunchParameters.nrc.photonCount + lane;

  PerRayData prd;
  prd.seed = tea<4>(lane, theLaunchParameters.iteration + 0x9e3779b9u);

  bdrcGatherIntegrator(prd, slot);
}
```

Add `#include "cache/bdrc/unified_features.h"` and `#include "surface_reference.h"` to `raygen.cu` if not already present.

- [ ] **Step 7: Build**

`bdrc_host.cu` and `Application.cpp` still reference the deleted `NrcLaunchBuffers` fields, so this step will not link yet. Compile only the OptiX device modules to check the shader code:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer_optix_lib -j$(nproc)
```

Expected: `raygen.cu`, `hit.cu`, `miss.cu` compile with no errors. The host link failure is expected and is resolved in Tasks 6 and 7.

- [ ] **Step 8: Commit**

```bash
git add src/cuda/
git commit -m "feat: camera, photon and gather streams for the unified transport cache

The gather raygen draws an area-uniform anchor, estimates direct light there
with a unit BSDF so the estimate is incident flux, and continues through any
non-diffuse chain to the first diffuse vertex, where it records the tail query
and throughput * f_s. The anchor doubles as its own reference anchor."
```

---

## Task 6: Host dispatch — one field, one optimizer step

Rewrite `bdrc_host` around the single cache: pack, EMA-infer the tails, assemble targets, one loss, one step, composite.

**Files:**
- Modify: `src/cuda/cache/bdrc/bdrc_host.h`
- Modify: `src/cuda/cache/bdrc/bdrc_host.cu`
- Modify: `src/HostKernels.h`, `src/cuda/cache/kernel_nrc.cu`
- Delete: `src/cuda/cache/bdrc/mask_ngp_cache.h`, `src/cuda/cache/bdrc/mask_ngp_cache.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `bdrcCreate(cudaStream_t, const DreConfig&)`, `bdrcAllocBuffers(BdrcSystem*, uint32_t numPixels, uint32_t gatherCount)`, accessors `bdrcRecQuery/RecBase/RecDensity/RecTailQuery/RecTailValid/RecFactor/CamQuery/CamPrefix/CamFactor/CamValid`, `bdrcGetLastLoss`, `bdrcGetPhotonLive`, `bdrcGetGatherLive`, `bdrcSetDreConfig`, and a `bdrcDispatch` with the signature below.
- Produces: `device_bdrc_composite(stream, bufferAccum, camPrefix, camFactor, fieldOutput, camValid, numPixels, iteration)`.
- Produces: `device_bdrc_debug(stream, bufferAccum, camFactor, fieldOutput, camValid, numPixels, iteration, debugMode)`.

- [ ] **Step 1: Replace the composite and debug kernels**

In `src/cuda/cache/kernel_nrc.cu` replace `k_bdrc_composite` / `device_bdrc_composite`:

```c
// pixel = prefix + T(cam->D) * f_s(D) * U(D).
// camFactor already carries T * f_s: the receiver BSDF is applied here, in the
// renderer, because the field stores incident flux, not outgoing radiance.
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
```

Replace `k_bdrc_debug` / `device_bdrc_debug` with a two-mode version (1 = the field's own contribution, `factor * U`; 2 = the raw field value `U`), and delete `k_bdrc_splat_endpoints` / `device_bdrc_splat_endpoints` entirely. Update the declarations in `src/HostKernels.h` to match and delete the splat declaration.

- [ ] **Step 2: Rewrite `bdrc_host.h`**

```cpp
// bdrc_host.h — host interface for the unified transport cache.
//
// One neural field holding incident flux per unit world area, supervised by a
// photon stream and a gather stream through one weighted density-ratio
// objective. Replaces the three-network eye/light/mask system.
#pragma once
#ifndef BDRC_HOST_H
#define BDRC_HOST_H

#include "unified_ngp_cache.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

struct BdrcSystem;  // opaque, defined in bdrc_host.cu

BdrcSystem* bdrcCreate(cudaStream_t stream, const DreConfig& dre);
void        bdrcDestroy(BdrcSystem* sys);

// numPixels sizes the camera and photon lanes; gatherCount sizes the gather
// lanes and is independent of the render resolution.
void bdrcAllocBuffers(BdrcSystem* sys, uint32_t numPixels, uint32_t gatherCount);

// Records array — lanes [0, photonCount) photons, [photonCount, lanes) gathers.
float* bdrcRecQuery     (const BdrcSystem* sys);
float* bdrcRecBase      (const BdrcSystem* sys);
float* bdrcRecDensity   (const BdrcSystem* sys);
float* bdrcRecTailQuery (const BdrcSystem* sys);
float* bdrcRecTailValid (const BdrcSystem* sys);
float* bdrcRecFactor    (const BdrcSystem* sys);

// Camera stream — pixel-indexed.
float*        bdrcCamQuery (const BdrcSystem* sys);
float*        bdrcCamPrefix(const BdrcSystem* sys);
float*        bdrcCamFactor(const BdrcSystem* sys);
unsigned int* bdrcCamValid (const BdrcSystem* sys);

// Diagnostics for the spec's stream-count and fit gates.
float    bdrcGetLastLoss   (const BdrcSystem* sys);
uint32_t bdrcGetPhotonLive (const BdrcSystem* sys);  // lanes with density > 0 last frame
uint32_t bdrcGetGatherLive (const BdrcSystem* sys);
float    bdrcGetLastDensity(const BdrcSystem* sys);  // p_r read back from lane 0

void bdrcSetDreConfig(BdrcSystem* sys, const DreConfig& dre);

void bdrcSaveWeights(const BdrcSystem* sys, const std::string& path);
bool bdrcLoadWeights(BdrcSystem* sys, cudaStream_t stream, const std::string& path);

// One per-frame dispatch, after the camera, photon and gather launches have
// completed. There is no warmup and no phase: the single fit runs from frame 0.
void bdrcDispatch(
    BdrcSystem*  sys,
    cudaStream_t stream,
    float4*      bufferAccum,
    int          numPixels,
    unsigned int iteration,           // resets on camera move; drives the film blend
    unsigned int trainingIteration,   // never resets; drives the LR schedule
    unsigned int lrAnnealIters,
    int          debugMode,
    bool         freezeTraining);

#endif // BDRC_HOST_H
```

- [ ] **Step 3: Rewrite `bdrc_host.cu`**

Delete every old buffer, kernel and phase branch. The new file holds: the LR schedule helper (kept unchanged), one `assembleTargets` kernel, and the dispatch.

```c
// Builds the per-lane density-ratio record from the traced records and the
// tail's EMA field value.
//
//   photon lane: weight = clamped power, scalePos = 1/photonCount, scaleRef = 0
//   gather lane: weight = (E_dir + factor * U_EMA(tail)) / p_r,
//                scalePos = scaleRef = 1/gatherCount
//
// The gather weight's division by p_r is the conversion that matters: it turns
// a value at a point we chose into a sample of a measure, which is the only
// currency in which it can be added to the photon stream.
static __global__ void k_bdrc_assemble_targets(
    float*        weight,        // [3 x lanes] out
    float*        scalePos,      // [lanes] out
    float*        scaleRef,      // [lanes] out
    const float*  recBase,       // [3 x lanes]
    const float*  recDensity,    // [lanes]
    const float*  recFactor,     // [3 x lanes]
    const float*  recTailValid,  // [lanes]
    const float*  tailOutput,    // [3 x lanes] EMA field value at the tail
    uint32_t      photonCount,
    uint32_t      gatherCount,
    uint32_t      lanes)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lanes) return;

    weight[3*i+0] = 0.0f; weight[3*i+1] = 0.0f; weight[3*i+2] = 0.0f;
    scalePos[i] = 0.0f;
    scaleRef[i] = 0.0f;

    const float q = recDensity[i];
    if (!(q > 0.0f)) return;   // a disabled lane keeps its denominator, adds nothing

    if (i < photonCount)
    {
        const float invN = (photonCount > 0u) ? 1.0f / (float)photonCount : 0.0f;
        weight[3*i+0] = recBase[3*i+0];
        weight[3*i+1] = recBase[3*i+1];
        weight[3*i+2] = recBase[3*i+2];
        scalePos[i]   = invN;
        // scaleRef stays 0: the photon stream has no anchor pairing available
        // and works as a measure or not at all.
        return;
    }

    const float invN = (gatherCount > 0u) ? 1.0f / (float)gatherCount : 0.0f;
    const float tail = recTailValid[i];
    for (int c = 0; c < 3; ++c)
    {
        float v = recBase[3*i+c];
        if (tail > 0.0f)
            v += recFactor[3*i+c] * tailOutput[3*i+c];
        weight[3*i+c] = fmaxf(v, 0.0f) / q;
    }
    scalePos[i] = invN;
    scaleRef[i] = invN;
}

// Counts lanes with a live reference density, for the stream-count gate.
static __global__ void k_bdrc_count_live(
    const float* recDensity, uint32_t begin, uint32_t end, unsigned int* outCount)
{
    const uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    if (recDensity[i] > 0.0f) atomicAdd(outCount, 1u);
}
```

`bdrcDispatch` body, in the demo's order:

```c
void bdrcDispatch(BdrcSystem* sys, cudaStream_t stream, float4* bufferAccum,
                  int numPixels, unsigned int iteration, unsigned int trainingIteration,
                  unsigned int lrAnnealIters, int debugMode, bool freezeTraining)
{
    const uint32_t lanes       = sys->lanes;        // padded to 256
    const uint32_t photonCount = sys->photonCount;
    const uint32_t gatherCount = sys->gatherCount;

    // 1. Tail queries -> EMA inference. The teacher reads the tails; only the
    //    fit below touches the live weights. A bootstrapped estimator that ends
    //    by reading the field's own EMA is unbiased for the measure the current
    //    field induces, and the true solution is the iteration's fixed point.
    {
        tcnn::GPUMatrixDynamic<float> in(sys->d_recTailQuery, UNIFIED_INPUT_DIM, lanes,
                                         tcnn::MatrixLayout::ColumnMajor);
        tcnn::GPUMatrix<float> out(sys->d_tailOutput, 3u, lanes);
        sys->field->query(stream, in, out, /*useEmaWeights=*/true);
    }

    // 2. Assemble per-lane (weight, p_r, 1/N_pos, 1/N_ref).
    device_bdrc_assemble_targets(stream, sys, lanes, photonCount, gatherCount);

    // 3. One loss over the whole batch -> one backward -> one optimizer step.
    if (!freezeTraining)
    {
        sys->field->setLearningRate(
            bdrcLrScheduleFactor(trainingIteration, BDRC_LR_WARMUP_ITERS,
                                 lrAnnealIters, BDRC_LR_MIN_FACTOR));

        tcnn::GPUMatrixDynamic<float> in(sys->d_recQuery, UNIFIED_INPUT_DIM, lanes,
                                         tcnn::MatrixLayout::ColumnMajor);
        sys->m_lastLoss = sys->field->dreAccumGrad(
            stream, in, sys->d_weight, sys->d_recDensity,
            sys->d_scalePos, sys->d_scaleRef, lanes);
        sys->field->optimizerStep(stream);
        sys->field->updateEma(stream);
    }

    // 4. Camera inference (EMA) and composite.
    {
        tcnn::GPUMatrixDynamic<float> in(sys->d_camQuery, UNIFIED_INPUT_DIM, sys->camLanes,
                                         tcnn::MatrixLayout::ColumnMajor);
        tcnn::GPUMatrix<float> out(sys->d_camOutput, 3u, sys->camLanes);
        sys->field->query(stream, in, out, /*useEmaWeights=*/true);
    }

    if (debugMode == 0)
        device_bdrc_composite(stream, bufferAccum, sys->d_camPrefix, sys->d_camFactor,
                              sys->d_camOutput, sys->d_camValid, numPixels, iteration);
    else
        device_bdrc_debug(stream, bufferAccum, sys->d_camFactor, sys->d_camOutput,
                          sys->d_camValid, numPixels, iteration, debugMode);

    // 5. Stream counts for the gate, then clear the per-frame state.
    device_bdrc_count_live(stream, sys);

    cudaMemsetAsync(sys->d_recDensity,   0, lanes * sizeof(float), stream);
    cudaMemsetAsync(sys->d_recTailValid, 0, lanes * sizeof(float), stream);
    cudaMemsetAsync(sys->d_camValid,     0, sys->camLanes * sizeof(unsigned int), stream);
    cudaStreamSynchronize(stream);
}
```

Allocation sizes `lanes = padTo256(numPixels + gatherCount)`, `camLanes = padTo256(numPixels)`. Every records buffer and the camera buffers are `cudaMemset` to zero at allocation so a never-written padding lane decodes as a disabled lane rather than as uninitialized memory — the exact failure that produced the mask cache's iteration-~400 NaN.

- [ ] **Step 4: Delete the mask cache**

```bash
git rm src/cuda/cache/bdrc/mask_ngp_cache.h src/cuda/cache/bdrc/mask_ngp_cache.cu
```

Remove both from `CMakeLists.txt`.

- [ ] **Step 5: Build**

`Application.cpp` still calls the old `bdrcDispatch` signature, so the host link fails until Task 7. Compile the library target:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc) 2>&1 | grep -E "bdrc_host|unified_ngp|kernel_nrc|error" | head -40
```

Expected: no errors from `bdrc_host.cu`, `unified_ngp_cache.cu` or `kernel_nrc.cu`; the only errors name `Application.cpp`.

- [ ] **Step 6: Commit**

```bash
git add -A src/ CMakeLists.txt
git commit -m "feat: unified per-frame dispatch — EMA tails, one loss, one optimizer step

Removes the three-cache system, the two-phase schedule, the compaction
machinery and the virtual-emission fold-in. Both streams reach one backward
through one records array; invalid lanes keep their launch denominator."
```

---

## Task 7: Application wiring, the gather SBT and the GUI

**Files:**
- Modify: `src/Application.h`, `src/Application.cpp`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `PGID_GATHER_RAYGENERATION`, `m_sbtGather`, `m_bdrcGatherCount`, `m_bdrcDre`, `m_bdrcReferenceOrigin`.

- [ ] **Step 1: Program group and SBT**

In `src/Application.h` add `PGID_GATHER_RAYGENERATION,` to `ProgramGroupId` after `PGID_LIGHT_PATH_RAYGENERATION`, and `OptixShaderBindingTable m_sbtGather = {};` beside `m_sbtLightPath`.

In `Application.cpp::initPipeline()`, beside the existing light-path raygen description:

```cpp
  pgd = &programGroupDescriptions[PGID_GATHER_RAYGENERATION];
  pgd->kind  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgd->flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  pgd->raygen.module            = m_modules[MODULE_ID_RAYGENERATION];
  pgd->raygen.entryFunctionName = "__raygen__bdrc_gather";
```

In `initSBT()`, mirror the `m_sbtLightPath.raygenRecord` allocation for `m_sbtGather`, and copy the shared miss/hitgroup/callables fields wherever `m_sbtLightPath` copies them (both sites, ~line 6835 and ~line 6965).

- [ ] **Step 2: Members**

In `src/Application.h`, delete `m_bdrcPhase`, `m_bdrcWarmup`, `m_bdrcLastEyeLoss`, `m_bdrcLastLightLoss`, `m_bdrcLastMaskLoss` and their history arrays. Add:

```cpp
  int       m_bdrcGatherCount = 65536;   // gather lanes per frame; independent of resolution
  float     m_bdrcPhotonClamp = 10000.0f;
  DreConfig m_bdrcDre;                   // generator, alpha, beta, power
  float3    m_bdrcReferenceOrigin = {};  // stamped once in initBDRC
  float     m_bdrcLastLoss = 0.0f;
  uint32_t  m_bdrcPhotonLive = 0u;
  uint32_t  m_bdrcGatherLive = 0u;
  float     m_bdrcLastDensity = 0.0f;
  float     m_bdrcLossHistory[128] = {};
  int       m_bdrcLossHistoryIdx = 0;
```

- [ ] **Step 3: Stamp the reference origin and create the field**

In `initBDRC()`, replace the old three-cache creation (and remove the Task 3 self-test probes) with:

```cpp
  // The reference origin is stamped once, here, and never updated. It is NOT
  // the live viewing camera: a label taken against an interactive camera would
  // flip for every point the user orbits past, silently relabelling the whole
  // learned function mid-fit. The scene's bounding-sphere centre is a
  // deterministic, camera-independent choice.
  if (m_sceneExtent.isValid())
  {
    const glm::vec3 c = m_sceneExtent.getCenter();
    m_bdrcReferenceOrigin = make_float3(c.x, c.y, c.z);
  }
  m_launchParameters.nrc.referenceOrigin = m_bdrcReferenceOrigin;

  m_bdrcSystem = bdrcCreate(m_cudaStream, m_bdrcDre);
```

- [ ] **Step 4: Buffer wiring**

In the resolution-change path, replace the old BDRC wiring block with:

```cpp
  if (m_nrc == 2 && m_bdrcSystem)
  {
    bdrcAllocBuffers(m_bdrcSystem, numPixels, static_cast<uint32_t>(m_bdrcGatherCount));

    NrcLaunchBuffers& b = m_launchParameters.nrc;
    b.photonCount  = numPixels;
    b.gatherCount  = static_cast<uint32_t>(m_bdrcGatherCount);
    b.recQuery     = bdrcRecQuery(m_bdrcSystem);
    b.recBase      = bdrcRecBase(m_bdrcSystem);
    b.recDensity   = bdrcRecDensity(m_bdrcSystem);
    b.recTailQuery = bdrcRecTailQuery(m_bdrcSystem);
    b.recTailValid = bdrcRecTailValid(m_bdrcSystem);
    b.recFactor    = bdrcRecFactor(m_bdrcSystem);
    b.camQuery     = bdrcCamQuery(m_bdrcSystem);
    b.camPrefix    = bdrcCamPrefix(m_bdrcSystem);
    b.camFactor    = bdrcCamFactor(m_bdrcSystem);
    b.camValid     = bdrcCamValid(m_bdrcSystem);
    b.photonClamp  = m_bdrcPhotonClamp;

    // Halved: the field holds a value per (point, side), and an anchor takes
    // either side with probability 1/2.
    b.referenceDensity = 0.5f * m_surfaceRefInvTotalArea;
    b.referenceOrigin  = m_bdrcReferenceOrigin;
  }
```

- [ ] **Step 5: Launch the gather raygen and call the new dispatch**

In `render()`, inside the `m_nrc == 2` branch, after the light-path launch and before `bdrcDispatch`:

```cpp
        if (m_sbtGather.raygenRecord && m_bdrcGatherCount > 0)
        {
          OPTIX_CHECK( m_api.optixLaunch(m_pipeline, m_cudaStreamLight,
                                          reinterpret_cast<CUdeviceptr>(m_d_launchParameters),
                                          sizeof(LaunchParameters),
                                          &m_sbtGather,
                                          static_cast<unsigned int>(m_bdrcGatherCount), 1, 1) );
        }
```

Replace the `bdrcDispatch` call with the new signature, delete the phase update and the temporary pixel-scan diagnostics, and read back the new counters:

```cpp
        bdrcDispatch(m_bdrcSystem, m_cudaStream, m_launchParameters.bufferAccum,
                     m_resolution.x * m_resolution.y, nrcIteration,
                     m_nrcTrainingIteration,
                     static_cast<unsigned int>(m_bdrcLrAnnealIters),
                     m_bdrcDebugMode, m_bdrcFreezeTraining);

        m_bdrcLastLoss    = bdrcGetLastLoss(m_bdrcSystem);
        m_bdrcPhotonLive  = bdrcGetPhotonLive(m_bdrcSystem);
        m_bdrcGatherLive  = bdrcGetGatherLive(m_bdrcSystem);
        m_bdrcLastDensity = bdrcGetLastDensity(m_bdrcSystem);

        if (nrcIteration % 100u == 0u)
          printf("[bdrc] iter=%u loss=%.6g photonLive=%u/%u gatherLive=%u/%u p_r=%.6g\n",
                 nrcIteration, m_bdrcLastLoss,
                 m_bdrcPhotonLive, (unsigned)(m_resolution.x * m_resolution.y),
                 m_bdrcGatherLive, (unsigned)m_bdrcGatherCount,
                 m_bdrcLastDensity);

        m_bdrcLossHistory[m_bdrcLossHistoryIdx] = m_bdrcLastLoss;
        m_bdrcLossHistoryIdx = (m_bdrcLossHistoryIdx + 1) % 128;
```

- [ ] **Step 6: GUI**

Replace the BDRC panel in `guiWindow()`:

```cpp
    ImGui::Text("Unified transport cache");
    ImGui::Text("photon lanes live: %u / %u", m_bdrcPhotonLive, (unsigned)(m_resolution.x*m_resolution.y));
    ImGui::Text("gather lanes live: %u / %u", m_bdrcGatherLive, (unsigned)m_bdrcGatherCount);
    ImGui::Text("reference density p_r: %.6g", m_bdrcLastDensity);
    ImGui::Text("loss: %.6g", m_bdrcLastLoss);
    ImGui::PlotLines("loss", m_bdrcLossHistory, 128, m_bdrcLossHistoryIdx, nullptr,
                     FLT_MAX, FLT_MAX, ImVec2(0, 60));

    static const char* generatorItems[] = { "logistic", "brier", "lsif", "bregman" };
    int gen = static_cast<int>(m_bdrcDre.objective);
    bool dreDirty = false;
    dreDirty |= ImGui::Combo("Generator", &gen, generatorItems, IM_ARRAYSIZE(generatorItems));
    m_bdrcDre.objective = static_cast<DreObjective>(gen);
    dreDirty |= ImGui::SliderFloat("alpha (noise ratio)", &m_bdrcDre.alpha, 1.0f, 10000.0f, "%.0f",
                                   ImGuiSliderFlags_Logarithmic);
    dreDirty |= ImGui::SliderFloat("beta",  &m_bdrcDre.beta,  0.01f, 100.0f, "%.3f",
                                   ImGuiSliderFlags_Logarithmic);
    if (m_bdrcDre.objective == DreObjective::Bregman)
      dreDirty |= ImGui::SliderFloat("bregman power", &m_bdrcDre.power, -1.0f, 4.0f);

    if (dreDirty && m_bdrcSystem)
    {
      // The raw-ratio members read u* = (beta/alpha) * sum_s p_s w_bar_s, so
      // they reject unequal constants. Pin beta to alpha rather than throwing
      // at the user from a slider callback.
      if (m_bdrcDre.objective == DreObjective::Lsif || m_bdrcDre.objective == DreObjective::Bregman)
        m_bdrcDre.beta = m_bdrcDre.alpha;
      bdrcSetDreConfig(m_bdrcSystem, m_bdrcDre);
    }

    if (ImGui::SliderInt("Gather lanes", &m_bdrcGatherCount, 0, 1 << 20))
      m_isDirtyResolution = true;   // reallocates the records array
    ImGui::SliderFloat("Photon clamp", &m_bdrcPhotonClamp, 1.0f, 1.0e6f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Freeze training", &m_bdrcFreezeTraining);

    static const char* bdrcDebugItems[] = { "Normal", "Cache contribution", "Raw field" };
    ImGui::Combo("BDRC debug", &m_bdrcDebugMode, bdrcDebugItems, IM_ARRAYSIZE(bdrcDebugItems));
```

Default `m_bdrcFreezeTraining` to `false` — there is no warmup any more and the single fit runs from frame 0.

- [ ] **Step 7: Build**

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake --build cmake-build-release-system --target GLTF_renderer -j$(nproc)
```

Expected: links cleanly.

- [ ] **Step 8: Commit**

```bash
git add src/Application.h src/Application.cpp
git commit -m "feat: wire the unified cache — gather SBT, stamped reference origin, unified GUI

Removes the two-phase schedule and the warmup. The gather raygen launches at
its own lane count, independent of render resolution, so the gather stream
does not move when the window or the camera does."
```

---

## Task 8: Run the spec's verification gates

**Files:** none — this task only runs and observes.

- [ ] **Step 1: Stream-count gate**

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere.glb
```

Select BDRC, leave training on, watch the console.

Expected: `photonLive` and `gatherLive` are both nonzero and stable frame to frame (`gatherLive` should sit at or just below `m_bdrcGatherCount`; `photonLive` is a smaller fraction, since only `L S+ D` paths deposit). Then orbit the camera and confirm `gatherLive` does not change — a gather count that tracks camera framing means the anchor draw is reading the view.

- [ ] **Step 2: Reference-density gate**

Expected: the printed `p_r` is one positive constant, equal to `0.5 / totalSceneSurfaceArea`, and it does not change as the camera moves or between frames. Cross-check the scene's area against the AABB printed at load.

- [ ] **Step 3: Fit gate**

Expected: `loss` decreases over the first few hundred frames and then stabilises. It must not diverge, sit at exactly 0, or go NaN. Switch the debug view to "Raw field" and confirm a plausible spatial pattern — brighter under the light and on lit walls, dark in shadow — rather than a uniformly zero or uniformly saturated image.

- [ ] **Step 4: Transport gate**

Expected: in "Cache contribution" the caustic under the glass sphere is present. Then:

```bash
./cmake-build-release-system/bin/GLTF_renderer -f data/cornell_caustic_glass_sphere_sds.glb
```

Expected: the SDS scene's equivalent region is not darker than the simple scene's.

- [ ] **Step 5: Camera-independence gate**

With training running, orbit continuously for ~30 seconds, then stop and let the film re-accumulate. Expected: the cache-only view stays consistent with what was already learned — it must not visibly degrade or re-converge. A fit that re-converges on camera motion means a label, coordinate or reference density is reading the live camera. If this fails, check in order: `bdrcSignCamLabel`'s origin argument, `bdrcPackQuery5`'s AABB arguments, and `grep -rn cameraP src/cuda/raygen.cu src/cuda/hit.cu`.

- [ ] **Step 6: Empty-scene gate (Review Focus 1)**

```bash
./cmake-build-release-system/bin/GLTF_renderer
```

with no file, or with a scene that has no lights. Expected: no crash, no NaN in the console, `photonLive = 0`, `gatherLive = 0` (or gather live with `p_r > 0` and zero weights if geometry exists but no lights), and the loss reads `0`.

- [ ] **Step 7: Record the outcome**

The spec sets the expectation plainly: image quality is expected to be *worse* than the three-cache BDRC until the gather stream is correct, and a worse image at an intermediate task is not by itself evidence of a bug. Write down which gates passed and which did not, with the console numbers, before drawing any conclusion about quality.

- [ ] **Step 8: Update the project guide**

`CLAUDE.md`'s BDRC sections describe the three-cache system, the two-phase schedule, and the removed buffers. Rewrite the "Bidirectional Radiance Cache (BDRC)" section, the `NrcLaunchBuffers` subfields list, the `PerRayData` BDRC fields table, and the `Application` member list to match what now exists.

- [ ] **Step 9: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for the unified transport cache"
```
