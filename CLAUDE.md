# GLTF_renderer - Project Guide for Claude

## Project Overview

**GLTF_renderer** is an NVIDIA OptiX-based bidirectional path tracer for the [Khronos glTF 2.0](https://github.com/KhronosGroup/glTF/tree/main/specification/2.0) file format. It implements physically-based global illumination using **bidirectional path tracing (BDPT)** with three rendering modes:

1. **Pure path tracing** — progressive Monte Carlo with next event estimation (NEE/MIS)
2. **NRC** (Neural Radiance Cache) — unidirectional path tracing accelerated by a neural cache for indirect illumination (Mueller et al. 2021)
3. **BDRC** (Bidirectional Radiance Cache) — bidirectional path tracing with three neural caches for caustic-aware rendering

Built on NVIDIA OptiX SDK (7.7.0+) and CUDA, with tiny-cuda-nn for the neural network caches. Copyright NVIDIA Corporation 2025.

The renderer supports drag-and-drop of `*.gltf`/`*.glb` files and `*.hdr` environment maps, interactive GUI via Dear ImGui, and a broad set of glTF 2.0 material extensions.

---

## Build System

- **CMake** (minimum 3.31), C++17 required (`std::filesystem`).
- **Languages**: CXX + CUDA.
- **Build directory**: project root `build/`.
- **Output**: `build/bin/GLTF_renderer` (executable) + `build/bin/GLTF_renderer_core/*.optixir` (OptiX device modules).

### Required Dependencies

| Dependency       | Purpose                                         |
|------------------|-------------------------------------------------|
| CUDA Toolkit 12+ | CUDA runtime, driver, NVCC compiler             |
| OptiX SDK 7.7+   | Ray tracing API (9.1, 9.0, 8.1, 8.0, 7.7 supported) |
| OpenGL + GLEW    | Display and CUDA-GL interop                     |
| GLFW             | Window/input management                         |
| fastgltf         | glTF 2.0 parsing (`3rdparty/fastgltf`)          |
| simdjson         | JSON parsing (used by fastgltf)                 |
| GLM              | Math library (`3rdparty/glm`)                   |
| STB              | Image loading (`3rdparty/stb`)                  |
| DevIL 1.8.0      | Image loading (screenshots, HDR)                |
| Dear ImGui       | GUI (vendored in `imgui/`)                      |
| tiny-cuda-nn     | Neural network inference/training (`3rdparty/tiny-cuda-nn`) |

### Build Steps (Linux)

```bash
# Set CUDACXX env var (required on Linux)
export CUDACXX=/usr/local/cuda/bin/nvcc

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install          # copies .optixir modules to bin/GLTF_renderer_core/
cp ../data/* bin/     # copies runtime assets
```

**Important**: `CUDACXX` environment variable **must** be set on Linux, or CMake will abort with a fatal error.

### CUDA Architecture Targets

- CUDA 13+: SM 75, 80, 86, 89, 90, 120 (Maxwell dropped)
- CUDA 12.x: SM 60, 70, 75, 80, 86, 89, 90, 120
- Set `DEBUG_ARCH TRUE` in CMakeLists.txt to compile only the minimum architecture (faster iteration).

### OptiX Module Format

- Prefers **OptiX IR** (`.optixir`) when CUDA >= 11.7 + OptiX >= 7.5.
- Falls back to **PTX** (`.ptx`) otherwise.
- Controlled by `USE_OPTIX_IR` cmake variable and `#define USE_OPTIX_IR` compile definition.
- The `GLTF_renderer_optix_lib` CMake OBJECT library compiles the OptiX device programs separately with `CUDA_OPTIX_COMPILATION ON`.

---

## Directory Structure

```
BidirectionalPathTracer/
├── CMakeLists.txt              # Build configuration
├── CLAUDE.md                   # This file
├── README.md                   # User-facing overview
├── src/
│   ├── main.cpp                # Entry point: GLFW init, main loop, Options parsing
│   ├── Application.cpp/.h      # Central application class (see below)
│   ├── Animation.cpp/.h        # glTF animation system
│   ├── Arena.cpp/.h            # GPU memory arena allocator (cuda::ArenaAllocator)
│   ├── Camera.cpp/.h           # Camera types (perspective, orthographic)
│   ├── CheckMacros.h           # CUDA/OptiX error-checking macros
│   ├── ConversionArguments.h   # Helper for attribute conversion
│   ├── DeviceBuffer.h          # RAII wrapper for GPU buffer allocations
│   ├── HostBuffer.h            # CPU-side buffer for mesh attribute data
│   ├── HostKernels.h           # CPU-side kernel dispatch helpers
│   ├── Light.h                 # Light types (dev::Light)
│   ├── Logger.h                # OptiX log callback
│   ├── Mesh.cpp/.h             # Host/device mesh and primitive classes
│   ├── MyAssert.h              # MY_ASSERT macro
│   ├── Node.h                  # Scene node (transform, TRS)
│   ├── Options.cpp/.h          # Command-line argument parsing
│   ├── Picture.cpp/.h          # Image loading (DevIL)
│   ├── Record.h                # SBT record structures
│   ├── SceneExtent.h           # World-space AABB tracking
│   ├── Skin.h                  # Skeletal skinning data (dev::Skin)
│   ├── Texture.cpp/.h          # CUDA texture object management
│   ├── Trackball.cpp/.h        # Orbit/pan/dolly camera controller
│   ├── Utils.cpp/.h            # Timer and misc utilities
│   │
│   ├── cuda/                   # All GPU-side code
│   │   ├── config.h            # Global shader configuration defines
│   │   ├── launch_parameters.h # LaunchParameters + NrcLaunchBuffers structs
│   │   ├── per_ray_data.h      # PerRayData payload struct
│   │   ├── hit_group_data.h    # HitGroupData SBT record payload
│   │   ├── geometry_data.h     # GeometryData: vertex buffer pointers
│   │   ├── material_data.h     # MaterialData: all PBR + extension parameters
│   │   ├── light_definition.h  # LightDefinition: all light type parameters
│   │   ├── function_indices.h  # Direct callable program indices
│   │   ├── bxdf_common.h       # Shared BSDF sampling/evaluation functions
│   │   ├── shader_common.h     # Shared shading utilities
│   │   ├── transform.h         # Transform helpers (object-to-world, normals)
│   │   ├── random.h            # PCG random number generator
│   │   ├── vector_math.h       # float2/float3/float4 math operators
│   │   │
│   │   ├── raygen.cu           # Ray generation: eye paths, light paths, NRC/BDRC logic
│   │   ├── miss.cu             # Miss programs (constant env, spherical HDR env)
│   │   ├── hit.cu              # Closest-hit and any-hit programs (triangles + spheres)
│   │   ├── exception.cu        # OptiX exception program
│   │   ├── light_sample.cu     # Direct callable programs for light sampling
│   │   │
│   │   ├── kernel_morphing.cu  # Native CUDA kernel: GPU morph target blending
│   │   ├── kernel_skinning.cu  # Native CUDA kernel: GPU skeletal skinning
│   │   ├── kernel_blend.cu     # Native CUDA kernel: general blending
│   │   │
│   │   └── cache/
│   │       ├── nrc_features.h      # NRC/BDRC feature packing (14-dim NRC, 3-dim BDRC)
│   │       ├── cache_helpers.h/.cu # Shared GPU kernels: compaction, accumulation, splatting
│   │       │
│   │       ├── nrc/                # Unidirectional Neural Radiance Cache
│   │       │   ├── nrc_host.h/.cu  # NRC lifecycle: allocate, train, inference
│   │       │   ├── nrc_config.h    # NRC configuration constants
│   │       │   └── kernel_nrc.cu   # NRC-specific training target kernels
│   │       │
│   │       └── bdrc/               # Bidirectional Radiance Cache system
│   │           ├── bdrc_host.h/.cu # BDRC lifecycle and per-frame dispatch
│   │           ├── base_ngp_cache.h/.cu  # Base hash-grid + MLP cache (tcnn)
│   │           ├── mask_ngp_cache.h/.cu  # Mask cache (DRE Brier loss)
│   │           ├── ngp_config.h    # Hash-grid / MLP / optimizer configuration
│   │           └── grad_scaler.h   # Loss-scale gradient scaler
│   │
│   └── imgui/                  # Dear ImGui (vendored, v1.9x)
│
├── data/                       # Runtime assets
│   ├── sheen_lut.hdr           # Precomputed sheen sampling weight LUT
│   └── *.glb                   # Test scenes (Cornell box variants, etc.)
│
└── doc/                        # Documentation and screenshots
    └── README.md               # Full glTF extension and feature reference
```

---

## Architecture Deep-Dive

### Application Class (`Application.h/.cpp`)

The `Application` class is the central orchestrator. Key responsibilities:

**Initialization sequence** (called from constructor):
1. `initOpenGL()` / `initGLSL()` — OpenGL display pipeline
2. `initCUDA()` — CUDA context, stream, UUID check for interop
3. `initOptiX()` — OptiX context, pipeline, SBT
4. `loadGLTF()` — Parse asset with fastgltf
5. `initRenderer()` — Build acceleration structures, upload data
6. `initNRC()` / `initBDRC()` — Allocate neural cache systems when enabled

**Per-frame loop** (`main.cpp`):
```
render()          -> optixLaunch() eye + light paths (async)
bdrcDispatch()    -> compaction, inference, training (BDRC mode)
nrcDispatch()     -> inference + training (NRC mode)
update()          -> CUDA-GL interop copy to display texture
guiWindow()       -> all GUI panels including cache debug views
display()         -> OpenGL fullscreen quad
```

**Key BDRC / NRC member data** (`Application.h`):
```cpp
NrcSystem*   m_nrcSystem    = nullptr;
BdrcSystem*  m_bdrcSystem   = nullptr;
int          m_nrc          = 0;     // 0=off, 1=NRC, 2=BDRC
int          m_bdrcGatherCount   = 65536; // gather lanes/frame, independent of resolution
float        m_bdrcPhotonClamp   = 10000.0f;
DreConfig    m_bdrcDre;                    // generator, alpha, beta, power
float3       m_bdrcReferenceOrigin = {};   // stamped once; never follows the camera
int          m_bdrcDebugMode = 0;    // 0=normal, 1=cache contribution, 2=raw field
bool         m_bdrcFreezeTraining = false; // no warmup: the fit runs from frame 0
uint32_t     m_nrcTrainingIteration = 0;   // Total training frames, never reset
```

### Acceleration Structure Pipeline

```
HostMesh (CPU) -> DeviceMesh (GPU GAS) -> Instance (transform + GAS index)
                                              |
                                      OptixInstance array
                                              |
                                       IAS (root traversable)
```

- `buildDeviceMeshAccel()` — builds per-mesh GAS (Geometry Acceleration Structure)
- `buildInstanceAccel()` — builds root IAS from all instances
- `KeyTuple` (`idxHostMesh`, `idxSkin`, `idxNode`) maps unique (mesh, skin, morph-state) combos to device meshes

### Mesh / Primitive System (`Mesh.h`)

The namespace `dev` separates from `fastgltf::` types:

- `HostPrimitive` — CPU buffers: indices, positions, normals, tangents, colors, texcoords[2], joints[2], weights[2], plus morph target variants
- `DevicePrimitive` — GPU `DeviceBuffer`s mirroring the above, plus morphed/skinned output buffers
- `DeviceMesh` — owns a GAS handle + list of `DevicePrimitive`s
- `HostMesh` — list of `HostPrimitive`s with morph weights
- Primitive types: `Triangles` (triangle meshes) and `Points` (rendered as OptiX built-in spheres)

### OptiX Pipeline (`Application.h` — `initPipeline()` / `initSBT()`)

**Modules** (compiled separately as OptiX IR/PTX):
| ID | File | Purpose |
|----|------|---------|
| `MODULE_ID_RAYGENERATION` | `raygen.cu` | Eye + light path tracing loop |
| `MODULE_ID_EXCEPTION` | `exception.cu` | Debug exception handler |
| `MODULE_ID_MISS` | `miss.cu` | Environment lighting (constant / HDR sphere) |
| `MODULE_ID_HIT` | `hit.cu` | Surface shading (triangles + spheres) |
| `MODULE_ID_LIGHT_SAMPLE` | `light_sample.cu` | Direct callables for light sampling |

**Program Groups** (`ProgramGroupId` enum):
- Raygen, Exception
- Miss Radiance, Miss Shadow
- Hit Radiance Triangles, Hit Shadow Triangles
- Hit Radiance Spheres, Hit Shadow Spheres
- Direct Callables: `PGID_LIGHT_ENV_CONSTANT`, `PGID_LIGHT_ENV_SPHERE`, `PGID_LIGHT_POINT`, `PGID_LIGHT_SPOT`, `PGID_LIGHT_DIRECTIONAL`

**SBT layout**: One raygen record, two miss records (radiance/shadow), hit records per mesh primitive (one per alpha mode variant), direct callable records per light type.

### Material System (`cuda/material_data.h`)

`MaterialData` covers the full glTF 2.0 PBR model + extensions:

| Feature | Extension |
|---------|-----------|
| PBR Metallic-Roughness | Core |
| IOR | `KHR_materials_ior` |
| Specular | `KHR_materials_specular` |
| Transmission | `KHR_materials_transmission` |
| Volume | `KHR_materials_volume` |
| Clearcoat | `KHR_materials_clearcoat` |
| Anisotropy | `KHR_materials_anisotropy` |
| Sheen | `KHR_materials_sheen` |
| Iridescence | `KHR_materials_iridescence` |
| Emissive Strength | `KHR_materials_emissive_strength` |
| Unlit | `KHR_materials_unlit` |
| Texture Transform | `KHR_texture_transform` |
| Alpha modes | Opaque / Mask / Blend |

Extension flags are stored as a bitfield (`flags`) using `FLAG_KHR_*` defines.

The sheen model uses a precomputed LUT: `data/sheen_lut.hdr` (loaded in `initSheenLUT()`).

### Animation System (`Animation.h/.cpp`)

- Supports time-based (real-time) and key-frame scrubbing modes
- `dev::Animation` holds samplers (input/output buffers, interpolation mode) and channels (target node, path)
- Morph target blending: `updateMorph()` → `kernel_morphing.cu` (GPU)
- Skeletal skinning: `updateSkin()` → `kernel_skinning.cu` (GPU), joint matrices computed on CPU

### Memory Management

- `cuda::ArenaAllocator` (`Arena.h`): custom GPU memory pool, default 128 MiB arenas
  - `USAGE_STATIC` — long-lived allocations (geometry, textures)
  - `USAGE_TEMP` — temporary allocations (AS build scratch)
- `DeviceBuffer` (`DeviceBuffer.h`): RAII wrapper with `d_ptr` and `size`, supports `grow()` for resizable allocations

### OpenGL-CUDA Interop Modes (`InteropMode` enum)

| Mode | Description | Performance |
|------|-------------|-------------|
| `INTEROP_OFF` (0) | Device-to-host + host-to-device via `glTexSubImage2D` | Slowest, most compatible |
| `INTEROP_PBO` (1) | Render into mapped PBO, device-to-device copy | Medium |
| `INTEROP_TEX` (2) | Device-to-device copy to CUDA texture array | Fast |
| `INTEROP_IMG` (3) | Render directly into mapped CUDA surface object | Fastest |

---

## Neural Radiance Cache (NRC)

NRC (Mueller et al. 2021) accelerates path tracing by caching indirect radiance in a small neural network. When `m_nrc == 1`:

- Each eye path is terminated early at depth `terminationDepth` (or by area-spread criterion).
- The surface state at termination (14-dim feature vector) is queried against the neural cache.
- A fraction of pixels (`NRC_TRAIN_FRACTION`) trace additional suffix bounces to build training targets.
- The cache is trained online each frame with those suffix-bootstrapped targets.

**Feature vector** (`cache/nrc_features.h`, 14 dimensions):
```
[0–2]   Position (AABB-normalized)
[3–4]   Shading normal (spherical angles, [0,1])
[5–6]   View direction (spherical angles, [0,1])
[7]     Linear roughness
[8–10]  Diffuse base color (RGB)
[11–13] Specular color / F0 proxy (RGB)
```

**Training buffers** (`NrcLaunchBuffers` in `launch_parameters.h`):
- `renderFeatures` — pixel-indexed feature vectors at termination vertex K
- `trainNextFeatures` — suffix terminal features (slot-indexed)
- `trainSuffixScaled` — `T(cam→K) × L_suffix` accumulated over suffix
- `trainThroughputAtK` — `T(cam→K)` pre-BSDF at vertex K
- `trainCount` — atomic counter for compact training slots

**Config** (`config.h`):
```c
NRC_TRAIN_FRACTION  0.03f   // ~3% of pixels generate training data
NRC_SUFFIX_STEPS    3       // extra bounces per training path
NRC_UNBIASED_RATE   16      // 1-in-16 paths bypass cache (prevents drift)
```

---

## Unified Transport Cache (BDRC mode, `m_nrc == 2`)

One neural field replaces the former three-network system (eye colour cache,
light colour cache, mask density cache). The field stores **incident flux per
unit world area** on diffuse receivers — not radiance, not outgoing, with no
BSDF baked in. The receiver BSDF is applied at the query, in the renderer, on
every path that reads the field.

Source of truth for the design: `docs/superpowers/specs/2026-09-22-unified-single-cache-design.md`.

### The field (`bdrc/unified_ngp_cache.h/.cu`)

| Property | Value |
|---|---|
| Input | 5 dims: `[0..2]` world position normalized into the scene AABB (hash grid), `[3]` `sign_cam`, `[4]` arrival side (both Identity) |
| Encoding | tcnn `Composite`/`Concatenation`: nested `HashGrid` (8 levels x 2 features, 16..~273, log2 hashmap 20) + nested `Identity` for the two labels |
| MLP | FullyFusedMLP, 2 x 64, LeakyReLU hidden, **no output activation** |
| Output | 3 channels; softplus is applied by `unified_ngp_cache::query()` and differentiated analytically by the loss kernel |
| Optimizer | Adam, lr 5e-3, gradClip 0.05, gradScaler on with growth, weightClip 1.0 |
| EMA | decay 0.99, start 100 |

`tcnn` must not apply softplus itself: the loss differentiates through it, and a
second application would square the activation.

### Decomposition

With `S` any vertex without a diffuse lobe:

```
U(x) = C(x) + T(x)

C(x)  caustic term: flux per area deposited at x by emitter walks whose
      vertices before x are all non-diffuse and number at least one (L S+ D)

T(x)  everything else, as one gather segment:
      T(x) = E_dir(x) + integral f_s(y) U(y) |cos| dw
```

The families `L -> D`, `L -> S+ -> D` and `L -> ... -> D -> S* -> D` are
disjoint and complete, and the recursion through `U(y)` carries every longer
path. Disjoint path families induce measures that add, so one network
supervised by both memorises their sum — neither stream needs the other's
density.

### The three streams (`cuda/raygen.cu`)

| Stream | Raygen | Lanes | Records |
|---|---|---|---|
| Camera | `__raygen__path_tracer` (BDRC branch of `integrator()`) | `numPixels` | `camQuery`, `camPrefix`, `camFactor = T(cam->D) * f_s(D)`, `camValid` |
| Photon | `__raygen__light_path_tracer` -> `bdrcPhotonIntegrator` | `m_bdrcPhotonCount` (default 262144), 1-D launch | `recQuery`, `recBase` = raw clamped power, `recDensity` |
| Gather | `__raygen__bdrc_gather` (own SBT `m_sbtGather`) | `m_bdrcGatherCount`, its own launch dimension | `recQuery` (anchor), `recBase` = E_dir + chain radiance, `recTailQuery`, `recFactor`, `recTailValid` |

**Camera** traces the prefix with ordinary NEE and MIS, stops at the first
diffuse vertex `D`, and records one query there. The pixel reads
`prefix + T(cam->D) * f_s(D) * U(D)`. There is no suppression and no
continuation: the path simply ends, so it can never traverse a `D S+` chain.

**Photon** keeps the ungated `L S+ D` grammar — one deposit per path, no shadow
ray, no dependence on the camera. It stores raw clamped power.

**Gather** draws an anchor from the global area-uniform surface CDF table,
picks a side with probability 1/2, estimates direct light there with a **unit
BSDF** (`bdrcUnitBsdfDirect`, so the estimate is incident flux rather than
outgoing radiance), then cosine-samples the hemisphere with `throughput = pi`
and continues through any non-diffuse chain to the first diffuse vertex `y`,
recording `factor = throughput * f_s(y)`. The anchor is both the reference
anchor and the gather start, which makes the gathered half of the fit a
regression at known points.

### Labels (`bdrc/unified_features.h`)

- `sign_cam` = `sign(dot(ng, referenceOrigin - x))`, where `referenceOrigin` is
  stamped once in `initBDRC()` from the scene bounding-sphere centre and
  **never** follows the viewing camera. A label taken against an interactive
  camera would flip for every point the user orbits past, silently relabelling
  the whole learned function mid-fit.
- arrival side = `prd.bdrcFrontFace`, captured in `hit.cu` **before** the
  normals are flipped to the ray side. The flipped normal cannot serve:
  `dot(flippedN, wo)` is non-negative by construction.

Layer and side stay separate inputs; their product would give a thin shell's
dark inner back layer the label of its lit outer front.

### The loss (`k_unified_dre_grad`)

One weighted density-ratio objective over both positive streams against one
surface reference measure. Per lane, per channel, with `u` the softplus density
and `q` the lane's reference density:

| Generator | positive term | reference term |
|---|---|---|
| logistic | `softplus(-logit)` | `softplus(logit)` |
| brier | `sigmoid(-logit)^2` | `sigmoid(logit)^2` |
| lsif | `-r` | `r^2 / 2` |
| bregman(a) | `-(r^(a-1) - 1)/(a-1)` | `(r^a - 1)/a` |

with `logit = log u + log beta - log q - log alpha` and `r = u/q`. Gradients are
taken with respect to `log u` and chained through softplus; the `raw < -20`
branch keeps `log u` finite for an untrained cell.

Because `dL/dOutput` is per-sample, photon lanes and gather lanes never alias:
both streams reach **one** `backward` and **one** optimizer step, with no
`Overwrite`/`Accumulate` sequencing.

### Per-frame dispatch (`bdrcDispatch`)

```
optixLaunch  eye paths      -> camera records
optixLaunch  photon paths   -> photon records      (m_cudaStreamLight)
optixLaunch  gather paths   -> gather records      (m_cudaStreamLight, gatherCount lanes)
EMA inference over recTailQuery
k_bdrc_assemble_targets     -> per-lane (RGB weight, p_r, 1/N_pos, 1/N_ref)
dreAccumGrad                -> one optimizer step -> EMA update
EMA inference over camQuery -> composite: camPrefix + camFactor * U
```

There is no two-phase warmup: the single fit runs from frame 0.

### Numerical contracts

1. **The objective stays plain.** No per-stream reweighting of any kind — it
   would move the fixed point off the stream sum.
2. **Raw-ratio members require `alpha == beta`.** `lsif` and `bregman` read
   `u* = (beta/alpha) * sum_s p_s w_bar_s`; the constructor rejects unequal
   constants and the GUI pins `beta` to `alpha`.
3. **`alpha` is the noise ratio.** The field runs 1e2-1e4 times the reference
   density, so at `alpha = 1` the posterior saturates. Default 1000.
4. **Each stream is normalized by its own launch count.** There is no
   compaction: an invalid lane writes `recDensity = 0`, contributes nothing, and
   keeps its denominator.
5. **Reference density is per unit world surface area, halved**
   (`0.5 * surfaceRef.invTotalArea`) because the field holds a value per
   (point, side).
6. **EMA teacher, live student.** Tail reads and every render/debug query pass
   `useEmaWeights = true`.
7. **The stored quantity is incident flux per area.** The receiver BSDF is
   applied at the query.
8. **The cache lives on diffuse lobes only.**
9. **Training is camera-independent.** Moving the viewing camera changes rays
   and film accumulation only.

### Buffer layout (`NrcLaunchBuffers` in `launch_parameters.h`)

Training lanes are one fixed-slot array: `[0, photonCount)` photons,
`[photonCount, photonCount + gatherCount)` gathers.

| Buffer | Dim | Content |
|---|---|---|
| `recQuery` | 5 x lanes | start/deposit query (position + 2 labels) |
| `recBase` | 3 x lanes | photon: clamped power. gather: `E_dir` + chain radiance |
| `recDensity` | lanes | `p_r`; 0 disables the lane |
| `recTailQuery` | 5 x lanes | gather tail query |
| `recTailValid` | lanes | 1 when the tail query is meaningful |
| `recFactor` | 3 x lanes | gather: `throughput * f_s` at the tail |
| `camQuery` | 5 x N | query at the first diffuse vertex D |
| `camPrefix` | 3 x N | radiance accumulated before and at D |
| `camFactor` | 3 x N | `T(cam->D) * f_s(D)` |
| `camValid` | N | 0 / 1 |
| `referenceDensity` | scalar | `0.5 * invTotalArea` |
| `referenceOrigin` | float3 | stamped once, never the live camera |
| `photonClamp` | scalar | upper bound on stored photon power |

`SurfaceReferenceTable` holds **world-space** triangles plus an area CDF and
`invTotalArea`. World space, not GAS-local: one field spans the whole scene, and
the gather stream shoots shadow and continuation rays from these anchors.

### Diagnostics

`bdrcDispatch` prints every 100 frames and the GUI mirrors it:

```
[bdrc] iter=N loss=L photonLive=a/b gatherLive=c/d p_r=P field(mean=M max=X nonzero=n/m)
```

`gatherLive` must not change when the camera moves; `p_r` must be one positive
constant; `field(mean,max)` distinguishes a plausible spatial pattern from a
uniformly zero or saturated one.

### Startup self-test (`unifiedSelfTestAll`)

Runs at `initBDRC()` when the probe is enabled. Places both streams at one point
so the analytic fixed point is `5q`, then checks every generator at two
conditioning regimes. At `q = 1e-1` all four land on `5q`. At the realistic
`q = 1e-4` only the posterior members land — the raw-ratio members' `r^2`
reference term is heavy-tailed when `u/q` spans orders of magnitude, which is a
property of the parameterization, not a defect, and is why the GUI defaults to
logistic.

## Shader Configuration (`cuda/config.h`)

Key compile-time switches:
- `USE_NEXT_EVENT_ESTIMATION 1` — MIS-weighted direct lighting (default) vs brute-force
- `USE_DEBUG_EXCEPTIONS 0` — Enable OptiX exception program (major perf hit when 1)
- `MICROFACET_MIN_ROUGHNESS 0.0014142f` — Below this, BSDF switches to specular
- `SCENE_EPSILON_SCALE 1.0e-7f` — Self-intersection epsilon scale
- `RT_DEFAULT_MAX 1.e16f` — Default ray tmax / "infinity"
- `NRC_TRAIN_FRACTION 0.03f` — Fraction of NRC-terminated pixels that trace training suffixes
- `NRC_SUFFIX_STEPS 3` — Extra bounces per training suffix path
- `NRC_UNBIASED_RATE 16` — 1-in-N training paths bypass cache bootstrap

`MAX_TRACE_DEPTH 2` is defined in `Application.h` — controls maximum ray recursion depth for the OptiX pipeline.

---

## LaunchParameters (`cuda/launch_parameters.h`)

Uploaded to device as `__constant__ LaunchParameters theLaunchParameters` in every `.cu` program. Contains:
- `handle` — root IAS traversable
- `lightDefinitions` / `numLights` — light array pointer
- `textureSheenLUT` — sheen weight LUT texture
- `surface` / `bufferAccum` / `bufferAccumLightPath` — output buffers
- `bufferPicking` — single int for material picking result
- `resolution`, `picking`, `iteration`, `sceneEpsilon`, `pathLengths`
- Render overrides: `directLighting`, `ambientOcclusion`, `showEnvironment`, `forceUnlit`
- Camera params: `cameraType`, `cameraP/U/V/W`
- `nrc` — full `NrcLaunchBuffers` struct (see above)

### `NrcLaunchBuffers` subfields:
- `enabled` — 0 = off, 1 = active
- `nrcType` — `NRC` or `BDRC`
- `nrcDebugMode` — NRC: 0 = normal, 1 = cache-only. BDRC: 0 = normal,
  1 = cache contribution (`camFactor * U`), 2 = raw field (`U`)
- `spreadC` / `terminationDepth` — NRC area-spread termination
- `aabbMin` / `aabbMax` — scene bounding box for position normalization
- All unified-cache buffers listed in the Buffer layout table above

## PerRayData (`cuda/per_ray_data.h`)

Carried as an OptiX payload pointer through raygen → hit → miss programs.

**Core fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `stack` | `MaterialStack[4]` | Nested material IOR/absorption stack |
| `idxStack` | `int` | Top of material stack |
| `seed` | `uint` | PCG RNG state |
| `pos` | `float3` | Current hit position (world space) |
| `wo` / `wi` | `float3` | Outgoing / incoming directions |
| `radiance` | `float3` | Accumulated radiance |
| `throughput` | `float3` | Path throughput (modulated by BSDF/PDF each bounce) |
| `pdf` | `float` | Last BSDF sample PDF (for MIS) |
| `flags` | `uint` | `FLAG_HIT`, `FLAG_SHADOW`, `FLAG_LIGHT_PATH` |
| `typeEvent` | `BsdfEventType` | Last BSDF event (DIFFUSE / GLOSSY / SPECULAR / REFLECTION / TRANSMISSION) |

**NRC fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `nrcN` | `float3` | Shading normal at termination vertex |
| `nrcRoughness` | `float` | Linear roughness at termination vertex |
| `nrcBaseColor` | `float3` | Diffuse base color at termination vertex |
| `nrcSpecColor` | `float3` | Specular F0 proxy at termination vertex |
| `nrcTerminated` | `bool` | Set when path was NRC-terminated |
| `throughputPreBsdf` | `float3` | `T(cam→this_vertex)` before BSDF application |

**BDRC fields**:
| Field | Type | Purpose |
|-------|------|---------|
| `bdrcNg` | `float3` | Raw, unflipped world-space geometric normal; feeds the `sign_cam` label |
| `bdrcFrontFace` | `bool` | Which side the ray arrived on; feeds the arrival-side label |

Both are captured in `hit.cu` immediately before the normals are flipped to the
ray side, because neither fact is recoverable afterwards.

**`BsdfEventType` flags**:
```c
BSDF_EVENT_DIFFUSE   = 1
BSDF_EVENT_GLOSSY    = 1 << 1   // GGX microfacet (rough surfaces)
BSDF_EVENT_SPECULAR  = 1 << 2   // Ideal mirror/glass (zero roughness)
BSDF_EVENT_REFLECTION   = 1 << 3
BSDF_EVENT_TRANSMISSION = 1 << 4
```
Note: `BSDF_EVENT_GLOSSY` and `BSDF_EVENT_SPECULAR` are distinct bits. The caustic suppression check uses both: `(typeEvent & (BSDF_EVENT_SPECULAR | BSDF_EVENT_GLOSSY)) != 0`.

---

## Neural Network Architecture (`bdrc/ngp_config.h`)

`ngp_config` builds the tcnn JSON. `labelDims > 0` switches `toEncodingJson()`
from a bare `HashGrid` to a `Composite`/`Concatenation` whose nested `HashGrid`
takes the leading `inputDim - labelDims` dims and whose nested `Identity` takes
the trailing label dims. Labels must not be hashed: a label is a discrete
condition on the function, not a position whose neighbourhood should be
interpolated.

`weightClip` is passed to Adam as `clipping_magnitude`. Unlike gradient
clipping it is unconditional — no weight the optimizer writes can overflow to
Inf on the fp32 -> half cast. That is the exact mechanism behind the
reproducible NaN by iteration ~400 that gradClip alone did not prevent.

The unified field's own configuration is in `makeUnifiedConfig()`
(`unified_ngp_cache.cu`); see the table in the Unified Transport Cache section.

## Command-Line Options

```
--file    (-f) <path>    glTF file to load
--width   (-w) <int>     Window client width
--height  (-h) <int>     Window client height
--launches (-l) <int>    Async launches per render call (default 1, max 1000)
--interop (-i) <0-3>     OpenGL-CUDA interop mode (default 0)
--punctual               Enable KHR_lights_punctual (default true)
--miss    (-m) <0|1|2>   Miss shader: 0=null, 1=constant, 2=spherical HDR env
--env     (-e) <path>    HDR environment map for miss mode 2
--sphere-radius <float>  Sphere radius fraction for point clouds (default 0.005)
```

---

## Key Coding Patterns

- **`dev::` namespace** — used for all application-side types to avoid collision with `fastgltf::` types
- **Move-only buffers** — `DeviceBuffer`, `HostBuffer`, `DevicePrimitive`, `DeviceMesh`, `HostMesh` are move-only (no copy), to safely manage GPU memory ownership
- **`isDirty` flags** — used on `HostMesh`, `DeviceMesh`, `Camera`, and global booleans (`m_isDirtyScene`, `m_isDirtyResolution`, `m_isDirtyLights`) to minimize unnecessary GPU work
- **`KeyTuple`** — maps (hostMeshIndex, skinIndex, nodeIndex) to unique device meshes; same mesh instanced with different skins/morph states gets separate GAS
- **Dual material arrays** — `m_materialsOrg` holds original asset values; `m_materials` holds GUI-modified values
- **Picking** — `bufferPicking` (single int on device) receives material index from closest-hit; `float2 picking` in LaunchParameters holds normalized screen coordinate
- **Benchmarking** — `BenchmarkMode::FPS` or `SAMPLES_PER_SECOND`, running average over last 100 values
- **BDRC buffer ownership** — all BDRC device buffers are owned by `BdrcSystem` (opaque struct in `bdrc_host.cu`); `Application.cpp` accesses them only through the accessor functions in `bdrc_host.h`