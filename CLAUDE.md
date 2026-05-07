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
int          m_bdrcPhase    = 0;     // 0=Phase 1 (mask warmup), 1=Phase 2 (color)
int          m_bdrcWarmup   = 2000;  // Frames before Phase 2 starts
int          m_bdrcDebugMode = 0;    // 0=normal, 1=eye, 2=light, 3=mask, 4=eye+light
bool         m_bdrcFreezeTraining = true; // Freeze weights; inference still runs
uint32_t     m_nrcTrainingIteration = 0;  // Total training frames, never reset
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

## Bidirectional Radiance Cache (BDRC)

BDRC extends NRC with bidirectional path tracing to correctly render caustics (light → specular → diffuse paths). When `m_nrc == 2`:

### Three-Cache Architecture

| Cache | Neural Net | Purpose |
|-------|-----------|---------|
| **Eye Color Cache** | `base_ngp_cache` (3-ch, ReLU, RelativeL2) | Caches `c_e(D)` — incoming radiance at first diffuse vertex D from eye paths |
| **Light Color Cache** | `base_ngp_cache` (3-ch, Softplus output, RelativeL2) | Caches caustic transport `c_l(Y)` — light-path contribution at caustic floor Y |
| **Mask Cache** | `mask_ngp_cache` (1-ch, Softplus output, DRE Brier loss) | Estimates `p_l(x)` — PDF that a point x lies on a light path's caustic floor |

All three caches use **hash-grid + FullyFusedMLP** via tiny-cuda-nn, with 3D position as input (AABB-normalized). See `ngp_config.h` for architecture defaults (6-level hash grid, 64-dim hidden MLP, Adam optimizer).

### Two-Phase Training (`bdrcPhase`)

**Phase 1 (`bdrcPhase == 0`)** — Mask warmup (~2000 frames):
- Mask cache trains to distinguish light-path endpoints from eye-path endpoints (DRE).
- Eye and light color caches run inference (EMA query) but are not trained.
- The loss combines eye-sample and light-sample Brier terms, weighted by `N_light / N_eye`.

**Phase 2 (`bdrcPhase == 1`)** — Color training:
- Mask cache is frozen (EMA-only updates, no gradient steps).
- Eye color cache trains on: `c_e(D) = (L_direct_at_D + L_indirect) / T(cam→D)`
- Light color cache trains on caustic transport: `c_l(Y) = T(cam→Y_pre) × f_Y_toward_specular`

### BDRC Rendering Equation

For each pixel:

```
pixel = renderPrefix + T(cam→D) × eyeCache.query(D)
```

where:
- `renderPrefix` = accumulated radiance before the NRC termination vertex D (includes NEE at all vertices up to and including D)
- `T(cam→D)` = path throughput pre-BSDF at D (`throughputPreBsdf`)
- `eyeCache.query(D)` = neural cache output `c_e(D)`

### Eye Path — Caustic Suppression

To avoid double-counting with the light cache (which handles `light→specular→diffuse` caustic paths), eye paths suppress contributions when traversing a `diffuse→specular` chain:

```cpp
// per_ray_data.h
bool eyePathFoundDiffuse;    // latches true once any diffuse vertex is hit
bool eyePathInSpecularChain; // true if previous BSDF bounce was specular or glossy
```

Hit.cu sets these after each BSDF sample:
```cpp
prd->eyePathFoundDiffuse    |= (typeEvent & BSDF_EVENT_DIFFUSE) != 0;
prd->eyePathInSpecularChain  = (typeEvent & (BSDF_EVENT_SPECULAR | BSDF_EVENT_GLOSSY)) != 0;
```

Emission and NEE at vertex `x` are suppressed when `eyePathFoundDiffuse && eyePathInSpecularChain`.

### Light Path Generation

Light paths are launched alongside eye paths with `FLAG_LIGHT_PATH` set in `prd.flags`. The raygen:
1. Samples the environment or area light for the light path origin.
2. Traces the light path until a specular-then-diffuse vertex Y is found.
3. Connects Y to the eye path's diffuse vertex D via `bdrcLightIntegrator()`.
4. Writes `lightEndpoints[pixelIdx]`, `lightTargets[pixelIdx]`, and PDFs into the BDRC buffers.

### Per-Frame Dispatch (`bdrcDispatch`)

Called once per frame after both launches complete:

1. **Compaction** — scatter-to-compact for sparse `lightValid` and `resampledValid` arrays.
2. **Inference** — eye cache query on render features, light cache query on light endpoints.
3. **Phase 1 training** — mask cache Brier loss step (eye + light samples).
4. **Phase 2 training** — eye and light color cache training steps (if not frozen).
5. **Compositing** — `bufferAccum += renderPrefix + T × eyeCacheOutput + lightCacheOutput`.

Training can be paused via `m_bdrcFreezeTraining` (GUI toggle) while inference continues.

### BDRC Buffer Layout (`NrcLaunchBuffers` in `launch_parameters.h`)

All buffers are `[dim × numPixels]` (pixel-indexed before compaction) or `[dim × batchSize]` (slot-indexed after compaction):

**Eye-path buffers** (written by `integrator()` in raygen.cu):
| Buffer | Dim | Content |
|--------|-----|---------|
| `renderFeatures` | 14 × N | NRC feature vector at termination vertex |
| `renderPrefix` | 3 × N | Radiance before vertex D (includes NEE at D) |
| `renderThroughput` | 3 × N | `T(cam→D)` pre-BSDF |
| `renderValid` | N | 1 if pixel has valid termination |
| `trainNextFeatures` | 14 × B | Suffix terminal features |
| `trainSuffixScaled` | 3 × B | `T(cam→D) × L_suffix` |
| `trainThroughputAtK` | 3 × B | `T(cam→D)` pre-BSDF (training denominator) |
| `trainNextValid` | B | 1 if suffix is valid |

**Light-path buffers** (written by `bdrcLightIntegrator()` in raygen.cu):
| Buffer | Dim | Content |
|--------|-----|---------|
| `lightEndpoints` | 3 × N | AABB-normalized position of caustic floor Y |
| `lightTargets` | 3 × N | `T(cam→Y_pre) × f_Y_toward_S` |
| `lightValid` | N | 1 if this pixel has a valid caustic sample |
| `light_p_sa` | N | Hemisphere PDF at Y (analytic) |
| `light_p_e_area` | N | Eye-path area PDF at Y: `p_e_sa × cosY / dist²` |
| `eyeVertexPos` | 3 × N | Position of eye-path vertex S |
| `eyeVertexNormals` | 3 × N | Normal at S |
| `eyeVertexThroughput` | 3 × N | `T(cam→D)` pre-BSDF |
| `resampledEndpoints` | 3 × N | AABB-normalized position of R |
| `resampledValid` | N | 1 if resampled endpoint is valid |
| `resampled_p_e_area` | N | Eye-path area PDF at R |

---

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
- `enabled` — 0=NRC off, 1=active
- `nrcType` — `NRC` or `BDRC`
- `nrcDebugMode` — 0=normal, 1=cache-only, 2=mask, 3=indirect
- `bdrcPhase` — 0=Phase 1, 1=Phase 2
- `spreadC` — area-spread termination constant (default 0.01)
- `terminationDepth` — fallback max depth
- `aabbMin` / `aabbMax` — scene bounding box for feature normalization
- All buffer pointers listed in the BDRC Buffer Layout table above

---

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
| `lightSpecularCount` | `int` | Specular bounce counter on light path |
| `lightDiffuseAfterSpecular` | `int` | Diffuse bounces after first specular (light path) |
| `eyePathFoundDiffuse` | `bool` | Latches true once any diffuse vertex hit (eye path) |
| `eyePathInSpecularChain` | `bool` | True if last BSDF was specular or glossy (eye path) |

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

All BDRC caches use the same `ngp_config` structure with these defaults:

**Hash-grid encoding**:
- 6 levels, 4 features/level, base resolution 16, per-level scale 2.0×
- Hash map size: 2^19 entries

**MLP**:
- 2 hidden layers × 64 neurons, FullyFusedMLP (tcnn)
- Hidden activation: ReLU (eye cache) / ReLU (light cache) / configurable (mask cache)
- Output activation: None/ReLU (eye cache), Softplus (light cache, mask cache)

**Optimizer**: Adam, lr=1e-2, β1=0.9, β2=0.99

**Gradient scaler**: enabled by default (initial scale 65536, growth factor 2×)

---

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