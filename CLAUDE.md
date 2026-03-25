# GLTF_renderer - Project Guide for Claude

## Project Overview

**GLTF_renderer** is an NVIDIA OptiX-based ray tracing viewer for the [Khronos glTF 2.0](https://github.com/KhronosGroup/glTF/tree/main/specification/2.0) file format. It implements a physically-based global illumination unidirectional path tracer using NVIDIA OptiX SDK (7.7.0+) and CUDA. Copyright NVIDIA Corporation 2025.

The renderer supports drag-and-drop of `*.gltf`/`*.glb` files and `*.hdr` environment maps, interactive GUI via Dear ImGui, and a broad set of glTF 2.0 material extensions.

---

## Build System

- **CMake** (minimum 3.31), C++17 required (`std::filesystem`).
- **Languages**: CXX + CUDA.
- **Build directory**: `apps/GLTF_renderer/build/` (standalone) or the shared `OptiX_Apps/build/`.
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

### Build Steps (Linux)

```bash
# Set CUDACXX env var (required on Linux)
export CUDACXX=/usr/local/cuda/bin/nvcc

cd apps/GLTF_renderer
mkdir build && cd build
cmake ..
make -j$(nproc)
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
GLTF_renderer/
├── main.cpp                    # Entry point: GLFW init, main loop, Options parsing
├── Application.cpp/.h          # Central application class (see below)
├── Animation.cpp/.h            # glTF animation system
├── Arena.cpp/.h                # GPU memory arena allocator (cuda::ArenaAllocator)
├── Camera.cpp/.h               # Camera types (perspective, orthographic)
├── CheckMacros.h               # CUDA/OptiX error-checking macros
├── ConversionArguments.h       # Helper for attribute conversion
├── DeviceBuffer.h              # RAII wrapper for GPU buffer allocations
├── HostBuffer.h                # CPU-side buffer for mesh attribute data
├── HostKernels.h               # CPU-side kernel dispatch helpers
├── Light.h                     # Light types (dev::Light)
├── Logger.h                    # OptiX log callback
├── Mesh.cpp/.h                 # Host/device mesh and primitive classes
├── MyAssert.h                  # MY_ASSERT macro
├── Node.h                      # Scene node (transform, TRS)
├── Options.cpp/.h              # Command-line argument parsing
├── Picture.cpp/.h              # Image loading (DevIL)
├── Record.h                    # SBT record structures
├── SceneExtent.h               # World-space AABB tracking
├── Skin.h                      # Skeletal skinning data (dev::Skin)
├── Texture.cpp/.h              # CUDA texture object management
├── Trackball.cpp/.h            # Orbit/pan/dolly camera controller
├── Utils.cpp/.h                # Timer and misc utilities
│
├── cuda/                       # All GPU-side code
│   ├── config.h                # Global shader configuration defines
│   ├── launch_parameters.h     # LaunchParameters struct (host <-> device)
│   ├── per_ray_data.h          # PerRayData payload struct
│   ├── hit_group_data.h        # HitGroupData SBT record payload
│   ├── geometry_data.h         # GeometryData: vertex buffer pointers
│   ├── material_data.h         # MaterialData: all PBR + extension parameters
│   ├── light_definition.h      # LightDefinition: all light type parameters
│   ├── function_indices.h      # Direct callable program indices
│   ├── bxdf_common.h           # Shared BSDF sampling/evaluation functions
│   ├── shader_common.h         # Shared shading utilities
│   ├── transform.h             # Transform helpers (object-to-world, normals)
│   ├── random.h                # PCG random number generator
│   ├── vector_math.h           # float2/float3/float4 math operators
│   │
│   ├── raygen.cu               # Ray generation program (path tracing loop)
│   ├── miss.cu                 # Miss programs (constant env, spherical HDR env)
│   ├── hit.cu                  # Closest-hit and any-hit programs (triangles + spheres)
│   ├── exception.cu            # OptiX exception program
│   ├── light_sample.cu         # Direct callable programs for light sampling
│   │
│   ├── kernel_morphing.cu      # Native CUDA kernel: GPU morph target blending
│   └── kernel_skinning.cu      # Native CUDA kernel: GPU skeletal skinning
│
├── imgui/                      # Dear ImGui (vendored, v1.9x)
│   ├── imgui*.cpp/.h           # Core ImGui source
│   ├── imgui_impl_glfw.*       # GLFW backend
│   └── imgui_impl_opengl3.*    # OpenGL3 backend
│
├── data/                       # Runtime assets
│   └── sheen_lut.hdr           # Precomputed sheen sampling weight LUT
│
└── doc/                        # Documentation and screenshots
    └── README.md               # Full feature documentation with images
```

---

## Architecture Deep-Dive

### Application Class (`Application.h/.cpp`)

The `Application` class is the central orchestrator. Key responsibilities:

**Initialization sequence** (called from constructor):
1. `initOpenGL()` / `initGLSL()` - OpenGL display pipeline
2. `initCUDA()` - CUDA context, stream, UUID check for interop
3. `initOptiX()` - OptiX context, pipeline, SBT
4. `loadGLTF()` - Parse asset with fastgltf
5. `initRenderer()` - Build acceleration structures, upload data

**Per-frame loop** (`main.cpp`):
```
render()       -> optixLaunch() (path tracing, async)
update()       -> CUDA-GL interop copy to display texture
guiNewFrame()  -> ImGui new frame
guiWindow()    -> all GUI panels
guiEventHandler() -> mouse/keyboard input
display()      -> OpenGL draw fullscreen quad
guiRender()    -> ImGui render
glfwSwapBuffers()
```

**Key member data**:
- `m_asset` (`fastgltf::Asset`) - the loaded glTF asset
- `m_nodes` (`std::vector<dev::Node>`) - shadow of asset nodes with local transforms
- `m_hostMeshes` / `m_deviceMeshes` - CPU and GPU mesh representations
- `m_instances` (`std::vector<dev::Instance>`) - flattened instance list with transforms
- `m_materials` / `m_materialsOrg` - current and original `MaterialData` arrays
- `m_images` (`cudaArray_t`) / `m_samplers` (`cudaTextureObject_t`) - texture data
- `m_animations`, `m_skins`, `m_cameras`, `m_lights` - asset sub-systems
- `m_ias` (`OptixTraversableHandle`) - root Instance Acceleration Structure
- `m_launchParameters` (`LaunchParameters`) - uploaded to GPU each frame

### Acceleration Structure Pipeline

```
HostMesh (CPU) -> DeviceMesh (GPU GAS) -> Instance (transform + GAS index)
                                              |
                                      OptixInstance array
                                              |
                                       IAS (root traversable)
```

- `buildDeviceMeshAccel()` - builds per-mesh GAS (Geometry Acceleration Structure)
- `buildInstanceAccel()` - builds root IAS from all instances
- GAS rebuild is triggered by: material alpha mode changes, skinning/morphing updates, scene switches
- `KeyTuple` (`idxHostMesh`, `idxSkin`, `idxNode`) maps unique (mesh, skin, morph-state) combos to device meshes

### Mesh / Primitive System (`Mesh.h`)

The namespace `dev` separates from `fastgltf::` types:

- `HostPrimitive` - CPU buffers: indices, positions, normals, tangents, colors, texcoords[2], joints[2], weights[2], plus morph target variants
- `DevicePrimitive` - GPU `DeviceBuffer`s mirroring the above, plus morphed/skinned output buffers
- `DeviceMesh` - owns a GAS handle + list of `DevicePrimitive`s
- `HostMesh` - list of `HostPrimitive`s with morph weights
- Primitive types: `Triangles` (triangle meshes) and `Points` (rendered as OptiX built-in spheres)

Attribute priority for positions (used for GAS build):
`positionsSkinned` > `positionsMorphed` > `positions`

### OptiX Pipeline (`Application.h` - `initPipeline()` / `initSBT()`)

**Modules** (compiled separately as OptiX IR/PTX):
| ID | File | Purpose |
|----|------|---------|
| `MODULE_ID_RAYGENERATION` | `raygen.cu` | Path tracing loop |
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

Each `MaterialData::Texture` holds a `cudaTextureObject_t`, texcoord index, and KHR_texture_transform (scale, rotation as (sin,cos), translation).

The sheen model uses a precomputed LUT: `data/sheen_lut.hdr` (loaded in `initSheenLUT()`).

### Animation System (`Animation.h/.cpp`)

- Supports time-based (real-time) and key-frame scrubbing modes
- `dev::Animation` holds samplers (input/output buffers, interpolation mode) and channels (target node, path)
- `updateAnimations()` evaluates all enabled animations for `m_timeCurrent`
- Morph target blending: `updateMorph()` -> `kernel_morphing.cu` (GPU CUDA kernel)
- Skeletal skinning: `updateSkin()` -> `kernel_skinning.cu` (GPU CUDA kernel), joint matrices computed on CPU, uploaded to `m_growSkinMatrices`

### Memory Management

- `cuda::ArenaAllocator` (`Arena.h`): custom GPU memory pool, default 128 MiB arenas
  - `USAGE_STATIC` - long-lived allocations (geometry, textures)
  - `USAGE_TEMP` - temporary allocations (AS build scratch)
- `DeviceBuffer` (`DeviceBuffer.h`): RAII wrapper with `d_ptr` and `size`, supports `grow()` for resizable allocations
- `HostBuffer` (`HostBuffer.h`): CPU-side buffer for mesh attribute staging

### OpenGL-CUDA Interop Modes (`InteropMode` enum)

| Mode | Description | Performance |
|------|-------------|-------------|
| `INTEROP_OFF` (0) | Device-to-host + host-to-device via `glTexSubImage2D` | Slowest, most compatible |
| `INTEROP_PBO` (1) | Render into mapped PBO, device-to-device copy | Medium |
| `INTEROP_TEX` (2) | Device-to-device copy to CUDA texture array | Fast |
| `INTEROP_IMG` (3) | Render directly into mapped CUDA surface object | Fastest |

Default is `INTEROP_OFF`. Change with `--interop (-i) <0|1|2|3>`.

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

## Shader Configuration (`cuda/config.h`)

Key compile-time switches:
- `USE_NEXT_EVENT_ESTIMATION 1` - MIS-weighted direct lighting (default) vs brute-force
- `USE_DEBUG_EXCEPTIONS 0` - Enable OptiX exception program (major perf hit when 1)
- `MICROFACET_MIN_ROUGHNESS 0.0014142f` - Below this, BSDF switches to specular
- `SCENE_EPSILON_SCALE 1.0e-7f` - Self-intersection epsilon scale
- `RT_DEFAULT_MAX 1.e16f` - Default ray tmax / "infinity"

`MAX_TRACE_DEPTH 2` is defined in `Application.h` - controls maximum ray recursion depth for the pipeline.

---

## LaunchParameters (`cuda/launch_parameters.h`)

Uploaded to device as `__constant__ LaunchParameters theLaunchParameters` in every `.cu` program. Contains:
- `handle` - root IAS traversable
- `lightDefinitions` / `numLights` - light array pointer
- `textureSheenLUT` - sheen weight LUT texture
- `surface` / `bufferAccum` - output (surface for INTEROP_IMG, buffer otherwise)
- `bufferPicking` - single int for material picking result
- `resolution`, `picking`, `iteration`, `sceneEpsilon`, `pathLengths`
- Render overrides: `directLighting`, `ambientOcclusion`, `showEnvironment`, `forceUnlit`
- Camera params: `cameraType`, `cameraP/U/V/W`

---

## Scene Graph

glTF scene graph is traversed recursively:
1. `traverseNodeTrafo()` - computes `node.matrixGlobal` for all nodes (must precede skin updates)
2. `traverseNode()` - creates/updates `dev::Instance`s and `dev::DeviceMesh`es
3. `traverseUpdateSceneExtent()` - computes world-space AABB for camera auto-placement

Scene switching (`initScene()` / `updateScene(rebuild=true)`) triggers a full AS rebuild. Animation-only updates (`updateScene(rebuild=false)`) only update instance transforms.

---

## Key Coding Patterns

- **`dev::` namespace** - used for all application-side types to avoid collision with `fastgltf::` types
- **Move-only buffers** - `DeviceBuffer`, `HostBuffer`, `DevicePrimitive`, `DeviceMesh`, `HostMesh` are move-only (no copy), to safely manage GPU memory ownership
- **`isDirty` flags** - used on `HostMesh`, `DeviceMesh`, `Camera`, and global booleans (`m_isDirtyScene`, `m_isDirtyResolution`, `m_isDirtyLights`) to minimize unnecessary GPU work
- **`KeyTuple`** - maps (hostMeshIndex, skinIndex, nodeIndex) to unique device meshes; same mesh instanced with different skins/morph states gets separate GAS
- **Dual material arrays** - `m_materialsOrg` holds original asset values; `m_materials` holds GUI-modified values
- **Picking** - `bufferPicking` (single int on device) receives material index from closest-hit; `float2 picking` in LaunchParameters holds normalized screen coordinate
- **Benchmarking** - `BenchmarkMode::FPS` or `SAMPLES_PER_SECOND`, running average over last 100 values