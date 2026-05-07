# GLTF_renderer — Bidirectional Path Tracer

An NVIDIA OptiX-based physically-based renderer implementing bidirectional path tracing for the [Khronos glTF 2.0](https://github.com/KhronosGroup/glTF/tree/main/specification/2.0) file format, with neural radiance caching for real-time caustic rendering.

## Features

### Rendering
- **Bidirectional Path Tracing** — eye and light paths traced simultaneously for correct caustic transport
- **Unidirectional Path Tracing** — standard progressive Monte Carlo with next event estimation (NEE) and MIS
- **Neural Radiance Cache (NRC)** — online-trained neural cache for indirect illumination (Mueller et al. 2021)
- **Bidirectional Radiance Cache (BDRC)** — three-cache neural system for caustic-aware rendering (eye cache + light cache + mask cache)
- **HDR Environment Maps** — importance-sampled spherical environment lighting
- **Ambient Occlusion** — screen-space occlusion via material occlusion texture

### Materials (glTF 2.0)
- **PBR Metallic-Roughness** — full core model (base color, metallic, roughness, normal, occlusion, emission)
- **KHR_materials_transmission** — dielectric transmission with roughness
- **KHR_materials_volume** — absorption and scattering in transmitted materials
- **KHR_materials_clearcoat** — double-lobe clearcoat layer
- **KHR_materials_iridescence** — thin-film iridescence
- **KHR_materials_anisotropy** — anisotropic GGX specular
- **KHR_materials_sheen** — fabric sheen with precomputed LUT
- **KHR_materials_specular** — specular color and intensity override
- **KHR_materials_ior** — custom index of refraction
- **KHR_materials_emissive_strength** — HDR emission multiplier
- **KHR_materials_unlit** — unlit (constant) material
- **KHR_texture_transform** — UV scale, rotation, translation per texture
- **Alpha modes** — opaque, alpha mask, alpha blend

### Animation & Geometry
- **Skeletal skinning** — GPU-accelerated joint matrix skinning (`kernel_skinning.cu`)
- **Morph targets** — GPU-accelerated blend shape animation (`kernel_morphing.cu`)
- **Node transforms** — translation, rotation, scale animations
- **Point clouds** — rendered as OptiX built-in spheres with configurable radius

### Platform
- **GPU Acceleration** — NVIDIA OptiX 7.7+ ray tracing with CUDA compute kernels
- **Interactive GUI** — Dear ImGui panels for materials, cameras, lights, animation, cache debug views
- **Drag-and-drop** — drop `.gltf`, `.glb`, or `.hdr` files directly onto the window
- **Screenshot** — PNG (tone-mapped, `P`) and HDR (linear, `H`) export

---

## Prerequisites

### Hardware & Drivers
- **GPU**: NVIDIA GPU with OptiX 7.7+ support
- **Driver**: Display driver supporting the chosen OptiX SDK version (see OptiX Release Notes)

### Software
- **Compiler**: GCC 11+ (Linux) or MSVC 2019/2022 (Windows)
- **CUDA Toolkit**: 12.0 or newer (13.0+ recommended)
- **OptiX SDK**: 7.7.0 or newer
- **CMake**: 3.31 or newer

### Linux System Libraries

```bash
sudo apt install \
  libgl1-mesa-dev \
  libglew-dev \
  libglfw3-dev \
  libdevil-dev \
  libsimdjson-dev
```

### Git Submodules

These dependencies are included as submodules and built automatically:

| Submodule | Location | Purpose |
|-----------|----------|---------|
| fastgltf | `3rdparty/fastgltf` | glTF 2.0 parser |
| glm | `3rdparty/glm` | Math library (header-only) |
| stb | `3rdparty/stb` | Image loading (header-only) |
| tiny-cuda-nn | `3rdparty/tiny-cuda-nn` | Neural network inference + training (NRC/BDRC) |

---

## Building

### Clone

```bash
git clone --recurse-submodules <repository-url>
cd BidirectionalPathTracer
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Linux

```bash
# Required: tell CMake where nvcc is
export CUDACXX=/usr/local/cuda/bin/nvcc

mkdir build && cd build

# Configure (specify OptiX path if not in default location)
cmake .. -DCMAKE_BUILD_TYPE=Release
# or: OPTIX91_PATH=/path/to/OptiX-SDK-9.1.0 cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install OptiX IR modules to bin/GLTF_renderer_core/
make install

# Copy runtime assets
cp ../data/* bin/
```

> **Note**: `CUDACXX` must be set before running cmake on Linux, or the build will abort.

### Faster Iteration Builds

In `CMakeLists.txt`, set `DEBUG_ARCH TRUE` to compile only the minimum CUDA architecture (skips all other SM targets). Cuts compilation time significantly during development.

---

## Running

### Basic Usage

```bash
cd build/bin
./GLTF_renderer --file path/to/scene.glb
```

### Command-Line Options

```
--file    (-f) <path>      glTF/GLB file to load at startup
--width   (-w) <int>       Window width (default: 1024)
--height  (-h) <int>       Window height (default: 1024)
--launches (-l) <int>      Async launches per frame (default: 1, max: 1000)
--interop (-i) <0|1|2|3>   OpenGL-CUDA interop mode (default: 0)
                             0=off (compatible), 1=PBO, 2=texture, 3=image (fastest)
--punctual                 Enable KHR_lights_punctual
--miss    (-m) <0|1|2>     Environment mode: 0=black, 1=constant, 2=HDR sphere
--env     (-e) <path>      HDR environment map (required for --miss 2)
--sphere-radius <float>    Point cloud sphere radius fraction (default: 0.005)
-nrc <0|1|2>               Neural radiance caching mode (default: 0)
                             0=off, 1=NRC (unidirectional), 2=BDRC (bidirectional)
```

### Examples

```bash
# Basic scene
./GLTF_renderer -f model.glb

# HDR environment with fast interop
./GLTF_renderer -f model.glb -m 2 -e environment.hdr -i 3

# High resolution
./GLTF_renderer -f model.gltf -w 1920 -h 1080
```

### Drag-and-Drop

Drop `.gltf`, `.glb`, or `.hdr` files directly onto the window at any time to load them.

---

## User Interaction

### Camera Controls
| Input | Action |
|-------|--------|
| Left mouse + drag | Orbit around center of interest |
| Middle mouse + drag | Pan |
| Right mouse + drag | Dolly (zoom) |
| Mouse wheel | Adjust field of view (1–179°) |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `Space` | Toggle GUI |
| `P` | Save PNG screenshot (tone-mapped) |
| `H` | Save HDR screenshot (linear) |
| `ESC` | Exit |

---

## Architecture

### Rendering Modes

The renderer supports three modes selectable from the GUI:

| Mode | Description |
|------|-------------|
| **Path Tracing** | Unidirectional Monte Carlo path tracing with NEE/MIS |
| **NRC** | Path tracing + Neural Radiance Cache for fast indirect illumination |
| **BDRC** | Bidirectional path tracing + three neural caches for caustic rendering |

### BDRC — Bidirectional Radiance Cache

BDRC addresses the core limitation of unidirectional path tracing: **caustics** (light focusing through specular surfaces onto diffuse receivers). It uses three neural networks trained online:

```
Eye Color Cache   — c_e(D): radiance arriving at first diffuse vertex D from eye paths
Light Color Cache — c_l(Y): caustic transport at floor Y of light paths
Mask Cache        — p_l(x): probability that position x lies on a light-path caustic floor
```

Each pixel is rendered as:

```
pixel = prefix_radiance + T(cam→D) × eyeCache(D)
```

where `prefix_radiance` includes direct illumination up to D, and the eye cache contributes the full indirect radiance including caustics via the light cache connection.

**Two-phase training**:
- **Phase 1** (~2000 frames): Mask cache trains alone via density ratio estimation (DRE).
- **Phase 2**: Mask frozen; eye and light color caches train on BDPT-sampled radiance values.

### NRC Feature Vector (14-dimensional)

| Dimensions | Feature |
|-----------|---------|
| 0–2 | World position (AABB-normalized) |
| 3–4 | Shading normal (spherical, [0,1]) |
| 5–6 | View direction (spherical, [0,1]) |
| 7 | Linear roughness |
| 8–10 | Diffuse base color (RGB) |
| 11–13 | Specular F0 proxy (RGB) |

BDRC caches use only the 3D position (dimensions 0–2).

### OptiX Pipeline

| Module | File | Role |
|--------|------|------|
| Raygen | `raygen.cu` | Eye + light path loops, NRC/BDRC termination |
| Closest-hit | `hit.cu` | BSDF evaluation, NEE, NRC surface state |
| Any-hit | `hit.cu` | Shadow ray occlusion |
| Miss | `miss.cu` | Environment lighting |
| Direct callables | `light_sample.cu` | Per-light-type sampling |

### CUDA Kernels

| File | Purpose |
|------|---------|
| `kernel_skinning.cu` | GPU skeletal animation |
| `kernel_morphing.cu` | GPU morph target blending |
| `cache/nrc/kernel_nrc.cu` | NRC training target assembly |
| `cache/cache_helpers.cu` | Buffer compaction and accumulation |

### OptiX Module Format

- **OptiX IR** (`.optixir`): CUDA 11.7+ + OptiX 7.5+ (default)
- **PTX** (`.ptx`): fallback for older versions
- Controlled by `USE_OPTIX_IR` cmake variable

### CUDA Architecture Targets

| CUDA Version | SM Targets |
|-------------|-----------|
| CUDA 13+ | SM 75, 80, 86, 89, 90, 120 |
| CUDA 12.x | SM 60, 70, 75, 80, 86, 89, 90, 120 |

---

## Project Structure

```
BidirectionalPathTracer/
├── src/
│   ├── Application.h/.cpp       # Main orchestrator (OptiX, BDRC/NRC dispatch, GUI)
│   ├── Animation.h/.cpp         # glTF animation system
│   ├── Mesh.h/.cpp              # Host/device mesh + GAS management
│   ├── Texture.h/.cpp           # CUDA texture object management
│   ├── Camera.h/.cpp            # Camera types + trackball controller
│   ├── Light.h                  # Light definitions
│   ├── Arena.h/.cpp             # GPU arena allocator
│   ├── Options.h/.cpp           # CLI argument parsing
│   │
│   ├── cuda/                    # GPU-side code (OptiX programs + CUDA kernels)
│   │   ├── config.h             # Compile-time flags (NEE, NRC hyperparams, epsilon)
│   │   ├── launch_parameters.h  # LaunchParameters + NrcLaunchBuffers
│   │   ├── per_ray_data.h       # PerRayData payload (BDPT fields, NRC state)
│   │   ├── material_data.h      # Full PBR material parameters
│   │   ├── raygen.cu            # Path tracing loop (eye + light paths)
│   │   ├── hit.cu               # Surface shading + NRC/BDRC integration
│   │   ├── miss.cu              # Environment lighting
│   │   ├── light_sample.cu      # Direct callable light samplers
│   │   ├── kernel_*.cu          # GPU animation kernels
│   │   │
│   │   └── cache/
│   │       ├── nrc_features.h   # Feature packing (14-dim NRC, 3-dim BDRC)
│   │       ├── cache_helpers.*  # Shared compaction + accumulation kernels
│   │       ├── nrc/             # Unidirectional NRC system
│   │       └── bdrc/            # Bidirectional BDRC system
│   │           ├── bdrc_host.*  # BDRC lifecycle + per-frame dispatch
│   │           ├── base_ngp_cache.*  # Hash-grid + MLP cache (tcnn)
│   │           ├── mask_ngp_cache.*  # Mask cache (DRE Brier loss)
│   │           └── ngp_config.h      # Network / optimizer configuration
│   │
│   └── imgui/                   # Dear ImGui (vendored)
│
├── data/
│   ├── sheen_lut.hdr            # Precomputed sheen LUT (required at runtime)
│   └── *.glb                    # Test scenes
│
├── 3rdparty/
│   ├── fastgltf/                # glTF 2.0 parser (submodule)
│   ├── glm/                     # Math library (submodule, header-only)
│   ├── stb/                     # Image loader (submodule, header-only)
│   └── tiny-cuda-nn/            # Neural network library (submodule)
│
├── CMakeLists.txt
├── CLAUDE.md                    # Technical reference for developers
└── doc/                         # Screenshots and feature documentation
```

---

## Troubleshooting

### Build

| Error | Fix |
|-------|-----|
| `CUDACXX not set` | `export CUDACXX=/usr/local/cuda/bin/nvcc` before cmake |
| `fastgltf not found` | `git submodule update --init --recursive` |
| OptiX modules not found at runtime | Run `make install` to copy `.optixir` files |
| CUDA arch mismatch | Set `DEBUG_ARCH TRUE` in CMakeLists.txt for single-arch build |

### Runtime

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Black screen | Missing `sheen_lut.hdr` | `cp data/* build/bin/` |
| Crash on launch | OptiX driver version mismatch | Update GPU driver |
| BDRC produces NaN | Extreme scene scale | Check `aabbMin`/`aabbMax` normalization |
| Slow training convergence | Phase 1 too short | Increase `m_bdrcWarmup` (default 2000 frames) |

---

## Performance Tips

1. **Interop mode**: `--interop 3` is fastest but requires driver support for CUDA surface objects.
2. **Multi-GPU**: Uses GPU 0 by default. Use `CUDA_VISIBLE_DEVICES=N` to select a different GPU.
3. **Debug build**: Set `DEBUG_ARCH TRUE` in CMakeLists.txt for fast single-SM compilation during development.
4. **BDRC freeze**: Toggle "Freeze training" in the GUI to run inference-only while scrubbing camera / scene parameters.
5. **Reduce depth**: Lower `MAX_TRACE_DEPTH` in `Application.h` (default: 2) for faster paths at cost of multi-bounce accuracy.

---

## Technical Reference

For detailed documentation on the BDRC architecture, NRC training buffers, PerRayData fields, shader configuration, and coding patterns, see [CLAUDE.md](CLAUDE.md).

---

## License

Copyright NVIDIA Corporation 2025. See [LICENSE.md](LICENSE.md) for details.

Third-party library licenses: see [third_party_oss_licenses.txt](third_party_oss_licenses.txt).

## Acknowledgments

Based on the [NVIDIA OptiX_Apps](https://github.com/NVIDIA/OptiX_Apps) sample repository.

- [OptiX SDK](https://developer.nvidia.com/optix) by NVIDIA
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by NVlabs
- [fastgltf](https://github.com/spnda/fastgltf) by spnda
- [glm](https://github.com/g-truc/glm) by G-Truc
- [stb](https://github.com/nothings/stb) by nothings
- [Dear ImGui](https://github.com/ocornut/imgui) by ocornut