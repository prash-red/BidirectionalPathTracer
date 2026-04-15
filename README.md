# GLTF_renderer - Bidirectional Path Tracer

An NVIDIA OptiX-based physically-based renderer implementing bidirectional path tracing for the [Khronos glTF 2.0](https://github.com/KhronosGroup/glTF/tree/main/specification/2.0) file format with Neural Radiance Cache support.

## Features

- **Physically-Based Rendering**: Full glTF 2.0 PBR metallic-roughness material model
- **Advanced Materials**: Support for numerous glTF extensions (transmission, volume, clearcoat, sheen, iridescence, anisotropy, IOR, specular, and more)
- **Global Illumination**: Progressive Monte Carlo path tracing with next event estimation
- **GPU Acceleration**: NVIDIA OptiX 7.7+ ray tracing with CUDA compute kernels
- **Animation**: Full support for glTF animations (skeletal skinning, morph targets, node transforms)
- **Neural Radiance Cache**: Optimized caching for indirect illumination using tiny-cuda-nn
- **HDR Environment Maps**: Importance-sampled spherical environment lighting
- **AI Denoising**: Optional OptiX denoiser for noise reduction
- **Interactive GUI**: Dear ImGui interface for real-time parameter adjustment

For a complete feature list and glTF extension support, see [CLAUDE.md](CLAUDE.md).

## Prerequisites

### Required

- **GPU**: NVIDIA GPU with OptiX 7.7+ support (Maxwell or newer, RTX cards recommended)
- **Driver**: Display driver supporting OptiX 7.7+ (see OptiX Release Notes)
- **Compiler**: GCC 11+ (Linux) or MSVC 2019/2022 (Windows)
- **CUDA Toolkit**: 12.0 or newer (13.0+ recommended)
- **OptiX SDK**: 7.7.0 or newer (8.1.0+ recommended)
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

The following dependencies are included as submodules:

- **fastgltf**: glTF 2.0 parser (built automatically)
- **glm**: OpenGL Mathematics library (header-only)
- **stb**: Image loading library (header-only)
- **tiny-cuda-nn**: Neural network library for radiance caching

## Building

### Linux

1. **Set CUDA environment variable** (required):
   ```bash
   export CUDACXX=/usr/local/cuda/bin/nvcc
   ```

2. **Clone the repository with submodules**:
   ```bash
   git clone --recurse-submodules <repository-url>
   cd BidirectionalPathTracer
   ```
   
   Or if already cloned:
   ```bash
   git submodule update --init --recursive
   ```

3. **Create build directory**:
   ```bash
   mkdir build
   cd build
   ```

4. **Configure with CMake**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```
   
   If OptiX is not in the default location, specify it:
   ```bash
   OPTIX91_PATH=/path/to/OptiX-SDK-9.1.0 cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

5. **Build**:
   ```bash
   make -j$(nproc)
   ```

6. **Install OptiX modules** (important):
   ```bash
   make install
   ```
   This copies the OptiX `.optixir` modules to `bin/GLTF_renderer_core/`.

7. **Copy data files**:
   ```bash
   cp ../data/* bin/
   ```

## Running

### Basic Usage

```bash
cd build/bin
./GLTF_renderer --file <path-to-gltf-file>
```

### Command Line Options

```
--file    (-f) <path>    glTF/GLB file to load
--width   (-w) <int>     Window width (default: 1024)
--height  (-h) <int>     Window height (default: 1024)
--launches (-l) <int>    Async launches per frame (default: 1, max: 1000)
--interop (-i) <0-3>     OpenGL-CUDA interop mode (default: 0)
                         0=off, 1=PBO, 2=texture, 3=image (fastest)
--punctual               Enable KHR_lights_punctual extension
--miss    (-m) <0|1|2>   Miss shader mode:
                         0=null (black), 1=constant, 2=spherical HDR
--env     (-e) <path>    HDR environment map for miss mode 2
--sphere-radius <float>  Point cloud sphere radius (default: 0.005)
```

### Examples

**Load a glTF model**:
```bash
./GLTF_renderer -f model.gltf
```

**With HDR environment**:
```bash
./GLTF_renderer -f model.glb -m 2 -e environment.hdr
```

**High-resolution with fast interop**:
```bash
./GLTF_renderer -f model.gltf -w 1920 -h 1080 -i 3
```

**Drag-and-drop**: The application supports drag-and-drop of `.gltf`, `.glb`, and `.hdr` files directly onto the window.

## User Interaction

### Mouse Controls
- **Left Mouse + Drag**: Orbit camera around center of interest
- **Middle Mouse + Drag**: Pan camera
- **Right Mouse + Drag**: Dolly (zoom in/out)
- **Mouse Wheel**: Adjust field of view (1-179 degrees)

### Keyboard Shortcuts
- **SPACE**: Toggle GUI display
- **P**: Save current frame as PNG (tone-mapped)
- **H**: Save current frame as HDR (linear)
- **ESC**: Exit application

## Architecture

### Key Components

- **OptiX Ray Tracing Pipeline**: 
  - Ray generation, miss, closest-hit, any-hit programs
  - Direct callable programs for BSDF evaluation and light sampling
  - GPU acceleration structures (GAS/IAS)

- **CUDA Kernels**:
  - `kernel_morphing.cu`: GPU morph target blending
  - `kernel_skinning.cu`: GPU skeletal animation
  - `kernel_nrc.cu`: Neural radiance cache queries

- **Material System**: Full PBR with 10+ glTF material extensions

- **Animation System**: Supports rotation, translation, scale, and morph target animations

### OptiX Module Format

- Uses **OptiX IR** (`.optixir`) with CUDA 11.7+ and OptiX 7.5+
- Falls back to **PTX** (`.ptx`) with older versions
- Controlled by `USE_OPTIX_IR` CMake variable

### CUDA Architecture Targets

- **CUDA 13+**: SM 75, 80, 86, 89, 90, 120
- **CUDA 12.x**: SM 60, 70, 75, 80, 86, 89, 90, 120

For faster iteration during development, set `DEBUG_ARCH TRUE` in CMakeLists.txt to compile only the minimum architecture.

## Known Issues

### Build Issues

**CMake Error: "CUDACXX not set"**  
Solution: `export CUDACXX=/usr/local/cuda/bin/nvcc` before running cmake

**fastgltf not found**  
Solution: Ensure submodules are initialized: `git submodule update --init --recursive`

**OptiX modules not found at runtime**  
Solution: Build the INSTALL target to copy `.optixir` files to the correct location

### Runtime Issues

**Black screen or crash on launch**  
- Verify OptiX driver compatibility
- Check that `data/sheen_lut.hdr` is in the same directory as the executable
- Ensure OptiX modules were copied via the INSTALL build target

**Performance issues**  
- Try different interop modes: `--interop 3` is fastest but may not work on all systems
- Reduce window resolution: `-w 1024 -h 1024`
- Disable animations if not needed

## Project Structure

```
BidirectionalPathTracer/
├── src/                    # Source code
│   ├── Application.cpp     # Main application class
│   ├── Animation.cpp       # glTF animation system
│   ├── Mesh.cpp            # Host/device mesh management
│   ├── cuda/               # CUDA/OptiX device code
│   │   ├── raygen.cu       # Path tracing loop
│   │   ├── hit.cu          # Surface shading
│   │   ├── miss.cu         # Environment lighting
│   │   ├── kernel_*.cu     # Native CUDA kernels
│   │   └── *.h             # Shared device headers
│   └── imgui/              # Dear ImGui GUI library
├── data/                   # Runtime assets (sheen LUT, etc.)
├── 3rdparty/               # Dependencies
│   ├── fastgltf/           # glTF parser (submodule, built)
│   ├── glm/                # Math library (submodule, header-only)
│   ├── stb/                # Image loader (submodule, header-only)
│   ├── tiny-cuda-nn/       # Neural networks (submodule, built)
│   └── CMake/              # CMake find modules
├── CMakeLists.txt          # Build configuration
└── CLAUDE.md               # Detailed project documentation
```

## Performance Tips

1. **Enable fast interop**: Use `--interop 3` on supported systems
2. **Multi-GPU**: The renderer currently uses a single GPU (ordinal 0). Use `CUDA_VISIBLE_DEVICES` to select a different GPU.
3. **Reduce ray depth**: Adjust `MAX_TRACE_DEPTH` in `Application.h` (default: 2)
4. **Disable features**: Turn off direct lighting, ambient occlusion, or denoising in the GUI for faster previews

## Support and Contributing

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for comprehensive technical documentation
- **Issues**: Report bugs via GitHub issues
- **Forum**: NVIDIA OptiX Developer Forum

## License

Copyright NVIDIA Corporation 2025. See [LICENSE.md](LICENSE.md) for details.

Third-party libraries have their own licenses. See [third_party_oss_licenses.txt](third_party_oss_licenses.txt).

## Acknowledgments

Based on the [NVIDIA OptiX_Apps](https://github.com/NVIDIA/OptiX_Apps) sample repository.

- OptiX SDK by NVIDIA
- [fastgltf](https://github.com/spnda/fastgltf) by spnda
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) by NVlabs
- [glm](https://github.com/g-truc/glm) by G-Truc
- [stb](https://github.com/nothings/stb) by nothings
- [Dear ImGui](https://github.com/ocornut/imgui) by ocornut