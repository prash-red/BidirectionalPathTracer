//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// sppm_host.cu — SPPM reference: photon hash grid, gather, progressive update.
//
// The progressive update is ported exactly from the sibling project's
// SPPMDensityCache.update (Hachisuka and Jensen 2009):
//
//   ratio = (N + alpha M) / (N + M)          (1 when M = 0)
//   tau  <- (tau + flux) * ratio
//   r    <- max(r * sqrt(ratio), r_min)
//   N    <- N + alpha M
//
//   pixel = eye_acc / iterations + tau / (pi r^2 N_emitted)
//

#include "sppm_host.h"
#include "../vector_math.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <cstdio>
#include <vector>

namespace
{

constexpr uint32_t THREADS = 256u;

uint32_t nextPow2(uint32_t v)
{
    uint32_t p = 1u;
    while (p < v && p < (1u << 31)) p <<= 1u;
    return p;
}

__device__ __forceinline__ int3 cellOf(const float3 p, const float3 origin, const float invH)
{
    return make_int3(static_cast<int>(floorf((p.x - origin.x) * invH)),
                     static_cast<int>(floorf((p.y - origin.y) * invH)),
                     static_cast<int>(floorf((p.z - origin.z) * invH)));
}

__device__ __forceinline__ uint32_t cellHash(const int3 c, const uint32_t mask)
{
    return ((static_cast<uint32_t>(c.x) * 73856093u) ^
            (static_cast<uint32_t>(c.y) * 19349663u) ^
            (static_cast<uint32_t>(c.z) * 83492791u)) & mask;
}

// One linked list per bucket, prepended with atomicExch: O(n), no sort.
__global__ void k_build_grid(const float3* __restrict__ phPos, uint32_t n,
                             float3 origin, float invH, uint32_t mask,
                             int* __restrict__ head, int* __restrict__ next)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t h = cellHash(cellOf(phPos[i], origin, invH), mask);
    next[i] = atomicExch(&head[h], static_cast<int>(i));
}

// Visible-point-centric: each pixel owns its own sums, so there are no
// atomics on the output. The cell size h is the largest radius of any pixel,
// so a visible point's disc always fits in the 3x3x3 cells around it.
__global__ void k_gather_update(
    const float3* __restrict__ phPos, const float3* __restrict__ phNg,
    const unsigned int* __restrict__ phFront, const float3* __restrict__ phPower,
    const int* __restrict__ head, const int* __restrict__ next,
    float3 origin, float invH, uint32_t mask,
    const float3* __restrict__ vpPos, const float3* __restrict__ vpNg,
    const unsigned int* __restrict__ vpFront, const float3* __restrict__ vpFactor,
    const unsigned int* __restrict__ vpActive, const float3* __restrict__ eyeRadiance,
    float3* __restrict__ tau, float* __restrict__ nAcc, float* __restrict__ radius,
    float3* __restrict__ eyeAcc, unsigned int* __restrict__ activeCount,
    float alpha, float rMin, uint32_t numPixels)
{
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= numPixels) return;

    const float r = radius[px];
    float3 flux = make_float3(0.0f);
    float  M    = 0.0f;

    if (vpActive[px])
    {
        atomicAdd(activeCount, 1u);

        const float3       x     = vpPos[px];
        const float3       ng    = vpNg[px];
        const unsigned int front = vpFront[px];
        const float        r2    = r * r;
        const int3         c     = cellOf(x, origin, invH);

        float3 sum = make_float3(0.0f);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int3 cn = make_int3(c.x + dx, c.y + dy, c.z + dz);
            for (int j = head[cellHash(cn, mask)]; j >= 0; j = next[j])
            {
                const float3 p = phPos[j];

                // Two of the 27 cells can share a bucket. Accept a photon only
                // under its own cell, or that bucket's photons count twice.
                const int3 cp = cellOf(p, origin, invH);
                if (cp.x != cn.x || cp.y != cn.y || cp.z != cn.z) continue;

                const float3 d = p - x;
                if (dot(d, d) >= r2) continue;

                // A photon on a differently-oriented surface (the far side of a
                // corner) or on the other face of a two-sided sheet lights a
                // different domain from this visible point.
                if (dot(phNg[j], ng) <= 0.0f) continue;
                if (phFront[j] != front) continue;

                sum += phPower[j];
                M   += 1.0f;
            }
        }

        // The receiver BSDF is applied here, per visible point: the deposits
        // hold incident power only.
        flux = vpFactor[px] * sum;
    }

    // Progressive update. An inactive pixel has M = 0 and ratio = 1, so its
    // state is carried unchanged while N_emitted still grows: a zero sample.
    const float N     = nAcc[px];
    const float ratio = (M > 0.0f) ? (N + alpha * M) / (N + M) : 1.0f;

    tau[px]    = (tau[px] + flux) * ratio;
    radius[px] = fmaxf(r * sqrtf(ratio), rMin);
    nAcc[px]   = N + alpha * M;
    eyeAcc[px] = eyeAcc[px] + eyeRadiance[px];
}

__global__ void k_reconstruct(float4* __restrict__ out,
                              const float3* __restrict__ eyeAcc,
                              const float3* __restrict__ tau,
                              const float* __restrict__ radius,
                              float invIterations, float photonsEmitted,
                              uint32_t numPixels)
{
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= numPixels) return;

    const float r     = radius[px];
    const float denom = fmaxf(M_PIf * r * r * photonsEmitted, 1.0e-20f);
    const float3 L    = eyeAcc[px] * invIterations + tau[px] / denom;
    out[px] = make_float4(L, 1.0f);
}

__global__ void k_fill_float(float* __restrict__ v, float value, uint32_t n)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = value;
}

} // namespace

// ============================================================================

// Initial photon store per emitted photon. Russian roulette ends a photon
// after a few deposits on average, so this is ample for most scenes; a scene
// that needs more grows the store (see sppmEndIteration).
static constexpr uint32_t SPPM_INITIAL_DEPOSITS_PER_PHOTON = 8u;

struct SppmSystem
{
    SppmConfig cfg;
    uint32_t   numPixels     = 0u;
    uint32_t   capacity      = 0u;   // photon store size; grows on overflow, see sppmEndIteration
    uint32_t   tableSize     = 0u;
    float      sceneDiagonal = 1.0f;
    float      r0            = 0.0f;  // world units
    float      rMin          = 0.0f;

    // per pixel, rewritten every iteration
    float3*       d_vpPos      = nullptr;
    float3*       d_vpNg       = nullptr;
    unsigned int* d_vpFront    = nullptr;
    float3*       d_vpFactor   = nullptr;
    unsigned int* d_vpActive   = nullptr;
    float3*       d_eyeRadiance = nullptr;

    // per pixel, persistent across iterations
    float3* d_tau    = nullptr;
    float*  d_nAcc   = nullptr;
    float*  d_radius = nullptr;
    float3* d_eyeAcc = nullptr;

    // photons
    float3*       d_phPos   = nullptr;
    float3*       d_phNg    = nullptr;
    unsigned int* d_phFront = nullptr;
    float3*       d_phPower = nullptr;
    unsigned int* d_phCount = nullptr;

    // grid
    int* d_head = nullptr;
    int* d_next = nullptr;

    unsigned int* d_activeCount = nullptr;

    SppmStats stats;

    template <typename T> static void freeOne(T*& p) { cudaFree(p); p = nullptr; }

    void freeAll()
    {
        freeOne(d_vpPos); freeOne(d_vpNg); freeOne(d_vpFront); freeOne(d_vpFactor);
        freeOne(d_vpActive); freeOne(d_eyeRadiance);
        freeOne(d_tau); freeOne(d_nAcc); freeOne(d_radius); freeOne(d_eyeAcc);
        freeOne(d_phPos); freeOne(d_phNg); freeOne(d_phFront); freeOne(d_phPower);
        freeOne(d_phCount); freeOne(d_head); freeOne(d_next); freeOne(d_activeCount);
    }

    ~SppmSystem() { freeAll(); }
};

// The photon store and the grid arrays sized by it (capacity and tableSize must
// be set). Kept separate so an overflow can regrow it without touching the
// per-pixel progressive state.
static void allocPhotonStore(SppmSystem* sys)
{
    SppmSystem::freeOne(sys->d_phPos); SppmSystem::freeOne(sys->d_phNg);
    SppmSystem::freeOne(sys->d_phFront); SppmSystem::freeOne(sys->d_phPower);
    SppmSystem::freeOne(sys->d_head); SppmSystem::freeOne(sys->d_next);

    const size_t C = sys->capacity;
    cudaMalloc(&sys->d_phPos,   C * sizeof(float3));
    cudaMalloc(&sys->d_phNg,    C * sizeof(float3));
    cudaMalloc(&sys->d_phFront, C * sizeof(unsigned int));
    cudaMalloc(&sys->d_phPower, C * sizeof(float3));
    cudaMalloc(&sys->d_head, static_cast<size_t>(sys->tableSize) * sizeof(int));
    cudaMalloc(&sys->d_next, C * sizeof(int));
}

SppmSystem* sppmCreate() { return new SppmSystem(); }
void        sppmDestroy(SppmSystem* sys) { delete sys; }

void sppmAllocBuffers(SppmSystem* sys, uint32_t numPixels, const SppmConfig& cfg,
                      float sceneDiagonal)
{
    sys->freeAll();

    sys->cfg           = cfg;
    sys->numPixels     = numPixels;
    sys->sceneDiagonal = (sceneDiagonal > 0.0f) ? sceneDiagonal : 1.0f;
    sys->r0            = cfg.initialRadius * sys->sceneDiagonal;
    sys->rMin          = cfg.minRadius     * sys->sceneDiagonal;
    // Sized for the EXPECTED deposits, not the worst case. A photon deposits at
    // most once per bounce, but with the depth limit shared with the renderer
    // (32 by default) the worst case would be photonsPerIter * 32 slots, about
    // 800 MB at the default photon count, while Russian roulette keeps the mean
    // at a few deposits per photon. sppmEndIteration grows the store whenever
    // an iteration overflows it.
    sys->capacity      = cfg.photonsPerIter * SPPM_INITIAL_DEPOSITS_PER_PHOTON;
    sys->tableSize     = nextPow2(2u * (sys->capacity > 0u ? sys->capacity : 1u));

    const size_t P = numPixels;

    cudaMalloc(&sys->d_vpPos,       P * sizeof(float3));
    cudaMalloc(&sys->d_vpNg,        P * sizeof(float3));
    cudaMalloc(&sys->d_vpFront,     P * sizeof(unsigned int));
    cudaMalloc(&sys->d_vpFactor,    P * sizeof(float3));
    cudaMalloc(&sys->d_vpActive,    P * sizeof(unsigned int));
    cudaMalloc(&sys->d_eyeRadiance, P * sizeof(float3));

    cudaMalloc(&sys->d_tau,    P * sizeof(float3));
    cudaMalloc(&sys->d_nAcc,   P * sizeof(float));
    cudaMalloc(&sys->d_radius, P * sizeof(float));
    cudaMalloc(&sys->d_eyeAcc, P * sizeof(float3));

    allocPhotonStore(sys);
    cudaMalloc(&sys->d_phCount, sizeof(unsigned int));
    cudaMalloc(&sys->d_activeCount, sizeof(unsigned int));

    cudaMemset(sys->d_vpActive,    0, P * sizeof(unsigned int));
    cudaMemset(sys->d_eyeRadiance, 0, P * sizeof(float3));
    cudaMemset(sys->d_phCount,     0, sizeof(unsigned int));

    sppmReset(sys, nullptr);
    cudaDeviceSynchronize();
}

void sppmReset(SppmSystem* sys, cudaStream_t stream)
{
    const uint32_t P = sys->numPixels;
    if (P == 0u) return;

    cudaMemsetAsync(sys->d_tau,    0, P * sizeof(float3), stream);
    cudaMemsetAsync(sys->d_nAcc,   0, P * sizeof(float),  stream);
    cudaMemsetAsync(sys->d_eyeAcc, 0, P * sizeof(float3), stream);
    k_fill_float<<<(P + THREADS - 1u) / THREADS, THREADS, 0, stream>>>(sys->d_radius, sys->r0, P);

    sys->stats = SppmStats{};
    sys->stats.minRadius = sys->r0;
    sys->stats.maxRadius = sys->r0;
}

void sppmFillLaunchBuffers(const SppmSystem* sys, SppmLaunchBuffers& b)
{
    b.vpPos       = sys->d_vpPos;
    b.vpNg        = sys->d_vpNg;
    b.vpFront     = sys->d_vpFront;
    b.vpFactor    = sys->d_vpFactor;
    b.vpActive    = sys->d_vpActive;
    b.eyeRadiance = sys->d_eyeRadiance;

    b.phPos          = sys->d_phPos;
    b.phNg           = sys->d_phNg;
    b.phFront        = sys->d_phFront;
    b.phPower        = sys->d_phPower;
    b.photonCount    = sys->d_phCount;
    b.photonCapacity = sys->capacity;

    b.photonsPerIter = sys->cfg.photonsPerIter;
    b.iteration      = sys->stats.iterations;
    b.photonClamp    = sys->cfg.photonClamp;
}

void sppmBeginIteration(SppmSystem* sys, cudaStream_t stream)
{
    cudaMemsetAsync(sys->d_phCount, 0, sizeof(unsigned int), stream);
}

void sppmEndIteration(SppmSystem* sys, cudaStream_t stream, float4* bufferAccum,
                      float3 gridOrigin)
{
    const uint32_t P = sys->numPixels;
    if (P == 0u || bufferAccum == nullptr) return;

    unsigned int deposits = 0u;
    cudaMemcpyAsync(&deposits, sys->d_phCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Overflow. The deposits past capacity were never stored, and gathering
    // the rest would drop exactly the photons that bounced longest: a silent
    // darkening (this used to clamp the count and carry on). Discard the whole
    // iteration instead: the per-pixel state is untouched and the image keeps
    // its previous reconstruction. Then regrow with headroom so it does not
    // recur. The next iteration traces fresh photons (seeds follow the
    // renderer's iteration counter).
    if (deposits > sys->capacity)
    {
        const uint64_t want = static_cast<uint64_t>(deposits) + deposits / 2u;
        sys->capacity  = static_cast<uint32_t>(want < 0xFFFFFFF0ull ? want : 0xFFFFFFF0ull);
        sys->tableSize = nextPow2(2u * sys->capacity);
        allocPhotonStore(sys);
        sys->stats.discarded += 1u;
        printf("[sppm] photon store overflow: %u deposits, regrown to %u (iteration discarded)\n",
               deposits, sys->capacity);
        fflush(stdout);
        return;
    }

    // Cell size = the largest radius of any pixel, so every disc fits in the
    // 27 cells around its centre. Pixels that never see photons keep r0, which
    // bounds this from above; it only affects speed, never the result.
    thrust::device_ptr<float> rBegin(sys->d_radius);
    const float h    = thrust::reduce(thrust::cuda::par.on(stream), rBegin, rBegin + P,
                                      0.0f, thrust::maximum<float>());
    const float hMin = thrust::reduce(thrust::cuda::par.on(stream), rBegin, rBegin + P,
                                      h, thrust::minimum<float>());
    const float invH = 1.0f / fmaxf(h, 1.0e-20f);
    const uint32_t mask = sys->tableSize - 1u;

    cudaMemsetAsync(sys->d_head, 0xFF, static_cast<size_t>(sys->tableSize) * sizeof(int), stream);
    if (deposits > 0u)
    {
        k_build_grid<<<(deposits + THREADS - 1u) / THREADS, THREADS, 0, stream>>>(
            sys->d_phPos, deposits, gridOrigin, invH, mask, sys->d_head, sys->d_next);
    }

    cudaMemsetAsync(sys->d_activeCount, 0, sizeof(unsigned int), stream);
    k_gather_update<<<(P + THREADS - 1u) / THREADS, THREADS, 0, stream>>>(
        sys->d_phPos, sys->d_phNg, sys->d_phFront, sys->d_phPower,
        sys->d_head, sys->d_next, gridOrigin, invH, mask,
        sys->d_vpPos, sys->d_vpNg, sys->d_vpFront, sys->d_vpFactor,
        sys->d_vpActive, sys->d_eyeRadiance,
        sys->d_tau, sys->d_nAcc, sys->d_radius, sys->d_eyeAcc, sys->d_activeCount,
        sys->cfg.alpha, sys->rMin, P);

    sys->stats.iterations     += 1u;
    sys->stats.photonsEmitted += static_cast<double>(sys->cfg.photonsPerIter);
    sys->stats.lastDeposits    = deposits;
    sys->stats.maxRadius       = h;
    sys->stats.minRadius       = hMin;

    k_reconstruct<<<(P + THREADS - 1u) / THREADS, THREADS, 0, stream>>>(
        bufferAccum, sys->d_eyeAcc, sys->d_tau, sys->d_radius,
        1.0f / static_cast<float>(sys->stats.iterations),
        static_cast<float>(sys->stats.photonsEmitted), P);

    unsigned int active = 0u;
    cudaMemcpyAsync(&active, sys->d_activeCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    sys->stats.lastActiveVps = active;
}

SppmStats sppmGetStats(const SppmSystem* sys) { return sys->stats; }
