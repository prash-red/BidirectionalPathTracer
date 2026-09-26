//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// bdrc_host.cu — the unified transport cache's per-frame dispatch.
//
// Compiled as CUDA so tiny-cuda-nn headers are available. Fully separate from
// nrc_host.cu, which the NRC mode keeps unchanged.
//

#include "bdrc_host.h"
#include "unified_ngp_cache.h"
#include "../../../HostKernels.h"
#include "../../launch_parameters.h"
#include "../../vector_math.h" // M_PIf, used by the LR schedule below

#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <vector>

#include <tiny-cuda-nn/gpu_matrix.h>

// ============================================================================
// LR schedule — cosine warmup then cosine anneal to a floor.
//
// Unchanged from the three-cache system except that it now drives one field.
// Returns a multiplier in [minFactor, 1] for base_ngp_cache::setLearningRate.
// ============================================================================

static float bdrcLrScheduleFactor(uint32_t iteration, uint32_t warmupIters,
                                  uint32_t totalIters, float minFactor)
{
    float warmup = 1.0f;
    if (warmupIters > 0u && iteration < warmupIters)
    {
        warmup = 0.5f * (1.0f - cosf(M_PIf * static_cast<float>(iteration)
                                           / static_cast<float>(warmupIters)));
    }

    if (totalIters <= warmupIters || iteration < warmupIters)
        return warmup;

    const uint32_t annealIters    = totalIters - warmupIters;
    const float    annealProgress = fminf(static_cast<float>(iteration - warmupIters)
                                          / static_cast<float>(annealIters), 1.0f);
    return minFactor + (1.0f - minFactor) * 0.5f * (1.0f + cosf(M_PIf * annealProgress));
}

constexpr uint32_t BDRC_LR_WARMUP_ITERS = 500u;
constexpr float    BDRC_LR_MIN_FACTOR   = 0.05f;

// tcnn batch-size granularity.
static constexpr uint32_t BDRC_GRAN = 256u;

static uint32_t padTo(uint32_t n, uint32_t g) { return (n + g - 1u) / g * g; }

// ============================================================================
// Kernels
// ============================================================================

// Builds the per-lane density-ratio record from the traced records and the
// tail's EMA field value.
//
//   photon lane: weight = clamped power, scalePos = 1/photonCount, scaleRef = 0
//   gather lane: weight = (E_dir + factor * U_EMA(tail)) / p_r,
//                scalePos = scaleRef = 1/gatherCount
//
// The gather weight's division by p_r is the conversion that matters: it turns
// a value at a point we chose into a sample of a measure, which is the only
// currency in which it can be added to the photon stream. The photon stream
// needs no such division because its deposit measure IS the flux measure by
// construction of the transport estimator.
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

    weight[3*i+0] = 0.0f;
    weight[3*i+1] = 0.0f;
    weight[3*i+2] = 0.0f;
    scalePos[i]   = 0.0f;
    scaleRef[i]   = 0.0f;

    const float q = recDensity[i];
    if (!(q > 0.0f))
        return;   // a disabled lane keeps its denominator and adds nothing

    if (i < photonCount)
    {
        const float invN = (photonCount > 0u) ? 1.0f / static_cast<float>(photonCount) : 0.0f;
        weight[3*i+0] = recBase[3*i+0];
        weight[3*i+1] = recBase[3*i+1];
        weight[3*i+2] = recBase[3*i+2];
        scalePos[i]   = invN;
        // scaleRef stays 0: the photon stream has no anchor pairing available
        // and works as a measure or not at all.
        return;
    }

    const float invN = (gatherCount > 0u) ? 1.0f / static_cast<float>(gatherCount) : 0.0f;
    const bool  tail = (recTailValid[i] > 0.0f);
    for (int c = 0; c < 3; ++c)
    {
        float v = recBase[3*i+c];
        if (tail)
            v += recFactor[3*i+c] * tailOutput[3*i+c];
        weight[3*i+c] = fmaxf(v, 0.0f) / q;
    }
    scalePos[i] = invN;
    scaleRef[i] = invN;
}

// Summarises the field's camera-query output over valid pixels, for the design
// note's fit gate: a field that is uniformly zero or uniformly saturated is the
// failure this distinguishes from a plausible spatial pattern.
static __global__ void k_bdrc_field_stats(
    const float*        fieldOutput,
    const unsigned int* camValid,
    int                 numPixels,
    float*              outSum,     // [3]: sum of luminance, max, count (as float)
    unsigned int*       outNonZero)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPixels || !camValid[i]) return;

    const float u = (fieldOutput[3*i+0] + fieldOutput[3*i+1] + fieldOutput[3*i+2]) / 3.0f;
    atomicAdd(outSum + 0, u);
    atomicAdd(outSum + 2, 1.0f);
    atomicMax(reinterpret_cast<int*>(outSum) + 1, __float_as_int(u));
    if (u > 1.0e-8f) atomicAdd(outNonZero, 1u);
}

// Counts lanes with a live reference density, and copies one live lane's
// density out, for the design note's stream-count and reference-density gates.
static __global__ void k_bdrc_count_live(
    const float*  recDensity,
    uint32_t      begin,
    uint32_t      end,
    unsigned int* outCount,
    float*        outDensity)
{
    const uint32_t i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    if (recDensity[i] > 0.0f)
    {
        atomicAdd(outCount, 1u);
        *outDensity = recDensity[i];
    }
}

// ============================================================================
// BdrcSystem
// ============================================================================

struct BdrcSystem
{
    std::unique_ptr<unified_ngp_cache> field;

    uint32_t numPixels   = 0u;
    uint32_t photonCount = 0u;
    uint32_t gatherCount = 0u;
    uint32_t lanes       = 0u;   // padTo(photonCount + gatherCount, 256)
    uint32_t camLanes    = 0u;   // padTo(numPixels, 256)

    // ---- Training records (lane-indexed) ----
    float* d_recQuery     = nullptr;  // [5 x lanes]
    float* d_recBase      = nullptr;  // [3 x lanes]
    float* d_recDensity   = nullptr;  // [lanes]
    float* d_recTailQuery = nullptr;  // [5 x lanes]
    float* d_recTailValid = nullptr;  // [lanes]
    float* d_recFactor    = nullptr;  // [3 x lanes]

    // ---- Derived per-lane loss inputs ----
    float* d_weight   = nullptr;      // [3 x lanes]
    float* d_scalePos = nullptr;      // [lanes]
    float* d_scaleRef = nullptr;      // [lanes]
    float* d_tailOut  = nullptr;      // [3 x lanes]

    // ---- Camera stream (pixel-indexed) ----
    float*        d_camQuery  = nullptr;  // [5 x camLanes]
    float*        d_camPrefix = nullptr;  // [3 x camLanes]
    float*        d_camFactor = nullptr;  // [3 x camLanes]
    unsigned int* d_camValid  = nullptr;  // [camLanes]
    float*        d_camOutput = nullptr;  // [3 x camLanes]

    // ---- Diagnostics ----
    unsigned int* d_liveCount   = nullptr;  // [2] photon, gather
    float*        d_liveDensity = nullptr;  // [1]

    float* d_fieldStats    = nullptr;  // [3] sum, max, count
    unsigned int* d_fieldNZ = nullptr;  // [1]

    float    m_fieldMean   = 0.0f;
    float    m_fieldMax    = 0.0f;
    uint32_t m_fieldNonZero = 0u;
    uint32_t m_fieldCount   = 0u;

    float    m_lastLoss    = 0.0f;
    uint32_t m_photonLive  = 0u;
    uint32_t m_gatherLive  = 0u;
    float    m_lastDensity = 0.0f;

    void freeAllBuffers()
    {
        for (float** p : { &d_recQuery, &d_recBase, &d_recDensity, &d_recTailQuery,
                           &d_recTailValid, &d_recFactor, &d_weight, &d_scalePos,
                           &d_scaleRef, &d_tailOut, &d_camQuery, &d_camPrefix,
                           &d_camFactor, &d_camOutput, &d_liveDensity, &d_fieldStats })
        {
            cudaFree(*p);
            *p = nullptr;
        }
        cudaFree(d_camValid);  d_camValid  = nullptr;
        cudaFree(d_liveCount); d_liveCount = nullptr;
        cudaFree(d_fieldNZ);   d_fieldNZ   = nullptr;
    }

    ~BdrcSystem() { freeAllBuffers(); }
};

// ============================================================================
// Lifecycle
// ============================================================================

BdrcSystem* bdrcCreate(cudaStream_t stream, const DreConfig& dre)
{
    auto* sys = new BdrcSystem();
    sys->field = std::make_unique<unified_ngp_cache>(dre, stream);
    return sys;
}

void bdrcDestroy(BdrcSystem* sys)
{
    delete sys;
}

void bdrcAllocBuffers(BdrcSystem* sys, uint32_t numPixels,
                      uint32_t photonCount, uint32_t gatherCount)
{
    sys->freeAllBuffers();

    sys->numPixels   = numPixels;
    sys->photonCount = photonCount;
    sys->gatherCount = gatherCount;
    sys->lanes       = padTo(photonCount + gatherCount, BDRC_GRAN);
    sys->camLanes    = padTo(numPixels, BDRC_GRAN);

    const uint32_t L = sys->lanes;
    const uint32_t C = sys->camLanes;

    auto alloc = [](float** p, size_t n) {
        cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float));
        // Zero on allocation so a padding lane that no raygen ever writes
        // decodes as a disabled lane rather than as whatever cudaMalloc handed
        // back. An uninitialized position feeding the network produces NaN
        // forward activations, and 0 * NaN = NaN in IEEE-754, so a correctly
        // zeroed gradient does not save the shared weights.
        cudaMemset(*p, 0, n * sizeof(float));
    };

    alloc(&sys->d_recQuery,     static_cast<size_t>(UNIFIED_INPUT_DIM) * L);
    alloc(&sys->d_recBase,      static_cast<size_t>(3) * L);
    alloc(&sys->d_recDensity,   L);
    alloc(&sys->d_recTailQuery, static_cast<size_t>(UNIFIED_INPUT_DIM) * L);
    alloc(&sys->d_recTailValid, L);
    alloc(&sys->d_recFactor,    static_cast<size_t>(3) * L);

    alloc(&sys->d_weight,   static_cast<size_t>(3) * L);
    alloc(&sys->d_scalePos, L);
    alloc(&sys->d_scaleRef, L);
    alloc(&sys->d_tailOut,  static_cast<size_t>(3) * L);

    alloc(&sys->d_camQuery,  static_cast<size_t>(UNIFIED_INPUT_DIM) * C);
    alloc(&sys->d_camPrefix, static_cast<size_t>(3) * C);
    alloc(&sys->d_camFactor, static_cast<size_t>(3) * C);
    alloc(&sys->d_camOutput, static_cast<size_t>(3) * C);
    alloc(&sys->d_liveDensity, 1);
    alloc(&sys->d_fieldStats, 3);

    cudaMalloc(reinterpret_cast<void**>(&sys->d_camValid), C * sizeof(unsigned int));
    cudaMemset(sys->d_camValid, 0, C * sizeof(unsigned int));

    cudaMalloc(reinterpret_cast<void**>(&sys->d_liveCount), 2u * sizeof(unsigned int));
    cudaMemset(sys->d_liveCount, 0, 2u * sizeof(unsigned int));

    cudaMalloc(reinterpret_cast<void**>(&sys->d_fieldNZ), sizeof(unsigned int));
    cudaMemset(sys->d_fieldNZ, 0, sizeof(unsigned int));
}

// ---- Accessors --------------------------------------------------------------

float* bdrcRecQuery     (const BdrcSystem* sys) { return sys->d_recQuery;     }
float* bdrcRecBase      (const BdrcSystem* sys) { return sys->d_recBase;      }
float* bdrcRecDensity   (const BdrcSystem* sys) { return sys->d_recDensity;   }
float* bdrcRecTailQuery (const BdrcSystem* sys) { return sys->d_recTailQuery; }
float* bdrcRecTailValid (const BdrcSystem* sys) { return sys->d_recTailValid; }
float* bdrcRecFactor    (const BdrcSystem* sys) { return sys->d_recFactor;    }

float*        bdrcCamQuery (const BdrcSystem* sys) { return sys->d_camQuery;  }
float*        bdrcCamPrefix(const BdrcSystem* sys) { return sys->d_camPrefix; }
float*        bdrcCamFactor(const BdrcSystem* sys) { return sys->d_camFactor; }
unsigned int* bdrcCamValid (const BdrcSystem* sys) { return sys->d_camValid;  }

float    bdrcGetLastLoss   (const BdrcSystem* sys) { return sys->m_lastLoss;    }
uint32_t bdrcGetPhotonLive (const BdrcSystem* sys) { return sys->m_photonLive;  }
uint32_t bdrcGetGatherLive (const BdrcSystem* sys) { return sys->m_gatherLive;  }
float    bdrcGetLastDensity(const BdrcSystem* sys) { return sys->m_lastDensity; }
float    bdrcGetFieldMean  (const BdrcSystem* sys) { return sys->m_fieldMean;   }
float    bdrcGetFieldMax   (const BdrcSystem* sys) { return sys->m_fieldMax;    }
uint32_t bdrcGetFieldNonZero(const BdrcSystem* sys){ return sys->m_fieldNonZero;}
uint32_t bdrcGetFieldCount (const BdrcSystem* sys) { return sys->m_fieldCount;  }

void bdrcSetDreConfig(BdrcSystem* sys, const DreConfig& dre)
{
    sys->field->setDreConfig(dre);
}

void bdrcSaveWeights(const BdrcSystem* sys, const std::string& path)
{
    sys->field->saveWeights(path);
}

bool bdrcLoadWeights(BdrcSystem* sys, cudaStream_t stream, const std::string& path)
{
    return sys->field->loadWeights(path, stream);
}

// ============================================================================
// Per-frame dispatch
// ============================================================================

void bdrcDispatch(
    BdrcSystem*  sys,
    cudaStream_t stream,
    float4*      bufferAccum,
    int          numPixels,
    unsigned int iteration,
    unsigned int trainingIteration,
    unsigned int lrAnnealIters,
    int          debugMode,
    bool         freezeTraining)
{
    const uint32_t lanes       = sys->lanes;
    const uint32_t photonCount = sys->photonCount;
    const uint32_t gatherCount = sys->gatherCount;
    const uint32_t camLanes    = sys->camLanes;

    if (lanes == 0u || camLanes == 0u)
        return;

    constexpr uint32_t THREADS = 256u;

    // ---- 1. Tail queries -> EMA inference -----------------------------------
    // The teacher reads the tails; only the fit below touches the live weights.
    {
        tcnn::GPUMatrixDynamic<float> in(sys->d_recTailQuery, UNIFIED_INPUT_DIM, lanes,
                                         tcnn::MatrixLayout::ColumnMajor);
        tcnn::GPUMatrix<float> out(sys->d_tailOut, 3u, lanes);
        sys->field->query(stream, in, out, /*useEmaWeights=*/true);
    }

    // ---- 2. Assemble per-lane (weight, p_r, 1/N_pos, 1/N_ref) ---------------
    {
        const uint32_t blocks = (lanes + THREADS - 1u) / THREADS;
        k_bdrc_assemble_targets<<<blocks, THREADS, 0, stream>>>(
            sys->d_weight, sys->d_scalePos, sys->d_scaleRef,
            sys->d_recBase, sys->d_recDensity, sys->d_recFactor,
            sys->d_recTailValid, sys->d_tailOut,
            photonCount, gatherCount, lanes);
    }

    // ---- 3. One loss over the whole batch -> one step -----------------------
    // Both streams accumulate into ONE optimizer step. dL/dOutput is per-sample,
    // so photon lanes and gather lanes never alias and no Overwrite/Accumulate
    // sequencing is needed.
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

    // ---- 4. Camera inference (EMA) and composite ----------------------------
    {
        tcnn::GPUMatrixDynamic<float> in(sys->d_camQuery, UNIFIED_INPUT_DIM, camLanes,
                                         tcnn::MatrixLayout::ColumnMajor);
        tcnn::GPUMatrix<float> out(sys->d_camOutput, 3u, camLanes);
        sys->field->query(stream, in, out, /*useEmaWeights=*/true);
    }

    // Field statistics over the camera queries, for the fit gate.
    {
        cudaMemsetAsync(sys->d_fieldStats, 0, 3u * sizeof(float),   stream);
        cudaMemsetAsync(sys->d_fieldNZ,    0, sizeof(unsigned int), stream);
        const uint32_t blocks = (static_cast<uint32_t>(numPixels) + THREADS - 1u) / THREADS;
        k_bdrc_field_stats<<<blocks, THREADS, 0, stream>>>(
            sys->d_camOutput, sys->d_camValid, numPixels, sys->d_fieldStats, sys->d_fieldNZ);
    }

    if (debugMode == 0)
    {
        device_bdrc_composite(stream, bufferAccum, sys->d_camPrefix, sys->d_camFactor,
                              sys->d_camOutput, sys->d_camValid, numPixels, iteration);
    }
    else
    {
        device_bdrc_debug(stream, bufferAccum, sys->d_camFactor, sys->d_camOutput,
                          sys->d_camValid, numPixels, iteration, debugMode);
    }

    // ---- 5. Stream counts, then clear the per-frame state --------------------
    cudaMemsetAsync(sys->d_liveCount, 0, 2u * sizeof(unsigned int), stream);
    if (photonCount > 0u)
    {
        const uint32_t blocks = (photonCount + THREADS - 1u) / THREADS;
        k_bdrc_count_live<<<blocks, THREADS, 0, stream>>>(
            sys->d_recDensity, 0u, photonCount, sys->d_liveCount + 0, sys->d_liveDensity);
    }
    if (gatherCount > 0u)
    {
        const uint32_t blocks = (gatherCount + THREADS - 1u) / THREADS;
        k_bdrc_count_live<<<blocks, THREADS, 0, stream>>>(
            sys->d_recDensity, photonCount, photonCount + gatherCount,
            sys->d_liveCount + 1, sys->d_liveDensity);
    }

    cudaMemsetAsync(sys->d_recDensity,   0, lanes    * sizeof(float),        stream);
    cudaMemsetAsync(sys->d_recTailValid, 0, lanes    * sizeof(float),        stream);
    cudaMemsetAsync(sys->d_camValid,     0, camLanes * sizeof(unsigned int), stream);

    unsigned int live[2] = { 0u, 0u };
    cudaStreamSynchronize(stream);
    cudaMemcpy(live, sys->d_liveCount, 2u * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sys->m_lastDensity, sys->d_liveDensity, sizeof(float), cudaMemcpyDeviceToHost);
    sys->m_photonLive = live[0];
    sys->m_gatherLive = live[1];

    float stats[3] = { 0.0f, 0.0f, 0.0f };
    cudaMemcpy(stats, sys->d_fieldStats, 3u * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sys->m_fieldNonZero, sys->d_fieldNZ, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    sys->m_fieldCount = static_cast<uint32_t>(stats[2]);
    sys->m_fieldMean  = (stats[2] > 0.0f) ? stats[0] / stats[2] : 0.0f;
    sys->m_fieldMax   = stats[1];
}
