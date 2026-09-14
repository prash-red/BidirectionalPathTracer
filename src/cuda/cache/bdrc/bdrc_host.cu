//
// BdrcHost.cu — Host-side BDRC system managing the three-cache BDRC pipeline.
// Compiled as CUDA (nvcc) so tcnn headers are available.
// Fully separate from NrcHost.cu — NrcHost.h/.cu are NOT modified.
//

#include "bdrc_host.h"
#include "base_ngp_cache.h"
#include "mask_ngp_cache.h"
#include "ngp_config.h"
#include "../../../HostKernels.h"
#include "../../launch_parameters.h"

#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include <tiny-cuda-nn/gpu_matrix.h>

// ============================================================================
// BdrcSystem — full definition (opaque to Application.cpp)
// ============================================================================

struct BdrcSystem
{
    std::unique_ptr<base_ngp_cache>  eyeNrc;    // 3-ch, ReLU hidden, no output activation, RelativeL2
    std::unique_ptr<base_ngp_cache>  lightNrc;  // 3-ch, Softplus output, RelativeL2
    std::unique_ptr<mask_ngp_cache>  maskNrc;   // 1-ch, Softplus output, DRE Brier loss

    float m_lastEyeLoss   = 0.0f;
    float m_lastLightLoss = 0.0f;
    float m_lastMaskLoss  = 0.0f;

    // BDRC light-path buffers
    float*        d_lightOutput     = nullptr; // [3 × batchSize]
    float*        d_maskOutput      = nullptr; // [1 × batchSize]
    uint32_t      m_numPixels       = 0;       // exact pixel count
    // Light-path buffers — one slot per pixel (no depth striding; see Task 5).
    float*        d_lightEndpoints  = nullptr; // [3 × batchSize]
    float*        d_lightTargets    = nullptr; // [3 × batchSize]
    float*        d_light_p_e_area  = nullptr; // [batchSize]
    unsigned int* d_lightValid      = nullptr; // [batchSize]  reset each frame
    // Constant label buffers
    float*        d_zeroTargets3    = nullptr; // [3 × batchSize] zeros
    float*        d_oneTargets1     = nullptr; // [1 × batchSize] ones
    float*        d_zeroTargets1    = nullptr; // [1 × batchSize] zeros
    // Eye cache inference/training output buffers
    float*        d_renderOutput    = nullptr; // [3 × batchSize] eye NRC query result
    float*        d_nrcNextOutput   = nullptr; // [3 × batchSize] suffix EMA query
    float*        d_trainTargets    = nullptr; // [3 × batchSize] built by accumulate kernel
    // Eye-path buffers (filled by raygen)
    float*        d_renderFeatures      = nullptr; // [3 × batchSize]
    float*        d_renderPrefix        = nullptr; // [3 × batchSize]
    float*        d_renderThroughput    = nullptr; // [3 × batchSize]
    unsigned int* d_renderValid         = nullptr; // [batchSize]
    float*        d_trainNextFeatures   = nullptr; // [3 × batchSize]
    float*        d_trainSuffixScaled   = nullptr; // [3 × batchSize]
    float*        d_trainSuffThroughput = nullptr; // [3 × batchSize]
    float*        d_trainThroughputAtK  = nullptr; // [3 × batchSize]
    unsigned int* d_trainNextValid      = nullptr; // [batchSize]
    float*        d_albedo              = nullptr; // [3 × batchSize] (optional) albedo at eye vertices for weighted losses
    float*        d_eyeVertexPos        = nullptr; // [3 × batchSize]
    float*        d_eyeVertexNormals    = nullptr; // [3 × batchSize]
    float*        d_eyeVertexThroughput = nullptr; // [3 × batchSize]
    float*        d_resampledEndpoints  = nullptr; // [3 × batchSize]
    unsigned int* d_resampledValid      = nullptr; // [batchSize]
    float*        d_resampled_p_sa      = nullptr; // [batchSize]
    float*        d_resampled_p_e_area  = nullptr; // [batchSize]  eye-path area PDF at R
    // Compact (slot-indexed) buffers built each frame by bdrcDispatch before training.
    float*        d_compactLightEndpoints    = nullptr; // [3 × batchSize]
    float*        d_compactLightTargets      = nullptr; // [3 × batchSize]
    unsigned int* d_compactLightCount        = nullptr; // [1] reset + filled by compaction kernel
    float*        d_compactLight_p_e_area    = nullptr; // [batchSize]  compacted eye-area PDF at Y
    float*        d_compactResampledEndpoints    = nullptr; // [3 × batchSize]
    unsigned int* d_compactResampledCount        = nullptr; // [1] reset + filled by compaction kernel
    float*        d_compactResampled_p_e_area    = nullptr; // [batchSize]  compacted eye-area PDF at R
    float*        d_debugTargetMatrix          = nullptr; // [3 × batchsize]

    // Virtual light emitter buffers (Task 9 — unwired plumbing for Tasks 10-11).
    float*        d_virtEmitPos        = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_virtEmitThroughput = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    unsigned int* d_virtEmitValid      = nullptr; // [VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_compactVirtEmitPos        = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    float*        d_compactVirtEmitThroughput = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize]
    unsigned int* d_compactVirtEmitCount       = nullptr; // [1]
    float*        d_virtEmitOutput            = nullptr; // [3 × VIRT_EMIT_MAX_DEPTH × batchSize] lightNrc EMA query result
    float*        d_virtEmitContribSum        = nullptr; // [3 × batchSize] per-training-slot summed compensation
    unsigned int* d_pixelToTrainSlot          = nullptr; // [numPixels] inverse map: pixel -> training slot, 0xFFFFFFFF if none
    unsigned int* d_virtEmitPixelIdx          = nullptr; // [VIRT_EMIT_MAX_DEPTH * batchSize] compact output: pixel index per compacted virt-emit entry

    void freeAllBuffers()
    {
        cudaFree(d_lightOutput);         d_lightOutput         = nullptr;
        cudaFree(d_maskOutput);          d_maskOutput          = nullptr;
        cudaFree(d_lightEndpoints);      d_lightEndpoints      = nullptr;
        cudaFree(d_lightTargets);        d_lightTargets        = nullptr;
        cudaFree(d_light_p_e_area);      d_light_p_e_area      = nullptr;
        cudaFree(d_lightValid);          d_lightValid          = nullptr;
        cudaFree(d_zeroTargets3);        d_zeroTargets3        = nullptr;
        cudaFree(d_oneTargets1);         d_oneTargets1         = nullptr;
        cudaFree(d_zeroTargets1);        d_zeroTargets1        = nullptr;
        cudaFree(d_renderOutput);        d_renderOutput        = nullptr;
        cudaFree(d_nrcNextOutput);       d_nrcNextOutput       = nullptr;
        cudaFree(d_trainTargets);        d_trainTargets        = nullptr;
        cudaFree(d_renderFeatures);      d_renderFeatures      = nullptr;
        cudaFree(d_renderPrefix);        d_renderPrefix        = nullptr;
        cudaFree(d_renderThroughput);    d_renderThroughput    = nullptr;
        cudaFree(d_renderValid);         d_renderValid         = nullptr;
        cudaFree(d_trainNextFeatures);   d_trainNextFeatures   = nullptr;
        cudaFree(d_trainSuffixScaled);   d_trainSuffixScaled   = nullptr;
        cudaFree(d_trainSuffThroughput); d_trainSuffThroughput = nullptr;
        cudaFree(d_trainThroughputAtK);  d_trainThroughputAtK  = nullptr;
        cudaFree(d_trainNextValid);      d_trainNextValid      = nullptr;
        cudaFree(d_albedo);              d_albedo              = nullptr;
        cudaFree(d_eyeVertexPos);        d_eyeVertexPos        = nullptr;
        cudaFree(d_eyeVertexNormals);    d_eyeVertexNormals    = nullptr;
        cudaFree(d_eyeVertexThroughput); d_eyeVertexThroughput = nullptr;
        cudaFree(d_resampledEndpoints);  d_resampledEndpoints  = nullptr;
        cudaFree(d_resampledValid);      d_resampledValid      = nullptr;
        cudaFree(d_resampled_p_sa);              d_resampled_p_sa              = nullptr;
        cudaFree(d_resampled_p_e_area);          d_resampled_p_e_area          = nullptr;
        cudaFree(d_compactLightEndpoints);       d_compactLightEndpoints       = nullptr;
        cudaFree(d_compactLightTargets);         d_compactLightTargets         = nullptr;
        cudaFree(d_compactLightCount);           d_compactLightCount           = nullptr;
        cudaFree(d_compactLight_p_e_area);       d_compactLight_p_e_area       = nullptr;
        cudaFree(d_compactResampledEndpoints);       d_compactResampledEndpoints       = nullptr;
        cudaFree(d_compactResampledCount);           d_compactResampledCount           = nullptr;
        cudaFree(d_compactResampled_p_e_area);       d_compactResampled_p_e_area       = nullptr;
        cudaFree(d_debugTargetMatrix);           d_debugTargetMatrix           = nullptr;

        cudaFree(d_virtEmitPos);                 d_virtEmitPos                 = nullptr;
        cudaFree(d_virtEmitThroughput);           d_virtEmitThroughput          = nullptr;
        cudaFree(d_virtEmitValid);                d_virtEmitValid               = nullptr;
        cudaFree(d_compactVirtEmitPos);           d_compactVirtEmitPos          = nullptr;
        cudaFree(d_compactVirtEmitThroughput);    d_compactVirtEmitThroughput   = nullptr;
        cudaFree(d_compactVirtEmitCount);         d_compactVirtEmitCount        = nullptr;
        cudaFree(d_virtEmitOutput);               d_virtEmitOutput              = nullptr;
        cudaFree(d_virtEmitContribSum);           d_virtEmitContribSum          = nullptr;
        cudaFree(d_pixelToTrainSlot);              d_pixelToTrainSlot            = nullptr;
        cudaFree(d_virtEmitPixelIdx);               d_virtEmitPixelIdx           = nullptr;
    }

    ~BdrcSystem() { freeAllBuffers(); }
};

// ============================================================================
// API implementation
// ============================================================================

BdrcSystem* bdrcCreate(cudaStream_t stream)
{
    auto* sys = new BdrcSystem();

    // Eye color cache — learns ĉ_e(x) ≈ L_eye(x).
    // Loss: L_eye = E_{x~p_e}[||ĉ_e(x) - c_e(x)||²].
    {
        ngp_config cfg;
        cfg.loss = "RelativeL2Luminance";
        cfg.gradClip = 0;
        cfg.lr = 3e-3f;
        sys->eyeNrc = std::make_unique<base_ngp_cache>(cfg, /*outputDim=*/3, stream);
        sys->eyeNrc->initEma(cfg.ema.decay, cfg.ema.startIter);
    }

    // Light color cache — learns ĉ_l(x)
    // Two loss terms applied in Phase 2:
    //   L_light = E_{x~p_l}[||ĉ_l(x) - c_l(x)||²]   (on light samples)
    //   L_zero  = E_{x~p_e}[||ĉ_l(x)||²]                                      (zero supervision at eye positions)
    // Both are standard L2.
    {
        ngp_config cfg;
        cfg.loss = "RelativeL2Luminance";
        cfg.gradClip = 0.5;
        cfg.lr = 3e-4f;
        sys->lightNrc = std::make_unique<base_ngp_cache>(cfg, /*outputDim=*/3, stream);
        sys->lightNrc->initEma(cfg.ema.decay, cfg.ema.startIter);
    }

    // Density mask cache — learns α(x) ∈ [0,1] discriminating eye vs light samples.
    // Loss: L_mask = E_{x~p_e}[α(x)²] + (N_l/N_e)·E_{x~p_l}[(α(x)-1)²]
    //   = L2 vs label-0 on eye samples  +  L2 vs label-1 on light samples (Phase 1).
    // Softplus output keeps α(x)≥0; the ratio N_l/N_e is 1 (equal batch sizes).
    {
        sys->maskNrc = std::make_unique<mask_ngp_cache>(stream);
    }

    return sys;
}

void bdrcDestroy(BdrcSystem* sys)
{
    delete sys;
}

void bdrcAllocBuffers(BdrcSystem* sys, uint32_t numPixels)
{
    sys->freeAllBuffers();

    const uint32_t inputDim  = static_cast<uint32_t>(ngp_config().inputDim); // 3 (position only)
    const uint32_t batchSize = (numPixels + 255u) / 256u * 256u;
    sys->m_numPixels = numPixels;

    // Light-path output buffers
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightOutput),    batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_maskOutput),     batchSize * 1u       * sizeof(float));

    // Constant label buffers
    cudaMalloc(reinterpret_cast<void**>(&sys->d_zeroTargets3), batchSize  * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_oneTargets1),  batchSize  * 1u * sizeof(float));  // needs to cover batchSize for mask training
    cudaMalloc(reinterpret_cast<void**>(&sys->d_zeroTargets1), batchSize  * 1u * sizeof(float));

    // Eye-cache output buffers
    cudaMalloc(reinterpret_cast<void**>(&sys->d_renderOutput),  batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_nrcNextOutput), batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainTargets),  batchSize * 3u * sizeof(float));

    // Eye-path buffers (filled by raygen)
    cudaMalloc(reinterpret_cast<void**>(&sys->d_renderFeatures),      batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_renderPrefix),        batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_renderThroughput),    batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_renderValid),         batchSize            * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextFeatures),   batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffixScaled),   batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffThroughput), batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainThroughputAtK),  batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextValid),      batchSize            * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_albedo),              batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_eyeVertexPos),        batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_eyeVertexNormals),  batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_eyeVertexThroughput), batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_resampledEndpoints),         batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_resampledValid),             batchSize      * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_resampled_p_sa),             batchSize      * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_resampled_p_e_area),         batchSize      * sizeof(float));
    // Light-path buffers — one slot per pixel (no depth striding; see Task 5).
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightEndpoints),             batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightTargets),               batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_light_p_e_area),             batchSize            * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_lightValid),                 batchSize            * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightEndpoints),      batchSize * inputDim * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightTargets),        batchSize * 3u       * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLightCount),                    1u          * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactLight_p_e_area),      batchSize            * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactResampledEndpoints),      batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactResampledCount),                1u       * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactResampled_p_e_area),    batchSize      * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_debugTargetMatrix),          batchSize * 3u * sizeof(float));

    // Virtual light emitter buffers (Task 9 — unwired plumbing for Tasks 10-11).
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitPos),                VIRT_EMIT_MAX_DEPTH * batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitThroughput),         VIRT_EMIT_MAX_DEPTH * batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitValid),              VIRT_EMIT_MAX_DEPTH * batchSize      * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactVirtEmitPos),         VIRT_EMIT_MAX_DEPTH * batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactVirtEmitThroughput),  VIRT_EMIT_MAX_DEPTH * batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_compactVirtEmitCount),                       1u              * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitOutput),             VIRT_EMIT_MAX_DEPTH * batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitContribSum),         batchSize * 3u * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_pixelToTrainSlot),           numPixels * sizeof(unsigned int));
    cudaMalloc(reinterpret_cast<void**>(&sys->d_virtEmitPixelIdx),          VIRT_EMIT_MAX_DEPTH * batchSize * sizeof(unsigned int));

    // Zero-init valid flags, counters, and zero-target buffers
    cudaMemset(sys->d_lightValid,               0, batchSize  * sizeof(unsigned int));
    cudaMemset(sys->d_compactLightCount,        0,          1u * sizeof(unsigned int));
    cudaMemset(sys->d_zeroTargets3,             0, batchSize * 3u * sizeof(float));
    cudaMemset(sys->d_zeroTargets1,             0, batchSize * 1u * sizeof(float));
    cudaMemset(sys->d_renderValid,              0, batchSize * sizeof(unsigned int));
    cudaMemset(sys->d_trainNextValid,           0, batchSize * sizeof(unsigned int));
    cudaMemset(sys->d_resampledValid,           0, batchSize * sizeof(unsigned int));
    cudaMemset(sys->d_resampled_p_sa,           0, batchSize * sizeof(float));
    cudaMemset(sys->d_resampled_p_e_area,       0, batchSize * sizeof(float));
    cudaMemset(sys->d_compactLight_p_e_area,    0, batchSize * sizeof(float));
    cudaMemset(sys->d_compactResampled_p_e_area,0, batchSize * sizeof(float));
    cudaMemset(sys->d_virtEmitValid,            0, VIRT_EMIT_MAX_DEPTH * batchSize * sizeof(unsigned int));
    cudaMemset(sys->d_compactVirtEmitCount,     0,          1u * sizeof(unsigned int));

    // Fill d_oneTargets1 with 1.0f (sized for batchSize to cover compact light training)
    std::vector<float> ones(batchSize, 1.0f);
    cudaMemcpy(sys->d_oneTargets1, ones.data(), batchSize * sizeof(float), cudaMemcpyHostToDevice);

    // fill debugTargetMatrix with blue values
    {
        std::vector<float> blue(batchSize * 3u);
        for (uint32_t i = 0; i < batchSize; i++) {
            blue[i * 3 + 0] = 1.0f;
            blue[i * 3 + 1] = 0.0f;
            blue[i * 3 + 2] = 0.0f;
        }
        cudaMemcpy(sys->d_debugTargetMatrix, blue.data(), batchSize * 3u * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// ---- Buffer accessors -------------------------------------------------------

float*        bdrcLightOutput       (const BdrcSystem* sys) { return sys->d_lightOutput;          }
float*        bdrcMaskOutput        (const BdrcSystem* sys) { return sys->d_maskOutput;           }
float*        bdrcRenderFeatures    (const BdrcSystem* sys) { return sys->d_renderFeatures;       }
float*        bdrcRenderPrefix      (const BdrcSystem* sys) { return sys->d_renderPrefix;         }
float*        bdrcRenderThroughput  (const BdrcSystem* sys) { return sys->d_renderThroughput;     }
unsigned int* bdrcRenderValid       (const BdrcSystem* sys) { return sys->d_renderValid;          }
float*        bdrcTrainNextFeatures (const BdrcSystem* sys) { return sys->d_trainNextFeatures;    }
float*        bdrcTrainSuffixScaled (const BdrcSystem* sys) { return sys->d_trainSuffixScaled;    }
float*        bdrcTrainSuffThroughput(const BdrcSystem* sys){ return sys->d_trainSuffThroughput;  }
float*        bdrcTrainThroughputAtK(const BdrcSystem* sys) { return sys->d_trainThroughputAtK;   }
unsigned int* bdrcTrainNextValid    (const BdrcSystem* sys) { return sys->d_trainNextValid;       }
float*        bdrcAlbedo            (const BdrcSystem* sys) { return sys->d_albedo;              }
float*        bdrcEyeVertexPos      (const BdrcSystem* sys) { return sys->d_eyeVertexPos;         }
float*        bdrcEyeVertexNormals  (const BdrcSystem* sys) { return sys->d_eyeVertexNormals;     }
float*        bdrcEyeVertexThroughput(const BdrcSystem* sys){ return sys->d_eyeVertexThroughput;  }
float*        bdrcResampledEndpoints(const BdrcSystem* sys) { return sys->d_resampledEndpoints;   }
unsigned int* bdrcResampledValid    (const BdrcSystem* sys) { return sys->d_resampledValid;       }
float*        bdrcResampled_p_sa         (const BdrcSystem* sys) { return sys->d_resampled_p_sa;             }
float*        bdrcResampled_p_e_area     (const BdrcSystem* sys) { return sys->d_resampled_p_e_area;         }
float*        bdrcLightEndpoints         (const BdrcSystem* sys) { return sys->d_lightEndpoints;             }
float*        bdrcLightTargets           (const BdrcSystem* sys) { return sys->d_lightTargets;               }
float*        bdrcLight_p_e_area         (const BdrcSystem* sys) { return sys->d_light_p_e_area;             }
unsigned int* bdrcLightValid             (const BdrcSystem* sys) { return sys->d_lightValid;                 }
float*        bdrcCompactResampledEndpoints(const BdrcSystem* sys){ return sys->d_compactResampledEndpoints; }
unsigned int* bdrcCompactResampledCount  (const BdrcSystem* sys) { return sys->d_compactResampledCount;      }
float*        bdrcCompactResampled_p_e_area(const BdrcSystem* sys){ return sys->d_compactResampled_p_e_area; }
float         bdrcGetLastEyeLoss  (const BdrcSystem* sys) { return sys->m_lastEyeLoss;   }
float         bdrcGetLastLightLoss(const BdrcSystem* sys) { return sys->m_lastLightLoss; }
float         bdrcGetLastMaskLoss (const BdrcSystem* sys) { return sys->m_lastMaskLoss;  }
float*        bdrcRenderOutput    (const BdrcSystem* sys) { return sys->d_renderOutput;  }
float*        bdrcVirtEmitPos        (const BdrcSystem* sys) { return sys->d_virtEmitPos;        }
float*        bdrcVirtEmitThroughput (const BdrcSystem* sys) { return sys->d_virtEmitThroughput; }
unsigned int* bdrcVirtEmitValid      (const BdrcSystem* sys) { return sys->d_virtEmitValid;       }

// ---- Compaction kernels -----------------------------------------------------
//
// Both buffers use AoS (col-major) layout: sample s has 3 features at buf[3*s..3*s+2].
// Each thread handles one sparse sample; valid samples atomically claim a dense slot.

static __global__ void bdrc_compact_light_kernel(
    float*              compactEndpoints,   // [3 × batchSize] AoS output
    float*              compactTargets,     // [3 × batchSize] AoS output
    float*              compactPeArea,      // [batchSize] output
    unsigned int*       compactCount,       // [1] must be 0 before launch
    const float*        sparseEndpoints,    // [3 × batchSize] AoS input
    const float*        sparseTargets,      // [3 × batchSize] AoS input
    const float*        sparsePeArea,       // [batchSize] input
    const unsigned int* valid,              // [batchSize]
    uint32_t            batchSize)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize || !valid[i]) return;
    const uint32_t slot = atomicAdd(compactCount, 1u);
    compactEndpoints[3u * slot + 0u] = sparseEndpoints[3u * i + 0u];
    compactEndpoints[3u * slot + 1u] = sparseEndpoints[3u * i + 1u];
    compactEndpoints[3u * slot + 2u] = sparseEndpoints[3u * i + 2u];
    compactTargets  [3u * slot + 0u] = sparseTargets  [3u * i + 0u];
    compactTargets  [3u * slot + 1u] = sparseTargets  [3u * i + 1u];
    compactTargets  [3u * slot + 2u] = sparseTargets  [3u * i + 2u];
    compactPeArea   [slot]            = sparsePeArea  [i];
}

static __global__ void bdrc_compact_resampled_kernel(
    float*              compact,              // [3 × batchSize] AoS output
    float*              compactPeArea,        // [batchSize] output
    unsigned int*       compactCount,         // [1] must be 0 before launch
    const float*        sparse,               // [3 × batchSize] AoS input
    const float*        sparsePeArea,         // [batchSize] input
    const unsigned int* valid,                // [batchSize]
    uint32_t            batchSize)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize || !valid[i]) return;
    const uint32_t slot = atomicAdd(compactCount, 1u);
    compact[3u * slot + 0u] = sparse[3u * i + 0u];
    compact[3u * slot + 1u] = sparse[3u * i + 1u];
    compact[3u * slot + 2u] = sparse[3u * i + 2u];
    compactPeArea[slot] = sparsePeArea[i];
}

// Compact the depth-strided virtual-emission buffer into a dense array,
// remembering which pixel each compacted entry maps back to so the
// scatter-add step can find the right eye-training-target slot.
static __global__ void bdrc_compact_virt_emit_kernel(
    float*              compactPos,          // [3 x cap] AoS output
    float*              compactWeight,       // [3 x cap] AoS output
    unsigned int*       compactPixelIdx,     // [cap] output - pixel index this entry belongs to
    unsigned int*       compactCount,        // [1] must be 0 before launch
    const float*        sparsePos,           // [3 x VIRT_EMIT_MAX_DEPTH x numPixels] input
    const float*        sparseWeight,        // [3 x VIRT_EMIT_MAX_DEPTH x numPixels] input
    const unsigned int* sparseValid,         // [VIRT_EMIT_MAX_DEPTH x numPixels] input
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

// Build inverse map: pixel -> training slot. (The forward map trainPixelIndices
// maps training slot -> pixel; this inverts it for O(1) lookup by pixel.)
static __global__ void bdrc_build_pixel_to_train_slot_kernel(
    unsigned int*       pixelToTrainSlot,   // [numPixels] output, init to 0xFFFFFFFF before launch
    const unsigned int* trainPixelIndices,  // [trainCount] input
    unsigned int        trainCount)
{
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= trainCount) return;
    pixelToTrainSlot[trainPixelIndices[s]] = s;
}

// Scatter compensation = lightOutput * weight into the eye training target for
// the pixel each compacted entry belongs to. Multiple virtual-emission entries
// can belong to the same pixel (one per re-entered specular chain along the
// suffix), so this uses atomicAdd.
static __global__ void bdrc_add_virt_emission_kernel(
    float*              trainSuffixScaled,  // [3 x batchSize] eye training target, slot-indexed
    const unsigned int* pixelToTrainSlot,   // [numPixels] inverse map, 0xFFFFFFFF if none
    const float*        compactWeight,      // [3 x compactCount]
    const float*        lightOutput,        // [3 x compactCount] EMA lightNrc query result
    const unsigned int* compactPixelIdx,    // [compactCount]
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

static void device_bdrc_compact_light(
    cudaStream_t        stream,
    float*              compactEndpoints,
    float*              compactTargets,
    float*              compactPeArea,
    unsigned int*       compactCount,
    const float*        sparseEndpoints,
    const float*        sparseTargets,
    const float*        sparsePeArea,
    const unsigned int* valid,
    uint32_t            batchSize)
{
    constexpr uint32_t THREADS = 256u;
    const uint32_t blocks = (batchSize + THREADS - 1u) / THREADS;
    bdrc_compact_light_kernel<<<blocks, THREADS, 0, stream>>>(
        compactEndpoints, compactTargets, compactPeArea, compactCount,
        sparseEndpoints, sparseTargets, sparsePeArea, valid, batchSize);
}

static void device_bdrc_compact_resampled(
    cudaStream_t        stream,
    float*              compact,
    float*              compactPeArea,
    unsigned int*       compactCount,
    const float*        sparse,
    const float*        sparsePeArea,
    const unsigned int* valid,
    uint32_t            batchSize)
{
    constexpr uint32_t THREADS = 256u;
    const uint32_t blocks = (batchSize + THREADS - 1u) / THREADS;
    bdrc_compact_resampled_kernel<<<blocks, THREADS, 0, stream>>>(
        compact, compactPeArea, compactCount,
        sparse, sparsePeArea, valid, batchSize);
}

// ---- Per-frame dispatch -----------------------------------------------------

void bdrcDispatch(
    BdrcSystem*   sys,
    cudaStream_t  stream,
    float4*       bufferAccum,
    int           numPixels,
    uint32_t      batchSize,
    unsigned int  iteration,
    int           bdrcPhase,
    int           debugMode,
    unsigned int* d_trainCount,
    float*        d_trainFeatures,
    const unsigned int* d_trainPixelIndices,
    bool          freezeTraining,
    float3        aabbMin,
    float3        aabbMax,
    float3        cameraP,
    float3        cameraU,
    float3        cameraV,
    float3        cameraW,
    int           resolutionX,
    int           resolutionY)
{
    // =========================================================================
    // Debug modes 5 and 6: endpoint-splat visualization.
    // Skip compaction, training, and inference. Just project endpoints to camera.
    // =========================================================================
    if (debugMode == 5 || debugMode == 6)
    {
        const float3 aabbRange = make_float3(aabbMax.x - aabbMin.x,
                                             aabbMax.y - aabbMin.y,
                                             aabbMax.z - aabbMin.z);

        // // Clear bufferAccum so atomicAdds start from zero each frame.
        // cudaMemsetAsync(bufferAccum, 0, static_cast<size_t>(numPixels) * sizeof(float4), stream);

        if (debugMode == 5)
        {
            // d_renderOutput was written by raygen (chRadiance NEE at R) — splat directly.
            device_bdrc_splat_endpoints(stream, bufferAccum,
                sys->d_resampledEndpoints, sys->d_renderOutput, sys->d_resampledValid,
                numPixels, aabbMin, aabbRange, cameraP, cameraU, cameraV, cameraW,
                resolutionX, resolutionY, iteration);
        }
        else // debugMode == 6
        {
            // Compact per-pixel light buffer, then splat the dense result.
            cudaMemsetAsync(sys->d_compactLightCount, 0, sizeof(unsigned int), stream);
            device_bdrc_compact_light(stream,
                sys->d_compactLightEndpoints, sys->d_compactLightTargets,
                sys->d_compactLight_p_e_area,
                sys->d_compactLightCount,
                sys->d_lightEndpoints, sys->d_lightTargets,
                sys->d_light_p_e_area,
                sys->d_lightValid, batchSize);
            cudaStreamSynchronize(stream);
            unsigned int lightCount6 = 0;
            cudaMemcpy(&lightCount6, sys->d_compactLightCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (lightCount6 > 0)
                device_bdrc_splat_endpoints(stream, bufferAccum,
                    sys->d_compactLightEndpoints, sys->d_compactLightTargets, nullptr,
                    (int)lightCount6, aabbMin, aabbRange, cameraP, cameraU, cameraV, cameraW,
                    resolutionX, resolutionY, iteration);
        }

        // Clear all valid flags and training counter (same as normal dispatch).
        cudaMemsetAsync(sys->d_renderValid,       0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_trainNextValid,     0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_lightValid,         0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_resampledValid,     0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(d_trainCount,              0, sizeof(unsigned int),             stream);
        cudaStreamSynchronize(stream);
        return;
    }

    // Compact per-pixel light buffers (written by raygen) → dense for tcnn training.
    cudaMemsetAsync(sys->d_compactLightCount, 0, sizeof(unsigned int), stream);
    device_bdrc_compact_light(stream,
        sys->d_compactLightEndpoints, sys->d_compactLightTargets,
        sys->d_compactLight_p_e_area,
        sys->d_compactLightCount,
        sys->d_lightEndpoints, sys->d_lightTargets,
        sys->d_light_p_e_area,
        sys->d_lightValid, batchSize);

    // Compact sparse pixel-indexed resampled (eye-side) buffers.
    cudaMemsetAsync(sys->d_compactResampledCount, 0, sizeof(unsigned int), stream);
    device_bdrc_compact_resampled(stream,
        sys->d_compactResampledEndpoints, sys->d_compactResampled_p_e_area,
        sys->d_compactResampledCount,
        sys->d_resampledEndpoints, sys->d_resampled_p_e_area,
        sys->d_resampledValid, batchSize);

    // Sync stream then batch-read all training counts in one pass.
    cudaStreamSynchronize(stream);
    unsigned int trainCount = 0, lightCount = 0, resampledCount = 0;
    cudaMemcpy(&trainCount,     d_trainCount,                 sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lightCount,     sys->d_compactLightCount,     sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&resampledCount, sys->d_compactResampledCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // printf("[bdrc] iter=%u resampled=%u light=%u train=%u\n",
    //        iteration, resampledCount, lightCount, trainCount);

    if (resampledCount > 0 && iteration % 100 == 0) {                                                                                                                                    
      const uint32_t n = std::min(resampledCount, 4u);                                                                                                                                   
      std::vector<float> tmp(n);                                                                                                                                                         
      cudaMemcpy(tmp.data(), sys->d_compactResampled_p_e_area,                                                                                                                           
                 n * sizeof(float), cudaMemcpyDeviceToHost);                                                                                                                             
      printf("[mask-dbg] resampled p_e_area[0..%u]:", n);                                                                                                                              
      for (auto v : tmp) printf(" %.3e", v);                                                                                                                                             
      printf("\n");                                                                                                                                                                    
    }     

    constexpr uint32_t GRAN = 256u;
    const auto padTo256 = [](uint32_t n) -> uint32_t {
        return n > 0u ? (n + GRAN - 1u) / GRAN * GRAN : 0u;
    };
    const uint32_t paddedTrainCount     = padTo256(trainCount);
    uint32_t       paddedLightCount = padTo256(lightCount);
    if (paddedLightCount > batchSize) paddedLightCount = batchSize;
    const uint32_t paddedResampledCount = padTo256(resampledCount);

    if (bdrcPhase == 0)
    {
        // ====================================================================
        // Phase 1: Train mask cache only.
        // Eye cache shown (EMA query) but not trained.
        // ====================================================================

        // 1. Train maskNrc — single combined optimizer step matching Python's formulation:
        //      loss = eye_loss + (N_light/N_eye) * light_loss
        //    Both means are normalized by N_eye, so both passes share gradScale =
        //    lossScale / (paddedWidth * N_eye).  Eye samples are also weighted by
        //    throughput luminance to match Python's eye_throughput_w.
        //    Zero-pad p_e_area tails so padding samples contribute zero gradient.
        if (!freezeTraining)
        {
            if (iteration % 100 == 0)                                                                                                                                                                  
                printf("[bdrc] iter=%u phase=%d freeze=%d resampled=%u light=%u train=%u\n",
                    iteration, bdrcPhase, (int)freezeTraining, resampledCount, lightCount, trainCount);

            // Both eye and light passes use normBatch = n_eye (resampledCount, the true
            // anchor/reference count this frame) so the composite loss is
            //   total = (sum_eye_terms + sum_light_terms) / n_eye
            bool didAccumGrad = false;
            float eyeLossSum   = 0.0f;
            float lightLossSum = 0.0f;

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

            // 2. maskNrc EMA update
            sys->maskNrc->updateEma(stream);
        }

        // 3. Query eyeNrc (EMA) on renderFeatures -> d_renderOutput
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_renderOutput, 3u, batchSize);
            sys->eyeNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // 4. (debug mode 3) Query maskNrc at eye positions -> d_maskOutput for visualization
        if (debugMode == 3)
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_maskOutput, 1u, batchSize);
            sys->maskNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // Light-cache visualization is also valid during Phase 1: if the user selects
        // debug mode 2 or 4, query the light cache so device_bdrc_debug() never reads
        // stale/uninitialized d_lightOutput from a previous frame.
        if (debugMode == 2 || debugMode == 4)
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_lightOutput, 3u, batchSize);
            sys->lightNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // 5. Accumulate or visualize eye NRC output.
        //    In debug mode go directly to device_bdrc_debug; device_nrc_accumulate would
        //    write prefix+throughput*eyeOutput into bufferAccum with potentially stale
        //    prefix/throughput values, corrupting the progressive average read by the
        //    debug kernel on the next frame.
        if (debugMode == 0)
        {
            device_nrc_accumulate(
                stream,
                bufferAccum,
                sys->d_renderPrefix,
                sys->d_renderThroughput,
                sys->d_renderOutput,
                nullptr,  // BDRC eye cache: no reflectance factorization
                sys->d_renderValid,
                numPixels,
                iteration);
        }
        else
        {
            device_bdrc_debug(
                stream,
                bufferAccum,
                sys->d_renderOutput,
                sys->d_lightOutput,  // populated on-demand in Phase 1 for light/combined debug views
                sys->d_maskOutput,
                sys->d_renderValid,
                numPixels,
                iteration,
                debugMode);
        }

        // 7. Clear valid flags and training counter
        cudaMemsetAsync(sys->d_renderValid,   0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_trainNextValid, 0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_lightValid,    0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_resampledValid, 0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(d_trainCount,          0, sizeof(unsigned int),             stream);
        cudaStreamSynchronize(stream);
    }
    else
    {
        // ====================================================================
        // Phase 2: Mask frozen (EMA-only); train eye + light color caches.
        // Eye-cache training uses the compact slot batch (paddedTrainCount entries)
        // written by handleBdrcCacheLogic — mirrors the NRC fractional design.
        // ====================================================================

        // 1. Query maskNrc (EMA) at render positions D -> d_maskOutput
        // p_l_area(D) is needed by the composite: L = prefix + T*ĉ_e + p_l_area(D)*ĉ_l
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_maskOutput, 1u, batchSize);
            sys->maskNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // 2. (No separate scaling: p_e_area is already baked into lightTargets by bdrcLightIntegrator.)

        // 3. Query eyeNrc (EMA) on renderFeatures -> d_renderOutput
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_renderOutput, 3u, batchSize);
            sys->eyeNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // 4. Query lightNrc (EMA) on renderFeatures -> d_lightOutput
        {
            tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 3u, batchSize,
                                                   tcnn::MatrixLayout::ColumnMajor);
            tcnn::GPUMatrix<float> outputMat(sys->d_lightOutput, 3u, batchSize);
            sys->lightNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
        }

        // 5. Write BDRC composite or debug view to bufferAccum
        // d_maskOutput already holds p_l_area(D) from step 1 — used by both composite and debug mode 3.
        if (debugMode == 0)
        {
            device_bdrc_composite(
                stream,
                bufferAccum,
                sys->d_renderPrefix,
                sys->d_renderThroughput,
                sys->d_renderOutput,
                sys->d_lightOutput,
                sys->d_maskOutput,
                sys->d_renderValid,
                numPixels,
                iteration);
        }
        else
        {
            device_bdrc_debug(
                stream,
                bufferAccum,
                sys->d_renderOutput,
                sys->d_lightOutput,
                sys->d_maskOutput,
                sys->d_renderValid,
                numPixels,
                iteration,
                debugMode);
        }

        // 6. Eye-cache training — compact batch only (paddedTrainCount ≤ batchSize).
        if (!freezeTraining && paddedTrainCount > 0)
        {
            // Zero the padding range in d_trainTargets before building targets so that
            // device_nrc_build_targets (clamped to trainCount) leaves padding as zero.
            if (paddedTrainCount > trainCount && paddedTrainCount <= batchSize)
            {
                cudaMemsetAsync(sys->d_trainTargets + 3u * trainCount, 0,
                                (paddedTrainCount - trainCount) * 3u * sizeof(float), stream);
            }

            // 6a. EMA query on compact suffix-terminal features -> d_nrcNextOutput
            {
                tcnn::GPUMatrixDynamic<float> inputMat(sys->d_trainNextFeatures, 3u, paddedTrainCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                tcnn::GPUMatrix<float> outputMat(sys->d_nrcNextOutput, 3u, paddedTrainCount);
                sys->eyeNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
            }

            // ---- Virtual light emitter: compensate the eye training target for
            // ---- the L S+ D [any]+ family that validForEyeCache suppressed.
            {
                const uint32_t totalVirtSlots = static_cast<uint32_t>(VIRT_EMIT_MAX_DEPTH) * (uint32_t)numPixels;
                cudaMemsetAsync(sys->d_compactVirtEmitCount, 0, sizeof(unsigned int), stream);
                constexpr uint32_t THREADS = 256u;
                uint32_t blocks = (totalVirtSlots + THREADS - 1u) / THREADS;
                bdrc_compact_virt_emit_kernel<<<blocks, THREADS, 0, stream>>>(
                    sys->d_compactVirtEmitPos, sys->d_compactVirtEmitThroughput,
                    sys->d_virtEmitPixelIdx,
                    sys->d_compactVirtEmitCount,
                    sys->d_virtEmitPos, sys->d_virtEmitThroughput, sys->d_virtEmitValid,
                    (uint32_t)numPixels, totalVirtSlots);

                cudaStreamSynchronize(stream);
                unsigned int virtCount = 0;
                cudaMemcpy(&virtCount, sys->d_compactVirtEmitCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

                if (virtCount > 0u)
                {
                    const uint32_t paddedVirtCount = padTo256(virtCount);
                    if (paddedVirtCount > virtCount)
                        cudaMemsetAsync(sys->d_compactVirtEmitPos + 3u * virtCount, 0,
                                        (paddedVirtCount - virtCount) * 3u * sizeof(float), stream);

                    tcnn::GPUMatrixDynamic<float> inp(sys->d_compactVirtEmitPos, 3u, paddedVirtCount,
                                                      tcnn::MatrixLayout::ColumnMajor);
                    tcnn::GPUMatrix<float> out(sys->d_virtEmitOutput, 3u, paddedVirtCount);
                    sys->lightNrc->query(stream, inp, out, /*useEmaWeights=*/true);

                    cudaMemsetAsync(sys->d_pixelToTrainSlot, 0xFF, (uint32_t)numPixels * sizeof(unsigned int), stream);
                    if (trainCount > 0u)
                    {
                        blocks = (trainCount + THREADS - 1u) / THREADS;
                        bdrc_build_pixel_to_train_slot_kernel<<<blocks, THREADS, 0, stream>>>(
                            sys->d_pixelToTrainSlot, d_trainPixelIndices, trainCount);
                    }

                    blocks = (virtCount + THREADS - 1u) / THREADS;
                    bdrc_add_virt_emission_kernel<<<blocks, THREADS, 0, stream>>>(
                        sys->d_trainSuffixScaled, sys->d_pixelToTrainSlot,
                        sys->d_compactVirtEmitThroughput, sys->d_virtEmitOutput,
                        sys->d_virtEmitPixelIdx, virtCount);
                }
            }

            // 6b. Build eye training targets — valid slots only.
            //     Padding slots were zeroed above; running on paddedTrainCount would overwrite
            //     those zeros with stale previous-frame data from the padding range.
            device_nrc_build_targets(
                stream,
                sys->d_trainTargets,
                sys->d_trainSuffixScaled,
                sys->d_trainSuffThroughput,
                sys->d_nrcNextOutput,
                sys->d_trainThroughputAtK,
                nullptr,  // BDRC eye cache: no reflectance factorization
                sys->d_trainNextValid,
                static_cast<int>(trainCount));

            // 6c. Train eyeNrc on compact features + targets (pure MC, no bootstrap)
            {
                tcnn::GPUMatrixDynamic<float> inputMat(d_trainFeatures, 3u, paddedTrainCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                tcnn::GPUMatrix<float> targetMat(sys->d_trainTargets, 3u, paddedTrainCount);
                sys->eyeNrc->trainStep(stream, inputMat, targetMat);
            }
        }

        // 7. eyeNrc EMA update
        if (!freezeTraining)
            sys->eyeNrc->updateEma(stream);

        // 8. Train lightNrc L_light at compact Y positions.
        //    target = tp_Y_pre × f_Y_toward_S  (written by bdrcLightIntegrator).
        sys->m_lastLightLoss = 0.0f;
        if (!freezeTraining)
        {
            if (paddedLightCount > 0u)
            {
                // Zero-pad the AoS target tail so tcnn sees target=0 for padding slots.
                if (paddedLightCount > lightCount)
                    cudaMemsetAsync(sys->d_compactLightTargets + 3u * lightCount, 0,
                                    (paddedLightCount - lightCount) * 3u * sizeof(float), stream);
                tcnn::GPUMatrixDynamic<float> inputMat(sys->d_compactLightEndpoints, 3u, paddedLightCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                tcnn::GPUMatrix<float> targetMat(sys->d_compactLightTargets, 3u, paddedLightCount);
                sys->m_lastLightLoss += sys->lightNrc->trainStep(stream, inputMat, targetMat, tcnn::GradientMode::Accumulate);
            }

            // 9. Train lightNrc L_zero: compact resampled R with zero targets.
            if (paddedResampledCount > 0u)
            {
                tcnn::GPUMatrixDynamic<float> inputMat(sys->d_compactResampledEndpoints, 3u, paddedResampledCount,
                                                       tcnn::MatrixLayout::ColumnMajor);
                tcnn::GPUMatrix<float> targetMat(sys->d_zeroTargets3, 3u, paddedResampledCount);
                sys->m_lastLightLoss += sys->lightNrc->trainStep(stream, inputMat, targetMat);
            }

            // 10. lightNrc EM0A update
            sys->lightNrc->updateEma(stream);
        }

        // 11. Clear valid flags and training counter
        cudaMemsetAsync(sys->d_renderValid,    0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_trainNextValid,  0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_lightValid,     0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(sys->d_resampledValid,  0, batchSize * sizeof(unsigned int), stream);
        cudaMemsetAsync(d_trainCount,           0, sizeof(unsigned int),             stream);
        cudaStreamSynchronize(stream);
    }
}

void bdrcSaveWeights(const BdrcSystem* sys,
                     const std::string& eyePath,
                     const std::string& lightPath,
                     const std::string& maskPath)
{
    sys->eyeNrc->saveWeights(eyePath);
    sys->lightNrc->saveWeights(lightPath);
    sys->maskNrc->saveWeights(maskPath);
}

bool bdrcLoadWeights(BdrcSystem*        sys,
                     cudaStream_t       stream,
                     const std::string& eyePath,
                     const std::string& lightPath,
                     const std::string& maskPath)
{
    bool ok = sys->eyeNrc->loadWeights(eyePath, stream);
    ok     &= sys->lightNrc->loadWeights(lightPath, stream);
    ok     &= sys->maskNrc->loadWeights(maskPath, stream);
    return ok;
}