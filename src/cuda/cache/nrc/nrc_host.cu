//
//
// NrcHost.cu — Host-side NRC system managing the Forward NRC cache.
// Compiled as CUDA (nvcc) so tcnn headers are available.
//

#include "nrc_host.h"
#include "nrc.h"
#include "nrc_config.h"
#include "../../../HostKernels.h"

#include <cuda_runtime.h>
#include <memory>

#include <tiny-cuda-nn/gpu_matrix.h>

// ============================================================================
// NrcSystem — full definition (opaque to Application.cpp)
// ============================================================================

struct NrcSystem
{
  // ---- Forward NRC (Mueller 2021, 14-dim composite encoding) ---------------
  std::unique_ptr<nrc>              fwdNrc;

  // ---- Shared render-query buffers (pixel-indexed) -------------------------
  float*        d_renderThroughput    = nullptr; // [3 × batchSize]
  float*        d_renderPrefix        = nullptr; // [3 × batchSize]
  float*        d_renderOutput        = nullptr; // [3 × batchSize] filled by query()
  float*        d_renderReflectance   = nullptr; // [3 × batchSize] clamp(α+β, 0.01) at K
  unsigned int* d_renderValid         = nullptr; // [batchSize]

  // ---- Forward NRC training buffers ----------------------------------------
  float*        d_renderFeatures      = nullptr; // [NRC_INPUT_DIM × batchSize] pixel-indexed
  float*        d_trainNextFeatures   = nullptr; // [NRC_INPUT_DIM × batchSize]
  float*        d_trainSuffixScaled   = nullptr; // [3 × batchSize]
  float*        d_trainSuffThroughput = nullptr; // [3 × batchSize]
  float*        d_trainThroughputAtK  = nullptr; // [3 × batchSize]
  unsigned int* d_trainNextValid      = nullptr; // [batchSize]
  float*        d_nrcNextOutput       = nullptr; // [3 × batchSize] EMA query for trainNext
  float*        d_trainTargets        = nullptr; // [3 × batchSize] self-training targets

  // ---- Compact training batch (Option B: stream compaction via atomicAdd) --
  unsigned int* d_trainCount          = nullptr; // [1] atomic counter
  unsigned int* d_trainPixelIndices   = nullptr; // [batchSize] pixel index per slot (diagnostic)
  float*        d_trainFeatures       = nullptr; // [NRC_INPUT_DIM × batchSize]  compact
  float*        d_trainReflectance    = nullptr; // [3 × batchSize] clamp(α+β, 0.01) per training slot

  // ---- Per-frame diagnostics ----------------------------------------------
  float        m_lastLoss        = 0.0f;
  unsigned int m_lastTrainCount  = 0u;

  void freeSharedBuffers()
  {
    cudaFree(d_renderThroughput);    d_renderThroughput    = nullptr;
    cudaFree(d_renderPrefix);        d_renderPrefix        = nullptr;
    cudaFree(d_renderOutput);        d_renderOutput        = nullptr;
    cudaFree(d_renderReflectance);   d_renderReflectance   = nullptr;
    cudaFree(d_renderValid);         d_renderValid         = nullptr;
  }

  void freeForwardBuffers()
  {
    cudaFree(d_renderFeatures);      d_renderFeatures      = nullptr;
    cudaFree(d_trainNextFeatures);   d_trainNextFeatures   = nullptr;
    cudaFree(d_trainSuffixScaled);   d_trainSuffixScaled   = nullptr;
    cudaFree(d_trainSuffThroughput); d_trainSuffThroughput = nullptr;
    cudaFree(d_trainThroughputAtK);  d_trainThroughputAtK  = nullptr;
    cudaFree(d_trainNextValid);      d_trainNextValid      = nullptr;
    cudaFree(d_nrcNextOutput);       d_nrcNextOutput       = nullptr;
    cudaFree(d_trainTargets);        d_trainTargets        = nullptr;
    cudaFree(d_trainCount);          d_trainCount          = nullptr;
    cudaFree(d_trainPixelIndices);   d_trainPixelIndices   = nullptr;
    cudaFree(d_trainFeatures);       d_trainFeatures       = nullptr;
    cudaFree(d_trainReflectance);    d_trainReflectance    = nullptr;
  }

  void freeAllBuffers()
  {
    freeSharedBuffers();
    freeForwardBuffers();
  }

  ~NrcSystem() { freeAllBuffers(); }
};

// ============================================================================
// Internal helpers
// ============================================================================

static void initForwardNrc(NrcSystem* sys, cudaStream_t stream)
{
  nrc_config config;
  config.ema.enabled   = true;
  config.ema.decay     = 0.999f;
  config.ema.startIter = 0;

  sys->fwdNrc = std::make_unique<nrc>(config, /*outputDim=*/3, stream);
  sys->fwdNrc->initEma(config.ema.decay, config.ema.startIter);
}

// ============================================================================
// API implementation
// ============================================================================

NrcSystem* nrcCreate(cudaStream_t stream)
{
  NrcSystem* sys = new NrcSystem();
  initForwardNrc(sys, stream);
  return sys;
}

void nrcDestroy(NrcSystem* sys)
{
  delete sys;
}

void nrcAllocBuffers(NrcSystem* sys, uint32_t batchSize)
{
  sys->freeAllBuffers();

  // ---- Shared buffers (always allocated) ------------------------------------
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderThroughput),    batchSize * 3u * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderPrefix),        batchSize * 3u * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderOutput),        batchSize * 3u * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderReflectance),   batchSize * 3u * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderValid),         batchSize      * sizeof(unsigned int));

  // ---- Forward NRC buffers -------------------------------------------------
  const uint32_t fwdInputDim = static_cast<uint32_t>(nrc_config().inputDim); // 14
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderFeatures),      batchSize * fwdInputDim * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextFeatures),   batchSize * fwdInputDim * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffixScaled),   batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffThroughput), batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainThroughputAtK),  batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextValid),      batchSize               * sizeof(unsigned int));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_nrcNextOutput),       batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainTargets),        batchSize * 3u          * sizeof(float));

  // ---- Compact training batch buffers --------------------------------------
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainCount),          sizeof(unsigned int));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainPixelIndices),   batchSize * sizeof(unsigned int));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainFeatures),       batchSize * fwdInputDim * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainReflectance),    batchSize * 3u          * sizeof(float));

  // Zero-initialise all flags and the training counter.
  cudaMemset(sys->d_renderValid,    0, batchSize * sizeof(unsigned int));
  cudaMemset(sys->d_trainNextValid, 0, batchSize * sizeof(unsigned int));
  cudaMemset(sys->d_trainCount,     0, sizeof(unsigned int));
}

// ---- Shared pointer accessors -----------------------------------------------

float*        nrcRenderThroughput   (const NrcSystem* sys) { return sys->d_renderThroughput;    }
float*        nrcRenderPrefix       (const NrcSystem* sys) { return sys->d_renderPrefix;        }
float*        nrcRenderOutput       (const NrcSystem* sys) { return sys->d_renderOutput;        }
float*        nrcRenderReflectance  (const NrcSystem* sys) { return sys->d_renderReflectance;   }
unsigned int* nrcRenderValid        (const NrcSystem* sys) { return sys->d_renderValid;         }

// ---- Forward NRC pointer accessors ------------------------------------------

float*        nrcRenderFeatures     (const NrcSystem* sys) { return sys->d_renderFeatures;      }
float*        nrcTrainNextFeatures  (const NrcSystem* sys) { return sys->d_trainNextFeatures;   }
float*        nrcTrainSuffixScaled  (const NrcSystem* sys) { return sys->d_trainSuffixScaled;   }
float*        nrcTrainSuffThroughput(const NrcSystem* sys) { return sys->d_trainSuffThroughput; }
float*        nrcTrainThroughputAtK (const NrcSystem* sys) { return sys->d_trainThroughputAtK;  }
unsigned int* nrcTrainNextValid     (const NrcSystem* sys) { return sys->d_trainNextValid;      }

// ---- Compact training batch accessors ---------------------------------------

unsigned int* nrcTrainCount        (const NrcSystem* sys) { return sys->d_trainCount;         }
unsigned int* nrcTrainPixelIndices (const NrcSystem* sys) { return sys->d_trainPixelIndices;  }
float*        nrcTrainFeatures     (const NrcSystem* sys) { return sys->d_trainFeatures;      }
float*        nrcTrainReflectance  (const NrcSystem* sys) { return sys->d_trainReflectance;   }

float        nrcGetLastLoss       (const NrcSystem* sys) { return sys->m_lastLoss;           }
unsigned int nrcGetLastTrainCount (const NrcSystem* sys) { return sys->m_lastTrainCount;     }

// ---- Per-frame dispatch -----------------------------------------------------

void nrcDispatch(NrcSystem*   sys,
                 cudaStream_t stream,
                 float4*      bufferAccum,
                 int          numPixels,
                 uint32_t     batchSize,
                 unsigned int iteration,
                 NrcDebugMode debugMode)
{
  // ========================================================================
  // Forward NRC dispatch (Mueller 2021) — compact training batch
  // ========================================================================
  //
  // Pipeline:
  //   1. Read trainCount D2H  (one sync: how many training slots raygen wrote)
  //   2. Render inference  (EMA, full pixel batch) → d_renderOutput
  //   3. Next-vertex inference (EMA, compact train batch) → d_nrcNextOutput
  //   4. Accumulate / debug-visualize (pixel batch)
  //   5. Build training targets (compact train batch)
  //   6. Training step (compact train batch)
  //   7. EMA update
  //   8. Clear valid flags and training counter
  //
  // The compact training batch uses only the ~3% of pixels that traced a
  // multi-vertex suffix in raygen, giving a faithful separation between the
  // render pass (all NRC-terminated pixels) and the training pass (the subset
  // that extended their path to collect a self-training target).

  // 1. Read training slot count written by raygen this frame.
  unsigned int trainCount = 0;
  cudaMemcpy(&trainCount, sys->d_trainCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  // Round up to tcnn BATCH_SIZE_GRANULARITY (256) so matrix dimensions are valid.
  constexpr uint32_t GRAN = 256u;
  const uint32_t paddedTrainCount = (trainCount + GRAN - 1u) / GRAN * GRAN;

  // Zero the padding range in d_trainTargets so tcnn sees target=0 for unused slots.
  if (paddedTrainCount > trainCount && paddedTrainCount <= batchSize)
  {
    cudaMemsetAsync(sys->d_trainTargets + 3u * trainCount, 0,
                    (paddedTrainCount - trainCount) * 3u * sizeof(float), stream);
  }

  // 2. Render inference with EMA weights on the full pixel batch.
  {
    tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 14u, batchSize,
                                           tcnn::MatrixLayout::ColumnMajor);
    tcnn::GPUMatrix<float> outputMat(sys->d_renderOutput, 3u, batchSize);
    sys->fwdNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
  }

  // 3. Accumulate / visualize (pixel batch).
  if (debugMode == NRC_DEBUG_OFF)
  {
    device_nrc_accumulate(
        stream, bufferAccum,
        sys->d_renderPrefix, sys->d_renderThroughput, sys->d_renderOutput,
        sys->d_renderReflectance, sys->d_renderValid,
        numPixels, iteration);
  }
  else
  {
    device_nrc_debug_accumulate(
        stream, bufferAccum,
        sys->d_renderOutput, sys->d_renderThroughput, sys->d_renderReflectance,
        sys->d_renderValid, numPixels, iteration, static_cast<int>(debugMode));
  }

  // 4–6. Training: only runs if there are training samples and not in cache-only debug mode.
  //      NRC_DEBUG_CACHE_ONLY freezes the network for a stable visualization.
  if (debugMode != NRC_DEBUG_CACHE_ONLY && paddedTrainCount > 0)
  {
    // 4. EMA inference on compact suffix terminal features → d_nrcNextOutput.
    {
      tcnn::GPUMatrixDynamic<float> inputMat(sys->d_trainNextFeatures, 14u, paddedTrainCount,
                                             tcnn::MatrixLayout::ColumnMajor);
      tcnn::GPUMatrix<float> outputMat(sys->d_nrcNextOutput, 3u, paddedTrainCount);
      sys->fwdNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
    }

    // 5. Build training targets from compact slot data — valid slots only.
    //    paddedTrainCount pads to 256 for tcnn, but the padding range was already zeroed
    //    above.  Running k_nrc_build_targets on the padding slots would overwrite those
    //    zeros with stale previous-frame data, injecting garbage into the training batch.
    device_nrc_build_targets(
        stream,
        sys->d_trainTargets,
        sys->d_trainSuffixScaled,
        sys->d_trainSuffThroughput,
        sys->d_nrcNextOutput,
        sys->d_trainThroughputAtK,
        sys->d_trainReflectance,
        sys->d_trainNextValid,
        static_cast<int>(trainCount));

    // 6. Training step on compact features + targets.
    {
      tcnn::GPUMatrixDynamic<float> inputMat(sys->d_trainFeatures, 14u, paddedTrainCount,
                                             tcnn::MatrixLayout::ColumnMajor);
      tcnn::GPUMatrix<float> targetMat(sys->d_trainTargets, 3u, paddedTrainCount);
      sys->m_lastLoss = sys->fwdNrc->trainStep(stream, inputMat, targetMat);
    }
    sys->m_lastTrainCount = trainCount;

    // 7. EMA update.
    sys->fwdNrc->updateEma(stream);
  }

  // 8. Clear per-frame state: valid flags and training counter.
  cudaMemsetAsync(sys->d_renderValid,    0, batchSize * sizeof(unsigned int), stream);
  cudaMemsetAsync(sys->d_trainNextValid, 0, batchSize * sizeof(unsigned int), stream);
  cudaMemsetAsync(sys->d_trainCount,     0, sizeof(unsigned int),             stream);
  cudaStreamSynchronize(stream);
}