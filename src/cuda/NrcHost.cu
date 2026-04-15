//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// NrcHost.cu — Host-side NRC system managing the Forward NRC cache.
// Compiled as CUDA (nvcc) so tcnn headers are available.
//

#include "NrcHost.h"
#include "Nrc.h"
#include "NrcConfig.h"
#include "../HostKernels.h"

#include <cuda_runtime.h>
#include <memory>

#include <tiny-cuda-nn/gpu_matrix.h>

// ============================================================================
// NrcSystem — full definition (opaque to Application.cpp)
// ============================================================================

struct NrcSystem
{
  // ---- Forward NRC (Mueller 2021, 14-dim composite encoding) ---------------
  std::unique_ptr<Nrc>              fwdNrc;

  // ---- Shared buffers -----------------------------------------------------
  float*        d_renderThroughput    = nullptr; // [3 × batchSize]
  float*        d_renderPrefix        = nullptr; // [3 × batchSize]
  float*        d_renderOutput        = nullptr; // [3 × batchSize] filled by query()
  unsigned int* d_renderValid         = nullptr; // [batchSize]

  // ---- Forward NRC buffers -------------------------------------------------
  float*        d_renderFeatures      = nullptr; // [NRC_INPUT_DIM × batchSize]
  float*        d_trainNextFeatures   = nullptr; // [NRC_INPUT_DIM × batchSize]
  float*        d_trainSuffixScaled   = nullptr; // [3 × batchSize]
  float*        d_trainSuffThroughput = nullptr; // [3 × batchSize]
  float*        d_trainThroughputAtK  = nullptr; // [3 × batchSize]
  unsigned int* d_trainNextValid      = nullptr; // [batchSize]
  float*        d_nrcNextOutput       = nullptr; // [3 × batchSize] EMA query for trainNext
  float*        d_trainTargets        = nullptr; // [3 × batchSize] self-training targets

  void freeSharedBuffers()
  {
    cudaFree(d_renderThroughput);    d_renderThroughput    = nullptr;
    cudaFree(d_renderPrefix);        d_renderPrefix        = nullptr;
    cudaFree(d_renderOutput);        d_renderOutput        = nullptr;
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
  NrcConfig config;
  config.ema.enabled   = true;
  config.ema.decay     = 0.999f;
  config.ema.startIter = 0;

  sys->fwdNrc = std::make_unique<Nrc>(config, /*outputDim=*/3, stream);
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
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderValid),         batchSize      * sizeof(unsigned int));

  // ---- Forward NRC buffers --------------------------------------------------
  const uint32_t fwdInputDim = static_cast<uint32_t>(NrcConfig().inputDim); // 14
  cudaMalloc(reinterpret_cast<void**>(&sys->d_renderFeatures),      batchSize * fwdInputDim * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextFeatures),   batchSize * fwdInputDim * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffixScaled),   batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainSuffThroughput), batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainThroughputAtK),  batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainNextValid),      batchSize               * sizeof(unsigned int));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_nrcNextOutput),       batchSize * 3u          * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&sys->d_trainTargets),        batchSize * 3u          * sizeof(float));

  // Zero-initialise valid flags.
  cudaMemset(sys->d_renderValid,    0, batchSize * sizeof(unsigned int));
  cudaMemset(sys->d_trainNextValid, 0, batchSize * sizeof(unsigned int));
}

// ---- Shared pointer accessors -----------------------------------------------

float*        nrcRenderThroughput   (const NrcSystem* sys) { return sys->d_renderThroughput;    }
float*        nrcRenderPrefix       (const NrcSystem* sys) { return sys->d_renderPrefix;        }
float*        nrcRenderOutput       (const NrcSystem* sys) { return sys->d_renderOutput;        }
unsigned int* nrcRenderValid        (const NrcSystem* sys) { return sys->d_renderValid;         }

// ---- Forward NRC pointer accessors ------------------------------------------

float*        nrcRenderFeatures     (const NrcSystem* sys) { return sys->d_renderFeatures;      }
float*        nrcTrainNextFeatures  (const NrcSystem* sys) { return sys->d_trainNextFeatures;   }
float*        nrcTrainSuffixScaled  (const NrcSystem* sys) { return sys->d_trainSuffixScaled;   }
float*        nrcTrainSuffThroughput(const NrcSystem* sys) { return sys->d_trainSuffThroughput; }
float*        nrcTrainThroughputAtK (const NrcSystem* sys) { return sys->d_trainThroughputAtK;  }
unsigned int* nrcTrainNextValid     (const NrcSystem* sys) { return sys->d_trainNextValid;      }

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
  // Forward NRC dispatch (Mueller 2021) - Optimized
  // ========================================================================

  // Pipeline overview (async stream execution):
  // 1. Render query (NN inference) -> d_renderOutput
  // 2. Next-vertex query (NN inference) -> d_nrcNextOutput
  // 3. Fused accumulate+build_targets kernel (single launch)
  // 4. Training step (NN) using d_trainTargets
  // 5. EMA update
  // 6. Clear valid flags
  //
  // Only one explicit sync at the end. Neural network ops internally sync when needed.

  // 1. Render inference with EMA weights: fill d_renderOutput (14-dim query).
  {
    tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 14u, batchSize,
                                           tcnn::MatrixLayout::ColumnMajor);
    tcnn::GPUMatrix<float> outputMat(sys->d_renderOutput, 3u, batchSize);
    sys->fwdNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
  }

  // 2. EMA inference on suffix vertices: fill d_nrcNextOutput for target computation.
  {
    tcnn::GPUMatrixDynamic<float> inputMat(sys->d_trainNextFeatures, 14u, batchSize,
                                           tcnn::MatrixLayout::ColumnMajor);
    tcnn::GPUMatrix<float> outputMat(sys->d_nrcNextOutput, 3u, batchSize);
    sys->fwdNrc->query(stream, inputMat, outputMat, /*useEmaWeights=*/true);
  }

  // 3. Fused composite/debug + build targets (single kernel launch).
  // The neural network queries above internally synchronize before returning,
  // so d_renderOutput and d_nrcNextOutput are ready.
  if (debugMode == NRC_DEBUG_OFF)
  {
    device_nrc_accumulate_and_build_targets(
        stream,
        bufferAccum,
        sys->d_renderPrefix,
        sys->d_renderThroughput,
        sys->d_renderOutput,
        sys->d_renderValid,
        sys->d_trainTargets,
        sys->d_trainSuffixScaled,
        sys->d_trainSuffThroughput,
        sys->d_nrcNextOutput,
        sys->d_trainThroughputAtK,
        sys->d_trainNextValid,
        numPixels,
        iteration);
  }
  else
  {
    device_nrc_debug_and_build_targets(
        stream,
        bufferAccum,
        sys->d_renderOutput,
        sys->d_renderThroughput,
        sys->d_renderValid,
        sys->d_trainTargets,
        sys->d_trainSuffixScaled,
        sys->d_trainSuffThroughput,
        sys->d_nrcNextOutput,
        sys->d_trainThroughputAtK,
        sys->d_trainNextValid,
        numPixels,
        iteration,
        static_cast<int>(debugMode));
  }

  // 4. Training step (d_trainTargets ready after fused kernel completes).
  {
    tcnn::GPUMatrixDynamic<float> inputMat(sys->d_renderFeatures, 14u, batchSize,
                                           tcnn::MatrixLayout::ColumnMajor);
    tcnn::GPUMatrix<float> targetMat(sys->d_trainTargets, 3u, batchSize);
    sys->fwdNrc->trainStep(stream, inputMat, targetMat);
  }

  // 5. EMA update.
  sys->fwdNrc->updateEma(stream);

  // 6. Clear valid flags and sync once at the end.
  cudaMemsetAsync(sys->d_renderValid,    0, batchSize * sizeof(unsigned int), stream);
  cudaMemsetAsync(sys->d_trainNextValid, 0, batchSize * sizeof(unsigned int), stream);
  cudaStreamSynchronize(stream);
}
