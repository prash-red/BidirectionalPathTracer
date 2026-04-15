/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * NrcCache.h — C++ equivalent of bdrc/cache/base.py (BaseCache + create_ngp).
 *
 * Uses tcnn::NetworkWithInputEncoding directly instead of create_from_config,
 * giving explicit control over the training pipeline with the NRC paper-style
 * Composite encoding and fully fused MLP configuration.
 *
 * Why not create_from_config / Trainer::training_step:
 *   create_from_config returns a TrainableModel whose Trainer::training_step
 *   evaluates the loss on the raw network output (pre-activation). With
 *   output_activation="None", that means loss(logits, target) — wrong.
 *   By owning NetworkWithInputEncoding directly and calling forward -> loss ->
 *   backward -> optimizer step explicitly, loss evaluation and optimizer updates
 *   stay fully explicit and match the runtime training loop.
 *
 * Python → C++ mapping:
 *   create_ngp()               → NetworkWithInputEncoding constructor
 *   BaseCache.__init__()       → NrcCache::NrcCache()
 *   BaseCache.step()           → NrcCache::trainStep()
 *   BaseCache.update_ema()     → NrcCache::updateEma()
 *   BaseCache.swap_ema_in/out  → NrcCache::swapEmaIn/Out()
 *   BaseCache.set_learning_rate→ NrcCache::setLearningRate()
 *   ColorCache3D.query()       → NrcCache::query()
 */

#pragma once

#ifndef NRC_H
#define NRC_H

#include "NrcConfig.h"

#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace tcnn {
template <typename T>
class NetworkWithInputEncoding;
}

// Shorthand for the half-precision type tcnn uses on SM75+ hardware.
// Guard against redefinition when multiple headers define NrcNetT.
#ifndef NRC_NET_T_DEFINED
#define NRC_NET_T_DEFINED
using NrcNetT = tcnn::network_precision_t;
#endif

// ============================================================================
// NrcCache
//
// Manages one Neural Radiance Cache instance:
//   Composite(TriangleWave + OneBlob + Identity) -> FullyFusedMLP -> RGB
//
// Training loop (mirrors bdrc/training/train_ngp.py):
//
//   NrcCache cache(config, 3, stream);
//
//   // Per-iteration (column-major: rows=features, cols=samples)
//   cache.trainStep(stream, inputBatch, targetBatch);   // fwd+loss+bwd+Adam
//   cache.updateEma(stream);                            // EMA shadow update
//
//   // Inference
//   cache.query(stream, inputBatch, outputBatch);       // net(x) -> float32
//
//   // EMA inference (mirrors bdrc swap_ema_in/out)
//   cache.swapEmaIn(stream);
//   cache.query(stream, inputBatch, outputBatch);
//   cache.swapEmaOut(stream);
// ============================================================================

class Nrc
{
public:
    // -------------------------------------------------------------------------
    // Construction
    //
    // Builds NetworkWithInputEncoding with separate encoding and network JSON,
    // initializes params, creates Adam optimizer and RelativeL2 loss.
    // -------------------------------------------------------------------------
    Nrc(const NrcConfig& config, int outputDim, cudaStream_t stream = nullptr);
    ~Nrc() = default;

    Nrc(const Nrc&)            = delete;
    Nrc& operator=(const Nrc&) = delete;
    Nrc(Nrc&&) noexcept                 = default;
    Nrc& operator=(Nrc&&) noexcept      = default;

    // -------------------------------------------------------------------------
    // query — inference pass
    //
    // Calls inference_mixed_precision and casts half -> float32.
    //
    // input  : [inputDim  × batchSize] float32, column-major
    // output : [outputDim × batchSize] float32, column-major — written in-place
    //
    // useEmaWeights: when true, uses inference_params which point to the EMA
    //   shadow buffer (set by swapEmaIn). Pass false for normal training-weight
    //   inference. Calling swapEmaIn/Out is the preferred explicit API.
    // -------------------------------------------------------------------------
    void query(
        cudaStream_t                          stream,
        const tcnn::GPUMatrixDynamic<float>&  input,
        tcnn::GPUMatrix<float>&               output,
        bool                                  useEmaWeights = false);

    // -------------------------------------------------------------------------
    // trainStep — one explicit forward → loss → backward → optimizer step
    //
    // One explicit pass:
    //   1. m_network->forward()
    //   2. m_loss->evaluate()          - RelativeL2/RelativeL2Luminance
    //   3. m_network->backward()
    //   4. m_optimizer->step()         - Adam update on m_paramsFP / m_params
    //
    // input  : [inputDim  × batchSize] float32, column-major
    // target : [outputDim × batchSize] float32, column-major
    //          (RelativeL2 kernel handles padded_output_width > outputDim internally)
    //
    // Returns mean loss (float) — involves one device→host sync.
    // -------------------------------------------------------------------------
    float trainStep(
        cudaStream_t                          stream,
        const tcnn::GPUMatrixDynamic<float>&  input,
        const tcnn::GPUMatrix<float>&         target);

    // -------------------------------------------------------------------------
    // setLearningRate — mirrors bdrc BaseCache.set_learning_rate(factor)
    // -------------------------------------------------------------------------
    void setLearningRate(float factor);

    // =========================================================================
    // EMA — mirrors bdrc/cache/base.py BaseCache EMA methods exactly
    // =========================================================================

    void initEma(float decay = 0.999f, int startIter = 0);

    // updateEma — ADS bias-corrected EMA (after startIter warm-up):
    //   eta_t = 1 - decay^t
    //   shadow_t = decay * eta_{t-1}/eta_t * shadow_{t-1} + (1-decay)/eta_t * params_t
    // Call after every trainStep(). No-op before startIter is reached.
    void updateEma(cudaStream_t stream);

    // swapEmaIn  — copy EMA shadow into m_params (and save old m_params).
    //   After this call, query() uses EMA weights.
    void swapEmaIn(cudaStream_t stream);

    // swapEmaOut — restore m_params saved by swapEmaIn.
    //   After this call, trainStep() / query() use training weights again.
    void swapEmaOut(cudaStream_t stream);

    [[nodiscard]] bool emaActive() const { return m_emaActive && m_emaInitialized; }

    // =========================================================================
    // Diagnostics
    // =========================================================================
    [[nodiscard]] size_t      numParams() const { return m_network ? m_network->n_params() : 0; }
    [[nodiscard]] std::string serialize(const std::string& name = "NrcCache") const;

    // Direct network access for advanced use (e.g. save/load weights)
    tcnn::NetworkWithInputEncoding<NrcNetT>& network() { return *m_network; }

private:
    NrcConfig m_config;
    int       m_outputDim = 0;

    // ---- Core tcnn objects (owned via shared_ptr for tcnn API compatibility) --
    std::shared_ptr<tcnn::NetworkWithInputEncoding<NrcNetT>> m_network;
    std::shared_ptr<tcnn::Optimizer<NrcNetT>>                m_optimizer;
    std::shared_ptr<tcnn::Loss<NrcNetT>>                     m_loss;

    // ---- Parameter buffers -------------------------------------------------
    // Layout: [ float * n_params | NrcNetT * n_params | NrcNetT * n_params ]
    //          ^m_paramsFP        ^m_params             ^m_paramGrads
    //
    // Mirrors the Trainer's m_params_buffer layout so the same optimizer
    // API (step(stream, loss_scale, fp, hp, grads)) works directly.
    tcnn::GPUMemory<uint8_t> m_paramsBuf;
    float*    m_paramsFP    = nullptr;
    NrcNetT*  m_params      = nullptr;  // half-precision training weights
    NrcNetT*  m_paramGrads  = nullptr;  // gradient accumulation buffer

    // ---- EMA state ---------------------------------------------------------
    bool  m_emaActive      = false;
    float m_emaDecay       = 0.999f;
    int   m_emaStartIter   = 0;
    int   m_emaUpdateCount = 0;
    bool  m_emaInitialized = false;

    tcnn::GPUMemory<NrcNetT> m_emaParamsBuf;    // EMA shadow
    tcnn::GPUMemory<NrcNetT> m_savedParamsBuf;  // saved training weights for swap
};

#endif // NRC_H