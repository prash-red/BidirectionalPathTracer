#ifndef GLTF_RENDERER_BASENGPCACHE_H
#define GLTF_RENDERER_BASENGPCACHE_H

#include "ngp_config.h"

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

class base_ngp_cache
{
public:
    base_ngp_cache(const ngp_config& config, int outputDim, cudaStream_t stream = nullptr);
    virtual ~base_ngp_cache() = default;

    base_ngp_cache(const base_ngp_cache&)            = delete;
    base_ngp_cache& operator=(const base_ngp_cache&) = delete;
    base_ngp_cache(base_ngp_cache&&) noexcept                 = default;
    base_ngp_cache& operator=(base_ngp_cache&&) noexcept      = default;

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
    virtual float trainStep(
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
    [[nodiscard]] std::string serialize(const std::string& name = "BaseNGPCache") const;

    // Direct network access for advanced use (e.g. save/load weights)
    tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>& network() { return *m_network; }

protected:
    ngp_config m_config;
    int       m_outputDim = 3;

    // ---- Core tcnn objects (owned via shared_ptr for tcnn API compatibility) --
    std::shared_ptr<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>> m_network;
    std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>>                m_optimizer;
    std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>>                     m_loss;

    // ---- Parameter buffers -------------------------------------------------
    // Layout: [ float * n_params | tcnn::network_precision_t * n_params | tcnn::network_precision_t * n_params ]
    //          ^m_paramsFP        ^m_params             ^m_paramGrads
    //
    // Mirrors the Trainer's m_params_buffer layout so the same optimizer
    // API (step(stream, loss_scale, fp, hp, grads)) works directly.
    tcnn::GPUMemory<uint8_t> m_paramsBuf;
    float*    m_paramsFP    = nullptr;
    tcnn::network_precision_t*  m_params      = nullptr;  // half-precision training weights
    tcnn::network_precision_t*  m_paramGrads  = nullptr;  // gradient accumulation buffer

    // ---- EMA state ---------------------------------------------------------
    bool  m_emaActive      = false;
    float m_emaDecay       = 0.999f;
    int   m_emaStartIter   = 0;
    int   m_emaUpdateCount = 0;
    bool  m_emaInitialized = false;

    tcnn::GPUMemory<tcnn::network_precision_t> m_emaParamsBuf;    // EMA shadow
    tcnn::GPUMemory<tcnn::network_precision_t> m_savedParamsBuf;  // saved training weights for swap
};



#endif //GLTF_RENDERER_BASENGPCACHE_H
