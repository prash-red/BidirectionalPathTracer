#ifndef GLTF_RENDERER_MASKNGPCACHE_H
#define GLTF_RENDERER_MASKNGPCACHE_H

#include "base_ngp_cache.h"

class mask_ngp_cache : public base_ngp_cache
{
public:
    explicit mask_ngp_cache(cudaStream_t stream = nullptr);

    float trainStep(
        cudaStream_t                         stream,
        const tcnn::GPUMatrixDynamic<float>& input,
        const tcnn::GPUMatrix<float>&        target) override;

    // Brier-score DRE loss with correct p_e_area in the denominator.
    //
    // Trains p_l_area(x) directly:
    //   alpha(x) = p_l_area(x) / (p_l_area(x) + p_e_area(x))
    //   label=0 (eye):   L = alpha²,        dL/d(p_l) =  2*p_l*p_e / (p_l+p_e)³
    //   label=1 (light): L = (alpha-1)²,    dL/d(p_l) = -2*p_e²    / (p_l+p_e)³
    //
    // d_p_e_area: [batchSize] device buffer, one area-measure PDF per sample.
    // label: 0 = eye/resampled samples, 1 = light samples.
    float dreLossTrainStep(
        cudaStream_t                         stream,
        const tcnn::GPUMatrixDynamic<float>& input,
        const float*                         d_p_e_area,
        int                                  label);

    // Accumulate DRE gradients without taking an optimizer step.
    //
    // normBatch: gradient normalizer — use paddedResampledCount for BOTH eye and light
    //   passes so the light term is implicitly balanced by (n_light/n_eye) as in Python.
    // d_throughputWeights: [batchSize] per-sample eye-vertex throughput luminance; nullptr = uniform 1.0.
    //   Pass for both eye (label=0) and light (label=1) passes to match Python's throughput weighting.
    // gradMode: GradientMode::Overwrite for first pass, Accumulate for subsequent passes.
    //
    // Call optimizerStep() once after all dreLossAccumGrad() calls for the frame.
    void dreLossAccumGrad(
        cudaStream_t                         stream,
        const tcnn::GPUMatrixDynamic<float>& input,
        const float*                         d_p_e_area,
        int                                  label,
        uint32_t                             normBatch,
        const float*                         d_throughputWeights,
        tcnn::GradientMode                   gradMode);

    // Single Adam optimizer step — call after all dreLossAccumGrad() calls.
    void optimizerStep(cudaStream_t stream);
};

#endif // GLTF_RENDERER_MASKNGPCACHE_H

