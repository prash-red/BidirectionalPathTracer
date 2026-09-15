#ifndef GLTF_RENDERER_MASKNGPCACHE_H
#define GLTF_RENDERER_MASKNGPCACHE_H

#include "base_ngp_cache.h"

class mask_ngp_cache : public base_ngp_cache
{
public:
    explicit mask_ngp_cache(cudaStream_t stream = nullptr);

    // Logistic NCE loss (density-ratio estimation) with correct p_e_area in the denominator.
    //
    // Trains p_l_area(x) directly via a binary logistic classifier between eye/reference
    // samples (label=0) and light/caustic samples (label=1):
    //   label=0 (eye):   L = -log(p_e / (p_l+p_e)),  dL/d(p_l) =  1 / (p_l+p_e)
    //   label=1 (light): L = -log(p_l / (p_l+p_e)),  dL/d(p_l) = -p_e / (p_l*(p_l+p_e))
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
    // normBatch: gradient normalizer — pass the shared resampled/anchor-sample count
    //   (resampledCount, clamped to >= 1) for BOTH eye and light passes, so the composite
    //   loss total = (sum_eye + sum_light) / resampledCount matches Python's
    //   alpha*mean(ref) + beta*(n_light/n_eye)*mean(pos) class-balance ratio.
    // d_throughputWeights: [batchSize] per-sample eye-vertex throughput luminance. This loss
    //   is intentionally unweighted by throughput — always pass nullptr for both eye
    //   (label=0) and light (label=1) passes.
    // gradMode: GradientMode::Overwrite for first pass, Accumulate for subsequent passes.
    //
    // Returns the sum of per-sample logistic-NCE loss values for this batch.
    // Call optimizerStep() once after all dreLossAccumGrad() calls for the frame.
    float dreLossAccumGrad(
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

