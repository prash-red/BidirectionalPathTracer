//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_ngp_cache — the single neural field of the unified transport cache.
//
// Stores incident flux per unit world area on diffuse receivers. Not radiance,
// not outgoing, no BSDF baked in: the receiver BSDF is applied at the query, in
// the renderer, on every path that reads the field.
//
// Input : 5 dims. [0..2] world position normalized into the scene AABB, fed to
//         a hash grid. [3] sign_cam — the face's orientation relative to a
//         FIXED reference origin, never the live camera. [4] arrival side.
//         Both labels bypass the grid through an Identity nested encoding.
//         Layer and side stay separate inputs; their product would give a thin
//         shell's dark inner back layer the label of its lit outer front.
// Output: 3 channels, raw from the MLP. softplus is applied by this class
//         (query()) and differentiated analytically by the loss kernel, which
//         is why the network itself carries no output activation — letting
//         tcnn apply softplus too would square the activation.
//
// Supervision: one weighted density-ratio objective over two positive streams
// (photon deposits and gather starts) against one surface reference measure.
// For any Bregman generator the fitted field satisfies U* = sum_s p_s w_bar_s,
// so disjoint path families supervising one network fit their sum. The
// objective must stay plain: any per-stream reweighting (self-normalized
// weights, a learned global scale, a proxy mixture, a detached reweighting)
// rescales one stream's mean weight against the other's and moves the fixed
// point off that sum.
//
#pragma once
#ifndef GLTF_RENDERER_UNIFIEDNGPCACHE_H
#define GLTF_RENDERER_UNIFIEDNGPCACHE_H

#include "base_ngp_cache.h"
#include "dre_config.h"

// The field's fixed input/output widths. Device code packing a query must use
// UNIFIED_INPUT_DIM as its per-lane stride (see unified_features.h).
#define UNIFIED_INPUT_DIM   5
#define UNIFIED_LABEL_DIMS  2
#define UNIFIED_OUTPUT_DIM  3

class unified_ngp_cache : public base_ngp_cache
{
public:
    explicit unified_ngp_cache(const DreConfig& dre, cudaStream_t stream = nullptr);

    // Rejects alpha != beta for the raw-ratio members: those read
    // u* = (beta/alpha) * sum_s p_s w_bar_s, so unequal constants silently
    // rescale the whole field off the stream sum.
    void setDreConfig(const DreConfig& dre);
    [[nodiscard]] const DreConfig& dreConfig() const { return m_dre; }

    // Inference with softplus applied in place, so callers read u directly.
    //
    // useEmaWeights selects the teacher. Every tail read, render query and
    // debug view passes true; only the fit below touches the live weights. A
    // bootstrapped estimator that ends by reading the field's own EMA is
    // unbiased for the measure the current field induces, and the true solution
    // is the fixed point of the iteration.
    void query(cudaStream_t                         stream,
               const tcnn::GPUMatrixDynamic<float>& input,
               tcnn::GPUMatrix<float>&              output,
               bool                                 useEmaWeights);

    // One forward + analytic dL/dOutput + backward over the WHOLE batch
    // (photon lanes and gather lanes together). Because dL/dOutput is
    // per-sample there is no aliasing between lanes, so both streams reach one
    // backward and one optimizer step without any Overwrite/Accumulate dance.
    //
    // d_weight   [3 x batchSize] RGB positive weight w
    // d_density  [batchSize]     reference density q; <= 0 disables the lane
    // d_scalePos [batchSize]     positive normalizer (1/N_pos), 0 if none
    // d_scaleRef [batchSize]     reference normalizer (1/N_ref), 0 if none
    //
    // Each stream is normalized by its own draw count, not by the live count of
    // valid samples in the frame: a frame with fewer valid samples must not
    // implicitly renormalize its gradient up. A disabled lane therefore keeps
    // its denominator and simply contributes nothing.
    //
    // Returns the summed per-lane loss (one device->host sync).
    float dreAccumGrad(cudaStream_t                         stream,
                       const tcnn::GPUMatrixDynamic<float>& input,
                       const float*                         d_weight,
                       const float*                         d_density,
                       const float*                         d_scalePos,
                       const float*                         d_scaleRef,
                       uint32_t                             batchSize);

    void optimizerStep(cudaStream_t stream);

    // Analytic fixed-point check, run once at construction.
    //
    // Both streams are placed at one input point. Photons carry weight 2 with
    // scalePos = 1/N; gathers carry weight 3 with scalePos = scaleRef = 1/N.
    // Then sum_s p_s w_bar_s = 5q, and every member of the family has that same
    // minimizer: the posterior members' fixed point does not move with alpha,
    // and the raw-ratio members read (beta/alpha) * 5q = 5q when alpha == beta.
    // Prints the recovered u against 5q and returns whether it lands within 15%.
    //
    // q selects the conditioning regime: near 1e-1 every generator is well
    // conditioned, while 1e-4 is the realistic ratio at which the raw-ratio
    // members' r^2 reference term becomes heavy-tailed.
    bool selfTest(cudaStream_t stream, float q = 1.0e-4f);

private:
    DreConfig m_dre;
};

#endif // GLTF_RENDERER_UNIFIEDNGPCACHE_H
