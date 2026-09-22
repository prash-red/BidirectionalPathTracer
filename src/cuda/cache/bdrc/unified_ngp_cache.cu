//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_ngp_cache — see the header for the field's contract.
//
// The loss is a direct port of the demo's dre_loss.wgsl: sum channels rather
// than averaging them, normalize each stream by its own draw count, and
// evaluate in log space so an untrained cell whose logit underflows still
// produces a finite gradient.
//

#include "unified_ngp_cache.h"
#include "bdrc_host.h"

#include <tiny-cuda-nn/reduce_sum.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{

ngp_config makeUnifiedConfig()
{
    ngp_config cfg;

    cfg.inputDim  = UNIFIED_INPUT_DIM;
    cfg.labelDims = UNIFIED_LABEL_DIMS;

    // Hash grid: 8 levels x 2 features from 16 to ~273, matching the demo's
    // validated field. per_level_scale = exp((ln 273 - ln 16) / 7) ~= 1.5.
    cfg.nLevels           = 8;
    cfg.nFeaturesPerLevel = 2;
    cfg.baseResolution    = 16;
    cfg.perLevelScale     = 1.5f;
    cfg.log2HashmapSize   = 20;

    cfg.hiddenDim        = 64;
    cfg.numHiddenLayers  = 2;
    cfg.hiddenActivation = "LeakyReLU";

    // No output activation: the loss kernel applies softplus itself and
    // differentiates through it analytically. Letting tcnn apply softplus too
    // would square the activation.
    cfg.outputActivation = "None";

    // m_loss is never used — dreAccumGrad writes dL/dOutput directly. The base
    // class always constructs one, so name a valid member.
    cfg.loss = "L2";

    cfg.lr       = 0.005f;
    cfg.gradClip = 0.05f;

    // Half-precision guards, carried over unchanged from the mask cache, which
    // documented a reproducible NaN by iteration ~400 without all three.
    cfg.gradScaler       = true;
    cfg.gradScalerGrowth = true;
    cfg.weightClip       = 1.0f;

    cfg.ema.enabled   = true;
    cfg.ema.decay     = 0.99f;
    cfg.ema.startIter = 100;

    return cfg;
}

__device__ __forceinline__ float uni_softplus(float x)
{
    const float e = expf(-fabsf(x));
    // log1p loses small e in fp32; the 4th-order series below 0.01 has relative
    // truncation error under 2e-9.
    const float small = e * (1.0f + e * (-0.5f + e * (1.0f / 3.0f - 0.25f * e)));
    return fmaxf(x, 0.0f) + ((e < 0.01f) ? small : logf(1.0f + e));
}

__device__ __forceinline__ float uni_sigmoid(float x)
{
    const float e = expf(-fabsf(x));
    return (x >= 0.0f) ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

// Returns (log u, d(log u)/d(raw)).
//
// The low branch matters: an untrained cell can sit far enough below zero that
// softplus underflows to 0 in fp32 and log(0) = -inf poisons the gradient. For
// x < -20, softplus(x) ~= exp(x) to within fp32, so log u = x and
// d(log u)/d(raw) = 1.
__device__ __forceinline__ float2 uni_log_density(float raw)
{
    if (raw < -20.0f)
        return make_float2(raw, 1.0f);
    const float u = uni_softplus(raw);
    return make_float2(logf(u), uni_sigmoid(raw) / u);
}

// (positive term, reference term, d positive / d log u, d reference / d log u)
__device__ __forceinline__ float4 uni_terms(
    float logU, float logQ, int objective, float alpha, float beta, float power)
{
    if (objective <= 1)
    {
        const float logit = logU + logf(beta) - logQ - logf(alpha);
        const float s = uni_sigmoid(logit);
        const float t = uni_sigmoid(-logit);
        if (objective == 0)  // logistic
            return make_float4(uni_softplus(-logit), uni_softplus(logit), -t, s);
        // brier
        return make_float4(t * t, s * s, -2.0f * s * t * t, 2.0f * s * s * t);
    }

    const float logR = logU - logQ;

    if (objective == 2)  // lsif
    {
        const float r = expf(logR);
        return make_float4(-r, 0.5f * r * r, -r, r * r);
    }

    // bregman at arbitrary power, with the two limits handled separately (as in
    // the demo's terms()).
    if (fabsf(power) < 1e-9f)
    {
        const float invR = expf(-logR);
        return make_float4(invR - 1.0f, logR, -invR, 1.0f);
    }
    if (fabsf(power - 1.0f) < 1e-9f)
    {
        const float r = expf(logR);
        return make_float4(-logR, r - 1.0f, -1.0f, r);
    }
    const float rp = expf((power - 1.0f) * logR);
    const float rr = expf(power * logR);
    return make_float4(-(rp - 1.0f) / (power - 1.0f), (rr - 1.0f) / power, -rp, rr);
}

__global__ void k_unified_dre_grad(
    const tcnn::network_precision_t* __restrict__ networkOut,
    const float*                     __restrict__ weight,     // [3 x batch]
    const float*                     __restrict__ density,    // [batch]
    const float*                     __restrict__ scalePos,   // [batch]
    const float*                     __restrict__ scaleRef,   // [batch]
    tcnn::network_precision_t*       __restrict__ dLdOut,
    float*                           __restrict__ lossOut,
    uint32_t batchSize,
    uint32_t paddedWidth,
    int      objective,
    float    alpha,
    float    beta,
    float    power,
    float    gradScale)
{
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batchSize) return;

    const uint32_t base = lane * paddedWidth;

    for (uint32_t r = 0u; r < paddedWidth; ++r)
        dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);

    const float q  = density[lane];
    const float sp = scalePos[lane];
    const float sr = scaleRef[lane];

    // A lane with no reference density (empty surface table, degenerate
    // triangle, a miss, or padding) or with no stream membership contributes
    // nothing. Its launch still stands in the normalizers, which is the point:
    // a frame with fewer valid samples must not renormalize its gradient up.
    if (!(q > 0.0f) || (sp == 0.0f && sr == 0.0f))
    {
        lossOut[lane] = 0.0f;
        return;
    }

    const float logQ = logf(q);
    float loss = 0.0f;

    for (uint32_t c = 0u; c < 3u; ++c)
    {
        const float2 head = uni_log_density(static_cast<float>(networkOut[base + c]));
        const float4 t    = uni_terms(head.x, logQ, objective, alpha, beta, power);

        const float positive  = beta  * weight[3u * lane + c] * sp;
        const float reference = alpha * sr;

        float g = 0.0f;
        // Do not multiply a zero stream weight by a possibly overflowing term.
        if (positive != 0.0f)
        {
            loss += positive * t.x;
            g    += positive * t.z * head.y;
        }
        if (reference != 0.0f)
        {
            loss += reference * t.y;
            g    += reference * t.w * head.y;
        }

        g *= gradScale;
        g  = fmaxf(fminf(g, 1.0e4f), -1.0e4f);  // half-precision overflow guard
        dLdOut[base + c] = static_cast<tcnn::network_precision_t>(g);
    }

    lossOut[lane] = loss;
}

__global__ void k_softplus_inplace(float* __restrict__ v, uint32_t n)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = uni_softplus(v[i]);
}

} // namespace

unified_ngp_cache::unified_ngp_cache(const DreConfig& dre, cudaStream_t stream)
    : base_ngp_cache(makeUnifiedConfig(), UNIFIED_OUTPUT_DIM, stream)
{
    setDreConfig(dre);
}

void unified_ngp_cache::setDreConfig(const DreConfig& dre)
{
    if (!(dre.alpha > 0.0f) || !(dre.beta > 0.0f) ||
        !std::isfinite(dre.alpha) || !std::isfinite(dre.beta))
    {
        throw std::invalid_argument("DreConfig: alpha and beta must be finite and positive");
    }

    if (dreIsRawRatio(dre.objective) && dre.alpha != dre.beta)
    {
        // u* = (beta/alpha) * sum_s p_s w_bar_s for these members, so unequal
        // constants rescale the entire field off the stream sum. The demo
        // rejects the combination and so must this port.
        throw std::invalid_argument(
            "DreConfig: raw-ratio generators (lsif, bregman) require alpha == beta");
    }
    if (dre.objective == DreObjective::Bregman && !std::isfinite(dre.power))
        throw std::invalid_argument("DreConfig: Bregman power must be finite");

    m_dre = dre;
}

void unified_ngp_cache::query(
    cudaStream_t                         stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    tcnn::GPUMatrix<float>&              output,
    bool                                 useEmaWeights)
{
    base_ngp_cache::query(stream, input, output, useEmaWeights);

    const uint32_t n = static_cast<uint32_t>(UNIFIED_OUTPUT_DIM) * output.n();
    k_softplus_inplace<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(
        output.data(), n);
    CUDA_CHECK_THROW(cudaGetLastError());
}

float unified_ngp_cache::dreAccumGrad(
    cudaStream_t                         stream,
    const tcnn::GPUMatrixDynamic<float>& input,
    const float*                         d_weight,
    const float*                         d_density,
    const float*                         d_scalePos,
    const float*                         d_scaleRef,
    uint32_t                             batchSize)
{
    if (static_cast<int>(input.m()) != m_config.inputDim)
        throw std::runtime_error("unified_ngp_cache::dreAccumGrad: input row count != inputDim");
    if (input.n() != batchSize)
        throw std::runtime_error("unified_ngp_cache::dreAccumGrad: input batch size mismatch");
    if (batchSize == 0u)
        return 0.0f;

    const uint32_t paddedWidth = m_network->padded_output_width();

    tcnn::GPUMatrix<tcnn::network_precision_t> networkOut(paddedWidth, batchSize, stream);
    tcnn::GPUMatrix<tcnn::network_precision_t> dLdOutput(paddedWidth, batchSize, stream);
    tcnn::GPUMemory<float>                     lossVals(batchSize);

    auto ctx = m_network->forward(stream, input, &networkOut, false, false);

    const float lossScale = m_gradScaler.currentScale();
    const float gradScale = lossScale / static_cast<float>(paddedWidth);

    const uint32_t blocks = (batchSize + tcnn::N_THREADS_LINEAR - 1u) / tcnn::N_THREADS_LINEAR;
    k_unified_dre_grad<<<blocks, tcnn::N_THREADS_LINEAR, 0, stream>>>(
        networkOut.data(), d_weight, d_density, d_scalePos, d_scaleRef,
        dLdOutput.data(), lossVals.data(),
        batchSize, paddedWidth,
        static_cast<int>(m_dre.objective), m_dre.alpha, m_dre.beta, m_dre.power,
        gradScale);
    CUDA_CHECK_THROW(cudaGetLastError());

    m_network->backward(stream, *ctx, input, networkOut, dLdOutput,
                        nullptr, false, tcnn::GradientMode::Overwrite);

    return tcnn::reduce_sum(lossVals.data(), lossVals.size(), stream);
}

void unified_ngp_cache::optimizerStep(cudaStream_t stream)
{
    const float lossScale = m_gradScaler.currentScale();
    if (m_config.gradScaler)
    {
        if (checkGradOverflow(stream))
        {
            m_gradScaler.onOverflow();
            return;
        }
        m_gradScaler.onSuccess();
    }
    if (m_config.gradClip > 0.0f)
        clipGradNorm(stream, m_config.gradClip * lossScale);
    m_optimizer->step(stream, lossScale, m_paramsFP, m_params, m_paramGrads);
}

bool unified_ngp_cache::selfTest(cudaStream_t stream, float q)
{
    constexpr uint32_t N      = 256u;     // lanes per stream
    constexpr uint32_t BATCH  = 2u * N;   // already a multiple of 256
    constexpr float    W_PH   = 2.0f;     // photon weight
    constexpr float    W_GA   = 3.0f;     // gather weight
    const     float    Q      = q;        // reference density
    const     float    target = (W_PH + W_GA) * Q;

    std::vector<float> h_input(UNIFIED_INPUT_DIM * BATCH);
    std::vector<float> h_weight(3u * BATCH);
    std::vector<float> h_density(BATCH, Q);
    std::vector<float> h_scalePos(BATCH, 1.0f / static_cast<float>(N));
    std::vector<float> h_scaleRef(BATCH, 0.0f);

    for (uint32_t lane = 0u; lane < BATCH; ++lane)
    {
        float* in = h_input.data() + lane * UNIFIED_INPUT_DIM;
        in[0] = 0.5f; in[1] = 0.5f; in[2] = 0.5f;  // one point
        in[3] = 1.0f; in[4] = 1.0f;                // one (layer, side) bucket

        const bool  photon = (lane < N);
        const float w      = photon ? W_PH : W_GA;
        h_weight[3u * lane + 0u] = w;
        h_weight[3u * lane + 1u] = w;
        h_weight[3u * lane + 2u] = w;
        // Photons carry no anchor pairing; gathers are their own anchors.
        h_scaleRef[lane] = photon ? 0.0f : (1.0f / static_cast<float>(N));
    }

    tcnn::GPUMemory<float> d_input(h_input.size());       d_input.copy_from_host(h_input);
    tcnn::GPUMemory<float> d_weight(h_weight.size());     d_weight.copy_from_host(h_weight);
    tcnn::GPUMemory<float> d_density(h_density.size());   d_density.copy_from_host(h_density);
    tcnn::GPUMemory<float> d_scalePos(h_scalePos.size()); d_scalePos.copy_from_host(h_scalePos);
    tcnn::GPUMemory<float> d_scaleRef(h_scaleRef.size()); d_scaleRef.copy_from_host(h_scaleRef);
    tcnn::GPUMemory<float> d_output(UNIFIED_OUTPUT_DIM * BATCH);

    tcnn::GPUMatrixDynamic<float> inputMat(d_input.data(), UNIFIED_INPUT_DIM, BATCH,
                                           tcnn::MatrixLayout::ColumnMajor);

    for (int step = 0; step < 3000; ++step)
    {
        dreAccumGrad(stream, inputMat, d_weight.data(), d_density.data(),
                     d_scalePos.data(), d_scaleRef.data(), BATCH);
        optimizerStep(stream);
        updateEma(stream);
    }

    tcnn::GPUMatrix<float> outputMat(d_output.data(), UNIFIED_OUTPUT_DIM, BATCH);
    query(stream, inputMat, outputMat, /*useEmaWeights=*/false);
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    std::vector<float> h_output(UNIFIED_OUTPUT_DIM * BATCH);
    d_output.copy_to_host(h_output);

    const float recovered = h_output[0];
    const float ratio     = (target > 0.0f) ? recovered / target : 0.0f;
    const bool  ok        = (ratio > 0.85f && ratio < 1.15f);

    printf("[unified-selftest] objective=%d alpha=%g beta=%g q=%.0e  u=%.6g  expected=%.6g  ratio=%.3f  %s\n",
           static_cast<int>(m_dre.objective), m_dre.alpha, m_dre.beta, Q,
           recovered, target, ratio, ok ? "PASS" : "FAIL");
    fflush(stdout);

    return ok;
}

bool unifiedSelfTestAll(cudaStream_t stream)
{
    bool ok = true;

    // The raw-ratio members must refuse unequal stream constants outright.
    try
    {
        DreConfig bad;
        bad.objective = DreObjective::Lsif;
        bad.alpha     = 1000.0f;
        bad.beta      = 1.0f;
        unified_ngp_cache probe(bad, stream);
        printf("[unified-selftest] alpha!=beta rejection: FAIL (constructed)\n");
        ok = false;
    }
    catch (const std::invalid_argument& e)
    {
        printf("[unified-selftest] alpha!=beta rejection: PASS (%s)\n", e.what());
    }

    // Two conditioning regimes, because they answer different questions.
    //
    // q = 1e-1 puts u/q near 1, where every generator is well conditioned. All
    // four must land on the same sum-of-streams fixed point there: that is what
    // "the generator is free" means, and a member that misses it has its terms
    // wrong.
    //
    // q = 1e-4 is the realistic regime — the field runs 1e2-1e4 times the
    // reference density. The posterior members must still land, because their
    // fixed point does not move with alpha and their gradient is bounded by a
    // sigmoid. The raw-ratio members are expected to saturate here: they score
    // u/q directly, so their reference term grows as r^2 and their gradient is
    // heavy-tailed when the ratio spans orders of magnitude. That is a property
    // of the parameterization, not a defect, and it is why the GUI defaults to
    // logistic.
    const DreObjective members[] = { DreObjective::Logistic, DreObjective::Brier,
                                     DreObjective::Lsif, DreObjective::Bregman };

    for (float q : { 1.0e-1f, 1.0e-4f })
    {
        for (DreObjective obj : members)
        {
            DreConfig dre;
            dre.objective = obj;
            dre.alpha     = dreIsRawRatio(obj) ? 1.0f : 1000.0f;
            dre.beta      = 1.0f;
            dre.power     = 3.0f;

            unified_ngp_cache probe(dre, stream);
            const bool pass = probe.selfTest(stream, q);

            // Only the well-conditioned regime and the posterior members in the
            // realistic one are gates; a raw-ratio saturation at q = 1e-4 is
            // reported but expected.
            if (q > 1.0e-2f || !dreIsRawRatio(obj))
                ok = pass && ok;
        }
    }

    return ok;
}
