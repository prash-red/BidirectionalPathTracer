// mask_ngp_cache — thin specialization of base_ngp_cache for the BDRC mask.

#include "mask_ngp_cache.h"

#include <tiny-cuda-nn/reduce_sum.h>

namespace
{
ngp_config makeMaskConfig()
{
	ngp_config cfg;
	cfg.outputActivation = "Softplus";
	cfg.ema.enabled = true;
	cfg.lr = 1e-5;
	cfg.gradClip = 0.05f;
	return cfg;
}

// Logistic-NCE gradient kernel — same physical fixed point as the Brier form
// it replaces (u* = p_l_area), but matches the current Python production
// density_loss="logistic": -log(sigma) for positives, -log(1-sigma) for
// references, where sigma = p_l / (p_l + p_e). Logistic damps the gradient
// less aggressively near the saturated caustic peak than Brier does, which
// the Python project measured as a +0.1-0.33 dB gain.
//
// label=0 (reference/eye anchor): loss = -log(p_e / (p_l + p_e))
//   dL/d(p_l) = 1 / (p_l + p_e)
// label=1 (light/positive):       loss = -log(p_l / (p_l + p_e))
//   dL/d(p_l) = -p_e / (p_l * (p_l + p_e))
//
// gradScale = lossScale / (paddedWidth * normBatch), same convention as before.
__global__ void k_dre_logistic_grad(
    const tcnn::network_precision_t* __restrict__ networkOut,
    const float*                     __restrict__ p_e_area,
    const float*                     __restrict__ throughputWeights,
    tcnn::network_precision_t*       __restrict__ dLdOut,
    float*                           __restrict__ lossOut,
    uint32_t batchSize,
    uint32_t paddedWidth,
    int      label,
    float    gradScale)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;

    const uint32_t base = i * paddedWidth;

    const float p_l = networkOut[base];
    const float p_e = p_e_area[i];
    const float tw  = throughputWeights ? throughputWeights[i] : 1.0f;

    if (p_e < 1e-10f) {
        for (uint32_t r = 0u; r < paddedWidth; ++r)
            dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);
        lossOut[i] = 0.0f;
        return;
    }

    const float p_l_safe = fmaxf(p_l, 1e-20f);
    const float denom    = fmaxf(p_l + p_e, 1e-10f);

    float grad, lv;
    if (label == 0) {
        grad = 1.0f / denom;
        lv   = -logf(p_e / denom);
    } else {
        grad = -p_e / (p_l_safe * denom);
        lv   = -logf(p_l_safe / denom);
    }

    grad *= gradScale * tw;

    lossOut[i] = tw * lv;

    dLdOut[base] = static_cast<tcnn::network_precision_t>(grad);
    for (uint32_t r = 1u; r < paddedWidth; ++r)
        dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);
}
} // namespace

mask_ngp_cache::mask_ngp_cache(cudaStream_t stream)
	: base_ngp_cache(makeMaskConfig(), /*outputDim=*/1, stream)
{
}

float mask_ngp_cache::dreLossAccumGrad(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const float* d_p_e_area,
	int label,
	uint32_t normBatch,
	const float* d_throughputWeights,
	tcnn::GradientMode gradMode)
{
	if (static_cast<int>(input.m()) != m_config.inputDim)
		throw std::runtime_error("mask_ngp_cache::dreLossAccumGrad: input row count != inputDim");

	const uint32_t batchSize   = input.n();
	const uint32_t paddedWidth = m_network->padded_output_width();

	tcnn::GPUMatrix<tcnn::network_precision_t> networkOut(paddedWidth, batchSize, stream);
	tcnn::GPUMatrix<tcnn::network_precision_t> dLdOutput(paddedWidth, batchSize, stream);

	float* d_lossOut = nullptr;
	CUDA_CHECK_THROW(cudaMalloc(&d_lossOut, batchSize * sizeof(float)));

	auto ctx = m_network->forward(stream, input, &networkOut, false, false);

	const float lossScale = m_gradScaler.currentScale();
	const float gradScale = lossScale / (float)(paddedWidth * normBatch);
	const uint32_t blocks = (batchSize + tcnn::N_THREADS_LINEAR - 1u) / tcnn::N_THREADS_LINEAR;
	k_dre_logistic_grad<<<blocks, tcnn::N_THREADS_LINEAR, 0, stream>>>(
	    networkOut.data(),
	    d_p_e_area,
	    d_throughputWeights,
	    dLdOutput.data(),
	    d_lossOut,
	    batchSize,
	    paddedWidth,
	    label,
	    gradScale);
	CUDA_CHECK_THROW(cudaGetLastError());

	m_network->backward(
	    stream, *ctx, input, networkOut, dLdOutput,
	    nullptr, false, gradMode);

	// Sync and sum per-sample loss values for display
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	std::vector<float> h_loss(batchSize);
	CUDA_CHECK_THROW(cudaMemcpy(h_loss.data(), d_lossOut, batchSize * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(d_lossOut);

	float lossSum = 0.0f;
	for (uint32_t i = 0u; i < batchSize; ++i)
		lossSum += h_loss[i];

	return lossSum;
}

void mask_ngp_cache::optimizerStep(cudaStream_t stream)
{
	const float lossScale = m_gradScaler.currentScale();
	if (m_config.gradScaler) {
		if (checkGradOverflow(stream)) {
			m_gradScaler.onOverflow();
			return;
		}
		m_gradScaler.onSuccess();
	}
	if (m_config.gradClip > 0.0f)
		clipGradNorm(stream, m_config.gradClip * lossScale);
	m_optimizer->step(stream, lossScale, m_paramsFP, m_params, m_paramGrads);
}

float mask_ngp_cache::dreLossTrainStep(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const float* d_p_e_area,
	int label)
{
	const float lossSum = dreLossAccumGrad(stream, input, d_p_e_area, label,
	                                       input.n(), nullptr, tcnn::GradientMode::Overwrite);
	optimizerStep(stream);
	return lossSum;
}