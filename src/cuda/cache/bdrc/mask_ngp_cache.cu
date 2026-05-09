// mask_ngp_cache — thin specialization of base_ngp_cache for the BDRC mask.

#include "mask_ngp_cache.h"

#include <tiny-cuda-nn/reduce_sum.h>

namespace
{
ngp_config makeMaskConfig()
{
	ngp_config cfg;
	cfg.outputActivation = "Softplus";
	cfg.ema.enabled = false;
	cfg.lr = 5e-5f;
	cfg.gradClip = 5.0f;
	return cfg;
}

// Brier-score DRE gradient kernel.
//
// For each sample i, the network outputs p_l_area (post-Softplus, in column i of
// networkOut).  Given the per-sample p_e_area and the sample label, the analytic
// gradient of the Brier loss w.r.t. p_l_area is:
//
//   label=0 (eye):    dL/d(p_l) =  2 * p_l * p_e / (p_l + p_e)³
//   label=1 (light):  dL/d(p_l) = -2 * p_e²     / (p_l + p_e)³
//
// gradScale = lossScale / (paddedWidth * normBatch) is pre-computed by the caller.
// Both eye and light passes use normBatch = n_eye so that the composite loss
//   total = (sum_eye_terms + sum_light_terms) / n_eye
//
// d_throughputWeights: per-sample eye-vertex throughput luminance (nullptr = uniform 1.0).
//   Applied to both eye and light passes to match Python's throughput weighting.
//
// lossOut: [batchSize] per-sample throughput-weighted Brier loss value
//   (tw * alpha^2 for eye, tw * (alpha-1)^2 for light).
//   Padding samples (p_e_area == 0) contribute 0.
//
// networkOut : [paddedWidth × batchSize] half, column-major — row 0 = p_l_area
// p_e_area   : [batchSize] float — area-measure eye PDF per sample
// dLdOut     : [paddedWidth × batchSize] half, column-major — filled in-place
__global__ void k_dre_brier_grad(
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

    const uint32_t base = i * paddedWidth;  // column-major: col i starts here

    const float p_l = networkOut[base];
    const float p_e = p_e_area[i];
    const float tw  = throughputWeights ? throughputWeights[i] : 1.0f;

    // Padding guard: p_e == 0 means this slot was zero-padded; contribute nothing.
    if (p_e < 1e-10f) {
        for (uint32_t r = 0u; r < paddedWidth; ++r)
            dLdOut[base + r] = static_cast<tcnn::network_precision_t>(0.0f);
        lossOut[i] = 0.0f;
        return;
    }

    const float denom    = p_l + p_e + 1e-8f;
    const float denom_cu = denom * denom * denom;

    float grad;
    if (label == 0) {
        grad = 2.0f * p_l * p_e / denom_cu;
    } else {
        grad = -2.0f * p_e * p_e / denom_cu;
    }

    grad *= gradScale * tw;

    // Per-sample throughput-weighted Brier loss
    const float alpha = p_l / denom;
    const float lv    = (label == 0) ? (alpha * alpha) : ((alpha - 1.0f) * (alpha - 1.0f));
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
	k_dre_brier_grad<<<blocks, tcnn::N_THREADS_LINEAR, 0, stream>>>(
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