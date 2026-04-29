// mask_ngp_cache — thin specialization of base_ngp_cache for the BDRC mask.

#include "mask_ngp_cache.h"

#include <tiny-cuda-nn/reduce_sum.h>

namespace
{
ngp_config makeMaskConfig()
{
	ngp_config cfg;
	cfg.outputActivation = "Softplus";
	cfg.useL2Loss = true;
	cfg.ema.enabled = true;
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
// normBatch should be paddedResampledCount for BOTH eye and light passes so that
// the light term is implicitly scaled by (n_light/n_eye) as in the Python formulation.
//
// d_throughputWeights: per-sample eye-vertex throughput luminance (nullptr = uniform 1.0).
//   Applied to both eye (label=0) and light (label=1) passes to match Python's throughput weighting.
//   For padding samples p_e_area is zeroed → grad = 0 regardless of weight.
//
// networkOut : [paddedWidth × batchSize] half, column-major — row 0 = p_l_area
// p_e_area   : [batchSize] float — area-measure eye PDF per sample
// dLdOut     : [paddedWidth × batchSize] half, column-major — filled in-place
__global__ void k_dre_brier_grad(
    const tcnn::network_precision_t* __restrict__ networkOut,
    const float*                     __restrict__ p_e_area,
    const float*                     __restrict__ throughputWeights,
    tcnn::network_precision_t*       __restrict__ dLdOut,
    uint32_t batchSize,
    uint32_t paddedWidth,
    int      label,
    float    gradScale)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;

    const uint32_t base = i * paddedWidth;  // column-major: col i starts here

    // Read p_l_area (row 0, col i).  Forward already applied Softplus → always ≥ 0.
    const float p_l = (float)networkOut[base];
    const float p_e = p_e_area[i];

    // Stable denominator: add small epsilon to avoid 0/0 when both are zero.
    const float denom    = p_l + p_e + 1e-8f;
    const float denom_cu = denom * denom * denom;

    float grad;
    if (label == 0) {
        // Eye sample: L = alpha²  →  dL/d(p_l) = 2 * p_l * p_e / (p_l+p_e)³
        grad = 2.0f * p_l * p_e / denom_cu;
    } else {
        // Light sample: L = (alpha-1)²  →  dL/d(p_l) = -2 * p_e² / (p_l+p_e)³
        grad = -2.0f * p_e * p_e / denom_cu;
    }

    grad *= gradScale;
    if (throughputWeights)
        grad *= throughputWeights[i];

    // Write gradient for row 0; zero out padding rows
    dLdOut[base] = (tcnn::network_precision_t)grad;
    for (uint32_t r = 1u; r < paddedWidth; ++r)
        dLdOut[base + r] = (tcnn::network_precision_t)(0.0f);
}
} // namespace

mask_ngp_cache::mask_ngp_cache(cudaStream_t stream)
	: base_ngp_cache(makeMaskConfig(), /*outputDim=*/1, stream)
{
}

float mask_ngp_cache::trainStep(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const tcnn::GPUMatrix<float>& target)
{
	if (static_cast<int>(input.m()) != m_config.inputDim) {
		throw std::runtime_error("BaseNGPCache::trainStep: input row count != inputDim");
	}
	if (static_cast<int>(target.m()) != m_outputDim) {
		throw std::runtime_error("BaseNGPCache::trainStep: target row count != outputDim");
	}
	if (input.n() != target.n()) {
		throw std::runtime_error("BaseNGPCache::trainStep: input/target batch size mismatch");
	}

	const uint32_t batchSize = input.n();
	const uint32_t paddedWidth = m_network->padded_output_width();

	tcnn::GPUMatrix<tcnn::network_precision_t> networkOut(paddedWidth, batchSize, stream);
	tcnn::GPUMatrix<float> lossVals(paddedWidth, batchSize, stream);
	tcnn::GPUMatrix<tcnn::network_precision_t> dLdOutput(paddedWidth, batchSize, stream);

	auto ctx = m_network->forward(
		stream,
		input,
		&networkOut,
		false,
		false);
	const float lossScale = tcnn::default_loss_scale<tcnn::network_precision_t>();
	m_loss->evaluate(stream, lossScale, networkOut, target, lossVals, dLdOutput);

	m_network->backward(
		stream,
		*ctx,
		input,
		networkOut,
		dLdOutput,
		nullptr,
		false,
		tcnn::GradientMode::Overwrite);

	m_optimizer->step(stream, lossScale, m_paramsFP, m_params, m_paramGrads);

	const float totalLoss = tcnn::reduce_sum(lossVals.data(), lossVals.n_elements(), stream);
	return totalLoss / static_cast<float>(static_cast<uint32_t>(m_outputDim) * batchSize);
}

void mask_ngp_cache::dreLossAccumGrad(
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

	auto ctx = m_network->forward(stream, input, &networkOut, false, false);

	const float lossScale = tcnn::default_loss_scale<tcnn::network_precision_t>();
	const float gradScale = lossScale / (float)(paddedWidth * normBatch);
	const uint32_t blocks = (batchSize + tcnn::N_THREADS_LINEAR - 1u) / tcnn::N_THREADS_LINEAR;
	k_dre_brier_grad<<<blocks, tcnn::N_THREADS_LINEAR, 0, stream>>>(
	    networkOut.data(),
	    d_p_e_area,
	    d_throughputWeights,
	    dLdOutput.data(),
	    batchSize,
	    paddedWidth,
	    label,
	    gradScale);
	CUDA_CHECK_THROW(cudaGetLastError());

	m_network->backward(
	    stream, *ctx, input, networkOut, dLdOutput,
	    nullptr, false, gradMode);
}

void mask_ngp_cache::optimizerStep(cudaStream_t stream)
{
	const float lossScale = tcnn::default_loss_scale<tcnn::network_precision_t>();
	m_optimizer->step(stream, lossScale, m_paramsFP, m_params, m_paramGrads);
}

float mask_ngp_cache::dreLossTrainStep(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const float* d_p_e_area,
	int label)
{
	dreLossAccumGrad(stream, input, d_p_e_area, label,
	                 input.n(), nullptr, tcnn::GradientMode::Overwrite);
	optimizerStep(stream);
	return 0.0f;
}