#include "base_ngp_cache.h"

#include "../cache_helpers.h"

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/random.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

base_ngp_cache::base_ngp_cache(const ngp_config& config, int outputDim, cudaStream_t /*stream*/)
	: m_config(config), m_outputDim(outputDim) {
	if (outputDim <= 0) {
		throw std::invalid_argument("BaseNGPCache: outputDim must be > 0");
	}
	if (config.inputDim <= 0) {
		throw std::invalid_argument("BaseNGPCache: config.inputDim must be > 0");
	}

	m_network = std::make_shared<tcnn::NetworkWithInputEncoding<tcnn::network_precision_t>>(
		static_cast<uint32_t>(config.inputDim),
		static_cast<uint32_t>(outputDim),
		config.toEncodingJson(),
		config.toNetworkJson());

	const size_t n = m_network->n_params();
	m_paramsBuf.resize(sizeof(float) * n + sizeof(tcnn::network_precision_t) * n * 2);
	m_paramsBuf.memset(0);

	m_paramsFP = reinterpret_cast<float*>(m_paramsBuf.data());
	m_params = reinterpret_cast<tcnn::network_precision_t*>(m_paramsBuf.data() + sizeof(float) * n);
	m_paramGrads = m_params + n;

	tcnn::pcg32 rng{1337};
	m_network->initialize_params(rng, m_paramsFP);

	k_fp32_to_half<<<tcnn::n_blocks_linear(static_cast<uint32_t>(n)), tcnn::N_THREADS_LINEAR>>>(
		m_paramsFP, m_params, static_cast<uint32_t>(n));
	CUDA_CHECK_THROW(cudaGetLastError());
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	m_network->set_params(m_params, m_params, m_paramGrads);

	m_optimizer.reset(tcnn::create_optimizer<tcnn::network_precision_t>(config.toOptimizerJson()));
	m_optimizer->allocate(m_network);

	m_loss.reset(tcnn::create_loss<tcnn::network_precision_t>(config.toLossJson()));

	if (config.ema.enabled) {
		initEma(config.ema.decay, config.ema.startIter);
	}
}

void base_ngp_cache::query(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	tcnn::GPUMatrix<float>& output,
	bool useEmaWeights) {
	const uint32_t batchSize = input.n();
	const uint32_t paddedWidth = m_network->padded_output_width();

	if (static_cast<int>(input.m()) != m_config.inputDim) {
		throw std::runtime_error("BaseNGPCache::query: input row count != inputDim");
	}
	if (static_cast<int>(output.m()) != m_outputDim || output.n() != batchSize) {
		throw std::runtime_error("BaseNGPCache::query: output shape mismatch");
	}

	tcnn::GPUMatrixDynamic<tcnn::network_precision_t> rawOutput(paddedWidth, batchSize, stream);
	m_network->inference_mixed_precision(stream, input, rawOutput, useEmaWeights);

	const uint32_t n = static_cast<uint32_t>(m_outputDim) * batchSize;
	k_cast_half_to_float_strided<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(
		rawOutput.data(),
		output.data(),
		static_cast<uint32_t>(m_outputDim),
		batchSize,
		paddedWidth);
	CUDA_CHECK_THROW(cudaGetLastError());
}

float base_ngp_cache::trainStep(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const tcnn::GPUMatrix<float>& target) {
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

void base_ngp_cache::setLearningRate(float factor) {
	m_optimizer->set_learning_rate(m_config.lr * factor);
}

void base_ngp_cache::initEma(float decay, int startIter) {
	if (!(decay >= 0.0f && decay < 1.0f)) {
		throw std::invalid_argument("BaseNGPCache::initEma: decay must be in [0, 1)");
	}
	if (startIter < 0) {
		throw std::invalid_argument("BaseNGPCache::initEma: startIter must be >= 0");
	}

	m_emaDecay = decay;
	m_emaStartIter = startIter;
	m_emaUpdateCount = 0;
	m_emaActive = true;
	m_emaInitialized = false;

	const size_t n = m_network->n_params();
	m_emaParamsBuf.resize(n);
}

void base_ngp_cache::updateEma(cudaStream_t stream) {
	if (!m_emaActive) {
		return;
	}

	++m_emaUpdateCount;
	if (m_emaUpdateCount <= m_emaStartIter) {
		return;
	}

	const size_t n = m_network->n_params();
	const int localStep = m_emaUpdateCount - m_emaStartIter;
	if (!m_emaInitialized) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			m_emaParamsBuf.data(),
			m_params,
			sizeof(tcnn::network_precision_t) * n,
			cudaMemcpyDeviceToDevice,
			stream));
		m_emaInitialized = true;
		return;
	}

	if (localStep <= 1) {
		return;
	}

	// ADS-style bias-corrected EMA:
	//   eta_t = 1 - a^t
	//   Wbar_t = a * eta_{t-1}/eta_t * Wbar_{t-1} + (1-a)/eta_t * W_t
	const float a = m_emaDecay;
	const float eta_t = 1.0f - std::pow(a, static_cast<float>(localStep));
	const float eta_prev = 1.0f - std::pow(a, static_cast<float>(localStep - 1));

	if (eta_t <= 0.0f) {
		return;
	}

	const float coeffOld = a * (eta_prev / eta_t);
	const float coeffNew = (1.0f - a) / eta_t;

	k_ema_update<<<tcnn::n_blocks_linear(static_cast<uint32_t>(n)), tcnn::N_THREADS_LINEAR, 0, stream>>>(
		m_params,
		m_emaParamsBuf.data(),
		static_cast<uint32_t>(n),
		coeffOld,
		coeffNew);
	CUDA_CHECK_THROW(cudaGetLastError());
}

void base_ngp_cache::swapEmaIn(cudaStream_t stream) {
	if (!m_emaActive || !m_emaInitialized) {
		return;
	}

	const size_t n = m_network->n_params();
	if (m_savedParamsBuf.size() != n) {
		m_savedParamsBuf.resize(n);
	}

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_savedParamsBuf.data(),
		m_params,
		sizeof(tcnn::network_precision_t) * n,
		cudaMemcpyDeviceToDevice,
		stream));

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_params,
		m_emaParamsBuf.data(),
		sizeof(tcnn::network_precision_t) * n,
		cudaMemcpyDeviceToDevice,
		stream));
}

void base_ngp_cache::swapEmaOut(cudaStream_t stream) {
	if (!m_emaActive || !m_emaInitialized || m_savedParamsBuf.size() == 0) {
		return;
	}

	const size_t n = m_network->n_params();
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_params,
		m_savedParamsBuf.data(),
		sizeof(tcnn::network_precision_t) * n,
		cudaMemcpyDeviceToDevice,
		stream));
}

std::string base_ngp_cache::serialize(const std::string& name) const {
	char buf[256];
	std::snprintf(
		buf,
		sizeof(buf),
		"%s: %zu params | encoding output width: %u | outputDim: %d%s",
		name.c_str(),
		numParams(),
		m_network ? m_network->num_encoded_dims() : 0u,
		m_outputDim,
		m_emaActive ? " [EMA active]" : "");
	return {buf};
}