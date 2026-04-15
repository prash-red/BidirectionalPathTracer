/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Nrc.cu — implementation of NRC network + training/inference helpers.
 */

#include "Nrc.h"

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/random.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

// -----------------------------------------------------------------------------
// CUDA helpers
// -----------------------------------------------------------------------------

__global__ void k_fp32_to_half(const float* __restrict__ src, NrcNetT* __restrict__ dst, uint32_t n) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		dst[i] = static_cast<NrcNetT>(src[i]);
	}
}

__global__ void k_cast_half_to_float_strided(
	const NrcNetT* __restrict__ src,
	float* __restrict__ dst,
	uint32_t outputDim,
	uint32_t batchSize,
	uint32_t stride) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= outputDim * batchSize) {
		return;
	}

	const uint32_t col = i / outputDim;
	const uint32_t row = i % outputDim;
	dst[i] = static_cast<float>(src[col * stride + row]);
}

__global__ void k_ema_update(
	const NrcNetT* __restrict__ current,
	NrcNetT* __restrict__ shadow,
	uint32_t n,
	float coeffOld,
	float coeffNew) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	const auto cur = static_cast<float>(current[i]);
	const auto old = static_cast<float>(shadow[i]);
	shadow[i] = static_cast<NrcNetT>(coeffOld * old + coeffNew * cur);
}

__global__ void k_normalize3d(float* __restrict__ positions, uint32_t n, float3 aabbMin, float3 aabbMax) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	const float3 ext = {
		aabbMax.x - aabbMin.x,
		aabbMax.y - aabbMin.y,
		aabbMax.z - aabbMin.z,
	};

	float* p = positions + i * 3;
	p[0] = fminf(fmaxf((p[0] - aabbMin.x) / (ext.x > 0.f ? ext.x : 1.f), 0.f), 1.f);
	p[1] = fminf(fmaxf((p[1] - aabbMin.y) / (ext.y > 0.f ? ext.y : 1.f), 0.f), 1.f);
	p[2] = fminf(fmaxf((p[2] - aabbMin.z) / (ext.z > 0.f ? ext.z : 1.f), 0.f), 1.f);
}

__global__ void k_normalize2d(float* __restrict__ uvs, uint32_t n, float invWidth, float invHeight) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	float* uv = uvs + i * 2;
	uv[0] = fminf(fmaxf(uv[0] * invWidth, 0.f), 1.f);
	uv[1] = fminf(fmaxf(uv[1] * invHeight, 0.f), 1.f);
}

__device__ inline uint32_t pcg_step(uint32_t& s) {
	const uint32_t old = s;
	s = old * 747796405u + 2891336453u;
	const uint32_t w = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
	return (w >> 22u) ^ w;
}

__device__ inline float pcg_float(uint32_t& s) {
	return static_cast<float>(pcg_step(s) >> 8) * (1.f / static_cast<float>(1u << 24));
}

__device__ inline float box_muller(uint32_t& s) {
	const float u1 = fmaxf(fminf(pcg_float(s), 1.f - 1e-6f), 1e-6f);
	const float u2 = pcg_float(s);
	return sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265358979f * u2);
}

__global__ void k_spn_noise3d(float* __restrict__ p, uint32_t n, float alpha, uint32_t seed) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	uint32_t s = seed ^ (i * 2654435761u);
	float* v = p + i * 3;
	v[0] = fminf(fmaxf(v[0] + alpha * box_muller(s), 0.f), 1.f);
	v[1] = fminf(fmaxf(v[1] + alpha * box_muller(s), 0.f), 1.f);
	v[2] = fminf(fmaxf(v[2] + alpha * box_muller(s), 0.f), 1.f);
}

__global__ void k_spn_noise2d(float* __restrict__ uv, uint32_t n, float alpha, uint32_t seed) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	uint32_t s = seed ^ (i * 2654435761u + 0xDEADBEEFu);
	float* v = uv + i * 2;
	v[0] = fminf(fmaxf(v[0] + alpha * box_muller(s), 0.f), 1.f);
	v[1] = fminf(fmaxf(v[1] + alpha * box_muller(s), 0.f), 1.f);
}

// -----------------------------------------------------------------------------
// NrcProjection launchers
// -----------------------------------------------------------------------------

namespace nrc {

void nrcNormalize3dBatch(float* d_pos, uint32_t n, float3 aabbMin, float3 aabbMax, cudaStream_t stream) {
	if (!n) {
		return;
	}
	k_normalize3d<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(d_pos, n, aabbMin, aabbMax);
	CUDA_CHECK_THROW(cudaGetLastError());
}

void nrcNormalize2dBatch(float* d_uvs, uint32_t n, int width, int height, cudaStream_t stream) {
	if (!n) {
		return;
	}
	const float iw = width > 0 ? 1.f / static_cast<float>(width) : 1.f;
	const float ih = height > 0 ? 1.f / static_cast<float>(height) : 1.f;
	k_normalize2d<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(d_uvs, n, iw, ih);
	CUDA_CHECK_THROW(cudaGetLastError());
}

void nrcApplySpnNoise3d(float* d_pos, uint32_t n, float alpha, uint32_t seed, cudaStream_t stream) {
	if (!n || alpha == 0.f) {
		return;
	}
	k_spn_noise3d<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(d_pos, n, alpha, seed);
	CUDA_CHECK_THROW(cudaGetLastError());
}

void nrcApplySpnNoise2d(float* d_uvs, uint32_t n, float alpha, uint32_t seed, cudaStream_t stream) {
	if (!n || alpha == 0.f) {
		return;
	}
	k_spn_noise2d<<<tcnn::n_blocks_linear(n), tcnn::N_THREADS_LINEAR, 0, stream>>>(d_uvs, n, alpha, seed);
	CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace nrc

// -----------------------------------------------------------------------------
// Nrc implementation
// -----------------------------------------------------------------------------

Nrc::Nrc(const NrcConfig& config, int outputDim, cudaStream_t /*stream*/)
	: m_config(config), m_outputDim(outputDim) {
	if (outputDim <= 0) {
		throw std::invalid_argument("Nrc: outputDim must be > 0");
	}
	if (config.inputDim <= 0) {
		throw std::invalid_argument("Nrc: config.inputDim must be > 0");
	}

	m_network = std::make_shared<tcnn::NetworkWithInputEncoding<NrcNetT>>(
		static_cast<uint32_t>(config.inputDim),
		static_cast<uint32_t>(outputDim),
		config.toEncodingJson(),
		config.toNetworkJson());

	const size_t n = m_network->n_params();
	m_paramsBuf.resize(sizeof(float) * n + sizeof(NrcNetT) * n * 2);
	m_paramsBuf.memset(0);

	m_paramsFP = reinterpret_cast<float*>(m_paramsBuf.data());
	m_params = reinterpret_cast<NrcNetT*>(m_paramsBuf.data() + sizeof(float) * n);
	m_paramGrads = m_params + n;

	tcnn::pcg32 rng{1337};
	m_network->initialize_params(rng, m_paramsFP);

	k_fp32_to_half<<<tcnn::n_blocks_linear(static_cast<uint32_t>(n)), tcnn::N_THREADS_LINEAR>>>(
		m_paramsFP, m_params, static_cast<uint32_t>(n));
	CUDA_CHECK_THROW(cudaGetLastError());
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	m_network->set_params(m_params, m_params, m_paramGrads);

	m_optimizer.reset(tcnn::create_optimizer<NrcNetT>(config.toOptimizerJson()));
	m_optimizer->allocate(m_network);

	m_loss.reset(tcnn::create_loss<NrcNetT>(config.toLossJson()));

	if (config.ema.enabled) {
		initEma(config.ema.decay, config.ema.startIter);
	}
}

void Nrc::query(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	tcnn::GPUMatrix<float>& output,
	bool useEmaWeights) {
	const uint32_t batchSize = input.n();
	const uint32_t paddedWidth = m_network->padded_output_width();

	if (static_cast<int>(input.m()) != m_config.inputDim) {
		throw std::runtime_error("Nrc::query: input row count != inputDim");
	}
	if (static_cast<int>(output.m()) != m_outputDim || output.n() != batchSize) {
		throw std::runtime_error("Nrc::query: output shape mismatch");
	}

	tcnn::GPUMatrixDynamic<NrcNetT> rawOutput(paddedWidth, batchSize, stream);
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

float Nrc::trainStep(
	cudaStream_t stream,
	const tcnn::GPUMatrixDynamic<float>& input,
	const tcnn::GPUMatrix<float>& target) {
	if (static_cast<int>(input.m()) != m_config.inputDim) {
		throw std::runtime_error("Nrc::trainStep: input row count != inputDim");
	}
	if (static_cast<int>(target.m()) != m_outputDim) {
		throw std::runtime_error("Nrc::trainStep: target row count != outputDim");
	}
	if (input.n() != target.n()) {
		throw std::runtime_error("Nrc::trainStep: input/target batch size mismatch");
	}

	const uint32_t batchSize = input.n();
	const uint32_t paddedWidth = m_network->padded_output_width();

	tcnn::GPUMatrix<NrcNetT> networkOut(paddedWidth, batchSize, stream);
	tcnn::GPUMatrix<float> lossVals(paddedWidth, batchSize, stream);
	tcnn::GPUMatrix<NrcNetT> dLdOutput(paddedWidth, batchSize, stream);

	auto ctx = m_network->forward(
		stream,
		input,
		&networkOut,
		false,
		false);

	const float lossScale = tcnn::default_loss_scale<NrcNetT>();
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

void Nrc::setLearningRate(float factor) {
	m_optimizer->set_learning_rate(m_config.lr * factor);
}

void Nrc::initEma(float decay, int startIter) {
	if (!(decay >= 0.0f && decay < 1.0f)) {
		throw std::invalid_argument("Nrc::initEma: decay must be in [0, 1)");
	}
	if (startIter < 0) {
		throw std::invalid_argument("Nrc::initEma: startIter must be >= 0");
	}

	m_emaDecay = decay;
	m_emaStartIter = startIter;
	m_emaUpdateCount = 0;
	m_emaActive = true;
	m_emaInitialized = false;

	const size_t n = m_network->n_params();
	m_emaParamsBuf.resize(n);
}

void Nrc::updateEma(cudaStream_t stream) {
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
			sizeof(NrcNetT) * n,
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

void Nrc::swapEmaIn(cudaStream_t stream) {
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
		sizeof(NrcNetT) * n,
		cudaMemcpyDeviceToDevice,
		stream));

	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_params,
		m_emaParamsBuf.data(),
		sizeof(NrcNetT) * n,
		cudaMemcpyDeviceToDevice,
		stream));
}

void Nrc::swapEmaOut(cudaStream_t stream) {
	if (!m_emaActive || !m_emaInitialized || m_savedParamsBuf.size() == 0) {
		return;
	}

	const size_t n = m_network->n_params();
	CUDA_CHECK_THROW(cudaMemcpyAsync(
		m_params,
		m_savedParamsBuf.data(),
		sizeof(NrcNetT) * n,
		cudaMemcpyDeviceToDevice,
		stream));
}

std::string Nrc::serialize(const std::string& name) const {
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

