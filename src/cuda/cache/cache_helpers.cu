#include "cache_helpers.h"

__global__ void k_fp32_to_half(const float* __restrict__ src, tcnn::network_precision_t* __restrict__ dst, uint32_t n) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		dst[i] = static_cast<tcnn::network_precision_t>(src[i]);
	}
}

__global__ void k_cast_half_to_float_strided(
	const tcnn::network_precision_t* __restrict__ src,
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
	const tcnn::network_precision_t* __restrict__ current,
	tcnn::network_precision_t* __restrict__ shadow,
	uint32_t n,
	float coeffOld,
	float coeffNew) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) {
		return;
	}

	const auto cur = static_cast<float>(current[i]);
	const auto old = static_cast<float>(shadow[i]);
	shadow[i] = static_cast<tcnn::network_precision_t>(coeffOld * old + coeffNew * cur);
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