#ifndef GLTF_RENDERER_CACHEHELPERS_H
#define GLTF_RENDERER_CACHEHELPERS_H

#include <tiny-cuda-nn/network_with_input_encoding.h>

namespace tcnn {
	template <typename T>
	class NetworkWithInputEncoding;
}

// -----------------------------------------------------------------------------
// CUDA helpers — declarations only; definitions live in CacheHelpers.cu
// -----------------------------------------------------------------------------

__global__ void k_fp32_to_half(const float* __restrict__ src, tcnn::network_precision_t* __restrict__ dst, uint32_t n);

__global__ void k_cast_half_to_float_strided(
	const tcnn::network_precision_t* __restrict__ src,
	float* __restrict__ dst,
	uint32_t outputDim,
	uint32_t batchSize,
	uint32_t stride);

__global__ void k_ema_update(
	const tcnn::network_precision_t* __restrict__ current,
	tcnn::network_precision_t* __restrict__ shadow,
	uint32_t n,
	float coeffOld,
	float coeffNew);

__global__ void k_normalize3d(float* __restrict__ positions, uint32_t n, float3 aabbMin, float3 aabbMax);

__global__ void k_normalize2d(float* __restrict__ uvs, uint32_t n, float invWidth, float invHeight);

// -----------------------------------------------------------------------------
// Device helpers — inline, safe to include in multiple TUs
// -----------------------------------------------------------------------------

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

__global__ void k_spn_noise3d(float* __restrict__ p, uint32_t n, float alpha, uint32_t seed);

__global__ void k_spn_noise2d(float* __restrict__ uv, uint32_t n, float alpha, uint32_t seed);

#endif //GLTF_RENDERER_CACHEHELPERS_H