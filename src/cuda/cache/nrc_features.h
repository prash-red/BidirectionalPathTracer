/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * nrc_features.h — Device-side NRC feature packing.
 *
 * Packs 14 surface attributes into the NRC input buffer (column-major,
 * rows = NRC_INPUT_DIM, cols = numPixels) following Mueller 2021 §4:
 *
 *   feat[0..2]   position, normalized to scene AABB     → TriangleWave(3, 12)
 *   feat[3..7]   normal_theta, normal_phi,              → OneBlob(5, 4)
 *                view_theta, view_phi, roughness
 *   feat[8..13]  baseColor.rgb, specularColor.rgb       → Identity(6)
 *
 * Layout: element (feature f, pixel p) = buf[p * NRC_INPUT_DIM + f]
 * This is column-major for a [NRC_INPUT_DIM × numPixels] tcnn matrix.
 *
 * Requires: vector_math.h included before this header (for normalize(), float3).
 */

#pragma once

#ifndef NRC_FEATURES_H
#define NRC_FEATURES_H

#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

#define NRC_INPUT_DIM   14
#define NRC_OUTPUT_DIM   3
#define BDRC_INPUT_DIM   3   // BDRC caches take position only (x,y,z normalized)

__device__ __forceinline__ void nrcPackFeatures(
    float*       buf,
    uint32_t     pixelIdx,
    const float3 pos,
    const float3 N,
    const float3 wo,
    float        roughness,
    const float3 baseColor,
    const float3 specularColor,
    const float3 aabbMin,
    const float3 aabbMax)
{
    float* d = buf + pixelIdx * NRC_INPUT_DIM;

    // [0,1,2] world-space position normalized to scene AABB
    const float rx = aabbMax.x - aabbMin.x;
    const float ry = aabbMax.y - aabbMin.y;
    const float rz = aabbMax.z - aabbMin.z;
    d[0] = fminf(fmaxf((pos.x - aabbMin.x) / (rx > 0.f ? rx : 1.f), 0.f), 1.f);
    d[1] = fminf(fmaxf((pos.y - aabbMin.y) / (ry > 0.f ? ry : 1.f), 0.f), 1.f);
    d[2] = fminf(fmaxf((pos.z - aabbMin.z) / (rz > 0.f ? rz : 1.f), 0.f), 1.f);

    // [3,4] shading normal as spherical angles in [0,1]
    const float3 n = normalize(N);
    d[3] = acosf(fminf(fmaxf(n.z, -1.f), 1.f)) * (float)M_1_PI;
    d[4] = (atan2f(n.y, n.x) * (float)M_1_PI + 1.f) * 0.5f;

    // [5,6] outgoing view direction (wo) as spherical angles in [0,1]
    const float3 v = normalize(wo);
    d[5] = acosf(fminf(fmaxf(v.z, -1.f), 1.f)) * (float)M_1_PI;
    d[6] = (atan2f(v.y, v.x) * (float)M_1_PI + 1.f) * 0.5f;

    // [7] linear roughness
    d[7] = roughness;

    // [8,9,10] diffuse base color
    d[8]  = baseColor.x;
    d[9]  = baseColor.y;
    d[10] = baseColor.z;

    // [11,12,13] specular color (F0 proxy)
    d[11] = specularColor.x;
    d[12] = specularColor.y;
    d[13] = specularColor.z;
}

// Packs one 3-dim BDRC position query: world position normalized into the
// SCENE AABB, clamped to [0,1].
//
// World space, not GAS-local: one neural field spans the whole scene, so two
// meshes must not share a coordinate. Normalizing each mesh by its own local
// AABB (the pre-2026-09 convention) collapsed every mesh onto the same unit
// cube and made two instances of one mesh indistinguishable to the field.
//
// Layout: element (feature f, lane i) at buf[i * BDRC_INPUT_DIM + f], i.e. a
// column-major [BDRC_INPUT_DIM x N] tcnn matrix.
__device__ __forceinline__ void bdrcPackFeatures(
    float*       buf,
    uint32_t     lane,
    const float3 worldPos,
    const float3 aabbMin,
    const float3 aabbMax)
{
    float* d = buf + lane * BDRC_INPUT_DIM;

    // A planar scene has zero extent on one axis; fall back to 1.0 so the
    // coordinate pins to 0 instead of producing inf/NaN that would poison
    // every hash-grid cell the lane touches.
    const float rx = aabbMax.x - aabbMin.x;
    const float ry = aabbMax.y - aabbMin.y;
    const float rz = aabbMax.z - aabbMin.z;
    d[0] = fminf(fmaxf((worldPos.x - aabbMin.x) / (rx > 0.f ? rx : 1.f), 0.f), 1.f);
    d[1] = fminf(fmaxf((worldPos.y - aabbMin.y) / (ry > 0.f ? ry : 1.f), 0.f), 1.f);
    d[2] = fminf(fmaxf((worldPos.z - aabbMin.z) / (rz > 0.f ? rz : 1.f), 0.f), 1.f);
}

#endif // NRC_FEATURES_H