//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// unified_features.h — device-side query packing for the unified field.
//
// One query is 5 floats: [0..2] world position normalized into the scene AABB
// (hash grid), [3] sign_cam, [4] arrival side (both Identity). Element
// (feature f, lane i) lives at buf[i * UNIFIED_QUERY_DIM + f], i.e. a
// column-major [5 x N] tcnn matrix.
//
#pragma once
#ifndef UNIFIED_FEATURES_H
#define UNIFIED_FEATURES_H

#include <cuda_runtime.h>

// Mirrors UNIFIED_INPUT_DIM in unified_ngp_cache.h. Kept as its own define so
// the OptiX device modules do not have to include a tiny-cuda-nn header.
#define UNIFIED_QUERY_DIM 5

// Face orientation against a FIXED reference origin.
//
// Never the live camera: this renderer's camera is interactive, and a label
// taken against it would flip for every point the user orbits past, silently
// relabelling the whole learned function mid-fit. The demo avoids this by
// construction because its authored camera owns the mapping while the viewing
// camera only changes rays; this port has no authored camera, so the fixed
// origin is explicit.
__device__ __forceinline__ float bdrcSignCamLabel(
    const float3 ng, const float3 worldPos, const float3 referenceOrigin)
{
    const float3 toOrigin = make_float3(referenceOrigin.x - worldPos.x,
                                        referenceOrigin.y - worldPos.y,
                                        referenceOrigin.z - worldPos.z);
    const float  d = ng.x * toOrigin.x + ng.y * toOrigin.y + ng.z * toOrigin.z;
    return (d >= 0.0f) ? 1.0f : -1.0f;
}

// Which of the two sides of the field this sample belongs to. For a ray this is
// the side it arrived on; for an anchor it is the coin flip. The field holds a
// value per (point, side), and the interior of a closed solid is a side of its
// own that fits to zero.
__device__ __forceinline__ float bdrcSideLabel(const bool frontFace)
{
    return frontFace ? 1.0f : -1.0f;
}

__device__ __forceinline__ void bdrcPackQuery5(
    float*       buf,
    unsigned int lane,
    const float3 worldPos,
    const float3 ng,
    const float  side,
    const float3 referenceOrigin,
    const float3 aabbMin,
    const float3 aabbMax)
{
    float* d = buf + lane * UNIFIED_QUERY_DIM;

    // A planar scene has zero extent on one axis; fall back to 1.0 so the
    // coordinate pins to 0 rather than producing inf/NaN.
    const float rx = aabbMax.x - aabbMin.x;
    const float ry = aabbMax.y - aabbMin.y;
    const float rz = aabbMax.z - aabbMin.z;
    d[0] = fminf(fmaxf((worldPos.x - aabbMin.x) / (rx > 0.f ? rx : 1.f), 0.f), 1.f);
    d[1] = fminf(fmaxf((worldPos.y - aabbMin.y) / (ry > 0.f ? ry : 1.f), 0.f), 1.f);
    d[2] = fminf(fmaxf((worldPos.z - aabbMin.z) / (rz > 0.f ? rz : 1.f), 0.f), 1.f);
    d[3] = bdrcSignCamLabel(ng, worldPos, referenceOrigin);
    d[4] = side;
}

#endif // UNIFIED_FEATURES_H
