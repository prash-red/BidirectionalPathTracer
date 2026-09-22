//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// dre_config.h — the unified field's generator selection.
//
// Deliberately free of tiny-cuda-nn (and therefore of nvcc): Application.cpp is
// a plain C++ translation unit and must be able to name a DreConfig to pass
// through bdrc_host.h's opaque interface without dragging tcnn's device headers
// into a host compile.
//
#pragma once
#ifndef GLTF_RENDERER_DRECONFIG_H
#define GLTF_RENDERER_DRECONFIG_H

// Density-ratio generator family. Same membership and numbering as the demo's
// dre_loss.wgsl OBJECTIVE override, because the scene recipes there disagree on
// which member is best and the choice is worth exposing per recipe.
enum class DreObjective
{
    Logistic = 0,  // posterior form; fixed point independent of alpha
    Brier    = 1,  // posterior form; fixed point independent of alpha
    Lsif     = 2,  // raw ratio; requires alpha == beta
    Bregman  = 3,  // raw ratio at arbitrary power; requires alpha == beta
};

struct DreConfig
{
    DreObjective objective = DreObjective::Logistic;

    // alpha is NCE's noise ratio and it matters. The field runs 1e2-1e4 times
    // the reference density, so at alpha = 1 the posterior saturates and the
    // shared head weighs the brightest regions least. The posterior members'
    // fixed point does not move with alpha; only the loss curvature does.
    float alpha = 1000.0f;
    float beta  = 1.0f;

    // Bregman power. |power - 2| < 1e-9 degenerates to Lsif, matching the demo.
    float power = 2.0f;
};

// True when the generator reads u* = (beta/alpha) * sum_s p_s w_bar_s, i.e.
// when unequal stream constants would rescale the whole field.
inline bool dreIsRawRatio(DreObjective o)
{
    return o == DreObjective::Lsif || o == DreObjective::Bregman;
}

#endif // GLTF_RENDERER_DRECONFIG_H
