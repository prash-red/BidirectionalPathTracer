#pragma once
#ifndef GRAD_SCALER_H
#define GRAD_SCALER_H

#include <tiny-cuda-nn/common.h>

// Per-cache dynamic loss scaler (mirrors Python GradScaler pattern).
//
// Usage per training step:
//   1. float lossScale = scaler.currentScale()      — scale the loss
//   2. forward + backward (loss already scaled)
//   3. if (config.gradScaler) checkGradOverflow()
//   4a. no overflow → scaler.onSuccess(); optimizer.step(lossScale)
//   4b. overflow    → scaler.onOverflow(); skip optimizer step
struct GradScaler
{
    bool  enabled          = false;
    float scale            = 65536.0f;
    float growthFactor     = 2.0f;
    bool  growthEnabled    = true;
    int   growthInterval   = 2000;   // consecutive clean steps before growing scale
    int   consecutiveClean = 0;

    void init(bool _enabled, float initScale, float _growthFactor, bool _growthEnabled)
    {
        enabled        = _enabled;
        scale          = initScale;
        growthFactor   = _growthFactor;
        growthEnabled  = _growthEnabled;
        consecutiveClean = 0;
    }

    // Returns the current dynamic scale when enabled, or the tcnn default when disabled.
    float currentScale() const
    {
        return enabled ? scale
                       : tcnn::default_loss_scale<tcnn::network_precision_t>();
    }

    // Call after a clean optimizer step (no overflow detected).
    void onSuccess()
    {
        if (!enabled) return;
        ++consecutiveClean;
        if (growthEnabled && consecutiveClean >= growthInterval) {
            scale *= growthFactor;
            consecutiveClean = 0;
        }
    }

    // Call when NaN/Inf was detected in gradients — skips step, halves scale.
    void onOverflow()
    {
        if (!enabled) return;
        scale /= growthFactor;
        if (scale < 1.0f) scale = 1.0f;
        consecutiveClean = 0;
    }
};

#endif // GRAD_SCALER_H