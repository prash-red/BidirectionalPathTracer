/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * NrcConfig.h — tiny-cuda-nn config for real-time Neural Radiance Caching
 * following Mueller et al. 2021.
 *
 * Key defaults in this file:
 *   - Encoding: Composite(TriangleWave(3 dims, 12 freqs), OneBlob(5 dims, 4 bins), Identity(rest))
 *   - Encoded width: 62 (runtime pads to 64 for tensor-core friendly FullyFusedMLP)
 *   - Network: FullyFusedMLP with 5 hidden layers, 64 neurons each, ReLU, linear RGB output
 *   - Optimizer: Adam
 *   - Loss: RelativeL2 (optionally RelativeL2Luminance for RGB)
 */

#pragma once

#ifndef NRCCONFIG_H
#define NRCCONFIG_H

#include <json/json.hpp>

#include <stdexcept>
#include <string>

struct EMAConfig
{
    bool  enabled   = false;
    float decay     = 0.999f;
    int   startIter = 0;
};

struct NrcConfig
{
    // Raw input layout expected by NRC feature packing.
    // 3 position + 5 non-linear appearance + 6 linear reflectance features = 14.
    int inputDim = 14;

    // Composite encoding blocks (Mueller 2021 NRC).
    int spatialDims           = 3;
    int spatialFrequencies    = 12; // 2^d hierarchy via TriangleWave basis.
    int appearanceDims        = 5;
    int appearanceBins        = 4;  // OneBlob quartic-kernel bins.

    // Encoded dimensions: 3*12 + 5*4 + 6 = 62, then padded to 64 upstream.
    int encodedFeatureDim     = 62;
    int paddedNetworkInputDim = 64;

    // Fully fused MLP: five hidden layers of 64 neurons.
    std::string networkType      = "FullyFusedMLP";
    std::string hiddenActivation = "ReLU";
    std::string outputActivation = "None";
    int         hiddenDim        = 64;
    int         numHiddenLayers  = 5;

    // Optimizer (Adam performs best for low per-frame sample budgets).
    float lr = 1e-2f;

    // Relative loss variant for RGB output.
    bool useLuminanceRelativeL2 = true;

    EMAConfig ema;

    nlohmann::json toEncodingJson() const
    {
        if (inputDim < spatialDims + appearanceDims) {
            throw std::invalid_argument("NGPConfig.inputDim is smaller than encoded sub-block dims");
        }

        return {
            {"otype", "Composite"},
            {"nested", nlohmann::json::array({
                {
                    {"n_dims_to_encode", spatialDims},
                    {"otype", "TriangleWave"},
                    {"n_frequencies", spatialFrequencies},
                },
                {
                    {"n_dims_to_encode", appearanceDims},
                    {"otype", "OneBlob"},
                    {"n_bins", appearanceBins},
                },
                {
                    {"otype", "Identity"},
                },
            })}
        };
    }

    nlohmann::json toNetworkJson() const
    {
        if (hiddenDim != 16 && hiddenDim != 32 && hiddenDim != 64 && hiddenDim != 128) {
            throw std::invalid_argument("FullyFusedMLP hiddenDim must be one of {16, 32, 64, 128}");
        }
        if (numHiddenLayers < 0) {
            throw std::invalid_argument("NGPConfig.numHiddenLayers must be >= 0");
        }

        return {
            {"otype",             networkType},
            {"activation",        hiddenActivation},
            {"output_activation", outputActivation},
            {"n_neurons",         hiddenDim},
            {"n_hidden_layers",   numHiddenLayers},
        };
    }

    nlohmann::json toLossJson() const
    {
        // tiny-cuda-nn implements the RelativeL2 normalization internally.
        return {
            {"otype", useLuminanceRelativeL2 ? "RelativeL2Luminance" : "RelativeL2"},
        };
    }

    nlohmann::json toOptimizerJson() const
    {
        return {
            {"otype",          "Adam"},
            {"learning_rate",  lr},
            {"beta1",          0.9f},
            {"beta2",          0.99f},
            {"epsilon",        1e-8f},
            {"l2_reg",         0.0f},
        };
    }
};

#endif // NRCCONFIG_H

