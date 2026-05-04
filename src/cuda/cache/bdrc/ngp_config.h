#ifndef GLTF_RENDERER_NGPCONFIG_H
#define GLTF_RENDERER_NGPCONFIG_H
#include "json/json.hpp"

struct EMAConfig
{
    bool  enabled   = false;
    float decay     = 0.999f;
    int   startIter = 0;
};

struct ngp_config
{
    // Base input layout for a 3D hash-grid cache.
    int inputDim = 3;

    // Hash grid encoding settings.
    int   nLevels           = 6;
    int   nFeaturesPerLevel = 4;
    float perLevelScale     = 2.0f;
    int   baseResolution    = 16;
    int   log2HashmapSize   = 19;

    // MLP settings.
    int         hiddenDim       = 64;
    int         numHiddenLayers = 2;
    float       lr              = 1e-2f;
    float       mlpLrScale      = 1.0f;
    std::string hiddenActivation = "ReLu";
    std::string outputActivation = "Softplus";

    // Optimizer settings.
    std::string optimizer   = "adam"; // "sgd" or "adam"
    float       momentum    = 0.9f;
    bool        amsgrad     = false;
    bool        decoupledWd = false;
    bool        lamb        = false;

    // Precision and gradient-scaling controls.
    bool  gradScaler             = true;
    float gradScalerInitScale    = 65536.0f;
    float gradScalerGrowthFactor = 2.0f;
    bool  gradScalerGrowth       = true;

    // Gradient clipping and regularization controls.
    float gradClip       = 0.0f;
    float weightDecayMlp = 0.0f;
    float weightDecayEnc = 0.0f;

    EMAConfig ema;

    nlohmann::json toEncodingJson() const
    {
        if (inputDim <= 0) {
            throw std::invalid_argument("NGPConfig.inputDim must be > 0");
        }
        if (nLevels <= 0 || nFeaturesPerLevel <= 0 || baseResolution <= 0 || log2HashmapSize <= 0) {
            throw std::invalid_argument("NGPConfig hash-grid parameters must be > 0");
        }
        if (perLevelScale <= 1.0f) {
            throw std::invalid_argument("NGPConfig.perLevelScale must be > 1.0");
        }

        return {
            {"otype", "HashGrid"},
            {"n_levels", nLevels},
            {"n_features_per_level", nFeaturesPerLevel},
            {"log2_hashmap_size", log2HashmapSize},
            {"base_resolution", baseResolution},
            {"per_level_scale", perLevelScale},
        };
    }

    // nlohmann::json toEncodingJson() const
    // {
    //     int spatialFrequencies    = 12; // 2^d hierarchy via TriangleWave basis.
    //     return {
    //                 {"otype", "TriangleWave"},
    //                 {"n_frequencies", spatialFrequencies}};
    // }

    nlohmann::json toNetworkJson() const
    {
        if (hiddenDim != 16 && hiddenDim != 32 && hiddenDim != 64 && hiddenDim != 128) {
            throw std::invalid_argument("FullyFusedMLP hiddenDim must be one of {16, 32, 64, 128}");
        }
        if (numHiddenLayers < 0) {
            throw std::invalid_argument("NGPConfig.numHiddenLayers must be >= 0");
        }


        return {
            {"otype",             "FullyFusedMLP"},
            {"activation",        hiddenActivation},
            {"output_activation", outputActivation},
            {"n_neurons",         hiddenDim},
            {"n_hidden_layers",   numHiddenLayers},
        };
    }

    nlohmann::json toLossJson() const
    {
        return {{"otype", "L2"}};
    }

    nlohmann::json toOptimizerJson() const
    {
        const std::string optimizerLower = optimizer;
        if (optimizerLower == "sgd" || optimizerLower == "SGD") {
            return {
                {"otype", "SGD"},
                {"learning_rate", lr},
                {"momentum", momentum},
                {"l2_reg", 0.0f},
            };
        }

        if (!(optimizerLower == "adam" || optimizerLower == "Adam")) {
            throw std::invalid_argument("NGPConfig.optimizer must be 'adam' or 'sgd'");
        }

        return {
            {"otype",          "Adam"},
            {"learning_rate",  lr * mlpLrScale},
            {"beta1",          momentum},
            {"beta2",          0.99f},
            {"epsilon",        1e-8f},
            {"l2_reg",         0.0f},
        };
    }
};

#endif //GLTF_RENDERER_NGPCONFIG_H
