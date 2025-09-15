#pragma once

// Include the main header with all the Swift FFI declarations
#include "../include/metal_sdpa_backend.h"
#include <string>
#include <optional>

namespace metal_sdpa {

// Supported output precision types
enum class OutputPrecision {
    FP32,  // Default - Float32
    FP16,  // Float16
    BF16   // BFloat16
};

// Configuration for quantized attention operations
struct QuantizationConfig {
    std::string precision = "int8";           // Quantization precision: "int8", "int4"
    OutputPrecision output_precision = OutputPrecision::FP32;  // Output buffer precision
    bool is_causal = false;                   // Causal masking
    std::optional<double> scale = std::nullopt; // Custom scaling factor

    // Convert string to OutputPrecision enum
    static OutputPrecision string_to_precision(const std::string& precision_str) {
        if (precision_str == "fp16" || precision_str == "float16") {
            return OutputPrecision::FP16;
        } else if (precision_str == "bf16" || precision_str == "bfloat16") {
            return OutputPrecision::BF16;
        } else {
            return OutputPrecision::FP32;  // Default fallback
        }
    }

    // Convert OutputPrecision enum to string
    static std::string precision_to_string(OutputPrecision precision) {
        switch (precision) {
            case OutputPrecision::FP16: return "fp16";
            case OutputPrecision::BF16: return "bf16";
            case OutputPrecision::FP32:
            default: return "fp32";
        }
    }
};

// Extension functions for MetalSDPABackend class (declared in main header)
torch::Tensor quantized_scaled_dot_product_attention_with_config(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const QuantizationConfig& config = QuantizationConfig{}
);

} // namespace metal_sdpa