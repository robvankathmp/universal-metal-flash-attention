# Universal Metal Flash Attention - Development Makefile
# Provides consistent build, test, and debug workflows

# Configuration
PROJECT_ROOT := $(shell cd ../.. && pwd)
VENV_PATH := $(PROJECT_ROOT)/.venv
SWIFT_BUILD_PATH := $(PROJECT_ROOT)/.build
FLUX_DIR := $(shell pwd)
PYTORCH_FFI_DIR := $(FLUX_DIR)/pytorch-custom-op-ffi

# Environment setup
SWIFT_BUILD_RELEASE := $(SWIFT_BUILD_PATH)/arm64-apple-macosx/release
SWIFT_BUILD_DEBUG := $(SWIFT_BUILD_PATH)/arm64-apple-macosx/debug
DYLD_LIBRARY_PATH := $(SWIFT_BUILD_RELEASE):$(SWIFT_BUILD_DEBUG)
PYTHON := $(VENV_PATH)/bin/python

# Default target
.PHONY: all
all: build test-single

# Build targets
.PHONY: build build-swift build-pytorch
build: build-swift build-pytorch

build-swift:
	@echo "🔨 Building Swift MFA library..."
	cd $(PROJECT_ROOT) && swift build
	@echo "✅ Swift build complete"

build-pytorch: build-swift
	@echo "🔨 Building PyTorch FFI extension..."
	cd $(PYTORCH_FFI_DIR) && \
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) $(PYTHON) setup.py build_ext --inplace
	@echo "✅ PyTorch extension build complete"

# Test targets - SHOWING THE PROBLEM
.PHONY: test-single test-multihead-broken
test-single: build-pytorch
	@echo "🧪 Testing single-head attention..."
	cd $(FLUX_DIR) && \
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) $(PYTHON) -c "\
import sys; sys.path.append('pytorch-custom-op-ffi'); \
import torch, metal_sdpa_extension; \
q=k=v=torch.randn(1,4,1,8,dtype=torch.float16)*0.1; \
result=metal_sdpa_extension.quantized_scaled_dot_product_attention(q,k,v,precision='int8'); \
print(f'✅ Single-head works: {result.shape}')"

test-multihead-broken: build-pytorch
	@echo "🚨 Testing multi-head attention (BROKEN - not real MHA)..."
	cd $(FLUX_DIR) && \
	DYLD_LIBRARY_PATH=$(DYLD_LIBRARY_PATH) $(PYTHON) -c "\
import sys; sys.path.append('pytorch-custom-op-ffi'); \
import torch, metal_sdpa_extension; \
q=k=v=torch.randn(1,4,4,8,dtype=torch.float16)*0.1; \
print('⚠️  This runs 4 single-head operations sequentially - NOT parallel MHA!'); \
result=metal_sdpa_extension.quantized_scaled_dot_product_attention(q,k,v,precision='int8'); \
print(f'🚨 Fake multi-head result: {result.shape}')"

# Show the proper MHA approach
.PHONY: show-real-mha-approach
show-real-mha-approach:
	@echo "🔧 REAL Multi-Head Attention should:"
	@echo "   1. Use MultiHeadAttention class from metal-flash-attention submodule"
	@echo "   2. Process all heads in parallel using proper BHSD layout"
	@echo "   3. Leverage GPU parallelism across heads"
	@echo "   4. Use efficient tensor reshaping and broadcasting"
	@echo ""
	@echo "   Current implementation just runs single-head 4 times = SLOWER!"

# Clean targets
.PHONY: clean
clean:
	cd $(PROJECT_ROOT) && swift package clean
	find $(FLUX_DIR) -name "*.so" -delete

.PHONY: help
help:
	@echo "Universal Metal Flash Attention - Development Makefile"
	@echo ""
	@echo "Commands:"
	@echo "  make build                    - Build Swift + PyTorch extension"
	@echo "  make test-single              - Test single-head (works)"
	@echo "  make test-multihead-broken    - Test fake multi-head (broken approach)"
	@echo "  make show-real-mha-approach   - Show what real MHA should look like"
	@echo "  make clean                    - Clean build artifacts"
