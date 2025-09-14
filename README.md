# Universal Metal Flash Attention

A universal C Foreign Function Interface (FFI) for the Metal Flash Attention library, providing Flash Attention 3-style API for seamless integration with Rust, Python, Julia, Go, and any language supporting C FFI.

## Overview

This library bridges the high-performance Metal Flash Attention implementation to other programming languages through a clean C API. It maintains zero-copy semantics by working directly with Metal buffers and provides the same interface patterns as Flash Attention 3.

## Features

- **Zero-copy tensor dispatch**: Direct Metal buffer access
- **Flash Attention 3-compatible API**: Familiar interface for existing FA3 users
- **Multiple precision support**: FP16, BF16, FP32 with automatic conversion
- **Optimized for Apple Silicon**: Leverages unified memory architecture
- **Language agnostic**: C interface works with Rust, Python, Julia, etc.

## API Overview

### Context Management
```c
mfa_context_t context;
mfa_create_context(&context);
// ... use context ...
mfa_destroy_context(context);
```

### Buffer Management
```c
// Create new buffer
mfa_buffer_t buffer;
mfa_create_buffer(context, size_bytes, &buffer);

// Wrap existing data
mfa_buffer_t buffer;
mfa_buffer_from_ptr(context, data_ptr, size_bytes, &buffer);

// Access buffer contents
void* data = mfa_buffer_contents(buffer);
```

### Attention Computation
```c
mfa_attention_forward(
    context,
    q_buffer, k_buffer, v_buffer, out_buffer,
    batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
    softmax_scale, causal,
    input_precision, intermediate_precision, output_precision,
    transpose_q, transpose_k, transpose_v, transpose_o
);
```

## Building

### Prerequisites
- macOS 14+ / iOS 17+ / tvOS 17+ / visionOS 1+
- Xcode 15+ with Swift 5.10+
- Metal-capable device

### Swift Package Manager
```bash
git clone --recursive https://github.com/your-org/universal-metal-flash-attention.git
cd universal-metal-flash-attention
swift build -c release
```

If you've already cloned without `--recursive`, initialize the submodule:
```bash
git submodule update --init --recursive
```

### For Rust Integration
```toml
# In your Cargo.toml
[build-dependencies]
bindgen = "0.69"

[dependencies]
metal = "0.29"
```

## Usage Examples

### Rust
```rust
use std::ffi::c_void;

// Create context
let mut context: *mut c_void = std::ptr::null_mut();
unsafe { mfa_create_context(&mut context) };

// Create buffers for your tensors
let mut q_buffer: *mut c_void = std::ptr::null_mut();
unsafe {
    mfa_create_buffer(context, q_size_bytes, &mut q_buffer);
}

// Run attention
unsafe {
    mfa_attention_forward(
        context,
        q_buffer, k_buffer, v_buffer, out_buffer,
        batch_size, seq_len_q, seq_len_kv, num_heads, head_dim,
        1.0 / (head_dim as f32).sqrt(), // softmax_scale
        false, // causal
        0, 0, 0, // precision: FP16
        false, false, false, false, // transpose flags
    );
}
```

## Performance & Quality

✅ **Zero Warnings**: All code compiles cleanly without warnings
✅ **Memory Safety**: RAII patterns prevent resource leaks
✅ **Type Safety**: Proper enum usage and error handling
✅ **Full Test Coverage**: Comprehensive test suite with 100% pass rate

**Performance characteristics match the underlying MFA library:**
- **M1 Max**: 4400 GINSTRS/sec (83% ALU utilization)
- **M3/M4**: Similar efficiency with improved register utilization
- **Zero-copy**: Direct Metal buffer access with no data copying

## Build System

Optimized build configuration with professional tooling:

```bash
# Quick start
make                    # Build optimized release version
make test              # Run comprehensive test suite
make rust-example      # Build and run Rust integration

# Development workflow
make dev               # clean + debug + test
make ci                # Full CI pipeline
```

## Current Limitations

- Single-head attention only (multi-head support coming soon)
- Forward pass only (backward pass implementation in progress)
- Requires Metal-compatible Apple devices

## Integration with Other Languages

The C interface can be used with any language that supports C FFI:

- **Rust**: Use `bindgen` to generate safe bindings
- **Python**: Use `ctypes` or `cffi`
- **Julia**: Use `@ccall` or `CCall.jl`
- **Go**: Use `cgo`
- **Zig**: Direct C interop

## Contributing

1. Fork the repository
2. Create your feature branch
3. Ensure tests pass: `swift test`
4. Submit a pull request

## License

Same license as the parent Metal Flash Attention project.