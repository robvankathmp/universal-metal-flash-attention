#!/usr/bin/env python3
"""
Test script to verify the fix for quantized attention tensor destructor crash.
This script tests the specific scenario: seq_len=2 with quantized attention.
"""
import sys
from pathlib import Path

import torch

# Add the build directory to sys.path to import the extension
build_path = Path(__file__).parent / "build" / "lib.macosx-15.0-arm64-cpython-312"
sys.path.insert(0, str(build_path))

try:
    import metal_sdpa_extension

    print("✅ Successfully imported metal_sdpa_extension")
except ImportError as e:
    print(f"❌ Failed to import metal_sdpa_extension: {e}")
    print(
        "Make sure to build the extension first with: python setup.py build_ext --inplace"
    )
    sys.exit(1)


def test_quantized_seq_len_2():
    """Test the specific crash scenario: quantized attention with seq_len=2"""
    print("\n🧪 Testing quantized attention with seq_len=2 (crash scenario)")

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available on this device")
        return False

    try:
        # Create tensors with seq_len=2 (the problematic case)
        batch_size = 1
        seq_len = 2  # This is the sequence length that causes the crash
        num_heads = 1
        head_dim = 64

        # Create 4D tensors [batch, seq_len, num_heads, head_dim]
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

        print(f"Input tensor shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

        # Test quantized attention - this should NOT crash after the fix
        result = metal_sdpa_extension.quantized_scaled_dot_product_attention(
            q, k, v, precision="int8", is_causal=False, scale=None
        )

        print(f"✅ Success! Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")

        # Verify output is not NaN
        if torch.isnan(result).any():
            print("⚠️  Warning: Output contains NaN values")
            return False

        print(f"Output range: [{result.min().item():.4f}, {result.max().item():.4f}]")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_quantized_seq_len_1():
    """Test with seq_len=1 (working case) for comparison"""
    print("\n🧪 Testing quantized attention with seq_len=1 (working case)")

    try:
        # Create tensors with seq_len=1 (known working case)
        batch_size = 1
        seq_len = 1  # This works
        num_heads = 1
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

        result = metal_sdpa_extension.quantized_scaled_dot_product_attention(
            q, k, v, precision="int8", is_causal=False, scale=None
        )

        print(f"✅ Success! Output shape: {result.shape}")
        print(f"Output range: [{result.min().item():.4f}, {result.max().item():.4f}]")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def main():
    print("🔧 Testing Fix for Quantized Attention Tensor Destructor Crash")
    print("=" * 70)

    # Register the backend
    metal_sdpa_extension.register_backend()

    try:
        # Test both cases
        seq_len_1_success = test_quantized_seq_len_1()
        seq_len_2_success = test_quantized_seq_len_2()

        print("\n" + "=" * 70)
        print("📊 RESULTS SUMMARY")
        print("=" * 70)
        print(f"seq_len=1: {'✅ PASS' if seq_len_1_success else '❌ FAIL'}")
        print(f"seq_len=2: {'✅ PASS' if seq_len_2_success else '❌ FAIL'}")

        if seq_len_1_success and seq_len_2_success:
            print("\n🎉 SUCCESS: Both test cases passed!")
            print("The tensor destructor crash has been fixed.")
        elif seq_len_1_success and not seq_len_2_success:
            print("\n⚠️  PARTIAL SUCCESS: seq_len=1 works but seq_len=2 still crashes")
            print("The fix may not be complete or there's another issue.")
        else:
            print("\n❌ FAILURE: Tests failed")

    finally:
        # Unregister the backend
        metal_sdpa_extension.unregister_backend()


if __name__ == "__main__":
    main()
