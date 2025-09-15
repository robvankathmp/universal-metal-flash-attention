#!/usr/bin/env python3

import torch
from metal_sdpa_backend import metal_sdpa_function

print("🔬 Testing FP32 fix for multi-head attention...")

# Test case that should use multi-head path (numHeads > 1)
batch_size = 1
seq_len_q = 1
seq_len_kv = 1
num_heads = 2  # This will trigger multi-head path
head_dim = 4

device = torch.device("cpu")

# Create simple test tensors (all ones for easy verification)
query = torch.ones(batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float32, device=device)
key = torch.ones(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device)
value = torch.ones(batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device)

print(f"Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
print(f"Input dtypes: Q={query.dtype}, K={key.dtype}, V={value.dtype}")

# Run the attention
output = metal_sdpa_function(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
print(f"Output values:\n{output}")

# Verify we got FP32 output
if output.dtype == torch.float32:
    print("✅ SUCCESS: Output is FP32!")
else:
    print(f"❌ FAILURE: Output is {output.dtype}, expected FP32")

# Verify values are reasonable (should be non-zero)
if torch.all(output != 0):
    print("✅ SUCCESS: Output contains non-zero values!")
else:
    print("❌ FAILURE: Output contains zeros")

print("🎯 FP32 fix test complete!")