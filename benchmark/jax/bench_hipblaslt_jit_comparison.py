"""
Benchmark Primus-Turbo hipBLASLt: JIT vs no-JIT comparison

This script tests whether Primus-Turbo gets similar JIT acceleration as grouped_gemm_jax turbo
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import sys

# Enable 64-bit
jax.config.update("jax_enable_x64", True)

print("=" * 80)
print("Primus-Turbo hipBLASLt JIT Comparison Test")
print("=" * 80)
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Import Primus-Turbo
try:
    from primus_turbo.jax.lax import grouped_gemm_hipblaslt
    print("✓ Primus-Turbo imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Primus-Turbo: {e}")
    sys.exit(1)

# Configuration (GateUp from bench_grouped_gemm_jax_pure.py)
num_experts = 16
tokens_per_expert = 4096
N, K = 4096, 2048

total_tokens = num_experts * tokens_per_expert

print(f"\nConfiguration:")
print(f"  num_experts: {num_experts}")
print(f"  tokens_per_expert: {tokens_per_expert}")
print(f"  total_tokens: {total_tokens}")
print(f"  K: {K}, N: {N}")

# Create input data
print("\nPreparing input data...")
key = jax.random.PRNGKey(42)
key, subkey1, subkey2 = jax.random.split(key, 3)

a = jax.random.normal(subkey1, (total_tokens, K), dtype=jnp.bfloat16) * 0.01
b = jax.random.normal(subkey2, (num_experts, N, K), dtype=jnp.bfloat16) * 0.01
group_lens = jnp.array([tokens_per_expert] * num_experts, dtype=jnp.int64)

print(f"  a shape: {a.shape}, dtype: {a.dtype}")
print(f"  b shape: {b.shape}, dtype: {b.dtype}")
print(f"  group_lens shape: {group_lens.shape}, dtype: {group_lens.dtype}")

NUM_WARMUP = 10
NUM_ITERS = 100

# Total FLOPs
total_flops = 2 * total_tokens * N * K  # Forward
total_flops_bwd = 3 * total_flops  # Forward + 2 backward GEMMs

# ============================================================================
# Test 1: Forward Pass - no JIT
# ============================================================================

print("\n" + "=" * 80)
print("Test 1: Forward Pass (no JIT)")
print("=" * 80)

def forward_no_jit(a, b, group_lens):
    return grouped_gemm_hipblaslt(a, b, group_lens, transB=True)

# Warmup
print(f"Warming up ({NUM_WARMUP} iterations)...")
for _ in range(NUM_WARMUP):
    out = forward_no_jit(a, b, group_lens)
    out.block_until_ready()

# Benchmark
print(f"Benchmarking ({NUM_ITERS} iterations)...")
start = time.perf_counter()
for _ in range(NUM_ITERS):
    out = forward_no_jit(a, b, group_lens)
    out.block_until_ready()
end = time.perf_counter()

fwd_time_no_jit = (end - start) * 1000 / NUM_ITERS
fwd_tflops_no_jit = (total_flops / 1e12) / (fwd_time_no_jit / 1000)

print(f"✓ Forward (no JIT):  {fwd_time_no_jit:.3f} ms | {fwd_tflops_no_jit:.2f} TFLOPS")

# ============================================================================
# Test 2: Forward Pass - WITH JIT
# ============================================================================

print("\n" + "=" * 80)
print("Test 2: Forward Pass (WITH JIT)")
print("=" * 80)

@jax.jit
def forward_jit(a, b, group_lens):
    return grouped_gemm_hipblaslt(a, b, group_lens, transB=True)

# Warmup
print(f"Warming up ({NUM_WARMUP} iterations)...")
for _ in range(NUM_WARMUP):
    out = forward_jit(a, b, group_lens)
    out.block_until_ready()

# Benchmark
print(f"Benchmarking ({NUM_ITERS} iterations)...")
start = time.perf_counter()
for _ in range(NUM_ITERS):
    out = forward_jit(a, b, group_lens)
    out.block_until_ready()
end = time.perf_counter()

fwd_time_jit = (end - start) * 1000 / NUM_ITERS
fwd_tflops_jit = (total_flops / 1e12) / (fwd_time_jit / 1000)

print(f"✓ Forward (JIT):     {fwd_time_jit:.3f} ms | {fwd_tflops_jit:.2f} TFLOPS")
print(f"  JIT Speedup: {fwd_time_no_jit / fwd_time_jit:.2f}x")

# ============================================================================
# Test 3: Backward Pass - no JIT
# ============================================================================

print("\n" + "=" * 80)
print("Test 3: Backward Pass (no JIT)")
print("=" * 80)

def loss_fn_no_jit(a, b):
    out = grouped_gemm_hipblaslt(a, b, group_lens, transB=True)
    return jnp.sum(out ** 2)

grad_fn_no_jit = jax.grad(loss_fn_no_jit, argnums=(0, 1))

# Warmup
print(f"Warming up ({NUM_WARMUP} iterations)...")
for _ in range(NUM_WARMUP):
    grad_a, grad_b = grad_fn_no_jit(a, b)
    grad_a.block_until_ready()
    grad_b.block_until_ready()

# Benchmark
print(f"Benchmarking ({NUM_ITERS} iterations)...")
start = time.perf_counter()
for _ in range(NUM_ITERS):
    grad_a, grad_b = grad_fn_no_jit(a, b)
    grad_a.block_until_ready()
    grad_b.block_until_ready()
end = time.perf_counter()

bwd_time_no_jit = (end - start) * 1000 / NUM_ITERS
bwd_only_time_no_jit = bwd_time_no_jit - fwd_time_no_jit
bwd_tflops_no_jit = (total_flops_bwd / 1e12) / (bwd_time_no_jit / 1000)

print(f"✓ Backward (no JIT): {bwd_time_no_jit:.3f} ms | {bwd_tflops_no_jit:.2f} TFLOPS (fwd+bwd)")
print(f"  Backward only:     {bwd_only_time_no_jit:.3f} ms")

# ============================================================================
# Test 4: Backward Pass - WITH JIT
# ============================================================================

print("\n" + "=" * 80)
print("Test 4: Backward Pass (WITH JIT)")
print("=" * 80)

@jax.jit
def loss_fn_jit(a, b):
    out = grouped_gemm_hipblaslt(a, b, group_lens, transB=True)
    return jnp.sum(out ** 2)

grad_fn_jit = jax.jit(jax.grad(loss_fn_jit, argnums=(0, 1)))

# Warmup
print(f"Warming up ({NUM_WARMUP} iterations)...")
for _ in range(NUM_WARMUP):
    grad_a, grad_b = grad_fn_jit(a, b)
    grad_a.block_until_ready()
    grad_b.block_until_ready()

# Benchmark
print(f"Benchmarking ({NUM_ITERS} iterations)...")
start = time.perf_counter()
for _ in range(NUM_ITERS):
    grad_a, grad_b = grad_fn_jit(a, b)
    grad_a.block_until_ready()
    grad_b.block_until_ready()
end = time.perf_counter()

bwd_time_jit = (end - start) * 1000 / NUM_ITERS
bwd_only_time_jit = bwd_time_jit - fwd_time_jit
bwd_tflops_jit = (total_flops_bwd / 1e12) / (bwd_time_jit / 1000)

print(f"✓ Backward (JIT):    {bwd_time_jit:.3f} ms | {bwd_tflops_jit:.2f} TFLOPS (fwd+bwd)")
print(f"  Backward only:     {bwd_only_time_jit:.3f} ms")
print(f"  JIT Speedup:       {bwd_time_no_jit / bwd_time_jit:.2f}x")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nForward Performance:")
print(f"  no JIT:  {fwd_time_no_jit:.3f} ms | {fwd_tflops_no_jit:.2f} TFLOPS")
print(f"  JIT:     {fwd_time_jit:.3f} ms | {fwd_tflops_jit:.2f} TFLOPS")
print(f"  Speedup: {fwd_time_no_jit / fwd_time_jit:.2f}x")

print("\nBackward Performance:")
print(f"  no JIT:  {bwd_time_no_jit:.3f} ms | {bwd_tflops_no_jit:.2f} TFLOPS")
print(f"  JIT:     {bwd_time_jit:.3f} ms | {bwd_tflops_jit:.2f} TFLOPS")
print(f"  Speedup: {bwd_time_no_jit / bwd_time_jit:.2f}x")

print("\n" + "=" * 80)
print("Comparison with grouped_gemm_jax turbo:")
print("=" * 80)
print("\ngrouped_gemm_jax turbo (from benchmark):")
print("  Forward (JIT):  1.018 ms | 1079.61 TFLOPS")
print("  Backward (JIT): 2.258 ms | 973.91 TFLOPS")
print("  JIT Speedup (bwd): 2.28x")

print("\nPrimus-Turbo hipBLASLt:")
print(f"  Forward (JIT):  {fwd_time_jit:.3f} ms | {fwd_tflops_jit:.2f} TFLOPS")
print(f"  Backward (JIT): {bwd_time_jit:.3f} ms | {bwd_tflops_jit:.2f} TFLOPS")
print(f"  JIT Speedup (bwd): {bwd_time_no_jit / bwd_time_jit:.2f}x")

# Performance gap
fwd_gap = (fwd_time_jit / 1.018 - 1) * 100
bwd_gap = (bwd_time_jit / 2.258 - 1) * 100
jit_speedup_gap = (2.28 / (bwd_time_no_jit / bwd_time_jit) - 1) * 100

print("\nPerformance Gap:")
print(f"  Forward (JIT):       {fwd_gap:+.1f}%  (negative is better)")
print(f"  Backward (JIT):      {bwd_gap:+.1f}%  (negative is better)")
print(f"  JIT Speedup (bwd):   {jit_speedup_gap:+.1f}%  (negative means less speedup)")

print("\n" + "=" * 80)
if bwd_time_jit / bwd_time_no_jit < 1.5:
    print("⚠️  WARNING: JIT speedup < 1.5x for backward pass!")
    print("    This suggests JIT optimization is not working as expected.")
    print("    grouped_gemm_jax turbo achieves 2.28x speedup.")
elif bwd_time_jit / bwd_time_no_jit >= 2.0:
    print("✓ JIT optimization is working well!")
    print(f"  Achieving {bwd_time_no_jit / bwd_time_jit:.2f}x backward speedup.")
else:
    print("⚠️  JIT optimization is working, but could be better.")
    print(f"    Current speedup: {bwd_time_no_jit / bwd_time_jit:.2f}x")
    print("    grouped_gemm_jax turbo: 2.28x")

print("=" * 80)

