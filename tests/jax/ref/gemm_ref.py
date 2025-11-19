###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp


def grouped_gemm_ref(a, b, group_lens, trans_b=True):
    """Reference implementation of grouped GEMM using JAX ops (matches PyTorch version)."""
    group_lens_np = jnp.array(group_lens)
    out = []
    start = 0
    for i in range(len(group_lens_np)):
        size = int(group_lens_np[i])
        rhs = b[i, :, :].T if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return jnp.concatenate(out, axis=0)


def grouped_gemm_variable_k_ref(a, b, group_lens, trans_a=True, trans_b=False):
    """Reference implementation of grouped GEMM with variable K."""
    assert trans_a == True and trans_b == False, "Only trans_a=True and trans_b=False are supported."
    group_lens_np = jnp.array(group_lens)
    B = len(group_lens_np)
    M = a.shape[1]
    N = b.shape[1]
    out = jnp.zeros((B, M, N), dtype=a.dtype)
    start = 0
    for i in range(B):
        size = int(group_lens_np[i])
        a_tmp = a[start : start + size, :].T
        b_tmp = b[start : start + size, :]
        out_tmp = a_tmp @ b_tmp
        out = out.at[i].set(out_tmp)
        start += size
    return out


def generate_grouped_gemm_group_lens(b, m, balance=True):
    """Generate group lengths (matches PyTorch version)."""
    if balance:
        return jnp.full((b,), m, dtype=jnp.int64)
    else:
        key = jax.random.PRNGKey(42)
        dist = 0.2 + 0.8 * jax.random.uniform(key, (b,))
        dist = dist / dist.sum()
        group_lens = (dist * b * m).astype(jnp.int64)
        error = b * m - group_lens.sum()
        group_lens = group_lens.at[-1].add(error)
        return group_lens
