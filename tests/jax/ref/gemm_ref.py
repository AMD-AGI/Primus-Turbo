###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp


def _grouped_gemm_ref_impl(a, b, group_lens_tuple, trans_b):
    out = []
    start = 0
    for i, size in enumerate(group_lens_tuple):
        rhs = b[i].T if trans_b else b[i]
        out.append(a[start : start + size] @ rhs)
        start += size
    return jnp.concatenate(out, axis=0)


def _grouped_gemm_ref_fwd_bwd_impl(a, b, group_lens_tuple, trans_b):
    def loss_fn(a, b):
        out = _grouped_gemm_ref_impl(a, b, group_lens_tuple, trans_b)
        return jnp.sum(out), out

    (_, out), (grad_a, grad_b) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(a, b)
    return out, grad_a, grad_b


_grouped_gemm_ref_fwd_bwd_jit = jax.jit(_grouped_gemm_ref_fwd_bwd_impl, static_argnums=(2, 3))


def grouped_gemm_ref_fwd_bwd(a, b, group_lens, trans_b=True):
    group_lens_tuple = tuple(int(x) for x in group_lens)
    return _grouped_gemm_ref_fwd_bwd_jit(a, b, group_lens_tuple, trans_b)


def generate_grouped_gemm_group_lens(b, m, balance=True):
    if balance:
        return jnp.full((b,), m, dtype=jnp.int64)
    key = jax.random.PRNGKey(42)
    dist = 0.2 + 0.8 * jax.random.uniform(key, (b,))
    dist = dist / dist.sum()
    group_lens = (dist * b * m).astype(jnp.int64)
    group_lens = group_lens.at[-1].add(b * m - group_lens.sum())
    return group_lens
