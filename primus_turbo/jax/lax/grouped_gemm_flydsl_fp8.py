###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FP8 (e4m3, per-tensor scales) grouped GEMM on the Primus-Turbo FlyDSL gfx950 kernels, callable from JAX.

Same bridge as flydsl_gmm.py (FlyDSL-compiled host launcher called through the XLA FFI trampoline); the kernels apply the
dequantisation scales (a_scale * b_scale, device scalars) in their epilogue and read the group offsets as int64 (passed as int32 pairs).

  grouped_gemm_fp8_flydsl(a[M,K] bf16, b[G,K,N] bf16, group_lens[G]) -> [M,N] bf16, differentiable (fp8 fwd, dA, dW)
"""

import functools

import jax
import jax.numpy as jnp

import primus_turbo.common.envs as envs

try:
    from primus_turbo.jax.lax import grouped_gemm_flydsl as F
except ImportError:
    import flydsl_gmm as F

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0


def _k8():
    from primus_turbo.flydsl.grouped_gemm import grouped_gemm_fp8_kernel as K8

    return K8


def _compile_with_hints(launch, args, out_index=2):
    import flydsl.compiler as flyc  # noqa: F401
    from flydsl.compiler.kernel_function import CompilationContext

    # The kernel module no longer re-exports these (FlyDSL autotune unification, #587).
    from primus_turbo.flydsl.utils.gemm_helper import compile_with_scratch_out

    K8 = _k8()
    with CompilationContext.compile_hints(K8._GG_SCHED_HINTS):
        return compile_with_scratch_out(launch, args, out_index=out_index)


def _dummies(torch, dev, a_shape, b_shape, c_shape, G, a_scale=True):
    z = lambda shape, dt: torch.zeros(*shape, dtype=dt, device=f"cuda:{dev}")
    return (
        z(a_shape, torch.int8),
        z(b_shape, torch.int8),
        z(c_shape, torch.bfloat16),
        z((1,), torch.float32),
        z((1,), torch.float32),
        z((2 * (G + 1),), torch.int32),
    )


@functools.lru_cache(maxsize=None)
def _nn8(M, N, K, G, bm=256, xcd=4, gm=8, gn=0):
    """out[rows] = a[rows] @ b[g]   (a [M,K] fp8, b [G,K,N] fp8)"""
    flyc, torch, _ = F._flydsl()
    K8 = _k8()

    def make_launch():
        return K8._compile_grouped_nn(
            K=K,
            G=G,
            BLOCK_M=bm,
            BLOCK_N=256,
            out_fp16=False,
            cbsz=0,
            blgp=0,
            num_xcd=xcd,
            group_m=gm,
            group_n=gn,
            persistent=False,
            agpr_inplace=True,
            store_cshuffle=False,
            sched_schedbar=False,
            nt_vmcnt=-1,
            i64_traverse=(K * N >= 2**32),
            N=N,
            beta_is_one=False,
            n_stride=0,
        )

    def make_args(dev):
        a, b, c, sa, sb, go = _dummies(torch, dev, (256, K), (G, K, N), (256, N), G)
        return (a, b, c, sa, sb, go, M, N, torch.cuda.current_stream())

    # args: (A, B, C, A_scale, B_scale, go32, m_total, c_n, stream); XLA buffers: a, b, a_scale, b_scale, go32 | out
    return F._Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 4, 5, 2], compile_fn=_compile_with_hints)


@functools.lru_cache(maxsize=None)
def _nt8(M, N, K, G, bm=256, xcd=4, gm=8, gn=0):
    """out[rows] = a[rows] @ b[g]^T  (a [M,K] fp8, b [G,N,K] fp8)"""
    flyc, torch, _ = F._flydsl()
    K8 = _k8()

    def make_launch():
        return K8._compile_grouped_nt(
            K=K,
            G=G,
            BLOCK_M=bm,
            BLOCK_N=256,
            out_fp16=False,
            cbsz=0,
            blgp=0,
            num_xcd=xcd,
            group_m=gm,
            group_n=gn,
            persistent=False,
            agpr_inplace=False,
            store_cshuffle=False,
            sched_schedbar=False,
            beta_is_one=False,
            nt_vmcnt=3,
            N=N,
            n_stride=0,
        )

    def make_args(dev):
        a, b, c, sa, sb, go = _dummies(torch, dev, (256, K), (G, N, K), (256, N), G)
        return (a, b, c, sa, sb, go, M, N, torch.cuda.current_stream())

    return F._Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 4, 5, 2], compile_fn=_compile_with_hints)


@functools.lru_cache(maxsize=None)
def _wg8(OUT_M, OUT_N, G, xcd=8, gm=4, chunk=8):
    """out[g] = lhs[rows_g]^T @ rhs[rows_g]   (lhs [M,OUT_M] fp8, rhs [M,OUT_N] fp8, out [G*OUT_M, OUT_N] bf16)"""
    flyc, torch, _ = F._flydsl()
    K8 = _k8()

    def make_launch():
        return K8._compile_grouped_tn_wgrad_masked(
            OUT_M=OUT_M,
            OUT_N=OUT_N,
            G=G,
            num_xcd=xcd,
            acc_mode="agpr",
            s2r_inline=True,
            out_fp16=False,
            cbsz=0,
            blgp=0,
            group_m=gm,
            store_cshuffle=True,
            chunk=chunk,
            i64_traverse=False,
            beta_is_one=False,
        )

    def make_args(dev):
        z = lambda shape, dt: torch.zeros(*shape, dtype=dt, device=f"cuda:{dev}")
        return (
            z((256, OUT_M), torch.int8),
            z((256, OUT_N), torch.int8),
            z((G * OUT_M, OUT_N), torch.bfloat16),
            z((1,), torch.float32),
            z((1,), torch.float32),
            z((2 * (G + 1),), torch.int32),
            torch.cuda.current_stream(),
        )

    # args: (A, B, C, A_scale, B_scale, go32, stream)
    return F._Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 4, 5, 2], compile_fn=_compile_with_hints)


def quantize(x):
    """per-tensor e4m3: returns (x_q, scale_inv f32[1]) with x ~= x_q * scale_inv"""
    amax = jnp.max(jnp.abs(x)).astype(jnp.float32)
    scale = FP8_MAX / jnp.maximum(amax, 1e-12)
    return (x.astype(jnp.float32) * scale).astype(FP8), (1.0 / scale).reshape(1)


def _go(group_lens):
    return F._i64_of(F._offsets(group_lens))


def f8_nn(aq, bq, sa, sb, group_lens, **kw):
    M, K = aq.shape
    G, _, N = bq.shape
    k = _nn8(M, N, K, G, **kw)
    (out,) = k.call(
        (aq, bq, sa, sb, _go(group_lens)), [jax.ShapeDtypeStruct((M, N), jnp.bfloat16)], zero_ret=1
    )
    return out


def f8_nt(aq, bq, sa, sb, group_lens, **kw):
    M, K = aq.shape
    G, N, _ = bq.shape
    k = _nt8(M, N, K, G, **kw)
    (out,) = k.call(
        (aq, bq, sa, sb, _go(group_lens)), [jax.ShapeDtypeStruct((M, N), jnp.bfloat16)], zero_ret=1
    )
    return out


WG_MAX_G = 64


def f8_wgrad(aq, dyq, sa, sdy, group_lens, **kw):
    M, K = aq.shape
    N = dyq.shape[1]
    G = group_lens.shape[0]
    offs = F._offsets(group_lens)
    if G <= WG_MAX_G:
        k = _wg8(K, N, G, **kw)
        (out,) = k.call(
            (aq, dyq, sa, sdy, F._i64_of(offs)), [jax.ShapeDtypeStruct((G * K, N), jnp.bfloat16)], zero_ret=1
        )
        return out.reshape(G, K, N)
    out = jnp.zeros((G * K, N), jnp.bfloat16)
    for c0 in range(0, G, WG_MAX_G):  # chunks of <= 64 groups into one aliased output (as in the bf16 bridge)
        gc = min(WG_MAX_G, G - c0)
        k = _wg8(K, N, gc, **kw)
        (out,) = k.call(
            (aq, dyq, sa, sdy, F._i64_of(offs[c0 : c0 + gc + 1]), out),
            [jax.ShapeDtypeStruct((G * K, N), jnp.bfloat16)],
            zero_ret=0,
            ptr_off=[0, 0, 0, 0, 0, 0, c0 * K * N * 2],
            buf_slot=list(k.buf_slot[:5]) + [-1] + [k.buf_slot[5]],
            aliases={5: 0},
        )
    return out.reshape(G, K, N)


def supported(a, b):
    """fp8 kernels: contraction dims K (fwd) and N (dA) multiples of 64 and > 128 (BLOCK_K=128 needs two k-iterations)."""
    K, N = a.shape[1], b.shape[2]
    return a.dtype == jnp.bfloat16 and K % 64 == 0 and K >= 192 and N % 64 == 0 and N >= 192


_warned = set()


def grouped_gemm_fp8_flydsl(a, b, group_lens):
    """FP8 grouped GEMM (bf16 in/out); shapes the fp8 kernels cannot handle use the bf16 FlyDSL path (with a warning)."""
    if supported(a, b):
        return _gg8(a, b, group_lens)
    key = (a.shape[1], b.shape[2], str(a.dtype))
    if key not in _warned:
        _warned.add(key)
        import warnings

        warnings.warn(
            f"grouped_gemm_fp8_flydsl: unsupported shape K={a.shape[1]} N={b.shape[2]} {a.dtype}; using the bf16 path"
        )
    return F.grouped_gemm_flydsl(a, b, group_lens)


@jax.custom_vjp
def _gg8(a, b, group_lens):
    return _fwd(a, b, group_lens)[0]


# PRIMUS_FP8_BWD: all (default: dA and dW in fp8) | dgrad (dA fp8, dW bf16) | bf16 (backward in bf16, only the forward in fp8)
_BWD = envs.PRIMUS_FP8_BWD


def _fwd(a, b, group_lens):
    rows = jax.lax.broadcasted_iota(jnp.int32, (a.shape[0], 1), 0)
    a = jnp.where(rows < jnp.sum(group_lens.astype(jnp.int32)), a, jnp.zeros_like(a))
    aq, sa = quantize(a)
    bq, sb = quantize(b)
    res = (aq, sa, bq, sb, group_lens) if _BWD == "all" else (aq, sa, bq, sb, group_lens, a, b)
    return f8_nn(aq, bq, sa, sb, group_lens), res


def _bwd(res, dy):
    aq, sa, bq, sb, group_lens = res[:5]
    rows = jax.lax.broadcasted_iota(jnp.int32, (dy.shape[0], 1), 0)
    dy = jnp.where(rows < jnp.sum(group_lens.astype(jnp.int32)), dy, jnp.zeros_like(dy))
    if _BWD == "all":
        dyq, sdy = quantize(dy)
        return f8_nt(dyq, bq, sdy, sb, group_lens), f8_wgrad(aq, dyq, sa, sdy, group_lens), None
    a, b = res[5], res[6]
    if _BWD == "dgrad":
        dyq, sdy = quantize(dy)
        return f8_nt(dyq, bq, sdy, sb, group_lens), F.gg_wgrad(a, dy, group_lens), None
    return F.gg_nt(dy, b, group_lens, zero_pad=True), F.gg_wgrad(a, dy, group_lens), None


_gg8.defvjp(_fwd, _bwd)
