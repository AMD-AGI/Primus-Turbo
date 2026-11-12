###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""JAX grouped GEMM (bf16) on the Primus-Turbo FlyDSL kernels (gfx950), callable inside jit/shard_map.

FlyDSL kernels are MLIR-JIT'd host launchers `void f(void **packed_args)`. We compile them once per shape (at trace
time, with tiny torch dummies, once per device so every GPU has its own loaded code object), snapshot the ABI slot
template, and call them from XLA through the generic C++ trampoline in flydsl_launch.cc, which patches in the XLA
buffer pointers and stream. Group sizes stay on the device; the per-tile (expert, row, count) table is built with jnp ops.

  grouped_gemm_flydsl(a[M,K], b[G,K,N], group_lens[G] int32) -> [M,N]   (rows past sum(group_lens) are zero)
with a custom VJP (dA via the NT kernel on the same weight tensor, dB via the variable-K kernel).
"""

import ctypes
import functools
import os
import subprocess
import threading

import jax
import jax.numpy as jnp
import numpy as np

import primus_turbo.common.envs as envs

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None
_LOCK = threading.RLock()
TARGET = "primus_flydsl_launch"


def _build_and_register():
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return
        src = os.path.join(_HERE, "flydsl_launch.cc")
        out_dir = envs.PRIMUS_TURBO_CACHE
        os.makedirs(out_dir, exist_ok=True)
        so = os.path.join(out_dir, "libflydsl_launch.so")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            tmp = so + f".{os.getpid()}"
            subprocess.check_call(
                [
                    "hipcc",
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-std=c++17",
                    f"-I{jax.ffi.include_dir()}",
                    src,
                    "-o",
                    tmp,
                ]
            )
            os.replace(tmp, so)
        _LIB = ctypes.cdll.LoadLibrary(so)
        jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(_LIB.FlyDSLLaunchHandler), platform="ROCM")


# --------------------------------------------------------------------------------------------------------------
# Compile helpers (torch + flydsl are only needed at trace time)
# --------------------------------------------------------------------------------------------------------------
def _flydsl():
    import flydsl.compiler as flyc
    import torch

    from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel as K

    return flyc, torch, K


def _snapshot(compiled, args, buf_arg_idx):
    """Return (fn_ptr, template_bytes, slot_off, buf_slot, stream_slot) for a flyc CompiledFunction."""
    from flydsl.compiler.jit_executor import build_abi_storage

    state = compiled._call_state
    spec = state._spec
    storages, _packed = build_abi_storage([ct for _i, ct, _f in spec])
    first_slot = {}
    for si, ((ai, ct, fill), st) in enumerate(zip(spec, storages)):
        if fill is not None:
            fill(args[ai], st)
        first_slot.setdefault(ai, si)
    offs, blob = [], bytearray()
    for st in storages:
        n = ctypes.sizeof(st)
        offs.append(len(blob))
        blob += ctypes.string_at(ctypes.addressof(st), n)
        blob += b"\0" * ((-len(blob)) % 8)
    stream_slot = first_slot.get(len(args) - 1, -1)
    buf_slot = [first_slot[i] for i in buf_arg_idx]
    fn = ctypes.cast(state._func_exe, ctypes.c_void_p).value
    return fn, bytes(blob), offs, buf_slot, stream_slot


class _Kernel:
    """Compiled launcher: one FlyDSL compile (dummy run on device 0), one loaded engine per device, ABI template."""

    def __init__(self, make_launch, make_args, buf_arg_idx, compile_fn=None):
        flyc, torch, _ = _flydsl()
        import sys

        # local ordinals only: with several hosts jax.devices() lists all processes' GPUs
        ndev = jax.local_device_count()
        sys.stderr.write(
            f"[flydsl_gmm] jax devices={ndev} torch device_count={torch.cuda.device_count()} "
            f"env={ {k: v for k, v in os.environ.items() if 'VISIBLE' in k} }\n"  # envs: allow
        )
        cur = torch.cuda.current_device()
        torch.cuda.set_device(0)
        launch = make_launch()
        args = make_args(0)
        compiled = compile_fn(launch, args) if compile_fn else flyc.compile(launch, *args)
        torch.cuda.synchronize()
        _fn0, self.blob, self.offs, self.buf_slot, self.stream_slot = _snapshot(compiled, args, buf_arg_idx)
        del args
        torch.cuda.empty_cache()
        art = compiled._keepalive
        self._keep = [compiled]
        self.fns = []
        for dev in range(ndev):
            # An ExecutionEngine loads the GPU code object on the device that is current while it initialises,
            # so every device gets its own engine built from the same compiled IR.
            torch.cuda.set_device(dev)
            a2 = object.__new__(type(art))
            a2.__dict__.update(art.__dict__)
            a2._engine = a2._func_exe = a2._module = a2._jit_module = a2._ctx = None
            a2._lock = threading.Lock()
            fn = a2._get_func_exe()
            self._keep.append(a2)
            self.fns.append(ctypes.cast(fn, ctypes.c_void_p).value)
        torch.cuda.set_device(cur)

    def call(self, operands, out_shapes, zero_ret=0, ptr_off=(), buf_slot=None, aliases=None):
        _build_and_register()
        call = jax.ffi.ffi_call(
            TARGET, out_shapes, vmap_method="sequential", input_output_aliases=aliases or {}
        )
        return call(
            *operands,
            fns=np.asarray(self.fns, np.int64),
            tmpl=np.frombuffer(self.blob, np.uint8).copy(),
            slot_off=np.asarray(self.offs, np.int32),
            buf_slot=np.asarray(self.buf_slot if buf_slot is None else buf_slot, np.int32),
            ptr_off=np.asarray(list(ptr_off) if len(ptr_off) else [0], np.int64),
            stream_slot=np.int32(self.stream_slot),
            zero_ret=np.int32(zero_ret),
        )


def _i32_view(torch, n_elems_bf16, dev):
    return torch.zeros(max(n_elems_bf16 // 2, 2), dtype=torch.int32, device=f"cuda:{dev}")


@functools.lru_cache(maxsize=None)
def _nt_kernel(m_upper, N, K, G, BM=256, BN=256, GROUP_M=4, num_xcd=8, xcd_band=32, cap_cu=0):
    flyc, torch, KM = _flydsl()

    def make_launch():
        return KM._compile_grouped_bf16_nt(
            m_upper,
            N,
            K,
            G,
            BLOCK_M=BM,
            BLOCK_N=BN,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
            ab_ty=KM._ab_ty(torch.bfloat16),
        )

    def make_args(dev):
        z = lambda n, dt=torch.bfloat16: torch.zeros(n, dtype=dt, device=f"cuda:{dev}")
        return (
            _i32_view(torch, 4096, dev),
            flyc.from_torch_tensor(z(G * N * K)),
            flyc.from_torch_tensor(torch.zeros(256, N, dtype=torch.bfloat16, device=f"cuda:{dev}")),
            z(3 * m_upper, torch.int32),
            N,
            torch.cuda.current_stream(),
        )

    # args order: (A, B, C, m_tiles, N, stream) ; XLA buffers: a, b, m_tiles | out
    return _Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 2])


@functools.lru_cache(maxsize=None)
def _nn_kernel(m_upper, N, K, G, BM=256, BN=256, GROUP_M=0, num_xcd=8, xcd_band=32, cap_cu=0):
    flyc, torch, KM = _flydsl()
    if GROUP_M == 0:
        GROUP_M = 8 if K * N * 2 > 24 << 20 else 4

    def make_launch():
        return KM._compile_grouped_bf16_nn(
            m_upper,
            N,
            K,
            G,
            BLOCK_M=BM,
            BLOCK_N=BN,
            GROUP_M=GROUP_M,
            num_xcd=num_xcd,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
            ab_ty=KM._ab_ty(torch.bfloat16),
        )

    def make_args(dev):
        z = lambda n, dt=torch.bfloat16: torch.zeros(n, dtype=dt, device=f"cuda:{dev}")
        return (
            _i32_view(torch, 4096, dev),
            flyc.from_torch_tensor(z(G * N * K)),
            flyc.from_torch_tensor(torch.zeros(256, N, dtype=torch.bfloat16, device=f"cuda:{dev}")),
            z(3 * m_upper, torch.int32),
            N,
            torch.cuda.current_stream(),
        )

    return _Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 2])


@functools.lru_cache(maxsize=None)
def _wgrad_kernel(OUT_M, OUT_N, G, BM=256, BN=256, num_xcd=8, group_m=4, xcd_band=32, cap_cu=0):
    flyc, torch, KM = _flydsl()

    def make_launch():
        return KM._compile_grouped_bf16_wgrad(
            OUT_M,
            OUT_N,
            G,
            BLOCK_M=BM,
            BLOCK_N=BN,
            num_xcd=num_xcd,
            ab_ty=KM._ab_ty(torch.bfloat16),
            out_fp16=False,
            trans_c=False,
            group_m=group_m,
            xcd_band=xcd_band,
            cap_cu=cap_cu,
        )

    def make_args(dev):
        z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=f"cuda:{dev}")
        return (
            _i32_view(torch, 4096, dev),
            _i32_view(torch, 4096, dev),
            flyc.from_torch_tensor(z(G, OUT_M, OUT_N)),
            z(G + 1, dt=torch.int64),
            z(G, dt=torch.int64),
            OUT_M,
            OUT_N,
            torch.cuda.current_stream(),
        )

    # args: (A, B, out, offsets, masked_k, OUT_M, OUT_N, stream); XLA buffers: a, b, offsets, masked_k | out
    return _Kernel(make_launch, make_args, buf_arg_idx=[0, 1, 3, 4, 2])


# --------------------------------------------------------------------------------------------------------------
# jnp helpers and ops
# --------------------------------------------------------------------------------------------------------------
def _offsets(group_lens):
    lens = group_lens.astype(jnp.int32)
    return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(lens)])


def _m_tiles(group_lens, block_m, upper):
    lens = group_lens.astype(jnp.int32)
    offs = _offsets(lens)[:-1]
    G = lens.shape[0]
    blocks = (lens + (block_m - 1)) // block_m
    first = jnp.cumsum(blocks) - blocks
    total = first[-1] + blocks[-1]
    idx = jnp.arange(upper, dtype=jnp.int32)
    g = jnp.clip(jnp.searchsorted(first, idx, side="right") - 1, 0, G - 1).astype(jnp.int32)
    local = idx - first[g]
    live = idx < total
    rows = jnp.where(live, jnp.clip(lens[g] - local * block_m, 0, block_m), 0)
    start = jnp.where(live, offs[g] + local * block_m, 0)
    return jnp.stack([g, start, rows], axis=1).astype(jnp.int32).reshape(-1)


def _upper(M, block_m, G):
    return (M + block_m - 1) // block_m + G


# Ring-of-experts EP hands the kernels a buffer of M = ep x (local rows) rows of which only ~M/ep are valid: the grid is sized by M, and the
# ~(M/BM) dead tiles cost ~3x the useful work (profile: 4 ms vs 0.95 ms per call at M=2.1M). PRIMUS_FLYDSL_ROWS_CAP_DIV=d launches a grid sized for
# M/d rows whenever sum(group_lens) <= M/d (exact), and the full grid otherwise (lax.cond on the device value), so dropless semantics are unchanged.
ROWS_CAP_DIV = envs.PRIMUS_FLYDSL_ROWS_CAP_DIV


def _capped(M, BM, G, group_lens, run):
    """run(up) -> out for a tile-grid upper bound `up`; picks the small grid when the live rows fit."""
    up = _upper(M, BM, G)
    cap_rows = (M // ROWS_CAP_DIV // BM) * BM if ROWS_CAP_DIV > 1 else 0
    if cap_rows < 4 * BM or _upper(cap_rows, BM, G) >= up:
        return run(up)
    up_small = _upper(cap_rows, BM, G)
    fits = jnp.sum(group_lens.astype(jnp.int32)) <= cap_rows
    return jax.lax.cond(fits, lambda: run(up_small), lambda: run(up))


def _i64_of(x32):
    """int32 vector -> the same values as little-endian int64, passed as int32 pairs (no global x64 needed)."""
    return jnp.stack([x32.astype(jnp.int32), jnp.zeros_like(x32, jnp.int32)], axis=-1).reshape(-1)


def _cfg(name, **kw):
    """Tile configuration overrides: env PRIMUS_FLYDSL_<NAME>='BM=128,GROUP_M=2' (applies to every shape)."""
    env = getattr(envs, f"PRIMUS_FLYDSL_{name}")
    for item in filter(None, env.split(",")):
        k, v = item.split("=")
        kw.setdefault(k, int(v))
    return kw


def _zero_pad(kw):
    z = kw.pop("zero_pad", None)
    if z is None:
        z = envs.PRIMUS_FLYDSL_ZERO_PAD
    return int(bool(z))


def gg_nn(a, b, group_lens, **kw):
    """out[rows] = a[rows] @ b[g]; a[M,K], b[G,K,N]."""
    M, K = a.shape
    G, _, N = b.shape
    zero = _zero_pad(kw)
    kw = _cfg("NN", **kw)
    BM = kw.get("BM", 256)

    def run(up):
        k = _nn_kernel(up, N, K, G, **kw)
        (out,) = k.call(
            (a, b, _m_tiles(group_lens, BM, up)), [jax.ShapeDtypeStruct((M, N), a.dtype)], zero_ret=zero
        )
        return out

    return _capped(M, BM, G, group_lens, run)


def gg_nt(a, b, group_lens, **kw):
    """out[rows] = a[rows] @ b[g]^T; a[M,K], b[G,N,K]."""
    M, K = a.shape
    G, N, _ = b.shape
    zero = _zero_pad(kw)
    kw = _cfg("NT", **kw)
    BM = kw.get("BM", 256)

    def run(up):
        k = _nt_kernel(up, N, K, G, **kw)
        (out,) = k.call(
            (a, b, _m_tiles(group_lens, BM, up)), [jax.ShapeDtypeStruct((M, N), a.dtype)], zero_ret=zero
        )
        return out

    return _capped(M, BM, G, group_lens, run)


# The variable-K kernel ranks groups with one 64-lane wave: groups >= 64 are mis-assigned (verified wrong from G=65).
# 64 is correct; larger values reproduce the old wrong-gradient bug, for sensitivity tests.
WG_MAX_G = envs.PRIMUS_FLYDSL_WG_MAX_G


def gg_wgrad(a, dy, group_lens, **kw):
    """out[g] = a[rows_g]^T @ dy[rows_g]; a[M,K], dy[M,N] -> [G,K,N]. Chunks of <= 64 groups per launch."""
    M, K = a.shape
    N = dy.shape[1]
    G = group_lens.shape[0]
    kw = _cfg("WG", **kw)
    offs = _offsets(group_lens)
    lens32 = group_lens.astype(jnp.int32)
    if G <= WG_MAX_G:
        k = _wgrad_kernel(K, N, G, **kw)
        (out,) = k.call(
            (a, dy, _i64_of(offs), _i64_of(lens32)), [jax.ShapeDtypeStruct((G, K, N), a.dtype)], zero_ret=1
        )
        return out
    out = jnp.zeros((G, K, N), a.dtype)
    for c0 in range(0, G, WG_MAX_G):
        gc = min(WG_MAX_G, G - c0)
        k = _wgrad_kernel(K, N, gc, **kw)
        # buffers: a, dy, offsets, masked_k, out_in (aliased, not passed to the kernel) | out
        (out,) = k.call(
            (a, dy, _i64_of(offs[c0 : c0 + gc]), _i64_of(lens32[c0 : c0 + gc]), out),
            [jax.ShapeDtypeStruct((G, K, N), a.dtype)],
            zero_ret=0,
            ptr_off=[0, 0, 0, 0, 0, c0 * K * N * a.dtype.itemsize],
            buf_slot=list(k.buf_slot[:4]) + [-1] + [k.buf_slot[4]],
            aliases={4: 0},
        )
    return out


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _gg_flydsl(a, b, group_lens, zero_pad=True):
    """zero_pad=False skips zeroing the output rows past sum(group_lens); only valid when those rows are unused
    (e.g. sum(group_lens) == M, which holds for dropless MoE without expert parallelism)."""
    return gg_nn(a, b, group_lens, zero_pad=zero_pad)


def _fwd(a, b, group_lens, zero_pad):
    return gg_nn(a, b, group_lens, zero_pad=zero_pad), (a, b, group_lens)


def _bwd(zero_pad, res, g):
    a, b, group_lens = res
    return gg_nt(g, b, group_lens, zero_pad=zero_pad), gg_wgrad(a, g, group_lens), None


_gg_flydsl.defvjp(_fwd, _bwd)


def supported(a, b):
    """The FlyDSL kernels need the contraction dims of every pass to be multiples of 64 and >= 128: K (fwd, dW rows are free)
    and N (dA contracts over N). Anything else would fail an assert while tracing."""
    K, N = a.shape[1], b.shape[2]
    return (
        a.dtype == jnp.bfloat16
        and b.dtype == jnp.bfloat16
        and K % 64 == 0
        and K >= 128
        and N % 64 == 0
        and N >= 128
    )


_warned = set()


def grouped_gemm_flydsl(a, b, group_lens, zero_pad=True):
    """Grouped GEMM out[rows_g] = a[rows_g] @ b[g] (bf16, rows sorted by group, device-side group sizes) with autodiff.
    Shapes the FlyDSL kernels cannot handle fall back to jax.lax.ragged_dot (with a warning)."""
    # Skipping the output memset (~2% of a Mixtral step) is only valid when sum(group_lens) == M, i.e. no padding rows are left
    # unwritten (checked with a NaN-poisoned allocator: test_nozero_poison.py). It is off by default; PRIMUS_FLYDSL_ALLOW_NOZERO=1
    # honours zero_pad=False (MaxText passes it for dropless runs without expert parallelism).
    if not envs.PRIMUS_FLYDSL_ALLOW_NOZERO:
        zero_pad = True
    if supported(a, b):
        return _gg_flydsl(a, b, group_lens, zero_pad)
    key = (a.shape[1], b.shape[2], str(a.dtype))
    if key not in _warned:
        _warned.add(key)
        import warnings

        warnings.warn(
            f"grouped_gemm_flydsl: unsupported shape/dtype K={a.shape[1]} N={b.shape[2]} {a.dtype}; using jax.lax.ragged_dot"
        )
    gl = group_lens.astype(jnp.int32)
    out = jax.lax.ragged_dot(a, b, gl)
    rows = jax.lax.broadcasted_iota(jnp.int32, (a.shape[0], 1), 0)
    return jnp.where(rows < jnp.sum(gl), out, jnp.zeros_like(out))
