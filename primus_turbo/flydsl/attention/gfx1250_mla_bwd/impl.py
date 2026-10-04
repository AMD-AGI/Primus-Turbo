"""flydsl_attn_bwd: the gfx1250 FlyDSL backward (k_delta, k_dkdv64, k_dqg96 + k_dqg in kernels.py)."""
import importlib.util as _ilu
import pathlib as _pl

_HERE = _pl.Path(__file__).resolve().parent


def _sibling(stem):
    """Import a module from THIS directory under a name unique to this directory.

    A plain `import kernels` binds `sys.modules["kernels"]`, so loading a second
    implementation in the same process would silently reuse the first one's module -- and
    with it the first one's JIT-compiled kernels. The directory is the identity, so the
    module name has to carry it.
    """
    name = f"{stem}__{abs(hash(str(_HERE)))}"
    spec = _ilu.spec_from_file_location(name, _HERE / f"{stem}.py")
    mod = _ilu.module_from_spec(spec)
    import sys as _sys
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_env = _sibling("_env")

import math
import os
import sys

import torch

_k = _sibling("kernels")

import flydsl.compiler as _flyc  # kernels.py has already put the pinned flydsl on sys.path

# FlyDSL's @flyc.jit __call__ re-derives the whole cache key on every launch (~0.27 ms of
# host time per call). flyc.compile(launcher, *args) performs the first launch and returns
# a CompiledFunction whose __call__ only refreshes the ctypes argument storage. Every
# shape-dependent argument is a runtime fx.Int32, so one compiled object serves every
# shape; the memo key (dtype and rank of each tensor, device) keeps a different dtype or
# layout from reusing it.
_COMPILED = {}


def _launch(name, launcher, args):
    """Launch `launcher(*args)`, via flyc.compile's fast path after the first call."""
    # id(launcher): one plan name may map to different launchers by shape (k_dqg96 vs k_dqg).
    key = (name, id(launcher),
           tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor)),
           args[0].device.index)
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    # flyc.compile() ISSUES this launch itself, so it must not be repeated here.
    fn = _flyc.compile(launcher, *args)
    if fn is None:                      # COMPILE_ONLY builds return None
        launcher(*args)
        return
    _COMPILED[key] = fn


_ENV_CHECKED = False

# The dQ chain (k_dqg) runs on the caller's stream AFTER the dK/dV chain (k_dkdv) by default.
# It may instead run on a SIDE stream concurrently with k_dkdv: both only read
# q/k/v/do/lse/delta (delta is written by k_delta on the main stream BEFORE the fork) and
# write disjoint outputs, so the fork is legal and the outputs are bitwise identical to the
# serial order. For MHA at DeepSeek-V3 shapes each kernel alone fills every SIMD (prod: 32768
# one-wave workgroups each), and running the two concurrently was 28% SLOWER than serial at
# b2 s4096 h128 (7.06 vs 5.52 ms) and no faster at b1 s4096 h64; the side stream paid off
# only for the hd128 GQA shape it was built for. Env switches, read once at import:
#   FLY_BWD_SIDE_STREAM=1    side stream: k_dqg concurrently with k_dkdv.
#   FLY_BWD_RECORD_STREAM=0  with the side stream, skip the record_stream() calls at the
#                            join (every block s2 touches was allocated on the caller's
#                            stream, which waits for s2 before we return).
DQ_SIDE_STREAM = os.environ.get("FLY_BWD_SIDE_STREAM", "0") != "0"
DQ_SIDE_RECORD = os.environ.get("FLY_BWD_RECORD_STREAM", "1") != "0"
_SIDE = {}
# Validation only: allocate delta/dq/dk/dv NaN-filled, so an element the kernels never write
# shows up as non-finite instead of as stale memory.
POISON = False


def _alloc(shape, device, dtype):
    if POISON:
        return torch.full(shape, float("nan"), device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype)


def _check_env_once():
    global _ENV_CHECKED
    if not _ENV_CHECKED:
        _env.assert_environment()
        print(f"[flydsl_bwd {_HERE.name}] DQ_SIDE_STREAM={int(DQ_SIDE_STREAM)} "
              f"DQ_SIDE_RECORD={int(DQ_SIDE_RECORD)}", file=sys.stderr, flush=True)
        _ENV_CHECKED = True


def _side_stream(dev):
    s = _SIDE.get(dev)
    if s is None:
        s = torch.cuda.Stream(device=dev)
        _SIDE[dev] = s
    return s


def _check(do, q, k, v, o, lse):
    """Shape/dtype/layout contract of the kernels; returns (b, sq, skv, hq, hkv)."""
    b, sq, hq, dqk = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    dv = v.shape[-1]
    assert (dqk, dv) == (_k.D_QK, _k.D_V), (
        f"these kernels are head dims (qk {_k.D_QK}, v {_k.D_V}) only, got ({dqk}, {dv})")
    assert k.shape == (b, skv, hkv, dqk) and v.shape == (b, skv, hkv, dv), (
        q.shape, k.shape, v.shape)
    assert o.shape == (b, sq, hq, dv) and do.shape == o.shape, (q.shape, o.shape, do.shape)
    assert lse.shape == (b, hq, sq), f"lse must be [B, Hq, Sq], got {tuple(lse.shape)}"
    assert hq % hkv == 0, f"heads_q {hq} is not a multiple of heads_kv {hkv}"
    # k_dkdv consumes query PAIRS of 32 rows; k_dqg tiles [0, q_split) by DQ_BQW and k_dqg96
    # [q_split, sq) by DQ_BQW96 (kernels.dq_split): a remainder would leave rows uncomputed and
    # write past the end.
    assert sq % 64 == 0 and sq % _k.DQ_BQW == 0, f"seqlen_q must be a multiple of 64, got {sq}"
    # k_dkdv64 consumes DKDV_NW*BLOCK_KV = 64-row kv blocks (one 32-row tile per wave).
    assert skv % _k.KV_STEP == 0 and skv % (_k.BLOCK_KV * _k.DKDV_NW) == 0, (
        f"seqlen_kv must be a multiple of {_k.BLOCK_KV * _k.DKDV_NW}, got {skv}")
    n_rows = b * sq * hq
    assert n_rows % _k.ROWS_DELTA == 0, (
        f"batch*seqlen_q*heads_q must be a multiple of {_k.ROWS_DELTA}, got {n_rows}")
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
    # Byte extents the kernels compute in 32-bit arithmetic or hard-code as descriptor
    # num_records (k_dqg: 1 GiB for q/do/dq, 256 MiB for lse/delta).
    assert max(q.numel(), o.numel()) * 2 <= (1 << 30), (
        "q/do/dq larger than k_dqg's 1 GiB descriptor extent")
    assert lse.numel() * 4 <= (1 << 28), "lse/delta larger than k_dqg's 256 MiB descriptor extent"
    assert max(k.numel(), v.numel()) * 2 < (1 << 31), (
        "k/v byte extent overflows k_dkdv's int32 descriptor size")
    assert n_rows * _k.D_V * 2 < (1 << 31), "o/do byte extent overflows k_delta's int32 size"
    return b, sq, skv, hq, hkv


def _plan(do, q, k, v, o, lse, softmax_scale, causal, stream, dq_stream=None):
    """Every launch flydsl_attn_bwd makes for these inputs, in issue order, plus the
    tensors it allocates. Used verbatim by the launcher below and by the compile-only gate
    (tools/flydsl/drivers/bwd_mla.py, meta tensors, stream=None), so the compiled set is
    exactly the launched set.

    launches: [(name, launcher, args, chain)], chain "main" (caller's stream) or "dq".
    """
    b, sq, skv, hq, hkv = _check(do, q, k, v, o, lse)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(_k.D_QK)
    g = hq // hkv
    n_rows = b * sq * hq
    c = int(bool(causal))
    dq_stream = stream if dq_stream is None else dq_stream
    delta = _alloc((b, hq, sq), q.device, torch.float32)
    dq = _alloc((b, sq, hq, _k.D_QK), q.device, q.dtype)
    dk = _alloc((b, skv, hkv, _k.D_QK), k.device, k.dtype)
    dv = _alloc((b, skv, hkv, _k.D_V), v.device, v.dtype)
    launches = [
        ("delta", _k.launch_delta,
         (do, o, delta, sq, hq, n_rows, n_rows // _k.ROWS_DELTA, stream), "main"),
        # k_dkdv64: 2 waves per 64-row kv block share one Q/dO ring (bwd_r2_a)
        ("dkdv", _k.launch_dkdv64,
         (q, k, v, do, lse, delta, dv, dk, float(softmax_scale),
          sq, skv, hq, hkv, g, sq // 16, skv - sq, c,
          skv // (_k.BLOCK_KV * _k.DKDV_NW), hkv, b, stream), "main"),
    ]
    grids = {"delta": (n_rows // _k.ROWS_DELTA, 1, 1),
             "dkdv": (hkv, skv // (_k.BLOCK_KV * _k.DKDV_NW), b)}
    # dQ chain: k_dqg96 (2 waves x 48 queries on one K/V ring, bwd_r2_b) over [q_split, sq),
    # longest-first, then the 32-query k_dqg over the head [0, q_split) (the shortest tiles). Plan
    # name "dqg" is the main dQ launch (k_dqg96; k_dqg when there is no 96-query part), "dqg_head"
    # the head launch. Both are single-kernel modules, so the compile-only dump/ISA table sees each
    # kernel on its own.
    q_split, n32, n96 = _k.dq_split(sq)
    assert q_split + n96 * _k.DQ_BQW96 == sq and n32 * _k.DQ_BQW == q_split, (sq, q_split, n32, n96)
    dq_args = (q, k, v, do, o, lse, delta, dq, float(softmax_scale),
               sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, c)
    if n96:
        launches.append(("dqg", _k.launch_dqg96, dq_args + (q_split, n96, hq, b, dq_stream), "dq"))
        grids["dqg"] = (hq, n96, b)
    if n32:
        nm = "dqg_head" if n96 else "dqg"
        launches.append((nm, _k.launch_dqg, dq_args + (0, n32, hq, b, dq_stream), "dq"))
        grids[nm] = (hq, n32, b)
    return {"launches": launches, "outputs": (dq, dk, dv), "delta": delta, "grids": grids,
            "dq_split": (q_split, n32, n96)}


def flydsl_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Flash-attention backward on gfx1250.

    q [B, Sq, Hq, D_QK], o/do [B, Sq, Hq, D_V], k [B, Skv, Hkv, D_QK], v [B, Skv, Hkv, D_V],
    all bf16 and contiguous (DeepSeek-V3 MLA: D_QK = 192, D_V = 128).
    lse     [B, Hq, Sq] fp32, NATURAL log, as the gfx1250 forward emits it.
    Returns (dq, dk, dv) in q/k/v's dtype, laid out like q/k/v.

    causal is BOTTOM-RIGHT: query i attends keys j <= i + (Skv - Sq).
    """
    _check_env_once()
    lse = lse.contiguous().float()
    stream = torch.cuda.current_stream()
    split = DQ_SIDE_STREAM
    s2 = _side_stream(q.device) if split else stream
    plan = _plan(do, q, k, v, o, lse, softmax_scale, causal, stream, s2)
    for name, fn, args, chain in plan["launches"]:
        _launch(name, fn, args)
        if name == "delta" and split:
            # fork: the dQ chain waits for everything already on the main stream (inputs,
            # lse.float(), k_delta) but not for k_dkdv, which is issued next.
            s2.wait_stream(stream)
    dq, dk, dv = plan["outputs"]
    if split:
        if DQ_SIDE_RECORD:
            for t in (q, k, v, do, o, lse, plan["delta"], dq):
                t.record_stream(s2)
        stream.wait_stream(s2)
    return dq, dk, dv


attn_bwd = flydsl_attn_bwd   # uniform name every loader uses
