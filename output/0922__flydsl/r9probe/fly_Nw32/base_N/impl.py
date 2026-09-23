"""op.reference.api, backed by the three gfx1250 FlyDSL kernels in kernels.py."""
import importlib.util as _ilu
import pathlib as _pl

_HERE = _pl.Path(__file__).resolve().parent


def _sibling(stem):
    """Import a module from THIS directory under a name unique to this directory.

    A plain `import kernels` binds `sys.modules["kernels"]`, so loading a second
    implementation in the same process silently reuses the first one's module -- and with
    it the first one's JIT-compiled kernels. The two arms then differ by under 0.05% and
    produce identical output, which is exactly what a real result looks like. The
    directory is the identity, so the module name has to carry it.
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

import torch

_k = _sibling("kernels")

import flydsl.compiler as _flyc   # r5.i1.g17; kernels.py has already put it on sys.path

_ENV_CHECKED = False

# r5.i1.g17 -- FlyDSL's @flyc.jit __call__ re-derives the whole cache key on EVERY launch:
# inspect.Signature.bind, a getattr_static/typing.instancecheck pass over every argument,
# a globals-drift scan and a re-read of the cache-invalidating env vars. Measured on this
# box (`_scratch/work/hostcost.py`, `hostprof.py`): a FLAT 0.266 ms of CPU per call at all
# three shapes -- 51.3% of the `fast` shape's whole measured latency, 14.0% of proxy, 1.6%
# of prod -- with 68% of it inside `_resolve_and_make_cache_key`.
#
# `flyc.compile(launcher, *args)` is FlyDSL's own documented fast path for exactly this:
# it performs the first launch, then returns a CompiledFunction whose __call__ does only
# "update pre-allocated ctypes storage (data_ptr / scalar extraction), invoke the JIT'd C
# function pointer -- no Signature.bind, no _resolve_and_make_cache_key, no cache lookup",
# quoted ~5 us. Constexpr arguments are baked; every argument these three launchers take
# that varies with the shape (Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, the three grid
# dimensions) is a runtime fx.Int32, so one compiled object serves every shape.
#
# The memo key is conservative anyway: dtype and rank of each tensor plus the device, so a
# different dtype or layout can never reuse a compiled object. It is a pure host-side
# change -- not one instruction of GPU code differs, and the outputs are bitwise identical.
_COMPILED = {}


def _launch(name, launcher, args):
    """Launch `launcher(*args)`, via flyc.compile's fast path after the first call."""
    key = (name, tuple((a.dtype, a.dim()) for a in args if isinstance(a, torch.Tensor)),
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


def _check_env_once():
    global _ENV_CHECKED
    if not _ENV_CHECKED:
        _env.assert_environment()
        _ENV_CHECKED = True


def flydsl_attn_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """Grouped-query flash-attention backward on gfx1250.

    q/o/do  [B, Sq, Hq, D] bf16      k/v  [B, Skv, Hkv, D] bf16
    lse     [B, Hq, Sq] fp32, NATURAL log, as aiter's gfx1250 forward emits it.
    Returns (dq, dk, dv) in q/k/v's dtype, laid out like q/k/v.

    causal is BOTTOM-RIGHT: query i attends keys j <= i + (Skv - Sq).
    """
    _check_env_once()
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    assert d == _k.D, f"this kernel is head_dim {_k.D} only, got {d}"
    assert hq % hkv == 0, f"heads_q {hq} is not a multiple of heads_kv {hkv}"
    assert sq % 32 == 0, (
        f"seqlen_q must be a multiple of 32 (k_dkdv consumes query tiles in pairs), "
        f"got {sq}")
    assert skv % _k.KV_STEP == 0, (
        f"seqlen_kv must be a multiple of {_k.KV_STEP}, got {skv}")
    assert skv % _k.BLOCK_KV == 0, (
        f"seqlen_kv must be a multiple of {_k.BLOCK_KV}, got {skv}")
    n_rows = b * sq * hq
    assert n_rows % _k.ROWS_DELTA == 0, (
        f"batch*seqlen_q*heads_q must be a multiple of {_k.ROWS_DELTA}, got {n_rows}")
    for name, t in (("do", do), ("q", q), ("k", k), ("v", v), ("o", o)):
        assert t.is_contiguous(), f"{name} must be contiguous"
        assert t.dtype == torch.bfloat16, f"{name} must be bf16, got {t.dtype}"
    lse = lse.contiguous().float()

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    g = hq // hkv
    stream = torch.cuda.current_stream()

    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    _launch("delta", _k.launch_delta,
            (do, o, delta, sq, hq, n_rows, n_rows // _k.ROWS_DELTA, stream))

    # r3.i2.g11 -- the kernels write bf16 straight out. Three `.to()` kernels and three
    # fp32 temporaries are gone; the accumulators are still fp32 inside the kernel.
    dk_o = torch.empty((b, skv, hkv, d), device=k.device, dtype=k.dtype)
    dv_o = torch.empty_like(dk_o)
    _launch("dkdv", _k.launch_dkdv,
            (q, k, v, do, lse, delta, dv_o, dk_o, float(softmax_scale),
             sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
             skv // _k.BLOCK_KV, hkv, b, stream))

    dq_o = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
    _launch("dq", _k.launch_dq,
            (q, k, v, do, lse, delta, dq_o, float(softmax_scale),
             sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
             sq // _k.BLOCK_Q, hq, b, stream))

    return dq_o, dk_o, dv_o


attn_bwd = flydsl_attn_bwd   # uniform name every loader uses
