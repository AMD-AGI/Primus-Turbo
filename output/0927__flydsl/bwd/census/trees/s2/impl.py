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


# r17.i1.g52 -- the nsp_q sweep constant. The shipped rule is `while wgs*nsp < 2048 and
# nsp < 16`, which is where the dkdv census's return vanishes. It is fitted on k_dkdv,
# whose post-split `fast` body still runs 4.1 loop trips; k_dq at fast runs 17.0, so at
# nsp_q = 16 it would run 1.06 trips per split -- the rule extrapolated outside its
# fitted range. The cap is therefore SWEPT on card over {4, 8, 16} rather than assumed,
# and the winner must be an interior point or the range is unconverged.
# MEASURED, round 17, rounds/017/_scratch/meas2 -- three same-session replica blocks,
# each a different arm-slot permutation, two independent copies of each arm, `fast`:
#     cap  4 -> 52.27 TF/s (n=6)    cap  8 -> 53.69 TF/s (n=6)    cap 16 -> 46.40 (n=3)
#     same-code control cur/ctrl -> 37.89 / 38.81, i.e. a 2.4% floor between identical
#     trees, so the 4-vs-8 gap is at the edge of it and the 8-vs-16 gap is far outside.
# The optimum is the INTERIOR point 8, so the {4, 8, 16} range is converged and the
# constant is shipped from a converged sweep rather than from the dkdv rule's cap.
# Why 16 loses while the dkdv census says 16: at nsp_q = 16 k_dq's 17.0 loop trips become
# 1.06 per split, so the per-workgroup prologue (Q/dO/lse/delta) and the fp32 partial
# store stop being amortised at all, and the fold reads 16 x dQ instead of 8 x.
_NSP_Q_CAP = 8


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
    # r13.i1.g42 -- SPLIT-K OVER k_dkdv's q LOOP, switched on by the GRID SIZE.
    # k_dkdv's workgroup count is (Skv/BLOCK_KV)*Hkv*B and one workgroup is one wave32,
    # so at 1 wave/SIMD (g38) the device holds 256*4 = 1024 of them at once. prod
    # launches 8192 -- eight dispatch waves, and the greedy hardware dispatcher balances
    # the 256:1 causal work skew to a modelled 100% (raw/census.txt). proxy launches
    # EXACTLY 1024: one wave, every workgroup resident from t=0, so the kernel ends when
    # its LONGEST workgroup ends and the modelled efficiency is 50.4%. fast launches 64
    # onto 1024 slots: 3.2%. Dispatch order cannot touch a one-wave grid; only more
    # workgroups can. So split the q loop `nsp` ways into an fp32 workspace and fold it
    # on the host -- the technique `flydsl/attention/techniques.md:363-374` measures at
    # "+18% here", with its own warning that it is worthless when the inner range is
    # already short, which is exactly why prod keeps nsp = 1 and the shipped bf16 path.
    _wgs = (skv // _k.BLOCK_KV) * hkv * b
    # r13.i1.g42(a) -- round 14: nsp is now DERIVED, not guessed. Round 13 picked it by
    # "round up to about 2048 workgroups, cap 8"; the split-K census
    # (rounds/014/1-opt/raw/census_sp.txt, zero card time, greedy list scheduling in the
    # REAL dispatch order grid=(nhkv*nsp, nblk, nb)) says the modelled optimum is
    # fast 16 / proxy 2 / prod 1, and the shipped cap of 8 put `fast` EXACTLY on the
    # endpoint of its own range -- which the corpus flags as the signature of a sweep
    # that has not converged (flydsl/attention/dead-ends.md:262-271, reopening such a
    # range was worth +18% there). Two dispatch waves is what the census actually wants:
    # one wave makes the makespan equal to the LONGEST workgroup (proxy nsp=1: 512 units
    # against a 258 lower bound, 50.4%), two waves let the greedy dispatcher balance the
    # 128:1 causal skew (proxy nsp=2: 264 units, 97.7%). The 16 cap is where the census's
    # own return vanishes and matches aiter's split_K = max(2, min(num_cu/tg, 16, ...)).
    nsp = 1
    # r23 -- 2048 encoded TWO dispatch waves of SINGLE-wave workgroups (residency
    # 1024). k_dkdv's workgroup is now FOUR waves, so residency is 256 and the
    # target divides by 4. Verified to reproduce today's nsp at all three scored
    # shapes: fast 16, proxy 2, prod 1. impl.py's k_dq rule below is NOT touched.
    while _wgs * nsp < 512 and nsp < 16:
        nsp *= 2
    if nsp == 1:
        dk_o = torch.empty((b, skv, hkv, d), device=k.device, dtype=k.dtype)
        dv_o = torch.empty_like(dk_o)
        _launch("dkdv", _k.launch_dkdv,
                (q, k, v, do, lse, delta, dv_o, dk_o, float(softmax_scale),
                 sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
                 skv // _k.BLOCK_KV, hkv, b, stream))
    else:
        dkp = torch.empty((nsp, b, skv, hkv, d), device=k.device, dtype=torch.float32)
        dvp = torch.empty_like(dkp)
        _launch("dkdv_sp", _k.launch_dkdv_sp,
                (q, k, v, do, lse, delta, dvp, dkp, float(softmax_scale),
                 sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
                 skv // _k.BLOCK_KV, hkv, b, nsp, hkv * nsp, stream))
        # r13.i1.g42(c) -- round 14: the fold-back is now ONE device kernel over BOTH
        # tensors writing bf16 straight out, replacing four torch launches
        # (sum -> fp32 temp -> .to(bf16), twice) that round 14 measured at 18.4% of the
        # `fast` shape and 6.4% of proxy, almost flat in nsp -- i.e. launch-and-latency
        # bound at 1.19 TB/s, not bandwidth bound. See kernels.py:k_redsp.
        # Same fixed ascending `sp` order as torch.sum(0), so determinism is unchanged
        # and the result is intended to be bitwise identical to the shipped path.
        dk_o = torch.empty((b, skv, hkv, d), device=k.device, dtype=k.dtype)
        dv_o = torch.empty_like(dk_o)
        _n_vec = (b * skv * hkv * d) // _k.RED_VEC
        _rblk = (_n_vec + _k.RED_THREADS - 1) // _k.RED_THREADS
        _launch("redsp", _k.launch_redsp,
                (dkp, dvp, dk_o, dv_o, _n_vec, nsp, _rblk, stream))

    # r17.i1.g52 -- SPLIT-K OVER k_dq's kv LOOP, derived from k_dq's OWN grid.
    # The `nsp` above is derived from the dkdv grid and was never applied to k_dq, which
    # has a different and much smaller one: ceil(Sq/BLOCK_Q)*Hq*B. At `fast` that is
    # 16*8*1 = 128 workgroups of one wave32 on 1024 SIMDs -- 87.5% of the machine idle --
    # while k_dq is 49.82% of that shape's kernel time
    # (rounds/017/1-profiling/kernel.yaml:55, SQ_WAVES = Grid/32 = 128 exactly). proxy
    # gives 64*32*1 = 2048 and prod 128*32*4 = 16384, so BOTH take nsp_q = 1 on the same
    # rule and run the shipped non-split k_dq object unchanged: the split cannot reach
    # the two shapes it would only cost. Same derivation as nsp above, deliberately --
    # the census that fitted it (rounds/014/1-opt/raw/census_sp.txt) is a statement about
    # dispatch waves on this part, not about which kernel is being dispatched.
    _wgs_q = ((sq + _k.BLOCK_Q - 1) // _k.BLOCK_Q) * hq * b
    nsp_q = 1
    while _wgs_q * nsp_q < 2048 and nsp_q < _NSP_Q_CAP:
        nsp_q *= 2
    if nsp_q == 1:
        dq_o = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
        _launch("dq", _k.launch_dq,
                (q, k, v, do, lse, delta, dq_o, float(softmax_scale),
                 sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                 sq // _k.BLOCK_Q, hq, b, stream))
    else:
        dqp = torch.empty((nsp_q, b, sq, hq, d), device=q.device, dtype=torch.float32)
        _launch("dq_sp", _k.launch_dq_sp,
                (q, k, v, do, lse, delta, dqp, float(softmax_scale),
                 sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                 sq // _k.BLOCK_Q, hq, b, nsp_q, hq * nsp_q, stream))
        dq_o = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
        _n_vec_q = (b * sq * hq * d) // _k.RED_VEC
        _rblk_q = (_n_vec_q + _k.RED_THREADS - 1) // _k.RED_THREADS
        _launch("redsp_q", _k.launch_redsp_q,
                (dqp, dq_o, _n_vec_q, nsp_q, _rblk_q, stream))

    return dq_o, dk_o, dv_o


attn_bwd = flydsl_attn_bwd   # uniform name every loader uses
