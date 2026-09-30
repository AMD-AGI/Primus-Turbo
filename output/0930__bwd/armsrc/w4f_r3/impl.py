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

# arm dqg_dkdv_streams -- run the dQ chain (k_dqg / k_dq / k_dq_sp+k_redsp_q) on a SIDE
# stream concurrently with the dK/dV chain (k_dkdv / k_dkdv_sp+k_redsp) on the caller's
# stream. With DQ_DFUSE = False (r29) the only dependence between the two chains is
# delta, which k_delta writes on the main stream BEFORE the fork; both chains only read
# q/k/v/do/lse/delta and write disjoint outputs, so the fork is legal. The intent is to
# let k_dqg's workgroups fill the causal tail of k_dkdv's dispatch waves (and vice
# versa). Kernels are unchanged, so outputs are bitwise identical to r29.
# The flyc.compile fast path does NOT bake the stream: fx.Stream's __c_abi_spec__ fill
# re-reads `.cuda_stream` on every call and its __cache_signature__ is the type only.
# With DQ_DFUSE = True, k_dkdv depends on k_dqg (delta), so that path stays serial.
DQ_SIDE_STREAM = True
# variant b: submit the dQ chain (side stream) BEFORE the dK/dV chain (main stream).
DQ_SIDE_FIRST = False
_SIDE = {}


# arm w4f -- the 4-wave fused-dQ k_dkdv (kernels.py:_dkdv_w4f_impl). W4F = True routes
# every call through k_delta -> zeroed fp32 dq workspace [B, Hq, Sq, D] -> k_dkdv_w4f (dK/dV
# + dQ atomics) -> k_dq_cvt; k_dqg is not launched. W4F = False is the s1 path, untouched.
W4F = True
# split-K over the q pairs for small grids (same mechanism as k_dkdv_sp). One 4-wave WG per
# CU at > 512 VGPR, 256 CUs: aim for >= 2 dispatch waves = 512 WGs, cap 16.
_W4F_SP_TARGET = 512
# debug/first-launch knob: an int forces nsp (1 = the prod kernel object k_dkdv_w4f at any
# shape, so the first card run exercises exactly what prod runs). None = the rule above.
# Env W4F_NSP_FORCE=<int> sets it without editing the file (first card run: 1).
import os as _os
W4F_NSP_FORCE = int(_os.environ["W4F_NSP_FORCE"]) if _os.environ.get("W4F_NSP_FORCE") else None


def _w4f_bwd(do, q, k, v, o, lse, softmax_scale, causal):
    b, sq, hq, d = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    assert sq % 32 == 0, f"w4f: seqlen_q must be a multiple of 32 (q pairs), got {sq}"
    assert skv % _k.W4_BKV == 0, f"w4f: seqlen_kv must be a multiple of {_k.W4_BKV}, got {skv}"
    g = hq // hkv
    stream = torch.cuda.current_stream()
    n_rows = b * sq * hq
    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    _launch("delta", _k.launch_delta,
            (do, o, delta, sq, hq, n_rows, n_rows // _k.ROWS_DELTA, stream))
    dqa = torch.zeros((b, hq, sq, d), device=q.device, dtype=torch.float32)
    wgs = (skv // _k.W4_BKV) * hkv * b
    nsp = 1
    while wgs * nsp < _W4F_SP_TARGET and nsp < 16:
        nsp *= 2
    if W4F_NSP_FORCE is not None:
        nsp = int(W4F_NSP_FORCE)
    dk_o = torch.empty((b, skv, hkv, d), device=k.device, dtype=k.dtype)
    dv_o = torch.empty_like(dk_o)
    if nsp == 1:
        _launch("dkdv_w4f", _k.launch_dkdv_w4f,
                (q, k, v, do, lse, delta, dv_o, dk_o, dqa, float(softmax_scale),
                 sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
                 skv // _k.W4_BKV, hkv, b, stream))
    else:
        dkp = torch.empty((nsp, b, skv, hkv, d), device=k.device, dtype=torch.float32)
        dvp = torch.empty_like(dkp)
        _launch("dkdv_w4f_sp", _k.launch_dkdv_w4f_sp,
                (q, k, v, do, lse, delta, dvp, dkp, dqa, float(softmax_scale),
                 sq, skv, hq, hkv, g, sq // 16, skv - sq, int(bool(causal)),
                 skv // _k.W4_BKV, hkv, b, nsp, hkv * nsp, stream))
        _n_vec = (b * skv * hkv * d) // _k.RED_VEC
        _rblk = (_n_vec + _k.RED_THREADS - 1) // _k.RED_THREADS
        _launch("redsp", _k.launch_redsp,
                (dkp, dvp, dk_o, dv_o, _n_vec, nsp, _rblk, stream))
    dq_o = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
    _nq = (b * sq * hq * d) // _k.RED_VEC
    _launch("dq_cvt", _k.launch_dq_cvt_t if _k.W4_DQ_TILED else _k.launch_dq_cvt,
            (dqa, dq_o, sq, hq, _nq, (_nq + _k.RED_THREADS - 1) // _k.RED_THREADS, stream))
    return dq_o, dk_o, dv_o


def _side_stream(dev):
    s = _SIDE.get(dev)
    if s is None:
        s = torch.cuda.Stream(device=dev)
        _SIDE[dev] = s
    return s


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
    # h33 defect a -- the assert guarded k_dkdv's query PAIR (32) but k_dq's grid is
    # ceil(Sq/BLOCK_Q) tiles while impl.py's launch passes sq // BLOCK_Q. At
    # sq % 64 == 32 the two disagree by one tile: dQ tile 0 is never computed and the
    # kernel writes 262,144 B past the end. BLOCK_Q is the binding multiple, so assert
    # it. Every scored and UT shape is already a multiple of 64, so nothing is rejected
    # that used to work correctly.
    assert sq % _k.BLOCK_Q == 0, (
        f"seqlen_q must be a multiple of {_k.BLOCK_Q} (k_dq consumes query tiles of "
        f"BLOCK_Q and the launch grid is sq // BLOCK_Q), got {sq}")
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
    if W4F:
        return _w4f_bwd(do, q, k, v, o, lse, softmax_scale, causal)
    g = hq // hkv
    stream = torch.cuda.current_stream()

    # lab-kdq: nsp_q is derived up front (same rule as below) so that the grouped k_dqg
    # can be chosen here; with DQ_DFUSE it computes delta itself and runs BEFORE k_dkdv.
    _wgs_q0 = ((sq + _k.BLOCK_Q - 1) // _k.BLOCK_Q) * hq * b
    _nsp_q0 = 1
    while _wgs_q0 * _nsp_q0 < 2048 and _nsp_q0 < _NSP_Q_CAP:
        _nsp_q0 *= 2
    use_g = (_nsp_q0 == 1 and hq % _k.DQ_NW == 0 and sq % _k.DQ_BQW == 0)
    fuse = use_g and _k.DQ_DFUSE

    def _dqg(out=None, st=stream):
        if out is None:
            out = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
        _launch("dqg", _k.launch_dqg,
                (q, k, v, do, o, lse, delta, out, float(softmax_scale),
                 sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                 sq // _k.DQ_BQW, hq // _k.DQ_NW, b, st))
        return out

    delta = torch.empty((b, hq, sq), device=q.device, dtype=torch.float32)
    if fuse:
        dq_g = _dqg()                   # writes delta, then k_dkdv reads it
    else:
        _launch("delta", _k.launch_delta,
                (do, o, delta, sq, hq, n_rows, n_rows // _k.ROWS_DELTA, stream))

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
    assert nsp_q == _nsp_q0
    if fuse:
        dq_o = dq_g
    else:
        dq_o = torch.empty((b, sq, hq, d), device=q.device, dtype=q.dtype)
    dqp = None
    if not fuse and not use_g and nsp_q != 1:
        dqp = torch.empty((nsp_q, b, sq, hq, d), device=q.device, dtype=torch.float32)

    # arm dqg_dkdv_streams -- fork: the dQ chain goes to a side stream that first waits
    # for everything already on the main stream (inputs, lse.float(), k_delta). All dQ
    # outputs/workspaces are allocated HERE on the main stream, before the fork.
    split = DQ_SIDE_STREAM and not fuse
    s2 = _side_stream(q.device) if split else stream
    if split:
        s2.wait_stream(stream)

    # arm dqg_dkdv_streams -- the dQ chain, on s2 (== stream when not split). Same
    # kernels, same arguments, same outputs as r29; only the queue differs.
    def _dq_chain():
        if fuse:
            pass
        elif use_g:
            _dqg(dq_o, s2)
        elif nsp_q == 1:
            _launch("dq", _k.launch_dq,
                    (q, k, v, do, lse, delta, dq_o, float(softmax_scale),
                     sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                     sq // _k.BLOCK_Q, hq, b, s2))
        else:
            _launch("dq_sp", _k.launch_dq_sp,
                    (q, k, v, do, lse, delta, dqp, float(softmax_scale),
                     sq, skv, hq, hkv, g, skv // _k.KV_STEP, skv - sq, int(bool(causal)),
                     sq // _k.BLOCK_Q, hq, b, nsp_q, hq * nsp_q, s2))
            _n_vec_q = (b * sq * hq * d) // _k.RED_VEC
            _rblk_q = (_n_vec_q + _k.RED_THREADS - 1) // _k.RED_THREADS
            _launch("redsp_q", _k.launch_redsp_q,
                    (dqp, dq_o, _n_vec_q, nsp_q, _rblk_q, s2))

    if DQ_SIDE_FIRST:        # variant b: dQ chain submitted before the dK/dV chain
        _dq_chain()

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
    while _wgs * nsp < 2048 and nsp < 16:
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

    if not DQ_SIDE_FIRST:
        _dq_chain()

    # join: every tensor the side stream touched is recorded on it (the caching
    # allocator must not hand its block to main-stream work that could run before s2
    # finishes), and the caller's stream waits for s2 before anything after us runs.
    if split:
        for t in (q, k, v, do, o, lse, delta, dq_o, dqp):
            if t is not None:
                t.record_stream(s2)
        stream.wait_stream(s2)

    return dq_o, dk_o, dv_o


attn_bwd = flydsl_attn_bwd   # uniform name every loader uses
