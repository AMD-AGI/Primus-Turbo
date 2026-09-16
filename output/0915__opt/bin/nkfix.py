"""Steer aten::mm away from this image's MT32x16x32 solutions.

WHY. hipBLASLt on this image ships no plain-bf16-GEMM tuning library for the NN transpose
combination, so those calls fall back on a sparse GridBased table whose nearest entry to our
shapes is N=1 -- and MT32x16x32 is the tile you would pick for a GEMV. 500-900 ms for a call
the right solution does in single digits.

Two call classes, found by profiling the 8-layer step and then reading which predicate
actually fired (the stats file below -- the first version of rule 2 matched ZERO times in a
real run because its premise about the operand layout was wrong):

  dgrad   A row-major, B contiguous (K,N).   Give mm an N-major B.
              torch.mm(a, b)              55.5 ms    69 TF/s
              b.t().contiguous().t()       2.4 ms  1628 TF/s     ~20x incl. copy

  wgrad   A is a transposed VIEW, B contiguous.  BOTH operands must change; either one
          alone leaves it on the bad tile, which is why the first attempt bought nothing:
              untouched (Av + Bc)         70.9 ms    54 TF/s
              B -> N-major only           55.5 ms    69 TF/s
              A -> contiguous only        61.4 ms    63 TF/s
              BOTH                         3.4 ms  1146 TF/s     9.4x incl. both copies
          lm_head, the biggest single call: 690 -> 23.2 ms, 14.7x net.

Bit-exact either way: every variant scores the same SQNR against an fp32 reference
(55.60-55.62 dB at K = 14336 / 32768 / 128256), because only operand layout changes.

COST, and it bit. The wgrad rule copies A -- 0.25 GiB for qkv, 0.88 for mlp, 7.83 for
lm_head. Free on the 8-layer config (32% memory, peak unchanged at 141.00 GiB because the
caching allocator reuses the block). On the 32-layer production config, which already sits at
88%, the lm_head copy SIGBUSed the process before step 1 and took the card with it (the dying
process keeps its KFD context; dmesg then shows MES failures and a GPU reset that does not
complete). That is why _headroom_ok exists: the rule now asks whether the copy fits before
making it, instead of assuming a config has room.

Note what the failure looks like from outside, because it is misleading: the dmesg trail is
MES INVALIDATE_TLBS failures and "MES might be in unrecoverable state", which reads like the
ring-buffer wedges. The actual cause is one line in the training log -- Signal 7 (SIGBUS).
Read the training log before theorising about the driver.

A WORKAROUND for a library coverage defect, same family as the HIPBLASLT_TENSILE_LIBPATH
mis-packaging. On an image with proper NN coverage every rule here is a pure loss. Re-measure,
never assume, elsewhere.

Implemented as a TorchDispatchMode: overriding aten::mm.default via torch.library makes the
captured "original" resolve back to the override and the fallback path recurses until the
stack dies. A dispatch mode gets re-entrancy handling for free.
"""
import os, torch
from torch.utils._python_dispatch import TorchDispatchMode

_MIN_BYTES = int(os.environ.get("NKFIX_MIN_BYTES", 4 << 20))
_RULE2 = os.environ.get("NKFIX_WGRAD", "1") not in ("", "0")
_HEADROOM = float(os.environ.get("NKFIX_HEADROOM", "0.5"))
# Rule 3: send wgrad to the FlyDSL gfx1250 WMMA GEMM instead of copying both operands.
# Off by default -- it needs the feat/gemm/gfx1250-flydsl-gemm branch merged, and a config
# where that import fails must behave exactly as before rather than erroring at the first mm.
_RULE3 = os.environ.get("NKFIX_FLYDSL_WGRAD", "") not in ("", "0")
# Check every Nth rule-3 output for non-finite values (0 = never). One run in eleven went
# loss=nan at step 10 -- a sudden jump, not a divergence: the seeded control run was at 9.71 on
# the same step. 720 isolated calls across three processes were bit-identical to torch.mm, so
# the fault is rare, nondeterministic, and not reproducible in isolation. Until it is
# understood, a run using rule 3 should be able to say whether it stayed finite rather than
# leaving it to the loss column to notice several steps later.
_RULE3_CHECK = int(os.environ.get("NKFIX_FLYDSL_CHECK", "0"))
_MM = torch.ops.aten.mm.default
stats = {"dgrad": 0, "wgrad": 0, "wgrad_skipped_oom": 0, "miss": 0,
         "wgrad_flydsl": 0, "flydsl_unavailable": 0, "flydsl_declined": 0,
         "flydsl_no_config": 0, "flydsl_nonfinite": 0}
_seen = {}


def _big(t):
    return t.numel() * t.element_size() >= _MIN_BYTES


def _headroom_ok(t):
    """Is there room to copy t without pushing the allocator over a cliff?

    Asked at call time rather than gated on a fixed size, because the same rule is free on a
    config at 32% memory and fatal on one at 88%. torch.cuda.mem_get_info reports the DEVICE's
    free bytes, which is what a fresh allocation actually draws on -- the caching allocator's
    own reserve is already excluded from it.

    The 2x is not padding for its own sake: contiguous() must hold the source and the
    destination simultaneously, so the transient requirement is twice the tensor. _HEADROOM
    then keeps a margin on top; at 0.5 the copy may claim at most half of what is free.
    """
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return True          # cannot tell -- behave as before rather than silently disabling
    need = t.numel() * t.element_size() * 2
    return need < free * _HEADROOM


# --- rule 3: FlyDSL TN for wgrad -------------------------------------------------------
#
# Why this exists. Rule 2 buys its speed with two copies -- dout^T and the activation -- and
# measurement (BOTTLENECK-SHIFT.md) put those at 137 ms of a 760 ms step, with NEITHER
# cacheable: both operands are freshly computed every layer every step. FlyDSL's TN kernel
# consumes the transposed view directly, so the copies simply do not happen. Measured on the
# real shapes with rule 2's copies included, as training pays them:
#
#     shape                     rule 2 (+copies)   FlyDSL tuned
#     o_proj  4096x32768x4096       441.7 TF/s       794.2       1.80x
#     mlp    14336x32768x4096       517.0            715.8       1.38x
#     mlp2    4096x32768x14336      506.3            726.6       1.44x
#     qkv     6144x32768x4096       470.6            625.9       1.33x
#     lm_head 128256x32768x4096     732.1            782.2       1.07x
#
# lm_head is the exception and it matters, because it is 29% of the wgrad FLOPs: rule 2 is
# already good there (732 vs 442-517 elsewhere), so the hybrid's win is 58 ms, not the 77 ms
# a mean over the other four shapes would have predicted.
#
# NOT used for dgrad. There FlyDSL NN measures 820-849 TF/s against rule 1's 1238-1627, and
# rule 1's only copy is a weight (12.5 ms/step). Rule 1 stays.

_fly = None          # gemm_gfx1250, or False once the import has failed
_fly_table = None    # {(M, N, K): cfg or False}, loaded from NKFIX_FLYDSL_TABLE


def _flydsl():
    global _fly
    if _fly is None:
        try:
            from primus_turbo.flydsl.gemm.gemm_gfx1250_kernel import gemm_gfx1250
            _fly = gemm_gfx1250
        except Exception:
            _fly = False
    return _fly


def _load_table():
    """Configs measured OFFLINE, one per exact (M, N, K). Absent shape -> rule 2.

    The first version of this rule called FlyDSL's autotune() lazily from inside the training
    step, and that run took the card down at step 2 with hipErrorLaunchFailure. Two reasons it
    must never happen again, independent of which config actually faulted:

      * autotune's measurement loop is `try: ... torch.cuda.synchronize() ... except: continue`.
        A candidate whose launch faults raises at that synchronize, gets swallowed, and the
        loop moves on -- but a HIP context that has taken an unspecified launch failure is
        dead, and catching the Python exception does not revive it. The next real GEMM reports
        the failure, and by then the cause is 30 configs back. It converts "the context is
        gone" into "that config was not a candidate".
      * Its cache key is (kind, layout, N, K), documented as leaving M out because "M is the
        token count". True for NT and NN. In TN it is inverted -- there M is out_features and
        K is the token count -- so o_proj's wgrad (M=4096) and kv's (M=1024) share one tuned
        config, and feasible_configs never constrains M at all, so nothing re-checks it.

    So: measure offline, key on the whole shape, and treat an unknown shape as out of scope
    rather than as something to go and tune mid-step.
    """
    global _fly_table
    if _fly_table is None:
        _fly_table = {}
        path = os.environ.get("NKFIX_FLYDSL_TABLE", "")
        if path:
            try:
                import json
                with open(path) as f:
                    for k, v in json.load(f).items():
                        M, N, K = (int(x) for x in k.split(","))
                        _fly_table[(M, N, K)] = v and dict(
                            tile=tuple(v["tile"]), m_warp=v["m_warp"],
                            n_warp=v["n_warp"], num_buffers=v["num_buffers"])
            except Exception as e:
                print("[nkfix] FlyDSL table unreadable (%r); rule 3 disabled" % (e,), flush=True)
    return _fly_table


def _flydsl_wgrad(a, b):
    """Run a wgrad mm through FlyDSL TN, or return None to fall through to rule 2.

    ``a`` is the (M, K) transposed view; FlyDSL wants its (K, M) base. Returning None rather
    than raising is deliberate: this sits in a dispatch mode on every mm in the model, so an
    unsupported shape must degrade to the path that already works.
    """
    gemm = _flydsl()
    if not gemm:
        stats["flydsl_unavailable"] += 1
        return None
    base = a.t()
    if not base.is_contiguous() or not b.is_contiguous():
        stats["flydsl_declined"] += 1
        return None
    cfg = _load_table().get((a.shape[0], b.shape[1], b.shape[0]))
    if not cfg:
        stats["flydsl_no_config"] += 1
        return None
    try:
        out = gemm(base, b, layout="tn", out_dtype=a.dtype, **cfg)
    except Exception:
        stats["flydsl_declined"] += 1
        return None
    if out is None:
        stats["flydsl_declined"] += 1
        return None
    stats["wgrad_flydsl"] += 1
    if _RULE3_CHECK and stats["wgrad_flydsl"] % _RULE3_CHECK == 0:
        # torch.isfinite().all() costs a full read plus a sync, so it is sampled rather than
        # run on every call. Falling back to rule 2 on a bad output would hide the event; the
        # point is to record that it happened, with the shape, while the run is still alive.
        if not torch.isfinite(out).all():
            stats["flydsl_nonfinite"] += 1
            print("[nkfix] NON-FINITE FlyDSL wgrad output: M=%d N=%d K=%d (call %d)"
                  % (a.shape[0], b.shape[1], b.shape[0], stats["wgrad_flydsl"]), flush=True)
    return out


class NKFix(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _MM and len(args) == 2:
            a, b = args
            if (a.dim() == 2 and b.dim() == 2 and a.dtype == b.dtype
                    and a.dtype in (torch.bfloat16, torch.float16)):
                if _big(a) or _big(b):
                    k = (tuple(a.shape), tuple(b.shape),
                         "Ac" if a.is_contiguous() else "Av",
                         "Bc" if b.is_contiguous() else "Bv")
                    _seen[k] = _seen.get(k, 0) + 1
                # Order matters: the wgrad test must come first. A wgrad call also has a
                # contiguous B, so the dgrad rule would claim it and leave it on the bad tile
                # -- which is exactly what the previous version did, silently.
                if _RULE2 and not a.is_contiguous() and b.is_contiguous() and _big(a):
                    # Rule 3 first when enabled: it needs no allocation at all, so it is also
                    # the right answer on a config too tight for rule 2's copy.
                    if _RULE3:
                        out = _flydsl_wgrad(a, b)
                        if out is not None:
                            return out
                    if _headroom_ok(a):
                        stats["wgrad"] += 1
                        return func(a.contiguous(), b.t().contiguous().t())
                    # No room for the copy. Fall through to the dgrad rule, which needs no
                    # extra allocation and is still worth 1.16-1.34x on a wgrad call -- far
                    # better than taking the SIGBUS.
                    stats["wgrad_skipped_oom"] += 1
                if b.is_contiguous() and _big(b):
                    stats["dgrad"] += 1
                    return func(a, b.t().contiguous().t())
            stats["miss"] += 1
        return func(*args, **kwargs)


_mode = None


def _report():
    # Which rule actually fired in a real run? The microbenchmark says rule 2 is worth 8-11x,
    # so if the end-to-end number does not move, the first thing to check is whether the
    # predicate ever matched -- not whether the rewrite works.
    # A file, not stdout: under the training launcher stdout goes through capture layers that
    # demonstrably swallow lines -- the same reason the ASM-backward gate writes a trace file.
    out = os.environ.get("NKFIX_STATS_FILE", "/tmp/nkfix_stats.txt")
    try:
        with open(out, "w") as f:
            f.write("stats: %r\n" % (stats,))
            for k, v in sorted(_seen.items(), key=lambda kv: -kv[1]):
                f.write("  %5d x  A%s %s  B%s %s\n" % (v, k[0], k[2], k[1], k[3]))
    except Exception as e:
        print("[nkfix] stats write failed: %r" % (e,), flush=True)


def install():
    global _mode
    if _mode is None:
        _mode = NKFix()
        _mode.__enter__()
        import atexit
        atexit.register(_report)
    return stats
