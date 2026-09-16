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
# Same sampling for rules 1 and 2. Scanning every finished run's log found loss=nan in 8 of 39
# runs that had nkfix installed and 0 of 35 that did not -- and the non-nkfix group had MORE
# total steps, so it had more exposure, not less (Fisher one-sided p ~ 0.003). That points at
# nkfix itself rather than at rule 3, which did not exist for six of the eight. So the check
# cannot stay on rule 3 alone: it has to be able to say WHICH rule's output went non-finite.
_CHECK = int(os.environ.get("NKFIX_CHECK", "0"))

# Detecting a rare corruption needs every call covered, and cannot afford a synchronize on every
# call. The first version sampled one call in N: with N=7 over 6840 calls it inspected 14% of
# them, so even a run that did go nan would most likely have missed the offending call. Raising
# it to every call would add a device sync per mm -- which serialises the pipeline and is exactly
# the kind of diagnostic that changes whether a race happens at all.
#
# So: fold every output into a running accumulator (asynchronous, no sync, and nan/inf is
# absorbing -- once it is in, it never leaves), and inspect the accumulator only every N calls.
# 100% of calls covered, one sync per N.
_acc = {}


def _finite_check(tag, out):
    if not _CHECK:
        return
    a = _acc.get(tag)
    if a is None:
        a = _acc[tag] = torch.zeros((), device=out.device, dtype=torch.float32)
    a += out.sum(dtype=torch.float32)          # async; no synchronize here
    stats["checked"] += 1
    if stats["checked"] % _CHECK:
        return
    if not torch.isfinite(a):                  # the only sync, once per N calls
        stats["nonfinite_" + tag] = stats.get("nonfinite_" + tag, 0) + 1
        print("[nkfix] NON-FINITE detected in rule %s by call %d (accumulator poisoned)"
              % (tag, stats["checked"]), flush=True)
        a.zero_()                              # re-arm, so a second event is still visible


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
                        out = func(a.contiguous(), b.t().contiguous().t())
                        _finite_check("wgrad", out)
                        return out
                    # No room for the copy. Fall through to the dgrad rule, which needs no
                    # extra allocation and is still worth 1.16-1.34x on a wgrad call -- far
                    # better than taking the SIGBUS.
                    stats["wgrad_skipped_oom"] += 1
                if b.is_contiguous() and _big(b):
                    stats["dgrad"] += 1
                    out = func(a, b.t().contiguous().t())
                    _finite_check("dgrad", out)
                    return out
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
