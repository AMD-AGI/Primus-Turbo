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

COST. The wgrad rule copies A -- 0.25 GiB for qkv, 0.88 for mlp, 7.83 for lm_head. Fine on
the 8-layer config (32% memory). NOT established as safe on the 32-layer production config,
which already sits at 88%.

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
_MM = torch.ops.aten.mm.default
stats = {"dgrad": 0, "wgrad": 0, "miss": 0}
_seen = {}


def _big(t):
    return t.numel() * t.element_size() >= _MIN_BYTES


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
                    stats["wgrad"] += 1
                    return func(a.contiguous(), b.t().contiguous().t())
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
