"""Steer aten::mm away from this image's MT32x16x32 solutions. Two rules, two directions.

WHY. hipBLASLt on this image has no plain-bf16-GEMM tuning library for some transpose
combinations, so those fall back on a sparse GridBased table whose nearest entry to our
shapes is N=1 -- and MT32x16x32 is the tile you would pick for a GEMV. Result: 16-65 million
workgroups and 500-900 ms for a call the right solution does in single-digit ms.

Two distinct bad cases, found by profiling the 8-layer step twice (before and after rule 1):

  Rule 1 -- dgrad. B contiguous (K,N).  Hand mm a B whose storage is N-major instead:
      torch.mm(a, b)                55.5 ms    69 TF/s
      b.t().contiguous().t()         2.4 ms  1628 TF/s     ~20x incl. copy

  Rule 2 -- wgrad. BOTH operands are transposed views (Alik_Bjlk). Here the fix is on A,
  not B, and making B contiguous makes it WORSE (54.5 TF/s):
      torch.mm(a, b)                55.6 ms    69 TF/s
      a.contiguous()                 3.3 ms  1160 TF/s   + 3.3 ms copy -> 8.4x net
      lm_head (A = 7.83 GiB):     513.6 -> 23.0 + 22.8   ->  11.2x net
  The zero-copy rewrite (B^T @ A^T)^T was tried and LOSES (0.83-0.95x) -- it lands on the
  same bad family. The copy is not avoidable; it is simply worth paying.

Both rules are bit-exact: the rewritten call and the original produce identical results
(maxdiff 0 measured at three shapes), because only the operand layout changes.

COST. Rule 2 allocates a transient copy of A -- 0.25 GiB for qkv wgrad, 0.88 for mlp,
7.83 for lm_head. Fine on the 8-layer config (32% memory). NOT obviously safe on the
32-layer production config, which already sits at 88%.

This is a WORKAROUND for a library packaging/coverage defect, in the same family as the
HIPBLASLT_TENSILE_LIBPATH mis-packaging. On an image with proper coverage both rules are a
pure loss (one extra copy). Re-measure, never assume, elsewhere.

Implemented as a TorchDispatchMode: overriding aten::mm.default via torch.library makes the
captured "original" resolve back to the override, so the fallback path recurses until the
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
                if b.is_contiguous() and _big(b):
                    stats["dgrad"] += 1
                    return func(a, b.t().contiguous().t())
                if _RULE2 and not b.is_contiguous() and not a.is_contiguous() and _big(a):
                    stats["wgrad"] += 1
                    return func(a.contiguous(), b)
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
