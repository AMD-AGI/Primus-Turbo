"""Route aten::mm's contiguous-(K,N) B through an N-major B on this box's hipBLASLt.

WHY. A step-10 torch profile of the 8-layer run puts 94% of GPU time in Tensile GEMM kernels,
and 97% of THAT in solutions with macro tile MT32x16x32 -- grids of 16-65 million 64-thread
workgroups, 592-681 ms for a single call. The same shapes, when hipBLASLt picks
MT256x256x128, run 16-21x faster. What flips the selector is B's physical major order:

  torch.mm(a(M,K), b(K,N) contiguous)          55.5 ms    69 TF/s   <- MT32x16x32
  F.linear(a(M,K), bT(N,K) contiguous)          2.4 ms  1628 TF/s   <- MT256x256x128
  transpose to produce bT                       0.4 ms
                                              --------------------- net 20x

F.linear is fast because it hands mm a NON-contiguous (K,N) view whose storage is N-major.
So the fix is to give mm the same thing: b.t().contiguous().t().

Costs no accuracy: both paths score SQNR 55.60 / 55.62 / 55.62 dB against an fp32 reference
at K = 14336 / 32768 / 128256 -- identical to two decimals. They differ from each other by
0.3-0.4% of peak, which is accumulation order, not bias.

A WORKAROUND for a solution-coverage gap in this image's Tensile library, same family as the
HIPBLASLT_TENSILE_LIBPATH mis-packaging in BLAS-FINDING.md. Re-measure, never assume, on any
other image -- on a library with proper coverage this would be a pure loss (one extra copy).

Implemented as a TorchDispatchMode rather than a library impl override: overriding
aten::mm.default makes the captured "original" resolve back to the override, so the fallback
path recurses until the stack blows. A mode gets re-entrancy handling for free.
"""
import os, torch
from torch.utils._python_dispatch import TorchDispatchMode

_MIN_BYTES = int(os.environ.get("NKFIX_MIN_BYTES", 4 << 20))
_MM = torch.ops.aten.mm.default
stats = {"hit": 0, "miss": 0}


class NKFix(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _MM and len(args) == 2:
            a, b = args
            if (a.dim() == 2 and b.dim() == 2 and b.is_contiguous()
                    and a.dtype in (torch.bfloat16, torch.float16) and a.dtype == b.dtype
                    and b.numel() * b.element_size() >= _MIN_BYTES):
                stats["hit"] += 1
                return func(a, b.t().contiguous().t())
            stats["miss"] += 1
        return func(*args, **kwargs)


_mode = None


def install():
    """Activate globally without a `with` block, so it can be turned on from a converter."""
    global _mode
    if _mode is None:
        _mode = NKFix()
        _mode.__enter__()
    return stats
