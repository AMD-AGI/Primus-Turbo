"""nkfix + a transpose cache keyed on tensor identity AND version.

MOTIVATION. The base workaround pays one b.t().contiguous() per mm call. For the dgrad of a
Linear, b IS the weight -- unchanged within a step and re-transposed once per layer per
backward. Caching it should (a) remove that copy from the critical path and (b) cut the rate
at which kernels are submitted, which matters because the run that wedged the card on 0915
died with "MES(0,0) ring buffer is full" -- a command-submission-rate failure mode, and the
base patch roughly doubles step throughput and therefore submission rate.

(b) is a HYPOTHESIS. The wedge has not been attributed to nkfix: it appeared on the 4th
consecutive nkfix run, while ~26 consecutive non-nkfix runs the same day never produced that
signature. Suggestive, not established, and n is small either way.

CORRECTNESS. The cache key includes `b._version`, which PyTorch bumps on every in-place
mutation -- so an optimizer step invalidates the entry automatically. Keying on data_ptr
alone would be WRONG: the allocator reuses addresses, so a freed weight's slot could serve a
different tensor's transpose. Entries hold a weakref to the source; the cache self-evicts
when the weight dies, and is capped so an unexpected key churn cannot grow without bound.
"""
import os, torch, weakref
from torch.utils._python_dispatch import TorchDispatchMode

_MIN_BYTES = int(os.environ.get("NKFIX_MIN_BYTES", 4 << 20))
_MAX_ENTRIES = int(os.environ.get("NKFIX_MAX_CACHE", 64))
_MM = torch.ops.aten.mm.default
stats = {"hit": 0, "miss": 0, "cache_hit": 0, "cache_evict": 0}
_cache = {}


def _nmajor(b):
    """Return a (K,N) view of b whose storage is N-major, reusing a cached transpose."""
    key = (b.data_ptr(), b._version, tuple(b.shape), b.dtype)
    ent = _cache.get(key)
    if ent is not None:
        src, bt = ent
        if src() is b:                      # same live tensor, not a recycled address
            stats["cache_hit"] += 1
            return bt.t()
        del _cache[key]
    bt = b.t().contiguous()
    if len(_cache) >= _MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))
        stats["cache_evict"] += 1
    _cache[key] = (weakref.ref(b), bt)
    return bt.t()


class NKFix(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _MM and len(args) == 2:
            a, b = args
            if (a.dim() == 2 and b.dim() == 2 and b.is_contiguous()
                    and a.dtype in (torch.bfloat16, torch.float16) and a.dtype == b.dtype
                    and b.numel() * b.element_size() >= _MIN_BYTES):
                stats["hit"] += 1
                return func(a, _nmajor(b))
            stats["miss"] += 1
        return func(*args, **kwargs)


_mode = None


def install():
    global _mode
    if _mode is None:
        _mode = NKFix()
        _mode.__enter__()
    return stats
