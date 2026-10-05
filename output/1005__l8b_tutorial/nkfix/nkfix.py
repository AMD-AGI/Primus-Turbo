"""nkfix for the B0 e2e: steer backward aten::mm off hipBLASLt's MT32x16x32 fallback tile.

Port of output/0915__opt/bin/nkfix.py (A0). NOTE the committed A0 file at HEAD lost its helper
block (_MM / stats / _big / _headroom_ok are referenced but undefined -- nanloc2 on A0 died with
NameError('stats')); this is rebuilt from the last complete version (1255557f) and changed:

  * No _headroom_ok / torch.cuda.mem_get_info per call. Instead every temporary copy is BOUNDED
    (NKFIX_CHUNK_BYTES, default 256 MiB) by splitting the GEMM along an OUTPUT dimension, so the
    K-reduction of every output element is unchanged. On the 32-layer config (88.3-88.4% memory,
    memguard at 88.5%) the unchunked lm_head wgrad would copy a 7.83 GiB operand.
  * Non-finite accounting is exact and asynchronous (count of non-finite elements per rule, for
    the rewritten call's OUTPUT and optionally its INPUTS, so an event is attributable: finite in
    -> non-finite out means the GEMM made it), inspected every NKFIX_CHECK_EVERY rewritten calls.

The image's hipBLASLt ships no plain-bf16 tuned library for the NN / (A-view, B-contig) transpose
combos, so dgrad (Ailk_Bljk) and wgrad (Ailk_Bjlk) fall back to MT32x16x32 at 50-78 TF/s; the
forward's Alik_Bljk lands on MT256x256x128 at 1.5-1.9 PF/s. The rewrite hands mm the same
logical matrices in the forward's physical layout (opprobe.log: 5-30x per call incl. copies).

  dgrad   A (M,K) contiguous, B (K,N) contiguous      -> mm(A, B.t().contiguous().t())
  wgrad   A = g.t() view,     B (K,N) contiguous      -> mm(A.contiguous(), B.t().contiguous().t())
          (order matters: wgrad also has a contiguous B, test it first)

Env: NKFIX_ENABLE=1 installs it through the Primus hook (primus-nkfix-hook.patch).
     NKFIX_MIN_BYTES (4 MiB), NKFIX_CHUNK_BYTES (256 MiB), NKFIX_CHECK (0 off / 1 outputs /
     2 outputs+inputs), NKFIX_CHECK_EVERY (64), NKFIX_STATS_FILE, NKFIX_RULES ("dgrad,wgrad"),
     NKFIX_SCRATCH (1), NKFIX_SHADOW (""|orig|nk), NKFIX_TRANSPOSE (triton|torch).
"""
import os
import sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_MIN_BYTES = int(os.environ.get("NKFIX_MIN_BYTES", 4 << 20))
_CHUNK = int(os.environ.get("NKFIX_CHUNK_BYTES", 256 << 20))
_CHECK = int(os.environ.get("NKFIX_CHECK", "0"))
_EVERY = int(os.environ.get("NKFIX_CHECK_EVERY", "64"))
_RULES = set(os.environ.get("NKFIX_RULES", "dgrad,wgrad").split(","))
# In-situ verification: NKFIX_SHADOW=orig|nk computes BOTH the untouched call and the rewrite on
# the real tensors of the real step, records how far apart they are, and returns the named one.
# (=orig makes the run numerically the unpatched run with the mode installed -- it separates
# "the rewrite changes results" from "installing a dispatch mode changes results".)
_SHADOW = os.environ.get("NKFIX_SHADOW", "")
_shadow = []       # (tag, shape key, device tensor [max|d|/rms(orig), frac>2ulp, sum(o), sum(k)])
_MM = torch.ops.aten.mm.default
_MM_OUT = torch.ops.aten.mm.out

stats = {"dgrad": 0, "wgrad": 0, "chunked": 0, "miss": 0,
         "checked": 0, "syncs": 0}
_seen = {}
_cnt = {}          # tag -> fp32 device scalar: running max |x| (NaN-absorbing, async)
_peak = {}         # tag -> largest finite max |x| seen at a sync
_events = []       # (call index, tag, count) found at a sync


def _big(t):
    return t.numel() * t.element_size() >= _MIN_BYTES


def _nmajor(b):
    """Same logical (K, N) matrix, physically N-major (the forward's B layout)."""
    return b.t().contiguous().t()


def _count(tag, t):
    # aminmax is a pure reduction (no numel-sized temporary, unlike isfinite on an 8 GiB operand)
    # and propagates NaN; |.|max also records how big the values got (garbage reads as 1e20+).
    # A transposed view is reduced through its contiguous base: aminmax on the view itself
    # materialises a contiguous copy (memprobe: +8 GiB on the lm_head wgrad A).
    if t.dim() == 2 and not t.is_contiguous() and t.t().is_contiguous():
        t = t.t()
    mn, mx = torch.aminmax(t)
    v = torch.maximum(mn.float().abs(), mx.float().abs())
    c = _cnt.get(tag)
    if c is None:
        _cnt[tag] = v.clone()
        _peak[tag] = 0.0
    else:
        torch.maximum(c, v, out=c)        # NaN is absorbing


def _check(a, b, out, rule):
    if not _CHECK:
        return
    _count(rule + ".out", out)
    if _CHECK >= 2:
        _count(rule + ".inA", a)
        _count(rule + ".inB", b)
    stats["checked"] += 1
    if stats["checked"] % _EVERY == 0:
        _sync_check()


def _sync_check():
    stats["syncs"] += 1
    for tag, c in _cnt.items():
        v = float(c.item())
        if v != v or v == float("inf"):
            _events.append((stats["checked"], tag, v))
            print("[nkfix] NON-FINITE in %s by rewritten call %d" % (tag, stats["checked"]),
                  file=sys.stderr, flush=True)
            c.zero_()
        else:
            _peak[tag] = max(_peak[tag], v)


# --- the rewrite -----------------------------------------------------------------------------
# All temporaries (A rows made contiguous, B made N-major, a column block of the output) live in
# ONE persistent scratch per (device, stream), NKFIX_CHUNK_BYTES each, allocated on first use and
# reused by every later call. So after the first backward the rewrite allocates nothing but the
# output tensor -- the same allocation the untouched mm makes. (A0's version allocated a fresh
# copy per call, up to 7.83 GiB for the lm_head A; its NaN runs clustered in the allocator warm-up
# steps 1-10. Whatever the mechanism there, this removes nkfix's own allocator churn from the
# picture.) Reuse is safe because every consumer of the scratch is enqueued on the same stream,
# after the previous consumer. NKFIX_SCRATCH=0 falls back to fresh allocations.
_USE_SCRATCH = os.environ.get("NKFIX_SCRATCH", "1") not in ("", "0")
_scratch = {}
# Operand transposes: a tiled Triton transpose (transpose_triton.py; bit-identical to torch's copy,
# 5-7x faster on these shapes -- tprobe_prod.log) unless NKFIX_TRANSPOSE=torch or the import fails.
_TRANSPOSE = os.environ.get("NKFIX_TRANSPOSE", "triton")
_tt = None
if _TRANSPOSE == "triton":
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from transpose_triton import transpose_into as _tt
    except Exception as e:                                   # pragma: no cover
        print("[nkfix] triton transpose unavailable (%r), using torch copy" % (e,), file=sys.stderr)
        _TRANSPOSE = "torch"


def _tcopy(y, x):
    """y (C, R) contiguous <- x.T for x (R, C) of any strides."""
    if _tt is not None and y.is_cuda:
        _tt(y, x)
    else:
        y.copy_(x.t())


def _buf(t, which, nelem):
    # "o" (a column block of the output) never needs more than M x cols: 64 MiB on these shapes
    cap = _CHUNK // 4 if which == "o" else _CHUNK
    if not _USE_SCRATCH or t.device.type != "cuda" or nelem * t.element_size() > cap:
        return torch.empty(nelem, device=t.device, dtype=t.dtype)
    key = (t.device, torch.cuda.current_stream(t.device).cuda_stream, t.dtype, which)
    s = _scratch.get(key)
    if s is None:
        s = _scratch[key] = torch.empty(cap // t.element_size(), device=t.device, dtype=t.dtype)
        stats["scratch_bytes"] = stats.get("scratch_bytes", 0) + cap
    return s[:nelem]


def _rewrite(a, b, copy_a):
    """mm(a, b) with B handed over N-major (and A row-major when copy_a): the forward's layout.

    Split along OUTPUT rows (only needed when A is copied) and OUTPUT columns so that no
    temporary exceeds NKFIX_CHUNK_BYTES; every output element keeps its full K reduction.
    """
    M, K = a.shape
    N = b.shape[1]
    es = a.element_size()
    unit = lambda: max(256, (_CHUNK // (K * es)) // 256 * 256)
    cols = N if K * N * es <= _CHUNK else unit()
    rows = M if (not copy_a or M * K * es <= _CHUNK) else unit()
    if cols < N or rows < M:
        stats["chunked"] += 1
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    for c0 in range(0, N, cols):
        c1 = min(N, c0 + cols); cn = c1 - c0
        bn = _buf(b, "b", K * cn).view(cn, K)
        _tcopy(bn, b[:, c0:c1])
        bn = bn.t()                                     # logical (K, cn), physically N-major
        for r0 in range(0, M, rows):
            r1 = min(M, r0 + rows); rn = r1 - r0
            if copy_a:
                ac = _buf(a, "a", rn * K).view(rn, K)
                _tcopy(ac, a[r0:r1].t())
            else:
                ac = a[r0:r1]
            if cols == N:
                _MM_OUT(ac, bn, out=out[r0:r1])
            else:
                o = _buf(a, "o", rn * cn).view(rn, cn)
                _MM_OUT(ac, bn, out=o)
                out[r0:r1, c0:c1].copy_(o)
    return out


def _wgrad(a, b):
    # a: (Mo, K) = g.t() view, b: (K, N) contiguous
    return _rewrite(a, b, copy_a=True)


def _dgrad(a, b):
    # a: (M, K) contiguous, b: (K, N) contiguous
    return _rewrite(a, b, copy_a=False)


def _shadow_cmp(tag, key, a, b, out):
    o = _MM(a, b)
    of, kf = o.float(), out.float()
    d = (of - kf).abs()
    m = torch.maximum(of.abs(), kf.abs()).clamp_min(2.0 ** -126)
    u = torch.exp2(torch.floor(torch.log2(m)) - 7)
    rms = of.pow(2).mean().sqrt()
    _shadow.append((tag, key, torch.stack([d.max() / (rms + 1e-30), (d > 2 * u).float().mean(),
                                           of.sum(), kf.sum(), (~torch.isfinite(o)).sum().float(),
                                           (~torch.isfinite(out)).sum().float()])))
    return o if _SHADOW == "orig" else out


class NKFix(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _MM and len(args) == 2 and not kwargs:
            a, b = args
            if (a.dim() == 2 and b.dim() == 2 and a.dtype == b.dtype
                    and a.dtype in (torch.bfloat16, torch.float16)
                    and a.device.type == "cuda" and (_big(a) or _big(b))):
                k = (tuple(a.shape), tuple(b.shape),
                     "Ac" if a.is_contiguous() else "Av", "Bc" if b.is_contiguous() else "Bv")
                _seen[k] = _seen.get(k, 0) + 1
                if ("wgrad" in _RULES and not a.is_contiguous() and a.t().is_contiguous()
                        and b.is_contiguous() and _big(a)):
                    stats["wgrad"] += 1
                    out = _wgrad(a, b)
                    _check(a, b, out, "wgrad")
                    if _SHADOW:
                        out = _shadow_cmp("wgrad", k, a, b, out)
                    return out
                if "dgrad" in _RULES and a.is_contiguous() and b.is_contiguous() and _big(b):
                    stats["dgrad"] += 1
                    out = _dgrad(a, b)
                    _check(a, b, out, "dgrad")
                    if _SHADOW:
                        out = _shadow_cmp("dgrad", k, a, b, out)
                    return out
            stats["miss"] += 1
        return func(*args, **kwargs)


_mode = None


def report():
    if _CHECK:
        try:
            _sync_check()
        except Exception as e:
            print("[nkfix] final check failed: %r" % (e,), file=sys.stderr, flush=True)
    path = os.environ.get("NKFIX_STATS_FILE", "/tmp/nkfix_stats.txt")
    try:
        with open(path, "w") as f:
            f.write("stats: %r\n" % (stats,))
            f.write("nonfinite_events: %r\n" % (_events,))
            f.write("max_abs_by_tag: %r\n" % ({k: "%.4g" % v for k, v in _peak.items()},))
            for k, v in sorted(_seen.items(), key=lambda kv: -kv[1]):
                f.write("  %6d x  A%s %s  B%s %s\n" % (v, k[0], k[2], k[1], k[3]))
            if _shadow:
                agg = {}
                for i, (tag, key, t) in enumerate(_shadow):
                    v = t.tolist()
                    f.write("shadow %5d %s A%s%s B%s%s maxd/rms=%.3e frac>2ulp=%.3e sum_o=%.6e sum_k=%.6e nonfin_o=%d nonfin_k=%d\n"
                            % (i, tag, key[0], key[2], key[1], key[3], v[0], v[1], v[2], v[3], v[4], v[5]))
                    g = agg.setdefault((tag, key), [0, 0.0, 0.0, 0, 0])
                    g[0] += 1; g[1] = max(g[1], v[0]); g[2] = max(g[2], v[1]); g[3] += int(v[4]); g[4] += int(v[5])
                f.write("shadow summary (mode=%s): calls, worst maxd/rms, worst frac>2ulp, nonfin orig, nonfin nk\n" % _SHADOW)
                for (tag, key), g in sorted(agg.items(), key=lambda kv: str(kv[0])):
                    f.write("  %s A%s%s B%s%s  n=%d  %.3e  %.3e  %d  %d\n" % (tag, key[0], key[2], key[1], key[3], *g))
    except Exception as e:
        print("[nkfix] stats write failed: %r" % (e,), file=sys.stderr, flush=True)


def install():
    global _mode
    if _mode is None:
        _mode = NKFix()
        _mode.__enter__()
        import atexit
        atexit.register(report)
        print("[nkfix] installed rules=%s min_bytes=%d chunk_bytes=%d check=%d/%d shadow=%r transpose=%s"
              % (sorted(_RULES), _MIN_BYTES, _CHUNK, _CHECK, _EVERY, _SHADOW, _TRANSPOSE), file=sys.stderr, flush=True)
    return stats
