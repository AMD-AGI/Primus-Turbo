#!/usr/bin/env python3
"""Canonical FLOP and byte counts for the ops this project optimizes.

An agent that counts FLOP by hand gets it wrong in ways that are hard to see: it drops the
factor of 2 on a multiply-add, counts a causal attention as if it were dense, forgets that
GQA reads each KV head once and not once per query head, or prices the backward as one GEMM
when it is five. Every one of those produces a believable TFLOPS number that is wrong by an
integer factor, and an integer factor is exactly what a roofline cannot survive.

So the counts live here, once, and everything else calls this:

    python3 tools/op_flops.py gemm --m 4096 --n 4096 --k 4096 --dtype bf16
    python3 tools/op_flops.py attention --batch 1 --heads 128 --kv-heads 16 \
        --seqlen 4096 --head-dim 64 --causal bottom-right
    python3 tools/op_flops.py grouped_gemm --group-m 512,512,1024 --n 4096 --k 4096

    from op_flops import gemm, attention
    r = gemm(m=4096, n=4096, k=4096, dtype="bf16")
    tflops = r.flop / seconds / 1e12

`flop` is *algorithmic* work -- what the operator owes, not what a kernel issued. A kernel
that recomputes, pads to a tile, or runs a masked-out block still owes the same FLOP, and
dividing by measured time is what makes achieved TFLOPS comparable between implementations.

`bytes_min` is the compulsory traffic at the DRAM level: every input read once, every output
written once, in the storage dtype. It is a lower bound, so `flop / bytes_min` is the highest
arithmetic intensity the operator can have -- a real kernel sits at or below it. Use it for
the roofline's ridge-point comparison; use measured counters for where the kernel actually is.
"""
import argparse
import sys
from dataclasses import dataclass

# Storage width. Accumulators are wider and are not counted -- bytes_min is DRAM traffic.
DTYPE_BYTES = {
    "fp32": 4,
    "tf32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "bf8": 1,
    "fp6": 0.75,
    "fp4": 0.5,
}


@dataclass
class Count:
    op: str
    flop: float
    bytes_min: float
    detail: str

    @property
    def intensity(self):
        """Algorithmic FLOP per compulsory DRAM byte -- the op's ceiling, not the kernel's."""
        return self.flop / self.bytes_min if self.bytes_min else float("nan")


def _b(dtype):
    if dtype not in DTYPE_BYTES:
        sys.exit("unknown dtype %r; known: %s" % (dtype, ", ".join(sorted(DTYPE_BYTES))))
    return DTYPE_BYTES[dtype]


def gemm(m, n, k, dtype="bf16", out_dtype=None, batch=1):
    """C[m,n] = A[m,k] @ B[k,n]. One multiply-add is 2 FLOP."""
    e, eo = _b(dtype), _b(out_dtype or dtype)
    flop = 2.0 * batch * m * n * k
    byt = batch * (m * k * e + k * n * e + m * n * eo)
    return Count("gemm", flop, byt, "batch=%d m=%d n=%d k=%d %s" % (batch, m, n, k, dtype))


def grouped_gemm(group_m, n, k, dtype="bf16", out_dtype=None):
    """Per-group A[m_g,k] @ B[k,n]. B is per group, which is what makes it not one big GEMM."""
    e, eo = _b(dtype), _b(out_dtype or dtype)
    total_m = float(sum(group_m))
    flop = 2.0 * total_m * n * k
    byt = total_m * k * e + len(group_m) * k * n * e + total_m * n * eo
    return Count(
        "grouped_gemm", flop, byt, "groups=%d total_m=%d n=%d k=%d %s" % (len(group_m), total_m, n, k, dtype)
    )


def _live_fraction(sq, skv, mode, window_left=-1, window_right=0):
    """Fraction of the score matrix the mask leaves live.

    Two masks compose, and both change the FLOP count:

    **Causal alignment.** Bottom-right puts the diagonal at the *end* of both axes, which is
    what a decode-style or ragged batch means by causal. Top-left puts it at the origin. The
    two agree only when sq == skv; at sq << skv they differ by a large factor, and taking the
    wrong one is a silent FLOP error of exactly that size.

    **Sliding window.** `window_left = W` keeps only the W keys before the diagonal (plus the
    diagonal itself), so a row has **W + 1** live keys, not W. That is Primus-Turbo's
    convention, not a choice made here: its eager reference masks
    `col_idx < row_idx + shift - window_size[0]` and its HipKittens kernel keeps
    `kv >= q + offset - window_left`, both inclusive. A reader coming from HuggingFace's
    `sliding_window` expects W; do not "fix" this to match. `window_right` extends it after. This is the one that matters most for
    a roofline: a window turns attention's work from quadratic in sequence length into linear,
    so at long sequence a windowed kernel does a *small fraction* of the dense FLOP. Counting
    it as dense understates achieved TFLOPS by that same fraction and puts the point in
    completely the wrong place on the plot.

    `window_left = -1` means unlimited, i.e. no window.
    """
    if mode in (None, "none", "full"):
        off, causal = 0, False
    elif mode == "top-left":
        off, causal = 0, True
    elif mode == "bottom-right":
        off, causal = skv - sq, True
    else:
        sys.exit("causal must be one of: none, top-left, bottom-right")

    windowed = window_left is not None and window_left >= 0
    if not causal and not windowed:
        return 1.0

    live = 0
    for i in range(sq):
        centre = i + off  # the diagonal key for this query row
        hi = min(centre + window_right, skv - 1) if causal else skv - 1
        if windowed:
            lo = max(centre - window_left, 0)
        else:
            lo = 0
        if hi >= lo:
            live += hi - lo + 1
    return live / float(sq * skv)


def attention(
    batch,
    heads,
    seqlen_q,
    seqlen_kv,
    head_dim,
    kv_heads=None,
    causal=None,
    dtype="bf16",
    backward=False,
    head_dim_v=None,
    window_left=-1,
    window_right=0,
):
    """Flash attention, forward or backward.

    Forward is two matmuls per live score element: S = Q K^T and O = P V, each
    2 * D FLOP per element. The softmax itself is O(sq*skv) elementwise work and is left
    out of `flop` -- it is not matmul work and counting it inflates achieved TFLOPS.

    Backward is five matmuls of the same shape family, so 5/2 of the forward's matmul FLOP.
    The row-wise `delta = rowsum(dO . O)` pre-pass is elementwise and likewise not counted.

    GQA does not change FLOP -- every query head still attends -- but it does change bytes:
    K and V are stored per *kv* head, so the compulsory read is kv_heads, not heads.
    """
    e = _b(dtype)
    kvh = kv_heads or heads
    dv = head_dim_v or head_dim
    live = _live_fraction(seqlen_q, seqlen_kv, causal, window_left, window_right)
    pairs = batch * heads * seqlen_q * seqlen_kv * live

    qk = 2.0 * pairs * head_dim
    pv = 2.0 * pairs * dv
    flop = (qk + pv) * (2.5 if backward else 1.0)

    q = batch * heads * seqlen_q * head_dim * e
    k = batch * kvh * seqlen_kv * head_dim * e
    v = batch * kvh * seqlen_kv * dv * e
    o = batch * heads * seqlen_q * dv * e
    byt = q + k + v + o
    if backward:
        # dO in, plus dQ / dK / dV out; LSE and delta are per row and negligible beside these
        byt += o + q + k + v
    win = "none" if window_left is None or window_left < 0 else "%d/%d" % (window_left, window_right)
    return Count(
        "attention" + ("_bwd" if backward else "_fwd"),
        flop,
        byt,
        "b=%d h=%d kvh=%d sq=%d skv=%d d=%d causal=%s window=%s live=%.4f %s"
        % (batch, heads, kvh, seqlen_q, seqlen_kv, head_dim, causal, win, live, dtype),
    )


def _report(c, seconds=None):
    print("%-16s %s" % ("op", c.op))
    print("%-16s %s" % ("shape", c.detail))
    print("%-16s %.6g  (%.3f GFLOP)" % ("flop", c.flop, c.flop / 1e9))
    print("%-16s %.6g  (%.3f MB)" % ("bytes_min", c.bytes_min, c.bytes_min / 1e6))
    print(
        "%-16s %.2f FLOP/byte  -- the op's ceiling, a kernel sits at or below it"
        % ("intensity_max", c.intensity)
    )
    if seconds:
        print("%-16s %.3f us" % ("time", seconds * 1e6))
        print("%-16s %.1f" % ("TFLOPS", c.flop / seconds / 1e12))
        print(
            "%-16s %.2f TB/s  -- at bytes_min, so a floor on the real rate"
            % ("BW_min", c.bytes_min / seconds / 1e12)
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="op", required=True)

    def add_common(p):
        p.add_argument("--seconds", type=float, help="measured kernel time, to also report achieved TFLOPS")

    g = sub.add_parser("gemm")
    g.add_argument("--m", type=int, required=True)
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--k", type=int, required=True)
    g.add_argument("--batch", type=int, default=1)
    g.add_argument("--dtype", default="bf16")
    add_common(g)
    g.add_argument("--out-dtype")

    gg = sub.add_parser("grouped_gemm")
    gg.add_argument("--group-m", required=True, help="comma-separated per-group M")
    gg.add_argument("--n", type=int, required=True)
    gg.add_argument("--k", type=int, required=True)
    gg.add_argument("--dtype", default="bf16")
    add_common(gg)
    gg.add_argument("--out-dtype")

    a = sub.add_parser("attention")
    a.add_argument("--batch", type=int, required=True)
    a.add_argument("--heads", type=int, required=True)
    a.add_argument("--kv-heads", type=int)
    a.add_argument("--seqlen", type=int, help="sets both q and kv")
    a.add_argument("--seqlen-q", type=int)
    a.add_argument("--seqlen-kv", type=int)
    a.add_argument("--head-dim", type=int, required=True)
    a.add_argument("--head-dim-v", type=int)
    a.add_argument("--causal", default="none", choices=["none", "top-left", "bottom-right"])
    a.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="sliding window: keys kept before the diagonal; -1 for none",
    )
    a.add_argument(
        "--window-right",
        type=int,
        default=0,
        help="keys kept after the diagonal; 0 for a strictly causal window",
    )
    a.add_argument("--backward", action="store_true")
    a.add_argument("--dtype", default="bf16")
    add_common(a)

    args = ap.parse_args()
    if args.op == "gemm":
        c = gemm(args.m, args.n, args.k, args.dtype, args.out_dtype, args.batch)
    elif args.op == "grouped_gemm":
        c = grouped_gemm(
            [int(x) for x in args.group_m.split(",")], args.n, args.k, args.dtype, args.out_dtype
        )
    else:
        sq = args.seqlen_q or args.seqlen
        skv = args.seqlen_kv or args.seqlen
        if not (sq and skv):
            sys.exit("attention needs --seqlen, or --seqlen-q and --seqlen-kv")
        c = attention(
            args.batch,
            args.heads,
            sq,
            skv,
            args.head_dim,
            args.kv_heads,
            args.causal,
            args.dtype,
            args.backward,
            args.head_dim_v,
            args.window_left,
            args.window_right,
        )
    _report(c, args.seconds)


if __name__ == "__main__":
    main()
