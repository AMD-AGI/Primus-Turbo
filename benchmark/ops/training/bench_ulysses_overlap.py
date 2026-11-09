###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Serial vs overlapped vs roofline for each Ulysses fused GEMM + all-to-all op.

For each of the four ops (fwd QKV GemmA2A, fwd o_proj A2AGemm, bwd o_proj dgrad GemmA2A + wgrad,
bwd QKV A2AGemm dgrad + wgrad) on one shape, all timed the same way (ranks aligned by a barrier,
the launch queued behind a GPU sleep so host time is hidden, median of iterations, max over ranks):

    gemm      the GEMM coupled to the exchange, hipBLASLt (``F.linear`` / ``@``)
    wgrad     the weight-grad GEMM of a backward op, hipBLASLt (not coupled to the exchange)
    a2a       ``all_to_all_single`` of the exchanged bytes alone (RCCL)
    pack      pack + unpack copies of the unfused path
    serial    gemm + a2a + wgrad: no overlap, no copies
    unfused   the unfused op as one chain (gemm, pack, a2a, unpack, wgrad)
    overlap   max(gemm + wgrad, a2a): perfect overlap of the measured parts
    fused     the Primus-Turbo fused op
    roofline  max(all FLOPs / GEMM peak, off-rank bytes / link peak), peaks measured below
    hidden    (serial - fused) / (serial - overlap): share of the hideable time that is hidden

Peaks: the GEMM peak is hipBLASLt on an 8192^3 bf16 GEMM; the link peak is the per-rank off-rank
bandwidth of a 1 GiB RCCL all-to-all.

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_ulysses_overlap.py --model qwen3-30b-a3b --seq 32768
"""

import argparse
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    _o_gemm_backward,
    _o_gemm_forward,
    _qkv_backward,
    _qkv_gemm_a2a_forward,
)
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

MODELS = {
    "llama3.1-8b": dict(hidden=4096, hq=32, hkv=8, d=128),
    "llama3.1-70b": dict(hidden=8192, hq=64, hkv=8, d=128),
    "wan2.1-14b": dict(hidden=5120, hq=40, hkv=40, d=128),
    "qwen3-30b-a3b": dict(hidden=2048, hq=32, hkv=4, d=128),
}


def timeit(fn, group, warmup=5, iters=30):
    times = []
    for it in range(warmup + iters):
        torch.cuda.synchronize()
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        if it >= warmup:
            times.append(start.elapsed_time(end) * 1e3)
    t = torch.tensor([statistics.median(times)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.item()


def a2a_fn(numel, group):
    inp = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    return lambda: dist.all_to_all_single(out, inp, group=group)


def measure_peaks(group):
    n = group.size()
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    gemm_us = timeit(lambda: a @ b, group)
    numel = (1 << 30) // 2
    a2a_us = timeit(a2a_fn(numel, group), group)
    flops = 2 * 8192**3 / (gemm_us * 1e-6)
    link = numel * 2 * (n - 1) / n / (a2a_us * 1e-6)
    return flops, link


def build_ops(cfg, seq, b, group):
    n = group.size()
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    s_local = seq // n
    T = s_local * b
    Kq, Ko = (hq + 2 * hkv) * d, hq * d
    helper = get_attention_cp_a2a_helper(b, seq, hq, hkv, d, d, 1, n)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    x = rnd(b, s_local, hidden)
    w_qkv = rnd(Kq, hidden, scale=0.02)
    w_o = rnd(hidden, Ko, scale=0.02)
    res = rnd(b, s_local, hidden)
    o = rnd(b, seq, hq // n, d)
    og = rnd(b, s_local, Ko)
    dy = rnd(T, hidden)
    dq, dk, dv = (rnd(b, seq, h // n, d) for h in (hq, hkv, hkv))
    dqkv = rnd(T, Kq)
    qkv_out = F.linear(x, w_qkv)

    def split_qkv(qkv):
        qkv = qkv.view(b, s_local, hkv, hq // hkv + 2, d)
        return qkv[..., : hq // hkv, :].reshape(b, s_local, hq, d), qkv[..., -2, :], qkv[..., -1, :]

    def a2a(t):
        out = torch.empty_like(t)
        dist.all_to_all_single(out, t, group=group)
        return out

    # --- fwd QKV ---
    def qkv_pack():
        packed = helper.combine_qkv_before_a2a(*split_qkv(qkv_out))
        return helper.splits_qkv_after_a2a(packed)

    def qkv_unfused():
        return helper.splits_qkv_after_a2a(a2a(helper.combine_qkv_before_a2a(*split_qkv(F.linear(x, w_qkv)))))

    # --- fwd o_proj ---
    def o_pack():
        return helper.reshape_o_after_a2a(helper.reshape_o_before_a2a(o))

    def o_unfused():
        og_ = helper.reshape_o_after_a2a(a2a(helper.reshape_o_before_a2a(o)))
        return F.linear(og_.reshape(b, s_local, Ko), w_o) + res

    # --- bwd o_proj ---
    do_local = rnd(T, Ko)

    def do_pack():
        return helper.reshape_do_after_a2a(helper.reshape_do_before_a2a(do_local.view(b, s_local, hq, d)))

    def do_unfused():
        do = helper.reshape_do_after_a2a(
            a2a(helper.reshape_do_before_a2a(F.linear(dy, w_o.t()).view(b, s_local, hq, d)))
        )
        return do, dy.t() @ og.view(T, Ko)

    # --- bwd QKV ---
    def dqkv_pack():
        return helper.split_dqkv_after_a2a(helper.combine_dqkv_before_a2a(dq, dk, dv))

    def dqkv_unfused():
        gq, gk, gv = (
            t.reshape(T, -1, d)
            for t in helper.split_dqkv_after_a2a(a2a(helper.combine_dqkv_before_a2a(dq, dk, dv)))
        )
        full = torch.cat(
            (gq.view(T, hkv, hq // hkv, d), gk.view(T, hkv, 1, d), gv.view(T, hkv, 1, d)), dim=2
        ).view(T, Kq)
        return full @ w_qkv, full.t() @ x.view(T, hidden)

    ctx = SimpleNamespace(
        saved_tensors=(x, w_qkv),
        hq=hq,
        hkv=hkv,
        d=d,
        group=group,
        weight_layout="interleaved",
        s_local=s_local,
        b=b,
        qkv_format="bshd",
        needs_input_grad=(True, True),
    )
    # name: (gemm, wgrad or None, exchanged elements, pack, unfused, fused, FLOPs)
    return {
        "fwd QKV (GemmA2A)": (
            lambda: F.linear(x, w_qkv),
            None,
            T * Kq,
            qkv_pack,
            qkv_unfused,
            lambda: _qkv_gemm_a2a_forward(x, w_qkv, hq, hkv, d, group, "bshd", "interleaved"),
            2 * T * hidden * Kq,
        ),
        "fwd o_proj (A2AGemm)": (
            lambda: F.linear(og, w_o) + res,
            None,
            T * Ko,
            o_pack,
            o_unfused,
            lambda: _o_gemm_forward(o, w_o, res, group, "bshd", True),
            2 * T * Ko * hidden,
        ),
        "bwd o_proj (GemmA2A+wgrad)": (
            lambda: F.linear(dy, w_o.t()),
            lambda: dy.t() @ og.view(T, Ko),
            T * Ko,
            do_pack,
            do_unfused,
            lambda: _o_gemm_backward(
                dy, w_o, og.view(T, Ko), group, hq // n, d, s_local, b, "bshd", True, True
            ),
            4 * T * Ko * hidden,
        ),
        "bwd QKV (A2AGemm+wgrad)": (
            lambda: dqkv @ w_qkv,
            lambda: dqkv.t() @ x.view(T, hidden),
            T * Kq,
            dqkv_pack,
            dqkv_unfused,
            lambda: _qkv_backward(ctx, dq, dk, dv),
            4 * T * Kq * hidden,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="qwen3-30b-a3b", choices=sorted(MODELS))
    parser.add_argument("--seq", nargs="+", type=int, default=[32768])
    parser.add_argument("--mbs", type=int, default=1)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()

    peak_flops, peak_link = measure_peaks(group)
    if rank == 0:
        print(
            f"peaks: GEMM {peak_flops / 1e12:.0f} TFLOP/s (hipBLASLt 8192^3), link {peak_link / 1e9:.0f} GB/s per rank off-rank (RCCL 1 GiB a2a)"
        )
    cols = ("gemm", "wgrad", "a2a", "pack", "serial", "unfused", "overlap", "fused", "roofline")
    for seq in args.seq:
        ops = build_ops(MODELS[args.model], seq, args.mbs, group)
        if rank == 0:
            print(f"\n== {args.model} seq={seq} ulysses={n} mbs={args.mbs}  (us, max over ranks)")
            print(f"  {'op':<28}" + "".join(f"{c:>9}" for c in cols) + f"{'hidden':>8}{'fused/roof':>11}")
        for name, (gemm, wgrad, numel, pack, unfused, fused, flops) in ops.items():
            t = {
                "gemm": timeit(gemm, group),
                "wgrad": timeit(wgrad, group) if wgrad else 0.0,
                "a2a": timeit(a2a_fn(numel, group), group),
                "pack": timeit(pack, group),
                "unfused": timeit(unfused, group),
                "fused": timeit(fused, group),
            }
            t["serial"] = t["gemm"] + t["a2a"] + t["wgrad"]
            t["overlap"] = max(t["gemm"] + t["wgrad"], t["a2a"])
            t["roofline"] = max(flops / peak_flops, numel * 2 * (n - 1) / n / peak_link) * 1e6
            hidden = (t["serial"] - t["fused"]) / max(t["serial"] - t["overlap"], 1e-9)
            if rank == 0:
                print(
                    f"  {name:<28}"
                    + "".join(f"{t[c]:>9.0f}" for c in cols)
                    + f"{hidden:>7.0%}{t['fused'] / t['roofline']:>10.2f}x"
                )
        del ops
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
