###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLA under Ulysses: per-head all-to-all vs latent all-gather, on DeepSeek-V3 dimensions.

per-head   up-project the local tokens for every head, then all-to-all q / k / v (forward) and
           dq / dk / dv back (backward); what ``flash_attn_usp_func`` / a GemmA2A would exchange
latent     all-gather [q_latent | c_kv | k_pe] of the local tokens, up-project the whole sequence
           for the local heads; backward reduce-scatters the latent grads

Columns (us, median over iterations, max over ranks; ranks aligned and host launch hidden):

    gemm      the GEMM(s) coupled to the exchange (hipBLASLt)
    wgrad     the weight-grad GEMMs (backward)
    comm      the RCCL collective of the exchanged bytes alone
    serial    gemm + wgrad + comm
    overlap   max(gemm + wgrad, comm): the best a fused GEMM + exchange can reach
    op        the whole stage as one call: plain torch ops (pack / copies included) per head,
              ``ulysses_mla_latent_up_proj`` (sbhd, RCCL or the fused AgGemm) for the latent forward
    roofline  max(FLOPs / GEMM peak, off-rank bytes / link peak)

plus the causal attention of the local heads over the full sequence as the scale of the stage.

    torchrun --nproc_per_node 4 benchmark/ops/training/bench_ulysses_mla_latent.py --seq 16384 32768
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from bench_ulysses_overlap import measure_peaks, timeit

from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import (
    UlyssesMLALatentUpProjFunction,
    ulysses_mla_latent_up_proj,
)
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

DSV3 = dict(heads=128, q_lora=1536, kv_lora=512, nope=128, rope=64, v=128)


def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).bfloat16()


def collective(kind, numel_in, group):
    """``numel_in``: elements this rank contributes (all-gather / all-to-all) or holds before the
    reduction (reduce-scatter)."""
    n = group.size()
    if kind == "all_gather":
        inp, out = rnd(numel_in), rnd(numel_in * n)
        return lambda: dist.all_gather_into_tensor(out, inp, group=group)
    if kind == "reduce_scatter":
        inp, out = rnd(numel_in), rnd(numel_in // n)
        return lambda: dist.reduce_scatter_tensor(out, inp, group=group)
    inp, out = rnd(numel_in), rnd(numel_in)
    return lambda: dist.all_to_all_single(out, inp, group=group)


def build(cfg, seq, b, group):
    n, rank = group.size(), group.rank()
    h, cq, ckv, dn, dr, dv = (cfg[k] for k in ("heads", "q_lora", "kv_lora", "nope", "rope", "v"))
    hl, dq = h // n, dn + dr
    s_local = seq // n
    T, S = s_local * b, seq * b
    wq, wkv = rnd(h * dq, cq, scale=0.02), rnd(h * (dn + dv), ckv, scale=0.02)
    wq_r, wkv_r = wq.view(n, -1, cq)[rank], wkv.view(n, -1, ckv)[rank]

    # --- per-head: local tokens, all heads ---
    lq_loc, lkv_loc = rnd(T, cq), rnd(T, ckv + dr)
    helper = get_attention_cp_a2a_helper(b, seq, h, h, dq, dv, 1, n)

    def ph_up():
        q = (lq_loc @ wq.t()).view(b, s_local, h, dq)
        kv = (lkv_loc[:, :ckv] @ wkv.t()).view(b, s_local, h, dn + dv)
        return q, kv

    def ph_fwd_unfused():
        q, kv = ph_up()
        k = torch.cat(
            (kv[..., :dn], lkv_loc[:, None, ckv:].view(b, s_local, 1, dr).expand(-1, -1, h, -1)), -1
        )
        packed = helper.combine_qkv_before_a2a(q, k, kv[..., dn:])
        out = torch.empty_like(packed)
        dist.all_to_all_single(out, packed, group=group)
        return helper.splits_qkv_after_a2a(out)

    ph_numel = T * h * (dq + dq + dv)
    gq_full, gk_full, gv_full = rnd(b, seq, hl, dq), rnd(b, seq, hl, dq), rnd(b, seq, hl, dv)
    gq_loc, gkv_loc = rnd(T, h * dq), rnd(T, h * (dn + dv))

    def ph_bwd_gemm():
        return gq_loc @ wq, gkv_loc @ wkv

    def ph_bwd_wgrad():
        return gq_loc.t() @ lq_loc, gkv_loc.t() @ lkv_loc[:, :ckv]

    def ph_bwd_unfused():
        packed = helper.combine_dqkv_before_a2a(gq_full, gk_full, gv_full)
        out = torch.empty_like(packed)
        dist.all_to_all_single(out, packed, group=group)
        dq_, dk_, dv_ = helper.split_dqkv_after_a2a(out)
        gkv = torch.cat((dk_[..., :dn], dv_), -1).reshape(T, -1)
        gq = dq_.reshape(T, -1)
        dk_pe = dk_[..., dn:].float().sum(2)
        return gq @ wq, gkv @ wkv, dk_pe, gq.t() @ lq_loc, gkv.t() @ lkv_loc[:, :ckv]

    # --- latent: whole sequence, local heads ---
    lq_full, lkv_full = rnd(S, cq), rnd(S, ckv + dr)
    q_lat = rnd(s_local, b, cq)
    kv_lat = rnd(s_local, b, ckv + dr)
    # weights that need grad: the forward keeps the gathered latents, as in training
    wq_g, wkv_g = wq.detach().requires_grad_(), wkv.detach().requires_grad_()

    def lat_up():
        return lq_full @ wq_r.t(), lkv_full[:, :ckv] @ wkv_r.t()

    def lat_fwd(fused):
        return lambda: ulysses_mla_latent_up_proj(
            q_lat, kv_lat, wq_g, wkv_g, h, group, dn, dr, dv, "sbhd", fused=fused
        )

    lat_numel = T * (cq + ckv + dr)
    gq_lat, gkv_lat = gq_full.view(S, -1), rnd(S, hl * (dn + dv))

    def lat_bwd_gemm():
        return gq_lat @ wq_r, gkv_lat @ wkv_r

    def lat_bwd_wgrad():
        return gq_lat.t() @ lq_full, gkv_lat.t() @ lkv_full[:, :ckv]

    ctx = SimpleNamespace(
        saved_tensors=(lq_full, lkv_full[:, :ckv], wq, wkv),
        dims=(hl, cq, ckv, dn, dr, dv),
        group=group,
        qkv_format="bshd",
        local_shape=(b, s_local, cq),
        needs_input_grad=(True, True, True, True),
    )

    def lat_bwd_unfused():
        return UlyssesMLALatentUpProjFunction.backward(ctx, gq_full, gk_full, gv_full)

    up_flops = 2 * T * h * (cq * dq + ckv * (dn + dv))
    q_att, k_att, v_att = rnd(b, seq, hl, dq), rnd(b, seq, hl, dq), rnd(b, seq, hl, dv)
    # name: (gemm, wgrad, (collective, numel), op, FLOPs)
    stages = {
        "per-head fwd (a2a)": (ph_up, None, ("all_to_all", ph_numel), ph_fwd_unfused, up_flops),
        "latent fwd (all-gather, rccl)": (lat_up, None, ("all_gather", lat_numel), lat_fwd(False), up_flops),
        "latent fwd (fused AgGemm)": (lat_up, None, ("all_gather", lat_numel), lat_fwd(True), up_flops),
        "per-head bwd (a2a)": (
            ph_bwd_gemm,
            ph_bwd_wgrad,
            ("all_to_all", ph_numel),
            ph_bwd_unfused,
            2 * up_flops,
        ),
        "latent bwd (reduce-scatter)": (
            lat_bwd_gemm,
            lat_bwd_wgrad,
            ("reduce_scatter", lat_numel * n),
            lat_bwd_unfused,
            2 * up_flops,
        ),
    }
    return stages, (q_att, k_att, v_att)


def attention_times(qkv, group):
    import primus_turbo.pytorch as pt

    q, k, v = (t.detach().requires_grad_() for t in qkv)

    def fwd():
        with torch.no_grad():
            return timeit(lambda: pt.ops.flash_attn_func(q, k, v, causal=True), group)

    def bwd():
        out = pt.ops.flash_attn_func(q, k, v, causal=True)
        do = torch.randn_like(out)
        return timeit(lambda: torch.autograd.grad(out, (q, k, v), do, retain_graph=True), group)

    times = []
    for direction, fn in (("fwd", fwd), ("bwd", bwd)):
        try:
            times.append(fn())
        except Exception as e:  # noqa: BLE001
            if group.rank() == 0:
                print(f"  attention {direction} skipped: {type(e).__name__}: {str(e)[:120]}")
            times.append(None)
    return times


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", nargs="+", type=int, default=[32768])
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--no-attn", action="store_true")
    parser.add_argument(
        "--attn-tflops", type=float, default=1232.0, help="causal MLA forward rate for the attention estimate"
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()

    peak_flops, peak_link = measure_peaks(group)
    if rank == 0:
        print(
            f"peaks: GEMM {peak_flops / 1e12:.0f} TFLOP/s, link {peak_link / 1e9:.0f} GB/s per rank off-rank"
        )
    cols = ("gemm", "wgrad", "comm", "serial", "overlap", "op", "roofline")
    for seq in args.seq:
        stages, qkv = build(DSV3, seq, args.mbs, group)
        if rank == 0:
            print(f"\n== DeepSeek-V3 MLA seq={seq} ulysses={n} mbs={args.mbs}  (us, max over ranks)")
            print(f"  {'stage':<30}" + "".join(f"{c:>9}" for c in cols) + f"{'off-rank MB':>13}")
        for name, (gemm, wgrad, (kind, numel), op, flops) in stages.items():
            t = {
                "gemm": timeit(gemm, group),
                "wgrad": timeit(wgrad, group) if wgrad else 0.0,
                "comm": timeit(collective(kind, numel, group), group),
                "op": timeit(op, group),
            }
            # bytes a rank receives from its peers
            off_rank = numel * 2 * (n - 1) / (1 if kind == "all_gather" else n)
            t["serial"] = t["gemm"] + t["wgrad"] + t["comm"]
            t["overlap"] = max(t["gemm"] + t["wgrad"], t["comm"])
            t["roofline"] = max(flops / peak_flops, off_rank / peak_link) * 1e6
            if rank == 0:
                print(f"  {name:<30}" + "".join(f"{t[c]:>9.0f}" for c in cols) + f"{off_rank / 1e6:>13.1f}")
        if not args.no_attn:
            fwd, bwd = attention_times(qkv, group)
            if rank == 0:
                fmt = lambda t: "-" if t is None else f"{t:.0f} us"  # noqa: E731
                print(f"  causal attention, {DSV3['heads'] // n} local heads: fwd {fmt(fwd)}, bwd {fmt(bwd)}")
        if rank == 0:
            heads = DSV3["heads"] // n
            dqk, dv = DSV3["nope"] + DSV3["rope"], DSV3["v"]
            fwd_flops = args.mbs * heads * seq * seq / 2 * 2 * (dqk + dv)
            print(
                f"  causal attention fwd at {args.attn_tflops:.0f} TFLOP/s: {fwd_flops / args.attn_tflops / 1e6:.0f} us"
            )
        del stages, qkv
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
