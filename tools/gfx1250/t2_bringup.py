#!/usr/bin/env python3
"""T2 bring-up: first GPU execution of the prebuilt gfx1250 ASM backward.

Nothing in asm_bwd_launcher.py has ever run on a card -- it was written while the box was
wedged -- so this is a bring-up, not a measurement. Deliberately minimal and outside the
tuning harness: the less machinery between this and the kernel, the less there is to blame
when the first run returns garbage, and this is the most likely thing in the session to
fault the GPU.

STAGED, because the risks are not equally suspect. T2-ASM-BACKWARD-SPEC.md ranks them:

  1. odo's compact kernarg layout -- the ONLY one of the three not confirmed by
     disassembly. It uses gfx1250 kernarg preload into SGPRs and issues zero s_load
     instructions, so "which offsets does it read" cannot be asked of the binary. Evidence
     is a transcription of the C++ packer plus the fact that 3 ptr + 11 u32 + 2 ptr = 84 B
     equals kernarg_segment_size exactly.
  2. byte vs element strides -- passing element strides does not error, it silently reads
     the wrong memory.
  3. dq_acc must be fp32 and zeroed (514 buffer_atomic_add_f32 accumulate into it).
  4. grid (ceil(Sk/128), nhead_q, batch), halved on the x axis for causal; block always 128.

Risk 1 is separately checkable and that is what --stage odo does: delta = rowsum(dO*O) is
three lines of torch, so the top suspect can be confirmed or killed WITHOUT launching the
main kernel at all. Only then does --stage full put the whole chain on the card.

Reference is computed in fp32 with explicit math rather than through the fused path, so a
disagreement cannot be blamed on another kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Before torch, exactly as tune_attention.py does: torch reads this at backend-selection
# time. Without it the fp32 reference matmul raises HIPBLAS_STATUS_INVALID_VALUE from
# hipblasLtMatmulAlgoGetHeuristic on this part -- and hipBLASLt is 10.5x slower than a
# Triton GEMM here anyway, so the reference should not be built on it either way.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The repo root too, so `import primus_turbo` resolves to this checkout rather than to
# whatever the image registered. The image's editable install is uninstalled in fa-repro
# (its .pth installs a MetaPathFinder that sys.path cannot shadow), but this keeps the
# script honest if it is ever run in a container where that was not done.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402

import asm_bwd_launcher as L  # noqa: E402

SHAPES = {
    # Start small. If the kernarg layout is wrong this is the blast radius.
    "tiny": dict(batch=1, seqlen=256, hq=8, hkv=2, d=128),
    "s128":  dict(batch=1, seqlen=128,  hq=8, hkv=2, d=128),
    "s256":  dict(batch=1, seqlen=256,  hq=8, hkv=2, d=128),
    "s384":  dict(batch=1, seqlen=384,  hq=8, hkv=2, d=128),
    "s512":  dict(batch=1, seqlen=512,  hq=8, hkv=2, d=128),
    "s768":  dict(batch=1, seqlen=768,  hq=8, hkv=2, d=128),
    "smoke": dict(batch=1, seqlen=1024, hq=8, hkv=2, d=128),
    # ratio=1. If dk/dv come right here and wrong under GQA, the fault is in how the
    # kernel's `ratio` field is used (or in the reference's grouping), not in the strides.
    "smoke-mha": dict(batch=1, seqlen=1024, hq=8, hkv=8, d=128),
    # Production GQA ratio (32/8=4) at a size whose fp32 reference still fits: the scores
    # matrix is 537 MB here against 34 GB at llama31-8b, which is why correctness is
    # established at this shape and only timing is taken at the production one.
    "gqa2k": dict(batch=1, seqlen=2048, hq=32, hkv=8, d=128),
    "mid": dict(batch=2, seqlen=4096, hq=32, hkv=8, d=128),
    "llama31-8b": dict(batch=4, seqlen=8192, hq=32, hkv=8, d=128),
}


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    got = got.float()
    num = (ref * ref).sum()
    den = ((ref - got) ** 2).sum()
    if den == 0:
        return float("inf")
    return float(10.0 * torch.log10(num / den))


def reference(q, k, v, causal=True):
    """fp32 forward with the gradients, plus the LSE the ASM kernels expect.

    Natural log, matching aiter's own test (it builds log(denom) + max and compares to the
    kernel's LSE with no base conversion) and matching what the vendored fused backward
    already consumes.
    """
    b, s, hq, d = q.shape
    hk = k.shape[2]
    scale = 1.0 / math.sqrt(d)
    q32 = q.float().transpose(1, 2)                      # [B,Hq,S,D]
    k32 = k.float().transpose(1, 2)
    v32 = v.float().transpose(1, 2)
    rep = hq // hk
    k32 = k32.repeat_interleave(rep, dim=1)
    v32 = v32.repeat_interleave(rep, dim=1)
    q32.requires_grad_(True)
    k32.requires_grad_(True)
    v32.requires_grad_(True)

    scores = torch.matmul(q32, k32.transpose(-1, -2)) * scale
    if causal:
        # Sq == Sk here, so bottom-right alignment and top-left agree. Keep them equal.
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)                # [B,Hq,S] natural log
    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, v32)                             # [B,Hq,S,D]
    return o, lse, (q32, k32, v32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="tiny", choices=sorted(SHAPES))
    ap.add_argument("--stage", default="odo", choices=["odo", "full", "time"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--co-variant", default="", choices=["", "_perf"],
                    help="swap in the _perf build of the main kernel; same symbol, same ABI")
    ap.add_argument("--dkdv-heads", default="kv", choices=["kv", "q"],
                    help="'q' allocates dk/dv with nhead_q slices and reduces host-side, "
                         "testing whether the kernel writes per-q-head partials.")
    args = ap.parse_args()

    cfg = SHAPES[args.shape]
    b, s, hq, hk, d = cfg["batch"], cfg["seqlen"], cfg["hq"], cfg["hkv"], cfg["d"]
    torch.manual_seed(args.seed)
    dev = "cuda"

    q = torch.randn(b, s, hq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, s, hk, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, s, hk, d, device=dev, dtype=torch.bfloat16)
    do = torch.randn(b, s, hq, d, device=dev, dtype=torch.bfloat16)

    t_start = time.time()
    # The fp32 reference is skipped for --stage time: at llama31-8b its scores matrix alone
    # is 34 GB, and that stage compares against the shipping champion instead.
    if args.stage != "time":
        o32, lse32, (q32, k32, v32) = reference(q, k, v, causal=True)
        o = o32.detach().transpose(1, 2).contiguous().to(torch.bfloat16)   # [B,S,Hq,D]
        lse = lse32.detach().contiguous().float()                          # [B,Hq,S]

    rec = {"shape": args.shape, "stage": args.stage, "batch": b, "seqlen": s,
           "hq": hq, "hkv": hk, "head_dim": d, "ok": False}

    if args.stage == "time":
        pass   # handled below; must NOT fall through the fp32-reference path
    elif args.stage == "odo":
        # delta = rowsum(dO * O), per kernel 1 of the three. Isolating it settles the one
        # layout that disassembly could not confirm, with the main kernel never launched.
        delta_ref = (do.float().transpose(1, 2) * o.float().transpose(1, 2)).sum(-1)  # [B,Hq,S]

        delta = torch.empty((b, hq, s), device=dev, dtype=torch.float32)
        delta.fill_(float("nan"))       # so "kernel wrote nothing" is distinguishable from zeros
        hip = L.HipModule()
        f_odo = hip.function(L.ASM_DIR / L.CO["odo"], L.SYMBOLS["odo"])

        def bs(t, elem):
            return (t.stride(0) * elem, t.stride(1) * elem, t.stride(2) * elem)

        b_o, s_o, h_o = bs(o, L.BF16)
        b_do, s_do, h_do = bs(do, L.BF16)
        h_lsed = s * L.FP32
        hip.launch(f_odo, ((s + L.TS_ODO - 1) // L.TS_ODO, hq, b), (L.BDX, 1, 1),
                   L.pack_compact(L.ODO_FIELDS, L.ODO_SIZE, {
                       "ptr_o": o.data_ptr(), "ptr_do": do.data_ptr(), "ptr_d": delta.data_ptr(),
                       "Hs_o": h_o, "BAs_o": b_o, "Seqs_o": s_o,
                       "Hs_do": h_do, "BAs_do": b_do, "Seqs_do": s_do,
                       "Hs_d": h_lsed, "BAs_d": hq * h_lsed, "Seqs_d": L.FP32,
                       "seqlen_q": s, "head_dim": d,
                       "ptr_qseq": 0, "ptr_qseq_padded": 0}),
                   torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        wrote = int((~torch.isnan(delta)).sum())
        rec.update({
            "delta_elems": delta.numel(),
            "delta_written": wrote,
            "delta_all_written": wrote == delta.numel(),
            "sqnr_delta_db": sqnr_db(delta_ref, torch.nan_to_num(delta)),
            "delta_ref_head": [round(float(x), 5) for x in delta_ref.flatten()[:4]],
            "delta_got_head": [round(float(x), 5) for x in delta.flatten()[:4]],
        })
        rec["ok"] = rec["delta_all_written"] and rec["sqnr_delta_db"] >= 40.0

    else:
        q32.grad = k32.grad = v32.grad = None
        o32.backward(do.float().transpose(1, 2))
        rep = hq // hk
        dq_ref = q32.grad.transpose(1, 2).contiguous()
        dk_ref = k32.grad.view(b, hk, rep, s, d).sum(2).transpose(1, 2).contiguous()
        dv_ref = v32.grad.view(b, hk, rep, s, d).sum(2).transpose(1, 2).contiguous()

        dq, dk, dv = L.asm_backward(q, k, v, o, do, lse, dkdv_heads=args.dkdv_heads,
                                    co_variant=args.co_variant)
        if args.dkdv_heads == "q" and rep > 1:
            dk = dk.view(b, s, hk, rep, d).sum(3).to(k.dtype)
            dv = dv.view(b, s, hk, rep, d).sum(3).to(v.dtype)
        torch.cuda.synchronize()
        rec.update({
            "sqnr_dq_db": sqnr_db(dq_ref, dq),
            "sqnr_dk_db": sqnr_db(dk_ref, dk),
            "sqnr_dv_db": sqnr_db(dv_ref, dv),
        })
        rec["ok"] = min(rec["sqnr_dq_db"], rec["sqnr_dk_db"], rec["sqnr_dv_db"]) >= 40.0

    if args.stage == "time":
        # Timing, plus correctness against the SHIPPING fused backward rather than an fp32
        # reference: at this shape the scores matrix alone is 34 GB, and the champion is the
        # thing the acceptance line is stated against anyway. Both run in one process on one
        # card, so the comparison is apples to apples.
        from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func

        qg = q.clone().requires_grad_(True)
        kg = k.clone().requires_grad_(True)
        vg = v.clone().requires_grad_(True)
        o_ref = flash_attn_func(qg, kg, vg, causal=True)
        torch.cuda.synchronize()

        def champ():
            for g in (qg, kg, vg):
                g.grad = None
            o_ref.backward(do, retain_graph=True)
            return qg.grad, kg.grad, vg.grad

        scale = 1.0 / math.sqrt(d)
        from primus_turbo.pytorch.ops.attention.flash_attn_interface import (
            triton_dense_forward,
        )
        o_t, lse_t = triton_dense_forward(q, k, v, softmax_scale=scale, causal=True)
        torch.cuda.synchronize()
        rep = hq // hk

        def asm_bwd():
            dq, dk, dv = L.asm_backward(q, k, v, o_t, do, lse_t, dkdv_heads="q",
                                        co_variant=args.co_variant)
            if rep > 1:
                dk = dk.view(b, s, hk, rep, d).float().sum(3).to(k.dtype)
                dv = dv.view(b, s, hk, rep, d).float().sum(3).to(v.dtype)
            return dq, dk, dv

        def timed(fn, iters, warmup):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            flush = torch.empty(64 * 1024 * 1024, device=dev, dtype=torch.float32)
            ts = []
            for _ in range(iters):
                flush.zero_()          # 256 MiB L2 flush per rep, as tune_attention does
                a = torch.cuda.Event(enable_timing=True)
                bb = torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                bb.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(bb))
            ts.sort()
            return ts[len(ts) // 2]

        cdq, cdk, cdv = champ()
        adq, adk, adv = asm_bwd()
        torch.cuda.synchronize()
        rec.update({
            "sqnr_dq_vs_champ_db": sqnr_db(cdq, adq),
            "sqnr_dk_vs_champ_db": sqnr_db(cdk, adk),
            "sqnr_dv_vs_champ_db": sqnr_db(cdv, adv),
        })
        rec["champ_ms"] = timed(champ, args.iters, args.warmup)
        rec["asm_ms"] = timed(asm_bwd, args.iters, args.warmup)
        rec["speedup"] = rec["champ_ms"] / rec["asm_ms"]
        rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / (1 << 30)
        rec["co_variant"] = args.co_variant or "(shipped)"
        rec["ok"] = min(rec["sqnr_dq_vs_champ_db"], rec["sqnr_dk_vs_champ_db"],
                        rec["sqnr_dv_vs_champ_db"]) >= 40.0

    rec["wall_s"] = round(time.time() - t_start, 2)
    rec["correct"] = rec["ok"]
    print(json.dumps(rec))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
