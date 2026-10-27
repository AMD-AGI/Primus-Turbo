###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fwd_ab.py -- GPU driver of the HK MLA forward code object: canary, bit-exactness gate and same-process A/B
against the shipped FlyDSL forward (M._forward_op).

    python fwd_ab.py --hsaco K.hsaco [--shape toy|s640|prod|b1|b4 | --b B --s S --h H --layout sbhd|bshd]
                     [--canary] [--no-check] [--no-perf] [--no-bwd] [--det-iters N] [--rounds 5] [--iters 30]
                     [--warmup 5] [--seed 0] [--tag T]

--canary: ONE workgroup (grid 1 x 1 x 1: B = 1, Sq = Skv = 256, H = 1, nqb = 1), poisoned outputs, compared bit for
bit with FlyDSL on the same head; nothing else runs. Otherwise:
CHECK  out / lse NaN-poisoned before the HK call -> every row finite; HK vs FlyDSL bit for bit (whole tensors, and
       the first and the last 256-query block of every head separately); determinism (2 calls + --det-iters
       back-to-back calls bitwise equal); the HK backward (impl.hk_attn_bwd) fed with HK's (out, lse) vs fed with
       FlyDSL's: dq / dk / dv bitwise equal (--no-bwd skips); SNR of both forwards vs the fp32 reference on
       three (batch, head) pairs (for the record).
PERF   CUDA events; FlyDSL and HK alternate in blocks of --iters back-to-back calls over --rounds rounds; median of
       the per-round means, the per-round values and how many rounds HK was faster.
Prints JSON lines META / CHECK / PERF / RESULT (RESULT carries the gate verdict). Inputs as scripts/hk/mla_ab.py
builds them (SBHD storage folded to [1, s, b * h, d] for b > 1)."""

import argparse
import hashlib
import json
import math
import statistics
import sys

import torch

sys.path.insert(0, "/home/lihuzhan/projects/2026_1012__dsv3/scripts/hk")
from variant_ab import kernel_from_hsaco  # noqa: E402

from primus_turbo.pytorch.kernels.attention import (  # noqa: E402
    attention_flydsl_gfx1250_mla_impl as M,
)

D_QK, D_V, BM = 192, 128, 256
SCALE = 0.1352337788608801
SHAPES = {
    "toy": (1, 256, 2, "sbhd"),
    "s640": (2, 640, 128, "sbhd"),
    "prod": (2, 4096, 128, "sbhd"),
    "b1": (1, 4096, 128, "bshd"),
    "b4": (4, 4096, 128, "sbhd"),
}
ARGS = [
    ("Q", "ptr"),
    ("K", "ptr"),
    ("V", "ptr"),
    ("O", "ptr"),
    ("LSE", "ptr"),
    ("scale", "f32"),
    ("Sq", "i32"),
    ("Skv", "i32"),
    ("H", "i32"),
    ("B", "i32"),
    ("nqb", "i32"),
]


def args_meta():
    out, off = [], 0
    for n, k in ARGS:
        sz = 8 if k == "ptr" else 4
        out.append({"name": n, "kind": k, "offset": off, "size": sz})
        off += sz
    return out, (off + 7) // 8 * 8


def out(tag, obj):
    print(tag, json.dumps(obj), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    ap.add_argument("--name", default="k_mla_fwd")
    ap.add_argument("--shape", default=None, choices=sorted(SHAPES))
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--layout", default="sbhd")
    ap.add_argument("--canary", action="store_true")
    ap.add_argument("--check-heads", default="0,63,127")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--det-iters", type=int, default=0)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-perf", action="store_true")
    ap.add_argument("--no-bwd", action="store_true")
    ap.add_argument("--no-ref", action="store_true", help="skip the fp32 CPU reference SNRs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.shape:
        a.b, a.s, a.h, a.layout = SHAPES[a.shape]
    if a.canary:
        a.b, a.s, a.h, a.layout = 1, 256, 1, "bshd"
    b, s, h = a.b, a.s, a.h
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(a.seed)

    def make(d):
        if a.layout == "sbhd":
            return torch.randn(s, b, h, d, generator=g).to(torch.bfloat16).transpose(0, 1)
        return torch.randn(b, s, h, d, generator=g).to(torch.bfloat16)

    q_c, k_c, v_c, do_c = make(D_QK), make(D_QK), make(D_V), make(D_V)

    def to_dev(t):
        if a.layout == "sbhd":
            return t.transpose(0, 1).contiguous().to(dev).transpose(0, 1)
        return t.contiguous().to(dev)

    q, k, v, dout = (to_dev(t) for t in (q_c, k_c, v_c, do_c))
    fold = b > 1 and q.transpose(0, 1).is_contiguous()
    qk, kk, vk = (M._to_kernel_layout(t, fold) for t in (q, k, v))
    dok = M._to_kernel_layout(dout, fold)
    kb, sq, kh, _ = qk.shape
    skv = kk.shape[1]
    if sq != skv or sq % 64 or kb * sq * kh * D_V * 2 >= 2**31:
        raise SystemExit(f"shape outside the HK forward's domain: Sq {sq} Skv {skv} B {kb} H {kh}")
    nqb = (sq + BM - 1) // BM
    meta, ksz = args_meta()
    kern = kernel_from_hsaco(a.hsaco, a.name, meta, 128, ksz)
    stream = torch.cuda.current_stream()
    out(
        "META",
        {
            "b": b,
            "s": s,
            "h": h,
            "layout": a.layout,
            "kernel_q": list(qk.shape),
            "hsaco": a.hsaco,
            "sha256": hashlib.sha256(open(a.hsaco, "rb").read()).hexdigest(),
            "grid": [nqb, kh, kb],
            "canary": a.canary,
            "tag": a.tag,
        },
    )

    def fly():
        return M._forward_op(qk, kk, vk, SCALE, True)

    def hk(poison=False):
        if poison:
            o = torch.full((kb, sq, kh, D_V), float("nan"), device=dev, dtype=torch.bfloat16)
            lse = torch.full((kb, kh, sq), float("nan"), device=dev, dtype=torch.float32)
        else:
            o = torch.empty((kb, sq, kh, D_V), device=dev, dtype=torch.bfloat16)
            lse = torch.empty((kb, kh, sq), device=dev, dtype=torch.float32)
        kern.call((nqb, kh, kb), 128, (qk, kk, vk, o, lse, SCALE, sq, skv, kh, kb, nqb), stream)
        return o, lse

    def neq(x, y):
        return int(
            (
                x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
                != y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
            ).sum()
        )

    def snr(ref, x):
        ref, x = ref.double(), x.double()
        n = (ref - x).pow(2).sum().item()
        return float("inf") if n == 0 else 10 * math.log10(ref.pow(2).sum().item() / n)

    res, gate = {}, {}
    if a.canary or not a.no_check:
        fo, fl = (t.clone() for t in fly())
        ho, hl = (t.clone() for t in hk(poison=True))
        torch.cuda.synchronize()
        res["hk_finite"] = bool(torch.isfinite(ho.float()).all()) and bool(torch.isfinite(hl).all())
        res["mismatch_vs_fly"] = {
            "out": neq(fo, ho),
            "lse": neq(fl, hl),
            "out_n": fo.numel(),
            "lse_n": fl.numel(),
        }
        lo_last = (nqb - 1) * BM
        res["mismatch_first_block"] = {
            "out": neq(fo[:, :BM], ho[:, :BM]),
            "lse": neq(fl[:, :, :BM], hl[:, :, :BM]),
        }
        res["mismatch_last_block"] = {
            "out": neq(fo[:, lo_last:], ho[:, lo_last:]),
            "lse": neq(fl[:, :, lo_last:], hl[:, :, lo_last:]),
        }
        gate["finite"] = res["hk_finite"]
        gate["bitwise_vs_fly"] = res["mismatch_vs_fly"]["out"] == 0 and res["mismatch_vs_fly"]["lse"] == 0
        if not a.canary:
            ho2, hl2 = (t.clone() for t in hk())
            torch.cuda.synchronize()
            res["hk_deterministic_2"] = torch.equal(ho, ho2) and torch.equal(hl, hl2)
            gate["deterministic"] = res["hk_deterministic_2"]
            if a.det_iters:
                bad = 0
                for _ in range(a.det_iters):
                    o3, l3 = hk()
                    bad += int(not (torch.equal(o3, ho) and torch.equal(l3, hl)))
                torch.cuda.synchronize()
                res["hk_det_backtoback"] = {"calls": a.det_iters, "mismatch": bad}
                gate["deterministic"] = gate["deterministic"] and bad == 0
            if not a.no_bwd:
                from primus_turbo.hipkittens.gfx1250_mla_bwd import impl as hk_impl

                reason = M.hk_mla_bwd_unsupported_reason(qk, kk, vk, True)
                if reason is not None:
                    res["bwd_skipped"] = reason
                else:
                    gr = {}
                    for key, (o_, l_) in (("fly", (fo, fl)), ("hk", (ho, hl))):
                        gr[key] = [
                            t.clone() for t in hk_impl.hk_attn_bwd(dok, qk, kk, vk, o_, l_, SCALE, True)
                        ]
                    torch.cuda.synchronize()
                    res["bwd_mismatch_hkfwd_vs_flyfwd"] = {
                        n: neq(x, y) for n, x, y in zip(("dq", "dk", "dv"), gr["fly"], gr["hk"])
                    }
                    gate["bwd_bitwise"] = all(v_ == 0 for v_ in res["bwd_mismatch_hkfwd_vs_flyfwd"].values())
                    del gr
            if not a.no_ref:
                pairs = [int(x) for x in a.check_heads.split(",") if x and int(x) < h]
                acc = {key: {"out": [], "lse": []} for key in ("fly", "hk")}
                views = {
                    "fly": (M._from_kernel_layout(fo, b), fl.view(b, h, sq)),
                    "hk": (M._from_kernel_layout(ho, b), hl.view(b, h, sq)),
                }
                for bi in range(b):
                    for hh in pairs:
                        qr, kr, vr = (t[bi : bi + 1, :, hh : hh + 1].float().cpu() for t in (q_c, k_c, v_c))
                        sc = torch.einsum("bqhd,bkhd->bhqk", qr, kr) * SCALE
                        sc = sc.masked_fill(torch.ones(sq, skv, dtype=torch.bool).triu(1), float("-inf"))
                        lse_r = torch.logsumexp(sc, -1)[0, 0]
                        o_r = torch.einsum("bhqk,bkhd->bqhd", sc.softmax(-1), vr)[0, :, 0]
                        for key, (ov, lv) in views.items():
                            acc[key]["out"].append(snr(o_r, ov[bi, :, hh].float().cpu()))
                            acc[key]["lse"].append(snr(lse_r, lv[bi, hh].float().cpu()))
                res["snr_min_vs_fp32"] = {
                    k_: {n: round(min(v_), 2) for n, v_ in d.items()} for k_, d in acc.items()
                }
            del ho2, hl2
        del fo, fl, ho, hl
    out("CHECK", res)

    if not a.canary and not a.no_perf:

        def tfn(fn, iters):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            ev[0].record()
            for _ in range(iters):
                fn()
            ev[1].record()
            ev[1].synchronize()
            return ev[0].elapsed_time(ev[1]) / iters

        fns = [("fly", fly), ("hk", hk)]
        for _, fn in fns:
            tfn(fn, a.warmup)
        rr = {n: [] for n, _ in fns}
        for _ in range(a.rounds):
            for n, fn in fns:
                rr[n].append(tfn(fn, a.iters))
        med = {n: statistics.median(v_) for n, v_ in rr.items()}
        flops = 2.0 * kb * kh * sq * skv * (D_QK + D_V) * 0.5
        perf = {f"{n}_ms": round(m_, 4) for n, m_ in med.items()}
        perf.update({f"{n}_rounds": [round(x, 4) for x in v_] for n, v_ in rr.items()})
        perf["speedup"] = round(med["fly"] / med["hk"], 4)
        perf["hk_faster_rounds"] = sum(1 for x, y in zip(rr["hk"], rr["fly"]) if x < y)
        perf["hk_tflops"] = round(flops / med["hk"] / 1e9, 1)
        out("PERF", perf)
    verdict = "PASS" if gate and all(gate.values()) else ("NO_CHECK" if not gate else "FAIL")
    out(
        "RESULT",
        {"gate": gate, "verdict": verdict, "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)},
    )


if __name__ == "__main__":
    main()
