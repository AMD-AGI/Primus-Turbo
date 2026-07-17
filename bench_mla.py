"""dsv4 sparse-MLA fwd+bwd benchmark: TFLOPS + SNR (vs triton oracle) over the 6
shapes (flash/pro x cr{0,4,128}) @seq=4096.

Just run it:   python bench_mla.py
Single shape:  python bench_mla.py --only pro:4
Tuning knobs:  --seq 4096 --warmup 10 --iters 20
"""

import argparse
import math
import os

os.environ.setdefault("PRIMUS_DSA_BWD_FLYDSL_DQ", "1")
os.environ.setdefault("PRIMUS_DSA_INTERM_FLYDSL", "1")
import torch

import bench_flydsl_cr as bc
from primus_turbo.flydsl.attention.sparse_mla_bwd import sparse_mla_bwd_v4_flydsl
from primus_turbo.flydsl.attention.sparse_mla_fwd import sparse_mla_fwd_v4_flydsl
from primus_turbo.triton.attention.sparse_mla import sparse_mla_bwd_v4_triton, sparse_mla_fwd_v4_triton

_SWA, _D = 128, 512
V = {"flash": 64, "pro": 128}


def snr(ref, t):
    ref, t = ref.float(), t.float()
    n = (ref - t).pow(2).mean()
    return 10 * torch.log10(ref.pow(2).mean() / n).item() if n > 0 else 99.0


def one(variant, cr, S, warm, it):
    H = V[variant]
    scale = 1 / math.sqrt(_D)
    if cr == 4:
        P = max(S // 4, 1)
        K = min({"flash": 512, "pro": 1024}[variant], P)
        topk = _SWA + K
    elif cr == 0:
        P, K, topk = 0, 0, _SWA
    else:
        P = max(S // cr, 1)
        K, topk = 0, _SWA + P
    gq, gkv, gtopk, sink, do = bc._build(cr, H, S, _D, K, P, _SWA)
    fwd_flop = 2.0 * S * H * topk * (_D + _D)  # QK + PV GEMMs

    # ---- forward ----
    def fwd():
        return sparse_mla_fwd_v4_flydsl(gq, gkv, gtopk, attn_sink=sink, kv_lora_rank=_D, scale=scale)

    of, lf = fwd()
    ot, lt = sparse_mla_fwd_v4_triton(gq, gkv, gtopk, attn_sink=sink, kv_lora_rank=_D, scale=scale)
    fsnr = snr(ot, of)
    ftf = fwd_flop / (bc._time(fwd, warm, it) * 1e-3) / 1e12

    # ---- backward (reuses the flydsl fwd output) ----
    def bwd():
        return sparse_mla_bwd_v4_flydsl(
            gq, gkv, of, do, gtopk, lf, attn_sink=sink, kv_lora_rank=_D, scale=scale
        )

    dqf, dkf, dsf = bwd()
    dqt, dkt, dst = sparse_mla_bwd_v4_triton(
        gq, gkv, ot, do, gtopk, lt, attn_sink=sink, kv_lora_rank=_D, scale=scale
    )
    bsnr = min(snr(dqt, dqf), snr(dkt, dkf))
    if dsf is not None and dst is not None:
        bsnr = min(bsnr, snr(dst, dsf))
    btf = 2.5 * fwd_flop / (bc._time(bwd, warm, it) * 1e-3) / 1e12
    return ftf, fsnr, btf, bsnr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--only", default="", help="filter e.g. 'pro:4' (variant:cr)")
    a = ap.parse_args()
    groups = [(v, cr) for v in ("flash", "pro") for cr in (0, 4, 128)]
    if a.only:
        fv, fc = a.only.split(":")
        groups = [(fv, int(fc))]

    print("  shape         fwd TF   SNR      bwd TF   SNR")
    ftfs, btfs, ok = [], [], True
    for v, cr in groups:
        ftf, fsnr, btf, bsnr = one(v, cr, a.seq, a.warmup, a.iters)
        fp, bp = fsnr > 40.0, bsnr > 35.0
        ok = ok and fp and bp
        print(
            f"  {v:5s} cr={cr:<3d}  {ftf:7.1f}  {fsnr:5.1f}dB {'ok' if fp else 'FAIL':4s}  "
            f"{btf:7.1f}  {bsnr:5.1f}dB {'ok' if bp else 'FAIL'}",
            flush=True,
        )
        ftfs.append(ftf)
        btfs.append(btf)
    if len(ftfs) > 1:
        print(
            f"  MEAN         {sum(ftfs) / len(ftfs):7.1f}          {sum(btfs) / len(btfs):7.1f}", flush=True
        )
    print(f"  {'ALL PASS' if ok else 'SOME FAIL'}", flush=True)
