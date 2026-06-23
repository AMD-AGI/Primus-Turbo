###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Benchmark / profile for the ported DeepSeek-V4 attention Triton kernels.

Covers the three per-layer attention types on DeepSeek-V4 shapes:

* dense / SWA   (``compress_ratio == 0``)  -> ``hca_attention``
* HCA           (``compress_ratio == 128``) -> ``hca_attention`` split-mask
* CSA           (``compress_ratio == 4``)   -> ``csa_attention_from_pool``

Two production model envelopes are supported (HuggingFace
``deepseek-ai/DeepSeek-V4-{Flash,Pro}`` config.json). They share
head_dim=512, MQA (K_H=1) and sliding_window=128; they differ only in the
attention head count and the indexer top-k:

    V4-Flash: num_attention_heads = 64,  index_topk = 512
    V4-Pro:   num_attention_heads = 128, index_topk = 1024

(Pro is wider/deeper -- hidden_size 7168, 61 layers, 384 experts -- but
those do not change the per-layer attention kernel shape.) Pool size is
P = S // compress_ratio; the CSA top-k actually used is min(index_topk, P).

For each shape we report forward and forward+backward latency (ms),
achieved TFLOP/s, and an SNR correctness check vs the eager reference.

Usage:
    python benchmark/ops/bench_deepseek_attention.py                       # both models
    python benchmark/ops/bench_deepseek_attention.py --model pro
    python benchmark/ops/bench_deepseek_attention.py --batch 1 2 --seqlen 4096 8192
    python benchmark/ops/bench_deepseek_attention.py --kinds dense csa
    python benchmark/ops/bench_deepseek_attention.py --num-heads 96 --index-topk 768
"""

import argparse
import csv
import os
from datetime import datetime

import torch
import torch.utils.benchmark as benchmark

# Disable FP32 atomic for better perf on gfx950 (matches bench_attention_turbo).
def _is_gfx950():
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 5


if torch.cuda.is_available() and _is_gfx950():
    os.environ["PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"] = "0"

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.ops.attention import (
    eager_hca_attention,
    eager_csa_attention,
    sliding_window_causal_mask,
    hca_attention,
    csa_attention,
    csa_attention_from_pool,
)

# Backend selection for the A/B comparison (design §7.2). ``None`` keeps the
# dispatcher default (Triton). The FlyDSL forward is gfx950 + D=512 only and
# falls back to Triton elsewhere / for the CSA paths.
_BACKEND_MAP = {"triton": BackendType.TRITON, "flydsl": BackendType.FLYDSL}

# DeepSeek-V4 production defaults shared by both Flash and Pro.
V4_HEAD_DIM = 512
V4_SWA_WINDOW = 128
V4_HCA_RATIO = 128
V4_CSA_RATIO = 4

# Per-model attention envelope (HuggingFace config.json). Only num_heads and
# index_topk differ between Flash and Pro at the attention-op level.
V4_MODELS = {
    "flash": {"num_heads": 64, "index_topk": 512},
    "pro": {"num_heads": 128, "index_topk": 1024},
}


def compute_snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    ref, actual = ref.float(), actual.float()
    signal = torch.norm(ref).pow(2)
    noise = torch.norm(ref - actual).pow(2)
    return (10 * torch.log10(signal / (noise + 1e-12))).item()


def _bench_ms(fn) -> float:
    """Median latency in milliseconds via torch.utils.benchmark."""
    t = benchmark.Timer(stmt="fn()", globals={"fn": fn})
    return t.blocked_autorange(min_run_time=1.0).median * 1e3


# ---------------------------------------------------------------------------
# Per-kind setup + flop accounting
# ---------------------------------------------------------------------------


def _setup_dense(B, S, H, D, dtype, dev, index_topk=512, backend=None):
    scale = D**-0.5
    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)

    def fwd():
        return hca_attention(
            q, k, v, sink=None, swa_window=V4_SWA_WINDOW, additive_mask=None,
            attn_dropout=0.0, training=True, scale=scale, backend=backend,
        )

    def ref():
        return eager_hca_attention(
            q.detach(), k.detach().expand(B, H, S, D), v.detach().expand(B, H, S, D),
            sink=None, swa_window=V4_SWA_WINDOW, additive_mask=None,
            attn_dropout=0.0, training=False, scale=scale,
        )

    # SWA: each query attends to ~window keys. 2 matmuls (QK, PV) * 2 FLOP.
    eff_k = min(S, V4_SWA_WINDOW)
    fwd_flops = 2 * 2 * B * H * S * eff_k * D
    return (q, k, v), fwd(), fwd, ref, fwd_flops


def _setup_hca(B, S, H, D, dtype, dev, index_topk=512, backend=None):
    scale = D**-0.5
    P = S // V4_HCA_RATIO
    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, 1, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool_k = torch.randn(B, 1, P, D, device=dev, dtype=dtype, requires_grad=True)
    pool_v = torch.randn(B, 1, P, D, device=dev, dtype=dtype, requires_grad=True)
    k_cat = torch.cat([k_local, pool_k], dim=2)
    v_cat = torch.cat([v_local, pool_v], dim=2)
    pool_mask = torch.zeros(S, P, device=dev, dtype=dtype)

    def fwd():
        return hca_attention(
            q, k_cat, v_cat, sink=None, swa_window=V4_SWA_WINDOW, additive_mask=pool_mask,
            attn_dropout=0.0, training=True, scale=scale, hca_local_seqlen=S, backend=backend,
        )

    def ref():
        local_mask = sliding_window_causal_mask(S, V4_SWA_WINDOW, device=dev, dtype=dtype)
        joint_mask = torch.cat([local_mask, pool_mask], dim=1)
        return eager_hca_attention(
            q.detach(), k_cat.detach().expand(B, H, S + P, D),
            v_cat.detach().expand(B, H, S + P, D), sink=None, swa_window=V4_SWA_WINDOW,
            additive_mask=joint_mask, attn_dropout=0.0, training=False, scale=scale,
        )

    # local SWA window + full pool P.
    eff_k = min(S, V4_SWA_WINDOW) + P
    fwd_flops = 2 * 2 * B * H * S * eff_k * D
    return (q, k_cat, v_cat), fwd(), fwd, ref, fwd_flops


def _setup_csa(B, S, H, D, dtype, dev, index_topk=512, backend=None):
    scale = D**-0.5
    P = S // V4_CSA_RATIO
    K = min(index_topk, P)
    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    pool = torch.randn(B, P, D, device=dev, dtype=dtype, requires_grad=True)
    topk = torch.randint(0, P, (B, S, K), device=dev, dtype=torch.int64)

    def fwd():
        return csa_attention_from_pool(
            q, k_local, v_local, pool, topk_idxs=topk, sink=None,
            swa_window=V4_SWA_WINDOW, attn_dropout=0.0, training=True, scale=scale, backend=backend,
        )

    def ref():
        bidx = torch.arange(B, device=dev).view(B, 1, 1)
        gathered = pool.detach()[bidx, topk]
        sparse_mask = torch.zeros(B, S, K, device=dev, dtype=dtype)
        return eager_csa_attention(
            q.detach(), k_local.detach(), v_local.detach(), gathered, sink=None,
            swa_window=V4_SWA_WINDOW, sparse_mask=sparse_mask, attn_dropout=0.0,
            training=False, scale=scale,
        )

    # local SWA window + K sparse keys.
    eff_k = min(S, V4_SWA_WINDOW) + K
    fwd_flops = 2 * 2 * B * H * S * eff_k * D
    return (q, k_local, v_local, pool), fwd(), fwd, ref, fwd_flops


def _setup_csa_gathered(B, S, H, D, dtype, dev, index_topk=512, backend=None):
    """CSA via the pre-gathered path (``csa_attention``). This is the form the
    ported FlyDSL CSA forward kernel handles (the 2.79x-over-Triton kernel),
    so ``--backends flydsl --kinds csa_gathered`` actually exercises FlyDSL
    (the ``csa`` / from-pool kind has no FlyDSL backend and falls back)."""
    scale = D**-0.5
    P = S // V4_CSA_RATIO
    K = min(index_topk, P)
    q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    k_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    v_local = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
    gathered = torch.randn(B, S, K, D, device=dev, dtype=dtype, requires_grad=True)
    sparse_mask = torch.zeros(B, S, K, device=dev, dtype=dtype)

    def fwd():
        return csa_attention(
            q, k_local, v_local, gathered, sink=None, swa_window=V4_SWA_WINDOW,
            sparse_mask=sparse_mask, attn_dropout=0.0, training=True, scale=scale, backend=backend,
        )

    def ref():
        return eager_csa_attention(
            q.detach(), k_local.detach(), v_local.detach(), gathered.detach(), sink=None,
            swa_window=V4_SWA_WINDOW, sparse_mask=sparse_mask, attn_dropout=0.0,
            training=False, scale=scale,
        )

    # local SWA window + K sparse keys.
    eff_k = min(S, V4_SWA_WINDOW) + K
    fwd_flops = 2 * 2 * B * H * S * eff_k * D
    return (q, k_local, v_local, gathered), fwd(), fwd, ref, fwd_flops


_SETUP = {
    "dense": _setup_dense,
    "hca": _setup_hca,
    "csa": _setup_csa,
    "csa_gathered": _setup_csa_gathered,
}


def profile_one(kind, B, S, H, D, dtype, dev, index_topk=512, backend=None):
    tensors, out0, fwd, ref, fwd_flops = _SETUP[kind](B, S, H, D, dtype, dev, index_topk, backend)
    bwd_flops = fwd_flops * 2.5

    # Correctness vs eager reference.
    with torch.no_grad():
        out_ref = ref()
    snr = compute_snr(out_ref, out0)

    grad_out = torch.randn_like(out0)

    def fwd_only():
        return fwd()

    def fwd_bwd():
        for t in tensors:
            t.grad = None
        o = fwd()
        o.backward(grad_out)

    torch.cuda.synchronize()
    fwd_ms = _bench_ms(fwd_only)
    fwdbwd_ms = _bench_ms(fwd_bwd)
    torch.cuda.synchronize()

    fwd_tflops = fwd_flops / (fwd_ms * 1e-3) / 1e12
    total_tflops = (fwd_flops + bwd_flops) / (fwdbwd_ms * 1e-3) / 1e12

    return {
        "kind": kind, "B": B, "S": S, "H": H, "D": D,
        "backend": "default" if backend is None else backend.name.lower(),
        "fwd_ms": fwd_ms, "fwdbwd_ms": fwdbwd_ms,
        "fwd_TFLOPs": fwd_tflops, "total_TFLOPs": total_tflops,
        "snr_dB": snr,
    }


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 attention benchmark")
    parser.add_argument("--model", type=str, nargs="+", default=["flash", "pro"],
                        choices=["flash", "pro"],
                        help="V4 model envelope(s): sets num_heads + index_topk.")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Override the model's num_attention_heads.")
    parser.add_argument("--index-topk", type=int, default=None,
                        help="Override the model's index_topk (CSA top-k cap).")
    parser.add_argument("--head-dim", type=int, default=V4_HEAD_DIM)
    parser.add_argument("--kinds", type=str, nargs="+", default=["dense", "hca", "csa"],
                        choices=["dense", "hca", "csa", "csa_gathered"])
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV path. Default: dpsk_attn_benchmark_result_{date}_{gpu}.csv")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--backends", type=str, nargs="+", default=["triton"],
                        choices=["triton", "flydsl"],
                        help="Attention backend(s) to compare (Triton vs FlyDSL; design §7.2).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / HIP device required for the V4 attention benchmark.")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = "cuda"
    D = args.head_dim

    backends = [_BACKEND_MAP[b] for b in args.backends]

    gpu_name = torch.cuda.get_device_name(0)
    test_id = 0
    print(f"Device: {gpu_name}")
    header = (
        f"{'model':6} {'kind':6} {'backend':8} {'B':>2} {'S':>6} {'H':>4} {'fwd_ms':>9} "
        f"{'fwdbwd_ms':>11} {'fwd_TFLOPs':>11} {'tot_TFLOPs':>11} {'SNR_dB':>8}"
    )

    rows = []
    for model in args.model:
        H = args.num_heads if args.num_heads is not None else V4_MODELS[model]["num_heads"]
        index_topk = (
            args.index_topk if args.index_topk is not None else V4_MODELS[model]["index_topk"]
        )
        print(
            f"\n=== DeepSeek-V4-{model.capitalize()}  dtype={args.dtype}  H={H}  D={D}  "
            f"SWA={V4_SWA_WINDOW}  index_topk={index_topk} ==="
        )
        print(header)
        print("-" * len(header))
        for kind in args.kinds:
            for backend in backends:
                for B in args.batch:
                    for S in args.seqlen:
                        try:
                            r = profile_one(kind, B, S, H, D, dtype, dev, index_topk, backend)
                        except torch.cuda.OutOfMemoryError:
                            print(f"{model:6} {kind:6} {backend.name.lower():8} {B:>2} {S:>6}  OOM")
                            torch.cuda.empty_cache()
                            continue
                        test_id += 1
                        r["TestID"] = test_id
                        r["GPU"] = gpu_name
                        r["model"] = model
                        rows.append(r)
                        print(
                            f"{model:6} {r['kind']:6} {r['backend']:8} {r['B']:>2} {r['S']:>6} "
                            f"{r['H']:>4} {r['fwd_ms']:>9.3f} {r['fwdbwd_ms']:>11.3f} "
                            f"{r['fwd_TFLOPs']:>11.1f} {r['total_TFLOPs']:>11.1f} {r['snr_dB']:>8.1f}"
                        )
                        torch.cuda.empty_cache()

    _write_csv(rows, args.output, gpu_name)
    return rows


def _write_csv(rows, output_csv, gpu_name):
    """Write the collected rows to a CSV (TestID first, for the suite merge)."""
    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"dpsk_attn_benchmark_result_{timestamp}_{gpu_name}.csv"
    fields = [
        "TestID", "GPU", "model", "kind", "backend", "B", "S", "H", "D",
        "fwd_ms", "fwdbwd_ms", "fwd_TFLOPs", "total_TFLOPs", "snr_dB",
    ]
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
