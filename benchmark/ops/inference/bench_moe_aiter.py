###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference MoE micro-benchmark for aiter's fused_moe (forward-only).

Same kernel vLLM/SGLang call on ROCm: ``aiter.fused_moe.fused_moe``. Supports
bf16 / fp8 (block, per-channel) / mxfp4 weights via --quant. Simulates one rank
under TP (shard intermediate) or EP (shard experts). Without --balanced, EP
ranks are imbalanced, so all ep_size ranks are swept and Time/TFLOPS/BW are
reported as a min~max range.

    python bench_moe_aiter.py [--model M] [--quant Q] [--tp-size N | --ep-size N] [--balanced] [--check]
"""

import argparse
import contextlib
import os
from datetime import datetime

import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

# MoE routed-expert shapes: hidden_size, moe_intermediate_size, n_routed_experts, topk.
# gpt-oss uses SwiGLU (we bench with SiLU) and K=2880 is not /128 (no fp8_block).
MODELS = {
    # https://huggingface.co/deepseek-ai/DeepSeek-R1  (671B total / 37B active)
    "deepseek-r1": {"hidden_size": 7168, "moe_intermediate_size": 2048, "n_routed_experts": 256, "topk": 8},
    # https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro  (1.6T total / 49B active; native fp4 experts, SwiGLU, top-6)
    "deepseek-v4-pro": {"hidden_size": 7168, "moe_intermediate_size": 3072, "n_routed_experts": 384, "topk": 6},
    # https://huggingface.co/moonshotai/Kimi-K2.6  (1T total / 32B active)
    "kimi-k2.6": {"hidden_size": 7168, "moe_intermediate_size": 2048, "n_routed_experts": 384, "topk": 8},
    # https://huggingface.co/zai-org/GLM-5.1  (754B total)
    "glm-5.1": {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 256, "topk": 8},
    # https://huggingface.co/MiniMaxAI/MiniMax-M2.7  (229B total)
    "minimax-m2.7": {"hidden_size": 3072, "moe_intermediate_size": 1536, "n_routed_experts": 256, "topk": 8},
    # https://huggingface.co/openai/gpt-oss-120b  (117B total / 5.1B active; SwiGLU, top-4)
    "gpt-oss-120b": {"hidden_size": 2880, "moe_intermediate_size": 2880, "n_routed_experts": 128, "topk": 4},
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  (1.02T total / 42B active)
    "mimo-v2.5-pro": {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 384, "topk": 8},
    # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking  (80B total / 3B active; 512 experts, top-10)
    "qwen3-next-80b-a3b": {"hidden_size": 2048, "moe_intermediate_size": 512, "n_routed_experts": 512, "topk": 10},
}

DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


@contextlib.contextmanager
def suppress_output():
    """Silence aiter's chatty per-call stdout/stderr at the fd level.

    Uses os.dup2 so both print() and logging (whose handlers may hold the
    original stream) are captured, keeping the result table readable.
    """
    with open(os.devnull, "w") as devnull:
        old_out, old_err = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_out, 1)
            os.dup2(old_err, 2)
            os.close(old_out)
            os.close(old_err)


def compute_snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    """Signal-to-Noise Ratio (dB); higher is better."""
    ref_f = ref.to(torch.float64)
    act_f = actual.to(torch.float64)
    signal = ref_f.norm().pow(2)
    noise = (ref_f - act_f).norm().pow(2)
    return 10 * torch.log10(signal / (noise + 1e-12)).item()


def make_routing(num_tokens, num_experts, topk, device):
    """Standard softmax top-k routing with renormalized weights.

    Done outside the timed region; only the routing *outputs* feed fused_moe.
    Returns (topk_weights[fp32], topk_ids[int32]).
    """
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def make_balanced_routing(num_tokens, num_experts, topk, device):
    """Deterministic perfectly-balanced routing: each expert receives an equal
    token share, so every EP rank has identical load. Weights are uniform.
    """
    ids = (torch.arange(num_tokens * topk, device=device) % num_experts).reshape(num_tokens, topk)
    weights = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    return weights, ids.to(torch.int32)


def torch_moe_ref(hidden, w1, w2, topk_weights, topk_ids, expert_offset=0):
    """Naive reference MoE (fp32 accumulation) for correctness only.

    Layouts match aiter fused_moe: hidden [T, K], w1 [local_E, 2*I, K]
    (gate||up), w2 [local_E, K, I], SiLU gating. ``expert_offset`` maps local
    weight row ``e`` to global expert id ``expert_offset + e`` (EP windows).
    """
    T, K = hidden.shape
    out = torch.zeros(T, K, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    w1f = w1.float()
    w2f = w2.float()
    for e in range(w1.shape[0]):
        mask = topk_ids == expert_offset + e
        if not mask.any():
            continue
        tok_idx, slot = mask.nonzero(as_tuple=True)
        gate_up = x[tok_idx] @ w1f[e].t()
        gate, up = gate_up.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up
        ye = h @ w2f[e].t()
        scale = topk_weights[tok_idx, slot].float().unsqueeze(-1)
        out.index_add_(0, tok_idx, ye * scale)
    return out


def make_expert_mask(num_experts, lo, local_E, device):
    """aiter EP expert_mask: int32 of shape (num_experts + 1,), 1 for this rank's
    local experts (global window [lo, lo + local_E)), 0 elsewhere + trailing
    sentinel. The window maps to local weight rows 0..local_E-1 via aiter's
    prefix-sum over the mask. See vLLM determine_expert_map().
    """
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    mask[lo : lo + local_E] = 1
    return mask


def fp8_dtype():
    """OCP fp8 (e4m3fn) on gfx950 (MI355X); e4m3fnuz on older CDNA (MI300)."""
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) == (9, 5):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def quantize_weights(w1, w2, quant):
    """Return (w1_k, w2_k, w1_scale, w2_scale, quant_type) ready for fused_moe.

    Weights are shuffled to the aiter kernel layout; scales are not. Activations
    are quantized dynamically inside fused_moe in all quantized modes.

    none      : bf16, no scales.
    fp8_block : per-128x128 block fp8 (DeepSeek convention).
    fp8_token : per-output-channel fp8 (per-token activations).
    fp4       : mxfp4 (fp4x2 weights + e8m0 block scales).
    """
    if quant == "none":
        w1_k = shuffle_weight(w1, layout=(16, 16))
        w2_k = shuffle_weight(w2, layout=(16, 16))
        return w1_k, w2_k, None, None, QuantType.No

    if quant == "fp8_block":
        dt = fp8_dtype()
        w1_q, w1_s = _block_quant_fp8(w1, dt)
        w2_q, w2_s = _block_quant_fp8(w2, dt)
        # Like bf16, the fp8 weights must be shuffled to the kernel layout; the
        # block scales stay un-shuffled (see aiter test_moe_blockscale.py).
        w1_k = shuffle_weight(w1_q, layout=(16, 16))
        w2_k = shuffle_weight(w2_q, layout=(16, 16))
        return w1_k, w2_k, w1_s, w2_s, QuantType.per_128x128

    if quant == "fp8_token":
        dt = fp8_dtype()
        w1_q, w1_s = _channel_quant_fp8(w1, dt)
        w2_q, w2_s = _channel_quant_fp8(w2, dt)
        w1_k = shuffle_weight(w1_q, layout=(16, 16))
        w2_k = shuffle_weight(w2_q, layout=(16, 16))
        return w1_k, w2_k, w1_s, w2_s, QuantType.per_Token

    if quant == "fp4":
        # mxfp4 (a4w4): fp4x2 weights + e8m0 block scales; the kernel quantizes
        # bf16 activations to fp4 internally. Mirrors aiter test_moe_2stage.py.
        tq = aiter.get_torch_quant(QuantType.per_1x32)

        def _q(w):
            w_qt, w_scale = tq(w, quant_dtype=dtypes.fp4x2)
            w_qt = w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2)  # fp4x2: 2 vals/byte
            return shuffle_weight(w_qt, layout=(16, 16)), fp4_utils.e8m0_shuffle(w_scale)

        w1_k, w1_s = _q(w1)
        w2_k, w2_s = _q(w2)
        return w1_k, w2_k, w1_s, w2_s, QuantType.per_1x32

    raise ValueError(f"Unknown quant: {quant}")


def _channel_quant_fp8(w, dt):
    """Per-output-channel fp8 quant for per_Token MoE.

    Returns (w_q[fp8], scale[fp32] of shape [E, N, 1]); activations are quantized
    per-token dynamically inside fused_moe.
    """
    scale = (w.abs().amax(dim=-1, keepdim=True) / torch.finfo(dt).max).clamp(min=1e-12)
    w_q = (w / scale).to(dt)
    return w_q.contiguous(), scale.to(torch.float32).contiguous()


def _block_quant_fp8(w, dt, block=128):
    """128x128 block fp8 quant (DeepSeek convention).

    Returns (w_q[fp8], scale_inv[fp32] of shape [E, N/128, K/128]) where
    ``w_q.float() * scale_inv`` recovers w; this is the layout aiter's
    per_128x128 fused_moe expects.
    """
    E, N, K = w.shape
    assert N % block == 0 and K % block == 0, f"({N},{K}) not divisible by {block}"
    fmax = torch.finfo(dt).max
    wb = w.view(E, N // block, block, K // block, block)
    scale_inv = (wb.abs().amax(dim=(2, 4)) / fmax).clamp(min=1e-12)  # [E, N/128, K/128]
    w_q = (wb / scale_inv[:, :, None, :, None]).to(dt).view(E, N, K)
    return w_q.contiguous(), scale_inv.to(torch.float32).contiguous()


def aiter_fused_moe(hidden, w1, w2, topk_weights, topk_ids, quant_type, w1_scale, w2_scale, expert_mask=None):
    return fused_moe(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_mask,
        ActivationType.Silu,
        quant_type,
        False,  # doweight_stage1
        w1_scale,
        w2_scale,
    )


def _per_expert_nbytes(t):
    """Stored bytes of one expert's slice (t[0]), correct for fp4x2 packing."""
    return t[0].numel() * t.element_size()


def _time_ms(fn, warmup=10, iters=50):
    """GPU time per call via CUDA events (excludes host/Python launch overhead).

    Launches `iters` calls back-to-back between two events so the elapsed GPU
    stream time reflects steady-state per-call cost.
    """
    with suppress_output():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_one(num_tokens, cfg, dtype, device, tp_size, ep_size, balanced, quant, check):
    K = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    E = cfg["n_routed_experts"]
    topk = cfg["topk"]

    # One rank's local shape: TP shards the intermediate, EP shards the experts.
    local_I = I // tp_size
    local_E = E // ep_size

    hidden = torch.randn(num_tokens, K, device=device, dtype=dtype)
    # Weight shapes are identical on every rank, so reuse one set; rank r just
    # masks the global-expert window [r*local_E, (r+1)*local_E) onto these rows.
    w1 = torch.randn(local_E, 2 * local_I, K, device=device, dtype=dtype) / (K**0.5)
    w2 = torch.randn(local_E, K, local_I, device=device, dtype=dtype) / (local_I**0.5)
    # Kernel-ready weights/scales; the torch reference always uses the original
    # (bf16, un-shuffled) weights, so fp8 modes show their quantization SNR.
    w1_k, w2_k, w1_scale, w2_scale, quant_type = quantize_weights(w1, w2, quant)

    routing = make_balanced_routing if balanced else make_routing
    topk_weights, topk_ids = routing(num_tokens, E, topk, device)

    # EP without balancing: every rank sees a different token load, so sweep all
    # ranks and report the spread. Otherwise a single rank is representative.
    ranks = range(ep_size) if (ep_size > 1 and not balanced) else [0]

    elem = torch.finfo(dtype).bits // 8
    times, tflops_list, bw_list = [], [], []
    snr = None
    for r in ranks:
        lo = r * local_E
        expert_mask = make_expert_mask(E, lo, local_E, device) if ep_size > 1 else None
        fn = lambda: aiter_fused_moe(
            hidden, w1_k, w2_k, topk_weights, topk_ids, quant_type, w1_scale, w2_scale, expert_mask
        )

        # Check every rank and keep the worst (min) SNR: one wrong rank
        # corrupts the whole MoE output. Empty windows (ref all zeros) skip.
        if check:
            with suppress_output():
                out = fn()
            ref = torch_moe_ref(hidden, w1, w2, topk_weights, topk_ids, lo)
            if ref.abs().any():
                s = compute_snr(ref, out.float())
                snr = s if snr is None else min(snr, s)

        time_ms = _time_ms(fn)

        # Token-expert pairs and activated experts in this rank's window.
        in_win = (topk_ids >= lo) & (topk_ids < lo + local_E)
        n = int(in_win.sum().item())
        active_experts = int(torch.unique(topk_ids[in_win]).numel())

        # FC1: [n,K]x[K,2*local_I], FC2: [n,local_I]x[local_I,K]
        flops = 2 * n * K * (2 * local_I) + 2 * n * local_I * K
        # Effective HBM traffic: activated-expert weights (read once, actual
        # stored bytes incl. scales) + bf16 activations in/out.
        per_expert_bytes = _per_expert_nbytes(w1_k) + _per_expert_nbytes(w2_k)
        if w1_scale is not None:
            per_expert_bytes += _per_expert_nbytes(w1_scale) + _per_expert_nbytes(w2_scale)
        weight_bytes = active_experts * per_expert_bytes
        act_bytes = 2 * num_tokens * K * elem  # hidden in + out

        times.append(time_ms)
        tflops_list.append(flops / (time_ms * 1e-3) / 1e12)
        bw_list.append((weight_bytes + act_bytes) / (time_ms * 1e-3) / 1e9)

    return times, tflops_list, bw_list, snr


def save_excel(path, args, cfg, tp_size, ep_size, rows):
    """Write a config sheet + a results sheet to an .xlsx file."""
    import pandas as pd

    meta = {
        "backend": "aiter",
        "model": args.model,
        "hidden_size": cfg["hidden_size"],
        "moe_intermediate_size": cfg["moe_intermediate_size"],
        "n_routed_experts": cfg["n_routed_experts"],
        "topk": cfg["topk"],
        "balanced": args.balanced,
        "seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
    }
    if not path.endswith(".xlsx"):
        path += ".xlsx"
    results_sheet = f"{args.quant}_tp{tp_size}_ep{ep_size}"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(meta.items(), columns=["field", "value"]).to_excel(
            writer, sheet_name="config", index=False
        )
        pd.DataFrame(rows).to_excel(writer, sheet_name=results_sheet, index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Inference MoE benchmark (aiter fused_moe)")
    # What to benchmark
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="deepseek-r1",
        help="MoE model whose routed-expert shapes to benchmark.",
    )
    parser.add_argument(
        "--quant",
        choices=["none", "fp8_block", "fp8_token", "fp4"],
        default="none",
        help="Quantization: none (bf16), fp8_block (per-128x128), fp8_token (per-channel), fp4 (mxfp4).",
    )
    # Parallelism (TP or EP, mutually exclusive)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size (shards intermediate). Mutually exclusive with --ep-size.",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=1,
        help="Expert-parallel size (shards experts). Mutually exclusive with --tp-size.",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Use balanced routing (all EP ranks identical). Default: imbalanced, sweep all ranks.",
    )
    # Sweep range
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=DEFAULT_TOKENS,
        help="Token counts to sweep.",
    )
    # Run options
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the SNR correctness check vs the torch reference (default: off).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible weights/routing (default: 0).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save results to this .xlsx file. If omitted, an auto-named file is used.",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a GPU."
    torch.manual_seed(args.seed)
    tp_size, ep_size = args.tp_size, args.ep_size
    assert tp_size >= 1 and ep_size >= 1, "tp/ep size must be >= 1"
    assert not (tp_size > 1 and ep_size > 1), "Enable either TP or EP, not both."

    device = "cuda"
    dtype = torch.bfloat16
    cfg = MODELS[args.model]
    assert cfg["moe_intermediate_size"] % tp_size == 0, "intermediate not divisible by tp_size"
    assert cfg["n_routed_experts"] % ep_size == 0, "experts not divisible by ep_size"

    print()
    print(f"  Model     : {args.model} (hidden={cfg['hidden_size']}, inter={cfg['moe_intermediate_size']}, "
          f"experts={cfg['n_routed_experts']}, topk={cfg['topk']})")
    print(f"  Precision : {args.quant}        Parallel : tp={tp_size} ep={ep_size}  balanced={args.balanced}")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print()

    def fmt(xs, prec, w):
        lo, hi = min(xs), max(xs)
        if lo == hi:
            return f"{lo:.{prec}f}"
        return f"{lo:>{w}.{prec}f} ~ {hi:<{w}.{prec}f}"

    cols = (("Tokens", 8), ("Time (ms)", 17), ("TFLOPS", 15), ("BW (GB/s)", 15), ("SNR (dB)", 8))
    header = " | ".join(f"{name:^{w}}" for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)

    rows = []
    for num_tokens in args.tokens:
        try:
            times, tflops_list, bw_list, snr = benchmark_one(
                num_tokens, cfg, dtype, device, tp_size, ep_size, args.balanced, args.quant, args.check
            )
            snr_str = "-" if snr is None else f"{snr:.1f}"
            print(
                f"{num_tokens:>8} | {fmt(times, 3, 7):>17} | {fmt(tflops_list, 1, 6):>15} | "
                f"{fmt(bw_list, 0, 6):>15} | {snr_str:>8}"
            )
            rows.append(
                {
                    "Tokens": num_tokens,
                    "Time (ms)": fmt(times, 3, 0),
                    "TFLOPS": fmt(tflops_list, 1, 0),
                    "BW (GB/s)": fmt(bw_list, 0, 0),
                    "SNR (dB)": snr_str,
                }
            )
        except Exception as e:
            print(f"{num_tokens:>8} | ERROR: {e}")
            rows.append({"Tokens": num_tokens, "Time (ms)": f"ERROR: {e}"})

    output = args.output
    if output is None:
        date = datetime.now().strftime("%Y%m%d")
        output = f"bench_moe_{args.model}_{args.quant}_tp{tp_size}_ep{ep_size}_{date}.xlsx"
    saved = save_excel(output, args, cfg, tp_size, ep_size, rows)
    print(f"\nSaved results to {saved}")


if __name__ == "__main__":
    main()
