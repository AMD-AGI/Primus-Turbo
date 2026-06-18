###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference MoE micro-benchmark for aiter's fused_moe (forward-only).

Same kernel vLLM/SGLang call on ROCm: ``aiter.fused_moe.fused_moe``. Supports
bf16 / fp8 (block, per-channel) / mxfp4 weights via --quant. Simulates one rank
under TP (shard intermediate) or EP (shard experts).

--variance (>=0) sets expert-load imbalance: the variance of per-expert load
normalized to mean 1 (squared coefficient of variation). 0=perfectly balanced;
larger=more hot experts (saturates at E/topk - 1). Under EP all ranks are swept
and the bottleneck (slowest) rank's Time/TFLOPS/BW are reported, with an Imbal
column = slowest/fastest rank time (1.0 = balanced; >1 = EP load imbalance).

    python bench_moe_aiter.py [--model M] [--quant Q] [--tp-size N | --ep-size N] [--variance V] [--check]
"""

import argparse
import contextlib
import math
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
    # https://huggingface.co/MiniMaxAI/MiniMax-M3  (428B total / 23B active; SwiGLU-OAI, top-4, MSA sparse attn)
    "minimax-m3": {"hidden_size": 6144, "moe_intermediate_size": 3072, "n_routed_experts": 128, "topk": 4},
    # https://huggingface.co/openai/gpt-oss-120b  (117B total / 5.1B active; SwiGLU, top-4)
    "gpt-oss-120b": {"hidden_size": 2880, "moe_intermediate_size": 2880, "n_routed_experts": 128, "topk": 4},
    # https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro  (1.02T total / 42B active)
    "mimo-v2.5-pro": {"hidden_size": 6144, "moe_intermediate_size": 2048, "n_routed_experts": 384, "topk": 8},
    # https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking  (80B total / 3B active; 512 experts, top-10)
    "qwen3-next-80b-a3b": {"hidden_size": 2048, "moe_intermediate_size": 512, "n_routed_experts": 512, "topk": 10},
}

DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

_USE_CUDAGRAPH = True  # time via CUDA-graph replay (matches SGLang/vLLM decode); set by main().


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


def make_routing(num_tokens, num_experts, topk, device, variance):
    """Routing whose load imbalance is set by ``variance``: the variance of the
    per-expert token count normalized to mean 1 (squared coefficient of
    variation). Load balance is defined per expert (each expert gets the same
    number of tokens), the standard MoE definition.

    variance == 0  -> *exact* per-expert balance: each token takes topk experts
        evenly spaced across the expert range (stride = E // topk), so every
        expert receives an identical token count (up to the unavoidable residue
        when num_tokens * topk < E, e.g. tiny batches) and the picks fan out
        across all EP ranks. This is deterministic -> no sampling jitter.

    variance > 0   -> per-expert popularity is a uniform/hot-set mix
        p_i = b/E + (1-b)/topk * [i in hot], whose load variance is exactly
        (1-b)^2 * (E/topk - 1); we invert that to hit the requested variance,
        saturating at var_max = E/topk - 1. The hot set is *fixed* (seed 1234,
        independent of num_tokens) so every batch size sees the same skew and
        rows are comparable. Per-token picks are sampled without replacement.

    Weights are uniform (combine weights don't affect cost).
    """
    weights = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    var_max = num_experts / topk - 1.0
    balance = 1.0 if (variance <= 0.0 or var_max <= 0.0) else 1.0 - math.sqrt(min(variance, var_max) / var_max)

    if balance >= 1.0:
        stride = max(1, num_experts // topk)
        offs = (torch.arange(num_tokens, device=device) % stride).unsqueeze(1)
        base = (torch.arange(topk, device=device) * stride).unsqueeze(0)
        ids = (offs + base) % num_experts  # topk distinct, evenly spaced
        return weights, ids.to(torch.int32)

    gen = torch.Generator(device=device).manual_seed(1234)
    hot = torch.randperm(num_experts, generator=gen, device=device)[:topk]
    pop = torch.full((num_experts,), balance / num_experts, device=device, dtype=torch.float32)
    pop[hot] += (1.0 - balance) / topk
    probs = pop.unsqueeze(0).expand(num_tokens, -1).contiguous()
    ids = torch.multinomial(probs, topk, replacement=False, generator=gen)
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


def _time_ms(fn, warmup=10, iters=50, use_graph=None):
    """GPU time per call via CUDA events (excludes host/Python launch overhead).

    With use_graph (default, controlled by --cudagraph) fn is captured into a
    CUDA graph and timed via replay, removing per-launch dispatch overhead and
    matching how SGLang/vLLM run decode. Falls back to eager launches if the
    kernel cannot be captured (e.g. an internal device->host sync).
    """
    if use_graph is None:
        use_graph = _USE_CUDAGRAPH
    with suppress_output():
        if use_graph:
            try:
                # Warmup on a side stream (required before capture; also triggers
                # any first-call JIT so it is not captured into the graph).
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(warmup):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                for _ in range(3):  # warm the replay
                    graph.replay()
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) / iters
            except Exception:
                pass  # capture unsupported for this kernel -> eager timing below

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


def benchmark_one(num_tokens, cfg, dtype, device, tp_size, ep_size, variance, quant, check):
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

    topk_weights, topk_ids = make_routing(num_tokens, E, topk, device, variance)

    # In EP each rank owns a different expert window; under load imbalance the
    # ranks get different token counts. We first measure each rank's *load*
    # (token-expert pairs = grouped-GEMM M, noise-free), then report the
    # bottleneck = heaviest rank, since a decode step waits for the slowest one.
    # Imbalance = max/min load over active ranks (1.00 = perfectly balanced).
    # Under TP a single rank owns all experts, so one rank is the whole picture.
    ranks = range(ep_size) if ep_size > 1 else [0]

    per_expert_bytes = _per_expert_nbytes(w1_k) + _per_expert_nbytes(w2_k)
    if w1_scale is not None:
        per_expert_bytes += _per_expert_nbytes(w1_scale) + _per_expert_nbytes(w2_scale)

    elem = torch.finfo(dtype).bits // 8

    # Per-rank load (no timing): (r, lo, local_m, local_tokens, active_experts).
    loads = []
    for r in ranks:
        lo = r * local_E
        in_win = (topk_ids >= lo) & (topk_ids < lo + local_E)
        local_m = int(in_win.sum().item())
        if local_m == 0:
            continue  # idle rank (no tokens routed here) -> never the bottleneck
        local_tokens = int(in_win.any(dim=1).sum().item())
        active_experts = int(torch.unique(topk_ids[in_win]).numel())
        loads.append((r, lo, local_m, local_tokens, active_experts))

    if not loads:
        return None

    ms = [l[2] for l in loads]
    imbalance = max(ms) / min(ms)
    # Bottleneck = heaviest-load rank; that rank's consistent triple is reported.
    r, lo, local_m, local_tokens, active_experts = max(loads, key=lambda l: l[2])

    weight_bytes = active_experts * per_expert_bytes  # activated-expert weights
    expert_mask = make_expert_mask(E, lo, local_E, device) if ep_size > 1 else None
    fn = lambda: aiter_fused_moe(
        hidden, w1_k, w2_k, topk_weights, topk_ids, quant_type, w1_scale, w2_scale, expert_mask
    )

    snr = None
    if check:
        with suppress_output():
            out = fn()
        ref = torch_moe_ref(hidden, w1, w2, topk_weights, topk_ids, lo)
        if ref.abs().any():
            snr = compute_snr(ref, out.float())

    time_ms = _time_ms(fn)

    # FC1: [local_m,K]x[K,2*local_I], FC2: [local_m,local_I]x[local_I,K]
    flops = 2 * local_m * K * (2 * local_I) + 2 * local_m * local_I * K
    # Effective HBM traffic: activated-expert weights + bf16 activations. Only
    # the tokens routed to this rank are read/written (EP touches a subset; TP
    # touches all of them).
    act_bytes = 2 * local_tokens * K * elem  # hidden in + out

    return {
        "time_ms": time_ms,
        "tflops": flops / (time_ms * 1e-3) / 1e12,
        "bw": (weight_bytes + act_bytes) / (time_ms * 1e-3) / 1e9,
        "imbalance": imbalance,
        "snr": snr,
    }


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
        "variance": args.variance,
        "cudagraph": args.cudagraph,
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
        "--variance",
        type=float,
        default=0.0,
        help="Expert-load imbalance: variance of per-expert load normalized to "
        "mean 1 (squared coefficient of variation). 0=perfectly balanced; larger "
        "= more hot experts. Saturates at E/topk - 1 (all tokens on the same topk).",
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
        "--no-cudagraph",
        dest="cudagraph",
        action="store_false",
        help="Time with eager launches instead of CUDA-graph replay (default: cudagraph on).",
    )
    parser.set_defaults(cudagraph=True)
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
    assert args.variance >= 0.0, "--variance must be >= 0"

    assert torch.cuda.is_available(), "This benchmark requires a GPU."
    torch.manual_seed(args.seed)
    global _USE_CUDAGRAPH
    _USE_CUDAGRAPH = args.cudagraph
    tp_size, ep_size = args.tp_size, args.ep_size
    assert tp_size >= 1 and ep_size >= 1, "tp/ep size must be >= 1"
    assert not (tp_size > 1 and ep_size > 1), "Enable either TP or EP, not both."

    device = "cuda"
    dtype = torch.bfloat16
    cfg = MODELS[args.model]
    assert cfg["moe_intermediate_size"] % tp_size == 0, "intermediate not divisible by tp_size"
    assert cfg["n_routed_experts"] % ep_size == 0, "experts not divisible by ep_size"

    report = "bottleneck rank" if ep_size > 1 else "single rank"
    print()
    print(f"  Model     : {args.model} (hidden={cfg['hidden_size']}, inter={cfg['moe_intermediate_size']}, "
          f"experts={cfg['n_routed_experts']}, topk={cfg['topk']})")
    print(f"  Precision : {args.quant}        Parallel : tp={tp_size} ep={ep_size}  variance={args.variance}")
    print(f"  Timing    : {'cudagraph' if args.cudagraph else 'eager'}  ({report}; Imbal = slowest/fastest rank)")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print()

    cols = (("Tokens", 8), ("Time (ms)", 11), ("TFLOPS", 9), ("BW (GB/s)", 10), ("Imbal", 6), ("SNR (dB)", 8))
    header = " | ".join(f"{name:^{w}}" for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)

    rows = []
    for num_tokens in args.tokens:
        try:
            res = benchmark_one(
                num_tokens, cfg, dtype, device, tp_size, ep_size, args.variance, args.quant, args.check
            )
            if res is None:
                print(f"{num_tokens:>8} | (no active ranks)")
                rows.append({"Tokens": num_tokens, "Time (ms)": None})
                continue
            snr_str = "-" if res["snr"] is None else f"{res['snr']:.1f}"
            print(
                f"{num_tokens:>8} | {res['time_ms']:>11.3f} | {res['tflops']:>9.1f} | "
                f"{res['bw']:>10.0f} | {res['imbalance']:>6.2f} | {snr_str:>8}"
            )
            rows.append(
                {
                    "Tokens": num_tokens,
                    "Time (ms)": round(res["time_ms"], 3),
                    "TFLOPS": round(res["tflops"], 1),
                    "BW (GB/s)": round(res["bw"], 0),
                    "Imbal": round(res["imbalance"], 2),
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
