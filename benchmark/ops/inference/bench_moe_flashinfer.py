###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inference MoE micro-benchmark for FlashInfer's trtllm-gen fused MoE (forward-only).

NVIDIA counterpart of ``bench_moe_aiter.py``; mirrors its structure one-to-one
(same MODELS, helpers, metrics, CLI, Excel layout) so AMD/aiter and NVIDIA/
FlashInfer numbers compare directly. Calls the TensorRT-LLM-gen fused-MoE kernels
FlashInfer dispatches on Blackwell (B200 / sm100):

  none      : bf16          -> trtllm_bf16_moe
  fp8_block : per-128x128    -> trtllm_fp8_block_scale_moe   (DeepSeek convention)
  fp8_token : per-tensor fp8 -> trtllm_fp8_per_tensor_scale_moe
  fp4       : mxfp4 (OCP)    -> trtllm_fp4_block_scale_moe    (aiter-comparable)
  nvfp4     : nvfp4          -> trtllm_fp4_block_scale_moe    (Blackwell flagship)

Simulates one rank under TP (shard intermediate) or EP (shard experts, via
local_expert_offset / local_num_experts). --variance (>=0) sets expert-load
imbalance; under EP all ranks are swept and the bottleneck (slowest) rank's
Time/TFLOPS/BW are reported, with an Imbal column = slowest/fastest rank time.

The trtllm-gen kernels need kernel-specific weight pre-processing (row reorder
for gated activation, shuffle_matrix_a, block-scale interleave, BlockMajorK block
layout, nvfp4 packing); we reuse FlashInfer's own public helpers so the layout
always matches the kernel. All prep is done outside the timed region.

NOTE: targets B200/sm100, written against FlashInfer's documented API and its
reference tests (tests/moe/test_trtllm_gen_fused_moe.py). The bf16 path is the
most direct; the fp8/fp4 scale plumbing should be validated on real Blackwell.

    python bench_moe_flashinfer.py [--model M] [--quant Q] [--tp-size N | --ep-size N] [--variance V] [--check]

================================ KNOWN ISSUES ================================
Two cross-backend comparability caveats vs bench_moe_aiter.py. These are tracked
in KNOWN_ISSUES below and are emitted onto extra rows of EVERY exported .xlsx so
that consumers of the numbers cannot miss them:

  [ISSUE-1] fp8_token granularity mismatch -- aiter --quant fp8_token is
    per-output-channel fp8 (QuantType.per_Token, scale [E,N,1]); flashinfer
    fp8_token maps to trtllm_fp8_per_tensor_scale_moe (one scalar per tensor).
    Different quantization granularity => NOT a like-for-like comparison.

  [ISSUE-2] activation-quant timing asymmetry -- aiter fuses activation
    quantization INSIDE the timed kernel, whereas this bench pre-quantizes the
    hidden states OUTSIDE the timed region (_quant_block_act_fp8 / fp4_quantize).
    So flashinfer Time/TFLOPS/BW EXCLUDE the activation-quant cost that aiter
    INCLUDES (affects fp8_block / fp8_token / nvfp4). Negligible at small T,
    but biases flashinfer favorably as the token count grows.
=============================================================================
"""

import argparse
import contextlib
import math
import os
from datetime import datetime

import torch
from flashinfer import (
    ActivationType,
    RoutingMethodType,
    fp4_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
)
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)

# MoE routed-expert shapes: hidden_size, moe_intermediate_size, n_routed_experts, topk.
MODELS = {
    # https://huggingface.co/deepseek-ai/DeepSeek-R1  (671B total / 37B active)
    "deepseek-r1": {"hidden_size": 7168, "moe_intermediate_size": 2048, "n_routed_experts": 256, "topk": 8},
    # https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro  (1.6T total / 49B active; native fp4 experts, SwiGLU, top-6)
    "deepseek-v4-pro": {
        "hidden_size": 7168,
        "moe_intermediate_size": 3072,
        "n_routed_experts": 384,
        "topk": 6,
    },
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
    "qwen3-next-80b-a3b": {
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "n_routed_experts": 512,
        "topk": 10,
    },
}

DEFAULT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

_USE_CUDAGRAPH = True  # time via CUDA-graph replay (matches SGLang/vLLM decode); set by main().

# Cross-backend comparability caveats vs bench_moe_aiter.py. Documented in the
# module docstring (KNOWN ISSUES) and written onto extra rows of every exported
# .xlsx (see save_excel). Each entry is (id, applies_to, summary).
KNOWN_ISSUES = [
    (
        "ISSUE-1",
        ("fp8_token",),
        "fp8_token granularity mismatch: aiter fp8_token is per-output-channel "
        "fp8 (QuantType.per_Token, scale [E,N,1]); flashinfer fp8_token is "
        "per-tensor (trtllm_fp8_per_tensor_scale_moe). Different granularity -> "
        "NOT a like-for-like comparison.",
    ),
    (
        "ISSUE-2",
        ("fp8_block", "fp8_token", "nvfp4"),
        "activation-quant timing asymmetry: aiter fuses activation quantization "
        "INSIDE the timed kernel; this bench pre-quantizes hidden states OUTSIDE "
        "the timed region. So flashinfer Time/TFLOPS/BW EXCLUDE activation-quant "
        "cost that aiter INCLUDES. Negligible at small T, biases flashinfer "
        "favorably as T grows.",
    ),
]

# Generic top-k routing: TopK -> Softmax over the selected logits (norm_topk_prob).
# DeepSeek-V3 group routing isn't modeled here (it barely affects kernel cost).
_ROUTING_METHOD = RoutingMethodType.Renormalize
_EPILOGUE_TILE_M = 128  # trtllm-gen kernel internal; used by the weight shuffle.
# Per-process cache of permute indices, keyed by tensor shape, reused across cells.
_PERMUTE_CACHE = {}


@contextlib.contextmanager
def suppress_output():
    """Silence chatty per-call stdout/stderr at the fd level."""
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
    """Routing whose load imbalance is set by ``variance`` -- identical model to
    bench_moe_aiter.make_routing (variance of per-expert load normalized to mean
    1; 0 = exact per-expert balance, larger = more hot experts, saturating at
    E/topk - 1). Returns (logits[bf16], topk_weights[fp32], topk_ids[int32]).

    aiter feeds (topk_weights, topk_ids) straight to the kernel; the trtllm-gen
    kernel instead routes internally from `logits`, so we synthesize logits that
    reproduce *exactly* this (ids, weights): set the selected experts' logits to
    the desired post-softmax weights' log (others to -inf-ish), so the kernel's
    Renormalize (TopK -> Softmax over the selected) recovers the same picks and
    the same renormalized weights. Weights are uniform (1/topk), matching aiter.
    """
    weights = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    var_max = num_experts / topk - 1.0
    balance = (
        1.0 if (variance <= 0.0 or var_max <= 0.0) else 1.0 - math.sqrt(min(variance, var_max) / var_max)
    )

    if balance >= 1.0:
        stride = max(1, num_experts // topk)
        offs = (torch.arange(num_tokens, device=device) % stride).unsqueeze(1)
        base = (torch.arange(topk, device=device) * stride).unsqueeze(0)
        ids = ((offs + base) % num_experts).to(torch.int32)  # topk distinct, evenly spaced
    else:
        gen = torch.Generator(device=device).manual_seed(1234)
        hot = torch.randperm(num_experts, generator=gen, device=device)[:topk]
        pop = torch.full((num_experts,), balance / num_experts, device=device, dtype=torch.float32)
        pop[hot] += (1.0 - balance) / topk
        probs = pop.unsqueeze(0).expand(num_tokens, -1).contiguous()
        ids = torch.multinomial(probs, topk, replacement=False, generator=gen).to(torch.int32)

    # Build logits that reproduce these picks under TopK->Softmax: selected slots
    # get log(weight) (uniform -> equal logits -> uniform softmax = same weights),
    # unselected get a large negative so they are never in the top-k.
    logits = torch.full((num_tokens, num_experts), -1e4, device=device, dtype=torch.float32)
    logits.scatter_(1, ids.to(torch.int64), 0.0)  # uniform selected logits -> Renormalize gives 1/topk
    return logits.to(torch.bfloat16), weights, ids.to(torch.int32)


def torch_moe_ref(hidden, w1, w2, topk_weights, topk_ids, expert_offset=0):
    """Naive reference MoE (fp32 accumulation) for correctness only. Always uses
    the original bf16 weights, so fp8/fp4 modes show their quantization SNR.
    Layout: hidden [T, K], w1 [local_E, 2*I, K] (gate||up), w2 [local_E, K, I]."""
    T, K = hidden.shape
    out = torch.zeros(T, K, dtype=torch.float32, device=hidden.device)
    x = hidden.float()
    w1f, w2f = w1.float(), w2.float()
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


# --------------------------------------------------------------------------- #
# Weight preparation (ported from FlashInfer tests/moe/test_trtllm_gen_fused_moe.py)
# --------------------------------------------------------------------------- #


def _swap_gate_up(w1):
    """Reorder w1 from [gate||up] (the torch_moe_ref / aiter convention, SiLU on
    the first half) to [up||gate], which is what the trtllm-gen MoE kernels expect
    (they apply SiLU to the *second* half; see sglang align_*_moe_weights_for_
    flashinfer_trtllm swap_w13_halves). Done on the bf16 weights before any
    quant/shuffle, so torch_moe_ref stays identical to the aiter benchmark."""
    E, two_I, K = w1.shape
    return w1.reshape(E, 2, two_I // 2, K).flip(dims=[1]).reshape(E, two_I, K).contiguous()


def _shuffle_bf16(w1, w2, num_experts):
    """bf16 BlockMajorK shuffle for trtllm_bf16_moe (gated-activation reorder +
    shuffle_matrix_a + convert_to_block_layout)."""
    g1, g2 = [], []
    for i in range(num_experts):
        pi = _maybe_get_cached_w3_w1_permute_indices(
            _PERMUTE_CACHE, w1[i].view(torch.uint8), _EPILOGUE_TILE_M, is_gated_act_gemm=True
        )
        t1 = w1[i].view(torch.uint8)[pi.to(w1.device)].contiguous()
        pi2 = get_w2_permute_indices_with_cache(_PERMUTE_CACHE, w2[i].view(torch.uint8), _EPILOGUE_TILE_M)
        t2 = w2[i].view(torch.uint8)[pi2.to(w2.device)].contiguous()
        t1 = convert_to_block_layout(t1.view(torch.uint8), 128)
        t2 = convert_to_block_layout(t2.view(torch.uint8), 128)
        g1.append(t1.view(torch.bfloat16))
        g2.append(t2.view(torch.bfloat16))
    return torch.stack(g1).contiguous(), torch.stack(g2).contiguous()


# fp4 flavors: fp4 == mxfp4 (OCP micro-scaling, e8m0 scales over 32-elt blocks,
# bf16 activations -- directly comparable to aiter's --quant fp4) and nvfp4
# (Blackwell flagship, fp8-e4m3 block scales over 16-elt blocks, fp4 activations).
_FP4_PARAMS = {"fp4": dict(sf_vec=32, ue8m0=True), "nvfp4": dict(sf_vec=16, ue8m0=False)}


def _quant_fp4(w, num_experts, sf_vec_size, ue8m0):
    """fp4 quantize a weight stack [E, N, K] -> (packed uint8 [E, N, K//2],
    fp8-container block scales [E, N, K//sf_vec_size], per-expert global scale).
    nvfp4: sf_vec=16, ue8m0=False (e4m3 scales); mxfp4: sf_vec=32, ue8m0=True (e8m0)."""
    qs, ss, gs = [], [], []
    for i in range(num_experts):
        amax = w[i].abs().amax().clamp(min=1e-6)
        gscale = (448.0 * 6.0) / amax.float()  # fp4 e2m1 global scale convention
        q, s = fp4_quantize(w[i], gscale, sf_vec_size, sf_use_ue8m0=ue8m0, is_sf_swizzled_layout=False)
        qs.append(q)
        ss.append(s)
        gs.append(gscale)
    return torch.stack(qs), torch.stack(ss), torch.stack(gs)


def _shuffle_fp4(wq, ws, num_experts, gated, num_elts_per_sf):
    """Shuffle fp4 packed weights + interleave block scales to kernel layout.
    Block scales are returned in the fp8-e4m3 container the kernel requires."""
    g1w, g1s = [], []
    for i in range(num_experts):
        pi = _maybe_get_cached_w3_w1_permute_indices(
            _PERMUTE_CACHE, wq[i].view(torch.uint8), _EPILOGUE_TILE_M, is_gated_act_gemm=gated
        )
        g1w.append(wq[i].view(torch.uint8)[pi.to(wq.device)].contiguous())
        psf = _maybe_get_cached_w3_w1_permute_indices(
            _PERMUTE_CACHE,
            ws[i].view(torch.uint8),
            _EPILOGUE_TILE_M,
            num_elts_per_sf=num_elts_per_sf,
            is_gated_act_gemm=gated,
        )
        si = block_scale_interleave(ws[i].view(torch.uint8)[psf.to(ws.device)].contiguous())
        g1s.append(si.view(torch.float8_e4m3fn))
    return torch.stack(g1w), torch.stack(g1s)


def _shuffle_fp4_w2(wq, ws, num_experts, num_elts_per_sf):
    g2w, g2s = [], []
    for i in range(num_experts):
        pi = get_w2_permute_indices_with_cache(_PERMUTE_CACHE, wq[i].view(torch.uint8), _EPILOGUE_TILE_M)
        g2w.append(wq[i].view(torch.uint8)[pi.to(wq.device)].contiguous())
        psf = get_w2_permute_indices_with_cache(
            _PERMUTE_CACHE, ws[i].view(torch.uint8), _EPILOGUE_TILE_M, num_elts_per_sf=num_elts_per_sf
        )
        si = block_scale_interleave(ws[i].view(torch.uint8)[psf.to(ws.device)].contiguous())
        g2s.append(si.view(torch.float8_e4m3fn))
    return torch.stack(g2w), torch.stack(g2s)


def _block_quant_fp8(w, dt, block=128):
    """128x128 block fp8 quant (DeepSeek convention). Returns (w_q[fp8],
    scale[fp32] of shape [E, N/128, K/128])."""
    E, N, K = w.shape
    assert N % block == 0 and K % block == 0, f"({N},{K}) not divisible by {block}"
    fmax = torch.finfo(dt).max
    wb = w.view(E, N // block, block, K // block, block)
    scale = (wb.abs().amax(dim=(2, 4)) / fmax).clamp(min=1e-12)
    w_q = (wb / scale[:, :, None, :, None]).to(dt).view(E, N, K)
    return w_q.contiguous(), scale.to(torch.float32).contiguous()


def fp8_dtype():
    return torch.float8_e4m3fn


def _quant_block_act_fp8(hidden, dt, block=128):
    """Per-(128-group, token) fp8 activation quant for fp8_block MoE. Returns
    (hidden_fp8 [T, K], scale[fp32] of shape [K//128, T])."""
    T, K = hidden.shape
    fmax = torch.finfo(dt).max
    hb = hidden.view(T, K // block, block).float()
    s = (hb.abs().amax(-1) / fmax).clamp(min=1e-12)  # [T, K//128]
    hq = (hb / s[:, :, None]).to(dt).view(T, K)
    return hq, s.t().contiguous()


def _estimate_inter_amax(hidden, w1_upgate):
    """Amax of the SwiGLU intermediate (expert 0), used to set the per-tensor fp8
    scale of the FC1 output. w1_upgate is [up||gate] (kernel order)."""
    with torch.no_grad():
        gu = hidden.float() @ w1_upgate[0].float().t()
        up, gate = gu.chunk(2, dim=-1)  # kernel order: up first, gate second
        inter = torch.nn.functional.silu(gate) * up
    return inter.abs().amax().clamp(min=1e-12).float()


def _time_ms(fn, warmup=10, iters=50, use_graph=None):
    """GPU time per call via CUDA events (excludes host/Python launch overhead).

    With use_graph (default, controlled by --cudagraph) fn is captured into a
    CUDA graph and timed via replay, removing per-launch dispatch overhead and
    matching how SGLang/vLLM run decode. Falls back to eager launches if the
    kernel cannot be captured (e.g. an internal device->host sync). Mirrors
    bench_moe_aiter._time_ms one-to-one."""
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


def _per_expert_nbytes(t):
    """Stored bytes of one expert's slice (t[0]), correct for fp4x2 packing."""
    return t[0].numel() * t.element_size()


def build_moe_fn(quant, hidden, w1, w2, logits, num_experts, topk, local_I, local_E, lo, device):
    """Quantize/shuffle weights and return (fn, per_expert_weight_bytes). All
    quantization/shuffle happens here, outside the timed region."""
    K = hidden.shape[1]
    # trtllm-gen kernels gate on the 2nd half of w1; the torch reference (and the
    # caller's w1) use [gate||up]. Reorder to [up||gate] for the kernel only.
    w1 = _swap_gate_up(w1)
    common = dict(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=hidden,
        num_experts=num_experts,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=local_I,
        local_expert_offset=lo,
        local_num_experts=local_E,
        routed_scaling_factor=None,
        routing_method_type=int(_ROUTING_METHOD),
        activation_type=int(ActivationType.Swiglu),
        do_finalize=True,
    )

    if quant == "none":
        g1, g2 = _shuffle_bf16(w1, w2, local_E)

        def fn():
            return trtllm_bf16_moe(
                gemm1_weights=g1,
                gemm2_weights=g2,
                use_shuffled_weight=True,
                weight_layout=int(WeightLayout.BlockMajorK),
                **common,
            )

        return fn, _per_expert_nbytes(g1) + _per_expert_nbytes(g2)

    if quant in _FP4_PARAMS:
        gated = True
        p = _FP4_PARAMS[quant]
        sf_vec, ue8m0, n_sf = p["sf_vec"], p["ue8m0"], p["sf_vec"]
        w1q, w1s, w1g = _quant_fp4(w1, local_E, sf_vec, ue8m0)
        w2q, w2s, w2g = _quant_fp4(w2, local_E, sf_vec, ue8m0)
        g1w, g1s = _shuffle_fp4(w1q, w1s, local_E, gated, n_sf)
        g2w, g2s = _shuffle_fp4_w2(w2q, w2s, local_E, n_sf)

        if quant == "nvfp4":
            # nvfp4: activations are also nvfp4 (packed uint8 + per-token e4m3
            # block scale). FC1 dequant folds in the activation global scale.
            a_gscale = (448.0 * 6.0) / hidden.abs().amax().clamp(min=1e-6).float()
            hq, hsf = fp4_quantize(hidden, a_gscale, 16, sf_use_ue8m0=False, is_sf_swizzled_layout=False)
            hs = hq
            hs_scale = hsf.view(torch.float8_e4m3fn).reshape(hidden.shape[0], K // 16)
            out1 = (1.0 / (w1g * a_gscale)).to(torch.float32)
            out2 = (1.0 / w2g).to(torch.float32)
        else:
            # fp4 == mxfp4: bf16 activations (no hidden scale); per-expert global-
            # scale corrections recover the e2m1 dequant (1/gscale).
            hs, hs_scale = hidden, None
            out1 = (1.0 / w1g).to(torch.float32)
            out2 = (1.0 / w2g).to(torch.float32)

        def fn():
            return trtllm_fp4_block_scale_moe(
                hidden_states=hs,
                hidden_states_scale=hs_scale,
                gemm1_weights=g1w,
                gemm1_weights_scale=g1s,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=g2w,
                gemm2_weights_scale=g2s,
                gemm2_bias=None,
                output1_scale_scalar=out1,
                output1_scale_gate_scalar=out1,
                output2_scale_scalar=out2,
                **{k: v for k, v in common.items() if k != "hidden_states"},
            )[0]

        return fn, (
            _per_expert_nbytes(g1w)
            + _per_expert_nbytes(g2w)
            + _per_expert_nbytes(g1s)
            + _per_expert_nbytes(g2s)
        )

    if quant == "fp8_block":
        dt = fp8_dtype()
        w1q, w1s = _block_quant_fp8(w1, dt)
        w2q, w2s = _block_quant_fp8(w2, dt)
        # Activations must be fp8: per-(128-group, token) quant, scale [K//128, T].
        hq, hs_scale = _quant_block_act_fp8(hidden, dt)

        def fn():
            return trtllm_fp8_block_scale_moe(
                hidden_states=hq,
                hidden_states_scale=hs_scale,
                gemm1_weights=w1q,
                gemm1_weights_scale=w1s,
                gemm2_weights=w2q,
                gemm2_weights_scale=w2s,
                use_shuffled_weight=False,
                weight_layout=int(WeightLayout.MajorK),
                **{k: v for k, v in common.items() if k != "hidden_states"},
            )

        return fn, (
            _per_expert_nbytes(w1q)
            + _per_expert_nbytes(w2q)
            + _per_expert_nbytes(w1s)
            + _per_expert_nbytes(w2s)
        )

    if quant == "fp8_token":  # per-tensor fp8 (dynamic per-tensor activations)
        dt = fp8_dtype()
        fmax = torch.finfo(dt).max
        s1 = (w1.abs().amax() / fmax).clamp(min=1e-12).float()
        s2 = (w2.abs().amax() / fmax).clamp(min=1e-12).float()
        w1q = (w1 / s1).to(dt).contiguous()
        w2q = (w2 / s2).to(dt).contiguous()
        # Per-tensor kernel needs the same gated-act reorder + shuffle_matrix_a as bf16.
        w1q = torch.stack(
            [
                shuffle_matrix_a(reorder_rows_for_gated_act_gemm(w1q[i]).view(torch.uint8), _EPILOGUE_TILE_M)
                for i in range(local_E)
            ]
        ).view(dt)
        w2q = torch.stack(
            [shuffle_matrix_a(w2q[i].view(torch.uint8), _EPILOGUE_TILE_M) for i in range(local_E)]
        ).view(dt)
        # Per-tensor activation scales (a1 = FC1 input, a2 = FC1-output fp8 scale).
        a1 = (hidden.abs().amax() / fmax).clamp(min=1e-12).float()
        hq = (hidden / a1).to(dt).contiguous()
        a2 = _estimate_inter_amax(hidden, w1) / fmax
        # sglang convention: out1 = s1*a1/a2, gate = s1*a1, out2 = a2*s2.
        out1 = (s1 * a1 / a2).reshape(1).expand(local_E).contiguous()
        out1g = (s1 * a1).reshape(1).expand(local_E).contiguous()
        out2 = (a2 * s2).reshape(1).expand(local_E).contiguous()

        def fn():
            return trtllm_fp8_per_tensor_scale_moe(
                hidden_states=hq,
                gemm1_weights=w1q,
                output1_scales_scalar=out1,
                output1_scales_gate_scalar=out1g,
                gemm2_weights=w2q,
                output2_scales_scalar=out2,
                use_routing_scales_on_input=False,
                **{k: v for k, v in common.items() if k != "hidden_states"},
            )

        return fn, (_per_expert_nbytes(w1q) + _per_expert_nbytes(w2q))

    raise ValueError(f"Unknown quant: {quant}")


def benchmark_one(num_tokens, cfg, dtype, device, tp_size, ep_size, variance, quant, check):
    K = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    E = cfg["n_routed_experts"]
    topk = cfg["topk"]

    # One rank's local shape: TP shards the intermediate, EP shards the experts.
    local_I = I // tp_size
    local_E = E // ep_size

    # trtllm-gen MoE requires both hidden and (per-rank) intermediate to be a
    # multiple of 128; gpt-oss-120b's 2880 is neither (2880 % 128 == 64), so the
    # kernel aborts (intermediate "% 128 == 0" assert; hidden has no valid tile
    # config / w2 shuffle asserts M % 128). vLLM and SGLang serve gpt-oss on
    # Blackwell by *weight padding* both dims up to a multiple of 256 (2880 ->
    # 3072) and feeding the padded shape to the same trtllm-gen FP4 kernel, then
    # trimming the output back to the true hidden. We mirror that: round K and
    # local_I up to 256. Padded gate/up/w2 rows and padded hidden columns are
    # zero, so SiLU(0)*0 == 0 and the zero hidden lanes contribute nothing -- the
    # reference runs on the same padded weights so SNR stays honest. Already-
    # aligned models round up to themselves (no-op). FLOP/byte accounting below
    # uses the *true* K/local_I, never the padded sizes, so padding never inflates
    # reported throughput.
    _ALIGN = 256
    K_pad = ((K + _ALIGN - 1) // _ALIGN) * _ALIGN
    local_I_pad = ((local_I + _ALIGN - 1) // _ALIGN) * _ALIGN

    hidden = torch.zeros(num_tokens, K_pad, device=device, dtype=dtype)
    hidden[:, :K] = torch.randn(num_tokens, K, device=device, dtype=dtype)
    w1 = torch.zeros(local_E, 2 * local_I_pad, K_pad, device=device, dtype=dtype)
    w2 = torch.zeros(local_E, K_pad, local_I_pad, device=device, dtype=dtype)
    # Fill only the live [gate||up] / w2 blocks; padded tail/lanes stay zero. w1
    # is [gate||up] so gate and up are padded independently (each I -> I_pad), and
    # the K dimension is padded to K_pad as well.
    w1[:, :local_I, :K] = torch.randn(local_E, local_I, K, device=device, dtype=dtype) / (K**0.5)
    w1[:, local_I_pad : local_I_pad + local_I, :K] = (
        torch.randn(local_E, local_I, K, device=device, dtype=dtype) / (K**0.5)
    )
    w2[:, :K, :local_I] = torch.randn(local_E, K, local_I, device=device, dtype=dtype) / (local_I**0.5)

    logits, topk_weights, topk_ids = make_routing(num_tokens, E, topk, device, variance)

    # In EP each rank owns a different expert window; under load imbalance the
    # ranks get different token counts. Measure each rank's load (token-expert
    # pairs = grouped-GEMM M, noise-free), then bench/report the bottleneck =
    # heaviest rank (a decode step waits for the slowest one). Imbal = max/min
    # load over active ranks. Under TP one rank owns all experts. Mirrors aiter.
    ranks = range(ep_size) if ep_size > 1 else [0]

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

    # Kernel sees the padded intermediate (w1/w2 are built at local_I_pad); flop/
    # byte accounting below stays on the true local_I.
    fn, per_expert_bytes = build_moe_fn(quant, hidden, w1, w2, logits, E, topk, local_I_pad, local_E, lo, device)

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
    # the tokens routed to this rank are read/written (EP touches a subset).
    weight_bytes = active_experts * per_expert_bytes
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
        "backend": "flashinfer",
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

    # Extra rows appended to the config sheet so the cross-backend comparability
    # caveats travel with every exported result and cannot be overlooked.
    # Stringify values so the round-trip stays faithful: Excel stores bool as a
    # distinct cell type, and since Python `bool` subclasses `int` (1 == True), a
    # mixed bool/int column reads ints equal to 1 back as True.
    config_items = [(k, str(v)) for k, v in meta.items()]
    config_items.append(("", ""))
    config_items.append(("KNOWN ISSUES", "cross-backend comparability vs bench_moe_aiter.py"))
    for iid, applies_to, summary in KNOWN_ISSUES:
        active = "APPLIES to this run" if args.quant in applies_to else "n/a for this run"
        config_items.append((f"{iid} [{active}]", summary))

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(config_items, columns=["field", "value"]).to_excel(
            writer, sheet_name="config", index=False
        )
        pd.DataFrame(rows).to_excel(writer, sheet_name=results_sheet, index=False)

        # Also surface the issues as their own sheet for visibility.
        issue_rows = [
            {
                "id": iid,
                "applies_to_quant": ", ".join(applies_to),
                "active_in_this_run": args.quant in applies_to,
                "summary": summary,
            }
            for iid, applies_to, summary in KNOWN_ISSUES
        ]
        pd.DataFrame(issue_rows).to_excel(writer, sheet_name="known_issues", index=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Inference MoE benchmark (FlashInfer trtllm fused_moe)")
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="deepseek-r1",
        help="MoE model whose routed-expert shapes to benchmark.",
    )
    parser.add_argument(
        "--quant",
        choices=["none", "fp8_block", "fp8_token", "fp4", "nvfp4"],
        default="none",
        help="Quantization: none (bf16), fp8_block (per-128x128), fp8_token (per-tensor), "
        "fp4 (mxfp4 OCP micro-scaling, aiter-comparable), nvfp4 (Blackwell flagship).",
    )
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
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=DEFAULT_TOKENS, help="Token counts to sweep."
    )
    parser.add_argument("--check", action="store_true", help="Run the SNR correctness check (default: off).")
    parser.add_argument(
        "--no-cudagraph",
        dest="cudagraph",
        action="store_false",
        help="Time with eager launches instead of CUDA-graph replay (default: cudagraph on).",
    )
    parser.set_defaults(cudagraph=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save to .xlsx (auto-named if omitted)."
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
    print(
        f"  Model     : {args.model} (hidden={cfg['hidden_size']}, inter={cfg['moe_intermediate_size']}, "
        f"experts={cfg['n_routed_experts']}, topk={cfg['topk']})"
    )
    print(f"  Precision : {args.quant}        Parallel : tp={tp_size} ep={ep_size}  variance={args.variance}")
    print(
        f"  Timing    : {'cudagraph' if args.cudagraph else 'eager'}  ({report}; Imbal = slowest/fastest rank)"
    )
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
            msg = str(e).strip() or type(e).__name__
            print(f"{num_tokens:>8} | ERROR: {msg}")
            rows.append({"Tokens": num_tokens, "Time (ms)": f"ERROR: {msg}"})

    output = args.output
    if output is None:
        date = datetime.now().strftime("%Y%m%d")
        output = f"bench_moe_fi_{args.model}_{args.quant}_tp{tp_size}_ep{ep_size}_{date}.xlsx"
    saved = save_excel(output, args, cfg, tp_size, ep_size, rows)
    print(f"\nSaved results to {saved}")


if __name__ == "__main__":
    main()
