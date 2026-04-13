###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared configurations for benchmark scripts."""

import re

import torch

###############################################################################
# Platform Detection
###############################################################################


def get_platform_info():
    """Detect the current platform (CUDA/ROCm) and return device info."""
    device_name = torch.cuda.get_device_name(0)

    if "AMD" in device_name or "MI" in device_name or "Radeon" in device_name:
        platform = "ROCm"
        match = re.search(r"(MI\d+[A-Za-z]*)", device_name)
        gpu_name = match.group(1) if match else device_name.split()[-1]
    else:
        platform = "CUDA"
        match = re.search(r"(H100|A100|A10|V100|RTX \d+|Tesla [A-Z]\d+)", device_name)
        gpu_name = match.group(1) if match else device_name.split()[-1]

    return platform, gpu_name


def check_allclose(out, out_ref, dtype, rtol=None, atol=None):
    """Check if two tensors are close within tolerance."""
    if rtol is None or atol is None:
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2
        else:  # bfloat16
            rtol, atol = 1e-2, 1e-2
    return torch.allclose(out, out_ref, rtol=rtol, atol=atol)


def compute_snr(ref, actual):
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    ref_f64 = ref.to(torch.float64)
    actual_f64 = actual.to(torch.float64)
    signal_power = ref_f64.norm().pow(2)
    noise_power = (ref_f64 - actual_f64).norm().pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).item()


###############################################################################
# Reference Implementations
###############################################################################


def gemm_ref(a, b, trans_b=True):
    """Reference GEMM using PyTorch native matmul."""
    b_mat = b.T if trans_b else b
    return a @ b_mat


def grouped_gemm_ref(a, b, group_lens, trans_b=True):
    """Reference grouped GEMM using PyTorch native matmul."""
    group_lens_cpu = group_lens.cpu().numpy()
    out = []
    start = 0
    for i, size in enumerate(group_lens_cpu):
        rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start : start + size, :] @ rhs)
        start += size
    return torch.cat(out)


###############################################################################
# Model Configurations
###############################################################################

# Dense Models
DenseModelConfigs = {
    # https://huggingface.co/meta-llama/Llama-2-7b/blob/main/config.json
    "Llama-2-7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
    },
    # https://huggingface.co/meta-llama/Llama-2-70b/blob/main/config.json
    "Llama-2-70B": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    # https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    "Llama-3.1-8B": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    # https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
    "Llama-3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/file/view/master/config.json
    "Qwen2.5-7B": {
        "seqlen": 8192,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct
    "Qwen2.5-72B": {
        "seqlen": 8192,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/config.json
    "Mistral-7B": {
        "seqlen": 4096,  # sliding_window
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}

# MoE Models
MoEModelConfigs = {
    # https://modelscope.cn/models/deepseek-ai/DeepSeek-V3/file/view/master/config.json
    "DeepSeek-V3": {
        "n_routed_experts": 256,
        "moe_intermediate_size": 2048,
        "hidden_size": 7168,
        # MLA attention config
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim_qk": 192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        "head_dim_v": 128,
        "seqlen": 4096,
        "num_experts": 256,
        "num_topk": 8,
    },
    # https://huggingface.co/LiquidAI/LFM2-8B-A1B
    "LFM2-8B-A1B": {
        "n_routed_experts": 32,
        "moe_intermediate_size": 1792,
        "hidden_size": 2048,
        # GQA attention config
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,  # kv_channels
        "seqlen": 4096,
        "num_experts": 32,
        "num_topk": 4,
        "ep_size_list": [8, 1],  # EP=8→B=4, EP=1→B=32
    },
    # /shared_nfs/kyle/test/Primus/examples/megatron/configs/MI300X/gpt_oss_20B-BF16-pretrain.yaml
    "gpt_oss_20B": {
        "n_routed_experts": 32,
        "moe_intermediate_size": 2880,
        "hidden_size": 2880,
        # GQA attention config
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim_qk": 128,  # qk_head_dim
        "head_dim_v": 64,    # kv_channels
        "seqlen": 4096,
        "num_experts": 32,
        "num_topk": 4,
        "ep_size_list": [8, 1],  # EP=8→B=4, EP=1→B=32
    },
}

###############################################################################
# Benchmark Constants
###############################################################################

BATCH_SIZE_LIST = [1, 2, 4]

# Grouped GEMM (MoE) configurations
GROUPED_GEMM_M_SIZE_LIST = [2048, 4096]
GROUPED_GEMM_EP_SIZE_LIST = [16, 8]

###############################################################################
# Test Case Generators
###############################################################################


def gen_gemm_test_cases(model_config):
    """Generate GEMM test cases from model config."""
    seq = model_config["seqlen"]
    hidden_size = model_config["hidden_size"]
    intermediate_size = model_config["intermediate_size"]
    num_attention_heads = model_config["num_attention_heads"]
    num_key_value_heads = model_config["num_key_value_heads"]
    head_dim = model_config["head_dim"]

    # [[m, n, k]...]
    gemm_shape_list = []
    # attn qkv pass
    gemm_shape_list.append(
        [
            seq,
            int((num_attention_heads + 2 * num_key_value_heads) * head_dim),
            hidden_size,
        ]
    )
    # attn out
    gemm_shape_list.append([seq, hidden_size, hidden_size])
    # mlp gate+up
    gemm_shape_list.append([seq, int(2 * intermediate_size), hidden_size])
    # mlp down
    gemm_shape_list.append([seq, hidden_size, intermediate_size])
    return gemm_shape_list


def gen_grouped_gemm_group_lens(b, m, balance: str = "balanced", num_topk: int = None):
    """Generate group lengths for grouped GEMM.

    balance="balanced" : all experts get exactly m tokens (uniform).
    balance="mild"     : random distribution, every expert gets tokens
                         (0.2 + 0.8*rand, normalized). Realistic MoE routing.
    balance="extreme"  : only num_topk experts receive tokens, rest get 0.
                         Simulates heavily skewed topk routing.
    """
    if balance == "balanced":
        return torch.full((b,), m, dtype=torch.int64)
    elif balance == "mild":
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens
    else:  # extreme
        total = b * m
        k = num_topk if (num_topk is not None and num_topk < b) else max(1, b // 8)
        group_lens = torch.zeros(b, dtype=torch.int64)
        hot = torch.randperm(b)[:k]
        group_lens[hot] = total // k
        group_lens[hot[0]] += total - group_lens.sum()  # fix rounding
        return group_lens


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    ep_size_list=None,
    num_topk: int = None,
):
    """Generate MoE test cases for grouped GEMM benchmark."""
    if ep_size_list is None:
        ep_size_list = GROUPED_GEMM_EP_SIZE_LIST

    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in ep_size_list:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in GROUPED_GEMM_M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
                    for balance in ["balanced", "mild", "extreme"]:
                        test_cases.append(
                            {
                                "Case": name,
                                "B": B,
                                "M": M,
                                "N": N,
                                "K": K,
                                "dtype": dtype,
                                "balance": balance,
                                "num_topk": num_topk if balance == "extreme" else None,
                            }
                        )
    return test_cases


def gen_grouped_gemm_test_cases():
    """Generate all grouped GEMM test cases for MoE models."""
    all_test_cases = []
    for name_prefix, config in MoEModelConfigs.items():
        test_cases = _generate_moe_test_cases(
            name_prefix=name_prefix,
            n_routed_experts=config["n_routed_experts"],
            moe_intermediate_size=config["moe_intermediate_size"],
            hidden_size=config["hidden_size"],
            ep_size_list=config.get("ep_size_list"),
            num_topk=config.get("num_topk"),
        )
        all_test_cases.extend(test_cases)
    return all_test_cases


def gen_attention_test_cases():
    """Generate attention test cases from model configs (deduplicated)."""
    seen = set()
    test_cases = []

    # From dense models
    for config in DenseModelConfigs.values():
        key = (
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            config["head_dim"],
            config["seqlen"],
        )
        if key not in seen:
            seen.add(key)
            test_cases.append(
                {
                    "num_head_q": config["num_attention_heads"],
                    "num_head_kv": config["num_key_value_heads"],
                    "head_dim_qk": config["head_dim"],
                    "head_dim_v": config["head_dim"],
                    "seqlen": config["seqlen"],
                }
            )

    # From MoE models (for MLA and other attention variants)
    for config in MoEModelConfigs.values():
        if "num_attention_heads" not in config:
            continue
        head_dim_qk = config.get("head_dim_qk", config.get("head_dim", 128))
        head_dim_v = config.get("head_dim_v", config.get("head_dim", 128))
        key = (
            config["num_attention_heads"],
            config["num_key_value_heads"],
            head_dim_qk,
            head_dim_v,
            config["seqlen"],
        )
        if key not in seen:
            seen.add(key)
            test_cases.append(
                {
                    "num_head_q": config["num_attention_heads"],
                    "num_head_kv": config["num_key_value_heads"],
                    "head_dim_qk": head_dim_qk,
                    "head_dim_v": head_dim_v,
                    "seqlen": config["seqlen"],
                }
            )

    return test_cases


def gen_deepep_test_cases():
    # From MoE models (for MLA and other attention variants)
    test_cases = []
    for name, config in MoEModelConfigs.items():
        test_cases.append(
            (name, config["seqlen"], config["hidden_size"], config["num_experts"], config["num_topk"])
        )

    return test_cases
