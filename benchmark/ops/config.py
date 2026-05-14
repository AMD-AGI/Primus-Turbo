###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared configurations for benchmark scripts."""

import json
import os
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
        "vocab_size": 32000,
    },
    # https://huggingface.co/meta-llama/Llama-2-70b/blob/main/config.json
    "Llama-2-70B": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    # https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
    "Llama-3.1-8B": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    # https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json
    "Llama-3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    # Llama-4 is an MoE model; these entries cover only its *dense* GEMMs
    # (attention QKV/O, dense FFN via ffn_hidden_size, lm_head). The routed-expert
    # (grouped) GEMMs are modeled separately in MoEModelConfigs. The 16E/128E
    # variants differ only in expert count, so their dense GEMM shapes are
    # identical. Hyperparameters from Primus configs/models/megatron/llama4_*.yaml.
    "Llama-4-17Bx16E": {
        "seqlen": 4096,
        "hidden_size": 5120,
        "intermediate_size": 16384,  # ffn_hidden_size (dense FFN)
        "num_attention_heads": 40,
        "num_key_value_heads": 8,  # num_query_groups
        "head_dim": 128,
        "vocab_size": 202048,
    },
    "Llama-4-17Bx128E": {
        "seqlen": 4096,
        "hidden_size": 5120,
        "intermediate_size": 16384,  # ffn_hidden_size (dense FFN)
        "num_attention_heads": 40,
        "num_key_value_heads": 8,  # num_query_groups
        "head_dim": 128,
        "vocab_size": 202048,
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/file/view/master/config.json
    "Qwen2.5-7B": {
        "seqlen": 8192,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
    },
    # https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct
    "Qwen2.5-72B": {
        "seqlen": 8192,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 152064,
    },
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/config.json
    "Mistral-7B": {
        "seqlen": 4096,  # sliding_window
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
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
    # https://modelscope.cn/models/deepseek-ai/DeepSeek-V2/file/view/master/config.json
    "DeepSeek-V2": {
        "n_routed_experts": 160,
        "moe_intermediate_size": 1536,
        "hidden_size": 5120,
        # MLA attention config
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim_qk": 192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        "head_dim_v": 128,
        "seqlen": 4096,
        "num_experts": 160,
        "num_topk": 8,
    },
    # https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite/file/view/master/config.json
    "DeepSeek-V2-Lite": {
        "n_routed_experts": 64,
        "moe_intermediate_size": 1408,
        "hidden_size": 2048,
        # MLA attention config
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim_qk": 192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        "head_dim_v": 128,
        "seqlen": 4096,
        "num_experts": 64,
        "num_topk": 8,
    },
    # https://huggingface.co/xai-org/grok-2/blob/main/config.json
    "Grok-2": {
        "n_routed_experts": 8,
        "moe_intermediate_size": 16384,
        "hidden_size": 8192,
        # Standard MHA
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "seqlen": 8192,
        "num_experts": 8,
        "num_topk": 2,
    },
    # https://modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507/file/view/master/config.json
    "Qwen3-30B-A3B": {
        "n_routed_experts": 128,
        "moe_intermediate_size": 2048,
        "hidden_size": 2048,
        # GQA attention config
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "seqlen": 8192,
        "num_experts": 128,
        "num_topk": 8,
    },
    # https://modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507
    "Qwen3-235B-A22B": {
        "n_routed_experts": 128,
        "moe_intermediate_size": 4096,
        "hidden_size": 4096,
        # GQA attention config
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "seqlen": 8192,
        "num_experts": 128,
        "num_topk": 8,
    },
    # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json
    "Mixtral-8x7B": {
        "n_routed_experts": 8,  # num_local_experts
        "moe_intermediate_size": 14336,
        "hidden_size": 4096,
        # GQA attention config
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "seqlen": 4096,
        "num_experts": 8,
        "num_topk": 2,
    },
    # https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/blob/main/config.json
    "Mixtral-8x22B": {
        "n_routed_experts": 8,  # num_local_experts
        "moe_intermediate_size": 16384,
        "hidden_size": 6144,
        # GQA attention config
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "seqlen": 8192,
        "num_experts": 8,
        "num_topk": 2,
    },
    # https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json
    "Kimi-K2": {
        "n_routed_experts": 384,
        "moe_intermediate_size": 2048,
        "hidden_size": 7168,
        # MLA attention config
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim_qk": 192,  # qk_nope_head_dim(128) + qk_rope_head_dim(64)
        "head_dim_v": 128,
        "seqlen": 4096,
        "num_experts": 384,
        "num_topk": 8,
    },
    # https://github.com/AMD-AGI/Primus/blob/main/primus/configs/models/megatron/moe_1T.yaml
    # 1T total params, 44B active params
    "MoE-1T": {
        "n_routed_experts": 224,
        "moe_intermediate_size": 1920,  # moe_ffn_hidden_size
        "hidden_size": 8192,
        # GQA attention config
        "num_attention_heads": 64,
        "num_key_value_heads": 8,  # num_query_groups
        "head_dim": 128,
        "seqlen": 4096,
        "num_experts": 224,
        "num_topk": 8,
    },
    # https://github.com/AMD-AGI/Primus/blob/main/primus/configs/models/megatron/moe_2T.yaml
    # 2T total params, 80B active params
    # "MoE-2T": {
    #     "n_routed_experts": 448,
    #     "moe_intermediate_size": 1920,  # moe_ffn_hidden_size
    #     "hidden_size": 8192,
    #     # GQA attention config
    #     "num_attention_heads": 64,
    #     "num_key_value_heads": 8,  # num_query_groups
    #     "head_dim": 128,
    #     "seqlen": 4096,
    # },
}

###############################################################################
# Benchmark Constants
###############################################################################

BATCH_SIZE_DEFAULT = [1, 2, 4]

# Batch sizes are crawled from Primus pretrain configs into batch_size_config.json
# (regenerate via crawl_batch_sizes.sh). The JSON records the exact Primus commit
# it was crawled from so values can be tracked back to an upstream revision.
BATCH_SIZE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "batch_size_config.json")


def _load_batch_size_config(path=BATCH_SIZE_CONFIG_PATH):
    """Load crawled batch sizes from JSON.

    Returns (config, meta) where ``config`` maps both:
      (gpu, model, dtype) -> [batch_sizes]   # per-config entries
      dtype               -> [batch_sizes]   # dtype-level fallback
    and ``meta`` is the JSON ``_meta`` block (source commit, crawl date, ...).

    On any failure the config is left empty so callers fall back to
    BATCH_SIZE_DEFAULT instead of crashing the benchmark.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[config] warning: could not load {path}: {exc}; using default batch sizes")
        return {}, {}

    config = {}
    for entry in data.get("entries", []):
        config[(entry["gpu"], entry["model"], entry["dtype"])] = entry["batch_sizes"]
    for dtype, sizes in data.get("dtype_default", {}).items():
        config[dtype] = sizes
    return config, data.get("_meta", {})


BATCH_SIZE_CONFIG, BATCH_SIZE_CONFIG_META = _load_batch_size_config()

# Deprecated alias – kept for backward compatibility
BATCH_SIZE_LIST = BATCH_SIZE_DEFAULT


GPU_NAME_MAP = {
    "MI300X": "MI30*",
    "MI300A": "MI30*",
    "MI308X": "MI30*",
    "MI350X": "MI35*",
    "MI355X": "MI35*",
}


def get_batch_sizes(model_name=None, dtype_name=None, gpu_name=None):
    """Look up batch sizes for a given model/dtype/GPU.

    The gpu_name is mapped via GPU_NAME_MAP to a config key
    (e.g. "MI355X" -> "MI35*") before lookup.

    Resolution (highest to lowest priority):
      (gpu, model, dtype) -> dtype -> default
    """
    gpu_key = GPU_NAME_MAP.get(gpu_name, gpu_name)
    if (gpu_key, model_name, dtype_name) in BATCH_SIZE_CONFIG:
        return BATCH_SIZE_CONFIG[(gpu_key, model_name, dtype_name)]
    if dtype_name in BATCH_SIZE_CONFIG:
        return BATCH_SIZE_CONFIG[dtype_name]
    return BATCH_SIZE_DEFAULT


# Grouped GEMM (MoE) configurations
GROUPED_GEMM_M_SIZE_LIST = [512, 1024, 2048, 4096, 8192, 16384]
GROUPED_GEMM_EP_SIZE_LIST = [32, 16, 8]

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
    vocab_size = model_config["vocab_size"]

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
    # vocab/embedding projection (lm_head)
    gemm_shape_list.append([seq, vocab_size, hidden_size])
    return gemm_shape_list


def gen_grouped_gemm_group_lens(b, m, balance: bool = True):
    """Generate group lengths for grouped GEMM."""
    if balance:
        return torch.full((b,), m, dtype=torch.int64)
    else:
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens


def _generate_moe_test_cases(
    name_prefix: str,
    n_routed_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
):
    """Generate MoE test cases for grouped GEMM benchmark."""
    test_cases = []
    shapes_dict = {
        f"{name_prefix}-GateUP": (2 * moe_intermediate_size, hidden_size),
        f"{name_prefix}-Down": (hidden_size, moe_intermediate_size),
    }

    for ep in GROUPED_GEMM_EP_SIZE_LIST:
        if n_routed_experts % ep != 0:
            continue
        B = n_routed_experts // ep
        if B < 1:
            continue
        for M in GROUPED_GEMM_M_SIZE_LIST:
            for name, (N, K) in shapes_dict.items():
                for dtype in [torch.bfloat16]:
                    test_cases.append(
                        {
                            "Case": name,
                            "B": B,
                            "M": M,
                            "N": N,
                            "K": K,
                            "dtype": dtype,
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
