###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared configurations for benchmark scripts."""

ModelConfigs = {
    "llama2-7b": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
    },
    "llama2-70b": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "llama3.1-8b": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
    "llama3.1-405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}

MBS_LIST = [1, 2, 4]


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

