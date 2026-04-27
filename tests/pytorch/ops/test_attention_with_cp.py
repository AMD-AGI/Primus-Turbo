###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import itertools
from typing import List

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal import common_distributed
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import instantiate_parametrized_tests

import primus_turbo.pytorch as pt
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.pytorch.test_utils import compute_snr

# loosen timeout for this test to avoid timeout failures
common_distributed.TIMEOUT_DEFAULT = 600

test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=4096, seqlen_kv=4096, num_head_q=40, num_head_kv=40, head_dim_qk=192, head_dim_v=128),
]


def shard_cp_input(input_tensors: List[torch.Tensor], cp_group, seq_dim=1) -> List[torch.Tensor]:
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    output_list = []
    for t in input_tensors:
        output_list.append(t.chunk(cp_size, seq_dim)[cp_rank].contiguous())

    return output_list


@instantiate_parametrized_tests
class AttentionWithCPTestCase(MultiProcessTestCase):
    """AttentionWithCPTestCase"""

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42)

    @skip_if_lt_x_gpu(2)
    def test_attention_with_cp(self):
        self._init_process()
        dtype = torch.bfloat16
        device = torch.device("cuda", self.rank)

        test_params = {
            "batch": [2],
            "config": test_cases,
            "causal": [True, False],
            "fp8": [True, False],
            "qkv_format": ["bshd", "sbhd", "bhsd"],
            "ulysses_degree": [1, 2, 4, 8],
        }

        for batch, config, causal, fp8, qkv_format, ulysses_degree in itertools.product(
            *[test_params[k] for k in test_params]
        ):
            if self.world_size % ulysses_degree != 0:
                continue
            ring_degree = self.world_size // ulysses_degree
            if fp8 and ring_degree != 1:
                # Ring attention is currently not supported for FP8
                continue
            if fp8 and qkv_format != "bshd":
                continue
            func = pt.ops.flash_attn_fp8_usp_func if fp8 else pt.ops.flash_attn_usp_func
            self.run_attn_with_cp(
                func,
                batch,
                config,
                causal,
                qkv_format,
                ulysses_degree,
                ring_degree,
                device,
                dtype,
            )
        dist.destroy_process_group()

    def run_attn_with_cp(
        self, func, batch, config, causal, qkv_format, ulysses_degree, ring_degree, device, dtype
    ):
        cp_group = dist.group.WORLD
        device_mesh = init_device_mesh(
            "cuda",
            (ring_degree, ulysses_degree),
            mesh_dim_names=("ring", "ulysses"),
        )

        seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
            config.seqlen_q,
            config.seqlen_kv,
            config.num_head_q,
            config.num_head_kv,
            config.head_dim_qk,
            config.head_dim_v,
        )
        if qkv_format == "sbhd":
            q_layout = (seqlen_q, batch, num_head_q, head_dim_qk)
            k_layout = (seqlen_kv, batch, num_head_kv, head_dim_qk)
            v_layout = (seqlen_kv, batch, num_head_kv, head_dim_v)
            seq_dim = 0
        elif qkv_format == "bhsd":
            q_layout = (batch, num_head_q, seqlen_q, head_dim_qk)
            k_layout = (batch, num_head_kv, seqlen_kv, head_dim_qk)
            v_layout = (batch, num_head_kv, seqlen_kv, head_dim_v)
            seq_dim = 2
        else:
            q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
            k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
            v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)
            seq_dim = 1

        query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
        query_ref = query.clone().detach().requires_grad_()
        key_ref = key.clone().detach().requires_grad_()
        value_ref = value.clone().detach().requires_grad_()

        sm_scale = query.shape[-1] ** (-0.5)
        o_ref = attention_vanilla_forward_pytorch_ref_impl(
            query_ref, key_ref, value_ref, sm_scale, causal, qkv_format
        )

        grad_ref = torch.randn(*o_ref.shape, device=device, dtype=dtype)
        o_ref.backward(grad_ref)
        o_ref, dq_ref, dk_ref, dv_ref = shard_cp_input(
            [o_ref, query_ref.grad, key_ref.grad, value_ref.grad], cp_group, seq_dim
        )

        query_local_token, key_local_token, value_local_token = shard_cp_input(
            [query, key, value], cp_group, seq_dim
        )

        # flash_attn_usp_func expects shape [b, s, h, d] with format encoded
        # in strides.  Transpose sharded inputs to the expected view layout.
        if qkv_format == "sbhd":
            query_local_token = query_local_token.transpose(0, 1)
            key_local_token = key_local_token.transpose(0, 1)
            value_local_token = value_local_token.transpose(0, 1)
        elif qkv_format == "bhsd":
            query_local_token = query_local_token.transpose(1, 2)
            key_local_token = key_local_token.transpose(1, 2)
            value_local_token = value_local_token.transpose(1, 2)

        kwargs = {}
        if func is pt.ops.flash_attn_usp_func:
            kwargs["qkv_format"] = qkv_format

        o = func(
            query_local_token,
            key_local_token,
            value_local_token,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            ulysses_group=device_mesh["ulysses"].get_group(),
            ring_group=device_mesh["ring"].get_group(),
            **kwargs,
        )
        grad = shard_cp_input([grad_ref], cp_group, seq_dim)[0]
        if qkv_format == "sbhd":
            grad = grad.transpose(0, 1)
        elif qkv_format == "bhsd":
            grad = grad.transpose(1, 2)
        o.backward(grad)

        if qkv_format == "sbhd":
            o = o.transpose(0, 1)
        elif qkv_format == "bhsd":
            o = o.transpose(1, 2)
        dq, dk, dv = shard_cp_input([query.grad, key.grad, value.grad], cp_group, seq_dim)
        out_snr = compute_snr(o_ref, o)
        query_grad_snr = compute_snr(dq_ref, dq)
        key_grad_snr = compute_snr(dk_ref, dk)
        value_grad_snr = compute_snr(dv_ref, dv)

        assert out_snr > 20, "out_snr too low"
        assert query_grad_snr > 15, "query_grad_snr too low"
        assert key_grad_snr > 15, "key_grad_snr too low"
        assert value_grad_snr > 15, "value_grad_snr too low"
