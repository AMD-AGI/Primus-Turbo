###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Variable-length (THD / cu_seqlens) Ulysses A2A context-parallel attention test.
# MultiProcessTestCase spawning world_size == device_count ranks; needs >= 2 GPUs.

import itertools
import os
import tempfile
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
from tests.pytorch.ref.attention_ref import attention_varlen_forward_pytorch_ref_impl
from tests.pytorch.test_utils import compute_snr

common_distributed.TIMEOUT_DEFAULT = 600


def make_cu_seqlens(seqlens: List[int], device) -> torch.Tensor:
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    return cu


def shard_thd_by_rank(t: torch.Tensor, cp_size: int, cp_rank: int) -> torch.Tensor:
    # Contiguous-by-rank token split: rank r owns [r*t//n : (r+1)*t//n].
    total = t.shape[0]
    assert total % cp_size == 0
    chunk = total // cp_size
    return t[cp_rank * chunk : (cp_rank + 1) * chunk].contiguous()


@instantiate_parametrized_tests
class AttentionVarlenWithCPTestCase(MultiProcessTestCase):
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
        # Per-rank Triton cache: ranks JIT-compile the same kernels concurrently and
        # would otherwise race on the shared ~/.triton/cache.
        os.environ["TRITON_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), f"triton_cache_rank{self.rank}")
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
    def test_attention_varlen_with_cp(self):
        self._init_process()
        dtype = torch.bfloat16
        device = self.device

        # Doc lengths sum to a multiple of every ulysses degree (contiguous-by-rank shard).
        # (num_head_q, num_head_kv): equal => MHA, hq>hkv => GQA.
        test_params = {
            "seqlens": [[512, 512, 1024], [256, 768, 256, 768]],
            "head_config": [(16, 16), (16, 8), (16, 4)],
            "head_dim": [128, 64],
            "causal": [True, False],
            "ulysses_degree": [2, 4, 8],
        }

        for seqlens, (hq, hkv), hd, causal, ud in itertools.product(*[test_params[k] for k in test_params]):
            # Run each ulysses degree that divides the world as world//ud independent groups.
            if self.world_size % ud != 0:
                continue
            total = sum(seqlens)
            if total % ud != 0 or hq % ud != 0 or hkv % ud != 0:
                continue
            self.run_varlen_attn_with_cp(seqlens, hq, hkv, hd, causal, ud, device, dtype)
        dist.destroy_process_group()

    def run_varlen_attn_with_cp(self, seqlens, hq, hkv, hd, causal, ulysses_degree, device, dtype):
        # world//ud independent ulysses groups of size ud. The leading "dp" dim just
        # replicates the (identically-seeded) problem; it is NOT a ring group.
        # Cache the mesh per degree so we don't recreate NCCL subgroups every config.
        if not hasattr(self, "_mesh_cache"):
            self._mesh_cache = {}
        if ulysses_degree not in self._mesh_cache:
            self._mesh_cache[ulysses_degree] = init_device_mesh(
                "cuda",
                (self.world_size // ulysses_degree, ulysses_degree),
                mesh_dim_names=("dp", "ulysses"),
            )
        device_mesh = self._mesh_cache[ulysses_degree]
        ulysses_group = device_mesh["ulysses"].get_group()
        ring_group = None

        total = sum(seqlens)
        cu = make_cu_seqlens(seqlens, device)
        max_seqlen = max(seqlens)
        sm_scale = hd ** (-0.5)

        # Global packed tensors [total, h, d]; same seed -> identical on all ranks.
        q = torch.randn(total, hq, hd, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(total, hkv, hd, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(total, hkv, hd, device=device, dtype=dtype, requires_grad=True)

        q_ref = q.clone().detach().requires_grad_()
        k_ref = k.clone().detach().requires_grad_()
        v_ref = v.clone().detach().requires_grad_()

        # Single-GPU varlen reference over the full packed sequence.
        o_ref = attention_varlen_forward_pytorch_ref_impl(q_ref, k_ref, v_ref, cu, cu, sm_scale, causal)
        grad_ref = torch.randn_like(o_ref)
        o_ref.backward(grad_ref)

        cp_size = ulysses_group.size()
        cp_rank = ulysses_group.rank()

        q_local = shard_thd_by_rank(q, cp_size, cp_rank).detach().requires_grad_()
        k_local = shard_thd_by_rank(k, cp_size, cp_rank).detach().requires_grad_()
        v_local = shard_thd_by_rank(v, cp_size, cp_rank).detach().requires_grad_()

        o = pt.ops.flash_attn_varlen_usp_func(
            q_local,
            k_local,
            v_local,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=(-1, -1),
            bias=None,
            alibi_slopes=None,
            deterministic=False,
            return_lse=False,
            return_attn_probs=False,
            ulysses_group=ulysses_group,
            ring_group=ring_group,
        )

        grad_local = shard_thd_by_rank(grad_ref, cp_size, cp_rank).contiguous()
        o.backward(grad_local)

        o_ref_local = shard_thd_by_rank(o_ref, cp_size, cp_rank)
        dq_ref_local = shard_thd_by_rank(q_ref.grad, cp_size, cp_rank)
        dk_ref_local = shard_thd_by_rank(k_ref.grad, cp_size, cp_rank)
        dv_ref_local = shard_thd_by_rank(v_ref.grad, cp_size, cp_rank)

        assert compute_snr(o_ref_local, o) > 20, "out_snr too low"
        assert compute_snr(dq_ref_local, q_local.grad) > 15, "query_grad_snr too low"
        assert compute_snr(dk_ref_local, k_local.grad) > 15, "key_grad_snr too low"
        assert compute_snr(dv_ref_local, v_local.grad) > 15, "value_grad_snr too low"
