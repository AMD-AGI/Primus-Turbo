import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.ops.utils.attention_utils import All2AllAttentionSharder
from tests.pytorch.ref.attention_ref import (
    AttnConfig,
    attention_vanilla_forward_pytorch_ref_impl,
)
from tests.test_utils import compute_snr

test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(
        seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128
    ),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


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
    @parametrize("batch", [4])
    @parametrize("config", test_cases)
    @parametrize("causal", [True])
    @parametrize("backend_type", ["triton"])
    @parametrize("cp_comm_type", ["a2a"])
    def test_attention_with_cp_fp8(self, batch, config, causal, backend_type, cp_comm_type):
        self._init_process()
        cp_group = dist.group.WORLD

        device = "cuda"
        dtype = torch.bfloat16
        if cp_comm_type == "a2a":
            input_sharder = All2AllAttentionSharder()
        else:
            raise NotImplementedError()

        seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
            config.seqlen_q,
            config.seqlen_kv,
            config.num_head_q,
            config.num_head_kv,
            config.head_dim_qk,
            config.head_dim_v,
        )
        q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
        k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
        v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

        query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
        key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
        query_ref = query.clone().detach().requires_grad_()
        key_ref = key.clone().detach().requires_grad_()
        value_ref = value.clone().detach().requires_grad_()

        sm_scale = query.shape[-1] ** (-0.5)
        o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)

        grad_ref = torch.randn(*o_ref.shape, device=device, dtype=dtype)
        o_ref.backward(grad_ref)
        o_ref, dq_ref, dk_ref, dv_ref = input_sharder.shard_cp_input(
            [o_ref, query_ref.grad, key_ref.grad, value_ref.grad], cp_group
        )

        # attention with CP
        cp_stream = torch.cuda.Stream()
        cp_param_bundle = {"cp_group": cp_group, "cp_stream": cp_stream, "cp_comm_type": cp_comm_type}

        query_local_token, key_local_token, value_local_token = input_sharder.shard_cp_input(
            [query, key, value], cp_group
        )

        o = pt.ops.attention_fp8_blockwise(
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
            backend_type=backend_type,
            cp_param_bundle=cp_param_bundle,
        )

        grad = input_sharder.shard_cp_input([grad_ref], cp_group)[0]
        o.backward(grad)

        dq, dk, dv = input_sharder.shard_cp_input([query.grad, key.grad, value.grad], cp_group)

        out_snr = compute_snr(o_ref, o)
        query_grad_snr = compute_snr(dq_ref, dq)
        key_grad_snr = compute_snr(dk_ref, dk)
        value_grad_snr = compute_snr(dv_ref, dv)

        assert out_snr > 20, "out_snr too low"
        assert query_grad_snr > 15, "query_grad_snr too low"
        assert key_grad_snr > 15, "key_grad_snr too low"
        assert value_grad_snr > 15, "value_grad_snr too low"
