from abc import ABC
from typing import List

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    get_f8_fwd_dtype,
)
from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M, FIXED_BLOCK_N

__all__ = ["block_scaling_node"]


def block_scaling_node(tensor, BLOCK_M=FIXED_BLOCK_M, float8_dtype=get_f8_fwd_dtype()):
    """
    Used to scale tensor in per-block mode

    Inputs:
        tensor(Tensor): bf16 tensor
        BLOCK_M(int): triton block size
        float8_dtype(Tensor.dtype): float8_dtype

    Output:
        fp8tensor(Tensor): tensor after blockwise quant
        unscale_tensor(Tensor): tensor for unscale quanted tensor from fp8 to bf16
    """
    tensor = tensor.permute(0, 2, 1, 3)  # [B, H, L, D]
    B, H, L, D = tensor.shape
    tensor = tensor.reshape(B, H, L // BLOCK_M, BLOCK_M, D).reshape(B, H, L // BLOCK_M, BLOCK_M * D)
    MAX_E4M3 = torch.finfo(float8_dtype).max
    scale = MAX_E4M3 / tensor.abs().max(dim=-1)[0]
    tensor = tensor * scale.reshape(scale.shape + (1,))
    tensor = tensor.clamp(-MAX_E4M3, MAX_E4M3)
    tensor = tensor.to(float8_dtype)
    tensor = tensor.reshape(B, H, L, D).permute(0, 2, 1, 3).contiguous()
    # [B, L, H, D]
    return tensor, 1.0 / scale.to(torch.float32).contiguous()


def blockwise_scaling_qkv_to_fp8(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, use_fp8: bool):
    """
    blockwise scaling qkv to fp8, which to be used for triton fp8 fa kernel
    """
    if use_fp8:
        # online quant
        range_v = torch.max(torch.abs(v))
        float8_fw = torch.float8_e4m3fnuz
        dtype_max = torch.finfo(float8_fw).max
        v_scale = dtype_max / range_v
        p_scale = dtype_max

        def check_and_convert(t, scale):
            finfo = torch.finfo(float8_fw)
            return (
                (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw)
                if t.dtype != float8_fw
                else t
            )

        q, q_scale = block_scaling_node(q, FIXED_BLOCK_M)
        k, k_scale = block_scaling_node(k, FIXED_BLOCK_N)
        v = check_and_convert(v, v_scale)
    else:
        use_fp8 = False
        q_scale = torch.tensor([1.0], device=q.device)
        k_scale = torch.tensor([1.0], device=q.device)
        v_scale = torch.tensor([1.0], device=q.device)
        p_scale = 1.0

    return q, k, v, p_scale, q_scale, k_scale, v_scale


class AttentionSharder(ABC):
    """AttentionSharder Interface"""

    def shard_cp_input(self, input_tensors: List[torch.Tensor], cp_group) -> List[torch.Tensor]:
        """
        Shard input from whole seq to specific cp rank, the implementation differ from different cp-comm type

        Inputs:
            input_tensors: tensors to shard as [Q, K, V]
            cp_groups: cp communication process group
        """


class All2AllAttentionSharder(AttentionSharder):
    """All2All AttentionSharder Impl"""

    def shard_cp_input(self, input_tensors: List[torch.Tensor], cp_group, seq_dim=1) -> List[torch.Tensor]:
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()

        output_list = []
        for t in input_tensors:
            output_list.append(t.chunk(cp_size, seq_dim)[cp_rank])

        return output_list


class AttentionCommunicator(ABC):
    """AttentionCommunicator Interface"""

    def data_exchange_over_cp_groups_async(
        self, send_buffers: List[torch.Tensor], recv_buffers: List[torch.Tensor], round: int
    ):
        pass

    def wait_data_exchange_done(self):
        pass


class All2AllAttentionCommunicator(AttentionCommunicator):
    """All2AllAttentionCommunicator Impl"""

    def __init__(self, cp_group):
        self.cp_group = cp_group
        self.done_flags = None

    def data_exchange_over_cp_groups_async(
        self, send_buffers: List[torch.Tensor], recv_buffers: List[torch.Tensor], round: int
    ):
        done_flags = []
        for i in range(len(send_buffers)):
            send_tensor = send_buffers[i].contiguous().flatten()
            recv_tensor = recv_buffers[i]
            done_flags.append(
                torch.distributed.all_to_all_single(
                    recv_tensor, send_tensor, group=self.cp_group, async_op=True
                )
            )

        self.done_flags = done_flags

    def wait_data_exchange_done(self):
        for done_flag in self.done_flags:
            done_flag.wait()
