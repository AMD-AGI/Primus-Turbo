from abc import ABC
from typing import List

import torch

from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (
    get_f8_fwd_dtype,
)
from primus_turbo.triton.attention.attention_kernel import FIXED_BLOCK_M


def _check_and_convert(t, scale, float8_fw):
    finfo = torch.finfo(float8_fw)
    return (t * scale).clamp(min=finfo.min, max=finfo.max).to(dtype=float8_fw) if t.dtype != float8_fw else t


def block_scaling_node(tensor, use_fp8, BLOCK_M=FIXED_BLOCK_M, float8_dtype=get_f8_fwd_dtype()):
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
    if use_fp8:
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
    else:
        scale = torch.tensor([1.0], device=tensor.device)
        return tensor, scale


def quant_v_get_p_scale(v, use_fp8: bool):
    """
    Get p_scale for quant_v_getp_scale
    """
    if use_fp8:
        range_v = torch.max(torch.abs(v))

        float8_fw = torch.float8_e4m3fnuz
        dtype_max = torch.finfo(float8_fw).max

        v_scale = dtype_max / range_v
        p_scale = dtype_max
        v = _check_and_convert(v, v_scale, float8_fw)

    else:
        v_scale = torch.tensor([1.0], device=v.device)
        p_scale = 1.0
        p_scale = 1.0

    return v, v_scale, p_scale


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
            output_list.append(t.chunk(cp_size, seq_dim)[cp_rank].contiguous())

        return output_list


class All2AllAttentionCommunicator:
    """All2AllAttentionCommunicator Impl"""

    cp_comm_streams = []

    def __init__(self, cp_group):
        self.cp_group = cp_group
        self.done_flags = None
        if len(All2AllAttentionCommunicator.cp_comm_streams) == 0:
            All2AllAttentionCommunicator.cp_comm_streams.append(torch.cuda.Stream())
            All2AllAttentionCommunicator.cp_comm_streams.append(torch.cuda.Stream())

    def data_exchange_over_cp_groups(
        self, send_buffers: List[torch.Tensor], before_all2all_funcs=None, after_all2all_funcs=None
    ):
        recv_buffers = [torch.empty_like(x) for x in send_buffers]

        cp_streams = All2AllAttentionCommunicator.cp_comm_streams
        for stream in cp_streams:
            stream.wait_stream(torch.cuda.current_stream())

        before_all2all_done_events = [torch.cuda.Event() for _ in range(len(send_buffers))]
        all2all_done_flags = [None for _ in range(len(send_buffers))]

        # 3-stages pipeline
        pipeline_rounds = len(send_buffers) + 2
        for i in range(pipeline_rounds):
            if i < pipeline_rounds - 2:
                with torch.cuda.stream(cp_streams[0]):
                    if before_all2all_funcs is not None:
                        send_buffers[i], recv_buffers[i] = before_all2all_funcs[i](
                            send_buffers[i], recv_buffers[i]
                        )
                    send_buffers[i] = send_buffers[i].contiguous().flatten()
                    before_all2all_done_events[i].record()

            if i > 0 and i < pipeline_rounds - 1:
                before_all2all_done_events[i - 1].wait()
                send_tensor = send_buffers[i - 1]
                recv_tensor = recv_buffers[i - 1]

                all2all_done_flags[i - 1] = torch.distributed.all_to_all_single(
                    recv_tensor, send_tensor, group=self.cp_group, async_op=True
                )

            if i > 1:
                with torch.cuda.stream(cp_streams[1]):
                    all2all_done_flags[i - 2].wait()
                    all2all_done_tensor = recv_buffers[i - 2]
                    if after_all2all_funcs is not None:
                        recv_buffers[i - 2] = after_all2all_funcs[i - 2](all2all_done_tensor)

        for stream in cp_streams:
            torch.cuda.current_stream().wait_stream(stream)
        return recv_buffers
