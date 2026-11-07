###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Persistent causal forward attention on the gfx950 dual-wave tile (sbhd, bf16, D = 64 / 128).

One CTA per CU claims ``(head group, q block, batch)`` work items from a counter, in the dispatch order
of the one-CTA-per-tile launch (head group fastest, q blocks longest first), and runs the same tile
body (``flash_attn_fwd.make_dualwave_swp_tile``) on each. The building block of a kernel whose
attention workers start on a tile as soon as its inputs land.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.compiler.kernel_function import CompilationContext

from primus_turbo.flydsl.attention.flash_attn_fwd import (
    DUALWAVE_SWP_COMPILE_HINTS,
    make_dualwave_swp_tile,
)
from primus_turbo.flydsl.ulysses.common import (
    claim_unit,
    exit_and_bump_epoch,
    i64_base,
)
from primus_turbo.flydsl.utils.attn_helper import (
    SCORE_BOUND_HEAD_DIM_MAX,
    SCORE_BOUND_KV_CHUNKS,
    DualwaveKernelContext,
    _make_dualwave_swp_traits,
    dtype_to_elem_type,
)
from primus_turbo.flydsl.utils.prims import ld
from primus_turbo.triton.attention.score_bound import k_sq_norm_max


def make_traits(num_heads, num_kv_heads, head_dim, emit_lse):
    """The production sbhd forward configuration (``attention_flydsl_impl._fwd_module``)."""
    return _make_dualwave_swp_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        waves_per_eu=2,
        dualwave_swp_enable_stagger=False,
        emit_lse=emit_lse,
        block_m=64 if head_dim == 64 else 128,
        sbhd=True,
    )


@ASTRewriter.transform
def _work_loop(thread_index, state_base, claim_slot, num_units, num_q_blocks, head_groups, run_tile):
    unit = claim_unit(thread_index, state_base, claim_slot)
    while unit < num_units:
        rest = unit // fx.Int32(head_groups)
        run_tile((unit % fx.Int32(head_groups), rest % num_q_blocks, rest // num_q_blocks))
        unit = claim_unit(thread_index, state_base, claim_slot)


@functools.lru_cache(maxsize=16)
def _build(num_heads, num_kv_heads, head_dim, emit_lse, grid):
    traits = make_traits(num_heads, num_kv_heads, head_dim, emit_lse)
    attn_tile = make_dualwave_swp_tile(traits)
    head_groups = traits.NUM_HEADS_Q // traits.Q_HEADS_PER_WG
    elem = dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem, traits.LDS_KV_TOTAL_SIZE, 16]
        claim: fx.Array[fx.Int32, 1, 16]

    @flyc.kernel(name="mega_attn_fwd_persistent_kernel", known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        ScoreBound: fx.Tensor,
        STATE: fx.Tensor,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        num_q_blocks: fx.Int32,
        num_units: fx.Int32,
    ):
        thread_index = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claim_slot = fx.Int32(fx.ptrtoint(lds.claim.ptr))
        state_base = i64_base(STATE)
        epoch = ld(state_base, fx.Int32(0), dtype=fx.T.i64)

        def run_tile(block_ids):
            ctx = DualwaveKernelContext(
                traits,
                Q,
                K,
                V,
                O,
                LSE,
                O,
                O,
                O,
                seq_len,
                seq_len,
                stride_q_n,
                stride_kv_n,
                fx.Int32(traits.HEAD_DIM),
                fx.Int32(0),
                SINK=O,
                ScoreBound=ScoreBound,
            )
            ctx._setup(SharedStorage, block_ids=block_ids, lds=lds)
            attn_tile(ctx)

        _work_loop(thread_index, state_base, claim_slot, num_units, num_q_blocks, head_groups, run_tile)
        exit_and_bump_epoch(thread_index, state_base, epoch, grid)

    passthrough = [
        ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
        ["no-nans-fp-math", "true"],
        ["unsafe-fp-math", "true"],
    ]

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        ScoreBound: fx.Tensor,
        STATE: fx.Tensor,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        num_q_blocks: fx.Int32,
        num_units: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            Q,
            K,
            V,
            O,
            LSE,
            ScoreBound,
            STATE,
            seq_len,
            stride_q_n,
            stride_kv_n,
            num_q_blocks,
            num_units,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "llvm.passthrough": passthrough if traits.DAZ else None,
            },
        ).launch(grid=(grid, 1, 1), block=(traits.BLOCK_SIZE, 1, 1), stream=stream)

    return traits, launch


_STATES = {}
_COMPILED = {}


def _state(device):
    """[epoch, exit counter, claim counter, unused] int64; the last CTA out resets the counters."""
    if device not in _STATES:
        _STATES[device] = torch.zeros(4, dtype=torch.int64, device=device)
    return _STATES[device]


def attn_fwd_persistent(q, k, v, return_lse=True):
    """``q`` [Sq, B, Hq, D], ``k`` / ``v`` [Sq, B, Hkv, D] contiguous bf16, causal, 1/sqrt(D) scale.
    Returns ``out`` like ``q`` and the LSE ``[B * Sq, Hq]`` fp32 (as ``flash_attn_sbhd_flydsl_forward``)."""
    Sq, B, Hq, D = q.shape
    Hkv = k.shape[2]
    assert k.shape[0] == Sq, "self-attention only"
    assert all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in (q, k, v))
    grid = torch.cuda.get_device_properties(q.device).multi_processor_count
    traits, launch = _build(Hq, Hkv, D, bool(return_lse), grid)
    score_bound = None
    if D <= SCORE_BOUND_HEAD_DIM_MAX:
        score_bound = k_sq_norm_max(k, SCORE_BOUND_KV_CHUNKS)
    out = torch.empty_like(q)
    lse = torch.empty((B * Sq, Hq), device=q.device, dtype=torch.float32) if return_lse else None
    num_q_blocks = -(-Sq // traits.BLOCK_M)
    num_units = (Hq // traits.Q_HEADS_PER_WG) * num_q_blocks * B
    args = (
        q,
        k,
        v,
        out,
        out if lse is None else lse,
        out if score_bound is None else score_bound,
        _state(q.device),
        Sq,
        B * Hq * D,
        B * Hkv * D,
        num_q_blocks,
        num_units,
    )
    stream = torch.cuda.current_stream()
    key = (Hq, Hkv, D, bool(return_lse), grid) + args[7:]
    fn = _COMPILED.get(key)
    if fn is None:
        with CompilationContext.compile_hints(DUALWAVE_SWP_COMPILE_HINTS):
            fn = flyc.compile(launch, *args, stream)
        _COMPILED[key] = fn
    fn(*args, stream)
    return out, lse
