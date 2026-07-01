###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fully fused mega MoE forward (FlyDSL) over a single symmetric buffer.

Pipeline (EP intra-node, mirrors the EP test):

  1. dispatch_grouped_gemm_impl -- fused prologue (build the cross-rank dispatch handle)
                                   + cross-rank dispatch PUSH + grouped L1 GEMM (NT)
  3. swiglu                     -- fused SwiGLU activation
  4. grouped_gemm_combine_impl  -- grouped L2 GEMM (NT) + cross-rank combine PUSH
  5. weighted topk reduce       -- y[token] = sum_k w[token, k] * comb[k, token]

Cross-rank pool + scratch memory is carved out of ONE cached symmetric main
buffer (``SymmBuffer``, sized by ``get_symm_buffer_size_for_mega_moe`` --
inspired by ``deep_gemm/mega``); the spin-wait flags AND the combine buffer
(``comb``) live in the uncached signal pad -- the reduce reads ``comb`` through
the cache, so a cached ``comb`` would serve stale locally-zeroed rows. The stage
kernels are torch custom ops from ``primus_turbo.pytorch.kernels.mega_moe`` and are
called directly in the forward.

``MegaMoEFusedFunction`` wraps the forward for autograd; backward (conjugate via
Dispatch<->Combine duality, mirroring ``MegaKernelFlyDSL/ops/mega_moe.py``) returns
grads for x / w1 / w2 / topk_weights:

  1. dispatch_grouped_gemm_impl(layout="nn") -- dispatch dy + L2 dgrad (grad_swiglu = dispatch_l2_grad @ w2)
  2. swiglu_backward(scale=dispatch_weights_in_buf, grad_gate) -- re-inject routing weight + gate grad
  3. dW2 = grouped_gemm_variable_k(dispatch_l2_grad, act_weighted, trans_a) ; dW1 = variable_k(grad_l1, pool_x, trans_a)
  4. grouped_gemm_combine_impl(layout="nn") -- 3-role L1 dgrad GEMM + combine PUSH + dx reduce
  5. grad_topk_weights[t,k] = grad_gate scattered cross-rank into combine_gate (the gate ride-along of step 4)

Naming mirrors triton_dist (``ep_a2a_fused_layer`` / ``ep_moe_fused``): the per-pool-row
routing weight lives in ``weight_recv_buf`` (used as ``dispatch_weights_in_buf``); the prologue
rides each token's routing weight cross-rank into it (no all_gather). Step 4 reuses the forward's
fused 3-role kernel (``grouped_gemm_combine_impl``), with the combine role also scattering
``grad_gate`` -> ``combine_gate`` for ``grad_topk_weights`` (mirrors ``MegaKernelFlyDSL/ops/mega_moe.py``).
"""

import os

import torch

from primus_turbo.flydsl.mega.swiglu_kernel import swiglu, swiglu_backward

# SymmBuffer + sizing helpers live in the FlyDSL layer; re-exported here (below) so
# existing ``mega_moe_fused`` importers keep working. The symm buffer itself is hidden:
# the stage kernels fetch it internally, and everything this op needs rides the handle /
# the dispatch return -- no direct ``get_symm_buffer_for_mega_moe`` use here.
from primus_turbo.flydsl.mega.symm_buffer import (
    SymmBuffer,
    get_symm_buffer_size_for_mega_moe,
)
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
    grouped_gemm_variable_k_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
    dispatch_grouped_gemm_impl,
)
from primus_turbo.pytorch.kernels.mega_moe.grouped_gemm_combine_impl import (
    grouped_gemm_combine_impl,
)

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_size_for_mega_moe",
    "MegaMoEFusedFunction",
    "mega_moe_fused",
]

_FLYDSL = BackendType.FLYDSL.value

# Debug-only: HOST barrier between backward ops to test the peer-arrival spin hypothesis.
_BWD_BARRIER = os.environ.get("MEGA_BWD_BARRIER", "0") == "1"
# Lightweight DEVICE cross-rank barrier before each backward cross-rank PUSH (DeepEP
# device_sync_kernel style). The backward reuse path skips the prologue barrier, so under
# dW2 wgrad backend: flydsl TN variable-k (default) vs Triton variable-k grouped GEMM.
# backward dispatch comm-CU split (GEMM-bound -> fewer comm CUs frees CUs for GEMM).
_DISPATCH_CU_BWD = int(os.environ.get("MEGA_DISPATCH_CU_BWD", "16"))


def _bwd_barrier(group):
    """Re-align all ranks (sync + collective) so dispatch PUSH never spins."""
    if not _BWD_BARRIER:
        return
    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)


# --------------------------------------------------------------------------- #
# Forward / backward: call the stage kernels directly over the symmetric buffer.
# --------------------------------------------------------------------------- #
class MegaMoEFusedFunction(torch.autograd.Function):
    """Wraps the fused mega MoE forward so its output joins the autograd graph.

    Backward (conjugate via Dispatch<->Combine duality) returns grads for x, w1,
    w2, and topk_weights; topk_idx / group / tiling args are non-differentiable.

    NOTE: the symmetric buffer is cached/shared per (group, shape, tiling), and
    ``backward`` mutates it in place. A second ``forward`` with the SAME shape
    before this op's ``backward`` (e.g. another same-shape layer, grad
    accumulation, or activation recompute) would clobber the shared buffer; only
    a single forward->backward per shape between collectives is safe."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult, num_experts):
        with torch.profiler.record_function("mega_moe_forward"):
            num_tokens, hidden_size = x.shape
            num_topk = topk_idx.shape[-1]
            # int64 end-to-end (combine reads topk i64); no-op for int64 callers, single
            # cheap cast for int32 callers -> downstream prologue/combine never re-cast.
            topk_idx = topk_idx.to(torch.int64)

            ctx.set_materialize_grads(False)

            # 1+2) fused prologue + cross-rank dispatch PUSH + grouped L1 GEMM (NT):
            # the dispatch builds/fetches the active symm workspace, runs the fused prologue
            # (resets scoreboard -> 0 / barrier_local -> -1 in-kernel before its cross-rank
            # barrier), then dispatches pool[M,H] @ w1[g,2I,H] -> l1_out. The handle carries the
            # comm handle + tile tables + the live symm buffer (reused in backward / read below).
            # sb_l2 self-resets in the combine push role; comb is fully overwritten by the push.
            # standard 4-output API: (l1_out, dispatch_x_in_buf, dispatch_weights_in_buf, handle).
            # forward uses the routing weight (swiglu scale) + handle (origin/meta snapshot tail
            # restored on backward reuse); the dispatched pool (dispatch_x_in_buf) is unused here.
            # tiling is fixed at the kernel default (block_m/n = 256).
            l1_out, _, dispatch_weights_in_buf, handle = dispatch_grouped_gemm_impl(
                x,
                w1,
                group,
                _FLYDSL,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                layout="nt",
            )
            # handle is the flat prologue tuple: [7]=tile_to_expert, [9]=group_lens, [10]=group_offs

            # 3) fused SwiGLU activation: l1_out[M,2I] -> act[M,I] (bound rides the active symm)
            act = swiglu(l1_out)

            # 4) grouped L2 GEMM (NT) + cross-rank combine PUSH + 3-role topk reduce -> y [num_tokens, hidden]
            # CU split tunable via env; reduce runs on empty GEMM blocks so dedicated region defaults to 0
            _combine_cu = int(os.environ.get("MEGA_COMBINE_CU", "64"))
            _reduce_cu = int(os.environ.get("MEGA_REDUCE_CU", "0"))
            y, _ = grouped_gemm_combine_impl(
                act,
                w2,
                list(handle),
                _FLYDSL,
                topk_indices=topk_idx.contiguous().view(-1),
                topk_weights=topk_weights.to(torch.float32).contiguous().view(-1),
                num_combine_cu=_combine_cu,
                num_reduce_cu=_reduce_cu,
                layout="nt",
                BM=block_m,
                BN=block_n,
            )

            # ---- stash everything backward needs ----
            if any(ctx.needs_input_grad):
                # per-pool-row routing weight (triton_dist: dispatch_weights_in_buf): the prologue
                # rode each token's routing weight cross-rank into weight_recv_buf[dest_row] (no
                # all_gather). Returned by the dispatch -> clone before a later layer overwrites it.
                dispatch_weights_in_buf = dispatch_weights_in_buf.clone()

                ctx.group = group  # backward dispatches resolve the PG by name via the impl
                ctx.num_tokens = num_tokens
                ctx.num_topk = num_topk
                ctx.block_m = block_m
                ctx.block_n = block_n
                # save the full prologue handle directly (slot [7]=tile_to_expert, [9]=group_lens,
                # [10]=group_offs for the variable-K wgrads; tail [11..13] = this layer's
                # origin/meta snapshots, restored into the symm buffer on backward reuse).
                # grad_topk_weights is produced in-kernel in backward (swiglu_backward grad_gate +
                # combine gate-scatter), so the forward combine buffer is NOT saved.
                ctx.handle_len = len(handle)
                ctx.save_for_backward(
                    *handle,  # full dispatch prologue tuple + origin/meta snapshot tail
                    x,  # raw source x; backward re-dispatches it into the pool for dW1
                    l1_out,  # swiglu input (swiglu_backward; dW2 B = act_weighted recomputed here)
                    dispatch_weights_in_buf,
                    w1,
                    w2,
                    topk_idx,
                )
            return y

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_y):
        """Conjugate of forward via Dispatch<->Combine duality; grads for x / w1 / w2 / topk_w."""
        with torch.profiler.record_function("mega_moe_backward"):
            # set_materialize_grads(False) -> grad_y is None when the output got no grad
            if grad_y is None:
                return (None,) * 10
            saved = ctx.saved_tensors
            # the saved prologue handle (slot [7]=tile_to_expert, [9]=group_lens, [10]=group_offs;
            # tail [11..13] = origin/meta snapshots) followed by the operands.
            handle = tuple(saved[: ctx.handle_len])
            (
                saved_x,
                l1_out,
                dispatch_weights_in_buf,
                w1,
                w2,
                topk_idx,
            ) = saved[ctx.handle_len :]
            # block_m-padded per-expert lengths / prefix for the variable-K wgrads
            group_lens, group_offs = handle[9], handle[10]
            # symm is hidden: the dispatch kernel copies this layer's origin/meta snapshots
            # (handle tail) device-side into the symm regions on reuse, so the subsequent combine
            # reads the right routing/meta even after a later layer clobbered the shared buffer.
            num_tokens, num_topk = ctx.num_tokens, ctx.num_topk
            dy = grad_y.contiguous().to(torch.bfloat16)
            triton_be = BackendType.TRITON.value

            # ===== STEP 1: combine^T (dispatch dy) + L2 dgrad — fused dispatch + NN GEMM =====
            # No cross-rank barrier: scoreboard self-resets EARLY in the forward dispatch (its last
            # GEMM reader) while the backward dispatch's peer signal is LATE (a full fwd-pass after the
            # prologue barrier), so the reuse does not race; pool/comb are likewise freed by forward
            # long before backward reuses them. (Holds while per-iter rank skew < ~one pass.)
            # dispatch dy into the pool (-> dispatch_l2_grad, unweighted), then grad_swiglu = dispatch_l2_grad @ w2 (NN).
            # reuse the forward's handle (no prologue) via the rebuilt handle tuple.
            # the nn dispatch returns the pool (now holding the dispatched dy rows) as dispatch_x_in_buf
            _bwd_barrier(ctx.group)
            grad_swiglu, dispatch_l2_grad, _, _ = dispatch_grouped_gemm_impl(
                dy,
                w2,
                ctx.group,
                _FLYDSL,
                handle=handle,
                layout="nn",
                num_dispatch_cu=_DISPATCH_CU_BWD,
            )
            _bwd_barrier(ctx.group)

            # ===== STEP 2: SwiGLU^T (re-inject routing weight) + gate grad (= grad_topk_weights/row) =====
            # grad_gate[r] = <grad_swiglu_unweighted[r], act_unweighted[r]> = grad_topk_weights of pair (t,k);
            # from the UNSCALED dact (independent of weight placement). swiglu_backward allocates
            # grad_gate internally (torch.empty, single store/row) and returns it (triton_dist dscale).
            # act_weighted = (recomputed fwd act) * routing weight is emitted by swiglu_backward (folds
            # the host saved_act*weight mul AND removes the forward saved_act clone).
            grad_l1, grad_gate, act_weighted = swiglu_backward(
                grad_swiglu,
                l1_out,
                scale=dispatch_weights_in_buf,
                return_gate=True,
                return_act_w=True,
            )

            # ===== dW2 = dispatch_l2_grad^T @ act_weighted (variable-K wgrad; weight folded into act_weighted) =====
            # handle carries int64 group meta -> no cast
            _bwd_barrier(ctx.group)
            dW2 = grouped_gemm_variable_k_impl(
                dispatch_l2_grad,
                act_weighted,
                group_lens,
                group_offs,
                trans_a=True,
                trans_b=False,
                trans_c=False,
                num_cu=None,
                default_backend=triton_be,
            )
            _bwd_barrier(ctx.group)

            # ===== STEP 3: L1 dgrad (grad_pool = grad_l1 @ w1, NN) + combine PUSH + dx reduce =====
            # 3-role fused (mirrors forward STEP 4); the combine role also scatters grad_gate ->
            # origin combine_gate[token*topk+k] (grad_topk_weights). grad_l1 carries the weight -> unweighted reduce.
            # sb_l2 self-resets in the combine push role; combine_gate's scatter is a per-slot
            # overwrite (dropped pairs masked to 0 below).
            # STEP3 is GEMM-bound (K=2I, twice the forward), so the combine fully hides under the
            # GEMM with FEWER dedicated combine CUs -> the GEMM keeps more CUs. (Forward STEP4 is
            # combine-bound and wants 64; sharing one value left ~0.4ms of GEMM CU-starvation here.)
            # Separate env MEGA_COMBINE_CU_BWD (default 20, swept best); reduce on empty GEMM blocks.
            _combine_cu = int(os.environ.get("MEGA_COMBINE_CU_BWD", "20"))
            _reduce_cu = int(os.environ.get("MEGA_REDUCE_CU", "0"))
            # dx flag re-arm: the combine reduce re-arms barrier_local -> -1 on consume (see
            # grouped_gemm_combine_bf16_kernel; requires num_reduce_cu == 0), so the forward reduce
            # already left the flags in the "wait" state for this backward STEP3 reuse -- no manual
            # prologue-style re-arm / cross-rank barrier needed here.
            # combine scatters grad_gate; the reduce folds masked grad_topk_weights and returns dx [T, hidden]
            _bwd_barrier(ctx.group)
            dx, grad_topk_weights_flat = grouped_gemm_combine_impl(
                grad_l1,
                w1,
                list(handle),
                _FLYDSL,
                topk_indices=topk_idx.contiguous().view(-1),
                topk_weights=None,
                grad_gate=grad_gate,
                num_combine_cu=_combine_cu,
                num_reduce_cu=_reduce_cu,
                layout="nn",
                BM=ctx.block_m,
                BN=ctx.block_n,
            )
            _bwd_barrier(ctx.group)

            # ===== dW1 = pool(x)^T @ grad_l1 (variable-K TN wgrad) =====
            # fwd saved raw x (not the pool clone) -> re-dispatch it into the pool and fuse the
            # TN wgrad: dW1[g] = x_pool[g]^T @ grad_l1[g]. trans_c stores [G, 2I, H] (W1-native).
            # tn dispatch & wgrad touch independent data; group_offs rides the handle slot [10].
            _bwd_barrier(ctx.group)
            dW1, _, _, _ = dispatch_grouped_gemm_impl(
                saved_x,
                grad_l1,
                ctx.group,
                _FLYDSL,
                handle=handle,
                layout="tn",
                trans_c=True,
                num_dispatch_cu=_DISPATCH_CU_BWD,
            )
            _bwd_barrier(ctx.group)

            # grad_topk_weights[t,k]: produced (masked + decoupled into a fresh buffer) by the combine reduce;
            # just reshape the [combine_slots] output to [num_tokens, num_topk].
            grad_topk_weights = grad_topk_weights_flat.view(num_tokens, num_topk)
            # grads for (x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult, num_experts)
            return (
                dx,
                None,
                grad_topk_weights,
                dW1.to(w1.dtype),
                dW2.to(w2.dtype),
                None,
                None,
                None,
                None,
                None,
            )


def mega_moe_fused(
    group, x, topk_idx, topk_weights, w1, w2, *, block_m=256, block_n=256, pool_mult=2, num_experts=None
) -> torch.Tensor:
    """One fully fused mega MoE forward; the symmetric buffer is fetched (and cached)
    internally by ``dispatch_grouped_gemm_impl`` from the tensor shapes + ``group``.

    ``x`` [num_tokens, hidden] bf16, ``topk_idx`` / ``topk_weights`` [num_tokens, num_topk],
    ``w1`` [experts_per_rank, 2*intermediate_hidden, hidden],
    ``w2`` [experts_per_rank, hidden, intermediate_hidden]. Returns y [num_tokens, hidden].
    The buffer is allocated once per (group, shape, tiling) and reused on later calls.
    ``num_experts`` defaults to ``experts_per_rank * world_size`` (the global expert count)."""
    if num_experts is None:
        num_experts = w1.shape[0] * group.size()
    return MegaMoEFusedFunction.apply(
        x, topk_idx, topk_weights, w1, w2, group, block_m, block_n, pool_mult, num_experts
    )
