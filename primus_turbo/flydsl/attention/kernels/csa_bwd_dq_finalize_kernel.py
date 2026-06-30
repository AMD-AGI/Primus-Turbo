# SPDX-License-Identifier: Apache-2.0
"""csa_bwd_dq_finalize: dgathered split-K finalize reduce.

Companion to the csa_bwd_full split-K scratch. The full backward kernel no
longer atomic_fadd's its gathered gradient into the fp32 ``DGATHERED`` word;
instead each of the ``num_groups = HQ/HEAD_BLOCK`` head-group programs writes a
DISJOINT ``group_id`` stripe of a packed-bf16 split scratch
``DGATHERED_SPLIT[B, Sq, K_topk, num_groups, D]`` with a plain (race-free)
packed 2xbf16 store -- no same-word atomic-RMW serialization and half the
accumulation store bytes.

This kernel is the conflict-free finalize pass: it sums the ``num_groups`` bf16
stripes of each ``(b, q, k_pos, d)`` element back into the fp32 ``DGATHERED``
output. It is a flat element-parallel pass (one thread per output element), so
its global reads are coalesced within a row and it adds exactly one extra pass
over the (half-byte) split scratch.
"""
from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, fly as _fly, llvm as _llvm
from flydsl.expr import const_expr  # noqa: E402
from primus_turbo.flydsl.attention.kernels.kernels_common import dtype_to_elem_type

KERNEL_NAME = "csa_bwd_dgathered_finalize_kernel"
_LLVM_GEP_DYNAMIC = -2147483648


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def build_csa_bwd_dgathered_finalize_module(
    head_dim,
    num_groups,
    dtype_str="bf16",
    block_threads=256,
):
    """Build the dgathered split-K finalize launcher.

    Inputs:
        SPLIT  bf16 [B, Sq, K_topk, num_groups, D]  (flat element view)
        OUT    f32  [B, Sq, K_topk, D]              (flat element view)
        n_elems: number of OUT elements (= B*Sq*K_topk*D)

    Reduces the ``num_groups`` contiguous-stride stripes of each OUT element.
    """
    gpu_arch = get_hip_arch()  # noqa: F841 (kept for parity / future arch gates)
    D = int(head_dim)
    G = int(num_groups)
    assert G >= 1, "num_groups must be >= 1"
    BLOCK = int(block_threads)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def csa_bwd_dgathered_finalize_kernel(
        SPLIT: fx.Tensor,
        OUT: fx.Tensor,
        n_elems: fx.Int32,
    ):
        elem_type = dtype_to_elem_type(dtype_str)
        f32_ty = T.f32
        fm_fast = arith.FastMathFlags.fast

        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=True)
        # SPLIT is read through a 64-bit flat GEP: its element count
        # B*Sq*K_topk*num_groups*D exceeds the 32-bit buffer voffset range
        # (4 GB) for wide MHA shapes, so a buffer_load voffset would wrap.
        split_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), SPLIT)

        bid = arith.index_cast(T.i32, fx.block_idx.x)
        tid = arith.index_cast(T.i32, fx.thread_idx.x)
        c_block = arith.constant(BLOCK, type=T.i32)
        e = arith.AddIOp(arith.MulIOp(bid, c_block).result, tid).result

        active = arith.cmpi(arith.CmpIPredicate.slt, e, n_elems)
        c_zero_i32 = arith.constant(0, type=T.i32)
        e_safe = arith.select(active, e, c_zero_i32)

        # row = e // D, col = e % D; SPLIT element of stripe g is
        # (row*G + g)*D + col. Accumulated in explicit 64-bit arithmetic so the
        # (row*G*D) product does not overflow i32 for large split scratches.
        e64 = arith.extsi(T.i64, e_safe)
        D64 = arith.constant(D, type=T.i64)
        row = arith.DivUIOp(e64, D64).result
        col = arith.RemUIOp(e64, D64).result
        gd64 = arith.constant(G * D, type=T.i64)
        base = arith.AddIOp(arith.MulIOp(row, gd64).result, col).result

        acc = arith.constant(0.0, type=f32_ty)
        for g in range_constexpr(G):
            off_i64 = arith.AddIOp(base, arith.constant(g * D, type=T.i64)).result
            gep = _llvm.GEPOp(
                _llvm_ptr_ty(), split_ptr, [off_i64],
                rawConstantIndices=[_LLVM_GEP_DYNAMIC],
                elem_type=elem_type, noWrapFlags=0,
            )
            v = _llvm.LoadOp(elem_type, gep.result).result
            v_f32 = arith.extf(f32_ty, v)
            acc = arith.AddFOp(acc, v_f32, fastmath=fm_fast).result

        _if = scf.IfOp(active, [], has_else=False)
        with ir.InsertionPoint(_if.then_block):
            buffer_ops.buffer_store(acc, out_rsrc, e_safe)
            scf.YieldOp([])

    @flyc.jit
    def launch_csa_bwd_dgathered_finalize(
        SPLIT: fx.Tensor,
        OUT: fx.Tensor,
        n_elems: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        from flydsl.expr.arith import ArithValue
        n_idx = ArithValue(n_elems).index_cast(T.index)
        grid_x = (n_idx + arith.index(BLOCK - 1)) // arith.index(BLOCK)
        launcher = csa_bwd_dgathered_finalize_kernel(SPLIT, OUT, n_elems)
        launcher.launch(
            grid=(grid_x, arith.index(1), arith.index(1)),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch_csa_bwd_dgathered_finalize
