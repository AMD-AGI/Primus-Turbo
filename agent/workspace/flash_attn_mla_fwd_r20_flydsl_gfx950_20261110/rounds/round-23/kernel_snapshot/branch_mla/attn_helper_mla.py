###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Shared MLIR-dialect helpers and ``s_waitcnt`` bit-field constants for the gfx950
dual-wave, software-pipelined flash-attention kernels.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TargetAddressSpace
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from primus_turbo.flydsl.utils import buffer_ops
from primus_turbo.flydsl.utils.prims import LOG2E

# FlyDSL's on-disk kernel cache hashes the kernel/launcher source, helpers from their own directory
# and closure scalars -- not this module (nor prims.py), so code emitted from here reaches that key
# only through DualwaveSwpTraits.cache_tag. Bump it with any change here that alters a generated
# kernel, or a warm cache keeps serving the old one.
CODEGEN_REV = 1062

# Room a later key has above a tile's row max before it forces another rebase.
REF_MARGIN = 96.0

# gfx950 hands workgroups to its XCDs round-robin in linear-id order, and each XCD has its own L2.
NUM_XCD = 8
# q-blocks of one head kept adjacent in an XCD's dispatch order, so they stream its K/V together.
# Sizes largest first: as many full groups of each size as the ranks ahead of the tail allow, the
# remainder going to the next size. 32 = the CTAs one XCD holds at once (32 CUs, 1 CTA per CU).
XCD_Q_GROUPS = (32, 16, 8)
# Last q ranks (shortest causal jobs) dispatched one rank at a time: a grouped tail would leave
# jobs of a group's different lengths in each XCD's final wave and idle CUs behind the longest.
XCD_TAIL_Q = 8


def host_job_order(nh, nq, causal):
    """(head, q block) of each in-batch linear CTA id, as `DualwaveKernelContext.block_item`
    maps it on the device; the persistent schedule replays this order on the host."""
    order = []
    for lin in range(nh * nq):
        if nh % NUM_XCD == 0:
            hpx = nh // NUM_XCD
            slot = lin // NUM_XCD
            nq_head = nq - XCD_TAIL_Q if nq > XCD_TAIL_Q else 0
            lo = 0
            h_local = q_rank = None
            for i, g in enumerate(XCD_Q_GROUPS):
                u = slot - lo * hpx
                grp, r = u // (g * hpx), u % (g * hpx)
                if i == 0 or slot >= lo * hpx:
                    h_local, q_rank = r // g, lo + grp * g + r % g
                lo = lo + (nq_head - lo) // g * g
            if slot >= lo * hpx:
                t = slot - lo * hpx
                h_local, q_rank = t % hpx, lo + t // hpx
            h = h_local * NUM_XCD + lin % NUM_XCD
        else:
            h, q_rank = lin % nh, lin // nh
        order.append((h, nq - 1 - q_rank if causal else q_rank))
    return order


# A reference at or below this is still the -3e38 floor of a row that has met no live key yet.
# Such a row seeds its QK with 0: seeding -reference would swamp every score it accumulates.
SEED_FLOOR_LIMIT = -1.0e30

# Slack on the Cauchy-Schwarz logit bound: the fp32 norm reductions on both sides, plus the
SCORE_BOUND_SLACK = 1.0 + 2.0**-6
SCORE_BOUND_KV_CHUNKS = 8
# Head dims the bound gate is armed for. Above it the kernel keeps the exact per-tile trigger
SCORE_BOUND_HEAD_DIM_MAX = 64


def _wave_any(cond):
    """Wave-uniform "some lane satisfies cond", as one ballot and one compare."""
    mask = rocdl.ballot(fx.Int64.ir_type, as_mlir_value(cond))
    return ArithValue(fx.Int64(mask) != fx.Int64(0))


def _if_wave(cond, vals, then_fn):
    """Wave-uniform branch yielding ``vals``; the else arm passes them through."""
    from flydsl._mlir.dialects import scf

    _v = [as_mlir_value(v) for v in vals]
    op = scf.IfOp(as_mlir_value(cond), [x.type for x in _v], has_else=True)
    with ir.InsertionPoint(op.regions[0].blocks[0]):
        scf.YieldOp([as_mlir_value(x) for x in then_fn()])
    if not op.regions[1].blocks:
        op.regions[1].blocks.append()
    with ir.InsertionPoint(op.regions[1].blocks[0]):
        scf.YieldOp(_v)
    return list(op.results)


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')")


# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3

_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'

# Wait and low-level ROCDL wrappers


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (n & _VMCNT_LO_MASK) | _LGKMCNT_EXPCNT_BASE | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    rocdl.s_waitcnt(val)


def _s_waitcnt(val):
    rocdl.s_waitcnt(val)


def _sched_barrier(val):
    rocdl.sched_barrier(val)


def _s_barrier():
    rocdl.s_barrier()


def _s_setprio(val):
    rocdl.s_setprio(val)


def _dualwave_sync_barrier():
    rocdl.s_barrier()


def _s_nop(x):
    if not isinstance(x, int) or not 0 <= x <= 15:
        raise ValueError("s_nop immediate must be a Python int in [0, 15]")
    llvm.inline_asm(ir.Type.parse("!llvm.void"), [], f"s_nop {x}", "", has_side_effects=True)


def _ds_read_tr16_b64(traits, result_type, base_ptr, imm_bytes, buf_id):
    """gfx950 ds_read_b64_tr_b16 at a constant byte offset from a per-lane V base.
    Emitted as the rocdl op, not inline asm: the op carries a real LDS memory effect so
    SIInsertWaitcnts places lgkmcnt itself, whereas asm would need hand-written waits.
    """
    scope_name = _dualwave_lds_scope("v", buf_id)
    ptr = buffer_ops.get_element_ptr(base_ptr, byte_offset=int(imm_bytes), elem_type=T.i8)
    return rocdl.ds_read_tr16_b64_(
        result_type,
        as_mlir_value(ptr),
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    )


# Arithmetic and inline-asm primitives


def _fadd(a, b, fm_fast):
    return arith.addf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fsub(a, b, fm_fast):
    return arith.subf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmul(a, b, fm_fast):
    return arith.mulf(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast)


def _fmax(a, b, fm_fast):
    return arith.MaxNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _fmin(a, b, fm_fast):
    return arith.MinNumFOp(as_mlir_value(a), as_mlir_value(b), fastmath=fm_fast).result


def _mfma_acc(a, b, c, _mma_atom, mfma_acc_vec_type):
    return fly.mma_atom_call_ssa([mfma_acc_vec_type], _mma_atom, a, b, c)


# The backend picks the VGPR MFMA form, so at 64 rows per wave the O accumulators (128 dwords)
# crowd S, P and the K/V packs out of the 256 arch VGPRs and every tile pays AGPR spill copies.
# Pinning the P*V accumulator to AGPRs in asm keeps O there for the whole loop. The hazard
# recognizer cannot see an MFMA inside asm, so any VALU read of O must first clear
# PV_AGPR_READ_NOPS wait states (`_pv_agpr_settle`).
PV_AGPR_READ_NOPS = 18


def _mfma16_acc_agpr(a, b, c):
    """16x16x32 MFMA with the accumulator c (vector<4xf32>) read and written in place in AGPRs."""
    return llvm.inline_asm(
        ir.Type.parse("vector<4xf32>"),
        [as_mlir_value(a), as_mlir_value(b), as_mlir_value(c)],
        "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $3",
        "=a,v,v,0",
        # Ordered against sched_barrier and the LDS reads: the scheduler cannot model asm, so
        # the P*V sites place these by hand.
        has_side_effects=True,
    )


def _pv_agpr_settle():
    """Wait out an asm P*V MFMA before a VALU read of its AGPR accumulator."""
    _sched_barrier(0)
    for n in (8, 8, PV_AGPR_READ_NOPS - 16):
        _s_nop(n - 1)
    _sched_barrier(0)


def _concat_vectors(lhs, rhs):
    lhs_vec = Vec(lhs)
    rhs_vec = Vec(rhs)
    return lhs_vec.shuffle(
        rhs_vec,
        list(range(lhs_vec.numel)) + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
    )


def _bitcast_i32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Int32.ir_type))


def _bitcast_f32(value):
    return as_mlir_value(ArithValue(value).bitcast(fx.Float32.ir_type))


def _apply_dualwave_mask_pair(s_values, rel_i32, neg_inf_i32, pair_thresholds, cmp):
    for p in range_constexpr(len(pair_thresholds)):
        thr_x, thr_y = pair_thresholds[p]
        idx_x = p * 2
        idx_y = p * 2 + 1
        x_bits = _bitcast_i32(s_values[idx_x])
        y_bits = _bitcast_i32(s_values[idx_y])
        asm_str = (
            f"v_cmp_{cmp}_i32_e64 $0, $6, {int(thr_x)}\n\t"
            f"v_cmp_{cmp}_i32_e64 $1, $6, {int(thr_y)}\n\t"
            "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
            "v_cndmask_b32_e64 $3, $5, $7, $1"
        )
        ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
        ret = llvm.inline_asm(
            ret_struct_ty,
            [
                as_mlir_value(x_bits),
                as_mlir_value(y_bits),
                as_mlir_value(rel_i32),
                as_mlir_value(neg_inf_i32),
            ],
            asm_str,
            "=s,=s,=v,=v,2,3,v,v,~{vcc}",
            has_side_effects=True,
        )
        new_x = llvm.extractvalue(T.i32, ret, [2])
        new_y = llvm.extractvalue(T.i32, ret, [3])
        s_values[idx_x] = _bitcast_f32(new_x)
        s_values[idx_y] = _bitcast_f32(new_y)


def _permlane16_swap_f32(x, y):
    """v_permlane16_swap_b32 on f32: lanes of even 16-lane rows return (x, x of row + 1), lanes of
    odd rows (y of row - 1, y)."""
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane16_swap(
        pair_i32_ty, _bitcast_i32(as_mlir_value(x)), _bitcast_i32(as_mlir_value(y)), False, False
    )
    return (
        _bitcast_f32(llvm.extractvalue(T.i32, swapped, [0])),
        _bitcast_f32(llvm.extractvalue(T.i32, swapped, [1])),
    )


def _swap_halves(dw):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False)
    lo_res = llvm.extractvalue(T.i32, swapped, [0])
    hi_res = llvm.extractvalue(T.i32, swapped, [1])
    return lo_res, hi_res


def _o_pack_2dw(traits, v_o, dc, store_group, elem_dtype):
    r_base = store_group * 4
    if const_expr(traits.DTYPE_STR == "bf16"):
        lo = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base],
            Vec(v_o[dc])[r_base + 1],
        )
        hi = rocdl.cvt_pk_bf16_f32(
            Vec(v_o[dc])[r_base + 2],
            Vec(v_o[dc])[r_base + 3],
        )
        return lo, hi

    o_f16 = []
    for i in range_constexpr(4):
        o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
    pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
    return as_mlir_value(pack[0]), as_mlir_value(pack[1])


def _packed_o_128_vec(traits, v_o, dc, g, lane_div_32, elem_dtype):
    d0_a, d1_a = _o_pack_2dw(traits, v_o, dc, 2 * g, elem_dtype)
    d0_b, d1_b = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, elem_dtype)
    is_hi_half = ArithValue(lane_div_32 != fx.Index(0))
    y0_a_lo, y0_a_hi = _swap_halves(d0_a)
    y1_a_lo, y1_a_hi = _swap_halves(d1_a)
    y0_b_lo, y0_b_hi = _swap_halves(d0_b)
    y1_b_lo, y1_b_hi = _swap_halves(d1_b)
    y0_a, y1_a = is_hi_half.select(y0_a_lo, y0_a_hi), is_hi_half.select(y1_a_lo, y1_a_hi)
    y0_b, y1_b = is_hi_half.select(y0_b_lo, y0_b_hi), is_hi_half.select(y1_b_lo, y1_b_hi)
    w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
    w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
    w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
    w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
    return Vec.from_elements([fx.Int32(w) for w in (w0, w1, w2, w3)], fx.Int32)


def _pin_regs(vals, reg):
    """Pass vals through an empty asm that defines them in register class reg ("=v" / "=a")."""
    irs = [as_mlir_value(v) for v in vals]
    constraints = ",".join([reg] * len(irs) + [str(i) for i in range(len(irs))])
    if len(irs) == 1:
        return [llvm.inline_asm(irs[0].type, irs, "", constraints, has_side_effects=True)]
    ret_ty = ir.Type.parse(f"!llvm.struct<({', '.join(str(v.type) for v in irs)})>")
    ret = llvm.inline_asm(ret_ty, irs, "", constraints, has_side_effects=True)
    return [llvm.extractvalue(irs[i].type, ret, [i]) for i in range(len(irs))]


def _anchor_v_o(traits, v_o):
    """Pin v_o accumulators at the current source position."""
    reg = "=a" if getattr(traits, "PV_AGPR", False) else "=v"
    return _pin_regs([v_o[dc] for dc in range_constexpr(traits.D_CHUNKS)], reg)


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([as_mlir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([as_mlir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _concat_tree(parts):
    """Concatenate a power-of-two list of vectors pairwise, ((a0 a1) (a2 a3)) ..."""
    while len(parts) > 1:
        parts = [_concat_vectors(parts[i], parts[i + 1]).ir_value() for i in range(0, len(parts), 2)]
    return parts[0]


def _numel(v):
    return Vec(as_mlir_value(v)).numel


def _reduce_score_pair(v_s, initial, reducer, fm_fast):
    s_lo, s_hi = v_s
    acc = initial
    for r in range_constexpr(16):
        acc = reducer(acc, s_lo[r], fm_fast)
    for r in range_constexpr(16):
        acc = reducer(acc, s_hi[r], fm_fast)
    return acc


def _reduce_vec_tree(vec, width, reducer, fm_fast):
    """Fold a power-of-two vector down to one lane-local scalar, halving the width each step so the reduction is log-depth and every step is one packed op."""
    cur = Vec(vec)
    while width > 1:
        half = width // 2
        lo = Vec(cur.shuffle(cur, [i for i in range(half)]).ir_value())
        hi = Vec(cur.shuffle(cur, [half + i for i in range(half)]).ir_value())
        cur = Vec(reducer(lo, hi, fm_fast))
        width = half
    return cur[0]


def _reduce_score_tree(v_s, reducer, fm_fast):
    n = len(v_s[0]) if isinstance(v_s[0], list) else _numel(v_s[0])
    return _reduce_list_tree([half[r] for half in v_s for r in range_constexpr(n)], reducer, fm_fast)


def _reduce_list_tree(cur, reducer, fm_fast):
    while len(cur) > 1:
        nxt = []
        for i in range_constexpr(0, len(cur), 3):
            grp = cur[i : i + 3]
            acc = grp[0]
            for x in grp[1:]:
                acc = reducer(acc, x, fm_fast)
            nxt.append(acc)
        cur = nxt
    return cur[0]


def _lane_pair_reduce(v, reducer, fm_fast):
    v_i32 = _bitcast_i32(v)
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return reducer(_bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32), fm_fast)


# Descriptor, LDS, and buffer helpers


def _lds_alias_scope_array(names):
    attrs = [f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>' for name in names]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def _dualwave_lds_scope(kind, buf_id):
    return f"lds_{kind}{buf_id}"


def _dualwave_lds_alias_scopes(name):
    return _lds_alias_scope_array([name])


def _dualwave_lds_noalias_scopes(name, scope_names):
    return _lds_alias_scope_array([scope_name for scope_name in scope_names if scope_name != name])


def _cu_load(div, idx, cu_atom, cu_v1i32):
    v = fly.copy_atom_call_ssa([cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx))))
    return fx.Index(Vec(v, (1,), fx.Int32)[0])


def _load_k_pack_aligned(traits, lds_kv_base_ptr, elem_idx, buf_id, kv_mfma_pack_type):
    scope_name = _dualwave_lds_scope("k", buf_id)
    byte_offset = elem_idx * traits.BF16_BYTES
    ptr = buffer_ops.get_element_ptr(lds_kv_base_ptr, byte_offset=byte_offset, elem_type=T.i8)
    return llvm.LoadOp(
        kv_mfma_pack_type,
        ptr,
        alignment=16,
        alias_scopes=_dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=_dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


def _lds_o_scopes(traits):
    # One-sided: scoped AA separates either way, and leaving "lds_o" out of the K / V noalias
    # lists keeps their metadata (and the loop schedule) as without the O stage.
    return dict(
        alias_scopes=_dualwave_lds_alias_scopes("lds_o"),
        noalias_scopes=_lds_alias_scope_array(traits.LDS_SCOPE_NAMES),
    )


def _lds_store_o_128(traits, pack_i32_vec, lane_ptr, imm_bytes):
    ptr = buffer_ops.get_element_ptr(lane_ptr, byte_offset=int(imm_bytes), elem_type=T.i8)
    llvm.StoreOp(pack_i32_vec.ir_value(), ptr, alignment=16, **_lds_o_scopes(traits))


def _lds_load_o_128(traits, lane_ptr, imm_bytes):
    ptr = buffer_ops.get_element_ptr(lane_ptr, byte_offset=int(imm_bytes), elem_type=T.i8)
    return Vec(llvm.LoadOp(Vec.make_type(4, fx.Int32), ptr, alignment=16, **_lds_o_scopes(traits)).result)


def _buffer_store_128(pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div):
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(_store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index))))


# Mapping and address helpers


def _qk16_k_steps(traits):
    """16x16x32 QK k-steps (32 wide); each spans two of the 16-wide K_STEPS_QK."""
    return traits.K_STEPS_QK // 2


# 16x16x32 QK k-steps whose d chunks are paired across the first and third 64-wide d blocks.
# k-chunk groups c = lane // 16 of 0 / 1 (and 2 / 3) share a ds_read_b128 phase on gfx950, which
# is conflict-free only when both read the same d % 64 of blocks 0 and 2 (8 lines apart, which
# lands on the same banks): at paired step s, group c reads d = 16 s + 128 (c % 2) + 8 (c // 2).
# No chunk of the middle block can pair that way, so its DMA stores 16-byte chunk m of the rows
# in lane row groups 0 and 3 at chunk m ^ 4 (`_k_mid_swizzled`), which moves those rows' banks
# by 64 bytes; the remaining steps then read d = 64 + 16 (s - 4) + 32 (c % 2) + 8 (c // 2).
QK16_PAIRED_STEPS = 4
# Q_LDS_STAGE: a stage row is the paired steps' part of a Q row, d blocks 0 and 2 (256 bytes).
Q_STAGE_ROW_BYTES = 256
Q_STAGE_ROW_CHUNKS = Q_STAGE_ROW_BYTES // 16
# Waves that split the last wave's stage DMAs between them.
Q_STAGE_LAST_SPLIT = 2


def _qk16_step_d(step):
    """Static part of the d a 16x16x32 QK k-step reads (see QK16_PAIRED_STEPS)."""
    if step < QK16_PAIRED_STEPS:
        return 16 * step
    return 64 + 16 * (step - QK16_PAIRED_STEPS)


def _k_mid_swizzled(line, row):
    """1 where LDS line `line`, row `row` holds its middle d block chunk-swizzled (QK16_PAIRED_STEPS)."""
    return (row % 2 + line // 4 + 1) % 2


def _v_pv16_chunk(chunk, row):
    """Global 16-byte V chunk the DMA stores at LDS chunk `chunk` of row `row`: chunk ^ 2 on odd rows."""
    odd = row % 2
    return chunk + 2 * odd - 4 * odd * ((chunk // 2) % 2)


# Raw 16x16 score half (32 kv) of a wave with nqt q tiles: element 4 (nqt kt + qt) + e is kv
# 16 kt + [0, 8, 4, 12][c] + e of the half (c = lane // 16), q = lane % 16 + 16 qt. The P*V B
# operand of q tile qt is its 8 elements.
def _pv16_qt_elems(qt, nqt=2):
    return [4 * qt + e for e in range(4)] + [4 * nqt + 4 * qt + e for e in range(4)]


def _permlane16_swap_i32(x, y):
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane16_swap(pair_i32_ty, as_mlir_value(x), as_mlir_value(y), False, False)
    return llvm.extractvalue(T.i32, swapped, [0]), llvm.extractvalue(T.i32, swapped, [1])


def _permlane32_swap_i32(x, y):
    """Lanes 0-31 return (x, x of lane + 32), lanes 32-63 (y of lane - 32, y)."""
    pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_i32_ty, as_mlir_value(x), as_mlir_value(y), False, False)
    return llvm.extractvalue(T.i32, swapped, [0]), llvm.extractvalue(T.i32, swapped, [1])


def _permlane32_swap_f32(x, y):
    a, b = _permlane32_swap_i32(_bitcast_i32(as_mlir_value(x)), _bitcast_i32(as_mlir_value(y)))
    return _bitcast_f32(a), _bitcast_f32(b)


def _k_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
    return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


def _v_buf_base(traits, buf_id):
    if const_expr(isinstance(buf_id, int)):
        return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
    return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER


class DualwaveSwpTraits:
    """Pure compile-time tile/layout constants for gfx950 DUALWAVE_SWP."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def cache_tag(self):
        return (
            CODEGEN_REV,
            self.NUM_HEADS_Q,
            self.NUM_HEADS_KV,
            self.HEAD_DIM,
            self.HEAD_DIM_V,
            self.BLOCK_M,
            self.Q_HEADS_PER_WG,
            self.CAUSAL,
            self.DTYPE_STR,
            self.WAVES_PER_EU,
            self.DAZ,
            self.DUALWAVE_SWP_MFMA_ROWSUM,
            self.DUALWAVE_SWP_SETPRIO,
            self.DUALWAVE_SWP_ENABLE_STAGGER,
            self.EMIT_LSE,
            self.WINDOW_LEFT,
            self.VARLEN,
            self.CROSS_SEQLEN,
            self.SBHD,
            self.HAS_SINK,
            getattr(self, "K_HEAD", None),
            self.QK_MFMA16,
            self.PV_MFMA16,
            self.ROWS_PER_WAVE,
            getattr(self, "FOLD_BARRIERS", None),
            getattr(self, "PERSISTENT", 0),
        )


def _make_dualwave_swp_traits(
    num_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    varlen=False,
    cross_seqlen=False,
    emit_lse=False,
    window_left=-1,
    block_m=None,
    gqa_merge=None,
    sbhd=False,
    has_sink=False,
    head_dim_v=None,
    qk_mfma16=False,
    pv_mfma16=False,
    rows_per_wave=32,
):
    """Build gfx950 DUALWAVE_SWP compile-time layout traits.

    ``head_dim`` is the Q/K head dim (QK reduction); ``head_dim_v`` the V/O head dim
    (defaults to ``head_dim``). ``rows_per_wave`` 64 gives each wave four 16-row q tiles, so a
    K/V pack read from LDS feeds twice the MFMAs (16x16x32 QK and P*V only)."""
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    assert rows_per_wave in (32, 64), f"rows_per_wave must be 32 or 64, got {rows_per_wave}"
    if rows_per_wave == 64:
        assert not (varlen or cross_seqlen or int(window_left) >= 0 or has_sink), (
            "64 rows per wave covers the dense causal/non-causal paths only"
        )
        gqa_merge = False
    warp_size = 64
    # Windowed default BLOCK_M=64 (fewer masked KV tiles; also avoids waves_per_eu=4).
    # Dense default stays 256. Callers can still pass block_m explicitly.
    if block_m is None:
        block_m = 64 if int(window_left) >= 0 else 256
    wave_row_groups = block_m // rows_per_wave
    block_n = 64
    k_sub_n = 32
    k_step_qk = 16
    d_chunk = 32
    gqa_group_size = num_heads // num_kv_heads

    # Global K/V DMA is 16B per lane; D_128B_SIZE is one 128B row in bf16 elements.
    bf16_bytes = 2
    d_128b_size = 64
    smem_linear_wave = warp_size * 16 // bf16_bytes
    smem_n_per_wave = smem_linear_wave // d_128b_size
    smem_n_rpt = block_n // smem_n_per_wave
    # Two q-heads of one kv-head share a CTA and thus one K/V LDS tile. Capped at SMEM_N_RPT
    # waves, one LDS line each, so the DMA still splits evenly.
    if gqa_merge is None:
        gqa_merge = (
            not dualwave_swp_enable_stagger and gqa_group_size % 2 == 0 and 2 * wave_row_groups <= smem_n_rpt
        )
    q_heads_per_wg = 2 if gqa_merge else 1
    num_waves = wave_row_groups * q_heads_per_wg
    smem_d_rpt_k = head_dim // d_128b_size
    smem_d_rpt_v = head_dim_v // d_128b_size
    # K/V LDS tiles are wave-linear rows with padding to avoid bank-aligned repeats.
    smem_k_line_stride = smem_linear_wave + 16 // bf16_bytes
    smem_v_line_stride = smem_linear_wave + 64 // bf16_bytes
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt_k * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt_v * smem_v_line_stride
    dualwave_swp_kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    varlen = bool(varlen)
    cross_seqlen = bool(cross_seqlen)
    # The running row sum lives in an MFMA accumulator fed by a ones A operand instead of a
    # VALU fold.
    # The ones-operand MFMA row sum folds the half-wave partner too, so it replaces a
    # permlane pair reduce per tile. Online needs its running sum corrected when the row
    # maximum moves, which is a multiply on the accumulator -- no reason to give it up.
    dualwave_swp_mfma_rowsum = True
    # Splitting the K/V LDS reads across the memory/compute cluster boundary keeps the main
    # loop inside the 4-waves-per-SIMD register budget, so ask for that budget instead of the
    # caller's floor. Stagger spends the same scheduling slack, so it keeps it.
    if (
        not dualwave_swp_enable_stagger
        and dualwave_swp_mfma_rowsum
        and head_dim == 64
        and block_m in (128, 256)
    ):
        waves_per_eu = max(waves_per_eu, 4)
    # QK on 16x16x32 MFMAs (None: auto), permlane16-swapped back into the 32x32x16 score layout
    # so softmax, P and P*V are unchanged. A floored seed (causal window) reads its own reference
    # out of the seed vector, which the swapped seed no longer holds per lane.
    if qk_mfma16 is None:
        qk_mfma16 = head_dim == 192 and not (causal and int(window_left) >= 0)
    assert not (qk_mfma16 and causal and int(window_left) >= 0), "QK_MFMA16 needs an unfloored seed"
    # P*V on 16x16x32 MFMAs as well (None: with QK_MFMA16). The raw QK accumulators are then
    # already the P*V B operand, so softmax, P and O stay in the 16x16 layout and nothing is
    # swapped per tile; per-row scalars (reference, final sum, LSE) keep one q per lane.
    if pv_mfma16 is None:
        pv_mfma16 = qk_mfma16
    assert not pv_mfma16 or (qk_mfma16 and dualwave_swp_mfma_rowsum), (
        "PV_MFMA16 needs QK_MFMA16 and the MFMA row sum"
    )
    assert not pv_mfma16 or head_dim_v == 128, "PV_MFMA16 maps P*V steps onto two 64-wide V d blocks"
    assert rows_per_wave == 32 or pv_mfma16, "64 rows per wave needs the 16x16x32 QK and P*V"

    return DualwaveSwpTraits(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        WARP_SIZE=warp_size,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=num_waves * warp_size,
        WAVE_ROW_GROUPS=wave_row_groups,
        Q_HEADS_PER_WG=q_heads_per_wg,
        ROWS_PER_WAVE=rows_per_wave,
        Q_TILES=rows_per_wave // 16,
        # Block rows of q tile qt of wave w: WAVE_ROW_STRIDE w + QT_ROW_STRIDE qt + [0, 16). The
        # kernel may interleave them (ROW_INTERLEAVE: 16 w + 64 qt) to share the causal diagonal.
        ROW_INTERLEAVE=False,
        QT_ROW_STRIDE=16,
        WAVE_ROW_STRIDE=rows_per_wave,
        PV_AGPR=rows_per_wave == 64,
        # One wave per SIMD: no partner wave hides the softmax VALU, so both compute clusters
        # interleave it with their own MFMAs (the P*V site takes its exp2 half speculatively).
        SOLO_WAVE=rows_per_wave == 64,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        K_STEP_QK=k_step_qk,
        K_STEPS_QK=head_dim // k_step_qk,
        D_CHUNK=d_chunk,
        D_CHUNKS=head_dim_v // d_chunk,
        PV_K_STEPS=k_sub_n // 16,
        MFMA_LANE_K=8,
        QK_MFMA16=bool(qk_mfma16),
        PV_MFMA16=bool(pv_mfma16),
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_MFMA_ROWSUM=dualwave_swp_mfma_rowsum,
        # With one wave per SIMD there is no partner to arbitrate against.
        DUALWAVE_SWP_SETPRIO=bool(dualwave_swp_setprio) and rows_per_wave != 64,
        DUALWAVE_SWP_ENABLE_STAGGER=bool(dualwave_swp_enable_stagger),
        EMIT_LSE=bool(emit_lse),
        WINDOW_LEFT=int(window_left),
        VARLEN=varlen,
        CROSS_SEQLEN=cross_seqlen,
        SBHD=bool(sbhd),
        HAS_SINK=bool(has_sink),
        DEFAULT_STRIDE_Q_N=num_heads * head_dim,
        DEFAULT_STRIDE_KV_N=num_kv_heads * head_dim,
        DEFAULT_STRIDE_V_N=num_kv_heads * head_dim_v,
        DEFAULT_STRIDE_O_N=num_heads * head_dim_v,
        DMA_BYTES=16,
        BF16_BYTES=bf16_bytes,
        D_128B_SIZE=d_128b_size,
        VEC_KV=8,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT_K=smem_d_rpt_k,
        SMEM_D_RPT_V=smem_d_rpt_v,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        DUALWAVE_SWP_KV_PER_BUFFER=dualwave_swp_kv_per_buffer,
        LDS_KV_TOTAL_SIZE=2 * dualwave_swp_kv_per_buffer,
        # The 64-row O store (`_store_final_o16x4`) goes through a per-wave LDS tile so each
        # buffer_store covers four whole q rows; rows are padded 16 B off the 64-bank period.
        O_LDS_STAGE=bool(pv_mfma16) and rows_per_wave == 64,
        O_STAGE_ROW_ELEMS=head_dim_v + 16 // bf16_bytes,
        LDS_O_STAGE_SIZE=num_waves * rows_per_wave * (head_dim_v + 16 // bf16_bytes),
        DUALWAVE_SWP_K_BUF_BASE=(0, dualwave_swp_kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + dualwave_swp_kv_per_buffer),
        # K LDS->VGPR reads: hi half jumps one N strip; ks uses inner then d_rpt stride.
        K_LDS_TO_REG_N_STRIP_STRIDE=256,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=16,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=smem_n_rpt * smem_k_line_stride,
        # V LDS->VGPR base: half-wave, lane quad, N group, lane-in-quad; immediates step across
        # K substeps, D-chunk pairs, and transpose-load pairs.
        V_LDS_TO_REG_HALF_WAVE_STRIDE=2176,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=smem_v_line_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=16,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=4,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=128,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=smem_n_rpt * smem_v_line_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=32,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=d_128b_size,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
    )


class DualwaveKernelContext:
    """Shared per-kernel state for the gfx950 dualwave attention helpers."""

    def __init__(
        self,
        traits,
        Q=None,
        K=None,
        V=None,
        O=None,
        DebugCounts=None,  # noqa: E741
        CuSeqQ=None,
        CuSeqKv=None,
        BlockTable=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
        block_table_stride=None,
        SINK=None,
        ScoreBound=None,
    ):
        self.traits = traits
        self.SINK = SINK
        self.ScoreBound = ScoreBound
        self.use_score_bound = ScoreBound is not None and traits.HEAD_DIM <= SCORE_BOUND_HEAD_DIM_MAX
        # Without the bound floor, a row whose prologue tile is wholly masked keeps the floor as its
        # reference; only a left window can still hand such a row live keys in a later tile.
        self.seed_floored = not self.use_score_bound and traits.CAUSAL and traits.WINDOW_LEFT >= 0
        self.bound_over_ref = None
        self.q_norm_sq = None
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.BlockTable = BlockTable
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime
        self.block_table_stride = block_table_stride

    def _setup(self, shared_storage, bind_block=True):
        """Per-CTA state; ``bind_block`` also binds the CTA's own job (see `bind_item`)."""
        head_dim_runtime = self.head_dim_runtime
        traits = self.traits
        self.NUM_DMA_K = traits.SMEM_D_RPT_K
        self.NUM_DMA_V = traits.SMEM_D_RPT_V
        # vmcnt drains are counted in DMA instructions, but one K/V tile costs
        # DMA_WAVE_REPS of them per wave, so keep the drains at a fixed number of tiles
        # in flight however many waves share the tile.
        _dma_reps = traits.SMEM_N_RPT // traits.NUM_WAVES
        self.VM_DRAIN_KV = (self.NUM_DMA_K + self.NUM_DMA_V) * _dma_reps // 2
        self.VM_DRAIN_V = self.NUM_DMA_V * _dma_reps // 2

        self.fm_fast = fx.arith.FastMathFlags.fast
        self.elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)
        self.q_load_i32x4_type = Vec.make_type(4, fx.Int32)
        self.v_lds_read_vec4_type = Vec.make_type(4, self.elem_dtype)
        self.kv_mfma_pack_type = Vec.make_type(8, self.elem_dtype)
        self.mfma_acc_vec_type = Vec.make_type(16, fx.Float32)
        self.rowsum_acc_vec_type = Vec.make_type(4, fx.Float32)
        self.c_zero_v4f32 = Vec.filled(4, 0.0, fx.Float32)
        self.c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)

        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        self.QT = traits.Q_TILES
        # P*V accumulators per D chunk (2 dv tiles x QT q tiles) and the MFMA row sum (QT tiles).
        self.c_zero_o = Vec.filled(8 * self.QT, 0.0, fx.Float32)
        self.c_zero_l16 = Vec.filled(4 * self.QT, 0.0, fx.Float32)
        self.ref_neg = None
        self.ref_seed_vec = None
        head_dim_f32 = fx.Float32(fx.Int32(head_dim_runtime))
        c_log2e_f = fx.Float32(LOG2E)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                as_mlir_value(fmath.rsqrt(head_dim_f32, fastmath=self.fm_fast)),
                as_mlir_value(c_log2e_f),
                fastmath=self.fm_fast,
            )
        )

        seq_len = self.seq_len
        seq_len_kv = self.seq_len_kv
        stride_q_n = self.stride_q_n
        stride_kv_n = self.stride_kv_n
        self.seq_len_v = fx.Index(seq_len)
        self.seq_len_kv_v = fx.Index(seq_len_kv)
        self.stride_q_n_v = fx.Index(stride_q_n)
        self.stride_kv_n_v = fx.Index(stride_kv_n)
        # V/O rows are the Q/K rows narrowed to HEAD_DIM_V, so only contiguous layouts are valid.
        if const_expr(traits.HEAD_DIM_V == traits.HEAD_DIM):
            self.stride_o_n_v = self.stride_q_n_v
            self.stride_v_n_v = self.stride_kv_n_v
        else:
            self.stride_o_n_v = self.stride_q_n_v // fx.Index(traits.HEAD_DIM) * fx.Index(traits.HEAD_DIM_V)
            self.stride_v_n_v = self.stride_kv_n_v // fx.Index(traits.HEAD_DIM) * fx.Index(traits.HEAD_DIM_V)

        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = buffer_ops.create_llvm_ptr(self.lds_kv_base_idx, address_space=3)
        if const_expr(traits.O_LDS_STAGE):
            self.lds_o_base_idx = fx.Index(fx.ptrtoint(lds.o.ptr))
            self.lds_o_base_ptr = buffer_ops.create_llvm_ptr(self.lds_o_base_idx, address_space=3)

        self.tid = fx.Index(gpu.thread_idx.x)
        self._bind_lane_layout(self.tid)

        _tid_i32 = as_mlir_value(fx.Int32(self.tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, as_mlir_value(fx.Int32(traits.WARP_SIZE))),
        )
        self.wave_id_uni = fx.Index(_wave_id_uni_i32)
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            self.wave_row_group_uni = self.wave_id_uni % traits.WAVE_ROW_GROUPS
        else:
            self.wave_row_group_uni = self.wave_id_uni

        self.buf_flags_i32 = fx.Int32(buffer_ops._get_buffer_flags())
        self.elem_ir = self.elem_dtype.ir_type
        # EMIT_LSE aliases the DebugCounts tensor slot as the fp32 LSE output
        # ([total_q, NUM_HEADS_Q] layout).
        if const_expr(traits.EMIT_LSE):
            _lse_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(self.DebugCounts)))
            _lse_addr_i64 = as_mlir_value(_lse_base_i64 + fx.Int64(fx.Int64(0)))
            self.lse_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _lse_addr_i64, num_records_bytes=as_mlir_value(fx.Int64(fx.Int64(0xFFFFFFFF)))
            )
        else:
            self.lse_rsrc = None

        # Sink is loaded in the epilogue (`load_sink_h`), not here: a prologue load stays
        # live across the KV loop and spills ~112 B of scratch on the sink+LSE build.
        self.sink_h = None

        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, self.elem_dtype))
        if const_expr(self.traits.QK_MFMA16):
            self.qk16_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, self.elem_dtype))
        if const_expr(self.traits.DUALWAVE_SWP_MFMA_ROWSUM):
            self.rowsum_mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, self.elem_dtype))
            # A operand of the 16x16x32 row-sum MFMA. Re-reading a packed P slice as 16x16x32
            # gives col = lane%16, k = 8*(lane//16)+i, so rows 0/8 tap the low q half and 4/12
            # the high one, leaving each lane's own q sum in D element 0 with no cross-lane move.
            _one_bits = 0x3F803F80 if const_expr(self.traits.DTYPE_STR == "bf16") else 0x3C003C00
            if const_expr(self.traits.PV_MFMA16):
                # A 16x16 P pack is already a [32 kv x 16 q] B operand: ones everywhere put
                # the q sum in every row of D, so each lane reads its own q with no tap rows.
                _dword = fx.Int32(_one_bits)
            else:
                _is_tap_row = ArithValue(fx.Index(self.lane % 4) == fx.Index(0))
                _half_match = ArithValue(fx.Index((self.lane // 4) % 2) == fx.Index((self.lane // 16) % 2))
                _dword = _is_tap_row.select(_half_match.select(fx.Int32(_one_bits), fx.Int32(0)), fx.Int32(0))
            self.rowsum_ones_a = Vec.from_elements([_dword] * 4, fx.Int32).bitcast(self.elem_dtype).ir_value()
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        self.lds_ptr_ty = fx.PointerType.get(self.elem_dtype.ir_type, 2, self.traits.DMA_BYTES)
        self.dma_wave_reps = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES

        self.kv_tile_size = traits.BLOCK_N

        self.k_dma_m0 = self.dma_m0_table(
            lambda b, w, d: self._dma_base(b, w, d, _k_buf_base, traits.SMEM_K_LINE_STRIDE), self.NUM_DMA_K
        )
        self.v_dma_m0 = self.dma_m0_table(
            lambda b, w, d: self._dma_base(b, w, d, _v_buf_base, traits.SMEM_V_LINE_STRIDE), self.NUM_DMA_V
        )
        if bind_block:
            self.bind_item(*self.block_item())

    def _bind_lane_layout(self, tid):
        """Lane / wave terms of thread `tid`: q rows, LDS read bases, mask and store lanes."""
        traits = self.traits
        self.wave_id = tid // traits.WARP_SIZE
        self.lane = tid % traits.WARP_SIZE
        self.lane_mod_32 = self.lane % 32
        self.lane_div_32 = self.lane // 32
        # Merged CTAs stack Q_HEADS_PER_WG sharers of one kv-head on top of the row groups:
        # the low wave bits pick the row group, the high bits pick the sharer.
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            self.wave_row_group = self.wave_id % traits.WAVE_ROW_GROUPS
        else:
            self.wave_row_group = self.wave_id
        self.wave_q_offset = self.wave_row_group * traits.WAVE_ROW_STRIDE

        if const_expr(traits.QK_MFMA16):
            # 16x16x32 A operand: row r = lane % 16 of kv tile kt holds kv 16 kt + [0, 8, 4, 12][r // 4]
            # + r % 4 (bits 2 and 3 of r swapped), which is the row order the permlane16 swap in
            # `_qk16_to_score_layout` needs to land on the 32x32x16 score layout. LDS element
            # (kv, d) sits at line kv % 8, row kv // 8 of that line, and 64-wide d block d // 64
            # K_LDS_TO_REG_KSTEP_OUTER_STRIDE further on. Per-lane d terms: QK16_PAIRED_STEPS.
            assert traits.SMEM_N_RPT == 8, "the middle-block K swizzle assumes 8 LDS lines"
            _r_grp = (self.lane % 16) // 4
            _c = self.lane // 16
            _line = (_r_grp // 2) * 4 + self.lane % 4
            _rows = _line * traits.SMEM_K_LINE_STRIDE + (_r_grp % 2) * traits.D_128B_SIZE
            self.k_lds_read_base_per_lane = (
                _rows + (_c % 2) * 2 * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (_c // 2) * traits.VEC_KV
            )
            _swz = _k_mid_swizzled(_line, _r_grp % 2)
            self.k_lds_read_base_middle = (
                _rows + ((_c % 2 + _swz) % 2) * 4 * traits.VEC_KV + (_c // 2) * traits.VEC_KV
            )
        else:
            self.k_lds_read_base_per_lane = (
                (self.lane_mod_32 % 8) * traits.SMEM_K_LINE_STRIDE
                + (self.lane_mod_32 // 8) * traits.D_128B_SIZE
                + self.lane_div_32 * traits.VEC_KV
            )
        self.v_lds_read_base_per_lane = (
            self.lane_div_32 * traits.V_LDS_TO_REG_HALF_WAVE_STRIDE
            + ((self.lane % 16) // 4) * traits.V_LDS_TO_REG_LANE_QUAD_STRIDE
            + ((self.lane // 16) % 2) * traits.V_LDS_TO_REG_N_GROUP_STRIDE
            + (self.lane % 4) * traits.V_LDS_TO_REG_LANE_IN_QUAD_STRIDE
        )
        if const_expr(traits.PV_MFMA16):
            # Raw 16x16 layout (`_pv16_qt_elems`): lane group c = lane // 16 holds kv 16 kt +
            # pv16_kv_offset + e, q = lane % 16 + 16 qt. The P*V A operand (V^T, one 16-dv tile) needs the
            # same kv: each ds_read_b64_tr_b16 gives a lane 4 kv rows (line kv % 8 of the 16-lane
            # group's quad, row kv // 8) at dv = lane % 16. Groups 0 / 1 (and 2 / 3) share a
            # read phase and sit on rows of opposite parity, so the V DMA stores odd rows with
            # 16-byte chunk m at m ^ 2 (`_v_pv16_chunk`): dv tile pairs swap places there, which
            # needs one base for even dv tiles and one for odd ones.
            _c = self.lane // 16
            self.pv16_c_odd = _c % 2
            # Per-row scalars sit one q per lane: q = lane % 32 (2 q tiles, each q on two lanes)
            # or q = lane (4 q tiles), so the lane's own q is in q tile c % QT of the raw layout.
            self.pv16_owned_qt = _c % traits.Q_TILES
            self.pv16_kv_offset = 8 * (_c % 2) + 4 * (_c // 2)
            _v_lane = (
                (4 * (_c // 2) + (self.lane % 16) // 4) * traits.SMEM_V_LINE_STRIDE
                + (_c % 2) * traits.D_128B_SIZE
                + (self.lane % 4) * 4
            )
            self.v16_read_base = (_v_lane + 16 * (_c % 2), _v_lane + 16 - 16 * (_c % 2))
            # Byte offsets of those bases per V buffer. Buffer 1 starts at the 64 KiB limit of the
            # DS offset field, so a constant base cannot fold there and every read would get its own
            # loop-invariant address register; an opaque base leaves only the read's offset.
            v16_off = [
                [fx.Int32((_v_buf_base(traits, b) + base) * traits.BF16_BYTES) for base in self.v16_read_base]
                for b in range(2)
            ]
            v16_off[1] = [fx.Int32(v) for v in _pin_regs(v16_off[1], "=v")]
            self.v16_read_off = v16_off

    def block_item(self):
        """(head, q block, batch) this CTA owns in the one-job-per-CTA grid."""
        traits = self.traits
        _nq = (self.seq_len_v + fx.Index(traits.BLOCK_M - 1)) // fx.Index(traits.BLOCK_M)
        _nh = traits.NUM_HEADS_Q // traits.Q_HEADS_PER_WG
        if const_expr(_nh % NUM_XCD == 0):
            # Each XCD owns every NUM_XCD-th head and walks them a group of q-blocks at a time, so
            # its resident CTAs read a few heads' K/V through its L2 rather than one CTA each.
            # Groups advance in q rank, which keeps the longest-first order below.
            _heads_per_xcd = _nh // NUM_XCD
            _lin = fx.Index(gpu.block_idx.y) * fx.Index(_nh) + fx.Index(gpu.block_idx.x)
            _slot = _lin // NUM_XCD
            _nq_head = ArithValue(_nq > fx.Index(XCD_TAIL_Q)).select(_nq - fx.Index(XCD_TAIL_Q), fx.Index(0))

            # Rank sections [lo, hi) per group size, then the rank-major tail; a slot takes the
            # last section starting at or below it (an empty section yields to the next one).
            _lo = fx.Index(0)
            _h_local = _q_rank = None
            for _i in range_constexpr(len(XCD_Q_GROUPS)):
                _g = XCD_Q_GROUPS[_i]
                _u = _slot - _lo * _heads_per_xcd
                _grp = _u // (_g * _heads_per_xcd)
                _r = _u - _grp * (_g * _heads_per_xcd)
                _h_sec = _r // _g
                _q_sec = _lo + _grp * _g + _r % _g
                if const_expr(_i == 0):
                    _h_local, _q_rank = _h_sec, _q_sec
                else:
                    _in = ArithValue(_slot >= _lo * _heads_per_xcd)
                    _h_local = _in.select(_h_sec, _h_local)
                    _q_rank = _in.select(_q_sec, _q_rank)
                _lo = _lo + (_nq_head - _lo) // _g * _g
            _t = _slot - _lo * _heads_per_xcd
            _in_tail = ArithValue(_slot >= _lo * _heads_per_xcd)
            _h_local = _in_tail.select(_t % _heads_per_xcd, _h_local)
            _q_rank = _in_tail.select(_lo + _t // _heads_per_xcd, _q_rank)
            h_idx = _h_local * NUM_XCD + _lin % NUM_XCD
        else:
            h_idx = fx.Index(gpu.block_idx.x)
            _q_rank = fx.Index(gpu.block_idx.y)
        if const_expr(traits.CAUSAL):
            # Dispatch order is the list-schedule order and causal work grows with q_block
            # (a window flattens but never inverts that), so walk q_block descending
            # (longest-processing-time-first). From seq_len, not grid_dim.y, to stay uniform.
            q_block_idx = _nq - fx.Index(1) - _q_rank
        else:
            q_block_idx = _q_rank
        return h_idx, q_block_idx, fx.Index(gpu.block_idx.z)

    def block_row0(self, q_block_idx):
        """First q row of a q block. ROW_INTERLEAVE pulls a block that runs past the sequence end
        (rounded up to the drain's two KV tiles) back to end there, so every causal band is the
        block's full diagonal; the rows it shares with the block before get the same values."""
        traits = self.traits
        row0 = q_block_idx * traits.BLOCK_M
        if const_expr(traits.ROW_INTERLEAVE):
            span = 2 * traits.BLOCK_N
            end = (self.seq_len_v + fx.Index(span - 1)) // fx.Index(span) * fx.Index(span)
            end = fx.Index(ArithValue(end < fx.Index(traits.BLOCK_M)).select(fx.Index(traits.BLOCK_M), end))
            last = end - fx.Index(traits.BLOCK_M)
            row0 = fx.Index(ArithValue(row0 < last).select(row0, last))
        return row0

    def bind_item(self, h_idx, q_block_idx, batch_idx):
        """Point every per-job value (heads, rows, tensor views, KV tile span) at one job."""
        traits = self.traits
        self.h_idx = h_idx
        self.q_block_idx = q_block_idx
        self.batch_idx = batch_idx
        self.q_start = self.block_row0(q_block_idx)
        tid = self._job_tid()
        self._bind_lanes(tid)
        if const_expr(getattr(traits, "PERSISTENT", 0)):
            self._bind_lane_layout(tid)

        self.h_kv_idx = self.h_idx % traits.NUM_HEADS_KV
        self.group_id = self.h_idx // traits.NUM_HEADS_KV
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            self.q_head_idx = (
                self.h_kv_idx * traits.GQA_GROUP_SIZE
                + self.group_id * traits.Q_HEADS_PER_WG
                + self.wave_id_uni // traits.WAVE_ROW_GROUPS
            )
        else:
            self.q_head_idx = self.h_kv_idx * traits.GQA_GROUP_SIZE + self.group_id
        self.kv_head_idx = self.h_kv_idx

        CuSeqQ = self.CuSeqQ
        CuSeqKv = self.CuSeqKv
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqQ), fx.make_layout(1, 1))
            _cuk_div = fx.logical_divide(fx.rocdl.make_buffer_tensor(CuSeqKv), fx.make_layout(1, 1))
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(_cuk_div, self.batch_idx + fx.Index(1), _cu_atom, _cu_v1i32)
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + fx.Index(1)) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv

        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = self.q_start * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM
        self.v_gmem_elem_offset = self.kv_head_idx * traits.HEAD_DIM_V

        hq, hkv, d_qk, d_v = traits.NUM_HEADS_Q, traits.NUM_HEADS_KV, traits.HEAD_DIM, traits.HEAD_DIM_V
        b, q_tok, kv_tok = self.batch_idx, self.q_tok_base, self.kv_tok_base
        self.q_div = self._batch_view(self.Q, self.seqlen_q_v, self.stride_q_n_v, hq, d_qk, b, q_tok)
        self.o_div = self._batch_view(self.O, self.seqlen_q_v, self.stride_o_n_v, hq, d_v, b, q_tok)
        self.k_div = self._batch_view(self.K, self.seqlen_kv_v, self.stride_kv_n_v, hkv, d_qk, b, kv_tok)
        self.v_div = self._batch_view(self.V, self.seqlen_kv_v, self.stride_v_n_v, hkv, d_v, b, kv_tok)

        self.num_kv_tiles = (self.seqlen_kv_v + self.kv_tile_size - 1) // self.kv_tile_size
        if const_expr(traits.CAUSAL):
            self.causal_end_raw_i32 = fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            causal_end_i32 = fx.Int32(
                ArithValue(self.causal_end_raw_i32 > fx.Int32(0)).select(self.causal_end_raw_i32, fx.Int32(0))
            )
            causal_num_tiles = (fx.Index(causal_end_i32) + self.kv_tile_size - 1) // self.kv_tile_size
            self.max_num_tiles = fx.Index(
                ArithValue(causal_num_tiles < self.num_kv_tiles).select(causal_num_tiles, self.num_kv_tiles)
            )
        else:
            self.causal_end_raw_i32 = None
            self.max_num_tiles = self.num_kv_tiles

        if const_expr(traits.CAUSAL and traits.WINDOW_LEFT >= 0):
            # Keep an odd KV span: two-tile drain admits N in {3,5,7,...}; even-rounding
            # added a fourth tile that the window mask then threw away.
            self.max_num_tiles = fx.Index(
                ArithValue(self.max_num_tiles < fx.Index(3)).select(fx.Index(3), self.max_num_tiles)
            )
            # SWA: start at the first KV tile that can intersect the left window, leaving
            # >= 3 tiles for the pipeline.
            swa_lo_raw = fx.Int32(self.q_start) + self.delta_i32 - fx.Int32(traits.WINDOW_LEFT)
            swa_lo_raw = fx.Int32(ArithValue(swa_lo_raw > fx.Int32(0)).select(swa_lo_raw, fx.Int32(0)))
            swa_lo_tile = fx.Index(swa_lo_raw) // fx.Index(traits.BLOCK_N)
            max_lo = fx.Index(
                ArithValue(self.max_num_tiles > fx.Index(3)).select(
                    self.max_num_tiles - fx.Index(3), fx.Index(0)
                )
            )
            swa_lo_tile = fx.Index(ArithValue(swa_lo_tile < max_lo).select(swa_lo_tile, max_lo))
            # If the span is even, grow by one tile. Prefer downward (window-masked);
            # at tile 0 grow upward (causal-masked).
            span = self.max_num_tiles - swa_lo_tile
            span_odd = span - (span // fx.Index(2)) * fx.Index(2)
            grow = fx.Index(1) - span_odd
            down = fx.Index(ArithValue(swa_lo_tile > fx.Index(0)).select(grow, fx.Index(0)))
            self.split_t0 = swa_lo_tile - down
            self.split_t_end = self.max_num_tiles + (grow - down)
        else:
            self.max_num_tiles = ((self.max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            self.max_num_tiles = fx.Index(
                ArithValue(self.max_num_tiles < fx.Index(4)).select(fx.Index(4), self.max_num_tiles)
            )
            self.split_t0 = 0
            self.split_t_end = self.max_num_tiles

        if const_expr(traits.VARLEN):
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                self.active = ArithValue(self.q_start < self.seqlen_q_v) & (
                    self.causal_end_raw_i32 > fx.Int32(0)
                )
            else:
                self.active = ArithValue(self.q_start < self.seqlen_q_v)
        elif const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
            self.active = ArithValue(self.causal_end_raw_i32 > fx.Int32(0))
        else:
            self.active = None

    def _job_tid(self):
        """The thread id; opaque in persistent jobs. Lane terms are job-invariant, so in the job
        loop LICM hoists them out and they spill across it (~70 VGPRs at D_QK 192); deriving them
        from an opaque tid per job keeps them next to their uses."""
        if const_expr(getattr(self.traits, "PERSISTENT", 0)):
            return fx.Index(_pin_regs([fx.Int32(self.tid)], "=v")[0])
        return self.tid

    def _bind_lanes(self, tid):
        """Lane terms of the global Q / K / V offsets."""
        traits = self.traits
        self.job_lane = tid % traits.WARP_SIZE
        self.dma_wave = tid // traits.WARP_SIZE
        self.n_in_warp = self.job_lane // traits.VEC_KV
        self.d_bucket = self.job_lane % traits.VEC_KV
        if const_expr(traits.PV_MFMA16):
            self.v16_dma_d_bucket = _v_pv16_chunk(self.d_bucket, self.n_in_warp)

    def _view(self, tensor, byte_off, nrec, layout):
        base_iter = fx.get_iter(tensor)
        base_i64 = fx.Int64(fx.ptrtoint(base_iter))
        shifted = fx.inttoptr(base_iter.type, base_i64 + fx.Int64(byte_off))
        buf_ptr_ty = fx.PointerType.get(
            elem_ty=self.elem_ir,
            address_space=_TargetAddressSpace.BufferDesc,
            alignment=base_iter.alignment,
        )
        buf_ptr = fx.make_ptr(
            buf_ptr_ty,
            [shifted, fx.Int16(0).ir_value(), fx.Int64(nrec).ir_value(), self.buf_flags_i32.ir_value()],
        )
        return fx.logical_divide(fx.make_view(buf_ptr, layout), fx.make_layout(1, 1))

    def _batch_view(self, tensor, seqlen, stride, sbhd_heads, head_dim, batch_idx, tok_base):
        traits = self.traits
        per_batch = seqlen * stride
        nrec_bytes = per_batch * fx.Index(traits.BF16_BYTES)
        layout = fx.make_layout(fx.Int32(per_batch), fx.Int32(1))
        # SBHD [S,B,H,D]: per-batch base is only heads*D (the seq-step lives in the row stride).
        if const_expr(traits.SBHD):
            off = batch_idx * fx.Index(sbhd_heads * head_dim) * fx.Index(traits.BF16_BYTES)
        else:
            off = tok_base * stride * fx.Index(traits.BF16_BYTES)
        return self._view(tensor, off, nrec_bytes, layout)

    def _dma_base(self, buf_id, w_rep, d, buf_base_fn, line_stride):
        traits = self.traits
        wave_id_uni = self.wave_id_uni + fx.Index(w_rep * self.traits.NUM_WAVES)
        lds_byte_base = self.lds_kv_base_idx + buf_base_fn(traits, buf_id) * traits.BF16_BYTES
        lds_addr = (
            lds_byte_base
            + wave_id_uni * (line_stride * traits.BF16_BYTES)
            + (d * traits.SMEM_N_RPT * line_stride * traits.BF16_BYTES)
        )
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(lds_addr)))

    def dma_m0_table(self, base_fn, count):
        return tuple(
            tuple(tuple(base_fn(buf, wr, d) for d in range(count)) for wr in range(self.dma_wave_reps))
            for buf in range(2)
        )

    def split_tile(self, offset_tiles=0):
        return self.split_t0 + fx.Index(offset_tiles)

    def tile_start(self, tile_idx):
        return tile_idx * self.traits.BLOCK_N

    # Pipeline helpers

    def load_all(self):
        self.bind_q_rows()
        return self.assemble_q(self.load_q_raw(self.q_div, self.q_gmem_elem_offset, self.wave_q_offset))

    def bind_q_rows(self):
        """This lane's q row (one q per lane) and the wave's first row, for the bound job."""
        traits = self.traits
        lane = self.job_lane
        if const_expr(traits.ROW_INTERLEAVE):
            self.q_row_in_block = self.wave_q_offset + lane // 16 * traits.QT_ROW_STRIDE + lane % 16
        elif const_expr(traits.Q_TILES == 4):
            self.q_row_in_block = self.wave_q_offset + lane
        else:
            self.q_row_in_block = self.wave_q_offset + lane % 32
        # The wave's first row (with ROW_INTERLEAVE its rows are not contiguous after it).
        self.q_start_pos_i32 = fx.Int32(self.q_start + self.wave_row_group_uni * traits.WAVE_ROW_STRIDE)
        self.q_row = self.q_start + self.q_row_in_block
        self.q_row_i32 = fx.Int32(self.q_row)

    def _q_job_view(self, h_idx, q_block_idx, batch_idx):
        """Q view and block element offset of the persistent job at (h_idx, q_block_idx, batch_idx)."""
        traits = self.traits
        assert traits.Q_HEADS_PER_WG == 1
        q_div = self._batch_view(
            self.Q,
            self.seq_len_v,
            self.stride_q_n_v,
            traits.NUM_HEADS_Q,
            traits.HEAD_DIM,
            batch_idx,
            batch_idx * self.seq_len_v,
        )
        q_head = (h_idx % traits.NUM_HEADS_KV) * traits.GQA_GROUP_SIZE + h_idx // traits.NUM_HEADS_KV
        offset = self.block_row0(q_block_idx) * self.stride_q_n_v + q_head * traits.HEAD_DIM
        return q_div, offset

    def prefetch_q_raw(self, h_idx, q_block_idx, batch_idx):
        """`load_q_raw` of the job at (h_idx, q_block_idx, batch_idx), ahead of its bind_item.
        Rebinds the lane terms; persistent jobs only (never varlen, one q head per WG)."""
        traits = self.traits
        tid = self._job_tid()
        self._bind_lanes(tid)
        q_div, offset = self._q_job_view(h_idx, q_block_idx, batch_idx)
        return self.load_q_raw(q_div, offset, tid // traits.WARP_SIZE * traits.WAVE_ROW_STRIDE)

    def stage_q_dma(self, h_idx, q_block_idx, batch_idx):
        """The paired k-steps' Q of the job at (h_idx, q_block_idx, batch_idx), by DMA into the
        waves' O stage tiles: d blocks 0 and 2 are lines 0 and 2 of every 384-byte Q row, so each
        quarter-wave moves whole lines where a gather spends a line on every 16 bytes.

        The last wave issues none: in the dense drain it runs the diagonal QK alone while the
        others wait at its barrier. Every other wave fills its own tile and the first
        Q_STAGE_LAST_SPLIT share the last one's. The owner may read a tile (`read_q_stage`) after
        the issuers' vmcnt wait and a barrier."""
        traits = self.traits
        assert traits.Q_LDS_STAGE and Q_STAGE_LAST_SPLIT < traits.NUM_WAVES
        tid = self._job_tid()
        q_div, offset = self._q_job_view(h_idx, q_block_idx, batch_idx)
        lane = tid % traits.WARP_SIZE
        wave = self.wave_id_uni
        wave_bytes = self._o_stage_wave_bytes
        rows_per_dma = traits.WARP_SIZE // Q_STAGE_ROW_CHUNKS
        dma_bytes = traits.WARP_SIZE * 16
        # Stage row r is q row r of its tile; its chunk p holds the row's part-A chunk
        # (p - r) % 16, so `read_q_stage` meets 16 distinct chunks per k-step. A DMA fills 4 rows
        # from a multiple of 4, so its lane chunks only depend on the DMA's index mod 4.
        chunk_offs = []
        for i in range_constexpr(4):
            j = (lane % 16 + 32 - lane // 16 - rows_per_dma * i) % 16
            chunk_offs.append(j % 8 * traits.VEC_KV + j // 8 * 128)

        # Stage rows come in q tiles of 16: past each, the block row jumps to the next q tile.
        tile_rows = 16
        wrap_rows = rows_per_dma + traits.QT_ROW_STRIDE - tile_rows

        def _issue(tile_bytes, first_row, n_dma):
            """n_dma DMAs from block row first_row, the start of a q tile."""
            m0_base = rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(self.lds_o_base_idx + tile_bytes)))
            # A running sum: separate i * stride products are job-invariant and get hoisted.
            row_elem = offset + (first_row + lane // 16) * self.stride_q_n_v
            row_step = rows_per_dma * self.stride_q_n_v
            wrap_step = row_step if wrap_rows == rows_per_dma else wrap_rows * self.stride_q_n_v
            for i in range_constexpr(n_dma):
                lds_ptr = fx.inttoptr(self.lds_ptr_ty, fx.Int32(m0_base) + fx.Int32(i * dma_bytes))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(q_div, (None, fx.Int32(row_elem + chunk_offs[i % 4])))
                fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(0))
                row_elem = row_elem + (wrap_step if (i + 1) * rows_per_dma % tile_rows == 0 else row_step)

        last = traits.NUM_WAVES - 1
        dmas_per_tile = traits.ROWS_PER_WAVE // rows_per_dma
        share = dmas_per_tile // Q_STAGE_LAST_SPLIT
        share_rows = share * rows_per_dma
        assert share_rows % tile_rows == 0

        @flyc.jit
        def _own():
            if wave < fx.Index(last):
                _issue(wave * wave_bytes, wave * traits.WAVE_ROW_STRIDE, dmas_per_tile)

        @flyc.jit
        def _last_share():
            if wave < fx.Index(Q_STAGE_LAST_SPLIT):
                _issue(
                    last * wave_bytes + wave * (share * dma_bytes),
                    last * traits.WAVE_ROW_STRIDE + wave * (share_rows // tile_rows * traits.QT_ROW_STRIDE),
                    share,
                )

        _own()
        _last_share()

    def prefetch_q_staged(self, h_idx, q_block_idx, batch_idx):
        """`prefetch_q_raw` with the paired k-steps' packs left as None, for `read_q_stage` to
        fill in from what `stage_q_dma` moved; only the middle block is gathered."""
        traits = self.traits
        tid = self._job_tid()
        self._bind_lanes(tid)
        q_div, offset = self._q_job_view(h_idx, q_block_idx, batch_idx)
        wave_q_offset = tid // traits.WARP_SIZE * traits.WAVE_ROW_STRIDE
        return self.load_q_raw(q_div, offset, wave_q_offset, keep=lambda ks16: ks16 >= QK16_PAIRED_STEPS)

    @property
    def _o_stage_wave_bytes(self):
        return self.traits.ROWS_PER_WAVE * self.traits.O_STAGE_ROW_ELEMS * self.traits.BF16_BYTES

    def read_q_stage(self, q_raw):
        """The packs `prefetch_q_staged` left as None, from the wave's O stage tile once its
        `stage_q_dma` is visible. The wave's own O store after this is ordered behind the reads."""
        traits = self.traits
        tid = self._job_tid()
        wave, lane = tid // traits.WARP_SIZE, tid % traits.WARP_SIZE
        # Lane group c reads part-A chunk 8 (c % 2) + 2 s + c // 2 of q row lane % 16 + 16 qt.
        c = lane // 16
        row = wave * self._o_stage_wave_bytes + lane % 16 * Q_STAGE_ROW_BYTES
        steps = _qk16_k_steps(traits)
        out = list(q_raw)
        for s in range_constexpr(QK16_PAIRED_STEPS):
            chunk = (c % 2 * 8 + c // 2 + 2 * s + lane % 16) % 16
            ptr = self.lds_o_byte_ptr(row + chunk * 16)
            for qt in range_constexpr(traits.Q_TILES):
                pack = _lds_load_o_128(traits, ptr, qt * 16 * Q_STAGE_ROW_BYTES)
                assert out[qt * steps + s] is None
                out[qt * steps + s] = pack.bitcast(self.elem_dtype).ir_value()
        return out

    def load_q_raw(self, q_div, q_elem_offset, wave_q_offset, keep=None):
        """This lane's Q packs as loaded (one vector per 16-byte load), before `assemble_q`; with
        `keep`, only the k-steps it accepts are loaded and the rest are None."""
        traits = self.traits
        lane = self.job_lane
        if const_expr(traits.Q_TILES == 4):
            q_row_in_block = wave_q_offset + lane
        else:
            q_row_in_block = wave_q_offset + lane % 32

        n_packs = traits.Q_TILES * _qk16_k_steps(traits) if traits.QK_MFMA16 else traits.K_STEPS_QK
        if const_expr(traits.QK_MFMA16):
            # Per-lane d terms of the 16x16x32 B operand, matching the K reads (QK16_PAIRED_STEPS).
            _c = lane // 16
            paired_d = (_c % 2) * 128 + (_c // 2) * traits.VEC_KV
            middle_d = (_c % 2) * 4 * traits.VEC_KV + (_c // 2) * traits.VEC_KV
        q_raw_packs = []
        for ks in range_constexpr(n_packs):
            if const_expr(keep is not None and not keep(ks % _qk16_k_steps(traits))):
                q_raw_packs.append(None)
                continue
            if const_expr(traits.QK_MFMA16):
                # 16x16x32 B operand, pack qt * _qk16_k_steps + ks16: q row 16 qt + lane % 16, the
                # d chunk K reads at k-step ks16 (QK16_PAIRED_STEPS).
                qt, ks16 = ks // _qk16_k_steps(traits), ks % _qk16_k_steps(traits)
                lane_d = paired_d if ks16 < QK16_PAIRED_STEPS else middle_d
                elem_index = q_elem_offset + (
                    (wave_q_offset + lane % 16 + traits.QT_ROW_STRIDE * qt) * self.stride_q_n_v
                    + (_qk16_step_d(ks16) + lane_d)
                )
            else:
                elem_index = q_elem_offset + (
                    q_row_in_block * self.stride_q_n_v
                    + (ks * self.traits.K_STEP_QK + lane // 32 * self.traits.MFMA_LANE_K)
                )
            q_i32_pack = fly.copy_atom_call_ssa(
                [self.q_load_i32x4_type],
                self.load_atom_128,
                fx.slice(q_div, (None, fx.Int32(elem_index))),
            )
            q_raw_packs.append(Vec(q_i32_pack, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value())
        return q_raw_packs

    def assemble_q(self, q_raw_packs):
        traits = self.traits
        n_packs = len(q_raw_packs)
        q_16_packs = []
        for pair in range_constexpr(n_packs // 2):
            q_16_packs.append(_concat_vectors(q_raw_packs[pair * 2], q_raw_packs[pair * 2 + 1]))

        q_32_packs = []
        for pair in range_constexpr(n_packs // 4):
            q_32_packs.append(_concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1]))

        q_all = q_32_packs[0]
        for pack in q_32_packs[1:]:
            q_all = _concat_vectors(q_all, pack)
        return Vec(q_all, (n_packs * traits.MFMA_LANE_K,), self.elem_dtype)

    def scale_all(self, q_all_bf16):
        traits = self.traits
        n_elems = q_all_bf16.numel
        fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
        v64f32_type = Vec.make_type(n_elems, fx.Float32)
        q_all_f32_op = llvm.FPExtOp(v64f32_type, as_mlir_value(q_all_bf16))
        q_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
        q_all_f32 = q_all_f32_op.result
        scale_vec = Vec.from_elements([self.c_sm_scale_log2e], fx.Float32).broadcast_to(n_elems)
        q_all_scaled_f32 = arith.mulf(
            as_mlir_value(scale_vec),
            as_mlir_value(q_all_f32),
            fastmath=self.fm_fast,
        )
        if const_expr(self.use_score_bound):
            n_half = traits.K_STEPS_QK * traits.MFMA_LANE_K
            sq = _fmul(Vec(q_all_scaled_f32), Vec(q_all_scaled_f32), self.fm_fast)
            acc = _reduce_vec_tree(sq, n_half, _fadd, self.fm_fast)
            self.q_norm_sq = _lane_pair_reduce(acc, _fadd, self.fm_fast)
        # Truncate one MFMA pack at a time: a 96-wide fptrunc (D_QK 192) legalizes down to an odd
        # width and scalarizes, leaving one bf16 per VGPR that is re-packed with v_perm at every
        # use (and spilled unpacked). Power-of-two packs select v_cvt_pk_bf16_f32 pairwise.
        q_f32_vec = Vec(q_all_scaled_f32)
        n_packs = n_elems // traits.MFMA_LANE_K
        q_packs = []
        for ks in range_constexpr(n_packs):
            base = ks * traits.MFMA_LANE_K
            q_slice = q_f32_vec.shuffle(q_f32_vec, [base + i for i in range(traits.MFMA_LANE_K)])
            trunc_op = llvm.FPTruncOp(Vec.make_type(traits.MFMA_LANE_K, self.elem_dtype), q_slice.ir_value())
            trunc_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            q_packs.append(trunc_op.result)
        if const_expr(traits.PV_AGPR):
            # Q is only ever an MFMA operand, which reads AGPRs directly; defined in VGPRs it is
            # parked in AGPRs anyway and copied back before every QK MFMA.
            q_packs = _pin_regs(q_packs, "=a")
        q_16_packs = []
        for pair in range_constexpr(n_packs // 2):
            q_16_packs.append(_concat_vectors(q_packs[pair * 2], q_packs[pair * 2 + 1]))
        q_32_packs = []
        for pair in range_constexpr(n_packs // 4):
            q_32_packs.append(_concat_vectors(q_16_packs[pair * 2], q_16_packs[pair * 2 + 1]))
        q_all_scaled_bf16 = q_32_packs[0]
        for pack in q_32_packs[1:]:
            q_all_scaled_bf16 = _concat_vectors(q_all_scaled_bf16, pack)
        return Vec(q_all_scaled_bf16, (n_elems,), self.elem_dtype)

    def score_bound(self):
        """Cauchy-Schwarz upper bound on every logit this lane's q row can ever produce."""
        traits = self.traits
        grp = self.kv_head_idx
        if const_expr(traits.SBHD):
            # SBHD K is [S, B, Hkv, D]: a row carries B*Hkv groups, batch-major.
            grp = self.batch_idx * fx.Index(traits.NUM_HEADS_KV) + grp
        n_chunks = SCORE_BOUND_KV_CHUNKS
        rsrc = buffer_ops.create_buffer_resource(
            self.ScoreBound,
            max_size=False,
            num_records_bytes=as_mlir_value((grp + fx.Index(1)) * fx.Index(n_chunks * 4)),
        )
        base = grp * fx.Index(n_chunks)
        kn_sq_vec = None
        for c in range_constexpr(0, n_chunks, 4):
            part = buffer_ops.buffer_load(rsrc, fx.Int32(base + fx.Index(c)), vec_width=4, dtype=fx.Float32)
            kn_sq_vec = part if kn_sq_vec is None else _fmax(Vec(kn_sq_vec), Vec(part), self.fm_fast)
        kn_sq = _reduce_vec_tree(kn_sq_vec, 4, _fmax, self.fm_fast)
        bound = fmath.sqrt(fx.Float32(_fmul(self.q_norm_sq, kn_sq, self.fm_fast)), fastmath=self.fm_fast)
        return _fmul(fx.Float32(bound), fx.Float32(SCORE_BOUND_SLACK), self.fm_fast)

    def floor_at_score_bound(self, row_max, bound):
        """Floor the prologue maximum at -bound, which is a valid row maximum by the same inequality."""
        return _fmax(row_max, _fsub(self.c_zero_f, bound, self.fm_fast), self.fm_fast)

    def arm_score_bound(self, bound, ref):
        """Latch, once, whether any row in this wave can outrun its reference."""
        self.bound_over_ref = _wave_any(ArithValue(fx.Float32(bound) > fx.Float32(ref)))

    def score_acc_seed(self):
        """QK accumulates into C, so seeding C with -seed_offset(reference) shifts the scores for free."""
        if self.ref_neg is None:
            return self.c_zero_o if self.traits.PV_MFMA16 else self.c_zero_v16f32
        if self.ref_seed_vec is None:
            self.ref_seed_vec = self._seed_vec(self.ref_neg)
        return self.ref_seed_vec

    def _seed_vec(self, ref_neg):
        if const_expr(not self.traits.QK_MFMA16):
            return Vec.from_elements([as_mlir_value(ref_neg)], fx.Float32).broadcast_to(16)
        return self._qt_pair_vec(ref_neg)

    def _qk16_chain(self, k_part, q_all, acc16, ks_lo, ks_hi):
        return self._qk16_halves([(k_part, acc16)], q_all, ks_lo, ks_hi)[0]

    def _qk16_both(self, k_lo, k_hi, q_all, s_lo, s_hi, ks_lo, ks_hi, qts=None):
        return tuple(self._qk16_halves([(k_lo, s_lo), (k_hi, s_hi)], q_all, ks_lo, ks_hi, qts))

    def qk_scores(self, v_s):
        """A finished QK's scores in the 32x32x16 layout every consumer expects.

        The 16x16x32 QK leaves its accumulators raw: swapping them inside the QK region keeps
        all of its K packs live at once (the scheduler stops streaming them), so the swap runs
        in a later region, after those packs are dead."""
        if const_expr(not self.traits.QK_MFMA16 or self.traits.PV_MFMA16):
            return v_s
        return tuple(self._qk16_to_score_layout(half) for half in v_s)

    def _qt_vals(self, x):
        """A per-row value held one q per lane, as this lane's value for each raw q tile
        (q = lane % 16 + 16 qt).

        Two q tiles (q = lane % 32): the owner of each q is this lane or its row-pair partner,
        and one swap returns exactly (qt0, qt1). Four (q = lane): the row-pair swap gives
        lane rows 0/1 (qt0, qt1) and rows 2/3 (qt2, qt3); the half-wave swaps of each then
        hand every lane both pairs."""
        a, b = _permlane16_swap_f32(x, x)
        if const_expr(self.QT == 2):
            return [a, b]
        v0, v2 = _permlane32_swap_f32(a, a)
        v1, v3 = _permlane32_swap_f32(b, b)
        return [v0, v1, v2, v3]

    def _qt_pair_vec(self, x):
        """A per-row value spread over a raw 16x16 half: [qt0 x 4, ..., qt(QT-1) x 4] x 2."""
        vals = self._qt_vals(x)
        return Vec.from_elements([v for v in vals for _ in range(4)] * 2, fx.Float32)

    def _row_max_owned(self, v_s):
        """Per-q maximum of a raw 16x16 score tile, back on one q per lane.

        Two q tiles: the swap hands even lane rows their partner's qt0 maximum and odd rows
        their partner's qt1 one, so one max pairs both q tiles; the half-wave swap then joins
        the other two lane groups, which hold the same q. Four: the half-wave swaps first join
        lane rows c and c ^ 2 for (qt0, qt2) and (qt1, qt3), leaving rows 0/1 with qt0/qt1 and
        rows 2/3 with qt2/qt3; the row-pair swap then joins c and c ^ 1, so lane row c ends up
        with the whole maximum of q tile c (q = lane)."""
        m = []
        for qt in range_constexpr(self.QT):
            vals = [half[i] for half in v_s for i in _pv16_qt_elems(qt, self.QT)]
            m.append(_reduce_list_tree(vals, _fmax, self.fm_fast))
        if const_expr(self.QT == 2):
            x, y = _permlane16_swap_f32(m[0], m[1])
            return _lane_pair_reduce(_fmax(x, y, self.fm_fast), _fmax, self.fm_fast)
        m02 = _fmax(*_permlane32_swap_f32(m[0], m[2]), self.fm_fast)
        m13 = _fmax(*_permlane32_swap_f32(m[1], m[3]), self.fm_fast)
        return _fmax(*_permlane16_swap_f32(m02, m13), self.fm_fast)

    def _qk16_halves(self, halves, q_all, ks_lo, ks_hi, qts=None):
        """16x16x32 QK over K slots [ks_lo, ks_hi) and q tiles `qts` (all by default) for
        (k_part, acc) 32-kv halves, on the raw accumulator layout (see `qk_scores`)."""
        traits = self.traits
        nqt = self.QT
        qts = tuple(range(nqt)) if qts is None else tuple(qts)
        accs = []
        for _, acc16 in halves:
            v = Vec(acc16)
            accs.append([v.shuffle(v, [4 * i + e for e in range(4)]).ir_value() for i in range(2 * nqt)])
        q_vec = Vec(q_all)
        for ks in range_constexpr(ks_lo, ks_hi):
            for qt in qts:
                base = (qt * _qk16_k_steps(traits) + ks // 2) * traits.MFMA_LANE_K
                q_pack = q_vec.shuffle(q_vec, [base + i for i in range(traits.MFMA_LANE_K)]).ir_value()
                a = nqt * (ks % 2) + qt
                for h in range_constexpr(len(halves)):
                    accs[h][a] = _mfma_acc(
                        halves[h][0][ks], q_pack, accs[h][a], self.qk16_mma_atom, self.rowsum_acc_vec_type
                    )
        return [_concat_tree(acc) for acc in accs]

    def _qk16_to_score_layout(self, raw16):
        """Swap qt1 of lane row 2m with qt0 of row 2m + 1: every lane then holds q = lane % 32, and the
        swap's first result is score element 8 t + j of kv tile t, its second element 8 t + 4 + j."""
        r = Vec(raw16)
        out = [None] * 16
        for t in range_constexpr(2):
            for j in range_constexpr(4):
                x, y = _permlane16_swap_f32(r[8 * t + j], r[8 * t + 4 + j])
                out[8 * t + j] = x
                out[8 * t + 4 + j] = y
        return Vec.from_elements(out, fx.Float32).ir_value()

    def set_seed_vec(self, value):
        """Adopt a loop-carried seed so the broadcast is not rebuilt every iteration."""
        self.ref_seed_vec = Vec(as_mlir_value(value))

    def lift_ref(self, m_row):
        return _fadd(m_row, fx.Float32(REF_MARGIN), self.fm_fast)

    def seed_offset(self, ref):
        """What the QK seed subtracts: the reference, or 0 while it is still the masked-row floor.

        fl(3e38 + q.k) is 3e38, so seeding a floored row would hand its first live tile identical
        scores. Seeding it with 0 keeps those scores exact; rebase_if_needed then lifts the row to
        that tile's maximum plus REF_MARGIN, as if the tile had been the prologue."""
        if const_expr(not self.seed_floored):
            return ref
        ref = fx.Float32(ref)
        return fx.Float32(ArithValue(ref > fx.Float32(SEED_FLOOR_LIMIT)).select(ref, self.c_zero_f))

    def set_ref(self, ref):
        self.ref_neg = _fsub(self.c_zero_f, self.seed_offset(ref), self.fm_fast)
        self.ref_seed_vec = None

    def qk(self, v_k, q_all_scaled_bf16, v_s=None, ks_range=None, qts=None):
        """`qts` (16x16x32 only): the q tiles to compute; the others keep their accumulators."""
        k_lo, k_hi = v_k
        ks_lo, ks_hi = (0, self.traits.K_STEPS_QK) if ks_range is None else ks_range
        if v_s is None:
            v_s_lo = self.score_acc_seed()
            v_s_hi = v_s_lo
        else:
            v_s_lo, v_s_hi = v_s
        if const_expr(self.traits.QK_MFMA16):
            return self._qk16_both(k_lo, k_hi, q_all_scaled_bf16, v_s_lo, v_s_hi, ks_lo, ks_hi, qts)
        assert qts is None
        for ks in range_constexpr(ks_lo, ks_hi):
            _q_vec = Vec(q_all_scaled_bf16)
            _base = ks * self.traits.MFMA_LANE_K
            q_pack = _q_vec.shuffle(_q_vec, [_base + i for i in range(self.traits.MFMA_LANE_K)]).ir_value()
            v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type)
            v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type)
        return (v_s_lo, v_s_hi)

    # Skip a 32-col QK half that is wholly -inf for this wave (window/causal edge).
    # Predicates are wave-uniform; the mask still writes -inf over skipped accs.
    def chunk_live(self, tile_idx):
        traits = self.traits
        kv0 = fx.Int32(self.tile_start(tile_idx))
        q_lo = fx.Int32(self.q_start_pos_i32 + self.delta_i32)
        q_hi = fx.Int32(q_lo + fx.Int32((traits.Q_TILES - 1) * traits.QT_ROW_STRIDE + 15))
        live_hi = ArithValue(fx.Int32(kv0 + fx.Int32(32)) <= q_hi)
        if const_expr(traits.WINDOW_LEFT >= 0):
            win_lo = fx.Int32(q_lo - fx.Int32(traits.WINDOW_LEFT))
            live_lo = ArithValue(fx.Int32(kv0 + fx.Int32(31)) >= win_lo)
        else:
            # Dense causal: LO half is dead when kv0 already exceeds the wave's last q row.
            live_lo = ArithValue(kv0 <= q_hi)
        return live_lo, live_hi

    def qk_live(self, v_k, q_all_scaled_bf16, live_lo, live_hi, v_s=None, ks_range=None):
        k_lo, k_hi = v_k
        traits = self.traits
        ks_lo, ks_hi = (0, traits.K_STEPS_QK) if ks_range is None else ks_range
        if v_s is None:
            s_lo_in = s_hi_in = self.score_acc_seed()
        else:
            s_lo_in, s_hi_in = v_s

        def _chain(k_part, acc):
            if const_expr(traits.QK_MFMA16):
                return self._qk16_chain(k_part, q_all_scaled_bf16, acc, ks_lo, ks_hi)
            for ks in range_constexpr(ks_lo, ks_hi):
                _q_vec = Vec(q_all_scaled_bf16)
                _base = ks * traits.MFMA_LANE_K
                q_pack = _q_vec.shuffle(_q_vec, [_base + i for i in range(traits.MFMA_LANE_K)]).ir_value()
                acc = _mfma_acc(k_part[ks], q_pack, acc, self.mma_atom, self.mfma_acc_vec_type)
            return acc

        if live_lo is None:
            v_s_lo = _chain(k_lo, s_lo_in)
        else:

            @flyc.jit
            def _qk_lo(v_s_lo, live_lo):
                if live_lo:
                    v_s_lo = _chain(k_lo, v_s_lo)
                return v_s_lo

            v_s_lo = _qk_lo(s_lo_in, live_lo)

        @flyc.jit
        def _qk_hi(v_s_hi, live_hi):
            if live_hi:
                v_s_hi = _chain(k_hi, v_s_hi)
            return v_s_hi

        v_s_hi = _qk_hi(s_hi_in, live_hi)
        return (v_s_lo, v_s_hi)

    def pv_step_k(self, step, v_p, v_v, v_o, after_mfma=None, qts=None):
        """`qts` (16x16x32 only): the q tiles to accumulate; the others keep their O."""
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        if const_expr(self.traits.PV_MFMA16):
            # Step s: kv half s // 2 against dv tiles 4 (s % 2) + dc. v_o[dc] holds the 16x16
            # accumulators [(dc, qt) for qt] + [(dc + 4, qt) for qt] of O^T.
            nqt = self.QT
            qts = tuple(range(nqt)) if qts is None else tuple(qts)
            assert after_mfma is None or len(qts) == nqt
            p_packs = v_p_lo if step < 2 else v_p_hi
            g = step % 2
            for dc in range_constexpr(self.traits.D_CHUNKS):
                v = Vec(v_o[dc])
                accs = [v.shuffle(v, [4 * i + e for e in range(4)]).ir_value() for i in range(2 * nqt)]
                for qt in qts:
                    if const_expr(self.traits.PV_AGPR):
                        accs[nqt * g + qt] = _mfma16_acc_agpr(v_pk[dc], p_packs[qt], accs[nqt * g + qt])
                    else:
                        accs[nqt * g + qt] = _mfma_acc(
                            v_pk[dc],
                            p_packs[qt],
                            accs[nqt * g + qt],
                            self.qk16_mma_atom,
                            self.rowsum_acc_vec_type,
                        )
                    if const_expr(after_mfma is not None):
                        after_mfma(dc * nqt + qt)
                v_o[dc] = _concat_tree(accs)
            return v_o
        if const_expr(step < 2):
            p_pk = v_p_lo[step]
        else:
            p_pk = v_p_hi[step - 2]
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pv(self, v_p, v_v, v_o):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o)
        return v_o

    def reduce_max(self, v_s):
        if const_expr(self.traits.PV_MFMA16):
            return self._row_max_owned(v_s)
        return _lane_pair_reduce(
            _reduce_score_pair(v_s, self.c_neg_inf, _fmax, self.fm_fast), _fmax, self.fm_fast
        )

    def floor_masked_max(self, row_max):
        return _fmax(row_max, self.c_neg_floor, self.fm_fast)

    def _exp2_clamped(self, x):
        """exp2 with P held at 1, which the hardware applies as an output modifier."""
        e = rocdl.exp2(T.f32, as_mlir_value(x))
        if self.ref_neg is None:
            return e
        lo = _fmax(fx.Float32(e), self.c_zero_f, self.fm_fast)
        return _fmin(fx.Float32(lo), fx.Float32(1.0), self.fm_fast)

    def spec_softmax(self, v_s, max_chunk=8, exps_per_piece=2):
        """The tile-max tree and first-half exp2 of ``v_s`` as small pieces of VALU work, for
        the caller to slot between asm MFMAs (which sched_group_barrier cannot see).

        Returns (pieces, result); after the last piece, result() is the ``spec`` that
        `rebase_if_needed` takes. The exp2 half is pinned before the branch, which would
        otherwise sink it into the only arm that uses it."""
        if isinstance(v_s[0], list):
            v_s = _score_lists_to_vecs(v_s)
        n = _numel(v_s[0])
        flat = [Vec(half)[r] for half in v_s for r in range_constexpr(n)]
        p_lo = [None] * n
        out = {}
        pieces = []

        # A running max: each piece folds its chunk into the maximum so far (max is exact in
        # any order), so no reduce of the chunk maxima is left for one long piece at the end.
        def _fold(c):
            vals = flat[c : c + max_chunk]
            if const_expr(c == 0):
                m, vals = _reduce_list_tree(vals[:3], _fmax, self.fm_fast), vals[3:]
            else:
                m = out["m"]
            for i in range_constexpr(0, len(vals), 2):
                m = _reduce_list_tree([m] + vals[i : i + 2], _fmax, self.fm_fast)
            out["m"] = m

        for c in range_constexpr(0, len(flat), max_chunk):
            pieces.append(lambda c=c: _fold(c))

        def _exps(r0):
            for r in range_constexpr(r0, r0 + exps_per_piece):
                p_lo[r] = self._exp2_clamped(Vec(v_s[0])[r])
            if const_expr(r0 + exps_per_piece == n):
                out["p"] = _pin_regs([Vec.from_elements(p_lo, fx.Float32)], "=v")[0]

        for r0 in range_constexpr(0, n, exps_per_piece):
            pieces.append(lambda r0=r0: _exps(r0))
        return pieces, lambda: (out["m"], out["p"])

    def finish_p_pieces(self, l_row, v_p, exps_per_piece=4):
        """The carried P's second exp2 half, then its cast and row sum, as pieces for the
        caller to slot into a QK; after the last piece, result() is (v_p, l_row)."""
        n = _numel(v_p[1])
        lo = [Vec(v_p[0])[r] for r in range_constexpr(n)]
        hi = [Vec(v_p[1])[r] for r in range_constexpr(n)]
        out = {}

        def _exps(r0):
            for r in range_constexpr(r0, r0 + exps_per_piece):
                hi[r] = self._exp2_clamped(hi[r])

        pieces = []
        for r0 in range_constexpr(0, n, exps_per_piece):
            pieces.append(lambda r0=r0: _exps(r0))
        pieces.append(lambda: out.update(r=self.cast_p_and_sum(l_row, (lo, hi), fence=False)))
        return pieces, lambda: out["r"]

    def exp2(self, v_s, start, length):
        n = _numel(v_s[0])
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(n)]
            lo_partial = []
            for r in range_constexpr(n):
                lo_partial.append(self._exp2_clamped(s_lo[r]))
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(n)]
        hi_full = []
        for r in range_constexpr(n):
            hi_full.append(self._exp2_clamped(Vec(v_s[1])[r]))
        return lo_partial, hi_full

    def cast_p_and_sum(self, l_row, v_p, fence=True):
        if const_expr(not self.traits.DUALWAVE_SWP_MFMA_ROWSUM):
            tile_sum = _lane_pair_reduce(
                _reduce_score_pair(v_p, self.c_zero_f, _fadd, self.fm_fast),
                _fadd,
                self.fm_fast,
            )
            l_row = _fadd(l_row, tile_sum, self.fm_fast)
            return self.cast_p(v_p), l_row

        """Pack P to bf16 and fold the tile into the row sum; the MFMA path feeds the packs
        themselves to a ones-matrix MFMA so numerator and denominator see the same bf16 values.
        Each 16-kv pack is one 16x16x32 MFMA against the ones A operand, which also folds the
        half-wave partner, so no permlane pair reduce is left; the fence keeps the packs from
        being hoisted and the scored fptrunc supplies the wait states."""
        v_p = self.cast_p(v_p)
        p_lo_packs, p_hi_packs = v_p
        if const_expr(fence):
            _sched_barrier(0)
        if const_expr(self.traits.PV_MFMA16):
            # One accumulator per q tile ([qt0 x 4, qt1 x 4, ...]); every row of D holds the q sum.
            l_v = Vec(l_row)
            l_acc = [l_v.shuffle(l_v, [4 * qt + e for e in range(4)]).ir_value() for qt in range(self.QT)]
            for packs in (p_lo_packs, p_hi_packs):
                for qt in range_constexpr(self.QT):
                    l_acc[qt] = _mfma_acc(
                        self.rowsum_ones_a,
                        packs[qt],
                        l_acc[qt],
                        self.rowsum_mma_atom,
                        self.rowsum_acc_vec_type,
                    )
            return v_p, _concat_tree(l_acc)
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            for pack in (p_lo_packs[pks], p_hi_packs[pks]):
                l_row = _mfma_acc(
                    self.rowsum_ones_a,
                    pack,
                    l_row,
                    self.rowsum_mma_atom,
                    self.rowsum_acc_vec_type,
                )
        return v_p, l_row

    def finish_row_sum(self, l_row):
        if const_expr(not self.traits.DUALWAVE_SWP_MFMA_ROWSUM):
            return l_row
        """Take the row sum out of the MFMA accumulator; D element 0 holds this lane's q."""
        if const_expr(self.traits.PV_MFMA16):
            # q = lane % 32 is qt0 of an even lane row and qt1 of an odd one; q = lane is q tile c.
            if const_expr(self.QT == 2):
                return ArithValue(fx.Index(self.pv16_c_odd) != fx.Index(0)).select(
                    Vec(l_row)[4], Vec(l_row)[0]
                )
            out = Vec(l_row)[0]
            for qt in range_constexpr(1, self.QT):
                out = ArithValue(fx.Index(self.pv16_owned_qt) == fx.Index(qt)).select(Vec(l_row)[4 * qt], out)
            return out
        return Vec(l_row)[0]

    def cast_p(self, v_p):
        traits = self.traits
        elem_dtype = self.elem_dtype

        def _pack_v8(f32_vals):
            if const_expr(traits.DTYPE_STR == "bf16"):
                # A vector fptrunc still selects v_cvt_pk_bf16_f32 pairwise, but as a scored op:
                # the backend places the pack-to-MFMA wait states itself, so consumers need no
                # hand fence (inline asm hides the VGPR def from GCNHazardRecognizer).
                f32_vec = Vec.from_elements([as_mlir_value(v) for v in f32_vals], fx.Float32)
                trunc_op = llvm.FPTruncOp(Vec.make_type(8, elem_dtype), as_mlir_value(f32_vec))
                trunc_op.operation.attributes["fastmathFlags"] = ir.Attribute.parse("#llvm.fastmath<fast>")
                return trunc_op.result
            f16_vals = []
            for i in range_constexpr(8):
                f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
            return Vec.from_elements(f16_vals, elem_dtype).ir_value()

        lo_partial_list, hi_full = v_p
        if const_expr(traits.PV_MFMA16):
            return tuple(
                [_pack_v8([half[i] for i in _pv16_qt_elems(qt, self.QT)]) for qt in range_constexpr(self.QT)]
                for half in (lo_partial_list, hi_full)
            )
        p_lo_packs = []
        p_hi_packs = []
        for pks in range_constexpr(traits.PV_K_STEPS):
            p_base = pks * 8
            lo_slice = [lo_partial_list[p_base + s] for s in range_constexpr(8)]
            hi_slice = hi_full[p_base : p_base + 8]
            p_lo_packs.append(_pack_v8(lo_slice))
            p_hi_packs.append(_pack_v8(hi_slice))
        return p_lo_packs, p_hi_packs

    def safe_l_inv(self, l_row):
        l_inv = rocdl.rcp(T.f32, as_mlir_value(l_row))
        return ArithValue(fx.Float32(l_row) > self.c_zero_f).select(l_inv, self.c_zero_f)

    def load_sink_h(self):
        """Load the per-q-head attention sink (epilogue, not prologue)."""
        traits = self.traits
        _sink_rsrc = buffer_ops.create_buffer_resource(
            self.SINK,
            max_size=False,
            num_records_bytes=as_mlir_value(fx.Index(traits.NUM_HEADS_Q) * fx.Index(4)),
        )
        return fx.Float32(buffer_ops.buffer_load(_sink_rsrc, self.q_head_idx, vec_width=1, dtype=fx.Float32))

    def finalize_o_scale(self, m_row, l_row):
        """Return (o_scale, m_out, l_out): the normalizing scale applied to the unnormalized O
        and the (max, denom) pair to store as LSE. m_row/l_row are in the log2 domain
        (l_row = sum_j 2^(s_j - m_row)).

        With HAS_SINK, fold the learned per-head sink (a virtual key with logit=sink_h,
        value=0) into the denominator, mirroring sparse_mla_fwd._epi_scalars:
            mf = max(m_row, sink_log2);  af = 2^(m_row-mf);  st = 2^(sink_log2-mf)
            l_out = l_row*af + st;  O *= af/l_out  (sink's value=0 -> no O contribution)
        Without HAS_SINK, af==1 and mf==m_row, so this is byte-identical to the plain 1/l path."""
        if const_expr(self.traits.HAS_SINK):
            fm = self.fm_fast
            sink_log2 = _fmul(self.load_sink_h(), fx.Float32(LOG2E), fm)
            mf = fx.Float32(_fmax(m_row, sink_log2, fm))
            af = fx.Float32(rocdl.exp2(T.f32, _fsub(m_row, mf, fm)))
            st = fx.Float32(rocdl.exp2(T.f32, _fsub(sink_log2, mf, fm)))
            l_out = fx.Float32(_fadd(_fmul(l_row, af, fm), st, fm))
            o_scale = fx.Float32(_fmul(af, self.safe_l_inv(l_out), fm))
            return o_scale, mf, l_out
        return self.safe_l_inv(l_row), m_row, l_row

    def scale_o(self, v_o, scale_scalar):
        if const_expr(self.traits.PV_AGPR):
            _pv_agpr_settle()
        if const_expr(self.traits.PV_MFMA16):
            scale_vec = self._qt_pair_vec(scale_scalar)
        else:
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec, self.fm_fast)

    def scores_for_softmax(self, v_s):
        return v_s

    def shift_scores(self, v_s, row_max):
        if isinstance(v_s[0], list):
            v_s = _score_lists_to_vecs(v_s)
        if const_expr(self.traits.PV_MFMA16):
            m_vec = self._qt_pair_vec(row_max)
            return tuple(_fsub(Vec(half), m_vec, self.fm_fast) for half in v_s)
        shifted = []
        for half in v_s:
            shifted.append(
                Vec.from_elements(
                    [as_mlir_value(_fsub(Vec(half)[r], row_max, self.fm_fast)) for r in range_constexpr(16)],
                    fx.Float32,
                ).ir_value()
            )
        return tuple(shifted)

    def rebase_if_needed(self, ref, l_row, v_s, v_o, pending=None, spec=None):
        """Raise the reference only for a tile that has outrun it, which is rare.

        With the Cauchy-Schwarz bound armed the trigger is loop-invariant, so the 32-score
        max tree moves inside the guard and only runs for a row whose bound clears the
        reference. Without it the tile's own maximum is the trigger and has to be reduced
        on every tile to find out.

        The accumulator correction this site owes is applied to whatever ``v_o`` holds when
        the branch runs, so the caller decides how much of the P*V it covers. A trigger that
        only resolves after the P*V has to place the branch there and correct its own site.
        A loop-invariant trigger can instead run the branch before the P*V and hand the
        correction to ``pending`` for the next site to pay, once its P*V has landed: that
        keeps the P*V MFMAs and the exp2 chain they hide in one scheduling region and still
        costs one branch per site, which only works because both sites test the same
        wave-uniform value.

        ``spec`` = (tile max, first-half exp2) from `spec_softmax`, already taken against
        the unmoved reference ahead of the branch so they could hide under the P*V; a firing
        tile redoes the exp2 on the shifted scores, so every bit matches the plain order. The
        first half then comes back as P, as from ``exp2(v_s, 0, n)``."""
        if isinstance(v_s[0], list):
            v_s = _score_lists_to_vecs(v_s)
        gated = self.bound_over_ref is not None
        # Only a row without the score-bound floor can sit at the floor, and those take the exact trigger.
        assert not (gated and self.seed_floored), "a floored seed needs the exact per-tile trigger"
        assert not (gated and spec is not None), "a speculative site needs the exact trigger"
        if const_expr(not gated):
            if const_expr(spec is not None):
                m_lane_always = spec[0]
            else:
                m_lane_always = _reduce_score_tree(v_s, _fmax, self.fm_fast)
            assert pending is None, "an exact trigger cannot pay another site's debt"
            if const_expr(self.seed_floored):
                # The scores are s - off, off being what the seed subtracted (see seed_offset). A row
                # outruns its reference once a score clears ref - off: +0 for a seeded row, and the
                # floor itself for one that has met no live key, which any live score clears.
                off = _fsub(self.c_zero_f, Vec(as_mlir_value(self.score_acc_seed()))[0], self.fm_fast)
                ref_over_off = _fsub(ref, off, self.fm_fast)
                fired = _wave_any(ArithValue(fx.Float32(m_lane_always) > fx.Float32(ref_over_off)))
            else:
                fired = _wave_any(ArithValue(fx.Float32(m_lane_always) > self.c_zero_f))
        else:
            fired = self.bound_over_ref
        seed_in = self.score_acc_seed()

        def _then():
            if const_expr(self.traits.PV_MFMA16):
                m_tile = self._row_max_owned(v_s)
            else:
                if const_expr(gated):
                    m_lane = _reduce_score_tree(v_s, _fmax, self.fm_fast)
                else:
                    m_lane = m_lane_always
                m_tile = _lane_pair_reduce(m_lane, _fmax, self.fm_fast)
            if const_expr(self.traits.CAUSAL):
                m_tile = self.floor_masked_max(m_tile)
            if const_expr(self.seed_floored):
                # New reference max(ref, off + m_tile + margin), carried relative to off. A floored
                # row adopts its first live tile's maximum + margin (r is 0 against an empty sum);
                # one whose tile is masked too keeps the floor, with r = 1.
                delta = _fmax(_fadd(m_tile, fx.Float32(REF_MARGIN), self.fm_fast), ref_over_off, self.fm_fast)
                r = fx.Float32(rocdl.exp2(T.f32, _fsub(ref_over_off, delta, self.fm_fast)))
                out = [_fadd(off, delta, self.fm_fast), self.scale_l_by(l_row, r)]
            else:
                delta = _fmax(
                    _fadd(m_tile, fx.Float32(REF_MARGIN), self.fm_fast), self.c_zero_f, self.fm_fast
                )
                r = fx.Float32(rocdl.exp2(T.f32, _fsub(self.c_zero_f, delta, self.fm_fast)))
                out = [_fadd(ref, delta, self.fm_fast), self.scale_l_by(l_row, r)]
            if const_expr(self.traits.PV_MFMA16):
                delta_vec = self._qt_pair_vec(delta)
                out += [_fsub(Vec(half), delta_vec, self.fm_fast) for half in v_s]
            else:
                for half in v_s:
                    out.append(
                        Vec.from_elements(
                            [
                                as_mlir_value(_fsub(Vec(half)[i], delta, self.fm_fast))
                                for i in range_constexpr(16)
                            ],
                            fx.Float32,
                        ).ir_value()
                    )
            if const_expr(spec is not None):
                out[2] = self.exp2((out[2], out[3]), 0, 16)[0]
            # The next tile's seed goes through seed_offset too: a row still at the floor after a
            # rebase other rows fired must stay unseeded.
            out.append(self._seed_vec(_fsub(self.c_zero_f, self.seed_offset(out[0]), self.fm_fast)))
            out += self._scale_acc(v_o, r if pending is None else pending[1])
            if const_expr(pending is not None):
                out.append(r)
            return out

        s_pass = list(v_s)
        if const_expr(spec is not None):
            s_pass[0] = spec[1]
        tail = list(v_o) + ([] if pending is None else [self.no_rescale()])
        res = _if_wave(fired, [ref, l_row] + s_pass + [seed_in] + tail, _then)
        self.set_ref(res[0])
        self.set_seed_vec(res[4])
        nd = self.traits.D_CHUNKS
        v_o_out = list(res[5 : 5 + nd])
        if const_expr(pending is None):
            return res[0], res[1], (res[2], res[3]), v_o_out
        return res[0], res[1], (res[2], res[3]), v_o_out, (fired, res[5 + nd])

    def no_rescale(self):
        """The neutral accumulator rescale a site hands on when the reference did not move."""
        return as_mlir_value(fx.Float32(1.0))

    def _scale_acc(self, v_o, rescale):
        if const_expr(self.traits.PV_AGPR):
            _pv_agpr_settle()
        if const_expr(self.traits.PV_MFMA16):
            r_vec = self._qt_pair_vec(rescale)
        else:
            r_vec = Vec.from_elements([as_mlir_value(rescale)], fx.Float32).broadcast_to(16)
        return [_fmul(Vec(v_o[dc]), r_vec, self.fm_fast) for dc in range_constexpr(self.traits.D_CHUNKS)]

    def rebase_acc(self, v_o, pending):
        """Apply a reference rise to the output accumulator, once the P*V measured against the old reference has been added to it."""
        fired, rescale = pending
        return list(_if_wave(fired, list(v_o), lambda: self._scale_acc(v_o, rescale)))

    def scale_l_by(self, l_row, rescale):
        """Apply a row-max correction to the running row sum, in whichever form it is held.

        The MFMA row sum lives in an accumulator vector rather than a scalar; only element 0
        is ever read, but the correction has to reach every lane the accumulator feeds."""
        if rescale is None:
            return l_row
        if const_expr(self.traits.PV_MFMA16):
            vals = self._qt_vals(rescale)
            return _fmul(
                Vec(l_row), Vec.from_elements([v for v in vals for _ in range(4)], fx.Float32), self.fm_fast
            )
        if const_expr(self.traits.DUALWAVE_SWP_MFMA_ROWSUM):
            return (Vec(l_row) * fx.Float32(rescale)).ir_value()
        return _fmul(l_row, rescale, self.fm_fast)

    def v_s_vec_to_lists(self, v_s):
        s_lo, s_hi = v_s
        n = _numel(s_lo)
        return (
            [Vec(s_lo)[r] for r in range_constexpr(n)],
            [Vec(s_hi)[r] for r in range_constexpr(n)],
        )

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None, *, do_causal=True, do_window=True):
        # Python bools: each instantiation emits only the requested edge.
        if q_row_i32 is None:
            q_row_i32 = self.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        if const_expr(traits.PV_MFMA16):
            # Raw 16x16 element (kt, qt, e) of half h is kv 32 h + 16 kt + pv16_kv_offset + e and
            # q = q_row + QT_ROW_STRIDE (qt - pv16_owned_qt) (q_row is the one-q-per-lane row).
            qt_rows = traits.QT_ROW_STRIDE
            rel0_i32 = fx.Int32(
                q_row_i32
                + self.delta_i32
                - kv_start_i32
                - fx.Int32(self.pv16_kv_offset)
                - fx.Int32(self.pv16_owned_qt) * fx.Int32(qt_rows)
            )
            neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
            pairs = [(0, 1), (2, 3), (16, 17), (18, 19)]
            edges = []
            if const_expr(do_causal):
                edges.append((0, "lt"))
            if const_expr(traits.WINDOW_LEFT >= 0 and do_window):
                edges.append((traits.WINDOW_LEFT, "gt"))
            for h, half in enumerate((s_lo, s_hi)):
                for qt in range_constexpr(self.QT):
                    idx = _pv16_qt_elems(qt, self.QT)
                    vals = [half[i] for i in idx]
                    for w, cmp in edges:
                        rel = fx.Int32(rel0_i32 + fx.Int32(qt_rows * qt - 32 * h - w))
                        _apply_dualwave_mask_pair(vals, rel, neg_inf_i32, pairs, cmp)
                    for i, v in zip(idx, vals):
                        half[i] = v
            return
        # lane>=32 has a larger n offset in the K-permuted P layout.
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        rel_lo_i32 = fx.Int32(q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32)
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)

        pair_thresholds = [(0, 1), (2, 3), (8, 9), (10, 11), (16, 17), (18, 19), (24, 25), (26, 27)]
        if const_expr(do_causal):
            _apply_dualwave_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds, "lt")
            _apply_dualwave_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds, "lt")
        # SWA left window: additionally mask columns below the window (phys_kv < q_row + delta - W).
        if const_expr(traits.WINDOW_LEFT >= 0 and do_window):
            w_i32 = fx.Int32(traits.WINDOW_LEFT)
            rel_win_lo_i32 = fx.Int32(rel_lo_i32 - w_i32)
            rel_win_hi_i32 = fx.Int32(rel_hi_i32 - w_i32)
            _apply_dualwave_mask_pair(s_lo, rel_win_lo_i32, neg_inf_i32, pair_thresholds, "gt")
            _apply_dualwave_mask_pair(s_hi, rel_win_hi_i32, neg_inf_i32, pair_thresholds, "gt")

    def causal_mask_prologue_if_needed(
        self, v_s, tile_idx=None, kv_end_pos=None, q_start_pos_i32=None, q_row_i32=None, *, kv_end_tile=None
    ):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        if kv_end_pos is None:
            end_tile = tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile
            kv_end_pos = self.tile_start(end_tile)
        if q_start_pos_i32 is None:
            q_start_pos_i32 = self.q_start_pos_i32
        if q_row_i32 is None:
            q_row_i32 = self.q_row_i32

        traits = self.traits
        # SWA lower-edge guard is deliberately conservative; _causal_mask_inplace is a no-op on tiles that need no masking.
        kv_start_pos_w = self.tile_start(tile_idx)

        @flyc.jit
        def _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32):
            s_lo, s_hi = v_s
            _need = q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos)
            if const_expr(traits.WINDOW_LEFT >= 0):
                # Window and causal edges never share a tile; apply each mask only when needed.
                _win_edge = (
                    q_start_pos_i32 + fx.Int32(traits.BLOCK_M) + self.delta_i32 - fx.Int32(traits.WINDOW_LEFT)
                )
                _need_win = ArithValue(fx.Int32(kv_start_pos_w) < _win_edge)
                if _need:
                    lo_list, hi_list = self.v_s_vec_to_lists((s_lo, s_hi))
                    self._causal_mask_inplace(
                        (lo_list, hi_list), tile_idx, q_row_i32=q_row_i32, do_window=False
                    )
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
                if _need_win:
                    lo_list, hi_list = self.v_s_vec_to_lists((s_lo, s_hi))
                    self._causal_mask_inplace(
                        (lo_list, hi_list), tile_idx, q_row_i32=q_row_i32, do_causal=False
                    )
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            else:
                if _need:
                    lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                    self._causal_mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                    s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos, q_start_pos_i32, q_row_i32)

    def causal_mask_split_prologue_if_needed(self, v_s, offset_tiles=0, end_offset_tiles=1):
        return self.causal_mask_prologue_if_needed(
            v_s,
            self.split_tile(offset_tiles),
            kv_end_tile=self.split_tile(end_offset_tiles),
        )

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        traits = self.traits
        seqlen_kv_i32 = self.seqlen_kv_i32

        @flyc.jit
        def _seq_pad_mask_if_needed(v_s, tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + fx.Index(1)) * traits.BLOCK_N
            if fx.Int32(kv_tile_end) > seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                if const_expr(self.traits.PV_MFMA16):
                    col_base = fx.Int32(tile_idx * self.traits.BLOCK_N) + fx.Int32(self.pv16_kv_offset)
                else:
                    col_base = fx.Int32(tile_idx * self.traits.BLOCK_N) + fx.Int32(
                        self.lane_div_32
                    ) * fx.Int32(4)
                for r in range_constexpr(len(lo_list)):
                    if const_expr(self.traits.PV_MFMA16):
                        col_lo = col_base + fx.Int32(16 * (r // (4 * self.QT)) + r % 4)
                    else:
                        col_lo = col_base + fx.Int32((r // 4) * 8 + (r % 4))
                    col_hi = col_lo + fx.Int32(32)
                    lo_list[r] = ArithValue(col_lo < self.seqlen_kv_i32).select(lo_list[r], self.c_neg_inf)
                    hi_list[r] = ArithValue(col_hi < self.seqlen_kv_i32).select(hi_list[r], self.c_neg_inf)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _seq_pad_mask_if_needed(v_s, tile_idx)

    def _load_kv(
        self, tile_start, buf_id, dma_m0, div, num_dma, src_base, row_stride, chunk_fn=None, pieces=None
    ):
        """`chunk_fn(d, line)`: the global 16-byte chunk this lane stores in d block d (None: its own).
        With ``pieces`` (a list) each DMA is appended there as a thunk for the caller to issue."""
        soffset = tile_start * row_stride
        # w_rep: each wave fills DMA_WAVE_REPS LDS lines (1 for 8-wave, 2 for 4-wave).
        for wr in range_constexpr(self.dma_wave_reps):
            eff_wave = self.dma_wave + fx.Index(wr * self.traits.NUM_WAVES)
            for d in range_constexpr(num_dma):
                # LDS line stride = SMEM_N_RPT (# lines, always 8), NOT NUM_WAVES: at the 8-wave
                # CTA they coincide, but a 4-wave CTA needs each wave to fill 2 lines, with
                # ``eff_wave`` carrying the effective line index (wave_id + w_rep*NUM_WAVES).
                n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + eff_wave
                d_bucket = self.d_bucket if chunk_fn is None else chunk_fn(d, eff_wave)
                global_d = d_bucket * self.traits.VEC_KV + d * self.traits.D_128B_SIZE
                src_elem = src_base + n_in_tile * row_stride + global_d
                # 128-bit global->LDS DMA; `src_elem` is voffset, `soffset` is scaled by the atom.
                lds_ptr = fx.inttoptr(self.lds_ptr_ty, fx.Int32(dma_m0[buf_id][wr][d]))
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(div, (None, fx.Int32(src_elem)))
                if const_expr(pieces is None):
                    fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(soffset))
                else:
                    pieces.append(
                        lambda src=src, dst=dst: fx.copy(self.dma_atom, src, dst, soffset=fx.Int32(soffset))
                    )

    def load_k_tile(self, tile_idx, buf_id, pieces=None):
        self._load_kv(
            self.tile_start(tile_idx),
            buf_id,
            self.k_dma_m0,
            self.k_div,
            self.NUM_DMA_K,
            self.kv_gmem_elem_offset,
            self.stride_kv_n_v,
            chunk_fn=self._k_dma_chunk if self.traits.QK_MFMA16 else None,
            pieces=pieces,
        )

    def prefetch_k_tile(self, h_idx, batch_idx, tile_idx, buf_id):
        """K tile `tile_idx` of the job at (head h_idx, batch_idx) into `buf_id`, ahead of that
        job's bind_item. Rebinds the lane terms; persistent jobs only (never varlen)."""
        traits = self.traits
        self._bind_lanes(self._job_tid())
        k_div = self._batch_view(
            self.K,
            self.seq_len_kv_v,
            self.stride_kv_n_v,
            traits.NUM_HEADS_KV,
            traits.HEAD_DIM,
            batch_idx,
            batch_idx * self.seq_len_kv_v,
        )
        self._load_kv(
            self.tile_start(tile_idx),
            buf_id,
            self.k_dma_m0,
            k_div,
            self.NUM_DMA_K,
            (h_idx % traits.NUM_HEADS_KV) * traits.HEAD_DIM,
            self.stride_kv_n_v,
            chunk_fn=self._k_dma_chunk if traits.QK_MFMA16 else None,
        )

    def _k_dma_chunk(self, d, line):
        if d != 1:
            return self.d_bucket
        return (self.d_bucket + 4 * _k_mid_swizzled(line, self.n_in_warp)) % self.traits.VEC_KV

    def load_k_split(self, offset_tiles, buf_id):
        self.load_k_tile(self.split_tile(offset_tiles), buf_id)

    def load_v_tile(self, tile_idx, buf_id, pieces=None):
        self._load_kv(
            self.tile_start(tile_idx),
            buf_id,
            self.v_dma_m0,
            self.v_div,
            self.NUM_DMA_V,
            self.v_gmem_elem_offset,
            self.stride_v_n_v,
            chunk_fn=(lambda d, line: self.v16_dma_d_bucket) if self.traits.PV_MFMA16 else None,
            pieces=pieces,
        )

    def load_v_split(self, offset_tiles, buf_id):
        self.load_v_tile(self.split_tile(offset_tiles), buf_id)

    def lds_load_k(self, buf_id, urk_base=None, ks_range=None, k_regs=None):
        """Read this buffer's K packs for k-steps [ks_range). Splitting the range keeps only half the K tile resident in registers at once."""
        assert urk_base is None or not self.traits.QK_MFMA16, "QK_MFMA16 reads K from its own two bases"
        if urk_base is None:
            urk_base = self.k_lds_read_base_per_lane
        ks_lo, ks_hi = (0, self.traits.K_STEPS_QK) if ks_range is None else ks_range
        k_base = _k_buf_base(self.traits, buf_id)
        if k_regs is None:
            k_lo = [None] * self.traits.K_STEPS_QK
            k_hi = [None] * self.traits.K_STEPS_QK
        else:
            k_lo, k_hi = k_regs

        for ks in range_constexpr(ks_lo, ks_hi):
            if const_expr(self.traits.QK_MFMA16):
                # Slot ks holds kv tile 2 * half + ks % 2 at 16x16x32 k-step ks // 2.
                ks16 = ks // 2
                step_d = _qk16_step_d(ks16)
                if ks16 < QK16_PAIRED_STEPS:
                    _idx_lo = urk_base + step_d
                else:
                    _idx_lo = (
                        self.k_lds_read_base_middle
                        + self.traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE
                        + (step_d - 64)
                    )
                _idx_lo = _idx_lo + k_base + (ks % 2) * 2 * self.traits.D_128B_SIZE
                _idx_hi = _idx_lo + 4 * self.traits.D_128B_SIZE
            else:
                ks_off = (ks // 4) * self.traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
                    ks % 4
                ) * self.traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE
                _idx_lo = k_base + urk_base + ks_off
                _idx_hi = _idx_lo + self.traits.K_LDS_TO_REG_N_STRIP_STRIDE
            k_lo[ks] = _load_k_pack_aligned(
                self.traits,
                self.lds_kv_base_ptr,
                _idx_lo,
                buf_id,
                self.kv_mfma_pack_type,
            )
            k_hi[ks] = _load_k_pack_aligned(
                self.traits,
                self.lds_kv_base_ptr,
                _idx_hi,
                buf_id,
                self.kv_mfma_pack_type,
            )
        return (k_lo, k_hi)

    def lds_load_v(self, buf_id, urv_base=None, substeps=None, packs=None, dcs=None):
        """`dcs`: the d-chunk range to read (PV_MFMA16 only; default all)."""
        if urv_base is None:
            urv_base = self.v_lds_read_base_per_lane
        v_base = _v_buf_base(self.traits, buf_id)
        ks_lo, ks_hi = (0, 4) if substeps is None else substeps
        if packs is None:
            packs = [[None] * self.traits.D_CHUNKS for _ in range(4)]
        if const_expr(self.traits.PV_MFMA16):
            assert urv_base is self.v_lds_read_base_per_lane, "PV_MFMA16 reads V from its own two bases"
            return self._lds_load_v16(ks_lo, ks_hi, packs, buf_id, dcs)
        assert dcs is None, "only the PV_MFMA16 read splits by d chunk"
        pair_off = self.traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * self.traits.BF16_BYTES
        lane_ptr = buffer_ops.get_element_ptr(
            self.lds_kv_base_ptr,
            byte_offset=as_mlir_value(fx.Int32((v_base + urv_base) * self.traits.BF16_BYTES)),
            elem_type=T.i8,
        )
        # k-substep major: the two reads of every P*V step land back to back, so the
        # backend's own lgkmcnt for step k does not also wait on the later substeps.
        for k_substep in range_constexpr(ks_lo, ks_hi):
            for dc in range_constexpr(self.traits.D_CHUNKS):
                dc_off = (dc // 2) * self.traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
                    dc % 2
                ) * self.traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE
                imm_lo = (
                    k_substep * self.traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + dc_off
                ) * self.traits.BF16_BYTES
                a = _ds_read_tr16_b64(self.traits, self.v_lds_read_vec4_type, lane_ptr, imm_lo, buf_id)
                b = _ds_read_tr16_b64(
                    self.traits, self.v_lds_read_vec4_type, lane_ptr, imm_lo + pair_off, buf_id
                )
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def _lds_load_v16(self, ks_lo, ks_hi, packs, buf_id, dcs=None):
        """V^T packs of P*V step s (kv half s // 2, dv tiles 4 (s % 2) + dc): kv tile 0 of the half,
        then kv tile 1 (16 kv = two LDS rows on), each a ds_read_b64_tr_b16."""
        traits = self.traits
        dc_lo, dc_hi = (0, traits.D_CHUNKS) if dcs is None else dcs
        lane_ptrs = [
            buffer_ops.get_element_ptr(self.lds_kv_base_ptr, byte_offset=as_mlir_value(off), elem_type=T.i8)
            for off in self.v16_read_off[buf_id]
        ]
        for step in range_constexpr(ks_lo, ks_hi):
            for dc in range_constexpr(dc_lo, dc_hi):
                imm = (
                    (step // 2) * 4 * traits.D_128B_SIZE
                    + (step % 2) * traits.SMEM_N_RPT * traits.SMEM_V_LINE_STRIDE
                    + 16 * (dc - dc % 2)
                ) * traits.BF16_BYTES
                a = _ds_read_tr16_b64(traits, self.v_lds_read_vec4_type, lane_ptrs[dc % 2], imm, buf_id)
                b = _ds_read_tr16_b64(
                    traits,
                    self.v_lds_read_vec4_type,
                    lane_ptrs[dc % 2],
                    imm + 2 * traits.D_128B_SIZE * traits.BF16_BYTES,
                    buf_id,
                )
                packs[step][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def _final_o_base(self, q_row):
        return q_row * self.stride_o_n_v + self.q_head_idx * self.traits.HEAD_DIM_V + self.lane_div_32 * 8

    def _final_o_global(self, o_base, dc, g):
        return o_base + (dc * self.traits.D_CHUNK + 2 * g * 8)

    def _store_o_128(self, o_base, pack_fn):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                _buffer_store_128(
                    pack_fn(dc, g),
                    self._final_o_global(o_base, dc, g),
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=self.o_div,
                )

    def zero_o_block_if_needed(self, causal_end_raw_i32=None):
        if causal_end_raw_i32 is None:
            causal_end_raw_i32 = self.causal_end_raw_i32
        q_start = self.q_start
        wave_q_offset = self.wave_q_offset
        lane_mod_32 = self.lane_mod_32
        seq_len_v = self.seq_len_v

        @flyc.jit
        def _zero_o_block_if_needed():
            if causal_end_raw_i32 <= fx.Int32(0):
                q_row_z = q_start + wave_q_offset + lane_mod_32
                c_zero_i = fx.Int32(0)
                zero_pack = Vec.from_elements([c_zero_i, c_zero_i, c_zero_i, c_zero_i], fx.Int32)
                if q_row_z < seq_len_v:
                    self._store_o_128(self._final_o_base(q_row_z), lambda dc, g: zero_pack)

        _zero_o_block_if_needed()

    @property
    def o_stores_per_lane(self):
        """buffer_store instructions `store_final_o` issues per lane (16 B each on every path)."""
        if self.traits.PV_MFMA16 and self.QT == 4:
            return self.traits.D_CHUNKS * 4
        return self.traits.D_CHUNKS * 2

    def store_final_o(self, v_o, q_row):
        if const_expr(self.traits.PV_MFMA16 and self.QT == 4):
            return self._store_final_o16x4(v_o, q_row)
        if const_expr(self.traits.PV_MFMA16):
            return self._store_final_o16(v_o, q_row)
        self._store_o_128(
            self._final_o_base(q_row),
            lambda dc, g: _packed_o_128_vec(self.traits, v_o, dc, g, self.lane_div_32, self.elem_dtype),
        )

    def _store_final_o16(self, v_o, q_row):
        """O^T accumulator (dv tile t, qt): this lane holds dv 16 t + 4 c + [0, 4) of q
        lane % 16 + 16 qt. Swapping qt1 of even lane rows with qt0 of odd ones gives every lane
        8 consecutive dv of its own q = q_row, dv 16 t + 8 (c // 2): one 16-byte store per tile."""
        traits = self.traits
        o_base = q_row * self.stride_o_n_v + self.q_head_idx * traits.HEAD_DIM_V + 8 * (self.lane // 32)
        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                x = _o_pack_2dw(traits, v_o, dc, 2 * g, self.elem_dtype)
                y = _o_pack_2dw(traits, v_o, dc, 2 * g + 1, self.elem_dtype)
                r0, r1 = _permlane16_swap_i32(x[0], y[0])
                s0, s1 = _permlane16_swap_i32(x[1], y[1])
                _buffer_store_128(
                    Vec.from_elements([fx.Int32(w) for w in (r0, s0, r1, s1)], fx.Int32),
                    o_base + 16 * (4 * g + dc),
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=self.o_div,
                )

    def _store_final_o16x4(self, v_o, q_row):
        """Four q tiles: lane row c holds dv 16 t + 4 c + [0, 4) of q lane % 16 + 16 qt. A 4x4
        transpose across lane rows (half-wave swaps join rows c and c ^ 2, then row-pair swaps
        c and c ^ 1) gives every lane all 16 dv of tile t for its own q = lane: two 16-byte stores.
        Those touch 64 q rows each, 16 B per row; with O_LDS_STAGE they go to the wave's LDS
        tile instead, which is then stored four whole rows (16 lanes each) per buffer_store."""
        traits = self.traits
        o_base = q_row * self.stride_o_n_v + self.q_head_idx * traits.HEAD_DIM_V
        if const_expr(traits.O_LDS_STAGE):
            # Fresh lane terms: CSE with the job's own would keep them live across the KV loop.
            tid = self._job_tid()
            lane = tid % traits.WARP_SIZE
            row_bytes = traits.O_STAGE_ROW_ELEMS * traits.BF16_BYTES
            wave_tile = tid // traits.WARP_SIZE * (traits.ROWS_PER_WAVE * row_bytes)
            stage_lane = self.lds_o_byte_ptr(wave_tile + lane * row_bytes)
        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                x = [_o_pack_2dw(traits, v_o, dc, 4 * g + qt, self.elem_dtype) for qt in range(4)]
                # z[k][a][w]: q tile 2 (lane // 32) + k, dv 8 a + 4 (c % 2) + 2 w of the tile.
                z = [[[None] * 2 for _ in range(2)] for _ in range(2)]
                for k in range_constexpr(2):
                    for w in range_constexpr(2):
                        z[k][0][w], z[k][1][w] = _permlane32_swap_i32(x[k][w], x[k + 2][w])
                for a in range_constexpr(2):
                    words = [None] * 4
                    for w in range_constexpr(2):
                        words[w], words[2 + w] = _permlane16_swap_i32(z[0][a][w], z[1][a][w])
                    pack = Vec.from_elements([fx.Int32(v) for v in words], fx.Int32)
                    dv = 16 * (dc + 4 * g) + 8 * a
                    if const_expr(traits.O_LDS_STAGE):
                        _lds_store_o_128(traits, pack, stage_lane, dv * traits.BF16_BYTES)
                    else:
                        _buffer_store_128(
                            pack,
                            o_base + dv,
                            _o_store_reg_128=self.o_store_reg_128,
                            _store_atom_128=self.store_atom_128,
                            o_div=self.o_div,
                        )
        if const_expr(traits.O_LDS_STAGE):
            # The wave reads back only its own tile, and its DS ops complete in order: no barrier.
            lanes_per_row = traits.HEAD_DIM_V * traits.BF16_BYTES // 16
            rows_per_store = traits.WARP_SIZE // lanes_per_row
            row, chunk = lane // lanes_per_row, lane % lanes_per_row
            read_lane = self.lds_o_byte_ptr(wave_tile + row * row_bytes + chunk * 16)
            # Stage row r is q row r of the wave (one q per lane): q tile r // 16, row r % 16.
            if const_expr(traits.ROW_INTERLEAVE):
                wave_row0 = q_row - lane // 16 * traits.QT_ROW_STRIDE - lane % 16
            else:
                wave_row0 = q_row - lane
            o_row = (wave_row0 + row) * self.stride_o_n_v + self.q_head_idx * traits.HEAD_DIM_V
            o_row = o_row + chunk * (16 // traits.BF16_BYTES)
            # A running sum: separate i * stride products are job-invariant and get hoisted.
            o_step = rows_per_store * self.stride_o_n_v
            wrap_rows = rows_per_store + traits.QT_ROW_STRIDE - 16
            wrap_step = o_step if wrap_rows == rows_per_store else wrap_rows * self.stride_o_n_v
            for i in range_constexpr(traits.ROWS_PER_WAVE // rows_per_store):
                _buffer_store_128(
                    _lds_load_o_128(traits, read_lane, i * rows_per_store * row_bytes),
                    o_row,
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=self.o_div,
                )
                o_row = o_row + (wrap_step if (i + 1) * rows_per_store % 16 == 0 else o_step)

    def lds_o_byte_ptr(self, byte_offset):
        return buffer_ops.get_element_ptr(
            self.lds_o_base_ptr, byte_offset=as_mlir_value(fx.Int32(byte_offset)), elem_type=T.i8
        )

    def store_lse(self, m_row, l_row, q_row):
        """LSE: m_row/l_row live in the log2 domain, hence LSE_natural =
        ln2 * (m_row + log2(l_row)); written as f32 per (row, head) into the DebugCounts slot."""
        traits = self.traits
        lse_rsrc = self.lse_rsrc
        q_tok_base = self.q_tok_base
        q_head_idx = self.q_head_idx
        seqlen_q_v = self.seqlen_q_v
        lane = self.lane
        fm_fast = self.fm_fast
        _ln2 = host_math.log(2.0)
        # Two q tiles keep each q on two lanes; four give every lane its own q.
        owner_lanes = 16 * self.QT

        @flyc.jit
        def _store_lse():
            if q_row < seqlen_q_v:
                if lane < fx.Index(owner_lanes):
                    log2_l = fx.Float32(fmath.log2(as_mlir_value(fx.Float32(l_row)), fastmath=fm_fast))
                    lse = _fmul(_fadd(m_row, log2_l, fm_fast), fx.Float32(_ln2), fm_fast)
                    elem_idx = (q_tok_base + q_row) * fx.Index(traits.NUM_HEADS_Q) + q_head_idx
                    _lse_f32_ir = as_mlir_value(fx.Float32(lse))
                    buffer_ops.buffer_store(_lse_f32_ir, lse_rsrc, as_mlir_value(fx.Int32(elem_idx)))

        _store_lse()


def _scale_sched_pairs(pairs, head_dim):
    """At head_dim 64, use 1/4 of the sched_group_barrier anchors (softmax work is D-independent)."""
    if head_dim != 64:
        return pairs
    return max(1, (pairs + 3) // 4)


def _sched_barrier_pairs(traits, pairs, valu_cnt, group, mask=None):
    # per (MFMA,second) pair: second op is VALU by default, EXP when ``mask`` is passed.
    mask = traits.SCHED_VALU_MASK if mask is None else mask
    # Both the MFMAs and the softmax work of a cluster scale with the wave's q tiles.
    pairs = _scale_sched_pairs(pairs, traits.HEAD_DIM) * (traits.Q_TILES // 2)
    for _ in range_constexpr(pairs):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(mask, valu_cnt, group)


def _interleave(traits, n_mfma, per_mfma, group, mask=None):
    """Unscaled (MFMA, per_mfma x mask) sched_group_barrier pairs; mask defaults to VALU."""
    mask = traits.SCHED_VALU_MASK if mask is None else mask
    for _ in range_constexpr(n_mfma):
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, group)
        rocdl.sched_group_barrier(mask, per_mfma, group)
