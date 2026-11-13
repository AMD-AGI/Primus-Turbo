###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Flash attention forward in FlyDSL, structured after FA4's ``FlashAttentionForwardSm80``
(``flash_attn/cute/flash_fwd.py``) and written for CDNA's wave64 MFMA (gfx942 first).

Layout. A CTA holds ``num_waves`` waves; each wave owns 32 query rows. Scores are
computed transposed, S^T = K . Q^T, with MFMA 32x32x8, so each lane holds one query
row (the accumulator column) and 16 of a 32-key tile's scores for it -- the rest sit
in lane ``^ 32``. That keeps the softmax reductions in-lane plus one shuffle, and lets
the probabilities feed the PV product without any data movement: accumulator group
``g`` (elements ``4g .. 4g+3``) is exactly the B fragment of reduction step ``g`` of
O^T = V^T . P^T. Q lives in registers for the whole loop; each KV tile is staged
global -> registers -> LDS once per CTA and read by every wave. Rows past a sequence's
end are loaded clamped to its last row (never out of bounds) and masked away.

Tensors follow FA4. Dense: q ``[B, Sq, Hq, D]``, k ``[B, Sk, Hk, D]``,
v ``[B, Sk, Hk, Dv]``, out ``[B, Sq, Hq, Dv]``, lse ``[B, Hq, Sq]`` (fp32, natural log).
With ``cu_seqlens_q`` the query side is packed instead -- q / out ``[Tq, H, D]``, lse
``[Hq, Tq]`` -- and likewise k / v with ``cu_seqlens_k``. ``seqused_q`` / ``seqused_k``
override the per-sequence lengths. Shapes and strides are read at runtime, so one
compiled kernel serves every length.

Mods follow FA4 (see ``mods.py``): ``score_mod`` gets the score times ``softmax_scale``
and runs before the change to base 2; ``mask_mod`` is and-ed into the causal / window /
length mask. Both are traced into the kernel and may read up to ``MAX_AUX_TENSORS``
aux tensors. ``learnable_sink`` (fp32 ``[Hq]``) adds one value-less logit per head to
the softmax denominator.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.extras import types as T
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.flex_attention.common import LOG2E, AttentionMask, BlockInfo, SeqlenInfo, Softmax
from primus_turbo.flydsl.flex_attention.torch_utils import (
    broadcast_heads,
    dequantize_fp8,
    flash_attn_combine,
    subtile_factors,
    to_native_fp8,
)

MFMA_M = 32
MAX_AUX_TENSORS = 4
FP8_DTYPES = ("fp8_e4m3", "fp8_e5m2")


def _k_step(dtype: str) -> int:
    """MFMA reduction depth: 32x32x8 for 16-bit inputs, 32x32x16 for fp8."""
    return 16 if dtype in FP8_DTYPES else 8


def _elem_type(dtype: str, fp8_ocp: bool):
    if dtype == "bf16":
        return fx.BFloat16
    if dtype == "fp16":
        return fx.Float16
    if dtype == "fp8_e4m3":
        return fx.Float8E4M3FN if fp8_ocp else fx.Float8E4M3FNUZ
    return fx.Float8E5M2 if fp8_ocp else fx.Float8E5M2FNUZ


def _arch_has_ocp_fp8() -> bool:
    import torch

    return "gfx95" in torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName


def _fp8_v_row(r):
    """LDS row of V-tile row ``r`` for fp8: swap bits 2 and 3 within each 16 keys.

    A lane's fp8 B fragment holds 8 consecutive k (``8 * (lane // 32) + i``), but its
    probabilities sit in accumulator rows ``4 * (lane // 32) + {0..3}`` and
    ``8 + 4 * (lane // 32) + {0..3}``. Permuting V's keys the same way lets P feed the
    PV product unchanged.
    """
    return r + fx.Int32(4) * ((r // fx.Int32(4)) % fx.Int32(2) - (r // fx.Int32(8)) % fx.Int32(2))


def _to_elem(values, elem, dtype: str):
    """fp32 values -> a vector of ``elem``. fp8 goes through the packed hardware
    conversion (arch-native format; plain ``truncf`` to fp8 has no lowering)."""
    if dtype not in FP8_DTYPES:
        return Vec.from_elements(values, fx.Float32).to(elem)
    cvt = rocdl.cvt_pk_bf8_f32 if dtype == "fp8_e5m2" else rocdl.cvt_pk_fp8_f32
    words = []
    for w in range(len(values) // 4):
        lo = cvt(T.i32(), values[4 * w], values[4 * w + 1], fx.Int32(0), False)
        words.append(cvt(T.i32(), values[4 * w + 2], values[4 * w + 3], lo, True))
    return Vec.from_elements(words, fx.Int32).bitcast(elem)


def _descales(fp8, q_descale, k_descale, v_descale, batch, kv_head):
    """``(q * k descale, v descale)`` for this (batch, kv head), FA3/FA4 semantics."""
    if not fp8:
        return None, None
    at = lambda t: fx.Float32(fx.slice(fx.rocdl.make_buffer_tensor(t), (batch, None))[kv_head])  # noqa: E731
    return at(q_descale) * at(k_descale), at(v_descale)


def _min(a, b):
    return (a < b).select(a, b)


# Static dispatch lives outside the kernel body: the AST rewriter turns an ``if`` there
# into branch closures that do not see the kernel's other locals.
def _apply_score_mod(score_mod, score, batch, head, q_pos, kv_pos, mask: AttentionMask):
    if score_mod is None:
        return score
    return score_mod(score, batch, head, mask.clamp_q(q_pos), mask.clamp_kv(kv_pos), mask.seqlen, mask.aux_tensors)


def _sink_log2(has_sink, sink, head):
    if not has_sink:
        return None
    return fx.Float32(fx.rocdl.make_buffer_tensor(sink)[head]) * fx.Float32(LOG2E)


class FlashAttentionForward:
    """Builds and caches the launcher for one compile-time configuration."""

    def __init__(
        self,
        dtype: str = "bf16",
        head_dim: int = 128,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        tile_n: int = 64,
        num_waves: int = 4,
        varlen_q: bool = False,
        varlen_k: bool = False,
        has_seqused_q: bool = False,
        has_seqused_k: bool = False,
        score_mod=None,
        mask_mod=None,
        num_aux_tensors: int = 0,
        has_learnable_sink: bool = False,
        block_sparse: bool = False,
        has_full_blocks: bool = False,
        sparse_block_size: tuple | None = None,
        paged_kv: bool = False,
        split_kv: bool = False,
        fp8_ocp: bool | None = None,
        mfma_k: int | None = None,
    ):
        if dtype not in ("bf16", "fp16") + FP8_DTYPES:
            raise ValueError(f"dtype must be bf16, fp16, fp8_e4m3 or fp8_e5m2, got {dtype}")
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        k_step = _k_step(dtype)
        if head_dim % k_step or head_dim_v % MFMA_M:
            raise ValueError(f"need head_dim % {k_step} == 0 and head_dim_v % {MFMA_M} == 0")
        # FP8 runs natively in the GPU's format: FNUZ on gfx942, OCP on gfx950.
        self.fp8_ocp = _arch_has_ocp_fp8() if dtype in FP8_DTYPES and fp8_ocp is None else bool(fp8_ocp)
        self.input_dtype = dtype
        # FlyDSL 0.3.4.1 has no e5m2 MFMA on either target (no FNUZ e5m2 type for gfx942;
        # its MFMA atom rejects OCP e5m2 on gfx950), so e5m2 inputs are dequantized
        # (exactly; only the descale multiply rounds) and run through the bf16 kernel.
        self.dequant_e5m2 = dtype == "fp8_e5m2"
        if self.dequant_e5m2:
            if paged_kv:
                raise NotImplementedError("paged KV with fp8 e5m2 (no e5m2 MFMA in FlyDSL 0.3.4.1)")
            dtype = "bf16"
        self.fp8 = dtype in FP8_DTYPES
        # MFMA reduction depth: 32x32x16 for fp8, and for 16-bit inputs on gfx950 (twice the
        # K of gfx942's 32x32x8); ``mfma_k`` overrides for 16-bit inputs.
        self.mfma_k = 16 if self.fp8 else (mfma_k or (16 if _arch_has_ocp_fp8() else 8))
        if self.mfma_k not in (8, 16) or head_dim % self.mfma_k:
            raise ValueError(f"mfma_k must be 8 or 16 and divide head_dim, got {self.mfma_k}")
        if tile_n % MFMA_M:
            raise ValueError(f"tile_n must be a multiple of {MFMA_M}")
        if is_causal and is_local:
            raise ValueError("pass causal with a left window as is_local with window_right = 0")
        if not 0 <= num_aux_tensors <= MAX_AUX_TENSORS:
            raise ValueError(f"at most {MAX_AUX_TENSORS} aux tensors, got {num_aux_tensors}")
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.num_aux_tensors = num_aux_tensors
        self.has_learnable_sink = has_learnable_sink
        self.dtype = dtype
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.tile_n = tile_n
        self.num_waves = num_waves
        self.varlen_q = varlen_q
        self.varlen_k = varlen_k
        self.has_seqused_q = has_seqused_q
        self.has_seqused_k = has_seqused_k
        self.tile_m = MFMA_M * num_waves
        self.num_threads = 64 * num_waves
        if block_sparse and (varlen_q or varlen_k):
            raise NotImplementedError("block sparsity with varlen is not supported yet")
        if paged_kv and varlen_k:
            raise ValueError("page_table is not supported with cu_seqlens_k (as in FA4)")
        self.paged_kv = paged_kv
        if split_kv and (block_sparse or has_learnable_sink):
            raise NotImplementedError("split-KV with block sparsity or a learnable sink is not supported")
        self.split_kv = split_kv
        self.block_sparse = block_sparse
        self.has_full_blocks = block_sparse and has_full_blocks
        self.sparse_block_size = sparse_block_size
        self.q_subtile, self.kv_subtile = subtile_factors(sparse_block_size, self.tile_m, tile_n)
        pack = 16 if self.fp8 else 8
        for d in (head_dim, head_dim_v):
            if (tile_n * d // pack) % self.num_threads:
                raise ValueError(f"KV tile {tile_n}x{d} does not split evenly over {self.num_threads} threads")
        self.launcher = self._build()
        # Paged single-token decode goes to FlashAttentionDecode (built on first use).
        self._decode_ok = (
            paged_kv and not varlen_q and not self.fp8 and not self.dequant_e5m2
            and qhead_per_kvhead <= 16 and head_dim == head_dim_v and head_dim % 32 == 0
        )
        self._decode = None

    def _build(self):
        D, DV, TN = self.head_dim, self.head_dim_v, self.tile_n
        NT, TM = self.num_threads, self.tile_m
        Q_PER_KV = self.qhead_per_kvhead
        fp8, dtype = self.fp8, self.dtype
        elem = _elem_type(dtype, self.fp8_ocp)
        KS = self.mfma_k  # reduction depth of one MFMA
        # gfx950 16-bit (KS = 16): P . V reads its V^T A fragments with the LDS transpose
        # read (ds_read_b64_tr_b16), two per fragment, instead of 8 single-element reads.
        VTR = not self.fp8 and KS == 16
        # With VTR, V's 16-byte chunks are XOR-swizzled by (stored) row: a 16-lane transpose
        # read covers 4 consecutive rows x 32 bytes (2 chunks), 256 B apart unswizzled (one
        # 8-bank set); chunk ^= (row & 3) << 1 puts the 4 rows on 4 different chunk pairs.
        V_SWZ = VTR and DV // 8 >= 8
        KPL = KS // 2  # k per lane in an A / B fragment
        PACK = 16 if fp8 else 8  # elements per 128-bit global / LDS access
        out_elem = fx.BFloat16 if fp8 else elem
        k_iters = TN * D // PACK // NT
        v_iters = TN * DV // PACK // NT
        block_info = BlockInfo(TM, TN, self.is_causal, self.is_local)
        is_causal, is_local = self.is_causal, self.is_local
        varlen_q, varlen_k = self.varlen_q, self.varlen_k
        has_seqused_q, has_seqused_k = self.has_seqused_q, self.has_seqused_k
        score_mod, mask_mod = self.score_mod, self.mask_mod
        num_aux, has_sink = self.num_aux_tensors, self.has_learnable_sink
        block_sparse, has_full_blocks = self.block_sparse, self.has_full_blocks
        Q_SUB, KV_SUB = self.q_subtile, self.kv_subtile
        paged_kv, split_kv = self.paged_kv, self.split_kv
        o_elem = fx.Float32 if split_kv else out_elem

        # K's 16-byte chunks are XOR-swizzled by row (chunk ^= row % 2^K_SWZ): unswizzled,
        # the 32 key rows of an A fragment are a multiple of 128 bytes apart and all read
        # the same LDS bank. A swizzle costs no LDS (padding would push a CTA past 32 KB).
        k_chunks = D // PACK
        K_SWZ = min(3, k_chunks.bit_length() - 1) if k_chunks & (k_chunks - 1) == 0 else 0

        # gfx950 16-bit without block lists: K / V double-buffered in LDS (2 x 32 KB per CTA; the
        # 160 KB LDS holds two CTAs). Tile n+1's global loads are in flight while tile n computes.
        DB = VTR and not block_sparse
        NBUF = 2 if DB else 1

        @fx.struct
        class SharedStorage:
            k: fx.Array[elem, NBUF * TN * D, 16]
            v: fx.Array[elem, NBUF * TN * DV, 16]

        @flyc.kernel(known_block_size=[NT, 1, 1])
        def kernel(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            O: fx.Tensor,  # noqa: E741
            LSE: fx.Tensor,
            CuSeqlensQ: fx.Tensor,
            CuSeqlensK: fx.Tensor,
            SeqUsedQ: fx.Tensor,
            SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor,
            Aux1: fx.Tensor,
            Aux2: fx.Tensor,
            Aux3: fx.Tensor,
            Sink: fx.Tensor,
            MaskCnt: fx.Tensor,
            MaskIdx: fx.Tensor,
            FullCnt: fx.Tensor,
            FullIdx: fx.Tensor,
            PageTable: fx.Tensor,
            QDescale: fx.Tensor,
            KDescale: fx.Tensor,
            VDescale: fx.Tensor,
            page_size: fx.Int32,
            num_splits: fx.Int32,
            seqlen_q_static: fx.Int32,
            seqlen_k_static: fx.Int32,
            softmax_scale: fx.Float32,
            window_left: fx.Int32,
            window_right: fx.Int32,
        ):
            tid = fx.thread_idx.x
            lane = tid % fx.Int32(64)
            wave = tid // fx.Int32(64)
            m_block = fx.Int32(fx.block_idx.x)
            head = fx.Int32(fx.block_idx.y)
            batch = fx.Int32(fx.block_idx.z) // num_splits
            split_idx = fx.Int32(fx.block_idx.z) % num_splits
            kv_head = head // fx.Int32(Q_PER_KV)

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, KS, elem))
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
            thr_mma = tiled_mma.get_slice(lane)
            # Coordinate views: an element's value is its own (row, col) index.
            b_coords = thr_mma.partition_B(fx.make_view(0, fx.make_layout((MFMA_M, KS), (1, MFMA_M))))
            c_rows = thr_mma.partition_C(fx.make_view(0, fx.make_layout((MFMA_M, MFMA_M), (1, 0))))
            c_cols = thr_mma.partition_C(fx.make_view(0, fx.make_layout((MFMA_M, MFMA_M), (0, 1))))

            def mfma(a_frag, b_frag, acc):
                c_frag = fx.make_fragment_like(c_rows, fx.Float32)
                c_frag.store(Vec(acc))
                fx.gemm(tiled_mma, c_frag, a_frag, b_frag, c_frag)
                return c_frag.load()

            copy_g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem)
            # A / B fragments are KS / 2 elements per lane: 64 bits, or 128 for 16-bit x16.
            frag128 = KPL * (8 if fp8 else 16) == 128
            copy_q = fx.make_copy_atom(fx.rocdl.BufferCopy128b() if frag128 else fx.rocdl.BufferCopy64b(), elem)
            # Four output elements per store: 16-bit O, or the fp32 split-KV partial O.
            copy_o = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b() if split_kv else fx.rocdl.BufferCopy64b(), o_elem
            )
            copy_s128 = fx.make_copy_atom(fx.UniversalCopy128b(), elem)
            copy_s64 = fx.make_copy_atom(fx.UniversalCopy128b() if frag128 else fx.UniversalCopy64b(), elem)
            # V^T reads are one element each (strided by DV in LDS).
            copy_s1 = fx.make_copy_atom(fx.UniversalCopy8b() if fp8 else fx.UniversalCopy16b(), elem)
            copy_k = fx.make_tiled_copy_A(copy_s64, tiled_mma).get_slice(lane)
            copy_vt = fx.make_tiled_copy_A(copy_s1, tiled_mma).get_slice(lane)
            a_coords = thr_mma.partition_A(fx.make_view(0, fx.make_layout((MFMA_M, KS), (1, MFMA_M))))
            copy_vtr = fx.make_copy_atom(rocdl.cdna4.LDSReadTrans16_64b(), elem) if VTR else None
            # A transpose read: each 16 lanes supply the addresses of a 4-key x 16-d block
            # (lane: key + (lane % 16) // 4, d + 4 * (lane % 4)) and lane j gets d column j.
            vtr_d = fx.Int32(16) * ((lane % fx.Int32(32)) // fx.Int32(16)) + fx.Int32(4) * (lane % fx.Int32(4))
            vtr_key = (lane % fx.Int32(16)) // fx.Int32(4) + fx.Int32(8) * (lane // fx.Int32(32))
            qk_descale, v_descale = _descales(fp8, QDescale, KDescale, VDescale, batch, kv_head)
            score_scale = softmax_scale * qk_descale if fp8 else softmax_scale

            q_t = fx.rocdl.make_buffer_tensor(Q)
            k_t = fx.rocdl.make_buffer_tensor(K)
            v_t = fx.rocdl.make_buffer_tensor(V)
            o_t = fx.rocdl.make_buffer_tensor(O)
            lse_t = fx.rocdl.make_buffer_tensor(LSE)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            k_layout = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(K_SWZ, PACK.bit_length() - 1, D.bit_length() - PACK.bit_length())),
                fx.make_layout((TN, D), (D, 1)),
            ) if K_SWZ else fx.make_layout((TN, D), (D, 1))
            k_lds_tiles = fx.flat_divide(fx.make_view(lds.k.ptr, k_layout), (MFMA_M, KS))
            k_lds_packs = fx.logical_divide(fx.make_view(lds.k.ptr, fx.make_layout(NBUF * TN * D, 1)), fx.make_layout(PACK, 1))

            def k_tiles_at(buf):
                """K tile views of LDS buffer ``buf`` (runtime index)."""
                base = fx.add_offset(lds.k.ptr, fx.make_int_tuple(buf * fx.Int32(TN * D)))
                return fx.flat_divide(fx.make_view(base, k_layout), (MFMA_M, KS))
            # V^T view of the row-major [TN, DV] V tile: (d, key) with strides (1, DV).
            vt_lds_tiles = fx.flat_divide(fx.make_view(lds.v.ptr, fx.make_layout((DV, TN), (1, DV))), (MFMA_M, KS))
            v_lds_packs = fx.logical_divide(fx.make_view(lds.v.ptr, fx.make_layout(NBUF * TN * DV, 1)), fx.make_layout(PACK, 1))

            seqlen = SeqlenInfo.create(
                batch,
                seqlen_q_static,
                seqlen_k_static,
                cu_seqlens_q=fx.rocdl.make_buffer_tensor(CuSeqlensQ) if varlen_q else None,
                cu_seqlens_k=fx.rocdl.make_buffer_tensor(CuSeqlensK) if varlen_k else None,
                seqused_q=fx.rocdl.make_buffer_tensor(SeqUsedQ) if has_seqused_q else None,
                seqused_k=fx.rocdl.make_buffer_tensor(SeqUsedK) if has_seqused_k else None,
            )
            aux_tensors = tuple(fx.rocdl.make_buffer_tensor(t) for t in (Aux0, Aux1, Aux2, Aux3)[:num_aux])
            mask = AttentionMask(
                seqlen, is_causal, is_local, window_left, window_right, mask_mod, batch, head, aux_tensors
            )

            # Helpers defined in the kernel body are AST-rewritten along with it: an
            # ``if`` becomes branch closures, so a ``return`` inside one would only leave
            # the branch. Choose with conditional expressions and return once.
            def q_side_row(t, pos, h):
                """Row ``pos`` of this sequence in a query-side tensor (q or out)."""
                coord = (seqlen.offset_q + pos, h, None) if varlen_q else (batch, pos, h, None)
                return fx.slice(t, coord)

            def out_row(pos):
                """Row of O, or of this split's partial O ``[num_splits, ...]``."""
                coord = (seqlen.offset_q + pos, head, None) if varlen_q else (batch, pos, head, None)
                return fx.slice(o_t, (split_idx,) + coord if split_kv else coord)

            def lse_head_row():
                coord = (head, None) if varlen_q else (batch, head, None)
                return fx.slice(lse_t, (split_idx,) + coord if split_kv else coord)

            pages = fx.slice(fx.rocdl.make_buffer_tensor(PageTable), (batch, None)) if paged_kv else None

            def kv_side_row(t, pos):
                coord = (
                    (fx.Int32(pages[pos // page_size]), pos % page_size, kv_head, None) if paged_kv
                    else (seqlen.offset_k + pos, kv_head, None) if varlen_k
                    else (batch, pos, kv_head, None)
                )
                return fx.slice(t, coord)

            def stage_kv(kv_base):
                """Copy the K and V tiles at ``kv_base`` to LDS buffer 0."""
                store_kv(load_kv(kv_base), fx.Int32(0))

            def store_kv(pending, buf):
                k_off, v_off = buf * fx.Int32(TN * D // PACK), buf * fx.Int32(TN * DV // PACK)
                for reg, is_k, idx in pending:
                    packs, off = (k_lds_packs, k_off) if is_k else (v_lds_packs, v_off)
                    fx.copy(copy_s128, reg, fx.slice(packs, (None, idx + off)))

            def load_kv(kv_base):
                """Global loads of the K and V tiles at ``kv_base`` (rows clamped to the
                sequence) into registers, with their LDS pack indices. All loads are issued
                before any store, so a tile pays one load latency rather than one per load."""
                last_key = seqlen.seqlen_k - fx.Int32(1)
                pending = []  # (register pack, LDS packs view, pack index)
                for it in range_constexpr(k_iters):
                    linear = fx.Int32(it * NT) + tid
                    key = _min(kv_base + linear // fx.Int32(D // PACK), last_key)
                    src = fx.logical_divide(kv_side_row(k_t, key), fx.make_layout(PACK, 1))
                    reg = fx.make_rmem_tensor(PACK, elem)
                    fx.copy(copy_g128, fx.slice(src, (None, linear % fx.Int32(D // PACK))), reg)
                    k_row, k_chunk = linear // fx.Int32(k_chunks), linear % fx.Int32(k_chunks)
                    k_pack = k_row * fx.Int32(k_chunks) + (k_chunk ^ (k_row & fx.Int32((1 << K_SWZ) - 1)))
                    pending.append((reg, True, k_pack))
                for it in range_constexpr(v_iters):
                    linear = fx.Int32(it * NT) + tid
                    row, col = linear // fx.Int32(DV // PACK), linear % fx.Int32(DV // PACK)
                    key = _min(kv_base + row, last_key)
                    src = fx.logical_divide(kv_side_row(v_t, key), fx.make_layout(PACK, 1))
                    reg = fx.make_rmem_tensor(PACK, elem)
                    fx.copy(copy_g128, fx.slice(src, (None, col)), reg)
                    lds_row = _fp8_v_row(row) if KS == 16 else row
                    v_col = col ^ ((lds_row & fx.Int32(3)) * fx.Int32(2)) if V_SWZ else col
                    pending.append((reg, False, lds_row * fx.Int32(DV // PACK) + v_col))
                return pending

            def vt_frag_elems(tile):
                a_frag = thr_mma.make_fragment_A(tile)
                fx.copy(copy_s1, copy_vt.partition_S(tile), copy_vt.retile(a_frag))
                return a_frag

            def vt_frag_tr(c, t, buf):
                """V^T A fragment (d block c, key step t) from two transpose reads of the
                row-major V tile: keys 8 * (lane // 32) + {0..3}, then {4..7}."""
                d = fx.Int32(c * MFMA_M) + vtr_d
                halves = []
                for h in range_constexpr(2):
                    row = fx.Int32(t * KS + 4 * h) + vtr_key
                    chunk = (d // fx.Int32(8)) ^ ((row & fx.Int32(3)) * fx.Int32(2)) if V_SWZ else d // fx.Int32(8)
                    idx = buf * fx.Int32(TN * DV) + row * fx.Int32(DV) + chunk * fx.Int32(8) + d % fx.Int32(8)
                    src = fx.make_view(fx.add_offset(lds.v.ptr, fx.make_int_tuple(idx)), fx.make_layout(4, 1))
                    dst = fx.make_rmem_tensor(fx.make_layout(4, 1), elem)
                    fx.copy(copy_vtr, src, dst)
                    halves.append(Vec(dst.load()))
                frag = fx.make_fragment_like(a_coords, elem)
                frag.store(halves[0].shuffle(halves[1], list(range(8))))
                return frag

            def sparse_n_block(idx_row, j):
                """Kernel tile of the ``j``-th entry of a block list, sparse blocks split in KV_SUB."""
                return fx.Int32(idx_row[j // fx.Int32(KV_SUB)]) * fx.Int32(KV_SUB) + j % fx.Int32(KV_SUB)

            def compute_one_n_block(n_block, q_frags, q_pos, row_max, row_sum, o_acc, apply_mask_mod=True,
                                    masked=True):
                fx.gpu.barrier()  # every wave is done reading the previous tile
                stage_kv(n_block * fx.Int32(TN))
                fx.gpu.barrier()
                return compute_tile(n_block, fx.Int32(0), q_frags, q_pos, row_max, row_sum, o_acc, apply_mask_mod,
                                    masked)

            def compute_one_n_block_db(n_block, n_min, n_max, q_frags, q_pos, row_max, row_sum, o_acc, masked=True):
                """Double-buffered tile: issue tile n+1's loads, compute tile n from its buffer,
                store tile n+1 into the other buffer (last read a tile ago, before the previous
                barrier), one barrier. The last tile re-stages itself into the idle buffer."""
                buf = (n_block - n_min) & fx.Int32(1)
                pending = load_kv(_min(n_block + fx.Int32(1), n_max - fx.Int32(1)) * fx.Int32(TN))
                res = compute_tile(n_block, buf, q_frags, q_pos, row_max, row_sum, o_acc, True, masked)
                store_kv(pending, fx.Int32(1) - buf)
                fx.gpu.barrier()
                return res

            def compute_tile(n_block, buf, q_frags, q_pos, row_max, row_sum, o_acc, apply_mask_mod, masked):
                kv_base = n_block * fx.Int32(TN)
                k_tiles = k_tiles_at(buf) if DB else k_lds_tiles
                zero16 = Vec.filled(16, 0.0, fx.Float32)
                scores, keep = [], []
                for half in range_constexpr(TN // MFMA_M):
                    acc = zero16
                    for ks in range_constexpr(D // KS):
                        tile = fx.slice(k_tiles, (None, None, half, ks))
                        a_frag = thr_mma.make_fragment_A(tile)
                        fx.copy(copy_s64, copy_k.partition_S(tile), copy_k.retile(a_frag))
                        acc = mfma(a_frag, q_frags[ks], acc)
                    acc = Vec(acc)
                    for j in range_constexpr(16):
                        kv_pos = kv_base + fx.Int32(half * MFMA_M) + fx.Int32(fx.get_scalar(c_rows[j]))
                        score = _apply_score_mod(score_mod, acc[j] * score_scale, batch, head, q_pos, kv_pos, mask)
                        scores.append(score * fx.Float32(LOG2E))
                        keep.append(mask.keep(q_pos, kv_pos, apply_mask_mod=apply_mask_mod) if masked else None)

                probs, new_max, new_sum, rescale = Softmax.online_softmax(row_max, row_sum, scores,
                                                                          keep if masked else None)
                rescale_vec = Vec.from_elements([rescale], fx.Float32).broadcast_to(16)
                o_acc = [Vec(o) * rescale_vec for o in o_acc]
                for half in range_constexpr(TN // MFMA_M):
                    for g in range_constexpr(MFMA_M // KS):
                        # Accumulator groups of 4 in order; with KS = 16 (fp8, or 16-bit on
                        # gfx950) two groups make one fragment, matching V's permuted rows
                        # (``_fp8_v_row``).
                        first = half * 16 + KPL * g
                        p_frag = fx.make_fragment_like(b_coords, elem)
                        p_frag.store(_to_elem(probs[first : first + KPL], elem, dtype))
                        # All d blocks' V^T fragments of this key step are read before their
                        # MFMAs, so the reads overlap instead of each MFMA waiting on its own.
                        t = half * (MFMA_M // KS) + g
                        a_frags = [vt_frag_tr(c, t, buf) if VTR else vt_frag_elems(fx.slice(vt_lds_tiles, (None, None, c, t)))
                                   for c in range_constexpr(DV // MFMA_M)]
                        for c in range_constexpr(DV // MFMA_M):
                            o_acc[c] = mfma(a_frags[c], p_frag, o_acc[c])
                return new_max, new_sum, o_acc

            # A varlen grid is sized for the longest sequence; shorter ones skip the
            # excess query blocks. The condition is uniform over the CTA, so the
            # barriers inside stay convergent.
            if m_block * fx.Int32(TM) < seqlen.seqlen_q:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, window_left, window_right)
                if split_kv:
                    n_block_min, n_block_max = BlockInfo.split(n_block_min, n_block_max, split_idx, num_splits)
                q_pos = m_block * fx.Int32(TM) + wave * fx.Int32(MFMA_M) + fx.Int32(fx.get_scalar(c_cols[0]))
                q_row = _min(q_pos, seqlen.seqlen_q - fx.Int32(1))

                # ---- Q: B fragments for every reduction step, register-resident ----
                q_k0 = fx.Int32(fx.get_scalar(b_coords[0])) // fx.Int32(MFMA_M)  # this lane's first k in a step
                q_packs = fx.logical_divide(q_side_row(q_t, q_row, head), fx.make_layout(KPL, 1))
                q_frags = []
                for ks in range_constexpr(D // KS):
                    reg = fx.make_rmem_tensor(KPL, elem)
                    fx.copy(copy_q, fx.slice(q_packs, (None, (fx.Int32(ks * KS) + q_k0) // fx.Int32(KPL))), reg)
                    frag = fx.make_fragment_like(b_coords, elem)
                    frag.store(Vec(reg.load()))
                    q_frags.append(frag)

                init = Softmax.initial_state() + [Vec.filled(16, 0.0, fx.Float32) for _ in range_constexpr(DV // MFMA_M)]
                results = init
                if block_sparse:
                    # FA4's lists: blocks needing mask_mod, then fully kept blocks without it.
                    # A sparse KV block spans KV_SUB kernel tiles, a sparse Q block Q_SUB m_blocks.
                    sparse_m = m_block // fx.Int32(Q_SUB)
                    mask_cnt = fx.Int32(fx.slice(fx.rocdl.make_buffer_tensor(MaskCnt), (batch, head, None))[sparse_m])
                    mask_idx = fx.slice(fx.rocdl.make_buffer_tensor(MaskIdx), (batch, head, sparse_m, None))
                    for n_iter, state in range(fx.Int32(0), mask_cnt * fx.Int32(KV_SUB), fx.Int32(1), init=results):
                        o_state = [state[2 + c] for c in range_constexpr(DV // MFMA_M)]
                        row_max, row_sum, o_acc = compute_one_n_block(
                            sparse_n_block(mask_idx, fx.Int32(n_iter)), q_frags, q_pos, state[0], state[1], o_state
                        )
                        results = yield [row_max, row_sum] + o_acc
                    if has_full_blocks:
                        full_cnt = fx.Int32(fx.slice(fx.rocdl.make_buffer_tensor(FullCnt), (batch, head, None))[sparse_m])
                        full_idx = fx.slice(fx.rocdl.make_buffer_tensor(FullIdx), (batch, head, sparse_m, None))
                        for n_iter, state in range(fx.Int32(0), full_cnt * fx.Int32(KV_SUB), fx.Int32(1), init=results):
                            o_state = [state[2 + c] for c in range_constexpr(DV // MFMA_M)]
                            row_max, row_sum, o_acc = compute_one_n_block(
                                sparse_n_block(full_idx, fx.Int32(n_iter)), q_frags, q_pos, state[0], state[1], o_state,
                                apply_mask_mod=False,
                            )
                            results = yield [row_max, row_sum] + o_acc
                else:
                    # Interior KV blocks (every key kept for every query of the block) skip the
                    # per-element mask; the masked blocks before and after them keep it. With a
                    # mask_mod every block is masked.
                    n_lo, n_hi = block_info.get_n_block_interior(
                        seqlen, m_block, window_left, window_right, n_block_min, n_block_max
                    ) if mask_mod is None else (n_block_max, n_block_max)
                    if DB:
                        # Prologue: the first tile into buffer 0 (the condition is uniform per CTA).
                        if n_block_min < n_block_max:
                            stage_kv(n_block_min * fx.Int32(TN))
                            fx.gpu.barrier()

                    def step(n, state, masked):
                        o_state = [state[2 + c] for c in range_constexpr(DV // MFMA_M)]
                        if_db = compute_one_n_block_db(n, n_block_min, n_block_max, q_frags, q_pos, state[0], state[1],
                                                       o_state, masked) if DB else None
                        return if_db if DB else compute_one_n_block(n, q_frags, q_pos, state[0], state[1], o_state,
                                                                    masked=masked)

                    for n_iter, state in range(n_block_min, n_lo, fx.Int32(1), init=init):
                        row_max, row_sum, o_acc = step(fx.Int32(n_iter), state, True)
                        results = yield [row_max, row_sum] + o_acc
                    for n_iter, state in range(n_lo, n_hi, fx.Int32(1), init=results):
                        row_max, row_sum, o_acc = step(fx.Int32(n_iter), state, False)
                        results = yield [row_max, row_sum] + o_acc
                    for n_iter, state in range(n_hi, n_block_max, fx.Int32(1), init=results):
                        row_max, row_sum, o_acc = step(fx.Int32(n_iter), state, True)
                        results = yield [row_max, row_sum] + o_acc

                # ---- Epilogue: normalize, store O (4 contiguous elements per group) and LSE ----
                inv_sum, lse = Softmax.finalize(
                    fx.Float32(results[0]), fx.Float32(results[1]), _sink_log2(has_sink, Sink, head)
                )
                o_scale = inv_sum * v_descale if fp8 else inv_sum
                inv_vec = Vec.from_elements([o_scale], fx.Float32).broadcast_to(16)
                if q_pos < seqlen.seqlen_q:
                    o_packs = fx.logical_divide(out_row(q_row), fx.make_layout(4, 1))
                    for c in range_constexpr(DV // MFMA_M):
                        o = Vec(results[2 + c]) * inv_vec
                        for g in range_constexpr(4):
                            d_start = fx.Int32(c * MFMA_M) + fx.Int32(fx.get_scalar(c_rows[4 * g]))
                            reg = fx.make_rmem_tensor(4, o_elem)
                            reg.store(Vec.from_elements([o[4 * g + i] for i in range_constexpr(4)], fx.Float32).to(o_elem))
                            fx.copy(copy_o, reg, fx.slice(o_packs, (None, d_start // fx.Int32(4))))
                    if lane < fx.Int32(MFMA_M):
                        lse_row = lse_head_row()
                        lse_row[seqlen.offset_q + q_row if varlen_q else q_row] = lse

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            O: fx.Tensor,  # noqa: E741
            LSE: fx.Tensor,
            CuSeqlensQ: fx.Tensor,
            CuSeqlensK: fx.Tensor,
            SeqUsedQ: fx.Tensor,
            SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor,
            Aux1: fx.Tensor,
            Aux2: fx.Tensor,
            Aux3: fx.Tensor,
            Sink: fx.Tensor,
            MaskCnt: fx.Tensor,
            MaskIdx: fx.Tensor,
            FullCnt: fx.Tensor,
            FullIdx: fx.Tensor,
            PageTable: fx.Tensor,
            QDescale: fx.Tensor,
            KDescale: fx.Tensor,
            VDescale: fx.Tensor,
            page_size: fx.Int32,
            num_splits: fx.Int32,
            seqlen_q_static: fx.Int32,
            seqlen_k_static: fx.Int32,
            num_m_blocks: fx.Int32,
            num_heads: fx.Int32,
            batch: fx.Int32,
            softmax_scale: fx.Float32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            kernel(
                Q, K, V, O, LSE, CuSeqlensQ, CuSeqlensK, SeqUsedQ, SeqUsedK,
                Aux0, Aux1, Aux2, Aux3, Sink, MaskCnt, MaskIdx, FullCnt, FullIdx, PageTable, QDescale, KDescale, VDescale, page_size, num_splits,
                seqlen_q_static, seqlen_k_static, softmax_scale, window_left, window_right,
                # Double buffering keeps a tile's loads in flight across the compute; hold the
                # kernel to 256 registers (2 waves / SIMD).
                value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": f"{NT},{NT}"} if DB else {},
            ).launch(grid=(num_m_blocks, num_heads, batch * num_splits), block=(NT, 1, 1), stream=stream)

        return launch

    def _run_decode(self, q, k, v, softmax_scale, out, lse, window_left, seqused_k, aux_tensors, learnable_sink,
                    page_table, block_sparse_tensors, stream):
        """One query token per sequence on a paged cache: FlashAttentionDecode (GQA packed,
        256-token partitions). Causal needs no mask here (the keys end at the query)."""
        import torch

        from primus_turbo.flydsl.flex_attention.flash_decode import FlashAttentionDecode

        if self._decode is None:
            self._decode = FlashAttentionDecode(
                dtype=self.dtype, head_dim=self.head_dim, qhead_per_kvhead=self.qhead_per_kvhead,
                is_local=self.is_local, score_mod=self.score_mod, mask_mod=self.mask_mod,
                num_aux_tensors=self.num_aux_tensors, has_learnable_sink=self.has_learnable_sink,
                block_sparse=self.block_sparse, has_full_blocks=self.has_full_blocks,
                sparse_block_size=self.tile_n * self.kv_subtile,
            )
        B, _, Hq, D = q.shape
        if seqused_k is None:
            seqused_k = torch.full((B,), page_table.shape[1] * k.shape[1], dtype=torch.int32, device=q.device)
        o3 = None if out is None else out.view(B, Hq, D)
        l2 = None if lse is None else lse.view(B, Hq)
        o3, l2 = self._decode(q.view(B, Hq, D), k, v, page_table, seqused_k, softmax_scale, out=o3, lse=l2,
                              window_left=window_left if self.is_local else -1, aux_tensors=aux_tensors,
                              learnable_sink=learnable_sink, block_sparse_tensors=self._decode_sparse(block_sparse_tensors),
                              stream=stream)
        return o3.view(B, 1, Hq, D), l2.view(B, Hq, 1)

    def _decode_sparse(self, bst):
        """The forward's block-sparse lists for the decode kernel (KV block in keys)."""
        if bst is None:
            return None
        if (bst.full_block_cnt is not None) != self.has_full_blocks:
            raise ValueError("full_block_cnt / full_block_idx do not match how this kernel was built")
        if subtile_factors(bst.block_size, self.tile_m, self.tile_n) != (self.q_subtile, self.kv_subtile):
            raise ValueError("block_sparse_tensors.block_size does not match how this kernel was built")
        return bst._replace(block_size=None)

    def __call__(
        self,
        q,
        k,
        v,
        softmax_scale,
        out=None,
        lse=None,
        window_left=-1,
        window_right=-1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        aux_tensors=(),
        learnable_sink=None,
        block_sparse_tensors=None,
        page_table=None,
        num_splits=1,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        stream=None,
    ):
        """Run the kernel; returns ``(out, lse)``. Varlen inputs need ``max_seqlen_q`` to
        size the grid. Absent optional tensors are passed as unused placeholders.

        Paged KV (FA4): k / v are ``[num_pages, page_size, Hk, D]`` and ``page_table``
        ``[B, max_pages]`` int32; sequence lengths come from ``seqused_k``, else
        ``max_pages * page_size``. ``num_splits < 1`` picks the split count with FA4's
        heuristic (needs ``split_kv=True``)."""
        import torch

        if (page_table is not None) != self.paged_kv:
            raise ValueError("page_table does not match how this kernel was built")
        if self._decode_ok and q.dim() == 4 and q.shape[1] == 1:
            if (block_sparse_tensors is not None) != self.block_sparse:
                raise ValueError("block_sparse_tensors does not match how this kernel was built")
            return self._run_decode(q, k, v, softmax_scale, out, lse, window_left, seqused_k, aux_tensors,
                                    learnable_sink, page_table, block_sparse_tensors, stream)
        if num_splits != 1 and not self.split_kv:
            raise ValueError("num_splits != 1 needs a kernel built with split_kv=True")
        descales = ()
        out_dtype = torch.bfloat16 if self.fp8 or self.dequant_e5m2 else q.dtype  # FA4: fp8 writes bf16
        if self.dequant_e5m2:
            q, k, v = dequantize_fp8(
                q, k, v, (q_descale, k_descale, v_descale), self.qhead_per_kvhead, cu_seqlens_q, cu_seqlens_k
            )
        elif self.fp8:
            q, k, v, descales = to_native_fp8(q, k, v, (q_descale, k_descale, v_descale), self.dtype, self.fp8_ocp)
        elif any(d is not None for d in (q_descale, k_descale, v_descale)):
            raise ValueError("q/k/v_descale are for fp8 inputs")
        if (cu_seqlens_q is not None) != self.varlen_q or (cu_seqlens_k is not None) != self.varlen_k:
            raise ValueError("cu_seqlens_q / cu_seqlens_k do not match how this kernel was built")
        if (seqused_q is not None) != self.has_seqused_q or (seqused_k is not None) != self.has_seqused_k:
            raise ValueError("seqused_q / seqused_k do not match how this kernel was built")
        aux_tensors = tuple(aux_tensors or ())
        if len(aux_tensors) != self.num_aux_tensors:
            raise ValueError(f"kernel built for {self.num_aux_tensors} aux tensors, got {len(aux_tensors)}")
        if (learnable_sink is not None) != self.has_learnable_sink:
            raise ValueError("learnable_sink does not match how this kernel was built")
        if (block_sparse_tensors is not None) != self.block_sparse:
            raise ValueError("block_sparse_tensors does not match how this kernel was built")
        if self.block_sparse:
            if (block_sparse_tensors.full_block_cnt is not None) != self.has_full_blocks:
                raise ValueError("full_block_cnt / full_block_idx do not match how this kernel was built")
            if subtile_factors(block_sparse_tensors.block_size, self.tile_m, self.tile_n) != (
                self.q_subtile, self.kv_subtile,
            ):
                raise ValueError("block_sparse_tensors.block_size does not match how this kernel was built")
        if self.varlen_q:
            total_q, num_heads, _ = q.shape
            batch = cu_seqlens_q.numel() - 1
            seqlen_q = int(max_seqlen_q)
            out_shape, lse_shape = (total_q, num_heads, self.head_dim_v), (num_heads, total_q)
        else:
            batch, seqlen_q, num_heads, _ = q.shape
            out_shape, lse_shape = (batch, seqlen_q, num_heads, self.head_dim_v), (batch, num_heads, seqlen_q)
        seqlen_k = 0 if self.varlen_k else k.shape[1]
        page_size = 1
        if self.paged_kv:
            if page_table.dtype != torch.int32 or page_table.shape[0] != batch:
                raise ValueError("page_table must be int32 [batch, max_pages]")
            page_size = k.shape[1]
            seqlen_k = page_table.shape[1] * page_size
        num_m_blocks = max((seqlen_q + self.tile_m - 1) // self.tile_m, 1)
        if num_splits < 1:
            # FA4's heuristic: enough splits to fill the CUs, none for short KV. Without
            # pack_gqa every query head is its own CTA.
            num_n_blocks = -(-(seqlen_k if seqlen_k else k.shape[0] // max(batch, 1)) // self.tile_n)
            num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
            total_m_blocks = max(batch * num_heads * num_m_blocks, 1)
            num_splits = 1 if num_n_blocks <= 4 else max(1, min(num_cus // total_m_blocks, 128, num_n_blocks))
        final_out, final_lse = out, lse
        if self.split_kv:
            out = torch.empty((num_splits, *out_shape), dtype=torch.float32, device=q.device)
            lse = torch.empty((num_splits, *lse_shape), dtype=torch.float32, device=q.device)
        if out is None:
            out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        if lse is None:
            lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
        placeholder = lse
        if learnable_sink is not None:
            learnable_sink = learnable_sink.to(torch.float32).contiguous()
        aux_slots = aux_tensors + (placeholder,) * (MAX_AUX_TENSORS - len(aux_tensors))
        sparse_slots = (placeholder,) * 4
        if self.block_sparse:
            bst = broadcast_heads(block_sparse_tensors, num_heads)
            sparse_slots = tuple(
                t if t is not None else placeholder
                for t in (bst.mask_block_cnt, bst.mask_block_idx, bst.full_block_cnt, bst.full_block_idx)
            )
        stream = torch.cuda.current_stream() if stream is None else stream
        self.launcher(
            q, k, v, out, lse,
            cu_seqlens_q if cu_seqlens_q is not None else placeholder,
            cu_seqlens_k if cu_seqlens_k is not None else placeholder,
            seqused_q if seqused_q is not None else placeholder,
            seqused_k if seqused_k is not None else placeholder,
            *aux_slots,
            learnable_sink if learnable_sink is not None else placeholder,
            *sparse_slots,
            page_table if page_table is not None else placeholder,
            *(descales or (placeholder,) * 3), page_size, int(num_splits),
            seqlen_q, seqlen_k, num_m_blocks, num_heads, batch,
            float(softmax_scale), int(window_left), int(window_right),
            stream=stream,
        )
        if not self.split_kv:
            return out, lse
        # Partial LSE is [splits, *lse_shape]; combine takes FA4's [splits, ..., S, H].
        out, lse_c = flash_attn_combine(out, lse.transpose(-1, -2), out=final_out, out_dtype=out_dtype)
        lse_c = lse_c.transpose(-1, -2)
        if final_lse is None:
            return out, lse_c.contiguous()
        final_lse.copy_(lse_c)
        return out, final_lse
