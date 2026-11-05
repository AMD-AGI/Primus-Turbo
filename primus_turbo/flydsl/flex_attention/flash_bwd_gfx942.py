###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deterministic flash attention backward in FlyDSL for gfx942 (MI300), after FA4's
``flash_bwd.py``. The gfx950 (MI355) backward is ``flash_bwd_gfx950.py``; the two evolve
separately.

FA4 accumulates dQ with atomics across KV blocks. To stay deterministic without
ordering semaphores, this splits the work into two kernels that each own their output:

* ``dq``: one CTA per query block, looping over KV blocks -- the forward's layout
  (S^T = K . Q^T, one query per lane) with dP^T = V . dO^T and dQ^T = K^T . dS^T.
* ``dkdv``: one CTA per KV block, looping over every query head of its GQA group and
  every query block -- the mirrored layout (S = Q . K^T, one key per lane) with
  dV^T = dO^T . P and dK^T = Q^T . dS.

Both recompute P from the forward's LSE. The preprocess (``D = rowsum(dO * O) - dLSE``,
FA4's ``D'``) and the learnable-sink gradient are small row reductions done on the host.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.flex_attention.common import LOG2E, AttentionMask, BlockInfo, SeqlenInfo
from primus_turbo.flydsl.flex_attention.torch_utils import broadcast_heads, subtile_factors

MFMA_M = 32
K_STEP = 8
MAX_AUX_TENSORS = 4


def _min(a, b):
    return (a < b).select(a, b)


# Static dispatch outside the kernel bodies (see flash_fwd._apply_score_mod).
def _mod_score(score_mod, score, batch, head, q_pos, kv_pos, mask: AttentionMask):
    if score_mod is None:
        return score
    return score_mod(score, batch, head, mask.clamp_q(q_pos), mask.clamp_kv(kv_pos), mask.seqlen, mask.aux_tensors)


def _mod_grad(score_mod_bwd, grad, score, batch, head, q_pos, kv_pos, mask: AttentionMask):
    if score_mod_bwd is None:
        return grad
    return score_mod_bwd(
        grad, score, batch, head, mask.clamp_q(q_pos), mask.clamp_kv(kv_pos), mask.seqlen, mask.aux_tensors
    )


def _store_score_grad(hook, grad_t, pad, dscore, valid, batch, head, q_pos, kv_pos, mask: AttentionMask):
    """Store what ``score_grad_hook`` returns; a declined store goes out of bounds (dropped)."""
    if hook is None:
        return
    idx, value, ok = hook(dscore, valid, batch, head, mask.clamp_q(q_pos), mask.clamp_kv(kv_pos), mask.seqlen,
                          mask.aux_tensors)
    grad_t[ok.select(idx, pad)] = fx.Float32(value)


def _lse_log2(lse):
    """LSE in base 2; an empty row's -inf becomes +inf so its P recomputes to 0."""
    return (lse == fx.Float32(float("-inf"))).select(fx.Float32(float("inf")), lse * fx.Float32(LOG2E))


class _Ops:
    """MFMA 32x32x8 atom, coordinate views and copy atoms for one lane (built in-kernel)."""

    def __init__(self, elem, lane, k_step=K_STEP):
        self.elem, self.k_step, self.kpl = elem, k_step, k_step // 2
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_M, k_step, elem))
        self.tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        self.thr_mma = self.tiled_mma.get_slice(lane)
        self.b_coords = self.thr_mma.partition_B(fx.make_view(0, fx.make_layout((MFMA_M, k_step), (1, MFMA_M))))
        self.c_rows = self.thr_mma.partition_C(fx.make_view(0, fx.make_layout((MFMA_M, MFMA_M), (1, 0))))
        self.c_cols = self.thr_mma.partition_C(fx.make_view(0, fx.make_layout((MFMA_M, MFMA_M), (0, 1))))
        self.copy_g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem)
        self.copy_g64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), elem)
        self.copy_g128_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        self.copy_s128 = fx.make_copy_atom(fx.UniversalCopy128b(), elem)
        # A / B fragments are k_step / 2 16-bit elements per lane: 64 bits, or 128 at k_step 16.
        frag128 = self.kpl * 16 == 128
        self.copy_frag_g = fx.make_copy_atom(fx.rocdl.BufferCopy128b() if frag128 else fx.rocdl.BufferCopy64b(), elem)
        self.copy_s64 = fx.make_copy_atom(fx.UniversalCopy128b() if frag128 else fx.UniversalCopy64b(), elem)
        self.copy_s16 = fx.make_copy_atom(fx.UniversalCopy16b(), elem)
        self.copy_rows = fx.make_tiled_copy_A(self.copy_s64, self.tiled_mma).get_slice(lane)
        self.copy_cols = fx.make_tiled_copy_A(self.copy_s16, self.tiled_mma).get_slice(lane)

    def row(self, j):
        """Accumulator row of element ``j`` (static per lane)."""
        return fx.Int32(fx.get_scalar(self.c_rows[j]))

    def col(self):
        return fx.Int32(fx.get_scalar(self.c_cols[0]))

    def mfma(self, a_frag, b_frag, acc):
        c_frag = fx.make_fragment_like(self.c_rows, fx.Float32)
        c_frag.store(Vec(acc))
        fx.gemm(self.tiled_mma, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load()

    def a_from_rows(self, tile):
        """A fragment of a row-major LDS tile (64-bit reads along K)."""
        frag = self.thr_mma.make_fragment_A(tile)
        fx.copy(self.copy_s64, self.copy_rows.partition_S(tile), self.copy_rows.retile(frag))
        return frag

    def a_from_cols(self, tile):
        """A fragment of a transposed LDS view (16-bit reads)."""
        frag = self.thr_mma.make_fragment_A(tile)
        fx.copy(self.copy_s16, self.copy_cols.partition_S(tile), self.copy_cols.retile(frag))
        return frag

    def b_frags_from_row(self, row, dim):
        """B fragments of one global row (this lane's column) for every K step over ``dim``."""
        k0 = fx.Int32(fx.get_scalar(self.b_coords[0])) // fx.Int32(MFMA_M)
        kpl, ks_ = self.kpl, self.k_step
        packs = fx.logical_divide(row, fx.make_layout(kpl, 1))
        frags = []
        for ks in range_constexpr(dim // ks_):
            reg = fx.make_rmem_tensor(kpl, self.elem)
            fx.copy(self.copy_frag_g, fx.slice(packs, (None, (fx.Int32(ks * ks_) + k0) // fx.Int32(kpl))), reg)
            frag = fx.make_fragment_like(self.b_coords, self.elem)
            frag.store(Vec(reg.load()))
            frags.append(frag)
        return frags

    def b_frag(self, values):
        frag = fx.make_fragment_like(self.b_coords, self.elem)
        frag.store(Vec.from_elements(values, fx.Float32).to(self.elem))
        return frag

    def store_transposed_acc(self, acc_tiles, row, scale=None, fp32=False):
        """Store accumulator tiles of shape [dim, 32 lanes] into this lane's global row
        (4 contiguous elements per group), in the element type or, with ``fp32``, as fp32."""
        packs = fx.logical_divide(row, fx.make_layout(4, 1))
        out_elem, atom = (fx.Float32, self.copy_g128_f32) if fp32 else (self.elem, self.copy_g64)
        for c in range_constexpr(len(acc_tiles)):
            vals = Vec(acc_tiles[c])
            if scale is not None:
                vals = vals * Vec.from_elements([scale], fx.Float32).broadcast_to(16)
            for g in range_constexpr(4):
                d_start = fx.Int32(c * MFMA_M) + self.row(4 * g)
                reg = fx.make_rmem_tensor(4, out_elem)
                reg.store(Vec.from_elements([vals[4 * g + i] for i in range_constexpr(4)], fx.Float32).to(out_elem))
                fx.copy(atom, reg, fx.slice(packs, (None, d_start // fx.Int32(4))))


def _swizzle_bits(dim):
    """Row bits XOR-ed into the 16-byte chunk index of a row-major ``[rows, dim]`` 16-bit
    LDS tile: unswizzled, the 32 rows of an A fragment are a multiple of 128 bytes apart
    and read one bank."""
    chunks = dim // 8
    return min(3, chunks.bit_length() - 1) if chunks & (chunks - 1) == 0 else 0


def _lds_view(ptr, shape, stride, dim):
    """View of a row-major ``[rows, dim]`` LDS tile (any shape / stride over its linear
    offsets, e.g. the transposed view), through the tile's chunk swizzle."""
    layout = fx.make_layout(shape, stride)
    bits = _swizzle_bits(dim)
    if bits:
        layout = fx.make_composed_layout(fx.static(fx.SwizzleType.get(bits, 3, dim.bit_length() - 4)), layout)
    return fx.make_view(ptr, layout)


def _load_rows(ops, tid, NT, rows, dim, row_of, lds_packs):
    """Global loads of ``rows`` x ``dim`` (row ``r`` read from ``row_of(r)``) into registers,
    with each 16-byte chunk's swizzled LDS place (see ``_lds_view``); ``_store_rows`` writes
    them. Loading every staged tile before storing any pays one load latency per stage."""
    chunks, mask = dim // 8, (1 << _swizzle_bits(dim)) - 1
    pending = []
    for it in range_constexpr(rows * dim // 8 // NT):
        linear = fx.Int32(it * NT) + tid
        row, chunk = linear // fx.Int32(chunks), linear % fx.Int32(chunks)
        src = fx.logical_divide(row_of(row), fx.make_layout(8, 1))
        reg = fx.make_rmem_tensor(8, ops.elem)
        fx.copy(ops.copy_g128, fx.slice(src, (None, chunk)), reg)
        pending.append((reg, lds_packs, row * fx.Int32(chunks) + (chunk ^ (row & fx.Int32(mask)))))
    return pending


def _store_rows(ops, pending):
    for reg, lds_packs, idx in pending:
        fx.copy(ops.copy_s128, reg, fx.slice(lds_packs, (None, idx)))


MIN_SPLIT_WORK = 128


class FlashAttentionBackward:
    """Builds the dQ and dK/dV launchers for one compile-time configuration.

    ``score_mod`` / ``score_mod_bwd`` / ``mask_mod`` follow ``mods.py``; ``mask_mod`` and
    block-sparse lists must match the forward. dQ uses the forward's (query-major)
    lists; dK/dV takes the KV-major lists from ``torch_utils.transpose``.
    """

    def __init__(
        self,
        dtype: str = "bf16",
        head_dim: int = 128,
        head_dim_v: int | None = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        varlen_q: bool = False,
        varlen_k: bool = False,
        has_seqused_q: bool = False,
        has_seqused_k: bool = False,
        score_mod=None,
        score_mod_bwd=None,
        mask_mod=None,
        num_aux_tensors: int = 0,
        block_sparse: bool = False,
        has_full_blocks: bool = False,
        sparse_block_size: tuple | None = None,
        score_grad_hook=None,
        num_waves: int = 4,
        dq_tile_n: int = 64,
        dkdv_tile_m: int = 64,
    ):
        if dtype not in ("bf16", "fp16"):
            raise ValueError(f"dtype must be bf16 or fp16, got {dtype}")
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        if head_dim % MFMA_M or head_dim_v % MFMA_M:
            raise ValueError(f"backward needs head_dim and head_dim_v multiples of {MFMA_M}")
        if is_causal and is_local:
            raise ValueError("pass causal with a left window as is_local with window_right = 0")
        if score_mod is not None and score_mod_bwd is None:
            raise ValueError("score_mod needs score_mod_bwd in the backward")
        if not 0 <= num_aux_tensors <= MAX_AUX_TENSORS:
            raise ValueError(f"at most {MAX_AUX_TENSORS} aux tensors, got {num_aux_tensors}")
        if block_sparse and (varlen_q or varlen_k):
            raise NotImplementedError("block sparsity with varlen is not supported yet")
        self.dtype, self.head_dim, self.head_dim_v = dtype, head_dim, head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal, self.is_local = is_causal, is_local
        self.varlen_q, self.varlen_k = varlen_q, varlen_k
        self.has_seqused_q, self.has_seqused_k = has_seqused_q, has_seqused_k
        self.score_mod, self.score_mod_bwd, self.mask_mod = score_mod, score_mod_bwd, mask_mod
        self.score_grad_hook = score_grad_hook
        self.mfma_k = K_STEP  # 32x32x8
        self.num_aux_tensors = num_aux_tensors
        self.block_sparse = block_sparse
        self.has_full_blocks = block_sparse and has_full_blocks
        self.num_threads = 64 * num_waves
        # dq: CTA of 32 * num_waves queries, KV tiles of dq_tile_n.
        self.dq_tile_m, self.dq_tile_n = MFMA_M * num_waves, dq_tile_n
        # dkdv: CTA of 32 * num_waves keys, query tiles of dkdv_tile_m.
        self.dkdv_tile_m, self.dkdv_tile_n = dkdv_tile_m, MFMA_M * num_waves
        self.dq_sub = subtile_factors(sparse_block_size, self.dq_tile_m, self.dq_tile_n)
        self.dkdv_sub = subtile_factors(sparse_block_size, self.dkdv_tile_m, self.dkdv_tile_n)
        for rows, dims in ((dq_tile_n, (head_dim, head_dim_v)), (dkdv_tile_m, (head_dim, head_dim_v))):
            for d in dims:
                if (rows * d // 8) % self.num_threads:
                    raise ValueError(f"tile {rows}x{d} does not split evenly over {self.num_threads} threads")
        if dkdv_tile_m > self.num_threads or dkdv_tile_m % MFMA_M:
            raise ValueError(f"dkdv_tile_m must be a multiple of {MFMA_M} and at most {self.num_threads}")
        self.dq_launcher = self._build_dq()
        self._dkdv_launchers = {}

    def _dkdv_launcher(self, split):
        if split not in self._dkdv_launchers:
            self._dkdv_launchers[split] = self._build_dkdv(split)
        return self._dkdv_launchers[split]

    # ------------------------------------------------------------------ dQ ----
    def _build_dq(self):
        D, DV, TN, TM, NT = self.head_dim, self.head_dim_v, self.dq_tile_n, self.dq_tile_m, self.num_threads
        KS = self.mfma_k
        Q_PER_KV = self.qhead_per_kvhead
        elem = fx.BFloat16 if self.dtype == "bf16" else fx.Float16
        block_info = BlockInfo(TM, TN, self.is_causal, self.is_local)
        is_causal, is_local = self.is_causal, self.is_local
        varlen_q, varlen_k = self.varlen_q, self.varlen_k
        has_seqused_q, has_seqused_k = self.has_seqused_q, self.has_seqused_k
        score_mod, score_mod_bwd, mask_mod = self.score_mod, self.score_mod_bwd, self.mask_mod
        score_grad_hook = self.score_grad_hook
        num_aux = self.num_aux_tensors
        block_sparse, has_full_blocks = self.block_sparse, self.has_full_blocks
        Q_SUB, KV_SUB = self.dq_sub

        @fx.struct
        class SharedStorage:
            k: fx.Array[elem, TN * D, 16]
            v: fx.Array[elem, TN * DV, 16]

        @flyc.kernel(known_block_size=[NT, 1, 1])
        def kernel(
            Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, dO: fx.Tensor, LSE: fx.Tensor, Dsum: fx.Tensor,
            dQ: fx.Tensor, ScoreGrad: fx.Tensor,
            CuSeqlensQ: fx.Tensor, CuSeqlensK: fx.Tensor, SeqUsedQ: fx.Tensor, SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor, Aux1: fx.Tensor, Aux2: fx.Tensor, Aux3: fx.Tensor,
            MaskCnt: fx.Tensor, MaskIdx: fx.Tensor, FullCnt: fx.Tensor, FullIdx: fx.Tensor,
            seqlen_q_static: fx.Int32, seqlen_k_static: fx.Int32, softmax_scale: fx.Float32,
            window_left: fx.Int32, window_right: fx.Int32, score_grad_pad: fx.Int32,
        ):
            tid = fx.thread_idx.x
            lane = tid % fx.Int32(64)
            wave = tid // fx.Int32(64)
            m_block = fx.Int32(fx.block_idx.x)
            head = fx.Int32(fx.block_idx.y)
            batch = fx.Int32(fx.block_idx.z)
            kv_head = head // fx.Int32(Q_PER_KV)
            ops = _Ops(elem, lane, KS)
            # The rewriter carries any kernel local whose method is called inside a
            # runtime ``if`` as branch state, so those calls go through plain names.
            lane_col, b_frags_of_row, store_acc = ops.col(), ops.b_frags_from_row, ops.store_transposed_acc

            q_t, k_t, v_t = (fx.rocdl.make_buffer_tensor(t) for t in (Q, K, V))
            do_t, dq_t = fx.rocdl.make_buffer_tensor(dO), fx.rocdl.make_buffer_tensor(dQ)
            lse_t, dsum_t = fx.rocdl.make_buffer_tensor(LSE), fx.rocdl.make_buffer_tensor(Dsum)
            # Flat fp32 gradient of the hook's target, bounds-checked: a declined store goes
            # to index score_grad_pad (its size), which the hardware drops.
            sg_t = fx.rocdl.make_buffer_tensor(
                ScoreGrad, max_size=False, num_records_bytes=fx.Int64(score_grad_pad) * fx.Int64(4)
            ) if score_grad_hook is not None else None

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            k_rows = fx.flat_divide(_lds_view(lds.k.ptr, (TN, D), (D, 1), D), (MFMA_M, KS))
            kt_cols = fx.flat_divide(_lds_view(lds.k.ptr, (D, TN), (1, D), D), (MFMA_M, KS))
            k_packs = fx.logical_divide(fx.make_view(lds.k.ptr, fx.make_layout(TN * D, 1)), fx.make_layout(8, 1))
            v_rows = fx.flat_divide(_lds_view(lds.v.ptr, (TN, DV), (DV, 1), DV), (MFMA_M, KS))
            v_packs = fx.logical_divide(fx.make_view(lds.v.ptr, fx.make_layout(TN * DV, 1)), fx.make_layout(8, 1))

            seqlen = SeqlenInfo.create(
                batch, seqlen_q_static, seqlen_k_static,
                cu_seqlens_q=fx.rocdl.make_buffer_tensor(CuSeqlensQ) if varlen_q else None,
                cu_seqlens_k=fx.rocdl.make_buffer_tensor(CuSeqlensK) if varlen_k else None,
                seqused_q=fx.rocdl.make_buffer_tensor(SeqUsedQ) if has_seqused_q else None,
                seqused_k=fx.rocdl.make_buffer_tensor(SeqUsedK) if has_seqused_k else None,
            )
            aux_tensors = tuple(fx.rocdl.make_buffer_tensor(t) for t in (Aux0, Aux1, Aux2, Aux3)[:num_aux])
            mask = AttentionMask(
                seqlen, is_causal, is_local, window_left, window_right, mask_mod, batch, head, aux_tensors
            )

            def q_side_row(t, pos):
                coord = (seqlen.offset_q + pos, head, None) if varlen_q else (batch, pos, head, None)
                return fx.slice(t, coord)

            def kv_side_row(t, pos):
                coord = (seqlen.offset_k + pos, kv_head, None) if varlen_k else (batch, pos, kv_head, None)
                return fx.slice(t, coord)

            def row_stat(t, pos):
                row = fx.slice(t, (head, None)) if varlen_q else fx.slice(t, (batch, head, None))
                return fx.Float32(row[seqlen.offset_q + pos if varlen_q else pos])

            def sparse_n_block(idx_row, j):
                return fx.Int32(idx_row[j // fx.Int32(KV_SUB)]) * fx.Int32(KV_SUB) + j % fx.Int32(KV_SUB)

            def one_n_block(n_block, q_frags, do_frags, q_pos, lse2, dsum, dq_acc, apply_mask_mod=True):
                kv_base = n_block * fx.Int32(TN)
                last_key = seqlen.seqlen_k - fx.Int32(1)
                fx.gpu.barrier()
                pending = _load_rows(ops, tid, NT, TN, D, lambda r: kv_side_row(k_t, _min(kv_base + r, last_key)), k_packs)
                pending += _load_rows(ops, tid, NT, TN, DV, lambda r: kv_side_row(v_t, _min(kv_base + r, last_key)), v_packs)
                _store_rows(ops, pending)
                fx.gpu.barrier()
                zero16 = Vec.filled(16, 0.0, fx.Float32)
                ds_all = []
                for half in range_constexpr(TN // MFMA_M):
                    s_acc, dp_acc = zero16, zero16
                    for ks in range_constexpr(D // KS):
                        s_acc = ops.mfma(ops.a_from_rows(fx.slice(k_rows, (None, None, half, ks))), q_frags[ks], s_acc)
                    for ks in range_constexpr(DV // KS):
                        dp_acc = ops.mfma(ops.a_from_rows(fx.slice(v_rows, (None, None, half, ks))), do_frags[ks], dp_acc)
                    s_acc, dp_acc = Vec(s_acc), Vec(dp_acc)
                    for j in range_constexpr(16):
                        kv_pos = kv_base + fx.Int32(half * MFMA_M) + ops.row(j)
                        scaled = s_acc[j] * softmax_scale
                        score = _mod_score(score_mod, scaled, batch, head, q_pos, kv_pos, mask)
                        keep = mask.keep(q_pos, kv_pos, apply_mask_mod=apply_mask_mod)
                        p = keep.select(fx.math.exp2(score * fx.Float32(LOG2E) - lse2), fx.Float32(0.0))
                        grad = p * (dp_acc[j] - dsum)
                        _store_score_grad(score_grad_hook, sg_t, score_grad_pad, grad,
                                          (q_pos < seqlen.seqlen_q) & (kv_pos < seqlen.seqlen_k),
                                          batch, head, q_pos, kv_pos, mask)
                        ds_all.append(_mod_grad(score_mod_bwd, grad, scaled, batch, head, q_pos, kv_pos, mask))
                for half in range_constexpr(TN // MFMA_M):
                    for g in range_constexpr(MFMA_M // KS):
                        first = half * 16 + (KS // 2) * g
                        ds_frag = ops.b_frag(ds_all[first : first + KS // 2])
                        for c in range_constexpr(D // MFMA_M):
                            step = half * (MFMA_M // KS) + g
                            a = ops.a_from_cols(fx.slice(kt_cols, (None, None, c, step)))
                            dq_acc[c] = ops.mfma(a, ds_frag, dq_acc[c])
                return dq_acc

            if m_block * fx.Int32(TM) < seqlen.seqlen_q:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, window_left, window_right)
                q_pos = m_block * fx.Int32(TM) + wave * fx.Int32(MFMA_M) + lane_col
                q_row = _min(q_pos, seqlen.seqlen_q - fx.Int32(1))
                q_frags = b_frags_of_row(q_side_row(q_t, q_row), D)
                do_frags = b_frags_of_row(q_side_row(do_t, q_row), DV)
                lse2 = _lse_log2(row_stat(lse_t, q_row))
                dsum = row_stat(dsum_t, q_row)

                init = [Vec.filled(16, 0.0, fx.Float32) for _ in range_constexpr(D // MFMA_M)]
                results = init
                if block_sparse:
                    sparse_m = m_block // fx.Int32(Q_SUB)
                    mask_cnt = fx.Int32(fx.slice(fx.rocdl.make_buffer_tensor(MaskCnt), (batch, head, None))[sparse_m])
                    mask_idx = fx.slice(fx.rocdl.make_buffer_tensor(MaskIdx), (batch, head, sparse_m, None))
                    for n_iter, state in range(fx.Int32(0), mask_cnt * fx.Int32(KV_SUB), fx.Int32(1), init=results):
                        acc = one_n_block(sparse_n_block(mask_idx, fx.Int32(n_iter)), q_frags, do_frags, q_pos, lse2,
                                          dsum, list(state))
                        results = yield acc
                    if has_full_blocks:
                        full_cnt = fx.Int32(fx.slice(fx.rocdl.make_buffer_tensor(FullCnt), (batch, head, None))[sparse_m])
                        full_idx = fx.slice(fx.rocdl.make_buffer_tensor(FullIdx), (batch, head, sparse_m, None))
                        for n_iter, state in range(fx.Int32(0), full_cnt * fx.Int32(KV_SUB), fx.Int32(1), init=results):
                            acc = one_n_block(sparse_n_block(full_idx, fx.Int32(n_iter)), q_frags, do_frags, q_pos,
                                              lse2, dsum, list(state), apply_mask_mod=False)
                            results = yield acc
                else:
                    for n_iter, state in range(n_block_min, n_block_max, fx.Int32(1), init=init):
                        acc = one_n_block(fx.Int32(n_iter), q_frags, do_frags, q_pos, lse2, dsum, list(state))
                        results = yield acc

                if q_pos < seqlen.seqlen_q:
                    store_acc(list(results), q_side_row(dq_t, q_row), softmax_scale)

        @flyc.jit
        def launch(
            Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, dO: fx.Tensor, LSE: fx.Tensor, Dsum: fx.Tensor,
            dQ: fx.Tensor, ScoreGrad: fx.Tensor,
            CuSeqlensQ: fx.Tensor, CuSeqlensK: fx.Tensor, SeqUsedQ: fx.Tensor, SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor, Aux1: fx.Tensor, Aux2: fx.Tensor, Aux3: fx.Tensor,
            MaskCnt: fx.Tensor, MaskIdx: fx.Tensor, FullCnt: fx.Tensor, FullIdx: fx.Tensor,
            seqlen_q_static: fx.Int32, seqlen_k_static: fx.Int32,
            num_m_blocks: fx.Int32, num_heads: fx.Int32, batch: fx.Int32,
            softmax_scale: fx.Float32, window_left: fx.Int32, window_right: fx.Int32, score_grad_pad: fx.Int32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            kernel(
                Q, K, V, dO, LSE, Dsum, dQ, ScoreGrad, CuSeqlensQ, CuSeqlensK, SeqUsedQ, SeqUsedK,
                Aux0, Aux1, Aux2, Aux3, MaskCnt, MaskIdx, FullCnt, FullIdx,
                seqlen_q_static, seqlen_k_static, softmax_scale, window_left, window_right, score_grad_pad,
            ).launch(grid=(num_m_blocks, num_heads, batch), block=(NT, 1, 1), stream=stream)

        return launch

    # --------------------------------------------------------------- dK / dV ----
    def _build_dkdv(self, split=1):
        """dK / dV, one CTA per (key tile, KV head x ``split``, batch). With ``split`` > 1
        each CTA sums G / split query heads of the GQA group and writes fp32 partials
        ``[split, ...]`` that the host adds in a fixed order (deterministic)."""
        D, DV, TM, TN, NT = self.head_dim, self.head_dim_v, self.dkdv_tile_m, self.dkdv_tile_n, self.num_threads
        KS = self.mfma_k
        G = self.qhead_per_kvhead
        HPS = G // split  # query heads per CTA
        partial = split > 1
        elem = fx.BFloat16 if self.dtype == "bf16" else fx.Float16
        block_info = BlockInfo(TM, TN, self.is_causal, self.is_local)
        is_causal, is_local = self.is_causal, self.is_local
        varlen_q, varlen_k = self.varlen_q, self.varlen_k
        has_seqused_q, has_seqused_k = self.has_seqused_q, self.has_seqused_k
        score_mod, score_mod_bwd, mask_mod = self.score_mod, self.score_mod_bwd, self.mask_mod
        num_aux = self.num_aux_tensors
        block_sparse, has_full_blocks = self.block_sparse, self.has_full_blocks
        Q_SUB, KV_SUB = self.dkdv_sub

        @fx.struct
        class SharedStorage:
            q: fx.Array[elem, TM * D, 16]
            do: fx.Array[elem, TM * DV, 16]
            lse2: fx.Array[fx.Float32, TM, 16]
            dsum: fx.Array[fx.Float32, TM, 16]

        @flyc.kernel(known_block_size=[NT, 1, 1])
        def kernel(
            Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, dO: fx.Tensor, LSE: fx.Tensor, Dsum: fx.Tensor,
            dK: fx.Tensor, dV: fx.Tensor,
            CuSeqlensQ: fx.Tensor, CuSeqlensK: fx.Tensor, SeqUsedQ: fx.Tensor, SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor, Aux1: fx.Tensor, Aux2: fx.Tensor, Aux3: fx.Tensor,
            MaskCnt: fx.Tensor, MaskIdx: fx.Tensor, FullCnt: fx.Tensor, FullIdx: fx.Tensor,
            seqlen_q_static: fx.Int32, seqlen_k_static: fx.Int32, softmax_scale: fx.Float32,
            window_left: fx.Int32, window_right: fx.Int32,
        ):
            tid = fx.thread_idx.x
            lane = tid % fx.Int32(64)
            wave = tid // fx.Int32(64)
            n_block = fx.Int32(fx.block_idx.x)
            kv_head = fx.Int32(fx.block_idx.y) // fx.Int32(split)
            split_idx = fx.Int32(fx.block_idx.y) % fx.Int32(split)
            first_head = kv_head * fx.Int32(G) + split_idx * fx.Int32(HPS)
            batch = fx.Int32(fx.block_idx.z)
            ops = _Ops(elem, lane, KS)
            # The rewriter carries any kernel local whose method is called inside a
            # runtime ``if`` as branch state, so those calls go through plain names.
            lane_col, b_frags_of_row, store_acc = ops.col(), ops.b_frags_from_row, ops.store_transposed_acc

            q_t, k_t, v_t = (fx.rocdl.make_buffer_tensor(t) for t in (Q, K, V))
            do_t, dk_t, dv_t = (fx.rocdl.make_buffer_tensor(t) for t in (dO, dK, dV))
            lse_t, dsum_t = fx.rocdl.make_buffer_tensor(LSE), fx.rocdl.make_buffer_tensor(Dsum)

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            q_rows = fx.flat_divide(_lds_view(lds.q.ptr, (TM, D), (D, 1), D), (MFMA_M, KS))
            qt_cols = fx.flat_divide(_lds_view(lds.q.ptr, (D, TM), (1, D), D), (MFMA_M, KS))
            q_packs = fx.logical_divide(fx.make_view(lds.q.ptr, fx.make_layout(TM * D, 1)), fx.make_layout(8, 1))
            do_rows = fx.flat_divide(_lds_view(lds.do.ptr, (TM, DV), (DV, 1), DV), (MFMA_M, KS))
            dot_cols = fx.flat_divide(_lds_view(lds.do.ptr, (DV, TM), (1, DV), DV), (MFMA_M, KS))
            do_packs = fx.logical_divide(fx.make_view(lds.do.ptr, fx.make_layout(TM * DV, 1)), fx.make_layout(8, 1))
            lse2_lds, dsum_lds = lds.lse2.ptr, lds.dsum.ptr

            seqlen = SeqlenInfo.create(
                batch, seqlen_q_static, seqlen_k_static,
                cu_seqlens_q=fx.rocdl.make_buffer_tensor(CuSeqlensQ) if varlen_q else None,
                cu_seqlens_k=fx.rocdl.make_buffer_tensor(CuSeqlensK) if varlen_k else None,
                seqused_q=fx.rocdl.make_buffer_tensor(SeqUsedQ) if has_seqused_q else None,
                seqused_k=fx.rocdl.make_buffer_tensor(SeqUsedK) if has_seqused_k else None,
            )
            aux_tensors = tuple(fx.rocdl.make_buffer_tensor(t) for t in (Aux0, Aux1, Aux2, Aux3)[:num_aux])

            def q_side_row(t, pos, h):
                coord = (seqlen.offset_q + pos, h, None) if varlen_q else (batch, pos, h, None)
                return fx.slice(t, coord)

            def kv_side_row(t, pos):
                coord = (seqlen.offset_k + pos, kv_head, None) if varlen_k else (batch, pos, kv_head, None)
                return fx.slice(t, coord)

            def out_row(t, pos):
                """dK / dV row, or this split's fp32 partial row ``[split, ...]``."""
                coord = (seqlen.offset_k + pos, kv_head, None) if varlen_k else (batch, pos, kv_head, None)
                return fx.slice(t, (split_idx,) + coord if partial else coord)

            def row_stat(t, pos, h):
                row = fx.slice(t, (h, None)) if varlen_q else fx.slice(t, (batch, h, None))
                return fx.Float32(row[seqlen.offset_q + pos if varlen_q else pos])

            def sparse_m_block(idx_row, j):
                return fx.Int32(idx_row[j // fx.Int32(Q_SUB)]) * fx.Int32(Q_SUB) + j % fx.Int32(Q_SUB)

            def one_m_block(m_block, head, k_frags, v_frags, kv_pos, dk_acc, dv_acc, apply_mask_mod=True):
                mask = AttentionMask(
                    seqlen, is_causal, is_local, window_left, window_right, mask_mod, batch, head, aux_tensors
                )
                q_base = m_block * fx.Int32(TM)
                last_q = seqlen.seqlen_q - fx.Int32(1)
                fx.gpu.barrier()
                pending = _load_rows(ops, tid, NT, TM, D, lambda r: q_side_row(q_t, _min(q_base + r, last_q), head), q_packs)
                pending += _load_rows(ops, tid, NT, TM, DV, lambda r: q_side_row(do_t, _min(q_base + r, last_q), head),
                                      do_packs)
                _store_rows(ops, pending)
                # Every thread writes row tid % TM: duplicates store the same value.
                r = tid % fx.Int32(TM)
                q_r = _min(q_base + r, last_q)
                lse2_lds[r] = _lse_log2(row_stat(lse_t, q_r, head))
                dsum_lds[r] = row_stat(dsum_t, q_r, head)
                fx.gpu.barrier()
                zero16 = Vec.filled(16, 0.0, fx.Float32)
                for sub in range_constexpr(TM // MFMA_M):
                    s_acc, dp_acc = zero16, zero16
                    for ks in range_constexpr(D // KS):
                        s_acc = ops.mfma(ops.a_from_rows(fx.slice(q_rows, (None, None, sub, ks))), k_frags[ks], s_acc)
                    for ks in range_constexpr(DV // KS):
                        dp_acc = ops.mfma(ops.a_from_rows(fx.slice(do_rows, (None, None, sub, ks))), v_frags[ks], dp_acc)
                    s_acc, dp_acc = Vec(s_acc), Vec(dp_acc)
                    probs, grads = [], []
                    for j in range_constexpr(16):
                        q_local = fx.Int32(sub * MFMA_M) + ops.row(j)
                        q_pos = q_base + q_local
                        scaled = s_acc[j] * softmax_scale
                        score = _mod_score(score_mod, scaled, batch, head, q_pos, kv_pos, mask)
                        keep = mask.keep(q_pos, kv_pos, apply_mask_mod=apply_mask_mod) & (q_pos < seqlen.seqlen_q)
                        lse2 = fx.Float32(lse2_lds[q_local])
                        p = keep.select(fx.math.exp2(score * fx.Float32(LOG2E) - lse2), fx.Float32(0.0))
                        grad = p * (dp_acc[j] - fx.Float32(dsum_lds[q_local]))
                        probs.append(p)
                        grads.append(_mod_grad(score_mod_bwd, grad, scaled, batch, head, q_pos, kv_pos, mask))
                    for g in range_constexpr(MFMA_M // KS):
                        p_frag = ops.b_frag(probs[(KS // 2) * g : (KS // 2) * (g + 1)])
                        ds_frag = ops.b_frag(grads[(KS // 2) * g : (KS // 2) * (g + 1)])
                        step = sub * (MFMA_M // KS) + g
                        for c in range_constexpr(DV // MFMA_M):
                            a = ops.a_from_cols(fx.slice(dot_cols, (None, None, c, step)))
                            dv_acc[c] = ops.mfma(a, p_frag, dv_acc[c])
                        for c in range_constexpr(D // MFMA_M):
                            a = ops.a_from_cols(fx.slice(qt_cols, (None, None, c, step)))
                            dk_acc[c] = ops.mfma(a, ds_frag, dk_acc[c])
                return dk_acc, dv_acc

            if n_block * fx.Int32(TN) < seqlen.seqlen_k:
                m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block, window_left, window_right)
                kv_pos = n_block * fx.Int32(TN) + wave * fx.Int32(MFMA_M) + lane_col
                kv_row = _min(kv_pos, seqlen.seqlen_k - fx.Int32(1))
                k_frags = b_frags_of_row(kv_side_row(k_t, kv_row), D)
                v_frags = b_frags_of_row(kv_side_row(v_t, kv_row), DV)

                NDK, NDV = D // MFMA_M, DV // MFMA_M
                init = [Vec.filled(16, 0.0, fx.Float32) for _ in range_constexpr(NDK + NDV)]
                results = init
                if block_sparse:
                    # KV-major lists: for each (batch, q head, sparse KV block), the query blocks.
                    sparse_n = n_block // fx.Int32(KV_SUB)
                    mask_cnt_t = fx.rocdl.make_buffer_tensor(MaskCnt)
                    mask_idx_t = fx.rocdl.make_buffer_tensor(MaskIdx)
                    for g_head in range_constexpr(HPS):
                        head = first_head + fx.Int32(g_head)
                        mask_cnt = fx.Int32(fx.slice(mask_cnt_t, (batch, head, None))[sparse_n])
                        mask_idx = fx.slice(mask_idx_t, (batch, head, sparse_n, None))
                        for m_iter, state in range(fx.Int32(0), mask_cnt * fx.Int32(Q_SUB), fx.Int32(1), init=results):
                            dk, dv = one_m_block(sparse_m_block(mask_idx, fx.Int32(m_iter)), head, k_frags, v_frags,
                                                 kv_pos, list(state[:NDK]), list(state[NDK:]))
                            results = yield dk + dv
                        if has_full_blocks:
                            full_cnt = fx.Int32(fx.slice(fx.rocdl.make_buffer_tensor(FullCnt), (batch, head, None))[sparse_n])
                            full_idx = fx.slice(fx.rocdl.make_buffer_tensor(FullIdx), (batch, head, sparse_n, None))
                            for m_iter, state in range(fx.Int32(0), full_cnt * fx.Int32(Q_SUB), fx.Int32(1), init=results):
                                dk, dv = one_m_block(sparse_m_block(full_idx, fx.Int32(m_iter)), head, k_frags, v_frags,
                                                     kv_pos, list(state[:NDK]), list(state[NDK:]), apply_mask_mod=False)
                                results = yield dk + dv
                else:
                    for g_head in range_constexpr(HPS):
                        head = first_head + fx.Int32(g_head)
                        for m_iter, state in range(m_block_min, m_block_max, fx.Int32(1), init=results):
                            dk, dv = one_m_block(fx.Int32(m_iter), head, k_frags, v_frags, kv_pos,
                                                 list(state[:NDK]), list(state[NDK:]))
                            results = yield dk + dv

                if kv_pos < seqlen.seqlen_k:
                    acc_tiles = list(results)
                    store_acc(acc_tiles[:NDK], out_row(dk_t, kv_row), softmax_scale, partial)
                    store_acc(acc_tiles[NDK:], out_row(dv_t, kv_row), None, partial)

        @flyc.jit
        def launch(
            Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, dO: fx.Tensor, LSE: fx.Tensor, Dsum: fx.Tensor,
            dK: fx.Tensor, dV: fx.Tensor,
            CuSeqlensQ: fx.Tensor, CuSeqlensK: fx.Tensor, SeqUsedQ: fx.Tensor, SeqUsedK: fx.Tensor,
            Aux0: fx.Tensor, Aux1: fx.Tensor, Aux2: fx.Tensor, Aux3: fx.Tensor,
            MaskCnt: fx.Tensor, MaskIdx: fx.Tensor, FullCnt: fx.Tensor, FullIdx: fx.Tensor,
            seqlen_q_static: fx.Int32, seqlen_k_static: fx.Int32,
            num_n_blocks: fx.Int32, num_kv_heads: fx.Int32, batch: fx.Int32,
            softmax_scale: fx.Float32, window_left: fx.Int32, window_right: fx.Int32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            kernel(
                Q, K, V, dO, LSE, Dsum, dK, dV, CuSeqlensQ, CuSeqlensK, SeqUsedQ, SeqUsedK,
                Aux0, Aux1, Aux2, Aux3, MaskCnt, MaskIdx, FullCnt, FullIdx,
                seqlen_q_static, seqlen_k_static, softmax_scale, window_left, window_right,
            ).launch(grid=(num_n_blocks, num_kv_heads, batch), block=(NT, 1, 1), stream=stream)

        return launch

    # ------------------------------------------------------------------ host ----
    def __call__(
        self,
        q, k, v, out, dout, lse, softmax_scale,
        dlse=None,
        window_left=-1,
        window_right=-1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        aux_tensors=(),
        learnable_sink=None,
        block_sparse_tensors=None,
        block_sparse_tensors_kv=None,
        dq=None,
        dk=None,
        dv=None,
        score_grad=None,
        stream=None,
    ):
        """Returns ``(dq, dk, dv, dsink)`` (``dsink`` is ``None`` without a sink).
        ``block_sparse_tensors`` are the forward's lists and ``block_sparse_tensors_kv``
        their KV-major transpose (``torch_utils.transpose``). With a ``score_grad_hook``,
        ``score_grad`` is its target's fp32 gradient buffer (zeroed by the caller); the hook's
        stores write it."""
        import torch

        if (cu_seqlens_q is not None) != self.varlen_q or (cu_seqlens_k is not None) != self.varlen_k:
            raise ValueError("cu_seqlens_q / cu_seqlens_k do not match how this kernel was built")
        if (seqused_q is not None) != self.has_seqused_q or (seqused_k is not None) != self.has_seqused_k:
            raise ValueError("seqused_q / seqused_k do not match how this kernel was built")
        aux_tensors = tuple(aux_tensors or ())
        if len(aux_tensors) != self.num_aux_tensors:
            raise ValueError(f"kernel built for {self.num_aux_tensors} aux tensors, got {len(aux_tensors)}")
        if self.block_sparse != (block_sparse_tensors is not None) or self.block_sparse != (
            block_sparse_tensors_kv is not None
        ):
            raise ValueError("block-sparse lists (both orders) do not match how this kernel was built")

        # Preprocess: D' = rowsum(dO * O) - dLSE, laid out like the LSE.
        dsum = (dout.float() * out.float()).sum(-1)  # [B, Sq, H] or [Tq, H]
        dsum = dsum.transpose(-1, -2).contiguous() if self.varlen_q else dsum.permute(0, 2, 1).contiguous()
        if dlse is not None:
            dsum = dsum - dlse.float()
        dsink = None
        if learnable_sink is not None:
            sink = learnable_sink.float().view(-1, 1) if self.varlen_q else learnable_sink.float().view(1, -1, 1)
            # The sink logit has no value vector, so dP = 0 there and dS_sink = -P_sink * D'.
            dsink = -(torch.exp(sink - lse) * dsum).sum(dim=-1)
            dsink = dsink if self.varlen_q else dsink.sum(0)
            dsink = dsink.to(learnable_sink.dtype)

        if self.varlen_q:
            num_heads = q.shape[1]
            batch = cu_seqlens_q.numel() - 1
            seqlen_q = int(max_seqlen_q)
        else:
            batch, seqlen_q, num_heads, _ = q.shape
        if self.varlen_k:
            num_kv_heads = k.shape[1]
            seqlen_k = int(max_seqlen_k)
        else:
            seqlen_k, num_kv_heads = k.shape[1], k.shape[2]
        # Rows past seqused_q / seqused_k get no gradient and are never written.
        dq, dk, dv = (
            torch.zeros_like(t) if g is None else g.zero_() for g, t in ((dq, q), (dk, k), (dv, v))
        )
        placeholder = lse
        common = (
            cu_seqlens_q if cu_seqlens_q is not None else placeholder,
            cu_seqlens_k if cu_seqlens_k is not None else placeholder,
            seqused_q if seqused_q is not None else placeholder,
            seqused_k if seqused_k is not None else placeholder,
            *(aux_tensors + (placeholder,) * (MAX_AUX_TENSORS - len(aux_tensors))),
        )

        def sparse_slots(bst):
            if bst is None:
                return (placeholder,) * 4
            bst = broadcast_heads(bst, num_heads)
            return tuple(
                t if t is not None else placeholder
                for t in (bst.mask_block_cnt, bst.mask_block_idx, bst.full_block_cnt, bst.full_block_idx)
            )

        static_q = 0 if self.varlen_q else seqlen_q
        static_k = 0 if self.varlen_k else seqlen_k
        stream = torch.cuda.current_stream() if stream is None else stream
        scalars = (float(softmax_scale), int(window_left), int(window_right))
        if (score_grad is not None) != (self.score_grad_hook is not None):
            raise ValueError("score_grad does not match how this kernel was built")
        sg_flat, sg_pad = placeholder, 0
        if score_grad is not None:
            if score_grad.dtype != torch.float32 or not score_grad.is_contiguous():
                raise ValueError("score_grad must be a contiguous fp32 tensor")
            sg_flat, sg_pad = score_grad.view(-1), score_grad.numel()
        self.dq_launcher(
            q, k, v, dout, lse, dsum, dq, sg_flat, *common, *sparse_slots(block_sparse_tensors),
            static_q, static_k, max(-(-seqlen_q // self.dq_tile_m), 1), num_heads, batch, *scalars, sg_pad,
            stream=stream,
        )
        num_n_blocks = max(-(-seqlen_k // self.dkdv_tile_n), 1)
        split = self._dkdv_split(num_n_blocks * num_kv_heads * batch, seqlen_q, int(window_left), int(window_right),
                                 q.device)
        dk_out, dv_out = dk, dv
        if split > 1:
            dk_out = torch.empty((split, *dk.shape), dtype=torch.float32, device=dk.device)
            dv_out = torch.empty((split, *dv.shape), dtype=torch.float32, device=dv.device)
            if self.varlen_k or self.has_seqused_k:  # rows past a sequence are never written
                dk_out.zero_()
                dv_out.zero_()
        self._dkdv_launcher(split)(
            q, k, v, dout, lse, dsum, dk_out, dv_out, *common, *sparse_slots(block_sparse_tensors_kv),
            static_q, static_k, num_n_blocks, num_kv_heads * split, batch, *scalars,
            stream=stream,
        )
        if split > 1:
            dk.copy_(dk_out.sum(0))
            dv.copy_(dv_out.sum(0))
        return dq, dk, dv, dsink

    def _dkdv_split(self, ctas, seqlen_q, window_left, window_right, device):
        """Power-of-two split of the GQA group across dK/dV CTAs (it fits one CTA per CU).
        Split until every CU has a CTA; keep splitting up to two per CU only while each CTA
        keeps at least ``MIN_SPLIT_WORK`` (query tile x head) iterations, since below that the
        fp32 partials cost more than the extra overlap gains."""
        import torch

        num_cus = torch.cuda.get_device_properties(device).multi_processor_count
        m_blocks = max(-(-seqlen_q // self.dkdv_tile_m), 1)
        if self.is_local and window_left >= 0:
            span = window_left + max(window_right, 0) + self.dkdv_tile_n
            m_iters = min(m_blocks, -(-span // self.dkdv_tile_m))
        elif self.is_causal:
            m_iters = max(m_blocks // 2, 1)
        else:
            m_iters = m_blocks
        work = m_iters * self.qhead_per_kvhead
        split = 1
        while self.qhead_per_kvhead % (split * 2) == 0:
            if ctas * split < num_cus or (ctas * split < 2 * num_cus and work // (split * 2) >= MIN_SPLIT_WORK):
                split *= 2
            else:
                break
        return split
