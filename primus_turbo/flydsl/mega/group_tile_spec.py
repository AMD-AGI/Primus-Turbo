###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Group (per-expert) grouped (per-expert) FP8 tile spec — the shared abstraction over
``gemm_tile_spec`` reused by the mega-MoE fusion kernels (dispatch+GEMM and
GEMM+combine) and the standalone grouped GEMM.

It is a CONFORMANT, IMMUTABLE ``TileSpec``: ``GroupFp8TileSpec`` subclasses
``DenseFp8TileSpec`` and reuses every stock per-stage hook + the shared
``run_uniform_k_pipeline``. The ONLY custom code is two hook overrides:

  * ``schedule``     -- the fusion kernels' tile-id map: the LINEAR no-sync
                        row-major map, offset past the front comm/comb blocks
                        (``num_comm_blocks``, baked at construction = config).
  * ``base_offsets`` -- the per-expert B slab. The flat weight is [G*N,K] (NT) or
                        [G*K,N] (NN/TN); in fp8-element units the expert-g offset
                        is the SAME ``g * K * c_n`` for all three layouts, added to
                        the stock B base (the dense layout policy handles the
                        row-major-vs-K-strided difference). ``g = tile_to_group[block_m]``.

The stock ``gemm_tile_spec`` external interface is left UNCHANGED. Rather than add
a new method, this spec OVERRIDES the existing ``emit`` hook with extra OPTIONAL
``group_res`` / ``lds`` params (LSP-compatible -- callers using the base
``TileSpec.emit`` signature still work), re-running the shared template so it can
thread the runtime ``group_res`` (the ``tile_to_group`` buffer -> ``base_offsets``,
no mutable spec state) and the caller-allocated ``lds``.

Like ``DenseFp8TileSpec``, the spec owns its launcher via ``build_launch`` -- here
the standalone grouped GEMM-only launcher (no comm fusion). The fused dispatch /
combine kernels are built separately (they add a comm role around the same
``spec.emit``)."""

import functools
from typing import Protocol, runtime_checkable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.buffer_ops import (
    _unwrap_value,
    buffer_load,
    create_buffer_resource,
    create_llvm_ptr,
    extract_base_index,
    get_element_ptr,
)

from primus_turbo.flydsl.gemm.gemm_tile_spec import (
    DenseFp8TileSpec, LinearNoSyncScheduler, TileSpec, _emit_if_then,
)
from primus_turbo.flydsl.utils.gemm_helper import ceildiv, make_value_attrs


# TN's inplace MFMA needs a pinned AGPR budget (the dense TN forces 128).
_LAYOUT_AGPR = {"nt": 0, "nn": 0, "tn": 128}


# ── shared scoreboard / cross-rank prims (the comm handshake; gemm_helper has none) ──
# Both fusion kernels (dispatch push+signal, combine signal+push) use the SAME
# release/acquire fences + relaxed atomic add / load. Vendored once here so the
# kernels reuse them; only ``dispatch_tile`` / ``combine_tile`` stay kernel-specific.
_ORD = _llvm.AtomicOrdering
_I4 = 4  # int32 byte stride / atomic alignment


def _elem_ptr_i32(tensor, idx):
    """LLVM ptr to int32 element ``tensor[idx]``."""
    base = create_llvm_ptr(extract_base_index(tensor, address_space=1), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _elem_ptr_i32_from_addr(addr_i64, idx):
    """LLVM ptr to int32 element at ``(addr_i64)[idx]`` (runtime peer base addr)."""
    base = create_llvm_ptr(_unwrap_value(addr_i64), 1)
    byte_off = _unwrap_value(idx * fx.Int32(_I4))
    return get_element_ptr(base, byte_offset=byte_off, elem_type=fx.T.i8())


def _fence_release():
    """System-scope release fence (L2 writeback) so peers/consumers see prior stores."""
    _llvm.fence(_ORD.release, syncscope=None)


def _fence_acquire():
    """System-scope acquire fence (L1 invalidate) so this wave reads fresh data."""
    _llvm.fence(_ORD.acquire, syncscope=None)


def _atomic_add(tensor, idx, val):
    """Relaxed atomic int32 add into local ``tensor[idx]`` (a scoreboard signal)."""
    ptr = _elem_ptr_i32(tensor, idx)
    _llvm.atomicrmw(_llvm.AtomicBinOp.add, ptr, _unwrap_value(val),
                    _ORD.monotonic, syncscope=None, alignment=_I4)


def _atomic_add_addr(addr_i64, idx, val):
    """Relaxed atomic int32 add to ``(addr_i64)[idx]`` (peer scoreboard, system scope)."""
    ptr = _elem_ptr_i32_from_addr(addr_i64, idx)
    _llvm.atomicrmw(_llvm.AtomicBinOp.add, ptr, _unwrap_value(val),
                    _ORD.monotonic, syncscope=None, alignment=_I4)


def _ld_relaxed(tensor, idx):
    """Relaxed atomic int32 load of ``tensor[idx]`` for spin polling."""
    ptr = _elem_ptr_i32(tensor, idx)
    op = _llvm.LoadOp(fx.T.i32(), ptr, ordering=_ORD.monotonic, syncscope=None, alignment=_I4)
    return fx.arith.ArithValue(op.result, signed=True)


# ──────────────────────────────────────────────────────────────────────
# GroupTileSpec contract -- aligned with (and extending) ``gemm_tile_spec``'s
# ``TileSpec``: SAME member set (launcher surface + per-stage hooks incl. ``emit``
# and ``build_launch``; NO new method names). The grouping is expressed only as:
#   * ``num_comm_blocks``  -- construction-config attr: the front comm/comb blocks
#                             the tile-id map skips (read by ``schedule``).
#   * ``base_offsets``     -- the existing hook, OVERRIDDEN with an extra OPTIONAL
#                             ``group_res`` (tile_to_group) for the per-expert B slab.
#   * ``emit``             -- the existing hook, OVERRIDDEN with extra OPTIONAL
#                             ``group_res`` / ``lds`` (LSP-compatible).
# ``build_launch`` (overridden to build the grouped launcher), ``schedule``,
# ``make_buffers`` / ... keep their TileSpec signatures.
# ──────────────────────────────────────────────────────────────────────
@runtime_checkable
class GroupTileSpec(TileSpec, Protocol):
    """Group grouped-GEMM tile spec contract: a stock ``TileSpec`` (same members)
    plus the ``num_comm_blocks`` config attr. The per-expert B slab flows through
    the standard ``TileSpec.emit(..., group_base=...)`` SCALAR seam (the caller
    looks up ``g_idx`` and passes ``g_idx * K * c_n``), so no hook signature needs
    extending -- grouping = a custom ``schedule`` (tile-id map) + that scalar."""

    num_comm_blocks: int  # front comm/comb blocks the tile-id map skips


class GroupFp8TileSpec(DenseFp8TileSpec):
    """Immutable mega grouped FP8 tile spec (NT/NN/TN). Subclasses
    ``DenseFp8TileSpec``, swaps in the LINEAR no-sync ``LinearNoSyncScheduler``
    (held, not an override) and overrides only ``build_launch`` (the grouped
    GEMM-only launcher); reuses EVERY per-stage hook unchanged. The per-expert B
    slab rides the stock ``emit(..., group_base=...)`` scalar seam (the caller
    looks up ``g_idx`` and passes ``g_idx * K * c_n``), so neither ``emit`` nor
    ``base_offsets`` nor ``schedule`` is overridden -- the shared uniform-K
    template + tile-map live in ONE place. Carries no runtime state. Satisfies the
    ``GroupTileSpec`` protocol. ``num_comm_blocks`` is construction config (front
    comm/comb blocks the tile-id map skips)."""

    def __init__(self, *, num_comm_blocks, **kw):
        super().__init__(**kw)
        self.num_comm_blocks = num_comm_blocks
        self.kernel_name = "grouped_" + self.layout
        # discriminate the JIT cache: distinct comm-split -> distinct tile-id map
        self.cache_tag = self.cache_tag + (num_comm_blocks,)
        # swap the dense XCD scheduler for the fused LINEAR no-sync tile-id map
        # (held, not an override of ``schedule``).
        self.scheduler = LinearNoSyncScheduler(num_comm_blocks=num_comm_blocks)

    @staticmethod
    def group_base(group_res, block_m, K, c_n):
        """The per-expert B slab scalar for ``emit(group_base=...)``: ``g * K * c_n``
        (fp8 elements), g = tile_to_group[block_m]. Identical slab for NT (flat
        B[G*N,K]: g*N*K) and NN/TN (flat B[G*K,N]: g*K*N) since N==c_n; the stock
        per-layout B base / b_k_mult absorb the row-major-vs-K-strided difference.
        Called caller-side (kernel body) where block_m is already known."""
        g_idx = buffer_load(group_res, block_m, vec_width=1, dtype=fx.T.i32())
        return g_idx * fx.Int32(K) * c_n

    # ── build_launch: the standalone grouped GEMM-only launcher (no comm) ───
    def build_launch(self, *, waves_per_eu=2, agpr_alloc=None):
        """Build the ``@flyc.kernel`` + ``@flyc.jit`` grouped GEMM-only launcher
        (the mega analog of ``DenseFp8TileSpec.build_launch``). The kernel reads the
        runtime ``TILE_TO_GROUP`` + ``NUM_TILE_BLOCKS`` (no-sync over-launch
        self-bound), lays out tiles with this spec's LINEAR tile-id map, and runs
        the grouped ``emit``. ``agpr_alloc=None`` -> the layout's AGPR budget (TN=128).
        A/C are flat byte/elem views; A is pool[M,K] (NT/NN) or [K,M] (TN), B is the
        flat per-expert weight, C is out[M,N] bf16; c_m = pool rows, c_n = N."""
        spec = self
        tag = self.cache_tag
        BLOCK_M, BLOCK_N = self.BLOCK_M, self.BLOCK_N
        lds_top = self.layout != "nn"   # NT/TN: top alloc; NN: alloc inside emit (under the guard)
        if agpr_alloc is None:
            agpr_alloc = _LAYOUT_AGPR[self.layout]

        # Same shape as DenseFp8TileSpec.build_launch: functional flyc.kernel form +
        # kernel.__name__ = self.kernel_name, dynamic ``if`` via the shared
        # ``_emit_if_then`` (the functional form skips the AST if-rewrite). Extra args
        # vs the dense kernel are the grouped runtime inputs (the spec owns its I/O):
        # TILE_TO_GROUP (-> group_res -> base_offsets) and NUM_TILE_BLOCKS (no-sync
        # self-bound). The guard is always-true for the standalone (exact grid) but
        # kept: NN needs its LDS allocated inside this conditional.
        def kernel(A, B, C, A_scale, B_scale, TILE_TO_GROUP: fx.Tensor,
                   NUM_TILE_BLOCKS: fx.Tensor, c_n: fx.Int32):
            _ = tag  # JIT cache-key discriminator; emits no IR
            block_index, _b, _c = fx.block_idx
            n_blocks = ceildiv(c_n, BLOCK_N)
            group_res = create_buffer_resource(TILE_TO_GROUP, max_size=True)
            ntb = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)
            real_tiles = buffer_load(ntb, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
            lds = fx.SharedAllocator().allocate(spec.shared_storage).peek() if lds_top else None
            block_m = block_index // n_blocks  # standalone: num_comm_blocks == 0

            def _emit_one():
                c_m_real = real_tiles * fx.Int32(BLOCK_M)
                # per-expert B slab scalar (caller-side lookup) -> stock emit seam
                gbase = spec.group_base(group_res, block_m, spec.K, c_n)
                spec.emit(A=A, B=B, C=C, A_scale=A_scale, B_scale=B_scale,
                          c_m=c_m_real, c_n=c_n, lds=lds, group_base=gbase)

            _emit_if_then(block_m < real_tiles, _emit_one)

        kernel.__name__ = self.kernel_name
        kernel.__qualname__ = kernel.__name__
        kernel = flyc.kernel(kernel, known_block_size=[512, 1, 1])

        @flyc.jit
        def launch(A, B, C, A_scale, B_scale, TILE_TO_GROUP, NUM_TILE_BLOCKS,
                   c_m: int, c_n: int, stream: fx.Stream = fx.Stream(None)):
            grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
            kernel(A, B, C, A_scale, B_scale, TILE_TO_GROUP, NUM_TILE_BLOCKS, c_n,
                   value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
                grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

        return launch


@functools.lru_cache(maxsize=256)
def make_group_tile_spec(*, layout, K, BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3,
                        num_comm_blocks=0, vmcnt_hint=2, out_fp16=False, act=None):
    """Host-side group tile-spec factory (cached). One ``GroupFp8TileSpec`` for all
    layouts (the dense ``layout`` policy is the only per-layout difference);
    GROUP_M/num_xcd/group_n are irrelevant (schedule over ridden) so fixed; TN uses a
    deeper tr8 drain hint. ``act`` = optional epilogue activation (e.g. "relu")."""
    vh = 3 if layout == "tn" else vmcnt_hint
    return GroupFp8TileSpec(num_comm_blocks=num_comm_blocks, layout=layout, K=K,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_M=1, num_xcd=1,
                           group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
                           b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=out_fp16, act=act)


@functools.lru_cache(maxsize=256)
def compile_grouped_gemm(layout, K, BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3,
                         waves_per_eu=2, agpr_alloc=None, act=None):
    """Build & cache the standalone grouped GEMM-only launcher (the mega analog of
    ``compile_dense_nt_tiled``): make the spec, return ``spec.build_launch``.
    ``act`` = optional epilogue activation (e.g. "relu")."""
    spec = make_group_tile_spec(layout=layout, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                               nt_vmcnt=nt_vmcnt, num_comm_blocks=0, act=act)
    return spec.build_launch(waves_per_eu=waves_per_eu, agpr_alloc=agpr_alloc)


# ── shared host-side tensor helpers ───────────────────────────────────────────
def as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    """Zero-copy flat int8 byte view (fp8 -> int8)."""
    if t.element_size() == 1 and t.dtype != torch.int8:
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def scalar_f32(scale: torch.Tensor, device) -> torch.Tensor:
    return scale.to(dtype=torch.float32, device=device).reshape(1)


def weight_layout(layout, weight_fp8):
    """Per-layout weight unpack -> (num_experts, K_contract, N_out, flat_byte_view).
    NT: W[G,N,K] flat [G*N,K]; NN/TN: W[G,K,N] flat [G*K,N]."""
    G = weight_fp8.shape[0]
    if layout == "nt":
        _, N, K = weight_fp8.shape
        flat = weight_fp8.reshape(G * N, K)
    elif layout in ("nn", "tn"):
        _, K, N = weight_fp8.shape
        flat = weight_fp8.reshape(G * K, N)
    else:
        raise NotImplementedError(f"layout {layout!r} not implemented yet")
    return G, K, N, as_i8_flat(flat)
