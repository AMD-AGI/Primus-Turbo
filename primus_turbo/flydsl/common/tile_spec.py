###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from collections import namedtuple
from typing import Protocol, runtime_checkable

# isort: off
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ReplaceIfWithDispatch
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl

# isort: on


def _emit_if_then(cond, then_fn):
    """Emit a dynamic ``if cond: then_fn()`` (the body-only AST rewrite's primitive)."""
    ReplaceIfWithDispatch.scf_if_dispatch(cond, then_fn)


# Per-tile global base offsets + k-step units (from ``base_offsets``).
TileOffsets = namedtuple("TileOffsets", "A0 A1 B0 B1 a_k_mult b_k_mult")

# Per-operand loaders from ``LoaderSpec.build``: gmem2lds = G2S, lds2reg = S2R.
Loaders = namedtuple("Loaders", "a_gmem2lds b_gmem2lds a_lds2reg b_lds2reg")

# Mainloop output for the epilogue. c00..c11 = the 2x2 quadrants (BLOCK//2 split).
AccumTile = namedtuple("AccumTile", "c00 c01 c10 c11 block_m block_n wave_m wave_n")

# Compile-time shape descriptor: all derived scalars, built once by ``derive_geometry``.
TileGeometry = namedtuple(
    "TileGeometry",
    "BLOCK_M BLOCK_N BLOCK_K K "
    "WAVES_M WAVES_N "  # wave grid
    "N_TILES_A N_TILES_B N_ACCUMS "  # MMA tiling
    "LDS_BLOCK_M LDS_BLOCK_N "  # LDS half
    "N_LDS_STEPS_A N_LDS_STEPS_B N_LDS_ROUNDS chunk_stride a_lds_size b_lds_size "  # layout LDS
    "K_ITERS K_TAIL",  # K-tail
)

# Per-layout LDS contribution to TileGeometry (from ``lds_geometry``).
LdsGeometry = namedtuple("LdsGeometry", "N_LDS_STEPS_A N_LDS_STEPS_B chunk_stride a_lds_size b_lds_size")


BLOCK_K = 128
INST_M = INST_N = 16  # MFMA 16x16x128 instruction M/N
WAVEFRONT = 64  # gfx950 wavefront size
QUAD = 2  # 2x2 LDS quadrant split (BLOCK//2)
HALVES = 2  # M/N halves per operand ring
# TN bank-spread LDS chunk stride (1024+32) -- removes the transpose-read bank conflict.
_LDS_CS = 1056
# gfx950 per-workgroup LDS limit.
_LDS_LIMIT_BYTES = 160 * 1024


def _add_group_base(off, group_base):
    """Add the per-expert B slab to a B offset (dense passes 0 -> no-op, identical IR)."""
    if isinstance(group_base, int) and group_base == 0:
        return off
    return off + group_base


# TileSpec contract: shared emit surface + held policies (swap a policy to customize).
@runtime_checkable
class TileSpec(Protocol):
    """Tile spec contract: shared emit surface + held sub-specs (swap to customize)."""

    cache_tag: tuple  # JIT cache-key discriminator
    kernel_name: str  # emitted symbol name
    block_tile: tuple  # (BLOCK_M, BLOCK_N, BLOCK_K) threadblock tile
    # (WARP_M, WARP_N, WARP_K) per-wave tile -> wave grid = quadrant/WARP
    warp_tile: tuple
    shared_storage: type  # @fx.struct LDS class

    # held sub-specs (each a swappable policy)
    layout_spec: "LayoutSpec"  # operand storage: base_offsets / global_swizzle / transpose
    dtype_spec: "DtypeSpec"  # numeric format + typed buffers + operand loaders
    mma_spec: "MmaSpec"  # the MFMA instruction + its build
    scheduler_spec: "SchedulerSpec"  # tile-id map + launch grid
    # PerTensorEpilogue (C store: scale + activation)
    epilogue_spec: object
    # InterleavedKPipeline (mainloop schedule + smem ring)
    pipeline_spec: object

    def emit(self, *, A, B, C, A_scale, B_scale, c_m, c_n, group_base=0, lds=None, epilogue_ctx=None) -> None:
        """Splice the tile inline. group_base = B slab (0=dense); lds / epilogue_ctx = fused seams."""
        ...


# Scheduler: tile-id -> (block_m, block_n) strategy (dense XCD swizzle / grouped linear).
@runtime_checkable
class SchedulerSpec(Protocol):
    """Tile-id strategy: ``map`` -> (block_m, block_n); ``grid`` -> launch grid_x."""

    cache_key: tuple

    def map(self, geom, *, c_m, c_n, n_blocks): ...

    def grid(self, geom, c_m, c_n): ...


class XcdSwizzleScheduler:
    """Dense map: XCD-aware PID remap + GROUP_M (or 2D band) swizzle for L2 reuse."""

    def __init__(self, *, GROUP_M, group_n, num_xcd):
        self.GROUP_M = GROUP_M
        self.group_n = group_n
        self.num_xcd = num_xcd

    @property
    def cache_key(self):
        return ("xcd", self.GROUP_M, self.group_n, self.num_xcd)

    def map(self, geom, *, c_m, c_n, n_blocks):
        """tile-id -> (block_m, block_n). group_n==0: 1D GROUP_M swizzle; >0: 2D band."""
        GROUP_M, group_n, num_xcd = self.GROUP_M, self.group_n, self.num_xcd
        num_pid_m = ceildiv(c_m, geom.BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        if group_n > 0:
            band_tiles = num_pid_m * group_n
            band = pid // band_tiles
            pid_in_band = pid % band_tiles
            band_n0 = band * group_n
            rem_n = n_blocks - band_n0
            band_w = arith.select(rem_n < group_n, rem_n, fx.Int32(group_n))
            nig = GROUP_M * band_w
            gid = pid_in_band // nig
            pig = pid_in_band % nig
            fpm = gid * GROUP_M
            rem_m = num_pid_m - fpm
            gsm = arith.select(rem_m < GROUP_M, rem_m, fx.Int32(GROUP_M))
            return fpm + (pig % gsm), band_n0 + (pig // gsm)
        nig = GROUP_M * n_blocks
        gid = pid // nig
        pig = pid % nig
        fpm = gid * GROUP_M
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GROUP_M, rem_m, fx.Int32(GROUP_M))
        return fpm + (pig % gsm), pig // gsm

    def grid(self, geom, c_m, c_n):
        return ceildiv(c_m, geom.BLOCK_M) * ceildiv(c_n, geom.BLOCK_N)


class LinearNoSyncScheduler:
    """Grouped map: linear row-major tile-id, offset past the front comm blocks."""

    def __init__(self, *, num_comm_cu):
        self.num_comm_cu = num_comm_cu

    @property
    def cache_key(self):
        return ("linear", self.num_comm_cu)

    def map(self, geom, *, c_m, c_n, n_blocks):
        tile_index = fx.block_idx.x - fx.Int32(self.num_comm_cu)
        return tile_index // n_blocks, tile_index % n_blocks

    def grid(self, geom, c_m, c_n):
        # exact tile count (over-launch self-bound guarded in the kernel body)
        return ceildiv(c_m, geom.BLOCK_M) * ceildiv(c_n, geom.BLOCK_N)


# Layout policy (NT/NN/TN): operand STORAGE only -- addressing + transpose + LDS tiling.
# Codegen/schedule quirks tuned per layout live with their consumer (mma / loader /
# pipeline), keyed by layout.name -- NOT here (those are not layout semantics).
@runtime_checkable
class LayoutSpec(Protocol):
    """Layout spec = operand storage: base_offsets / global_swizzle / lds_geometry +
    the transpose each operand's major-ness implies. No codegen/schedule flags."""

    name: str
    cache_key: tuple

    # storage fact: operand is K-major in memory -> needs transpose feeding MFMA
    a_transpose: bool
    b_transpose: bool

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K): ...
    def base_offsets(self, geom, *, block_m, block_n, c_m, c_n, group_base=0): ...

    def global_swizzle(self, geom, *, lane_id, wave_id, c_m, c_n): ...


class NTLayout:
    """NT: A [M,K] K-contig, B [N,K] K-contig (= B^T of [K,N])."""

    name = "nt"

    # storage: which operand is K-major and needs a transpose feeding MFMA
    a_transpose = False
    b_transpose = False

    @property
    def cache_key(self):
        return (self.name,)

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K):
        """-> LdsGeometry(N_LDS_STEPS_A, N_LDS_STEPS_B, chunk_stride, a_lds_size, b_lds_size)."""
        N_LDS_STEPS_A = LDS_BLOCK_M // WAVEFRONT
        N_LDS_STEPS_B = LDS_BLOCK_N // WAVEFRONT
        return LdsGeometry(N_LDS_STEPS_A, N_LDS_STEPS_B, 1024, LDS_BLOCK_M * BLOCK_K, LDS_BLOCK_N * BLOCK_K)

    def base_offsets(self, geom, *, block_m, block_n, c_m, c_n, group_base=0):
        """NT: A [M,K] & B [N,K] both row-major K-contig -> *K bases, unit step."""
        BLOCK_M, BLOCK_N, K = geom.BLOCK_M, geom.BLOCK_N, geom.K
        LDS_BLOCK_M, LDS_BLOCK_N = geom.LDS_BLOCK_M, geom.LDS_BLOCK_N
        A0 = (block_m * BLOCK_M) * K
        A1 = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0 = (block_n * BLOCK_N) * K
        B1 = (block_n * BLOCK_N + LDS_BLOCK_N) * K
        return TileOffsets(A0, A1, _add_group_base(B0, group_base), _add_group_base(B1, group_base), 1, 1)

    def global_swizzle(self, geom, *, lane_id, wave_id, c_m, c_n):
        """NT: both A and B row-major K-contig."""
        K, R = geom.K, geom.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        return gl_off_a, gl_off_b


class NNLayout:
    """NN: A [M,K], B [K,N]. B is K-row strided + tr16 transpose S2R; A is NT."""

    name = "nn"

    # storage: B is K-major (B [K,N]) -> tr16 transpose; A is NT
    a_transpose = False
    b_transpose = True

    @property
    def cache_key(self):
        return (self.name,)

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K):
        N_LDS_STEPS_A = LDS_BLOCK_M // WAVEFRONT
        N_LDS_STEPS_B = LDS_BLOCK_N // WAVEFRONT
        return LdsGeometry(N_LDS_STEPS_A, N_LDS_STEPS_B, 1024, LDS_BLOCK_M * BLOCK_K, LDS_BLOCK_N * BLOCK_K)

    def base_offsets(self, geom, *, block_m, block_n, c_m, c_n, group_base=0):
        BLOCK_M, BLOCK_N, K = geom.BLOCK_M, geom.BLOCK_N, geom.K
        LDS_BLOCK_M, LDS_BLOCK_N = geom.LDS_BLOCK_M, geom.LDS_BLOCK_N
        A0 = (block_m * BLOCK_M) * K
        A1 = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0 = block_n * BLOCK_N + 0
        B1 = block_n * BLOCK_N + LDS_BLOCK_N
        return TileOffsets(A0, A1, _add_group_base(B0, group_base), _add_group_base(B1, group_base), 1, c_n)

    def global_swizzle(self, geom, *, lane_id, wave_id, c_m, c_n):
        K, R = geom.K, geom.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, R, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, R)
        return gl_off_a, gl_off_b


class TNLayout:
    """TN: A [K,M], B [K,N]. Both K-row strided + tr8 + inplace MFMA + bank-spread LDS."""

    name = "tn"

    # storage: both operands K-major (A [K,M], B [K,N]) -> tr8/tr16 transpose
    a_transpose = True
    b_transpose = True

    @property
    def cache_key(self):
        return (self.name,)

    def lds_geometry(self, *, LDS_BLOCK_M, LDS_BLOCK_N, BLOCK_K):
        # tr8 transpose load spans K_log [0,128) -> >= 2 G2S rounds / 16K slot.
        N_LDS_STEPS_A = max(LDS_BLOCK_M // WAVEFRONT, 2)
        N_LDS_STEPS_B = LDS_BLOCK_N // WAVEFRONT
        a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
        b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS
        return LdsGeometry(N_LDS_STEPS_A, N_LDS_STEPS_B, _LDS_CS, a_lds_size, b_lds_size)

    def base_offsets(self, geom, *, block_m, block_n, c_m, c_n, group_base=0):
        BLOCK_M, BLOCK_N = geom.BLOCK_M, geom.BLOCK_N
        LDS_BLOCK_M, LDS_BLOCK_N = geom.LDS_BLOCK_M, geom.LDS_BLOCK_N
        A0 = block_m * BLOCK_M + 0
        A1 = block_m * BLOCK_M + LDS_BLOCK_M
        B0 = block_n * BLOCK_N + 0
        B1 = block_n * BLOCK_N + LDS_BLOCK_N
        return TileOffsets(A0, A1, _add_group_base(B0, group_base), _add_group_base(B1, group_base), c_m, c_n)

    def global_swizzle(self, geom, *, lane_id, wave_id, c_m, c_n):
        R = geom.N_LDS_ROUNDS
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, R)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, R)
        return gl_off_a, gl_off_b


_LAYOUT_SPECS = {"nt": NTLayout(), "nn": NNLayout(), "tn": TNLayout()}


# ── Operand dtype vs MMA atom: two policies (split per the dtype axis). They are
# locked together per precision (fp8 dtype <-> fp8 MFMA), so the atom's format is
# fed from the dtype's cbsz/blgp at build time. v1 ships fp8; bf16/fp4 add siblings.


# Dtype policy: operand numeric format + typed buffers + the gmem->lds->reg loaders.
# Precision-agnostic surface: any concrete numeric format (cbsz/blgp, fp4 scales, ...)
# stays private to the implementation and is fed to its paired mma via build_mfma(dtype=).
@runtime_checkable
class DtypeSpec(Protocol):
    """Operand dtype: LDS element, typed A/B buffer views, and the G2S/S2R operand
    loaders (transpose-aware, keyed by layout). Numeric-format details are private."""

    name: str
    lds_element: object  # SharedStorage @fx.struct array element type
    cache_key: tuple

    def make_buffers(self, *, A, B): ...

    def build_loaders(
        self,
        geom,
        *,
        layout,
        a_div,
        b_div,
        gl_off_a,
        gl_off_b,
        wave_id,
        wave_m,
        wave_n,
        vmcnt_hint,
        b_inline_asm_load,
    ): ...


# MMA policy: the MFMA instruction (its inst_shape feeds geometry) + the atom build.
@runtime_checkable
class MmaSpec(Protocol):
    """MMA atom: ``inst_shape`` (M, N, K) drives geometry; ``build_mfma`` builds the
    concrete atom, operand formats read off the paired dtype + inplace from the layout."""

    inst_shape: tuple
    cache_key: tuple

    def build_mfma(self, geom, *, layout, dtype): ...


# The operand dtype: numeric format (E4M3/E5M2/hybrid) + typed buffers + loaders
# (bringing operands gmem->lds->reg is the dtype's job).
class Fp8Dtype:
    """fp8 operand dtype: E4M3 / E5M2 / hybrid (cbsz->srcA, blgp->srcB) format,
    typed A/B buffer views, and the G2S/S2R operand loaders."""

    name = "fp8"
    lds_element = fx.Float8E4M3FN  # SharedStorage @fx.struct array element
    # loader codegen: which layouts force inline-asm S2R (TN's tr8)
    _ASM_LAYOUTS = frozenset({"tn"})

    def __init__(self, *, cbsz, blgp):
        self.cbsz = cbsz  # E5M2 on srcA (else E4M3)
        self.blgp = blgp  # E5M2 on srcB (else E4M3)

    @property
    def loader_ir_type(self):
        return fx.Float8E4M3FN.ir_type  # G2S / S2R / make_buffers element type

    @property
    def cache_key(self):
        return (self.name, self.cbsz, self.blgp)

    def make_buffers(self, *, A, B):
        """raw A/B fp8 tensors -> (a_div, b_div) buffer views."""
        F8_IR_t = self.loader_ir_type
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        return a_div, b_div

    def build_loaders(
        self,
        geom,
        *,
        layout,
        a_div,
        b_div,
        gl_off_a,
        gl_off_b,
        wave_id,
        wave_m,
        wave_n,
        vmcnt_hint,
        b_inline_asm_load,
    ):
        """G2S + transpose-aware S2R loaders (element type = this, transpose = layout)."""
        F8 = self.loader_ir_type
        cs = geom.chunk_stride
        # loader owns the inline-asm decision (keyed by layout)
        force_inline_asm = layout.name in self._ASM_LAYOUTS
        a_gmem2lds = G2SLoader(a_div, gl_off_a, geom.N_LDS_STEPS_A, F8, wave_id, chunk_stride=cs)
        b_gmem2lds = G2SLoader(b_div, gl_off_b, geom.N_LDS_STEPS_B, F8, wave_id, chunk_stride=cs)
        if layout.a_transpose:
            a_lds2reg = S2RLoaderTr(
                wave_m,
                geom.N_TILES_A,
                geom.LDS_BLOCK_M // 2,
                inline_asm=force_inline_asm,
                vmcnt_hint=vmcnt_hint,
                chunk_stride=cs,
            )
        else:
            a_lds2reg = S2RLoader(wave_m, geom.N_TILES_A)
        if layout.b_transpose:
            b_inline = force_inline_asm or b_inline_asm_load
            b_lds2reg = S2RLoaderTr(
                wave_n,
                geom.N_TILES_B,
                32,
                inline_asm=b_inline,
                vmcnt_hint=vmcnt_hint,
                chunk_stride=cs,
            )
        else:
            b_lds2reg = S2RLoader(wave_n, geom.N_TILES_B)
        return Loaders(a_gmem2lds, b_gmem2lds, a_lds2reg, b_lds2reg)


# The MMA atom: the MFMA instruction (its inst_shape feeds geometry) + the build of
# the concrete atom, configured by the dtype format (cbsz/blgp) + layout (inplace).
class Fp8MmaAtom:
    """fp8 MFMA 16x16x128 atom: declares inst_shape and builds the atom; its operand
    formats come from the dtype's cbsz/blgp, its inplace mode from the layout."""

    inst_shape = (16, 16, 128)  # MFMA 16x16x128 (M, N, K)
    # mma codegen: which layouts use asm-inplace MFMA (accum in AGPR)
    _INPLACE_LAYOUTS = frozenset({"tn"})

    @property
    def cache_key(self):
        return ("mfma_16x16x128",)

    def build_mfma(self, geom, *, layout, dtype):
        # operand fp8 fmt selectors come from the paired dtype (cbsz->srcA, blgp->srcB)
        cbsz, blgp = dtype.cbsz, dtype.blgp
        inplace = layout.name in self._INPLACE_LAYOUTS
        mfma = Mfma16x16x128(geom.N_TILES_A, geom.N_TILES_B)
        if inplace:
            # asm-inplace MFMA (accum in AGPR); cbsz/blgp select srcA/srcB fmt
            mfma.set_inplace_asm(cbsz, blgp)
        elif cbsz or blgp:
            # E5M2 / hybrid: rebuild atom with per-operand fp8 fmt (cbsz->srcA, blgp->srcB)
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))
        return mfma


# Epilogue: C-store strategy. Per-tensor scaled store + optional f32 node chain (EVT).
def _epi_relu(v):
    """ReLU node: max(v, 0) on the post-scale f32 accumulator."""
    return v.maximumf(fx.Float32(0.0))


# Activation registry for the epilogue node chain (name -> f32->f32 emit fn).
_EPILOGUE_ACTS = {"relu": _epi_relu}


class PerTensorEpilogue:
    """Per-tensor scaled store (bf16/fp16) + optional f32 node chain (``nodes=()`` -> plain)."""

    def __init__(self, *, out_fp16, nodes=(), cache_key=()):
        self.out_fp16 = out_fp16
        self.nodes = tuple(nodes)
        # cache key: out_fp16 always keyed; act-node name rides along only when present
        act_key = tuple(cache_key) if nodes else ()
        self.cache_key = (out_fp16, *act_key)

    def _elem_fn(self):
        """Fold the node chain into one ``f32 -> f32`` callable (None if empty)."""
        if not self.nodes:
            return None
        nodes = self.nodes

        def fn(v):
            for node in nodes:
                v = node(v)
            return v

        return fn

    def build(self, geom, *, A_scale, B_scale, C, c_m, c_n, mfma, epilogue_ctx=None):
        out_ty = fx.Float16 if self.out_fp16 else fx.BFloat16
        return StoreCPerTensor(
            A_scale,
            B_scale,
            C,
            c_m,
            c_n,
            mfma.idx,
            geom.N_TILES_A,
            geom.N_TILES_B,
            out_ty,
            elem_fn=self._elem_fn(),
        )

    def consume(self, geom, *, store_c, accum, epilogue_ctx=None):
        """Store the accumulator tile (override for scatter / split-K)."""
        N_TILES_A, N_TILES_B = geom.N_TILES_A, geom.N_TILES_B
        LDS_BLOCK_M, LDS_BLOCK_N = geom.LDS_BLOCK_M, geom.LDS_BLOCK_N
        wave_n_offset = accum.wave_n * (N_TILES_B * INST_N)
        wave_m_offset = accum.wave_m * (N_TILES_A * INST_M)
        base_row = accum.block_m * geom.BLOCK_M + wave_m_offset
        base_col = accum.block_n * geom.BLOCK_N + wave_n_offset
        store_c.store(accum.c00, base_row + 0, base_col + 0)
        store_c.store(accum.c01, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(accum.c10, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(accum.c11, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


# The mainloop axis -- stages, smem ring, and the hand-authored run().
class InterleavedKPipeline:
    """Hand-scheduled 2-stage mainloop: interleaves G2S/S2R/MFMA to hide global-load latency."""

    name = "interleaved_k"
    stages = 2  # fixed 2; reserved for future multistage

    # pipeline owns its per-layout mainloop schedule (drains/mask empirically tuned).
    # (wants_end_iter_drain, main_b0_no_drain, mask_a_in_tail)
    _SCHED = {
        "nt": (True, False, True),
        "nn": (False, False, True),
        "tn": (False, True, False),
    }

    @property
    def cache_key(self):
        return (self.name, self.stages)

    def shared_storage(self, geom, *, lds_elem, halves=HALVES):
        """LDS struct: stages*halves buffers per operand. Field order MUST be op->stage->half."""
        # halves fixed at 2 (the M/N quadrant split)
        fields = {}
        for op, size in (("A", geom.a_lds_size), ("B", geom.b_lds_size)):
            for s in range(self.stages):  # stage-major
                for h in range(halves):  # half-minor
                    fields[f"{op}_lds_s{s}_h{h}"] = fx.Array[lds_elem, size, 16]
        return fx.struct(type("SharedStorage", (), {"__annotations__": fields}))

    def rings(self, lds, halves=HALVES):
        """lds -> (a_rings, b_rings); each ``halves`` lists, ``stages`` deep, indexed ring[k%S]."""

        # halves fixed at 2 (the M/N quadrant split)
        def build(op):
            return tuple(
                [getattr(lds, f"{op}_lds_s{s}_h{h}") for s in range(self.stages)] for h in range(halves)
            )

        return build("A"), build("B")

    def run(
        self,
        geom,
        *,
        layout_name,
        nt_vmcnt,
        a_rings,
        b_rings,
        loaders,
        mfma,
        A0_gl_offset,
        A1_gl_offset,
        B0_gl_offset,
        B1_gl_offset,
        a_k_mult,
        b_k_mult,
        lane_id,
        wave_m,
        wave_n,
        block_m,
        block_n,
    ):
        """Uniform-K pipeline: prelude -> main K-loop -> epilog1 -> epilog2 (K-tail) -> AccumTile.
        Per-layout drains/mask come from this pipeline's own _SCHED (not from the layout)."""
        S = self.stages
        a_h0, a_h1 = a_rings
        b_h0, b_h1 = b_rings
        a_gmem2lds, b_gmem2lds, a_lds2reg, b_lds2reg = loaders  # G2S / S2R per operand
        N_LDS_STEPS_A = geom.N_LDS_STEPS_A
        N_LDS_STEPS_B = geom.N_LDS_STEPS_B
        geom.N_TILES_A
        geom.N_TILES_B
        N_ACCUMS = geom.N_ACCUMS
        geom.LDS_BLOCK_M
        geom.LDS_BLOCK_N
        geom.BLOCK_M
        geom.BLOCK_N
        BLOCK_K = geom.BLOCK_K
        K_ITERS = geom.K_ITERS
        K_TAIL = geom.K_TAIL
        # per-layout schedule owned by this pipeline (not layout semantics)
        wants_end_iter_drain, main_b0_no_drain, mask_a = self._SCHED[layout_name]
        end_iter_drain = nt_vmcnt if wants_end_iter_drain else None

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude: k=0 -> ring[0], k=1 -> ring[1] (a-h1 ring[1] lazily on first main iter).
        a_cur0 = a_h0[0]
        a_cur1 = a_h1[0]
        a_next0 = a_h0[1]
        b_cur0 = b_h0[0]
        b_cur1 = b_h1[0]
        b_next0 = b_h0[1]
        b_next1 = b_h1[1]
        b_gmem2lds.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * b_k_mult)
        a_gmem2lds.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * a_k_mult)
        b_gmem2lds.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * b_k_mult)
        a_gmem2lds.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * a_k_mult)

        _emit_if_then(wave_m == 1, lambda: rocdl.s_barrier())

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_gmem2lds.load(b_next0, B0_gl_offset + 1 * BLOCK_K * b_k_mult)
        a_gmem2lds.load(a_next0, A0_gl_offset + 1 * BLOCK_K * a_k_mult)
        b_gmem2lds.load(b_next1, B1_gl_offset + 1 * BLOCK_K * b_k_mult)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # Main K-loop: s2r {a0,b0,b1,a1} -> 4 mma, interleaved with k+1 / k+2 prefetch
        for k in range_constexpr(K_ITERS - 2):
            a_cur0 = a_h0[k % S]
            a_cur1 = a_h1[k % S]
            a_next1 = a_h1[(k + 1) % S]
            b_cur0 = b_h0[k % S]
            b_cur1 = b_h1[k % S]
            a_cur0_pf = a_h0[(k + S) % S]
            b_cur0_pf = b_h0[(k + S) % S]
            b_cur1_pf = b_h1[(k + S) % S]
            if main_b0_no_drain:
                # TN: b0 drain covered by the following a0 load
                b0_frag = b_lds2reg.load(b_cur0, drain=False)
            else:
                b0_frag = b_lds2reg.load(b_cur0)
            a0_frag = a_lds2reg.load(a_cur0)
            a_gmem2lds.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * a_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_lds2reg.load(b_cur1)
            b_gmem2lds.load(b_cur0_pf, B0_gl_offset + (k + 2) * BLOCK_K * b_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_lds2reg.load(a_cur1)
            a_gmem2lds.load(a_cur0_pf, A0_gl_offset + (k + 2) * BLOCK_K * a_k_mult)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_gmem2lds.load(b_cur1_pf, B1_gl_offset + (k + 2) * BLOCK_K * b_k_mult)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            if end_iter_drain is not None and end_iter_drain >= 0:
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string=f"s_waitcnt vmcnt({end_iter_drain})",
                    constraints="",
                    has_side_effects=True,
                )  # end-of-iter G2S drain

        # Epilog 1 (k = K_ITERS - 2); the a-h1 load is the c10/c11 stale-a1 fix
        k = K_ITERS - 2
        a_cur0 = a_h0[k % S]
        a_cur1 = a_h1[k % S]
        a_next1 = a_h1[(k + 1) % S]
        b_cur0 = b_h0[k % S]
        b_cur1 = b_h1[k % S]
        b_next0 = b_h0[(k + 1) % S]
        b0_frag = b_lds2reg.load(b_cur0)
        a0_frag = a_lds2reg.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_lds2reg.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_lds2reg.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_lds2reg.load(b_next0)
        a_gmem2lds.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * a_k_mult)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Epilog 2 (k = K_ITERS - 1): K-tail. NT/NN mask invalid A K-cols; TN needs none.
        k = K_ITERS - 1
        a_cur0 = a_h0[k % S]
        a_cur1 = a_h1[k % S]
        b_cur1 = b_h1[k % S]
        a0_frag = a_lds2reg.load(a_cur0)
        if mask_a:
            a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_lds2reg.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_lds2reg.load(a_cur1)
        if mask_a:
            a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Hand the accumulator tile to the epilogue
        return AccumTile(c00_frag, c01_frag, c10_frag, c11_frag, block_m, block_n, wave_m, wave_n)


# Central legality gate (CUTLASS can_implement): host-side, emits no IR.
def validate_tile_config(*, layout, K, block_tile, b_inline_asm_load=False, agpr_alloc=None):
    """Raise if the tile config is illegal. Called from every spec ctor + the nn launcher."""
    BLOCK_M, BLOCK_N, BLOCK_K = block_tile
    assert layout in ("nt", "nn", "tn")
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"
    if b_inline_asm_load and agpr_alloc is not None and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0 (a compiler-decided "
            "AGPR count conflicts with the inline-asm operand constraints); "
            "pin AGPR to a nonzero value such as 32."
        )


# Geometry derivation: leaves -> one immutable TileGeometry (pure function).
def derive_geometry(layout_spec, mma_spec, *, K, block_tile, warp_tile):
    """(layout_spec, mma_spec, shape leaves) -> TileGeometry. See docs/GEOMETRY_AXIS.md."""
    BLOCK_M, BLOCK_N, BLOCK_K = block_tile
    WARP_M, WARP_N, WARP_K = warp_tile
    # MMA instruction shape comes from the dtype/mma spec (single source of truth).
    INST_M, INST_N, INST_K = mma_spec.inst_shape
    assert BLOCK_K == INST_K, f"BLOCK_K {BLOCK_K} must equal MMA K {INST_K}"
    # Fixed 2x2 LDS double-half split (the c00..c11 quadrants); only block_tile /
    # warp_tile vary. Hierarchy: block -> 2x2 quadrant (LDS half) -> wave
    # (warp_tile = the real contiguous per-wave output tile) -> 16x16 MMA.
    LDS_BLOCK_M = BLOCK_M // QUAD
    LDS_BLOCK_N = BLOCK_N // QUAD
    assert (
        LDS_BLOCK_M % WARP_M == 0 and LDS_BLOCK_N % WARP_N == 0
    ), "warp_tile must divide the quadrant (BLOCK // 2) along M/N"
    # Wave grid = CUTLASS WarpCount within a quadrant (quadrant / WarpShape) -- the
    # swappable 2x4 / 2x2 arrangement. Per-wave MMA tiles = WarpShape / instruction.
    WAVES_M = LDS_BLOCK_M // WARP_M
    WAVES_N = LDS_BLOCK_N // WARP_N
    N_TILES_A = WARP_M // INST_M
    N_TILES_B = WARP_N // INST_N
    # layout-specific LDS sub-geometry (TN forces >= 2 G2S rounds + bank-spread)
    lds = layout_spec.lds_geometry(LDS_BLOCK_M=LDS_BLOCK_M, LDS_BLOCK_N=LDS_BLOCK_N, BLOCK_K=BLOCK_K)
    return TileGeometry(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K=K,
        WAVES_M=WAVES_M,
        WAVES_N=WAVES_N,
        N_TILES_A=N_TILES_A,
        N_TILES_B=N_TILES_B,
        N_ACCUMS=N_TILES_A * N_TILES_B,
        LDS_BLOCK_M=LDS_BLOCK_M,
        LDS_BLOCK_N=LDS_BLOCK_N,
        N_LDS_STEPS_A=lds.N_LDS_STEPS_A,
        N_LDS_STEPS_B=lds.N_LDS_STEPS_B,
        N_LDS_ROUNDS=max(lds.N_LDS_STEPS_A, lds.N_LDS_STEPS_B),
        chunk_stride=lds.chunk_stride,
        a_lds_size=lds.a_lds_size,
        b_lds_size=lds.b_lds_size,
        # K-tail: ceil(K/128) iters; last iter length K_TAIL (0 = exact)
        K_ITERS=(K + BLOCK_K - 1) // BLOCK_K,
        K_TAIL=K % BLOCK_K,
    )


# Dense FP8 tile spec: geometry + held policies + shared storage.
class DenseFp8TileSpec:
    """Dense FP8 per-tensor tile spec (satisfies TileSpec); ``emit`` splices inline."""

    def __init__(
        self,
        *,
        layout,
        K,
        block_tile,
        warp_tile,
        GROUP_M,
        num_xcd,
        group_n,
        nt_vmcnt,
        vmcnt_hint,
        b_inline_asm_load,
        cbsz,
        blgp,
        out_fp16,
        act=None,
    ):
        # unpack block leaf for local geometry math
        BLOCK_M, BLOCK_N, BLOCK_K = block_tile
        # central legality gate (agpr rule validated by the nn launcher)
        validate_tile_config(layout=layout, K=K, block_tile=block_tile)
        # compile-time geometry, derived once and threaded into every emit method
        layout_spec = _LAYOUT_SPECS[layout]
        # dtype (numeric format + buffers + loaders) and mma atom: two held policies,
        # locked together per precision. mma supplies inst_shape for geometry.
        dtype_spec = Fp8Dtype(cbsz=cbsz, blgp=blgp)
        mma_spec = Fp8MmaAtom()
        geom = derive_geometry(layout_spec, mma_spec, K=K, block_tile=block_tile, warp_tile=warp_tile)
        self.geom = geom
        assert geom.N_ACCUMS > 0
        a_lds_size = geom.a_lds_size
        b_lds_size = geom.b_lds_size

        # pipeline/mainloop policy (held): schedule body + SharedStorage. Race-fix
        # flags are NOT copied here -- run() reads them off its own _SCHED at use time.
        pipeline = InterleavedKPipeline()
        SharedStorage = pipeline.shared_storage(geom, lds_elem=dtype_spec.lds_element)
        # LDS budget guard (gfx950 160KB): fail fast on an oversized tile
        lds_bytes = pipeline.stages * 2 * (a_lds_size + b_lds_size)
        assert lds_bytes <= _LDS_LIMIT_BYTES, (
            f"LDS {lds_bytes} bytes (BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}) exceeds the "
            f"{_LDS_LIMIT_BYTES}-byte gfx950 limit; reduce the block size."
        )

        self.layout = layout
        # block/warp leaves are the contract attrs; derived scalars live in self.geom
        self.block_tile = block_tile
        self.warp_tile = warp_tile
        # tile-id scheduler (held); grouped specs swap in LinearNoSyncScheduler
        self.scheduler_spec = XcdSwizzleScheduler(GROUP_M=GROUP_M, group_n=group_n, num_xcd=num_xcd)
        # C-store epilogue (held); act=None -> plain scaled store
        _nodes = () if act is None else (_EPILOGUE_ACTS[act],)
        self.epilogue_spec = PerTensorEpilogue(
            out_fp16=out_fp16, nodes=_nodes, cache_key=() if act is None else (act,)
        )
        self.nt_vmcnt = nt_vmcnt
        self.vmcnt_hint = vmcnt_hint
        self.b_inline_asm_load = b_inline_asm_load
        self.cbsz = cbsz
        self.blgp = blgp
        self.out_fp16 = out_fp16

        # shared_storage is the LDS struct emit allocates
        self.shared_storage = SharedStorage

        # layout policy (held): operand storage / addressing
        self.layout_spec = layout_spec
        # dtype policy (held): numeric format + typed buffers + operand loaders
        self.dtype_spec = dtype_spec
        # mma-atom policy (held): the MFMA instruction + its build
        self.mma_spec = mma_spec
        # pipeline/mainloop policy (held)
        self.pipeline_spec = pipeline

        # JIT cache-key discriminator (assembled from each held policy's cache_key)
        self.cache_tag = self._assemble_cache_tag()

        # Emitted symbol name (per TileSpec protocol).
        self.kernel_name = "kernel_dense_" + layout

    # K exposed off self.geom (mega reads spec.K); BLOCK_M/N via spec.block_tile
    @property
    def K(self):
        return self.geom.K

    # ──────────────────────────────────────────────────────────────────
    def _assemble_cache_tag(self):
        """Cache-key tuple = leaf choices only (geometry scalars + each held policy's cache_key)."""
        tag = (
            "dense_fp8_v2",
            self.K,
            *self.block_tile,
            *self.warp_tile,
            self.nt_vmcnt,
            self.vmcnt_hint,
            self.b_inline_asm_load,
        )
        # held policies only; loaders are derived (built by dtype), out_fp16 rides epilogue_spec
        for p in (
            self.layout_spec,
            self.dtype_spec,
            self.mma_spec,
            self.scheduler_spec,
            self.epilogue_spec,
            self.pipeline_spec,
        ):
            tag += tuple(p.cache_key)
        return tag

    # ──────────────────────────────────────────────────────────────────
    def emit(self, *, A, B, C, A_scale, B_scale, c_m, c_n, group_base=0, lds=None, epilogue_ctx=None):
        """Splice the tile inline: setup -> per-stage policy calls -> pipeline.
        group_base / lds / epilogue_ctx are the grouped/fused seams (dense passes none)."""
        layout = self.layout_spec
        geom = self.geom  # the immutable shape value threaded into every emit method
        # materialize thread_idx.x before any lazy tr S2R (nn/tn); position is load-bearing
        if layout.a_transpose or layout.b_transpose:
            _ = str(fx.thread_idx.x)

        n_blocks = ceildiv(c_n, geom.BLOCK_N)

        if lds is None:
            lds = fx.SharedAllocator().allocate(self.shared_storage).peek()
        # buffer rings (per operand-half, stage-deep)
        a_rings, b_rings = self.pipeline_spec.rings(lds)

        lane_id = fx.thread_idx.x % WAVEFRONT
        wave_id = fx.thread_idx.x // WAVEFRONT
        wave_m = wave_id // geom.WAVES_N
        wave_n = wave_id % geom.WAVES_N

        block_m, block_n = self.scheduler_spec.map(geom, c_m=c_m, c_n=c_n, n_blocks=n_blocks)

        (
            A0_gl_offset,
            A1_gl_offset,
            B0_gl_offset,
            B1_gl_offset,
            a_k_mult,
            b_k_mult,
        ) = layout.base_offsets(
            geom, block_m=block_m, block_n=block_n, c_m=c_m, c_n=c_n, group_base=group_base
        )

        a_div, b_div = self.dtype_spec.make_buffers(A=A, B=B)
        gl_off_a, gl_off_b = layout.global_swizzle(geom, lane_id=lane_id, wave_id=wave_id, c_m=c_m, c_n=c_n)
        # mma owns inplace (keyed by layout); operand formats read off the paired dtype
        mfma = self.mma_spec.build_mfma(geom, layout=layout, dtype=self.dtype_spec)
        # loaders = dtype x layout product (element type = dtype, transpose = layout)
        loaders = self.dtype_spec.build_loaders(
            geom,
            layout=layout,
            a_div=a_div,
            b_div=b_div,
            gl_off_a=gl_off_a,
            gl_off_b=gl_off_b,
            wave_id=wave_id,
            wave_m=wave_m,
            wave_n=wave_n,
            vmcnt_hint=self.vmcnt_hint,
            b_inline_asm_load=self.b_inline_asm_load,
        )
        store_c = self.epilogue_spec.build(
            geom,
            A_scale=A_scale,
            B_scale=B_scale,
            C=C,
            c_m=c_m,
            c_n=c_n,
            mfma=mfma,
            epilogue_ctx=epilogue_ctx,
        )

        # mainloop -> accumulator tile -> epilogue store
        accum = self.pipeline_spec.run(
            geom,
            layout_name=layout.name,
            nt_vmcnt=self.nt_vmcnt,
            a_rings=a_rings,
            b_rings=b_rings,
            loaders=loaders,
            mfma=mfma,
            A0_gl_offset=A0_gl_offset,
            A1_gl_offset=A1_gl_offset,
            B0_gl_offset=B0_gl_offset,
            B1_gl_offset=B1_gl_offset,
            a_k_mult=a_k_mult,
            b_k_mult=b_k_mult,
            lane_id=lane_id,
            wave_m=wave_m,
            wave_n=wave_n,
            block_m=block_m,
            block_n=block_n,
        )
        self.epilogue_spec.consume(geom, store_c=store_c, accum=accum, epilogue_ctx=epilogue_ctx)
