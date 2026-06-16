###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped FP8 GEMM (NT), FlyDSL.

Role-specialized single kernel: ``block_index < comm_blocks`` blocks push token
rows to peer pools over XGMI and signal a per-pool-block scoreboard; the
remaining blocks each compute one NT output tile of the grouped FP8 GEMM
(``A=pool[M,K]`` fp8, per-expert ``B=weight[G,N,K]`` fp8 -> ``C=out[M,N]`` bf16),
spinning on the scoreboard until their pool block is filled. The comm latency is
hidden under the MFMA-bound GEMM.

This is packaged as ``DispatchGroupFP8TileSpec(GroupFp8TileSpec)`` (see
``group_tile_spec``): its ``build_launch`` emits BOTH roles in one kernel -- the
GEMM tile (the GEMM role reuses the parent ``spec.emit(..., lds=..., group_res=...)``
unchanged -> every stock per-stage hook + ``run_uniform_k_pipeline``) and the comm
push (the ``dispatch_tile`` closure, the ONLY kernel-specific code). The base
``GroupFp8TileSpec`` overrides only ``schedule`` (the LINEAR no-sync tile-id map,
offset past the comm blocks) and ``base_offsets`` (the per-expert B slab
``g*K*c_n``); the runtime ``tile_to_group`` flows in through ``group_res``, so the
spec carries no mutable state.

``dispatch_tile`` stays a closure inside ``build_launch``'s ``@flyc.kernel`` body
(not a spec method): its dynamic ``for``/``while``/``if`` only lower under the
kernel AST rewriter. The per-tile GEMM op sequence is bit-identical to the
standalone dense kernel and stays at its peak; this spec adds only the dispatch
fusion (comm push + the ``num_comm_blocks`` tile-id offset).

Per-tensor A/B scale (matches the dense tile spec's ``StoreCPerTensor``)."""

import functools
import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
    create_buffer_resource_from_addr,
)

from primus_turbo.flydsl.mega.group_tile_spec import (
    _LAYOUT_AGPR,
    _atomic_add_addr,
    _fence_acquire,
    _fence_release,
    _ld_relaxed as _ld_relaxed_sys,
    GroupFp8TileSpec,
    compile_grouped_gemm,
    as_i8_flat as _as_i8_flat,
    scalar_f32 as _scalar,
    weight_layout as _weight_layout,
)
from primus_turbo.flydsl.utils.gemm_helper import make_value_attrs

# Scoreboard / cross-rank prims are shared from group_tile_spec (imported above);
# only dispatch_tile (the comm push) stays kernel-specific.


_VEC = 8               # fp8 bytes per lane per push step
_GATHER_LANES = 128    # push lanes (XGMI-saturating)
_BLOCK_THREADS = 512   # 8 waves (wave_m x wave_n = 2 x 4) — the tile-spec block size


# ──────────────────────────────────────────────────────────────────────
# Fused dispatch + grouped GEMM, as a GroupFp8TileSpec subclass.
# ──────────────────────────────────────────────────────────────────────
class DispatchGroupFP8TileSpec(GroupFp8TileSpec):
    """Fused cross-rank dispatch PUSH + grouped FP8 GEMM, expressed as a
    ``GroupFp8TileSpec`` subclass. The GEMM tile reuses the parent
    ``emit``/``schedule``/``base_offsets`` UNCHANGED; ``build_launch`` emits BOTH
    roles in one kernel -- the GEMM tile (gemm_tile) and the comm push
    (``dispatch_tile``). Only ``dispatch_tile`` is kernel-specific.

    Extra construction config (vs the base): ``out_features`` (-> n_blocks / grid),
    ``pool_capacity`` (-> the no-sync over-launch bound + peer-pool record bytes),
    ``num_comm`` (the comm-task count for the round-robin comm role)."""

    def __init__(self, *, out_features, pool_capacity, num_comm, **kw):
        super().__init__(**kw)
        self.out_features = out_features
        self.pool_capacity = pool_capacity
        self.num_comm = num_comm
        self.kernel_name = "dispatch_grouped_" + self.layout
        self.cache_tag = self.cache_tag + (out_features, pool_capacity, num_comm)

    def build_launch(self, *, waves_per_eu=2, agpr_alloc=None):
        """Build the fused ``@flyc.kernel`` + ``@flyc.jit`` launcher: the front
        ``num_comm_blocks`` blocks run ``dispatch_tile`` (push + scoreboard signal),
        the rest each compute one GEMM tile via the stock ``self.emit`` (spinning on
        the scoreboard first). Decorator form (NOT the base's functional form): the
        comm push has dynamic while/if that need the @flyc.kernel AST rewriter."""
        spec = self
        layout = self.layout
        hidden_size = self.K
        BLOCK_M, BLOCK_N = self.BLOCK_M, self.BLOCK_N
        out_features = self.out_features
        pool_capacity = self.pool_capacity
        num_comm = self.num_comm
        num_comm_blocks = self.num_comm_blocks
        if agpr_alloc is None:
            agpr_alloc = _LAYOUT_AGPR[layout]
        assert hidden_size % (_GATHER_LANES * _VEC) == 0, "hidden must be a multiple of 1024 (push step)"
        assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
        assert out_features % BLOCK_N == 0, "out_features must be a multiple of BLOCK_N"
        n_blocks = out_features // BLOCK_N
        worst_case_tiles = pool_capacity // BLOCK_M   # no-sync over-launch bound (LINEAR map)

        # comm-role push geometry. fp8 token bytes are pushed as i32 words (v8i8 raw
        # buffer_load does not legalize; v2i32 = same 8 bytes/lane = a legal dwordx2).
        _VEC_I32 = _VEC // 4
        hidden_i32 = hidden_size // 4
        cols_per_step_i32 = _GATHER_LANES * _VEC_I32
        chunk_count = hidden_i32 // cols_per_step_i32   # == hidden_size // 1024
        pool_record_bytes = pool_capacity * hidden_size  # fp8 = 1 byte
        use_acquire = os.environ.get("MEGA_FP8_ACQ", "0") == "1"

        @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
        def dispatch_grouped(INPUT_TOKENS: fx.Tensor, COMM_DESTINATION: fx.Tensor, COMM_START: fx.Tensor,
                 COMM_COUNT: fx.Tensor, COMM_SOURCE_OFFSET: fx.Tensor, SOURCE_TOKENS: fx.Tensor,
                 POOL_PTRS: fx.Tensor, SCOREBOARD_PTRS: fx.Tensor, POOL: fx.Tensor,
                 WEIGHTS: fx.Tensor, OUTPUT: fx.Tensor, A_SCALE: fx.Tensor, B_SCALE: fx.Tensor,
                 TILE_TO_GROUP: fx.Tensor, SCOREBOARD: fx.Tensor, EXPECTED: fx.Tensor,
                 NUM_TILE_BLOCKS: fx.Tensor, c_n: fx.Int32):
            _ = spec.cache_tag  # JIT cache-key discriminator; emits no IR
            thread_index = fx.thread_idx.x
            block_index, _b, _c = fx.block_idx
            comm_block_count = fx.Int32(num_comm_blocks)
            # NT/TN: LDS at the top (unconditional); NN: lds=None -> emit allocs it
            # inside the guard. Ternary (not an if-statement, which the kernel AST
            # rewriter would dispatch dynamically). Opposite codegen sensitivity per layout.
            lds = fx.SharedAllocator().allocate(spec.shared_storage).peek() if layout != "nn" else None

            # ===== COMM role resources =====
            input_resource = create_buffer_resource(INPUT_TOKENS, max_size=True)
            destination_resource = create_buffer_resource(COMM_DESTINATION, max_size=True)
            comm_start_resource = create_buffer_resource(COMM_START, max_size=True)
            comm_count_resource = create_buffer_resource(COMM_COUNT, max_size=True)
            comm_source_offset_resource = create_buffer_resource(COMM_SOURCE_OFFSET, max_size=True)
            source_tokens_resource = create_buffer_resource(SOURCE_TOKENS, max_size=True)
            pool_address_resource = create_buffer_resource(POOL_PTRS, max_size=True)
            scoreboard_address_resource = create_buffer_resource(SCOREBOARD_PTRS, max_size=True)
            # ===== GEMM role resources =====
            group_resource = create_buffer_resource(TILE_TO_GROUP, max_size=True)
            expected_resource = create_buffer_resource(EXPECTED, max_size=True)
            num_tile_blocks_resource = create_buffer_resource(NUM_TILE_BLOCKS, max_size=True)

            def dispatch_tile(task_index):
                # push fp8 token rows to the peer pool, release, then signal scoreboard
                destination_rank = buffer_load(destination_resource, task_index, vec_width=1, dtype=fx.T.i32())
                dest_row_start = buffer_load(comm_start_resource, task_index, vec_width=1, dtype=fx.T.i32())
                token_count = buffer_load(comm_count_resource, task_index, vec_width=1, dtype=fx.T.i32())
                source_offset = buffer_load(comm_source_offset_resource, task_index, vec_width=1, dtype=fx.T.i32())
                pool_address = buffer_load(pool_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
                scoreboard_address = buffer_load(scoreboard_address_resource, destination_rank, vec_width=1, dtype=fx.T.i64())
                peer_pool = create_buffer_resource_from_addr(pool_address, num_records_bytes=pool_record_bytes)
                if thread_index < fx.Int32(_GATHER_LANES):
                    for row_index in range(token_count):
                        source_row = buffer_load(source_tokens_resource, source_offset + row_index, vec_width=1, dtype=fx.T.i32())
                        dest_row = dest_row_start + row_index
                        chunk_values = []
                        for chunk_index in fx.range_constexpr(chunk_count):
                            column = fx.Int32(chunk_index * cols_per_step_i32) + thread_index * fx.Int32(_VEC_I32)
                            chunk_values.append(buffer_load(input_resource, source_row * fx.Int32(hidden_i32) + column, vec_width=_VEC_I32, dtype=fx.T.i32()))
                        for chunk_index in fx.range_constexpr(chunk_count):
                            column = fx.Int32(chunk_index * cols_per_step_i32) + thread_index * fx.Int32(_VEC_I32)
                            buffer_store(chunk_values[chunk_index], peer_pool, dest_row * fx.Int32(hidden_i32) + column)
                fx.rocdl.s_waitcnt(0)
                fx.gpu.barrier()
                if thread_index == fx.Int32(0):
                    # _fence_release()   # ensure the pushed pool rows are visible before the signal
                    first_block = dest_row_start // fx.Int32(BLOCK_M)
                    last_block = (dest_row_start + token_count - fx.Int32(1)) // fx.Int32(BLOCK_M)
                    block_span = last_block - first_block + fx.Int32(1)
                    for block_offset in range(block_span):
                        _atomic_add_addr(scoreboard_address, first_block + block_offset, fx.Int32(1))

            if block_index < comm_block_count:
                # COMM role: round-robin share of the comm tasks
                local_task_count = (fx.Int32(num_comm) - block_index + comm_block_count - fx.Int32(1)) // comm_block_count
                for task_iteration in range(local_task_count):
                    dispatch_tile(block_index + task_iteration * comm_block_count)
            else:
                # GEMM role: custom LINEAR tile-id map; no-sync over-launch self-bound
                # by num_tile_blocks (padding tiles early-exit). block_m only -> the
                # scoreboard spin; the stock spec.emit re-derives (block_m, block_n).
                tile_index = block_index - comm_block_count
                block_m = tile_index // fx.Int32(n_blocks)
                real_tiles = buffer_load(num_tile_blocks_resource, fx.Int32(0), vec_width=1, dtype=fx.T.i32())
                if block_m < real_tiles:
                    c_m_real = real_tiles * fx.Int32(BLOCK_M)
                    # spin until every comm task on this pool block has signalled
                    expected_count = buffer_load(expected_resource, block_m, vec_width=1, dtype=fx.T.i32())
                    if thread_index == fx.Int32(0):
                        signal = _ld_relaxed_sys(SCOREBOARD, block_m)
                        while signal < expected_count:
                            fx.rocdl.s_sleep(fx.Int32(2))
                            signal = _ld_relaxed_sys(SCOREBOARD, block_m)
                    fx.gpu.barrier()
                    # if use_acquire:
                    #     _fence_acquire()
                    # standard immutable emit; per-expert B slab via the scalar seam
                    gbase = spec.group_base(group_resource, block_m, hidden_size, c_n)
                    spec.emit(A=POOL, B=WEIGHTS, C=OUTPUT, A_scale=A_SCALE, B_scale=B_SCALE,
                              c_m=c_m_real, c_n=c_n, lds=lds, group_base=gbase)

        @flyc.jit
        def launch(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT, COMM_SOURCE_OFFSET,
                   SOURCE_TOKENS, POOL_PTRS, SCOREBOARD_PTRS, POOL, WEIGHTS, OUTPUT, A_SCALE, B_SCALE,
                   TILE_TO_GROUP, SCOREBOARD, EXPECTED, NUM_TILE_BLOCKS, c_n: int,
                   stream: fx.Stream = fx.Stream(None)):
            grid_size = num_comm_blocks + worst_case_tiles * n_blocks
            dispatch_grouped(INPUT_TOKENS, COMM_DESTINATION, COMM_START, COMM_COUNT, COMM_SOURCE_OFFSET,
                 SOURCE_TOKENS, POOL_PTRS, SCOREBOARD_PTRS, POOL, WEIGHTS, OUTPUT, A_SCALE, B_SCALE,
                 TILE_TO_GROUP, SCOREBOARD, EXPECTED, NUM_TILE_BLOCKS, c_n,
                 value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512")).launch(
                grid=(grid_size, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

        return launch


@functools.lru_cache(maxsize=256)
def make_dispatch_tile_spec(*, layout, K, out_features, pool_capacity, num_comm,
                            BLOCK_M=256, BLOCK_N=256, nt_vmcnt=3, num_comm_blocks=0,
                            vmcnt_hint=2):
    """Cached ``DispatchGroupFP8TileSpec`` factory (mirrors ``make_group_tile_spec``
    + the dispatch config). TN uses a deeper tr8 drain hint."""
    vh = 3 if layout == "tn" else vmcnt_hint
    return DispatchGroupFP8TileSpec(
        out_features=out_features, pool_capacity=pool_capacity, num_comm=num_comm,
        num_comm_blocks=num_comm_blocks, layout=layout, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        GROUP_M=1, num_xcd=1, group_n=0, nt_vmcnt=nt_vmcnt, vmcnt_hint=vh,
        b_inline_asm_load=False, cbsz=0, blgp=0, out_fp16=False)


@functools.lru_cache(maxsize=256)
def _compile(layout, out_features, hidden_size, pool_capacity, BLOCK_M, BLOCK_N, comm_blocks,
             num_comm, nt_vmcnt=3, waves_per_eu=2, agpr_alloc=0):
    spec = make_dispatch_tile_spec(layout=layout, K=hidden_size, out_features=out_features,
                                   pool_capacity=pool_capacity, num_comm=num_comm,
                                   BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, nt_vmcnt=nt_vmcnt,
                                   num_comm_blocks=int(comm_blocks))
    return spec.build_launch(waves_per_eu=waves_per_eu, agpr_alloc=agpr_alloc)


# The standalone grouped GEMM-only launcher now lives on the spec:
# ``GroupFp8TileSpec.build_launch`` (via ``compile_grouped_gemm``), used by
# ``grouped_gemm_fp8_only`` below. (_as_i8_flat / _scalar / _weight_layout are
# imported from group_tile_spec.)


# Per-shape autotune candidates (comm_blocks, nt_vmcnt, agpr_alloc, waves_per_eu).
# The comm/GEMM CU split dominates (each comm CU pushes at HBM/XGMI, each GEMM CU
# does MFMA); sweep it widely. nt_vmcnt (G2S drain) / waves are secondary.
_DISPATCH_CANDIDATES = [
    (16, 3, 0, 2), (24, 3, 0, 2), (32, 3, 0, 2), (40, 3, 0, 2),
    (48, 3, 0, 2), (56, 3, 0, 2), (64, 3, 0, 2), (80, 3, 0, 2), (96, 3, 0, 2),  # CU-split sweep
    (48, 4, 0, 2), (48, 2, 0, 2), (64, 4, 0, 2),                                # G2S drain depth
    (48, 3, 0, 1), (48, 3, -256, 2),                                            # occupancy probes
]
_DISPATCH_AUTOTUNE_CACHE: dict = {}


def dispatch_grouped_gemm_fp8(x_fp8, comm, pool_ptrs, scoreboard_ptrs, pool_fp8, weight_fp8,
                              output, tile_to_group, scoreboard, expected, mblk_dev, *,
                              a_scale, b_scale, layout="nt", BM=256, BN=256, comm_blocks=32,
                              nt_vmcnt=3, autotune=False, autotune_reset=None):
    """Fused cross-rank dispatch PUSH + grouped FP8 GEMM.

    ``layout`` selects the GEMM: ``nt`` (fwd L1, weight [G,N,K]) or ``nn`` (bwd
    dgrad, weight [G,K,N]). ``x_fp8`` [num_src_tokens, K] fp8 source tokens;
    ``pool_fp8`` [pool_cap, K] fp8 landing pool; ``output`` [pool_cap, N] bf16;
    per-tensor ``a_scale`` / ``b_scale``. ``comm`` carries
    dest/start/cnt/srcoff/src_tokens/num_comm. Scoreboard must be zeroed first."""
    G, K_contract, out_features, weight_bytes = _weight_layout(layout, weight_fp8)
    hidden_size = x_fp8.size(1)
    assert K_contract == hidden_size, f"weight K={K_contract} != activation K={hidden_size}"
    pool_capacity = pool_fp8.size(0)
    c_n = out_features
    device = x_fp8.device

    sa = _scalar(a_scale, device)
    sb = _scalar(b_scale, device)
    # source tokens pushed as i32 words (legal dwordx2 load); pool/weight read as fp8 bytes
    x_i32 = x_fp8.contiguous().view(torch.int32).view(-1)
    pool_bytes = _as_i8_flat(pool_fp8)
    output_flat = output.contiguous().view(-1)

    pos_args = (x_i32, comm.dest, comm.start, comm.cnt, comm.srcoff, comm.src_tokens,
                pool_ptrs, scoreboard_ptrs, pool_bytes, weight_bytes, output_flat, sa, sb,
                tile_to_group, scoreboard, expected, mblk_dev, c_n)

    if autotune:
        key = (layout, out_features, hidden_size, pool_capacity, BM, BN, int(comm.num_comm))
        cached = _DISPATCH_AUTOTUNE_CACHE.get(key)
        if cached is None:
            cached = _autotune(layout, pos_args, output_flat, out_features, hidden_size,
                               pool_capacity, BM, BN, int(comm.num_comm), scoreboard, autotune_reset)
            _DISPATCH_AUTOTUNE_CACHE[key] = cached
        launch, _cfg = cached
    else:
        launch = _compile(layout, out_features, hidden_size, pool_capacity, BM, BN,
                          int(comm_blocks), int(comm.num_comm), int(nt_vmcnt))
    launch(*pos_args, stream=torch.cuda.current_stream())
    return output


def _autotune(layout, pos_args, finite_view, out_features, hidden_size, pool_capacity, BM, BN,
              num_comm, scoreboard, reset):
    """Bench the candidates with a per-iter scoreboard reset; return (launch, cfg)."""
    if reset is None:
        reset = scoreboard.zero_
    stream = torch.cuda.current_stream()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best_us, best = float("inf"), None
    for comm_blocks, nt_vmcnt, agpr, waves in _DISPATCH_CANDIDATES:
        try:
            launch = _compile(layout, out_features, hidden_size, pool_capacity, BM, BN,
                              int(comm_blocks), num_comm, int(nt_vmcnt), int(waves), int(agpr))
            reset(); launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            if not torch.isfinite(finite_view.view(-1)[:1024].float()).all().item():
                continue
            for _ in range(2):
                reset(); launch(*pos_args, stream=stream)
            torch.cuda.synchronize()
            us_total = 0.0
            for _ in range(20):
                reset()
                e0.record(); launch(*pos_args, stream=stream); e1.record()
                torch.cuda.synchronize()
                us_total += e0.elapsed_time(e1) * 1000.0
            us = us_total / 20
            if us < best_us:
                best_us, best = us, (launch, (comm_blocks, nt_vmcnt, agpr, waves))
        except Exception:
            continue
    if best is None:
        raise RuntimeError("dispatch_grouped_gemm_fp8 autotune found no working cfg")
    return best


def grouped_gemm_fp8_only(pool_fp8, weight_fp8, output, tile_to_group, mblk_dev, *,
                          a_scale, b_scale, layout="nt", BM=256, BN=256, nt_vmcnt=3,
                          waves_per_eu=2, agpr_alloc=None, act=None):
    """Pure grouped FP8 GEMM (no dispatch) — the compute-peak baseline.

    NT/NN: ``pool_fp8`` is A=[M,K] (output rows = M = pool rows). TN: ``pool_fp8``
    is A=[K,M] (output rows = M = A's columns); ``output`` is [M,N]."""
    G, K_contract, out_features, weight_bytes = _weight_layout(layout, weight_fp8)
    if layout == "tn":
        hidden_size = pool_fp8.size(0)        # A=[K,M]
        pool_capacity = pool_fp8.size(1)      # output rows = M
    else:
        hidden_size = pool_fp8.size(1)        # A=[M,K]
        pool_capacity = pool_fp8.size(0)      # output rows = M
    assert K_contract == hidden_size, f"weight K={K_contract} != A K={hidden_size}"
    if agpr_alloc is None:
        agpr_alloc = _LAYOUT_AGPR[layout]     # TN inplace MFMA needs 128 AGPRs
    device = pool_fp8.device
    sa = _scalar(a_scale, device)
    sb = _scalar(b_scale, device)
    pool_bytes = _as_i8_flat(pool_fp8)
    output_flat = output.contiguous().view(-1)
    # spec-owned launcher (GroupFp8TileSpec.build_launch); c_m = pool rows (grid bound),
    # mblk_dev = real tile-blocks (runtime self-bound), c_n = out_features.
    launch = compile_grouped_gemm(layout, hidden_size, BM, BN, int(nt_vmcnt),
                                  int(waves_per_eu), int(agpr_alloc), act=act)
    launch(pool_bytes, weight_bytes, output_flat, sa, sb, tile_to_group, mblk_dev,
           pool_capacity, out_features, stream=torch.cuda.current_stream())
    return output
