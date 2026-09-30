"""gfx1250 FlyDSL flash-attention backward -- the three kernels, batched and GQA-aware.

Copied from Primus-Turbo/output/0917__flydsl/kernels/ (see PROVENANCE.md) and extended
with exactly what those bring-up files did not have: a batch/head grid, GQA, and -- for
dq -- a runtime kv loop and the causal mask. Nothing else is changed; the data path, the
fragment layouts and the LDS staging are the validated originals.

    delta[b,h,s] = sum_d dO*O                              k_delta_bshd  (odo, verbatim)
    dV[kv,d] = sum_q P^T dO ; dK[kv,d] = sum_q dS^T Q      k_dkdv        (+ b/h/GQA)
    dQ[q,d]  = sum_kv dS K                                 k_dq          (+ kv loop, causal)

Layouts: q/o/do [B, Sq, Hq, D] bf16; k/v [B, Skv, Hkv, D] bf16; lse/delta [B, Hq, Sq] fp32
natural log; dq [B, Sq, Hq, D] fp32; dk/dv [B, Skv, Hkv, D] fp32.

GQA is reduced INSIDE k_dkdv: one workgroup owns a kv tile of one kv head and streams every
query tile of all `G = Hq/Hkv` q heads that share it, accumulating into the same registers.
So every output element is written exactly once, with no atomics and no host reduction --
the determinism gate in op.config.determinism holds by construction.

Divisibility is asserted, not handled: Sq % 16 == 0, Skv % 32 == 0. The bring-up files
assert the same way and every shape in op.shape divides.
"""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

# _env sets sys.path (flydsl 0.3.2, aiter) and TORCH_BLAS_PREFER_HIPBLASLT. It must run
# before flydsl/aiter are imported, and it is loaded by path so two implementation
# directories in one process never share a module object.
_n = f"_env__{abs(hash(str(_pl.Path(__file__).resolve().parent)))}"
_sp = _ilu.spec_from_file_location(_n, _pl.Path(__file__).resolve().parent / "_env.py")
_env = _ilu.module_from_spec(_sp)
_sys.modules[_n] = _env
_sp.loader.exec_module(_env)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops      # w4f: Q/dO TDM ring

from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir

D = 128
DV8 = D // 8                 # vec8 tiles per row of one head
NDT = D // 32                # WMMA k-steps to contract 128
NDO = D // 16                # 16-wide output tiles across d
NKV = 2                      # r1.i6.g06: 16-row kv sub-tiles per workgroup
NST = 2 * NKV * NDO          # r7.i1.g21: dV/dK accumulators carried by k_dkdv
WAVE = 32                    # gfx1250 dispatches wave32; gfx942's odo says 64 here
LOG2E = 1.4426950408889634
NEG = -3.0e38
VF_KV = False                # lab-kdq: fma softmax + scale-at-store in k_dkdv
VF_Q = False                 # lab-kdq: same in k_dqg
KV_U2 = False                # lab-kdq: k_dkdv qloop_full unrolled by 2
TRORDER = True               # 0930 dkdv_trorder: fenced 4-region emission of k_dkdv's final DS phase
QDO_BURST = False            # 0930 dkdv_trorder_b: all 32 Q/dO staging stores as one fenced burst
BLOCK_KV = 32               # r1.i6.g06: kv rows one workgroup owns (was 16)
S_ROW_B = BLOCK_KV * 2 + 16  # r4.i1.g16: 64 -> 80 B. 16 dwords is a 4-way bank
                             # collision at 64 banks; 20 dwords walks all 64.
X_ROW_B = D * 2 + 16         # r4.i1.g16: 256 -> 272 B. 256 B = 64 dwords is EXACTLY
                             # the full 64-way collision stride on gfx1250's 64x4B
                             # LDS: every staged row starts on bank 0, so the 16
                             # rows a ds_load_tr16_b128 phase touches all collide.
                             # 68 dwords gives row*4 mod 64 -- 16 distinct groups.

DELTA_THREADS = 256
LANES_PER_ROW = D // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


# ---- shared helpers, verbatim from the bring-up files ------------------------------
def _bv(t, nb, dt, vec=1):
    nt = (1 << 31) // (vec * (dt.width // 8))
    b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=fx.Int64(nb))
    return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))


def _atom(dt, vec):
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)


def _ldv(buf, tile, dt, vec):
    f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
    fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
    return f.load()


def _ld1(buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.copy_atom_call(_atom(dt, 1), fx.slice(buf, (idx, None)), f)
    return f.load()[0]


def _st1(v, buf, idx, dt):
    f = fx.make_rmem_tensor(fx.make_layout(1, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements([v], dtype=dt), f)
    fx.copy_atom_call(_atom(dt, 1), f, fx.slice(buf, (idx, None)))


def _stv(vals, buf, tile, dt):
    """Store a `len(vals)`-wide vector of `dt` at vector-index `tile`. Mirrors _ldv."""
    n = len(vals)
    f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
    fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
    fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (tile, None)))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


# ==================================================================== odo ==========
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32. Verbatim from odo."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    g_do = _bv(DO, n_rows * (D * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D * 2), fx.BFloat16, 8)
    g_delta = _bv(DEL, n_rows * 4, fx.Float32)

    tile = bid * (ROWS_DELTA * DV8) + tid
    do_vecs = [_ldv(g_do, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
               for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tile + u * (ROWS_PER_PASS * DV8), fx.BFloat16, 8).ir_value()
              for u in range_constexpr(PASSES_PER_WG)]

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(4):
            acc = acc + fx.gpu.shuffle_xor(acc, 1 << sft, WAVE)
        idx = bid * fx.Int32(ROWS_DELTA) + u * fx.Int32(ROWS_PER_PASS) + row_in_group
        ok = (lane_in_row == fx.Int32(0)) & (idx < n_rows)
        b = idx // (S * H)
        rem = idx - b * (S * H)
        s_ = rem // H
        h = rem - s_ * H
        dst = (b * H + h) * S + s_
        _st1(acc, g_delta, ok.select(dst, n_rows), fx.Float32)


@flyc.jit
def launch_delta(DO, O, DEL, S: fx.Int32, H: fx.Int32, n_rows: fx.Int32,
                 nblk: fx.Int32, stream: fx.Stream):
    k_delta_bshd(DO, O, DEL, S, H, n_rows).launch(
        grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream)


# =================================================================== dkdv =========
# r13.i1.g42 -- SPLIT-K OVER THE q LOOP, and why the body became a helper.
# The census (rounds/013/1-opt/raw/census.txt) says k_dkdv's grid is exactly ONE dispatch
# wave at proxy (1024 workgroups on 256 CU x 4 SIMD at 1 wave/SIMD, g38) while its causal
# work skew is 128:1 -- so makespan = the LONGEST workgroup and half the machine's time is
# idle wait. Ordering cannot fix a one-wave dispatch; only splitting can. PARTIAL=True
# splits the unmasked q-pair range `nsp` ways, each split writing its own fp32 workspace
# slice, reduced on the host. PARTIAL=False is the shipped path, byte-for-byte, and is what
# prod uses (8 dispatch waves, already 100% balanced -- splitting there is pure cost).
def _dkdv_impl(PARTIAL, Q, K, V, DO, LSE, DEL, DV_, DK,
               scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nsp):
    """One wave owns a 16-row kv tile of one kv head and streams every (q head, q tile).

    grid = (Skv/16, Hkv, B). The accumulators persist across the G q heads that share this
    kv head, so the GQA reduction happens in registers -- atomic-free, written once.
    """
    lane = fx.Int32(fx.thread_idx.x)
    # h8 PROBE (throwaway, never shipped): x and y are swapped so that adjacent
    # workgroup ids walk kv HEADS instead of kv TILES.  Same work, same output,
    # only the dispatch order changes -- if the ~107 GB of requested Q/dO bytes
    # are actually served by L2/MALL because concurrent workgroups share (hkv,b),
    # this scatter destroys that sharing and must slow down.  If HBM really is
    # supplying all 107 GB already, this is a no-op.
    if PARTIAL:
        _x = fx.Int32(fx.block_idx.x)           # kv head * nsp + split
        hkv = _x // nsp
        sp = _x - hkv * nsp
    else:
        hkv = fx.Int32(fx.block_idx.x)          # kv head
        sp = fx.Int32(0)
    bid = fx.Int32(fx.block_idx.y)              # kv tile
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)

    # r1.i7.g07 -- ADDRESS CLAMP, not an algorithm change. Every descriptor here used to
    # carry a flat 1 GiB num_records, so a load one element past a tensor was ISSUED and
    # walked to whatever page came next (h2). The odo kernel already proves the hardware
    # bound works on this part: it steers its own OOB store with `ok.select(dst, n_rows)`
    # against `_bv(DEL, n_rows * 4, ...)`. Giving every buffer its TRUE byte extent makes
    # any over-read return 0 instead of faulting -- and a clamp that ever hits a LIVE
    # access craters SQNR, so it cannot hide a real bug.
    nq_b = B_ * Sq * Hq * fx.Int32(D * 2)       # q / do  bf16 [B, Sq, Hq, D]
    nkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)    # k / v   bf16 [B, Skv, Hkv, D]
    nl_b = B_ * Hq * Sq * fx.Int32(4)           # lse / delta fp32 [B, Hq, Sq]
    # r3.i2.g11 -- dK/dV are stored as BF16 directly. The caller used to take an fp32
    # buffer and convert it with `.to(bf16)`, which is one extra full-tensor read+write
    # and one extra kernel launch per output; the value that reaches the caller is the
    # same fp32 accumulator rounded once either way.
    ndkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)   # dk / dv bf16 [B, Skv, Hkv, D]
    g_q = _bv(Q, nq_b, fx.BFloat16, 8)
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    g_do = _bv(DO, nq_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)
    if PARTIAL:
        _np_b = nsp * B_ * Skv * Hkv * fx.Int32(D * 4)   # [nsp, B, Skv, Hkv, D] fp32
        g_dv = _bv(DV_, _np_b, fx.Float32)
        g_dk = _bv(DK, _np_b, fx.Float32)
    else:
        g_dv = _bv(DV_, ndkv_b, fx.BFloat16)
        g_dk = _bv(DK, ndkv_b, fx.BFloat16)

    rs_q = Hq * fx.Int32(DV8)                   # vec8 tiles between consecutive q rows
    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    # r1.i6.g06. The workgroup owns BLOCK_KV key rows as NKV 16-row WMMA accumulator
    # sets. Q/dO staging is untouched -- still 16 query rows -- so the whole LDS increment
    # is the [q][kv] P and dS tiles going 16 -> 32 columns: 9216 -> 10240 B, which is
    # still 5 x 2048 B of granularity and still the 32-workgroup residency rung.
    # r12.i1.g39 -- LDS SEGMENT SEPARATION. gfx1250's LDS is organised in 64 KB
    # segments served by TWO 256 B/cycle read ports; two reads that fall in different
    # segments can be served in the same cycle, two in the same segment cannot
    # (`optimization/techniques/6-gfx1250-cdna5-mechanisms.md:252-268`). The output
    # GEMM's A ring (P, dS) and B ring (dO, Q) were adjacent inside one 22528 B block,
    # so all 40 transposing reads of a body contended for one port. Put the B ring at
    # offset 0 and the A ring one whole segment up: adding exactly 65536 flips bit 16
    # of the address, so the two rings land in ADJACENT segments whatever the
    # workgroup's physical LDS base is -- no alignment assumption.
    # Occupancy is unchanged: 70656 B gives floor(327680/70656) = 4 workgroups per CU,
    # and VGPR already caps it at 4 (740 VGPR, 32768 registers/SIMD -> 1 wave/SIMD,
    # r12.i0.g38). The SMALL ring is the one placed high, which is what keeps the
    # total under the 4-workgroup rung.
    LDS_SEG = 65536
    smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_do = _lds0
    lds_q = lds_do + fx.Int32(32 * X_ROW_B)
    lds_p = _lds0 + fx.Int32(LDS_SEG)
    lds_ds = lds_p + fx.Int32(32 * S_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    def gfrag2(buf, base, rs, r, dt):
        """gfrag's two halves before the shuffle: cols [c, c+8) and [c+16, c+24) of
        row r+row, where c = half*8 + dt*32. Each is exactly one 16-byte LDS chunk."""
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8),
                _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    # K and V fragments for this kv tile are invariant over every query and every q head.
    # r7.i1.g21 -- PREFETCH. The 32 Q/dO buffer_load_b128 an iteration consumes are
    # issued one iteration EARLY and carried in the scf.for state, so the first wait
    # that consumes them sits ~450 instructions after issue instead of 2. Issue order
    # is unchanged; only the distance to the consuming s_wait_loadcnt moves.
    def _ldqd(qt, gh):
        qh = hkv * G + gh
        base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
        q0 = qt * fx.Int32(32)
        out = []
        # r16.i1.g48 -- the 4 LSE/delta buffer_load_b32 join g21's cross-iteration
        # prefetch, and are issued FIRST so they sit at the HEAD of the loadcnt FIFO.
        # r16.i0.attrib measured the in-place versions at a load-to-use distance of
        # 114 slots -- the shortest in the body -- and attributed 687 of .LBB0_8's 777
        # stall cycles (88.4%) to the single wait that consumes them, while the 32
        # b128 g21 already covers contribute exactly zero.
        # ⚠ This is the family of the dead r10.i1.g27 (-14.6%). g27 died because "the
        # consumption point was pushed past all 36/36 landing" (dead_ends.md:196) --
        # an ORDERING failure, not a depth one. Head-of-FIFO placement is what keeps
        # the consuming wait PARTIAL (0x20: 32 b128 still in flight) instead of 0x0.
        base_l = (bat * Hq + qh) * Sq
        for hh in range_constexpr(2):
            qg = q0 + fx.Int32(hh * 16) + row
            # carried as 1-wide vectors: the scf.for carried tuple is _ir()'d, and
            # _to_raw only accepts vector values.
            out.append(_ldv(g_lse, base_l + qg, fx.Float32, 1))
            out.append(_ldv(g_del, base_l + qg, fx.Float32, 1))
        # r16.i4.g51 -- g48 died at its own zero-card-time gate because the SCHEDULER
        # sank the 32 b128 below the b32 consumption point, turning the consuming wait
        # from 0x22 (partial) into 0x0 (full drain) -- i.e. straight back into the dead
        # r10.i1.g27. Source order is not issue order. This fence (mask=0: nothing may
        # cross) is what pins the 4 b32 ahead of the 32 b128 in the ISA.
        rocdl.sched_barrier(0)
        for hh in range_constexpr(2):
            qh0 = q0 + fx.Int32(hh * 16)
            for buf in (g_q, g_do):
                for dt in range_constexpr(NDT):
                    a, b = gfrag2(buf, base_q, rs_q, qh0, dt)
                    out.append(a)
                    out.append(b)
        return out

    kf = [[gfrag(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
          for kh in range(NKV)]
    vf = [[gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
          for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    # CAUSAL TILE SKIP. Under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so this workgroup's kv tile [kv0, kv0+16) is untouched by every
    # query tile below qt_start = max(0, (kv0 - cshift) // 16). Tiles below it contributed
    # exactly zero to dK/dV (p = exp2(NEG*LOG2E) = 0), so skipping them is not an
    # approximation -- the result is bit-identical, only the work is gone.
    nqt2 = nqt // fx.Int32(2)                   # query tiles come in PAIRS now

    def _clampqt(t):
        """h33 defect c -- pin a prefetch's query pair into [0, nqt2-1]."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # r6.i2.g20 -- MASK-SPLIT + GQA LOOP TRANSPOSE, the k_dkdv half of g19.
    # (a) LOOP TRANSPOSE. The iteration index was q-head-OUTER
    #     (gh = ii // nqp_eff), so each of the G q heads was swept over the whole query
    #     range before the next began. It is now query-pair-OUTER, q-head-INNER
    #     (qi = ii // G, gh = ii - qi*G). Two reasons: it puts every masked iteration at
    #     the FRONT of the sequence, which is what makes (b) a clean loop split; and the
    #     G q heads of one GQA group occupy G*D*2 = 1 KiB of CONTIGUOUS bytes inside one
    #     q row, so sweeping them back to back reads 1 KiB per row instead of revisiting
    #     the same row G times a whole Sq sweep apart.
    #     ⚠ This reorders the fp32 accumulation of dK/dV over (q head, query pair), so
    #     the arm is NOT bitwise identical to the incumbent -- the only shipped change in
    #     this job that is not. It is still exactly deterministic run to run (a fixed
    #     order, no atomics), which is what op.config.determinism gates.
    # (b) MASK-SPLIT. Under bottom-right causality exactly ONE query pair per q head
    #     straddles this workgroup's kv tile; every later pair has kv0+BLOCK_KV-1 <= q0 +
    #     cshift, so the predicate is provably false for all 8 elements of all 4 tiles.
    #     In the shipped ISA the mask costs 28 v_cmp_gt_i32 + 32 v_cndmask_b32 +
    #     33 s_and_b32 = 93 of the loop body's 643 instructions (14.5%), dead on
    #     255/256 of the iterations at prod. The loop body is 64 WMMA of 643
    #     instructions -- 10% matrix -- so issue slots, not matrix work, are what it is
    #     short of.
    def _body(acc, pre, qt, gh, do_mask, qt_n, gh_n, carry=True):
        # r16.i1.g48 moved qh/base_l into _ldqd with the LSE/delta loads.
        q0 = qt * fx.Int32(32)

        # r9.i1.g25 -- the masked body does NOT carry g21's prefetch. h14: the two
        # scf.for bodies had identically-shaped carried tuples and the allocator put
        # them in DISJOINT register blocks, duplicating ~200 VGPR. The masked body
        # runs G*nmaskp = 4 iterations at prod against the full body's ~508, so a
        # prefetch cannot pay for itself there anyway. With carry=False it loads its
        # own Q/dO and returns only the accumulators, so the two tuples stop being
        # the same shape and the full body's 128-dword prefetch block no longer has
        # to coexist with a masked-body copy of itself.
        if const_expr(carry):
            nxt = _ldqd(qt_n, gh_n)
        else:
            nxt = None
            pre = _ldqd(qt, gh)

        # r2.i2.g09 -- THE STAGING BURST IS GONE. The 32 query rows of Q and dO this
        # iteration needs were being read from global TWICE: once by the staging loop
        # (16 b128 per lane, each) and once by gfrag for the S/P GEMM -- 32 of the 64
        # loop-body buffer_load_b128. gfrag's halves tile the [32][D] LDS image
        # exactly (row hh*16+row, byte col half*16 + dt*64, second half at +32), so
        # the tile is written from the registers the GEMM already holds. No new
        # barrier, no new dependence: these stores sit where the staging loop sat,
        # ahead of the same fx.barrier().
        # g06's reuse (Q/dO fragments feed BOTH kv sub-tiles) now happens twice,
        # once per 16-query half, because g07 stages 32 query rows.
        def _stage(hh, qp, dp):
            xo = ((fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B)
                  + half * fx.Int32(16))
            for dt in range_constexpr(NDT):
                for u in range_constexpr(2):
                    o = xo + fx.Int32(dt * 64 + u * 32)
                    llvm_dialect.store(fx.as_ir_value(dp[dt][u]),
                                       create_llvm_ptr(lds_do + o, address_space=3))
                    llvm_dialect.store(fx.as_ir_value(qp[dt][u]),
                                       create_llvm_ptr(lds_q + o, address_space=3))

        def _qd(hh):
            qp = [(pre[4 + hh * 16 + dt * 2], pre[4 + hh * 16 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            dp = [(pre[4 + hh * 16 + 8 + dt * 2], pre[4 + hh * 16 + 8 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            return qp, dp

        # 0930 dkdv_trorder_b (QDO_BURST) -- the hh loop is split: all 32 Q/dO staging
        # stores are ONE fenced burst here, right after the prefetch, so none of them can
        # be sunk into the final DS group ahead of the tr16 loads (r29 sank the 16 hh=1
        # stores there). Same stores, same addresses, same values; only issue position.
        if const_expr(QDO_BURST):
            rocdl.sched_barrier(0)
            for hh in range_constexpr(2):
                _stage(hh, *_qd(hh))
            rocdl.sched_barrier(0)
        for hh in range_constexpr(2):
            qp, dp = _qd(hh)
            if const_expr(not QDO_BURST):
                _stage(hh, qp, dp)
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            # r16.i1.g48 -- prefetched an iteration early; see _ldqd. q_glob stays
            # (pure VALU, no load) because the causal mask below still needs it.
            q_glob = q0 + fx.Int32(hh * 16) + row
            lse_q = pre[hh * 2][0]
            del_q = pre[hh * 2 + 1][0]
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    s_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kf[kh][dt]), _ir(qfr[dt]), s_acc,
                        reuseA=False, reuseB=False).result
                    p_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vf[kh][dt]), _ir(dfr[dt]), p_acc,
                        reuseA=False, reuseB=False).result
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                # causal, BOTTOM-RIGHT: query q attends kv <= q + (Skv - Sq).
                kvb = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8)
                if const_expr(VF_KV):
                    # lab-kdq VF: exp2(s*scale*log2e - lse*log2e) as ONE fma per element,
                    # and dS stored WITHOUT the softmax scale (applied once to the dK
                    # accumulator at the store). Deterministic; not bitwise vs r19h.
                    _c1 = scale * fx.Float32(LOG2E)
                    _nl = lse_q * fx.Float32(-LOG2E)
                    tt = [fx.Float32(fx.fma(sv[si], _c1, _nl)) for si in range(8)]
                    if const_expr(do_mask):
                        tt = [((kvb + fx.Int32(si) > q_glob + cshift)
                               & (causal != fx.Int32(0))).select(fx.Float32(NEG), tt[si])
                              for si in range(8)]
                    pf = [_exp2(tt[si]) for si in range(8)]
                    p_l = [x.to(fx.BFloat16) for x in pf]
                    ds_l = [(pf[si] * (pv_[si] - del_q)).to(fx.BFloat16) for si in range(8)]
                else:
                    if const_expr(do_mask):
                        masked = [
                            ((kvb + fx.Int32(si) > q_glob + cshift)
                             & (causal != fx.Int32(0))
                             ).select(fx.Float32(NEG), sv[si] * scale)
                            for si in range(8)
                        ]
                    else:
                        masked = [sv[si] * scale for si in range(8)]
                    pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
                    p_l = [x.to(fx.BFloat16) for x in pf]
                    ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16)
                            for si in range(8)]
                off = ((fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B)
                       + fx.Int32(kh * 32) + half * fx.Int32(16))
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(p_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_p + off, address_space=3))
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(ds_l, dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ds + off, address_space=3))
        # r8.i1.g23+g24 -- fx.barrier() DELETED. block=(32,1,1): the workgroup is ONE
        # 32-lane wave, so s_barrier is semantically a no-op and LLVM already removes
        # it (0 `s_barrier` in the shipped ISA). What survives is the conservative
        # ALL-COUNTER waitcnt the backend inserts FOR the barrier before removing it --
        # `s_wait_loadcnt_dscnt 0x0`. Its dscnt half is the real LDS RAW and the backend
        # re-derives it from the memory dependence; its loadcnt half has no dependence
        # at all here and is exactly what truncates r7.i1.g21's prefetch cover to 299
        # instructions instead of a full iteration.

        # 32 queries staged, so the contraction is FULL: rows [lane_r] and
        # [lane_r+16] concatenate in lane into a v16 operand, no padding zeros.
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(base + fx.Int32(16 * rowb),
                                         address_space=3))), list(range(16)))

        # The B operands (dO, Q) are shared by BOTH kv sub-tiles -- one transpose
        # load feeds two WMMAs instead of one. The A operands differ only by a
        # 16-column (32-byte) offset into the same [q][kv] tile.
        if const_expr(TRORDER):
            # 0930 dkdv_trorder -- FENCED 4-REGION EMISSION. r29 issued all 32 b_do/b_q
            # tr loads first and the 8 a_p/a_ds loads last; DS completes in order, so the
            # first dV WMMA (needs a_p kh0) waited at s_wait_dscnt 0x6 for all 34 earlier
            # loads plus the 23 preceding stores (187 cyc/iter, report_dkdv.md L107).
            # Regions, each closed by sched_barrier(0):
            #   R1  a_p(kh0) + b_do                 (loads only)
            #   R2  a_ds(kh0) + b_q                 (loads, then fence, then 8 dV kh0 WMMA)
            #   R3  a_p(kh1) + a_ds(kh1)            (loads, then 8 dK kh0 WMMA)
            #   R4  16 kh1 WMMA
            # Every new[i] is still ONE WMMA on acc[i] with the same operands, so the
            # per-slot accumulation is unchanged: bitwise identical to r29.
            def _a(base, kh):
                col = lane_c * fx.Int32(2) + fx.Int32(kh * 32)
                return tr(base + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)

            def _b(base, dtile):
                c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
                return tr(base + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)

            new = [None] * (2 * NKV * NDO)

            def _mm(kh, a, bs, which):
                for dtile in range_constexpr(NDO):
                    k = (which * NKV + kh) * NDO + dtile
                    new[k] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(a), _ir(bs[dtile]), acc[k],
                        reuseA=(dtile > 0), reuseB=False).result

            rocdl.sched_barrier(0)
            a_p0 = _a(lds_p, 0)                                            # R1
            b_do = [_b(lds_do, dtile) for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            a_ds0 = _a(lds_ds, 0)                                          # R2
            b_q = [_b(lds_q, dtile) for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            _mm(0, a_p0, b_do, 0)
            rocdl.sched_barrier(0)
            a_p1 = _a(lds_p, 1)                                            # R3
            a_ds1 = _a(lds_ds, 1)
            _mm(0, a_ds0, b_q, 1)
            rocdl.sched_barrier(0)
            _mm(1, a_p1, b_do, 0)                                          # R4
            _mm(1, a_ds1, b_q, 1)
        else:
            b_do = []
            b_q = []
            for dtile in range_constexpr(NDO):
                c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
                b_do.append(tr(lds_do + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B))
                b_q.append(tr(lds_q + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B))
            new = [None] * (2 * NKV * NDO)
            for kh in range_constexpr(NKV):
                col = lane_c * fx.Int32(2) + fx.Int32(kh * 32)
                a_p = tr(lds_p + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)
                a_ds = tr(lds_ds + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)
                # r19.i1.g59 -- A-OPERAND REUSE. gfx1250 WMMA carries OPSEL[2]/OPSEL_HI[2]
                # as A-reuse / B-reuse hints (arch/gfx1250/isa.md); the intrinsic already
                # exposes them and this job has always passed False. The hint is only legal
                # when the PRECEDING matrix instruction used an IDENTICAL operand, so the
                # interleaved (a_p, a_ds) emission order could never use it. Split into two
                # runs of 8: within a run A is one 16x32 subtile held across all 8 dtiles
                # (corpus walk order: hold the operand that is >= 1 subtile wide, tie -> A),
                # so instructions 2..8 of each run may reuse it. B still changes every op.
                for dtile in range_constexpr(NDO):
                    new[kh * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(a_p), _ir(b_do[dtile]), acc[kh * NDO + dtile],
                        reuseA=(dtile > 0), reuseB=False).result
                for dtile in range_constexpr(NDO):
                    new[(NKV + kh) * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(a_ds), _ir(b_q[dtile]), acc[(NKV + kh) * NDO + dtile],
                        reuseA=(dtile > 0), reuseB=False).result
        # r8.i1.g23+g24 -- fx.barrier() DELETED. block=(32,1,1): the workgroup is ONE
        # 32-lane wave, so s_barrier is semantically a no-op and LLVM already removes
        # it (0 `s_barrier` in the shipped ISA). What survives is the conservative
        # ALL-COUNTER waitcnt the backend inserts FOR the barrier before removing it --
        # `s_wait_loadcnt_dscnt 0x0`. Its dscnt half is the real LDS RAW and the backend
        # re-derives it from the memory dependence; its loadcnt half has no dependence
        # at all here and is exactly what truncates r7.i1.g21's prefetch cover to 299
        # instructions instead of a full iteration.
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            st = list(carried)
            final = yield _body(st, None, qt0 + qi, ii - qi * G, True,
                                None, None, False)
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            jj = ii + fx.Int32(1)
            # h33 defect c (r19h port of round 24's clamp) -- k_dkdv's Q/dO descriptors
            # carry TRUE extents (r1.i7.g07), so this unclamped ii+1 is a live OOB READ
            # held back only by the descriptor: the last iteration prefetches a pair
            # that does not exist. Clamp exactly as k_dq's kvloop_full does. Value-neutral:
            # the loads a clamp redirects are consumed by iterations that never run.
            jj = (jj < n).select(jj, n - fx.Int32(1))
            qj = jj // G
            st = list(carried)
            final = yield _body(st[:NST], [fx.Vector(v) for v in st[NST:]],
                                qt0 + qi, ii - qi * G, False,
                                qt0 + qj, jj - qj * G)
        return final



    # lab-kdq KV_U2 -- qloop_full UNROLLED BY 2 (the k_dq U2 mechanism): two bodies per
    # scf.for trip so the second prefetch is allocated onto the dead carried set and the
    # 64 back-edge v_mov_b64 of the carried Q/dO prefetch go away. n2 = pairs; the odd
    # leftover iteration (only when G*pairs is odd; never at G = 4) runs in qloop_tail,
    # which loads its own Q/dO like qloop_mask but without the mask. Same iterations,
    # same order, same arithmetic: bitwise identical to the one-body loop.
    @flyc.jit
    def qloop_full2(state, n2, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n2), 1, init=state):
            ii = fx.Int32(it) * fx.Int32(2)
            i1 = ii + fx.Int32(1)
            jj = ii + fx.Int32(2)
            jj = (jj < n2 * fx.Int32(2)).select(jj, n2 * fx.Int32(2) - fx.Int32(1))
            qi = ii // G
            q1 = i1 // G
            qj = jj // G
            st = list(carried)
            mid = _body(st[:NST], [fx.Vector(v) for v in st[NST:]],
                        qt0 + qi, ii - qi * G, False, qt0 + q1, i1 - q1 * G)
            final = yield _body(mid[:NST], [fx.Vector(v) for v in mid[NST:]],
                                qt0 + q1, i1 - q1 * G, False,
                                qt0 + qj, jj - qj * G)
        return final

    @flyc.jit
    def qloop_tail(state, n, i0, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it) + i0
            qi = ii // G
            final = yield _body(list(carried), None, qt0 + qi, ii - qi * G, False,
                                None, None, False)
        return final

    def _qloop_full_any(state, n, qt0):
        if KV_U2:
            n2 = n // fx.Int32(2)
            out = qloop_full2(state, n2, qt0)
            return qloop_tail(list(out)[:NST], n - n2 * fx.Int32(2), n2 * fx.Int32(2), qt0)
        return qloop_full(state, n, qt0)
    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is
    # attended by the pair's SMALLEST query: kv0 + BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0 + BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0 + fx.Int32(BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NKV * NDO)]
    # r7.i1.g21: the prologue issue. Both loops start at the same (qt, gh) when
    # nmaskp == 0, and when it is not, qloop_mask's last body prefetches exactly
    # index G*nmaskp -- qloop_full's first iteration -- so the handoff is the state.
    if PARTIAL:
        # Exactly ONE query pair per q head is masked and it is the FIRST one
        # (qp_start), so it stays with split 0 and the mask-split of g19/g20 survives
        # untouched. The remaining `_fn` unmasked pairs are cut into contiguous
        # chunks; every pair in that range costs the same, so contiguous is balanced.
        _fn = nqp_eff - nmaskp
        _fn = (_fn < fx.Int32(0)).select(fx.Int32(0), _fn)
        _ch = (_fn + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = _fn - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        _mk = (sp != fx.Int32(0)).select(fx.Int32(0), nmaskp)
        _qt0 = qp_start + nmaskp + _off
        out = qloop_mask(init, G * _mk, qp_start)
        # h33 defect c -- the prologue issues iteration 0 unconditionally, even when the
        # chunk is empty (_cnt == 0). Clamp the query pair to the last legal one.
        # Value-neutral for every iteration that actually runs.
        _pc0 = _clampqt(_qt0)
        out = list(out) + [_ir(v) for v in _ldqd(_pc0, fx.Int32(0))]
        out = _qloop_full_any(out, G * _cnt, _qt0)
        base_o = ((sp * B_ + bat) * Skv * Hkv) * fx.Int32(D) + hkv * fx.Int32(D)
    else:
        out = qloop_mask(init, G * nmaskp, qp_start)
        # r9.i1.g25: the prologue issue now sits BETWEEN the loops, for qloop_full's
        # first iteration (qi = 0, gh = 0) at query pair qp_start + nmaskp.
        # h33 defect c -- same clamp on the non-split path.
        _pc0 = _clampqt(qp_start + nmaskp)
        out = list(out) + [_ir(v) for v in _ldqd(_pc0, fx.Int32(0))]
        out = _qloop_full_any(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
        base_o = bat * Skv * Hkv * fx.Int32(D) + hkv * fx.Int32(D)
    if PARTIAL:
        # fp32 workspace path (nsp > 1). LEFT EXACTLY AS IT WAS -- see r27.i3.g86 below
        # for why the wide-store staging cannot be applied here: gfx1250's transposing
        # LDS loads stop at 16-bit (`arch/gfx1250/isa.md:63-70`: TR16/TR8/TR6/TR4, no
        # 32-bit form), so an fp32 accumulator tile can only be staged row-major, and a
        # row-major stage costs 64 ds_write_b32 to save 48 buffer_store -- a net loss on
        # the issue roof (h16) before it is built.
        for kh in range_constexpr(NKV):
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                for si in range_constexpr(8):
                    kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
                    idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
                    _st1(ov[si], g_dv, idx, fx.Float32)
                    _st1(ok_[si], g_dk, idx, fx.Float32)
    else:
        # r27.i3.g86 -- STAGE THE dK/dV EPILOGUE THROUGH THE FINISHED LDS RING.
        # The WMMA output layout gives each lane ONE d-column and 8 kv rows, so the
        # bf16 epilogue could only ever emit the narrowest store the ISA has: 256
        # `buffer_store_b16` per workgroup (counted in round 27's own ISA dump --
        # `buffer_store_b64`/`b128`: zero). Each one covers 32 B of a 128 B line.
        #
        # The accumulator is transposed through LDS instead, and the transpose is
        # FREE in both directions:
        #   write  -- the image is COLUMN major, M[d][kv], so a lane's 8 kv rows for
        #             one d column are contiguous: one ds_write_b128 per d tile.
        #   read   -- `ds_load_tr16_b128` gives lane l element e `src[(l//16)*8+e, l%16]`
        #             (jobs/...HINTS.md:46), i.e. reading M column-major returns ROW
        #             major: lane l holds kv row l%16 and 8 CONSECUTIVE d columns.
        # 64 narrow stores per (kv half, output) become 8 ds_write_b128 + 8
        # ds_load_tr16_b128 + 8 buffer_store_b128: 256 -> 96 instructions, of which
        # VMEM 256 -> 32.
        #
        # It costs no LDS: the Q/dO ring (17408 B at offset 0) is dead here, and the
        # four staging images are 4 * 128 * EPI_CB = 24576 B, still inside segment 0,
        # so the allocation and therefore the 4-workgroup occupancy rung are unchanged.
        # EPI_CB = 48 B = 32 B of kv row + the g16 pad: 12 dwords walks c*12 mod 64,
        # 16 distinct bank groups, the same granularity X_ROW_B was chosen for.
        EPI_CB = 48
        g_dv8 = _bv(DV_, ndkv_b, fx.BFloat16, 8)
        g_dk8 = _bv(DK, ndkv_b, fx.BFloat16, 8)
        for kh in range_constexpr(NKV):
            lds_ev = _lds0 + fx.Int32((kh * 2 + 0) * 128 * EPI_CB)
            lds_ek = _lds0 + fx.Int32((kh * 2 + 1) * 128 * EPI_CB)
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                o = (fx.Int32(dtile * 16) + row) * fx.Int32(EPI_CB) + half * fx.Int32(16)
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(
                        [ov[si].to(fx.BFloat16) for si in range_constexpr(8)],
                        dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ev + o, address_space=3))
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(
                        [ok_[si].to(fx.BFloat16) for si in range_constexpr(8)],
                        dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ek + o, address_space=3))
            # lane_r / lane_c are the transposing-load address form already used by
            # `tr()` in the body: lane l addresses row (l//16)*8 + l%8, column
            # ((l//8)%2)*8, and receives column l%16 of rows (l//16)*8 + e.
            gt = base_kv + (kv0 + fx.Int32(kh * 16) + row) * rs_kv
            for sub in range_constexpr(NDO):
                a = ((fx.Int32(sub * 16) + lane_r) * fx.Int32(EPI_CB)
                     + lane_c * fx.Int32(2))
                vv = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
                kk = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
                t = gt + fx.Int32(sub * 2) + half
                _stv([vv[e] for e in range_constexpr(8)], g_dv8, t, fx.BFloat16)
                _stv([kk[e] for e in range_constexpr(8)], g_dk8, t, fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
           LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
           scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
           G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32, B_: fx.Int32):
    _dkdv_impl(False, Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv,
               G, nqt, cshift, causal, B_, fx.Int32(1))


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dkdv_sp(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
              LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
              scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32,
              Hkv: fx.Int32, G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32,
              causal: fx.Int32, B_: fx.Int32, nsp: fx.Int32):
    _dkdv_impl(True, Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv,
               G, nqt, cshift, causal, B_, nsp)


@flyc.jit
def launch_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
           nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream)


@flyc.jit
def launch_dkdv_sp(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                   Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
                   G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                   nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, nsp: fx.Int32,
                   nxs: fx.Int32, stream: fx.Stream):
    # nxs = nhkv * nsp, computed on the host: grid.x carries the split index in its low
    # digits so the kv tile stays on grid.y and longest-first dispatch (g03) survives.
    k_dkdv_sp(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
              nqt, cshift, causal, nb, nsp).launch(
        grid=(nxs, nblk, nb), block=(32, 1, 1), stream=stream)


# ===================================================================== dq =========
KV_STEP = 32                 # two 16-kv tiles: exactly one WMMA contraction of dS
NKT = KV_STEP // 16
BLOCK_Q = 64                 # r3.i4.g13: next rung of g10 (was 32)
NQ = BLOCK_Q // 16


# r17.i1.g52 -- SPLIT-K OVER k_dq's kv LOOP, the k_dq mirror of r13.i1.g42. k_dq's
# workgroup count is ceil(Sq/BLOCK_Q)*Hq*B and one workgroup is one wave32, so at
# 1 wave/SIMD the device holds 1024 at once. prod launches 16384 and proxy 2048 --
# both already saturate the dispatcher and keep nsp_q = 1 and the shipped bf16 path,
# byte for byte. `fast` launches 128 onto 1024 SIMDs: 87.5% of the machine is idle
# while k_dq takes 49.82% of that shape's kernel time
# (rounds/017/1-profiling/kernel.yaml:55). Splitting is the only thing that can add
# workgroups to a grid that small. PARTIAL=True cuts the UNMASKED kv range `nsp` ways,
# each split writing its own fp32 [nsp, B, Sq, Hq, D] workspace slice, folded by
# k_redsp_q. PARTIAL=False is the shipped path and is what prod and proxy use.
def _dq_impl(PARTIAL, Q, K, V, DO, LSE, DEL, DQ,
             scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, B_, nsp):
    """One wave owns BLOCK_Q queries of one q head and streams every 32-key block.

    grid = (Sq/16, Hq, B). The A operand of the dQ GEMM comes free: the two kv-tile dS^T
    accumulators concatenate in lane into one v16, so the only LDS traffic is staging K.
    """
    lane = fx.Int32(fx.thread_idx.x)
    # r12.i2.g40 -- LONGEST-FIRST DISPATCH, the k_dq mirror of r1.i3.g03. Under
    # bottom-right causality query tile t streams ceil((t*BLOCK_Q + BLOCK_Q + cshift)
    # / KV_STEP) kv blocks, so work grows MONOTONICALLY with t. The grid was
    # (Sq/BLOCK_Q, Hq, B) with the query tile FASTEST-varying and ASCENDING, i.e.
    # shortest-first with the longest tiles dispatched last -- the exact inverse of
    # what g03 measured at +2.60% / +2.68% prod for k_dkdv, where making the cheap
    # axis fastest-varying and the expensive axis slow and longest-first was the whole
    # change. Here: q head becomes grid.x (cheap, uniform work, 32 wide), the query
    # tile becomes grid.y and is walked DESCENDING, so the longest tiles are issued
    # first and the tail of the dispatch is made of the cheapest ones.
    # Purely a dispatch-order change: every workgroup computes exactly the same dQ
    # rows from the same inputs in the same order, no atomics -- bitwise identical.
    # r13.i2.g43 -- XCD-MAJOR q-head remap. /sys/class/kfd/kfd/topology/nodes/2/
    # properties reports `num_xcc 8` for this gfx1250 (simd_count 1024,
    # gfx_target_version 120500): the corpus said the XCD count was undocumented, the
    # KFD topology says it is 8. Workgroups are handed to XCDs round-robin on the
    # linearised id (x fastest), so with grid.x = Hq = 32 the q heads on one XCD are
    # {c, c+8, c+16, c+24} -- four DIFFERENT kv heads, so each XCD streams 4 kv heads'
    # K and V through its slice of L2. Permuting x by (x%8)*(Hq/8) + x/8 puts q heads
    # {4c..4c+3} on XCD c instead: exactly one kv head per XCD, K/V footprint per XCD
    # cut 4x. This is a BIJECTION of grid.x whenever Hq % 8 == 0 (32 at proxy/prod, 8
    # at fast where it degenerates to the identity), so it is a pure relabelling --
    # same workgroups, same inputs, same order inside each -- bitwise identical.
    # grid.x carries uniform work, so g40's longest-first on grid.y is untouched.
    # r17.i1.g52: grid.x carries the split index in its low digits, exactly as
    # launch_dkdv_sp does, so the kv tile stays on grid.y and g40's longest-first
    # descending dispatch survives. The g43 XCD remap below is applied to the DECODED
    # q head, not to the fused id, so it is still the same bijection of [0, Hq).
    _nx = fx.Int32(8)
    if PARTIAL:
        _fx = fx.Int32(fx.block_idx.x)          # q head * nsp + split
        _bx = _fx // nsp
        sp = _fx - _bx * nsp
    else:
        _bx = fx.Int32(fx.block_idx.x)
        sp = fx.Int32(0)
    qh = (Hq % _nx == fx.Int32(0)).select((_bx % _nx) * (Hq // _nx) + _bx // _nx, _bx)
    bid = ((Sq + fx.Int32(BLOCK_Q - 1)) // fx.Int32(BLOCK_Q)
           - fx.Int32(1) - fx.Int32(fx.block_idx.y))    # query tile, descending
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(BLOCK_Q)
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    if PARTIAL:
        # [nsp, B, Sq, Hq, D] fp32. TRUE byte extent, not the flat 1 GiB the bf16 path
        # carries: the workspace is the one buffer in this kernel whose size depends on
        # nsp, and an over-read of it would land in another split's slice.
        # h33 defect b -- this product is evaluated in Int32. At prod with nsp_q = 8 it
        # is 8*4*8192*32*512 = 4.398e11, which wraps to 0, and a descriptor with
        # num_records 0 drops EVERY dQ write silently. Compute the extent in Int64.
        g_dq = _bv(DQ, (nsp.to(fx.Int64) * B_.to(fx.Int64) * Sq.to(fx.Int64)
                        * Hq.to(fx.Int64) * fx.Int64(D * 4)), fx.Float32)
    else:
        g_dq = _bv(DQ, 1 << 30, fx.BFloat16)   # r3.i2.g11: bf16 output, no host .to()

    rs_q = Hq * fx.Int32(DV8)
    rs_kv = Hkv * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    def gfrag2(buf, base, rs, r, dt):
        """gfrag's two halves before the shuffle: cols [c, c+8) and [c+16, c+24) of
        row r+row, where c = half*8 + dt*32. Each is exactly one 16-byte LDS chunk."""
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8),
                _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    # Q, dO, lse and delta are invariant over the whole kv loop: hoist them.
    # r3.i1.g10 -- the workgroup owns BLOCK_Q = 32 queries as NQ 16-row halves. K and V
    # (global fragments AND the LDS staging AND the tr16 transpose of it) are invariant
    # over queries, so every one of them now serves twice the work.
    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
          for qh_ in range(NQ)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
           for qh_ in range(NQ)]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQ)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQ)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    # r9.i2.g26 -- PREFETCH, the g21 mechanism in k_dq. The 32 K/V buffer_load_b128
    # an iteration consumes are issued one iteration EARLY and carried in the
    # kvloop_full scf.for state. Built with g25's lesson already applied: only the
    # FULL loop carries it. The masked loop (98.4% dead at prod, a handful of
    # iterations) loads its own, so the two carried tuples are not the same shape and
    # the allocator has no identical block to duplicate.
    def _ldkv(kv0):
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                a, b = gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                c, d = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                out += [a, b, c, d]
        return out

    # r6.i1.g19 -- MASK-SPLIT. The causal compare-and-select was emitted for EVERY kv
    # block, but under bottom-right causality only the last one or two blocks a query
    # block touches straddle the diagonal; in every earlier block the predicate
    # `kv > q + cshift` is provably false for all 64 elements. Counted in the shipped
    # ISA, the mask costs 56 v_cmp_gt_i32 + 64 v_cndmask_b32 + 64 s_and_b32 = 184 of the
    # loop body's 782 instructions (23.5%), and it is dead on 98.4% of the iterations at
    # the prod shape (sum(2*bid) / sum(2*bid+2) over bid = 0..127). The loop is split:
    # [0, nfull) with no mask emitted at all, then [nfull, nkvt_eff) with it. Same
    # blocks, same order, same arithmetic on every live element -- BITWISE IDENTICAL,
    # only the dead predicate is gone. The kernel's loop body is 96 WMMA out of 782
    # instructions (12% matrix), so what it is short of is issue slots, not matrix work.
    def _body(acc, pre, kv0, do_mask, kv0_n, carry=True):
        if const_expr(carry):
            nxt = _ldkv(kv0_n)
        else:
            nxt = None
            pre = _ldkv(kv0)

        # r3.i3.g12 -- THE K STAGING LOOP IS GONE. K's 32 rows x D were read from
        # global TWICE: once by this loop (16 b128 per lane) and once by gfrag for
        # the S = K Q^T GEMM -- 16 of the 48 loop-body buffer_load_b128. gfrag's two
        # halves tile the [32][D] LDS image exactly (row kt*16+row, byte col
        # half*16 + dt*64, second half at +32), so the tile is now written from the
        # registers the S GEMM already holds. Same rows, same bytes, same order:
        # bit-identical. No new barrier (the stores sit where the staging loop sat,
        # ahead of the same fx.barrier()) and no new dependence. This is r2.i2.g09's
        # mechanism in the other kernel; X_ROW_B's g16 padding, which was already
        # shipped, is what makes these 16 ds_store_b128 conflict-free.
        ds_halves = [[] for _ in range_constexpr(NQ)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQ)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQ)]
            ko = ((fx.Int32(kt * 16) + row) * fx.Int32(X_ROW_B)
                  + half * fx.Int32(16))
            for dt in range_constexpr(NDT):
                # ONE pair of global K/V fragments now feeds BOTH query halves.
                # r9.i2.g26: they come from the carried prefetch, not from a load here.
                pi = (kt * NDT + dt) * 4
                kp = (pre[pi], pre[pi + 1])
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32),
                                        address_space=3))
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQ):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_],
                        reuseA=False, reuseB=False).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_],
                        reuseA=False, reuseB=False).result
            for qh_ in range_constexpr(NQ):
                sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                if const_expr(do_mask):
                    masked = [
                        ((kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                          > q_glob[qh_] + cshift) & (causal != fx.Int32(0))
                         ).select(fx.Float32(NEG), sv[si] * scale)
                        for si in range(8)
                    ]
                else:
                    masked = [sv[si] * scale for si in range(8)]
                pf = [_exp2((masked[si] - lse_q[qh_]) * fx.Float32(LOG2E))
                      for si in range(8)]
                ds_halves[qh_].append(
                    [(pf[si] * (pv_[si] - del_q[qh_]) * scale).to(fx.BFloat16)
                     for si in range(8)])
        # FREE: the two kv-tile accumulators concatenate in-lane into the dS A-operand.
        a_ds = [fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                        dtype=fx.BFloat16)
                for qh_ in range_constexpr(NQ)]
        # r8.i1.g23+g24 -- fx.barrier() DELETED. block=(32,1,1): the workgroup is ONE
        # 32-lane wave, so s_barrier is semantically a no-op and LLVM already removes
        # it (0 `s_barrier` in the shipped ISA). What survives is the conservative
        # ALL-COUNTER waitcnt the backend inserts FOR the barrier before removing it --
        # `s_wait_loadcnt_dscnt 0x0`. Its dscnt half is the real LDS RAW and the backend
        # re-derives it from the memory dependence; its loadcnt half has no dependence
        # at all here and is exactly what truncates r7.i1.g21's prefetch cover to 299
        # instructions instead of a full iteration.

        new = [None] * (NQ * NDO)
        for dtile in range_constexpr(NDO):
            base = (lds_k + lane_r * fx.Int32(X_ROW_B)
                    + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2))
            b_k = fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3))),
                list(range(16)))
            for qh_ in range_constexpr(NQ):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_k), acc[qh_ * NDO + dtile],
                    reuseA=False, reuseB=False).result
        # r8.i1.g23+g24 -- fx.barrier() DELETED. block=(32,1,1): the workgroup is ONE
        # 32-lane wave, so s_barrier is semantically a no-op and LLVM already removes
        # it (0 `s_barrier` in the shipped ISA). What survives is the conservative
        # ALL-COUNTER waitcnt the backend inserts FOR the barrier before removing it --
        # `s_wait_loadcnt_dscnt 0x0`. Its dscnt half is the real LDS RAW and the backend
        # re-derives it from the memory dependence; its loadcnt half has no dependence
        # at all here and is exactly what truncates r7.i1.g21's prefetch cover to 299
        # instructions instead of a full iteration.
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new

    if PARTIAL:
        # r17.i1.g52: identical body, with the iteration index rebased onto this
        # split's chunk. The prefetch clamp stays RELATIVE (jj in [0, n)) and the base
        # is added after it, so it still never reaches past this chunk's last block.
        @flyc.jit
        def kvloop_full(state, n, it0):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(st[:NQ * NDO],
                                    [fx.Vector(v) for v in st[NQ * NDO:]],
                                    (ii + it0) * fx.Int32(KV_STEP), False,
                                    (jj + it0) * fx.Int32(KV_STEP))
            return final
    else:
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                # r9.i2.g26: k_dq's descriptors carry a flat 1 GiB num_records (no g07
                # clamp here), so a prefetch past the last block would be a live OOB
                # read. The last iteration re-issues its own block instead.
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(st[:NQ * NDO],
                                    [fx.Vector(v) for v in st[NQ * NDO:]],
                                    ii * fx.Int32(KV_STEP), False,
                                    jj * fx.Int32(KV_STEP))
            return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), None,
                                (fx.Int32(it) + it0) * fx.Int32(KV_STEP), True,
                                None, False)
        return final


    # CAUSAL TILE SKIP, the mirror of k_dkdv's. Queries [q0, q0+16) reach at most key
    # q0 + 15 + cshift, so every kv block past that is fully masked and contributes
    # exactly zero to dQ. Bit-identical, not an approximation.
    _lim = (q0 + fx.Int32(BLOCK_Q) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)

    # Block `it` is fully unmasked iff its largest key index is attended by this
    # block's SMALLEST query: it*KV_STEP + KV_STEP-1 <= q0 + cshift, i.e.
    # it < (q0 + cshift + 1) // KV_STEP. cshift = Skv - Sq can be negative (Sq > Skv is
    # a ut shape), so the numerator is guarded before the division.
    _t = q0 + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NQ * NDO)]
    if PARTIAL:
        # r17.i1.g52. The UNMASKED range [0, nfull) is cut into nsp contiguous chunks;
        # every block in it costs the same, so contiguous is balanced. The masked tail
        # [nfull, nkvt_eff) is at most two blocks and is NOT cut -- it stays whole on
        # the LAST split, which makes the fold order over sp = 0..nsp-1 exactly the
        # ascending kv-block order the unsplit kernel accumulates in.
        _ch = (nfull + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = nfull - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        # The prologue prefetch is issued whether or not this split has work, and k_dq's
        # K/V descriptors carry a flat 1 GiB num_records, so an empty split at _off >=
        # nkvt_eff would issue a LIVE out-of-bounds read. Clamp the prefetch address;
        # the loaded value is dead in that case.
        _pf = (_off < nkvt_eff).select(_off, nkvt_eff - fx.Int32(1))
        _mk = (sp == nsp - fx.Int32(1)).select(nkvt_eff - nfull, fx.Int32(0))
        out = kvloop_full(init + [_ir(v) for v in _ldkv(_pf * fx.Int32(KV_STEP))],
                          _cnt, _off)
        out = kvloop_mask(list(out)[:NQ * NDO], _mk, nfull)
        base_o = ((sp * B_ + bat) * Sq * Hq) * fx.Int32(D) + qh * fx.Int32(D)
    else:
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], nfull)
        out = kvloop_mask(list(out)[:NQ * NDO], nkvt_eff - nfull, nfull)
        base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQ):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                # The address is written out TWICE on purpose. Hoisting it into a shared
                # `_idx` emits the same ops in a different ORDER (the address algebra
                # ahead of the vector.extract instead of behind it), and that alone moved
                # the non-split k_dq's final ISA by 9 instructions -- which breaks this
                # candidate's byte-identical-prod gate. Same ops, same order, same binary.
                if PARTIAL:
                    _st1(ov[si], g_dq,
                         base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                         fx.Float32)
                else:
                    _st1(ov[si].to(fx.BFloat16), g_dq,
                         base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                         fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
         LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
         scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
         G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    _dq_impl(False, Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
             nkvt, cshift, causal, fx.Int32(1), fx.Int32(1))


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq_sp(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
            LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
            scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32,
            Hkv: fx.Int32, G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32,
            causal: fx.Int32, B_: fx.Int32, nsp: fx.Int32):
    _dq_impl(True, Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
             nkvt, cshift, causal, B_, nsp)


@flyc.jit
def launch_dq(Q, K, V, DO, LSE, DEL, DQ, scale: fx.Float32,
              Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
              nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
              nblk: fx.Int32, nhq: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dq(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
         nkvt, cshift, causal).launch(
        grid=(nhq, nblk, nb), block=(32, 1, 1), stream=stream)   # r12.i2.g40


@flyc.jit
def launch_dq_sp(Q, K, V, DO, LSE, DEL, DQ, scale: fx.Float32,
                 Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
                 G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                 nblk: fx.Int32, nhq: fx.Int32, nb: fx.Int32, nsp: fx.Int32,
                 nxq: fx.Int32, stream: fx.Stream):
    # nxq = nhq * nsp, computed on the host -- same shape as launch_dkdv_sp's nxs.
    k_dq_sp(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
            nkvt, cshift, causal, nb, nsp).launch(
        grid=(nxq, nblk, nb), block=(32, 1, 1), stream=stream)



# ================================================================ dq (lab-kdq) =====
# lab-kdq -- GROUPED / LOW-VGPR k_dq ("k_dqg"), used by impl.py on the non-split path
# (nsp_q == 1: proxy and prod). The split path (fast) keeps k_dq_sp unchanged.
#
#   DQ_NW    waves per workgroup. Wave w of group x owns q head x*DQ_NW + w, so with
#            DQ_NW == G (= 4 on every scored shape) the NW waves of one workgroup are the
#            G q heads that share ONE kv head, on the same query tile: they stream the
#            IDENTICAL K/V rows with the IDENTICAL trip count, on the same CU, so 3 of 4
#            K/V fetches can hit the CU-local vector cache instead of L2. There is no
#            cross-wave communication at all: every wave has its own LDS K slice and its
#            own registers, so there is no barrier anywhere (the 4-wave barrier cost of
#            h44-h53 does not apply).
#   DQ_BQW   queries per wave (64 = r19h; 32 halves the accumulators and the Q/dO
#            fragments so that, without the K/V prefetch, the wave fits <= 512 VGPR,
#            i.e. 2 waves per SIMD).
#   DQ_PF    carry the one-iteration K/V prefetch (r9.i2.g26) or load in-iteration.
#   DQ_DFUSE compute delta = rowsum(dO * O) in the prologue from the dO fragments the
#            wave already holds plus one O fragment load, use it in-register, and write
#            it to DEL for k_dkdv (which then runs AFTER k_dqg); k_delta is not launched.
DQ_NW = 1
DQ_BQW = 64
DQ_PF = True
DQ_DFUSE = False
DQ_U2 = True
# dqg_tailpf: in the U2 loop, issue body 2's K/V prefetch (block ii+2, the one that lands
# in the loop-carried registers) inside BODY 1, right where body 1's carried K/V set dies,
# instead of at the end of body 2 (lead 2-18 WMMA in r29). Same _ldkv(jj*KV_STEP) values,
# same clamp -> bitwise. None = r29; "a" = one 32-load clump after body 1's S/dP loop;
# "b" = per kt, 16 loads once that kt half of the carried set is dead, fenced with a
# sched_barrier(DQ_TAILPF_MASK) on both sides.
DQ_TAILPF = "b"
# sched_barrier mask around the prefetch (only VALU|SALU|TRANS may cross). Tried at compile:
# 0x78F (spec; only VMEM blocked) is a no-op -- the fence floats past the WMMAs and the
# loads sink to the end of body 2 exactly as in r29. 0x786 (VMEM+WMMA blocked, DS free)
# pins the loads but lets body 2's K ds_stores rise to WMMA ~8, so the head clump is
# waited on ~8 WMMAs after issue. 0x406 also blocks DS and keeps the head-clump lead.
DQ_TAILPF_MASK = 0x406


def _dqg_impl(Q, K, V, DO, O, LSE, DEL, DQ,
              scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal):
    NW, BQW, NQW = DQ_NW, DQ_BQW, DQ_BQW // 16
    tid = fx.Int32(fx.thread_idx.x)
    if NW > 1:
        lane = tid % fx.Int32(WAVE)
        wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                            (tid // fx.Int32(WAVE)).ir_value()))
    else:
        lane = tid
        wave = fx.Int32(0)
    # Head GROUP on grid.x; the r13.i2.g43 XCD-major remap is applied to the group index
    # (a bijection of [0, ngrp) whenever ngrp % 8 == 0, identity otherwise), so with
    # Hq = 32: NW = 4 -> 8 groups, group c = kv head c on XCD c; NW = 2 -> 16 groups,
    # XCD c gets groups 2c, 2c+1 = q heads 4c..4c+3 = kv head c; NW = 1 = r19h's remap.
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    ngrp = Hq // fx.Int32(NW)
    grp = (ngrp % _nx == fx.Int32(0)).select((_gx % _nx) * (ngrp // _nx) + _gx // _nx, _gx)
    qh = grp * fx.Int32(NW) + wave
    bid = (Sq // fx.Int32(BQW) - fx.Int32(1) - fx.Int32(fx.block_idx.y))  # descending
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(BQW)
    hkv = qh // G

    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_k = _bv(K, 1 << 30, fx.BFloat16, 8)
    g_v = _bv(V, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.BFloat16)
    if DQ_DFUSE:
        g_o = _bv(O, 1 << 30, fx.BFloat16, 8)

    rs_q = Hq * fx.Int32(DV8)
    rs_kv = Hkv * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(NW * KV_STEP * X_ROW_B)
    lds_k = fx.Int32(fx.ptrtoint(smem.peek().ptr)) + wave * fx.Int32(KV_STEP * X_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    def gfrag2(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8),
                _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
          for qh_ in range(NQW)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
           for qh_ in range(NQW)]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQW)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    if DQ_DFUSE:
        # gfrag gives lane (row, half) columns [8h+32dt, +8) and [8h+32dt+16, +8) of its
        # row for dt = 0..3: 64 of the 128 columns; the lane with the other `half` (lane
        # ^ 16) holds the other 64. So delta(row) = own partial + partner's partial.
        # fp32 addition is commutative, so both lanes hold the bitwise-same total and
        # both store it to the same address (a benign same-value duplicate).
        del_q = []
        for qh_ in range(NQW):
            e0 = fx.Float32(0.0)
            e1 = fx.Float32(0.0)
            for dt in range(NDT):
                o16 = gfrag(g_o, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt)
                d16 = dof[qh_][dt]
                for i in range(8):
                    e0 = e0 + fx.Float32(d16[2 * i]) * fx.Float32(o16[2 * i])
                    e1 = e1 + fx.Float32(d16[2 * i + 1]) * fx.Float32(o16[2 * i + 1])
            part = e0 + e1
            tot = part + fx.gpu.shuffle_xor(part, 16, WAVE)
            _st1(tot, g_del, base_l + q_glob[qh_], fx.Float32)
            del_q.append(tot)
    else:
        del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    _c1q = scale * fx.Float32(LOG2E)                    # lab-kdq VF_Q
    _nlq = [lse_q[qh_] * fx.Float32(-LOG2E) for qh_ in range(NQW)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    def _ldkv_kt(kv0, kt):
        out = []
        for dt in range_constexpr(NDT):
            a, b = gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
            c, d = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
            out += [a, b, c, d]
        return out

    def _ldkv(kv0):
        out = []
        for kt in range_constexpr(NKT):
            out += _ldkv_kt(kv0, kt)
        return out

    def _body(acc, pre, kv0, do_mask, kv0_n, carry=True, pf2=None, nxt_in=None):
        # dqg_tailpf. pf2: kv0 of an EXTRA prefetch issued inside this body, returned
        # after `nxt`. nxt_in: this body's prefetch was already issued (by the previous
        # body's pf2); return it instead of loading.
        nxt2 = []
        if const_expr(nxt_in is not None):
            nxt = nxt_in
        elif const_expr(carry):
            nxt = _ldkv(kv0_n)
        else:
            nxt = None
            pre = _ldkv(kv0)
        ds_halves = [[] for _ in range_constexpr(NQW)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            ko = ((fx.Int32(kt * 16) + row) * fx.Int32(X_ROW_B)
                  + half * fx.Int32(16))
            for dt in range_constexpr(NDT):
                pi = (kt * NDT + dt) * 4
                kp = (pre[pi], pre[pi + 1])
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32),
                                        address_space=3))
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQW):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_],
                        reuseA=False, reuseB=False).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_],
                        reuseA=False, reuseB=False).result
            if const_expr(pf2 is not None and DQ_TAILPF == "b"):
                # the kt half of `pre` is dead here: its K ds_store and its S/dP WMMAs
                # are all above. Fence = no VMEM and no WMMA may cross (VALU/DS may).
                rocdl.sched_barrier(DQ_TAILPF_MASK)
                nxt2 += _ldkv_kt(pf2, kt)
                rocdl.sched_barrier(DQ_TAILPF_MASK)
            for qh_ in range_constexpr(NQW):
                sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                if const_expr(VF_Q):
                    tt = [fx.Float32(fx.fma(sv[si], _c1q, _nlq[qh_])) for si in range(8)]
                    if const_expr(do_mask):
                        tt = [((kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                                > q_glob[qh_] + cshift) & (causal != fx.Int32(0))
                               ).select(fx.Float32(NEG), tt[si]) for si in range(8)]
                    pf = [_exp2(tt[si]) for si in range(8)]
                    ds_halves[qh_].append(
                        [(pf[si] * (pv_[si] - del_q[qh_])).to(fx.BFloat16)
                         for si in range(8)])
                else:
                    if const_expr(do_mask):
                        masked = [
                            ((kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                              > q_glob[qh_] + cshift) & (causal != fx.Int32(0))
                             ).select(fx.Float32(NEG), sv[si] * scale)
                            for si in range(8)
                        ]
                    else:
                        masked = [sv[si] * scale for si in range(8)]
                    pf = [_exp2((masked[si] - lse_q[qh_]) * fx.Float32(LOG2E))
                          for si in range(8)]
                    ds_halves[qh_].append(
                        [(pf[si] * (pv_[si] - del_q[qh_]) * scale).to(fx.BFloat16)
                         for si in range(8)])
        if const_expr(pf2 is not None and DQ_TAILPF == "a"):
            rocdl.sched_barrier(DQ_TAILPF_MASK)
            nxt2 = _ldkv(pf2)
            rocdl.sched_barrier(DQ_TAILPF_MASK)
        a_ds = [fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                        dtype=fx.BFloat16)
                for qh_ in range_constexpr(NQW)]
        new = [None] * (NQW * NDO)
        for dtile in range_constexpr(NDO):
            base = (lds_k + lane_r * fx.Int32(X_ROW_B)
                    + (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2))
            b_k = fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3))),
                list(range(16)))
            for qh_ in range_constexpr(NQW):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_k), acc[qh_ * NDO + dtile],
                    reuseA=False, reuseB=False).result
        if const_expr(carry):
            return new + [_ir(v) for v in nxt] + [_ir(v) for v in nxt2]
        return new

    if DQ_PF and DQ_U2 and DQ_TAILPF is not None:
        # dqg_tailpf: as U2 below, but body 2's prefetch (block jj = ii+2, clamped) is
        # issued by body 1 (pf2) where body 1's carried K/V set dies, and handed to
        # body 2 as nxt_in. Values and clamp identical to U2 -> bitwise.
        NP = 4 * NKT * NDT
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it) * fx.Int32(2)
                jj = ii + fx.Int32(2)
                jj = (jj < n * fx.Int32(2)).select(jj, n * fx.Int32(2) - fx.Int32(1))
                st = list(carried)
                mid = _body(st[:NQW * NDO], [fx.Vector(v) for v in st[NQW * NDO:]],
                            ii * fx.Int32(KV_STEP), False,
                            (ii + fx.Int32(1)) * fx.Int32(KV_STEP),
                            pf2=jj * fx.Int32(KV_STEP))
                final = yield _body(mid[:NQW * NDO],
                                    [fx.Vector(v) for v in mid[NQW * NDO:NQW * NDO + NP]],
                                    (ii + fx.Int32(1)) * fx.Int32(KV_STEP), False,
                                    None,
                                    nxt_in=[fx.Vector(v) for v in mid[NQW * NDO + NP:]])
            return final
    elif DQ_PF and DQ_U2:
        # lab-kdq U2 -- the full loop UNROLLED BY 2. With one body per scf.for trip the
        # carried prefetch is loaded while the previous one is still being consumed, so
        # it lands in fresh registers and the back-edge copies it into the loop-carried
        # ones (65 v_mov_b64 per trip). Two bodies per trip: the first body's prefetch
        # goes to a temporary set, the second body's prefetch is issued after the carried
        # set is dead and can be allocated onto it directly. n = number of PAIRS; blocks
        # [0, 2n); the odd leftover block (if any) runs in kvloop_mask, whose causal
        # predicate is identically false on it -- same blocks, same order, bitwise.
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it) * fx.Int32(2)
                jj = ii + fx.Int32(2)
                jj = (jj < n * fx.Int32(2)).select(jj, n * fx.Int32(2) - fx.Int32(1))
                st = list(carried)
                mid = _body(st[:NQW * NDO], [fx.Vector(v) for v in st[NQW * NDO:]],
                            ii * fx.Int32(KV_STEP), False,
                            (ii + fx.Int32(1)) * fx.Int32(KV_STEP))
                final = yield _body(mid[:NQW * NDO],
                                    [fx.Vector(v) for v in mid[NQW * NDO:]],
                                    (ii + fx.Int32(1)) * fx.Int32(KV_STEP), False,
                                    jj * fx.Int32(KV_STEP))
            return final
    elif DQ_PF:
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                ii = fx.Int32(it)
                jj = ii + fx.Int32(1)
                jj = (jj < n).select(jj, n - fx.Int32(1))
                st = list(carried)
                final = yield _body(st[:NQW * NDO],
                                    [fx.Vector(v) for v in st[NQW * NDO:]],
                                    ii * fx.Int32(KV_STEP), False,
                                    jj * fx.Int32(KV_STEP))
            return final
    else:
        @flyc.jit
        def kvloop_full(state, n):
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                final = yield _body(list(carried), None,
                                    fx.Int32(it) * fx.Int32(KV_STEP), False,
                                    None, False)
            return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), None,
                                (fx.Int32(it) + it0) * fx.Int32(KV_STEP), True,
                                None, False)
        return final

    _lim = (q0 + fx.Int32(BQW) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0 + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NQW * NDO)]
    if DQ_PF and DQ_U2:
        npair = nfull // fx.Int32(2)
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], npair)
        nfull = npair * fx.Int32(2)             # the mask loop takes the odd leftover
    elif DQ_PF:
        # prologue prefetch of block 0: always a real block (nkvt_eff >= 1).
        out = kvloop_full(init + [_ir(v) for v in _ldkv(fx.Int32(0))], nfull)
    else:
        out = kvloop_full(init, nfull)
    out = kvloop_mask(list(out)[:NQW * NDO], nkvt_eff - nfull, nfull)
    base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            ovs = ([ov[si] * scale for si in range(8)] if VF_Q
                   else [ov[si] for si in range(8)])   # lab-kdq VF_Q: scale at store
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                _st1(ovs[si].to(fx.BFloat16), g_dq,
                     base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                     fx.BFloat16)


@flyc.kernel(known_block_size=[DQ_NW * 32, 1, 1])
def k_dqg(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor, O: fx.Tensor,
          LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
          scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
          G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    _dqg_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
              nkvt, cshift, causal)


@flyc.jit
def launch_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale: fx.Float32,
               Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
               nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
               nblk: fx.Int32, ngrp: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    # grid = (Hq / DQ_NW head groups, Sq / DQ_BQW query tiles, B)
    k_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
          nkvt, cshift, causal).launch(
        grid=(ngrp, nblk, nb), block=(DQ_NW * 32, 1, 1), stream=stream)

_G07_CLAMP = True   # r1.i7.g07: k_dkdv takes the batch count as its last argument


# ================================================================= redsp ==========
# r13.i1.g42(c) -- round 14. The split-K fold-back used to be four torch launches on the
# host: `dkp.sum(0)` -> fp32 temp -> `.to(bf16)`, twice. Round 14 PRICED it for the first
# time (rounds/014/1-opt/raw/reduce.out, and facts.md had recorded that nobody ever had):
# at the `fast` shape with nsp=8 it is **0.0287 ms against a 0.156 ms whole-op time --
# 18.4% of the shape** -- and it is almost FLAT in nsp (nsp=1 0.0246, nsp=16 0.0327), i.e.
# it is LAUNCH-AND-LATENCY bound, not bandwidth bound: 33.6 MB in 0.0327 ms is 1.19 TB/s
# against a part whose flat-pass roofline the corpus puts near 6 TB/s.
#
# The corpus names exactly this shape of mistake and exactly this fix:
# "reducing a strided axis with a generic library sum ran 4.5 TB/s across two launches,
#  while one flat pass folding both tensors reached about 6 TB/s ... The reduce is a
#  bandwidth kernel; lay it out as one, putting the split axis where the reduction reads
#  contiguously and folding all output tensors into one pass."
#      -- knowledge/backends/flydsl/attention/techniques.md:405-414
# and upstream ships precisely this as a separate kernel: `flash_attn_bwd_slotred_kernel`,
# folding the q_split dK/dV slots at 19.2 us against a 1725 us wall
# (flydsl/attention/recipes/hd128.md:903,935).
#
# So: ONE kernel, ONE pass, BOTH tensors, bf16 written straight out -- no fp32 temporary,
# which also removes 2*N*4 bytes of round-trip per tensor that torch's `sum -> .to()`
# pays. The summation order is `sp = 0, 1, .. nsp-1` ascending and fixed, which is the
# whole determinism guarantee (techniques.md:405); it is the SAME order `torch.sum(0)`
# used, so this is intended to be bitwise identical to the shipped path, not merely close.
RED_THREADS = 256
RED_VEC = 4                  # fp32 dwordx4 in, bf16 x4 out


@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_redsp(DKP: fx.Tensor, DVP: fx.Tensor, DK: fx.Tensor, DV: fx.Tensor,
            n_vec: fx.Int32, nsp: fx.Int32):
    """dk[i] = sum_{sp<nsp} dkp[sp, i], dv likewise; fp32 in, bf16 out, one pass."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * fx.Int32(RED_THREADS) + tid          # index in units of RED_VEC elements

    # h2: bound by an EXPLICIT predicate, never by the buffer descriptor. Every shape this
    # op accepts divides RED_THREADS*RED_VEC exactly (D = 128), so the clamp never fires;
    # it is here so that a shape that does not divide is still correct rather than lucky.
    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dkp = _bv(DKP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dvp = _bv(DVP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dk = _bv(DK, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)
    g_dv = _bv(DV, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            kacc = fx.Vector(carried[0])
            vacc = fx.Vector(carried[1])
            off = t + ii * n_vec
            kx = _ldv(g_dkp, off, fx.Float32, RED_VEC)
            vx = _ldv(g_dvp, off, fx.Float32, RED_VEC)
            kn = [kacc[c] + kx[c] for c in range_constexpr(RED_VEC)]
            vn = [vacc[c] + vx[c] for c in range_constexpr(RED_VEC)]
            final = yield [_ir(fx.Vector.from_elements(kn, dtype=fx.Float32)),
                           _ir(fx.Vector.from_elements(vn, dtype=fx.Float32))]
        return final

    init = [_ir(fx.Vector.filled(RED_VEC, 0.0, fx.Float32)) for _ in range(2)]
    out = redloop(init, nsp)
    kacc = fx.Vector(out[0])
    vacc = fx.Vector(out[1])
    dst = ok.select(tile, n_vec)      # off-range lanes are parked past the view
    _stv([kacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dk, dst,
         fx.BFloat16)
    _stv([vacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dv, dst,
         fx.BFloat16)


@flyc.jit
def launch_redsp(DKP, DVP, DK, DV, n_vec: fx.Int32, nsp: fx.Int32,
                 nblk: fx.Int32, stream: fx.Stream):
    k_redsp(DKP, DVP, DK, DV, n_vec, nsp).launch(
        grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)


# r17.i1.g52 -- the dQ fold. k_redsp's shape with ONE tensor instead of two: dQ is a
# single output, so there is nothing to co-fold and the second stream would be dead
# traffic. Everything else is deliberately identical, because the determinism guarantee
# lives in the ORDER: `sp = 0, 1, .. nsp-1` ascending and fixed, one flat pass, fp32 in
# and bf16 straight out with no fp32 temporary. k_dq_sp keeps the masked tail whole on
# the LAST split (kernels.py:_dq_impl), so folding ascending in sp adds this query's kv
# blocks in ascending kv order -- the same order the unsplit k_dq accumulates them in,
# with the split boundaries as the only difference.
@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_redsp_q(DQP: fx.Tensor, DQ: fx.Tensor, n_vec: fx.Int32, nsp: fx.Int32):
    """dq[i] = sum_{sp<nsp} dqp[sp, i]; fp32 in, bf16 out, one pass."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * fx.Int32(RED_THREADS) + tid          # index in units of RED_VEC elements

    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))

    g_dqp = _bv(DQP, nsp * n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_dq = _bv(DQ, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)

    @flyc.jit
    def redloop_q(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qacc = fx.Vector(carried[0])
            off = t + ii * n_vec
            qx = _ldv(g_dqp, off, fx.Float32, RED_VEC)
            qn = [qacc[c] + qx[c] for c in range_constexpr(RED_VEC)]
            final = yield [_ir(fx.Vector.from_elements(qn, dtype=fx.Float32))]
        return final

    init = [_ir(fx.Vector.filled(RED_VEC, 0.0, fx.Float32))]
    out = redloop_q(init, nsp)
    # k_redsp carries TWO vectors and gets a sequence back; this loop carries ONE and
    # flyc.jit unpacks a single-element carried state to the value itself at the return
    # boundary (inside the body `carried[0]` still indexes a sequence). Normalise.
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    qacc = fx.Vector(out[0])
    dst = ok.select(tile, n_vec)      # off-range lanes are parked past the view
    _stv([qacc[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_dq, dst,
         fx.BFloat16)


@flyc.jit
def launch_redsp_q(DQP, DQ, n_vec: fx.Int32, nsp: fx.Int32,
                   nblk: fx.Int32, stream: fx.Stream):
    k_redsp_q(DQP, DQ, n_vec, nsp).launch(
        grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)


# ================================================================ w4f: 4-wave fused dQ ====
# arm w4f (0930) -- aiter-ASM-style workgroup: FOUR waves own BLOCK_KV_WG = 128 kv rows of
# one (batch, kv head); wave w keeps its own 32 rows' K/V fragments resident exactly as
# the 1-wave k_dkdv does, and ALSO computes dQ, so k_dqg (which recomputed S and dP) is
# gone: 5 GEMMs per (32q x 32kv) unit instead of 7.
#
# Per global step s (one 32-query pair x one q head of the GQA group; pair-outer,
# head-inner, the same order as k_dkdv so dK/dV accumulate in the same order):
#   (A) prefetch LSE/delta of step s+1 (4 buffer_load_b32 per lane, carried)
#   (B) 32 fp32 SCOPE_DEV atomics of step s-1's dQ slice (DEFERRED one step), emitted one
#       per S/dP WMMA in source order
#   (C) S^T/dP^T (32 WMMA) on Q/dO B operands CARRIED in VGPRs (read back from ring slot
#       s%3 during step s-1's post-barrier DS burst -- dkdv_tdm3's readback)
#   (D) softmax; P -> the wave's PRIVATE P tile, dS -> the SHARED dS buffer s%2 (own cols),
#       the 8 stores as one burst
#   (E) s_wait_dscnt 0 + s_wait_tensorcnt 0, then s_barrier_signal  (split barrier B_s)
#   (F) dV/dK (32 WMMA, TRORDER regions), all operands local to the wave or landed
#   (G) s_barrier_wait B_s; TDM the Q/dO tile of step s+2 into slot (s+2)%3 (each wave
#       copies 8 of the 32 rows of dO and of Q: 2 tensor_load_to_lds per wave per step)
#   (H) one DS burst: dS A fragments (all 128 kv) + readback of slot (s+1)%3 for step s+1;
#       dQ[32q][32d slice of wave w] = dS[32q][128kv] . K[128kv][32d] (16 WMMA; B = K^T
#       fragments RESIDENT since the prologue) -> carried to step s+1's atomics.
# ONE split barrier per step. Why it suffices (all four hazards; proven by bounds_proof.py):
#   dS RAW   : writers wait dscnt 0 before signal B_s, readers read after wait B_s.
#   dS WAR   : buffer s%2 is rewritten at step s+2, after wait B_{s+1}; every reader's
#              dQ_s loads retire before its signal B_{s+1} (dscnt 0).
#   ring RAW : TDM(s+1) is waited (tensorcnt 0) by its issuing wave before signal B_s,
#              step s+1 reads slot (s+1)%3 after wait B_s.
#   ring WAR : TDM(s+2) overwrites slot (s-1)%3 after wait B_s; every read of that slot
#              (readback in step s-2, dV/dK tr16 in step s-1) retires before the reader's
#              signal B_{s-1} / B_s (dscnt 0).
# No tensorcnt wait relies on in-order TDM completion: at every wait exactly one TDM batch
# is outstanding per wave.
W4 = 4                           # waves per workgroup
W4_THREADS = W4 * WAVE           # 128
W4_BKV = W4 * BLOCK_KV           # 128 kv rows per workgroup
# w4f_relax: 4-slot ring, TDM issued THREE steps ahead (step s+3 after wait B_s) and the
# per-step wait before signal B_s relaxed to s_wait_tensorcnt 2 = retire only TDM(s+1) (the
# slot read back next), leaving TDM(s+2) (2 ops) in flight -> ~2 steps of TDM lead instead
# of ~1. Relies on in-order TDM retirement (the premise dkdv_tdm/tdm3 are card-proven on).
W4_RELAX = True
# w4f_noatom ABLATION (output WRONG): no dQ atomics at all; the dQ WMMAs stay (their result is
# loop-carried and the final value is sunk into dV accumulator 0 so nothing is DCE'd).
W4_ABL_NOATOM = False
W4_RING = 4 if W4_RELAX else 3   # Q/dO ring slots
W4_LEAD = W4_RING - 1            # TDM(s + W4_LEAD) issued after wait B_s
W4_TW_LOOP = 2 if W4_RELAX else 0          # tensorcnt threshold before each loop signal
W4_STAGE_B = 2 * 32 * X_ROW_B    # 17408: one slot = dO [32][272 B] @0, Q @8704
W4_DSBUF_B = 32 * X_ROW_B        # 8704: one shared dS buffer [32 q][128 kv] bf16, 272 B rows
W4_P_B = 32 * S_ROW_B            # 2560: one wave's private P tile [32 q][32 kv]
W4_OFF_RING = 0                  # [0, 52224)          LDS segment 0 (B operands)
W4_OFF_DS = 131072 if W4_RELAX else 65536   # dS double buffer (A operands) -- relax: [131072, 148480)
W4_OFF_P = W4_OFF_DS + 2 * W4_DSBUF_B          # [82944, 93184)
W4_OFF_K = 196608 if W4_RELAX else 131072   # K image, prologue only
W4_K_B = W4_BKV * X_ROW_B        # 34816
W4_LDS = W4_OFF_K + W4_K_B       # 165888 B <= 327680
W4_EPI_CB = 48
W4_EPI_B = 4 * 128 * W4_EPI_CB   # 24576: one wave's dK/dV epilogue staging (as r27.i3.g86)
W4_DEFER = True                  # dQ atomics issued one step late, 1 per S/dP WMMA
W4_ATOM_SPREAD = True            # sched_group_barrier pattern {1 WMMA, 1 VMEM-write} x 32 over the
                                 # S/dP phase: one atomic per WMMA gap, as the ASM does
W4_ATOM_IMM = True               # atomics as llvm.atomicrmw on ONE per-step base pointer + constant
                                 # GEPs (immediate offsets), instead of 32 per-atomic index VGPRs
assert W4_RING * W4_STAGE_B <= W4_OFF_DS
assert W4_OFF_P + W4 * W4_P_B <= W4_OFF_K
assert W4 * W4_EPI_B <= W4_LDS


def _dkdv_w4f_impl(PARTIAL, Q, K, V, DO, LSE, DEL, DV_, DK, DQA,
                   scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nsp):
    """grid = (Hkv [*nsp], Skv/128, B), block = 128 (4 waves). See the section comment."""
    assert not VF_KV, "w4f keeps the r29 softmax so dK/dV stay bitwise vs s1"
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid % fx.Int32(WAVE)
    # wave index as an SGPR: the TDM descriptor and every LDS slice base must be uniform.
    wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid // fx.Int32(WAVE)))
    if PARTIAL:
        _x = fx.Int32(fx.block_idx.x)
        hkv = _x // nsp
        sp = _x - hkv * nsp
    else:
        hkv = fx.Int32(fx.block_idx.x)
        sp = fx.Int32(0)
    bid = fx.Int32(fx.block_idx.y)              # 128-row kv block, ascending = longest first
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kvw0 = bid * fx.Int32(W4_BKV)               # workgroup's first kv row
    kv0 = kvw0 + wv * fx.Int32(BLOCK_KV)        # this wave's first kv row

    nkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)
    nl_b = B_ * Hq * Sq * fx.Int32(4)
    ndkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)
    if PARTIAL:
        _np_b = nsp * B_ * Skv * Hkv * fx.Int32(D * 4)
        g_dv = _bv(DV_, _np_b, fx.Float32)
        g_dk = _bv(DK, _np_b, fx.Float32)
    # dq_acc fp32 [B, Hq, Sq, D] (head-major so the 32 atomics of a step differ only by
    # immediate offsets), zeroed by impl.py. Plain global view, NOT a buffer tensor:
    # BufferAtomicAdd is SCOPE_CU and buffer UniversalAtomic aborts (h42); this lowers to
    # global_atomic_add_f32 ... scope:SCOPE_DEV (FUSED5, proven in ISA).
    g_dqa = fx.Tensor(fx.make_view(fx.get_iter(DQA),
                                   fx.make_layout((1 << 30, 1), (1, 1))))
    _dq_atom = fx.make_copy_atom(
        fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent), fx.Float32)

    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    smem = fx.SharedAllocator().allocate(W4_LDS)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ring = _lds0 + fx.Int32(W4_OFF_RING)
    lds_dsb = _lds0 + fx.Int32(W4_OFF_DS)
    lds_p = _lds0 + fx.Int32(W4_OFF_P) + wv * fx.Int32(W4_P_B)
    lds_k = _lds0 + fx.Int32(W4_OFF_K)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)

    def gfrag2(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return (_ldv(buf, t, fx.BFloat16, 8),
                _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8))

    def tr(base, rowb):
        return fx.Vector(rocdl.ds_load_tr16_b128(
            v8b, create_llvm_ptr(base, address_space=3))).shuffle(
            fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * rowb),
                                     address_space=3))), list(range(16)))

    def _l8(addr):
        return fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(addr, address_space=3)))

    def _st8(vals, addr):
        llvm_dialect.store(fx.as_ir_value(fx.Vector.from_elements(vals, dtype=fx.BFloat16)),
                           create_llvm_ptr(addr, address_space=3))

    def _split_barrier_signal(tw=0):
        # Explicit waits, then the RAW signal: fx.barrier()'s workgroup release fence would
        # also drain loadcnt and storecnt, i.e. the in-flight dQ atomics (BARRIER-CENSUS F1).
        rocdl.sched_barrier(0)
        rocdl.s_wait_dscnt(0)
        tdm_ops.tensor_wait(tw)
        rocdl.s_barrier_signal(-1)
        rocdl.sched_barrier(0)

    def _split_barrier_wait():
        rocdl.sched_barrier(0)
        rocdl.s_barrier_wait(-1)
        rocdl.sched_barrier(0)

    # ---- Q/dO TDM: wave w copies rows [8w, 8w+8) of the step's dO and Q tiles ----------
    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(elem_ty=fx.BFloat16.ir_type,
                                    address_space=fx.AddressSpace.Shared, alignment=16)
    _q_rs_el = Hq * fx.Int32(D)

    def _tdm_qdo(qt, gh, slot_off):
        qh = hkv * G + gh
        r0 = qt * fx.Int32(32) + wv * fx.Int32(8)
        off = (fx.Int64(bat * Sq + r0) * fx.Int64(Hq) + fx.Int64(qh)) * fx.Int64(D)
        valid = Sq - r0                          # >= 8 for every issued tile (bounds_proof)
        lb0 = lds_ring + slot_off + wv * fx.Int32(8 * X_ROW_B)
        for src, lb in ((DO, lb0), (Q, lb0 + fx.Int32(32 * X_ROW_B))):
            g_view = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), off),
                                            fx.make_layout((8, D), (D, 1))))
            atom = fx.rocdl.cdna5.make_tdm_atom(
                g_view, [valid, None], strides=[_q_rs_el, None], num_warps=1,
                pad_interval=D, pad_amount=X_ROW_EL - D)
            l_view = fx.Tensor(fx.make_view(fx.inttoptr(_lds_bf_ty, lb),
                                            fx.make_layout((8, D), (X_ROW_EL, 1))))
            fx.copy_atom_call(atom, g_view, l_view)

    # ---- step schedule: WORKGROUP-uniform (only kvw0, not the wave's kv0, enters) -------
    nqt2 = nqt // fx.Int32(2)

    def _clampqt(t):
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kvw0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start
    nqp_eff = (nqp_eff < fx.Int32(0)).select(fx.Int32(0), nqp_eff)
    # pair qt is unmasked for the WHOLE 128-row block iff kvw0+127 <= qt*32 + cshift
    _u = kvw0 + fx.Int32(W4_BKV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))
    if PARTIAL:
        _fn = nqp_eff - nmaskp
        _ch = (_fn + nsp - fx.Int32(1)) // nsp
        _off = sp * _ch
        _cnt = _fn - _off
        _cnt = (_cnt < fx.Int32(0)).select(fx.Int32(0), _cnt)
        _cnt = (_cnt < _ch).select(_cnt, _ch)
        _mk = (sp != fx.Int32(0)).select(fx.Int32(0), nmaskp)
        p_begin = qp_start + (sp != fx.Int32(0)).select(nmaskp + _off, fx.Int32(0))
    else:
        _mk = nmaskp
        _cnt = nqp_eff - nmaskp
        p_begin = qp_start
    n_mask = G * _mk                             # masked steps come first
    n_all = G * (_mk + _cnt)

    def _tile(s):
        """(qt, gh) of step s, clamped into [0, n_all-1] and qt into [0, nqt2-1]."""
        sc = (s < n_all).select(s, n_all - fx.Int32(1))
        sc = (sc < fx.Int32(0)).select(fx.Int32(0), sc)
        qi = sc // G
        return _clampqt(p_begin + qi), sc - qi * G

    def _ldl(qt, gh):
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        base_l = (bat * Hq + qh) * Sq
        out = []
        for hh in range_constexpr(2):
            qg = q0 + fx.Int32(hh * 16) + row
            out.append(_ldv(g_lse, base_l + qg, fx.Float32, 1))
            out.append(_ldv(g_del, base_l + qg, fx.Float32, 1))
        return out

    def _dq_atomics(dq, qt, gh):
        """32 closures, one fp32 atomic each: dQ[q][d] for this wave's 32 d columns.
        C layout of dq[qh16*2+j]: lane (row, half) holds d = 32*wv + 16*j + row and
        q = qt*32 + qh16*16 + half*8 + si."""
        qh = hkv * G + gh
        base = (((bat * Hq + qh) * Sq + qt * fx.Int32(32) + half * fx.Int32(8))
                * fx.Int32(D) + wv * fx.Int32(32) + row)
        fns = []
        if const_expr(W4_ATOM_IMM):
            # same element index as below, split as base + CONSTANT: the constant goes into
            # an inbounds GEP so LLVM folds it into the instruction's immediate offset.
            pb = fx.to_llvm_ptr(fx.add_offset(fx.get_iter(DQA), base))
            for qh16 in range_constexpr(2):
                for j in range_constexpr(2):
                    for si in range_constexpr(8):
                        def _g(qh16=qh16, j=j, si=si):
                            v = fx.Vector(dq[qh16 * 2 + j])[si]
                            p = llvm_dialect.GEPOp(
                                pb.type, pb, [], [(qh16 * 16 + si) * D + j * 16],
                                fx.Float32.ir_type, llvm_dialect.GEPNoWrapFlags.inbounds).result
                            llvm_dialect.atomicrmw(
                                llvm_dialect.AtomicBinOp.fadd, p, fx.as_ir_value(v),
                                llvm_dialect.AtomicOrdering.monotonic,
                                syncscope=rocdl.SyncScope.Agent, alignment=4)
                        fns.append(_g)
            return fns
        for qh16 in range_constexpr(2):
            for j in range_constexpr(2):
                for si in range_constexpr(8):
                    def _f(qh16=qh16, j=j, si=si):
                        v = fx.Vector(dq[qh16 * 2 + j])[si]
                        idx = base + fx.Int32((qh16 * 16 + si) * D + j * 16)
                        f = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
                        fx.memref_store_vec(fx.Vector.from_elements([v], dtype=fx.Float32), f)
                        fx.copy_atom_call(_dq_atom, f, fx.slice(g_dqa, (idx, None)))
                    fns.append(_f)
        return fns

    # ---- prologue ----------------------------------------------------------------------
    # TDM steps 0 and 1 first (slots 0/1), then K/V fragments, K image, one barrier.
    _t0 = _tile(fx.Int32(0))
    _t1 = _tile(fx.Int32(1))
    _tdm_qdo(_t0[0], _t0[1], fx.Int32(0))
    _tdm_qdo(_t1[0], _t1[1], fx.Int32(W4_STAGE_B))
    if const_expr(W4_RELAX):
        _t2 = _tile(fx.Int32(2))
        _tdm_qdo(_t2[0], _t2[1], fx.Int32(2 * W4_STAGE_B))
    k2 = [[gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
          for kh in range(NKV)]
    kf = [[k2[kh][dt][0].shuffle(k2[kh][dt][1], list(range(16))) for dt in range(NDT)]
          for kh in range(NKV)]
    vf = []
    for kh in range(NKV):
        _r = []
        for dt in range(NDT):
            a, b = gfrag2(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt)
            _r.append(a.shuffle(b, list(range(16))))
        vf.append(_r)
    for kh in range_constexpr(NKV):
        ko = ((wv * fx.Int32(BLOCK_KV) + fx.Int32(kh * 16) + row) * fx.Int32(X_ROW_B)
              + half * fx.Int32(16))
        for dt in range_constexpr(NDT):
            for u in range_constexpr(2):
                llvm_dialect.store(fx.as_ir_value(k2[kh][dt][u]),
                                   create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32),
                                                   address_space=3))
    pre0 = _ldl(_t0[0], _t0[1])
    # relax: TDM(1), TDM(2) (4 ops) may stay in flight; only TDM(0) (slot 0) is needed now
    _split_barrier_signal(4 if W4_RELAX else 0)
    _split_barrier_wait()
    # K^T B fragments for dQ, resident for the whole kernel: kt[j][kc] covers
    # d in [32*wv + 16*j, +16) x kv in [kvw0 + 32*kc, +32).
    kt = [[tr(lds_k + (fx.Int32(kc * 32) + lane_r) * fx.Int32(X_ROW_B)
              + (lane_c + wv * fx.Int32(32) + fx.Int32(j * 16)) * fx.Int32(2), X_ROW_B)
           for kc in range_constexpr(W4)] for j in range_constexpr(2)]

    def _rdqd(slot_off):
        """dkdv_tdm3's readback: the 32 ds_load_b128 of one ring slot's S/dP B operands,
        order hh, {Q, dO}, dt, u -> index hh*16 + {0, 8} + dt*2 + u."""
        lq = lds_ring + slot_off + fx.Int32(32 * X_ROW_B)
        ld = lds_ring + slot_off
        out = []
        for hh in range_constexpr(2):
            xo = ((fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16))
            for base in (lq, ld):
                for dt in range_constexpr(NDT):
                    for u in range_constexpr(2):
                        out.append(_l8(base + xo + fx.Int32(dt * 64 + u * 32)))
        return out

    def _body(st, s, do_mask):
        acc = st[:NST]
        dqp = st[NST:NST + 4]
        pre = [fx.Vector(v) for v in st[NST + 4:NST + 8]]
        qd = [fx.Vector(v) for v in st[NST + 8:NST + 40]]
        qt, gh = _tile(s)
        q0 = qt * fx.Int32(32)
        qtn, ghn = _tile(s + fx.Int32(1))
        nxt = _ldl(qtn, ghn)
        pend = []
        if const_expr(W4_DEFER and not W4_ABL_NOATOM):
            qtp, ghp = _tile(s - fx.Int32(1))
            pend = _dq_atomics(dqp, qtp, ghp)
        lds_do = lds_ring + (s % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B)
        lds_q = lds_do + fx.Int32(32 * X_ROW_B)
        lds_ds = lds_dsb + (s % fx.Int32(2)) * fx.Int32(W4_DSBUF_B)

        # (C)+(D) S^T/dP^T on the CARRIED operands (read back from slot s%3 during step
        # s-1's DS burst, dkdv_tdm3), softmax -- s1's arithmetic verbatim. The 8 P/dS
        # stores are deferred into one burst (dkdv_tdm3).
        stores = []
        for hh in range_constexpr(2):
            qfr = [qd[hh * 16 + dt * 2].shuffle(qd[hh * 16 + dt * 2 + 1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            dfr = [qd[hh * 16 + 8 + dt * 2].shuffle(qd[hh * 16 + 8 + dt * 2 + 1],
                                                    list(range(16)))
                   for dt in range_constexpr(NDT)]
            q_glob = q0 + fx.Int32(hh * 16) + row
            lse_q = pre[hh * 2][0]
            del_q = pre[hh * 2 + 1][0]
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT):
                    s_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kf[kh][dt]), _ir(qfr[dt]), s_acc,
                        reuseA=False, reuseB=False).result
                    if pend:
                        pend.pop(0)()
                    p_acc = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vf[kh][dt]), _ir(dfr[dt]), p_acc,
                        reuseA=False, reuseB=False).result
                    if pend:
                        pend.pop(0)()
                sv, pv_ = fx.Vector(s_acc), fx.Vector(p_acc)
                kvb = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8)
                if const_expr(do_mask):
                    masked = [
                        ((kvb + fx.Int32(si) > q_glob + cshift)
                         & (causal != fx.Int32(0))
                         ).select(fx.Float32(NEG), sv[si] * scale)
                        for si in range(8)
                    ]
                else:
                    masked = [sv[si] * scale for si in range(8)]
                pf = [_exp2((masked[si] - lse_q) * fx.Float32(LOG2E)) for si in range(8)]
                p_l = [x.to(fx.BFloat16) for x in pf]
                ds_l = [(pf[si] * (pv_[si] - del_q) * scale).to(fx.BFloat16)
                        for si in range(8)]
                stores.append((p_l, lds_p + (fx.Int32(hh * 16) + row) * fx.Int32(S_ROW_B)
                               + fx.Int32(kh * 32) + half * fx.Int32(16)))
                stores.append((ds_l, lds_ds + (fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B)
                               + wv * fx.Int32(64) + fx.Int32(kh * 32) + half * fx.Int32(16)))
        if const_expr(W4_DEFER and W4_ATOM_SPREAD and not W4_ABL_NOATOM):
            for _i in range_constexpr(32):
                rocdl.sched_group_barrier(0x008, 1, 0)      # 1 WMMA
                rocdl.sched_group_barrier(0x040, 1, 0)      # 1 VMEM write (the atomic)
        rocdl.sched_barrier(0)
        for vals, addr in stores:
            _st8(vals, addr)
        assert not pend

        # (E) split barrier B_s: signal (relax: retire only TDM(s+1))
        _split_barrier_signal(W4_TW_LOOP)

        # (F) dV/dK -- s1's TRORDER regions; A from the private P tile and this wave's own
        # columns of the shared dS buffer, B from ring slot s%3.
        def _a(base, rowb, kh):
            return tr(base + lane_r * fx.Int32(rowb) + lane_c * fx.Int32(2)
                      + fx.Int32(kh * 32), rowb)

        def _b(base, dtile):
            c = (lane_c + fx.Int32(dtile * 16)) * fx.Int32(2)
            return tr(base + lane_r * fx.Int32(X_ROW_B) + c, X_ROW_B)

        lds_dsw = lds_ds + wv * fx.Int32(64)
        new = [None] * NST

        def _mm(kh, a, bs, which):
            for dtile in range_constexpr(NDO):
                k = (which * NKV + kh) * NDO + dtile
                new[k] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a), _ir(bs[dtile]), acc[k],
                    reuseA=(dtile > 0), reuseB=False).result

        a_p0 = _a(lds_p, S_ROW_B, 0)                                   # R1
        b_do = [_b(lds_do, dtile) for dtile in range_constexpr(NDO)]
        rocdl.sched_barrier(0)
        a_ds0 = _a(lds_dsw, X_ROW_B, 0)                                # R2
        b_q = [_b(lds_q, dtile) for dtile in range_constexpr(NDO)]
        rocdl.sched_barrier(0)
        _mm(0, a_p0, b_do, 0)
        rocdl.sched_barrier(0)
        a_p1 = _a(lds_p, S_ROW_B, 1)                                   # R3
        a_ds1 = _a(lds_dsw, X_ROW_B, 1)
        _mm(0, a_ds0, b_q, 1)
        rocdl.sched_barrier(0)
        _mm(1, a_p1, b_do, 0)                                          # R4
        _mm(1, a_ds1, b_q, 1)

        # (G) wait B_s, then refill the slot step s-1 used with step s+2's tile
        _split_barrier_wait()
        qt2, gh2 = _tile(s + fx.Int32(W4_LEAD))
        _tdm_qdo(qt2, gh2, ((s + fx.Int32(W4_LEAD)) % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B))
        rocdl.sched_barrier(0)

        # (H) one DS burst: the dS A fragments of dQ_s (all 128 kv columns, written by all
        # four waves before B_s), then the readback of slot (s+1)%3 (landed and visible
        # since B_s) as step s+1's carried S/dP operands; then the 16 dQ WMMAs (2 q halves
        # x 2 d tiles, 4-deep chains over the 128 kv).
        afrs = []
        for qh16 in range_constexpr(2):
            xo = (fx.Int32(qh16 * 16) + row) * fx.Int32(X_ROW_B) + half * fx.Int32(16)
            afrs.append([_l8(lds_ds + xo + fx.Int32(kc * 64)).shuffle(
                             _l8(lds_ds + xo + fx.Int32(kc * 64 + 32)), list(range(16)))
                         for kc in range_constexpr(W4)])
        rb = _rdqd(((s + fx.Int32(1)) % fx.Int32(W4_RING)) * fx.Int32(W4_STAGE_B))
        rocdl.sched_barrier(0)
        dq = []
        for qh16 in range_constexpr(2):
            for j in range_constexpr(2):
                c = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for kc in range_constexpr(W4):
                    c = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(afrs[qh16][kc]), _ir(kt[j][kc]), c,
                        reuseA=False, reuseB=False).result
                dq.append(c)
        if const_expr(not W4_DEFER and not W4_ABL_NOATOM):
            for f in _dq_atomics(dq, qt, gh):
                f()
            dq = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(4)]
        return new + dq + [_ir(v) for v in nxt] + [_ir(v) for v in rb]

    @flyc.jit
    def w4loop_mask(state, n, i0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), fx.Int32(it) + i0, True)
        return final

    @flyc.jit
    def w4loop_full(state, n, i0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), fx.Int32(it) + i0, False)
        return final

    init = ([_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NST)]
            + [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(4)]
            + [_ir(v) for v in pre0]
            + [_ir(v) for v in _rdqd(fx.Int32(0))])
    out = w4loop_mask(init, n_mask, fx.Int32(0))
    out = w4loop_full(list(out), n_all - n_mask, n_mask)
    out = list(out)

    # flush the last step's deferred dQ (zeros to a clamped, in-bounds tile if n_all == 0)
    if const_expr(W4_ABL_NOATOM):
        # sink: dV acc 0 += sum of the last step's dQ tiles (WRONG output, keeps the WMMAs)
        _sk = fx.Vector(out[0])
        _dqs = [fx.Vector(out[NST + i]) for i in range(4)]
        out[0] = _ir(fx.Vector.from_elements(
            [_sk[e] + _dqs[0][e] + _dqs[1][e] + _dqs[2][e] + _dqs[3][e] for e in range(8)],
            dtype=fx.Float32))
    elif const_expr(W4_DEFER):
        _tl = _tile(n_all - fx.Int32(1))
        for f in _dq_atomics(out[NST:NST + 4], _tl[0], _tl[1]):
            f()
    # epilogue barrier: every wave's TDM retired and every LDS read done before any wave
    # overwrites the ring/dS/P area with its epilogue staging.
    _split_barrier_signal()
    _split_barrier_wait()

    if PARTIAL:
        base_o = ((sp * B_ + bat) * Skv * Hkv) * fx.Int32(D) + hkv * fx.Int32(D)
        for kh in range_constexpr(NKV):
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                for si in range_constexpr(8):
                    kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
                    idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
                    _st1(ov[si], g_dv, idx, fx.Float32)
                    _st1(ok_[si], g_dk, idx, fx.Float32)
    else:
        # r27.i3.g86's LDS-transposed bf16 epilogue, one 24576 B staging area per wave.
        g_dv8 = _bv(DV_, ndkv_b, fx.BFloat16, 8)
        g_dk8 = _bv(DK, ndkv_b, fx.BFloat16, 8)
        lds_e = _lds0 + wv * fx.Int32(W4_EPI_B)
        for kh in range_constexpr(NKV):
            lds_ev = lds_e + fx.Int32((kh * 2 + 0) * 128 * W4_EPI_CB)
            lds_ek = lds_e + fx.Int32((kh * 2 + 1) * 128 * W4_EPI_CB)
            for dtile in range_constexpr(NDO):
                ov = fx.Vector(out[kh * NDO + dtile])
                ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
                o = ((fx.Int32(dtile * 16) + row) * fx.Int32(W4_EPI_CB)
                     + half * fx.Int32(16))
                _st8([ov[si].to(fx.BFloat16) for si in range_constexpr(8)], lds_ev + o)
                _st8([ok_[si].to(fx.BFloat16) for si in range_constexpr(8)], lds_ek + o)
            gt = base_kv + (kv0 + fx.Int32(kh * 16) + row) * rs_kv
            for sub in range_constexpr(NDO):
                a = ((fx.Int32(sub * 16) + lane_r) * fx.Int32(W4_EPI_CB)
                     + lane_c * fx.Int32(2))
                vv = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
                kk = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
                t = gt + fx.Int32(sub * 2) + half
                _stv([vv[e] for e in range_constexpr(8)], g_dv8, t, fx.BFloat16)
                _stv([kk[e] for e in range_constexpr(8)], g_dk8, t, fx.BFloat16)


@flyc.kernel(known_block_size=[W4_THREADS, 1, 1])
def k_dkdv_w4f(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
               LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
               DQA: fx.Tensor,
               scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
               G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
               B_: fx.Int32):
    _dkdv_w4f_impl(False, Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv,
                   G, nqt, cshift, causal, B_, fx.Int32(1))


@flyc.kernel(known_block_size=[W4_THREADS, 1, 1])
def k_dkdv_w4f_sp(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
                  LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
                  DQA: fx.Tensor,
                  scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32,
                  Hkv: fx.Int32, G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32,
                  causal: fx.Int32, B_: fx.Int32, nsp: fx.Int32):
    _dkdv_w4f_impl(True, Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv,
                   G, nqt, cshift, causal, B_, nsp)


@flyc.jit
def launch_dkdv_w4f(Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale: fx.Float32,
                    Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                    nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                    nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dkdv_w4f(Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv, G,
               nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(W4_THREADS, 1, 1), stream=stream)


@flyc.jit
def launch_dkdv_w4f_sp(Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale: fx.Float32,
                       Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
                       G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                       nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, nsp: fx.Int32,
                       nxs: fx.Int32, stream: fx.Stream):
    k_dkdv_w4f_sp(Q, K, V, DO, LSE, DEL, DV_, DK, DQA, scale, Sq, Skv, Hq, Hkv, G,
                  nqt, cshift, causal, nb, nsp).launch(
        grid=(nxs, nblk, nb), block=(W4_THREADS, 1, 1), stream=stream)


# w4f dQ convert: fp32 [B, Hq, Sq, D] workspace -> bf16 [B, Sq, Hq, D], one flat pass in
# OUTPUT order (vec4 per thread). Every shape here divides RED_THREADS*RED_VEC; the
# predicate parks off-range lanes past the view anyway.
@flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
def k_dq_cvt(DQA: fx.Tensor, DQ: fx.Tensor, Sq: fx.Int32, Hq: fx.Int32, n_vec: fx.Int32):
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    tile = bid * fx.Int32(RED_THREADS) + tid
    ok = tile < n_vec
    t = ok.select(tile, fx.Int32(0))
    D4 = D // RED_VEC
    d4 = t % fx.Int32(D4)
    r = t // fx.Int32(D4)                 # (b*Sq + s)*Hq + h
    h = r % Hq
    bs = r // Hq
    b = bs // Sq
    s_ = bs - b * Sq
    src = ((b * Hq + h) * Sq + s_) * fx.Int32(D4) + d4
    g_a = _bv(DQA, n_vec * fx.Int32(RED_VEC * 4), fx.Float32, RED_VEC)
    g_q = _bv(DQ, n_vec * fx.Int32(RED_VEC * 2), fx.BFloat16, RED_VEC)
    x = _ldv(g_a, src, fx.Float32, RED_VEC)
    dst = ok.select(tile, n_vec)
    _stv([x[c].to(fx.BFloat16) for c in range_constexpr(RED_VEC)], g_q, dst, fx.BFloat16)


@flyc.jit
def launch_dq_cvt(DQA, DQ, Sq: fx.Int32, Hq: fx.Int32, n_vec: fx.Int32,
                  nblk: fx.Int32, stream: fx.Stream):
    k_dq_cvt(DQA, DQ, Sq, Hq, n_vec).launch(
        grid=(nblk, 1, 1), block=(RED_THREADS, 1, 1), stream=stream)
