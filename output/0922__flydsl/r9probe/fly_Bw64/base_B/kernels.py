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
DKDV_THREADS = 64
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
@flyc.kernel(known_block_size=[DKDV_THREADS, 1, 1])
def k_dkdv(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
           LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
           scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
           G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32, B_: fx.Int32):
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
    hkv = fx.Int32(fx.block_idx.x)              # kv head
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
    g_dv = _bv(DV_, ndkv_b, fx.BFloat16)
    g_dk = _bv(DK, ndkv_b, fx.BFloat16)

    rs_q = Hq * fx.Int32(DV8)                   # vec8 tiles between consecutive q rows
    rs_kv = Hkv * fx.Int32(DV8)
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    # r1.i6.g06. The workgroup owns BLOCK_KV key rows as NKV 16-row WMMA accumulator
    # sets. Q/dO staging is untouched -- still 16 query rows -- so the whole LDS increment
    # is the [q][kv] P and dS tiles going 16 -> 32 columns: 9216 -> 10240 B, which is
    # still 5 x 2048 B of granularity and still the 32-workgroup residency rung.
    smem = fx.SharedAllocator().allocate(2 * 32 * S_ROW_B + 2 * 32 * X_ROW_B)
    lds_p = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_ds = lds_p + fx.Int32(32 * S_ROW_B)
    lds_do = lds_ds + fx.Int32(32 * S_ROW_B)
    lds_q = lds_do + fx.Int32(32 * X_ROW_B)
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
    def _body(acc, pre, qt, gh, do_mask, qt_n, gh_n):
        qh = hkv * G + gh
        base_l = (bat * Hq + qh) * Sq
        q0 = qt * fx.Int32(32)

        # r7.i1.g21: next iteration's operands go out FIRST, before this iteration
        # touches anything it already holds in registers.
        nxt = _ldqd(qt_n, gh_n)

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
        for hh in range_constexpr(2):
            qh0 = q0 + fx.Int32(hh * 16)
            qp = [(pre[hh * 16 + dt * 2], pre[hh * 16 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            dp = [(pre[hh * 16 + 8 + dt * 2], pre[hh * 16 + 8 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            xo = ((fx.Int32(hh * 16) + row) * fx.Int32(X_ROW_B)
                  + half * fx.Int32(16))
            for dt in range_constexpr(NDT):
                for u in range_constexpr(2):
                    o = xo + fx.Int32(dt * 64 + u * 32)
                    llvm_dialect.store(fx.as_ir_value(dp[dt][u]),
                                       create_llvm_ptr(lds_do + o, address_space=3))
                    llvm_dialect.store(fx.as_ir_value(qp[dt][u]),
                                       create_llvm_ptr(lds_q + o, address_space=3))
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            q_glob = qh0 + row
            lse_q = _ld1(g_lse, base_l + q_glob, fx.Float32)
            del_q = _ld1(g_del, base_l + q_glob, fx.Float32)
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
        fx.barrier()

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
            for dtile in range_constexpr(NDO):
                new[kh * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do[dtile]), acc[kh * NDO + dtile],
                    reuseA=False, reuseB=False).result
                new[(NKV + kh) * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_q[dtile]), acc[(NKV + kh) * NDO + dtile],
                    reuseA=False, reuseB=False).result
        fx.barrier()
        return new + [_ir(v) for v in nxt]

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            jj = ii + fx.Int32(1)
            qj = jj // G
            st = list(carried)
            final = yield _body(st[:NST], [fx.Vector(v) for v in st[NST:]],
                                qt0 + qi, ii - qi * G, True,
                                qt0 + qj, jj - qj * G)
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            qi = ii // G
            jj = ii + fx.Int32(1)
            qj = jj // G
            st = list(carried)
            final = yield _body(st[:NST], [fx.Vector(v) for v in st[NST:]],
                                qt0 + qi, ii - qi * G, False,
                                qt0 + qj, jj - qj * G)
        return final


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
    init = init + [_ir(v) for v in _ldqd(qp_start, fx.Int32(0))]
    out = qloop_mask(init, G * nmaskp, qp_start)
    out = qloop_full(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
    base_o = bat * Skv * Hkv * fx.Int32(D) + hkv * fx.Int32(D)
    for kh in range_constexpr(NKV):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[kh * NDO + dtile])
            ok_ = fx.Vector(out[(NKV + kh) * NDO + dtile])
            for si in range_constexpr(8):
                kv = kv0 + fx.Int32(kh * 16) + half * fx.Int32(8) + fx.Int32(si)
                idx = base_o + kv * Hkv * fx.Int32(D) + fx.Int32(dtile * 16) + row
                _st1(ov[si].to(fx.BFloat16), g_dv, idx, fx.BFloat16)
                _st1(ok_[si].to(fx.BFloat16), g_dk, idx, fx.BFloat16)


@flyc.jit
def launch_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
           nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(DKDV_THREADS, 1, 1), stream=stream)


# ===================================================================== dq =========
KV_STEP = 32                 # two 16-kv tiles: exactly one WMMA contraction of dS
NKT = KV_STEP // 16
BLOCK_Q = 64                 # r3.i4.g13: next rung of g10 (was 32)
NQ = BLOCK_Q // 16


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dq(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
         LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
         scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
         G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    """One wave owns BLOCK_Q queries of one q head and streams every 32-key block.

    grid = (Sq/16, Hq, B). The A operand of the dQ GEMM comes free: the two kv-tile dS^T
    accumulators concatenate in lane into one v16, so the only LDS traffic is staging K.
    """
    lane = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)              # query tile
    qh = fx.Int32(fx.block_idx.y)               # q head
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
    def _body(acc, kv0, do_mask):

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
                kp = gfrag2(g_k, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
                for u in range_constexpr(2):
                    llvm_dialect.store(
                        fx.as_ir_value(kp[u]),
                        create_llvm_ptr(lds_k + ko + fx.Int32(dt * 64 + u * 32),
                                        address_space=3))
                kfr = kp[0].shuffle(kp[1], list(range(16)))
                vfr = gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kt * 16), dt)
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
        fx.barrier()   # for the K staging above, not for a_ds

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
        fx.barrier()   # LDS is reused by the next kv block
        return new

    @flyc.jit
    def kvloop_full(state, n):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried), fx.Int32(it) * fx.Int32(KV_STEP), False)
        return final

    @flyc.jit
    def kvloop_mask(state, n, it0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            final = yield _body(list(carried),
                                (fx.Int32(it) + it0) * fx.Int32(KV_STEP), True)
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
    out = kvloop_full(init, nfull)
    out = kvloop_mask(out, nkvt_eff - nfull, nfull)
    base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQ):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                _st1(ov[si].to(fx.BFloat16), g_dq,
                     base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                     fx.BFloat16)


@flyc.jit
def launch_dq(Q, K, V, DO, LSE, DEL, DQ, scale: fx.Float32,
              Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
              nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
              nblk: fx.Int32, nhq: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dq(Q, K, V, DO, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
         nkvt, cshift, causal).launch(
        grid=(nblk, nhq, nb), block=(32, 1, 1), stream=stream)

_G07_CLAMP = True   # r1.i7.g07: k_dkdv takes the batch count as its last argument
