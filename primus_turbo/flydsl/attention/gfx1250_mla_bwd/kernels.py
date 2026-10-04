"""gfx1250 FlyDSL flash-attention backward: k_delta, k_dkdv, k_dqg (wave32 WMMA, bf16).

    delta[b,h,s] = sum_d dO*O                                   k_delta_bshd
    dV[kv,:] = sum_q P^T dO ;  dK[kv,:] = sum_q dS^T Q           k_dkdv  (one wave per kv tile)
    dQ[q,:]  = sum_kv dS K                                       k_dqg   (one wave per q tile)

Layouts: q/o/do [B, Sq, Hq, D] bf16; k/v [B, Skv, Hkv, D] bf16; lse/delta [B, Hq, Sq] fp32,
natural log; dq/dk/dv bf16 like q/k/v. Causal is bottom-right (query i sees keys
j <= i + Skv - Sq). GQA is reduced inside k_dkdv (the G q heads of one kv head stream through
the same accumulators): no atomics, every output element written once, deterministic.

k_dkdv stages Q/dO through a 3-stage Tensor-Data-Mover LDS ring (prefetch two iterations
ahead) and reads the next iteration's S/dP B operands back into VGPRs one iteration early;
k_dqg does the same with a 3-stage K/V ring. Both run one wave32 per workgroup, so there is
no s_barrier anywhere. The design history behind every choice is in PROVENANCE.md.
"""
import importlib.util as _ilu
import pathlib as _pl
import sys as _sys

# _env sets sys.path (flydsl 0.3.4.1). It must run before flydsl is imported, and it is
# loaded by path so two implementation directories in one process never share a module.
_n = f"_env__{abs(hash(str(_pl.Path(__file__).resolve().parent)))}"
_sp = _ilu.spec_from_file_location(_n, _pl.Path(__file__).resolve().parent / "_env.py")
_env = _ilu.module_from_spec(_sp)
_sys.modules[_n] = _env
_sp.loader.exec_module(_env)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect  # UNSTABLE(gfx1250): LDS load/store
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops  # UNSTABLE(gfx1250): tensor_wait


def create_llvm_ptr(value, address_space=1):
    """Raw LLVM pointer from an i32 address (1 = global, 3 = LDS); aiter kernels_common's."""
    space = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}.get(address_space, address_space)
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
    return fx.as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, value)))


def _ir(v):
    """A typed flydsl value as a raw ir.Value (aiter tensor_shim._to_raw)."""
    return fx.as_ir_value(v)


D = 128
DV8 = D // 8                 # vec8 tiles per row of one head
NDT = D // 32                # WMMA k-steps to contract 128
NDO = D // 16                # 16-wide output tiles across d
NKV = 2                      # 16-row kv sub-tiles per k_dkdv workgroup
NST = 2 * NKV * NDO          # dV/dK accumulators carried by k_dkdv
WAVE = 32                    # gfx1250 dispatches wave32
LOG2E = 1.4426950408889634
NEG = -3.0e38
TDM_DEPTH = 3                # Q/dO (k_dkdv) and K/V (k_dqg) LDS ring stages
BLOCK_KV = 32                # kv rows one k_dkdv workgroup owns
S_ROW_B = BLOCK_KV * 2 + 16  # padded P/dS LDS row: 80 B = 20 dwords walks all 64 banks
X_ROW_B = D * 2 + 16         # padded Q/dO/K/V LDS row: 272 B; 256 B would put every row on
                             # bank 0 (64-way conflict on the 16 rows of a tr16 read)

DELTA_THREADS = 256
LANES_PER_ROW = D // 8
ROWS_PER_PASS = DELTA_THREADS // LANES_PER_ROW
ROWS_DELTA = 32
PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS


# ---- shared helpers ------------------------------------------------------------------
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


# ================================================================== delta ==========
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32."""
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
def _dkdv_impl(Q, K, V, DO, LSE, DEL, DV_, DK,
               scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_):
    """One wave owns a BLOCK_KV-row kv tile of one kv head and streams every (q head, q pair).

    grid = (Hkv, Skv/BLOCK_KV, B). The accumulators persist across the G q heads that share
    this kv head, so the GQA reduction happens in registers -- atomic-free, written once.
    """
    lane = fx.Int32(fx.thread_idx.x)
    hkv = fx.Int32(fx.block_idx.x)              # kv head (cheap axis fastest)
    bid = fx.Int32(fx.block_idx.y)              # kv tile
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)

    # TRUE byte extents on every k_dkdv descriptor: an over-read returns 0 instead of
    # walking into the next page, and a clamp that hit a LIVE access would crater SQNR.
    nkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)    # k / v   bf16 [B, Skv, Hkv, D]
    nl_b = B_ * Hq * Sq * fx.Int32(4)           # lse / delta fp32 [B, Hq, Sq]
    ndkv_b = B_ * Skv * Hkv * fx.Int32(D * 2)   # dk / dv bf16 [B, Skv, Hkv, D]
    g_k = _bv(K, nkv_b, fx.BFloat16, 8)
    g_v = _bv(V, nkv_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)

    rs_kv = Hkv * fx.Int32(DV8)                 # vec8 tiles between consecutive kv rows
    base_kv = bat * Skv * rs_kv + hkv * fx.Int32(DV8)

    # LDS: the Q/dO ring at offset 0 (segment 0), the P/dS tiles exactly one 64 KB segment
    # up, so the output GEMM's A reads (P, dS) and B reads (dO, Q) use different LDS read
    # ports whatever the workgroup's physical LDS base is.
    LDS_SEG = 65536
    smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_p = _lds0 + fx.Int32(LDS_SEG)
    lds_ds = lds_p + fx.Int32(32 * S_ROW_B)
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    # LSE/delta prefetch address: voffset = 4*row (loop-invariant VGPR) + soffset (SGPR).
    rsrc_lse = rocdl.get_buffer_rsrc(fx.get_iter(g_lse))
    rsrc_del = rocdl.get_buffer_rsrc(fx.get_iter(g_del))
    voff_l = row * fx.Int32(4)

    def _ldl(qt, gh, trim=True):
        """The 4 LSE/delta loads (hh x {lse, delta}) of query pair qt, q head hkv*G+gh,
        carried one iteration early. trim=False (masked loop): plain per-lane addresses."""
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        out = []
        base_l = (bat * Hq + qh) * Sq
        if const_expr(trim):
            # byte address 4*(base_l + q0 + hh*16 + row) split as voffset = 4*row +
            # soffset = 4*(base_l + q0) + 64*hh (bounds_proof.py A3: same address)
            sb = fx.Int32(rocdl.readfirstlane(
                fx.Int32.ir_type, ((base_l + q0) * fx.Int32(4)).ir_value()))
            for hh in range_constexpr(2):
                so = sb + fx.Int32(hh * 64)
                for rs_ in (rsrc_lse, rsrc_del):
                    x = fx.Float32(rocdl.raw_ptr_buffer_load(fx.Float32.ir_type, rs_, voff_l, so))
                    out.append(fx.Vector.from_elements([x], dtype=fx.Float32))
            return out
        for hh in range_constexpr(2):
            qg = q0 + fx.Int32(hh * 16) + row
            # carried as 1-wide vectors (the scf.for carried tuple must be vector values)
            out.append(_ldv(g_lse, base_l + qg, fx.Float32, 1))
            out.append(_ldv(g_del, base_l + qg, fx.Float32, 1))
        return out

    # Q/dO staging through the Tensor Data Mover into a TDM_DEPTH-stage LDS ring. One TDM
    # op per tensor writes the [32 q rows][D] bf16 tile: row r at r*X_ROW_B, element c at
    # 2c, 16 B pad per row (pad_interval = D elements, pad_amount = 8 elements). Stage s is
    # [s*QDO_B, (s+1)*QDO_B): dO at +0, Q at +32*X_ROW_B. Tile origin element
    # ((bat*Sq + q0)*Hq + qh)*D, outer stride Hq*D, outer extent Sq - q0 (>= 32 for every
    # issued tile: Sq % 64 == 0 is asserted by impl.py and qt is clamped to [0, nqt2-1]).
    QDO_B = 2 * 32 * X_ROW_B
    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(elem_ty=fx.BFloat16.ir_type,
                                    address_space=fx.AddressSpace.Shared, alignment=16)
    _q_rs_el = Hq * fx.Int32(D)                 # elements between consecutive q rows

    def _tdm_qdo(qt, gh, stage_off):
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        off = (fx.Int64(bat * Sq + q0) * fx.Int64(Hq) + fx.Int64(qh)) * fx.Int64(D)
        valid = Sq - q0
        for src, lb in ((DO, _lds0 + stage_off),
                        (Q, _lds0 + stage_off + fx.Int32(32 * X_ROW_B))):
            g_view = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), off),
                                            fx.make_layout((32, D), (D, 1))))
            atom = fx.rocdl.cdna5.make_tdm_atom(
                g_view, [valid, None], strides=[_q_rs_el, None], num_warps=1,
                pad_interval=D, pad_amount=X_ROW_EL - D)
            l_view = fx.Tensor(fx.make_view(fx.inttoptr(_lds_bf_ty, lb),
                                            fx.make_layout((32, D), (X_ROW_EL, 1))))
            fx.copy_atom_call(atom, g_view, l_view)

    # K and V fragments of this kv tile are invariant over every query and every q head.
    kf = [[gfrag(g_k, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
          for kh in range(NKV)]
    vf = [[gfrag(g_v, base_kv, rs_kv, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT)]
          for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # Lane-only parts of the two DS-phase address families; every per-op difference is a
    # compile-time constant ISel folds into the 16-bit DS immediate.
    lb_tr = lane_r * fx.Int32(X_ROW_B) + lane_c * fx.Int32(2)   # b_do/b_q tr16
    lb_rd = row * fx.Int32(X_ROW_B) + half * fx.Int32(16)        # _rdqd readback

    # Causal tile skip: under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so query pairs below qp_start contribute exactly zero to dK/dV.
    nqt2 = nqt // fx.Int32(2)                   # query tiles come in PAIRS

    def _clampqt(t):
        """Pin a prefetch's query pair into [0, nqt2-1]."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0 - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # Iteration order is query-pair-OUTER, q-head-INNER (qi = ii // G, gh = ii % G), which
    # puts every masked iteration at the front: exactly one query pair per q head straddles
    # this kv tile, so the mask runs in its own loop (qloop_mask) and the hot loop
    # (qloop_full) carries no mask at all.
    def _rdqd(stage_off, rbase=None):
        """The 32 ds_load_b128 that read one stage's S/dP B operands. Entry hh*16 + dt*2 + u
        is Q row hh*16+row, bytes half*16 + dt*64 + u*32; entry hh*16 + 8 + dt*2 + u the
        same for dO. Every address is rbase + a constant (bounds_proof.py A2)."""
        if const_expr(rbase is None):
            rbase = _lds0 + stage_off + lb_rd
        out = []
        for hh in range_constexpr(2):
            for qo in (32 * X_ROW_B, 0):          # Q then dO
                for dt in range_constexpr(NDT):
                    for u in range_constexpr(2):
                        out.append(fx.Vector(llvm_dialect.load(
                            v8b, create_llvm_ptr(
                                rbase + fx.Int32(hh * 16 * X_ROW_B + qo + dt * 64 + u * 32),
                                address_space=3))))
        return out

    def _body(acc, pre, qt, gh, do_mask, qt_n, gh_n, carry=True,
              cur_off=None, nxt_off=None, pf_qt=None, pf_gh=None, qd=None, rb_off=None):
        q0 = qt * fx.Int32(32)

        # carry=True (qloop_full): this iteration's S/dP B operands arrive in VGPRs (qd),
        # read back from stage it%3 during iteration it-1 (or by the prologue). The TDM for
        # iteration it+2 goes into stage (it+2)%3 == (it-1)%3 at the top, with NO wait: that
        # stage's last readers (tr16 of it-1, readback in it-2) were consumed by WMMAs of
        # it-1, hence retired. The tensor wait sits at the readback below.
        # carry=False (qloop_mask): load its own tile into stage 0 and wait for it.
        tr_base = None
        rb_base = None
        if const_expr(carry):
            # TDM first, then the LSE/delta prefetch: their soffset SGPRs are then not
            # recycled by the TDM descriptor SALU right after the loads.
            rocdl.sched_barrier(0)
            _tdm_qdo(pf_qt, pf_gh, nxt_off)
            rocdl.sched_barrier(0)
            nxt = _ldl(qt_n, gh_n)
        else:
            nxt = None
            cur_off = fx.Int32(0)
            pre = _ldl(qt, gh, trim=False)
            _tdm_qdo(qt, gh, cur_off)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
        lds_do = _lds0 + cur_off

        # The S/dP B operands, read back from the TDM image: lane (row, half) reads row
        # hh*16+row, bytes half*16 + dt*64 (+32).
        ops = qd if const_expr(carry) else _rdqd(cur_off)
        pst = []                    # P/dS stores deferred into one DS burst
        for hh in range_constexpr(2):
            qp = [(ops[hh * 16 + dt * 2], ops[hh * 16 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            dp = [(ops[hh * 16 + 8 + dt * 2], ops[hh * 16 + 8 + dt * 2 + 1])
                  for dt in range_constexpr(NDT)]
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16)))
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
                pst.append((fx.Vector.from_elements(p_l, dtype=fx.BFloat16), lds_p + off))
                pst.append((fx.Vector.from_elements(ds_l, dtype=fx.BFloat16), lds_ds + off))
        # ONE DS phase per iteration. Fence off the S/dP/softmax part, then issue: the 8 P/dS
        # stores, the A-operand tr16 loads, the B-operand tr16 loads, and (carry) the tensor
        # wait + 32 ds_load_b128 readback of the NEXT iteration's stage. DS completes in
        # order, so the first dK/dV WMMA waits only for the stores + its own operands.
        # The DS-phase bases are computed BEFORE the fence so their VALU->DS latency hides
        # under the S/dP phase.
        tr_base = lds_do + lb_tr
        if const_expr(carry):
            rb_base = _lds0 + rb_off + lb_rd
        rocdl.sched_barrier(0)
        for v_, a_ in pst:
            llvm_dialect.store(fx.as_ir_value(v_), create_llvm_ptr(a_, address_space=3))
        # One wave per workgroup: no barrier between the P/dS stores and their tr16 reads;
        # the backend derives the dscnt wait from the LDS dependence.

        # 32 queries staged, so the contraction is FULL: rows [lane_r] and [lane_r+16]
        # concatenate in lane into a v16 operand.
        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))).shuffle(
                fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(base + fx.Int32(16 * rowb),
                                         address_space=3))), list(range(16)))

        # The B operands (dO, Q) are shared by BOTH kv sub-tiles; the A operands differ
        # only by a 16-column (32-byte) offset into the same [q][kv] tile.
        def _a_tr(kh, base_):
            col = lane_c * fx.Int32(2) + fx.Int32(kh * 32)
            return tr(base_ + lane_r * fx.Int32(S_ROW_B) + col, S_ROW_B)

        if const_expr(carry):
            # DS issue order = the dK/dV WMMA consumption order (kh0: a_p x b_do,
            # a_ds x b_q; then kh1's A operands), each group fenced so the scheduler keeps it.
            a_pk = [None] * NKV
            a_dsk = [None] * NKV
            a_pk[0] = _a_tr(0, lds_p)
            rocdl.sched_barrier(0)
            b_do = [tr(tr_base + fx.Int32(dtile * 32), X_ROW_B) for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            a_dsk[0] = _a_tr(0, lds_ds)
            rocdl.sched_barrier(0)
            b_q = [tr(tr_base + fx.Int32(32 * X_ROW_B + dtile * 32), X_ROW_B)
                   for dtile in range_constexpr(NDO)]
            rocdl.sched_barrier(0)
            for kh in range_constexpr(1, NKV):
                a_pk[kh] = _a_tr(kh, lds_p)
                a_dsk[kh] = _a_tr(kh, lds_ds)
        else:
            a_pk = []
            a_dsk = []
            for kh in range_constexpr(NKV):
                a_pk.append(_a_tr(kh, lds_p))
                a_dsk.append(_a_tr(kh, lds_ds))
            b_do = []
            b_q = []
            for dtile in range_constexpr(NDO):
                b_do.append(tr(tr_base + fx.Int32(dtile * 32), X_ROW_B))
                b_q.append(tr(tr_base + fx.Int32(32 * X_ROW_B + dtile * 32), X_ROW_B))
        if const_expr(carry):
            # Outstanding TDM here: stage (it+1)%3 (issued in it-1) and (it+2)%3 (issued at
            # the top of this iteration). tensor_wait(2) retires the older pair, the one
            # about to be read (TDM ops retire in order).
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(2)
            rocdl.sched_barrier(0)
            rb = _rdqd(rb_off, rb_base)
        rocdl.sched_barrier(0)
        new = [None] * (2 * NKV * NDO)
        for kh in range_constexpr(NKV):
            a_p = a_pk[kh]
            a_ds = a_dsk[kh]
            # A-operand reuse hint: within a run of NDO WMMAs, A is one 16x32 subtile held
            # across all dtiles, so instructions 2..NDO of each run may reuse it.
            for dtile in range_constexpr(NDO):
                new[kh * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do[dtile]), acc[kh * NDO + dtile],
                    reuseA=(dtile > 0), reuseB=False).result
            for dtile in range_constexpr(NDO):
                new[(NKV + kh) * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_q[dtile]), acc[(NKV + kh) * NDO + dtile],
                    reuseA=(dtile > 0), reuseB=False).result
        if const_expr(carry):
            # Fence the dK/dV WMMAs off from the back-edge copies of the LSE/delta prefetch,
            # so their s_wait_loadcnt sits at the END of the body.
            rocdl.sched_barrier(0)
            return new + [_ir(v) for v in nxt] + [_ir(v) for v in rb]
        return new

    # Division-free loop index: the flat iteration index ii == qc*G + gc (0 <= gc < G) is
    # carried in the scf.for state as two uniform i32 counters advanced by a wrap
    # gc+1 == G -> (qc+1, 0); the ring stage offset cur = (ii % 3) * QDO_B is a third.
    # bounds_proof.py replays the counters against the division form for every iteration.
    def _wrap(qc, gc):
        g1 = gc + fx.Int32(1)
        nw = g1 < G
        return fx.Int32(nw.select(qc, qc + fx.Int32(1))), fx.Int32(nw.select(g1, fx.Int32(0)))

    @flyc.jit
    def qloop_mask(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            st = list(carried)
            qi, gh = fx.Int32(st[NST]), fx.Int32(st[NST + 1])
            qn, gn = _wrap(qi, gh)
            res = _body(st[:NST], None, qt0 + qi, gh, True,
                        None, None, False) + [fx.as_ir_value(qn), fx.as_ir_value(gn)]
            final = yield res
        return final

    @flyc.jit
    def qloop_full(state, n, qt0):
        final = state
        for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
            ii = fx.Int32(it)
            jj = ii + fx.Int32(1)
            st = list(carried)
            # counters (cur, qi, gh) at st[-3:], LSE/delta at st[NST:NST+4], carried B
            # operands at st[NST+4:-3].
            cur, qi, gh = fx.Int32(st[-3]), fx.Int32(st[-2]), fx.Int32(st[-1])
            qn, gn = _wrap(qi, gh)                     # decomposition of ii+1
            live = jj < n                              # else jj clamps to n-1 == ii
            qj = fx.Int32(live.select(qn, qi))
            gj = fx.Int32(live.select(gn, gh))
            q2, g2 = _wrap(qn, gn)                     # decomposition of ii+2
            live2 = (ii + fx.Int32(2)) < n             # else kk clamps to n-1 == jj
            pf_qt = qt0 + fx.Int32(live2.select(q2, qj))
            pf_gh = fx.Int32(live2.select(g2, gj))
            # prefetch stage (ii+2)%3 == (ii-1)%3: the stage before cur
            nxo = fx.Int32((cur == fx.Int32(0)).select(
                fx.Int32(2 * QDO_B), cur - fx.Int32(QDO_B)))
            # (ii+1)%3: next iteration's stage == this iteration's readback stage
            ncur = fx.Int32((cur == fx.Int32(2 * QDO_B)).select(
                fx.Int32(0), cur + fx.Int32(QDO_B)))
            res = _body(st[:NST], [fx.Vector(v) for v in st[NST:NST + 4]],
                        qt0 + qi, gh, False,
                        qt0 + qj, gj, True,
                        cur, nxo, pf_qt, pf_gh,
                        [fx.Vector(v) for v in st[NST + 4:-3]], ncur)
            res = res + [fx.as_ir_value(ncur), fx.as_ir_value(qn), fx.as_ir_value(gn)]
            final = yield res
        return final

    def _tdm_prologue(qt0, n):
        """Fill stages 0..1 for qloop_full's first two iterations: stage 0 gets iteration 0's
        tile (qt0, gh 0), stage 1 iteration min(1, n-1)'s, clamped to a legal query pair, so
        an empty or 1-iteration loop still issues only in-bounds tiles (never read). Then
        retire stage 0 and read back iteration 0's B operands."""
        _tdm_qdo(_clampqt(qt0), fx.Int32(0), fx.Int32(0))
        # j1 = max(min(1, n-1), 0) is 1 iff n > 1, and 1 == wrap(0, 0).
        q1w, g1w = _wrap(fx.Int32(0), fx.Int32(0))
        two = fx.Int32(1) < n
        q1 = fx.Int32(two.select(q1w, fx.Int32(0)))
        g1 = fx.Int32(two.select(g1w, fx.Int32(0)))
        _tdm_qdo(_clampqt(qt0 + q1), g1, fx.Int32(QDO_B))
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(2)
        rocdl.sched_barrier(0)
        return [_ir(v) for v in _rdqd(fx.Int32(0))]

    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is attended
    # by the pair's SMALLEST query: kv0 + BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0 + BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0 + fx.Int32(BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(2 * NKV * NDO)]
    # every loop starts at ii = 0: (qi, gh) = (0, 0), ring stage offset cur = 0.
    _z2 = [fx.as_ir_value(fx.Int32(0)), fx.as_ir_value(fx.Int32(0))]
    _z3 = [fx.as_ir_value(fx.Int32(0))] + _z2
    out = qloop_mask(init + _z2, G * nmaskp, qp_start)
    # The LSE/delta prefetch and the TDM prologue sit BETWEEN the loops, for qloop_full's
    # first iteration (qi = 0, gh = 0) at query pair qp_start + nmaskp (clamped: the
    # prologue issues even when the full loop is empty).
    _pc0 = _clampqt(qp_start + nmaskp)
    out = list(out)[:NST] + [_ir(v) for v in _ldl(_pc0, fx.Int32(0))]
    out = out + _tdm_prologue(qp_start + nmaskp, G * (nqp_eff - nmaskp)) + _z3
    out = qloop_full(out, G * (nqp_eff - nmaskp), qp_start + nmaskp)
    # Retire the last iteration's (clamped) prefetch -- or the prologue's, when the loop ran
    # 0 times -- before the epilogue reuses the start of LDS, which overlaps the ring.
    tdm_ops.tensor_wait(0)

    # Epilogue: stage the dK/dV accumulators through the finished ring as COLUMN-major
    # images M[d][kv] (a lane's 8 kv rows of one d column are contiguous: one ds_write_b128
    # per d tile) and read them back with ds_load_tr16_b128, which returns ROW-major (lane l:
    # kv row l%16, 8 consecutive d columns), so every global store is a 16-byte
    # buffer_store_b128. EPI_CB = 48 B = 32 B of kv row + 16 B pad (12 dwords: 16 distinct
    # bank groups).
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
        # lane_r / lane_c are the transposing-load address form of tr(): lane l addresses
        # row (l//16)*8 + l%8, column ((l//8)%2)*8, and receives column l%16 of rows
        # (l//16)*8 + e.
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
    _dkdv_impl(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv,
               G, nqt, cshift, causal, B_)


@flyc.jit
def launch_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    k_dkdv(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
           nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(32, 1, 1), stream=stream)


# ===================================================================== dqg ========
# k_dqg -- one wave owns DQ_BQW queries of one q head and streams every KV_STEP-row kv
# block through a 3-stage TDM K/V LDS ring. Global loop index i (full loop, then masked):
#   stage i%3        : tile i  -> K region read by the dQ tr16
#   stage (i+1)%3    : tile min(i+1, N-1) -> read back (ds_load_b128) into the carried S/dP
#                      A operands of iteration i+1
#   stage (i+2)%3    : TDM for tile min(i+2, N-1), issued at the top of iteration i
# N = nkvt_eff. The mask loop continues the SAME ring, so the full->mask handoff is just the
# carried state. No VMEM in the hot loop.
KV_STEP = 32                 # kv rows per k_dqg iteration: one WMMA contraction of dS
NKT = KV_STEP // 16
DQ_BQW = 64                  # queries per k_dqg wave
KV_B = 2 * KV_STEP * X_ROW_B          # one ring stage: K [32][X_ROW_B] + V at +32*X_ROW_B
DQT_TW = 2 * (TDM_DEPTH - 2)          # tensorcnt allowed at the readback: 2 TDM ops per stage
# Full-loop schedule ("ck"): kt0 S/dP WMMAs; kt1 S/dP in NDT chunks {8 WMMAs of dtile j +
# softmax/dS (kt0, qh j)}; the DS phase (tr16 | tensor_wait | readback); softmax/dS
# (kt1, qh 0); dQ qh-major in chunks {NDO dQ WMMAs of qh j + softmax/dS (kt1, qh j+1)}.
# Chunks end in sched_barrier(0); inside a chunk sched_group_barrier {1 WMMA, 2 VALU,
# 1 TRANS} per WMMA.
DQT_VT_SGB = (2, 1)


def _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ,
                  scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal):
    BQW, NQW = DQ_BQW, DQ_BQW // 16
    lane = fx.Int32(fx.thread_idx.x)
    # XCD-major q-head remap: workgroups go to the 8 XCDs round-robin on the linear id, so
    # x -> (x%8)*(Hq/8) + x/8 puts adjacent q heads (one kv head under GQA) on one XCD.
    # A bijection of [0, Hq) whenever Hq % 8 == 0, else the identity.
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    ngrp = Hq
    qh = (ngrp % _nx == fx.Int32(0)).select((_gx % _nx) * (ngrp // _nx) + _gx // _nx, _gx)
    # longest-first: the query tile is walked DESCENDING on grid.y
    bid = (Sq // fx.Int32(BQW) - fx.Int32(1) - fx.Int32(fx.block_idx.y))
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0 = bid * fx.Int32(BQW)
    hkv = qh // G

    # Fake 1 GiB / 256 MiB extents (impl.py asserts every tensor fits): never "fix" these to
    # the k_dkdv formula -- k_dqg has no batch-count argument.
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.BFloat16)

    rs_q = Hq * fx.Int32(DV8)
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(TDM_DEPTH * KV_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
          for qh_ in range(NQW)]
    dof = [[gfrag(g_do, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT)]
           for qh_ in range(NQW)]
    q_glob = [q0 + fx.Int32(qh_ * 16) + row for qh_ in range(NQW)]
    lse_q = [_ld1(g_lse, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    del_q = [_ld1(g_del, base_l + q_glob[qh_], fx.Float32) for qh_ in range(NQW)]
    # softmax row constants, hoisted out of the kv loops:
    #   p = exp2(fma(s, scale*LOG2E, -lse*LOG2E)); ds = bf16(p * fma(dp, scale, -delta*scale))
    _c1q = scale * fx.Float32(LOG2E)
    _nlq = [lse_q[qh_] * fx.Float32(-LOG2E) for qh_ in range(NQW)]
    _ndq = [del_q[qh_] * (fx.Float32(0.0) - scale) for qh_ in range(NQW)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # lane-only parts of the two DS address families; every per-op difference is a
    # compile-time immediate (max 32*X_ROW_B + 16*X_ROW_B + 7*64 + 32 + 224 < 65536).
    lb_tr = lane_r * fx.Int32(X_ROW_B) + lane_c * fx.Int32(2)   # dQ B operand (K) tr16
    lb_rd = row * fx.Int32(X_ROW_B) + half * fx.Int32(16)        # S/dP A readback

    X_ROW_EL = X_ROW_B // 2
    _lds_bf_ty = fx.PointerType.get(elem_ty=fx.BFloat16.ir_type,
                                    address_space=fx.AddressSpace.Shared, alignment=16)
    _kv_rs_el = Hkv * fx.Int32(D)               # elements between consecutive kv rows

    def _tdm_kv(kv0p, stage_off):
        """K then V tile [kv0p, kv0p+32) x [0, D) of (bat, hkv) into stage_off / +32 rows.
        Origin element ((bat*Skv + kv0p)*Hkv + hkv)*D, row stride Hkv*D, outer extent
        Skv - kv0p >= 32 (kv0p <= (nkvt-1)*32, bounds_proof.py T1)."""
        off = (fx.Int64(bat * Skv + kv0p) * fx.Int64(Hkv) + fx.Int64(hkv)) * fx.Int64(D)
        valid = Skv - kv0p
        for src, lb in ((K, _lds0 + stage_off),
                        (V, _lds0 + stage_off + fx.Int32(KV_STEP * X_ROW_B))):
            g_view = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), off),
                                            fx.make_layout((KV_STEP, D), (D, 1))))
            atom = fx.rocdl.cdna5.make_tdm_atom(
                g_view, [valid, None], strides=[_kv_rs_el, None], num_warps=1,
                pad_interval=D, pad_amount=X_ROW_EL - D)
            l_view = fx.Tensor(fx.make_view(fx.inttoptr(_lds_bf_ty, lb),
                                            fx.make_layout((KV_STEP, D), (X_ROW_EL, 1))))
            fx.copy_atom_call(atom, g_view, l_view)

    def _rdkv(stage_off):
        """32 ds_load_b128: entry (kt*NDT + dt)*4 + {0,1,2,3} = K u0, K u1, V u0, V u1 of
        row kt*16+row, bytes half*16 + dt*64 + u*32."""
        rb = _lds0 + stage_off + lb_rd
        out = []
        for kt in range_constexpr(NKT):
            for dt in range_constexpr(NDT):
                for vo in (0, KV_STEP * X_ROW_B):
                    for u in range_constexpr(2):
                        out.append(fx.Vector(llvm_dialect.load(
                            v8b, create_llvm_ptr(
                                rb + fx.Int32(kt * 16 * X_ROW_B + vo + dt * 64 + u * 32),
                                address_space=3))))
        return out

    def _bks(cur):
        tb = _lds0 + cur + lb_tr
        out = []
        for dtile in range_constexpr(NDO):
            base = tb + fx.Int32(dtile * 32)
            out.append(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * X_ROW_B), address_space=3))),
                list(range(16))))
        return out

    def _smx(sv, pv_, qh_, kt, kv0, do_mask):
        """softmax/dS of one (kt, qh) -> 8 bf16."""
        tt = [fx.Float32(fx.fma(sv[si], _c1q, _nlq[qh_])) for si in range(8)]
        if const_expr(do_mask):
            tt = [((kv0 + fx.Int32(kt * 16) + half * fx.Int32(8) + fx.Int32(si)
                    > q_glob[qh_] + cshift) & (causal != fx.Int32(0))
                   ).select(fx.Float32(NEG), tt[si]) for si in range(8)]
        pf = [_exp2(tt[si]) for si in range(8)]
        return [(pf[si] * fx.Float32(fx.fma(pv_[si], scale, _ndq[qh_])))
                .to(fx.BFloat16) for si in range(8)]

    def _sgb(nw):
        # {1 WMMA, DQT_VT_SGB[0] VALU, DQT_VT_SGB[1] TRANS} x nw
        for _ in range_constexpr(nw):
            rocdl.sched_group_barrier(0x008, 1, 0)
            if const_expr(DQT_VT_SGB[0] > 0):
                rocdl.sched_group_barrier(0x002, DQT_VT_SGB[0], 0)
            if const_expr(DQT_VT_SGB[1] > 0):
                rocdl.sched_group_barrier(0x400, DQT_VT_SGB[1], 0)

    def _body_ck(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        # Full loop: the WMMA<->softmax interleave described at DQT_VT_SGB.
        assert NQW == NDT == 4 and NKT == 2
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        sp_acc = []
        ds_halves = [[] for _ in range_constexpr(NQW)]
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            for dt in range_constexpr(NDT):
                pi = (kt * NDT + dt) * 4
                kfr = pre[pi].shuffle(pre[pi + 1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQW):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_],
                        reuseA=False, reuseB=False).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_],
                        reuseA=False, reuseB=False).result
                if const_expr(kt == 1):
                    # chunk dt: this dtile's 8 kt1 WMMAs + softmax/dS of (kt0, qh = dt)
                    s0, p0 = sp_acc[0]
                    ds_halves[dt].append(_smx(fx.Vector(s0[dt]), fx.Vector(p0[dt]),
                                              dt, 0, kv0, False))
                    _sgb(2 * NQW)
                    rocdl.sched_barrier(0)
            if const_expr(kt == 0):
                rocdl.sched_barrier(0)            # kt0 S/dP WMMAs: their own region
            sp_acc.append((s_acc, p_acc))
        s1, p1 = sp_acc[1]
        # the DS phase
        rocdl.sched_barrier(0)
        b_ks = _bks(cur)
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(DQT_TW)
        rocdl.sched_barrier(0)
        rb = _rdkv(ncur)
        rocdl.sched_barrier(0)
        ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
        rocdl.sched_barrier(0)
        new = [None] * (NQW * NDO)
        for qh_ in range_constexpr(NQW):
            a_ds = fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                           dtype=fx.BFloat16)
            for dtile in range_constexpr(NDO):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_ks[dtile]), acc[qh_ * NDO + dtile],
                    reuseA=False, reuseB=False).result
            if const_expr(qh_ + 1 < NQW):
                ds_halves[qh_ + 1].append(_smx(fx.Vector(s1[qh_ + 1]),
                                               fx.Vector(p1[qh_ + 1]),
                                               qh_ + 1, 1, kv0, False))
                _sgb(NDO)
            rocdl.sched_barrier(0)
        return new + [_ir(v) for v in rb]

    def _body(acc, pre, kv0, do_mask, pf_kv0, cur, nxo, ncur):
        if const_expr(not do_mask):
            return _body_ck(acc, pre, kv0, pf_kv0, cur, nxo, ncur)
        # Masked loop: S/dP WMMAs | DS phase | softmax VALU (hides the DS latency) + dQ.
        # TDM for tile min(i+2, N-1) into stage (i-1)%3: no wait, that stage's last readers
        # (dQ tr16 of i-1, readback in i-2) were consumed by WMMAs of i-1.
        rocdl.sched_barrier(0)
        _tdm_kv(pf_kv0, nxo)
        rocdl.sched_barrier(0)
        sp_acc = []
        for kt in range_constexpr(NKT):
            s_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            p_acc = [_ir(fx.Vector.filled(8, 0.0, fx.Float32))
                     for _ in range_constexpr(NQW)]
            for dt in range_constexpr(NDT):
                pi = (kt * NDT + dt) * 4
                kfr = pre[pi].shuffle(pre[pi + 1], list(range(16)))
                vfr = pre[pi + 2].shuffle(pre[pi + 3], list(range(16)))
                for qh_ in range_constexpr(NQW):
                    s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_],
                        reuseA=False, reuseB=False).result
                    p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                        v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_],
                        reuseA=False, reuseB=False).result
            sp_acc.append((s_acc, p_acc))

        def _softmax():
            ds_halves = [[] for _ in range_constexpr(NQW)]
            for kt in range_constexpr(NKT):
                s_acc, p_acc = sp_acc[kt]
                for qh_ in range_constexpr(NQW):
                    sv, pv_ = fx.Vector(s_acc[qh_]), fx.Vector(p_acc[qh_])
                    ds_halves[qh_].append(_smx(sv, pv_, qh_, kt, kv0, do_mask))
            return ds_halves

        # ONE DS phase: dQ B operands (tr16 of this stage's K), then retire the stage of tile
        # min(i+1, N-1) (outstanding: that stage and the one issued above; TDM retires in
        # order, so tensor_wait(DQT_TW) retires exactly the older one) and read it back.
        rocdl.sched_barrier(0)
        b_ks = _bks(cur)
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(DQT_TW)
        rocdl.sched_barrier(0)
        rb = _rdkv(ncur)
        rocdl.sched_barrier(0)
        ds_halves = _softmax()
        a_ds = [fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                        dtype=fx.BFloat16)
                for qh_ in range_constexpr(NQW)]
        new = [None] * (NQW * NDO)
        for qh_ in range_constexpr(NQW):
            for dtile in range_constexpr(NDO):
                new[qh_ * NDO + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_ks[dtile]), acc[qh_ * NDO + dtile],
                    reuseA=False, reuseB=False).result
        return new + [_ir(v) for v in rb]

    NACC = NQW * NDO
    NP = 4 * NKT * NDT

    def _mkloop(do_mask):
        @flyc.jit
        def kvloop(state, n, i0, nlast):
            # carried: acc[NACC] + readback A operands[NP] + cur (ring offset of stage i%3)
            final = state
            for it, carried in range(fx.Index(fx.Int32(0)), fx.Index(n), 1, init=state):
                st = list(carried)
                cur = fx.Int32(st[-1])
                ii = fx.Int32(it) + i0
                kk = ii + fx.Int32(TDM_DEPTH - 1)
                kk = (kk < nlast).select(kk, nlast)              # min(i+2, N-1)
                nxo = fx.Int32((cur == fx.Int32(0)).select(      # (i+2)%3 == (i-1)%3
                    fx.Int32((TDM_DEPTH - 1) * KV_B), cur - fx.Int32(KV_B)))
                ncur = fx.Int32((cur == fx.Int32((TDM_DEPTH - 1) * KV_B)).select(
                    fx.Int32(0), cur + fx.Int32(KV_B)))         # (i+1)%3
                res = _body(st[:NACC], [fx.Vector(v) for v in st[NACC:NACC + NP]],
                            ii * fx.Int32(KV_STEP), do_mask,
                            kk * fx.Int32(KV_STEP), cur, nxo, ncur)
                final = yield res + [fx.as_ir_value(ncur)]
            return final
        return kvloop

    kvloop_full = _mkloop(False)
    kvloop_mask = _mkloop(True)

    _lim = (q0 + fx.Int32(BQW) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0 + cshift + fx.Int32(1)
    _nf = (_t < fx.Int32(0)).select(fx.Int32(0), _t // fx.Int32(KV_STEP))
    _nf = (_nf < nkvt_eff).select(_nf, nkvt_eff)
    nfull = (causal != fx.Int32(0)).select(_nf, nkvt_eff)
    nlast = nkvt_eff - fx.Int32(1)                       # >= 0: nkvt_eff >= 1

    # prologue: stage s = tile min(s, N-1) for s = 0..1; retire stage 0, read it back.
    _tdm_kv(fx.Int32(0), fx.Int32(0))
    for s_ in range_constexpr(1, TDM_DEPTH - 1):
        t1 = (fx.Int32(s_) < nlast).select(fx.Int32(s_), nlast)
        _tdm_kv(t1 * fx.Int32(KV_STEP), fx.Int32(s_ * KV_B))
    rocdl.sched_barrier(0)
    tdm_ops.tensor_wait(DQT_TW)
    rocdl.sched_barrier(0)
    rb0 = _rdkv(fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NACC)]
    out = kvloop_full(init + [_ir(v) for v in rb0] + [fx.as_ir_value(fx.Int32(0))],
                      nfull, fx.Int32(0), nlast)
    out = kvloop_mask(list(out), nkvt_eff - nfull, nfull, nlast)
    # retire the last two (clamped, never read) prefetches before the wave ends: the
    # workgroup's LDS must not be written after it is released.
    tdm_ops.tensor_wait(0)
    base_o = bat * Sq * Hq * fx.Int32(D) + qh * fx.Int32(D)
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO):
            ov = fx.Vector(out[qh_ * NDO + dtile])
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                _st1(ov[si].to(fx.BFloat16), g_dq,
                     base_o + q_i * Hq * fx.Int32(D) + fx.Int32(dtile * 16) + row,
                     fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dqg(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor, O: fx.Tensor,
          LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
          scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
          G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32):
    _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
                  nkvt, cshift, causal)


@flyc.jit
def launch_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale: fx.Float32,
               Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
               nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
               nblk: fx.Int32, ngrp: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    # grid = (Hq q heads, Sq / DQ_BQW query tiles, B)
    k_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
          nkvt, cshift, causal).launch(
        grid=(ngrp, nblk, nb), block=(32, 1, 1), stream=stream)
