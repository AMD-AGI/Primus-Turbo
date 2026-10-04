"""gfx1250 FlyDSL flash-attention backward: k_delta, k_dkdv, k_dqg (wave32 WMMA, bf16).

    delta[b,h,s] = sum_d dO*O                                   k_delta_bshd
    dV[kv,:] = sum_q P^T dO ;  dK[kv,:] = sum_q dS^T Q           k_dkdv64 (two waves per 64-row kv
                                                                 block, one shared Q/dO ring)
    dQ[q,:]  = sum_kv dS K                                       k_dqg96 (two waves x 48 queries
                                                                 share one K/V ring) + k_dqg
                                                                 (32-query head tiles)

Head dims: D_QK for q/k/dq/dk (S = Q K^T contracts D_QK), D_V for v/o/do/dv (dP = dO V^T
contracts D_V); DeepSeek-V3 MLA is D_QK = 192 (128 nope + 64 rope), D_V = 128.
Layouts: q [B, Sq, Hq, D_QK], o/do [B, Sq, Hq, D_V], k [B, Skv, Hkv, D_QK], v [B, Skv, Hkv,
D_V], all bf16; lse/delta [B, Hq, Sq] fp32, natural log; dq/dk/dv bf16 like q/k/v. Causal is
bottom-right (query i sees keys j <= i + Skv - Sq). GQA is reduced inside k_dkdv (the G q
heads of one kv head stream through the same accumulators): no atomics, every output
element written once, deterministic.

k_dkdv stages Q/dO through a 3-stage Tensor-Data-Mover LDS ring (prefetch two iterations
ahead) and reads the next iteration's S/dP B operands back into VGPRs one iteration early;
k_dqg does the same with a 3-stage K/V ring. k_delta, k_dqg and the legacy one-wave k_dkdv
run one wave32 per workgroup with no s_barrier; k_dkdv64 runs two waves (one per SIMD) over ONE
Q/dO ring and synchronises it with one workgroup barrier per full-loop iteration (protocol at
_dkdv_impl); k_dqg96 runs two waves on one K/V ring (each TDMs half of every tile) with one
workgroup barrier per kv step (_wg_sync). The design history behind every choice is in
PROVENANCE.md.
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
from flydsl._mlir import ir as _mlir_ir  # UNSTABLE(gfx1250): gpu.barrier memfence attribute
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


D_QK = 192                   # q/k head dim
D_V = 128                    # v/o head dim
DV8_QK = D_QK // 8           # vec8 tiles per row: q, k, dq, dk
DV8_V = D_V // 8             # vec8 tiles per row: v, o, do, dv
NDT_QK = D_QK // 32          # WMMA k-steps contracting S = K Q^T
NDT_V = D_V // 32            # WMMA k-steps contracting dP = V dO^T
NDT_MAX = max(NDT_QK, NDT_V)
NDO_QK = D_QK // 16          # 16-wide output tiles across d: dK, dQ
NDO_V = D_V // 16            # 16-wide output tiles across d: dV
NDO_MAX = max(NDO_QK, NDO_V)
SAME_D = D_QK == D_V         # one shared address family when the row strides agree
NKV = 2                      # 16-row kv sub-tiles per k_dkdv workgroup
NST = NKV * (NDO_V + NDO_QK)  # dV then dK accumulators carried by k_dkdv
WAVE = 32                    # gfx1250 dispatches wave32
LOG2E = 1.4426950408889634
NEG = -3.0e38
TDM_DEPTH = 3                # Q/dO (k_dkdv) and K/V (k_dqg) LDS ring stages
BLOCK_KV = 32                # kv rows one k_dkdv workgroup owns
S_ROW_B = BLOCK_KV * 2 + 16  # padded P/dS LDS row: 80 B = 20 dwords walks all 64 banks
# Padded LDS rows of the TDM-staged tiles: D*2 + 16 bytes. An unpadded 256 B row puts every
# row on bank 0 (64-way conflict on the 16 rows of a tr16 read); +16 B makes row r start at
# dword 4r (D=128, 68-dword stride) or 36r (D=192, 100-dword stride) mod 64 -- 16 distinct
# 4-bank groups either way.
XK_ROW_B = D_QK * 2 + 16     # q / k rows (400 B at D_QK = 192)
XV_ROW_B = D_V * 2 + 16      # do / v rows (272 B at D_V = 128)


def _pow2_segments(d):
    """[(col0, width)] covering [0, d) with power-of-two widths, largest first. A TDM pad
    interval must be a power-of-two number of dwords, so a 192-wide row takes two ops
    (128 + 64), each padded to the same LDS row stride."""
    segs, c0 = [], 0
    while c0 < d:
        w = 1 << ((d - c0).bit_length() - 1)
        segs.append((c0, w))
        c0 += w
    return segs


for _d in (D_QK, D_V):
    assert _d % 32 == 0, _d
    for _c0, _w in _pow2_segments(_d):
        # descriptor fields: pad_interval = log2(w/2 dwords) - 1 in 3 bits, pad_amount =
        # (row - w)/2 dwords - 1 in 7 bits (flydsl tdm_ops.compute_padding_encoding)
        assert 4 <= _w <= 512 and 1 <= (_d + 8 - _w) // 2 <= 128, (_d, _w)

# k_dkdv LDS: the Q/dO ring in segment 0 -- per stage the [32][D_V] dO tile at +0 and the
# [32][D_QK] Q tile at +QOFF -- and the P/dS tiles one 64 KB segment up (LDS_SEG).
LDS_SEG = 65536
QOFF = 32 * XV_ROW_B
QDO_B = 32 * (XV_ROW_B + XK_ROW_B)           # 21504 B at 192/128 (17408 at 128/128)
TDM_OPS_QDO = len(_pow2_segments(D_V)) + len(_pow2_segments(D_QK))  # TDM ops per stage
TW_QDO = TDM_OPS_QDO * (TDM_DEPTH - 2)      # tensorcnt that retires all but the newest stage
NQE = 2 * NDT_QK             # carried Q readback entries per 16-query half (u = 0, 1)
NHB = NQE + 2 * NDT_V        # + dO entries
DK0 = NKV * NDO_V            # dK accumulators follow the dV ones
EPI_CB = 48                  # epilogue d-row: 32 B of kv row + 16 B pad (12 dwords)
EPI_W = NKV * (D_V + D_QK) * EPI_CB          # one wave's dV + dK epilogue images (30720 B)
# k_dkdv64: DKDV_NW waves per workgroup; wave w owns kv rows [kv0g + 32w, kv0g + 32w + 32) of the
# workgroup's DKDV_NW*BLOCK_KV-row block. The waves share ONE Q/dO ring (TDM issued cooperatively,
# num_warps = DKDV_NW: wave w moves rows [16w, 16w + 16) of every 32-row tile); each keeps its own
# P/dS tiles at LDS_SEG + w*PDS_B and its own epilogue image at w*EPI_W inside the dead ring.
DKDV_NW = 2
PDS_B = 2 * 32 * S_ROW_B     # one wave's P + dS tiles (5120 B)
BAR_KH = NKV // 2            # k_dkdv64 full loop: the ring barrier sits before kv sub-tile BAR_KH's WMMAs
assert TDM_DEPTH * QDO_B <= LDS_SEG, "the Q/dO ring must stay inside LDS segment 0"
assert DKDV_NW * EPI_W <= TDM_DEPTH * QDO_B, "epilogue images must fit the dead ring"
assert 32 % DKDV_NW == 0 and (DKDV_NW & (DKDV_NW - 1)) == 0, "TDM num_warps splits 32 rows evenly"

DELTA_THREADS = 256
LANES_PER_ROW = D_V // 8
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


def _lds_barrier():
    """Workgroup barrier whose fences cover LDS only (gpu.barrier memfence [workgroup]): the
    release side drains this wave's DS ops (s_wait_dscnt 0) but not its in-flight global
    loads (the LSE/delta prefetch), which a plain gpu.barrier() also waits for."""
    fx.gpu.barrier(address_spaces=_mlir_ir.Attribute.parse("[#gpu.address_space<workgroup>]"))


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _tdm_rows(src, off, d, rows, valid, rs_el, lds_row0, lds_ty, num_warps=1):
    """TDM the [rows][d] bf16 tile at element `off` of `src` (row stride rs_el elements,
    outer extent `valid` rows) into LDS at byte address lds_row0, row stride d*2+16 bytes:
    one TDM op per power-of-two column segment, so len(_pow2_segments(d)) ops. num_warps > 1
    is the cooperative form: every wave of the workgroup issues the same ops and wave w moves
    rows [w*rows/num_warps, (w+1)*rows/num_warps) (FlyDSL splits the outer dim by wave_id; the
    per-wave outer extent clamp is valid - w*rows/num_warps). Used by k_dkdv64 (Q/dO ring) and
    k_dqg96 (K/V ring): each wave's tensorcnt then counts only its own len(_pow2_segments(d))
    ops (FlyDSL GFX1250 TDM lowering: per-wave global/LDS offset from rocdl.wave_id, the fwd's
    num_warps 8 pattern)."""
    x_el = d + 8
    for c0, w in _pow2_segments(d):
        g_off = off if c0 == 0 else off + fx.Int64(c0)
        lb = lds_row0 if c0 == 0 else lds_row0 + fx.Int32(c0 * 2)
        g_view = fx.Tensor(fx.make_view(fx.add_offset(fx.get_iter(src), g_off),
                                        fx.make_layout((rows, w), (d, 1))))
        atom = fx.rocdl.cdna5.make_tdm_atom(
            g_view, [valid, None], strides=[rs_el, None], num_warps=num_warps,
            pad_interval=w, pad_amount=x_el - w)
        l_view = fx.Tensor(fx.make_view(fx.inttoptr(lds_ty, lb),
                                        fx.make_layout((rows, w), (x_el, 1))))
        fx.copy_atom_call(atom, g_view, l_view)


# ================================================================== delta ==========
@flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
def k_delta_bshd(DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor,
                 S: fx.Int32, H: fx.Int32, n_rows: fx.Int32):
    """delta[b, h, s] = sum_d dO[b, s, h, d] * O[b, s, h, d], fp32."""
    tid = fx.Int32(fx.thread_idx.x)
    bid = fx.Int32(fx.block_idx.x)
    g_do = _bv(DO, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_o = _bv(O, n_rows * (D_V * 2), fx.BFloat16, 8)
    g_delta = _bv(DEL, n_rows * 4, fx.Float32)

    tile = bid * (ROWS_DELTA * DV8_V) + tid
    do_vecs = [_ldv(g_do, tile + u * (ROWS_PER_PASS * DV8_V), fx.BFloat16, 8).ir_value()
               for u in range_constexpr(PASSES_PER_WG)]
    o_vecs = [_ldv(g_o, tile + u * (ROWS_PER_PASS * DV8_V), fx.BFloat16, 8).ir_value()
              for u in range_constexpr(PASSES_PER_WG)]

    lane_in_row = tid % fx.Int32(LANES_PER_ROW)
    row_in_group = tid // fx.Int32(LANES_PER_ROW)
    for u in range_constexpr(PASSES_PER_WG):
        do8 = fx.Vector(do_vecs[u])
        o8 = fx.Vector(o_vecs[u])
        e0 = fx.Float32(0.0)
        e1 = fx.Float32(0.0)
        for c in range_constexpr(4):   # 8 elements of the lane's vec8, 2 chains
            e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
            e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
        acc = e0 + e1
        for sft in range_constexpr(LANES_PER_ROW.bit_length() - 1):   # LANES_PER_ROW lanes/row
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
               scale, Sq, Skv, Hq, Hkv, G, nqt, cshift, causal, B_, nw=1):
    """One wave owns a BLOCK_KV-row kv tile of one kv head and streams every (q head, q pair).

    nw (compile-time) waves per workgroup. nw = 1: the legacy k_dkdv, grid = (Hkv,
    Skv/BLOCK_KV, B), no barrier. nw = DKDV_NW: k_dkdv64, grid = (Hkv, Skv/(nw*BLOCK_KV), B);
    wave w (rocdl.wave_id, an SGPR) owns kv rows kv0g + w*BLOCK_KV + [0, BLOCK_KV) and every
    wave streams the SAME (q pair, q head) sequence -- the workgroup's -- out of ONE Q/dO ring.
    The accumulators persist across the G q heads that share this kv head, so the GQA
    reduction happens in registers -- atomic-free, written once.

    k_dkdv64 ring protocol. Every wave runs the same instruction stream: the wave index only
    offsets addresses (kv rows, P/dS tiles, epilogue image, its TDM half), no branch depends on
    it, and every trip count comes from the block ids and kernel args, so every wave takes the
    same barriers: 2 per masked iteration + 2 in the prologue + 1 per full iteration + 1 before
    the epilogue (bounds_proof.py K6 replays them). BARRIER = _lds_barrier(): LDS-only
    workgroup release fence (s_wait_dscnt 0: this wave's DS ops retired; TDM retirement is the
    explicit tensor_wait before it) + s_barrier_signal/wait + acquire fence.
      masked iteration: BARRIER (WAR: every wave's reads of stage 0 retired) -> TDM own half of
        the tile into stage 0 -> tensor_wait(0) -> BARRIER (RAW) -> readback/tr16 of stage 0.
      prologue: BARRIER (WAR vs the masked loop) -> TDM stages 0, 1 -> tensor_wait(TW_QDO)
        (own half of stage 0) -> BARRIER (RAW) -> readback of stage 0.
      full iteration ii: TDM into (ii+2)%3 at the top [WAR: its last readers -- readback in
        ii-2, tr16 in ii-1 -- retired before BARRIER(ii-1)]; tr16 of stage ii%3; dK/dV WMMAs of
        kv sub-tiles < BAR_KH; tensor_wait(TW_QDO) (own half of stage (ii+1)%3); BARRIER(ii);
        readback of stage (ii+1)%3 [RAW: every wave's half retired before BARRIER(ii)]; WMMAs
        of kv sub-tiles >= BAR_KH (they hide the readback latency).
      exit: tensor_wait(0) -> BARRIER (every wave's last TDM landed, every ring read retired)
        -> the epilogue images overwrite the ring.
    """
    if const_expr(nw == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(WAVE)
    hkv = fx.Int32(fx.block_idx.x)              # kv head (cheap axis fastest)
    bid = fx.Int32(fx.block_idx.y)              # kv tile (nw = 1) / kv block (nw > 1)
    bat = fx.Int32(fx.block_idx.z)              # batch
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    kv0 = bid * fx.Int32(BLOCK_KV)
    kv0g = kv0                                  # the workgroup's first kv row (loop bounds)
    if const_expr(nw > 1):
        wv = fx.Int32(rocdl.wave_id())          # wave in the workgroup, [0, nw), SGPR
        kv0g = bid * fx.Int32(nw * BLOCK_KV)
        kv0 = kv0g + wv * fx.Int32(BLOCK_KV)    # this wave's kv tile

    # TRUE byte extents on every k_dkdv descriptor: an over-read returns 0 instead of
    # walking into the next page, and a clamp that hit a LIVE access would crater SQNR.
    nk_b = B_ * Skv * Hkv * fx.Int32(D_QK * 2)  # k / dk bf16 [B, Skv, Hkv, D_QK]
    nv_b = nk_b if SAME_D else B_ * Skv * Hkv * fx.Int32(D_V * 2)  # v / dv [.., D_V]
    nl_b = B_ * Hq * Sq * fx.Int32(4)           # lse / delta fp32 [B, Hq, Sq]
    g_k = _bv(K, nk_b, fx.BFloat16, 8)
    g_v = _bv(V, nv_b, fx.BFloat16, 8)
    g_lse = _bv(LSE, nl_b, fx.Float32)
    g_del = _bv(DEL, nl_b, fx.Float32)

    rs_k = Hkv * fx.Int32(DV8_QK)               # vec8 tiles between consecutive k rows
    base_k = bat * Skv * rs_k + hkv * fx.Int32(DV8_QK)
    rs_v = rs_k if SAME_D else Hkv * fx.Int32(DV8_V)
    base_v = base_k if SAME_D else bat * Skv * rs_v + hkv * fx.Int32(DV8_V)

    # LDS: the Q/dO ring at offset 0 (segment 0), the P/dS tiles exactly one 64 KB segment
    # up, so the output GEMM's A reads (P, dS) and B reads (dO, Q) use different LDS read
    # ports whatever the workgroup's physical LDS base is.
    # nw > 1: one shared ring, a P/dS tile pair per wave (wave w at LDS_SEG + w*PDS_B).
    smem = fx.SharedAllocator().allocate(LDS_SEG + nw * PDS_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    lds_p = _lds0 + fx.Int32(LDS_SEG)
    if const_expr(nw > 1):
        lds_p = lds_p + wv * fx.Int32(PDS_B)
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
            # soffset = 4*(base_l + q0) + 64*hh: the same address (bounds_proof.py K3)
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

    # Q/dO staging through the Tensor Data Mover into a TDM_DEPTH-stage LDS ring. Stage s
    # is [s*QDO_B, (s+1)*QDO_B): the [32 q rows][D_V] dO tile at +0 (row stride XV_ROW_B),
    # the [32][D_QK] Q tile at +QOFF (row stride XK_ROW_B); element c of a row at byte 2c.
    # Each tile is TDM_OPS_QDO-many ops in total, one per power-of-two column segment
    # (_tdm_rows). Tile origin element ((bat*Sq + q0)*Hq + qh)*D, outer stride Hq*D, outer
    # extent Sq - q0 (>= 32 for every issued tile: Sq % 64 == 0 is asserted by impl.py and
    # qt is clamped to [0, nqt2-1]).
    _lds_bf_ty = fx.PointerType.get(elem_ty=fx.BFloat16.ir_type,
                                    address_space=fx.AddressSpace.Shared, alignment=16)
    _q_rs_v = Hq * fx.Int32(D_V)                # elements between consecutive do rows
    _q_rs_k = _q_rs_v if SAME_D else Hq * fx.Int32(D_QK)

    def _tdm_qdo(qt, gh, stage_off):
        qh = hkv * G + gh
        q0 = qt * fx.Int32(32)
        row0 = fx.Int64(bat * Sq + q0) * fx.Int64(Hq) + fx.Int64(qh)
        off_v = row0 * fx.Int64(D_V)
        off_k = off_v if SAME_D else row0 * fx.Int64(D_QK)
        valid = Sq - q0
        lb_v = _lds0 + stage_off
        lb_k = _lds0 + stage_off + fx.Int32(QOFF)
        # nw > 1: cooperative, wave w moves rows [16w, 16w + 16) (own tensorcnt)
        _tdm_rows(DO, off_v, D_V, 32, valid, _q_rs_v, lb_v, _lds_bf_ty, nw)
        _tdm_rows(Q, off_k, D_QK, 32, valid, _q_rs_k, lb_k, _lds_bf_ty, nw)

    # K and V fragments of this kv tile are invariant over every query and every q head.
    kf = [[gfrag(g_k, base_k, rs_k, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_QK)]
          for kh in range(NKV)]
    vf = [[gfrag(g_v, base_v, rs_v, kv0 + fx.Int32(kh * 16), dt) for dt in range(NDT_V)]
          for kh in range(NKV)]
    lane_r = (lane // fx.Int32(16)) * fx.Int32(8) + lane % fx.Int32(8)
    lane_c = ((lane // fx.Int32(8)) % fx.Int32(2)) * fx.Int32(8)
    # Lane-only parts of the DS-phase address families (dO rows / Q rows); every per-op
    # difference is a compile-time constant ISel folds into the 16-bit DS immediate
    # (bounds_proof.py: max immediate < 65536).
    lb_tr_v = lane_r * fx.Int32(XV_ROW_B) + lane_c * fx.Int32(2)   # b_do tr16
    lb_rd_v = row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)        # dO readback
    lb_tr_k = lb_tr_v if SAME_D else lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)
    lb_rd_k = lb_rd_v if SAME_D else row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)

    def _rd_bases(stage_off):
        """(Q base, dO base) of a stage's readback: rows are addressed base + constant."""
        b_v = _lds0 + stage_off + lb_rd_v
        return (b_v, b_v) if SAME_D else (_lds0 + stage_off + lb_rd_k, b_v)

    # Causal tile skip: under bottom-right causal, kv row j is attended only by queries
    # q >= j - cshift, so query pairs below qp_start contribute exactly zero to dK/dV. All loop
    # bounds use the WORKGROUP's kv rows [kv0g, kv0g + nw*BLOCK_KV) (block id only): a pair
    # below qp_start is fully masked for every wave, a pair from qp_start + nmaskp on fully
    # unmasked for every wave; inside the masked pairs each wave masks with its own kv0.
    nqt2 = nqt // fx.Int32(2)                   # query tiles come in PAIRS

    def _clampqt(t):
        """Pin a prefetch's query pair into [0, nqt2-1]."""
        t = (t < nqt2).select(t, nqt2 - fx.Int32(1))
        return (t < fx.Int32(0)).select(fx.Int32(0), t)

    _c = kv0g - cshift
    qp_start = ((_c < fx.Int32(0)).select(fx.Int32(0), _c)) // fx.Int32(32)
    qp_start = (causal != fx.Int32(0)).select(qp_start, fx.Int32(0))
    nqp_eff = nqt2 - qp_start

    # Iteration order is query-pair-OUTER, q-head-INNER (qi = ii // G, gh = ii % G), which
    # puts every masked iteration at the front: exactly one query pair per q head straddles
    # this kv tile, so the mask runs in its own loop (qloop_mask) and the hot loop
    # (qloop_full) carries no mask at all.
    def _rdqd(stage_off, rbase=None):
        """The 2*NHB ds_load_b128 that read one stage's S/dP B operands. Entry
        hh*NHB + dt*2 + u is Q row hh*16+row, bytes half*16 + dt*64 + u*32; entry
        hh*NHB + NQE + dt*2 + u the same for dO. Every address is a base + a constant."""
        if const_expr(rbase is None):
            rbase = _rd_bases(stage_off)
        out = []
        for hh in range_constexpr(2):
            for rb_, img, xrow, ndt in ((rbase[0], QOFF, XK_ROW_B, NDT_QK),
                                        (rbase[1], 0, XV_ROW_B, NDT_V)):   # Q then dO
                for dt in range_constexpr(ndt):
                    for u in range_constexpr(2):
                        out.append(fx.Vector(llvm_dialect.load(
                            v8b, create_llvm_ptr(
                                rb_ + fx.Int32(hh * 16 * xrow + img + dt * 64 + u * 32),
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
            if const_expr(nw > 1):
                # WAR: every wave's reads of stage 0 (previous masked iteration) retired
                rocdl.sched_barrier(0)
                _lds_barrier()
                rocdl.sched_barrier(0)
            _tdm_qdo(qt, gh, cur_off)
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            rocdl.sched_barrier(0)
            if const_expr(nw > 1):
                _lds_barrier()                # RAW: every wave's half of the tile landed
                rocdl.sched_barrier(0)
        lds_do = _lds0 + cur_off

        # The S/dP B operands, read back from the TDM image: lane (row, half) reads row
        # hh*16+row, bytes half*16 + dt*64 (+32).
        ops = qd if const_expr(carry) else _rdqd(cur_off)
        pst = []                    # P/dS stores deferred into one DS burst
        for hh in range_constexpr(2):
            qp = [(ops[hh * NHB + dt * 2], ops[hh * NHB + dt * 2 + 1])
                  for dt in range_constexpr(NDT_QK)]
            dp = [(ops[hh * NHB + NQE + dt * 2], ops[hh * NHB + NQE + dt * 2 + 1])
                  for dt in range_constexpr(NDT_V)]
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT_QK)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT_V)]
            q_glob = q0 + fx.Int32(hh * 16) + row
            lse_q = pre[hh * 2][0]
            del_q = pre[hh * 2 + 1][0]
            for kh in range_constexpr(NKV):
                s_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                p_acc = _ir(fx.Vector.filled(8, 0.0, fx.Float32))
                for dt in range_constexpr(NDT_MAX):
                    if const_expr(dt < NDT_QK):     # S^T = K Q^T contracts D_QK
                        s_acc = rocdl.wmma_f32_16x16x32_bf16(
                            v8f, _ir(kf[kh][dt]), _ir(qfr[dt]), s_acc,
                            reuseA=False, reuseB=False).result
                    if const_expr(dt < NDT_V):      # dP^T = V dO^T contracts D_V
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
        tr_v = lds_do + lb_tr_v
        tr_k = tr_v if SAME_D else lds_do + lb_tr_k
        if const_expr(carry):
            rb_base = _rd_bases(rb_off)
        rocdl.sched_barrier(0)
        for v_, a_ in pst:
            llvm_dialect.store(fx.as_ir_value(v_), create_llvm_ptr(a_, address_space=3))
        # The P/dS tiles are private to the wave (one per wave at nw > 1): no barrier between
        # the P/dS stores and their tr16 reads; the backend derives the dscnt wait from the
        # LDS dependence.

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
            b_do = [tr(tr_v + fx.Int32(dtile * 32), XV_ROW_B) for dtile in range_constexpr(NDO_V)]
            rocdl.sched_barrier(0)
            a_dsk[0] = _a_tr(0, lds_ds)
            rocdl.sched_barrier(0)
            b_q = [tr(tr_k + fx.Int32(QOFF + dtile * 32), XK_ROW_B)
                   for dtile in range_constexpr(NDO_QK)]
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
            for dtile in range_constexpr(NDO_MAX):
                if const_expr(dtile < NDO_V):
                    b_do.append(tr(tr_v + fx.Int32(dtile * 32), XV_ROW_B))
                if const_expr(dtile < NDO_QK):
                    b_q.append(tr(tr_k + fx.Int32(QOFF + dtile * 32), XK_ROW_B))
        if const_expr(carry and nw == 1):
            # Outstanding TDM here: stage (it+1)%3 (issued in it-1) and (it+2)%3 (issued at
            # the top of this iteration), TDM_OPS_QDO ops each. Waiting down to TDM_OPS_QDO
            # retires the older stage, the one about to be read (TDM ops retire in order).
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(TW_QDO)
            rocdl.sched_barrier(0)
            rb = _rdqd(rb_off, rb_base)
        rocdl.sched_barrier(0)
        new = [None] * NST
        for kh in range_constexpr(NKV):
            if const_expr(carry and nw > 1 and kh == BAR_KH):
                # k_dkdv64: the ring barrier, after the WMMAs of kv sub-tiles < BAR_KH (their
                # run hides the drain of this iteration's tr16 reads). tensor_wait retires
                # this wave's half of stage (it+1)%3; the barrier's release fence drains its
                # DS reads of stage it%3 (WAR for the TDM at the top of it+1); after it every
                # wave's half of (it+1)%3 is in LDS, so the readback may start.
                rocdl.sched_barrier(0)
                tdm_ops.tensor_wait(TW_QDO)
                rocdl.sched_barrier(0)
                _lds_barrier()
                rocdl.sched_barrier(0)
                rb = _rdqd(rb_off, rb_base)
                rocdl.sched_barrier(0)
            a_p = a_pk[kh]
            a_ds = a_dsk[kh]
            # A-operand reuse hint: within a run of WMMAs, A is one 16x32 subtile held
            # across all dtiles, so instructions 2.. of each run may reuse it.
            for dtile in range_constexpr(NDO_V):                       # dV += P^T dO
                new[kh * NDO_V + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_p), _ir(b_do[dtile]), acc[kh * NDO_V + dtile],
                    reuseA=(dtile > 0), reuseB=False).result
            for dtile in range_constexpr(NDO_QK):                      # dK += dS^T Q
                new[DK0 + kh * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_q[dtile]), acc[DK0 + kh * NDO_QK + dtile],
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
        if const_expr(nw > 1):
            # WAR: every wave's reads of stage 0 in the masked loop retired
            rocdl.sched_barrier(0)
            _lds_barrier()
            rocdl.sched_barrier(0)
        _tdm_qdo(_clampqt(qt0), fx.Int32(0), fx.Int32(0))
        # j1 = max(min(1, n-1), 0) is 1 iff n > 1, and 1 == wrap(0, 0).
        q1w, g1w = _wrap(fx.Int32(0), fx.Int32(0))
        two = fx.Int32(1) < n
        q1 = fx.Int32(two.select(q1w, fx.Int32(0)))
        g1 = fx.Int32(two.select(g1w, fx.Int32(0)))
        _tdm_qdo(_clampqt(qt0 + q1), g1, fx.Int32(QDO_B))
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(TW_QDO)
        rocdl.sched_barrier(0)
        if const_expr(nw > 1):
            _lds_barrier()                    # RAW: every wave's half of stage 0 landed
            rocdl.sched_barrier(0)
        return [_ir(v) for v in _rdqd(fx.Int32(0))]

    # Query pair `qt` is fully unmasked iff this workgroup's LARGEST key index is attended
    # by the pair's SMALLEST query: kv0g + nw*BLOCK_KV-1 <= qt*32 + cshift, i.e.
    # qt >= ceil((kv0g + nw*BLOCK_KV-1 - cshift)/32). cshift = Skv - Sq can be negative.
    _u = kv0g + fx.Int32(nw * BLOCK_KV - 1) - cshift
    _qsf = (_u < fx.Int32(0)).select(fx.Int32(0), (_u + fx.Int32(31)) // fx.Int32(32))
    _qsf = (_qsf < nqt2).select(_qsf, nqt2)
    _nm = _qsf - qp_start
    _nm = (_nm < fx.Int32(0)).select(fx.Int32(0), _nm)
    _nm = (_nm < nqp_eff).select(_nm, nqp_eff)
    nmaskp = (causal != fx.Int32(0)).select(_nm, fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NST)]
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
    _epi0 = _lds0
    if const_expr(nw > 1):
        # every wave's TDM landed and every wave's ring reads (incl. the dead final readback)
        # retired before any wave's image overwrites the ring; wave w's image at w*EPI_W
        rocdl.sched_barrier(0)
        _lds_barrier()
        rocdl.sched_barrier(0)
        _epi0 = _lds0 + wv * fx.Int32(EPI_W)

    # Epilogue: stage the dK/dV accumulators through the finished ring as COLUMN-major
    # images M[d][kv] (a lane's 8 kv rows of one d column are contiguous: one ds_write_b128
    # per d tile) and read them back with ds_load_tr16_b128, which returns ROW-major (lane l:
    # kv row l%16, 8 consecutive d columns), so every global store is a 16-byte
    # buffer_store_b128. Per kv sub-tile kh: the dV image (D_V rows) then the dK image
    # (D_QK rows), EPI_CB bytes per d row.
    g_dv8 = _bv(DV_, nv_b, fx.BFloat16, 8)
    g_dk8 = _bv(DK, nk_b, fx.BFloat16, 8)
    for kh in range_constexpr(NKV):
        lds_ev = _epi0 + fx.Int32(kh * (D_V + D_QK) * EPI_CB)
        lds_ek = _epi0 + fx.Int32((kh * (D_V + D_QK) + D_V) * EPI_CB)
        for dtile in range_constexpr(NDO_MAX):
            o = (fx.Int32(dtile * 16) + row) * fx.Int32(EPI_CB) + half * fx.Int32(16)
            if const_expr(dtile < NDO_V):
                ov = fx.Vector(out[kh * NDO_V + dtile])
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(
                        [ov[si].to(fx.BFloat16) for si in range_constexpr(8)],
                        dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ev + o, address_space=3))
            if const_expr(dtile < NDO_QK):
                ok_ = fx.Vector(out[DK0 + kh * NDO_QK + dtile])
                llvm_dialect.store(
                    fx.as_ir_value(fx.Vector.from_elements(
                        [ok_[si].to(fx.BFloat16) for si in range_constexpr(8)],
                        dtype=fx.BFloat16)),
                    create_llvm_ptr(lds_ek + o, address_space=3))
        # lane_r / lane_c are the transposing-load address form of tr(): lane l addresses
        # row (l//16)*8 + l%8, column ((l//8)%2)*8, and receives column l%16 of rows
        # (l//16)*8 + e.
        kvr = kv0 + fx.Int32(kh * 16) + row
        gt_v = base_v + kvr * rs_v
        gt_k = gt_v if SAME_D else base_k + kvr * rs_k
        for sub in range_constexpr(NDO_MAX):
            a = ((fx.Int32(sub * 16) + lane_r) * fx.Int32(EPI_CB)
                 + lane_c * fx.Int32(2))
            if const_expr(sub < NDO_V):
                vv = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ev + a, address_space=3)))
            if const_expr(sub < NDO_QK):
                kk = fx.Vector(rocdl.ds_load_tr16_b128(
                    v8b, create_llvm_ptr(lds_ek + a, address_space=3)))
            if const_expr(sub < NDO_V):
                _stv([vv[e] for e in range_constexpr(8)], g_dv8,
                     gt_v + fx.Int32(sub * 2) + half, fx.BFloat16)
            if const_expr(sub < NDO_QK):
                _stv([kk[e] for e in range_constexpr(8)], g_dk8,
                     gt_k + fx.Int32(sub * 2) + half, fx.BFloat16)


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


# k_dkdv64: DKDV_NW = 2 waves per workgroup (one per SIMD at 1 wave/SIMD), 64 kv rows per
# Q/dO fetch, so the Q/dO TDM bytes and LDS writes per dK/dV FLOP halve. Same body as k_dkdv
# (each wave keeps its 887-VGPR register plan), one shared ring, barriers per _dkdv_impl.
@flyc.kernel(known_block_size=[WAVE * DKDV_NW, 1, 1])
def k_dkdv64(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor,
             LSE: fx.Tensor, DEL: fx.Tensor, DV_: fx.Tensor, DK: fx.Tensor,
             scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
             G: fx.Int32, nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32, B_: fx.Int32):
    _dkdv_impl(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv,
               G, nqt, cshift, causal, B_, DKDV_NW)


@flyc.jit
def launch_dkdv64(Q, K, V, DO, LSE, DEL, DV_, DK, scale: fx.Float32,
                  Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                  nqt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                  nblk: fx.Int32, nhkv: fx.Int32, nb: fx.Int32, stream: fx.Stream):
    # grid = (Hkv kv heads, Skv/64 kv blocks, B); kv block g ascending = longest first (causal)
    k_dkdv64(Q, K, V, DO, LSE, DEL, DV_, DK, scale, Sq, Skv, Hq, Hkv, G,
             nqt, cshift, causal, nb).launch(
        grid=(nhkv, nblk, nb), block=(WAVE * DKDV_NW, 1, 1), stream=stream)


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
# queries per k_dqg wave: at D_QK = 192, 64 needs 1024 VGPR + 148 spilled (the accumulators
# and Q/dO fragments grow 1.5x); 32 halves both (709 VGPR, no spill).
DQ_BQW = 64 if D_QK <= 128 else 32
NQW = DQ_BQW // 16
# k_dqg48: the same kernel at 48 queries per wave (NQW48 = 3 query sub-tiles). Per 32-row kv step
# it issues the same 64 DS loads and 20 KB of TDM as the 32-query kernel but 96 WMMA (S 36, dP 24,
# dQ 36) instead of 64. Sq % 48 != 0 in general, so the 32-query k_dqg covers the head [0, q_split)
# and k_dqg48 the rest; dq_split() is the single source of that split (impl._plan, bounds_proof).
DQ_BQW48 = 48
NQW48 = DQ_BQW48 // 16
# k_dqg96: DQ_NWAVE waves per workgroup, each the k_dqg48 body over its own 48 queries, consume
# ONE K/V TDM ring: 96 queries per K/V fetch. Wave w owns [q0 + 48w, q0 + 48w + 48), q0 =
# q_off + bid*96; the kv range runs to the last wave's diagonal (both waves run the same trip
# count; wave 0's extra steps are masked through do_mask). One barrier per kv step.
DQ_NWAVE = 2
DQ_BQW96 = DQ_NWAVE * DQ_BQW48
DQ_SPLITS = (0, 32, 64)      # q_split candidates: multiples of DQ_BQW; lcm(32, 96) = 96 > 64


def dq_split(sq, bqw=DQ_BQW96):
    """(q_split, n32, nm) for Sq = sq: k_dqg runs n32 = q_split/32 tiles over queries [0, q_split),
    the main dQ kernel (bqw 96: k_dqg96, the launched one; 48: k_dqg48) nm = (sq - q_split)/bqw
    tiles over [q_split, sq); q_split is the smallest of DQ_SPLITS with (sq - q_split) % bqw == 0
    (exists for every sq % 32 == 0: sq % 96 in {0, 32, 64}, sq % 48 in {0, 32, 16})."""
    assert DQ_BQW == 32, "the k_dqg48/k_dqg96 split assumes the 32-query head kernel (D_QK 192)"
    for qs in DQ_SPLITS:
        if qs <= sq and (sq - qs) % bqw == 0:
            return qs, qs // DQ_BQW, (sq - qs) // bqw
    raise ValueError(f"no k_dqg/main split for seqlen_q {sq} at {bqw} queries per tile")


# Workgroup sync of k_dqg96, one per kv step and one in the prologue. Every LDS read this wave
# issued before it (the dQ tr16 of stage i%3, the readback of i-1) has retired (dscnt 0), and
# every TDM op of this wave except the newest stage has retired (the caller's tensor_wait).
# Raw split barrier with nothing between signal and wait; sched_barrier(0) on both sides keeps
# every DS/TDM op on its side (checked on the ISA by tools/flydsl/isa_ring_barrier_check.py).
# DQ_BARRIER_FENCE: workgroup release/acquire fences around signal/wait (gpu.barrier()'s
# semantics), so no IR pass moves an LDS access across the barrier either. Measured at compile
# time: the fences add no s_wait_tensorcnt 0 (the newest ring stage stays in flight) and no wait
# the kernel did not already have (prologue loadcnt 0 is needed by the softmax constants anyway).
DQ_BARRIER_FENCE = True


def _wg_sync():
    # UNSTABLE(gfx1250): s_wait_dscnt, raw s_barrier_signal/wait (id -1), llvm.fence
    rocdl.sched_barrier(0)
    rocdl.s_wait_dscnt(0)
    if const_expr(DQ_BARRIER_FENCE):
        llvm_dialect.fence(llvm_dialect.AtomicOrdering.release, syncscope="workgroup")
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)
    if const_expr(DQ_BARRIER_FENCE):
        llvm_dialect.fence(llvm_dialect.AtomicOrdering.acquire, syncscope="workgroup")
    rocdl.sched_barrier(0)


VOFF = KV_STEP * XK_ROW_B    # ring stage: K [32][D_QK] at +0, V [32][D_V] at +VOFF
KV_B = KV_STEP * (XK_ROW_B + XV_ROW_B)       # one ring stage (21504 B at 192/128)
TDM_OPS_KV = len(_pow2_segments(D_QK)) + len(_pow2_segments(D_V))   # TDM ops per stage
DQT_TW = TDM_OPS_KV * (TDM_DEPTH - 2)        # tensorcnt that retires all but the newest stage
assert TDM_DEPTH * KV_B <= LDS_SEG
# Carried S/dP A operands (read back one iteration early), in issue order: per (kt, dt)
# K u0, K u1 (dt < NDT_QK) then V u0, V u1 (dt < NDT_V) -- the 16 B chunks of row kt*16+row
# at bytes half*16 + dt*64 + u*32.
_RD_ORDER = [(kt, dt, w, u) for kt in range(NKT) for dt in range(NDT_MAX) for w in "kv"
             if dt < (NDT_QK if w == "k" else NDT_V) for u in range(2)]
_RD_IDX = {e: i for i, e in enumerate(_RD_ORDER)}
NP = len(_RD_ORDER)
# Full-loop schedule ("ck"): kt0 S/dP WMMAs; kt1 S/dP in NDT_MAX chunks {the WMMAs of dtile j
# + softmax/dS (kt0, qh j)}; the DS phase (tr16 | tensor_wait | readback); softmax/dS
# (kt1, qh 0); dQ qh-major in chunks {NDO_QK dQ WMMAs of qh j + softmax/dS (kt1, qh j+1)}.
# Chunks end in sched_barrier(0); inside a chunk sched_group_barrier {1 WMMA, 2 VALU,
# 1 TRANS} per WMMA.
DQT_VT_SGB = (2, 1)


def _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ,
                  scale, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, nqw,
                  nwave=1):
    # nqw (compile-time): 16-query sub-tiles per wave (NQW 2 = k_dqg, NQW48 3 = k_dqg48/96).
    # nwave (compile-time): waves per workgroup sharing the K/V ring (1, or DQ_NWAVE = k_dqg96).
    # The workgroup owns queries [q0g, q0g + nwave*16*nqw), q0g = q_off + bid*BQWG, bid in
    # [0, ntile) (runtime); wave wv owns [q0, q0 + BQW), q0 = q0g + wv*BQW. nwave == 1 is the
    # single-wave kernel unchanged (q0 == q0g, lane == thread id, no barrier).
    NQW = nqw
    BQW = 16 * NQW
    BQWG = BQW * nwave
    if const_expr(nwave == 1):
        lane = fx.Int32(fx.thread_idx.x)
    else:
        _tid = fx.Int32(fx.thread_idx.x)
        lane = _tid % fx.Int32(WAVE)
        # wave index in the workgroup (threads [32w, 32w+32) form wave w), made uniform (SGPR)
        wv = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                          (_tid // fx.Int32(WAVE)).ir_value()))
    # XCD-major q-head remap: workgroups go to the 8 XCDs round-robin on the linear id, so
    # x -> (x%8)*(Hq/8) + x/8 puts adjacent q heads (one kv head under GQA) on one XCD.
    # A bijection of [0, Hq) whenever Hq % 8 == 0, else the identity.
    _nx = fx.Int32(8)
    _gx = fx.Int32(fx.block_idx.x)
    ngrp = Hq
    qh = (ngrp % _nx == fx.Int32(0)).select((_gx % _nx) * (ngrp // _nx) + _gx // _nx, _gx)
    # longest-first: the query tile is walked DESCENDING on grid.y (grid.y == ntile)
    bid = ntile - fx.Int32(1) - fx.Int32(fx.block_idx.y)
    bat = fx.Int32(fx.block_idx.z)
    row = lane % fx.Int32(16)
    half = lane // fx.Int32(16)
    q0g = q_off + bid * fx.Int32(BQWG)
    if const_expr(nwave == 1):
        q0 = q0g
    else:
        q0 = q0g + wv * fx.Int32(BQW)
    hkv = qh // G

    # Fake 1 GiB / 256 MiB extents (impl.py asserts every tensor fits): never "fix" these to
    # the k_dkdv formula -- k_dqg has no batch-count argument.
    g_q = _bv(Q, 1 << 30, fx.BFloat16, 8)
    g_do = _bv(DO, 1 << 30, fx.BFloat16, 8)
    g_lse = _bv(LSE, 1 << 28, fx.Float32)
    g_del = _bv(DEL, 1 << 28, fx.Float32)
    g_dq = _bv(DQ, 1 << 30, fx.BFloat16)

    rs_q = Hq * fx.Int32(DV8_QK)                # vec8 tiles between consecutive q rows
    base_q = bat * Sq * rs_q + qh * fx.Int32(DV8_QK)
    rs_o = rs_q if SAME_D else Hq * fx.Int32(DV8_V)
    base_o = base_q if SAME_D else bat * Sq * rs_o + qh * fx.Int32(DV8_V)
    base_l = (bat * Hq + qh) * Sq

    smem = fx.SharedAllocator().allocate(TDM_DEPTH * KV_B)
    _lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v8f = fx.Vector.make_type(8, fx.Float32)

    def gfrag(buf, base, rs, r, dt):
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
        return _ldv(buf, t, fx.BFloat16, 8).shuffle(
            _ldv(buf, t + fx.Int32(2), fx.BFloat16, 8), list(range(16)))

    qf = [[gfrag(g_q, base_q, rs_q, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_QK)]
          for qh_ in range(NQW)]
    dof = [[gfrag(g_do, base_o, rs_o, q0 + fx.Int32(qh_ * 16), dt) for dt in range(NDT_V)]
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
    # lane-only parts of the DS address families; every per-op difference is a compile-time
    # immediate (bounds_proof.py: max immediate < 65536).
    lb_tr = lane_r * fx.Int32(XK_ROW_B) + lane_c * fx.Int32(2)    # dQ B operand (K) tr16
    lb_rd_k = row * fx.Int32(XK_ROW_B) + half * fx.Int32(16)       # S/dP A readback, K rows
    lb_rd_v = lb_rd_k if SAME_D else row * fx.Int32(XV_ROW_B) + half * fx.Int32(16)

    _lds_bf_ty = fx.PointerType.get(elem_ty=fx.BFloat16.ir_type,
                                    address_space=fx.AddressSpace.Shared, alignment=16)
    _kv_rs_k = Hkv * fx.Int32(D_QK)             # elements between consecutive k rows
    _kv_rs_v = _kv_rs_k if SAME_D else Hkv * fx.Int32(D_V)

    def _tdm_kv(kv0p, stage_off):
        """K then V tile [kv0p, kv0p+32) of (bat, hkv) into stage_off / +VOFF: TDM_OPS_KV
        ops. Origin element ((bat*Skv + kv0p)*Hkv + hkv)*D, row stride Hkv*D, outer extent
        Skv - kv0p >= 32 (kv0p <= (nkvt-1)*32, bounds_proof.py Q1). nwave > 1: each wave
        issues its KV_STEP/nwave-row block of every op (bounds_proof.py W1)."""
        row0 = fx.Int64(bat * Skv + kv0p) * fx.Int64(Hkv) + fx.Int64(hkv)
        off_k = row0 * fx.Int64(D_QK)
        off_v = off_k if SAME_D else row0 * fx.Int64(D_V)
        valid = Skv - kv0p
        lb_k = _lds0 + stage_off
        lb_v = _lds0 + stage_off + fx.Int32(VOFF)
        _tdm_rows(K, off_k, D_QK, KV_STEP, valid, _kv_rs_k, lb_k, _lds_bf_ty, nwave)
        _tdm_rows(V, off_v, D_V, KV_STEP, valid, _kv_rs_v, lb_v, _lds_bf_ty, nwave)

    def _rdkv(stage_off):
        """NP ds_load_b128 in _RD_ORDER: K/V row kt*16+row, bytes half*16 + dt*64 + u*32."""
        rb_k = _lds0 + stage_off + lb_rd_k
        rb_v = rb_k if SAME_D else _lds0 + stage_off + lb_rd_v
        out = []
        for kt, dt, w, u in _RD_ORDER:
            if w == "k":
                a = rb_k + fx.Int32(kt * 16 * XK_ROW_B + dt * 64 + u * 32)
            else:
                a = rb_v + fx.Int32(VOFF + kt * 16 * XV_ROW_B + dt * 64 + u * 32)
            out.append(fx.Vector(llvm_dialect.load(v8b, create_llvm_ptr(a, address_space=3))))
        return out

    def _bks(cur):
        """dQ GEMM B operands: K^T of this stage, NDO_QK v16 (rows kv 0..15 | 16..31)."""
        tb = _lds0 + cur + lb_tr
        out = []
        for dtile in range_constexpr(NDO_QK):
            base = tb + fx.Int32(dtile * 32)
            out.append(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base, address_space=3))
            ).shuffle(fx.Vector(rocdl.ds_load_tr16_b128(
                v8b, create_llvm_ptr(base + fx.Int32(16 * XK_ROW_B), address_space=3))),
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

    def _sdp(pre, kt, dt, s_acc, p_acc):
        """k-step dt of S^T = K Q^T (dt < NDT_QK) and dP^T = V dO^T (dt < NDT_V), all qh."""
        if const_expr(dt < NDT_QK):
            kfr = pre[_RD_IDX[(kt, dt, "k", 0)]].shuffle(
                pre[_RD_IDX[(kt, dt, "k", 1)]], list(range(16)))
        if const_expr(dt < NDT_V):
            vfr = pre[_RD_IDX[(kt, dt, "v", 0)]].shuffle(
                pre[_RD_IDX[(kt, dt, "v", 1)]], list(range(16)))
        for qh_ in range_constexpr(NQW):
            if const_expr(dt < NDT_QK):
                s_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(kfr), _ir(qf[qh_][dt]), s_acc[qh_],
                    reuseA=False, reuseB=False).result
            if const_expr(dt < NDT_V):
                p_acc[qh_] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(vfr), _ir(dof[qh_][dt]), p_acc[qh_],
                    reuseA=False, reuseB=False).result

    def _body_ck(acc, pre, kv0, pf_kv0, cur, nxo, ncur):
        # Full loop: the WMMA<->softmax interleave described at DQT_VT_SGB.
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
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
                if const_expr(kt == 1):
                    # chunk dt: this dtile's kt1 WMMAs + softmax/dS of (kt0, qh = dt)
                    if const_expr(dt < NQW):
                        s0, p0 = sp_acc[0]
                        ds_halves[dt].append(_smx(fx.Vector(s0[dt]), fx.Vector(p0[dt]),
                                                  dt, 0, kv0, False))
                    _sgb(NQW * (int(dt < NDT_QK) + int(dt < NDT_V)))
                    rocdl.sched_barrier(0)
            if const_expr(kt == 0):
                rocdl.sched_barrier(0)            # kt0 S/dP WMMAs: their own region
            sp_acc.append((s_acc, p_acc))
        for j in range_constexpr(NDT_MAX, NQW):  # kt0 rows without a kt1 chunk (NQW > NDT)
            s0, p0 = sp_acc[0]
            ds_halves[j].append(_smx(fx.Vector(s0[j]), fx.Vector(p0[j]), j, 0, kv0, False))
        s1, p1 = sp_acc[1]
        # the DS phase
        rocdl.sched_barrier(0)
        b_ks = _bks(cur)
        rocdl.sched_barrier(0)
        if const_expr(nwave == 1):
            tdm_ops.tensor_wait(DQT_TW)
            rocdl.sched_barrier(0)
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
            ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
            rocdl.sched_barrier(0)
        else:
            # k_dqg96: softmax/dS (kt1, qh 0) covers the tr16 latency, then this wave's half
            # of stage ncur retires (tensor_wait), the tr16 drains and the workgroup syncs
            # (_wg_sync): both halves of ncur landed, every wave's reads of stage cur retired
            # (the TDM at the top of i+1 targets it). Only then the readback of ncur.
            ds_halves[0].append(_smx(fx.Vector(s1[0]), fx.Vector(p1[0]), 0, 1, kv0, False))
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        new = [None] * (NQW * NDO_QK)
        for qh_ in range_constexpr(NQW):
            a_ds = fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                           dtype=fx.BFloat16)
            for dtile in range_constexpr(NDO_QK):
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds), _ir(b_ks[dtile]), acc[qh_ * NDO_QK + dtile],
                    reuseA=False, reuseB=False).result
            if const_expr(qh_ + 1 < NQW):
                ds_halves[qh_ + 1].append(_smx(fx.Vector(s1[qh_ + 1]),
                                               fx.Vector(p1[qh_ + 1]),
                                               qh_ + 1, 1, kv0, False))
                _sgb(NDO_QK)
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
            for dt in range_constexpr(NDT_MAX):
                _sdp(pre, kt, dt, s_acc, p_acc)
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
        if const_expr(nwave == 1):
            tdm_ops.tensor_wait(DQT_TW)
            rocdl.sched_barrier(0)
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
            ds_halves = _softmax()
        else:
            # k_dqg96: the softmax VALU covers the tr16 latency before the drain + sync (as in
            # _body_ck); the readback of ncur follows the sync.
            ds_halves = _softmax()
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(DQT_TW)
            _wg_sync()
            rb = _rdkv(ncur)
            rocdl.sched_barrier(0)
        a_ds = [fx.Vector.from_elements(ds_halves[qh_][0] + ds_halves[qh_][1],
                                        dtype=fx.BFloat16)
                for qh_ in range_constexpr(NQW)]
        new = [None] * (NQW * NDO_QK)
        for qh_ in range_constexpr(NQW):
            for dtile in range_constexpr(NDO_QK):
                new[qh_ * NDO_QK + dtile] = rocdl.wmma_f32_16x16x32_bf16(
                    v8f, _ir(a_ds[qh_]), _ir(b_ks[dtile]), acc[qh_ * NDO_QK + dtile],
                    reuseA=False, reuseB=False).result
        return new + [_ir(v) for v in rb]

    NACC = NQW * NDO_QK

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

    # kv range of the WORKGROUP tile [q0g, q0g + BQWG) (== the wave's when nwave == 1): every
    # wave runs the same nfull full + (nkvt_eff - nfull) masked steps, so the barrier counts
    # match; rows of an earlier wave past their own diagonal are masked by do_mask (ds = 0).
    _lim = (q0g + fx.Int32(BQWG) + cshift + fx.Int32(KV_STEP - 1)) // fx.Int32(KV_STEP)
    _lim = (_lim < fx.Int32(1)).select(fx.Int32(1), _lim)
    _lim = (_lim < nkvt).select(_lim, nkvt)
    nkvt_eff = (causal != fx.Int32(0)).select(_lim, nkvt)
    _t = q0g + cshift + fx.Int32(1)
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
    if const_expr(nwave > 1):
        _wg_sync()                                       # both halves of stage 0 landed
    rocdl.sched_barrier(0)
    rb0 = _rdkv(fx.Int32(0))

    init = [_ir(fx.Vector.filled(8, 0.0, fx.Float32)) for _ in range(NACC)]
    out = kvloop_full(init + [_ir(v) for v in rb0] + [fx.as_ir_value(fx.Int32(0))],
                      nfull, fx.Int32(0), nlast)
    out = kvloop_mask(list(out), nkvt_eff - nfull, nfull, nlast)
    # retire the last two (clamped, never read) prefetches before the wave ends: the
    # workgroup's LDS must not be written after it is released.
    tdm_ops.tensor_wait(0)
    base_dq = bat * Sq * Hq * fx.Int32(D_QK) + qh * fx.Int32(D_QK)
    for qh_ in range_constexpr(NQW):
        for dtile in range_constexpr(NDO_QK):
            ov = fx.Vector(out[qh_ * NDO_QK + dtile])
            for si in range_constexpr(8):
                q_i = q0 + fx.Int32(qh_ * 16) + half * fx.Int32(8) + fx.Int32(si)
                _st1(ov[si].to(fx.BFloat16), g_dq,
                     base_dq + q_i * Hq * fx.Int32(D_QK) + fx.Int32(dtile * 16) + row,
                     fx.BFloat16)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dqg(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor, O: fx.Tensor,
          LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
          scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
          G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
          q_off: fx.Int32, ntile: fx.Int32):
    _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
                  nkvt, cshift, causal, q_off, ntile, NQW)


@flyc.kernel(known_block_size=[32, 1, 1])
def k_dqg48(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor, O: fx.Tensor,
            LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
            scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
            G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
            q_off: fx.Int32, ntile: fx.Int32):
    _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
                  nkvt, cshift, causal, q_off, ntile, NQW48)


@flyc.jit
def launch_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale: fx.Float32,
               Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
               nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
               q_off: fx.Int32, ntile: fx.Int32, ngrp: fx.Int32, nb: fx.Int32,
               stream: fx.Stream):
    # grid = (Hq q heads, ntile DQ_BQW-query tiles from q_off, B)
    k_dqg(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
          nkvt, cshift, causal, q_off, ntile).launch(
        grid=(ngrp, ntile, nb), block=(32, 1, 1), stream=stream)


@flyc.jit
def launch_dqg48(Q, K, V, DO, O, LSE, DEL, DQ, scale: fx.Float32,
                 Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                 nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                 q_off: fx.Int32, ntile: fx.Int32, ngrp: fx.Int32, nb: fx.Int32,
                 stream: fx.Stream):
    # grid = (Hq q heads, ntile DQ_BQW48-query tiles from q_off, B)
    k_dqg48(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
            nkvt, cshift, causal, q_off, ntile).launch(
        grid=(ngrp, ntile, nb), block=(32, 1, 1), stream=stream)


@flyc.kernel(known_block_size=[DQ_NWAVE * WAVE, 1, 1])
def k_dqg96(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, DO: fx.Tensor, O: fx.Tensor,
            LSE: fx.Tensor, DEL: fx.Tensor, DQ: fx.Tensor,
            scale: fx.Float32, Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32,
            G: fx.Int32, nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
            q_off: fx.Int32, ntile: fx.Int32):
    _dqg_tdm_impl(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
                  nkvt, cshift, causal, q_off, ntile, NQW48, DQ_NWAVE)


@flyc.jit
def launch_dqg96(Q, K, V, DO, O, LSE, DEL, DQ, scale: fx.Float32,
                 Sq: fx.Int32, Skv: fx.Int32, Hq: fx.Int32, Hkv: fx.Int32, G: fx.Int32,
                 nkvt: fx.Int32, cshift: fx.Int32, causal: fx.Int32,
                 q_off: fx.Int32, ntile: fx.Int32, ngrp: fx.Int32, nb: fx.Int32,
                 stream: fx.Stream):
    # grid = (Hq q heads, ntile DQ_BQW96-query tiles from q_off, B), DQ_NWAVE waves per workgroup
    k_dqg96(Q, K, V, DO, O, LSE, DEL, DQ, scale, Sq, Skv, Hq, Hkv, G,
            nkvt, cshift, causal, q_off, ntile).launch(
        grid=(ngrp, ntile, nb), block=(DQ_NWAVE * WAVE, 1, 1), stream=stream)
