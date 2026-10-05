#!/usr/bin/env python3
"""CPU bounds proof for kernels.py (stdlib only: no torch, no flydsl, no GPU).

    python3 bounds_proof.py [--modes two_wave,one_wave] [shape ...]      default: every shape, both modes

The layout constants and helpers (_pow2_segments, _RD_ORDER, ...) are exec'd from
kernels.py itself, and the launch geometry (impl._geometry, SMALL_GRID_WAVES) from impl.py
itself, so the proof cannot drift from the kernel's constants or the host's dispatch; the
per-thread index formulas are re-derived here line by line from kernels.py and replayed for
every lane, every loop iteration and every kv/query tile of the extreme (head, batch) workgroups.

Modes (small-grid fallback, bwd_r4_c's rule on the head-group launches): every shape is replayed
with BOTH launch sets impl._geometry can return -- two_wave (k_dkdv64 + k_dqg96/k_dqg head, the
launches of bwd_r3_a with bwd_r4_a's head groups) and one_wave (k_dkdv nw = 1, grid (Hkv, Skv/32,
B), + one head-grouped k_dqg over [0, Sq): bwd_c1's plan) -- obtained by running
_geometry with both chains' SMALL_GRID_WAVES forced to 0 / 2**62, so every kernel is in bounds
whatever the thresholds; G1 then checks which set the real per-chain thresholds launch.
  G1  dispatch: per chain small iff its one-wave workgroup count (k_dkdv B*Hkv*Skv/32, k_dqg
      B*Hq*Sq/32) < SMALL_GRID_WAVES[chain]; the launched geometry equals the replayed mode of that
      decision; the two_wave set equals bwd_r3_a's _plan (k_dkdv64 nblk Skv/64, dq_split's
      k_dqg96 + head) and the one_wave set bwd_c1's (k_dkdv nblk Skv/32, k_dqg ntile Sq/32,
      q_off 0); the dkdv grid tiles [0, Skv) exactly; the decision is fold-invariant
      (_geometry(B, S, S, H, H) == _geometry(1, S, S, B*H, B*H)); expected selections: proxy,
      prod (= b2h128), prodfold [1, 4096, 256], b4h64, sg32, thr_dq_at -> two_wave; toy, fast, rect,
      ..., thr_below -> one_wave; sg16, thr_at, thr_dq_below -> two-wave k_dkdv + one-wave dQ;
      gqa_mixed, gqa_hg -> one-wave k_dkdv + two-wave dQ.

Checks (each prints its count; any violation raises). k_dkdv64: DKDV_NW = 2 waves per
workgroup, grid (Hkv, Skv/64, B), wave w owns kv rows kv0g + 32w + [0, 32); K1..K5 are also
replayed for the one-wave k_dkdv (nw = 1: one wave per workgroup owns kv rows 32*bid + [0, 32),
whole-tile TDM, the single-wave ring of bwd_c1/r1_b with no barrier: K2 in-order tensorcnt).
  K1  k_dkdv TDM tiles: q pair in [0, Sq/32), outer extent Sq - q0 >= 32 and the per-wave
      clamp Sq - q0 - 16w >= 16 (TDM ignores bounds), every wave's 16-row slice of every
      segment inside q / do; LDS data writes inside their stage image, the waves' slices and
      the segments tile each image row's [0, 2D) and all 32 rows exactly, ring in segment 0.
  K2  k_dkdv ring schedule with TDM_OPS_QDO ops per stage per wave (each wave its own
      tensorcnt) and waits TW_QDO / 0 (in-order retirement), replayed for both waves. The full
      loop prefetches tile min(ii+3, n-1) into stage ii%3 right after BARRIER(ii) (behind the
      readback); the prologue fills stages 0, 1 before its RAW barrier and stage 2 after it.
  K3  k_dkdv LDS reads (readback, tr16, P/dS) inside written bytes; DS immediates < 64 KiB;
      per-wave P/dS tiles and epilogue images disjoint, inside the allocation / dead ring;
      K/V fragment and LSE/delta loads in bounds (each wave its own kv rows).
  K4  causal split at WORKGROUP level (block id only): for EACH wave's 32 kv rows, query
      pairs < qp_start fully masked, pairs >= qp_start + nmaskp fully unmasked; division-free
      (qi, gh) counters == (ii // G, ii % G) for every ii.
  K5  dk/dv: every output element written exactly once (all workgroups, both waves, lanes).
  K6  k_dkdv64 barrier protocol: both waves run one program (no wave-dependent branch or trip
      count: source scan of _dkdv_impl), so per wave #barriers == 2*G*nmaskp + 2 + n_full + 1;
      phase model (ops between two barriers run in any cross-wave order, the barrier's release
      fence drains DS): a read hits a stage whose every wave-slice was retired by its issuing
      wave before an earlier barrier and not re-targeted since; no TDM targets a stage read in
      the same phase; at the epilogue barrier no TDM is pending and the ring is not read again.
  Q1  k_dqg TDM tiles (kv block <= nkvt-1, extent >= 32, K/V elements in bounds), LDS
      writes; Q2 ring schedule with TDM_OPS_KV ops per stage, waits DQT_TW / 0;
      Q3 LDS reads (readback, tr16) inside written bytes, DS immediates; Q/dO fragment
      and LSE/delta loads in bounds, inside k_dqg's fake 1 GiB / 256 MiB extents;
      Q4 causal split (nfull iterations fully unmasked, nkvt_eff covers every attended
      kv); Q5 dq written exactly once; the XCD q-head remap is a bijection.
  Q6  k_dqg / k_dqg96 split (impl._geometry's dq launch list: kernels.dq_split in two_wave mode,
      one k_dqg over [0, Sq) in one_wave mode): k_dqg96 workgroup
      tiles (96 rows, q0g = q_split + bid*96; wave w owns [q0g + 48w, q0g + 48w + 48)) and k_dqg
      tiles (32 rows, q0 = q_off + bid*32) over bid = ntile-1-grid.y partition [0, Sq) exactly
      (per-wave ranges); per workgroup and per ROW (brute force) the kv range [0, 32*nkvt_eff) (from the
      workgroup tile, the same for both waves) holds every attended key, iterations < nfull are
      fully unmasked for every row of the workgroup, every masked-loop iteration has a masked
      entry; at cshift 0 a 96-row workgroup has 3 masked steps (32-row: 1); per wave the masked
      steps split into partial / fully masked (wave 0: 1-2 fully masked steps, whose ds is 0:
      exp2(NEG) = 0; the last wave: none), recorded in the log.
  W1  k_dqg96 TDM per wave (num_warps 2: wave w issues rows [16w, 16w+16) of every op): global
      elements in bounds, per-wave extent Skv - kv0p - 16w >= 16, and the two waves' LDS writes
      tile each stage image exactly like the single-wave TDM (SPAN_K / SPAN_V).
  W2  k_dqg96 2-wave ring protocol with one barrier per kv step: every LDS read sees both halves
      of the right tile retired (RAW across waves through a barrier), no TDM write overlaps an
      unretired read of either wave (WAR), no two TDM writes of a stage half in flight (WAW),
      equal barrier counts (no hang), tensorcnt 0 at exit; happens-before = program order in a
      wave, barrier epochs across waves; N = 1..160 plus every shape's trip counts. Mutations
      (drain dropped, readback above the barrier, wait one stage too deep, TDM above the WAR
      barrier) must each be caught (N1..N4).
  R1  head-group launch (bwd_r4_a; kernels.HEAD_GROUP / head_group, impl._plan): for every launched
      kernel (k_dkdv64 over Hkv, k_dqg96 and the head k_dqg over Hq) and every target in
      R_TARGETS, hg = head_group(nh, target) divides nh, the grid (hg, tiles, B*nh/hg) has the
      workgroup count of (nh, tiles, B), and the kernel's decode of (block_idx.x, block_idx.z)
      (k_dkdv64: z*hg + x; k_dqg*: z*hg + the in-group XCD remap of x, then bat = hb // nh,
      h = hb - bat*nh) is a bijection onto [0, B) x [0, nh) with grid.y passed through, so every
      workgroup runs exactly one (bat, head, tile) of r3_a and the K/Q replays above (which take
      (bat, head, tile) as inputs) cover every launched workgroup; i32 intermediates < 2^31;
      hg == nh reproduces r3_a's (bat, head) of every workgroup exactly; hg % 8 == 0 keeps all
      workgroups of one head on one XCD (linear id % 8 constant), counted. In the one_wave set the
      single k_dqg over [0, Sq) is head-grouped the same way, and the one-wave k_dkdv has no head
      group (grid (Hkv, Skv/32, B), (bat, hkv) = (block_idx.z, block_idx.x), source-tied).
  R2  traversal equivalence: on the fold launch [1, s, 256, d] (prodfold) at target 128 (bwd_r4_a's
      default) the sequence of (folded head b*128 + h, tile) over the linear workgroup id equals the
      sequence r3_a's b2 h128 launch walks, for k_dkdv64, k_dqg96 and k_dqg; at the default target
      HEAD_GROUP (bwd_r5_hg64: 64) it equals the sequence the contiguous b2 h128 launch walks at that
      same target; and at target 0 the grids and decodes are r3_a's for every shape.
"""
import pathlib
import re
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE / "kernels.py").read_text()
IMPL_SRC = (HERE / "impl.py").read_text()


def _block(start, end, src=None):
    """Source from the line that starts with `start` to the next line starting with `end`."""
    src = SRC if src is None else src
    i = src.index("\n" + start) + 1
    j = src.index("\n" + end, i) + 1
    return src[i:j]


C = {}
exec(_block("D_QK = ", "def _bv("), C)                 # noqa: S102  head dims, LDS layout
exec(_block("KV_STEP = 32", "def _dqg_tdm_impl("), C)  # noqa: S102  k_dqg constants
g = C.get
D_QK, D_V, BLOCK_KV, KV_STEP, TDM_DEPTH = g("D_QK"), g("D_V"), g("BLOCK_KV"), g("KV_STEP"), g("TDM_DEPTH")
XK, XV, S_ROW_B, LDS_SEG = g("XK_ROW_B"), g("XV_ROW_B"), g("S_ROW_B"), g("LDS_SEG")
QOFF, QDO_B, TDM_OPS_QDO, TW_QDO = g("QOFF"), g("QDO_B"), g("TDM_OPS_QDO"), g("TW_QDO")
NDT_QK, NDT_V, NDO_QK, NDO_V, NKV, EPI_CB = (g("NDT_QK"), g("NDT_V"), g("NDO_QK"), g("NDO_V"),
                                             g("NKV"), g("EPI_CB"))
DQ_BQW, VOFF, KV_B, TDM_OPS_KV, DQT_TW = g("DQ_BQW"), g("VOFF"), g("KV_B"), g("TDM_OPS_KV"), g("DQT_TW")
RD_ORDER, NKT, NQW = g("_RD_ORDER"), g("NKT"), g("NQW")
DQ_BQW48, NQW48, dq_split = g("DQ_BQW48"), g("NQW48"), g("dq_split")
DQ_NWAVE, DQ_BQW96 = g("DQ_NWAVE"), g("DQ_BQW96")
segs = g("_pow2_segments")
DKDV_NW, PDS_B, EPI_W, BAR_KH = g("DKDV_NW"), g("PDS_B"), g("EPI_W"), g("BAR_KH")
HEAD_GROUP, head_group = g("HEAD_GROUP"), g("head_group")
R_TARGETS = (0, 8, 16, 24, 32, 64, 96, 128, 256)
B2H128_TARGET = 128             # bwd_r4_a's default: prodfold at 128 IS r3_a's b2h128 grid (R2)
assert HEAD_GROUP in R_TARGETS and B2H128_TARGET in R_TARGETS
ALLOC_KV = LDS_SEG + DKDV_NW * PDS_B
ROWS_W = 32 // DKDV_NW          # rows of every 32-row Q/dO tile one wave's TDM moves
LANES = range(32)

# impl._geometry (the host dispatch, bwd_r4_c) exec'd from impl.py with the kernel constants it
# reads through `_k`; the block is pure Python by contract (no torch).
_KNS = types.SimpleNamespace(BLOCK_KV=BLOCK_KV, DQ_BQW=DQ_BQW, DKDV_NW=DKDV_NW, NQW=NQW,
                             NQW48=NQW48, DQ_NWAVE=DQ_NWAVE, dq_split=dq_split)
IG = {"_k": _KNS, "os": types.SimpleNamespace(environ={})}  # the default threshold, not the caller's env
exec(_block("N_CU = ", "# Validation only", IMPL_SRC), IG)  # noqa: S102
SMALL_GRID_WAVES = IG["SMALL_GRID_WAVES"]
assert SMALL_GRID_WAVES == {"dkdv": 1024, "dq": 2048}, SMALL_GRID_WAVES   # impl.py's measured crossovers
MODES = {"two_wave": 0, "one_wave": 1 << 62}      # both chains' SMALL_GRID_WAVES forced: never / always small


def geometry(B, Sq, Skv, Hq, Hkv, thr=None):
    """impl._geometry at the real threshold, or with SMALL_GRID_WAVES forced to `thr`."""
    if thr is None:
        return IG["_geometry"](B, Sq, Skv, Hq, Hkv)
    keep = IG["SMALL_GRID_WAVES"]
    IG["SMALL_GRID_WAVES"] = {"dkdv": thr, "dq": thr}
    try:
        return IG["_geometry"](B, Sq, Skv, Hq, Hkv)
    finally:
        IG["SMALL_GRID_WAVES"] = keep


def lane_rc(lane):
    return lane % 16, lane // 16                                     # row, half


def lane_tr(lane):
    return (lane // 16) * 8 + lane % 8, ((lane // 8) % 2) * 8         # lane_r, lane_c


SHAPES = {  # name: (B, Sq, Skv, Hq, Hkv, causal)
    "toy": (1, 256, 256, 2, 2, 1), "fast": (1, 1024, 1024, 8, 8, 1),
    "proxy": (1, 4096, 4096, 64, 64, 1), "prod": (2, 4096, 4096, 128, 128, 1),
    "rect": (1, 256, 384, 2, 2, 1), "edge64": (2, 64, 64, 2, 2, 1),
    "gqa": (2, 512, 512, 8, 2, 1), "noncausal": (1, 256, 256, 2, 2, 0),
    "rect_short_kv": (1, 384, 256, 2, 2, 1),
    "s128": (1, 128, 128, 2, 2, 1), "s192": (2, 192, 192, 2, 2, 1),
    # bwd_r4_a: the fold launch of prod (Megatron b2 h128 SBHD views as [1, s, 256, d]), the fold
    # of gqa, and a GQA shape whose q launch is head-grouped (hg 128 of 256, B 2) while the kv
    # launch keeps r3_a's grid
    "prodfold": (1, 4096, 4096, 256, 256, 1), "gqafold": (1, 512, 512, 16, 4, 1),
    "gqa_hg": (2, 128, 128, 256, 64, 1),
    # small-grid fallback (bwd_r4_c's rule on the head-group launches): b4h64 (prod's bytes), the
    # threshold edges (992 / 1024 one-wave workgroups), a GQA shape whose chains decide differently
    # (k_dkdv 512 small, k_dqg 4096 not), the second small-grid A/B shape b1 s2048 h16 (1024: two-wave)
    # and the card-coverage shapes of the tests (sq % 256 != 0, sq < skv rectangles, b2 folds)
    "b4h64": (4, 4096, 4096, 64, 64, 1),
    "thr_below": (1, 1024, 1024, 31, 31, 1), "thr_at": (1, 1024, 1024, 32, 32, 1),
    "thr_dq_below": (1, 1024, 1024, 62, 62, 1), "thr_dq_at": (1, 1024, 1024, 64, 64, 1),
    "sg32": (1, 2048, 2048, 32, 32, 1),
    "gqa_mixed": (1, 2048, 2048, 64, 8, 1), "sg16": (1, 2048, 2048, 16, 16, 1),
    "s64": (1, 64, 64, 2, 2, 1), "b2s64fold": (1, 64, 64, 4, 4, 1),
    "r64x128": (1, 64, 128, 2, 2, 1), "r128x384": (1, 128, 384, 2, 2, 1),
    "r192x448fold": (1, 192, 448, 4, 4, 1),
}
# G1: the launch set impl._geometry must select per shape at the real threshold:
# (dkdv one-wave?, dq one-wave?)
EXPECT_SMALL = {"proxy": (False, False), "prod": (False, False), "prodfold": (False, False),
                "b4h64": (False, False), "sg32": (False, False), "thr_dq_at": (False, False),
                "sg16": (False, True), "thr_at": (False, True), "thr_dq_below": (False, True),
                "gqa_mixed": (True, False), "gqa_hg": (True, False),
                "toy": (True, True), "fast": (True, True), "rect": (True, True), "edge64": (True, True),
                "gqa": (True, True), "gqafold": (True, True), "noncausal": (True, True),
                "rect_short_kv": (True, True), "s64": (True, True), "s128": (True, True),
                "s192": (True, True), "b2s64fold": (True, True), "r64x128": (True, True),
                "r128x384": (True, True), "r192x448fold": (True, True), "thr_below": (True, True)}
COUNT = {}


def ok(cond, tag, msg=""):
    if not cond:
        raise AssertionError(f"{tag} violated: {msg}")
    COUNT[tag] = COUNT.get(tag, 0) + 1


# --------------------------------------------------------------- static layout facts
def tdm_image_writes(d, xrow):
    """Bytes the TDM ops of one [32][d] image write (data only: the pad is skipped, not
    written), relative to the image base; and the per-row coverage check."""
    spans = []
    for c0, w in segs(d):
        ok(w & (w - 1) == 0 and 2 <= w // 2 <= 256, "K1", f"pad interval {w}")
        pad_dw = (d + 8 - w) // 2
        ok(1 <= pad_dw <= 128 and (d + 8 - w) % 2 == 0, "K1", f"pad amount {pad_dw}")
        ok(w + (d + 8 - w) == xrow // 2, "K1", "segment stride == image row stride")
        for r in range(32):
            spans.append((r * xrow + 2 * c0, r * xrow + 2 * c0 + 2 * w))
    # segments tile each row's data [0, 2d) exactly, no overlap
    for r in range(32):
        row = sorted((a - r * xrow, b - r * xrow) for a, b in spans
                     if r * xrow <= a < (r + 1) * xrow)
        ok(row[0][0] == 0 and row[-1][1] == 2 * d
           and all(row[i][1] == row[i + 1][0] for i in range(len(row) - 1)), "K1", f"row tiling {row}")
    return spans


SPAN_K = tdm_image_writes(D_QK, XK)
SPAN_V = tdm_image_writes(D_V, XV)
# cooperative TDM (num_warps = DKDV_NW): wave w's ops write image rows [w*ROWS_W, (w+1)*ROWS_W)
# (FlyDSL CopyAtom: warps[0] = DKDV_NW, bpw[0] = 32/DKDV_NW, lds offset w*bpw[0]*row stride);
# the waves' slices partition the 32 rows.
ok(32 % DKDV_NW == 0 and DKDV_NW & (DKDV_NW - 1) == 0, "K1", "TDM warp split")
for d_, xr in ((D_QK, XK), (D_V, XV)):
    rows_w = [set(range(w * ROWS_W, (w + 1) * ROWS_W)) for w in range(DKDV_NW)]
    ok(set().union(*rows_w) == set(range(32)) and sum(map(len, rows_w)) == 32, "K1", "wave slices")
    for w in range(DKDV_NW):
        lds_off = w * ROWS_W * xr          # = warpOff * ldsStride[0] * 2 B
        ok(lds_off == w * ROWS_W * ((d_ + 8) * 2), "K1", "slice LDS offset")
ok(QOFF == 32 * XV and QDO_B == QOFF + 32 * XK, "K1", "k_dkdv stage = dO image + Q image")
ok(TDM_DEPTH * QDO_B <= LDS_SEG, "K1", "Q/dO ring inside segment 0")
ok(VOFF == 32 * XK and KV_B == VOFF + 32 * XV, "Q1", "k_dqg stage = K image + V image")
ok(TDM_DEPTH * KV_B <= LDS_SEG, "Q1", "K/V ring inside segment 0")
ok(TDM_OPS_QDO == len(segs(D_QK)) + len(segs(D_V)) == TDM_OPS_KV, "K2", "ops per stage")
# the wait immediates themselves are validated by the ring replays (K2, Q2)

# bank groups of the tr16 / readback row starts (perf, not safety): 16 distinct groups
for xrow in (XK, XV):
    ok(len({(r * xrow // 4) % 64 // 4 for r in range(16)}) == 16, "K3", f"bank groups {xrow}")


def written(img_spans, img_base, lo, hi):
    """[lo, hi) lies inside one data span of the image at img_base."""
    return any(img_base + a <= lo and hi <= img_base + b for a, b in img_spans)


# k_dkdv LDS reads relative to the stage base, and their DS immediates
DS_IMM = []
RD_KV = []      # (lo, hi) of every readback/tr16 byte range, stage-relative, per lane
for lane in LANES:
    row, half = lane_rc(lane)
    lr, lc = lane_tr(lane)
    for hh in range(2):
        for dt in range(NDT_QK):
            for u in range(2):
                imm = hh * 16 * XK + QOFF + dt * 64 + u * 32
                a = row * XK + half * 16 + imm
                DS_IMM.append(imm)
                ok(written(SPAN_K, QOFF, a, a + 16), "K3", f"Q readback {a}")
                RD_KV.append((a, a + 16))
        for dt in range(NDT_V):
            for u in range(2):
                imm = hh * 16 * XV + dt * 64 + u * 32
                a = row * XV + half * 16 + imm
                DS_IMM.append(imm)
                ok(written(SPAN_V, 0, a, a + 16), "K3", f"dO readback {a}")
                RD_KV.append((a, a + 16))
    for dtile in range(NDO_V):
        for r2 in (0, 16):
            imm = dtile * 32 + r2 * XV
            a = lr * XV + lc * 2 + imm
            DS_IMM.append(imm)
            ok(written(SPAN_V, 0, a, a + 16), "K3", f"b_do tr16 {a}")
    for dtile in range(NDO_QK):
        for r2 in (0, 16):
            imm = QOFF + dtile * 32 + r2 * XK
            a = lr * XK + lc * 2 + imm
            DS_IMM.append(imm)
            ok(written(SPAN_K, QOFF, a, a + 16), "K3", f"b_q tr16 {a}")
    # P/dS tiles [32 q][BLOCK_KV kv] bf16 at LDS_SEG + w*PDS_B (+32*S_ROW_B for dS), per wave
    for hh in range(2):
        for kh in range(NKV):
            off = (hh * 16 + row) * S_ROW_B + kh * 32 + half * 16
            ok(off + 16 <= 32 * S_ROW_B and (off % S_ROW_B) + 16 <= BLOCK_KV * 2, "K3", "P/dS store")
    for kh in range(NKV):
        for r2 in (0, 16):
            a = lr * S_ROW_B + lc * 2 + kh * 32 + r2 * S_ROW_B
            ok(a + 16 <= 32 * S_ROW_B and (a % S_ROW_B) + 16 <= BLOCK_KV * 2, "K3", "P/dS tr16")
    # epilogue images: per kh dV [D_V rows] then dK [D_QK rows], EPI_CB bytes per d row
    for kh in range(NKV):
        ev = kh * (D_V + D_QK) * EPI_CB
        ek = ev + D_V * EPI_CB
        for dtile in range(max(NDO_V, NDO_QK)):
            o = (dtile * 16 + row) * EPI_CB + half * 16
            if dtile < NDO_V:
                ok(ev + o + 16 <= ek, "K3", "dV image store")
            if dtile < NDO_QK:
                ok(ek + o + 16 <= ek + D_QK * EPI_CB, "K3", "dK image store")
            ok(o % EPI_CB + 16 <= 32, "K3", "image store stays in the 32 B kv row")
        for sub in range(max(NDO_V, NDO_QK)):
            a = (sub * 16 + lr) * EPI_CB + lc * 2
            if sub < NDO_V:
                ok(ev + a + 16 + 0 <= ek and (a % EPI_CB) + 16 <= 32, "K3", "dV image tr16")
            if sub < NDO_QK:
                ok(ek + a + 16 <= ek + D_QK * EPI_CB, "K3", "dK image tr16")
ok(NKV * (D_V + D_QK) * EPI_CB == EPI_W and DKDV_NW * EPI_W <= TDM_DEPTH * QDO_B <= LDS_SEG, "K3",
   "per-wave epilogue images [w*EPI_W, (w+1)*EPI_W) disjoint, inside the dead ring")
ok(PDS_B == 2 * 32 * S_ROW_B and LDS_SEG + DKDV_NW * PDS_B == ALLOC_KV and ALLOC_KV <= 160 * 1024, "K3",
   "per-wave P/dS tiles [LDS_SEG + w*PDS_B, +PDS_B) disjoint, inside the allocation; 2 workgroups/CU fit")
ok(max(DS_IMM) < 65536 and 2 * QDO_B + max(DS_IMM) + 16 <= LDS_SEG + 2 * 32 * S_ROW_B, "K3",
   f"DS immediate {max(DS_IMM)}")
# one-wave k_dkdv (nw = 1): allocation LDS_SEG + PDS_B, its P/dS tiles at LDS_SEG (wave 0's slot
# above) and its epilogue image at 0 (wave 0's image above), whole-tile TDM (rows [0, 32))
ok(EPI_W <= TDM_DEPTH * QDO_B and LDS_SEG + PDS_B <= ALLOC_KV and LDS_SEG + PDS_B <= 80 * 1024, "K3",
   "nw = 1: epilogue image inside the dead ring, P/dS tiles inside the 70656 B allocation")

# k_dqg LDS reads relative to the stage base
DSQ_IMM = []
RD_Q = []
for lane in LANES:
    row, half = lane_rc(lane)
    lr, lc = lane_tr(lane)
    for kt, dt, w, u in RD_ORDER:
        if w == "k":
            imm = kt * 16 * XK + dt * 64 + u * 32
            a = row * XK + half * 16 + imm
            ok(written(SPAN_K, 0, a, a + 16), "Q3", f"K readback {a}")
        else:
            imm = VOFF + kt * 16 * XV + dt * 64 + u * 32
            a = row * XV + half * 16 + imm
            ok(written(SPAN_V, VOFF, a, a + 16), "Q3", f"V readback {a}")
        DSQ_IMM.append(imm)
        RD_Q.append((a, a + 16))
    for dtile in range(NDO_QK):
        for r2 in (0, 16):
            imm = dtile * 32 + r2 * XK
            a = lr * XK + lc * 2 + imm
            DSQ_IMM.append(imm)
            ok(written(SPAN_K, 0, a, a + 16), "Q3", f"K tr16 {a}")
ok(len(RD_ORDER) == 2 * NKT * (NDT_QK + NDT_V), "Q3", "readback count")
ok(max(DSQ_IMM) < 65536, "Q3", f"DS immediate {max(DSQ_IMM)}")


class Ring:
    """TDM tensorcnt model: ops retire in order; wait(n) retires all but the newest n."""

    def __init__(self, tag, ops_per_stage, nst):
        self.tag, self.k, self.q = tag, ops_per_stage, []
        self.stage_tile = [None] * nst       # tile whose TDM ops last targeted the stage
        self.pending = [0] * nst             # unretired ops per stage
        self.reads = [0] * nst               # reads of the stage not yet consumed

    def issue(self, stage, tile):
        ok(self.reads[stage] == 0, self.tag, f"TDM into stage {stage} with an unconsumed read")
        self.stage_tile[stage] = tile
        for _ in range(self.k):
            self.q.append(stage)
            self.pending[stage] += 1
        ok(len(self.q) <= 2 * self.k, self.tag, "more than two stages in flight")

    def wait(self, n):
        while len(self.q) > n:
            self.pending[self.q.pop(0)] -= 1

    def read(self, stage, tile):
        ok(self.pending[stage] == 0, self.tag, f"read of stage {stage} with TDM in flight")
        ok(self.stage_tile[stage] == tile, self.tag, f"stage {stage} holds {self.stage_tile[stage]} not {tile}")
        self.reads[stage] += 1

    def consume(self, stage):
        self.reads[stage] = 0


def wrap(qc, gc, G):
    g1 = gc + 1
    return (qc, g1) if g1 < G else (qc + 1, 0)


class Ring2:
    """k_dkdv64 Q/dO ring shared by DKDV_NW waves (K2, K6). Every wave issues TDM_OPS_QDO ops per
    stage for its own 16-row slice and retires them with its own tensorcnt (in order: wait(n)
    leaves the newest n). Ops between two barriers form a PHASE whose cross-wave order is
    arbitrary; the barrier's release fence drains every wave's DS reads. Hence: a stage is
    readable iff at the last barrier every wave's slice of it was retired and nothing targeted
    it since; a TDM may not target a stage any wave reads in the same phase."""

    def __init__(self, nw, nst, k):
        self.nw, self.k = nw, k
        self.q = [[] for _ in range(nw)]               # per wave: stage of each unretired op
        self.slice_tile = [[None] * nw for _ in range(nst)]
        self.visible = [None] * nst                   # tile readable since the last barrier
        self.phase_tdm, self.phase_read = set(), set()
        self.nbar = [0] * nw

    def issue(self, stage, tile):
        ok(stage not in self.phase_read, "K6", f"TDM into stage {stage} read in the same phase")
        for w in range(self.nw):
            self.q[w] += [stage] * self.k
            self.slice_tile[stage][w] = tile
            ok(len(self.q[w]) <= 2 * self.k, "K2", "more than two stages in flight")
        self.phase_tdm.add(stage)
        self.visible[stage] = None

    def wait(self, n):
        for w in range(self.nw):
            while len(self.q[w]) > n:
                self.q[w].pop(0)

    def barrier(self):
        for w in range(self.nw):
            self.nbar[w] += 1
        for st in range(len(self.visible)):
            done = all(st not in self.q[w] for w in range(self.nw))
            tiles = set(self.slice_tile[st])
            if done and len(tiles) == 1 and None not in tiles:
                self.visible[st] = self.slice_tile[st][0]
        self.phase_tdm, self.phase_read = set(), set()

    def read(self, stage, tile):
        ok(stage not in self.phase_tdm, "K6", f"read of stage {stage} TDM'd in the same phase")
        ok(self.visible[stage] == tile, "K2", f"stage {stage} visible {self.visible[stage]} not {tile}")
        self.phase_read.add(stage)

    def epilogue(self):
        ok(all(not q for q in self.q), "K6", "TDM pending when the epilogue overwrites the ring")
        ok(not self.phase_tdm and not self.phase_read, "K6", "ring traffic in the epilogue phase")


# Static scan of _dkdv_impl (K6): the wave index `wv` only offsets addresses, every `if` is a
# compile-time const_expr (no scf.if, so no wave-dependent branch), the loop trip counts derive
# from kv0g (block id) only, and the barrier call sites are the six of the protocol.
_DKDV_SRC = _block("def _dkdv_impl(", "@flyc.kernel(known_block_size=[32, 1, 1])")
_WV_OK = (r"^wv = fx\.Int32\(rocdl\.wave_id\(\)\)", r"^kv0 = kv0g \+ wv \* fx\.Int32\(BLOCK_KV\)",
          r"^lds_p = lds_p \+ wv \* fx\.Int32\(PDS_B\)", r"^_epi0 = _lds0 \+ wv \* fx\.Int32\(EPI_W\)")
for ln in _DKDV_SRC.splitlines():
    t = ln.split("#", 1)[0].strip()
    if re.search(r"\bwv\b", t):
        ok(any(re.match(p_, t) for p_ in _WV_OK), "K6", f"wave index used outside addressing: {t}")
    if re.match(r"^(el)?if\b", t):
        ok("const_expr(" in t, "K6", f"runtime branch in _dkdv_impl: {t}")
    if re.match(r"^while\b", t):
        ok(False, "K6", f"while loop in _dkdv_impl: {t}")
for pat in (r"_c = kv0g - cshift", r"_u = kv0g \+ fx\.Int32\(nw \* BLOCK_KV - 1\) - cshift",
            r"qloop_mask\(init \+ _z2, G \* nmaskp, qp_start\)",
            r"qloop_full\(out, G \* \(nqp_eff - nmaskp\), qp_start \+ nmaskp\)"):
    ok(len(re.findall(pat, _DKDV_SRC)) == 1, "K6", f"trip-count source {pat}")
ok(sum(ln.split("#", 1)[0].strip() == "_lds_barrier()" for ln in _DKDV_SRC.splitlines()) == 6, "K6",
   "six barrier call sites (2 mask, 2 prologue, 1 full, 1 exit)")
ok(len(re.findall(r"const_expr\(carry and nw > 1 and kh == BAR_KH\)", _DKDV_SRC)) == 1
   and 0 < BAR_KH < NKV, "K6", "full-loop barrier between the kv sub-tiles' WMMAs")


# ------------------------------------------------------------------- k_dkdv replay
def dkdv_shape(B, Sq, Skv, Hq, Hkv, causal, written_kv, nw=DKDV_NW):
    """nw = DKDV_NW: k_dkdv64 (grid (Hkv, Skv/64, B)); nw = 1: k_dkdv (grid (Hkv, Skv/32, B))."""
    G = Hq // Hkv
    nqt = Sq // 16
    nqt2 = nqt // 2
    cshift = Skv - Sq
    n_q = B * Sq * Hq
    ldl_max = B * Hq * Sq
    ok(nw in (1, DKDV_NW), "K1", f"k_dkdv waves {nw}")
    NW, BKG = nw, nw * BLOCK_KV                    # waves, kv rows per workgroup
    RW = 32 // NW                                  # TDM rows per wave (cooperative num_warps = nw)
    ok(Skv % BKG == 0, "K1", f"Skv % {BKG} (impl._check: Skv % 64)")

    def clampqt(t):
        t = t if t < nqt2 else nqt2 - 1
        return 0 if t < 0 else t

    for bat in sorted({0, B - 1}):
        for hkv in sorted({0, Hkv - 1}):
            for bid in range(Skv // BKG):
                kv0g = bid * BKG                      # block id only
                _c = kv0g - cshift
                qp_start = (0 if _c < 0 else _c) // 32 if causal else 0
                nqp_eff = nqt2 - qp_start
                _u = kv0g + BKG - 1 - cshift
                qsf = 0 if _u < 0 else (_u + 31) // 32
                qsf = min(qsf, nqt2)
                nm = min(max(qsf - qp_start, 0), nqp_eff)
                nmaskp = nm if causal else 0
                kv0w = [kv0g + w * BLOCK_KV for w in range(NW)]
                # K4: masked/unmasked classification by brute force, for EACH wave's 32 kv rows
                for w in range(NW):
                    kv0 = kv0w[w]
                    for qt in range(nqt2):
                        full = (not causal) or (kv0 + BLOCK_KV - 1 <= 32 * qt + cshift)
                        none = causal and (kv0 > 32 * qt + 31 + cshift)
                        if qt < qp_start:
                            ok(none, "K4", f"wave {w} pair {qt} < qp_start {qp_start} not fully masked")
                        elif qt >= qp_start + nmaskp:
                            ok(full, "K4", f"wave {w} pair {qt} in the full loop has masked entries")

                def tdm(qt, gh, stage):
                    ok(0 <= qt < nqt2 and 0 <= gh < G, "K1", f"tile ({qt},{gh})")
                    q0 = qt * 32
                    ok(Sq - q0 >= 32, "K1", "TDM outer extent < 32")
                    qh = hkv * G + gh
                    for w in range(NW):                  # wave w's slice: rows q0 + w*RW + [0, RW)
                        ok(Sq - q0 - w * RW >= RW, "K1", "per-wave TDM extent clamp")
                        row0 = (bat * Sq + q0 + w * RW) * Hq + qh
                        for d in (D_V, D_QK):
                            for c0, w_ in segs(d):
                                first = row0 * d + c0
                                last = first + (RW - 1) * Hq * d + w_ - 1
                                ok(0 <= first and last < n_q * d, "K1", f"TDM global [{first},{last}]")
                    return (qt, gh)

                def ldl(qt, gh):
                    qh = hkv * G + gh
                    base_l = (bat * Hq + qh) * Sq
                    for hh in range(2):
                        for row in range(16):
                            idx = base_l + qt * 32 + hh * 16 + row
                            ok(0 <= idx < ldl_max, "K3", "LSE/delta index")

                if NW == 1:
                    # k_dkdv (nw = 1, bwd_c1/r1_b's instruction stream: no barrier; one wave issues
                    # and reads every stage, in-order tensorcnt): bwd_r1_b's single-wave replay.
                    ring = Ring("K2", TDM_OPS_QDO, TDM_DEPTH)
                    # qloop_mask: own tile -> stage 0, wait 0, read stage 0 (consumed in-iteration)
                    qi = gh = 0
                    for ii in range(G * nmaskp):
                        ok((qi, gh) == divmod(ii, G), "K4", "mask-loop counters")
                        ldl(qp_start + qi, gh)
                        t = tdm(qp_start + qi, gh, 0)
                        ring.issue(0, t)
                        ring.wait(0)
                        ring.read(0, t)
                        ring.consume(0)
                        qi, gh = wrap(qi, gh, G)
                    n = G * (nqp_eff - nmaskp)
                    qt0 = qp_start + nmaskp
                    ldl(clampqt(qt0), 0)
                    # prologue: tile(0) -> s0, tile(min(1, n-1)) -> s1, wait, read back s0
                    ring.issue(0, tdm(clampqt(qt0), 0, 0))
                    q1w, g1w = wrap(0, 0, G)
                    q1, g1 = (q1w, g1w) if 1 < n else (0, 0)
                    ring.issue(1, tdm(clampqt(qt0 + q1), g1, 1))
                    ring.wait(TW_QDO)
                    ring.read(0, (clampqt(qt0), 0))
                    cur, qi, gh = 0, 0, 0
                    for ii in range(n):
                        ok((qi, gh) == divmod(ii, G), "K4", "full-loop counters")
                        st = cur
                        qn, gn = wrap(qi, gh, G)
                        qj, gj = (qn, gn) if ii + 1 < n else (qi, gh)
                        q2, g2 = wrap(qn, gn, G)
                        pf = (q2, g2) if ii + 2 < n else (qj, gj)
                        kk = min(ii + 2, n - 1)
                        ok(pf == divmod(kk, G), "K4", "prefetch counters == min(ii+2, n-1)")
                        nxo = 2 if cur == 0 else cur - 1
                        ncur = 0 if cur == 2 else cur + 1
                        ok(nxo == (ii + 2) % 3 and ncur == (ii + 1) % 3 and st == ii % 3, "K2",
                           "stage rotation")
                        ring.issue(nxo, tdm(qt0 + pf[0], pf[1], nxo))   # top of the body, no wait
                        ldl(qt0 + qj, gj)
                        ring.consume(st)                     # carried readback -> S/dP WMMAs
                        ring.read(st, (qt0 + qi, gh))       # tr16 of this iteration's stage
                        ring.consume(st)                     # consumed by the dK/dV WMMAs
                        ring.wait(TW_QDO)
                        ring.read(ncur, (qt0 + qj, gj))      # readback for ii+1 (min(ii+1, n-1))
                        cur, qi, gh = ncur, qn, gn
                    ring.wait(0)
                    ok(not ring.q, "K2", "tensorcnt 0 at exit")     # epilogue overwrites the ring
                else:
                    ring = Ring2(NW, TDM_DEPTH, TDM_OPS_QDO)
                    # qloop_mask: BARRIER, own tile -> stage 0, wait 0, BARRIER, read stage 0
                    qi = gh = 0
                    for ii in range(G * nmaskp):
                        ok((qi, gh) == divmod(ii, G), "K4", "mask-loop counters")
                        ldl(qp_start + qi, gh)
                        ring.barrier()
                        t = tdm(qp_start + qi, gh, 0)
                        ring.issue(0, t)
                        ring.wait(0)
                        ring.barrier()
                        ring.read(0, t)                      # readback + tr16, consumed in-iteration
                        qi, gh = wrap(qi, gh, G)
                    n = G * (nqp_eff - nmaskp)
                    qt0 = qp_start + nmaskp
                    ldl(clampqt(qt0), 0)
                    # prologue: BARRIER, tile(0) -> s0, tile(min(1, n-1)) -> s1, wait, BARRIER, read s0
                    ring.barrier()
                    ring.issue(0, tdm(clampqt(qt0), 0, 0))
                    q1w, g1w = wrap(0, 0, G)
                    q1, g1 = (q1w, g1w) if 1 < n else (0, 0)
                    ring.issue(1, tdm(clampqt(qt0 + q1), g1, 1))
                    ring.wait(TW_QDO)
                    ring.barrier()
                    ring.read(0, (clampqt(qt0), 0))
                    q2w, g2w = wrap(q1w, g1w, G)
                    q2p, g2p = (q2w, g2w) if 2 < n else (q1, g1)
                    ok((q2p, g2p) == divmod(max(min(2, n - 1), 0), G), "K4", "prologue stage-2 tile == min(2, n-1)")
                    ring.issue(2, tdm(clampqt(qt0 + q2p), g2p, 2))
                    cur, qi, gh = 0, 0, 0
                    for ii in range(n):
                        ok((qi, gh) == divmod(ii, G), "K4", "full-loop counters")
                        st = cur
                        qn, gn = wrap(qi, gh, G)
                        qj, gj = (qn, gn) if ii + 1 < n else (qi, gh)
                        q2, g2 = wrap(qn, gn, G)
                        pf2 = (q2, g2) if ii + 2 < n else (qj, gj)
                        q3, g3 = wrap(q2, g2, G)
                        pf = (q3, g3) if ii + 3 < n else pf2
                        kk = min(ii + 3, n - 1)
                        ok(pf == divmod(kk, G), "K4", "prefetch counters == min(ii+3, n-1)")
                        ncur = 0 if cur == 2 else cur + 1
                        ok(ncur == (ii + 1) % 3 and st == ii % 3, "K2", "stage rotation")
                        ldl(qt0 + qj, gj)                    # LSE/delta prefetch at the top
                        ring.read(st, (qt0 + qi, gh))       # tr16 of this iteration's stage
                        ring.wait(TW_QDO)                    # after the WMMAs of kv sub-tiles < BAR_KH
                        ring.barrier()
                        ring.read(ncur, (qt0 + qj, gj))      # readback for ii+1 (min(ii+1, n-1))
                        ring.issue(st, tdm(qt0 + pf[0], pf[1], st))   # prefetch ii+3 -> stage ii%3, after BARRIER(ii)
                        cur, qi, gh = ncur, qn, gn
                    ring.wait(0)
                    ring.barrier()                           # epilogue barrier
                    ring.epilogue()
                    # K6: every wave took the same barriers, as many as the protocol says
                    nb_exp = 2 * G * nmaskp + 2 + n + 1
                    ok(all(x == nb_exp for x in ring.nbar), "K6", f"barriers {ring.nbar} != {nb_exp}")
                # K/V fragments of each wave's tile, dk/dv epilogue stores (vec8 index)
                rs_k, rs_v = Hkv * D_QK // 8, Hkv * D_V // 8
                base_k = bat * Skv * rs_k + hkv * D_QK // 8
                base_v = bat * Skv * rs_v + hkv * D_V // 8
                for kv0 in kv0w:
                    for kh in range(NKV):
                        for lane in LANES:
                            row, half = lane_rc(lane)
                            for base, rs, ndt, d in ((base_k, rs_k, NDT_QK, D_QK), (base_v, rs_v, NDT_V, D_V)):
                                for dt in range(ndt):
                                    for t in (base + (kv0 + kh * 16 + row) * rs + half + dt * 4,):
                                        for tt in (t, t + 2):
                                            ok(0 <= tt * 8 and tt * 8 + 8 <= B * Skv * Hkv * d, "K3", "K/V frag")
                                            ok((tt * 8) % d + 8 <= d, "K3", "K/V frag inside its row")
                            for sub in range(max(NDO_V, NDO_QK)):
                                for base, rs, nd, d, key in ((base_v, rs_v, NDO_V, D_V, "v"),
                                                             (base_k, rs_k, NDO_QK, D_QK, "k")):
                                    if sub < nd:
                                        t = base + (kv0 + kh * 16 + row) * rs + sub * 2 + half
                                        for e in range(8):
                                            el = t * 8 + e
                                            ok(0 <= el < B * Skv * Hkv * d, "K5", "dk/dv store")
                                            written_kv[key][el] = written_kv[key].get(el, 0) + 1


def dkdv_cover(B, Sq, Skv, Hq, Hkv, nw=DKDV_NW):
    """K5 exactly-once over ALL workgroups, by closed form (cheap): each (bat, hkv, bid, w, kh,
    row, sub, half) writes 8 consecutive elements of row kv0g + 32w + 16kh + row; the map is
    injective and onto the workgroup's nw*32 kv rows x d."""
    for key, d, nd in (("v", D_V, NDO_V), ("k", D_QK, NDO_QK)):
        seen = set()
        for w in range(nw):
            for kvr in range(BLOCK_KV):             # kh*16 + row
                for sub in range(nd):
                    for half in range(2):
                        c = (sub * 2 + half) * 8    # column of the 8-element run
                        ok(c + 8 <= d, "K5", "run inside the row")
                        seen.add((w * BLOCK_KV + kvr, c))
        ok(len(seen) == nw * BLOCK_KV * d // 8, "K5", f"d{key} tile coverage {len(seen)}")


# -------------------------------------------------------------------- k_dqg replay
def tdm_wave_blocks(nwave):
    """W1: per-wave TDM blocks of one stage (num_warps = nwave splits the 32 rows into nwave
    contiguous blocks, wave w = rows [w*32/nwave, (w+1)*32/nwave)). Returns {w: rows}."""
    ok(KV_STEP % nwave == 0, "W1", "rows split evenly")
    bpw = KV_STEP // nwave
    blocks = {w: list(range(w * bpw, (w + 1) * bpw)) for w in range(nwave)}
    for d, xrow, span in ((D_QK, XK, SPAN_K), (D_V, XV, SPAN_V)):
        got = []
        for w, rows in blocks.items():
            for c0, wd in segs(d):
                for r in rows:
                    got.append((r * xrow + 2 * c0, r * xrow + 2 * c0 + 2 * wd))
        ok(sorted(got) == sorted(span), "W1", f"per-wave LDS writes != single-wave image d{d}")
    return blocks


WAVE_BLOCKS = tdm_wave_blocks(DQ_NWAVE)


def ring2_events(N, nwave, mut=None):
    """Program-order event list per wave of k_dqg96 (kernels._dqg_tdm_impl, nwave > 1):
    ('tdm', stage, tile) | ('twait', n) | ('dscnt0',) | ('bar',) | ('read', stage, tile, kind) |
    ('use', kind) (a WMMA consuming that read kind: the compiler's s_wait_dscnt) | ('end',).
    mut: None, 'no_drain', 'read_above_bar', 'deep_wait', 'tdm_above_bar'."""
    TW = DQT_TW
    out = {}
    for w in range(nwave):
        ev = [("tdm", 0, 0), ("tdm", 1, min(1, N - 1)), ("twait", TW)]
        ev += [("dscnt0",), ("bar",), ("read", 0, 0, "rb")]
        for i in range(N):
            cur, nxo, ncur = i % 3, (i + 2) % 3, (i + 1) % 3
            ev.append(("tdm", nxo, min(i + 2, N - 1)))
            ev.append(("use", "rb"))                         # S/dP WMMAs: carried readback
            ev.append(("read", cur, i, "tr"))                # dQ B operands (tr16)
            ev.append(("twait", TW + 3 if mut == "deep_wait" else TW))
            if mut == "read_above_bar":
                ev.append(("read", ncur, min(i + 1, N - 1), "rb"))
            if mut != "no_drain":
                ev.append(("dscnt0",))
            ev.append(("bar",))
            if mut != "read_above_bar":
                ev.append(("read", ncur, min(i + 1, N - 1), "rb"))
            ev.append(("use", "tr"))                         # dQ WMMAs
        ev.append(("twait", 0))
        ev.append(("end",))
        out[w] = ev
    if mut == "tdm_above_bar":
        # wave 1 issues its prefetch for iteration i+1 BEFORE barrier i (no WAR protection)
        ev = out[1]
        for k in range(len(ev) - 1, 0, -1):
            if ev[k][0] == "bar" and k + 3 < len(ev) and ev[k + 3][0] == "tdm":
                ev.insert(k, ev.pop(k + 3))
        out[1] = ev
    return out


class Viol(Exception):
    pass


def ring2_check(N, nwave, mut=None):
    """W2 happens-before replay. Returns the number of barriers per wave; raises Viol."""
    evs = ring2_events(N, nwave, mut)
    owner = {w: w for w in range(nwave)}             # half w of every stage is written by wave w
    # per wave: annotate epoch (barriers passed) and completion indices
    writes = []      # (stage, half, tile, wave, issue_pos, done_pos)  pos = (wave, idx, epoch)
    reads = []       # (stage, tile, wave, issue_pos, done_pos, kind)
    nbar = {}
    for w, ev in evs.items():
        ep = 0
        q = []           # outstanding TDM ops of this wave: [write index] (one entry per op)
        open_reads = []  # read indices not yet completed
        last_kind = {}
        for k, e in enumerate(ev):
            pos = (w, k, ep)
            if e[0] == "tdm":
                for wr in writes:
                    if wr[3] == w and wr[0] == e[1] and wr[5] is None:
                        raise Viol(f"WAW: wave {w} TDM into stage {e[1]} with one in flight")
                writes.append([e[1], w, e[2], w, pos, None])
                for _ in range(TDM_OPS_KV):
                    q.append(len(writes) - 1)
                if len(q) > 2 * TDM_OPS_KV:
                    raise Viol("more than two stages in flight")
            elif e[0] == "twait":
                while len(q) > e[1]:
                    wi = q.pop(0)
                    if wi not in q:
                        writes[wi][5] = pos
            elif e[0] == "read":
                reads.append([e[1], e[2], w, pos, None, e[3]])
                open_reads.append(len(reads) - 1)
            elif e[0] == "use":
                # in-order LDS returns: waiting for the newest read of this kind retires all
                # older reads of this wave too
                idx = [r for r in open_reads if reads[r][5] == e[1]]
                if idx:
                    newest = max(idx)
                    for r in [r for r in open_reads if r <= newest]:
                        reads[r][4] = pos
                    open_reads = [r for r in open_reads if r > newest]
            elif e[0] == "dscnt0":
                for r in open_reads:
                    reads[r][4] = pos
                open_reads = []
            elif e[0] == "bar":
                ep += 1
            elif e[0] == "end":
                if q:
                    raise Viol(f"wave {w} ends with TDM in flight")
        nbar[w] = ep
    if len(set(nbar.values())) != 1:
        raise Viol(f"barrier counts differ {nbar} (hang)")

    def hb(a, b):
        if a is None or b is None:
            return False
        return a[1] < b[1] if a[0] == b[0] else a[2] < b[2]

    for st, tile, w, ri, rd, kind in reads:
        for half in range(nwave):
            ws = [x for x in writes if x[0] == st and x[1] == half]
            seen = None
            for x in ws:
                if hb(x[5], ri):
                    seen = x if seen is None or x[4][1] > seen[4][1] else seen
                elif not hb(rd, x[4]):
                    raise Viol(f"RAW/WAR: {kind} of stage {st} by wave {w} at {ri} races TDM of "
                               f"half {half} (tile {x[2]}) by wave {x[3]} issued {x[4]}")
            if seen is None or seen[2] != tile:
                raise Viol(f"RAW: {kind} of stage {st} half {half} by wave {w} expects tile {tile}, "
                           f"sees {None if seen is None else seen[2]}")
    return nbar[0]


def ring2_selftest():
    for N in range(1, 161):
        nb = ring2_check(N, DQ_NWAVE)
        ok(nb == N + 1, "W2", f"barriers {nb} for N {N}")
    for tag, mut in (("N1", "no_drain"), ("N2", "read_above_bar"), ("N3", "deep_wait"),
                     ("N4", "tdm_above_bar")):
        caught = 0
        for N in (1, 2, 3, 4, 7, 32):
            try:
                ring2_check(N, DQ_NWAVE, mut)
            except Viol:
                caught += 1
        ok(caught > 0, tag, f"mutation {mut} not caught")
        print(f"  mutation {tag} {mut}: caught at {caught}/6 trip counts", flush=True)


def dqg_shape(B, Sq, Skv, Hq, Hkv, causal, written_q, stats, launches):
    """launches: impl._geometry(...)["dq"], [(nqw, nwave, q_off, ntile)] in issue order."""
    G = Hq // Hkv
    nkvt = Skv // KV_STEP
    cshift = Skv - Sq
    # XCD remap bijection
    if Hq % 8 == 0:
        m = sorted((x % 8) * (Hq // 8) + x // 8 for x in range(Hq))
        ok(m == list(range(Hq)), "Q5", "XCD remap is a bijection")
    # Q6: the launches impl._plan issues for the dQ chain, (nqw, nwave, q_off, ntile) in order
    q_split, n32, n96 = dq_split(Sq)
    ok(q_split in (0, 32, 64) and q_split + DQ_BQW96 * n96 == Sq and DQ_BQW * n32 == q_split
       and NQW48 * 16 * DQ_NWAVE == DQ_BQW96 and NQW * 16 == DQ_BQW, "Q6",
       f"split {q_split} {n32} {n96}")
    ok(len(launches) >= 1 and all((nqw, nw_) in ((NQW48, DQ_NWAVE), (NQW, 1)) and ntile >= 1
                                  for nqw, nw_, _, ntile in launches), "Q6", f"dq launches {launches}")
    tiles = []                                   # (nqw, nwave, q0g) of every workgroup row of grid.y
    for nqw, nwave, q_off, ntile in launches:
        bqwg = 16 * nqw * nwave
        bids = [ntile - 1 - y for y in range(ntile)]          # kernel: bid = ntile-1-block_idx.y
        ok(sorted(bids) == list(range(ntile)), "Q5", "descending tile walk")
        for bid in bids:
            q0g = q_off + bid * bqwg
            ok(0 <= q0g and q0g + bqwg <= Sq and q0g % 16 == 0, "Q6", f"tile [{q0g}, {q0g + bqwg})")
            tiles.append((nqw, nwave, q0g))
        q0s = [q_off + b_ * bqwg for b_ in bids]
        ok(all(q0s[i] > q0s[i + 1] for i in range(len(q0s) - 1)), "Q6", "longest-first order")
    cov = sorted((q0g + w * 16 * nqw, q0g + (w + 1) * 16 * nqw)
                 for nqw, nwave, q0g in tiles for w in range(nwave))
    ok(cov[0][0] == 0 and cov[-1][1] == Sq and all(cov[i][1] == cov[i + 1][0]
                                                   for i in range(len(cov) - 1)),
       "Q6", "per-wave dQ tiles partition [0, Sq)")
    trips = set()
    for bat in sorted({0, B - 1}):
        for qh in sorted({0, Hq - 1}):
            hkv = qh // G
            for nqw, nwave, q0g in tiles:
                BQW = 16 * nqw
                BQWG = BQW * nwave
                # kv range of the WORKGROUP tile (kernels._dqg_tdm_impl), shared by its waves
                lim = (q0g + BQWG + cshift + KV_STEP - 1) // KV_STEP
                lim = min(max(lim, 1), nkvt)
                nkvt_eff = lim if causal else nkvt
                t_ = q0g + cshift + 1
                nf = 0 if t_ < 0 else t_ // KV_STEP
                nf = min(nf, nkvt_eff)
                nfull = nf if causal else nkvt_eff
                nlast = nkvt_eff - 1
                ok(nlast >= 0 and nfull <= nkvt_eff, "Q4", "loop bounds")
                rows_wg = range(q0g, q0g + BQWG)
                for q in (q0g, q0g + BQWG - 1):
                    hi = min(q + cshift, Skv - 1) if causal else Skv - 1
                    ok(hi < KV_STEP * nkvt_eff, "Q4", "kv coverage")
                for i in range(nfull):
                    ok((not causal) or KV_STEP * i + KV_STEP - 1 <= q0g + cshift, "Q4", "full iteration masked")
                if bat == 0 and qh == 0:         # Q6 per row (tile math is (bat, qh)-independent)
                    for q in rows_wg:
                        att = range(0, min(q + cshift, Skv - 1) + 1) if causal else range(Skv)
                        ok(len(att) == 0 or att[-1] < KV_STEP * nkvt_eff, "Q6", f"row {q} kv coverage")
                        for i in range(nfull):
                            ok(KV_STEP * i + KV_STEP - 1 in att, "Q6", f"row {q} full iteration {i}")
                    if causal:
                        for i in range(nfull, nkvt_eff):
                            ok(any(KV_STEP * i + j > q + cshift for q in rows_wg
                                   for j in range(KV_STEP)), "Q6", f"masked iteration {i} has no mask")
                        nm = nkvt_eff - nfull
                        if cshift == 0:
                            ok(nm == (3 if nwave == DQ_NWAVE else 1), "Q6",
                               f"masked steps {nm} for a {BQWG}-row tile at {q0g}")
                        ok(nm <= 4, "Q6", f"masked steps {nm} > 4")
                        # per wave: partial vs fully masked masked-loop steps
                        for w in range(nwave):
                            rows = range(q0g + w * BQW, q0g + (w + 1) * BQW)
                            full_m = sum(all(KV_STEP * i + j > q + cshift for q in rows
                                             for j in range(KV_STEP))
                                         for i in range(nfull, nkvt_eff))
                            unm = sum(all(KV_STEP * i + j <= q + cshift for q in rows
                                          for j in range(KV_STEP))
                                      for i in range(nfull, nkvt_eff))
                            key = (nwave, w, nm, full_m, unm)
                            stats[key] = stats.get(key, 0) + 1
                            if nwave == DQ_NWAVE and cshift == 0 and q0g + cshift >= 0:
                                ok(full_m == (1 if w == 0 else 0), "Q6",
                                   f"wave {w} fully masked steps {full_m} at {q0g}")
                trips.add(nkvt_eff)

                def tdm(kb, w=None):
                    ok(0 <= kb <= nkvt - 1, "Q1", f"kv block {kb}")
                    kv0p = kb * KV_STEP
                    ok(Skv - kv0p >= KV_STEP, "Q1", "TDM outer extent")
                    rows = range(KV_STEP) if w is None else WAVE_BLOCKS[w]
                    if w is not None:
                        ok(Skv - kv0p - rows[0] >= len(rows), "W1", "per-wave TDM extent")
                    for d in (D_QK, D_V):
                        for c0, wd in segs(d):
                            row0 = (bat * Skv + kv0p + rows[0]) * Hkv + hkv
                            first = row0 * d + c0
                            last = first + (len(rows) - 1) * Hkv * d + wd - 1
                            ok(0 <= first and last < B * Skv * Hkv * d, "W1" if w is not None else "Q1",
                               "TDM global")
                    return kb

                if nwave == 1:
                    ring = Ring("Q2", TDM_OPS_KV, TDM_DEPTH)
                    ring.issue(0, tdm(0))
                    for s_ in range(1, TDM_DEPTH - 1):
                        ring.issue(s_, tdm(min(s_, nlast)))
                    ring.wait(DQT_TW)
                    ring.read(0, 0)
                    cur = 0
                    for ii in range(nkvt_eff):
                        kk = min(ii + TDM_DEPTH - 1, nlast)
                        nxo = (TDM_DEPTH - 1) if cur == 0 else cur - 1
                        ncur = 0 if cur == TDM_DEPTH - 1 else cur + 1
                        ok(cur == ii % 3 and nxo == (ii + 2) % 3 and ncur == (ii + 1) % 3, "Q2", "rotation")
                        ring.issue(nxo, tdm(kk))              # top of the body, no wait
                        ring.consume(cur)                     # carried K/V A operands -> S/dP WMMAs
                        ring.read(cur, ii)                    # dQ B operand tr16 of K(ii)
                        ring.wait(DQT_TW)
                        ring.read(ncur, min(ii + 1, nlast))   # readback for ii+1
                        ring.consume(cur)                     # tr16 operands -> dQ WMMAs
                        cur = ncur
                    ring.wait(0)
                    ok(not ring.q, "Q2", "tensorcnt 0 at exit")
                else:
                    # every TDM tile the 2-wave replay issues, per wave block, in bounds
                    for kb in sorted({0, min(1, nlast)} | {min(i + 2, nlast) for i in range(nkvt_eff)}):
                        for w in range(nwave):
                            tdm(kb, w)
                # Q/dO fragments, LSE/delta, dQ stores: per wave
                base_l = (bat * Hq + qh) * Sq
                for w in range(nwave):
                    q0 = q0g + w * BQW
                    for lane in LANES:
                        row, half = lane_rc(lane)
                        for qh_ in range(nqw):
                            qg = q0 + qh_ * 16 + row
                            ok(0 <= base_l + qg < B * Hq * Sq and (base_l + qg) * 4 < (1 << 28), "Q3", "lse")
                            for d, ndt in ((D_QK, NDT_QK), (D_V, NDT_V)):
                                rs = Hq * d // 8
                                base = bat * Sq * rs + qh * d // 8
                                for dt in range(ndt):
                                    t = base + qg * rs + half + dt * 4
                                    for tt in (t, t + 2):
                                        ok(tt * 8 + 8 <= B * Sq * Hq * d and tt * 16 < (1 << 30), "Q3", "Q/dO frag")
                                        ok((tt * 8) % d + 8 <= d, "Q3", "frag inside its row")
                            base_dq = bat * Sq * Hq * D_QK + qh * D_QK
                            for dtile in range(NDO_QK):
                                for si in range(8):
                                    q_i = q0 + qh_ * 16 + half * 8 + si
                                    el = base_dq + q_i * Hq * D_QK + dtile * 16 + row
                                    ok(0 <= el < B * Sq * Hq * D_QK and el * 2 < (1 << 30), "Q5", "dq store")
                                    written_q[el] = written_q.get(el, 0) + 1
    # W2 on this shape's 2-wave trip counts (the self-test covers N = 1..160 as well)
    for N in sorted(trips):
        ring2_check(N, DQ_NWAVE)
        COUNT["W2"] = COUNT.get("W2", 0) + 1


# ------------------------------------------------------------------ head-group launch
def _xcd_remap(x, ngrp):
    """kernels._dqg_tdm_impl: x -> (x%8)*(ngrp/8) + x/8 when ngrp % 8 == 0, else x."""
    return (x % 8) * (ngrp // 8) + x // 8 if ngrp % 8 == 0 else x



# R1 source tie: the decode / grid / plan lines hg_decode, hg_launches and impl._plan replay
_IMPL_SRC = (HERE / "impl.py").read_text()
for src, pats in (
        (SRC, (r"_hb = fx\.Int32\(fx\.block_idx\.z\) \* HG \+ fx\.Int32\(fx\.block_idx\.x\)\n"
               r"\s+bat = _hb // Hkv\n\s+hkv = _hb - bat \* Hkv\n",
               r"ngrp = HG\n",
               r"qh = \(ngrp % _nx == fx\.Int32\(0\)\)\.select\(\(_gx % _nx\) \* \(ngrp // _nx\) \+ _gx // _nx, _gx\)",
               r"_hb = fx\.Int32\(fx\.block_idx\.z\) \* HG \+ qh\n\s+bat = _hb // Hq\n\s+qh = _hb - bat \* Hq\n",
               r"nqt, cshift, causal, nb, hg\)\.launch\(\n\s+grid=\(hg, nblk, ngz\)",
               r"nkvt, cshift, causal, q_off, ntile, hg\)\.launch\(\n\s+grid=\(hg, ntile, ngz\), block=\(32,",
               r"nkvt, cshift, causal, q_off, ntile, hg\)\.launch\(\n\s+grid=\(hg, ntile, ngz\), block=\(DQ_NWAVE",
               r"G, nqt, cshift, causal, B_, DKDV_NW, HG\)", r"nkvt, cshift, causal, q_off, ntile, NQW, 1, HG\)",
               r"nkvt, cshift, causal, q_off, ntile, NQW48, DQ_NWAVE, HG\)",
               # the one-wave k_dkdv: no head group (grid (Hkv, Skv/32, B), (bat, hkv) = (z, x))
               r"G, nqt, cshift, causal, B_\)\n\n\n@flyc\.jit\ndef launch_dkdv\(",
               r"nqt, cshift, causal, nb\)\.launch\(\n\s+grid=\(nhkv, nblk, nb\), block=\(32, 1, 1\)")),
        (_IMPL_SRC, (r"hg_kv, hg_q = _k\.head_group\(hkv, tgt\), _k\.head_group\(hq, tgt\)",
                     r"ngz_kv, ngz_q = b \* hkv // hg_kv, b \* hq // hg_q",
                     r"_k\.launch_dkdv64, dkdv_args \+ \(hg_kv, ngz_kv, b, stream\)",
                     r"_k\.launch_dkdv, dkdv_args \+ \(hkv, b, stream\)",
                     r"sq, skv, hq, hkv, g, sq // 16, skv - sq, c, nblk\)\n",
                     r"fn = _k\.launch_dqg96 if nwave == _k\.DQ_NWAVE else _k\.launch_dqg\n",
                     r"launches\.append\(\(nm, fn, dq_args \+ \(q_off, ntile, hg_q, ngz_q, dq_stream\), \"dq\"\)\)",
                     r"tgt = HEAD_GROUP if head_group is None else int\(head_group\)"))):
    for pat in pats:
        ok(len(re.findall(pat, src)) == 1, "R1", f"source line {pat}")


def hg_decode(kind, x, z, hg, nh):
    """(bat, head) the kernel derives from (block_idx.x, block_idx.z) under head group hg."""
    xx = x if kind == "kv" else _xcd_remap(x, hg)
    hb = z * hg + xx
    ok(0 <= hb < (1 << 31) and z * hg < (1 << 31), "R1", "i32 head index")
    bat = hb // nh
    return bat, hb - bat * nh


def r3a_decode(kind, x, z, nh):
    """r3_a: grid (nh, tiles, B); k_dkdv (bat, hkv) = (z, x), k_dqg* (z, XCD remap of x over nh)."""
    return z, (x if kind == "kv" else _xcd_remap(x, nh))


def hg_launches(B, Sq, Skv, Hq, Hkv, mode="two_wave"):
    """impl._plan's head-grouped launches in launch set `mode`: (name, kind, nh, tiles). The
    one-wave k_dkdv has no head group (grid (Hkv, Skv/32, B), r3_a's decode): not listed here,
    checked by headgroup_shape on its own."""
    g_ = geometry(B, Sq, Skv, Hq, Hkv, MODES[mode])
    nw, nblk = g_["dkdv"]
    out = [("dkdv", "kv", Hkv, nblk)] if nw == DKDV_NW else []
    for i, (_, _, _, ntile) in enumerate(g_["dq"]):
        out.append(("dqg" if i == 0 else "dqg_head", "q", Hq, ntile))
    if mode == "two_wave":                         # == bwd_r4_a's list (dq_split)
        q_split, n32, n96 = dq_split(Sq)
        ok([(a, b) for a, _, _, b in out] == [("dkdv", Skv // (DKDV_NW * BLOCK_KV))]
           + ([("dqg", n96)] if n96 else []) + ([("dqg_head" if n96 else "dqg", n32)] if n32 else []),
           "R1", f"two_wave launches {out}")
    return out


def hg_order(kind, B, nh, tiles, target):
    """(bat*nh + head, tile) of every workgroup in linear-id order (x fastest, then y, then z)."""
    hg = head_group(nh, target)
    ngz = B * nh // hg
    seq = []
    for z in range(ngz):
        for y in range(tiles):
            for x in range(hg):
                bat, h = hg_decode(kind, x, z, hg, nh)
                seq.append((bat * nh + h, y))
    return seq


def headgroup_shape(B, Sq, Skv, Hq, Hkv, name, mode="two_wave"):
    launched = {}
    g_ = geometry(B, Sq, Skv, Hq, Hkv, MODES[mode])
    if g_["dkdv"][0] == 1:
        # one-wave k_dkdv: grid (Hkv, Skv/32, B), (bat, hkv) = (block_idx.z, block_idx.x)
        img = {(z, x) for z in range(B) for x in range(Hkv)}
        ok(len(img) == B * Hkv, "R1", "k_dkdv decode onto [0,B)x[0,Hkv)")
        launched["dkdv"] = (Hkv, g_["dkdv"][1], B)
    for lname, kind, nh, tiles in hg_launches(B, Sq, Skv, Hq, Hkv, mode):
        for target in R_TARGETS:
            hg = head_group(nh, target)
            ok(1 <= hg <= nh and nh % hg == 0 and (B * nh) % hg == 0, "R1", f"{lname} hg {hg} of {nh}")
            ok(hg == nh or (hg % 8 == 0 and hg <= target), "R1", f"{lname} hg {hg} target {target}")
            ok(target > 0 and nh > target or hg == nh, "R1", f"{lname} identity rule hg {hg}")
            ngz = B * nh // hg
            ok(hg * tiles * ngz == nh * tiles * B, "R1", "workgroup count unchanged")
            img = {}
            xcd = {}
            for z in range(ngz):
                for x in range(hg):
                    bat, h = hg_decode(kind, x, z, hg, nh)
                    ok(0 <= bat < B and 0 <= h < nh, "R1", f"{lname} decode ({x},{z}) -> ({bat},{h})")
                    ok((bat, h) not in img, "R1", f"{lname} ({bat},{h}) twice")
                    img[(bat, h)] = (x, z)
                    if hg == nh:
                        ok((bat, h) == r3a_decode(kind, x, z, nh), "R1", f"{lname} hg == nh is r3_a's map")
                    if hg % 8 == 0:
                        for y in range(tiles):
                            xcd.setdefault((bat, h), set()).add((x + hg * (y + tiles * z)) % 8)
            ok(len(img) == B * nh, "R1", f"{lname} decode onto [0,B)x[0,nh)")
            if hg % 8 == 0:
                ok(all(len(v) == 1 for v in xcd.values()), "R1", f"{lname} head split across XCDs")
                COUNT["R1_xcd"] = COUNT.get("R1_xcd", 0) + 1
            if target == HEAD_GROUP:
                launched[lname] = (hg, tiles, ngz)
            if target == 0:
                ok(hg == nh and ngz == B, "R2", f"{lname} target 0 is r3_a's grid")
    return launched


def fold_equivalence():
    """R2: prodfold at target 128 walks (folded head, tile) exactly like r3_a's b2h128; prodfold at
    the default target HEAD_GROUP walks like the contiguous b2h128 launch at the same target."""
    Bp, S, H = 2, 4096, 128
    for (lname, kind, nh, tiles), (lname2, kind2, nh2, tiles2) in zip(
            hg_launches(1, S, S, Bp * H, Bp * H), hg_launches(Bp, S, S, H, H)):
        ok((lname, kind, tiles) == (lname2, kind2, tiles2) and nh == Bp * nh2, "R2", "launch lists")
        fold = hg_order(kind, 1, nh, tiles, B2H128_TARGET)
        ref = []                                   # r3_a grid (H, tiles, Bp) over BSHD b2 h128
        for z in range(Bp):
            for y in range(tiles):
                for x in range(H):
                    bat, h = r3a_decode(kind, x, z, H)
                    ref.append((bat * H + h, y))   # = the folded head of (bat, h)
        ok(fold == ref, "R2", f"{lname}: fold traversal != b2h128 traversal")
        old = hg_order(kind, 1, nh, tiles, 0)      # r3_a on the fold launch, for the record
        same = sum(a == b for a, b in zip(old, ref))
        print(f"[R2] {lname}: prodfold@{B2H128_TARGET} == b2h128 order over {len(ref)} workgroups "
              f"(r3_a's fold order agrees at {same})", flush=True)
        fold_d = hg_order(kind, 1, nh, tiles, HEAD_GROUP)
        ref_d = hg_order(kind, Bp, nh2, tiles2, HEAD_GROUP)   # (bat*128 + h, tile) = folded head
        ok(fold_d == ref_d, "R2", f"{lname}: fold@{HEAD_GROUP} != b2h128@{HEAD_GROUP} traversal")
        agree = sum(a == b for a, b in zip(fold_d, ref))
        print(f"[R2] {lname}: prodfold@{HEAD_GROUP} == b2h128@{HEAD_GROUP} order over {len(ref_d)} "
              f"workgroups (agrees with r3_a's b2h128 order at {agree})", flush=True)


def r3a_geometry(B, Sq, Skv, Hq, Hkv):
    """bwd_r3_a's _plan dispatch, restated (G1 reference): k_dkdv64 + dq_split's k_dqg96 / head."""
    q_split, n32, n96 = dq_split(Sq)
    return {"dkdv": (DKDV_NW, Skv // (BLOCK_KV * DKDV_NW)),
            "dq": (([(NQW48, DQ_NWAVE, q_split, n96)] if n96 else []) + ([(NQW, 1, 0, n32)] if n32 else []))}


def c1_geometry(B, Sq, Skv, Hq, Hkv):
    """bwd_c1's _plan dispatch, restated (G1 reference): k_dkdv nblk Skv/32, one k_dqg ntile Sq/32."""
    return {"dkdv": (1, Skv // BLOCK_KV), "dq": [(NQW, 1, 0, Sq // DQ_BQW)]}


def g1_dispatch(name, B, Sq, Skv, Hq, Hkv):
    """G1: which launch set the real threshold selects; returns {'two_wave'|'one_wave' per chain}."""
    geo = geometry(B, Sq, Skv, Hq, Hkv)
    want = (B * Hkv * (Skv // BLOCK_KV) < SMALL_GRID_WAVES["dkdv"], B * Hq * (Sq // DQ_BQW) < SMALL_GRID_WAVES["dq"])
    ok(geo["small"] == want, "G1", f"{name}: small {geo['small']} != {want}")
    if name in EXPECT_SMALL:
        ok(geo["small"] == EXPECT_SMALL[name], "G1", f"{name}: small {geo['small']} != expected {EXPECT_SMALL[name]}")
    two, one = geometry(B, Sq, Skv, Hq, Hkv, MODES["two_wave"]), geometry(B, Sq, Skv, Hq, Hkv, MODES["one_wave"])
    ok(two["small"] == (False, False) and one["small"] == (True, True), "G1", "forced modes")
    r3, c1 = r3a_geometry(B, Sq, Skv, Hq, Hkv), c1_geometry(B, Sq, Skv, Hq, Hkv)
    ok(two["dkdv"] == r3["dkdv"] and two["dq"] == r3["dq"], "G1", f"{name}: two_wave set != bwd_r3_a's {two} {r3}")
    ok(one["dkdv"] == c1["dkdv"] and one["dq"] == c1["dq"], "G1", f"{name}: one_wave set != bwd_c1's {one} {c1}")
    ok(geo["dkdv"] == (one if want[0] else two)["dkdv"] and geo["dq"] == (one if want[1] else two)["dq"], "G1",
       f"{name}: launched geometry is not the replayed mode of its decision")
    for g_ in (two, one):
        nw, nblk = g_["dkdv"]
        ok(nblk * nw * BLOCK_KV == Skv and nblk >= 1, "G1", f"{name}: dkdv grid ({Hkv}, {nblk}, {B}) tiles Skv")
    # fold invariance: [B, S, H, D] views of SBHD storage launch as [1, S, B*H, D] (MHA and GQA alike)
    gf = geometry(1, Sq, Skv, B * Hq, B * Hkv)
    ok(gf["small"] == geo["small"] and gf["dkdv"] == geo["dkdv"] and gf["dq"] == geo["dq"], "G1",
       f"{name}: fold launch decides differently {gf} vs {geo}")
    return {"dkdv": "one_wave" if want[0] else "two_wave", "dq": "one_wave" if want[1] else "two_wave"}


def main(names, modes):
    print(f"[G1] SMALL_GRID_WAVES {SMALL_GRID_WAVES} (impl.py); modes {modes}", flush=True)
    print("[W2] 2-wave ring protocol self-test, N = 1..160 + mutations", flush=True)
    ring2_selftest()
    fold_equivalence()
    for name in names:
        B, Sq, Skv, Hq, Hkv, causal = SHAPES[name]
        assert Sq % 64 == 0 and Sq % DQ_BQW == 0 and Skv % 64 == 0 and Hq % Hkv == 0  # impl.py _check
        assert (B * Sq * Hq) % 32 == 0                                                  # ROWS_DELTA
        sel = g1_dispatch(name, B, Sq, Skv, Hq, Hkv)
        geo = geometry(B, Sq, Skv, Hq, Hkv)
        print(f"[{name}] launched: dkdv {sel['dkdv']} {geo['dkdv']} (one-wave WGs {B * Hkv * (Skv // BLOCK_KV)}), "
              f"dq {sel['dq']} {geo['dq']} (one-wave WGs {B * Hq * (Sq // DQ_BQW)}); "
              f"dq_split {dq_split(Sq)} (q_split, n32, n96)", flush=True)
        for mode in modes:
            g_ = geometry(B, Sq, Skv, Hq, Hkv, MODES[mode])
            nw = g_["dkdv"][0]
            hgl = headgroup_shape(B, Sq, Skv, Hq, Hkv, name, mode)
            wkv = {"k": {}, "v": {}}
            wq = {}
            stats = {}
            dkdv_shape(B, Sq, Skv, Hq, Hkv, causal, wkv, nw)
            dkdv_cover(B, Sq, Skv, Hq, Hkv, nw)
            dqg_shape(B, Sq, Skv, Hq, Hkv, causal, wq, stats, g_["dq"])
            # exactly-once on the replayed (extreme) workgroups: every replayed element once
            ok(all(v == 1 for v in wkv["k"].values()) and all(v == 1 for v in wkv["v"].values()),
               "K5", "dk/dv element written twice")
            ok(all(v == 1 for v in wq.values()), "Q5", "dq element written twice")
            nb, nh = len({0, B - 1}), len({0, Hkv - 1})
            ok(len(wkv["k"]) == nb * nh * Skv * D_QK and len(wkv["v"]) == nb * nh * Skv * D_V,
               "K5", f"dk/dv coverage {len(wkv['k'])}")
            ok(len(wq) == len({0, B - 1}) * len({0, Hq - 1}) * Sq * D_QK, "Q5", "dq coverage")
            print(f"[{name}/{mode}] B{B} Sq{Sq} Skv{Skv} Hq{Hq} Hkv{Hkv} causal{causal}: dkdv nw {nw}, dq "
                  f"{g_['dq']}; grids at HEAD_GROUP {HEAD_GROUP} (hg, tiles, B*nh/hg): "
                  + ", ".join(f"{k}={v}" for k, v in hgl.items())
                  + "; masked-step classes (nwave, wave, masked, fully_masked, unmasked): count = "
                  + ", ".join(f"{k}: {v}" for k, v in sorted(stats.items())), flush=True)
    print("checks:", " ".join(f"{k}={v}" for k, v in sorted(COUNT.items())))
    print(f"layout: D_QK {D_QK} D_V {D_V} rows {XK}/{XV} B, segments qk {segs(D_QK)} v {segs(D_V)}; "
          f"k_dkdv64 {DKDV_NW} waves, stage {QDO_B} B x {TDM_DEPTH} = {TDM_DEPTH * QDO_B} B shared, "
          f"{TDM_OPS_QDO} TDM ops/stage/wave ({ROWS_W} rows each), P/dS {PDS_B} B/wave, alloc {ALLOC_KV} B, "
          f"epilogue {EPI_W} B/wave, "
          f"wait {TW_QDO}; k_dqg stage {KV_B} B x {TDM_DEPTH} = {TDM_DEPTH * KV_B} B, {TDM_OPS_KV} ops, "
          f"wait {DQT_TW}; k_dqg96 {DQ_NWAVE} waves x {KV_STEP // DQ_NWAVE} TDM rows; "
          f"DS imm max {max(DS_IMM)} / {max(DSQ_IMM)}")
    print("BOUNDS_PROOF PASS")


if __name__ == "__main__":
    args = sys.argv[1:]
    modes = list(MODES)
    if args[:1] == ["--modes"]:
        modes, args = args[1].split(","), args[2:]
        assert modes and all(m in MODES for m in modes), modes
    main(args or list(SHAPES), modes)
