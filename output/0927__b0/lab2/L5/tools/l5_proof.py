"""CPU safety proof for lever L5 (causal-aligned q-tile origin), no GPU.

Integer model of every index expression the L5 edit touches, transcribed from
  rot/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py  (_q_row_origin, _packed_tile_indices,
      _core_attention kv-range / clean-split, LSE store, _softmax masks)
  rot/flydsl_fwd/fmha_b16_buffer_managers.py      (QManager16bV2.load_q_to_vgpr_part1,
      OManager16bV3._warp_addrs + valid_rows>0 burst gate)
run for BOTH the champion (align=False) and L5 (align=True). Checks, per config:

 P1 origin: every warp_row0 in [0, gx*BLOCK_M), multiple of RPW (no wave straddles the wrap,
    Q-loader precondition gqa|RPW or RPW|gqa is unchanged).
 P2 bijection: the multiset of all (bx, warp, row) packed rows == range(gx*BLOCK_M), so every
    valid (seq, head) is computed by exactly one lane-row; padding rows map to seq >= q_len.
 P3 loads: Q TDM reads seq in [seq0, seq0 + max(q_len-seq0, 0)) -> 0 <= seq < q_len only.
 P4 stores: O (V3) writes packed rows [wb, wb+min(valid,RPW)) iff valid>0, LSE writes lane rows
    with seq<q_len; every write is a valid (seq, head), and each valid (seq, head) is written
    exactly once for O and once for LSE.
 P5 masks unchanged: for every valid row, the kv positions that survive the kernel's per-region
    masks over [start_tile, n_tiles) equal the reference band
      [max(0, s+off-wl), min(s+off+wr, kv_len-1)]  (causal/window)  or  [0, kv_len-1]
    and lie inside [0, kv_len_wg) (never a zero-filled K row). Champion and L5 both equal the
    reference, hence equal each other.
 P6 cost: KV-tile visits sum_WG (n_tiles - start_tile), champion vs L5.
"""
import itertools
import sys

import numpy as np

BLOCK_M, RPW, NB, R, WM = 256, 32, 64, 2, 16


def origin(bx, w, gx, q_len, g, align):
    """(warp_row0, lo, hi) exactly as _q_row_origin (bx, w are numpy arrays)."""
    base = bx * BLOCK_M
    if not align:
        return base + w * RPW, base, base + BLOCK_M - 1
    tot = gx * BLOCK_M
    pad = max(tot - q_len * g, 0)
    pad_w = ((pad // RPW) % (BLOCK_M // RPW)) * RPW
    r = base + w * RPW - pad_w
    r = np.where(r < 0, r + tot, r)
    lo = np.maximum(base - pad_w, 0)
    hi = base + BLOCK_M - 1 - pad_w
    return r, lo, hi, pad_w


def check(q_len, kv_len, g, gx, mask_left, mask_right, wl, wr, align):
    tot = gx * BLOCK_M
    bx = np.arange(gx)[:, None]                  # [gx,1]
    w = np.arange(8)[None, :]                    # [1,8]
    o = origin(bx, w, gx, q_len, g, align)
    wr0, lo, hi = o[0], o[1], o[2]
    wr0 = np.broadcast_to(wr0, (gx, 8))
    # P1
    assert (wr0 >= 0).all() and (wr0 < tot).all(), "P1 range"
    assert (wr0 % RPW == 0).all(), "P1 align"
    # P2: rows of every lane-row  (row = warp_row0 + qt*16 + lane%16, qt<R, lane%16<16)
    rows = (wr0[:, :, None] + np.arange(RPW)[None, None, :])        # [gx,8,32]
    flat = np.sort(rows.ravel())
    assert np.array_equal(flat, np.arange(tot)), "P2 bijection"
    nvalid = q_len * g
    seq = rows // g
    valid = rows < nvalid
    assert np.array_equal(valid, seq < q_len), "P2 valid<->seq<q_len"
    # P3: Q loader per warp (seq0 = warp_row0//g; gqa>RPW -> n_seq=1)
    n_head = min(g, RPW); n_seq = RPW // n_head
    seq0 = wr0 // g
    nsv = np.maximum(q_len - seq0, 0)
    assert (seq0 >= 0).all(), "P3 seq0>=0"
    rd_hi = seq0 + np.minimum(nsv, n_seq)       # exclusive
    assert (rd_hi[nsv > 0] <= q_len).all(), "P3 read past q_len"  # nsv==0 -> TDM reads nothing
    # P4: O store (V3)
    vr = nvalid - wr0
    o_written = []
    for b in range(gx):
        for ww in range(8):
            if vr[b, ww] > 0:
                o_written.append(np.arange(wr0[b, ww], wr0[b, ww] + min(vr[b, ww], RPW)))
    o_written = np.concatenate(o_written) if o_written else np.zeros(0, int)
    assert (o_written >= 0).all() and (o_written < nvalid).all(), "P4 O OOB"
    assert np.array_equal(np.sort(o_written), np.arange(nvalid)), "P4 O exactly-once"
    lse_rows = rows[valid]
    assert np.array_equal(np.sort(lse_rows), np.arange(nvalid)), "P4 LSE exactly-once"
    # WG kv partition (per bx), exactly as _core_attention
    off = kv_len - q_len
    lo = np.broadcast_to(lo, (gx, 1))[:, 0]; hi = np.broadcast_to(hi, (gx, 1))[:, 0]
    assert (lo >= 0).all() and (hi >= 0).all()
    if mask_right:
        wg_max = np.minimum(hi // g, q_len - 1)
        kv_len_wg = np.maximum(np.minimum(wg_max + off + wr + 1, kv_len), 1)
    else:
        kv_len_wg = np.full(gx, kv_len)
    n_tiles = -(-kv_len_wg // NB)
    if mask_left:
        wg_min = lo // g
        kv_lo = np.maximum(wg_min + off - wl, 0)
        start = np.minimum(kv_lo // NB, n_tiles - 1)
    else:
        start = np.zeros(gx, int)
    n_last = n_tiles - 1
    if mask_right:
        qmax_min = np.maximum(lo // g + off + wr, 0)
        clean_hi = (qmax_min + 1) // NB
    else:
        clean_hi = n_tiles.copy()
    clean_hi = np.maximum(np.minimum(clean_hi, n_last), start)
    if mask_left:
        wg_max2 = np.minimum(hi // g, q_len - 1)
        qmin_max = np.maximum(wg_max2 + off - wl, 0)
        clean_lo = (qmin_max + NB - 1) // NB
    else:
        clean_lo = start.copy()
    clean_lo = np.minimum(np.maximum(clean_lo, start), clean_hi)
    # P5: per valid row, surviving kv interval(s) vs reference band
    BIG = 1 << 40
    vb, vw, vi = np.nonzero(valid)
    s = seq[vb, vw, vi]
    qmax = s + off + wr if mask_right else None
    qmin = s + off - wl if mask_left else None
    ref_lo = np.maximum(qmin, 0) if mask_left else np.zeros_like(s)
    ref_hi = np.minimum(qmax, kv_len - 1) if mask_right else np.full_like(s, kv_len - 1)
    got_lo = np.full_like(s, BIG); got_hi = np.full_like(s, -BIG); covered = np.zeros_like(s)
    regions = []
    if mask_left:
        regions.append((start, clean_lo, True, True, False))
    regions.append((clean_lo, clean_hi, False, False, False))
    regions.append((clean_hi, n_tiles, mask_left, mask_right, True))
    for (t0, t1, ml, mr, kvl) in regions:
        a = t0[vb] * NB; b = t1[vb] * NB - 1     # positions [a, b]
        lo_i = a.copy(); hi_i = b.copy()
        if ml and mask_left:
            lo_i = np.maximum(lo_i, qmin)
        if mr and mask_right:
            ub = np.minimum(qmax, kv_len - 1) if kvl else qmax
            hi_i = np.minimum(hi_i, ub)
        if kvl and not (mr and mask_right):
            hi_i = np.minimum(hi_i, kv_len - 1)
        ne = lo_i <= hi_i
        covered += np.where(ne, hi_i - lo_i + 1, 0)
        got_lo = np.where(ne, np.minimum(got_lo, lo_i), got_lo)
        got_hi = np.where(ne, np.maximum(got_hi, hi_i), got_hi)
    ref_n = np.maximum(ref_hi - ref_lo + 1, 0)
    nonempty = ref_n > 0
    assert np.array_equal(covered, ref_n), "P5 attended count != band"
    assert (got_lo[nonempty] == ref_lo[nonempty]).all() and (got_hi[nonempty] == ref_hi[nonempty]).all(), "P5 band"
    assert (got_hi[nonempty] < kv_len_wg[vb][nonempty]).all(), "P5 read zero-filled K"
    return int((n_tiles - start).sum()), (o[3] if align else 0)


def main():
    gqas = (1, 2, 4, 8, 16, 32, 64)
    modes = {"causal": (False, True, 0, 0), "nc": (False, False, 0, 0),
             "win128": (True, True, 128, 0)}
    n = 0; worst = {}
    # exhaustive small q_len, sq == skv and sq != skv, BSHD (gx from q_len) and THD (gx from max_sq)
    sqs = list(range(1, 400)) + [511, 512, 513, 1000, 1023, 1024, 1025, 2047, 4095, 4096, 4097, 8100, 8191, 8192]
    for g, sq, (mname, (ml, mr, wl, wr)) in itertools.product(gqas, sqs, modes.items()):
        for skv in {sq, sq + 7, sq + 100, max(1, sq - 5) if mname == "nc" else sq}:
            gx_b = -(-(sq * g) // BLOCK_M)
            for gx, lay in ((gx_b, "bshd"), (gx_b + 1 + (sq % 3), "thd")):
                c0, _ = check(sq, skv, g, gx, ml, mr, wl, wr, False)
                c1, pw = check(sq, skv, g, gx, ml, mr, wl, wr, True)
                n += 1
                if lay == "bshd" and pw == 0:
                    assert c0 == c1, "pad_w==0 but visits changed"
                k = (mname, g)
                worst.setdefault(k, [0, 0, 0])
                worst[k][0] += c0; worst[k][1] += c1; worst[k][2] += (c1 > c0)
    print(f"configs checked (x2 arms): {n}; P1-P5 hold for champion and L5 in every one")
    print("aggregate KV-tile visits over the sweep (champion, L5, #configs where L5 visits MORE):")
    for k, v in sorted(worst.items()):
        print(f"  {k[0]:7s} gqa={k[1]:2d}  {v[0]:10d} {v[1]:10d}  ({(v[1]-v[0])/v[0]*100:+.2f}%)  worse_in={v[2]}")
    print("scored shapes (B folds out; visits per (b, kv_head)):")
    for name, (sq, g) in {"fast": (1024, 4), "proxy": (4096, 4), "prod": (8192, 4)}.items():
        gx = -(-(sq * g) // BLOCK_M)
        c0, _ = check(sq, sq, g, gx, False, True, 0, 0, False)
        c1, pw = check(sq, sq, g, gx, False, True, 0, 0, True)
        r0 = origin(np.arange(gx)[:, None], np.arange(8)[None, :], gx, sq, g, False)[0]
        r1 = origin(np.arange(gx)[:, None], np.arange(8)[None, :], gx, sq, g, True)[0]
        print(f"  {name:5s} sq={sq} gqa={g} gx={gx} pad={gx*BLOCK_M-sq*g} pad_w={pw} "
              f"row map identical={np.array_equal(r0, r1)} visits {c0} -> {c1}")
    print("ragged examples (causal, sq == skv):")
    for sq, g in ((8100, 4), (8100, 1), (8000, 1), (8160, 1), (4000, 2), (3000, 1), (8100, 8)):
        gx = -(-(sq * g) // BLOCK_M)
        c0, _ = check(sq, sq, g, gx, False, True, 0, 0, False)
        c1, pw = check(sq, sq, g, gx, False, True, 0, 0, True)
        print(f"  sq={sq} gqa={g} pad={gx*BLOCK_M-sq*g} pad_w={pw} visits {c0} -> {c1} ({(c1-c0)/c0*100:+.2f}%)")
    print("ragged examples (causal bottom-right, sq != skv):")
    for sq, skv, g in ((8100, 8192, 4), (1000, 4096, 4), (4000, 4096, 1), (8100, 8192, 1)):
        gx = -(-(sq * g) // BLOCK_M)
        c0, _ = check(sq, skv, g, gx, False, True, 0, 0, False)
        c1, pw = check(sq, skv, g, gx, False, True, 0, 0, True)
        print(f"  sq={sq} skv={skv} gqa={g} pad_w={pw} visits {c0} -> {c1} ({(c1-c0)/c0*100:+.2f}%)")
    print("PROOF_OK")


if __name__ == "__main__":
    sys.exit(main())
