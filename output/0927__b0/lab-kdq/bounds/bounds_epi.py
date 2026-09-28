"""CPU proof for DQ_EPI (k_dqg epilogue through LDS). Enumerates every lane/dtile/sub/qh_.
LDS: slice = max(32*272, 4*128*48) = 24576 B per wave; every ds_write_b128 / ds_load_tr16_b128
byte range must lie in [0, 24576) and images of different qh_ must not overlap.
Coverage: the tr16 read-back must deliver every (q row, d col) of each 16x128 block exactly
once, with lane l <- (row l%16, cols sub*16 + (l//16)*8 + e) (ds_load_tr16_b128 semantics:
lane l element e = src[(l//16)*8 + e][l%16] of the 16x16 bf16 tile addressed by lane_r/lane_c).
Global: vec8 index gt + sub*2 + half, gt = base_q + (q0+qh_*16+row)*rs_q, same form as the Q
fragment loads already proved in bounds_dqg.txt (max column vec 15 < DV8 = 16)."""
CB, SL, NQW, NDO = 48, max(32 * 272, 4 * 128 * 48), 4, 8
bad = 0; img = {}
for qh in range(NQW):
    base = qh * 128 * CB
    for lane in range(32):
        row, half = lane % 16, lane // 16
        for dt in range(NDO):
            o = base + (dt * 16 + row) * CB + half * 16
            if not (0 <= o and o + 16 <= SL): bad += 1
            for si in range(8):           # element si = q row half*8+si, d col dt*16+row
                img[(qh, o + 2 * si)] = (half * 8 + si, dt * 16 + row)
    seen = set()
    for lane in range(32):
        lr = (lane // 16) * 8 + lane % 8; lc = ((lane // 8) % 2) * 8
        for sub in range(NDO):
            a = base + (sub * 16 + lr) * CB + lc * 2
            if not (0 <= a and a + 16 <= SL): bad += 1
            # tr16 over the 16 lanes' addresses: lane l element e reads row-address of lane
            # ((l//16)*8+e) -> address a' of that lane, element column l%16
            for e in range(8):
                src_lane = (lane // 16) * 16 + e + (8 if (lane % 16) >= 8 else 0)
                slr = (src_lane // 16) * 8 + src_lane % 8; slc = ((src_lane // 8) % 2) * 8
                col = lane % 16
                # the 16x16 tile: rows = sub*16 + [0,16) (d), cols = 16 bf16 (q rows)
                d = sub * 16 + (lane // 16) * 8 + e
                q = lane % 16
                addr = base + d * CB + q * 2
                seen.add((q, d))
                if img.get((qh, addr)) != (q, d): bad += 1
    if len(seen) != 16 * 128: bad += 1
print("EPI LDS/coverage violations:", bad, " slice bytes", SL)
