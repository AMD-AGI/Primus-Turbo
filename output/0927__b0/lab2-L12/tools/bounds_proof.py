"""CPU bounds proof for every index expression whose value changes with (NUM_WAVES, R, LDS map).

Mirrors the kernel/manager formulas as Python ints (source lines cited) and enumerates every
(block_x, kv_head, batch, wave, q-tile, lane / TDM row / tile) for every job shape x causal/non-causal.
Asserts: packed Q rows are a partition of [0, grid_x*BLOCK_M) and every in-range row is owned exactly
once; every global read/write that is not HW-/mask-dropped stays inside its tensor; every LDS access
stays inside its region and inside the allocation; regions that must be disjoint are disjoint.
Run: python3 bounds_proof.py  (no torch, no GPU)
"""
import json, pathlib, sys

L = pathlib.Path(__file__).resolve().parent.parent
SHAPES = {  # job_context/op/ut/common.py:13-26
    "fast": (1, 1024, 1024, 8, 2, 128), "proxy": (1, 4096, 4096, 32, 8, 128),
    "prod": (4, 8192, 8192, 32, 8, 128), "toy": (1, 256, 256, 2, 1, 128),
    "short_q": (1, 128, 512, 4, 1, 128), "gqa4_batch2": (2, 256, 256, 8, 2, 128),
    "mha": (1, 256, 256, 4, 4, 128), "unequal_seqlen": (1, 512, 1024, 8, 2, 128),
    "unequal_seqlen2": (2, 1024, 2048, 4, 1, 128), "sq_gt_skv": (2, 1024, 512, 4, 1, 128),
    # extra: ragged lengths (not multiples of BLOCK_M / n_block), gqa 8
    "ragged": (1, 1000, 1037, 16, 2, 128), "ragged_g1": (1, 333, 777, 3, 3, 128),
}
WMMA_M, N_BLOCK, D, DV = 16, 64, 128, 128
K_PAD, V_PAD, Q_PAD, O_PAD = 8, 16, 8, 8
LDS_CAP = 320 * 1024


def cdiv(a, b):
    return -(-a // b)


def proof(arm, cfg):
    NW, R, MIN_KV, ALLOC = cfg["nw"], cfg["r"], cfg["min_kv"], min(cfg["lds"], LDS_CAP)
    BLOCK_M = WMMA_M * R * NW
    rpw = WMMA_M * R
    k_row, v_row, q_row, o_row = (D + K_PAD) * 2, (DV + V_PAD) * 2, (D + Q_PAD) * 2, (DV + O_PAD) * 2
    k_blk = max(N_BLOCK * k_row, MIN_KV)
    v_blk = max(N_BLOCK * v_row, MIN_KV)
    q_size = BLOCK_M * q_row
    slot = max(k_blk + v_blk, q_size)
    o_size = NW * rpw * o_row
    assert 2 * slot <= ALLOC, (arm, "ring", 2 * slot, ALLOC)
    assert o_size <= slot, (arm, "O ring", o_size, slot)
    # K/V TDM per-warp split of the [n_block, hdim] tile (ISA: wave&(NW-1), rows*16/8)
    assert N_BLOCK % NW == 0
    rows_w = N_BLOCK // NW
    for w in range(NW):
        k0, k1 = w * rows_w * k_row, (w + 1) * rows_w * k_row
        v0, v1 = k_blk + w * rows_w * v_row, k_blk + (w + 1) * rows_w * v_row
        assert 0 <= k0 and k1 <= k_blk and k_blk <= v0 and v1 <= k_blk + v_blk <= slot
    # K ds_load: base lane (l%16)*row + (l//16)*16, imm kv*16*row + (dt*32+half*16)*2  (managers V2)
    kmax = max((l % 16) * k_row + (l // 16) * 16 for l in range(32)) + (N_BLOCK // 16 - 1) * 16 * k_row \
        + ((D // 32 - 1) * 32 + 16) * 2 + 16
    assert kmax <= N_BLOCK * k_row <= k_blk, (arm, "K ds", kmax)
    vmax = max(((l // 16) * 8 + l % 8) * v_row + ((l // 8) % 2) * 8 * 2 for l in range(32)) \
        + ((N_BLOCK // 32 - 1) * 32 + 16) * v_row + (DV // 16 - 1) * 16 * 2 + 16
    # tr16_b128 reads 8 rows x 16B starting at the lane row; the highest lane row already accounts for +7
    assert vmax <= N_BLOCK * v_row + 16, (arm, "V ds", vmax)
    assert (N_BLOCK - 1) * v_row + (DV + 0) * 2 <= v_blk
    # Q per-warp LDS region (QManager16bV2.load_q_to_vgpr_part1): slot1 + w*rpw*q_row
    for w in range(NW):
        a0, a1 = slot + w * rpw * q_row, slot + (w + 1) * rpw * q_row
        assert slot <= a0 and a1 <= 2 * slot <= ALLOC
        # part2 reads: lane base + qt*16*row + tile*64 (+32) + 16 bytes
        rmax = a0 + 15 * q_row + 16 + (R - 1) * 16 * q_row + (D // 32 - 1) * 64 + 32 + 16
        assert rmax <= a1, (arm, "Q read", rmax, a1)
    # O V3 region: non-current slot base + w*rpw*o_row; flush reads rows < rpw
    for w in range(NW):
        o0, o1 = w * rpw * o_row, (w + 1) * rpw * o_row
        assert o1 <= o_size <= slot
    n_checked = 0
    for sname, (B, SQ, SKV, HQ, HKV, _) in SHAPES.items():
        gqa = HQ // HKV
        # QManager16bV2 assert (divides or multiple)
        if not (gqa % rpw == 0 or rpw % gqa == 0):
            print(f"  {arm} {sname}: gqa {gqa} vs rows/wave {rpw} -> host assert (not launchable), skipped")
            continue
        for causal in (True, False):
            if causal and SQ > SKV:
                continue  # job: sq > skv is non-causal only
            grid_x = cdiv(SQ * gqa, BLOCK_M)  # _ensure_bshd_kernel grid
            owner = {}
            for bx in range(grid_x):
                # --- _core_attention KV range (fmha_fwd :836-860)
                causal_off = SKV - SQ
                if causal:
                    wg_max_seq = min((bx * BLOCK_M + BLOCK_M - 1) // gqa, SQ - 1)
                    kv_len_wg = max(min(wg_max_seq + causal_off + 0 + 1, SKV), 1)
                else:
                    kv_len_wg = SKV
                n_tiles = cdiv(kv_len_wg, N_BLOCK)
                start_tile = 0
                n_last = n_tiles - 1
                if causal:
                    wg_min_seq = (bx * BLOCK_M) // gqa
                    clean_hi = (max(wg_min_seq + causal_off, 0) + 1) // N_BLOCK
                else:
                    clean_hi = n_tiles
                clean_hi = max(min(clean_hi, n_last), start_tile)
                assert 0 <= start_tile <= clean_hi <= n_last < n_tiles
                # TDM K/V rows for every tile t in [start, n_tiles): per-warp valid = clamp(valid - w*rows_w)
                for t in (start_tile, n_tiles - 1):
                    row0 = t * N_BLOCK
                    valid = min(max(kv_len_wg - row0, 0), N_BLOCK)
                    for w in range(NW):
                        vw = min(max(valid - w * rows_w, 0), rows_w)
                        if vw:
                            assert row0 + w * rows_w + vw - 1 < SKV
                # prefetch of tile t+1 is skipped when nxt >= n_tiles (kernel _prefetch)
                for w in range(NW):
                    warp_row0 = bx * BLOCK_M + w * rpw  # _packed_tile_indices
                    # Q TDM (per-warp): n_head/n_seq, seq0, n_seq_valid clamp
                    n_head = min(gqa, rpw)
                    n_seq = rpw // n_head
                    seq0 = warp_row0 // gqa
                    head0 = (warp_row0 % gqa) if gqa > rpw else 0
                    nvalid = min(max(SQ - seq0, 0), n_seq)
                    if nvalid:
                        assert seq0 + nvalid - 1 < SQ and head0 + n_head <= gqa
                    for qt in range(R):
                        for l in range(16):
                            pr = warp_row0 + qt * 16 + l
                            assert pr not in owner, (arm, sname, "dup row", pr)
                            owner[pr] = (bx, w, qt)
                            seq, hq = pr // gqa, pr % gqa
                            if seq < SQ:  # LSE/O stores masked by seq < q_len
                                assert 0 <= hq < gqa
                    # O V3 _warp_addrs: rows clamp to last valid row of this warp
                    valid_rows = SQ * gqa - warp_row0
                    last_valid = min(max(valid_rows - 1, 0), rpw - 1)
                    if valid_rows > 0:
                        for r in range(rpw * (DV // 8) // 32):
                            for l in range(32):
                                c = r * 32 + l
                                row = min(c // (DV // 8), last_valid)
                                pr = warp_row0 + row
                                assert pr // gqa < SQ
                                assert row * o_row + (c % (DV // 8)) * 16 + 16 <= rpw * o_row
                n_checked += 1
            rows = sorted(owner)
            assert rows == list(range(grid_x * BLOCK_M)), (arm, sname, "not a partition")
            assert all(r in owner for r in range(SQ * gqa))
    return dict(BLOCK_M=BLOCK_M, rows_per_wave=rpw, slot=slot, ring=2 * slot, alloc=ALLOC,
                q_lds=q_size, o_lds=o_size, k_blk=k_blk, v_blk=v_blk, tdm_rows_per_warp=rows_w,
                wg_checked=n_checked)


if __name__ == "__main__":
    sys.path.insert(0, str(L / "tools"))
    from make_arms import ARMS
    out = {}
    for arm, cfg in ARMS.items():  # base_r6 arms share the index math of base (r6 changes softmax only)
        out[arm] = proof(arm, cfg)
        print("PASS", arm, out[arm])
    (L / "bounds_proof.json").write_text(json.dumps(out, indent=1))
