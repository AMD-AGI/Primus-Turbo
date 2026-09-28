"""CPU bounds proof for k_dqg's index expressions (lab-kdq). Pure python, no torch, no GPU.

Mirrors kernels.py:_dqg_impl literally for every workgroup/wave of every UT shape x causal
mode (as if the grouped path applied everywhere, which is stronger than impl's rule, which
only routes proxy/prod to it) and every arm config. Checks, in element units of each tensor:
Q/dO/O vec8 loads, K/V vec8 loads (incl. the prologue and carried prefetch), LSE/DEL
scalar loads + the DFUSE DEL store, DQ bf16 stores, LDS byte offsets vs the allocation.
"""
SHAPES = {"fast": (1, 1024, 1024, 8, 2), "proxy": (1, 4096, 4096, 32, 8),
          "prod": (4, 8192, 8192, 32, 8), "toy": (1, 128, 128, 2, 1),
          "gqa4_small": (2, 128, 128, 8, 2), "mha": (1, 256, 256, 4, 4),
          "unequal_seqlen": (1, 512, 1024, 8, 2), "unequal_seqlen_2": (2, 1024, 2048, 4, 1),
          "sq_gt_skv": (2, 1024, 512, 4, 1)}
D, DV8, KV_STEP, X_ROW_B, NDT, NDO, NKT = 128, 16, 32, 272, 4, 8, 2
CONFIGS = [(4, 64, True), (2, 64, True), (1, 64, True), (1, 32, False), (4, 32, False),
           (1, 32, True), (4, 32, True)]


def check(shape, causal, NW, BQW, PF):
    b, sq, skv, hq, hkv = SHAPES[shape]
    if hq % NW or sq % BQW or skv % 32:
        return None
    G = hq // hkv; cshift = skv - sq; nkvt = skv // KV_STEP
    ngrp = hq // NW; NQW = BQW // 16
    nq_vec = b * sq * hq * DV8; nkv_vec = b * skv * hkv * DV8; nl = b * hq * sq
    ndq = b * sq * hq * D
    heads_seen = set(); rows_seen = 0; viol = []; nacc = 0
    for gx in range(ngrp):
        grp = (gx % 8) * (ngrp // 8) + gx // 8 if ngrp % 8 == 0 else gx
        for gy in range(sq // BQW):
            bid = sq // BQW - 1 - gy
            q0 = bid * BQW
            lim = (q0 + BQW + cshift + KV_STEP - 1) // KV_STEP
            lim = max(lim, 1); lim = min(lim, nkvt)
            nk_eff = lim if causal else nkvt
            t = q0 + cshift + 1
            nf = 0 if t < 0 else t // KV_STEP
            nf = min(nf, nk_eff)
            nfull = nf if causal else nk_eff
            # kv blocks touched: loop blocks [0, nk_eff), prefetch: prologue 0, carried
            # jj = min(ii+1, nfull-1) for ii in [0, nfull) -> all within [0, max(nfull,1))
            blocks = set(range(nk_eff)) | ({0} if PF else set())
            if PF:
                blocks |= {min(ii + 1, nfull - 1) for ii in range(nfull)}
            kvmax = max(blocks) * KV_STEP + KV_STEP - 1   # max kv row loaded
            kvmin = min(blocks) * KV_STEP
            for bat in range(b):
                for wave in range(NW):
                    qh = grp * NW + wave
                    heads_seen.add((bat, qh, bid))
                    hk = qh // G
                    rs_q, rs_kv = hq * DV8, hkv * DV8
                    base_q = bat * sq * rs_q + qh * DV8
                    base_kv = bat * skv * rs_kv + hk * DV8
                    base_l = (bat * hq + qh) * sq
                    # Q/dO/O: rows q0 + qh_*16 + row, row 0..15; vec half + dt*4 (+2), +7 lanes? no:
                    # a vec8 index addresses 8 elements; max vec = base + r*rs + 1 + 12 + 2
                    rmax = q0 + (NQW - 1) * 16 + 15
                    vq = [base_q + r * rs_q + h + dt * 4 + e for r in (q0, rmax)
                          for h in (0, 1) for dt in (0, 3) for e in (0, 2)]
                    vk = [base_kv + r * rs_kv + h + dt * 4 + e for r in (kvmin, kvmax)
                          for h in (0, 1) for dt in (0, 3) for e in (0, 2)]
                    vl = [base_l + r for r in (q0, rmax)]
                    vd = [(bat * sq * hq * D + qh * D) + r * hq * D + dtile * 16 + row
                          for r in (q0, rmax) for dtile in (0, NDO - 1) for row in (0, 15)]
                    nacc += 1
                    if not (0 <= min(vq) and max(vq) < nq_vec): viol.append(("q", shape, qh, bid))
                    if not (0 <= min(vk) and max(vk) < nkv_vec): viol.append(("kv", shape, qh, bid, kvmax))
                    if not (0 <= min(vl) and max(vl) < nl): viol.append(("l", shape, qh, bid))
                    if not (0 <= min(vd) and max(vd) < ndq): viol.append(("dq", shape, qh, bid))
                    if kvmax >= skv: viol.append(("kvrow", shape, qh, bid, kvmax))
                    if not (0 <= qh < hq): viol.append(("qh", shape, qh))
                    # LDS: wave*KV_STEP*X_ROW_B + ko + dt*64+u*32 + 16 bytes (store), tr loads
                    st_max = wave * KV_STEP * X_ROW_B + (16 + 15) * X_ROW_B + 16 + 3 * 64 + 32 + 16
                    ld_max = (wave * KV_STEP * X_ROW_B + 15 * X_ROW_B + (8 + 7 * 16) * 2
                              + 16 * X_ROW_B + 16)
                    lo = wave * KV_STEP * X_ROW_B
                    if st_max > (wave + 1) * KV_STEP * X_ROW_B or ld_max > (wave + 1) * KV_STEP * X_ROW_B:
                        viol.append(("lds", wave, st_max, ld_max))
    cover = len(heads_seen) == b * hq * (sq // BQW)
    return nacc, viol, cover


print("config(NW,BQW,PF) shape causal -> waves checked, violations, full (b,qh,tile) coverage")
bad = 0
for cfg in CONFIGS:
    for s in SHAPES:
        for c in ((False,) if s == "sq_gt_skv" else (True, False)):
            r = check(s, c, *cfg)
            if r is None:
                print(f"{cfg} {s:17s} causal={c}: not eligible (hq%NW or sq%BQW)"); continue
            n, v, cov = r
            bad += len(v) + (not cov)
            print(f"{cfg} {s:17s} causal={c}: waves={n} viol={len(v)} cover={cov} {v[:3]}")
print("TOTAL violations+uncovered:", bad)
