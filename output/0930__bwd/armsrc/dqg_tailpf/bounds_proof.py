"""dqg_tailpf bounds proof (host, no torch). The arm changes NO index/address/predicate
expression: body 1 now issues _ldkv(jj*KV_STEP) (same jj, same clamp) that r29's body 2
issued. This enumerates, for every CTA / trip / lane / kt / dt, the vec8 element index of
every K/V load of the U2 loop (head prefetch ii+1, tail prefetch jj, prologue block 0) and
checks 0 <= idx and idx+1 (b128 = 8 bf16 = one vec8 unit... gfrag2 loads units t and t+2)
< b*Skv*Hkv*DV8, and that every loaded kv row < Skv."""
D, DV8, KV_STEP, BQW, NKT, NDT = 128, 16, 32, 64, 2, 4
def check(b, s, hq, hkv, causal):
    sq = skv = s; G = hq // hkv; cshift = skv - sq; nkvt = skv // KV_STEP
    rs_kv = hkv * DV8; lim_units = b * skv * hkv * DV8
    nload = 0; maxrow = -1
    for bid in range(sq // BQW):
        q0 = bid * BQW
        lim = (q0 + BQW + cshift + KV_STEP - 1) // KV_STEP
        lim = max(lim, 1); lim = min(lim, nkvt)
        nkvt_eff = lim if causal else nkvt
        t = q0 + cshift + 1
        nf = 0 if t < 0 else t // KV_STEP
        nf = min(nf, nkvt_eff)
        nfull = nf if causal else nkvt_eff
        n = nfull // 2
        blocks = {0}
        for it in range(n):
            ii = 2 * it; jj = ii + 2
            jj = jj if jj < 2 * n else 2 * n - 1
            blocks |= {ii + 1, jj}
        for blk in blocks:
            kv0 = blk * KV_STEP
            for kt in range(NKT):
                for lane in range(32):
                    row, half = lane % 16, lane // 16
                    r = kv0 + kt * 16 + row
                    assert 0 <= r < skv, (b, s, bid, blk, r)
                    maxrow = max(maxrow, r)
            nload += 1
    # address: base_kv + r*rs_kv + half + dt*4 (+2), base_kv = (bat*skv)*rs_kv + hkv_i*DV8
    for bat in (0, b - 1):
        for hk in (0, hkv - 1):
            base = bat * skv * rs_kv + hk * DV8
            for r in (0, maxrow):
                for half in (0, 1):
                    for dt in range(NDT):
                        for u in (0, 2):
                            idx = base + r * rs_kv + half + dt * 4 + u
                            assert 0 <= idx < lim_units, idx
    return nload, maxrow
for shp in [(4, 8192, 32, 8), (1, 1024, 8, 2), (1, 128, 2, 1)]:
    for causal in (1, 0):
        nl, mr = check(*shp, causal)
        print(f"b{shp[0]} s{shp[1]} hq{shp[2]} hkv{shp[3]} causal={causal}: "
              f"blocks={nl} max kv row={mr} < {shp[1]}  OK")
print("ALL IN BOUNDS")
