"""CPU proof for the two unroll-by-2 loop rewrites (DQ_U2 in k_dqg, KV_U2 in k_dkdv).

Both rewrites only re-index an existing loop over iteration indices [0, n), so the proof
is about index sequences, for every n that can occur (0..4096 covers every UT shape:
k_dq nkvt_eff <= Skv/32 = 256, k_dkdv G*nqp <= 4*256 = 1024):
  (1) the CONSUMED sequence (full loop + leftover loop) is exactly 0, 1, .., n-1, in order;
  (2) every PREFETCH index lies in [0, n-1] (or is the prologue load of index 0 whose
      address is the original's -- k_dq block 0 / k_dkdv _clampqt(...) -- unchanged);
  (3) each carried prefetch is consumed by the iteration it was issued for (pipeline);
  (4) the ORIGINAL loops' prefetch indices are a superset-compatible reference: every
      U2 prefetch index also occurs among the original loop's prefetch indices.
k_dq: the full range is [0, nfull); pairs np = nfull//2 run in kvloop_full, and the mask
loop runs [2np, nkvt_eff) -- (1) is checked over (nfull, nkvt_eff) pairs with
nkvt_eff - nfull in {0, 1, 2} (at most two masked blocks, kernels.py) plus a wide sweep.
k_dkdv: the full range [0, n) is split into pairs [0, 2*n2) + tail [2*n2, n).
"""
bad = 0; cases = 0


def orig(n):
    pf = [min(i + 1, n - 1) for i in range(n)]
    return list(range(n)), pf


def u2(n):
    n2 = n // 2
    consumed, pf, carried = [], [], (0 if n > 0 else None)
    for p in range(n2):
        ii = 2 * p
        assert carried == ii, (n, p, carried)          # (3)
        consumed.append(ii); mid = ii + 1; pf.append(mid)
        consumed.append(mid)
        jj = min(ii + 2, 2 * n2 - 1); pf.append(jj); carried = jj
    for i in range(2 * n2, n):                        # leftover loop, loads its own
        consumed.append(i)
    return consumed, pf


for n in range(0, 4097):
    cases += 1
    c, pf = u2(n)
    oc, opf = orig(n)
    if c != list(range(n)): bad += 1; print("consumed mismatch", n)
    if any(not (0 <= x <= n - 1) for x in pf): bad += 1; print("prefetch OOR", n)
    if n and set(pf) - set(opf) - {n - 1}: bad += 1; print("new prefetch index", n)

# k_dq split of [0, nkvt_eff) into full pairs + mask loop, for every (nfull, nmask<=2) and a sweep
for nk in range(1, 300):
    for nf in range(max(0, nk - 2), nk + 1):
        cases += 1
        npair = nf // 2
        seq = list(range(2 * npair)) + list(range(2 * npair, nk))
        if seq != list(range(nk)): bad += 1; print("dq split mismatch", nk, nf)
        # the leftover full block run in the MASK loop must be fully unmasked there too:
        # it is < nfull, so its causal predicate is false by nfull's definition (checked
        # numerically in bounds_dqg-style below for every UT shape)
print(f"index-sequence cases {cases}, failures {bad}")

# leftover full block in kvloop_mask: predicate kv > q + cshift must be false for EVERY
# element when block < nfull, for every UT shape / q tile / BQW in {32, 64}.
SH = {"fast": (1024, 1024), "proxy": (4096, 4096), "prod": (8192, 8192), "toy": (128, 128),
      "gqa4_small": (128, 128), "mha": (256, 256), "unequal_seqlen": (512, 1024),
      "unequal_seqlen_2": (1024, 2048)}
viol = 0; chk = 0
for s, (sq, skv) in SH.items():
    cshift = skv - sq; nkvt = skv // 32
    for BQW in (32, 64):
        for bid in range(sq // BQW):
            q0 = bid * BQW
            lim = min(max((q0 + BQW + cshift + 31) // 32, 1), nkvt)
            t = q0 + cshift + 1
            nfull = min(0 if t < 0 else t // 32, lim)
            for blk in range(nfull):
                chk += 1
                if blk * 32 + 31 > q0 + cshift:       # largest key vs smallest query
                    viol += 1
print(f"leftover-in-mask-loop predicate checks {chk}, live-mask violations {viol}")
print("TOTAL", bad + viol)
