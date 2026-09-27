"""CPU proof of the prototype's tile partition, TDM ring and wait counts (bshd causal, mask_left=False).
Mirrors the kernel's integer expressions; enumerates every WG (block_x) for several shapes."""
import itertools
BLOCK_M, NB, GQA = 256, 64, 4
LDS_CAP, TPT = 327680, 2  # TDM ops per tile per wave (K + V at hdim 128)

def cdiv(a, b): return -(-a // b)

def wg(block_x, q_len, kv_len, U, D):
    causal_off = kv_len - q_len
    wg_max_seq = min((block_x * BLOCK_M + BLOCK_M - 1) // GQA, q_len - 1)
    kv_len_wg = max(min(wg_max_seq + causal_off + 1, kv_len), 1)
    n_tiles = cdiv(kv_len_wg, NB); start = 0
    n_last = n_tiles - 1
    wg_min_seq = (block_x * BLOCK_M) // GQA
    qmax_min = max(wg_min_seq + causal_off, 0)
    clean_hi = max(min((qmax_min + 1) // NB, n_last), start)
    clean_lo = min(max(start, start), clean_hi)
    order, masked = [], []
    if U == 2:
        n_pairs = (clean_hi - clean_lo) // 2
        clean_hi2 = clean_lo + 2 * n_pairs
        assert clean_hi2 <= clean_hi and clean_hi - clean_hi2 in (0, 1)
        for p in range(n_pairs):
            order += [clean_lo + 2 * p, clean_lo + 2 * p + 1]
        clean_hi = clean_hi2
    else:
        order += list(range(clean_lo, clean_hi))
    for t in range(clean_hi, n_tiles):
        order.append(t); masked.append(t)
    assert order == list(range(start, n_tiles)), (block_x, order)
    # every tile that needs masking (any query row q with kv index > q+causal_off) is in the masked loop
    for t in range(start, n_tiles):
        needs = (t + 1) * NB - 1 > wg_min_seq + causal_off or (t + 1) * NB > kv_len
        assert not needs or t in masked, (block_x, t)
    # ---- TDM ring simulation (per wave, in-order tensorcnt) ----
    slot = lambda t: (t - start) % D
    inflight = []  # FIFO of (tile, n_ops)
    resident = set(); readers = {}  # slot -> tile being read this iteration
    def issue(t):
        assert 0 <= slot(t) < D and D * SLOT(D) <= LDS_CAP
        inflight.append(t)
    issue(start)
    for j in range(1, D - 1):
        if start + j < n_tiles: issue(start + j)
    Q_SLOT = D - 1
    def wait(keep_tiles):
        while len(inflight) > keep_tiles: resident.add(inflight.pop(0))
    # prologue wait
    wait(1 if (D == 3 and start + 1 < n_tiles) else 0)
    assert start in resident
    for t in range(start, n_tiles):
        # top of tile: wait + barrier
        keep = 1 if (D == 3 and t + 1 < n_tiles) else 0
        wait(keep)
        assert t in resident, (block_x, t)
        # barrier: every wave finished reading tile t-1's slot; prefetch target
        nxt = t + D - 1
        if nxt < n_tiles:
            tgt = slot(nxt)
            live = {slot(t)} | {slot(x) for x in inflight}
            assert tgt not in live, (block_x, t, tgt, live)
            issue(nxt)
    assert not inflight
    o_slot = (n_tiles - start) % D
    assert o_slot != slot(n_tiles - 1)
    return n_tiles, len(masked)

def SLOT(D):
    floor = min(64 * 1024, (LDS_CAP // (2 * D)) // 1024 * 1024)
    k = max(64 * (128 + 8) * 2, floor); v = max(64 * (128 + 16) * 2, floor)
    q = 256 * (128 + 8) * 2; o = 8 * 32 * (128 + 8) * 2
    s = max(k + v, q); assert o <= s; return s

for D in (2, 3):
    print(f"N_KV_PP={D}: slot={SLOT(D)} B, {D} slots = {D*SLOT(D)} B <= {LDS_CAP}")
n = 0
for (b, s), U, D in itertools.product([(4, 8192), (1, 1024), (1, 64), (1, 100), (1, 1088), (1, 200), (1, 4000)], (1, 2), (2, 3)):
    q_len = kv_len = s
    gx = cdiv(s * GQA, BLOCK_M)
    extra = 0
    for bx in range(gx):
        nt, nm = wg(bx, q_len, kv_len, U, D); n += 1; extra = max(extra, nm)
    print(f"s={s:5d} U={U} D={D}: {gx} WGs ok, max masked tiles/WG = {extra}")
print("ALL OK", n, "WG schedules checked")
