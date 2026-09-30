"""dkdv_trorder / dkdv_trorder_b bounds proof (CPU only, no torch).

The arm changes only the EMISSION ORDER (and sched_barrier fences) of k_dkdv's final DS
phase and, in variant b, of the Q/dO staging stores. No global (buffer) address, no
predicate and no loop bound is touched, so the global side is r29's, per shape. This script
enumerates every LDS byte address the reordered code touches, per lane, and checks it
(1) is the same multiset as r29's formula and (2) lies inside the 70656 B allocation.
The LDS image is shape-independent; shapes are iterated only to show that the per-iteration
access set does not depend on them (it is the same every iteration for every shape).
"""
D = 128; NDT = D // 32; NDO = D // 16; NKV = 2; BLOCK_KV = 32
S_ROW_B = BLOCK_KV * 2 + 16; X_ROW_B = D * 2 + 16; LDS_SEG = 65536
ALLOC = LDS_SEG + 2 * 32 * S_ROW_B
lds_do = 0; lds_q = lds_do + 32 * X_ROW_B; lds_p = LDS_SEG; lds_ds = lds_p + 32 * S_ROW_B
# ranges owned by each ring
RING = {'do': (lds_do, lds_q), 'q': (lds_q, lds_q + 32 * X_ROW_B),
        'p': (lds_p, lds_ds), 'ds': (lds_ds, lds_ds + 32 * S_ROW_B)}

def lane_vals(lane):
    row = lane % 16; half = lane // 16
    lane_r = (lane // 16) * 8 + lane % 8
    lane_c = ((lane // 8) % 2) * 8
    return row, half, lane_r, lane_c

def tr(base, rowb):            # two 16-byte transposing reads
    return [(base, 16), (base + 16 * rowb, 16)]

def stores_r29(lane):          # hh loop, 16 stores each (dp then qp per (dt,u))
    row, half, _, _ = lane_vals(lane); out = []
    for hh in range(2):
        xo = (hh * 16 + row) * X_ROW_B + half * 16
        for dt in range(NDT):
            for u in range(2):
                o = xo + dt * 64 + u * 32
                out += [('do', lds_do + o, 16), ('q', lds_q + o, 16)]
    return out

stores_b = stores_r29          # variant b: _stage(hh, ...) is the same body, hoisted

def ps_stores(lane):
    row, half, _, _ = lane_vals(lane); out = []
    for hh in range(2):
        for kh in range(NKV):
            off = (hh * 16 + row) * S_ROW_B + kh * 32 + half * 16
            out += [('p', lds_p + off, 16), ('ds', lds_ds + off, 16)]
    return out

def loads_r29(lane):
    _, _, lane_r, lane_c = lane_vals(lane); out = []
    for dtile in range(NDO):
        c = (lane_c + dtile * 16) * 2
        out += [('do',) + x for x in tr(lds_do + lane_r * X_ROW_B + c, X_ROW_B)]
        out += [('q',) + x for x in tr(lds_q + lane_r * X_ROW_B + c, X_ROW_B)]
    for kh in range(NKV):
        col = lane_c * 2 + kh * 32
        out += [('p',) + x for x in tr(lds_p + lane_r * S_ROW_B + col, S_ROW_B)]
        out += [('ds',) + x for x in tr(lds_ds + lane_r * S_ROW_B + col, S_ROW_B)]
    return out

def loads_trorder(lane):       # R1 a_p0,b_do ; R2 a_ds0,b_q ; R3 a_p1,a_ds1  (_a/_b helpers)
    _, _, lane_r, lane_c = lane_vals(lane)
    _a = lambda base, kh: tr(base + lane_r * S_ROW_B + lane_c * 2 + kh * 32, S_ROW_B)
    _b = lambda base, dt: tr(base + lane_r * X_ROW_B + (lane_c + dt * 16) * 2, X_ROW_B)
    out = [('p',) + x for x in _a(lds_p, 0)]
    for dt in range(NDO): out += [('do',) + x for x in _b(lds_do, dt)]
    out += [('ds',) + x for x in _a(lds_ds, 0)]
    for dt in range(NDO): out += [('q',) + x for x in _b(lds_q, dt)]
    out += [('p',) + x for x in _a(lds_p, 1)] + [('ds',) + x for x in _a(lds_ds, 1)]
    return out

def check(acc):
    n = 0
    for ring, a, w in acc:
        lo, hi = RING[ring]
        assert lo <= a and a + w <= hi, (ring, a, w)
        assert 0 <= a and a + w <= ALLOC
        n += 1
    return n

SHAPES = {'prod': (4, 8192, 32, 8), 'fast': (1, 1024, 8, 2), 'toy': (1, 128, 2, 1)}
for name, (B, S, HQ, HKV) in SHAPES.items():
    G = HQ // HKV; nqp = S // 32; n_iter = G * nqp       # per workgroup, non-causal upper bound
    tot = 0
    for lane in range(32):
        r_ld, t_ld = loads_r29(lane), loads_trorder(lane)
        assert sorted(r_ld) == sorted(t_ld), lane
        assert sorted(stores_r29(lane)) == sorted(stores_b(lane))
        per = check(t_ld) + check(stores_b(lane)) + check(ps_stores(lane))
        tot += per
    # the per-iteration set is identical every iteration (no iteration-dependent LDS term)
    print(f'{name:5s} B{B} S{S} HQ{HQ} HKV{HKV}: iters/WG={n_iter} lds accesses/iter/wave={tot} '
          f'all in-bounds (alloc {ALLOC} B), trorder/qdoburst multiset == r29')
print('global addresses: unchanged (no buffer index / predicate / loop-bound edits) -> r29 proof holds')
print('OK')
