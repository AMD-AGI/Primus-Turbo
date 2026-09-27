"""CPU fp32 emulation of one query row's online softmax: base (defer 8.0, peer-sum per tile) vs
prototype (branch-free, per-lane partial d, one peer at the end), incl. sink seed and big-logit rows."""
import numpy as np
rng = np.random.default_rng(0)
L2E = np.float32(1.4426950408889634)
def run(s, v, defer, lane, sink=None, nb=64):
    f = np.float32
    m = f(sink) if sink is not None else f(-1e30)
    d = np.array([f(1.0), f(0.0)] if (sink is not None and lane) else [f(1.0 if sink is not None else 0.0)] * 2, np.float32)
    o = np.zeros(v.shape[1], np.float32)
    for t in range(0, len(s), nb):
        st, vt = s[t:t+nb], v[t:t+nb]
        halves = [st[:nb//2], st[nb//2:]]  # lane l / l^16 halves (layout detail irrelevant to the algebra)
        rmax = f(max(h.max() for h in halves))
        mfull = max(m, rmax)
        mnew = mfull if (not defer or rmax - m > 8.0) else m
        corr = np.exp2((m - mnew) * L2E, dtype=np.float32)
        p = np.exp2(st * L2E - mnew * L2E, dtype=np.float32)
        ps = [p[:nb//2].sum(dtype=np.float32), p[nb//2:].sum(dtype=np.float32)]
        if lane:
            d = corr * d + np.array(ps, np.float32)
        else:
            tot = f(ps[0] + ps[1]); d = np.array([corr * d[0] + tot] * 2, np.float32)
        o = corr * o + p.astype(np.float32) @ vt
        m = mnew
    dfin = f(d[0] + d[1]) if lane else d[0]
    return o / dfin, m + np.log(dfin)
worst = 0
for trial in range(200):
    n = 64 * rng.integers(1, 40)
    scale = [1, 5, 30, 80][trial % 4]            # large-logit rows force frequent rescales
    s = (rng.standard_normal(n) * scale).astype(np.float32)
    if trial % 7 == 0: s = np.sort(s)            # monotone-increasing max: worst case for defer
    v = rng.standard_normal((n, 16)).astype(np.float32)
    sink = float(rng.standard_normal()) if trial % 3 == 0 else None
    ref_w = np.exp(np.float64(s) - max(s.max(), sink if sink is not None else -np.inf))
    den = ref_w.sum() + (np.exp(sink - max(s.max(), sink)) if sink is not None else 0)
    ref = (ref_w @ v) / den
    for defer, lane in ((True, False), (False, True), (True, True)):
        o, lse = run(s, v, defer, lane, sink)
        err = np.abs(o - ref).max() / (np.abs(ref).max() + 1e-30)
        worst = max(worst, err)
print("max rel err vs fp64 reference over 200 rows x 3 variants:", worst)
assert worst < 1e-5
print("OK")
