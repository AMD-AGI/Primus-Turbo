"""CPU part (no card): fp64 reference on the sampled rows saved by card_run.py, SQNR per arm.

usage: ref_eval.py TAG   (TAG = data/<shape>_<causal|full>)
Writes data/<TAG>/eval.json (card summary merged with dB numbers).
"""
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ARMS = ["r4", "r6", "r6_as", "r6_nvs"]
torch.set_num_threads(32)


def sqnr(ref, got):
    ref, got = ref.double(), got.double()
    fin = torch.isfinite(got).all()
    if not fin:
        return float("-inf")
    num = ref.pow(2).mean()
    den = (ref - got).pow(2).mean()
    if den == 0:
        return float("inf")
    return float(10 * torch.log10(num / den))


E7, TH = math.exp(7.0), 8.0


def trigger_sim(sc):
    """Row-level estimate of the r6 trigger on tiles >= 1 (tile 0 always fires from the -1e30 seed).
    m follows r4's row-level rule (rescale when tile max - m > 8); the kernel's ballot couples the rows
    of a wave, so the real per-wave rate is >= this per-row rate. Returns (fired, visited)."""
    nr, skv = sc.shape
    T = (skv + 63) // 64
    pad = torch.full((nr, T * 64), float("-inf"), dtype=sc.dtype); pad[:, :skv] = sc
    st = pad.view(nr, T, 4, 2, 8)          # tile, kvt(16), half, 8
    m = st[:, 0].amax(dim=(1, 2, 3))
    fired = visited = 0
    for t in range(1, T):
        x = st[:, t]
        vis = torch.isfinite(x).any(dim=(1, 2, 3)) & torch.isfinite(m)
        half = torch.exp(x - m.view(nr, 1, 1, 1)).sum(dim=(1, 3))        # [nr, 2]
        trig = (half > E7).any(1) & vis
        fired += int(trig.sum()); visited += int(vis.sum())
        tmax = x.amax(dim=(1, 2, 3))
        m = torch.where(vis & (tmax - m > TH), tmax, m)
    return fired, visited


def reference(s):
    q, k, v = s["q"].double(), s["k"].double(), s["v"].double()  # q [nb, nr, nh, d], k/v [nb, skv, nhk, d]
    nb, nr, nh, d = q.shape
    scale = 1.0 / math.sqrt(d)
    shift = s["skv"] - s["sq"]
    hk_index = {h: s["hk"].index(h // s["gq"]) for h in s["hs"]}
    o = torch.empty(nb, nr, nh, v.shape[-1], dtype=torch.float64)
    trig = [0, 0]
    lse = torch.empty(nb, nh, nr, dtype=torch.float64)
    rows = torch.tensor(s["rows"]).unsqueeze(1)
    kj = torch.arange(s["skv"]).unsqueeze(0)
    for bi in range(nb):
        for hi, h in enumerate(s["hs"]):
            kb, vb = k[bi, :, hk_index[h]], v[bi, :, hk_index[h]]
            sc = (q[bi, :, hi] @ kb.T) * scale
            if s["causal"]:
                sc = sc.masked_fill(kj > rows + shift, float("-inf"))
            f_, v_ = trigger_sim(sc); trig[0] += f_; trig[1] += v_
            l = torch.logsumexp(sc, -1)
            o[bi, :, hi] = torch.exp(sc - l.unsqueeze(1)) @ vb
            lse[bi, hi] = l
    return o, lse, trig


tag = sys.argv[1]
dd = HERE / "data" / tag
card = {r["kind"]: r for r in json.loads((dd / "card_summary.json").read_text())}
out = []
for kind, rec in card.items():
    s = torch.load(dd / f"{kind}.pt")
    ro, rl, trig = reference(s)
    rec["trig_fired"], rec["trig_visited"] = trig
    rec["ref_lse_finite"] = bool(torch.isfinite(rl).all())
    for a in ARMS:
        o, l = s[f"o_{a}"].double(), s[f"lse_{a}"].double()
        rec[f"{a}_db_o"] = sqnr(ro, o)
        rec[f"{a}_db_lse"] = sqnr(rl, l) if rec["ref_lse_finite"] else float("nan")
        fin = torch.isfinite(l) & torch.isfinite(rl)
        rec[f"{a}_maxerr_lse"] = float((l - rl).abs()[fin].max()) if fin.any() else float("nan")
        rec[f"{a}_nonfinite_sampled"] = int((~torch.isfinite(o)).sum() + (~torch.isfinite(l)).sum())
    out.append(rec)
    print(f"{tag:16s} {kind:16s} o dB r4 {rec['r4_db_o']:7.2f} r6 {rec['r6_db_o']:7.2f} as {rec['r6_as_db_o']:7.2f}"
          f" nvs {rec['r6_nvs_db_o']:7.2f} | lse dB r4 {rec['r4_db_lse']:7.2f} r6 {rec['r6_db_lse']:7.2f}"
          f" | bit r6=as {rec['bit_r6_r6_as']} r6=r4 {rec['bit_r6_r4']} rows r6!=nvs {rec['rowsdiff_r6_r6_nvs']} trig {trig[0]}/{trig[1]}",
          flush=True)
(dd / "eval.json").write_text(json.dumps(out, indent=1))
