"""Card part: ONE (shape, causal) per process. Runs every arm on every adversarial kind, compares arms
on the card (bitwise / max diff / finiteness), and saves the sampled rows (+ the inputs they need) for
the CPU fp64 reference (ref_eval.py).

usage: card_run.py SHAPE CAUSAL(0|1) [kinds,comma,separated]
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

from adv_inputs import INFORMATIONAL, KINDS, SHAPES, make, sample_spec  # noqa: E402

ARMS = ["r4", "r6", "r6_as", "r6_nvs"]


def load_impl(d):
    d = Path(d).resolve()
    name = "vr6_impl_" + str(abs(hash(str(d))))
    spec = importlib.util.spec_from_file_location(name, d / "impl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.attn_fwd


shape, causal = sys.argv[1], bool(int(sys.argv[2]))
kinds = sys.argv[3].split(",") if len(sys.argv) > 3 else KINDS
fns = {a: load_impl(HERE / "arms" / a) for a in ARMS}
b, sq, skv, hq, hkv, d = SHAPES[shape]
bs, hs, rows = sample_spec(shape)
tag = f"{shape}_{'causal' if causal else 'full'}"
outdir = HERE / "data" / tag
outdir.mkdir(parents=True, exist_ok=True)
summary = []


def biteq(x, y):
    return bool(torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                            y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32)))


def rows_differ(oa, la, ob, lb):
    # per (b, h, q) row: any bit difference in o or lse
    do = (oa.view(torch.int16) != ob.view(torch.int16)).any(-1).permute(0, 2, 1)  # b,h,q
    dl = la.view(torch.int32) != lb.view(torch.int32)
    return int((do | dl).sum())


for kind in kinds:
    t0 = time.time()
    q, k, v = make(shape, kind)
    qc, kc, vc = q.cuda(), k.cuda(), v.cuda()
    outs = {}
    for a in ARMS:
        o, l = fns[a](qc, kc, vc, causal=causal)
        torch.cuda.synchronize()
        outs[a] = (o.clone(), l.clone())
        del o, l
    torch.cuda.synchronize()
    rec = {"shape": shape, "causal": causal, "kind": kind, "informational": kind in INFORMATIONAL}
    for a in ARMS:
        o, l = outs[a]
        rec[f"{a}_finite_o"] = int(torch.isfinite(o.float()).sum())
        rec[f"{a}_finite_lse"] = int(torch.isfinite(l).sum())
    rec["numel_o"], rec["numel_lse"] = outs["r4"][0].numel(), outs["r4"][1].numel()
    for x, y in (("r6", "r4"), ("r6", "r6_as"), ("r6", "r6_nvs")):
        (ox, lx), (oy, ly) = outs[x], outs[y]
        rec[f"bit_{x}_{y}"] = biteq(ox, oy) and biteq(lx, ly)
        rec[f"maxdo_{x}_{y}"] = float((ox.float() - oy.float()).abs().nan_to_num(float("inf")).max())
        fin = torch.isfinite(lx) & torch.isfinite(ly)
        rec[f"maxdlse_{x}_{y}"] = float((lx[fin] - ly[fin]).abs().max()) if fin.any() else float("nan")
        rec[f"rowsdiff_{x}_{y}"] = rows_differ(ox, lx, oy, ly)
    rec["rows_total"] = b * hq * sq
    # sampled rows for the CPU reference
    hk = sorted({h // (hq // hkv) for h in hs})
    save = {"q": q[bs][:, rows][:, :, hs].clone(), "k": k[bs][:, :, hk].clone(), "v": v[bs][:, :, hk].clone(),
            "bs": bs, "hs": hs, "hk": hk, "rows": rows, "sq": sq, "skv": skv, "gq": hq // hkv, "causal": causal}
    ri = torch.tensor(rows, device="cuda")
    for a in ARMS:
        o, l = outs[a]
        save[f"o_{a}"] = o[bs][:, ri][:, :, hs].cpu()
        save[f"lse_{a}"] = l[bs][:, hs][:, :, ri].cpu()
    torch.save(save, outdir / f"{kind}.pt")
    rec["sec"] = round(time.time() - t0, 2)
    summary.append(rec)
    print("CARD", json.dumps(rec), flush=True)
    del outs, qc, kc, vc
    torch.cuda.empty_cache()

(outdir / "card_summary.json").write_text(json.dumps(summary, indent=1))
print("CARD_DONE", tag, flush=True)
