#!/usr/bin/env python3
"""L21 adversarial large-logit correctness suite (hint f16/h16). ONE shape per process.

    adv.py <shape> <out.json> <arm> [<arm> ...]      (arm = dir name under arms/, or 'champ')

For each case: bf16 q/k/v built so the scaled logits s = scale * q.k have a planted structure
(row max far above/below any fixed max, drifting across KV tiles, ranges past the fp32 exp2
overflow/underflow walls, one-key rows, huge outliers), plus plain randn. Every arm (champion
included) is compared against an fp64 reference computed from the SAME bf16 inputs, and
against the champion; each arm runs twice for bitwise determinism.
"""
import json, math, os, sys, time
from pathlib import Path

OP = Path("/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/"
          "gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op")
CHAMP = OP / "current"
L = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OP)); sys.path.insert(0, str(OP / "ut"))
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"
import torch  # noqa: E402
from common import load_impl  # noqa: E402

SHAPES = {  # b, sq, skv, hq, hkv, d, causal
    "toy": (1, 256, 256, 4, 1, 128, True),
    "A": (1, 2048, 2048, 8, 2, 128, True),     # square causal: row 0 sees exactly one key
    "B": (1, 1024, 2048, 8, 2, 128, True),     # bottom-right offset 1024
    "C": (1, 1024, 1024, 8, 2, 128, False),    # non-causal (other binary)
}
shape, out_path, arm_names = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
b, sq, skv, hq, hkv, D, causal = SHAPES[shape]
dev = "cuda"
S16 = 1.0 / 16.0            # power-of-two scale: folding it into bf16 Q is exact
DEF = 1.0 / math.sqrt(D)


def gen(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32)


def base(seed, qs=1.0, ks=1.0):
    r = gen(seed)
    return r(b, sq, hq, D) * qs, r(b, skv, hkv, D) * ks, r(b, skv, hkv, D)


def plant(q, k, a, bvec):
    """dim 0 carries the planted logit: s_ij = scale*(a_i * bvec_j + noise)."""
    q = q.clone(); k = k.clone()
    q[..., 0] = a if torch.is_tensor(a) else torch.full_like(q[..., 0], a)
    k[..., 0] = bvec.view(1, skv, 1).expand(b, skv, hkv) if torch.is_tensor(bvec) else bvec
    return q, k


def kpos():
    return torch.arange(skv, device=dev, dtype=torch.float32)


def cases():
    """(name, scale, q, k, v). Planted logit X (natural units) = scale * a * b."""
    C = []
    q, k, v = base(1); C.append(("randn", DEF, q, k, v))
    q, k, v = base(2, 4.0, 1.0); C.append(("randn_x4", DEF, q, k, v))        # logit std ~ 4*1
    q, k, v = base(3, 4.0, 4.0); C.append(("randn_x16", DEF, q, k, v))       # std ~ 16, range ~+-70
    q, k, v = base(4, 8.0, 8.0); C.append(("randn_x64", DEF, q, k, v))       # std ~ 64, range ~+-280
    for X in (50.0, 90.0, 200.0, 1.0e4):   # whole row far ABOVE any fixed max
        q, k, v = base(10); q, k = plant(q, k, 16.0, X); C.append((f"const_hi_{X:g}", S16, q, k, v))
    for X in (50.0, 90.0, 110.0, 200.0, 1.0e4):   # whole row far BELOW (exp2 underflow at ~-87/-103)
        q, k, v = base(11); q, k = plant(q, k, 16.0, -X); C.append((f"const_lo_{X:g}", S16, q, k, v))
    P = 640  # drift starts at KV tile 10: first tiles low, later tiles +X (row max drifts UP)
    for X in (30.0, 45.0, 60.0, 90.0, 150.0, 1.0e3):
        q, k, v = base(12); bv = torch.where(kpos() >= P, X, 0.0)
        q, k = plant(q, k, 16.0, bv); C.append((f"drift_up_{X:g}", S16, q, k, v))
    for X in (90.0, 1.0e3):            # first tiles +X, later tiles ~0 (row max drifts DOWN)
        q, k, v = base(13); bv = torch.where(kpos() < P, X, 0.0)
        q, k = plant(q, k, 16.0, bv); C.append((f"drift_down_{X:g}", S16, q, k, v))
    # per-row ramps: s_ij = scale*a_i*(j/skv)*16*R -- row i spans [0, X_i] with X_i up to R,
    # crossing 2^64 (44.4), fp32 exp overflow (88.7) and 2^128/log2e walls at different rows
    for R, sgn in ((300.0, 1.0), (300.0, -1.0), (1.0e4, 1.0)):
        q, k, v = base(14)
        a = torch.linspace(0.0, 1.0, sq, device=dev).view(1, sq, 1).expand(b, sq, hq) * 16.0
        q, k = plant(q, k, a, sgn * R * kpos() / skv)
        C.append((f"ramp_{'up' if sgn > 0 else 'down'}_{R:g}", S16, q, k, v))
    for X in (100.0, 1.0e4):           # sparse huge outliers at late keys
        q, k, v = base(15); bv = torch.zeros(skv, device=dev)
        bv[torch.tensor([700, 1100, 1500, skv - 1], device=dev) % skv] = X
        q, k = plant(q, k, 16.0, bv); C.append((f"outlier_{X:g}", S16, q, k, v))
    for X in (1.0e4, -1.0e4, 200.0, -200.0):   # key 0 extreme: row 0 of a square causal shape
        q, k, v = base(16); bv = torch.zeros(skv, device=dev); bv[0] = X  # sees ONLY key 0
        q, k = plant(q, k, 16.0, bv); C.append((f"key0_{X:g}", S16, q, k, v))
    return [(n, s, q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)) for n, s, q, k, v in C]


def reference(q, k, v, scale):
    """fp64 reference from the same bf16 inputs. o [b,sq,hq,D], lse [b,hq,sq] natural log."""
    g = hq // hkv
    o = torch.empty(b, sq, hq, D, dtype=torch.float64, device=dev)
    lse = torch.empty(b, hq, sq, dtype=torch.float64, device=dev)
    i = torch.arange(sq, device=dev).view(sq, 1); j = torch.arange(skv, device=dev).view(1, skv)
    allowed = (j <= i + (skv - sq)) if causal else torch.ones(sq, skv, dtype=torch.bool, device=dev)
    for bb in range(b):
        for h in range(hq):
            qq = q[bb, :, h].double(); kk = k[bb, :, h // g].double(); vv = v[bb, :, h // g].double()
            s = (qq @ kk.T) * scale
            s = s.masked_fill(~allowed, float("-inf"))
            l_ = torch.logsumexp(s, dim=1)
            o[bb, :, h] = torch.softmax(s, dim=1) @ vv
            lse[bb, h] = l_
    return o, lse


def sqnr(ref, got):
    ref = ref.double(); got = got.double()
    num = (ref ** 2).sum().item(); den = ((ref - got) ** 2).sum().item()
    return float("inf") if den == 0 else 10 * math.log10(max(num, 1e-300) / den)


def metrics(ro, rl, o, l):
    fo = torch.isfinite(o.float()); fl = torch.isfinite(l.float())
    m = {"o_finite": int(fo.sum()), "o_numel": o.numel(), "lse_finite": int(fl.sum()), "lse_numel": l.numel()}
    if int(fo.sum()) == o.numel() and int(fl.sum()) == l.numel():
        m["o_db"] = sqnr(ro, o)
        m["lse_db"] = sqnr(rl, l)
        d = (l.double() - rl).abs()
        m["lse_maxabs"] = float(d.max())
        m["lse_maxrel"] = float((d / (1.0 + rl.abs())).max())
        m["o_maxabs"] = float((o.double() - ro).abs().max())
    else:
        m["o_db"] = m["lse_db"] = float("-inf"); m["lse_maxabs"] = m["lse_maxrel"] = m["o_maxabs"] = float("inf")
    return m


def main():
    fns = {}
    for a in arm_names:
        p = CHAMP if a == "champ" else (L / "arms" / a)
        fns[a] = load_impl(p)
        print(f"# arm {a} -> {p}", flush=True)
    print(f"# shape {shape} {SHAPES[shape]} device {torch.cuda.get_device_properties(0).gcnArchName} "
          f"OE_PHYS_GPU={os.environ.get('OE_PHYS_GPU')}", flush=True)
    res = {"shape": shape, "dims": SHAPES[shape], "arms": arm_names, "cases": {}}
    only = os.environ.get("ADV_ONLY")
    for name, scale, q, k, v in cases():
        if only and name not in only.split(","):
            continue
        ro, rl = reference(q, k, v, scale)
        rf = bool(torch.isfinite(ro).all() and torch.isfinite(rl).all())
        row = {"scale": scale, "ref_finite": rf, "ref_lse_max": float(rl.max()), "ref_lse_min": float(rl.min()),
               "arms": {}}
        outs = {}
        for a in arm_names:
            o1, l1 = fns[a](q, k, v, softmax_scale=scale, causal=causal)
            o1, l1 = o1.clone(), l1.clone()
            o2, l2 = fns[a](q, k, v, softmax_scale=scale, causal=causal)
            torch.cuda.synchronize()
            m = metrics(ro, rl, o1, l1)
            m["deterministic"] = bool(torch.equal(o1.view(torch.int16), o2.view(torch.int16))
                                      and torch.equal(l1.view(torch.int32), l2.view(torch.int32)))
            outs[a] = (o1, l1)
            row["arms"][a] = m
        if "champ" in outs:
            oc, lc = outs["champ"]
            for a in arm_names:
                o1, l1 = outs[a]
                row["arms"][a]["bitwise_vs_champ"] = bool(torch.equal(o1.view(torch.int16), oc.view(torch.int16))
                                                           and torch.equal(l1.view(torch.int32), lc.view(torch.int32)))
        res["cases"][name] = row
        line = " | ".join(f"{a}: o {m['o_db']:6.2f} lse {m['lse_maxabs']:.2e} fin {m['o_finite'] == m['o_numel']}"
                          f"{' det' if m['deterministic'] else ' NONDET'}{' =c' if m.get('bitwise_vs_champ') else ''}"
                          for a, m in row["arms"].items())
        print(f"CASE {name:16s} lse[{row['ref_lse_min']:.3g},{row['ref_lse_max']:.3g}] {line}", flush=True)
        del outs
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps(res, indent=1))
    print("ADV_DONE", flush=True)


if __name__ == "__main__":
    main()
