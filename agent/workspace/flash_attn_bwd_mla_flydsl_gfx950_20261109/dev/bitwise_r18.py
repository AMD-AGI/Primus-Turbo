"""MLA backward of the package at MLA_PKG_ROOT, saved per shape: python bitwise_r18.py dump <out.pt> [reps]
Compare two dumps: python bitwise_r18.py cmp <a.pt> <b.pt> (dK/dV bitwise, dQ SNR of b vs a and of a's
repeats against each other)."""
import os
import sys

import torch

SHAPES = ((1, 1024), (1, 2112), (1, 4096), (1, 4224), (2, 4096), (1, 8192))


def snr(a, b):
    a, b = a.float(), b.float()
    return (10 * torch.log10(b.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-30))).item()


def dump(out, reps):
    sys.path.insert(0, os.environ.get("MLA_PKG_ROOT", "/perf_apps/xiaoming/megaattention_overlap_bench"))
    from flydsl_mla import mla_fwd_sbhd
    from flydsl_mla.ops_bwd import mla_bwd_sbhd

    res = {}
    for b, s in SHAPES:
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k = (torch.randn((s, b, 128, 192), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
        v = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
        do = torch.randn((s, b, 128, 128), device="cuda", dtype=torch.bfloat16, generator=g)
        o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
        runs = [mla_bwd_sbhd(do, q, k, v, o, lse) for _ in range(reps)]
        torch.cuda.synchronize()
        res[(b, s)] = [[t.cpu() for t in r] for r in runs]
    torch.save(res, out)


def cmp(pa, pb):
    a, b = torch.load(pa), torch.load(pb)
    for sh in SHAPES:
        ra, rb = a[sh], b[sh]
        dq0, dk0, dv0 = ra[0]
        rep = [snr(r[0], dq0) for r in ra[1:]]
        print(
            "B%d S%d dK bitwise %s dV bitwise %s | dQ snr b vs a %s dB | a repeats %s dB | b repeats dK/dV bitwise %s"
            % (
                sh + (
                    all(torch.equal(r[1], dk0) for r in rb),
                    all(torch.equal(r[2], dv0) for r in rb),
                    " ".join("%.2f" % snr(r[0], dq0) for r in rb),
                    " ".join("%.2f" % x for x in rep),
                    all(torch.equal(r[1], rb[0][1]) and torch.equal(r[2], rb[0][2]) for r in rb),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        cmp(sys.argv[2], sys.argv[3])
