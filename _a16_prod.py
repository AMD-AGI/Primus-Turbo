#!/usr/bin/env python3
"""a16 on the PRODUCTION path (`_DQ_A16`), correctness + wall, one arm per process.

    MODE=ref  _a16_prod.py <B> <Hq> <Hkv> <S> <D> [reps]   -> /tmp/prod_ref.pt
    MODE=a16  _a16_prod.py <B> <Hq> <Hkv> <S> <D> [reps]   -> SNR against it

Two processes because the impl caches the built launcher, and warm because the first call of
a compiled launcher dispatches its grid twice (which an accumulated image, unlike a store,
notices).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import primus_turbo.flydsl.attention.flash_attn_bwd as M

B, Hq, Hkv, S, D = (int(x) for x in sys.argv[1:6])
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 20
MODE = os.environ.get("MODE", "a16")
if MODE == "ref":
    M._DQ_A16 = False
for _k, _v in (("BKV", "_A16_BLOCK_KV"), ("BQ", "_A16_BLOCK_Q"), ("QSP", "_A16_Q_SPLIT"), ("QSP8", "_A16_Q_SPLIT_G8")):
    if os.environ.get(_k):
        setattr(M, _v, int(os.environ[_k]))

if os.environ.get("LEG") == "nounp":  # body + fill only: prices the un-permute pass
    M._unpermute_dq_a16 = lambda *a, **kw: None
if os.environ.get("KW"):  # KW=g3_st_n=1,g2d=3 -- builder kwargs on the a16 body
    _ov = {a.split("=")[0]: int(a.split("=")[1]) for a in os.environ["KW"].split(",")}
    _bld = M.build_flash_attn_bwd_dkdv_module
    M.build_flash_attn_bwd_dkdv_module = lambda **kw: _bld(**{**kw, **_ov})

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_backward_impl as fb,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (  # noqa: E402
    flash_attn_sbhd_flydsl_forward_impl as ff,
)

g = torch.Generator().manual_seed(7)
mk = lambda h: torch.randn(S, B, h, D, generator=g, dtype=torch.bfloat16).cuda()
q, k, v = mk(Hq), mk(Hkv), mk(Hkv)
do = torch.randn(S, B, Hq, D, generator=g, dtype=torch.bfloat16).cuda()
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lse_h = lse.view(B, S, Hq).permute(0, 2, 1)


def _run():
    return fb(do, q, k, v, o, lse_h, causal=True, window_size=(-1, -1))


def _snr(ref, got):
    ref, got = ref.float(), got.float()
    n = (ref - got).pow(2).mean()
    return float("inf") if n == 0 else 10 * torch.log10(ref.pow(2).mean() / n).item()


_run()  # cold: never read, never timed
out = [x.clone() for x in _run()[:3]]
det = all(torch.equal(a, b) for a, b in zip(out, _run()[:3]))
torch.cuda.synchronize()
best = 1e9
for _ in range(REPS):
    t0 = time.perf_counter()
    _run()
    torch.cuda.synchronize()
    best = min(best, (time.perf_counter() - t0) * 1e3)
print("  %-4s %.4f ms  repeatable=%s" % (MODE, best, det))
if MODE == "ref":
    torch.save({"o": [x.cpu() for x in out], "t": best}, "/tmp/prod_ref.pt")
elif MODE == "a16" and os.path.exists("/tmp/prod_ref.pt"):
    r = torch.load("/tmp/prod_ref.pt")
    print(
        "  SNR dq %.2f  dk %.2f  dv %.2f dB   ref %.4f  a16 %.4f  delta %+.2f%%"
        % (
            *[_snr(a.cuda(), b) for a, b in zip(r["o"], out)],
            r["t"],
            best,
            100 * (best / r["t"] - 1),
        )
    )
