"""Round-3 correctness probe for the wgrad pair-major RHS feed.

The scored bench fills every E8M0 scale with 127 (=1.0), so a scale-to-row permutation
bug there is invisible: it stays finite and bitwise deterministic. This probe uses RANDOM
per-1x32 scales, so the paired preshuffle and the paired B feed have to agree row by row,
and checks small shapes that exercise every boundary variant (half-M, half-N, both, and a
non-pair-safe OUT_N that must keep the masked scalar store).

usage: _probe_r3_num.py
"""

import sys

import torch

from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_variable_k_flydsl_kernel as mx_wgrad,
)

DEV = "cuda"
F8 = torch.float8_e4m3fn
# (OUT_M, OUT_N, G, per-expert M): OUT_N 384/896 -> residue 128 (pair-safe, half-N),
# 512 -> aligned, 320 -> residue 64 (NOT pair-safe: the masked scalar store must stay).
CASES = ((384, 384, 3, 512), (384, 512, 2, 640), (256, 896, 2, 512), (384, 320, 2, 512))


def deq(x, sc):
    """fp8 [R, M] with E8M0 [R, M//32] -> f32, one scale per 1x32 block along M."""
    e = sc.to(torch.int32) - 127
    return x.float() * torch.exp2(e.float()).repeat_interleave(32, dim=1)


def ref(l, lsc, r, rsc, offs, G):
    a, b = deq(l, lsc), deq(r, rsc)
    return torch.stack([a[:, offs[g] : offs[g + 1]] @ b[:, offs[g] : offs[g + 1]].T for g in range(G)])


def snr(out, want):
    e = (out.float() - want).pow(2).mean()
    return float("inf") if e == 0 else 10 * torch.log10(want.pow(2).mean() / e).item()


def main():
    torch.manual_seed(0)
    worst, bad = 1e9, []
    for OUT_M, OUT_N, G, per in CASES:
        mtot = G * per
        l = (torch.randn(OUT_M, mtot, device=DEV) * 0.5).to(F8)
        r = (torch.randn(OUT_N, mtot, device=DEV) * 0.5).to(F8)
        lsc = torch.randint(122, 133, (OUT_M, mtot // 32), dtype=torch.uint8, device=DEV)
        rsc = torch.randint(122, 133, (OUT_N, mtot // 32), dtype=torch.uint8, device=DEV)
        offs = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
        offs[1:] = torch.full((G,), per, dtype=torch.int64, device=DEV).cumsum(0)
        o1 = mx_wgrad(l, lsc, r, rsc, offs, OUT_M, OUT_N, G, torch.bfloat16, -1, 4, True).clone()
        o2 = mx_wgrad(l, lsc, r, rsc, offs, OUT_M, OUT_N, G, torch.bfloat16, -1, 4, False)
        det = torch.equal(o1, o2)
        want = ref(l, lsc, r, rsc, offs.tolist(), G)
        s = snr(o1, want)
        worst = min(worst, s)
        print(f"  M={OUT_M} N={OUT_N} G={G} per={per}: SNR={s:6.1f} dB  det={det}")
        if s < 40.0 or not det:
            bad.append((OUT_M, OUT_N, G, per, s, det))
    print("NUM:", "OK" if not bad else f"FAIL {bad}", f"worst={worst:.1f} dB")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
