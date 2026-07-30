"""Widened-shape correctness probe for the grouped mxfp8 group->tile decode.

The campaign bench gate only covers g=2 / uniform M_g / 512-aligned group starts, so it
cannot see a broken group-find on non-uniform, non-64-aligned or empty groups. This probe
runs fwd (NT) and wgrad (variable-K TN) against an fp32 dequant reference on those shapes.
"""
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_flydsl_kernel as fly_mx,
    grouped_gemm_mxfp8_variable_k_flydsl_kernel as fly_vark_mx,
)
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
THR = 25.0


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


def scv(*s):
    return torch.randint(122, 132, s, dtype=torch.uint8, device=DEV)


def deq(x, s, ax):
    return x.float() * torch.pow(2.0, s.float() - 127.0).repeat_interleave(32, dim=ax)


def snr(ref, x):
    ref, x = ref.float(), x.float()
    p = (ref * ref).mean()
    n = ((ref - x) ** 2).mean()
    return float("inf") if n == 0 else 10 * torch.log10(p / n).item()


def offs(mg):
    o = [0]
    for m in mg:
        o.append(o[-1] + m)
    return torch.tensor(o, dtype=torch.int64, device=DEV)


CASES = {
    "uniform-512al": [1024, 1024, 1024, 1024],
    "nonuniform-512al": [512, 2048, 512, 1024],
    "nonuniform-64al": [64, 512, 128, 1024, 320, 192],
    "nonuniform-32al-not64": [96, 160, 224, 288],
    "empty-group": [512, 0, 256, 128],
    "empty-first": [0, 1024, 256],
    "empty-last": [1024, 256, 0],
    "single-group": [1536],
    "skew-32g": [4096] + [128] * 31,
    "skew-32g-tail-empty": [4096, 2048, 1024] + [128] * 26 + [0, 0, 0],
    "ragged-32g": [128 * (1 + (i * 7) % 11) for i in range(32)],
}


def run(name, mg, A=1024, B=256, num_cu=-1):
    o = offs(mg)
    m = int(o[-1].item())
    g = len(mg)
    out = []
    # fwd (NT): C[m, B] = A[m, A] @ W[g, B, A]^T. Contract: M_g is a multiple of 32.
    a, w = f8(m, A), f8(g, B, A)
    a_s, w_s = scv(m, A // 32), scv(g, B, A // 32)
    ref = torch.cat(
        [deq(a[int(o[i]) : int(o[i + 1])], a_s[int(o[i]) : int(o[i + 1])], 1) @ deq(w[i], w_s[i], 1).t()
         for i in range(g)]
    )
    got = fly_mx(a, a_s, w, w_s, o, B, A, out_dtype=torch.bfloat16, num_cu=num_cu)
    out.append(("fwd", snr(ref, got)))
    # wgrad (variable-K TN): C[g, A, B] = L[A, m] @ R[B, m]^T per group. Contract: the
    # per-group contraction is padded to a multiple of BLOCK_K=128; skip the shapes that
    # violate it (they are out of the kernel's documented domain).
    if all(x % 128 == 0 for x in mg):
        lm, rm = f8(A, m), f8(B, m)
        l_s, r_s = scv(A, m // 32), scv(B, m // 32)
        ld, rd = deq(lm, l_s, 1), deq(rm, r_s, 1)
        ref = torch.stack(
            [ld[:, int(o[i]) : int(o[i + 1])] @ rd[:, int(o[i]) : int(o[i + 1])].t() for i in range(g)]
        )
        got = fly_vark_mx(lm, l_s, rm, r_s, o, A, B, g, out_dtype=torch.bfloat16, num_cu=-1)
        out.append(("wgrad", snr(ref, got)))
    else:
        out.append(("wgrad", float("nan")))
    ok = all(s >= THR for k, s in out if s == s)
    print(f"{name:24s} g={g:2d} m={m:6d} | " + " | ".join(f"{k} {s:6.2f}dB" for k, s in out)
          + ("  OK" if ok else "  ***FAIL***"))
    return ok


# (N, num_cu) sweep. N=256 is BLOCK_N-aligned (full-quadrant body); N=384 is the gpt-oss
# x.5*256 flavour whose last N-block is half padding, so it takes the reduced-quadrant body.
# Both have an N residue of 0 or exactly BLOCK_N/2, so the store drops its column mask
# (_COL_SAFE); N=320 (residue 64, reduced body) and N=448 (residue 192, full body) are the
# residues that keep the mask, and their last N-block overhangs N -- an over-wide store
# lands in the next output row and shows up as an SNR break.
# num_cu>0 selects the persistent grid, which walks the tile space in an scf.for.
VARIANTS = [(256, -1), (384, -1), (384, 256), (320, -1), (448, -1)]


if __name__ == "__main__":
    torch.manual_seed(0)
    allok = True
    for N, ncu in VARIANTS:
        print(f"--- N={N} num_cu={ncu}")
        for name, mg in CASES.items():
            try:
                allok &= run(name, mg, B=N, num_cu=ncu)
            except Exception as e:  # unsupported shape -> report, don't mask
                print(f"{name:24s} EXC {type(e).__name__}: {e}")
                allok = False
    print("ALL_OK" if allok else "SOME_FAILED")
    sys.exit(0 if allok else 1)
