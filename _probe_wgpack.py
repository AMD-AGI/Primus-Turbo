"""Widened-shape correctness probe for the grouped mxfp8 wgrad scale packing + boundary tiles.

The campaign bench gate pins the wgrad to pack=1 and only ever feeds g=2 / uniform M_g /
512-aligned group starts, so it cannot see a packing that assumes an aligned per-group
contraction start. This probe runs the variable-K wgrad at pack=1 and pack=4 against an
fp32 dequant reference on non-uniform, 128-aligned-but-not-512, single-odd-start, empty
and ragged group layouts, and checks that (a) both SNRs pass, (b) the two packings agree
bytewise, and (c) preshuffle=False (the timed path) matches the full call bytewise.
"""
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
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


# Every M_g is a multiple of BLOCK_K=128 (the kernel's documented contract); what varies is
# the ALIGNMENT of the resulting group starts, which is what pack>1 used to assume.
CASES = {
    "uniform-512al": [1024, 1024, 1024, 1024],
    "nonuniform-512al": [512, 2048, 512, 1024],
    "128al-not512": [128, 384, 640, 896],
    "single-odd-start": [128, 1024, 1024, 1024],
    "odd-start-mid": [1024, 128, 1024, 512],
    "all-odd-starts": [384, 384, 384, 384],
    "empty-first": [0, 1024, 256],
    "empty-mid": [512, 0, 256, 128],
    "empty-last": [1024, 256, 0],
    "single-group": [1536],
    "g32-skew": [4096] + [128] * 31,
    "g32-ragged": [128 * (1 + (i * 7) % 11) for i in range(32)],
}

# (OUT_M, OUT_N): all four boundary gate combinations. x.5*256 reproduces the gpt-oss
# 2944/5760 flavour, where the last M-block / N-block is half padding and the kernel runs
# a reduced-quadrant body -- a wrongly dropped quadrant shows up as an SNR crash here.
SHAPES = [(1024, 256), (1152, 384), (1152, 512), (1024, 384)]


def run(name, mg, OUT_M, OUT_N):
    o = offs(mg)
    m = int(o[-1].item())
    g = len(mg)
    lm, rm = f8(OUT_M, m), f8(OUT_N, m)
    l_s, r_s = scv(OUT_M, m // 32), scv(OUT_N, m // 32)
    ld, rd = deq(lm, l_s, 1), deq(rm, r_s, 1)
    ref = torch.stack([ld[:, int(o[i]) : int(o[i + 1])] @ rd[:, int(o[i]) : int(o[i + 1])].t()
                       for i in range(g)])

    def call(pack, preshuffle=True):
        return fly_vark_mx(lm, l_s, rm, r_s, o, OUT_M, OUT_N, g, out_dtype=torch.bfloat16,
                           num_cu=-1, pack=pack, preshuffle=preshuffle)

    got1, got4 = call(1), call(4)
    s1, s4 = snr(ref, got1), snr(ref, got4)
    same = bool(torch.equal(got1.view(torch.uint8), got4.view(torch.uint8)))
    ng = bool(torch.equal(got4.view(torch.uint8), call(4, preshuffle=False).view(torch.uint8)))
    ok = s1 >= THR and s4 >= THR and same and ng
    print(f"{name:18s} OUT={OUT_M:5d}x{OUT_N:4d} g={g:2d} m={m:6d} | pack1 {s1:6.2f}dB | "
          f"pack4 {s4:6.2f}dB | eq={same} ng={ng}" + ("  OK" if ok else "  ***FAIL***"))
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    allok = True
    for OUT_M, OUT_N in SHAPES:
        for name, mg in CASES.items():
            try:
                allok &= run(name, mg, OUT_M, OUT_N)
            except Exception as e:  # unsupported shape -> report, don't mask
                print(f"{name:18s} OUT={OUT_M:5d}x{OUT_N:4d} EXC {type(e).__name__}: {e}")
                allok = False
    print("ALL_OK" if allok else "SOME_FAILED")
    sys.exit(0 if allok else 1)
