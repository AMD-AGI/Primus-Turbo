"""Optimize r1: what did the new wgrad cfg race actually adopt, and at what ratios?

Calls the production entry point once per scored wgrad shape (which runs the race) and prints the
split_k gate, the adopted (gm, xcd, gn) and the per-point cand/base ratios the race measured.
"""

import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.utils.gemm_helper import _robust_ab_ratio, _robust_time

DEV = "cuda"
G = 32
PER = 4096
H = 2944
F8 = torch.float8_e4m3fn


def main():
    ncu = torch.cuda.get_device_properties(DEV).multi_processor_count
    for OUT_N in (5760, 2944):
        MTOT = G * PER
        offs = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
        offs[1:] = torch.full((G,), PER, dtype=torch.int64, device=DEV).cumsum(0)
        lhs = (torch.randn(H, MTOT, device=DEV) * 0.5).to(F8)
        rhs = (torch.randn(OUT_N, MTOT, device=DEV) * 0.5).to(F8)
        ls = torch.full((H, MTOT // 32), 127, dtype=torch.uint8, device=DEV)
        rs = torch.full((OUT_N, MTOT // 32), 127, dtype=torch.uint8, device=DEV)
        pays = MK._wgrad_split_pays(H, OUT_N, G, 256, 256, ncu)
        MK.grouped_gemm_mxfp8_variable_k_flydsl_kernel(lhs, ls, rhs, rs, offs, H, OUT_N, G)
        torch.cuda.synchronize()
        key = [k for k in MK._GWG_CFG_CACHE if k[1] == OUT_N]
        print(f"OUT_N={OUT_N} split_k={pays} adopted={MK._GWG_CFG_CACHE[key[0]]}", flush=True)

        # ratios the race saw, re-measured on the same canonical points
        args = (None, None, torch.empty((G, H, OUT_N), dtype=torch.bfloat16, device=DEV))
        ws = MK._wgrad_split_ws(H, OUT_N, G, torch.device(DEV), torch.bfloat16, BLOCK_M=256, BLOCK_N=256)
        args = args + (None,) * 5 + (ws,) + (None,) * 5 + (torch.cuda.current_stream(),)
        for pm, skew in MK._GWG_PM_CANON:
            targs, _ = MK._canon_wgrad_targs(args, H, OUT_N, G, 4, pm, skew)
            base = MK._get_wgrad_launch(
                H, OUT_N, G, 256, 256, *MK._GWG_DEFAULT_CFG, 0, 0, False, 4, True, pays
            )
            _robust_time(base, targs, warmup=40, reps=1, iters=20)
            out = []
            for cfg in MK._GWG_CANDS:
                cand = MK._get_wgrad_launch(H, OUT_N, G, 256, 256, *cfg, 0, 0, False, 4, True, pays)
                out.append(f"{cfg}={_robust_ab_ratio(base, cand, targs, warmup=10, reps=3, iters=20):.4f}")
            print(f"  pm={pm} skew={int(skew)} cand/base: {' '.join(out)}", flush=True)
        del lhs, rhs, ls, rs
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
