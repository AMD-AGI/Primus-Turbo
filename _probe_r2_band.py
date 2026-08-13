"""Optimize r1: is the adopted (gm=4, gn=2) band the best of its family?

Times extra (group_m, group_n) points against the adopted cfg with interleaved A/B on the scored
wgrad gate_up shape, at the production tokens/group AND at both race points, so a candidate is only
worth adding to _GWG_CANDS if it wins everywhere. Ratio < 1 = candidate faster.
"""

import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.utils.gemm_helper import _robust_ab_ratio, _robust_time

DEV = "cuda"
G = 32
H = 2944
REF = (4, 1, 2)
CANDS = ((8, 1, 2), (8, 1, 4), (4, 1, 4), (6, 1, 2), (3, 1, 2), (4, 1, 6))


def main():
    OUT_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5760
    ncu = torch.cuda.get_device_properties(DEV).multi_processor_count
    pays = MK._wgrad_split_pays(H, OUT_N, G, 256, 256, ncu)
    C = torch.empty((G, H, OUT_N), dtype=torch.bfloat16, device=DEV)
    ws = MK._wgrad_split_ws(H, OUT_N, G, torch.device(DEV), torch.bfloat16, BLOCK_M=256, BLOCK_N=256)
    args = (None, None, C) + (None,) * 5 + (ws,) + (None,) * 5 + (torch.cuda.current_stream(),)

    def launch_of(cfg):
        return MK._get_wgrad_launch(H, OUT_N, G, 256, 256, *cfg, 0, 0, False, 4, True, pays)

    print(f"OUT_N={OUT_N} split_k={pays} ref={REF}", flush=True)
    for pm, skew in ((4096, False), (2048, False), (2048, True)):
        targs, _ = MK._canon_wgrad_targs(args, H, OUT_N, G, 4, pm, skew)
        base = launch_of(REF)
        t = _robust_time(base, targs, warmup=60, reps=3, iters=20)
        out = []
        for cfg in CANDS:
            try:
                r = _robust_ab_ratio(base, launch_of(cfg), targs, warmup=10, reps=3, iters=20)
                out.append(f"{cfg[0]}/{cfg[2]}={r:.4f}")
            except Exception as e:  # noqa: BLE001
                out.append(f"{cfg[0]}/{cfg[2]}=ERR({type(e).__name__})")
        print(f"  pm={pm} skew={int(skew)} ref={t*1e3:.1f}us  gm/gn: {' '.join(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
