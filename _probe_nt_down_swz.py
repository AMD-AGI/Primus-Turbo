"""Diagnostic: widen the NT swizzle search on the ONE weak campaign shape (down, N=2944,
K=2944) beyond the 4 autotune candidates -- including the gn>0 N-band, which was judged on
tensorwise MoE shapes in 2026-07 but never on this shape. Drift-immune interleaved A/B vs the
base cfg at both race points. Anything clearing the 1.5% adoption margin at BOTH points is a
candidate-list swap. Read-only w.r.t. production code.
"""
import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.utils.gemm_helper import _robust_ab_ratio, _robust_time

DEV = "cuda"
G, K, N, M = 32, 2944, 2944, 131072
BASE = (256, 4, 4, 0)
# (bm, gm, xcd, gn)
GRID = [
    (256, 8, 4, 0), (256, 16, 4, 0), (256, 2, 4, 0),
    (256, 4, 2, 0), (256, 8, 8, 0), (256, 16, 8, 0),
    (256, 4, 4, 2), (256, 4, 4, 4), (256, 8, 4, 4), (256, 4, 8, 4),
]

a8 = torch.randint(0, 127, (M, K), device=DEV, dtype=torch.int8)
b8 = torch.randint(0, 127, (G * N, K), device=DEV, dtype=torch.int8)
out = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
a_raw = torch.randint(120, 128, (M, K // 32), device=DEV, dtype=torch.uint8).view(torch.int32).reshape(-1)
b_raw = torch.randint(120, 128, (G * N, K // 32), device=DEV, dtype=torch.uint8).view(torch.int32).reshape(-1)
stream = torch.cuda.current_stream()
a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
go = (torch.arange(0, G + 1, dtype=torch.int64, device=DEV) * (M // G)).view(torch.int32)
args = (
    a8, b8, out, a_raw, b_raw, a_sp, b_sp, go, go, M,
    a_ngrp * 64, N, a_blocks, a_ngrp, ((M + 255) // 256 + G) * ((N + 255) // 256), stream,
)

points = [MK._canon_nt_targs(args, K, G, N, pm, skew) for pm, skew in MK._GNT_PM_CANON]
base = MK._get_nt_launch(K, G, N, *BASE, 0, 0, False, False)
base(*points[0][0])
torch.cuda.synchronize()
refs = [(t, o.detach().clone().float()) for t, o in points]
_robust_time(base, points[0][0])

print(f"base={BASE}  margin={MK._GNT_AT_MARGIN}  (ratio = cand/base, <1 faster)", flush=True)
for cfg in GRID:
    try:
        ln = MK._get_nt_launch(K, G, N, *cfg, 0, 0, False, False)
        rs, snr_ok = [], True
        for (targs, out_view), (_, ref) in zip(points, refs):
            ln(*targs)
            torch.cuda.synchronize()
            o = out_view.detach().float()
            rn = float((ref * ref).sum().item()) or 1.0
            if float(((o - ref) ** 2).sum().item()) / rn >= (2e-2**2):
                snr_ok = False
            rs.append(_robust_ab_ratio(base, ln, targs))
        tag = "ADOPT" if (max(rs) < MK._GNT_AT_MARGIN and snr_ok) else "     "
        print(
            f"  {str(cfg):20s} bal {rs[0]:.4f}  skew {rs[1]:.4f}  "
            f"gmean {(rs[0]*rs[1])**0.5:.4f}  num_ok={snr_ok}  {tag}",
            flush=True,
        )
    except Exception as ex:
        print(f"  {str(cfg):20s} FAILED {type(ex).__name__}: {str(ex)[:70]}", flush=True)
