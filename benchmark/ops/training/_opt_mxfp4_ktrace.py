"""Isolated dual-quant timing/BW probe (NOT the scored ruler).

Times the FlyDSL fused dual on the two fat shapes on the critical path:
  * grad_out dual  [4096, 28672]  row_rht=T col_rht=T  2d=F
  * B-weight dual  [28672, 8192]  row_2d=T col_2d=T col_rht=T
and (optionally) the C++ dual for the same, reporting us/call + effective BW.

Run in-container:
  python -u benchmark/ops/training/_opt_mxfp4_ktrace.py
"""

import torch

import primus_turbo.pytorch.kernels.quantization.quantization_impl  # noqa: F401
from primus_turbo.flydsl.quant.mxfp4_quant_kernel import get_dual_cast

ITERS = 50
WARMUP = 10


def _traffic_bytes(R, C):
    # read bf16 (2B) + write row fp4 (0.5B) + col fp4 (0.5B) + 2 scale bytes/32
    return R * C * 2 + R * C // 2 + R * C // 2 + 2 * (R * C // 32)


def time_fly(R, C, row_rht, col_rht, row_2d, col_2d):
    x_i32 = torch.randn((R, C), dtype=torch.bfloat16, device="cuda").view(torch.int32)
    ro = torch.empty((R, C // 8), dtype=torch.int32, device="cuda")
    rs = torch.empty((R, C // 32), dtype=torch.uint8, device="cuda")
    co = torch.empty((C, R // 8), dtype=torch.int32, device="cuda")
    cs = torch.empty((C, R // 32), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream()
    fn, grid_x = get_dual_cast(R, C, row_rht, col_rht, row_2d, col_2d)
    for _ in range(WARMUP):
        fn(x_i32, ro, rs, co, cs, R, C, grid_x, stream)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(ITERS):
        fn(x_i32, ro, rs, co, cs, R, C, grid_x, stream)
    en.record()
    torch.cuda.synchronize()
    us = st.elapsed_time(en) * 1000.0 / ITERS
    bw = _traffic_bytes(R, C) / (us * 1e-6) / 1e12
    return us, bw


if __name__ == "__main__":
    cases = [
        ("grad_out", 4096, 28672, True, True, False, False),
        ("B-weight", 28672, 8192, False, True, True, True),
    ]
    for name, R, C, rr, cr, r2, c2 in cases:
        us, bw = time_fly(R, C, rr, cr, r2, c2)
        print(f"[{name} {R}x{C}] FlyDSL {us:.1f} us/call  {bw:.2f} TB/s")
