"""Diagnostic: build the NT fwd kernel for the campaign `down` shape (N=2944 -> half-N
boundary block) and, for contrast, an N=3072-style aligned build, so the emitted ISA can be
compared for register pressure / spill / code size. Read-only w.r.t. production code.
"""
import os
import sys

import torch

import primus_turbo.pytorch  # noqa: F401
import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK

DEV = "cuda"
G, K = 32, 2944
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2944
M = 131072

os.environ.setdefault("FLYDSL_DUMP_IR", "1")

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
ln = MK._get_nt_launch(K, G, N, 256, 4, 4, 0, 0, 0, False, False)
ln(*args)
torch.cuda.synchronize()
print(f"ok N={N}", flush=True)
