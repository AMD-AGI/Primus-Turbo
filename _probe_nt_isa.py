"""Dump the mxfp8 NT main kernel ISA for one campaign shape and print the instruction mix.

Usage: FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/ntisa FLYDSL_RUNTIME_ENABLE_CACHE=0 python _probe_nt_isa.py [N] [K]
"""
import os
import re
import sys
from collections import Counter

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.low_precision import float8_e4m3

DEV = "cuda"
G, M = 32, 8192
CFG = (256, 4, 4, 0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2944
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2944


def f8(*s):
    t = torch.empty(s, dtype=float8_e4m3, device=DEV)
    t.view(torch.uint8).random_(0, 64)
    return t


a, w = f8(M, K), f8(G, N, K)
a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=DEV)
w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=DEV)
o = torch.arange(0, G + 1, dtype=torch.int64, device=DEV) * (M // G)
stream = torch.cuda.current_stream()
a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, DEV, stream)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=DEV)
args = (
    a.view(torch.int8),
    w.view(torch.int8),
    c,
    a_s.view(torch.int32).reshape(-1),
    w_s.view(torch.int32).reshape(-1),
    a_sp,
    b_sp,
    o.view(torch.int32),
    o.view(torch.int32),
    M,
    a_ngrp * 64,
    N,
    a_blocks,
    a_ngrp,
    ((M + 255) // 256 + G) * ((N + 255) // 256),
    stream,
)
MK._get_nt_launch(K, G, N, *CFG, 0, 0, False, False, preshuffle=True)(*args)
torch.cuda.synchronize()
print(f"ran N={N} K={K}", flush=True)

root = os.environ.get("FLYDSL_DUMP_DIR", "/root/.flydsl/debug")
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d, "21_final_isa.s")
    if not os.path.exists(p) or "mxfp8_nt" not in d:
        continue
    txt = open(p).read()
    ops = Counter()
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith((".", "/", ";", "//")) or s.endswith(":"):
            continue
        ops[s.split()[0]] += 1
    print(f"\n===== {d} =====")
    for key in (
        "num_vgpr",
        "num_agpr",
        "private_seg_size",
        "numbered_sgpr",
        "group_segment_fixed_size",
        "occupancy",
    ):
        for mm in re.finditer(rf"\.?\b\w*{key}\w*\b[ =:]+(\S+)", txt):
            print(f"  {key}: {mm.group(1)}")
            break
    tot = sum(ops.values())
    print(f"  total instructions: {tot}")
    for pat in (
        "buffer_load_dword$",
        "buffer_load_dwordx2$",
        "buffer_load_dwordx4$",
        "buffer_load_ubyte",
        "buffer_load_.*lds",
        "buffer_store_short",
        "buffer_store_dword",
        "ds_read",
        "ds_write",
        "s_barrier",
        "v_mfma",
        "scratch_",
        "v_cmp",
        "v_cndmask",
        "s_waitcnt",
    ):
        n = sum(v for kk, v in ops.items() if re.match(pat.rstrip("$") + ("$" if pat.endswith("$") else ""), kk))
        print(f"  {pat:26s} {n}")
    print("  top-20:", ops.most_common(20))
