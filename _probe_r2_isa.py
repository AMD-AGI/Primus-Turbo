"""Optimize r1: where does the wgrad split-K machinery's 2.3-3.2% go?  ISA histogram diff.

_probe_r2_wgdist showed the split-K path costs 2.3% (gate_up) / 3.2% (down) of the wgrad kernel
even on distributions where the runtime policy returns s == 1 (no window), while replacing the
O(G) policy scan itself with one load is worth 0%.  So the cost is in the per-tile slice
bookkeeping / grid extension, not the scan.  This dumps the final ISA for the same kernel with
the split path on and off and prints the instruction histogram of each, so the cost has a name.

usage: FLYDSL_DUMP_IR=1 _probe_r2_isa.py <OUT_N>
"""

import collections
import glob
import os
import shutil
import sys

import torch

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK
from primus_turbo.flydsl.grouped_gemm.gemm_fp8_grouped_kernel import _wgrad_split_ws

DEV = "cuda"
G = 32
PER = 4096
MTOT = G * PER
H = 2944
F8 = torch.float8_e4m3fn
PACK = 4
DBG = "/root/.flydsl/debug"

KEYS = (
    "TOTAL",
    "v_mfma",
    "ds_read",
    "ds_write",
    "vmem_load",
    "vmem_store",
    "s_barrier",
    "s_setprio",
    "s_waitcnt",
    "  vmcnt(0)",
    "  lgkmcnt(0)",
    "s_nop",
    "valu_other",
    "salu_other",
    "s_mov/s_cselect",
    "v_cndmask",
    "readfirstlane",
)


def hist(path):
    c = collections.Counter()
    for line in open(path):
        line = line.strip()
        if not line or line.startswith((".", ";", "//", "/*")) or line.endswith(":"):
            continue
        op = line.split()[0]
        if op.startswith("v_mfma"):
            c["v_mfma"] += 1
        elif op.startswith("ds_read"):
            c["ds_read"] += 1
        elif op.startswith("ds_write"):
            c["ds_write"] += 1
        elif op.startswith(("buffer_load", "global_load")):
            c["vmem_load"] += 1
        elif op.startswith(("buffer_store", "global_store")):
            c["vmem_store"] += 1
        elif op.startswith("s_barrier"):
            c["s_barrier"] += 1
        elif op.startswith("s_setprio"):
            c["s_setprio"] += 1
        elif op.startswith("s_waitcnt"):
            c["s_waitcnt"] += 1
            if "vmcnt(0)" in line:
                c["  vmcnt(0)"] += 1
            if "lgkmcnt(0)" in line:
                c["  lgkmcnt(0)"] += 1
        elif op.startswith("s_nop"):
            c["s_nop"] += 1
        elif op.startswith("v_"):
            c["valu_other"] += 1
        elif op.startswith("s_"):
            c["salu_other"] += 1
        if op.startswith(("s_mov", "s_cselect")):
            c["s_mov/s_cselect"] += 1
        if op.startswith("v_cndmask"):
            c["v_cndmask"] += 1
        if "readfirstlane" in op:
            c["readfirstlane"] += 1
        c["TOTAL"] += 1
    return c


def main():
    OUT_N = int(sys.argv[1])
    OUT_M = H
    torch.manual_seed(0)
    offs = torch.zeros(G + 1, dtype=torch.int64, device=DEV)
    offs[1:] = torch.full((G,), PER, dtype=torch.int64, device=DEV).cumsum(0)
    lhs = (torch.randn(OUT_M, MTOT, device=DEV) * 0.5).to(F8)
    rhs = (torch.randn(OUT_N, MTOT, device=DEV) * 0.5).to(F8)
    a_raw = torch.full((OUT_M * MTOT // 32,), 127, dtype=torch.uint8, device=DEV).view(torch.int32)
    b_raw = torch.full((OUT_N * MTOT // 32,), 127, dtype=torch.uint8, device=DEV).view(torch.int32)
    a8, b8 = lhs.view(torch.int8), rhs.view(torch.int8)
    out = torch.empty((G, OUT_M, OUT_N), dtype=torch.bfloat16, device=DEV)
    stream = torch.cuda.current_stream()
    K128 = MTOT // 128
    a_sp, b_sp = MK._get_grouped_wgrad_workspace(OUT_M, OUT_N, K128, G, PACK, DEV, stream)
    a_ngrp = (OUT_M + 63) // 64
    b_ngrp = ((OUT_N + 255) // 256) * 4
    n_ck = K128 // MK._PRESHUF_KT + G
    a_blocks = a_ngrp * n_ck
    pre_grid = a_blocks + b_ngrp * n_ck
    ws = _wgrad_split_ws(OUT_M, OUT_N, G, DEV, torch.bfloat16, BLOCK_M=256, BLOCK_N=256)
    args = (a8, b8, out, a_raw, b_raw, a_sp, b_sp, offs.view(torch.int32), ws, MTOT, K128, n_ck, a_blocks, pre_grid, stream)

    _geom = MK._wgrad_split_geom
    cols = {}
    for name, off in (("split", False), ("nosplit", True)):
        shutil.rmtree(DBG, ignore_errors=True)
        MK._wgrad_split_geom = (lambda *a: (1, 1, 1, 0, 0)) if off else _geom
        try:
            MK._compile_grouped_mxfp8_wgrad_fused(
                OUT_M, OUT_N, G, 256, 256, 4, 1, 0, 0, 0, False, pack=PACK, preshuffle=False
            )(*args)
            torch.cuda.synchronize()
        finally:
            MK._wgrad_split_geom = _geom
        best = None
        for f in glob.glob(f"{DBG}/*/21_final_isa.s"):
            n = sum(1 for line in open(f) if line.strip().startswith("v_mfma"))
            if best is None or n > best[0]:
                best = (n, f)
        c = hist(best[1])
        meta = [
            line.strip()
            for line in open(best[1])
            if line.strip().startswith((".vgpr_count", ".agpr_count", ".sgpr_count", ".vgpr_spill_count", ".group_segment_fixed_size", ".occupancy"))
        ]
        cols[name] = c
        print(f"=== {name}: {os.path.basename(os.path.dirname(best[1]))}  {' '.join(meta)}", flush=True)
    print(f"{'key':22s} {'split':>8s} {'nosplit':>8s} {'delta':>8s}")
    for k in KEYS:
        a, b = cols["split"][k], cols["nosplit"][k]
        print(f"{k:22s} {a:8d} {b:8d} {a-b:8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
