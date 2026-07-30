#!/usr/bin/env python3
"""Compile the grouped mxfp8 NT kernel with FLYDSL_DUMP_IR=1 and report the authoritative
register metadata plus a prologue / mainloop / epilogue instruction histogram."""
import glob
import os
import subprocess
import sys

os.environ["FLYDSL_DUMP_IR"] = "1"

import torch  # noqa: E402

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK  # noqa: E402
import primus_turbo.pytorch  # noqa: F401,E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2944
K, G, M = 2944, 32, 8192
dev, stream = "cuda", torch.cuda.current_stream()

a = torch.zeros((M, K), dtype=torch.int8, device=dev)
w = torch.zeros((G, N, K), dtype=torch.int8, device=dev)
a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev)
w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device=dev)
o = (torch.arange(0, G + 1, dtype=torch.int64, device=dev) * (M // G)).view(torch.int32)
a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, dev, stream)
c = torch.zeros((M, N), dtype=torch.bfloat16, device=dev)
MK._get_nt_launch(K, G, N, 256, 4, 4, 0, 0, 0, False, False, preshuffle=True)(
    a, w, c, a_s.view(torch.int32).reshape(-1), w_s.view(torch.int32).reshape(-1), a_sp, b_sp,
    o, o, M, a_ngrp * 64, N, a_blocks, a_ngrp, ((M + 255) // 256 + G) * ((N + 255) // 256), stream,
)
torch.cuda.synchronize()

best, blen = None, -1
for f in glob.glob("/root/.flydsl/debug/**/*", recursive=True):
    if not os.path.isfile(f):
        continue
    try:
        t = open(f, errors="ignore").read()
    except OSError:
        continue
    if "v_mfma_scale" in t and "s_waitcnt" in t and len(t) > blen:
        best, blen = f, len(t)
print(f"# dump: {best}", flush=True)
KERN = "kernel_grouped_mxfp8_nt_1"
txt = open(best, errors="ignore").read()
tail = txt[txt.index(f".name:           {KERN}") - 2000: txt.index(f".name:           {KERN}") + 800]
for ln in tail.splitlines():
    if any(k in ln for k in (".num_vgpr", ".num_agpr", "vgpr_spill_count", "sgpr_spill_count",
                             "private_segment_fixed_size", "group_segment_fixed_size", ".sgpr_count")):
        print("  " + ln.strip())
subprocess.run([sys.executable, "_isa_region.py", best, KERN])
