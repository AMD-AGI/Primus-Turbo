#!/usr/bin/env python3
"""Same region histogram as _isa_mxnt.py, for the grouped mxfp8 variable-K wgrad kernel."""
import glob
import os
import re
import subprocess
import sys

os.environ["FLYDSL_DUMP_IR"] = "1"

import torch  # noqa: E402

import primus_turbo.pytorch  # noqa: F401,E402
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (  # noqa: E402
    grouped_gemm_mxfp8_variable_k_flydsl_kernel as fly_vark_mx,
)
from primus_turbo.pytorch.core.low_precision import float8_e4m3  # noqa: E402

OUT_M, OUT_N = 2944, int(sys.argv[1]) if len(sys.argv) > 1 else 2944
G, M = 32, 8192
dev = "cuda"
o = (torch.arange(0, G + 1, dtype=torch.int64, device=dev) * (M // G))
lhs = torch.zeros((OUT_M, M), dtype=torch.uint8, device=dev).view(float8_e4m3)
rhs = torch.zeros((OUT_N, M), dtype=torch.uint8, device=dev).view(float8_e4m3)
ls = torch.full((OUT_M, M // 32), 127, dtype=torch.uint8, device=dev)
rs = torch.full((OUT_N, M // 32), 127, dtype=torch.uint8, device=dev)
fly_vark_mx(lhs, ls, rhs, rs, o, OUT_M, OUT_N, G, out_dtype=torch.bfloat16, num_cu=-1)
torch.cuda.synchronize()

best, blen = None, -1
for f in glob.glob("/root/.flydsl/debug/**/*", recursive=True):
    if not os.path.isfile(f) or not f.endswith(".s"):
        continue
    t = open(f, errors="ignore").read()
    if "kernel_grouped_mxfp8_wgrad" in t and len(t) > blen:
        best, blen = f, len(t)
print(f"# dump: {best}", flush=True)
txt = open(best, errors="ignore").read()
KERN = sorted(set(re.findall(r"^(kernel_grouped_mxfp8_wgrad_\d+):", txt, re.M)))[-1]
tail = txt[txt.index(f".name:           {KERN}") - 2000: txt.index(f".name:           {KERN}") + 800]
for ln in tail.splitlines():
    if any(k in ln for k in (".num_vgpr", ".num_agpr", "vgpr_spill_count", "sgpr_spill_count",
                             "private_segment_fixed_size", "group_segment_fixed_size", ".sgpr_count")):
        print("  " + ln.strip())
subprocess.run([sys.executable, "_isa_region.py", best, KERN])
