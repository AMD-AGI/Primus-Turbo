#!/usr/bin/env python3
"""Dump the mxfp8 NT ISA and print a compact opcode trace (waitcnt / barrier / LDS-DMA /
scale load / ds_read / mfma), so the emitted wait sequence of the full body and of the
half-N boundary body can be compared instruction by instruction."""
import glob
import os
import re
import sys

os.environ["FLYDSL_DUMP_IR"] = "1"

import torch  # noqa: E402

import primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel as MK  # noqa: E402
import primus_turbo.pytorch  # noqa: F401,E402

HN = int(sys.argv[1]) if len(sys.argv) > 1 else 0
K, N, G = 2944, 2944, 32
MK._build_grouped_mxfp8_nt_kernel(K=K, G=G, N=N, hn_mode=HN)  # build only; dump on compile
M = 4096
a = torch.zeros((M, K), dtype=torch.int8, device="cuda")
w = torch.zeros((G, N, K), dtype=torch.int8, device="cuda")
a_s = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
w_s = torch.full((G, N, K // 32), 127, dtype=torch.uint8, device="cuda")
o = torch.arange(0, G + 1, dtype=torch.int64, device="cuda") * (M // G)
stream = torch.cuda.current_stream()
a_sp, b_sp, a_blocks, a_ngrp = MK._get_grouped_mx_workspace(M, N, K // 128, G, "cuda", stream)
c = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")
MK._get_nt_launch(K, G, N, 256, 4, 4, 0, 0, 0, False, False, preshuffle=True, hn_mode=HN)(
    a, w, c, a_s.view(torch.int32).reshape(-1), w_s.view(torch.int32).reshape(-1), a_sp, b_sp,
    o.view(torch.int32), o.view(torch.int32), M, a_ngrp * 64, N, a_blocks, a_ngrp,
    ((M + 255) // 256 + G) * ((N + 255) // 256), stream,
)
torch.cuda.synchronize()

PATS = [
    (re.compile(r"^\s*s_waitcnt\s+(.*)$"), lambda m: f"WAIT[{m.group(1).strip()}]"),
    (re.compile(r"^\s*s_barrier"), lambda m: "BAR"),
    (re.compile(r"^\s*buffer_load_dwordx4 .*lds"), lambda m: "G2S"),
    (re.compile(r"^\s*buffer_load_dword\b"), lambda m: "SCL"),
    (re.compile(r"^\s*ds_read"), lambda m: "DSR"),
    (re.compile(r"^\s*v_mfma_scale"), lambda m: "MFMA"),
    (re.compile(r"^\s*buffer_store"), lambda m: "ST"),
    (re.compile(r"^\s*s_cbranch\w*\s+(\S+)"), lambda m: f"BR->{m.group(1)}"),
    (re.compile(r"^\s*s_branch\s+(\S+)"), lambda m: f"JMP->{m.group(1)}"),
    (re.compile(r"^(\.?\w+):\s*(;.*)?$"), lambda m: f"@{m.group(1)}"),
    (re.compile(r"^\s*s_setprio\s+(\d)"), lambda m: f"PRIO{m.group(1)}"),
]

best, txt = None, None
for f in glob.glob("/root/.flydsl/debug/**/*", recursive=True):
    if not os.path.isfile(f):
        continue
    try:
        t = open(f, errors="ignore").read()
    except OSError:
        continue
    if "v_mfma_scale" in t and "s_waitcnt" in t:
        if best is None or len(t) > len(txt):
            best, txt = f, t
print(f"# dump: {best}", flush=True)
seq = []
for line in txt.splitlines():
    for pat, fn in PATS:
        m = pat.match(line)
        if m:
            seq.append(fn(m))
            break
out, i = [], 0
while i < len(seq):
    j = i
    while j < len(seq) and seq[j] == seq[i]:
        j += 1
    out.append(seq[i] if j - i == 1 else f"{seq[i]}x{j-i}")
    i = j
print(f"# {len(seq)} tracked instrs -> {len(out)} runs", flush=True)
print(" ".join(out))
