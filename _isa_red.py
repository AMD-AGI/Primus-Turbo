#!/usr/bin/env python3
"""dqred's own register allocation, with an optional builder-kwarg override.

The fold's waves co-reside with the fused body, which takes 462 of the SIMD's 512, so what
the fold can afford is 50 dwords TOTAL across however many waves are wanted per SIMD.

usage: _isa_red.py [k=v,...]
"""
import glob, os, re, sys
over = {}
if len(sys.argv) > 1 and sys.argv[1]:
    for kv in sys.argv[1].split(","):
        k, v = kv.split("=")
        over[k] = int(v) if v.lstrip("-").isdigit() else v
tag = "base" if not over else "_".join(f"{k}{v}" for k, v in over.items())
DUMP = f"/tmp/isa_red_{tag}"
os.environ.update(FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=DUMP, FLYDSL_RUNTIME_ENABLE_CACHE="0")
os.system(f"rm -rf {DUMP}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import primus_turbo.flydsl.attention.flash_attn_bwd as M

if over:
    _b = M.build_flash_attn_bwd_dqred_module
    M.build_flash_attn_bwd_dqred_module = lambda **kw: _b(**{**kw, **over})
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)

B, S, HQ, HKV, D = 2, 8192, 64, 8, 128
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
fb(do, q, k, v, o, lse.view(B, S, HQ).permute(0, 2, 1), causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
for p in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    kn = os.path.basename(os.path.dirname(p))
    if "dqred" not in kn and "dkdv" not in kn:
        continue
    b = open(p).read()
    reg = {n: int(re.search(rf"[.;] ?{n}: *(\d+)", b).group(1))
           for n in ("vgpr_count", "agpr_count", "sgpr_count", "vgpr_spill_count")
           if re.search(rf"[.;] ?{n}: *(\d+)", b)}
    print(f"{kn:34s} tag={tag:10s} {reg}")
    if "dqred" in kn:
        print("     loads", b.count("buffer_load_dwordx4"), " vmcnt-waits", b.count("s_waitcnt vmcnt"),
              " free-after-body", 512 - 462, " waves@this", (512 - 462) // max(reg["vgpr_count"], 1))
