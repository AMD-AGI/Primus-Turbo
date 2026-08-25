#!/usr/bin/env python3
"""Static ISA of the D128 dkdv kernel with/without an arbitrary builder-kwarg override.

usage: _isa_g2w32.py <D> [k=v,k=v]
"""
import collections, glob, math, os, re, sys

D = int(sys.argv[1])
over = {}
if len(sys.argv) > 2 and sys.argv[2]:
    for kv in sys.argv[2].split(","):
        k, v = kv.split("=")
        over[k] = int(v) if v.lstrip("-").isdigit() else v
tag = "base" if not over else "_".join(f"{k}{v}" for k, v in over.items())
DUMP = f"/tmp/isa_g2_{D}_{tag}"
os.environ["FLYDSL_DUMP_IR"] = "1"
os.environ["FLYDSL_DUMP_DIR"] = DUMP
os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
os.system(f"rm -rf {DUMP}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import primus_turbo.flydsl.attention.flash_attn_bwd as M
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fbwd,
    flash_attn_sbhd_flydsl_forward_impl as ffwd,
)

if over:
    _b = M.build_flash_attn_bwd_dkdv_module
    M.build_flash_attn_bwd_dkdv_module = lambda **kw: _b(**{**kw, **over})

DEV, DT = "cuda", torch.bfloat16
B, S, HQ, HKV = 2, 8192, 64, 8
mk = lambda H: torch.randn(S, B, H, D, device=DEV, dtype=DT)
q, k, v = mk(HQ), mk(HKV), mk(HKV)
do = torch.randn_like(q)
o, lse = ffwd(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
fbwd(do, q, k, v, o, lse.view(B, S, HQ).permute(0, 2, 1), causal=True, window_size=(-1, -1))
torch.cuda.synchronize()

for path in sorted(glob.glob(os.path.join(DUMP, "**", "21_final_isa.s"), recursive=True)):
    kname = os.path.basename(os.path.dirname(path))
    blob = open(path).read()
    ops = re.findall(r"^\s+([a-z][\w.]*)", blob, re.M)
    c = collections.Counter(ops)
    mf = collections.Counter(o for o in ops if o.startswith("v_mfma"))
    meta = {k: int(re.search(rf"; {k}: (\d+)", blob).group(1))
            for k in ("vgpr_count", "agpr_count", "vgpr_spill_count", "sgpr_count")
            if re.search(rf"; {k}: (\d+)", blob)}
    print(f"== {kname}  tag={tag}  bytes={len(blob)}")
    print("   total_instr", len(ops), " mfma", sum(mf.values()), dict(mf))
    print("   ds_read", sum(v for k, v in c.items() if k.startswith("ds_read")),
          " ds_write", sum(v for k, v in c.items() if k.startswith("ds_write")),
          " vmem", sum(v for k, v in c.items() if k.startswith(("buffer_", "global_", "scratch_"))),
          " permlane", sum(v for k, v in c.items() if "permlane" in k),
          " valu", sum(v for k, v in c.items() if k.startswith("v_") and not k.startswith("v_mfma")))
    print("   meta", meta)
