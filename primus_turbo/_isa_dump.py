"""Dump final ISA for the backward kernels and print a VALU/MFMA histogram.

Usage (remote container):
    FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python primus_turbo/_isa_dump.py
Then greps /tmp/isa/kernel_*/21_final_isa.s.
"""
import glob
import os
import re

import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_backward_impl,
    attention_flydsl_forward_impl,
)

torch.manual_seed(0)
B, S, Hq, Hkv, D = 1, 8192, 64, 8, 64
dev, dt = "cuda", torch.bfloat16
scale = 1.0 / (D**0.5)

q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
dout = torch.randn(B, S, Hq, D, device=dev, dtype=dt)

out, lse = attention_flydsl_forward_impl(q, k, v, scale, True)
attention_flydsl_backward_impl(dout, q, k, v, out, lse, scale, True)
torch.cuda.synchronize()

dump_dir = os.environ.get("FLYDSL_DUMP_DIR", "/root/.flydsl/debug")
print("dump_dir:", dump_dir)

# Instruction families to count.
FAMILIES = [
    ("v_mfma", re.compile(r"\bv_mfma")),
    ("v_pk_add_f32", re.compile(r"\bv_pk_add_f32")),
    ("v_pk_mul_f32", re.compile(r"\bv_pk_mul_f32")),
    ("v_pk_fma_f32", re.compile(r"\bv_pk_fma_f32")),
    ("v_pk_*(any)", re.compile(r"\bv_pk_")),
    ("v_add_f32", re.compile(r"\bv_add_f32")),
    ("v_mul_f32", re.compile(r"\bv_mul_f32")),
    ("v_fma_f32", re.compile(r"\bv_fmac?_f32|\bv_fma_f32")),
    ("v_cvt_i32_f32", re.compile(r"\bv_cvt_i32_f32")),
    ("v_cvt_pk_bf16", re.compile(r"\bv_cvt_pk_bf16")),
    ("v_cvt(any)", re.compile(r"\bv_cvt_")),
    ("v_dual", re.compile(r"\bv_dual_")),
    ("ds_read", re.compile(r"\bds_read")),
    ("ds_write", re.compile(r"\bds_write")),
    ("buffer_load", re.compile(r"\bbuffer_load")),
    ("s_barrier", re.compile(r"\bs_barrier")),
    ("scratch_load", re.compile(r"\bscratch_load")),
    ("scratch_store", re.compile(r"\bscratch_store")),
]

for isa in sorted(glob.glob(os.path.join(dump_dir, "kernel_*", "21_final_isa.s"))):
    name = os.path.basename(os.path.dirname(isa))
    with open(isa) as f:
        txt = f.read()
    print("=" * 70)
    print(name)
    meta = re.findall(r"(num_vgpr|num_agpr|vgpr_spill|sgpr_spill|num_sgpr|group_segment_fixed_size|private_segment_fixed_size)\D*(\d+)", txt)
    md = {}
    for kk, vv in meta:
        md.setdefault(kk, vv)
    print("  meta:", md)
    for fam, rx in FAMILIES:
        c = len(rx.findall(txt))
        if c:
            print(f"    {fam:20s} {c}")
