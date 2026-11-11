"""Run the MLA backward of the package at MLA_PKG_ROOT once (B1 S1K, 128 heads, causal) so that
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> writes its ISA. Usage: python isa_dump_bwd.py [key=value ...]
(extra args are passed to mla_bwd_sbhd as dkdv builder overrides; ints/bools parsed)."""

import ast
import os
import sys

import torch

sys.path.insert(0, os.environ.get("MLA_PKG_ROOT", "/perf_apps/xiaoming/megaattention_overlap_bench"))
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
S = int(os.environ.get("ISA_S", "1024"))
q = torch.randn((S, 1, 128, 192), device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn((S, 1, 128, 128), device="cuda", dtype=torch.bfloat16)
do = torch.randn_like(v)
o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
mla_bwd_sbhd(do, q, k, v, o, lse, **over)
torch.cuda.synchronize()
print("compiled", over)
