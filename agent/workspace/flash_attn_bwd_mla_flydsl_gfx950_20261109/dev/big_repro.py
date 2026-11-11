"""One MLA forward + backward at a large shape, to surface int32 overflows.
Usage: python big_repro.py [B] [S] [H] [q_split]"""

import sys

import torch

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

B, S, H = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (1, 65536, 128)
QS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
q, k = (torch.randn((S, B, H, 192), device="cuda", dtype=torch.bfloat16) for _ in range(2))
v, do = (torch.randn((S, B, H, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)
dq, dk, dv = mla_bwd_sbhd(do, q, k, v, o, lse, q_split=QS)
torch.cuda.synchronize()
print("ok", [bool(torch.isfinite(t).all()) for t in (dq, dk, dv)], [t.float().norm().item() for t in (dq, dk, dv)])
