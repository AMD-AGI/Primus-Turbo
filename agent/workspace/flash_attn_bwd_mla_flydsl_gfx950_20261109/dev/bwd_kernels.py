"""Per-kernel device time of the FlyDSL MLA backward and the aiter a16 backward (torch.profiler, 5 calls).
Usage: python bwd_kernels.py [B] [S] [H] [key=value ...]   (extra args: mla_bwd_sbhd overrides)"""

import ast
import os
import sys

os.environ.setdefault("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "0")
sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")

import torch  # noqa: E402
from flydsl_mla import mla_fwd_sbhd  # noqa: E402
from flydsl_mla.ops_bwd import mla_bwd_sbhd  # noqa: E402

import primus_turbo.pytorch as pt  # noqa: E402
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager  # noqa: E402

argv = [a for a in sys.argv[1:] if "=" not in a]
over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
B, S, H = (int(x) for x in argv[:3]) if len(argv) >= 3 else (2, 4096, 128)
q, k = (torch.randn((S, B, H, 192), device="cuda", dtype=torch.bfloat16) for _ in range(2))
v, do = (torch.randn((S, B, H, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2))
o, lse = mla_fwd_sbhd(q, k, v, causal=True, return_lse=True)

GlobalBackendManager.set_attn_backend(BackendType.AITER)
leaves = [t.permute(1, 0, 2, 3).detach().requires_grad_() for t in (q, k, v)]
oa = pt.ops.flash_attn_func(*leaves, softmax_scale=192**-0.5, causal=True)
dob = do.permute(1, 0, 2, 3)

cases = {
    "flydsl": lambda: mla_bwd_sbhd(do, q, k, v, o, lse, **over),
    "aiter_a16": lambda: torch.autograd.grad(oa, leaves, dob, retain_graph=True),
}
for name, fn in cases.items():
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    evs = [e for e in prof.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in evs) / 5
    print(f"== {name} B{B} S{S} H{H} {over}: kernel sum {tot:.1f} us")
    for e in sorted(evs, key=lambda e: -e.device_time_total):
        print(f"   {e.count // 5:3d}x {e.device_time_total / 5:9.1f} us  {100 * e.device_time_total / 5 / tot:5.1f}%  {e.key[:90]}")
