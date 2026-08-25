"""Is the 1.2 ms hole host-side? Time the CPU cost of one bwd launch with no sync."""
import sys, time, torch
sys.path.insert(0, "/workspace/code/tensorwise/Primus-Turbo")
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)
B, Hq, Hkv, S, D = 2, 64, 8, 8192, 128
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv); do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lh = lse.view(B, S, Hq).permute(0, 2, 1)
for _ in range(10):
    fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
cpu = []
for _ in range(20):
    t = time.perf_counter(); fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1))
    cpu.append((time.perf_counter() - t) * 1e3)
torch.cuda.synchronize()
cpu.sort()
print("CPU launch ms: min %.3f med %.3f max %.3f" % (cpu[0], cpu[len(cpu)//2], cpu[-1]))
ev = lambda: (torch.cuda.Event(True), torch.cuda.Event(True))
best = 1e9
for _ in range(20):
    s, e = ev(); s.record(); fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1)); e.record()
    torch.cuda.synchronize(); best = min(best, s.elapsed_time(e))
print("GPU ms: %.4f" % best)
