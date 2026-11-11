"""aiter MLA backward time and kernels vs the layout of dout (B2 S4K H128, 192/128, causal, sbhd leaves).
Variants: dout = randn_like(out) (out's own strides) and dout = sbhd-stored [S,B,H,Dv] permuted to bshd."""

import torch

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager

B, S, H, DQK, DV = 2, 4096, 128, 192, 128
GlobalBackendManager.set_attn_backend(BackendType.AITER)
q, k, v = (torch.randn((S, B, H, d), device="cuda", dtype=torch.bfloat16).permute(1, 0, 2, 3).requires_grad_() for d in (DQK, DQK, DV))
out = pt.ops.flash_attn_func(q, k, v, softmax_scale=DQK**-0.5, causal=True)
print("out shape", tuple(out.shape), "stride", out.stride())
variants = {
    "randn_like(out)": torch.randn_like(out),
    "sbhd permuted": torch.randn((S, B, H, DV), device="cuda", dtype=torch.bfloat16).permute(1, 0, 2, 3),
    "bshd contiguous": torch.randn((B, S, H, DV), device="cuda", dtype=torch.bfloat16),
}
flops = B * S * (S + 1) // 2 * H * 2 * (3 * DQK + 2 * DV)
for name, g in variants.items():
    fn = lambda: torch.autograd.grad(out, (q, k, v), g, retain_graph=True)  # noqa: E731
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(20):
        fn()
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / 20
    print(f"== dout {name}: stride {g.stride()}  bwd {ms:.3f} ms  {flops / ms / 1e9:.1f} TFLOPS")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    for e in sorted(prof.key_averages(), key=lambda e: -e.device_time_total):
        if e.device_time_total > 0:
            print(f"   {e.count // 5:3d}x {e.device_time_total / 5:8.1f} us  {e.key[:100]}")
