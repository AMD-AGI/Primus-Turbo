import sys, torch
sys.path.insert(0, "/workspace/code/tensorwise/Primus-Turbo")
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl as fb, flash_attn_sbhd_flydsl_forward_impl as ff)
B, Hq, Hkv, S, D = 2, 64, 8, 8192, 128
mk = lambda H: torch.randn(S, B, H, D, device="cuda", dtype=torch.bfloat16)
q, k, v = mk(Hq), mk(Hkv), mk(Hkv); do = torch.randn_like(q)
o, lse = ff(q, k, v, causal=True, window_size=(-1, -1), return_lse=True)
lh = lse.view(B, S, Hq).permute(0, 2, 1)
for _ in range(8):
    fb(do, q, k, v, o, lh, causal=True, window_size=(-1, -1))
torch.cuda.synchronize()
