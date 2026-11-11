"""Forked bwd (head_dim_v=None) must reproduce Turbo's D128 backward bit for bit."""

import sys

import torch

sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")
import primus_turbo.flydsl.attention.flash_attn_bwd as tb  # noqa: E402
from flydsl_mla import flash_attn_bwd_mla as fb  # noqa: E402

torch.manual_seed(0)
B, S, H, D = 2, 4096, 16, 128
scale = D**-0.5
dev = "cuda"
q = torch.randn(S, B, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(S, B, H, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(S, B, H, D, device=dev, dtype=torch.bfloat16)
do = torch.randn(S, B, H, D, device=dev, dtype=torch.bfloat16)
o = torch.randn(S, B, H, D, device=dev, dtype=torch.bfloat16)
lse = (torch.randn(B, H, S, device=dev, dtype=torch.float32) * 0.5 + 8.0).contiguous()


def run(mod, det):
    return mod.flydsl_varlen_backward(do, q, k, v, o, lse, B, S, S, H, H, D, scale, sbhd=True, deterministic=det)


ok = True
for det in (True, False):
    a = run(tb, det)
    b = run(fb, det)
    torch.cuda.synchronize()
    for name, x, y in zip(("dq", "dk", "dv"), a, b):
        same = torch.equal(x, y)
        md = (x.float() - y.float()).abs().max().item()
        print(f"det={det} {name}: bitwise={same} maxdiff={md:.3e}")
        if det or name != "dq":
            ok &= same

# The MLA-shaped configuration (block_kv 128, block_q 32, a16, no pair) at D128.
kw = dict(
    num_heads=H, head_dim=D, causal=True, dtype_str="bf16", sm_scale=scale, num_kv_heads=H,
    window_left=-1, q_split=2, block_kv=128, batch_size=B, sbhd=True, waves_per_eu=1, g2d=2,
    dma_grp=1, pf_ring=False, square=True, kv_halves=1, q_pref=True, g3_defer=True, mfma_tie=0,
    g3_dbat=2, g3_kreg=False, k_reg=True, g3_st_at=0, g3_st_n=1, g3_sb=1, block_q=32, flat_wg=256,
    wsq_a16=True,
)
outs = []
for mod in (tb, fb):
    l = mod.build_flash_attn_bwd_dkdv_module(**kw)
    st = torch.cuda.current_stream()
    qf = tb._prescale_q(q, scale)
    lsef = (lse * -tb._LOG2E).reshape(-1)
    delta = (-(o.float() * do.float()).sum(-1)).permute(1, 2, 0).contiguous().reshape(-1)
    ws_dk = torch.empty(2, S, B, H, D, device=dev, dtype=torch.bfloat16)
    ws_dv = torch.empty(2, S, B, H, D, device=dev, dtype=torch.bfloat16)
    img = torch.zeros(B * S * H * D, device=dev, dtype=torch.bfloat16)
    cu = torch.empty(1, device=dev, dtype=torch.int32)
    l(qf, k.reshape(-1), v.reshape(-1), do.reshape(-1), lsef, delta, ws_dk.reshape(-1), ws_dv.reshape(-1), cu, cu, img, B, S, S, 0, st)
    img.zero_()
    l(qf, k.reshape(-1), v.reshape(-1), do.reshape(-1), lsef, delta, ws_dk.reshape(-1), ws_dv.reshape(-1), cu, cu, img, B, S, S, 0, st)
    torch.cuda.synchronize()
    outs.append((ws_dk, ws_dv, img))
for name, x, y in zip(("ws_dk", "ws_dv", "img"), outs[0], outs[1]):
    same = torch.equal(x, y)
    md = (x.float() - y.float()).abs().max().item()
    print(f"mla-shaped D128 {name}: bitwise={same} maxdiff={md:.3e}")
    if name != "img":
        ok &= same
print("REGRESS", "PASS" if ok else "FAIL")
