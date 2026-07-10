"""Determinism diagnostic: run the FlyDSL backward twice on identical inputs and
report which of dq/dk/dv differ (and by how much)."""
import torch

from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    attention_flydsl_backward_impl,
    attention_flydsl_forward_impl,
)

torch.manual_seed(0)
B, S, Hq, Hkv, D = 2, 1024, 64, 8, 64
dev, dt = "cuda", torch.bfloat16
sc = 1.0 / (D**0.5)
q = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
k = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
v = torch.randn(B, S, Hkv, D, device=dev, dtype=dt)
do = torch.randn(B, S, Hq, D, device=dev, dtype=dt)
o, l = attention_flydsl_forward_impl(q, k, v, sc, True)


def run():
    dq, dk, dv = attention_flydsl_backward_impl(do, q, k, v, o, l, sc, True)
    torch.cuda.synchronize()
    return dq.clone(), dk.clone(), dv.clone()


a = run()
for _ in range(3):
    b = run()
    for name, x, y in zip(("dq", "dk", "dv"), a, b):
        # dk/dv: [B, S, Hkv, D]; dq: [B, S, Hq, D]
        xf, yf = x.float(), y.float()
        d = (xf != yf) & ~(xf.isnan() & yf.isnan())
        n = int(d.sum())
        if n:
            # which S rows differ (collapse B, H, D)
            rows = d.any(dim=-1).any(dim=-1)  # [B, S]
            srows = rows[0].nonzero().flatten()
            mod256 = (srows % 256)
            in_tile1 = int(((mod256 >= 128)).sum())
            in_tile0 = int(((mod256 < 128)).sum())
            print(f"{name}: rows={srows.numel()} tile0(<128)={in_tile0} tile1(>=128)={in_tile1} "
                  f"first_rows={srows[:8].tolist()} mod256={sorted(set(mod256.tolist()))[:12]}")
print("done")
