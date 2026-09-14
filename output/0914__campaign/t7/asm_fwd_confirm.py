"""Timing + in-process agreement check: the timed ASM call must also be the correct one."""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
if os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo")
import torch  # noqa: E402

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
CAUSAL, SCALE = True, D ** -0.5


def sqnr_db(r, g):
    r, g = r.detach().double(), g.detach().double()
    return float(10 * torch.log10(r.norm().pow(2) / ((r - g).norm().pow(2) + 1e-30)))


def timed_ms(fn, iters=20, warmup=5):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_(); torch.cuda.synchronize()
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort(); return ts[len(ts) // 2]


torch.manual_seed(0)
q = torch.randn(B, S, HQ, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
qd, kd, vd = q.detach(), k.detach(), v.detach()

import aiter.ops.mha as M  # noqa: E402
from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType, GlobalBackendManager, PrecisionType)
from primus_turbo.pytorch.ops import flash_attn_func as turbo_fa  # noqa: E402
GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)

asm = lambda: M.fmha_fwd_with_sink_asm(qd, kd, vd, SCALE, CAUSAL, True)  # noqa: E731
tur = lambda: turbo_fa(q, k, v, causal=CAUSAL)                            # noqa: E731
asm(); tur(); torch.cuda.synchronize()

res = {"asm_ms": round(timed_ms(asm), 4), "turbo_ms": round(timed_ms(tur), 4)}
# the SAME callables that were just timed, now compared
o_asm, l_asm = asm()
o_tur = tur()
res["sqnr_asm_vs_turbo_db"] = round(sqnr_db(o_tur.float(), o_asm.float()), 3)
res["asm_out_finite"] = bool(torch.isfinite(o_asm).all())
res["lse_shape"] = list(l_asm.shape)
res["lse_dtype"] = str(l_asm.dtype)
print(json.dumps(res), flush=True)
