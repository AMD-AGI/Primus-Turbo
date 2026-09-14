"""Soak: is the ASM forward's 1.42 ms a cold-card artifact?

Cools the card first, then alternates ASM and turbo for many rounds. turbo is the
throttle anchor: the session champion measured it at 2.579 ms, so a round where turbo
reads near that is an unthrottled round and a round where it does not is not evidence.
"""
from __future__ import annotations
import json, os, sys, time
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
if os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
sys.path.insert(0, "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo")
COOL = int(os.environ.get("COOL_S", "300"))
print(f"cooling {COOL}s", flush=True)
time.sleep(COOL)
import torch  # noqa: E402

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
CAUSAL, SCALE = True, D ** -0.5


def timed_ms(fn, iters=20, warmup=3):
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
    ts.sort(); return round(ts[len(ts) // 2], 4)


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

# turbo FIRST, while the card is coldest: if turbo is going to look good anywhere it is
# here, so this cannot be accused of handing the cold window to the ASM kernel.
for r in range(10):
    t = timed_ms(tur)
    a = timed_ms(asm)
    print(json.dumps({"round": r, "turbo_ms": t, "asm_ms": a,
                      "ratio_asm_over_turbo": round(a / t, 3)}), flush=True)
