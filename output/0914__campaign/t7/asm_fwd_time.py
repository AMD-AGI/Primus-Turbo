"""T7 timing-only: ASM fwd vs turbo fwd vs aiter Triton fwd, interleaved, same process."""
from __future__ import annotations
import json, os, sys

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
if os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
REPO = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo"
sys.path.insert(0, REPO)
import torch  # noqa: E402

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
CAUSAL, SCALE = True, D ** -0.5


def timed_ms(fn, iters=20, warmup=5):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()
        torch.cuda.synchronize()
        st.record(); fn(); en.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    ts.sort()
    return ts[len(ts) // 2]


torch.manual_seed(0)
q = torch.randn(B, S, HQ, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

import aiter.ops.mha as M  # noqa: E402
from aiter.ops.triton.attention.mha import flash_attn_func as aiter_fa  # noqa: E402
from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType, GlobalBackendManager, PrecisionType)
from primus_turbo.pytorch.ops import flash_attn_func as turbo_fa  # noqa: E402
GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)

qd, kd, vd = q.detach(), k.detach(), v.detach()
cands = {
    "asm_lse":   lambda: M.fmha_fwd_with_sink_asm(qd, kd, vd, SCALE, CAUSAL, True),
    "asm_nolse": lambda: M.fmha_fwd_with_sink_asm(qd, kd, vd, SCALE, CAUSAL, False),
    "turbo":     lambda: turbo_fa(q, k, v, causal=CAUSAL),
    "aiter_tri": lambda: aiter_fa(q, k, v, causal=CAUSAL),
}
for fn in cands.values():          # one-time JIT / autotune, outside the timing
    fn()
torch.cuda.synchronize()

rounds = {n: [] for n in cands}
for r in range(3):
    for n, fn in cands.items():
        rounds[n].append(round(timed_ms(fn), 4))
        torch.cuda.empty_cache()
out = {n: {"reps": ts, "median": sorted(ts)[1]} for n, ts in rounds.items()}
print(json.dumps(out), flush=True)
