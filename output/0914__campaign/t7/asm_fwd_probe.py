"""T7 scratch probe: aiter prebuilt gfx1250 ASM forward on our production shape.

Scratch only. Patches nothing in the installed aiter or the source clone.
Numerics first (chunked fp32 reference, per (b, hq)), timing second.
"""
from __future__ import annotations
import json, os, sys, time

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
if os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]

REPO = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
CAUSAL = True
SCALE = D ** -0.5
res = {"shape": [B, S, HQ, HKV, D], "causal": CAUSAL}


def sqnr_db(ref, got):
    r, g = ref.detach().double(), got.detach().double()
    return float(10 * torch.log10(r.norm().pow(2) / ((r - g).norm().pow(2) + 1e-30)))


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
dev = "cuda"
q = torch.randn(B, S, HQ, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, S, HKV, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, S, HKV, D, device=dev, dtype=torch.bfloat16)

import aiter.ops.mha as M  # noqa: E402
res["aiter_mha"] = M.__file__

# --- 1. run the ASM forward -------------------------------------------------
t0 = time.time()
out, lse = M.fmha_fwd_with_sink_asm(q, k, v, SCALE, CAUSAL, True)
torch.cuda.synchronize()
res["first_call_s"] = round(time.time() - t0, 2)
res["out_shape"] = list(out.shape)
res["lse_shape"] = list(lse.shape)
res["out_finite"] = bool(torch.isfinite(out).all())
res["lse_finite"] = bool(torch.isfinite(lse).all())
out = out.float()
lse = lse.float()

# --- 2. chunked fp32 reference, out + lse ----------------------------------
g = HQ // HKV
ref_out = torch.empty(B, S, HQ, D, device=dev, dtype=torch.float32)
ref_lse = torch.empty(B, HQ, S, device=dev, dtype=torch.float32)
mask = torch.ones(S, S, dtype=torch.bool, device=dev).tril()
for bi in range(B):
    for h in range(HQ):
        hk = h // g
        qi = q[bi, :, h, :].float()
        ki = k[bi, :, hk, :].float()
        vi = v[bi, :, hk, :].float()
        sc = (qi @ ki.transpose(0, 1)) * SCALE
        sc.masked_fill_(~mask, float("-inf"))
        ref_lse[bi, h, :] = torch.logsumexp(sc, dim=-1)
        p = torch.softmax(sc, dim=-1)
        del sc
        ref_out[bi, :, h, :] = p @ vi
        del p
del mask

res["sqnr_out_db"] = round(sqnr_db(ref_out, out), 3)
res["sqnr_lse_db"] = round(sqnr_db(ref_lse, lse), 3)
res["lse_max_abs_err"] = float((ref_lse - lse).abs().max())
res["lse_allclose_1e-2"] = bool(torch.allclose(ref_lse, lse, rtol=1e-2, atol=1e-2))
# per-head SQNR spread: a gqa-ratio bug shows up as a subset of heads being wrong
per_h = [round(sqnr_db(ref_out[:, :, h, :], out[:, :, h, :]), 2) for h in range(HQ)]
res["sqnr_out_per_head_min"] = min(per_h)
res["sqnr_out_per_head_max"] = max(per_h)
res["sqnr_out_per_head"] = per_h
del ref_out, ref_lse, out, lse
torch.cuda.empty_cache()

print(json.dumps(res), flush=True)
if res["sqnr_out_db"] < 40.0:
    print(json.dumps({"verdict": "NUMERICS FAIL, not timing"}), flush=True)
    sys.exit(2)

# --- 3. timing, same session, same card ------------------------------------
timings = {}
timings["asm_fwd_ms"] = round(
    timed_ms(lambda: M.fmha_fwd_with_sink_asm(q, k, v, SCALE, CAUSAL, True)), 4)
timings["asm_fwd_nolse_ms"] = round(
    timed_ms(lambda: M.fmha_fwd_with_sink_asm(q, k, v, SCALE, CAUSAL, False)), 4)

from aiter.ops.triton.attention.mha import flash_attn_func as aiter_fa  # noqa: E402
timings["aiter_triton_fwd_ms"] = round(
    timed_ms(lambda: aiter_fa(q, k, v, causal=CAUSAL)), 4)

from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType, GlobalBackendManager, PrecisionType)
from primus_turbo.pytorch.ops import flash_attn_func as turbo_fa  # noqa: E402
GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)
qn, kn, vn = q.detach(), k.detach(), v.detach()
timings["turbo_fwd_ms"] = round(timed_ms(lambda: turbo_fa(qn, kn, vn, causal=CAUSAL)), 4)

res["timings_ms"] = timings
print(json.dumps(res), flush=True)
