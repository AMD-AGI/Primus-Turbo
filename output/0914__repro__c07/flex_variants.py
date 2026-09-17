"""flex_attention variant sweep. One variant per process. Prints one json line."""
import argparse, json, os, sys

p = argparse.ArgumentParser()
p.add_argument("--variant", default="base")
p.add_argument("--timer", default="events", choices=["events", "events_noflush", "timeit"])
p.add_argument("--iters", type=int, default=20)
p.add_argument("--warmup", type=int, default=5)
a = p.parse_args()

if os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch
import torch.utils.benchmark as tbench
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128


def timed_ms(fn, iters=a.iters, warmup=a.warmup):
    if a.timer == "timeit":
        for _ in range(warmup * 4):
            fn()
        torch.cuda.synchronize()
        return tbench.Timer(stmt="fn()", globals={"fn": fn}).timeit(100).mean * 1e3
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times, start, end = [], torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        if a.timer == "events":
            flush.zero_()
        torch.cuda.synchronize()
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
hkv = HKV
q = torch.randn(B, HQ, S, D, device=dev, dtype=dt, requires_grad=True)
k = torch.randn(B, hkv, S, D, device=dev, dtype=dt, requires_grad=True)
v = torch.randn(B, hkv, S, D, device=dev, dtype=dt, requires_grad=True)
do = torch.randn(B, HQ, S, D, device=dev, dtype=dt)

causal = lambda b, h, qi, kj: qi >= kj  # noqa: E731
bm = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
kw = dict(block_mask=bm, enable_gqa=True)
compile_kw = dict(dynamic=False)
note = ""

if a.variant == "base":
    pass
elif a.variant == "maxautotune":
    compile_kw["mode"] = "max-autotune-no-cudagraphs"
elif a.variant == "eager":
    compile_kw = None
elif a.variant == "score_mod":
    # causal via score_mod, no block_mask (dense work: no block skipping)
    kw = dict(score_mod=lambda s, b, h, qi, kj: torch.where(qi >= kj, s, -float("inf")),
              enable_gqa=True)
elif a.variant == "expandkv":
    k = k.detach().repeat_interleave(HQ // HKV, dim=1).requires_grad_()
    v = v.detach().repeat_interleave(HQ // HKV, dim=1).requires_grad_()
    kw = dict(block_mask=bm, enable_gqa=False)
elif a.variant == "maxautotune_expandkv":
    k = k.detach().repeat_interleave(HQ // HKV, dim=1).requires_grad_()
    v = v.detach().repeat_interleave(HQ // HKV, dim=1).requires_grad_()
    kw = dict(block_mask=bm, enable_gqa=False)
    compile_kw["mode"] = "max-autotune-no-cudagraphs"
elif a.variant == "warps4":
    kw = dict(block_mask=bm, enable_gqa=True, kernel_options={"num_warps": 4})
elif a.variant == "warps4_fwdonly":
    kw = dict(block_mask=bm, enable_gqa=True, kernel_options={"fwd_num_warps": 4})
elif a.variant == "bm128":
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev,
                           BLOCK_SIZE=128)
    kw = dict(block_mask=bm, enable_gqa=True)
else:
    raise SystemExit("unknown variant " + a.variant)

fa = flex_attention if compile_kw is None else torch.compile(flex_attention, **compile_kw)
fwd = lambda: fa(q, k, v, **kw)  # noqa: E731
out = fwd()
fwd_ms = timed_ms(fwd)


def bwd():
    for t in (q, k, v):
        t.grad = None
    out.backward(do, retain_graph=True)


bwd()
bwd_ms = timed_ms(bwd)

fwd_flops = 2 * B * S * S * HQ * (D + D) // 2
if a.variant == "score_mod":
    fwd_flops *= 2  # no block skipping: full dense score matrix is computed
bwd_flops = fwd_flops * 2.5
tot = fwd_ms + bwd_ms
print(json.dumps({
    "variant": a.variant, "timer": a.timer, "note": note,
    "fwd_ms": round(fwd_ms, 4), "bwd_ms": round(bwd_ms, 4), "total_ms": round(tot, 4),
    "fwd_tflops": round(fwd_flops / (fwd_ms * 1e-3) / 1e12, 1),
    "bwd_tflops": round(bwd_flops / (bwd_ms * 1e-3) / 1e12, 1),
    "total_tflops": round((fwd_flops + bwd_flops) / (tot * 1e-3) / 1e12, 1),
    "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
}))
