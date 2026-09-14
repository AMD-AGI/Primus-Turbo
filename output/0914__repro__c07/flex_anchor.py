"""torch flex_attention anchor, same shape/methodology as tools/gfx1250/tune_attention.py.

Timing, FLOP accounting and the fwd/bwd split are copied from that harness so the number
lands on the same ladder: CUDA-event timed, median of `iters`, L2 flushed between reps.
"""
import json, os, sys

if "--gpu" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
elif os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
ITERS, WARMUP = 20, 5


def timed_ms(fn, iters=ITERS, warmup=WARMUP):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times, start, end = [], torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()
        torch.cuda.synchronize()
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    # flex wants [B, H, S, D]; the turbo harness generates [B, S, H, D].
    q = torch.randn(B, HQ, S, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, HKV, S, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, HKV, S, D, device=dev, dtype=dt, requires_grad=True)
    do = torch.randn(B, HQ, S, D, device=dev, dtype=dt)

    causal = lambda b, h, qi, kj: qi >= kj  # noqa: E731
    block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device=dev)
    fa = torch.compile(flex_attention, dynamic=False)

    # Inductor's default heuristic picks num_warps=8 for the flex backward template on this
    # part; its own autotuner picks 4, which is 2.4x faster at identical tiles. Left as an
    # option rather than hardcoded so the default path stays inspectable.
    ko = {}
    if "--num-warps" in sys.argv:
        ko["num_warps"] = int(sys.argv[sys.argv.index("--num-warps") + 1])

    fwd = lambda: fa(q, k, v, block_mask=block_mask, enable_gqa=True, kernel_options=ko)  # noqa: E731
    out = fwd()
    fwd_ms = timed_ms(fwd)

    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(do, retain_graph=True)

    bwd()
    bwd_ms = timed_ms(bwd)

    fwd_flops = 2 * B * S * S * HQ * (D + D) // 2  # causal
    bwd_flops = fwd_flops * 2.5
    total = fwd_ms + bwd_ms
    print(json.dumps({
        "impl": "torch-flex", "batch": B, "seqlen": S, "hq": HQ, "hkv": HKV, "head_dim": D,
        "gpu": os.environ.get("HIP_VISIBLE_DEVICES", "all"),
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        # Not gated here: see flex_correct.py, which runs the same four-tensor fp32
        # SQNR check the turbo harness uses (out 53.67 / dq 52.23 / dk 52.29 / dv 52.71).
        "correct": None, "kernel_options": ko, "fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "total_ms": total,
        "fwd_tflops": fwd_flops / (fwd_ms * 1e-3) / 1e12,
        "bwd_tflops": bwd_flops / (bwd_ms * 1e-3) / 1e12,
        "total_tflops": (fwd_flops + bwd_flops) / (total * 1e-3) / 1e12,
        "bwd_share": bwd_ms / total,
        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
