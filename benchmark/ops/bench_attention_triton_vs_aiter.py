"""Direct comparison of Triton vs AITER attention backends on BF16."""
import torch
import torch.utils.benchmark as benchmark

from primus_turbo.pytorch.ops.attention.flash_attn_interface import (
    AiterFlashAttnFunc,
    TritonFlashAttnFunc,
)

CONFIGS = [
    # (batch, seqlen, nheads_q, nheads_kv, head_dim_qk, head_dim_v, causal)
    (2, 2048, 32, 32, 128, 128, True),
    (2, 4096, 32, 32, 128, 128, True),
    (2, 4096, 32, 32, 128, 128, False),
    (2, 8192, 32, 32, 128, 128, True),
    (2, 4096, 64, 8, 128, 128, True),
    (4, 4096, 48, 8, 128, 128, True),
    (1, 4096, 64, 64, 192, 128, True),
]


def run_benchmark():
    device = "cuda"
    dtype = torch.bfloat16

    print(f"{'Config':<55} {'Triton Fwd':>12} {'AITER Fwd':>12} {'Ratio':>8}")
    print("-" * 95)

    for B, S, Hq, Hkv, Dqk, Dv, causal in CONFIGS:
        sm_scale = Dqk**-0.5
        q = torch.randn(B, S, Hq, Dqk, device=device, dtype=dtype, requires_grad=False)
        k = torch.randn(B, S, Hkv, Dqk, device=device, dtype=dtype, requires_grad=False)
        v = torch.randn(B, S, Hkv, Dv, device=device, dtype=dtype, requires_grad=False)

        fwd_flops = 2 * B * S * S * Hq * (Dqk + Dv)
        if causal:
            fwd_flops //= 2

        triton_fn = lambda: TritonFlashAttnFunc.apply(
            q, k, v, 0.0, sm_scale, causal, (-1, -1), None, None, False, False, False, False,
        )
        aiter_fn = lambda: AiterFlashAttnFunc.apply(
            q, k, v, 0.0, sm_scale, causal, (-1, -1), None, None, False, False, False, False, None, 1, None,
        )

        for _ in range(5):
            triton_fn()
            aiter_fn()
        torch.cuda.synchronize()

        t_triton = benchmark.Timer(stmt="fn()", globals={"fn": triton_fn}).timeit(50).mean * 1e3
        t_aiter = benchmark.Timer(stmt="fn()", globals={"fn": aiter_fn}).timeit(50).mean * 1e3

        tflops_triton = fwd_flops / (t_triton * 1e-3) / 1e12
        tflops_aiter = fwd_flops / (t_aiter * 1e-3) / 1e12
        ratio = tflops_triton / tflops_aiter

        tag = f"B={B} S={S} H={Hq}/{Hkv} D={Dqk}/{Dv} causal={causal}"
        print(f"{tag:<55} {tflops_triton:>10.1f}T {tflops_aiter:>10.1f}T {ratio:>7.1%}")


if __name__ == "__main__":
    run_benchmark()
