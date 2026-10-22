###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU grouped GEMM baseline at the shapes one BF16 Mega MoE rank runs.

Per rank Mega runs two grouped GEMMs over its local experts: GateUp ``[M_e, H] x [2I, H]^T`` and
Down ``[M_e, I] x [H, I]^T``. The forward is an NT grouped GEMM, the backward an NN (dgrad) plus a
variable-K (wgrad) one -- the three kernel families Mega's GEMM tiles have to be built from.

Each case reports forward and backward time / TFLOPS per backend after an allclose check against a
per-expert torch matmul, plus one large hipBLASLt ``torch.mm`` as a practical dense-bf16 ceiling.

Usage::

    python tools/bench_mega_gemm_gfx1250.py                    # DeepSeek-V3, EP4 and EP8
    python tools/bench_mega_gemm_gfx1250.py --ep 4 --m 1024 -o mega_gemm.csv
"""

import argparse
import sys
from pathlib import Path

import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark" / "ops" / "training"))
from config import (  # noqa: E402
    MoEModelConfigs,
    check_allclose,
    gen_grouped_gemm_group_lens,
    grouped_gemm_ref,
)

_BACKENDS = {"flydsl": BackendType.FLYDSL, "triton": BackendType.TRITON}


def _time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(start, end):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(start, end))[iters // 2]


def _dense_ceiling(warmup, iters):
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    ms = _time_ms(lambda: torch.mm(a, b), warmup, iters)
    return 2 * 8192**3 / (ms * 1e-3) / 1e12


def _run_case(backend, B, M, N, K, balance, warmup, iters):
    torch.manual_seed(0)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=balance).to("cuda")
    x = torch.randn((B * M, K), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = torch.randn((B, N, K), dtype=torch.bfloat16, device="cuda", requires_grad=True) / K**0.5
    w = w.detach().requires_grad_()
    GlobalBackendManager.set_grouped_gemm_backend(backend)
    try:
        out = turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True)
        grad_out = torch.randn_like(out)
        out.backward(grad_out, retain_graph=True)
        x_ref = x.detach().clone().requires_grad_()
        w_ref = w.detach().clone().requires_grad_()
        out_ref = grouped_gemm_ref(x_ref, w_ref, group_lens, trans_b=True)
        out_ref.backward(grad_out)
        ok = (
            check_allclose(out.detach(), out_ref.detach(), torch.bfloat16)
            and check_allclose(x.grad, x_ref.grad, torch.bfloat16)
            and check_allclose(w.grad, w_ref.grad, torch.bfloat16)
        )
        fwd_ms = _time_ms(lambda: turbo.ops.grouped_gemm(x, w, group_lens, trans_b=True), warmup, iters)

        def backward():
            # fresh grads each call: accumulating into .grad would time a bf16 add kernel as well
            x.grad = w.grad = None
            out.backward(grad_out, retain_graph=True)

        bwd_ms = _time_ms(backward, warmup, iters)
    finally:
        GlobalBackendManager.set_grouped_gemm_backend(None)
    flops = 2 * B * M * N * K
    return ok, fwd_ms, flops / (fwd_ms * 1e-3) / 1e12, bwd_ms, 2 * flops / (bwd_ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="DeepSeek-V3", choices=sorted(MoEModelConfigs))
    parser.add_argument("--ep", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--m", type=int, nargs="+", default=[256, 512, 1024, 2048], help="tokens per expert")
    parser.add_argument("--backends", nargs="+", default=list(_BACKENDS), choices=list(_BACKENDS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("-o", "--output", default=None, help="CSV path")
    args = parser.parse_args()

    cfg = MoEModelConfigs[args.model]
    H, I, E = cfg["hidden_size"], cfg["moe_intermediate_size"], cfg["n_routed_experts"]
    print(f"device: {torch.cuda.get_device_name()} ({torch.cuda.get_device_properties(0).gcnArchName})")
    print(f"dense bf16 ceiling (hipBLASLt 8192^3): {_dense_ceiling(args.warmup, args.iters):.1f} TFLOPS")

    rows = []
    header = f"{'case':7s} {'EP':>3s} {'B':>4s} {'M':>5s} {'dist':5s} {'backend':7s} {'check':5s} " + (
        f"{'fwd ms':>8s} {'fwd TF':>7s} {'bwd ms':>8s} {'bwd TF':>7s}"
    )
    print(header)
    for ep in args.ep:
        B = E // ep
        for M in args.m:
            for case, (N, K) in (("GateUp", (2 * I, H)), ("Down", (H, I))):
                for balance in (True, False):
                    for name in args.backends:
                        ok, fwd_ms, fwd_tf, bwd_ms, bwd_tf = _run_case(
                            _BACKENDS[name], B, M, N, K, balance, args.warmup, args.iters
                        )
                        dist = "bal" if balance else "skew"
                        check = "PASS" if ok else "FAIL"
                        print(
                            f"{case:7s} {ep:3d} {B:4d} {M:5d} {dist:5s} {name:7s} {check:5s} "
                            f"{fwd_ms:8.3f} {fwd_tf:7.1f} {bwd_ms:8.3f} {bwd_tf:7.1f}",
                            flush=True,
                        )
                        rows.append(
                            dict(
                                case=case, ep=ep, B=B, M=M, N=N, K=K, dist=dist, backend=name, check=check,
                                fwd_ms=fwd_ms, fwd_tflops=fwd_tf, bwd_ms=bwd_ms, bwd_tflops=bwd_tf,
                            )
                        )
    if args.output:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
