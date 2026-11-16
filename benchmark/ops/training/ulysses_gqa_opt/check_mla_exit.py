"""Correctness of the exit ops (A2AGemm o_proj fwd, GemmA2A dO + dW_o bwd) on MLA widths.

torchrun --nproc_per_node 8 check_mla_exit.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(
    0,
    os.environ.get(
        "TURBO_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
    ),
)
from tests.pytorch.ops.test_ulysses_fused_a2a import check_o_gemm

# name: (heads, v head dim, hidden)
MODELS = {"deepseek-v3": (128, 128, 7168), "kimi-k2": (64, 128, 7168)}


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    failures = []
    for name, (hq, dv, hidden) in MODELS.items():
        # 300: not a multiple of any tile; 512 / 2048: whole tiles, one / several waves of dw tiles
        cases = [(300, "sbhd", True), (512, "sbhd", True), (2048, "bshd", True), (512, "sbhd", False)]
        if "--long" in sys.argv:
            # CP4 / CP2 at 128K
            cases += [(32768, "sbhd", True), (65536 // max(1, n // 2), "sbhd", True)]
        for s_local, fmt, res in cases:
            before = len(failures)
            check_o_gemm(group, hq, dv, 1, s_local, hidden, fmt, res, 2, failures)
            if rank == 0:
                status = "ok" if len(failures) == before else "FAIL"
                print(f"{name} hq={hq} s/n={s_local} {fmt} residual={res}: {status}", flush=True)
    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if rank == 0:
        bad = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(bad) if bad else f"mla exit: all cases passed (n={n})")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if not any(gathered) else 1)


if __name__ == "__main__":
    main()
