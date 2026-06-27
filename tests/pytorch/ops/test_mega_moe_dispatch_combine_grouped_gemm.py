###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 end-to-end MoE test for the fused FlyDSL mega kernels.

Compares the fully fused mega pipeline against a turbo (DeepEP) baseline on an
EP8 expert-parallel forward, for both load-balanced and round-robin routing.

The mega pipeline wires four FlyDSL kernels over HIP-IPC symmetric memory:

  1. dispatch_prologue        -- build the cross-rank dispatch plan from topk
  2. dispatch_grouped_gemm    -- cross-rank dispatch PUSH + grouped L1 GEMM (NT)
  3. swiglu (epilogue)        -- fused SwiGLU activation
  4. grouped_gemm_combine     -- grouped L2 GEMM (NT) + cross-rank combine PUSH

The routing weight is applied at the final combine reduction (W2 is linear, so
weighting the down-proj output equals weighting its input).

Run inside dev_primus (8 GPUs):
  PYTHONPATH=<...>/Primus-Turbo python \
      tests/pytorch/ops/test_mega_moe_dispatch_combine_grouped_gemm.py \
      --num-processes 8 [--perf] [--mode load_balanced|round_robin|both]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch.nn.functional as F  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402

# fused mega MoE forward + single-allocation symmetric buffer (implementation under test)
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1) Benchmark helper: warmup, then an L2-cache flush before each timed iter.
# ─────────────────────────────────────────────────────────────────────────────
_L2_FLUSH_BUF = None


def _l2_flush():
    """Evict the L2 cache by overwriting a large scratch buffer (~256 MB)."""
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def bench_kineto(fn, *, warmup=5, num_tests=20, group=None, flush_l2=True):
    """Profile ``fn`` and return per-GPU-kernel timings (mirrors deep_ep's bench_kineto).

    The L2 cache is flushed before every captured call (the flush is a memset, which
    is filtered out of the kernel aggregation below). Returns a list of
    (kernel_name, avg_us_per_call) for every CUDA kernel, sorted by total time desc.
    All ranks must drive the profiler (fn is collective); callers print on rank 0 only.
    """
    for _ in range(warmup):
        fn()

    if flush_l2:
        _l2_flush()  # cold L2 before each call (memset, excluded from breakdown)

    torch.cuda.synchronize()
    if group is not None:
        group.barrier()

    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for _ in range(2):  # one wait step, then one active step capturing num_tests calls
            for _ in range(num_tests):
                fn()
            torch.cuda.synchronize()
            prof.step()

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        prof.export_chrome_trace(tmp.name)
        events = json.loads(Path(tmp.name).read_text())["traceEvents"]

    # aggregate device-side kernel durations AND launch counts by name over num_tests calls
    agg: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for ev in events:
        if ev.get("cat") in ("kernel", "Kernel") and "dur" in ev:
            agg[ev["name"]] = agg.get(ev["name"], 0.0) + float(ev["dur"])
            cnt[ev["name"]] = cnt.get(ev["name"], 0) + 1
    # (name, avg_us_per_iter, avg_launches_per_iter)
    breakdown = [(name, dur / num_tests, cnt[name] / num_tests) for name, dur in agg.items()]
    breakdown.sort(key=lambda t: -t[1])
    return breakdown


def _diff_breakdown(total_breakdown, base_breakdown, *, eps=0.05):
    """Per-kernel backward time, isolating backward = (fwd+bwd) - fwd.

    Backward is gated by LAUNCH COUNT, not time: a kernel is a backward kernel
    only if it launches MORE times in fwd+bwd than in fwd-only. This drops
    forward-only kernels (e.g. prologue_k_0, swiglu_kernel_0) that backward never
    calls but whose per-call device time inflates in the fwd+bwd run (each iter's
    prologue stalls on the prior iter's backward), which a pure time-diff would
    otherwise mis-attribute to backward. For count-confirmed backward kernels the
    reported time is still the time diff (approximate: the two runs profile the
    forward independently), so small diffs are noise."""
    base_us = {name: us for name, us, _ in base_breakdown}
    base_n = {name: n for name, _, n in base_breakdown}
    diff_breakdown = []
    for name, us, n in total_breakdown:
        bwd_launches = n - base_n.get(name, 0.0)
        if bwd_launches < 0.5:  # not launched (more) by backward -> forward-only, drop
            continue
        bwd_us = us - base_us.get(name, 0.0)
        if bwd_us > eps:
            diff_breakdown.append((name, bwd_us, bwd_launches))
    diff_breakdown.sort(key=lambda t: -t[1])
    return diff_breakdown


def _print_breakdown(title, breakdown):
    # breakdown rows are (name, us[, launches_per_iter]); launches optional
    total_us = sum(row[1] for row in breakdown) or 1.0
    print(title)
    for row in breakdown:
        name, us = row[0], row[1]
        n = row[2] if len(row) > 2 else 1.0
        print(f"  {us:9.3f} us  {100.0 * us / total_us:6.2f}%  x{n:4.1f}  {name[:74]}")
    print(f"  {total_us:9.3f} us  100.00%         TOTAL")


# ─────────────────────────────────────────────────────────────────────────────
# 2) Test-data generator: load_balanced and round_robin routing.
# ─────────────────────────────────────────────────────────────────────────────
def generate_routing(num_tokens, num_topk, num_experts, mode, *, device="cuda", seed=0):
    """Return (topk_idx[T,K] int64, topk_w[T,K] f32) for the requested routing mode.

    load_balanced: softmax-topk of random positive scores (abs + 1).
    round_robin:   experts assigned by a contiguous arange % num_experts pattern.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "load_balanced":
        scores = torch.rand(num_tokens, num_experts, generator=g, device=device).abs() + 1
        topk_w, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1)
        topk_idx = topk_idx.to(torch.int64)
        topk_w = topk_w.to(torch.float32)
    elif mode == "round_robin":
        topk_idx = (
            torch.arange(num_tokens * num_topk, device=device).view(num_tokens, num_topk) % num_experts
        ).to(torch.int64)
        # routing weights: softmax over per-token random scores for the K slots
        topk_w = torch.rand(num_tokens, num_topk, generator=g, device=device).softmax(-1).to(torch.float32)
    else:
        raise ValueError(f"unknown routing mode: {mode}")
    return topk_idx, topk_w


def _global_weights(E, I, H, device):
    """Deterministic global expert weights, identical on every rank (then sliced).

    Init scaled by 1/sqrt(K_contract) so activation magnitudes are O(1) at any shape
    (std(acc1) ~ 2) -- keeps the mega kernel's SwiGLU clamp inert, so the unclamped
    turbo baseline and the clamped mega agree."""
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


# ─────────────────────────────────────────────────────────────────────────────
# 3) Baseline reference: turbo DeepEP dispatch + grouped_gemm + SwiGLU + combine.
# ─────────────────────────────────────────────────────────────────────────────
def make_baseline_reference(group, *, num_experts, num_topk, hidden, inter, W1, W2):
    """Build the canonical turbo EP-MoE forward (mirrors _turbo_ep_moe in test_overlap_e2e):
    DeepEP dispatch -> grouped_gemm -> SwiGLU -> grouped_gemm -> DeepEP combine.

    ``gate_logits`` is the precomputed [T, E] probs (token_dispatch gathers token_probs
    at ``indices`` and carries them as permuted_probs; combine applies no weight), built
    once by the caller to keep the per-step path scatter-free."""
    dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=num_topk,
        ep_group=group,
        permute_fusion=True,
        deepep_num_use_cu=80,
    )
    fc1, fc2 = W1, W2  # [epr, 2I, H], [epr, H, I]

    def _turbo_forward(x, topk_idx, gate_logits, w1, w2):
        permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
            x, gate_logits, indices=topk_idx
        )
        group_lens = tokens_per_expert.to(device=x.device, dtype=torch.int64)
        fc1_out = _turbo_gg(permuted_hidden, w1, group_lens, trans_b=True)
        gate, up = fc1_out.chunk(2, dim=-1)
        inter = (F.silu(gate.float()) * up.float() * permuted_probs.unsqueeze(-1)).to(x.dtype)
        fc2_out = _turbo_gg(inter, w2, group_lens, trans_b=True)
        return dispatcher.token_combine(fc2_out)

    def baseline_reference(x, topk_idx, gate_logits):
        return _turbo_forward(x, topk_idx, gate_logits, fc1, fc2)

    # grad-enabled forward (leaf weights) for backward timing
    def baseline_grad_forward(x, topk_idx, gate_logits):
        x_leaf = x.detach().requires_grad_(True)
        w1_leaf = fc1.detach().requires_grad_(True)
        w2_leaf = fc2.detach().requires_grad_(True)
        return _turbo_forward(x_leaf, topk_idx, gate_logits, w1_leaf, w2_leaf)

    return baseline_reference, baseline_grad_forward


def _gate3(g, ref):
    g, r = g.float().flatten(), ref.float().flatten()
    cos = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
    rel = float((g - r).norm() / (r.norm() + 1e-12))
    return cos, rel, (cos >= 0.99 and rel <= 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Per-process entry.
# ─────────────────────────────────────────────────────────────────────────────
def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8407"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    epr = E // world
    assert E % world == 0, "num_experts must be divisible by world_size"
    assert K <= E, "num_topk cannot exceed num_experts"

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]

    # global weights (shared across ranks), sliced to this rank's experts
    W1g, W2g = _global_weights(E, I, H, "cuda")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous()

    # turbo baseline + mega fused symmetric buffer (allocate once, reuse per step)
    baseline_reference, baseline_grad_forward = make_baseline_reference(
        group, num_experts=E, num_topk=K, hidden=H, inter=I, W1=W1, W2=W2
    )
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        block_m=args.bm,
        block_n=args.bn,
    )

    for mode in modes:
        torch.manual_seed(7 + rank)
        x = (torch.randn((T, H), device="cuda", dtype=torch.float32)).bfloat16()
        topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)
        # turbo needs a [T, E] probs; build it once (out of the timed path), like the ref
        gate_logits = torch.zeros(T, E, dtype=torch.float32, device="cuda")
        gate_logits.scatter_(1, topk_idx, topk_w)

        # warmup both (no autograd anywhere in this test)
        # mega_moe_fused fetches the cached symmetric buffer internally (same key as `symm`)
        with torch.no_grad():
            y_mega = mega_moe_fused(group, x, topk_idx, topk_w, W1, W2, block_m=args.bm, block_n=args.bn)
            y_turbo = baseline_reference(x, topk_idx, gate_logits)
        torch.cuda.synchronize()
        group.barrier()
        symm.assert_capacity()  # fail loudly rather than silently drop rows

        # ---- correctness: mega vs turbo (DeepEP) baseline ----
        res = {
            "mega_vs_turbo": _gate3(y_mega, y_turbo),
        }
        ok = res["mega_vs_turbo"][2]
        gathered = [None] * world
        dist.all_gather_object(gathered, (rank, mode, res, ok), group=group)
        if rank == 0:
            print(f"\n[{mode}] gate-3 (cos / rel_rmse / ok):")
            for r, _mode, rr, _ok in sorted(gathered, key=lambda t: t[0]):
                line = "  ".join(f"{n}: {v[0]:.5f}/{v[1]:.4f}/{v[2]}" for n, v in rr.items())
                print(f"  rank={r}  {line}")
            print(f"[{mode}]", "PASS" if all(g[3] for g in gathered) else "FAIL")
        # every rank asserts the global verdict -> a failure propagates through spawn
        assert all(g[3] for g in gathered), f"[{mode}] gate-3 FAILED"

        # ---- perf: per-kernel breakdown of the mega forward + backward (kineto) ----
        # backward is isolated by subtraction: profile the (grad-enabled) forward and
        # forward+backward, then diff. The same forward closure feeds both passes, so
        # forward kernels (incl. the save_for_backward clones) cancel exactly.
        if args.perf:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

            def _forward():
                x_leaf = x.detach().requires_grad_(True)
                w1_leaf = W1.detach().requires_grad_(True)
                w2_leaf = W2.detach().requires_grad_(True)
                return mega_moe_fused(
                    group, x_leaf, topk_idx, topk_w, w1_leaf, w2_leaf, block_m=args.bm, block_n=args.bn
                )

            fwd_breakdown = bench_kineto(_forward, num_tests=args.perf_iters, group=group)
            fwdbwd_breakdown = bench_kineto(
                lambda: _forward().backward(grad_y), num_tests=args.perf_iters, group=group
            )
            if rank == 0:
                bwd_breakdown = _diff_breakdown(fwdbwd_breakdown, fwd_breakdown)
                _print_breakdown(f"[{mode}] mega FORWARD per-kernel breakdown (us/call, %):", fwd_breakdown)
                _print_breakdown(f"[{mode}] mega BACKWARD per-kernel breakdown (us/call, %):", bwd_breakdown)

            # wall-clock e2e (captures host-sync/barrier cost the kineto kernel-sum filters out)
            import time as _time

            def _walltime(fn, iters=args.perf_iters, warmup=5):
                for _ in range(warmup):
                    fn()

                _l2_flush()
                torch.cuda.synchronize()
                group.barrier()
                t0 = _time.perf_counter()
                for _ in range(iters):
                    fn()
                torch.cuda.synchronize()
                return (_time.perf_counter() - t0) / iters * 1e6  # us/iter

            def _turbo_forward_grad():
                return baseline_grad_forward(x, topk_idx, gate_logits)

            # mega wall-clock
            wt_fwd = _walltime(_forward)
            wt_fb = _walltime(lambda: _forward().backward(grad_y))
            # turbo baseline wall-clock
            tb_fwd = _walltime(lambda: baseline_reference(x, topk_idx, gate_logits))
            tb_fb = _walltime(lambda: _turbo_forward_grad().backward(grad_y))
            if rank == 0:
                wt_bwd, tb_bwd = wt_fb - wt_fwd, tb_fb - tb_fwd
                print(
                    f"[{mode}] WALL-CLOCK us/iter (mega): fwd={wt_fwd:.1f}  fwd+bwd={wt_fb:.1f}  bwd={wt_bwd:.1f}"
                )
                print(
                    f"[{mode}] WALL-CLOCK us/iter (turbo): fwd={tb_fwd:.1f}  fwd+bwd={tb_fb:.1f}  bwd={tb_bwd:.1f}"
                )
                print(
                    f"[{mode}] SPEEDUP (turbo/mega): fwd={tb_fwd / wt_fwd:.3f}x  "
                    f"bwd={tb_bwd / wt_bwd:.3f}x  e2e={tb_fb / wt_fb:.3f}x"
                )
        torch.cuda.synchronize()
        group.barrier()

    symm.destroy()
    dist.destroy_process_group()


def _build_argparser():
    # Defaults: DeepSeek-V3 BF16, EP8, 8192 tokens/rank
    # (H=7168, I=2048, E=256 routed, topk=8 -> 32 experts/rank).
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--num-topk", type=int, default=8)
    ap.add_argument("--num-tokens", type=int, default=8192)
    ap.add_argument("--bm", type=int, default=256)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--mode", choices=["load_balanced", "round_robin", "both"], default="both")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--perf-iters", type=int, default=50)
    return ap


# ─────────────────────────────────────────────────────────────────────────────
# pytest entry: spawns the EP8 group; a child AssertionError propagates here.
# ─────────────────────────────────────────────────────────────────────────────
_WORLD = 8


@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs for EP{_WORLD}")
def test_mega_moe_dispatch_combine_grouped_gemm():
    args = _build_argparser().parse_args(
        [
            "--num-tokens",
            "1024",
            "--num-topk",
            "4",
            "--num-experts",
            "32",
            "--hidden",
            "2048",
            "--inter",
            "1024",
            "--mode",
            "both",
        ]
    )
    # spawn re-raises any child exception (gate-3 / capacity asserts) in the parent
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)
