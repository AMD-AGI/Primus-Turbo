###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 end-to-end MoE test: fused FlyDSL mega pipeline vs turbo (DeepEP) baseline, both routing modes."""

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

# Megatron-LM lives under the sibling Primus repo
_MEGATRON_LM = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Primus", "third_party", "Megatron-LM")
)
if _MEGATRON_LM not in sys.path:
    sys.path.insert(0, _MEGATRON_LM)

from megatron.core.fusions.fused_bias_swiglu import (  # noqa: E402
    weighted_bias_swiglu_impl,
)
from megatron.core.tensor_parallel.random import (  # noqa: E402
    get_cuda_rng_tracker,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.transformer.moe.moe_utils import apply_random_logits  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402

# fused mega MoE forward + symmetric buffer (under test)
from primus_turbo.flydsl.mega.symm_buffer import (  # noqa: E402
    get_symm_buffer_for_mega_moe,
)
from primus_turbo.pytorch.ops import grouped_gemm as _turbo_gg  # noqa: E402
from primus_turbo.pytorch.ops.moe.mega_moe_fused import mega_moe_fused  # noqa: E402

# 1) Benchmark helper: warmup, then L2-cache flush before each timed iter.
_L2_FLUSH_BUF = None


def _l2_flush():
    """Evict the L2 cache by overwriting a large scratch buffer (~256 MB)."""
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None:
        _L2_FLUSH_BUF = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    _L2_FLUSH_BUF.zero_()


def bench_kineto(fn, *, warmup=5, num_tests=20, group=None, flush_l2=True):
    """Profile fn, return per-kernel (name, avg_us_per_call) sorted by total time desc."""
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

    # aggregate kernel durations and launch counts by name
    agg: dict[str, float] = {}
    cnt: dict[str, int] = {}
    # include memcpy but not memset (l2_flush memset excluded)
    for ev in events:
        if ev.get("cat") in ("kernel", "Kernel", "gpu_memcpy") and "dur" in ev:
            agg[ev["name"]] = agg.get(ev["name"], 0.0) + float(ev["dur"])
            cnt[ev["name"]] = cnt.get(ev["name"], 0) + 1
    # (name, avg_us_per_iter, avg_launches_per_iter)
    breakdown = [(name, dur / num_tests, cnt[name] / num_tests) for name, dur in agg.items()]
    breakdown.sort(key=lambda t: -t[1])
    return breakdown


def bench_kineto_dist(fn, *, warmup=5, num_tests=50, group=None, flush_l2=False):
    """Profile fn as a continuous loop (no flush/sync, training-like); return {kernel: [dur_us,...]}."""
    for _ in range(warmup):
        fn()
    if flush_l2:
        _l2_flush()
    torch.cuda.synchronize()
    if group is not None:
        group.barrier()

    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for _ in range(2):  # wait step, then active step capturing the continuous loop
            for _ in range(num_tests):
                fn()
            torch.cuda.synchronize()
            prof.step()

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        prof.export_chrome_trace(tmp.name)
        events = json.loads(Path(tmp.name).read_text())["traceEvents"]

    dist_by_name: dict[str, list] = {}
    for ev in events:
        if ev.get("cat") in ("kernel", "Kernel", "gpu_memcpy") and "dur" in ev:
            dist_by_name.setdefault(ev["name"], []).append(float(ev["dur"]))
    return dist_by_name


def _pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def _print_dist(title, dist_by_name, *, top=20):
    # sort kernels by total time (sum) desc; show the per-call spread
    rows = sorted(dist_by_name.items(), key=lambda kv: -sum(kv[1]))
    print(title)
    print(f"  {'min':>9} {'avg':>9} {'p50':>9} {'p90':>9} {'p99':>9} {'max':>9} {'n':>5}  kernel")
    for name, vals in rows[:top]:
        avg = sum(vals) / len(vals)
        print(
            f"  {min(vals):9.1f} {avg:9.1f} {_pct(vals, 0.5):9.1f} {_pct(vals, 0.9):9.1f} "
            f"{_pct(vals, 0.99):9.1f} {max(vals):9.1f} {len(vals):5d}  {name[:60]}"
        )


def _diff_breakdown(total_breakdown, base_breakdown, *, eps=0.05):
    """Per-kernel backward time = (fwd+bwd) - fwd, gated by launch count to drop fwd-only kernels."""
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


# 2) Test-data generator: load_balanced and round_robin routing.
def generate_routing(num_tokens, num_topk, num_experts, mode, *, device="cuda", seed=0):
    """Return (topk_idx[T,K] int64, topk_w[T,K] f32) for load_balanced or round_robin routing."""
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "load_balanced":
        # Mirror Megatron moe_router_force_load_balancing: random normal logits.
        logits = torch.empty(num_tokens, num_experts, device=device, dtype=torch.float32)
        logits = apply_random_logits(logits)
        topk_w, topk_idx = torch.topk(logits.softmax(-1), num_topk, dim=-1)
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
    """Deterministic global expert weights (same on every rank), scaled 1/sqrt(K) to keep SwiGLU clamp inert."""
    g = torch.Generator(device=device).manual_seed(1234)
    W1 = torch.randn((E, 2 * I, H), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
    W2 = torch.randn((E, H, I), generator=g, device=device, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
    return W1, W2


# 3) Baseline reference: turbo DeepEP dispatch + grouped_gemm + SwiGLU + combine.
def make_baseline_reference(group, *, num_experts, num_topk, hidden, inter, W1, W2):
    """Build the turbo EP-MoE forward: DeepEP dispatch -> grouped_gemm -> SwiGLU -> grouped_gemm -> combine."""
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
        inter = weighted_bias_swiglu_impl(fc1_out, None, permuted_probs.unsqueeze(-1)).to(x.dtype)
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

    # grad-enabled forward exposing the leaves so callers can read their grads (gradcheck)
    def baseline_grad_forward_leaves(x, topk_idx, gate_logits):
        x_leaf = x.detach().requires_grad_(True)
        w1_leaf = fc1.detach().requires_grad_(True)
        w2_leaf = fc2.detach().requires_grad_(True)
        y = _turbo_forward(x_leaf, topk_idx, gate_logits, w1_leaf, w2_leaf)
        return y, x_leaf, w1_leaf, w2_leaf

    return baseline_reference, baseline_grad_forward, baseline_grad_forward_leaves


def _gate3(g, ref):
    g, r = g.float().flatten(), ref.float().flatten()
    cos = float(torch.dot(g, r) / (g.norm() * r.norm() + 1e-12))
    rel = float((g - r).norm() / (r.norm() + 1e-12))
    return cos, rel, (cos >= 0.99 and rel <= 0.05)


# Per-process entry.
def _run(local_rank, world, args):
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8407"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", init_method=f"tcp://{ip}:{port}", world_size=world, rank=local_rank)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    group = dist.new_group(list(range(world)))
    rank = dist.get_rank()

    # register rank-specific EP RNG state for apply_random_logits()
    _tracker = get_cuda_rng_tracker()
    _ep_rng = get_expert_parallel_rng_tracker_name()
    if _ep_rng not in _tracker.get_states():
        _tracker.add(_ep_rng, 1024 + 100 * rank)

    H, I, E, K, T = args.hidden, args.inter, args.num_experts, args.num_topk, args.num_tokens
    epr = E // world
    assert E % world == 0, "num_experts must be divisible by world_size"
    assert K <= E, "num_topk cannot exceed num_experts"

    modes = ["load_balanced", "round_robin"] if args.mode == "both" else [args.mode]

    # global weights on CPU, only this rank's slice to GPU (full set OOMs)
    W1g, W2g = _global_weights(E, I, H, "cpu")
    W1 = W1g[rank * epr : (rank + 1) * epr].contiguous().cuda()
    W2 = W2g[rank * epr : (rank + 1) * epr].contiguous().cuda()
    # keep global weights on CPU for the analytic dtw ref (bwd only)
    if not args.bwd:
        del W1g, W2g

    # turbo baseline + mega fused symmetric buffer (allocate once, reuse per step)
    baseline_reference, baseline_grad_forward, baseline_grad_forward_leaves = make_baseline_reference(
        group, num_experts=E, num_topk=K, hidden=H, inter=I, W1=W1, W2=W2
    )
    symm = get_symm_buffer_for_mega_moe(
        group,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
    )

    for mode in modes:
        torch.manual_seed(7 + rank)
        x = (torch.randn((T, H), device="cuda", dtype=torch.float32)).bfloat16()
        topk_idx, topk_w = generate_routing(T, K, E, mode, device="cuda", seed=100 + rank)
        # turbo needs [T, E] probs; build once out of the timed path
        gate_logits = torch.zeros(T, E, dtype=torch.float32, device="cuda")
        gate_logits.scatter_(1, topk_idx, topk_w)

        # warmup both (mega_moe_fused fetches the cached symm buffer internally)
        with torch.no_grad():
            y_mega = mega_moe_fused(group, x, topk_idx, topk_w, W1, W2)
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

        # ---- backward gradcheck: mega grads vs the turbo baseline (same grad_y) ----
        if args.bwd:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
            # mega fused grads (x/W1/W2/topk_w leaves)
            x_m = x.detach().requires_grad_(True)
            w1_m = W1.detach().requires_grad_(True)
            w2_m = W2.detach().requires_grad_(True)
            tw_m = topk_w.detach().requires_grad_(True)
            y_m = mega_moe_fused(group, x_m, topk_idx, tw_m, w1_m, w2_m)
            dx_m, dW1_m, dW2_m, dtw_m = torch.autograd.grad(y_m, [x_m, w1_m, w2_m, tw_m], grad_y)
            # turbo baseline grads (same grad_y, same local shard)
            y_t, x_t, w1_t, w2_t = baseline_grad_forward_leaves(x, topk_idx, gate_logits)
            dx_t, dW1_t, dW2_t = torch.autograd.grad(y_t, [x_t, w1_t, w2_t], grad_y)
            # analytic dtw ref: dtw[t,k] = <grad_y[t], o[t,k]>, o = unweighted route output
            dtw_ref = torch.zeros_like(topk_w)
            gyf = grad_y.float()
            for e in range(E):
                pos = (topk_idx == e).nonzero(as_tuple=False)  # [n, 2] (token, k) pairs
                if pos.numel() == 0:
                    continue
                tks = pos[:, 0]
                xe = x[tks].float()
                w1e = W1g[e].float().cuda()  # global expert weights live on CPU
                w2e = W2g[e].float().cuda()
                gate_e, up_e = (xe @ w1e.t()).chunk(2, dim=-1)
                o_e = (F.silu(gate_e) * up_e) @ w2e.t()  # [n, H]
                dtw_ref[pos[:, 0], pos[:, 1]] = (gyf[tks] * o_e).sum(-1)
            bres = {
                "dx": _gate3(dx_m, dx_t),
                "dW1": _gate3(dW1_m, dW1_t),
                "dW2": _gate3(dW2_m, dW2_t),
                "dtw": _gate3(dtw_m, dtw_ref),
            }
            bgathered = [None] * world
            dist.all_gather_object(bgathered, (rank, bres), group=group)
            if rank == 0:
                print(f"[{mode}] BACKWARD gradcheck vs turbo (cos / rel_rmse / ok):")
                for r, rr in sorted(bgathered, key=lambda t: t[0]):
                    line = "  ".join(f"{n}: {v[0]:.5f}/{v[1]:.4f}/{v[2]}" for n, v in rr.items())
                    print(f"  rank={r}  {line}")
            # gate dW1/dW2/dtw both modes; dx asserted only under load_balanced (round_robin noisy)
            gated = ["dW1", "dW2", "dtw"] + (["dx"] if mode == "load_balanced" else [])
            # round_robin dx is not gated (known noisy) but a regression should not be silent.
            if rank == 0 and mode != "load_balanced":
                dx_bad = [r for r, rr in bgathered if not rr["dx"][2]]
                if dx_bad:
                    print(f"[{mode}] WARNING: round_robin dx off-tolerance on ranks {dx_bad} (not gated)")
            bad = [
                (r, {k: v[2] for k, v in rr.items()})
                for r, rr in bgathered
                if not all(rr[k][2] for k in gated)
            ]
            assert not bad, f"[{mode}] backward gradcheck failed ({'/'.join(gated)}): {bad}"

        # ---- train-loop: continuous fwd+bwd (no L2 flush), per-call min/avg/max/p99 to catch bwd spikes ----
        if args.train_loop:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

            # N forwards (same x, don't chain outputs) then one backward runs all N backwards
            x_leaf = x.detach().requires_grad_(True)
            w1_leaf = W1.detach().requires_grad_(True)
            w2_leaf = W2.detach().requires_grad_(True)

            def _train_step():
                ys = [
                    mega_moe_fused(group, x_leaf, topk_idx, topk_w, w1_leaf, w2_leaf)
                    for _ in range(args.num_layers)
                ]
                # N backwards back-to-back
                torch.autograd.backward(ys, [grad_y] * len(ys))

            mega_dist = bench_kineto_dist(
                _train_step, num_tests=args.perf_iters, group=group, flush_l2=args.train_flush_l2
            )

            def _turbo_train_step():
                ys = [baseline_grad_forward(x, topk_idx, gate_logits) for _ in range(args.num_layers)]
                torch.autograd.backward(ys, [grad_y] * len(ys))

            turbo_dist = bench_kineto_dist(
                _turbo_train_step, num_tests=args.perf_iters, group=group, flush_l2=args.train_flush_l2
            )
            if rank == 0:
                flush = "L2-flush ON" if args.train_flush_l2 else "no L2-flush (training-like)"
                tag = f"{flush}, L={args.num_layers}, x{args.perf_iters} iters"
                _print_dist(f"[{mode}] mega TRAIN-LOOP per-call dist us ({tag}):", mega_dist)
                _print_dist(f"[{mode}] turbo TRAIN-LOOP per-call dist us ({tag}):", turbo_dist)

        # ---- perf: per-kernel breakdown of mega fwd + bwd (bwd isolated by fwd+bwd minus fwd) ----
        if args.perf:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

            def _forward():
                x_leaf = x.detach().requires_grad_(True)
                w1_leaf = W1.detach().requires_grad_(True)
                w2_leaf = W2.detach().requires_grad_(True)
                return mega_moe_fused(group, x_leaf, topk_idx, topk_w, w1_leaf, w2_leaf)

            fwd_breakdown = bench_kineto(_forward, num_tests=args.perf_iters, group=group)
            fwdbwd_breakdown = bench_kineto(
                lambda: _forward().backward(grad_y), num_tests=args.perf_iters, group=group
            )
            if rank == 0:
                bwd_breakdown = _diff_breakdown(fwdbwd_breakdown, fwd_breakdown)
                _print_breakdown(f"[{mode}] mega FORWARD per-kernel breakdown (us/call, %):", fwd_breakdown)
                _print_breakdown(f"[{mode}] mega BACKWARD per-kernel breakdown (us/call, %):", bwd_breakdown)

            # turbo (DeepEP) baseline per-kernel breakdown (same kineto path as mega)
            def _turbo_fwd():
                return baseline_grad_forward(x, topk_idx, gate_logits)

            turbo_fwd_breakdown = bench_kineto(_turbo_fwd, num_tests=args.perf_iters, group=group)
            turbo_fwdbwd_breakdown = bench_kineto(
                lambda: _turbo_fwd().backward(grad_y), num_tests=args.perf_iters, group=group
            )
            if rank == 0:
                turbo_bwd_breakdown = _diff_breakdown(turbo_fwdbwd_breakdown, turbo_fwd_breakdown)
                _print_breakdown(
                    f"[{mode}] turbo FORWARD per-kernel breakdown (us/call, %):", turbo_fwd_breakdown
                )
                _print_breakdown(
                    f"[{mode}] turbo BACKWARD per-kernel breakdown (us/call, %):", turbo_bwd_breakdown
                )

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
    # defaults: DeepSeek-V3 BF16, EP8, 8192 tokens/rank (H=7168, I=2048, E=256, topk=8)
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
    ap.add_argument("--bwd", action="store_true", help="run mega-vs-turbo backward gradcheck")
    ap.add_argument("--perf-iters", type=int, default=50)
    ap.add_argument(
        "--train-loop",
        action="store_true",
        help="continuous fwd+bwd loop (no L2 flush); per-call min/avg/max/p99 to catch bwd spikes",
    )
    ap.add_argument(
        "--train-flush-l2", action="store_true", help="flush L2 each iter in --train-loop (default off)"
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="--train-loop: stack N MoE layers (N forwards then N backwards, like training)",
    )
    return ap


# pytest entry: spawns the EP8 group; a child AssertionError propagates here.
_WORLD = 8


@pytest.mark.skipif(torch.cuda.device_count() < _WORLD, reason=f"needs {_WORLD} GPUs for EP{_WORLD}")
def test_mega_moe_dispatch_combine_grouped_gemm():
    # production DeepSeek-V3 EP8 shape (bwd flaky at small shapes); --bwd asserts dW1/dW2 all modes, dx/dtw balanced
    args = _build_argparser().parse_args(
        [
            "--num-tokens",
            "8192",
            "--num-topk",
            "8",
            "--num-experts",
            "256",
            "--hidden",
            "7168",
            "--inter",
            "2048",
            "--mode",
            "both",
            "--bwd",
        ]
    )
    # spawn re-raises any child exception (gate-3 / backward asserts) in the parent
    torch.multiprocessing.spawn(_run, args=(_WORLD, args), nprocs=_WORLD)


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    torch.multiprocessing.spawn(_run, args=(args.num_processes, args), nprocs=args.num_processes)
