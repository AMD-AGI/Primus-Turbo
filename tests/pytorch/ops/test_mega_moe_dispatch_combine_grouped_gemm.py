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

# Megatron-LM lives under the sibling Primus repo, not on the default path
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
    # include memcpy (gpu_memcpy) but NOT memset (the _l2_flush memset is excluded)
    for ev in events:
        if ev.get("cat") in ("kernel", "Kernel", "gpu_memcpy") and "dur" in ev:
            agg[ev["name"]] = agg.get(ev["name"], 0.0) + float(ev["dur"])
            cnt[ev["name"]] = cnt.get(ev["name"], 0) + 1
    # (name, avg_us_per_iter, avg_launches_per_iter)
    breakdown = [(name, dur / num_tests, cnt[name] / num_tests) for name, dur in agg.items()]
    breakdown.sort(key=lambda t: -t[1])
    return breakdown


def bench_kineto_dist(fn, *, warmup=5, num_tests=50, group=None, flush_l2=False):
    """Profile ``fn`` as a CONTINUOUS loop and return PER-CALL durations per kernel.

    Unlike ``bench_kineto`` (which flushes the L2 before each call and reports the
    per-kernel AVERAGE), this runs ``fn`` back-to-back with NO L2 flush and NO
    per-iter sync -- mirroring a real training loop, where the backward of one
    iter overlaps the next iter's forward and they contend for XGMI / L2. The full
    per-call duration list lets the caller report min/avg/max/p99 to catch spikes
    (e.g. dispatch_grouped_gemm backward jumping 4ms -> 9ms) that an average hides.

    Returns {kernel_name: [dur_us, ...]} aggregating every captured call.
    """
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

    load_balanced: softmax-topk of Megatron force-load-balancing random logits
                   (apply_random_logits -> rank-seeded normal logits).
    round_robin:   experts assigned by a contiguous arange % num_experts pattern.
    """
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

    # Register a rank-specific expert-parallel RNG state so apply_random_logits()
    # (used by generate_routing's load_balanced mode) can fork it.
    _tracker = get_cuda_rng_tracker()
    _ep_rng = get_expert_parallel_rng_tracker_name()
    if _ep_rng not in _tracker.get_states():
        _tracker.add(_ep_rng, 1024 + 100 * rank)

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

        # ---- probe: is the NN dgrad GEMM (d_swiglu = d_l2y @ w2) the broken stage? ----
        if args.probe_dswiglu:
            from primus_turbo.flydsl.mega.swiglu_kernel import swiglu_backward
            from primus_turbo.pytorch.core.backend import BackendType
            from primus_turbo.pytorch.kernels.mega_moe.dispatch_grouped_gemm_impl import (
                dispatch_grouped_gemm_impl,
            )

            with torch.no_grad():
                # forward dispatch builds the symm pool + handle (same as mega_moe_fused STEP1+2)
                _l1, _, _, handle = dispatch_grouped_gemm_impl(
                    x,
                    W1,
                    group,
                    BackendType.FLYDSL.value,
                    topk_idx=topk_idx.to(torch.int64),
                    topk_weights=topk_w,
                    layout="nt",
                )
                l1_out = _l1.clone()
                weight = symm.weight_recv_buf.clone().float()  # per-pool-row routing weight
                dy = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
                # STEP1 backward: NN dispatch of dy -> d_swiglu = d_l2y @ w2
                d_swiglu, _, _, _ = dispatch_grouped_gemm_impl(
                    dy,
                    W2,
                    group,
                    BackendType.FLYDSL.value,
                    handle=handle,
                    layout="nn",
                )
                d_l2y = symm.pool.clone()  # pool now holds the dispatched dy rows
                group_lens, group_offs = handle[9].tolist(), handle[10].tolist()
                Iq = I

                # (a) NN dgrad GEMM: d_swiglu = d_l2y @ w2
                ds_ref = torch.zeros_like(d_swiglu)
                for e in range(epr):
                    s, L = int(group_offs[e]), int(group_lens[e])
                    if L > 0:
                        ds_ref[s : s + L] = (d_l2y[s : s + L].float() @ W2[e].float()).to(d_swiglu.dtype)

                # (b) swiglu_backward in-situ: grad_l1 (dgate|dup) + act_w
                grad_l1, grad_gate, act_w = swiglu_backward(
                    d_swiglu, l1_out, scale=symm.weight_recv_buf, return_gate=True, return_act_w=True
                )
                # device-time the bounded swiglu_backward in isolation (rank 0)
                if rank == 0:
                    m_real = int(symm.meta_scalars[1].item()) * symm.block_m
                    for _ in range(20):
                        swiglu_backward(
                            d_swiglu, l1_out, scale=symm.weight_recv_buf, return_gate=True, return_act_w=True
                        )
                    torch.cuda.synchronize()
                    _e0 = torch.cuda.Event(True)
                    _e1 = torch.cuda.Event(True)
                    _e0.record()
                    for _ in range(100):
                        swiglu_backward(
                            d_swiglu, l1_out, scale=symm.weight_recv_buf, return_gate=True, return_act_w=True
                        )
                    _e1.record()
                    torch.cuda.synchronize()
                    _us = _e0.elapsed_time(_e1) / 100 * 1000
                    print(
                        f"[probe] swiglu_backward bounded: {_us:.1f} us  "
                        f"ACC1.shape={tuple(l1_out.shape)} m_real={m_real}",
                        flush=True,
                    )
                gate = l1_out[:, :Iq].float()
                up = l1_out[:, Iq:].float()
                sig = torch.sigmoid(gate)
                s_silu = gate * sig
                d = d_swiglu.float() * weight.unsqueeze(-1)
                dgate_ref = d * up * (sig * (1 + gate * (1 - sig)))
                dup_ref = d * s_silu
                gl1_ref = torch.cat([dgate_ref, dup_ref], dim=-1).to(grad_l1.dtype)
                actw_ref = (s_silu * up * weight.unsqueeze(-1)).to(act_w.dtype)

                mask = d_l2y.abs().sum(-1) > 0

                # (c) dW1 tn-wgrad STEP4: re-dispatch x into pool, dW1[e] = grad_l1[e]^T @ x_pool[e]
                dW1, _, _, _ = dispatch_grouped_gemm_impl(
                    x,
                    grad_l1,
                    group,
                    BackendType.FLYDSL.value,
                    handle=handle,
                    layout="tn",
                    trans_c=True,
                )
                x_pool = symm.pool.clone()
                xv = x_pool.abs().sum(-1) > 0  # valid (dispatched) rows
                gl = grad_l1.float().clone()
                gl[~xv] = 0  # invalid rows contribute nothing
                xp = x_pool.float().clone()
                xp[~xv] = 0
                dW1_ref = torch.zeros_like(dW1)
                for e in range(epr):
                    s, L = int(group_offs[e]), int(group_lens[e])
                    if L > 0:
                        dW1_ref[e] = (gl[s : s + L].T @ xp[s : s + L]).to(dW1.dtype)  # [2I, H]

                checks = {
                    "d_swiglu(NN gemm)": _gate3(d_swiglu[mask], ds_ref[mask]),
                    "act_w(swiglu_bwd)": _gate3(act_w[mask], actw_ref[mask]),
                    "grad_l1(swiglu_bwd)": _gate3(grad_l1[mask], gl1_ref[mask]),
                    "dW1(tn wgrad)": _gate3(dW1, dW1_ref),
                }
                gd = [None] * world
                dist.all_gather_object(gd, (rank, checks, int(mask.sum())), group=group)
                if rank == 0:
                    print(f"[{mode}] PROBE backward stages vs torch ref (cos/rel/ok):")
                    for r, cc, nrows in sorted(gd, key=lambda t: t[0]):
                        line = "  ".join(f"{n}: {v[0]:.5f}/{v[1]:.4f}/{v[2]}" for n, v in cc.items())
                        print(f"  rank={r} rows={nrows}  {line}")

        # ---- backward gradcheck: mega grads vs the turbo baseline (same grad_y) ----
        if args.bwd:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)
            # mega fused grads (fixed routing; x/W1/W2/topk_w leaves so grad_topk_weights is exercised)
            x_m = x.detach().requires_grad_(True)
            w1_m = W1.detach().requires_grad_(True)
            w2_m = W2.detach().requires_grad_(True)
            tw_m = topk_w.detach().requires_grad_(True)
            y_m = mega_moe_fused(group, x_m, topk_idx, tw_m, w1_m, w2_m, block_m=args.bm, block_n=args.bn)
            dx_m, dW1_m, dW2_m, dtw_m = torch.autograd.grad(y_m, [x_m, w1_m, w2_m, tw_m], grad_y)
            # turbo baseline grads (same grad_y, same local shard)
            y_t, x_t, w1_t, w2_t = baseline_grad_forward_leaves(x, topk_idx, gate_logits)
            dx_t, dW1_t, dW2_t = torch.autograd.grad(y_t, [x_t, w1_t, w2_t], grad_y)
            # analytic grad_topk_weights ref: dtw[t,k] = <grad_y[t], o[t,k]>, o = UNWEIGHTED route
            # output (swiglu(x@W1[e])@W2[e]); computed per-expert over the GLOBAL weights.
            dtw_ref = torch.zeros_like(topk_w)
            gyf = grad_y.float()
            for e in range(E):
                pos = (topk_idx == e).nonzero(as_tuple=False)  # [n, 2] (token, k) pairs
                if pos.numel() == 0:
                    continue
                tks = pos[:, 0]
                xe = x[tks].float()
                gate_e, up_e = (xe @ W1g[e].float().t()).chunk(2, dim=-1)
                o_e = (F.silu(gate_e) * up_e) @ W2g[e].float().t()  # [n, H]
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

        # ---- train-loop: reproduce the training case (continuous fwd+bwd, no L2 flush) ----
        # Real training runs fwd->bwd->fwd->bwd with no sync/flush between iters, so the
        # backward of one iter overlaps the next iter's forward and they contend for XGMI.
        # Profile that continuous loop and report per-call min/avg/max/p99 to catch the
        # dispatch_grouped_gemm backward spiking 4ms -> 9ms that the averaged --perf hides.
        if args.train_loop:
            grad_y = torch.randn((T, H), device="cuda", dtype=torch.bfloat16)

            # Simulate the multi-layer training shape: run N MoE-layer forwards (all
            # consuming the SAME well-conditioned x so routing stays balanced -- do NOT
            # chain outputs, that diverges and deadlocks, see memory note
            # project_mega_multilayer_hang_rootcause), keeping every autograd graph
            # alive, then a single backward() runs all N backwards in reverse order
            # (layer N..1). The N layers share one symm buffer (per-layer origin/meta
            # snapshots restore it in backward), so the backward dispatch/combine of
            # later layers overlaps earlier layers' work -- the real contention the
            # single-layer loop lacks.
            x_leaf = x.detach().requires_grad_(True)
            w1_leaf = W1.detach().requires_grad_(True)
            w2_leaf = W2.detach().requires_grad_(True)

            def _train_step():
                ys = [
                    mega_moe_fused(
                        group, x_leaf, topk_idx, topk_w, w1_leaf, w2_leaf, block_m=args.bm, block_n=args.bn
                    )
                    for _ in range(args.num_layers)
                ]
                # N forwards done; now N backwards back-to-back (reverse order)
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
    ap.add_argument("--bwd", action="store_true", help="run mega-vs-turbo backward gradcheck")
    ap.add_argument("--probe-dswiglu", action="store_true", help="probe the NN dgrad GEMM d_swiglu=d_l2y@w2")
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
