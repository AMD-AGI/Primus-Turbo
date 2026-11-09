###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-stage timing of a Ulysses (all-to-all context-parallel) transformer layer.

One layer is an attention block -- QKV GEMM, pack + all-to-all + unpack, attention,
pack + all-to-all + unpack, o_proj GEMM + residual -- followed by an optional dense
MLP. Forward and backward are timed stage by stage with CUDA events, using the same
pack/unpack helpers as Primus-Turbo's ``flash_attn_usp_func``.

From the stage times it bounds what fusing each all-to-all into its neighbouring GEMM
(GemmA2A: GEMM then all-to-all; A2AGemm: all-to-all then GEMM) could save, assuming the
fused kernel writes the post-all-to-all layout directly:

    bound = pack + unpack + min(gemm, isolated all-to-all)

The isolated all-to-all is timed with the ranks aligned by a barrier. Whatever the in-layer
all-to-all takes beyond that is waiting for slower ranks (mostly attention skew), which a
fused kernel has to wait for as well.

``--fused`` runs the layer a second time with Primus-Turbo's fused ops
(``ulysses_qkv_gemm_a2a`` / ``ulysses_a2a_o_gemm``) and reports each fused stage, max over
ranks, against the unfused GEMM + pack + all-to-all + unpack it replaces. Both paths use
Megatron's interleaved ``[ng, hq/ng + 2, d]`` QKV weight.

Example (single node, Ulysses degree 8):

    torchrun --nproc_per_node 8 benchmark/ops/training/bench_ulysses_a2a.py \
        --model llama3.1-8b --seq 32768 65536 131072 --fused --report-csv-path ulysses_a2a.csv
"""

import argparse
import csv
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.pytorch.ops import flash_attn_func
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    ulysses_a2a_o_gemm,
    ulysses_qkv_gemm_a2a,
)
from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
    get_attention_cp_a2a_helper,
)

MODEL_PARAMS_TABLE = {
    "llama3.1-8b": dict(hidden=4096, hq=32, hkv=8, d=128, ffn=14336, glu=True, causal=True),
    "llama3.1-70b": dict(hidden=8192, hq=64, hkv=8, d=128, ffn=28672, glu=True, causal=True),
    # self-attention + FFN of a DiT block; cross-attention and modulation are left out
    "wan2.1-14b": dict(hidden=5120, hq=40, hkv=40, d=128, ffn=13824, glu=False, causal=False),
    # MoE: the MLP stage is a dense SwiGLU as wide as the activated experts (top-k x expert ffn),
    # i.e. expert compute without EP dispatch/combine, routing or grouped-GEMM overhead.
    "qwen3-30b-a3b": dict(hidden=2048, hq=32, hkv=4, d=128, ffn=8 * 768, glu=True, causal=True),
    "qwen3-235b-a22b": dict(hidden=4096, hq=64, hkv=4, d=128, ffn=8 * 1536, glu=True, causal=True),
    "mixtral-8x7b": dict(hidden=4096, hq=32, hkv=8, d=128, ffn=2 * 14336, glu=True, causal=True),
}

STAGES = (
    "qkv_gemm",
    "pre_pack",
    "pre_a2a",
    "pre_unpack",
    "attn",
    "post_pack",
    "post_a2a",
    "post_unpack",
    "o_gemm",
    "mlp",
)

# (label, phase, gemm stage, all-to-all stage prefix, fusion kind)
FUSIONS = (
    ("fwd pre : qkv GEMM -> a2a", "fwd", "qkv_gemm", "pre", "GemmA2A"),
    ("fwd post: a2a -> o_proj GEMM", "fwd", "o_gemm", "post", "A2AGemm"),
    ("bwd post: o_proj dgrad/wgrad -> a2a", "bwd", "o_gemm", "post", "GemmA2A"),
    ("bwd pre : a2a -> qkv dgrad/wgrad", "bwd", "qkv_gemm", "pre", "A2AGemm"),
)

# Fraction of the bound reached by a measured fused-kernel implementation (pre side
# GemmA2A, post side A2AGemm including its boundary barrier); used for the "expected" range.
ETA = {"GemmA2A": (0.78, 0.87), "A2AGemm": (0.64, 0.67)}

# (label, phase, stages of the unfused path that one fused stage replaces)
FUSED_STAGES = (
    ("fwd QKV GEMM + a2a", "fwd", ("qkv_gemm", "pre_pack", "pre_a2a", "pre_unpack")),
    ("fwd a2a + o_proj GEMM", "fwd", ("post_pack", "post_a2a", "post_unpack", "o_gemm")),
    ("bwd o_proj GEMM + a2a", "bwd", ("o_gemm", "post_pack", "post_a2a", "post_unpack")),
    ("bwd a2a + QKV GEMM", "bwd", ("pre_pack", "pre_a2a", "pre_unpack", "qkv_gemm")),
    ("fwd attn", "fwd", ("attn",)),
    ("bwd attn", "bwd", ("attn",)),
    ("fwd mlp", "fwd", ("mlp",)),
    ("bwd mlp", "bwd", ("mlp",)),
)


class _Recorder:
    """CUDA events in issue order, per phase; a stage's time is the gap to the previous event."""

    def __init__(self):
        self.enabled = False
        self.events = {"fwd": [], "bwd": []}

    def reset(self, enabled):
        self.enabled = enabled
        self.events = {"fwd": [], "bwd": []}

    def rec(self, phase, name):
        if not self.enabled or name is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events[phase].append((name, event))

    def durations_us(self, phase):
        events = self.events[phase]
        out = {}
        for (_, prev), (name, event) in zip(events, events[1:]):
            out[name] = out.get(name, 0.0) + prev.elapsed_time(event) * 1e3
        return out


REC = _Recorder()


class _Mark(torch.autograd.Function):
    """Identity that records ``fwd_name`` when reached in forward and ``bwd_name`` in backward."""

    @staticmethod
    def forward(ctx, fwd_name, bwd_name, *tensors):
        ctx.bwd_name = bwd_name
        REC.rec("fwd", fwd_name)
        return tuple(t.view_as(t) for t in tensors)

    @staticmethod
    def backward(ctx, *grads):
        REC.rec("bwd", ctx.bwd_name)
        return (None, None, *grads)


def mark(fwd_name, bwd_name, *tensors):
    out = _Mark.apply(fwd_name, bwd_name, *tensors)
    return out if len(tensors) > 1 else out[0]


class _PreA2A(torch.autograd.Function):
    """[b, s/n, h, d] q/k/v -> [b, s, h/n, d]; backward is the mirror on dq/dk/dv."""

    @staticmethod
    def forward(ctx, q, k, v, helper, group):
        ctx.helper, ctx.group = helper, group
        qkv = helper.combine_qkv_before_a2a(q, k, v)
        REC.rec("fwd", "pre_pack")
        out = torch.empty_like(qkv)
        dist.all_to_all_single(out, qkv, group=group)
        REC.rec("fwd", "pre_a2a")
        q, k, v = helper.splits_qkv_after_a2a(out)
        REC.rec("fwd", "pre_unpack")
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        dqkv = ctx.helper.combine_dqkv_before_a2a(dq.contiguous(), dk.contiguous(), dv.contiguous())
        REC.rec("bwd", "pre_pack")
        out = torch.empty_like(dqkv)
        dist.all_to_all_single(out, dqkv, group=ctx.group)
        REC.rec("bwd", "pre_a2a")
        dq, dk, dv = ctx.helper.split_dqkv_after_a2a(out)
        REC.rec("bwd", "pre_unpack")
        return dq, dk, dv, None, None


class _PostA2A(torch.autograd.Function):
    """[b, s, h/n, d] attention output -> [b, s/n, h, d]; backward is the mirror on do."""

    @staticmethod
    def forward(ctx, o, helper, group):
        ctx.helper, ctx.group = helper, group
        o = helper.reshape_o_before_a2a(o)
        REC.rec("fwd", "post_pack")
        out = torch.empty_like(o)
        dist.all_to_all_single(out, o, group=group)
        REC.rec("fwd", "post_a2a")
        out = helper.reshape_o_after_a2a(out)
        REC.rec("fwd", "post_unpack")
        return out

    @staticmethod
    def backward(ctx, do):
        do = ctx.helper.reshape_do_before_a2a(do.contiguous())
        REC.rec("bwd", "post_pack")
        out = torch.empty_like(do)
        dist.all_to_all_single(out, do, group=ctx.group)
        REC.rec("bwd", "post_a2a")
        do = ctx.helper.reshape_do_after_a2a(out)
        REC.rec("bwd", "post_unpack")
        return do, None, None


class UlyssesLayer:
    def __init__(self, cfg, seq, mbs, group, use_mlp, fused=False, dtype=torch.bfloat16):
        n = group.size()
        self.cfg, self.mbs, self.group, self.use_mlp, self.fused = cfg, mbs, group, use_mlp, fused
        self.s_local = seq // n
        hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]

        def weight(out_features, in_features):
            w = torch.randn(out_features, in_features, device="cuda", dtype=dtype) * 0.02
            return w.requires_grad_()

        self.w_qkv = weight((hq + 2 * hkv) * d, hidden)
        self.w_o = weight(hidden, hq * d)
        self.params = [self.w_qkv, self.w_o]
        if use_mlp:
            ffn = cfg["ffn"]
            self.w_fc1 = weight(ffn * (2 if cfg["glu"] else 1), hidden)
            self.w_fc2 = weight(hidden, ffn)
            self.params += [self.w_fc1, self.w_fc2]
        self.helper = get_attention_cp_a2a_helper(mbs, seq, hq, hkv, d, d, 1, n)

    def __call__(self, x):
        cfg, b, s = self.cfg, self.mbs, self.s_local
        hq, hkv, d = cfg["hq"], cfg["hkv"], cfg["d"]

        x = mark("start", "qkv_gemm", x)
        if self.fused:
            q, k, v = ulysses_qkv_gemm_a2a(x, self.w_qkv, hq, hkv, d, self.group, "bshd", "interleaved")
            q, k, v = mark("qkv_gemm", "attn", q, k, v)
        else:
            qkv = F.linear(x, self.w_qkv).view(b, s, hkv, hq // hkv + 2, d)
            q = qkv[..., : hq // hkv, :].reshape(b, s, hq, d)
            k, v = qkv[..., -2, :], qkv[..., -1, :]
            q, k, v = mark("qkv_gemm", None, q, k, v)
            q, k, v = _PreA2A.apply(q, k, v, self.helper, self.group)
            q, k, v = mark(None, "attn", q, k, v)
        o = flash_attn_func(q, k, v, causal=cfg["causal"])
        if self.fused:
            o = mark("attn", "o_gemm", o)
            y = ulysses_a2a_o_gemm(o, self.w_o, x, self.group, "bshd")
        else:
            o = mark("attn", None, o)
            o = _PostA2A.apply(o, self.helper, self.group)
            o = mark(None, "o_gemm", o)
            y = F.linear(o.reshape(b, s, hq * d), self.w_o) + x
        if not self.use_mlp:
            return mark("o_gemm", None, y)

        y = mark("o_gemm", "mlp", y)
        h = F.linear(y, self.w_fc1)
        if cfg["glu"]:
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = F.gelu(h, approximate="tanh")
        out = F.linear(h, self.w_fc2) + y
        return mark("mlp", None, out)


def _barrier():
    torch.cuda.synchronize()
    dist.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()


def time_a2a_isolated_us(numel, group, warmup, iters):
    """All-to-all alone, ranks aligned by a barrier first, so peer skew is excluded."""
    inp = torch.randn(numel, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for it in range(warmup + iters):
        _barrier()
        start.record()
        dist.all_to_all_single(out, inp, group=group)
        end.record()
        torch.cuda.synchronize()
        if it >= warmup:
            times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def _all_reduce(values, op):
    t = torch.tensor(values, device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return t.tolist()


def run_config(model, seq, mbs, group, use_mlp, warmup, iters, fused=False):
    cfg = MODEL_PARAMS_TABLE[model]
    n = group.size()
    for what, size in (("seq", seq), ("q heads", cfg["hq"]), ("kv heads", cfg["hkv"])):
        assert size % n == 0, f"{what}={size} must be divisible by the Ulysses degree {n}"

    layer = UlyssesLayer(cfg, seq, mbs, group, use_mlp, fused)
    x0 = torch.randn(mbs, seq // n, cfg["hidden"], device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x0)

    per_iter = {"fwd": [], "bwd": []}
    for it in range(warmup + iters):
        REC.reset(enabled=it >= warmup)
        for p in layer.params:
            p.grad = None
        x = x0.clone().requires_grad_()
        _barrier()
        # keep the GPU busy while the host queues the layer, as the previous layer does in training
        torch.cuda._sleep(2_000_000)
        out = layer(x)
        REC.rec("bwd", "start")
        out.backward(dy)
        torch.cuda.synchronize()
        if REC.enabled:
            for phase in per_iter:
                per_iter[phase].append(REC.durations_us(phase))

    keys = [(phase, st) for phase in ("fwd", "bwd") for st in STAGES if st in per_iter[phase][0]]
    local = [statistics.median(d[st] for d in per_iter[phase]) for phase, st in keys]
    local_totals = [statistics.median(sum(d.values()) for d in per_iter[phase]) for phase in ("fwd", "bwd")]

    tokens = mbs * seq // n
    iso_local = [
        time_a2a_isolated_us(tokens * (cfg["hq"] + 2 * cfg["hkv"]) * cfg["d"], group, warmup, iters),
        time_a2a_isolated_us(tokens * cfg["hq"] * cfg["d"], group, warmup, iters),
    ]

    mean = [v / n for v in _all_reduce(local, dist.ReduceOp.SUM)]
    stage_max = _all_reduce(local, dist.ReduceOp.MAX)
    # The critical rank of a phase is the one with the most compute (attention + MLP): the others
    # wait for it at the next exchange, so its own exchange stages carry no waiting.
    per_rank = [None] * n
    dist.all_gather_object(per_rank, local, group=group)
    stage_crit = {}
    for phase in ("fwd", "bwd"):
        compute = [
            sum(v for (p, st), v in zip(keys, r) if p == phase and st in ("attn", "mlp")) for r in per_rank
        ]
        crit = per_rank[max(range(n), key=compute.__getitem__)]
        stage_crit[phase] = {st: v for (p, st), v in zip(keys, crit) if p == phase}
    totals = _all_reduce(local_totals, dist.ReduceOp.MAX)
    iso = [v / n for v in _all_reduce(iso_local, dist.ReduceOp.SUM)]

    del layer, x0, dy
    torch.cuda.empty_cache()
    stages = {"fwd": {}, "bwd": {}}
    stages_max = {"fwd": {}, "bwd": {}}
    for (phase, st), v, vmax in zip(keys, mean, stage_max):
        stages[phase][st] = v
        stages_max[phase][st] = vmax
    return (
        stages,
        stages_max,
        {"fwd": totals[0], "bwd": totals[1]},
        {"pre": iso[0], "post": iso[1]},
        stage_crit,
    )


def analyze(stages, totals, iso):
    rows = []
    for label, phase, gemm, comm, kind in FUSIONS:
        st = stages[phase]
        pack, a2a, unpack = st[f"{comm}_pack"], st[f"{comm}_a2a"], st[f"{comm}_unpack"]
        base = st[gemm] + pack + a2a + unpack
        bound = pack + unpack + min(st[gemm], iso[comm])
        lo, hi = (bound * e for e in ETA[kind])
        rows.append(
            dict(label=label, kind=kind, base=base, bound=bound, copies=pack + unpack, expected=(lo, hi))
        )
    layer_total = totals["fwd"] + totals["bwd"]
    comm = {
        part: sum(stages[phase][f"{c}_{part}"] for phase in ("fwd", "bwd") for c in ("pre", "post"))
        for part in ("pack", "a2a", "unpack")
    }
    wire = 2 * (iso["pre"] + iso["post"])
    breakdown = dict(
        copies=comm["pack"] + comm["unpack"],
        wire=wire,
        wait=comm["a2a"] - wire,
    )
    return rows, layer_total, breakdown


def report(model, seq, mbs, n, use_mlp, stages, stages_max, totals, iso):
    cfg = MODEL_PARAMS_TABLE[model]
    tokens = mbs * seq // n
    rows, layer_total, breakdown = analyze(stages, totals, iso)
    comm_total = sum(breakdown.values())

    print(f"\n== {model}  seq={seq}  ulysses={n}  mbs={mbs}  tokens/rank={tokens}  mlp={use_mlp} ==")
    print("per layer, us; mean over ranks of each rank's median (max rank in brackets)")
    print(f"  {'stage':<12}{'fwd':>18}{'bwd':>18}")
    for st in STAGES:
        if st not in stages["fwd"]:
            continue
        cells = [f"{stages[p][st]:8.0f} [{stages_max[p][st]:6.0f}]" for p in ("fwd", "bwd")]
        print(f"  {st:<12}{cells[0]:>18}{cells[1]:>18}")
    print(f"  {'layer total':<12}{totals['fwd']:>18.0f}{totals['bwd']:>18.0f}   (max rank)")

    for side, heads in (("pre", cfg["hq"] + 2 * cfg["hkv"]), ("post", cfg["hq"])):
        nbytes = tokens * heads * cfg["d"] * 2
        busbw = nbytes * (n - 1) / n / (iso[side] * 1e-6) / 1e9
        print(
            f"  a2a {side:<4} isolated {iso[side]:7.0f} us  ({nbytes / 2**20:.0f} MiB/rank, "
            f"{busbw:.0f} GB/s); in-layer fwd {stages['fwd'][f'{side}_a2a']:.0f} / "
            f"bwd {stages['bwd'][f'{side}_a2a']:.0f} us"
        )

    print(
        f"  pack + a2a + unpack, fwd+bwd: {comm_total:.0f} us = {comm_total / layer_total:.1%} of layer "
        f"(copies {breakdown['copies']:.0f}, wire {breakdown['wire']:.0f}, "
        f"waiting for slower ranks {breakdown['wait']:.0f})"
    )
    print(f"  {'fusion':<38}{'kind':>9}{'base':>8}{'bound':>8}{'copies':>8}{'expected':>14}")
    total_bound = total_lo = total_hi = 0.0
    for r in rows:
        lo, hi = r["expected"]
        total_bound += r["bound"]
        total_lo += lo
        total_hi += hi
        print(
            f"  {r['label']:<38}{r['kind']:>9}{r['base']:>8.0f}{r['bound']:>8.0f}{r['copies']:>8.0f}"
            f"{lo:>7.0f}-{hi:<6.0f}"
        )
    print(
        f"  saving per layer: bound {total_bound:.0f} us ({total_bound / layer_total:.1%}), "
        f"expected {total_lo:.0f}-{total_hi:.0f} us "
        f"({total_lo / layer_total:.1%}-{total_hi / layer_total:.1%}) of {layer_total:.0f} us fwd+bwd; "
        f"pack/unpack alone {breakdown['copies']:.0f} us ({breakdown['copies'] / layer_total:.1%})"
    )

    row = {"Model": model, "Seq": seq, "Ulysses": n, "MBS": mbs, "MLP": use_mlp}
    for phase in ("fwd", "bwd"):
        for st, v in stages[phase].items():
            row[f"{phase}_{st}_us"] = round(v, 1)
        row[f"{phase}_total_us"] = round(totals[phase], 1)
    row["a2a_pre_isolated_us"] = round(iso["pre"], 1)
    row["a2a_post_isolated_us"] = round(iso["post"], 1)
    for r, (_, phase, _, comm, _) in zip(rows, FUSIONS):
        row[f"bound_{phase}_{comm}_us"] = round(r["bound"], 1)
    row["copies_us"] = round(breakdown["copies"], 1)
    row["a2a_wire_us"] = round(breakdown["wire"], 1)
    row["a2a_wait_us"] = round(breakdown["wait"], 1)
    row["bound_total_us"] = round(total_bound, 1)
    row["bound_pct"] = round(100 * total_bound / layer_total, 2)
    row["expected_pct_lo"] = round(100 * total_lo / layer_total, 2)
    row["expected_pct_hi"] = round(100 * total_hi / layer_total, 2)
    return row


def report_fused(model, seq, mbs, n, base, fused):
    """Each fused stage against the unfused stages it replaces; ``base`` / ``fused`` are
    ``(mean over ranks, max over ranks, layer totals, critical rank)`` from :func:`run_config`.
    The mean and max include ranks waiting for the critical (slowest-compute) rank; the critical
    rank's own exchange stages do not, so that column is what the kernels add to the layer."""
    print(f"\n== fused vs unfused: {model}  seq={seq}  ulysses={n}  mbs={mbs} ==")
    print("per layer, us; each rank's median: critical rank | mean over ranks [max rank]")
    print(f"  {'stage':<26}{'unfused':>24}{'fused':>24}{'speedup':>9}")
    row = {"Model": model, "Seq": seq, "Ulysses": n, "MBS": mbs}
    for label, phase, parts in FUSED_STAGES:
        fused_stage = [p for p in parts if p in fused[0][phase]]
        if not fused_stage:
            continue
        b_mean, b_max = (sum(st[phase][p] for p in parts) for st in base[:2])
        b_crit = sum(base[3][phase][p] for p in parts)
        f_mean, f_max = (st[phase][fused_stage[0]] for st in fused[:2])
        f_crit = fused[3][phase][fused_stage[0]]
        cells = [
            f"{c:6.0f} | {m:6.0f} [{x:5.0f}]"
            for c, m, x in ((b_crit, b_mean, b_max), (f_crit, f_mean, f_max))
        ]
        print(f"  {label:<26}{cells[0]:>24}{cells[1]:>24}{b_crit / f_crit:>8.2f}x")
        key = label.replace(" + ", "_").replace(" ", "_")
        for name, vals in (("unfused", (b_crit, b_mean, b_max)), ("fused", (f_crit, f_mean, f_max))):
            row.update({f"{key}_{name}_{k}_us": round(v, 1) for k, v in zip(("crit", "mean", "max"), vals)})
    for phase in ("fwd", "bwd"):
        b_tot, f_tot = base[2][phase], fused[2][phase]
        print(
            f"  {phase + ' layer total':<26}{b_tot:>24.0f}{f_tot:>24.0f}{b_tot / f_tot:>8.2f}x   (max rank)"
        )
        row.update({f"{phase}_total_unfused_us": round(b_tot, 1), f"{phase}_total_fused_us": round(f_tot, 1)})
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", nargs="+", default=["llama3.1-8b"], choices=sorted(MODEL_PARAMS_TABLE))
    parser.add_argument(
        "--seq", nargs="+", type=int, default=[32768, 65536, 131072], help="total sequence length"
    )
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--no-mlp", action="store_true", help="time the attention block only")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--report-csv-path", type=str, default=None)
    parser.add_argument("--fused", action="store_true", help="also time the fused GemmA2A / A2AGemm layer")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    rank, n = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42 + rank)

    results = []
    for model in args.model:
        for seq in args.seq:
            stages, stages_max, totals, iso, crit = run_config(
                model, seq, args.mbs, group, not args.no_mlp, args.warmup, args.iters
            )
            if args.fused:
                fused_stages, fused_max, fused_totals, _, fused_crit = run_config(
                    model, seq, args.mbs, group, not args.no_mlp, args.warmup, args.iters, fused=True
                )
            if rank == 0:
                row = report(model, seq, args.mbs, n, not args.no_mlp, stages, stages_max, totals, iso)
                if args.fused:
                    row.update(
                        report_fused(
                            model,
                            seq,
                            args.mbs,
                            n,
                            (stages, stages_max, totals, crit),
                            (fused_stages, fused_max, fused_totals, fused_crit),
                        )
                    )
                results.append(row)

    if rank == 0 and args.report_csv_path and results:
        fieldnames = list(dict.fromkeys(k for r in results for k in r))
        with open(args.report_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    _barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
