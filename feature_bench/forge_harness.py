###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared harness for the forge-loop measurement drivers.

forge-loop runs a driver as a black box and reads it over stdout only, so
everything the drivers need at runtime lives here or beside them -- never in a
file that only exists while the campaign is being set up.

What this module provides:
  * torchrun self-launch (the mega kernels are EP8 collectives)
  * graph-timed benchmarking (forge rejects eager timing; it counts replays)
  * correctness, delegated to benchmark/ops/training/bench_mega_moe.py
    (see reference_snrs)

Correctness used to be a golden fingerprint: snapshot the pristine kernel's
output once, reduce it to a small order-invariant projection (project() below),
and compare later runs against it. It does not work on this build. A golden
written by a driver fails on the very next run of the same code -- dispatch_nn
worst of all, at a forced 0 dB because the projected row count itself moves --
and all three drivers failed their own goldens the same way. The fingerprint is
left here, unused, because the reasoning behind it is still the right reasoning
for a snapshot-based check and re-deriving it would be wasteful.

What replaced it is the gate the kernels are actually developed against: build
the reference *inside the run*, over the same handle and the same symmetric
buffer, and compare only the real pool rows. The pool layout is then identical
on both sides, so the nondeterminism cancels instead of being fingerprinted.

The same nondeterminism governs the *inputs*: any operand indexed by pool row
must be built with pool_keyed_operand(), never with a raw randn.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
WORLD = 8
PROJ_DIM = 32
PROJ_ROWS = 64
PROJ_SEED = 20260814
# Bump whenever project()'s output changes shape or meaning, so a stale golden
# is rejected with a clear message instead of a confusing unpack error.
GOLDEN_FORMAT = 4

# Correctness runs a small shape: fast, and the golden stays a few hundred KB.
# hidden must satisfy the dispatch tile's `hidden * 2 % 1024 == 0` assert.
CORRECT_SHAPE = dict(num_tokens=256, hidden=1024, inter=512, num_experts=64, num_topk=4)
# Benchmark runs the real DeepSeek-V3 BF16 MoE shape, 8192 tokens per EP rank.
BENCH_SHAPE = dict(num_tokens=8192, hidden=7168, inter=2048, num_experts=256, num_topk=8)

NUM_GROUPS = 8
GROUP_TOPK = 4
ROUTING_SCALE = 2.5
SEED = 1234


# --------------------------------------------------------------------------
# launch
# --------------------------------------------------------------------------


def resolve_nproc() -> int:
    """Rank count to self-launch: what forge asked for, not a driver-side guess.

    forge exports its --nproc-per-node as FORGE_NPROC_PER_NODE. Launching a
    different count would benchmark a world nobody asked for, so honour it and
    let the caller's world-size check reject a mismatch loudly.
    """
    try:
        value = int(os.environ.get("FORGE_NPROC_PER_NODE") or 0)
    except ValueError:
        value = 0
    return value or WORLD


def relaunch_under_torchrun(nproc: int | None = None) -> None:
    """Re-exec the calling driver under torchrun when it was started bare.

    Not `start_new_session`: forge kills the whole process group on timeout and
    a detached torchrun would survive holding its GPUs.
    """
    if "RANK" in os.environ:
        return
    nproc = resolve_nproc() if nproc is None else nproc
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        os.path.abspath(sys.argv[0]),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))


def init_dist():
    """Bind this rank's device and join the process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.set_default_device("cuda")
    return dist.group.WORLD, dist.get_rank(), dist.get_world_size()


def shutdown_dist():
    """Drop the symmetric buffer and the group; a wedged one poisons the next stage.

    Runs in a `finally`, so it must never be the thing that fails the driver:
    teardown noise would be reported as a correctness or bench failure.
    """
    from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

    try:
        get_symm_buffer_for_mega_moe().destroy()
    except Exception:  # no active buffer, or a teardown that does not concern us
        pass
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def configure_env():
    """Env the drivers rely on. Must run before primus_turbo is imported."""
    # The disk cache hashes the top-level kernel but not the inlined tile helpers,
    # so a kernel edit would silently reuse stale code.
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    if os.environ.get("FORGE_CLEAR_TUNE_CACHE", "0") == "1":
        _clear_tune_cache()


def _clear_tune_cache():
    """Drop autotune's on-disk configs (only when the config space itself changed).

    The autotune *directory* is what carries the name; the files inside it are
    `_compiled_<kernel>.json`, so match on the directory and take every json in
    it. Their key is the problem shape with no source hash, which is why this
    exists at all: without a wipe, every candidate reuses the configs the
    baseline picked.
    """
    for root in (Path.home() / ".flydsl", HERE.parent):
        if not root.exists():
            continue
        for tune_dir in root.rglob("autotune"):
            if not tune_dir.is_dir():
                continue
            for path in tune_dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    pass


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def deepseek_routing(num_tokens, num_experts, num_topk, *, device="cuda"):
    """DeepSeek-V3 group-limited sigmoid top-k routing. Deterministic given the seed."""
    scores = torch.sigmoid(torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32))
    grouped = scores.view(num_tokens, NUM_GROUPS, num_experts // NUM_GROUPS)
    group_scores = grouped.topk(max(num_topk // GROUP_TOPK, 1), dim=-1).values.sum(dim=-1)
    selected_groups = group_scores.topk(GROUP_TOPK, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, NUM_GROUPS, num_experts // NUM_GROUPS)
        .reshape(num_tokens, num_experts)
    )
    topk_weight, topk_idx = torch.topk(scores.masked_fill(~expert_mask, float("-inf")), num_topk, dim=-1)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx.to(torch.int64), (topk_weight * ROUTING_SCALE).to(torch.float32)


def seed_rank(rank: int):
    """Per-rank deterministic seed: same inputs on every re-run of the driver."""
    torch.manual_seed(SEED + rank)


def pool_keyed_operand(symm, cols, *, scale, tokens, topk, world, dtype=torch.bfloat16):
    """A pool-shaped operand whose values follow the row's source token.

    Any operand indexed by pool row has to be built this way. A raw randn is not
    reproducible across processes: which token lands in which pool row comes
    from a device atomic, so the golden run and every later run pair different
    operand values with the same token, and the result differs by far more than
    any kernel change. The SNR would then be noise, and no candidate could ever
    be kept.

    The prologue records where each row came from (pool_src_rank/pool_src_slot,
    written as `token * num_topk + slot`), so keying off that pair makes the
    value a function of the token instead of the slot. Rows the prologue never
    wrote hold stale data and their key is meaningless -- they are clamped into
    range rather than masked, because no kernel reads a padding row.
    """
    slots = tokens * topk
    key = symm.pool_src_rank.long().clamp(0, world - 1) * slots + symm.pool_src_slot.long().clamp(
        0, slots - 1
    )
    # Cheap keyed pseudo-random: a real gather would need a world-sized table
    # (8 x 8192 x 8 rows) that does not fit at the bench shape.
    lanes = torch.arange(cols, device=key.device, dtype=torch.float32)
    values = torch.sin(key.unsqueeze(1).float() * 0.7071 + lanes * 0.0131)
    return values.mul_(scale).to(dtype)


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------


def project(out: torch.Tensor, *, ordered: bool = False):
    """Reduce an output to a small fingerprint. Returns ``(rows, matrix, gram)``.

    Pass ordered=True whenever the output's row order is deterministic, and only
    then. Combine writes row i of its output from token i, so its rows are fixed
    and the fingerprint keeps them: a bug that lands the right values in the
    wrong tokens is a routing bug, and the order-invariant form below scores it
    100 dB. Dispatch is the opposite case -- the pool slot a token lands in comes
    from a device atomic, so the same inputs give the same *set* of rows in a
    different order every run, and comparing element-wise would report two
    correct runs as uncorrelated.

    Order-invariant form: drop the padding rows, project the columns through a
    fixed random matrix (cheap, and keeps the fingerprint small), sort each
    projected column independently, keep PROJ_ROWS quantiles. The per-column
    sort is what makes it stable -- a swap of two rows moves each column's value
    by one slot in an already-sorted list instead of rewriting everything -- but
    it also throws away which value sat with which, so the Gram matrix rides
    along to catch values landing in the wrong row.

    The row count rides alongside rather than inside the tensor: folded in, its
    magnitude (~1e3) would dwarf the projected values and inflate every SNR.
    """
    flat = out.reshape(-1, out.shape[-1]).float()
    if ordered:
        gen = torch.Generator(device=flat.device).manual_seed(PROJ_SEED)
        right = torch.randn(flat.shape[-1], PROJ_DIM, generator=gen, device=flat.device, dtype=torch.float32)
        projected = flat @ right
        rows = projected.shape[0]
        gram = (projected.T @ projected) / max(rows, 1)
        return rows, projected.cpu(), gram.cpu()

    kept = flat[flat.abs().sum(dim=-1) > 0]
    if kept.numel() == 0:
        # .cpu() matters: the drivers set a cuda default device, so this would
        # otherwise be the one fingerprint on a different device than the golden,
        # and compare() would raise instead of reporting the zero output as 0 dB.
        zeros = torch.zeros(PROJ_ROWS, PROJ_DIM, device="cpu")
        return 0, zeros, torch.zeros(PROJ_DIM, PROJ_DIM, device="cpu")

    gen = torch.Generator(device=kept.device).manual_seed(PROJ_SEED)
    right = torch.randn(kept.shape[-1], PROJ_DIM, generator=gen, device=kept.device, dtype=torch.float32)
    projected = kept @ right
    reduced = torch.sort(projected, dim=0).values

    rows = reduced.shape[0]
    idx = torch.linspace(0, rows - 1, min(PROJ_ROWS, rows), device=kept.device).long()
    quantiles = reduced[idx]
    if quantiles.shape[0] < PROJ_ROWS:
        pad = torch.zeros(PROJ_ROWS - quantiles.shape[0], PROJ_DIM, device=kept.device)
        quantiles = torch.cat([quantiles, pad])

    # The quantiles alone are blind to a routing bug: sorting each column
    # independently also throws away which value sat with which, so a change
    # that lands the right numbers in the wrong rows keeps them identical. The
    # Gram matrix is the missing half -- row-order-invariant by construction,
    # but built from within-row products, so mixing rows moves it.
    gram = (projected.T @ projected) / rows
    return rows, quantiles.cpu(), gram.cpu()


def compare(ref, test) -> float:
    """SNR between two project() fingerprints, reported as the weaker half.

    A row-count change fails outright: it is a different result, not a noisier
    one, and an SNR over mismatched populations means nothing.
    """
    ref_rows, ref_q, ref_g = ref
    test_rows, test_q, test_g = test
    if ref_rows != test_rows:
        return 0.0
    return min(snr_db(ref_q, test_q), snr_db(ref_g, test_g))


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB. 100 dB means bit-identical."""
    ref = ref.float()
    test = test.float()
    if ref.shape != test.shape:
        return 0.0
    noise = test - ref
    sig_p = torch.mean(ref * ref).item()
    noise_p = torch.mean(noise * noise).item()
    # Signal first. An all-zero reference matched by an all-zero candidate is
    # two kernels that wrote nothing, not a perfect score.
    if sig_p <= 0:
        return 0.0
    if noise_p <= 0:
        return 100.0
    return 10.0 * math.log10(sig_p / noise_p)


def golden_path(name: str) -> Path:
    """Durable location beside the driver, so it survives every re-run."""
    return HERE / f"forge_golden_{name}.pt"


def save_golden(name: str, rank: int, projections: dict, group):
    """Gather every rank's projections onto rank 0 and write one file.

    Refused inside a campaign. The golden is the only thing standing between a
    wrong kernel and a KEEP verdict, and it is an untracked file no revert
    restores -- an optimizer that regenerates it has quietly replaced the
    reference with its own output and every later SNR reads 100 dB.
    """
    if os.environ.get("FORGE_NPROC_PER_NODE") or os.environ.get("GRAPH_PROBE_OUT"):
        raise RuntimeError(
            "refusing to write a golden from inside a forge run; regenerate it by hand on the pristine kernel"
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, projections, group=group)
    if rank == 0:
        payload = {
            "format": GOLDEN_FORMAT,
            "per_rank": gathered,
            "proj_dim": PROJ_DIM,
            "proj_rows": PROJ_ROWS,
            "proj_seed": PROJ_SEED,
        }
        torch.save(payload, golden_path(name))
        print(f"golden written: {golden_path(name)}", flush=True)
    dist.barrier(group)


def load_golden(name: str, rank: int, cases=()) -> dict:
    """Load this rank's reference, refusing anything the current code can't read."""
    path = golden_path(name)
    if not path.exists():
        raise FileNotFoundError(f"missing golden {path}; run the driver once with --make-golden")
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != GOLDEN_FORMAT:
        raise RuntimeError(
            f"{path} is format {payload.get('format')}, this driver needs {GOLDEN_FORMAT}; "
            "regenerate it with --make-golden"
        )
    # The projection parameters are part of the reference, not of the code that
    # reads it: change PROJ_SEED alone and every candidate silently scores 0 dB.
    for key, want in (("proj_dim", PROJ_DIM), ("proj_rows", PROJ_ROWS), ("proj_seed", PROJ_SEED)):
        if payload.get(key) != want:
            raise RuntimeError(
                f"{path} was written with {key}={payload.get(key)}, this driver uses {want}; "
                "regenerate it with --make-golden"
            )
    if rank >= len(payload["per_rank"]):
        raise RuntimeError(f"{path} covers {len(payload['per_rank'])} ranks, this run has rank {rank}")
    golden = payload["per_rank"][rank]
    missing = [case for case in cases if case not in golden]
    if missing:
        raise RuntimeError(f"{path} has no reference for {missing}; regenerate it with --make-golden")
    return golden


def baseline_path(name: str) -> Path:
    """Where the pristine per-case times live, beside the golden."""
    return HERE / f"forge_baseline_{name}.json"


def save_baseline(name: str, case_times: dict, rank: int, group):
    """Record the pristine per-case times a driver-side regression guard reads.

    Refused inside a campaign, for the same reason save_golden is: recorded from
    a candidate's run it would enshrine that candidate's regression as the floor,
    and the guard would then wave through everything slower than the kernel the
    campaign started from.
    """
    if os.environ.get("FORGE_NPROC_PER_NODE") or os.environ.get("GRAPH_PROBE_OUT"):
        raise RuntimeError(
            "refusing to write a baseline from inside a forge run; record it by hand on the pristine kernel"
        )
    if rank == 0:
        payload = {"case_ms": {str(case): float(ms) for case, ms in case_times.items()}}
        baseline_path(name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"baseline written: {baseline_path(name)}", flush=True)
    dist.barrier(group)


def load_baseline(name: str) -> dict:
    """Per-case pristine times, or {} when none was ever recorded.

    Missing is not an error: the guard it feeds is an extra gate on top of the
    measurement, and a driver run by hand before the baseline exists should still
    report its timings rather than refuse to run.
    """
    path = baseline_path(name)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {str(case): float(ms) for case, ms in (payload.get("case_ms") or {}).items()}


# --------------------------------------------------------------------------
# correctness: the benchmark script's in-process reference
# --------------------------------------------------------------------------

# gate3 passes at rel_rmse <= 0.05, which is this in dB. A campaign's
# --snr-threshold set at or below it reproduces the benchmark's own verdict.
REF_GATE_DB = 26.02
REF_MODEL = "DeepSeek-V3"
REF_TOKENS = 8192


def load_reference_module():
    """Import benchmark/ops/training/bench_mega_moe.py, the accuracy authority.

    Its own directory has to be importable: it does `from config import ...` and
    expects to have been started from there. Appended rather than inserted --
    `config` is a name worth not hoisting over anything else. Imported on demand
    so a bench or profile run does not pay for pandas and tabulate.

    It is under benchmark/, which forge's workspace policy protects from edits,
    so a campaign cannot weaken the oracle it is judged by.
    """
    ref_dir = HERE.parent / "benchmark" / "ops" / "training"
    if str(ref_dir) not in sys.path:
        sys.path.append(str(ref_dir))
    import bench_mega_moe

    return bench_mega_moe


def snr_from_check(check) -> float:
    """One gate3 verdict as the dB figure forge's threshold reads.

    rel_rmse is ||out - ref|| / ||ref||, so -20log10(rel) is the same SNR
    snr_db() computes. Capped at 100 dB like snr_db, because these references
    land on rel = 0.0 exactly. A stage gate3 failed reports 0 dB rather than its
    own rel: forge reads one number, and the cosine is the half of gate3 that
    number cannot otherwise carry.
    """
    if check is None or not bool(check.ok):
        return 0.0
    return min(-20.0 * math.log10(max(float(check.rel), 1e-12)), 100.0)


def reference_snrs(group, rank: int, modes, *, model=REF_MODEL, tokens=REF_TOKENS, iters=1) -> dict:
    """Run the benchmark's accuracy gate for `modes`; return {case_id: dB}.

    `modes` is a sequence of (mode name, {benchmark stage key: our case id}) --
    one entry per operator the caller measures, in pipeline order.

    Everything up to and including the per-rank seed mirrors the benchmark's own
    case loop, so the stages see exactly the inputs their references were
    written for. `iters` only feeds the stage runner's timing calls, which the
    drivers do not read; one is enough to reach the accuracy probe.
    """
    from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

    ref = load_reference_module()
    ref_args = ref._build_parser().parse_args(
        ["--mode", modes[0][0], "--num-tokens", str(tokens), "--iters", str(iters)]
    )
    cases = ref.gen_moe_test_cases([model])
    if not cases:
        raise ValueError(f"unknown reference model {model!r}")
    ref.apply_case(ref_args, cases[0])

    snrs = {}
    for mode_name, stage_cases in modes:
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=ref_args.num_experts,
            num_max_tokens_per_rank=ref_args.num_tokens,
            num_topk=ref_args.num_topk,
            hidden=ref_args.hidden,
            intermediate_hidden=ref_args.inter,
        )
        ref.sync_ranks(group)
        # Per-rank seed so ranks get distinct tokens/routing, exactly as the
        # benchmark does before it calls profile().
        torch.manual_seed(rank)
        checks = ref.MODES[mode_name].profile(group, ref_args, symm)["checks"]
        for stage, case in stage_cases.items():
            snrs[case] = snr_from_check(checks.get(stage))
        ref.sync_ranks(group)
        # Hand the next mode back what this one used. profile() drops its own
        # context on return, but the caching allocator keeps the blocks, and a
        # caller running both modes needs the peak of one, not the sum: at the
        # DeepSeek-V3 shape the difference is what makes it fit next to somebody
        # else's job on the same cards.
        torch.cuda.empty_cache()
    return snrs


def report_snr(per_case_snr: dict, rank: int, group):
    """Print the WORST rank's worst case: one wrong rank is a wrong collective."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, per_case_snr, group=group)
    if rank != 0:
        return
    values = [v for d in gathered if d for v in d.values()]
    if not values:
        raise RuntimeError("no case produced an SNR; the driver measured nothing")
    worst = min(values)
    for case in sorted(per_case_snr):
        across = [d[case] for d in gathered if case in d]
        print(f"case_snr: {case} {min(across):.2f}")
    print(f"SNR: {worst:.2f} dB")
    print(f"allclose: {bool(worst >= 30.0)}")


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def cuda_graph_bench(step, *, warmup, iters, group, dirty=None, verify=None):
    """Capture `step` into a HIP graph and time replays.

    The ranks are re-synced before every timed replay: back-to-back launches let
    them drift, and the drift then shows up inside the kernel as time spent
    waiting on peer pushes rather than as real work.

    `dirty(out)` runs before every replay and `verify(out)` once at the end;
    both act on the graph-owned output tensor, which is the only way to tell a
    real replay apart from a silent no-op.
    """
    # Warm up eagerly first: this is where the flydsl JIT compiles and the
    # autotuner benches its configs, neither of which can happen under capture.
    for _ in range(max(warmup, 1)):
        step()
        torch.cuda.synchronize()
        dist.barrier(group)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier(group)
    with torch.cuda.graph(graph):
        out = step()
    torch.cuda.synchronize()
    dist.barrier(group)

    # One untimed replay so the first timed sample is not the odd one out.
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier(group)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier(group)
        if dirty is not None:
            # Fenced on both sides. Before: a peer finishing the last replay may
            # still be pushing into the buffers we are about to zero. After: a
            # peer that scrubbed faster would already be replaying and pushing
            # into ours, and our memset would erase what it just delivered.
            dirty(out)
            torch.cuda.synchronize()
            dist.barrier(group)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    # Barrier first, raise after, and raise on every rank. Raising before the
    # collective would leave the other seven inside it until the stage timeout
    # (300 s for bench) instead of failing the candidate in seconds.
    bad = torch.tensor([0 if verify is None or verify(out) else 1], device="cuda")
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group)
    dist.barrier(group)
    if int(bad.item()):
        raise RuntimeError("graph replay did not produce a valid result")
    # The graph rides along deliberately: `out` lives in the graph's private
    # memory pool, so the caller must be able to keep both alive together.
    return {"times_ms": times, "out": out, "graph": graph}


def report_case(case_id: str, rounds: list, rank: int, group, tag: str = ""):
    """Print the per-round samples and the case time; return it on rank 0.

    Each round is reduced with a per-quantile MAX over the ranks' sorted
    samples: a collective is only as fast as its slowest participant, and
    matching sample k to sample k keeps the spread meaningful instead of mixing
    ranks. The case time is then the median of the per-round medians, not of the
    pooled samples -- pooling hides the between-round drift that repeating the
    measurement exists to expose.

    `tag` rides along on the case line: "unscored" tells forge to measure and
    show the case but keep it out of the KEEP score. The return value is for a
    driver that aggregates several cases into one scored total -- it has to sum
    them from here rather than print its own case line, because forge fails a
    run that reports the same case id twice.
    """

    def _slowest(samples):
        local = torch.tensor(sorted(samples), dtype=torch.float64, device="cuda")
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=group)
        return local.cpu().tolist()

    medians = []
    for samples in rounds:
        slowest = _slowest(samples)
        medians.append(slowest[len(slowest) // 2])
        if rank == 0:
            for value in slowest:
                print(f"wall_ms: {value:.6f}")
    if rank != 0:
        return None
    medians.sort()
    case_ms = medians[len(medians) // 2]
    print(f"case_ms: {case_id} {case_ms:.6f}{' ' + tag if tag else ''}", flush=True)
    return case_ms


def add_common_args(parser):
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1, help="median over N in-process repeats")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--bn", type=int, default=256)
    # Correctness knobs: which geometry the in-process reference runs at. There
    # is no --make-golden any more; see this module's docstring.
    parser.add_argument("--ref-model", default=REF_MODEL, help="MoE model the correctness reference runs at")
    parser.add_argument("--ref-tokens", type=int, default=REF_TOKENS, help="tokens per rank for correctness")
    parser.add_argument(
        "--ref-iters",
        type=int,
        default=1,
        help="the reference runner times each stage before checking it; the drivers do not read "
        "those timings, so one iteration is enough",
    )
    return parser
