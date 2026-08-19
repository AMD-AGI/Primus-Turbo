###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""forge-loop measurement driver for the mega MoE layer: dispatch AND combine.

The two operators are one campaign's worth of work, not two: they share
ep_intranode, symm_buffer, prims and the GEMM tile, they hand each other the
dispatch handle, and they read and write the same pool. Optimised apart, a pool
layout that pays for itself only across the pair is a change neither campaign
can propose, and a regression one side lands in a shared file is invisible to
the other. This driver measures both, so one iteration sees both.

Five cases -- the GEMM legs of one MoE layer step, minus dW2:
    dispatch_nt   forward L1        x @ w1          -> pool activations
    dispatch_nn   backward L2 dgrad dy @ w2         -> pool grad
    dispatch_tn   backward dW1      pool(x)^T @ g   -> per-expert weight grad
    combine_nt    forward L2        act @ w2        -> combine + topk reduce
    combine_nn    backward L1 dgrad g  @ w1         -> combine + gate-grad scatter

Scored as one number, guarded per side:
  * `case_ms: megamoe_total` is the only scored case, so forge's mean case
    speedup is exactly the total-time speedup.
  * the five legs are reported `unscored` -- visible to the agent, the analysis
    pass and the logs, but out of the KEEP score.
  * a leg slower than the recorded pristine baseline by more than --tol fails
    the run, which is how "and neither side regresses" gets enforced: forge
    cannot express it, but it does report a non-zero driver exit as a failed
    candidate and hands the agent this driver's own words for why. The gate is
    armed only when the total is at or under its floor -- see _guard_regression.
    FORGE_MEGAMOE_GUARD=off disables it; FORGE_MEGAMOE_TOL sets the tolerance.

Correctness has no golden. It defers to the in-process reference in
benchmark/ops/training/bench_mega_moe.py, which is the accuracy contract these
kernels are actually developed against -- see run_correctness for why the
golden fingerprint cannot be used on this build.

Usage (forge-loop drives these, the driver launches its own 8 ranks):
    python forge_driver_megamoe.py                                 # SNR
    python forge_driver_megamoe.py --warmup 10 --iters 30 --bench-mode
    python forge_driver_megamoe.py --profile-run

One-off, before the campaign starts, on the pristine kernels:
    python forge_driver_megamoe.py --warmup 10 --iters 30 --repeat 3 \
        --bench-mode --make-baseline
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import forge_harness as fh  # noqa: E402

fh.configure_env()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

# The per-operator drivers are the definition of what each case *is*; importing
# them keeps one description of the operands, the step and the scrub instead of
# a third copy that drifts from both.
import forge_driver_combine as C  # noqa: E402
import forge_driver_dispatch as D  # noqa: E402

NAME = "megamoe"
TOTAL_CASE = "megamoe_total"
# Wide enough that ordinary run-to-run drift does not read as a regression --
# every candidate is measured three times and any one of them tripping this
# throws the candidate away. Measured on an idle box: case medians land within
# 1% of the recorded floor.
DEFAULT_TOL = 0.03
# Non-zero and not 1: 1 is "the driver raised", and the two want telling apart.
GUARD_EXIT = 2
# torchrun prints a SIGTERM traceback for all seven surviving ranks after rank 0
# exits, and forge keeps only the last 2000 characters of the run's output --
# so the reason gets pushed out of the window by the noise about the symptom.
# Rank 0 leaves it here and the parent re-emits it once torchrun is done.
GUARD_NOTE = fh.HERE / "forge_megamoe_guard.txt"

CASES = tuple([D.CASES[layout] for layout in D.LAYOUTS] + [C.CASES[layout] for layout in C.LAYOUTS])

# Correctness is delegated to benchmark/ops/training/bench_mega_moe.py; this maps
# its per-mode stage keys onto the case ids the rest of this driver reports.
REF_MODES = (
    ("dispatch_grouped_gemm", {"fwd": "dispatch_nt", "bwd": "dispatch_nn", "wgrad": "dispatch_tn"}),
    ("grouped_gemm_combine", {"fwd": "combine_nt", "bwd": "combine_nn"}),
)
# The campaign's --snr-threshold is set to forge_harness.REF_GATE_DB so that
# forge's verdict is the benchmark's verdict; see run_forge_megamoe.sh.


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
#
# One phase per operator, in pipeline order. Each returns the same shape so the
# three run modes below can stay operator-agnostic; the adapters exist only
# because combine needs the flattened topk indices threaded into its step.


def _dispatch_phase(shape, group, rank, world, bn):
    symm, handle, operands = D.build_context(shape, group, rank, world)
    steps = {D.CASES[layout]: D.make_step(layout, operands, group, handle, bn) for layout in D.LAYOUTS}
    # ordered=False: dispatch writes in pool space, whose row order the prologue's
    # slot atomics decide anew on every process. See forge_harness.project().
    return {"symm": symm, "steps": steps, "scrub": D.scrub(symm), "ordered": False}


def _combine_phase(shape, group, rank, world, bn):
    symm, handle, operands, topk_indices_flat = C.build_context(shape, group, rank, world)
    steps = {
        C.CASES[layout]: C.make_step(layout, operands, handle, topk_indices_flat, bn)
        for layout in C.LAYOUTS
    }
    # ordered=True: combine's output row i comes from token i, so row order is
    # part of the answer.
    return {"symm": symm, "steps": steps, "scrub": C.scrub(symm), "ordered": True}


PHASES = (_dispatch_phase, _combine_phase)


def _release(ctx, group):
    """Drop the phase's graphs and hand the next one a clean pool.

    The symmetric buffer is a process-wide singleton, so without this the
    combine phase inherits whatever dispatch left in the pool -- and since the
    campaign is allowed to change dispatch, combine's SNR would then move for
    reasons that have nothing to do with combine. Fenced on both sides like
    every other scrub here: a peer may still be pushing into these buffers.

    One memset per phase, not per replay -- the per-replay cost the bench mode
    refuses to pay (a 7.6 GB pool zeroed 90 times) is not this.
    """
    symm = ctx["symm"]
    torch.cuda.synchronize()
    dist.barrier(group)
    for buffer in (symm.dispatch_token_pool, symm.l2_token_buffer, symm.combine_token_buffer):
        buffer.zero_()
    torch.cuda.synchronize()
    dist.barrier(group)
    ctx.clear()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


def run_correctness(args, group, rank, world):
    """Gate both operators the way bench_mega_moe.py gates them.

    Both modes in one call, in pipeline order, so a candidate is validated on
    the pair it is scored on. The reference itself -- and why it replaced the
    golden -- is forge_harness.reference_snrs.

    check_accuracy has already reduced each stage across ranks, so every rank
    holds the same numbers; report_snr gathers again, which costs nothing and
    keeps this driver's output the same shape as the other two.
    """
    snrs = fh.reference_snrs(
        group, rank, REF_MODES, model=args.ref_model, tokens=args.ref_tokens, iters=args.ref_iters
    )
    fh.report_snr(snrs, rank, group)
    return 0


def run_bench(args, group, rank, world):
    case_times = {}
    if rank == 0:
        GUARD_NOTE.unlink(missing_ok=True)
    for phase in PHASES:
        ctx = phase(fh.BENCH_SHAPE, group, rank, world, args.bn)
        for case, step in ctx["steps"].items():
            rounds = []
            for _ in range(max(args.repeat, 1)):
                result = fh.cuda_graph_bench(
                    step,
                    warmup=args.warmup,
                    iters=args.iters,
                    group=group,
                    # No scrub here; see the per-operator drivers for why.
                    dirty=None,
                    verify=lambda out: bool(torch.isfinite(out.reshape(-1)[:4096].float()).all()),
                )
                rounds.append(result["times_ms"])
                del result
            case_ms = fh.report_case(case, rounds, rank, group, tag="unscored")
            if case_ms is not None:
                case_times[case] = case_ms
        _release(ctx, group)

    # The scored line comes last and on its own: every leg is measured before
    # anything is judged, so a rejected run still shows the full picture.
    if rank == 0:
        case_times[TOTAL_CASE] = sum(case_times.values())
        print(f"case_ms: {TOTAL_CASE} {case_times[TOTAL_CASE]:.6f}", flush=True)

    if args.make_baseline:
        fh.save_baseline(NAME, case_times, rank, group)
        return 0
    return _guard_regression(case_times, rank, group, args.tol)


def _guard_regression(case_times, rank, group, tol):
    """Fail the run when either operator is slower than the pristine baseline.

    forge scores this driver on megamoe_total alone, so on its own it would keep
    a candidate that bought a large win on one operator by giving a little back
    on the other. That is the half of "total faster AND neither side regresses"
    that has to live here. A non-zero exit is the supported way to say it: forge
    reports it as a failed candidate and attaches the tail of this output.

    Only armed when the total is no slower than the floor, which is the one
    thing that keeps this gate from taking down the whole campaign. A run whose
    total regressed cannot be kept by forge anyway (KEEP needs the total speedup
    above the incumbent), so rejecting it here adds nothing -- while a machine
    with someone else's job on it slows every case at once and would otherwise
    trip this during forge's own baseline measurement, where a failure is fatal
    rather than a rejected candidate. That is not hypothetical: it is how the
    first megamoe campaign died, two minutes in.
    """
    bad = torch.zeros(1, device="cuda")
    if rank == 0:
        baseline = fh.load_baseline(NAME)
        reference_total = baseline.get(TOTAL_CASE)
        measured_total = case_times.get(TOTAL_CASE)
        if not baseline:
            print(
                f"note: no {fh.baseline_path(NAME).name}; per-case regression guard is off",
                flush=True,
            )
        elif os.environ.get("FORGE_MEGAMOE_GUARD", "").strip().lower() in {"0", "off", "false"}:
            print("note: FORGE_MEGAMOE_GUARD is off; per-case regression guard skipped", flush=True)
        elif reference_total and measured_total and measured_total > reference_total:
            print(
                f"note: total {measured_total:.3f} ms is above the {reference_total:.3f} ms floor "
                "(nothing here can be kept); per-case regression guard not applied",
                flush=True,
            )
        else:
            lines = []
            for case in sorted(case_times):
                reference = baseline.get(case)
                if not reference:
                    continue
                delta = case_times[case] / reference - 1.0
                if delta > tol:
                    lines.append(
                        f"REGRESSION: {case} {case_times[case]:.6f} ms vs baseline "
                        f"{reference:.6f} ms ({delta * 100:+.1f}%, tol {tol * 100:.1f}%)"
                    )
            if lines:
                lines.append(
                    "bench rejected: the total improved by making one operator slower, "
                    "which this campaign does not accept"
                )
                report = "\n".join(lines)
                print(report, flush=True)
                GUARD_NOTE.write_text(report + "\n")
                bad += 1

    # Every rank has to learn the verdict. Rank 0 exiting alone would leave the
    # other seven waiting inside the next collective until the stage timeout,
    # turning a two-second rejection into a five-minute one.
    dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group)
    dist.barrier(group)
    return GUARD_EXIT if int(bad.item()) else 0


def _profile_selection(selected):
    """Which cases --profile-run should run."""
    if not selected:
        # Both operators' forward legs: the profile of a joint campaign should
        # show both kernels and their real share of the layer, not one of them.
        return (D.CASES["nt"], C.CASES["nt"])
    if selected == TOTAL_CASE:
        return CASES
    if selected not in CASES:
        raise ValueError(
            f"unknown --profile-case {selected!r}; choose one of {', '.join(CASES)} or {TOTAL_CASE}"
        )
    return (selected,)


def run_profile(args, group, rank, world):
    wanted = _profile_selection(args.profile_case)
    for phase in PHASES:
        ctx = phase(fh.BENCH_SHAPE, group, rank, world, args.bn)
        for case, step in ctx["steps"].items():
            if case not in wanted:
                continue
            for _ in range(3):
                step()
                torch.cuda.synchronize()
                dist.barrier(group)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            dist.barrier(group)
        _release(ctx, group)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fh.add_common_args(parser)
    # Kept off add_common_args: the per-operator drivers implement neither, and
    # --profile-case in particular is probed for by grepping a driver's --help.
    parser.add_argument(
        "--make-baseline",
        action="store_true",
        help="record the current per-case times as the regression guard's floor",
    )
    parser.add_argument(
        "--profile-case",
        default="",
        help=f"narrow --profile-run to one case ({TOTAL_CASE} means all five)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        # `or`, not a get() default: the runner exports the variable whether or
        # not the user set it, so an empty string is the common case.
        default=float(os.environ.get("FORGE_MEGAMOE_TOL") or DEFAULT_TOL),
        help="per-case regression tolerance, as a fraction of the baseline time",
    )
    args = parser.parse_args()
    # Recording the floor is a benchmark that happens to write a file, and it
    # has to be measured the way the campaign will measure it.
    if args.make_baseline:
        args.bench_mode = True

    # Clear the note here, before the ranks exist, so that finding one after
    # the run can only mean this run's guard wrote it.
    if "RANK" not in os.environ:
        GUARD_NOTE.unlink(missing_ok=True)
    try:
        fh.relaunch_under_torchrun()
    except SystemExit as stop:
        # Only the parent gets here. torchrun's epilogue is the last thing on
        # stderr and forge keeps only the tail, so say it again, after it, on
        # the same stream -- otherwise the campaign log shows seven SIGTERM
        # tracebacks and no reason.
        #
        # Keyed on the note, not on the exit code: torchrun reports its own
        # exit 1 for any worker failure, so rank 0's GUARD_EXIT never reaches
        # here (it is still worth returning -- it is what torchrun prints as
        # the root cause's exitcode).
        if stop.code and GUARD_NOTE.exists():
            print(GUARD_NOTE.read_text().rstrip(), file=sys.stderr, flush=True)
        raise

    group, rank, world = fh.init_dist()
    if world != fh.WORLD:
        raise ValueError(f"this driver requires EP{fh.WORLD}, got EP{world}")
    try:
        if args.profile_run:
            return run_profile(args, group, rank, world)
        if args.bench_mode:
            return run_bench(args, group, rank, world)
        return run_correctness(args, group, rank, world)
    finally:
        fh.shutdown_dist()


if __name__ == "__main__":
    sys.exit(main())
