###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Benchmark driver for ``DeepEPTokenDispatcher`` across EP backends.

Opt-in performance pass: this script does **nothing** unless ``--bench`` is
passed. With ``--bench`` and a model preset (``--model deepseek-v3``, etc.),
it sweeps the EP backends registered in Primus-Turbo (``MORI`` / ``UCCL`` /
``TURBO`` / ``DEEP_EP``) for a fixed ``(num_tokens, hidden, num_experts,
num_topk)`` shape and reports, per backend:

  * end-to-end ``token_dispatch`` / ``token_combine`` wall-clock latency
    (us) + per-rank bandwidth (GB/s);
  * a per-module kineto kernel-time breakdown -- ``dispatch``, ``permute``,
    ``unpermute``, ``combine`` -- and the end-to-end sum of those four.

MORI is the only backend that runs a post-processing kernel
(``ConvertDispatchOutputKernel``) after its EP all-to-all, so its
``dispatch`` column is printed as ``a + b = total`` (EP kernel + post-process
kernel). Every other backend prints a single dispatch number.

Launch model is ``torchrun`` with ``world_size == EP_SIZE``::

    torchrun --standalone --nproc_per_node=8 bench_token_dispatcher.py \\
        --bench --model deepseek-v3 --backends MORI,TURBO

Multi-node (EP=16 across 2 x 8):

    EP_SIZE=16 torchrun --nnodes=2 --nproc_per_node=8 \\
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \\
        bench_token_dispatcher.py --bench --model deepseek-v3
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
    _BACKEND_REGISTRY,
    clear_backend_instances,
)

# Hard-coded so the script works against older Primus-Turbo wheels that
# pre-date the ``ENV_EP_BACKEND`` constant in ``primus_turbo.common.constants``.
ENV_EP_BACKEND = "PRIMUS_TURBO_EP_BACKEND"
ENV_AUTO_TUNE = "PRIMUS_TURBO_AUTO_TUNE"

# Pseudo-backend that runs the dispatcher with PRIMUS_TURBO_AUTO_TUNE=1 and
# no PRIMUS_TURBO_EP_BACKEND pinned, so the real autotune path inside
# moe_dispatch_combine_impl picks the fastest registered backend on the
# first dispatch for the given shape.
AUTOTUNE_LABEL = "AUTOTUNE"

# Model presets: numbers come from each model's published HF config.
MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    "deepseek-v3": dict(num_experts=256, num_topk=8, hidden=7168),
    "kimi-k2": dict(num_experts=384, num_topk=8, hidden=7168),
    "qwen3-235b-a22b": dict(num_experts=128, num_topk=8, hidden=4096),
    "gpt-oss-120b": dict(num_experts=128, num_topk=4, hidden=2880),
}


# ---------------------------------------------------------------------------
# Kineto kernel taxonomy
# ---------------------------------------------------------------------------
# Five buckets, classified by first-substring-match against the (lowercased)
# kernel name. Order matters: MORI's ``ConvertDispatchOutputKernel`` must hit
# ``ep_dispatch_post`` before falling into ``ep_dispatch``'s catch-all
# ``dispatch<`` pattern. Anything not in these buckets (memcpy, elementwise
# mul, fill_, etc.) is discarded -- it's noise from the bench harness, not
# part of the dispatcher's per-module cost.
KERNEL_BUCKETS: Dict[str, List[str]] = {
    # MORI's post-dispatch convert kernel(s). This is *only* what makes
    # MORI's "dispatch" column render as ``base + post = total``; for every
    # other backend this bucket stays at 0.
    "ep_dispatch_post": [
        "convertdispatchoutputkernel",
        "convert_dispatch_output",
        "localexpertcountkernel",
        "local_expert_count",
        "compute_expert_token_info",
    ],
    "ep_dispatch": [
        # mori
        "epdispatchintranodekernel",
        "epdispatchinternodekernel",
        "epdispatchinternodev1",
        "epdispatchlowlatencyasync",
        "ep_dispatch_kernel",
        "mori::dispatch",
        "mori_dispatch",
        # turbo / deep_ep
        "notify_dispatch",
        "cached_notify_dispatch",
        "get_dispatch_layout",
        # uccl
        "uccl::ep::dispatch",
        # catch-all (must stay last)
        "dispatch<",
        "::dispatch(",
    ],
    "ep_combine": [
        "epcombineintranodekernel",
        "epcombineinternodekernel",
        "epcombineinternodev1",
        "epcombinelowlatencyasync",
        "ep_combine_kernel",
        "mori::combine",
        "mori_combine",
        "convertcombineinputkernel",
        "convert_combine_input",
        "cached_notify_combine",
        "uccl::ep::combine",
        "combine<",
        "::combine(",
    ],
    "permute": [
        # turbo triton permute pipeline.
        "_row_id_map_pass",
        "_permute_kernel",
        "_indices_to_multihot_kernel",
        "make_row_id_map",
        "permute_with_mask_map",
    ],
    "unpermute": [
        "_unpermute_kernel",
        "unpermute_with_mask_map",
    ],
}


def _classify(name: str) -> Optional[str]:
    nlow = name.lower()
    for bucket, patterns in KERNEL_BUCKETS.items():
        if any(p in nlow for p in patterns):
            return bucket
    return None


# ---------------------------------------------------------------------------
# Distributed bring-up
# ---------------------------------------------------------------------------


def init_dist() -> Tuple[int, int, dist.ProcessGroup]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return rank, world_size, dist.new_group(list(range(world_size)))


def _print0(*args, **kwargs) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, flush=True, **kwargs)


# ---------------------------------------------------------------------------
# Backend selection + dispatcher build
# ---------------------------------------------------------------------------


def resolve_backends(requested: List[str]) -> List[str]:
    out: List[str] = []
    for name in requested:
        up = name.strip().upper()
        if up == AUTOTUNE_LABEL:
            out.append(up)
            continue
        cls = _BACKEND_REGISTRY.get(up)
        if cls is None:
            _print0(f"[skip] unknown backend '{name}' (known: {list(_BACKEND_REGISTRY)} + {AUTOTUNE_LABEL})")
            continue
        try:
            available = bool(cls.is_available())
        except Exception:
            available = False
        if not available:
            _print0(f"[skip] backend '{up}' not importable in this container")
            continue
        out.append(up)
    if not out:
        raise RuntimeError(f"No EP backend from {requested} is available.")
    return out


def build_dispatcher(
    backend: str,
    group: dist.ProcessGroup,
    num_experts: int,
    num_topk: int,
    num_sms: int,
) -> "turbo.modules.DeepEPTokenDispatcher":
    # AUTOTUNE: drop the pinned backend env, set the autotune env, let
    # moe_dispatch_combine_impl sweep on the first dispatch.
    if backend == AUTOTUNE_LABEL:
        os.environ.pop(ENV_EP_BACKEND, None)
        os.environ[ENV_AUTO_TUNE] = "1"
    else:
        os.environ[ENV_EP_BACKEND] = backend
        os.environ.pop(ENV_AUTO_TUNE, None)
    clear_backend_instances()
    return turbo.modules.DeepEPTokenDispatcher(
        num_experts=num_experts,
        router_topk=num_topk,
        ep_group=group,
        permute_fusion=True,
        deepep_num_use_cu=num_sms,
    )


# ---------------------------------------------------------------------------
# Shape + inputs
# ---------------------------------------------------------------------------


@dataclass
class Shape:
    num_tokens: int
    hidden: int
    num_experts: int
    num_topk: int
    dtype: torch.dtype = torch.bfloat16


def make_inputs(shape: Shape) -> Tuple[torch.Tensor, torch.Tensor]:
    hs = torch.randn(
        (shape.num_tokens, shape.hidden),
        dtype=shape.dtype,
        device="cuda",
    )
    # Uniform topk distribution so combine reduction exactly cancels dispatch
    # (matters for correctness checks; here it just makes the workload
    # deterministic across backends).
    probs = (
        torch.ones((shape.num_tokens, shape.num_experts), dtype=torch.float32, device="cuda") / shape.num_topk
    )
    return hs, probs


def _bandwidth_gbs(shape: Shape, t_s: float) -> float:
    """Per-rank GB/s upper bound. Each token replicates to ``num_topk``
    destinations during dispatch (and reduces back during combine)."""
    if t_s <= 0:
        return 0.0
    nbytes = (
        shape.num_tokens * shape.num_topk * shape.hidden * torch.tensor([], dtype=shape.dtype).element_size()
    )
    return nbytes / 1e9 / t_s


# ---------------------------------------------------------------------------
# Wall-clock latency: time the dispatcher's public API end-to-end.
# ---------------------------------------------------------------------------


@dataclass
class WallClock:
    dispatch_us: float
    combine_us: float
    dispatch_gbs: float
    combine_gbs: float

    @property
    def e2e_us(self) -> float:
        return self.dispatch_us + self.combine_us


def _events(n: int) -> List[torch.cuda.Event]:
    return [torch.cuda.Event(enable_timing=True) for _ in range(n)]


def _mean_s(starts: List[torch.cuda.Event], ends: List[torch.cuda.Event]) -> float:
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / len(starts) * 1e-3


@torch.no_grad()
def bench_wallclock(
    dispatcher,
    shape: Shape,
    *,
    num_warmup: int,
    num_tests: int,
) -> WallClock:
    """Time ``token_dispatch`` and ``token_combine`` in separate CUDA-event
    windows. Each window is preceded by a 256 MB L2-flush -- MI355X's
    Infinity Cache would otherwise mask the comm latency."""
    hs, probs = make_inputs(shape)
    flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

    def one_pair() -> None:
        p, _, w = dispatcher.token_dispatch(hs, probs)
        dispatcher.token_combine((p * w.unsqueeze(-1)).to(shape.dtype))

    for _ in range(num_warmup):
        one_pair()
    torch.cuda.synchronize()

    ds, de = _events(num_tests), _events(num_tests)
    cs, ce = _events(num_tests), _events(num_tests)
    for i in range(num_tests):
        flush.zero_()
        ds[i].record()
        p, _, w = dispatcher.token_dispatch(hs, probs)
        de[i].record()
        # Local elementwise (mul + cast) is intentionally outside both windows
        # but still issued so the dispatcher handle is consumed before combine.
        p2 = (p * w.unsqueeze(-1)).to(shape.dtype)
        flush.zero_()
        cs[i].record()
        dispatcher.token_combine(p2)
        ce[i].record()
    torch.cuda.synchronize()

    d_s, c_s = _mean_s(ds, de), _mean_s(cs, ce)
    return WallClock(
        dispatch_us=d_s * 1e6,
        combine_us=c_s * 1e6,
        dispatch_gbs=_bandwidth_gbs(shape, d_s),
        combine_gbs=_bandwidth_gbs(shape, c_s),
    )


# ---------------------------------------------------------------------------
# Per-module kineto breakdown
# ---------------------------------------------------------------------------


@dataclass
class ModuleBreakdown:
    """Per-iter kineto kernel time (us), per module. For non-MORI backends
    ``ep_dispatch_post`` stays at 0."""

    ep_dispatch: float = 0.0
    ep_dispatch_post: float = 0.0
    permute: float = 0.0
    unpermute: float = 0.0
    ep_combine: float = 0.0

    @property
    def dispatch_total(self) -> float:
        return self.ep_dispatch + self.ep_dispatch_post

    @property
    def e2e_us(self) -> float:
        return self.dispatch_total + self.permute + self.unpermute + self.ep_combine


def _aggregate_trace(
    prof: "torch.profiler.profile",
    *,
    num_active: int,
    trace_path: Optional[str],
) -> ModuleBreakdown:
    cleanup = trace_path is None
    if trace_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        tmp.close()
        trace_path = tmp.name
    else:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(trace_path)
    try:
        with open(trace_path) as fh:
            data = json.load(fh)
    finally:
        if cleanup:
            try:
                os.unlink(trace_path)
            except OSError:
                pass

    bucket_us: Dict[str, float] = {b: 0.0 for b in KERNEL_BUCKETS}
    # Chrome-trace ``ph == "X"`` is a complete (start+dur) event; the kernel
    # category appears as "kernel" or "Kernel" across torch versions; ``dur``
    # is in microseconds.
    for ev in data.get("traceEvents", []):
        if ev.get("ph") != "X" or ev.get("cat") not in ("kernel", "Kernel"):
            continue
        dur = float(ev.get("dur", 0.0))
        if dur <= 0.0:
            continue
        bucket = _classify(ev.get("name", ""))
        if bucket is not None:
            bucket_us[bucket] += dur

    n = max(num_active, 1)
    return ModuleBreakdown(
        ep_dispatch=bucket_us["ep_dispatch"] / n,
        ep_dispatch_post=bucket_us["ep_dispatch_post"] / n,
        permute=bucket_us["permute"] / n,
        unpermute=bucket_us["unpermute"] / n,
        ep_combine=bucket_us["ep_combine"] / n,
    )


@torch.no_grad()
def bench_kineto(
    dispatcher,
    shape: Shape,
    *,
    num_warmup: int,
    num_active: int,
    trace_path: Optional[str] = None,
) -> ModuleBreakdown:
    """Run a torch.profiler kineto pass on a dispatch+combine pair, then
    bucket every CUDA/HIP kernel into the 5 KERNEL_BUCKETS."""
    hs, probs = make_inputs(shape)

    def one_pair() -> None:
        p, _, w = dispatcher.token_dispatch(hs, probs)
        dispatcher.token_combine((p * w.unsqueeze(-1)).to(shape.dtype))

    # External warmup so autotune / jit doesn't land inside the profiled window.
    for _ in range(num_warmup):
        one_pair()
    torch.cuda.synchronize()

    # The standard kineto trick: a "wait" tick before the "active" tick so the
    # first iteration's lazy autotune is excluded from the measurement.
    sched = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(2):
            for _ in range(num_active):
                one_pair()
            torch.cuda.synchronize()
            prof.step()

    return _aggregate_trace(prof, num_active=num_active, trace_path=trace_path)


# ---------------------------------------------------------------------------
# Per-backend orchestration
# ---------------------------------------------------------------------------


@dataclass
class Result:
    backend: str
    wall: Optional[WallClock] = None
    modules: Optional[ModuleBreakdown] = None
    err: Optional[str] = None


def run_backend(
    backend: str,
    *,
    group: dist.ProcessGroup,
    shape: Shape,
    args: argparse.Namespace,
) -> Result:
    def build():
        return build_dispatcher(
            backend,
            group,
            shape.num_experts,
            shape.num_topk,
            args.num_sms,
        )

    try:
        wall = bench_wallclock(
            build(),
            shape,
            num_warmup=args.num_warmup,
            num_tests=args.num_tests,
        )
        # Fresh dispatcher so the kineto trace isn't polluted by in-flight
        # async work the wall-clock pass left behind (MORI in particular).
        trace = None
        if args.kineto_trace_dir:
            trace = os.path.join(
                args.kineto_trace_dir,
                f"{args.model or 'custom'}_ep{group.size()}_{backend}" f"_rank{dist.get_rank()}.json",
            )
        modules = bench_kineto(
            build(),
            shape,
            num_warmup=max(2, args.num_warmup // 2),
            num_active=max(5, args.num_tests // 2),
            trace_path=trace,
        )
        return Result(backend=backend, wall=wall, modules=modules)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return Result(backend=backend, err=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Width of the kineto "dispatch" column so MORI's ``a + b = total`` form and
# the single-number form of other backends line up. Wide enough for
# ``XXXX.X + XXXX.X = XXXXX.X``.
_DISPATCH_COL_W = 26


def _fmt_dispatch_cell(backend: str, m: ModuleBreakdown) -> str:
    """MORI gets ``a + b = total``; everyone else gets just the total."""
    if backend == "MORI" and m.ep_dispatch_post > 0.0:
        cell = f"{m.ep_dispatch:6.1f} + {m.ep_dispatch_post:5.1f} = {m.dispatch_total:6.1f}"
    else:
        cell = f"{m.dispatch_total:6.1f}"
    return cell.rjust(_DISPATCH_COL_W)


def render_summary(results: List[Result], shape: Shape, world_size: int) -> str:
    lines: List[str] = []

    # --- block 1: wall-clock latency / bandwidth ---
    lines.append("=" * 110)
    lines.append(f"Wall-clock latency / bandwidth  (EP={world_size}, per rank)")
    lines.append("-" * 110)
    lines.append(
        f"  {'backend':<8}  {'dispatch (us)':>14}  {'BW (GB/s)':>10}  "
        f"{'combine (us)':>14}  {'BW (GB/s)':>10}  {'e2e (us)':>10}"
    )
    for r in results:
        if r.err:
            lines.append(f"  {r.backend:<8}  ERROR: {r.err[:80]}")
            continue
        w = r.wall
        lines.append(
            f"  {r.backend:<8}  {w.dispatch_us:>14.1f}  {w.dispatch_gbs:>10.1f}  "
            f"{w.combine_us:>14.1f}  {w.combine_gbs:>10.1f}  {w.e2e_us:>10.1f}"
        )

    # --- block 2: per-module kineto breakdown ---
    lines.append("")
    lines.append("=" * 110)
    lines.append(f"Per-module kernel time (kineto, per iter, us). " f"MORI dispatch = ep_dispatch + post")
    lines.append("-" * 110)
    lines.append(
        f"  {'backend':<8}  {'dispatch':>{_DISPATCH_COL_W}}  "
        f"{'permute':>8}  {'unpermute':>10}  {'combine':>8}  {'e2e':>8}"
    )
    for r in results:
        if r.err or r.modules is None:
            continue
        m = r.modules
        lines.append(
            f"  {r.backend:<8}  {_fmt_dispatch_cell(r.backend, m)}  "
            f"{m.permute:>8.1f}  {m.unpermute:>10.1f}  "
            f"{m.ep_combine:>8.1f}  {m.e2e_us:>8.1f}"
        )
    lines.append("=" * 110)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark DeepEPTokenDispatcher across EP backends with per-module "
            "kineto breakdown. Opt-in: requires --bench."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="Enable the benchmark pass. Without it the script prints help and exits.",
    )

    g = p.add_argument_group("model / shape")
    g.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_PRESETS),
        help="MoE preset (sets num_experts/num_topk/hidden).",
    )
    g.add_argument("--num-experts", type=int, default=None)
    g.add_argument("--num-topk", type=int, default=None)
    g.add_argument("--hidden", type=int, default=None)
    g.add_argument("--num-tokens", type=int, default=4096, help="Tokens per rank (default 4096).")

    g = p.add_argument_group("bench knobs")
    g.add_argument(
        "--backends",
        type=str,
        default="MORI,TURBO",
        help="Comma-separated subset of MORI/UCCL/TURBO/DEEP_EP/AUTOTUNE. "
        "AUTOTUNE runs with PRIMUS_TURBO_AUTO_TUNE=1 (no pin) and lets "
        "the runtime pick the fastest registered backend.",
    )
    g.add_argument("--num-warmup", type=int, default=5)
    g.add_argument("--num-tests", type=int, default=20)
    g.add_argument("--num-sms", type=int, default=32, help="num_use_cu / num_sms for the EP buffer.")
    g.add_argument(
        "--kineto-trace-dir",
        type=str,
        default=None,
        help="If set, dump per-(model,EP,backend,rank) chrome traces here.",
    )

    p.add_argument("--list-models", action="store_true", help="Print built-in model presets and exit.")
    return p


def _resolve_shape(args: argparse.Namespace) -> Shape:
    if args.model is not None:
        preset = MODEL_PRESETS[args.model]
        for k in ("num_experts", "num_topk", "hidden"):
            if getattr(args, k) is None:
                setattr(args, k, preset[k])
    missing = [k for k in ("num_experts", "num_topk", "hidden") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"need --model or all of {missing} (e.g. --model deepseek-v3)")
    return Shape(
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        num_experts=args.num_experts,
        num_topk=args.num_topk,
    )


def main() -> int:
    args = _build_parser().parse_args()

    if args.list_models:
        for name, cfg in MODEL_PRESETS.items():
            print(f"{name:<20} {cfg}")
        return 0

    if not args.bench:
        print(
            "Benchmark pass is opt-in. Pass --bench to enable it.\n"
            "Try: --list-models to see available model presets, or --help.",
            file=sys.stderr,
        )
        return 0

    shape = _resolve_shape(args)
    _, world_size, group = init_dist()
    backends = resolve_backends([b for b in args.backends.split(",") if b.strip()])

    _print0("=" * 78)
    _print0(f"DeepEPTokenDispatcher bench  (EP={world_size})")
    _print0(f"  model        = {args.model or '<custom>'}")
    _print0(
        f"  shape        = tokens={shape.num_tokens}/rank, hidden={shape.hidden},"
        f" experts={shape.num_experts}, topk={shape.num_topk}"
    )
    _print0(f"  bench iters  = {args.num_warmup} warmup + {args.num_tests} timed")
    _print0(f"  backends     = {backends}")
    _print0("=" * 78)

    results: List[Result] = []
    for backend in backends:
        dist.barrier(group=group)
        _print0(f"\n[{backend}] starting ...")
        t0 = time.perf_counter()
        r = run_backend(backend, group=group, shape=shape, args=args)
        dist.barrier(group=group)
        if r.err:
            _print0(f"[{backend}] ERROR after {time.perf_counter() - t0:.1f}s: {r.err}")
        else:
            _print0(
                f"[{backend}] done in {time.perf_counter() - t0:.1f}s  "
                f"(wall e2e={r.wall.e2e_us:.1f}us, kineto e2e={r.modules.e2e_us:.1f}us)"
            )
        results.append(r)
        clear_backend_instances()
        torch.cuda.synchronize()

    if dist.get_rank() == 0:
        print()
        print(render_summary(results, shape, world_size))

    dist.barrier(group=group)
    dist.destroy_process_group()
    return 0 if all(r.err is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
