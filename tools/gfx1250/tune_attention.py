###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measure one attention kernel config on gfx1250, correctness first, then time.

One process, one config, one JSON line on stdout. The tuning loop runs this once per
candidate rather than sweeping in-process, because a bad candidate on this part can take
the whole process down (see --health-check) and an in-process sweep would lose the
results that came before it.

Usage
-----
    # baseline: whatever the kernel ships with
    python3 tools/gfx1250/tune_attention.py --shape llama31-8b

    # one candidate
    python3 tools/gfx1250/tune_attention.py --shape llama31-8b \
        --tune "fwd:num_stages=2;bwd:num_warps=2"

    # cheaper proxy shape for a sweep round
    python3 tools/gfx1250/tune_attention.py --shape llama31-8b-s4096 --tune num_warps=2

Why it is built the way it is
-----------------------------
Four things below are defences against failure modes that were actually observed on this
hardware and in aiter's Triton MHA, not hypothetical hardening:

1. SQNR is computed separately for out, dq, dk AND dv against an fp32 reference. An
   output-only or dk/dv-only check is not a weaker version of this -- it is a check that
   actively rewards two known-wrong configs. In aiter's one-kernel backward, raising
   BLOCK_N1 alone halves the launch grid that the dq half also rides on, so dq covers half
   the query rows: that config measured 1.31x faster with dq at 9.59 dB while dk and dv
   stayed perfect. The threshold is not what catches it; the coverage is.
2. The config that was asked for is asserted to be the config that ran, before any timing.
   Triton's autotune list is built once at import from the environment, so a spec that
   arrives late, or is misspelled, silently measures the default -- which looks like a flat
   sweep where every candidate returns the same time.
3. dmesg is checked for GPU faults before and after. Deliberately not rocm-smi: when this
   card wedges, one of the stuck tasks is inside amdgpu_info_ioctl, so a health check that
   shells out to rocm-smi HANGS instead of reporting.
4. Timing is CUDA-event based over a fixed iteration count with the cache cleared between
   reps, and fwd and bwd are timed separately because the backward is 73-76% of the time
   and a combined number hides which half moved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# The tuning spec is read by attention_kernel at import, when @triton.autotune builds its
# config list. Set it before anything imports primus_turbo.
_TUNE_ENV = "PRIMUS_TURBO_ATTN_TRITON_TUNE"
if "--tune" in sys.argv:
    os.environ[_TUNE_ENV] = sys.argv[sys.argv.index("--tune") + 1]

# Route torch's own matmuls to rocBLAS rather than hipBLASLt.
#
# CORRECTED 2026-09-14. This comment previously said the gfx1250 Tensile library "is simply
# absent" from amdprimus:gfx1250-20260910 and that an fp32 matmul therefore raises
# HIPBLAS_STATUS_INVALID_VALUE. That is wrong on both counts. The image ships 46 gfx1250
# bf16 Tensile solutions under _rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250/
# (the build asserts >= 46 and fails otherwise), torch reports the backend as Cublaslt, and
# nothing raises.
#
# The reason to keep this setting is the one that survived measurement: on this part
# hipBLASLt is simply SLOW. Same tensors, same process, 8192^3 bf16 -- hipBLASLt 113.0
# TFLOP/s against a naive Triton GEMM at 1190.1 with max_abs_err 0.0, a 10.5x gap. Putting
# the tuned library first on LD_LIBRARY_PATH moves it by under 1%, and so does flipping
# TORCH_BLAS_PREFER_HIPBLASLT between 0 and 1 -- so it is neither a packaging bug nor a
# silent fallback to rocBLAS. Either way it is not something the fp32 reference should be
# built on. Set before torch is imported; torch reads it at backend-selection time.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

# Pin this process to one GPU, hard, before torch initialises.
#
# On a multi-GPU box the isolation has to be at the driver level, not a torch.cuda
# device index: op-evolve has no GPU lock, and a candidate that lands on a card another
# stream is using does not raise -- it records a low number, and that number becomes the
# champion the next round has to beat. HIP_VISIBLE_DEVICES makes the other cards
# invisible, so a stray allocation cannot reach them.
#
# Set GPU=<n> (or pass --gpu) and every stream is fenced to its own card. Torch then sees
# exactly one device and every "cuda" below means that one.
if "--gpu" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
elif os.environ.get("GPU"):
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["GPU"]

# Same reason as _TUNE_ENV above: the ASM forward gate reads its off switch at import.
#
# This exists because there is otherwise no way to measure the Triton forward. Since the
# dispatcher started choosing the ASM forward on its own, --impl turbo, fused and asm all
# reach it: measured on c07-1 they are 19.222 / 19.298 / 19.285 ms, a 0.4% spread that is
# the noise floor rather than an A/B, with a forward of ~1.55 ms where the Triton forward
# is ~5.2 ms at this clock. Any claim about what the ASM forward is worth needs the off run.
if "--asm-fwd" in sys.argv and sys.argv[sys.argv.index("--asm-fwd") + 1] == "off":
    os.environ["PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD"] = "1"

# Same reason as _TUNE_ENV above: the vendored fused backward reads its config at import.
if "--fused-tune" in sys.argv:
    os.environ["PRIMUS_TURBO_FUSED_MHA_BWD_TUNE"] = sys.argv[sys.argv.index("--fused-tune") + 1]

# Make the checkout this script lives in the primus_turbo that gets imported.
# Running a script puts the SCRIPT's directory on sys.path, not the repo root, so without
# this an editable install elsewhere in the image wins and the harness silently measures a
# different checkout than the one being edited -- which is exactly the kind of "the config
# did not apply" failure assert_config_applied exists to catch, one level further out.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
# Hq, Hkv, D, causal and dtype define which kernel you are tuning -- they set the tiling,
# the GQA group reduction, the mask logic and the numerics. Only batch and sequence move.
SHAPES = {
    # P0: the production shape. Llama-3.1-8B, MBS=4, seq 8192. One training step is
    # exactly 32 forward + 32 backward calls of this and nothing else (GBS=MBS so there
    # is one micro-batch, and activation checkpointing is off).
    "llama31-8b": dict(batch=4, seqlen=8192, hq=32, hkv=8, d=128),
    # P1: tuning proxy, exactly 1/4 the FLOPs. Preserves the causal block structure, the
    # GQA 4:1 group reduction and the occupancy regime; does not exercise the long K loop.
    # A config that wins here must be confirmed on llama31-8b before it becomes champion.
    "llama31-8b-s4096": dict(batch=4, seqlen=4096, hq=32, hkv=8, d=128),
    # P2: half cost at full sequence length. Also the shape commit c1325c7e reported
    # 220.6 TFLOP/s on, which the in-tree bench measured at 131.7 for the same kernel.
    "llama31-8b-b2": dict(batch=2, seqlen=8192, hq=32, hkv=8, d=128),
    # Smoke: seconds, for wiring changes.
    "smoke": dict(batch=1, seqlen=1024, hq=8, hkv=2, d=128),
    # Extra points for the shape-gate table. The tile that wins at s=8192 loses at s=1024,
    # so the dispatcher needs the crossover, not just the endpoints. Same heads/D/dtype
    # throughout -- only batch and sequence move, or it is a different kernel.
    "gate-s1024": dict(batch=4, seqlen=1024, hq=32, hkv=8, d=128),
    "gate-s2048": dict(batch=4, seqlen=2048, hq=32, hkv=8, d=128),
    "gate-s16384": dict(batch=1, seqlen=16384, hq=32, hkv=8, d=128),
    "gate-b1": dict(batch=1, seqlen=8192, hq=32, hkv=8, d=128),
    "gate-b8": dict(batch=8, seqlen=4096, hq=32, hkv=8, d=128),
    # Shapes that straddle fused_backward_eligible's _MIN_PARALLEL_WORK = 32 on
    # batch * num_q_heads. The threshold ships and has never been measured on this host;
    # these four bracket it so the dispatch rule can be checked rather than assumed.
    #   b1h32 = 32 (at the threshold, eligible)   b4h8  = 32 (at the threshold, eligible)
    #   b2h8  = 16 (below, declines)              b1h8  =  8 (well below, declines)
    "par-b1h32": dict(batch=1, seqlen=4096, hq=32, hkv=8, d=128),
    "par-b4h8": dict(batch=4, seqlen=4096, hq=8, hkv=2, d=128),
    "par-b2h8": dict(batch=2, seqlen=4096, hq=8, hkv=2, d=128),
    "par-b1h8": dict(batch=1, seqlen=4096, hq=8, hkv=2, d=128),
}

_FAULT_PATTERNS = (
    # NOT a bare "MES(": "MES(0, 0) ring buffer is full" is routine backpressure and
    # matching it reports a healthy card as faulted. The wedge signature is MES
    # failing to RESPOND to a message, then "wait for reset ack".
    "failed to respond to msg",
    "GPU Hang",
    "wait for reset ack",
    "Memory access fault",
    "ring gfx timeout",
    "GPU reset begin",
)


def dmesg_faults() -> list[str]:
    """Recent GPU fault lines, or [] if dmesg is unreadable.

    Not rocm-smi: a wedged card leaves tasks stuck in amdgpu_info_ioctl, which is what
    rocm-smi calls, so it hangs rather than reporting.
    """
    try:
        out = subprocess.run(
            ["dmesg", "--time-format", "iso"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    return [ln for ln in out.splitlines()[-400:] if any(p in ln for p in _FAULT_PATTERNS)]


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Signal-to-quantization-noise in dB, computed in fp64 so the metric is not itself
    the thing being measured."""
    ref64, got64 = ref.detach().double(), got.detach().double()
    signal = ref64.norm().pow(2)
    noise = (ref64 - got64).norm().pow(2)
    return float(10 * torch.log10(signal / (noise + 1e-30)))


def reference_fwd_bwd(q, k, v, do, causal: bool):
    """Exact fp32 attention, forward and analytic backward, one (batch, q-head) at a time.

    Chunked, and analytic rather than autograd, for two reasons that both bite at the
    production shape. The full score tensor [B, Hq, S, S] in fp32 is 34 GB at b=4 s=8192
    before any intermediate, and an autograd graph over it holds several of those at once.
    Per (batch, q-head) the same tensor is 268 MB.

    dk and dv are accumulated across all G = Hq/Hkv query heads that share a kv head. A
    reference that forgets that sum is wrong by a factor of G -- smoothly, so it passes a
    loose gate -- which is also why the SQNR check may not sample query heads.

    The math is the standard FA-2 backward:
        p[i, j]  = exp(s[i, j] - lse[i])            for j <= i, else 0
        delta[i] = sum_d do[i, d] * o[i, d]
        dp[i, j] = dot(do[i, :], v[j, :])
        ds[i, j] = p[i, j] * (dp[i, j] - delta[i])
        dv[j, :] = sum_{i >= j} p[i, j] * do[i, :]
        dk[j, :] = scale * sum_{i >= j} ds[i, j] * q[i, :]
        dq[i, :] = scale * sum_{j <= i} ds[i, j] * k[j, :]
    """
    b, s, hq, d = q.shape
    hkv = k.shape[2]
    g = hq // hkv
    scale = d**-0.5

    out = torch.empty(b, s, hq, d, device=q.device, dtype=torch.float32)
    dq = torch.empty(b, s, hq, d, device=q.device, dtype=torch.float32)
    dk = torch.zeros(b, s, hkv, d, device=q.device, dtype=torch.float32)
    dv = torch.zeros(b, s, hkv, d, device=q.device, dtype=torch.float32)

    if causal:
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()

    for bi in range(b):
        for h in range(hq):
            hk = h // g
            qi = q[bi, :, h, :].detach().float()        # [s, d]
            ki = k[bi, :, hk, :].detach().float()
            vi = v[bi, :, hk, :].detach().float()
            doi = do[bi, :, h, :].detach().float()

            scores = (qi @ ki.transpose(0, 1)) * scale  # [s, s]
            if causal:
                scores.masked_fill_(~mask, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            del scores
            oi = p @ vi
            out[bi, :, h, :] = oi

            delta = (doi * oi).sum(-1, keepdim=True)    # [s, 1]
            dp = doi @ vi.transpose(0, 1)               # [s, s]
            ds = p * (dp - delta)
            del dp

            dv[bi, :, hk, :] += p.transpose(0, 1) @ doi
            del p
            dk[bi, :, hk, :] += scale * (ds.transpose(0, 1) @ qi)
            dq[bi, :, h, :] = scale * (ds @ ki)
            del ds

    return out, dq, dk, dv


def timed_ms(fn, iters: int, warmup: int) -> float:
    """Mean ms over `iters`, CUDA-event timed, cache cleared between reps."""
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()  # evict L2 so every rep sees a cold cache
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    # Median rather than mean: a single descheduled rep should not move the number.
    return times[len(times) // 2]


def assert_config_applied(spec: str) -> dict:
    """Prove the spec reached the kernel before anything is timed.

    Triton builds its autotune list at import. A spec that arrived after the import, or a
    key that did not parse, leaves the shipped default in place and every candidate in the
    sweep then returns the same time -- a flat result that looks like "this knob does
    nothing" rather than like a broken harness.
    """
    from primus_turbo.triton.attention import attention_kernel as ak

    got = {
        "fwd": [
            dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
            for c in ak.get_autotune_fwd_configs()[0]
        ],
        "bwd": [
            dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
            for c in ak.get_autotune_bwd_configs()[0]
        ],
    }
    if not spec:
        return got

    for half in ("fwd", "bwd"):
        want = ak._parse_tune_spec(spec, half)
        if want in (None, "sweep"):
            continue
        for key, value in want[0].items():
            actual = [c.get(key) for c in got[half]]
            if actual != [value] * len(actual):
                raise SystemExit(
                    f"config did not apply: {half} {key} is {actual}, asked for {value}. "
                    f"{_TUNE_ENV}={os.environ.get(_TUNE_ENV)!r}"
                )
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="llama31-8b", choices=sorted(SHAPES))
    ap.add_argument("--tune", default="", help=f"value for {_TUNE_ENV}")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sqnr-min", type=float, default=50.0)
    ap.add_argument("--asm-fwd", default="auto", choices=["auto", "off"],
                    help="'off' forces the Triton forward by disabling the ASM gate. Read at "
                         "import (see the top of this file), so it must be a process-level "
                         "flag rather than a runtime one. Off-only: eligibility is a "
                         "capability question and forcing 'on' would only move the failure.")
    ap.add_argument("--bwd-path", default="auto", choices=["auto", "twokernel"],
                    help="'twokernel' forces the IN-TREE two-kernel Triton backward "
                         "(_bwd_kernel_dkdv + _bwd_kernel_dq). It is needed because the "
                         "shipped dispatcher never reaches that path at a training shape: "
                         "flash_attn_interface.py:387 prefers dense_fused_backward whenever "
                         "fused_backward_eligible() says yes, which at b*Hq >= 32 and "
                         "seqlen_k >= 512 is always. The fused backward does not read "
                         "PRIMUS_TURBO_ATTN_TRITON_TUNE, so a bwd: sweep left on 'auto' "
                         "returns the SAME time for every candidate -- the flat result that "
                         "reads as 'this knob does nothing'. Forcing it also forces the "
                         "Triton forward, because use_asm_fwd is gated on the same predicate.")
    ap.add_argument("--impl", default="turbo",
                    choices=["turbo", "aiter", "fused", "asm", "asmbwd", "flydsl"],
                    help="turbo = Primus-Turbo's in-tree Triton backend (the PR target). "
                         "aiter = AITER's Triton MHA, the alternative seed the plan named. "
                         "flydsl = AITER's FlyDSL gfx1250 forward (its default path on this "
                         "arch) paired with the vendored fused backward, so the number that "
                         "moves is the forward. There is no FlyDSL backward on gfx1250. "
                         "Same shape, same fp32 reference, same SQNR gate, same timer -- the "
                         "only way the two numbers are comparable is if everything but the "
                         "kernel is identical code.")
    ap.add_argument("--aiter-fwd-stages", type=int, default=0,
                    help="override aiter's forward num_stages (0 = shipped config)")
    ap.add_argument("--aiter-bwd-warps", type=int, default=0,
                    help="override aiter's backward num_warps (0 = shipped config)")
    ap.add_argument("--fused-tune", default="",
                    help="override the VENDORED fused backward config (sets "
                         "PRIMUS_TURBO_FUSED_MHA_BWD_TUNE before import)")
    ap.add_argument("--aiter-bwd-cfg", default="",
                    help="override keys on aiter's fused backward config, e.g. "
                         "'BLOCK_M1=64,BLOCK_N2=64'. NOTE the pairing constraint: the launch "
                         "grid is sized by BLOCK_N1 and the same grid serves the dq half, "
                         "which is tiled by BLOCK_M2, so N1 must equal M2 (and symmetrically "
                         "M1 == N2). Breaking it yields a config that measures FASTER with "
                         "dq silently covering half the query rows -- the four-tensor SQNR "
                         "gate is what catches it.")
    ap.add_argument("--determinism-reps", type=int, default=0,
                    help="run fwd+bwd N times in-process and compare every output BITWISE "
                         "against rep 0. Determinism needs no fp32 reference -- dropping it "
                         "makes a rep ~100x cheaper, which is what makes a >=500-rep study "
                         "practical. Catches both a silent corruption (a rep that differs) "
                         "and a non-deterministic accumulation (a tensor that never settles).")
    ap.add_argument("--correctness-only", action="store_true",
                    help="run the SQNR check and skip timing. For race characterisation: "
                         "repeat this many times and count failures. An intermittent wrong "
                         "answer is invisible to a single-shot check.")
    ap.add_argument("--skip-correctness", action="store_true",
                    help="time only. The loop must NOT use this: an unchecked candidate "
                         "can be fast because it computes less.")
    ap.add_argument("--gpu", default="",
                    help="pin to this GPU via HIP_VISIBLE_DEVICES, read before torch imports. "
                         "Driver-level rather than a torch device index, so a stray "
                         "allocation cannot reach another stream's card -- there is no GPU "
                         "lock, and a contended measurement does not raise, it just records "
                         "a low number that becomes the next champion to beat.")
    ap.add_argument("--shapes", default="",
                    help="comma-separated shapes to measure IN ONE PROCESS, e.g. "
                         "'llama31-8b,llama31-8b-b2'. Amortises the ~10-15 s of import and "
                         "Triton compile that otherwise dominates a sweep -- at one process "
                         "per candidate the GPU sits at ~12%% utilisation because most of "
                         "the wall clock is startup, not measurement. Only safe for things "
                         "that do NOT change import-time state: shape and impl are fine, "
                         "--tune and --fused-tune are read at import and still need their "
                         "own process.")
    ap.add_argument("--json", default="", help="also write the result object here")
    args = ap.parse_args()

    if args.shapes:
        # Re-enter main() once per shape in this process. Everything import-time (the tune
        # spec, the vendored config, the module graph) is already resolved and shared; only
        # the tensors and the measurement differ.
        rc = 0
        for _shape in [x.strip() for x in args.shapes.split(",") if x.strip()]:
            args.shape = _shape
            args.shapes = ""
            rc |= _measure(args)
        return rc
    return _measure(args)


def _measure(args) -> int:
    result: dict = {
        "shape": args.shape,
        "tune": args.tune,
        "dtype": args.dtype,
        "causal": not args.no_causal,
        "ok": False,
        "wall_start": time.time(),
    }

    pre = dmesg_faults()
    result["dmesg_faults_before"] = len(pre)

    from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
    from primus_turbo.pytorch.core.utils import is_gfx1250
    from primus_turbo.pytorch.ops import flash_attn_func

    import primus_turbo

    result["turbo_path"] = primus_turbo.__file__
    # Whether the ASM forward gate was disabled for this row. impl_note is a hardcoded string
    # assigned from --impl before anything runs, so it labels intent, not what executed; it
    # still says "turbo forward" on rows that took the ASM one. This field is read from the
    # env the gate itself reads, so an A/B can be audited from the ledger alone.
    result["asm_fwd"] = "off" if os.environ.get("PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD", "") not in ("", "0") else "auto"
    # Which physical card this row was measured on. Without it a 4-stream ledger cannot be
    # audited after the fact, and per-GPU clock differences get attributed to the config.
    result["gpu"] = os.environ.get("HIP_VISIBLE_DEVICES", "all")
    result["arch"] = torch.cuda.get_device_properties(0).gcnArchName
    result["is_gfx1250"] = bool(is_gfx1250())
    result["configs"] = assert_config_applied(args.tune)

    s = SHAPES[args.shape]
    b, sq, hq, hkv, d = s["batch"], s["seqlen"], s["hq"], s["hkv"], s["d"]
    causal = not args.no_causal
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    result.update(batch=b, seqlen=sq, hq=hq, hkv=hkv, head_dim=d)

    torch.manual_seed(0)
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(b, sq, hkv, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(b, sq, hkv, d, device="cuda", dtype=dtype, requires_grad=True)
    do = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype)

    # These four impls need attention backends that live on dev/lhz/attn (the aiter ASM
    # forward, the ASM backward launcher, and the vendored fused backward) and are NOT on
    # this branch, which was cut from main to keep the FlyDSL line separate. Say so, rather
    # than letting a bare ModuleNotFoundError surface a hundred lines into a measurement.
    if args.impl in ("fused", "asm", "asmbwd", "flydsl"):
        import importlib.util as _ilu

        _needed = {
            "fused": ["attention_fused_bwd_impl"],
            "asm": ["attention_asm_fwd_impl", "attention_fused_bwd_impl"],
            "asmbwd": ["attention_asm_fwd_impl", "attention_asm_bwd_impl"],
            "flydsl": ["attention_fused_bwd_impl"],
        }[args.impl]
        _missing = [
            m
            for m in _needed
            if _ilu.find_spec(f"primus_turbo.pytorch.kernels.attention.{m}") is None
        ]
        if _missing:
            raise SystemExit(
                f"--impl {args.impl} needs {', '.join(_missing)}, which this branch does not "
                "carry (they are part of the ASM/fused line on dev/lhz/attn). For a "
                "FlyDSL-vs-ASM forward comparison that needs neither, use "
                "output/0917__flydsl/bin/stage1_fwd_ab.py -- both of its arms come from "
                "aiter, which also sidesteps the flydsl 0.2.4/0.3.2 conflict."
            )

    if args.impl == "fused":
        # What would actually ship: Primus-Turbo's own forward, with the vendored fused
        # backward swapped in for the in-tree two-kernel one. Patched at the op layer rather
        # than wired into the dispatcher, so this measures the kernel without pre-committing
        # the dispatch change.
        from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
            dense_fused_backward,
        )
        from primus_turbo.pytorch.ops.attention import flash_attn_interface as _fai

        def _fused_bwd(do_, q_, k_, v_, o_, lse_, softmax_scale=None, causal=True,
                       sink=None, window_size=(-1, -1)):
            dq_, dk_, dv_ = dense_fused_backward(
                do_, q_, k_, v_, o_, lse_, softmax_scale, causal, window_size
            )
            return dq_, dk_, dv_, None

        _fai.triton_dense_backward = _fused_bwd
        result["impl_note"] = "turbo forward + vendored fused backward"
        _impl_fwd = None
    elif args.impl == "asm":
        # aiter's PREBUILT gfx1250 ASM forward, paired with the vendored fused backward.
        #
        # These two compose without an adapter, which is the whole reason this pairing is
        # worth measuring: the ASM forward returns LSE as plain [B, Hq, Sq] fp32 in natural
        # log, and dense_fused_backward documents exactly that as its accepted form. The
        # in-tree two-kernel backward would NOT compose -- it consumes turbo's packed
        # [B, Hq, 2*Sq] lse/delta scratch, where LSE and delta interleave every
        # FIXED_BLOCK_M rows, so substituting there needs a scatter into the LSE half.
        #
        # Measured as an autograd Function rather than two separate timings, so the total
        # is a measurement and not the sum of two halves taken in different sessions.
        from aiter.ops.mha import fmha_fwd_with_sink_asm
        from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
            dense_fused_backward,
        )

        _asm_scale = (q.shape[-1]) ** -0.5

        class _AsmFwdFusedBwd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q_, k_, v_):
                o_, lse_ = fmha_fwd_with_sink_asm(q_, k_, v_, _asm_scale, causal, True)
                ctx.save_for_backward(q_, k_, v_, o_, lse_)
                return o_

            @staticmethod
            def backward(ctx, do_):
                q_, k_, v_, o_, lse_ = ctx.saved_tensors
                dq_, dk_, dv_ = dense_fused_backward(
                    do_.contiguous(), q_, k_, v_, o_, lse_, _asm_scale, causal, (-1, -1)
                )
                return dq_, dk_, dv_

        result["impl_note"] = "aiter prebuilt gfx1250 ASM forward + vendored fused backward"
        _impl_fwd = lambda: _AsmFwdFusedBwd.apply(q, k, v)  # noqa: E731
    elif args.impl == "asmbwd":
        # Both halves from aiter's prebuilt gfx1250 ASM: the forward, and the three-kernel
        # backward brought up on 0915 (odo -> dqdkdv -> dq_convert), launched by hand
        # because no Python arch gate in aiter reaches the backward kernels --
        # can_impl_fmha_v3_bwd starts from get_gfx() == "gfx942" and the only widening is
        # for gfx950, so gfx1250 can never select them.
        #
        # dk/dv are allocated per q head and reduced here. The main kernel's grid is
        # (kv_tiles, nhead_q, batch), so under GQA `ratio` workgroups own the same dk/dv
        # tile and race; measured at ratio=4 that costs dk/dv about -0.3 dB while dq stays
        # correct. The reduction is inside the timed region because it is part of the cost.
        import sys as _sys
        _sys.path.insert(0, os.path.join(_REPO_ROOT, "tools", "gfx1250"))
        import asm_bwd_launcher as _abl
        from aiter.ops.mha import fmha_fwd_with_sink_asm

        _asm_scale = (q.shape[-1]) ** -0.5
        _rep = q.shape[2] // k.shape[2]

        class _AsmFwdAsmBwd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q_, k_, v_):
                o_, lse_ = fmha_fwd_with_sink_asm(q_, k_, v_, _asm_scale, causal, True)
                ctx.save_for_backward(q_, k_, v_, o_, lse_)
                return o_

            @staticmethod
            def backward(ctx, do_):
                q_, k_, v_, o_, lse_ = ctx.saved_tensors
                dq_, dk_, dv_ = _abl.asm_backward(
                    q_, k_, v_, o_, do_.contiguous(), lse_, _asm_scale, dkdv_heads="q"
                )
                if _rep > 1:
                    b_, s_, _, d_ = dk_.shape
                    hk_ = k_.shape[2]
                    dk_ = dk_.view(b_, s_, hk_, _rep, d_).float().sum(3).to(k_.dtype)
                    dv_ = dv_.view(b_, s_, hk_, _rep, d_).float().sum(3).to(v_.dtype)
                return dq_, dk_, dv_

        result["impl_note"] = "aiter prebuilt gfx1250 ASM forward + aiter prebuilt ASM backward"
        _impl_fwd = lambda: _AsmFwdAsmBwd.apply(q, k, v)  # noqa: E731
    elif args.impl == "flydsl":
        # AITER's FlyDSL gfx1250 forward -- the DEFAULT path on this arch -- paired with the
        # vendored fused backward, the same backward `--impl asm` uses. Pairing it that way
        # is the point: `asm` and `flydsl` then differ in the forward and nothing else, so
        # the delta between them is attributable.
        #
        # Entry point is flydsl_flash_attn_batch_func, NOT flydsl_flash_attn_func. The latter
        # is the gfx1201 RDNA4 kernel and returns no LSE, so it cannot be paired with any
        # backward. The batch entry is BSHD [B,S,H,D] -- the layout this harness already
        # holds -- gates on get_gfx() == "gfx1250", and returns LSE as [B, nheads_q, S_q]
        # fp32, which is the form dense_fused_backward documents as accepted.
        #
        # Its GQA condition is `nheads_q % nheads_kv == 0`, so Llama-3.1-8B's G=4 is in
        # scope here. Turbo's own _gqa_group_ok requires a power of two in [8, 256] and is
        # what keeps this shape off turbo's FlyDSL path -- a different gate, not this one.
        #
        # IT RETURNS None WHEN IT CANNOT SERVE THE CONFIGURATION rather than raising. Left
        # unchecked, an unsupported shape would fall through to whatever the caller does
        # next and the run would report a number for a kernel that never executed. Raise.
        from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func
        from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
            dense_fused_backward,
        )

        _fdsl_scale = (q.shape[-1]) ** -0.5

        _probe = flydsl_flash_attn_batch_func(
            q.detach(), k.detach(), v.detach(),
            softmax_scale=_fdsl_scale, causal=causal, return_lse=True,
        )
        if _probe is None:
            raise SystemExit(
                "flydsl_flash_attn_batch_func returned None for this configuration: "
                f"b={b} s={sq} hq={hq} hkv={hkv} d={d} dtype={dtype} causal={causal}, "
                f"arch={result.get('arch')}. It declines rather than raising, so this is a "
                "refusal to serve the shape, not a failure. Do not fall back silently."
            )
        _probe_out, _probe_lse = _probe
        result["flydsl_lse_shape"] = list(_probe_lse.shape)
        result["flydsl_lse_dtype"] = str(_probe_lse.dtype)
        del _probe, _probe_out, _probe_lse

        class _FlydslFwdFusedBwd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q_, k_, v_):
                o_, lse_ = flydsl_flash_attn_batch_func(
                    q_, k_, v_, softmax_scale=_fdsl_scale, causal=causal, return_lse=True,
                )
                ctx.save_for_backward(q_, k_, v_, o_, lse_)
                return o_

            @staticmethod
            def backward(ctx, do_):
                q_, k_, v_, o_, lse_ = ctx.saved_tensors
                dq_, dk_, dv_ = dense_fused_backward(
                    do_.contiguous(), q_, k_, v_, o_, lse_, _fdsl_scale, causal, (-1, -1)
                )
                return dq_, dk_, dv_

        # NOT YET VERIFIED: whether this kernel's LSE is natural log (what
        # dense_fused_backward expects, and what the ASM forward emits) or log2 -- the
        # kernel carries a LOG2E constant. A mismatch does not raise; it produces smoothly
        # wrong gradients. The four-tensor SQNR gate is what catches it, which is the reason
        # dq/dk/dv are gated here and not just `out`.
        result["impl_note"] = "aiter FlyDSL gfx1250 forward + vendored fused backward"
        _impl_fwd = lambda: _FlydslFwdFusedBwd.apply(q, k, v)  # noqa: E731
    elif args.impl == "aiter":
        from aiter.ops.triton._triton_kernels.attention import mha as _amha
        from aiter.ops.triton.attention.mha import flash_attn_func as _aiter_fa

        base = dict(_amha._get_config(False, dtype))
        fwd_cfg = dict(base)
        if args.aiter_fwd_stages:
            fwd_cfg["num_stages"] = args.aiter_fwd_stages

        # The backward has no config argument -- it calls a zero-arg _get_config that is
        # @functools.lru_cache'd. An override that does not clear the cache is SILENTLY
        # IGNORED, and the signature of that failure is a sweep where every candidate
        # returns the same time. Patch, clear, then assert what comes back.
        # TWO modules matter here. The config FUNCTION lives in _triton_kernels..., but the
        # backward WRAPPER did `from ... import _get_config` at import time, so it holds its
        # own binding. Patching only the source module leaves the wrapper calling the
        # original -- the override is accepted, `_get_config()` returns the new value, and
        # the launch still uses the old one. That produced a six-config sweep with a 0.6%
        # spread before it was caught. Patch the WRAPPER's binding.
        from aiter.ops.triton._triton_kernels.attention import mha_onekernel_bwd as _abwd_src
        from aiter.ops.triton.attention import mha_onekernel_bwd as _abwd_wrap

        _abwd = _abwd_src

        result["aiter_bwd_config_shipped"] = dict(_abwd_src._get_config())
        if args.aiter_bwd_cfg:
            base_bwd = _abwd_src._get_config()
            patched = {kk: dict(vv) for kk, vv in base_bwd.items()}
            for item in args.aiter_bwd_cfg.split(","):
                key, value = (x.strip() for x in item.split("=", 1))
                patched["onekernel"][key] = int(value)
            ok = patched["onekernel"]
            if ok.get("BLOCK_N1") != ok.get("BLOCK_M2") or ok.get("BLOCK_M1") != ok.get("BLOCK_N2"):
                raise SystemExit(
                    f"aiter bwd pairing violated (need N1==M2 and M1==N2): {ok}. "
                    "That config measures fast because dq covers only part of the query axis."
                )
            if args.aiter_bwd_warps:
                ok["num_warps"] = args.aiter_bwd_warps
            _abwd_src._get_config.cache_clear()
            _abwd_src._get_config = lambda: patched
            _abwd_wrap._get_config = lambda: patched   # the binding the launch actually reads
            got = _abwd_wrap._get_config()["onekernel"]
            for item in args.aiter_bwd_cfg.split(","):
                key, value = (x.strip() for x in item.split("=", 1))
                if got.get(key) != int(value):
                    raise SystemExit(f"aiter bwd override did not apply: {key}={got.get(key)}")
            result["aiter_bwd_config"] = got
        elif args.aiter_bwd_warps:
            # The backward config is nested: {"preprocess_kernel": {...}, "onekernel": {...}}.
            # num_warps lives on "onekernel"; setting it at the top level would be accepted
            # silently and change nothing.
            base_bwd = _abwd_src._get_config()
            patched = {k: dict(v) for k, v in base_bwd.items()}
            patched["onekernel"]["num_warps"] = args.aiter_bwd_warps
            _abwd_src._get_config.cache_clear()
            _abwd_src._get_config = lambda: patched
            _abwd_wrap._get_config = lambda: patched
            got = _abwd_wrap._get_config()
            if got["onekernel"].get("num_warps") != args.aiter_bwd_warps:
                raise SystemExit(f"aiter bwd override did not apply: {got}")
            result["aiter_bwd_config"] = got

        result["aiter_fwd_config"] = fwd_cfg
        _impl_fwd = lambda: _aiter_fa(q, k, v, causal=causal, config=fwd_cfg)  # noqa: E731
    else:
        _impl_fwd = None

    if args.bwd_path == "twokernel":
        if args.impl != "turbo":
            raise SystemExit("--bwd-path twokernel only applies to --impl turbo")
        from primus_turbo.pytorch.ops.attention import flash_attn_interface as _fai

        # Patch the NAME the dispatcher reads (it was imported into this module at import
        # time), not the definition in attention_fused_bwd_impl -- patching the source
        # module leaves the binding at flash_attn_interface.py:387 pointing at the original
        # and the override is silently ignored. Same failure shape as aiter's _get_config.
        _fai.fused_backward_eligible = lambda *a, **kw: False
        result["bwd_path"] = "twokernel(forced)"
    else:
        result["bwd_path"] = "auto"

    # Pin TRITON. Without this the dispatcher picks, and a round would not know which
    # kernel it just measured.
    GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)
    try:
        if not args.skip_correctness:
            out = (_impl_fwd() if _impl_fwd else flash_attn_func(q, k, v, causal=causal))
            out.backward(do)
            dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
            q.grad = k.grad = v.grad = None

            ref_out, ref_dq, ref_dk, ref_dv = reference_fwd_bwd(q, k, v, do, causal)
            sq_db = {
                "out": sqnr_db(ref_out, out.float()),
                "dq": sqnr_db(ref_dq, dq.float()),
                "dk": sqnr_db(ref_dk, dk.float()),
                "dv": sqnr_db(ref_dv, dv.float()),
            }
            result["sqnr_db"] = sq_db
            # All four, independently. See the module docstring for why dk/dv can be
            # perfect while dq is noise.
            result["correct"] = all(x >= args.sqnr_min for x in sq_db.values())
            if not result["correct"]:
                result["failed_tensors"] = [t for t, x in sq_db.items() if x < args.sqnr_min]
                # ok=True: the measurement RAN and produced a verdict. Only a process that
                # died leaves ok False. The distinction matters downstream -- a wrong answer
                # belongs in the "fast but incorrect" bucket, where it is visible and can
                # never win, not lumped in with compile errors as "failed to run".
                result["ok"] = True
                result["wall_s"] = time.time() - result["wall_start"]
                print(json.dumps(result))
                return 2

        if args.determinism_reps:
            # Bitwise, not SQNR: a tensor that is merely "close" run to run is already
            # non-deterministic, and SQNR against a reference cannot distinguish "this
            # kernel wobbles" from "this kernel is slightly inaccurate".
            ref = None
            mismatches = {"out": 0, "dq": 0, "dk": 0, "dv": 0}
            nan_reps = 0
            for rep in range(args.determinism_reps):
                o = flash_attn_func(q, k, v, causal=causal)
                o.backward(do)
                got = {"out": o.detach().clone(), "dq": q.grad.clone(),
                       "dk": k.grad.clone(), "dv": v.grad.clone()}
                q.grad = k.grad = v.grad = None
                if any(not torch.isfinite(t).all() for t in got.values()):
                    nan_reps += 1
                if ref is None:
                    ref = got
                    continue
                for name, t in got.items():
                    if not torch.equal(t, ref[name]):
                        mismatches[name] += 1
                del got
            result["determinism"] = {
                "reps": args.determinism_reps,
                "bitwise_mismatch_reps": mismatches,
                "nonfinite_reps": nan_reps,
                "deterministic": sum(mismatches.values()) == 0 and nan_reps == 0,
            }
            result["ok"] = True
            result["wall_s"] = time.time() - result["wall_start"]
            print(json.dumps(result))
            return 0 if result["determinism"]["deterministic"] else 3

        if args.correctness_only:
            result["ok"] = True
            result["wall_s"] = time.time() - result["wall_start"]
            print(json.dumps(result))
            return 0

        fwd = _impl_fwd or (lambda: flash_attn_func(q, k, v, causal=causal))  # noqa: E731
        out = fwd()
        bwd = lambda: out.backward(do, retain_graph=True)  # noqa: E731

        result["fwd_ms"] = timed_ms(fwd, args.iters, args.warmup)
        result["bwd_ms"] = timed_ms(bwd, args.iters, args.warmup)
    finally:
        GlobalBackendManager.set_attn_backend(None, PrecisionType.BF16_FP16_FP32)

    # Repo's own accounting (benchmark/ops/training/bench_attention_turbo.py): forward is
    # 2*B*Sq*Sk*Hq*(Dqk+Dv), halved for square causal; backward is 2.5x forward. Hq, not
    # Hkv -- GQA saves bytes, not math.
    fwd_flops = 2 * b * sq * sq * hq * (d + d)
    if causal:
        fwd_flops //= 2
    bwd_flops = fwd_flops * 2.5
    total_ms = result["fwd_ms"] + result["bwd_ms"]

    result["fwd_tflops"] = fwd_flops / (result["fwd_ms"] * 1e-3) / 1e12
    result["bwd_tflops"] = bwd_flops / (result["bwd_ms"] * 1e-3) / 1e12
    result["total_ms"] = total_ms
    result["total_tflops"] = (fwd_flops + bwd_flops) / (total_ms * 1e-3) / 1e12
    result["bwd_share"] = result["bwd_ms"] / total_ms
    # A Llama-3.1-8B step is 32 layers x (1 fwd + 1 bwd) at this shape.
    result["per_step_ms"] = total_ms * 32
    result["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30

    post = dmesg_faults()
    result["dmesg_faults_after"] = len(post)
    if len(post) > len(pre):
        result["new_dmesg_faults"] = post[len(pre):]

    result["ok"] = True
    result["wall_s"] = time.time() - result["wall_start"]
    print(json.dumps(result))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
