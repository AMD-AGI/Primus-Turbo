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

# Route torch's own matmuls to rocBLAS rather than hipBLASLt. Two independent reasons, and
# the first one is fatal rather than slow: in amdprimus:gfx1250-20260910 the hipBLASLt
# Tensile library for this arch is simply absent
# (_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/TensileLibrary_lazy_gfx1250.dat), so
# an fp32 matmul raises HIPBLAS_STATUS_INVALID_VALUE and the correctness reference cannot
# run at all. Second, where hipBLASLt IS present on this part it has been measured at
# 91.5 TFLOP/s against Triton's 1002.7 on the same tensors in the same process, so it is
# not something the reference should be built on either way.
# Set before torch is imported; torch reads it at backend-selection time.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

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
}

_FAULT_PATTERNS = (
    "MES(",
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
    ap.add_argument("--json", default="", help="also write the result object here")
    args = ap.parse_args()

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

    # Pin TRITON. Without this the dispatcher picks, and a round would not know which
    # kernel it just measured.
    GlobalBackendManager.set_attn_backend(BackendType.TRITON, PrecisionType.BF16_FP16_FP32)
    try:
        if not args.skip_correctness:
            out = flash_attn_func(q, k, v, causal=causal)
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

        fwd = lambda: flash_attn_func(q, k, v, causal=causal)  # noqa: E731
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
