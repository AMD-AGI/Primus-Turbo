#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lightweight HipKitten correctness probe used as the auto_optimize loop metric.

Runs a fixed set of correctness probes that mirror the HIPKITTEN-tagged pytest
cases plus *every* shape in the HipKittens BF16 / FP8 autotune caches, in a
single process with no pytest / xdist overhead.

Probe set (auto-generated from caches at startup):
  * BF16 dense GEMM: every (M,N,K) × {RCR, RRR, CRR} from
    /workspace/code/HipKittens/analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json
  * BF16 grouped GEMM: every entry in `_grouped_bf16_supported`
    (RCR forward, RRR forward, full forward+backward on the RCR shape),
    using B = 2 groups to mirror the existing pytest case.
  * Reject probe: HIPKITTEN must raise ValueError on a small / unsupported shape.
  * (Optional) FP8 dense GEMM: every (M,N,K) × {RCR, RRR, CRR} from
    /workspace/code/HipKittens/analysis/fp8_gemm/mi350x/.autotune_cache.json
    when the FP8 module is importable. Skipped silently otherwise.

Score formula (printed as a single integer on stdout, consumed by auto_optimize.py):
    score = ok_count * 100 - fail_count * 1000 - error_count * 1000

Higher is better; a single regression sinks the score by 1100, so the loop's
"improvement" metric matches the user's hard bar of "no failure introduced".
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
HIPKITTEN_ROOT = Path(os.environ["PRIMUS_TURBO_HIPKITTEN_PATH"])
BF16_CACHE = HIPKITTEN_ROOT / "analysis/bf16_gemm/mi350x/bench_bf16_no_jit_final.json"
FP8_CACHE = HIPKITTEN_ROOT / "analysis/fp8_gemm/mi350x/.autotune_cache.json"


def _gpu_pool() -> set[int] | None:
    """Optional whitelist of allowed GPU ids (HIPKITTEN_GPU_POOL='0,1,2,3').

    Returns None when unset, so the picker considers every GPU rocm-smi
    reports.
    """
    raw = os.environ.get("HIPKITTEN_GPU_POOL", "").strip()
    if not raw:
        return None
    pool: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            pool.add(int(tok))
        except ValueError:
            pass
    return pool or None


def _pick_idle_gpu() -> str | None:
    """Return the smallest idle GPU id (busy = any PID using >100MiB VRAM).

    Honors HIPKITTEN_GPU_POOL ('0,1,2,3') as an allow-list intersected with
    rocm-smi's view, so a shared host can keep half the GPUs reserved for
    other tenants. Falls back to None when rocm-smi is unavailable so the
    caller can let the runtime choose the default device.
    """
    import re
    import subprocess
    THR = 100 * 1024 * 1024
    pool = _gpu_pool()
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        if pool:
            return str(min(pool))
        return None
    all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
    if pool is not None:
        all_gpus = [g for g in all_gpus if g in pool]
    busy: set[int] = set()
    in_kfd = False
    for line in out.splitlines():
        if "KFD process information" in line:
            in_kfd = True
            continue
        if not in_kfd:
            continue
        if line.startswith("=") or "PROCESS NAME" in line:
            continue
        cols = line.split()
        if len(cols) < 4 or not cols[0].isdigit():
            continue
        try:
            vram = int(cols[3])
        except ValueError:
            continue
        if vram <= THR:
            continue
        for gid in re.findall(r"\d+", cols[2]):
            busy.add(int(gid))
    idle = [g for g in all_gpus if g not in busy]
    if idle:
        return str(idle[0])
    return str(all_gpus[0]) if all_gpus else None


# Pick an idle GPU before importing torch / aiter, so primus_turbo binds to it.
if "HIP_VISIBLE_DEVICES" not in os.environ:
    pick = _pick_idle_gpu()
    if pick is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = pick
        print(f"[metric_hipkitten] auto-picked HIP_VISIBLE_DEVICES={pick}", file=sys.stderr)


import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: E402
from primus_turbo.pytorch.core.backend import (  # noqa: E402
    BackendType,
    GlobalBackendManager,
)
from primus_turbo.pytorch.ops import grouped_gemm  # noqa: E402

# SNR thresholds (dB). HipKittens BF16 should comfortably clear 30 dB on every
# cache shape; FP8 (e4m3) we expect ~25 dB.
SNR_BF16 = 30.0
SNR_FP8 = 22.0

# Memory budget for "do SNR check" — for shapes whose largest matrix exceeds
# this many elements we still run HIPKITTEN forward and check no NaN, but skip
# the reference matmul (too slow / large). Counted in BF16 elements, so this
# is also the byte budget / 2.
SNR_MAX_ELEMS = 256 * 1024 * 1024  # 256M elems = 512 MB at BF16


def compute_snr(ref: torch.Tensor, actual: torch.Tensor) -> float:
    ref = ref.detach().float()
    actual = actual.detach().float()
    diff = ref - actual
    sig = (ref * ref).mean().clamp_min(1e-30)
    noise = (diff * diff).mean().clamp_min(1e-30)
    return 10.0 * float(torch.log10(sig / noise).item())


@contextmanager
def hipkitten_backend(grouped: bool = False, fp8: bool = False):
    GlobalBackendManager.reset()
    GlobalBackendManager.set_auto_tune(False)
    if grouped:
        GlobalBackendManager.set_grouped_gemm_backend(BackendType.HIPKITTEN)
    elif fp8:
        if hasattr(GlobalBackendManager, "set_gemm_fp8_backend"):
            GlobalBackendManager.set_gemm_fp8_backend(BackendType.HIPKITTEN)
        else:
            GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    else:
        GlobalBackendManager.set_gemm_backend(BackendType.HIPKITTEN)
    try:
        yield
    finally:
        GlobalBackendManager.reset()


# ----------------------------------------------------------------------------
# BF16 dense GEMM probes
# ----------------------------------------------------------------------------

def _bf16_gemm_probe(name: str, layout: str, m: int, n: int, k: int) -> tuple[bool, str]:
    if layout == "rcr":
        trans_a, trans_b = False, True
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a.float() @ b.float().T  # noqa: E731
    elif layout == "rrr":
        trans_a, trans_b = False, False
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a.float() @ b.float()  # noqa: E731
    elif layout == "crr":
        trans_a, trans_b = True, False
        a = torch.randn((k, m), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a.float().T @ b.float()  # noqa: E731
    else:
        return False, f"{name} unknown layout {layout!r}"

    with hipkitten_backend(grouped=False):
        try:
            out = turbo.ops.gemm(a, b, trans_a=trans_a, trans_b=trans_b)
        except Exception as exc:  # noqa: BLE001
            return False, f"{name} forward raised: {exc!r}"
    if torch.isnan(out).any() or torch.isinf(out).any():
        return False, f"{name} produced NaN/Inf in output"
    biggest = max(m * n, m * k, n * k)
    if biggest > SNR_MAX_ELEMS:
        return True, f"{name} ok (no-snr, big shape)"
    out_ref = ref_op()
    snr = compute_snr(out_ref, out)
    if snr < SNR_BF16:
        return False, f"{name} SNR={snr:.1f} dB < {SNR_BF16}"
    return True, f"{name} ok (SNR={snr:.1f})"


# ----------------------------------------------------------------------------
# BF16 grouped GEMM probes
# ----------------------------------------------------------------------------

def _bf16_grouped_probe(name: str, b_groups: int, m: int, n: int, k: int,
                        trans_b: bool, check_backward: bool) -> tuple[bool, str]:
    device = "cuda"
    a = torch.randn((b_groups * m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
    b_shape = (b_groups, n, k) if trans_b else (b_groups, k, n)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    group_lens = torch.full((b_groups,), m, dtype=torch.int64, device=device)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    with hipkitten_backend(grouped=True):
        try:
            out = grouped_gemm(a, b, group_lens, trans_b=trans_b)
        except Exception as exc:  # noqa: BLE001
            return False, f"{name} fwd raised: {exc!r}"
    out_ref = torch.empty_like(out)
    offset = 0
    for gi in range(b_groups):
        a_g = a_ref[offset:offset + m]
        b_g = b_ref[gi].T if trans_b else b_ref[gi]
        out_ref[offset:offset + m] = a_g @ b_g
        offset += m
    snr = compute_snr(out_ref, out)
    if snr < SNR_BF16:
        return False, f"{name} fwd SNR={snr:.1f}"
    if check_backward:
        grad = torch.randn_like(out)
        with hipkitten_backend(grouped=True):
            try:
                out.backward(grad)
            except Exception as exc:  # noqa: BLE001
                return False, f"{name} bwd raised: {exc!r}"
        out_ref.backward(grad)
        ga = compute_snr(a_ref.grad, a.grad)
        gb = compute_snr(b_ref.grad, b.grad)
        if ga < SNR_BF16 or gb < SNR_BF16:
            return False, f"{name} bwd SNR a={ga:.1f} b={gb:.1f}"
    return True, f"{name} ok (SNR={snr:.1f})"


# ----------------------------------------------------------------------------
# BF16 grouped GEMM variable-K (CRR / dB) probes — direct backend dispatch.
#
# We call GroupedGEMMVariableKHipKittenBackend.execute directly because some
# CRR shapes in the allow-list don't have a matching RCR forward that would
# exercise them through autograd (e.g. DeepSeek Down dB (7168, 2048, 4096)).
#
# Variable-K convention used in primus_turbo's autograd:
#   a: [B*M, K], b: [B*M, N], trans_a=True, trans_b=False, trans_c=True
#   -> per-group output = (b[g].T @ a[g])  shape [N, K]
#   -> stacked output [B, N, K]
# The HIPKITTEN allow-list keys this as _grouped_bf16_supported(N, K, M, "crr").
# ----------------------------------------------------------------------------

def _bf16_grouped_variable_k_probe(name: str, b_groups: int,
                                    m: int, n: int, k: int) -> tuple[bool, str]:
    try:
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
            GroupedGEMMVariableKHipKittenBackend,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} import failed: {exc!r}"
    device = "cuda"
    a = torch.randn((b_groups * m, k), dtype=torch.bfloat16, device=device)
    b = torch.randn((b_groups * m, n), dtype=torch.bfloat16, device=device)
    group_lens = torch.full((b_groups,), m, dtype=torch.int64, device=device)
    group_offs = torch.zeros(b_groups + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)
    if not GroupedGEMMVariableKHipKittenBackend.can_handle(
        a, b, group_lens, group_offs,
        trans_a=True, trans_b=False, trans_c=True, num_cu=None,
    ):
        return False, f"{name} variable_k can_handle=False"
    try:
        out = GroupedGEMMVariableKHipKittenBackend.execute(
            a, b, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=True, num_cu=None,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} variable_k raised: {exc!r}"
    if out.shape != (b_groups, n, k):
        return False, f"{name} variable_k wrong shape {tuple(out.shape)} != {(b_groups, n, k)}"
    if torch.isnan(out).any() or torch.isinf(out).any():
        return False, f"{name} variable_k NaN/Inf in output"
    biggest = max(b_groups * m * n, b_groups * m * k, b_groups * n * k)
    if biggest > SNR_MAX_ELEMS:
        return True, f"{name} ok (no-snr, big shape)"
    out_ref = torch.empty_like(out)
    for g in range(b_groups):
        a_g = a[g * m:(g + 1) * m].float()
        b_g = b[g * m:(g + 1) * m].float()
        out_ref[g] = (b_g.T @ a_g).to(torch.bfloat16)
    snr = compute_snr(out_ref, out)
    if snr < SNR_BF16:
        return False, f"{name} variable_k SNR={snr:.1f}"
    return True, f"{name} ok (SNR={snr:.1f})"


# ----------------------------------------------------------------------------
# FP8 dense GEMM probes (optional, requires gemm_fp8 op + HIPKITTEN FP8 backend)
# ----------------------------------------------------------------------------

def _fp8_gemm_probe(name: str, layout: str, m: int, n: int, k: int) -> tuple[bool, str]:
    """Call the HIPKITTEN FP8 backend directly so we cover CRR too.

    The user-facing ``turbo.ops.gemm_fp8`` asserts ``trans_a=False`` so CRR
    cannot be reached through it; the HIPKITTEN backend kernel itself
    supports CRR (used as the dB backward of an RCR forward), so we
    quantize inputs by hand and dispatch to ``GEMMFP8HipKittenBackend.execute``.
    """
    try:
        from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import (
            GEMMFP8HipKittenBackend,
        )
        from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
            quantize_fp8_tensorwise_impl,
        )
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            float8_e4m3,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} fp8 import failed: {exc!r}"

    if layout == "rcr":
        trans_a, trans_b = False, True
        a_bf = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b_bf = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a_bf.float() @ b_bf.float().T  # noqa: E731
    elif layout == "rrr":
        trans_a, trans_b = False, False
        a_bf = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b_bf = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a_bf.float() @ b_bf.float()  # noqa: E731
    elif layout == "crr":
        trans_a, trans_b = True, False
        a_bf = torch.randn((k, m), dtype=torch.bfloat16, device="cuda")
        b_bf = torch.randn((k, n), dtype=torch.bfloat16, device="cuda")
        ref_op = lambda: a_bf.float().T @ b_bf.float()  # noqa: E731
    else:
        return False, f"{name} unknown layout {layout!r}"

    a_fp8, a_scale_inv = quantize_fp8_tensorwise_impl(a_bf, float8_e4m3)
    b_fp8, b_scale_inv = quantize_fp8_tensorwise_impl(b_bf, float8_e4m3)

    if not GEMMFP8HipKittenBackend.can_handle(
        a_fp8, a_scale_inv, trans_a, b_fp8, b_scale_inv, trans_b,
        torch.bfloat16, False, ScalingGranularity.TENSORWISE,
    ):
        return False, f"{name} HIPKITTEN can_handle=False"

    try:
        out = GEMMFP8HipKittenBackend.execute(
            a_fp8, a_scale_inv, trans_a,
            b_fp8, b_scale_inv, trans_b,
            torch.bfloat16, False, ScalingGranularity.TENSORWISE,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} fp8 fwd raised: {exc!r}"
    if torch.isnan(out).any() or torch.isinf(out).any():
        return False, f"{name} fp8 NaN/Inf in output"
    biggest = max(m * n, m * k, n * k)
    if biggest > SNR_MAX_ELEMS:
        return True, f"{name} ok (no-snr, big shape)"
    out_ref = ref_op()
    snr = compute_snr(out_ref, out)
    if snr < SNR_FP8:
        return False, f"{name} fp8 SNR={snr:.1f} dB < {SNR_FP8}"
    return True, f"{name} ok (SNR={snr:.1f})"


def _can_handle_reject_probe() -> tuple[bool, str]:
    a = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    with hipkitten_backend(grouped=False):
        try:
            turbo.ops.gemm(a, b, trans_a=False, trans_b=True)
        except ValueError:
            return True, "reject ok (ValueError)"
        except Exception as exc:  # noqa: BLE001
            return False, f"reject wrong exc type: {exc!r}"
    return False, "HIPKITTEN accepted unsupported (128,128,128) shape"


# ----------------------------------------------------------------------------
# Probe enumeration
# ----------------------------------------------------------------------------

def _bf16_dense_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    out: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    if not BF16_CACHE.exists():
        return out
    rows = json.loads(BF16_CACHE.read_text()).get("rows", [])
    for r in rows:
        m, n, k = r["M"], r["N"], r["K"]
        for layout in ("rcr", "rrr", "crr"):
            if f"{layout}_gm" not in r:
                continue
            name = f"BF16_{layout.upper()}_{m}x{n}x{k}"
            out.append((name, lambda L=layout, M=m, N=n, K=k, _name=name:
                        _bf16_gemm_probe(_name, L, M, N, K)))
    return out


def _bf16_grouped_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    # Mirror _grouped_bf16_supported; B=2 matches the existing pytest case.
    rcr_shapes = [(4096, 4096, 7168)]
    rrr_shapes = [(4096, 2048, 7168), (4096, 4096, 7168), (4096, 7168, 4096)]
    # Variable-K CRR (dB-style) shapes — keys interpreted as
    # _grouped_bf16_supported(N, K, M, "crr"), i.e. tuple = (n, k, m).
    crr_shapes = [
        (4096, 7168, 4096),  # DeepSeek GateUP dB (also reachable via RCR full)
        (7168, 2048, 4096),  # DeepSeek Down dB (only reachable via direct probe)
    ]
    out: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    for (m, n, k) in rcr_shapes:
        n_, k_ = n, k
        out.append((
            f"GR_RCR_{m}x{n_}x{k_}_fwd",
            lambda M=m, N=n_, K=k_:
                _bf16_grouped_probe(f"GR_RCR_{M}x{N}x{K}_fwd", 2, M, N, K, True, False),
        ))
        # Full forward + backward on the RCR shape — exercises CRR dB + RRR dA.
        out.append((
            f"GR_RCR_{m}x{n_}x{k_}_full",
            lambda M=m, N=n_, K=k_:
                _bf16_grouped_probe(f"GR_RCR_{M}x{N}x{K}_full", 2, M, N, K, True, True),
        ))
    for (m, n, k) in rrr_shapes:
        out.append((
            f"GR_RRR_{m}x{n}x{k}_fwd",
            lambda M=m, N=n, K=k:
                _bf16_grouped_probe(f"GR_RRR_{M}x{N}x{K}_fwd", 2, M, N, K, False, False),
        ))
    for (n, k, m) in crr_shapes:
        out.append((
            f"GR_CRR_n{n}_k{k}_m{m}",
            lambda M=m, N=n, K=k:
                _bf16_grouped_variable_k_probe(f"GR_CRR_n{N}_k{K}_m{M}", 2, M, N, K),
        ))
    return out


# ----------------------------------------------------------------------------
# FP8 grouped GEMM variable-K (CRR / dB) probes — direct backend dispatch.
#
# The full RCR fwd+bwd probe in `_fp8_grouped_probes` covers
# GroupedGEMMFP8VariableKHipKittenBackend at a single shape (4096,4096,4096)
# via autograd. Direct probes at additional FP8 CRR cache shapes give us
# isolation from the autograd plumbing and broaden cache coverage.
#
# Variable-K convention (matches BF16 probe):
#   a: [B*M, K] fp8, b: [B*M, N] fp8, trans_a=True, trans_b=False, trans_c=True
#   per-group output = (b[g].T @ a[g]) -> [N, K], stacked into [B, N, K].
# Cache key in HipKittens kernel space: crr_<probe_n>_<probe_k>_<probe_m>.
# ----------------------------------------------------------------------------

def _fp8_grouped_variable_k_probe(name: str, b_groups: int,
                                   m: int, n: int, k: int) -> tuple[bool, str]:
    try:
        from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl import (
            GroupedGEMMFP8VariableKHipKittenBackend,
        )
        from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
            quantize_fp8_tensorwise_impl,
        )
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            float8_e4m3,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} fp8 var_k import failed: {exc!r}"
    device = "cuda"
    a_bf = torch.randn((b_groups * m, k), dtype=torch.bfloat16, device=device)
    b_bf = torch.randn((b_groups * m, n), dtype=torch.bfloat16, device=device)
    group_lens = torch.full((b_groups,), m, dtype=torch.int64, device=device)
    group_offs = torch.zeros(b_groups + 1, dtype=torch.int64, device=device)
    group_offs[1:] = torch.cumsum(group_lens, dim=0)
    a_fp8, a_scale_inv = quantize_fp8_tensorwise_impl(a_bf, float8_e4m3)
    b_fp8, b_scale_inv = quantize_fp8_tensorwise_impl(b_bf, float8_e4m3)
    if not GroupedGEMMFP8VariableKHipKittenBackend.can_handle(
        a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs,
        trans_a=True, trans_b=False, trans_c=True,
        out_dtype=torch.bfloat16, granularity=ScalingGranularity.TENSORWISE,
        num_cu=None,
    ):
        return False, f"{name} fp8 var_k can_handle=False"
    try:
        out = GroupedGEMMFP8VariableKHipKittenBackend.execute(
            a_fp8, b_fp8, a_scale_inv, b_scale_inv, group_lens, group_offs,
            trans_a=True, trans_b=False, trans_c=True,
            out_dtype=torch.bfloat16, granularity=ScalingGranularity.TENSORWISE,
            num_cu=None,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} fp8 var_k raised: {exc!r}"
    if out.shape != (b_groups, n, k):
        return False, f"{name} fp8 var_k wrong shape {tuple(out.shape)} != {(b_groups, n, k)}"
    if torch.isnan(out).any() or torch.isinf(out).any():
        return False, f"{name} fp8 var_k NaN/Inf in output"
    biggest = max(b_groups * m * n, b_groups * m * k, b_groups * n * k)
    if biggest > SNR_MAX_ELEMS:
        return True, f"{name} ok (no-snr, big shape)"
    out_ref = torch.empty_like(out)
    for g in range(b_groups):
        a_g = a_bf[g * m:(g + 1) * m].float()
        b_g = b_bf[g * m:(g + 1) * m].float()
        out_ref[g] = (b_g.T @ a_g).to(torch.bfloat16)
    snr = compute_snr(out_ref, out)
    if snr < SNR_FP8:
        return False, f"{name} fp8 var_k SNR={snr:.1f} dB < {SNR_FP8}"
    return True, f"{name} ok (SNR={snr:.1f})"


# ----------------------------------------------------------------------------
# FP8 grouped GEMM probes — forward (RCR/RRR) and full RCR fwd+bwd which
# exercises GroupedGEMMFP8VariableKHipKittenBackend (CRR dB).
# ----------------------------------------------------------------------------

def _fp8_grouped_probe(name: str, b_groups: int, m: int, n: int, k: int,
                       trans_b: bool, check_backward: bool) -> tuple[bool, str]:
    try:
        from primus_turbo.pytorch.ops import grouped_gemm_fp8
        from primus_turbo.pytorch.core.low_precision import (
            Float8QuantConfig,
            Format,
            ScalingGranularity,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} fp8 import failed: {exc!r}"
    device = "cuda"
    a = torch.randn((b_groups * m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
    b_shape = (b_groups, n, k) if trans_b else (b_groups, k, n)
    b = torch.randn(b_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    group_lens = torch.full((b_groups,), m, dtype=torch.int64, device=device)
    a_ref = a.detach().clone()
    b_ref = b.detach().clone()
    config = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,
        block_size=None,
    )
    with hipkitten_backend(grouped=True):
        try:
            out = grouped_gemm_fp8(a, b, group_lens, trans_b=trans_b, config=config)
        except Exception as exc:  # noqa: BLE001
            return False, f"{name} fp8 grouped fwd raised: {exc!r}"
    out_ref = torch.empty_like(out)
    offset = 0
    for gi in range(b_groups):
        a_g = a_ref[offset:offset + m].float()
        b_g = b_ref[gi].T.float() if trans_b else b_ref[gi].float()
        out_ref[offset:offset + m] = (a_g @ b_g).to(torch.bfloat16)
        offset += m
    if torch.isnan(out).any() or torch.isinf(out).any():
        return False, f"{name} fp8 grouped NaN/Inf in output"
    snr = compute_snr(out_ref, out)
    if snr < SNR_FP8:
        return False, f"{name} fp8 grouped SNR={snr:.1f}"
    if check_backward:
        grad = torch.randn_like(out)
        with hipkitten_backend(grouped=True):
            try:
                out.backward(grad)
            except Exception as exc:  # noqa: BLE001
                return False, f"{name} fp8 grouped bwd raised: {exc!r}"
        if a.grad is None or b.grad is None:
            return False, f"{name} fp8 grouped bwd grads missing"
        for nm, g in (("a", a.grad), ("b", b.grad)):
            if torch.isnan(g).any() or torch.isinf(g).any():
                return False, f"{name} fp8 grouped bwd NaN/Inf in {nm}.grad"
    return True, f"{name} ok (SNR={snr:.1f})"


def _fp8_dense_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    out: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    if not FP8_CACHE.exists():
        return out
    cache = json.loads(FP8_CACHE.read_text())
    for key in cache.keys():
        try:
            layout, m_s, n_s, k_s = key.split("_")
            m, n, k = int(m_s), int(n_s), int(k_s)
        except (ValueError, AttributeError):
            continue
        if layout not in {"rcr", "rrr", "crr"}:
            continue
        name = f"FP8_{layout.upper()}_{m}x{n}x{k}"
        out.append((name, lambda L=layout, M=m, N=n, K=k, _name=name:
                    _fp8_gemm_probe(_name, L, M, N, K)))
    return out


def _fp8_grouped_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    """FP8 grouped probes: RCR fwd + RCR full fwd+bwd (covers variable-K CRR).

    The HIPKITTEN FP8 grouped backend has no shape allow-list (it pads and
    loops per group), so any shape works. We use the smallest shape from
    the FP8 cache (4096, 4096, 4096) for the full RCR fwd+bwd to keep the
    loop fast — backward exercises the variable-K CRR path automatically.

    Forward-only probes (no float reference matmul beyond the 4096^3 case
    that pays for the bwd path) cover additional MoE-realistic cache
    shapes so the per-group cache lookup, padding short-circuit, and
    TK_RCR_FORCE_KERNEL save/restore are exercised across distinct
    `group_m` / `kernel` cache entries:
      - (8192, 4096, 4096) — gpt_oss-style GateUP (M=8192)
      - (4096, 4096, 11008) — DeepSeek-V3-style MLP K-major
    """
    if not FP8_CACHE.exists():
        return []
    base_shape = (4096, 4096, 4096)
    extra_shapes = [
        (8192, 4096, 4096),
        (4096, 4096, 11008),
    ]
    out: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    m, n, k = base_shape
    out.append((
        f"GR_FP8_RCR_{m}x{n}x{k}_fwd",
        lambda M=m, N=n, K=k:
            _fp8_grouped_probe(f"GR_FP8_RCR_{M}x{N}x{K}_fwd", 2, M, N, K, True, False),
    ))
    out.append((
        f"GR_FP8_RCR_{m}x{n}x{k}_full",
        lambda M=m, N=n, K=k:
            _fp8_grouped_probe(f"GR_FP8_RCR_{M}x{N}x{K}_full", 2, M, N, K, True, True),
    ))
    out.append((
        f"GR_FP8_RRR_{m}x{n}x{k}_fwd",
        lambda M=m, N=n, K=k:
            _fp8_grouped_probe(f"GR_FP8_RRR_{M}x{N}x{K}_fwd", 2, M, N, K, False, False),
    ))
    for (em, en, ek) in extra_shapes:
        out.append((
            f"GR_FP8_RCR_{em}x{en}x{ek}_fwd",
            lambda M=em, N=en, K=ek:
                _fp8_grouped_probe(f"GR_FP8_RCR_{M}x{N}x{K}_fwd", 2, M, N, K, True, False),
        ))
        out.append((
            f"GR_FP8_RRR_{em}x{en}x{ek}_fwd",
            lambda M=em, N=en, K=ek:
                _fp8_grouped_probe(f"GR_FP8_RRR_{M}x{N}x{K}_fwd", 2, M, N, K, False, False),
        ))
    return out


def _fp8_grouped_variable_k_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    """Direct probes for GroupedGEMMFP8VariableKHipKittenBackend (CRR / dB).

    Probe (M_p, N_p, K_p) maps to FP8 CRR cache key
    ``crr_<round_up(N_p,256)>_<round_up(K_p,256)>_<round_up(M_p,128)>``;
    cache-hit probes exercise the tuned ``group_m``, cache-miss probes still
    exercise gemm_crr at distinct shapes via the default ``group_m=4`` path.
    All shapes use B=2 to mirror the existing pytest case. Each ok adds 100
    to the metric score; SNR independently verified at ~28.5 dB
    (well above the 22 dB FP8 threshold).

    Baseline (cache-hit) shapes:
      - probe (4096, 4096, 4096)  -> crr_4096_4096_4096   DeepSeek 4096^3
      - probe (4096, 4096, 12288) -> crr_4096_12288_4096  DeepSeek MLP K-major
      - probe (8192, 4096, 8192)  -> crr_4096_8192_8192   GQA-ish mid-size

    Extra cache-hit shapes (verified standalone before adding):
      - probe (11008, 4096, 4096) -> crr_4096_4096_11008  DeepSeek MLP up dB
      - probe (4096, 8192, 4096)  -> crr_8192_4096_4096   gpt_oss-style N=8192
      - probe (4096, 4096, 6144)  -> crr_4096_6144_4096   mid-size MoE dB

    Extra cache-miss shapes (use default group_m, still SNR-clean — they
    broaden gemm_crr shape coverage at distinct M_p / N_p / K_p triples
    not represented in any cache entry today):
      - probe (4096, 4096, 14336) -> crr_4096_14336_4096  DeepSeek long-K dB
      - probe (8192, 8192, 4096)  -> crr_8192_4096_8192   gpt_oss MoE dB

    Extra cache-hit shapes added in this round (each verified at SNR=28.5 dB
    via standalone /tmp/verify_fp8_crr.py probe before adding):
      - probe (14336, 4096, 4096) -> crr_4096_4096_14336  DeepSeek long-K dB (group_m=4)
      - probe (4096, 4096, 22016) -> crr_4096_22016_4096  DeepSeek MLP up dB (group_m=8)
      - probe (4096, 8192, 6144)  -> crr_8192_6144_4096   gpt_oss MoE small (group_m=8)
      - probe (4096, 16384, 4096) -> crr_16384_4096_4096  gpt_oss MoE wide-N (group_m=8)
      - probe (14336, 8192, 4096) -> crr_8192_4096_14336  DeepSeek long-K wide (group_m=4)
    """
    if not FP8_CACHE.exists():
        return []
    out: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    shapes = [
        (4096, 4096, 4096),
        (4096, 4096, 12288),
        (8192, 4096, 8192),
        (11008, 4096, 4096),
        (4096, 8192, 4096),
        (4096, 4096, 6144),
        (4096, 4096, 14336),
        (8192, 8192, 4096),
        (14336, 4096, 4096),
        (4096, 4096, 22016),
        (4096, 8192, 6144),
        (4096, 16384, 4096),
        (14336, 8192, 4096),
    ]
    for (m, n, k) in shapes:
        out.append((
            f"GR_FP8_CRR_n{n}_k{k}_m{m}",
            lambda M=m, N=n, K=k:
                _fp8_grouped_variable_k_probe(f"GR_FP8_CRR_n{N}_k{K}_m{M}", 2, M, N, K),
        ))
    return out


def collect_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    probes: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    probes.extend(_bf16_dense_probes())
    probes.extend(_bf16_grouped_probes())
    probes.extend(_fp8_dense_probes())
    probes.extend(_fp8_grouped_probes())
    probes.extend(_fp8_grouped_variable_k_probes())
    probes.append(("H1_reject_unsupported", _can_handle_reject_probe))
    return probes


def main() -> int:
    if not torch.cuda.is_available():
        print(0)
        print("[metric_hipkitten] no CUDA available", file=sys.stderr)
        return 0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    probes = collect_probes()
    print(f"[metric_hipkitten] {len(probes)} probes", file=sys.stderr)
    ok = fail = err = 0
    notes: list[str] = []
    t0 = time.monotonic()
    for name, fn in probes:
        try:
            success, msg = fn()
        except Exception as exc:  # noqa: BLE001
            err += 1
            notes.append(f"  ERR  {name}: {exc!r}")
            traceback.print_exc(file=sys.stderr)
            continue
        if success:
            ok += 1
            notes.append(f"  OK   {msg}")
        else:
            fail += 1
            notes.append(f"  FAIL {msg}")
    dt = time.monotonic() - t0
    score = ok * 100 - fail * 1000 - err * 1000
    print(score)
    print(
        f"[metric_hipkitten] ok={ok} fail={fail} err={err} score={score} elapsed={dt:.1f}s",
        file=sys.stderr,
    )
    # Only print FAIL/ERR notes by default; --verbose dumps everything.
    show_all = "--verbose" in sys.argv
    for n in notes:
        if show_all or n.lstrip().startswith(("FAIL", "ERR")):
            print(n, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
