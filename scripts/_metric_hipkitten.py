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


def _pick_idle_gpu() -> str | None:
    """Return the smallest idle GPU id (busy = any PID using >100MiB VRAM).

    Falls back to None when rocm-smi is unavailable so the caller can let
    the runtime choose the default device.
    """
    import re
    import subprocess
    THR = 100 * 1024 * 1024
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showuse", "--showpids"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
    except Exception:
        return None
    all_gpus = sorted({int(m) for m in re.findall(r"^GPU\[(\d+)\]", out, flags=re.M)})
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
    return out


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


def collect_probes() -> list[tuple[str, Callable[[], tuple[bool, str]]]]:
    probes: list[tuple[str, Callable[[], tuple[bool, str]]]] = []
    probes.extend(_bf16_dense_probes())
    probes.extend(_bf16_grouped_probes())
    probes.extend(_fp8_dense_probes())
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
