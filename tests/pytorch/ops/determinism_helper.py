###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared harness for the bit-exact determinism suite (``test_gemm_determinism.py``).

One case runs ``DET_REPEATS`` times in process: run 0 is cold (dispatcher picks, FlyDSL
tuner picks and the module-level FlyDSL workspaces are all dropped first), the rest are
warm. Before every run the caching allocator's free pools are refilled with a sentinel
that changes from run to run, and so are the cached FlyDSL workspaces -- once before the
forward and again between forward and backward, since several of them are written by the
forward's shapes but read by the backward. A kernel that reads memory it never wrote
therefore sees a different value in every run and shows up as a mismatch.

Every run is compared to run 0 bit for bit (integer views, so -0.0 vs +0.0 and NaN
payloads count), all mismatches are collected before failing, and correctness against an
fp32 reference is reported as SNR on its own line so the two verdicts never mix.

Autotune is never forced: it follows ``PRIMUS_TURBO_AUTO_TUNE`` (default off).
"""

import importlib
import json
import os
import time
import zlib
from typing import Callable, Dict, List, Optional

import pytest
import torch

from primus_turbo.pytorch.core.backend import AutoKernelDispatcher
from tests.pytorch.test_utils import compute_snr

DET_REPEATS = 4

# Bytes near 0x7f decode to moderate finite E8M0 scales, so an unwritten scale slot stays
# finite but sentinel-dependent. As 16-bit floats several of them are huge or NaN, which is
# just as visible.
POISON_SENTINELS = [0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x7C, 0x83, 0x7B, 0x84]

_POOL_LARGE_BYTES = 1 << 30
_POOL_LARGE_BLOCKS = 2
_POOL_SMALL_BYTES = 1 << 20  # the largest size the caching allocator serves from its small pool
_POOL_SMALL_BLOCKS = 64

# (module, attribute) of every module-level FlyDSL workspace cache that holds pure scratch.
# Handoff flags and work-steal counters must stay zero at rest and are left alone.
_WORKSPACE_CACHES = [
    ("primus_turbo.flydsl.gemm.gemm_mxfp8_kernel", "_MXFP8_WS_CACHE"),
    ("primus_turbo.flydsl.gemm.gemm_mxfp8_kernel", "_MXFP8_SPLIT_WS_CACHE"),
    ("primus_turbo.flydsl.gemm.gemm_mxfp4_kernel", "_MXFP4_SCALE_WS"),
    ("primus_turbo.flydsl.gemm.gemm_mxfp4_kernel", "_MXFP4_SPLIT_WS"),
    ("primus_turbo.flydsl.gemm.gemm_fp8_kernel", "_TN4_WS_CACHE"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel", "_GNT_WS_CACHE"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_kernel", "_GWG_WS_CACHE"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel", "_GMXFP4_WS_CACHE"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp4_kernel", "_GMXFP4_WGRAD_WS_CACHE"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_mxfp8_glu_kernel", "_DGLU_DACT_WS"),
    ("primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8_kernel", "_WGRAD_SPLIT_WS_CACHE"),
]

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _mid_poison_enabled() -> bool:
    return os.environ.get("PRIMUS_TURBO_DET_MID_POISON", "1") != "0"


def case_seed(nodeid: str) -> int:
    return zlib.crc32(nodeid.encode())


def make_generator(seed: int, device="cuda") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def randn(shape, dtype, gen: torch.Generator, scale: float = 1.0, normalize: bool = False) -> torch.Tensor:
    t = torch.randn(shape, dtype=torch.float32, device=gen.device, generator=gen)
    if normalize:
        t = t / t.abs().max()
    return (t * scale).to(dtype)


def unbalanced_group_lens(B: int, M: int, seed: int, device="cuda") -> torch.Tensor:
    """Ragged lengths summing to B * M, every group non-empty, from a local generator."""
    gen = torch.Generator().manual_seed(seed)
    dist = 0.2 + 0.8 * torch.rand(B, generator=gen)
    dist /= dist.sum()
    lens = (dist * B * M).to(torch.int64)
    lens[-1] += B * M - lens.sum()
    assert int(lens.min()) > 0
    return lens.to(device)


def _workspace_caches():
    """Yield ``(module, attr, cache)`` for every entry of ``_WORKSPACE_CACHES``.

    Only a missing optional dependency (FlyDSL itself) skips an entry. A Primus-Turbo module or
    cache that no longer exists under its listed name raises: skipping it would quietly stop
    poisoning and clearing that workspace while every case kept passing.
    """
    for mod_name, attr in _WORKSPACE_CACHES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            if (e.name or "").startswith("primus_turbo"):
                raise ImportError(f"{mod_name} is gone; update _WORKSPACE_CACHES") from e
            continue
        cache = getattr(mod, attr, None)
        if not isinstance(cache, dict):
            raise AttributeError(f"{mod_name}.{attr} is not a workspace dict; update _WORKSPACE_CACHES")
        yield mod, attr, cache


def _workspace_tensors():
    for mod, attr, cache in _workspace_caches():
        for key, value in list(cache.items()):
            if attr == "_WGRAD_SPLIT_WS_CACHE":
                # [bands * OUT_M + flag_rows, OUT_N]; the trailing flag rows are zero at rest, so
                # their count is rebuilt from the cache key, checked against the tensor's width.
                _, _, out_m, out_n, groups, block_m, block_n = key
                if value.shape[1] != out_n:
                    raise AssertionError(f"_WGRAD_SPLIT_WS_CACHE key layout changed: {key}")
                tiles = -(-out_m // block_m) * -(-out_n // block_n)
                frows = mod._wgrad_flag_rows(groups * tiles, out_n)
                yield value[: value.shape[0] - frows]
                continue
            stack = [value]
            while stack:
                v = stack.pop()
                if isinstance(v, torch.Tensor):
                    yield v
                elif isinstance(v, (tuple, list)):
                    stack.extend(v)


def poison_workspaces(byte: int) -> None:
    for t in _workspace_tensors():
        if t.is_cuda and t.numel() > 0 and t.is_contiguous():
            t.view(torch.uint8).fill_(byte)


def clear_workspaces() -> None:
    for _, _, cache in _workspace_caches():
        cache.clear()


def _cached_free_block_sizes(device_index: int) -> List[int]:
    """Request sizes that, allocated largest first, land exactly on every free cached block.

    Segments that still hold a live tensor (an earlier run's result, a workspace) survive
    ``empty_cache`` with their free parts holding whatever was last written there, so those
    parts have to be refilled too or a later run could read the same stale bytes as run 0.
    """
    sizes = []
    for seg in torch.cuda.memory_snapshot():
        if seg["device"] != device_index:
            continue
        small = seg["segment_type"] == "small"
        for blk in seg["blocks"]:
            if blk["state"] != "inactive":
                continue
            size = blk["size"]
            if small:
                while size > 0:
                    sizes.append(min(size, _POOL_SMALL_BYTES))
                    size -= _POOL_SMALL_BYTES
            elif size > _POOL_SMALL_BYTES:  # smaller large-pool remainders serve no request
                sizes.append(size)
    return sorted(sizes, reverse=True)


def poison_allocator(byte: int, device="cuda") -> None:
    """Fill every free cached block, plus fresh large and small segments, with ``byte``."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    sizes = _cached_free_block_sizes(index)
    sizes += [_POOL_LARGE_BYTES] * _POOL_LARGE_BLOCKS + [_POOL_SMALL_BYTES] * _POOL_SMALL_BLOCKS
    blocks = [torch.empty(s, dtype=torch.uint8, device=device) for s in sizes]
    for b in blocks:
        b.fill_(byte)
    torch.cuda.synchronize()
    del blocks


def reset_cold() -> None:
    """Drop every pick and workspace a previous case left behind, keeping pinned backends."""
    torch.cuda.synchronize()
    AutoKernelDispatcher.clear_all_caches()
    try:
        from primus_turbo.flydsl.autotune import runtime

        runtime.clear()
    except ImportError:
        pass
    clear_workspaces()


RunOnce = Callable[[Callable[[], None]], Dict[str, torch.Tensor]]


def run_repeats(run_once: RunOnce, repeats: int = DET_REPEATS) -> List[Dict[str, torch.Tensor]]:
    """Run ``run_once(mid)`` ``repeats`` times; ``mid`` must be called between fwd and bwd."""
    reset_cold()
    results = []
    for i in range(repeats):
        sentinel = POISON_SENTINELS[i % len(POISON_SENTINELS)]
        poison_allocator(sentinel)
        poison_workspaces(sentinel)

        def mid(sentinel=sentinel):
            if _mid_poison_enabled():
                poison_workspaces(sentinel)

        out = run_once(mid)
        torch.cuda.synchronize()
        results.append({k: v.detach() for k, v in out.items()})
    return results


def _int_view(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(_INT_VIEW[t.element_size()])


def compare_bitwise(results: List[Dict[str, torch.Tensor]]) -> List[dict]:
    """Every (tensor, run) that differs from run 0, with enough detail to triage it."""
    ref = results[0]
    mismatches = []
    for i in range(1, len(results)):
        for name, t0 in ref.items():
            ti = results[i][name]
            assert ti.shape == t0.shape and ti.dtype == t0.dtype, name
            diff = _int_view(t0) != _int_view(ti)
            n_bad = int(diff.sum())
            if n_bad == 0:
                continue
            # Only the mismatched elements, and only where both sides are finite: inf - inf is
            # nan, and a nan anywhere would hide the real largest difference.
            flat = diff.reshape(-1)
            f0, fi = t0.float().reshape(-1)[flat], ti.float().reshape(-1)[flat]
            finite = torch.isfinite(f0) & torch.isfinite(fi)
            absdiff = (f0[finite] - fi[finite]).abs()
            ref_abs = t0.float().abs()
            ref_abs = ref_abs[torch.isfinite(ref_abs)]
            first = torch.nonzero(diff.reshape(t0.shape))[0].tolist()
            mismatches.append(
                {
                    "tensor": name,
                    "runs": f"run0 vs run{i}",
                    "mismatched": n_bad,
                    "nonfinite": int((~finite).sum()),
                    "numel": t0.numel(),
                    "max_abs_diff": float(absdiff.max()) if absdiff.numel() else float("nan"),
                    "max_abs_ref": float(ref_abs.max()) if ref_abs.numel() else float("nan"),
                    "first_index": first,
                }
            )
    return mismatches


def _report_path() -> Optional[str]:
    path = os.environ.get("PRIMUS_TURBO_DET_REPORT")
    if not path:
        return None
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return f"{path}.{worker}" if worker else path


def _write_record(record: dict) -> None:
    path = _report_path()
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def check_determinism(
    nodeid: str,
    run_once: RunOnce,
    reference: Optional[Dict[str, torch.Tensor]] = None,
    snr_floor: Optional[float] = None,
    snr_views: Optional[Dict[str, Callable[[Dict[str, torch.Tensor]], torch.Tensor]]] = None,
    repeats: int = DET_REPEATS,
) -> List[Dict[str, torch.Tensor]]:
    """Run, compare every repeat to run 0, report SNR separately, then fail if either failed.

    ``reference`` maps names to fp32 references for the SNR report; a name is looked up in
    run 0's results unless ``snr_views`` derives it from them. A pinned backend that refuses
    the inputs skips the case as unsupported.
    """
    t_start = time.time()
    try:
        results = run_repeats(run_once, repeats)
    except ValueError as e:
        if "cannot handle" not in str(e):
            raise
        _write_record({"nodeid": nodeid, "status": "unsupported", "error": str(e)[:500]})
        pytest.skip(f"unsupported: {str(e)[:200]}")
    mismatches = compare_bitwise(results)

    snr = {}
    for name, ref in (reference or {}).items():
        view = (snr_views or {}).get(name)
        got = view(results[0]) if view is not None else results[0][name]
        snr[name] = compute_snr(ref.float(), got.float())
    low_snr = {k: v for k, v in snr.items() if snr_floor is not None and not v > snr_floor}

    for m in mismatches:
        print(
            f"[det-mismatch] {nodeid} {m['tensor']} {m['runs']}: {m['mismatched']}/{m['numel']} "
            f"(non-finite {m['nonfinite']}) max|diff|={m['max_abs_diff']:.6g} "
            f"max|ref|={m['max_abs_ref']:.6g} first={m['first_index']}"
        )
    print(f"[det-snr] {nodeid} " + " ".join(f"{k}={v:.2f}dB" for k, v in snr.items()))

    status = "nondeterministic" if mismatches else ("snr_fail" if low_snr else "deterministic")
    _write_record(
        {
            "nodeid": nodeid,
            "status": status,
            "repeats": repeats,
            "mismatches": mismatches,
            "snr": snr,
            "snr_floor": snr_floor,
            "secs": round(time.time() - t_start, 2),
        }
    )

    msgs = []
    if mismatches:
        tensors = sorted({m["tensor"] for m in mismatches})
        msgs.append(f"non-deterministic across {repeats} runs: {tensors} ({len(mismatches)} mismatches)")
    if low_snr:
        msgs.append(f"SNR below {snr_floor} dB: {low_snr}")
    assert not msgs, "; ".join(msgs)
    return results
