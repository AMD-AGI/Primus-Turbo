###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared harness for the bit-exact determinism cases (``test_det_*``, ``--deterministic-only``).

The cases live next to each op's other tests (``test_gemm*.py``, ``test_grouped_gemm*.py``,
``test_grouped_mlp_*.py``); this module holds what they share. One case runs ``DET_REPEATS`` times in process: run 0 is cold (dispatcher picks, FlyDSL
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
import torch.nn.functional as F

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendType,
    GlobalBackendManager,
)
from primus_turbo.pytorch.core.low_precision import (
    MXFP8_BLOCK_SIZE,
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp4_support,
    check_mxfp8_support,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from tests.pytorch.ref.gemm_ref import grouped_gemm_ref
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


# ---------------------------------------------------------------------------
# What the test_det_* cases share across modules.
# ---------------------------------------------------------------------------
DET_DTYPES = [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")]


@pytest.fixture(name="det_seed")
def _det_seed(request):
    """The case's seed, with the backend registry reset on both sides of the case.

    Import ``_det_seed`` into a test module so pytest discovers ``det_seed`` there.
    """
    GlobalBackendManager.reset()
    yield case_seed(request.node.nodeid)
    GlobalBackendManager.reset()


def require_gfx950(what: str) -> None:
    if get_device_compute_capability() != (9, 5):
        pytest.skip(f"{what} is gfx950-only")


def skip_flydsl_on_gfx942(backend) -> None:
    if backend == BackendType.FLYDSL and get_device_compute_capability() == (9, 4):
        pytest.skip("FlyDSL backends are not supported on gfx942")


def require_mxfp8() -> None:
    ok, reason = check_mxfp8_support()
    if not ok:
        pytest.skip(reason)


def require_mxfp4() -> None:
    ok, reason = check_mxfp4_support()
    if not ok:
        pytest.skip(reason)


def mx8_config(fmt: Format = Format.E4M3) -> Float8QuantConfig:
    return Float8QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=fmt,
        block_size=MXFP8_BLOCK_SIZE,
        scale_dtype=ScaleDtype.E8M0,
    )


def fp4_config(scale_rounding_mode: int = 0) -> Float4QuantConfig:
    return Float4QuantConfig(
        granularity=ScalingGranularity.MX_BLOCKWISE,
        format=Format.E2M1_X2,
        block_size=32,
        scale_dtype=ScaleDtype.E8M0,
        use_gradient_sr=False,
        scale_rounding_mode=scale_rounding_mode,
    )


def fp32_leaves(*tensors) -> List[torch.Tensor]:
    return [t.detach().float().requires_grad_(True) for t in tensors]


def dense_ref(a0, b0, g, trans_a: bool = False, trans_b: bool = True) -> Dict[str, torch.Tensor]:
    a, b = fp32_leaves(a0, b0)
    out = (a.t() if trans_a else a) @ (b.t() if trans_b else b)
    out.backward(g.float())
    return {"out": out.detach(), "dgrad": a.grad, "wgrad": b.grad}


def grouped_ref(a0, b0, group_lens, trans_b: bool, g) -> Dict[str, torch.Tensor]:
    a, b = fp32_leaves(a0, b0)
    out = grouped_gemm_ref(a, b, group_lens, trans_b)
    out.backward(g.float())
    return {"out": out.detach(), "dgrad": a.grad, "wgrad": b.grad}


# Fused grad-accum (beta=1) wgrad into main_grad, all NT with bf16 operands.
ACCUM_DENSE_SHAPE = (2048, 4096, 1536)  # (M, N, K)
ACCUM_GROUPED_SHAPE = (8, 512, 2048, 1536)  # (B, M, N, K)


def check_fused_grad_accum(nodeid, seed, call, shape, main_grad_dtype, snr_floor, microbatches) -> None:
    """main_grad = previous + sum of each microbatch's wgrad, written by the beta=1 epilogue.

    ``call(a, b, group_lens)`` runs the op with ``fuse_bgrad_accum_pattern="megatron"``; a
    4-tuple ``shape`` makes it grouped. Each run starts from the same non-zero ``previous``;
    the microbatches use different activations and output gradients but the same weight
    leaf, as in pipeline training.
    """
    grouped = len(shape) == 4
    gen = make_generator(seed)
    if grouped:
        B, M, N, K = shape
        group_lens = unbalanced_group_lens(B, M, seed)
        rows, b_shape = B * M, (B, N, K)
    else:
        M, N, K = shape
        group_lens = None
        rows, b_shape = M, (N, K)
    b0 = randn(b_shape, torch.bfloat16, gen, normalize=True)
    a0s = [randn((rows, K), torch.bfloat16, gen, normalize=True) for _ in range(microbatches)]
    gs = [randn((rows, N), torch.bfloat16, gen) for _ in range(microbatches)]
    previous = randn(b_shape, main_grad_dtype, gen)

    def run_once(mid):
        b = b0.clone().requires_grad_(True)
        b.main_grad = previous.clone()
        b.grad_added_to_main_grad = False
        res = {}
        for i in range(microbatches):
            a = a0s[i].clone().requires_grad_(True)
            out = call(a, b, group_lens)
            mid()
            out.backward(gs[i])
            res[f"out{i}"], res[f"dgrad{i}"] = out, a.grad
        assert b.grad_added_to_main_grad is True, "weight must be flagged during forward"
        assert b.grad.shape == b.shape and b.grad.dtype == b.dtype, "dummy wgrad must match the weight"
        res["main_grad"] = b.main_grad
        return res

    ref, wgrad_sum = {}, torch.zeros(b_shape, dtype=torch.float32, device="cuda")
    for i in range(microbatches):
        if grouped:
            r = grouped_ref(a0s[i], b0, group_lens, True, gs[i])
        else:
            r = dense_ref(a0s[i], b0, gs[i])
        ref[f"out{i}"], ref[f"dgrad{i}"] = r["out"], r["dgrad"]
        wgrad_sum += r["wgrad"]
    ref["main_grad_delta"] = wgrad_sum
    views = {"main_grad_delta": lambda res: res["main_grad"].float() - previous.float()}

    check_determinism(nodeid, run_once, ref, snr_floor=snr_floor, snr_views=views)


# GLU: gated activation, per-expert fp32 reference and the grouped MLP harness.
_GATES = {"silu": F.silu, "gelu": lambda t: F.gelu(t, approximate="tanh")}


def glu(l1, activation: str, clamp_limit: Optional[float]) -> torch.Tensor:
    gate, up = torch.chunk(l1, 2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    return _GATES[activation](gate) * up


def grouped_mlp_ref(x0, w10, w20, probs0, group_lens, activation, clamp_limit, cotangent):
    x, w1, w2, probs = fp32_leaves(x0, w10, w20, probs0)
    offs = [0] + torch.cumsum(group_lens, 0).tolist()
    outs = []
    for e in range(w1.shape[0]):
        lo, hi = offs[e], offs[e + 1]
        act = glu(x[lo:hi] @ w1[e].t(), activation, clamp_limit) * probs[lo:hi, None]
        outs.append(act @ w2[e].t())
    out = torch.cat(outs)
    out.backward(cotangent.float())
    return {
        "out": out.detach(),
        "grad_x": x.grad,
        "grad_w1": w1.grad,
        "grad_w2": w2.grad,
        "grad_probs": probs.grad,
    }


def check_grouped_mlp(nodeid, fn, x0, w10, w20, probs0, cotangent, ref, snr_floor) -> None:
    """``fn(x, w1, w2, probs)`` runs the fused grouped MLP; output and all four gradients."""

    def run_once(mid):
        x, w1, w2, probs = [t.clone().requires_grad_(True) for t in (x0, w10, w20, probs0)]
        out = fn(x, w1, w2, probs)
        mid()
        out.backward(cotangent.to(out.dtype))
        return {
            "out": out,
            "grad_x": x.grad,
            "grad_w1": w1.grad,
            "grad_w2": w2.grad,
            "grad_probs": probs.grad,
        }

    check_determinism(nodeid, run_once, ref, snr_floor=snr_floor)
