###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Adapted from AITER (https://github.com/ROCm/aiter, MIT License),
# aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py and common.py.
# Modified by the Primus-Turbo team; see NOTICE.md in this directory.
###############################################################################

"""Runtime utilities and shared ``@triton.jit`` helpers for the flex-attention kernels:
GPU architecture detection, global configuration flags, the LDS budget, block-sparse
knob selection, and the aux-tensor / score-gradient hook dispatch."""

import functools
import json
import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

AutotuneMode = Literal["off", "on", "sweep"]

__all__ = [
    "AUTOTUNE",
    "DEBUG",
    "USE_EXP2",
    # Global config
    "AutotuneMode",
    # Runtime info
    "get_arch",
    "aux_kernel_args",
    "mask_aux_kernel_args",
    "bs_tensor_strides",
    "pick_knobs",
    "time_launch",
]


# -------------------------------
# GPU Architecture
# -------------------------------
ArchFamily = Literal["cdna", "rdna"]

CDNA_ARCHS = frozenset({"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"})
RDNA_ARCHS = frozenset(
    {
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)


@dataclass(frozen=True)
class GpuArch:
    """GPU architecture information."""

    name: str  # e.g., "gfx942", "gfx1100"
    family: ArchFamily | None = None

    @property
    def is_rdna(self) -> bool:
        return self.family == "rdna"

    @property
    def cu_count(self) -> int:
        """Get the number of compute units on the current GPU."""
        return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


# -------------------------------
# Global Variables
# -------------------------------
# --- MLA / large head_dim: LDS budget -------------------------------------------
# CDNA3 (gfx942) gives 64 KiB of LDS per workgroup. The tuned configs assume
# head_dim <= 256, where the Q tile always fits; MLA shapes (head_dim_v up to 512,
# head_dim_qk up to 512) overflow it. Measured on MI300X, Triton's reported LDS
# requirement for attn_fwd is exactly BLOCK_M * padded_head_dim_qk * elem_size
# (131072 B for BLOCK_M=128, padded 512, bf16 -- and halving BLOCK_M made it fit),
# and for the fused backward it is the same product with BLOCK_N1 / BLOCK_M2 in
# place of BLOCK_M. So model the budget on that single dominant tile.
LDS_LIMIT_BYTES = 64 * 1024


def max_block_for_lds(padded_head_dim: int, elem_size: int) -> int:
    """Largest power-of-two sequence-block that keeps the dominant tile inside LDS.

    Returns 0 when even a block of 16 cannot fit, which the callers surface as a
    clear "head_dim too large" error rather than a raw Triton OutOfResources.
    """
    raw = LDS_LIMIT_BYTES // max(padded_head_dim * elem_size, 1)
    block = 1 << (raw.bit_length() - 1) if raw >= 1 else 0  # round down to pow2
    return block if block >= 16 else 0


AUTOTUNE: AutotuneMode = (
    "on"
    if os.environ.get("PRIMUS_TURBO_FLEX_ATTENTION_AUTOTUNE", "1").lower() in ("1", "true", "yes", "on")
    else "off"
)

# User override config json for attn_fwd.
# Note: Ignored if PRIMUS_TURBO_FLEX_ATTENTION_AUTOTUNE is enabled.
#
# e.g. PRIMUS_TURBO_FLEX_ATTENTION_FWD_CONFIG_JSON='{"BLOCK_M":32,"BLOCK_N":32,"waves_per_eu":1,"PRE_LOAD_V":false,"num_stages":1,"num_warps":4}'
FWD_CONF_OVERRIDE = None
try:
    conf_json = os.getenv("PRIMUS_TURBO_FLEX_ATTENTION_FWD_CONFIG_JSON")
    if conf_json:
        conf = json.loads(conf_json)
        FWD_CONF_OVERRIDE = triton.Config(
            conf,
            num_stages=conf.pop("num_stages", 1),
            num_warps=conf.pop("num_warps", 4),
        )
except Exception as e:  # noqa: BLE001
    logger.warning(f"PRIMUS_TURBO_FLEX_ATTENTION_FWD_CONFIG_JSON parse error: {e}")

# Unified debug level:
#   0 = off (default)
#   1 = basic debug info (shapes, tensor stats, kernel params)
#   2 = detailed debug (includes Triton interpreter prints in kernels)
#
# Set via: PRIMUS_TURBO_FLEX_ATTENTION_DEBUG=0|1|2
DEBUG: int = int(os.environ.get("PRIMUS_TURBO_FLEX_ATTENTION_DEBUG", "0"))
# Printing every autotune result is debug output; it must not be forced on for the whole process.
if DEBUG > 0:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
if DEBUG >= 2:
    os.environ["TRITON_INTERPRET"] = "1"
USE_EXP2 = True


# -------------------------------
# Runtime info
# -------------------------------
@functools.cache
def get_arch() -> GpuArch:
    """Get the current GPU architecture."""
    try:
        name: str = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        # No GPU available (e.g. import-only on Windows/CPU)
        return GpuArch(name="unknown")
    if name in CDNA_ARCHS:
        return GpuArch(name=name, family="cdna")
    elif name in RDNA_ARCHS:
        return GpuArch(name=name, family="rdna")
    else:
        return GpuArch(name=name)


# -------------------------------
# Block-sparse knob selection
# -------------------------------
# Block-sparse pins the kernel tiles to the sparsity granularity, so it cannot use
# triton.autotune -- that would pick its own tiles. The remaining knobs (waves_per_eu
# and friends) are still free, and the best value is strongly per-shape: measured across
# block size, sequence length, head dim and architecture, waves_per_eu swings results by
# up to 2x in *either* direction, and the winner at block 64 is the loser at block 128.
#
# Wrapping the kernel in a second autotuner does work, but costs 8-19 us on every launch
# (triton short-circuits to a lean path only when it holds a single config; with several
# it re-extracts its key and looks up its cache each time). On a 75 us kernel that
# overhead exceeds the win. So: choose once per shape, cache the winner, and let every
# later launch go down the normal lean path.


def time_launch(launch, warmup: int = 3, reps: int = 7) -> float:
    """Median wall time of ``launch()`` in milliseconds."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def pick_knobs(cache: dict, key, candidates, launch):
    """Return the fastest candidate for ``key``, measuring once and caching the winner.

    NOT YET VERIFIED TO PAY OFF. Correctness is established (the full suite passes with
    it active on gfx942, and selection is one-time: one measurement over 50 calls). What
    is missing is evidence that it is *faster* anywhere. On gfx942 it measures neutral,
    0.94-1.01x over six shapes, which is expected there -- that architecture has little
    headroom, and the forward simply re-picks the default, so both arms run identical
    kernels. The case with real headroom is the gfx950 block-64 backward, where the
    config this selects is worth 1.18-1.32x and no free rule can capture it (the same
    setting is 1.41x slower at head_dim 128). That run is outstanding.
    To settle it: bench/bench_pick_knobs.py on an MI355X node, against the AUTOTUNE=off
    arm. If gfx950 does not show the win, drop this -- it would be the fourth mechanism
    in this area whose overhead exceeded what it recovered.

    ``launch(candidate)`` must run the kernel; it is called repeatedly during selection,
    which is safe because these kernels *store* their outputs rather than accumulating
    into them. A candidate that fails to compile for this shape is skipped rather than
    raising, so an unusable block size just does not get chosen.
    """
    hit = cache.get(key)
    if hit is not None:
        return hit
    best, best_ms = None, float("inf")
    for cand in candidates:
        try:
            ms = time_launch(lambda c=cand: launch(c))
        except Exception:  # noqa: BLE001 - an unusable config is simply not selected
            continue
        if ms < best_ms:
            best, best_ms = cand, ms
    if best is None:
        best = candidates[0]  # nothing measured; fall back and let the real launch raise
    cache[key] = best
    return best


def bs_tensor_strides(prefix: str, tensor) -> dict:
    """Kwargs for one block-sparse list's (batch, head, row) strides.

    All the launchers pass block-sparse count/index tensors to the kernel as a flat
    pointer plus separate stride_*_b/_h/_m kwargs (Triton kernels take plain pointers,
    not tensor objects, so the strides have to travel alongside). ``tensor`` is one of
    those count/index tensors, or None when this call has no block-sparse list of this
    kind -- the kernel then never dereferences the corresponding pointer, so the actual
    stride values don't matter, only that they're present and valid ints.
    """
    if tensor is None:
        return {f"stride_{prefix}_b": 0, f"stride_{prefix}_h": 0, f"stride_{prefix}_m": 0}
    return {
        f"stride_{prefix}_b": tensor.stride(0),
        f"stride_{prefix}_h": tensor.stride(1),
        f"stride_{prefix}_m": tensor.stride(2),
    }


MAX_AUX_TENSORS = 4


def aux_kernel_args(aux_tensors, prefix: str = "AUX", count: str = "NUM_AUX") -> dict:
    """Kwargs passing up to MAX_AUX_TENSORS aux tensors to a kernel.

    Unused slots stay None, so the kernel specializes them away; ``count`` tells
    apply_score_mod / apply_mask_mod how many to hand to the mod. The score_mod's
    tensors use the default names, the mask_mod's ``MASK_AUX*`` / ``NUM_MASK_AUX``.
    """
    aux = list(aux_tensors or ())
    if len(aux) > MAX_AUX_TENSORS:
        raise ValueError(f"at most {MAX_AUX_TENSORS} aux_tensors are supported, got {len(aux)}")
    slots = aux + [None] * (MAX_AUX_TENSORS - len(aux))
    return {**{f"{prefix}{i}": t for i, t in enumerate(slots)}, count: len(aux)}


def mask_aux_kernel_args(aux_tensors) -> dict:
    return aux_kernel_args(aux_tensors, "MASK_AUX", "NUM_MASK_AUX")


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    return pid


@triton.jit
def apply_mask_mod(
    MASK_MOD: tl.constexpr,
    b,
    h,
    q_idx,
    kv_idx,
    MASK_AUX0,
    MASK_AUX1,
    MASK_AUX2,
    MASK_AUX3,
    NUM_MASK_AUX: tl.constexpr,
):
    """Call ``MASK_MOD`` with the first ``NUM_MASK_AUX`` aux tensor pointers appended."""
    if NUM_MASK_AUX == 0:
        keep = MASK_MOD(b, h, q_idx, kv_idx)
    elif NUM_MASK_AUX == 1:
        keep = MASK_MOD(b, h, q_idx, kv_idx, MASK_AUX0)
    elif NUM_MASK_AUX == 2:
        keep = MASK_MOD(b, h, q_idx, kv_idx, MASK_AUX0, MASK_AUX1)
    elif NUM_MASK_AUX == 3:
        keep = MASK_MOD(b, h, q_idx, kv_idx, MASK_AUX0, MASK_AUX1, MASK_AUX2)
    else:
        keep = MASK_MOD(b, h, q_idx, kv_idx, MASK_AUX0, MASK_AUX1, MASK_AUX2, MASK_AUX3)
    return keep


@triton.jit
def apply_score_mod(
    SCORE_MOD: tl.constexpr,
    score,
    b,
    h,
    q_idx,
    kv_idx,
    AUX0,
    AUX1,
    AUX2,
    AUX3,
    NUM_AUX: tl.constexpr,
):
    """Call ``SCORE_MOD`` with the first ``NUM_AUX`` aux tensor pointers appended."""
    if NUM_AUX == 0:
        out = SCORE_MOD(score, b, h, q_idx, kv_idx)
    elif NUM_AUX == 1:
        out = SCORE_MOD(score, b, h, q_idx, kv_idx, AUX0)
    elif NUM_AUX == 2:
        out = SCORE_MOD(score, b, h, q_idx, kv_idx, AUX0, AUX1)
    elif NUM_AUX == 3:
        out = SCORE_MOD(score, b, h, q_idx, kv_idx, AUX0, AUX1, AUX2)
    else:
        out = SCORE_MOD(score, b, h, q_idx, kv_idx, AUX0, AUX1, AUX2, AUX3)
    return out


@triton.jit
def apply_score_grad_hook(
    SCORE_GRAD_HOOK: tl.constexpr,
    dscore,
    valid,
    b,
    h,
    q_idx,
    kv_idx,
    GRAD,
    AUX0,
    AUX1,
    AUX2,
    AUX3,
    NUM_AUX: tl.constexpr,
):
    """Call ``SCORE_GRAD_HOOK`` with its output buffer and the first ``NUM_AUX`` aux pointers."""
    if NUM_AUX == 0:
        SCORE_GRAD_HOOK(dscore, valid, b, h, q_idx, kv_idx, GRAD)
    elif NUM_AUX == 1:
        SCORE_GRAD_HOOK(dscore, valid, b, h, q_idx, kv_idx, GRAD, AUX0)
    elif NUM_AUX == 2:
        SCORE_GRAD_HOOK(dscore, valid, b, h, q_idx, kv_idx, GRAD, AUX0, AUX1)
    elif NUM_AUX == 3:
        SCORE_GRAD_HOOK(dscore, valid, b, h, q_idx, kv_idx, GRAD, AUX0, AUX1, AUX2)
    else:
        SCORE_GRAD_HOOK(dscore, valid, b, h, q_idx, kv_idx, GRAD, AUX0, AUX1, AUX2, AUX3)
