###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 HipKittens Project Contributors
#
# Adapted from HipKittens (https://github.com/HazyResearch/HipKittens).
# Modified by the Primus-Turbo team.
###############################################################################

"""HipKittens attention.

gfx950 (MI355X): dense flash attention, forward and backward, compiled into ``primus_turbo.pytorch._C`` from
``csrc/kernels/attention/hipkittens`` (``hipkittens_attn_supported_impl``, ``flash_attn_sbhd_hipkittens_*_impl``).

gfx1250 (MI455X): DeepSeek-V3 MLA attention (head dims qk 192 / v 128), forward and backward (``hk_mla_fwd``,
``hk_mla_bwd``). The kernels are code objects built from ``csrc/kernels/attention/hipkittens/gfx1250_mla`` into
``primus_turbo/lib/hk_gfx1250_mla`` (see the README there) and launched through the HIP runtime torch loaded.
``attention_flydsl_gfx1250_mla_impl`` picks them per pass in place of the FlyDSL MLA kernels.
"""

import ctypes
import functools
import hashlib
import math
import os
import struct
import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.utils import is_gfx950

__all__ = [
    "hipkittens_attn_supported_impl",
    "flash_attn_sbhd_hipkittens_forward_impl",
    "flash_attn_sbhd_hipkittens_backward_impl",
    "hk_mla_unsupported_reason",
    "hk_mla_fwd",
    "hk_mla_bwd",
]

# =============================================================================
# gfx950: dense attention
#
# The kernels admit only what hipkittens_attn_supported_impl accepts: bf16, contiguous SBHD [s, b, h, d], head dim
# 64 or 128, bottom-right causal with an optional left window, Sq <= Skv, no sink, varlen, dropout, bias or alibi.
# Outside that they read out of bounds or leave output unwritten instead of failing, so the checks are load-bearing.
# =============================================================================

SUPPORTED_HEAD_DIMS = (64, 128)

# A buffer descriptor's num_records is 32-bit, so a global tensor at or above 4 GiB wraps -- and at an exact multiple
# it wraps to ZERO, silently discarding every store. This bounds the split-K workspaces below.
_WS_LIMIT = 2**32

# The backward always takes the split dq + dkdv pair: the fused single-pass variant in the kernels is 2.2x to 3.6x
# slower wherever its workspace fits, and silently wrong (_WS_LIMIT) where it does not.

# The launcher has one `switch` case per split and head dim; a split it has no case for runs the NSPLIT == 1 kernel
# against a grid built for n_split. Must stay in lockstep with that switch.
_DKDV_SPLIT_VALUES = {64: (1, 2, 4, 8), 128: (1, 2, 4, 5, 8)}


def _ops():
    return torch.ops.primus_turbo_cpp_extension


@functools.lru_cache(maxsize=None)
def _ops_built() -> bool:
    """Whether this build carries the kernels: they are named *_gfx950.cu, so a build whose offload archs exclude
    gfx950 never registers the ops (is_gfx950() reads the card, not the build)."""
    return hasattr(_ops(), "hk_attn_fwd_d64")


@functools.lru_cache(maxsize=None)
def _blocks(head_dim: int) -> dict:
    """The tile sizes the kernels were compiled with; the sequence axes are padded to these."""
    b = _ops().hk_attn_block_sizes(head_dim)
    # Positional (the kernel decides the order); FUSED_* belong to the undispatched fused backward.
    keys = ("FWD_Q", "FWD_KV", "DQ_Q", "DQ_KV", "DKDV_Q", "DKDV_KV", "PREP_Q", "FUSED_Q", "FUSED_KV")
    return dict(zip(keys, (int(x) for x in b)))


def _q_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_Q"], b["DQ_Q"], b["DKDV_Q"], b["PREP_Q"])


def _kv_align(head_dim: int) -> int:
    b = _blocks(head_dim)
    return math.lcm(b["FWD_KV"], b["DQ_KV"], b["DKDV_KV"])


def hipkittens_attn_supported_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    sink: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """Whether the gfx950 kernels take SBHD ``q`` [Sq, B, Hq, D], ``k`` / ``v`` [Skv, B, Hkv, D], and if not, why."""
    if not (torch.cuda.is_available() and is_gfx950()):
        return False, "hipkittens attention is gfx950-only"
    if not _ops_built():
        return False, "hipkittens attention was not built (gfx950 is not among the offload archs)"
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        # The kernels declare gl<bf16, ...>: another 16-bit dtype would be reinterpreted, not converted.
        return False, f"hipkittens attention is bf16-only, got {q.dtype}"
    if not causal:
        return False, "hipkittens attention implements causal masking only"
    if sink is not None:
        return False, "hipkittens attention does not implement a learned sink"
    if dropout_p != 0.0:
        return False, "hipkittens attention does not implement dropout"
    if bias is not None or alibi_slopes is not None:
        return False, "hipkittens attention does not implement bias / alibi"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False, "hipkittens attention takes dense 4-D SBHD tensors, not varlen/THD packing"
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False, "hipkittens attention needs contiguous [s, b, h, d]; it derives strides from the shape"
    if len(window_size) > 1 and int(window_size[1]) not in (0, -1):
        return False, "hipkittens attention implements a left window only"

    Sq, B, Hq, D = q.shape
    Skv, Bk, Hkv, Dk = k.shape
    if (B, D) != (Bk, Dk) or v.shape != k.shape:
        return False, f"shape mismatch q={tuple(q.shape)} k={tuple(k.shape)}"
    if D not in SUPPORTED_HEAD_DIMS:
        return False, f"hipkittens attention supports head dim {SUPPORTED_HEAD_DIMS}, got {D}"
    if Hq % Hkv != 0:
        return False, f"Hq={Hq} must be a multiple of Hkv={Hkv}"
    if Sq > Skv:
        # The leading Sq - Skv query rows attend to no key: the forward returns early for them and leaves those
        # output rows uninitialized, so the shape is refused on both passes.
        return False, f"hipkittens attention does not support Sq > Skv yet (Sq={Sq}, Skv={Skv})"
    return True, ""


def _require_supported(q, k, v, *, causal, window_size) -> None:
    ok, why = hipkittens_attn_supported_impl(q, k, v, causal=causal, window_size=window_size)
    if not ok:
        raise NotImplementedError(why)


def _pad_seq(t: torch.Tensor, target: int) -> torch.Tensor:
    if t.shape[0] == target:
        return t
    out = t.new_zeros((target, *t.shape[1:]))
    out[: t.shape[0]] = t
    return out


def flash_attn_sbhd_hipkittens_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    return_lse: bool = True,
):
    """SBHD forward. Returns ``(out, lse)``, lse ``[B, Hq, 1, Sq]`` natural log, or just ``out``."""
    _require_supported(q, k, v, causal=causal, window_size=window_size)
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    window_left = int(window_size[0])
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    # The kernels read a ragged tail tile as unspecified bytes, which the MFMA turns into NaN even where the mask has
    # zeroed P, so ragged lengths are zero-padded; Sq / Skv stay the true lengths in the launch, so the mask still
    # excludes every padded position.
    b = _blocks(D)
    Sq_pad = -(-Sq // b["FWD_Q"]) * b["FWD_Q"]
    Skv_pad = -(-Skv // b["FWD_KV"]) * b["FWD_KV"]
    q_k, k_k, v_k = _pad_seq(q, Sq_pad), _pad_seq(k, Skv_pad), _pad_seq(v, Skv_pad)

    out = torch.empty_like(q_k)
    # LSE is written a whole Q tile at a time: a ragged Sq would spill into the next (b, h) slice.
    lse_pad = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    op = _ops().hk_attn_fwd_d64 if D == 64 else _ops().hk_attn_fwd_d128
    op(q_k, k_k, v_k, out, lse_pad, Sq, Skv, B, Hq, Hkv, window_left, float(softmax_scale))
    out, lse = out[:Sq], lse_pad[..., :Sq]
    return (out, lse) if return_lse else out


@functools.lru_cache(maxsize=None)
def _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad) -> int:
    """How many ways to split the GQA head group for dK/dV: the kernel's selector, bounded here because this layer
    allocates the partials (a disagreement would read past their end)."""
    n = int(_ops().hk_attn_dkdv_head_split(D, Sq, Skv, B, Hq, Hkv, window_left))
    if n not in _DKDV_SPLIT_VALUES[D]:
        return 1
    while n > 1 and (n * Skv_pad * B * Hkv * D * 2) >= _WS_LIMIT:
        n //= 2
    return n


def flash_attn_sbhd_hipkittens_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SBHD backward: ``dout/q/out`` [Sq, B, Hq, D], ``k/v`` [Skv, B, Hkv, D] bf16, ``lse`` the forward's
    ``[B, Hq, 1, Sq]`` (or ``[B, Hq, Sq]``) fp32. Returns ``(dq, dk, dv)`` in the layouts of q, k and v."""
    _require_supported(q, k, v, causal=causal, window_size=window_size)
    if dout.dtype is not torch.bfloat16 or out.dtype is not torch.bfloat16:
        raise NotImplementedError("hipkittens attention is bf16-only")
    if dout.shape != q.shape or out.shape != q.shape:
        raise ValueError("hipkittens attention backward: dout/out must have q's shape")
    for name, t in (("dout", dout), ("out", out)):
        if not t.is_contiguous():
            raise ValueError(f"hipkittens attention backward: {name} must be contiguous")

    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    window_left = int(window_size[0])
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    Sq_pad = -(-Sq // _q_align(D)) * _q_align(D)
    Skv_pad = -(-Skv // _kv_align(D)) * _kv_align(D)
    q_k, k_k, v_k = _pad_seq(q, Sq_pad), _pad_seq(k, Skv_pad), _pad_seq(v, Skv_pad)
    o_k, do_k = _pad_seq(out, Sq_pad), _pad_seq(dout, Sq_pad)

    lse_flat = lse.reshape(B, Hq, 1, -1)
    if lse_flat.shape[-1] != Sq:
        raise ValueError(f"hipkittens attention backward: lse last axis {lse_flat.shape[-1]} != Sq {Sq}")
    lse_k = torch.zeros((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)
    lse_k[..., :Sq] = lse_flat

    dq = torch.empty_like(q_k)
    dk = torch.empty_like(k_k)
    dv = torch.empty_like(v_k)
    # Prep writes delta and lneg over exactly [0, Sq_pad); a padded row carries delta = 0 and lse = 0 and adds exactly
    # zero to dK/dV.
    delta = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)
    lneg = torch.empty((B, Hq, 1, Sq_pad), device=q.device, dtype=torch.float32)

    n_split = _dkdv_split(Sq, Skv, B, Hq, Hkv, window_left, D, Skv_pad)
    if n_split > 1:
        wsk = torch.empty((n_split * Skv_pad, B, Hkv, D), device=q.device, dtype=k_k.dtype)
        wsv = torch.empty_like(wsk)
    else:
        wsk = wsv = dk  # never dereferenced at n_split == 1; the op schema takes tensors, not optionals

    op = _ops().hk_attn_bwd_d64 if D == 64 else _ops().hk_attn_bwd_d128
    op(
        q_k,
        k_k,
        v_k,
        o_k,
        do_k,
        dq,
        dk,
        dv,
        lse_k,
        delta,
        lneg,
        wsk,
        wsv,
        Sq,
        Skv,
        B,
        Hq,
        Hkv,
        window_left,
        float(softmax_scale),
        n_split,
    )
    return dq[:Sq], dk[:Skv], dv[:Skv]


# =============================================================================
# gfx1250: DeepSeek-V3 MLA attention (qk 192 / v 128, bf16, MHA, bottom-right causal)
#
# Forward k_mla_fwd: 256-query workgroups (4 waves) over a 4-stage TDM ring of 64-row K / V tiles, one q block per
# workgroup longest first, or one persistent workgroup per CU when the static schedule covers the shape. It runs the
# FlyDSL forward's arithmetic in its order: out and lse are bitwise equal to FlyDSL's.
# Backward: k_delta (delta = rowsum(dO * O) and the row constants of k_dkdv96ds), k_dkdv96ds (dK / dV over 96-row kv
# blocks, then 64-row tails, storing every bf16 dS^T tile into a scratch; workgroups take blocks from a ticket counter,
# longest first) and k_dqds_c (dQ = dS K from that scratch). Every output element is written by one workgroup, so
# both passes are deterministic.
# =============================================================================

MLA_D_QK, MLA_D_V = 192, 128
_MLA_FWD_ROWS = 256  # k_mla_fwd queries per workgroup
_MLA_XCDS = 8  # persistent forward: head-batch groups, one per XCD under the round-robin dispatch
_MLA_DKDV_ROWS = 96  # k_dkdv96ds kv rows per workgroup
_MLA_DQ_ROWS, _MLA_DQ_HEAD_ROWS = 384, 256  # k_dqds_c queries per workgroup, and of its merged head tile
_MLA_HEAD_GROUP = 32  # head-group target of the k_dkdv96ds / k_dqds_c grids
_MLA_DS_GUARD = 65536  # bytes before and after the dS layout; the ticket counter ends the front guard
_MLA_MAX_BH_S = 512 * 4096  # largest batch * heads * seqlen the backward's index arithmetic is verified for
_MLA_MIN_BWD_SCALE = 2.0**-100  # the backward accumulates dV from bf16(scale * P)
# kernel: (workgroup size, packed arguments, sha256 of the code object), the signatures and pins of
# csrc/.../gfx1250_mla/manifest.json (a test keeps them equal); only a code object with its pin is loaded:
#   k_mla_fwd   Q K V O LSE, scale, Sq Skv H B nqb npw
#   k_delta     DO O DELTA LSE NL ND, log2(scale), S H
#   k_dkdv96ds  Q K V DO NL ND DV DK DS, scale, Sq Skv Hq Hkv G nqt cshift causal B HG ds_mode
#   k_dqds_c    DS K DQ, Sq Skv Hq Hkv G nkvt cshift causal q_off ntile HG
_MLA_KERNELS = {
    "k_mla_fwd": (128, "<5Qf6i", "c29ac43ff14eed04d201fb458c0064ce5771ed25678e262524e4a48dfe364553"),
    "k_delta": (256, "<6Qf2i", "61d3bc136087a0160a26ae139f0701cc74d8775091d972ff8c15651473dfff89"),
    "k_dkdv96ds": (64, "<9Qf11i", "abeea38ec402459872df2c8640f5249e817d9df31d5b40326903426d20bbffbe"),
    "k_dqds_c": (128, "<3Q11i", "c182284e1ed7d0c1530e24f433ce719cdb950aea2fb7fd2020149e4921051b67"),
}
_MLA_PASS_KERNELS = {"fwd": ("k_mla_fwd",), "bwd": ("k_delta", "k_dkdv96ds", "k_dqds_c")}
_MLA_HSACO_DIR = Path(__file__).resolve().parents[3] / "lib" / "hk_gfx1250_mla"


def _mla_scale(softmax_scale: Optional[float]) -> float:
    return MLA_D_QK**-0.5 if softmax_scale is None else float(softmax_scale)


def _mla_dkdv_blocks(skv: int) -> int:
    """k_dkdv96ds kv blocks for a multiple of 32 above 96: 96-row ones, then one 64-row tail for skv % 96 == 64 or
    two for 32 (the last 128 rows)."""
    return -(-skv // _MLA_DKDV_ROWS)


def _mla_dq_split(sq: int) -> Tuple[int, int, int]:
    """(q_off, nhead, ntile): 384-query tiles over [q_off, sq), one merged 256-query head tile over [0, q_off)."""
    q_off = sq % _MLA_DQ_ROWS
    return q_off, int(q_off > 0), (sq - q_off) // _MLA_DQ_ROWS


def _mla_ds_bytes(b: int, sq: int, skv: int, h: int, causal: bool) -> int:
    """Size of the dS^T layout (csrc/.../include/dslayout.h): b * h regions of 1 KiB blocks of 16 kv rows x 32
    queries, without the blocks the bottom-right causal mask leaves empty."""
    nqp, nkt, m = sq // 32, skv // 16, (skv - sq) // 32

    def tri(n):
        return (n // 2) * (n // 2 - 1) + (n % 2) * (n // 2)

    skip = 0
    if causal:
        skip = tri(nkt) - m * nkt if m <= 0 else (tri(nkt - 2 * m) if nkt > 2 * m else 0)
    return b * h * (nkt * nqp - skip) * 1024


def _mla_shape_reason(pass_: str, b: int, sq: int, skv: int, h: int, hkv: int, causal: bool, scale: float):
    """Why the HIP MLA kernels of ``pass_`` cannot take this problem, beyond the FlyDSL MLA gate's rules, or None."""
    if h != hkv:
        return f"MHA only (heads_q == heads_kv), got {h} / {hkv}"
    if not causal:
        return "causal only"
    if skv < sq:
        return f"seqlen_kv {skv} < seqlen_q {sq}"
    if pass_ == "fwd":
        if sq != skv:
            return f"seqlen_q must equal seqlen_kv, got {sq} / {skv}"
        if sq % 64:
            return f"seqlen must be a multiple of 64, got {sq}"
        if b * sq * h * MLA_D_QK * 2 >= 1 << 31:
            return "q / k exceed the 32-bit buffer offsets"
        return None
    if sq % 32 or skv % 32:
        return f"seqlen_q / seqlen_kv must be multiples of 32, got {sq} / {skv}"
    if sq % _MLA_DQ_ROWS not in (0, _MLA_DQ_HEAD_ROWS):
        return f"seqlen_q % {_MLA_DQ_ROWS} must be 0 or {_MLA_DQ_HEAD_ROWS}, got {sq}"
    if b * h * skv > _MLA_MAX_BH_S:
        return f"batch * heads * seqlen {b * h * skv} exceeds the verified {_MLA_MAX_BH_S}"
    # Comparisons only (NaN fails), so torch.compile can trace a symbolic scale.
    if not _MLA_MIN_BWD_SCALE <= scale < math.inf:
        return f"softmax_scale must be finite and at least {_MLA_MIN_BWD_SCALE}, got {scale}"
    return None


@functools.lru_cache(maxsize=None)
def _hip_runtime() -> ctypes.CDLL:
    """The libamdhip64 torch loaded, typed for the module API: launches through another copy would run in another
    HIP runtime, outside torch's contexts and streams."""
    with open("/proc/self/maps") as f:
        paths = {line[line.index("/") :].strip() for line in f if "/libamdhip64.so" in line}
    if len({os.path.realpath(p) for p in paths}) != 1:
        raise RuntimeError(f"expected one libamdhip64 mapped by torch, found {sorted(paths)}")
    hip = ctypes.CDLL(sorted(paths)[0], mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    p, u = ctypes.c_void_p, ctypes.c_uint
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(p), p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(p), p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [p, u, u, u, u, u, u, u, p, ctypes.POINTER(p), ctypes.POINTER(p)]
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    return hip


def _hip_check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {_hip_runtime().hipGetErrorString(rc).decode()} ({rc})")


class _HipKernel:
    """The pinned code object of a kernel in _MLA_HSACO_DIR, loaded once per device, launched on a torch stream."""

    _lock = threading.Lock()

    def __init__(self, name: str):
        self.name = name
        self.block, layout, pin = _MLA_KERNELS[name]
        path = _MLA_HSACO_DIR / f"{name}.hsaco"
        if not path.is_file():
            raise RuntimeError(
                f"{path} is not built (GPU_ARCHS=gfx1250 and the toolchain manifest.json pins)"
            )
        image = path.read_bytes()
        if hashlib.sha256(image).hexdigest() != pin:
            raise RuntimeError(f"{path} is not the pinned code object: stale or foreign, rebuild it")
        self.args = struct.Struct(layout)
        self.image = ctypes.create_string_buffer(image)
        self.functions = {}  # device index -> hipFunction_t

    def load(self, device: int) -> ctypes.c_void_p:
        with self._lock:
            if device not in self.functions:
                module, fn = ctypes.c_void_p(), ctypes.c_void_p()
                hip = _hip_runtime()
                _hip_check(hip.hipModuleLoadData(ctypes.byref(module), self.image), f"loading {self.name}")
                _hip_check(hip.hipModuleGetFunction(ctypes.byref(fn), module, self.name.encode()), self.name)
                self.functions[device] = fn
            return self.functions[device]

    def __call__(self, grid: Tuple[int, int, int], values: tuple, stream: torch.cuda.Stream) -> None:
        fn = self.functions.get(stream.device_index)
        if fn is None:
            fn = self.load(stream.device_index)
        args = self.args.pack(*(x.data_ptr() if isinstance(x, torch.Tensor) else x for x in values))
        buf, size = ctypes.create_string_buffer(args, len(args)), ctypes.c_size_t(len(args))
        # HIP_LAUNCH_PARAM_BUFFER_POINTER, buf, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END
        extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(buf), 2, ctypes.addressof(size), 3)
        rc = _hip_runtime().hipModuleLaunchKernel(
            fn, *grid, self.block, 1, 1, 0, stream.cuda_stream, None, extra
        )
        _hip_check(rc, f"launching {self.name}")


_MLA_LOADED = {}  # kernel name -> _HipKernel


def _mla_kernel(name: str) -> _HipKernel:
    kernel = _MLA_LOADED.get(name)
    if kernel is None:
        kernel = _MLA_LOADED[name] = _HipKernel(name)
    return kernel


_MLA_UNAVAILABLE = {}


@torch.compiler.assume_constant_result  # the gate runs in compiled regions; the answer is fixed per process
def hk_mla_unavailable_reason(pass_: str) -> Optional[str]:
    """None when the code objects of ``pass_`` ("fwd" / "bwd") are the pinned ones and load on the current device,
    else why not."""
    if pass_ not in _MLA_UNAVAILABLE:
        reason = None
        try:
            for name in _MLA_PASS_KERNELS[pass_]:
                _mla_kernel(name).load(torch.cuda.current_device())
        except (OSError, RuntimeError) as e:
            reason = f"the HipKittens MLA {pass_} kernels are unavailable: {e}"
        _MLA_UNAVAILABLE[pass_] = reason
    return _MLA_UNAVAILABLE[pass_]


def hk_mla_unsupported_reason(
    pass_: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float] = None,
) -> Optional[str]:
    """Why ``hk_mla_fwd`` / ``hk_mla_bwd`` (``pass_`` "fwd" / "bwd") cannot take kernel-layout q / k / v that pass the
    FlyDSL MLA gate (``flydsl_gfx1250_mla_unsupported_reason``), or None."""
    b, sq, h, _ = q.shape
    _, skv, hkv, _ = k.shape
    reason = _mla_shape_reason(pass_, b, sq, skv, h, hkv, bool(causal), _mla_scale(softmax_scale))
    return reason if reason is not None else hk_mla_unavailable_reason(pass_)


def _mla_check(q, k, v, o=None, do=None, lse=None) -> Tuple[int, int, int, int, int]:
    """(b, sq, skv, hq, hkv) of contiguous [b, s, h, d] bf16 q / k / v (o / do) and [b, hq, sq] fp32 lse."""
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    want = [(q, (b, sq, hq, MLA_D_QK)), (k, (b, skv, hkv, MLA_D_QK)), (v, (b, skv, hkv, MLA_D_V))]
    if o is not None:
        want += [(o, (b, sq, hq, MLA_D_V)), (do, (b, sq, hq, MLA_D_V)), (lse, (b, hq, sq))]
    for t, shape in want:
        dtype = torch.float32 if t is lse else torch.bfloat16
        if t.shape != shape or t.dtype != dtype or t.device != q.device or not t.is_contiguous():
            raise ValueError(
                f"expected a contiguous {dtype} {shape} tensor on {q.device}, got {t.dtype} {tuple(t.shape)} on "
                f"{t.device}{'' if t.is_contiguous() else ' (strided)'}"
            )
    if min(b, sq, skv, hq) <= 0:
        raise ValueError(f"empty problem: q {tuple(q.shape)}, k {tuple(k.shape)}")
    return b, sq, skv, hq, hkv


@functools.lru_cache(maxsize=None)
def _cu_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def hk_mla_persistent_workgroups(b: int, s: int, h: int, cus: int) -> int:
    """Grid of the persistent forward for [b, s, h] on ``cus`` CUs (one workgroup each), or 0 for one q block per
    workgroup. Workgroup w serves head-batch group w % 8 and, in step t, head-batch 8 (R t + r) + w % 8 with q block
    i on even steps and nqb - 1 - i on odd ones (w / 8 = r nqb + i, R = heads per group and step), so every two
    steps cost each workgroup nqb + 1 q blocks. It covers the shape when every group gets a whole, even number of
    steps."""
    nqb, per_group, hb = -(-s // _MLA_FWD_ROWS), cus // _MLA_XCDS, b * h
    if cus % _MLA_XCDS or s % _MLA_FWD_ROWS or per_group % nqb or hb % _MLA_XCDS:
        return 0
    steps, rem = divmod(hb // _MLA_XCDS, per_group // nqb)
    return 0 if rem or steps % 2 else cus


def hk_mla_fwd(q, k, v, softmax_scale=None, causal=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """MLA forward: q / k [B, S, H, 192], v [B, S, H, 128], contiguous bf16 (``hk_mla_unsupported_reason``). Returns
    out [B, S, H, 128] bf16 and lse [B, H, S] fp32 (natural log), bitwise those of the FlyDSL forward."""
    b, s, skv, h, hkv = _mla_check(q, k, v)
    reason = _mla_shape_reason("fwd", b, s, skv, h, hkv, bool(causal), _mla_scale(softmax_scale))
    if reason is not None:
        raise NotImplementedError(f"hk_mla_fwd: {reason}")
    nqb = -(-s // _MLA_FWD_ROWS)
    npw = hk_mla_persistent_workgroups(b, s, h, _cu_count(q.device))
    out = torch.empty((b, s, h, MLA_D_V), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((b, h, s), device=q.device, dtype=torch.float32)
    values = (q, k, v, out, lse, _mla_scale(softmax_scale), s, s, h, b, nqb, npw)
    with torch.cuda.device(q.device):
        _mla_kernel("k_mla_fwd")((npw, 1, 1) if npw else (nqb, h, b), values, torch.cuda.current_stream())
    return out, lse


class _MlaScratchError(RuntimeError):
    pass


_MLA_SCRATCH_OOM_BYTES = math.inf  # the smallest dS scratch whose allocation failed


def _mla_scratch(nbytes: int, device: torch.device) -> torch.Tensor:
    """The dS scratch from the caching allocator, on the current stream. A failed allocation costs the allocator a
    synchronize and its whole cache, so after one, a scratch at least as large is refused without trying again."""
    global _MLA_SCRATCH_OOM_BYTES
    if nbytes < _MLA_SCRATCH_OOM_BYTES:
        try:
            return torch.empty((nbytes // 2,), device=device, dtype=torch.bfloat16)
        except torch.OutOfMemoryError:
            _MLA_SCRATCH_OOM_BYTES = nbytes
    raise _MlaScratchError(
        f"no memory for a {_MLA_SCRATCH_OOM_BYTES}-byte dS scratch, so none that large is tried again "
        "(PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES splits the call by heads instead)"
    )


def _mla_head_group(nh: int) -> int:
    """Heads per k_dkdv96ds / k_dqds_c grid group: nh when nh <= _MLA_HEAD_GROUP, else the largest multiple of 8 (one
    head's workgroups on one XCD) that divides nh and is at most _MLA_HEAD_GROUP, else nh."""
    if nh <= _MLA_HEAD_GROUP:
        return nh
    return next((d for d in range(_MLA_HEAD_GROUP, 7, -8) if nh % d == 0), nh)


def _hk_mla_bwd_plan(do, q, k, v, o, lse, scale, causal, max_scratch_bytes):
    """(launches [(kernel, grid, argument values)], scratch, (dq, dk, dv)) of hk_mla_bwd; works on meta tensors.

    The dS scratch is guard | layout | guard. Over ``max_scratch_bytes`` the dK / dV and dQ kernels run per chunk of
    heads of one batch (B = 1, the grid's z cut to the chunk, the tensors offset to its first (batch, head)), every
    chunk on the same scratch: the gradients are bitwise those of one call. _MlaScratchError when no chunk fits."""
    b, sq, skv, hq, hkv = _mla_check(q, k, v, o, do, lse)
    reason = _mla_shape_reason("bwd", b, sq, skv, hq, hkv, bool(causal), scale)
    if reason is not None:
        raise NotImplementedError(f"hk_mla_bwd: {reason}")
    c = int(bool(causal))
    hg_kv, hg_q = _mla_head_group(hkv), _mla_head_group(hq)

    def scratch_bytes(batches, heads):
        return _mla_ds_bytes(batches, sq, skv, heads, c) + 2 * _MLA_DS_GUARD

    heads = hq  # per (dK / dV, dQ) launch pair
    chunked = max_scratch_bytes is not None and scratch_bytes(b, hq) > max_scratch_bytes
    if chunked:
        step = math.lcm(hg_kv, hg_q)
        fits = [n for n in range(hq, 0, -step) if hq % n == 0 and scratch_bytes(1, n) <= max_scratch_bytes]
        if not fits:
            raise _MlaScratchError(f"the dS scratch of {step} heads exceeds {max_scratch_bytes} bytes")
        heads = fits[0]
    nbytes = scratch_bytes(1 if chunked else b, heads)
    scratch = _mla_scratch(nbytes, q.device)  # first: no other allocation when it does not fit
    delta, nl, nd = (torch.empty((b, hq, sq), device=q.device, dtype=torch.float32) for _ in range(3))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if not chunked:
        parts = [((q, k, v, do, nl, nd, dv, dk, dq), b)]
    else:
        parts = []
        for bi in range(b):
            for h0 in range(0, hq, heads):
                bshd = (slice(bi, bi + 1), slice(None), slice(h0, h0 + heads))  # [b, s, h, d] tensors
                bhs = (slice(bi, bi + 1), slice(h0, h0 + heads))  # [b, h, s] tensors
                ins = tuple(t[bshd] for t in (q, k, v, do)) + (nl[bhs], nd[bhs])
                parts.append((ins + tuple(t[bshd] for t in (dv, dk, dq)), 1))
    ds = scratch[_MLA_DS_GUARD // 2 : (nbytes - _MLA_DS_GUARD) // 2]
    q_off, nhead, ntile = _mla_dq_split(sq)
    # k_delta also writes NL = -log2(e) * lse + log2(scale) and ND = -delta: k_dkdv96ds starts its dP chains from ND and
    # its softmax yields scale * P, so dS = (scale * P) * (dP - delta) is one multiply.
    launches = [("k_delta", (sq // 64, b * hq, 1), (do, o, delta, lse, nl, nd, math.log2(scale), sq, hq))]
    for (q_, k_, v_, do_, nl_, nd_, dv_, dk_, dq_), bb in parts:
        dkdv = (q_, k_, v_, do_, nl_, nd_, dv_, dk_, ds, scale, sq, skv, hq, hkv, hq // hkv, sq // 16)
        dkdv += (skv - sq, c, bb, hg_kv, 1)  # ds_mode 1: dS^T stores bounded at each wave's true extent
        dqv = (ds, k_, dq_, sq, skv, hq, hkv, hq // hkv, skv // 32, skv - sq, c, q_off, ntile, hg_q)
        launches.append(("k_dkdv96ds", (hg_kv, _mla_dkdv_blocks(skv), bb * heads // hg_kv), dkdv))
        launches.append(("k_dqds_c", (hg_q, ntile + nhead, bb * heads // hg_q), dqv))
    return launches, scratch, (dq, dk, dv)


def hk_mla_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True, max_scratch_bytes=None):
    """MLA backward: q [B, Sq, H, 192], o / do [B, Sq, H, 128], k [B, Skv, H, 192], v [B, Skv, H, 128] contiguous
    bf16, lse [B, H, Sq] fp32 (natural log). Returns (dq, dk, dv) laid out like q / k / v.

    The dS scratch (4.33 GB at b2 s4096 h128) comes from the caching allocator on the current stream; over
    ``max_scratch_bytes`` (default $PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES, unset: no limit) the call is split by
    heads (see _hk_mla_bwd_plan). When not even one head group fits, or the scratch cannot be allocated (_mla_scratch),
    the call runs the FlyDSL backward."""
    scale = _mla_scale(softmax_scale)
    if max_scratch_bytes is None:
        max_scratch_bytes = envs.PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES
    with torch.cuda.device(q.device):
        try:
            launches, scratch, grads = _hk_mla_bwd_plan(do, q, k, v, o, lse, scale, causal, max_scratch_bytes)
        except _MlaScratchError as e:
            reason = str(e)
        else:
            stream = torch.cuda.current_stream()
            ticket = scratch.view(torch.int32)[(_MLA_DS_GUARD - 64) // 4 : _MLA_DS_GUARD // 4]
            for name, grid, values in launches:
                if name == "k_dkdv96ds":
                    ticket.zero_()  # k_dkdv96ds's work counter; it hands out work, so it does not change the result
                _mla_kernel(name)(grid, values, stream)
            return grads
        # Outside the except clauses: the failed allocations are released by now.
        from primus_turbo.flydsl.attention.gfx1250_mla import bwd as flydsl_bwd

        why = flydsl_bwd.shape_unsupported_reason(q.shape[0], q.shape[1], k.shape[1], q.shape[2])
        if why is not None:
            raise NotImplementedError(f"hk_mla_bwd: {reason}; FlyDSL cannot take it either: {why}")
        logger.warning(f"hk_mla_bwd: {reason}; this backward runs on FlyDSL", once=True)
        return flydsl_bwd.flash_attn_mla_bwd(do, q, k, v, o, lse, scale, causal)
