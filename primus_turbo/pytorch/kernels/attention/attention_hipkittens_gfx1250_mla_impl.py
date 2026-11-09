###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HipKittens kernels of the gfx1250 (MI455X) DeepSeek-V3 MLA attention backward: ``hk_mla_bwd``.

q / k head dim 192, v head dim 128, bf16, MHA, bottom-right causal; the tensors of the FlyDSL MLA kernels
(``attention_flydsl_gfx1250_mla_impl``, which runs this backward when PRIMUS_TURBO_MLA_BWD_IMPL is ``hk``).
The kernels are code objects built from ``csrc/kernels/attention/hipkittens/gfx1250_mla`` into
``primus_turbo/lib/hk_gfx1250_mla`` (see the README there), launched through the HIP runtime torch loaded:
  k_delta     NL = -log2(e) * lse and ND = -scale * delta, delta = rowsum(dO * O)
  k_dkdv96ds  dK, dV over 96-row kv blocks, then 64-row tail blocks; stores every bf16 dS^T tile it feeds the dK
              WMMA into a scratch
  k_dqds_c    dQ = dS K from that scratch: 384-query workgroups plus one merged 256-query head tile
They run the FlyDSL backward's arithmetic in its order, so the gradients are bitwise FlyDSL's, and every output
element is written by one workgroup, so the backward is deterministic.
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

from primus_turbo.common.logger import logger

__all__ = ["hk_mla_unsupported_reason", "hk_mla_bwd"]

MLA_D_QK, MLA_D_V = 192, 128
_MLA_DKDV_ROWS = 96  # k_dkdv96ds kv rows per workgroup
_MLA_DQ_ROWS, _MLA_DQ_HEAD_ROWS = 384, 256  # k_dqds_c queries per workgroup, and of its merged head tile
_MLA_HEAD_GROUP = 32  # head-group target of the k_dkdv96ds / k_dqds_c grids
_MLA_MAX_BH_S = 512 * 4096  # largest batch * heads * seqlen the kernels' index arithmetic is verified for
# kernel: (workgroup size, packed arguments, sha256 of the code object), the signatures and pins of
# csrc/.../gfx1250_mla/manifest.json (a test keeps them equal); only a code object with its pin is loaded:
#   k_delta     DO O LSE NL ND, scale, S H
#   k_dkdv96ds  Q K V DO NL ND DV DK DS, scale, Sq Skv Hq Hkv G nqt cshift causal B HG ds_mode
#   k_dqds_c    DS K DQ, Sq Skv Hq Hkv G nkvt cshift causal q_off ntile HG
_MLA_KERNELS = {
    "k_delta": (256, "<5Qf2i", "cc6c67a7c5475ddaab90f23e75eee07fc3e24fdf0dc3e1483c9128dc3cda2aa8"),
    "k_dkdv96ds": (64, "<9Qf11i", "620c95a97780f4dec86b8ffd135e9efec7e3c5195bb98cf4eba8f187356dd2c0"),
    "k_dqds_c": (128, "<3Q11i", "ccd6c8e68afd30e588477bf2fae710cd0cd50b03555238ac2cb4040106fbbd60"),
}
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
    # under the mask, kv16 row k >= 2m (seqlen_kv >= seqlen_q: m >= 0) skips its first k // 2 - m q pairs
    r = max(nkt - 2 * m, 0)
    skip = (r // 2) * (r // 2 - 1) + (r % 2) * (r // 2) if causal else 0
    return b * h * (nkt * nqp - skip) * 1024


def _mla_shape_reason(b: int, sq: int, skv: int, h: int, hkv: int, causal: bool) -> Optional[str]:
    """Why the kernels cannot take this problem, beyond the FlyDSL MLA gate's rules, or None. The reasons name no
    sizes: under torch.compile the sizes may be symbolic, and formatting one would specialize it."""
    if h != hkv:
        return "MHA only (heads_q == heads_kv)"
    if not causal:
        return "causal only"
    if skv < sq:
        return "seqlen_kv < seqlen_q"
    if sq % 32 or skv % 32:
        return "seqlen_q / seqlen_kv must be multiples of 32"
    if sq % _MLA_DQ_ROWS not in (0, _MLA_DQ_HEAD_ROWS):
        return f"seqlen_q % {_MLA_DQ_ROWS} must be 0 or {_MLA_DQ_HEAD_ROWS}"
    if b * h * skv > _MLA_MAX_BH_S:
        return f"batch * heads * seqlen_kv exceeds the verified {_MLA_MAX_BH_S}"
    return None


@functools.lru_cache(maxsize=None)
def _hip_runtime() -> ctypes.CDLL:
    """The libamdhip64 torch loaded, typed for the module API: launches through another copy would run in another
    HIP runtime, outside torch's contexts and streams. Found by its soname without loading anything: the dynamic
    linker returns the first copy loaded under that name, torch's, also when a library maps another one later."""
    if torch.version.hip is None:
        raise RuntimeError("torch is not a ROCm build")
    soname = f"libamdhip64.so.{torch.version.hip.split('.')[0]}"
    hip = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)  # OSError when torch did not load it
    p, u = ctypes.c_void_p, ctypes.c_uint
    hip.hipModuleLoadData.argtypes = [ctypes.POINTER(p), p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(p), p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [p, u, u, u, u, u, u, u, p, ctypes.POINTER(p), ctypes.POINTER(p)]
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipGetLastError.argtypes = []
    return hip


def _hip_check(rc: int, what: str) -> None:
    if rc != 0:
        hip = _hip_runtime()
        hip.hipGetLastError()  # clears the error HIP keeps per thread: torch's next launch check would raise it
        raise RuntimeError(f"{what} failed: {hip.hipGetErrorString(rc).decode()} ({rc})")


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
                with torch.cuda.device(device):  # hipModuleLoadData loads into the current device
                    _hip_check(
                        hip.hipModuleLoadData(ctypes.byref(module), self.image), f"loading {self.name}"
                    )
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


_MLA_UNAVAILABLE = {}  # device index -> why the kernels cannot run there, or None


@torch.compiler.assume_constant_result  # the gate runs in compiled regions; the answer is fixed per device
def hk_mla_unavailable_reason(device: Optional[int] = None) -> Optional[str]:
    """None when the code objects are the pinned ones and load on ``device`` (default: the current one), else why
    not."""
    if device is None:
        try:
            device = torch.cuda.current_device()
        except RuntimeError as e:  # no usable GPU
            return f"the HipKittens MLA backward kernels are unavailable: {e}"
    if device not in _MLA_UNAVAILABLE:
        reason = None
        try:
            for name in _MLA_KERNELS:
                _mla_kernel(name).load(device)
        except (OSError, RuntimeError) as e:
            reason = f"the HipKittens MLA backward kernels are unavailable: {e}"
        _MLA_UNAVAILABLE[device] = reason
    return _MLA_UNAVAILABLE[device]


def hk_mla_unsupported_reason(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Optional[str]:
    """Why ``hk_mla_bwd`` cannot take kernel-layout q / k / v that pass the FlyDSL MLA gate
    (``flydsl_gfx1250_mla_unsupported_reason``), or None."""
    b, sq, h, _ = q.shape
    _, skv, hkv, _ = k.shape
    reason = _mla_shape_reason(b, sq, skv, h, hkv, bool(causal))
    return reason if reason is not None else hk_mla_unavailable_reason(q.device.index)


def _mla_check(q, k, v, o, do, lse) -> Tuple[int, int, int, int, int]:
    """(b, sq, skv, hq, hkv) of contiguous [b, s, h, d] bf16 q / k / v / o / do and [b, hq, sq] fp32 lse."""
    b, sq, hq, _ = q.shape
    skv, hkv = k.shape[1], k.shape[2]
    want = [
        (q, (b, sq, hq, MLA_D_QK)),
        (k, (b, skv, hkv, MLA_D_QK)),
        (v, (b, skv, hkv, MLA_D_V)),
        (o, (b, sq, hq, MLA_D_V)),
        (do, (b, sq, hq, MLA_D_V)),
        (lse, (b, hq, sq)),
    ]
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


class _MlaScratchError(RuntimeError):
    pass


_MLA_OOM_BYTES = {}  # device -> the smallest dS scratch whose allocation (with the outputs) failed there


def _mla_buffers(q, k, v, b, sq, skv, hq, causal):
    """The dS scratch, NL / ND and (dq, dk, dv), from the caching allocator on the current stream. A failed
    allocation costs the allocator a synchronize and its whole cache, so after one on a device, a scratch at least
    as large is not tried there again: _MlaScratchError, and the call runs on FlyDSL."""
    nbytes = _mla_ds_bytes(b, sq, skv, hq, causal)
    failed = _MLA_OOM_BYTES.get(q.device, math.inf)
    if nbytes >= failed:
        raise _MlaScratchError(
            f"no memory for a {nbytes}-byte dS scratch: {failed} bytes did not fit on {q.device}"
        )
    try:
        ds = torch.empty((nbytes // 2,), device=q.device, dtype=torch.bfloat16)  # first: the largest
        nl, nd = (torch.empty((b, hq, sq), device=q.device, dtype=torch.float32) for _ in range(2))
        grads = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    except torch.OutOfMemoryError:
        _MLA_OOM_BYTES[q.device] = nbytes
        raise _MlaScratchError(f"no memory for a {nbytes}-byte dS scratch and the gradients") from None
    return ds, nl, nd, grads


def _mla_head_group(nh: int) -> int:
    """Heads per k_dkdv96ds / k_dqds_c grid group: nh when nh <= _MLA_HEAD_GROUP, else the largest multiple of 8 (one
    head's workgroups on one XCD) that divides nh and is at most _MLA_HEAD_GROUP, else nh."""
    if nh <= _MLA_HEAD_GROUP:
        return nh
    return next((d for d in range(_MLA_HEAD_GROUP, 7, -8) if nh % d == 0), nh)


def _hk_mla_bwd_plan(do, q, k, v, o, lse, scale, causal):
    """(launches [(kernel, grid, argument values)], (dq, dk, dv)) of hk_mla_bwd; works on meta tensors.
    _MlaScratchError when the buffers cannot be allocated."""
    b, sq, skv, hq, hkv = _mla_check(q, k, v, o, do, lse)
    reason = _mla_shape_reason(b, sq, skv, hq, hkv, bool(causal))
    if reason is not None:
        raise NotImplementedError(f"hk_mla_bwd: {reason}: q {tuple(q.shape)}, k {tuple(k.shape)}")
    c = int(bool(causal))
    ds, nl, nd, (dq, dk, dv) = _mla_buffers(q, k, v, b, sq, skv, hq, c)
    hg_kv, hg_q = _mla_head_group(hkv), _mla_head_group(hq)
    q_off, nhead, ntile = _mla_dq_split(sq)
    dkdv = (q, k, v, do, nl, nd, dv, dk, ds, scale, sq, skv, hq, hkv, hq // hkv, sq // 16)
    dkdv += (skv - sq, c, b, hg_kv, 1)  # ds_mode 1: the dS^T stores, bounded at each wave's true extent
    dqv = (ds, k, dq, sq, skv, hq, hkv, hq // hkv, skv // 32, skv - sq, c, q_off, ntile, hg_q)
    launches = [
        ("k_delta", (sq // 64, b * hq, 1), (do, o, lse, nl, nd, scale, sq, hq)),
        ("k_dkdv96ds", (hg_kv, _mla_dkdv_blocks(skv), b * hkv // hg_kv), dkdv),
        ("k_dqds_c", (hg_q, ntile + nhead, b * hq // hg_q), dqv),
    ]
    return launches, (dq, dk, dv)


def hk_mla_bwd(do, q, k, v, o, lse, softmax_scale=None, causal=True):
    """MLA backward: q [B, Sq, H, 192], o / do [B, Sq, H, 128], k [B, Skv, H, 192], v [B, Skv, H, 128] contiguous
    bf16, lse [B, H, Sq] fp32 (natural log). Returns (dq, dk, dv) laid out like q / k / v.

    The dS scratch (4.33 GB at b2 s4096 h128) comes from the caching allocator on the current stream. When it or the
    outputs cannot be allocated (_mla_buffers), the call runs the FlyDSL backward."""
    scale = _mla_scale(softmax_scale)
    lse = lse.contiguous().float()  # as the FlyDSL backward takes it
    with torch.cuda.device(q.device):
        try:
            launches, grads = _hk_mla_bwd_plan(do, q, k, v, o, lse, scale, causal)
        except _MlaScratchError as e:
            reason = str(e)
        else:
            stream = torch.cuda.current_stream()
            for name, grid, values in launches:
                _mla_kernel(name)(grid, values, stream)
            return grads
        # Outside the except clauses: the failed allocations are released by now.
        from primus_turbo.flydsl.attention.gfx1250_mla import bwd as flydsl_bwd

        why = flydsl_bwd.shape_unsupported_reason(q.shape[0], q.shape[1], k.shape[1], q.shape[2])
        if why is not None:
            raise NotImplementedError(f"hk_mla_bwd: {reason}; FlyDSL cannot take it either: {why}")
        logger.warning(f"hk_mla_bwd: {reason}; this backward runs on FlyDSL", once=True)
        return flydsl_bwd.flash_attn_mla_bwd(do, q, k, v, o, lse, scale, causal)
