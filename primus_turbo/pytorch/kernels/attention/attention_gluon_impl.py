###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opaque PyTorch custom-op boundary for the gfx950 Gluon attention kernel."""

import functools
import sys
from importlib import import_module as _import_module
from importlib import metadata
from typing import Tuple

import torch

_RAW_MODULE = "primus_turbo.gluon.attention.f16_fa_gfx950_rotated_4cluster"

_custom_op = functools.partial(
    torch.library.custom_op,
    tags=(torch._C.Tag.cudagraph_unsafe,),
)


def _installed_triton_version() -> str:
    loaded_triton = sys.modules.get("triton")
    module_version = getattr(loaded_triton, "__version__", None)
    if module_version is not None:
        return str(module_version)

    for distribution in ("triton", "pytorch-triton-rocm"):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pass
    return "unknown"


def _is_gluon_capability_import_error(exc: ImportError) -> bool:
    missing_name = getattr(exc, "name", None)
    if isinstance(exc, ModuleNotFoundError) and missing_name is not None:
        if missing_name == _RAW_MODULE or _RAW_MODULE.startswith(f"{missing_name}."):
            return True
    if missing_name is not None:
        return missing_name == "triton" or missing_name.startswith("triton.")

    message = str(exc).lower()
    return "triton" in message or "cdna4" in message or "gluon compiler" in message


def _load_flash_attn_gluon_raw():
    try:
        raw_module = _import_module(_RAW_MODULE)
    except ImportError as exc:
        if not _is_gluon_capability_import_error(exc):
            raise
        triton_version = _installed_triton_version()
        raise RuntimeError(
            "Gluon flash attention is unavailable: "
            f"Triton {triton_version} does not provide the required "
            "triton.experimental.gluon CDNA4 compiler support. "
            "Install a compatible AMD Triton build with Gluon CDNA4 support."
        ) from exc
    return raw_module.flash_attn_gluon_raw


def _validate_gluon_inputs(q, k, v, qkv_format):
    if qkv_format not in ("bshd", "bhsd"):
        raise ValueError(f"qkv_format must be 'bshd' or 'bhsd', got {qkv_format!r}")

    tensors = (q, k, v)
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("q, k, and v must have unit stride in the head dimension")
    if qkv_format == "bshd":
        layout_is_valid = all(tensor.is_contiguous() for tensor in tensors)
    else:
        layout_is_valid = all(tensor.transpose(1, 2).is_contiguous() for tensor in tensors)
    if not layout_is_valid:
        raise ValueError(f"q, k, and v storage does not match public qkv_format={qkv_format!r}")


@_custom_op(
    "primus_turbo::flash_attn_gluon_forward",
    mutates_args=(),
    device_types="cuda",
)
def _flash_attn_gluon_forward_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    qkv_format: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.cuda.device(q.device):
        # The raw API has no stream argument: Triton reads PyTorch's implicit current
        # stream. Resolve it after selecting q's device and keep it live across import/JIT
        # and launch.
        _caller_stream = torch.cuda.current_stream(q.device)
        flash_attn_gluon_raw = _load_flash_attn_gluon_raw()

        if qkv_format == "bshd":
            raw_q, raw_k, raw_v = q, k, v
        else:
            raw_q = q.transpose(1, 2)
            raw_k = k.transpose(1, 2)
            raw_v = v.transpose(1, 2)

        raw_out, lse = flash_attn_gluon_raw(
            raw_q,
            raw_k,
            raw_v,
            softmax_scale=softmax_scale,
            causal=causal,
            qkv_format=qkv_format,
        )
        out = raw_out if qkv_format == "bshd" else raw_out.transpose(1, 2)
        return out, lse


@_flash_attn_gluon_forward_op.register_fake
def _flash_attn_gluon_forward_op_fake(q, k, v, softmax_scale, causal, qkv_format):
    raw_q = q if qkv_format == "bshd" else q.transpose(1, 2)
    raw_out = torch.empty_strided(
        raw_q.shape,
        raw_q.stride(),
        dtype=raw_q.dtype,
        device=raw_q.device,
    )
    out = raw_out if qkv_format == "bshd" else raw_out.transpose(1, 2)
    lse = q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)
    return out, lse


def flash_attn_gluon_forward_impl(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qkv_format="bshd",
):
    _validate_gluon_inputs(q, k, v, qkv_format)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return _flash_attn_gluon_forward_op(q, k, v, scale, bool(causal), qkv_format)
