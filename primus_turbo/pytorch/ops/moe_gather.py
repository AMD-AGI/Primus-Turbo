###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-consumer MoE gather handoffs for grouped MXFP4 quantization.

The forward placeholder must be consumed by the gather-aware quantizer. Backward
wrappers materialize the original gather when an ordinary tensor consumer needs
values, retaining alias and in-place mutation semantics.
"""

import os
import weakref

import torch
from torch.utils._pytree import tree_map

# A short-lived single-consumer handoff. Holding the Storage object prevents
# address reuse until the consumer claims the entry, without retaining a
# placeholder TensorImpl or changing dummy-wgrad ownership.
_PERMUTED_ACTIVATION_SEAM_TABLE = {}


def register_permuted_activation_seam(placeholder, src, dest2src, permuted_probs, row_id_map=None):
    key = placeholder.data_ptr()
    if key in _PERMUTED_ACTIVATION_SEAM_TABLE:
        raise RuntimeError("Unconsumed permute/quantize handoff")
    _PERMUTED_ACTIVATION_SEAM_TABLE[key] = (
        placeholder.untyped_storage(),
        tuple(placeholder.shape),
        tuple(placeholder.stride()),
        placeholder.dtype,
        placeholder.device,
        (src, dest2src, permuted_probs, _new_backward_gather_plan(src, dest2src, row_id_map)),
    )


def lookup_permuted_activation_seam(x, include_plan=False):
    if not isinstance(x, torch.Tensor):
        return None
    key = x.data_ptr()
    entry = _PERMUTED_ACTIVATION_SEAM_TABLE.get(key)
    if entry is None:
        if x.ndim == 2 and x.shape[0] > 1 and x.stride(0) == 0:
            raise RuntimeError("Unmaterialized permute placeholder has no gather metadata")
        return None
    storage, shape, stride, dtype, device, payload = entry
    if (
        storage is not x.untyped_storage()
        or shape != tuple(x.shape)
        or stride != tuple(x.stride())
        or dtype != x.dtype
        or device != x.device
    ):
        raise RuntimeError("Permute/quantize handoff storage or layout mismatch")
    del _PERMUTED_ACTIVATION_SEAM_TABLE[key]
    return payload if include_plan else payload[:3]


# Deferred no-probs unpermute gradient. All aliases share authoritative
# materialized storage once any ordinary tensor operation needs its values.

_BACKWARD_GATHER_OUTPUTS = {}
_BACKWARD_GATHER_AUDIT = {}


def _gather_count(name):
    if os.environ.get("GPTOSS_BACKWARD_GATHER_AUDIT") == "1":
        _BACKWARD_GATHER_AUDIT[name] = _BACKWARD_GATHER_AUDIT.get(name, 0) + 1


def _gather_signature(x):
    return (
        x.untyped_storage(),
        x.storage_offset(),
        tuple(x.shape),
        tuple(x.stride()),
        x.dtype,
        x.device,
        x._version,
    )


class _BackwardGatherPlan:
    def __init__(self, src, d2s, rowmap):
        self.src_shape = tuple(src.shape)
        self.shape = (d2s.numel(), src.shape[1])
        self.dtype, self.device = src.dtype, src.device
        self.d2s, self.rowmap = d2s, rowmap
        self.d2s_signature = _gather_signature(d2s)
        self.rowmap_signature = _gather_signature(rowmap)

    def validate(self):
        if (
            _gather_signature(self.d2s) != self.d2s_signature
            or _gather_signature(self.rowmap) != self.rowmap_signature
        ):
            raise RuntimeError("Backward gather routing metadata was mutated")


def _new_backward_gather_plan(src, d2s, rowmap):
    if not (
        os.environ.get("GPTOSS_FUSED_BACKWARD_PERMUTE_QUANT") == "1"
        and torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and os.environ.get("PRIMUS_TP") == "1"
        and os.environ.get("PRIMUS_EP") == "1"
        and os.environ.get("MOE_SKIP_IDENTITY_SORT") == "1"
        and rowmap is not None
        and type(src) is torch.Tensor
        and src.dtype == torch.bfloat16
        and src.is_cuda
        and src.is_contiguous()
        and src.ndim == 2
        and src.shape[1] == 2880
        and 0 < src.shape[0] <= 32768
        and d2s.dtype == torch.int32
        and d2s.is_contiguous()
        and d2s.ndim == 1
        and d2s.numel() == src.shape[0] * 4
        and rowmap.dtype == torch.int32
        and rowmap.is_contiguous()
        and tuple(rowmap.shape) == (src.shape[0], 65)
        and d2s.device == src.device == rowmap.device
    ):
        return None
    return _BackwardGatherPlan(src, d2s, rowmap)


def enroll_backward_gather_output(out, plan, owner):
    if plan is None or type(out) is not torch.Tensor or not out.is_contiguous():
        return
    if tuple(out.shape) != plan.shape or out.dtype != plan.dtype or out.device != plan.device:
        return
    key = out.data_ptr()

    def cleanup(ref):
        entry = _BACKWARD_GATHER_OUTPUTS.get(key)
        if entry is not None and entry[0] is ref:
            del _BACKWARD_GATHER_OUTPUTS[key]

    ref = weakref.ref(owner, cleanup)
    _BACKWARD_GATHER_OUTPUTS[key] = (ref, _gather_signature(out), plan)
    _gather_count("enrolled")


def claim_backward_gather_output(inp, rowmap, probs, padding, tokens, experts, hidden):
    if type(inp) is not torch.Tensor:
        return None
    entry = _BACKWARD_GATHER_OUTPUTS.pop(inp.data_ptr(), None)
    if entry is None:
        return None
    ref, signature, plan = entry
    if (
        ref() is None
        or signature != _gather_signature(inp)
        or probs is not None
        or padding is not None
        or experts != 32
        or hidden != 2880
        or tokens != plan.src_shape[0]
        or _gather_signature(rowmap) != plan.rowmap_signature
    ):
        return None
    plan.validate()
    _gather_count("claimed")
    return plan


class _BackwardGatherState:
    def __init__(self, source, plan):
        # Retain the actual TensorImpl, not only detach(): autograd must not
        # steal and mutate this still-needed upstream buffer during fanout.
        self.source, self.plan = source, plan
        self.source_signature = _gather_signature(source)
        self.materialized = None

    def validate_source(self):
        if _gather_signature(self.source) != self.source_signature:
            raise RuntimeError("Deferred MoE source gradient was mutated")

    def dense(self):
        if self.materialized is None:
            self.validate_source()
            self.plan.validate()
            p = self.plan
            self.materialized = torch.ops.te_moe.unpermute_mask_map_bwd_no_probs(
                self.source, p.rowmap, None, p.src_shape[0], 32, p.shape[0], p.shape[1]
            )
            _gather_count("materialized")
        return self.materialized


class _BackwardGatherTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, state):
        p = state.plan
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            p.shape,
            strides=(p.shape[1], 1),
            storage_offset=0,
            dtype=p.dtype,
            device=p.device,
            layout=torch.strided,
            requires_grad=False,
        )
        out._gather_state = state
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return cls(args[0]._gather_state)
        if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default) and tuple(
            args[1]
        ) == tuple(args[0].shape):
            return cls(args[0]._gather_state)
        # Metadata mutation cannot update every retained lazy alias consistently.
        # Fail explicitly rather than exposing invalid logical tensor metadata.
        if func._schema.name in {
            "aten::resize_",
            "aten::resize_as_",
            "aten::set_",
            "aten::as_strided_",
            "aten::transpose_",
            "aten::t_",
            "aten::squeeze_",
            "aten::unsqueeze_",
        }:
            raise RuntimeError("Metadata mutation of a deferred MoE gradient is unsupported")
        wrappers = []

        def unwrap(x):
            if isinstance(x, cls):
                wrappers.append(x)
                return x._gather_state.dense()
            return x

        actual_args = tree_map(unwrap, args)
        actual_kwargs = tree_map(unwrap, kwargs)
        result = func(*actual_args, **actual_kwargs)
        # Preserve Tensor identity for add_, copy_, and out= schemas. Ordinary
        # views may escape as dense aliases; subsequent mutations remain visible
        # because every wrapper shares the same cached dense tensor.
        if any(a.alias_info is not None and a.alias_info.is_write for a in func._schema.arguments):

            def restore(x):
                for wrapper in wrappers:
                    if x is wrapper._gather_state.materialized:
                        return wrapper
                return x

            result = tree_map(restore, result)
        return result


def make_backward_gather(source, plan):
    if (
        plan is None
        or torch.is_grad_enabled()
        or type(source) is not torch.Tensor
        or tuple(source.shape) != plan.src_shape
        or source.dtype != plan.dtype
        or source.device != plan.device
        or not source.is_contiguous()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    plan.validate()
    _gather_count("created")
    return _BackwardGatherTensor(_BackwardGatherState(source, plan))


def resolve_backward_gather(grad, plan):
    if not isinstance(grad, _BackwardGatherTensor):
        return grad, {}
    state = grad._gather_state
    if state.materialized is not None or state.plan is not plan or torch.is_grad_enabled():
        return state.dense(), {}
    plan.validate()
    state.validate_source()
    stream = torch.cuda.current_stream(state.source.device)
    state.source.record_stream(stream)
    plan.d2s.record_stream(stream)
    _gather_count("consumed")
    return state.source, {"dest2src": plan.d2s, "permuted_probs": None, "total_m": plan.shape[0]}
