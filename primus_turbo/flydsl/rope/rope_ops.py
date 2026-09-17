###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Autograd wrapper + lazy shim installer for the FlyDSL RoPE kernels.

Forward (campaign H3, workstream A "port", OPTIMIZE r3/r4) replaced
``fused_qkv_rope_forward`` with ``flydsl_qkv_rope_forward``. TE's own
``FusedQKVRoPEFunc.backward`` only needs ``(q_freqs, k_freqs)`` saved from
forward, not the original qkv, so a forward that produces bit-compatible
Q/K/V and saves the same two tensors keeps gradients exact regardless of
which kernel computes backward.

Backward (workstream B "port", OPTIMIZE r9, L1) now has an env-gated choice
in ``_te_rope_backward`` below: TE's ``tex.fused_qkv_rope_backward`` (default)
or ``flydsl_qkv_rope_backward`` -- the inverse-RoPE FlyDSL kernel, i.e. the
same rotate-half math with sin negated (the forward 2x2 rotation transposed).
Gated by ``PRIMUS_TURBO_FLYDSL_ROPE_BWD=1`` (default off) so the two arms of
a same-round A/B are byte-identical source, differing only by the env flag
at launch time -- see that constant's definition for the measured numbers
this round's prototype produced (SNR, device time) before this port.

Shim mechanics (proven in the campaign's round-2 probe ``r2_p1_callsite.py``):
Megatron's ``attention`` module binds ``apply_fused_qkv_rotary_pos_emb`` via
``from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb``
at *import time*, so patching TE's module after that point is a silent no-op --
callers already hold Megatron's own reference. The shim must patch Megatron's
module attribute directly, and it must do so lazily: Megatron is not in
``sys.modules`` at ``import primus_turbo`` time, only once some Megatron code
has actually run.

Call site (OPTIMIZE r4, corrected from an earlier plan): NOT
``primus_turbo/pytorch/ops/gemm_fp4.py`` -- that file is both inert (repo
edits under ``pytorch/**`` are not bind-mounted, goal.md section 1) and
clobbered from deps at every training launch, and grepping the deployed
copy this round found zero references to this function anywhere in the
repo. The real call site is
``gemm_epilogue_helper.py::install_rope_shim_retry_hook()``, which patches
``StoreCPlain.__init__`` (in the safe, never-clobbered landing zone) to call
``maybe_install_rope_shim()`` after every GEMM-kernel-shape construction --
guaranteed to run after Megatron's attention module is built, on the same
schedule ``install_h2perm_hook`` already uses for a different patch target.

Guarded by ``PRIMUS_TURBO_FLYDSL_ROPE=1`` (default off): the TE path stays the
production default until a transfer run confirms step time and RoPE-family ms
both improve (this round only validates SNR + isolated latency on GPUs 0-1).
"""

import os

import torch

from primus_turbo.flydsl.rope.rope_kernel import flydsl_qkv_rope_backward, flydsl_qkv_rope_forward

_ROPE_SHIM_ENV = "PRIMUS_TURBO_FLYDSL_ROPE"

# Backward A/B switch (campaign 20260914_113401, OPTIMIZE r9, L1). Default off
# so the standing production behavior (TE backward) is unchanged unless a
# launch explicitly sets this -- same convention as _ROPE_SHIM_ENV above.
# Same-round A/B arms flip ONLY this env var; the source tree is identical.
_ROPE_BWD_ENV = "PRIMUS_TURBO_FLYDSL_ROPE_BWD"


def _te_rope_backward(dq, dk, dv, q_freqs, k_freqs, qkv_split_arg_list):
    """Backward for ``_FlyDSLFusedQKVRoPEFunc``: TE's kernel by default, or
    the FlyDSL inverse-RoPE kernel when ``PRIMUS_TURBO_FLYDSL_ROPE_BWD=1``.

    The TE branch mirrors
    ``transformer_engine.pytorch.attention.rope.FusedQKVRoPEFunc.backward``
    exactly (same positional args, same defaults for tensor_format/
    interleaved/cp_size/cp_rank), so gradients are bit-identical to the stock
    TE path. The FlyDSL branch (``flydsl_qkv_rope_backward``, this round's
    port of ``r8_replan_probe2.py::_make_bwd_kernel``/``_build_launchers``)
    computes the same packed ``dQKV`` gradient with the forward kernel's
    rotate-half matrix transposed (sin negated) -- measured this round at
    production shape [8192,4,8,768], seed 1234, on two independent GPUs:
    overall SNR 101.17 dB (dQ 100.72 dB, dK 99.24 dB, dV bit-exact) against a
    45 dB gate, and device time 0.17169-0.17502 ms/call (5.494-5.601 ms/step
    at 32 launches/step) vs TE's own 0.33718-0.33595 ms/call
    (10.79-10.75 ms/step) on the same two cards -- see goal.md section 2.2/2.3
    for the full table this port is transcribed from.

    ``.contiguous()`` on dq/dk/dv is called once here (not per-branch) so
    both paths see identical inputs; in production this is a no-op (rope_bwd's
    predecessor is ``ck_fused_attn::dk_dv_reduce`` with no copy kernel in
    between, so the gradients arrive already contiguous).
    """
    dq = dq.contiguous()
    dk = dk.contiguous()
    dv = dv.contiguous()

    if os.environ.get(_ROPE_BWD_ENV, "0") == "1":
        return flydsl_qkv_rope_backward(dq, dk, dv, q_freqs, k_freqs, qkv_split_arg_list)

    import transformer_engine.pytorch.attention.rope as te_rope
    from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

    tex = te_rope.tex
    return tex.fused_qkv_rope_backward(
        dq,
        dk,
        dv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        QKVFormat["sbhd"],
        False,  # interleaved
        1,  # cp_size
        0,  # cp_rank
    )


class _FlyDSLFusedQKVRoPEFunc(torch.autograd.Function):
    """Forward: FlyDSL kernel (this file's target). Backward: TE's existing kernel."""

    @staticmethod
    def forward(ctx, qkv, q_freqs, k_freqs, qkv_split_arg_list):
        if q_freqs.dtype != torch.float32:
            q_freqs = q_freqs.float()
        if k_freqs.dtype != torch.float32:
            k_freqs = k_freqs.float()
        q_freqs = q_freqs.contiguous()
        k_freqs = k_freqs.contiguous()
        assert qkv.is_contiguous(), "QKV tensor should be contiguous."

        q_out, k_out, v_out = flydsl_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list)

        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.qkv_split_arg_list = qkv_split_arg_list
        return q_out, k_out, v_out

    @staticmethod
    def backward(ctx, dq, dk, dv):
        q_freqs, k_freqs = ctx.saved_tensors
        grad_qkv = _te_rope_backward(dq, dk, dv, q_freqs, k_freqs, ctx.qkv_split_arg_list)
        return grad_qkv, None, None, None


def flydsl_fused_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list):
    """Autograd-wrapped FlyDSL RoPE forward with a TE-backward fallback.
    Drop-in replacement for ``apply_fused_qkv_rotary_pos_emb`` at the
    (qkv, q_freqs, k_freqs, qkv_split_arg_list) call shape Megatron actually uses.
    """
    return _FlyDSLFusedQKVRoPEFunc.apply(qkv, q_freqs, k_freqs, qkv_split_arg_list)


def _shape_supported(qkv, q_freqs, k_freqs, qkv_split_arg_list):
    """Best-effort compatibility check so the shim can fall back to TE instead
    of asserting, for any shape/dtype this kernel was not built for."""
    if qkv.ndim != 4 or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
        return False
    if len(qkv_split_arg_list) != 3:
        return False
    q_size, k_size, v_size = qkv_split_arg_list
    if k_size != 128 or v_size != 128 or q_size % 128 != 0:
        return False
    if qkv.shape[-1] != q_size + k_size + v_size:
        return False
    if q_freqs.shape[-1] != 128 or k_freqs.shape[-1] != 128:
        return False
    return True


def install_rope_shim(attention_module):
    """Install the idempotent FlyDSL RoPE-forward shim onto an already-imported
    ``megatron.core.transformer.attention`` module object. Safe to call more
    than once (no-op after the first successful install).
    """
    if getattr(attention_module, "_primus_turbo_rope_shim", False):
        return

    orig = attention_module.apply_fused_qkv_rotary_pos_emb

    def _flydsl_apply_fused_qkv_rotary_pos_emb(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        tensor_format="sbhd",
        start_positions=None,
        interleaved=False,
        cu_seqlens=None,
        cp_size=1,
        cp_rank=0,
    ):
        # Only the exact combination this kernel was built for takes the FlyDSL
        # path (matches the Megatron call site: 4 positional args, everything
        # else at its default). Anything else -- CP>1, start_positions,
        # interleaved, non-sbhd, or an unsupported shape -- falls back to the
        # original (TE) implementation so behavior never silently changes.
        if (
            tensor_format == "sbhd"
            and start_positions is None
            and not interleaved
            and cp_size == 1
            and _shape_supported(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        ):
            return flydsl_fused_qkv_rope_forward(qkv, q_freqs, k_freqs, qkv_split_arg_list)
        return orig(
            qkv,
            q_freqs,
            k_freqs,
            qkv_split_arg_list,
            tensor_format,
            start_positions,
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )

    _flydsl_apply_fused_qkv_rotary_pos_emb._primus_turbo_orig = orig
    attention_module.apply_fused_qkv_rotary_pos_emb = _flydsl_apply_fused_qkv_rotary_pos_emb
    attention_module._primus_turbo_rope_shim = True
    # One-time, rank-visible confirmation (matches the existing style of
    # primus_turbo.py's "gemm backend is set to flydsl" notice). OPTIMIZE r4
    # found the install trigger can silently no-op under a warm FlyDSL cache
    # with no other symptom -- a profiler trace or launch count is the only
    # way to notice, and both are expensive/unavailable mid-run. This print
    # is cheap (fires once per process) and makes success directly greppable
    # in rank logs without needing a kernel trace.
    print("[primus_turbo] FlyDSL RoPE-forward shim installed onto megatron attention module", flush=True)


_rope_shim_attempted = False


def maybe_install_rope_shim():
    """Lazy, idempotent installer -- call on every ``gemm_fp4()`` invocation.

    Cheap once installed (single bool check). Before that, cheap per call
    (env lookup + one ``sys.modules`` dict lookup): Megatron's ``attention``
    module is not present in ``sys.modules`` at ``import primus_turbo`` time
    (proven in ``r2_p1_callsite.py``), so this must keep checking lazily
    rather than installing once at import.
    """
    global _rope_shim_attempted
    if _rope_shim_attempted:
        return
    if os.environ.get(_ROPE_SHIM_ENV, "0") != "1":
        return
    import sys

    mod = sys.modules.get("megatron.core.transformer.attention")
    if mod is None:
        return  # Megatron not imported yet; retry on the next gemm_fp4() call.
    install_rope_shim(mod)
    assert getattr(mod, "_primus_turbo_rope_shim", False), "FlyDSL RoPE shim install did not take effect"
    _rope_shim_attempted = True
    # Cheap, always-on success signal (pitfalls.md 20260914_113401 r3: a
    # silent best-effort hook that succeeds is indistinguishable from one
    # that never ran until you profile-harvest a full training run). One
    # write, one process each; never worth guarding behind another env var.
    try:
        with open("/tmp/primus_turbo_rope_shim_installed", "a") as f:
            f.write(f"pid={os.getpid()} module={mod.__name__}\n")
    except Exception:
        pass
