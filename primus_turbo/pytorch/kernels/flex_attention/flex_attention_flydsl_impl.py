###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlyDSL backend of turbo flex attention (``primus_turbo.flydsl.flex_attention``).

Maps a call's mask and mods onto the FlyDSL kernels' static configuration and caches
one compiled kernel per configuration. A built-in mask on the fast path becomes the
kernel's causal / window flags; any other mask runs its FlyDSL twin (see
``flex_attention_dual``) over the block mask's lists at block size 128 (varlen:
elementwise over every block, as the FlyDSL kernels have no varlen block lists yet).
The backward is deterministic: dQ and dK/dV are computed by separate kernels, no
atomics.
"""

import importlib.util
import weakref
from typing import Optional

import torch

from primus_turbo.pytorch.ops.attention.flex_attention_dual import flydsl_twin

__all__ = ["flydsl_unsupported_reason", "flydsl_forward", "flydsl_backward"]

SPARSE_BLOCK = 128
MAX_AUX_TENSORS = 4

_FORWARDS: dict = {}
_BACKWARDS: dict = {}
_KV_MAJOR: dict = {}


def _flydsl():
    from primus_turbo.flydsl import flex_attention

    return flex_attention


def flydsl_unsupported_reason(cfg, block_mask, q) -> Optional[str]:
    """Why this call cannot run on the FlyDSL backend, or None if it can."""
    if importlib.util.find_spec("flydsl") is None:
        return "this environment: FlyDSL is not installed"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {q.dtype}"
    if cfg.score_grad_hook is not None and flydsl_twin(cfg.score_grad_hook) is None:
        return "score_grad_hook has no FlyDSL twin (register one with dual_mod)"
    if cfg.score_mod is not None and flydsl_twin(cfg.score_mod) is None:
        return "score_mod has no FlyDSL twin (register one with dual_mod)"
    if cfg.score_mod_bwd is not None and flydsl_twin(cfg.score_mod_bwd) is None:
        return "score_mod_bwd has no FlyDSL twin (register one with dual_mod)"
    mask_aux = ()
    if block_mask is not None and block_mask._fast_path is None:
        if flydsl_twin(block_mask.mask_mod) is None:
            return "mask_mod has no FlyDSL twin (register one with dual_mod)"
        mask_aux = block_mask.aux_tensors
    if len(cfg.aux_tensors) + len(mask_aux) > MAX_AUX_TENSORS:
        return f"more than {MAX_AUX_TENSORS} aux tensors (score_mod and mask_mod together)"
    if cfg.num_splits > 1 and block_mask is not None and block_mask._fast_path is None:
        return "num_splits > 1 with a block-sparse mask"
    return None


def _with_aux_slice(mod, lo, hi, is_mask):
    """Give a mod only its own aux tensors (score_mod's come first, then mask_mod's)."""
    if mod is None:
        return None
    if is_mask:
        return lambda b, h, q_idx, kv_idx, seqlen, aux: mod(b, h, q_idx, kv_idx, seqlen, aux[lo:hi])
    return lambda s, b, h, q_idx, kv_idx, seqlen, aux: mod(s, b, h, q_idx, kv_idx, seqlen, aux[lo:hi])


def _with_aux_slice_bwd(mod, lo, hi):
    if mod is None:
        return None
    return lambda g, s, b, h, q_idx, kv_idx, seqlen, aux: mod(g, s, b, h, q_idx, kv_idx, seqlen, aux[lo:hi])


def _with_aux_slice_hook(hook, lo, hi):
    if hook is None:
        return None
    return lambda g, ok, b, h, q_idx, kv_idx, seqlen, aux: hook(g, ok, b, h, q_idx, kv_idx, seqlen, aux[lo:hi])


def _mask_setup(cfg, block_mask):
    """``(flags, window, mask twin, mask aux, block-sparse lists or None)``."""
    if block_mask is None:
        return {}, (-1, -1), None, (), None
    if block_mask._fast_path is not None:
        causal, left, right = block_mask._fast_path
        if causal and left < 0:
            return {"is_causal": True}, (-1, -1), None, (), None
        if left < 0 and right < 0:
            return {}, (-1, -1), None, (), None
        return {"is_local": True}, (left, 0 if causal else right), None, (), None
    mask_fly = flydsl_twin(block_mask.mask_mod)
    if cfg.varlen:
        return {}, (-1, -1), mask_fly, block_mask.aux_tensors, None
    return {}, (-1, -1), mask_fly, block_mask.aux_tensors, block_mask.plan_at(SPARSE_BLOCK)


def _sparse_tensors(plan, batch, heads):
    bst = _flydsl().BlockSparseTensors
    plan = plan.expand(batch, heads)
    return bst(plan.mask_block_cnt, plan.mask_block_idx, plan.full_block_cnt, plan.full_block_idx,
               (SPARSE_BLOCK, SPARSE_BLOCK))


def _kv_major(plan, num_kv_blocks):
    """KV-major lists for the dK/dV kernel, memoized per plan (as the Triton backend's)."""
    key = id(plan.mask_block_idx)
    hit = _KV_MAJOR.get(key)
    if hit is None:
        from primus_turbo.flydsl.flex_attention.torch_utils import transpose

        fwd = _flydsl().BlockSparseTensors(
            plan.mask_block_cnt, plan.mask_block_idx, plan.full_block_cnt, plan.full_block_idx,
            (SPARSE_BLOCK, SPARSE_BLOCK),
        )
        hit = transpose(fwd, num_kv_blocks)
        _KV_MAJOR[key] = hit
        weakref.finalize(plan.mask_block_idx, _KV_MAJOR.pop, key, None)
    return hit


def _common(cfg, q, k, v, block_mask):
    flags, window, mask_fly, mask_aux, plan = _mask_setup(cfg, block_mask)
    n_score = len(cfg.aux_tensors)
    aux = tuple(cfg.aux_tensors) + tuple(mask_aux)
    heads_q, heads_kv = (q.shape[1], k.shape[1]) if cfg.varlen else (q.shape[2], k.shape[2])
    static = dict(
        dtype="bf16" if q.dtype == torch.bfloat16 else "fp16",
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        qhead_per_kvhead=heads_q // heads_kv,
        varlen_q=cfg.varlen,
        varlen_k=cfg.varlen,
        num_aux_tensors=len(aux),
        **flags,
    )
    mods = (
        flydsl_twin(cfg.score_mod) if cfg.score_mod is not None else None,
        flydsl_twin(cfg.score_mod_bwd) if cfg.score_mod_bwd is not None else None,
        mask_fly,
        n_score,
        len(aux),
    )
    return static, window, mods, aux, plan, heads_q


def flydsl_forward(q, k, v, out, lse, cfg, block_mask):
    """q/k/v/out in ``bshd`` (dense) or ``thd`` (varlen) views; writes out and lse."""
    static, window, mods, aux, plan, heads_q = _common(cfg, q, k, v, block_mask)
    score_fly, _, mask_fly, n_score, n_aux = mods
    split = cfg.num_splits > 1
    key = (tuple(sorted(static.items())), score_fly, mask_fly, n_score, plan is not None,
           plan is not None and plan.full is not None, split)
    fwd = _FORWARDS.get(key)
    if fwd is None:
        fwd = _flydsl().FlashAttentionForward(
            **static,
            score_mod=_with_aux_slice(score_fly, 0, n_score, False),
            mask_mod=_with_aux_slice(mask_fly, n_score, n_aux, True),
            block_sparse=plan is not None,
            has_full_blocks=plan is not None and plan.full is not None,
            sparse_block_size=(SPARSE_BLOCK, SPARSE_BLOCK) if plan is not None else None,
            split_kv=split,
        )
        _FORWARDS[key] = fwd
    batch = q.shape[0] if not cfg.varlen else None
    kwargs = {}
    if cfg.varlen:
        kwargs = dict(cu_seqlens_q=cfg.cu_seqlens_q, cu_seqlens_k=cfg.cu_seqlens_k, max_seqlen_q=cfg.max_seqlen_q)
    if plan is not None:
        kwargs["block_sparse_tensors"] = _sparse_tensors(plan, batch, heads_q)
    fwd(q, k, v, cfg.scale, out=out, lse=lse, window_left=window[0], window_right=window[1],
        aux_tensors=aux, num_splits=cfg.num_splits, **kwargs)


def flydsl_backward(dout, q, k, v, out, lse, dq, dk, dv, cfg, block_mask, dlse=None, score_grad=None):
    """Same views as ``flydsl_forward``; writes dq / dk / dv, and ``score_grad`` (zeroed,
    the ``score_grad_target``'s gradient) through the hook's FlyDSL twin when given."""
    static, window, mods, aux, plan, heads_q = _common(cfg, q, k, v, block_mask)
    score_fly, score_bwd_fly, mask_fly, n_score, n_aux = mods
    hook_fly = flydsl_twin(cfg.score_grad_hook) if score_grad is not None else None
    key = (tuple(sorted(static.items())), score_fly, score_bwd_fly, mask_fly, n_score, plan is not None,
           plan is not None and plan.full is not None, hook_fly)
    bwd = _BACKWARDS.get(key)
    if bwd is None:
        bwd = _flydsl().FlashAttentionBackward(
            **static,
            score_mod=_with_aux_slice(score_fly, 0, n_score, False),
            score_mod_bwd=_with_aux_slice_bwd(score_bwd_fly, 0, n_score),
            mask_mod=_with_aux_slice(mask_fly, n_score, n_aux, True),
            block_sparse=plan is not None,
            has_full_blocks=plan is not None and plan.full is not None,
            sparse_block_size=(SPARSE_BLOCK, SPARSE_BLOCK) if plan is not None else None,
            score_grad_hook=_with_aux_slice_hook(hook_fly, 0, n_score),
        )
        _BACKWARDS[key] = bwd
    kwargs = {}
    if cfg.varlen:
        kwargs = dict(cu_seqlens_q=cfg.cu_seqlens_q, cu_seqlens_k=cfg.cu_seqlens_k,
                      max_seqlen_q=cfg.max_seqlen_q, max_seqlen_k=cfg.max_seqlen_k)
    if plan is not None:
        batch = q.shape[0]
        kwargs["block_sparse_tensors"] = _sparse_tensors(plan, batch, heads_q)
        kwargs["block_sparse_tensors_kv"] = _expand_bst(_kv_major(plan, plan.num_kv_blocks), batch, heads_q)
    sg = score_grad
    if sg is not None and sg.dtype != torch.float32:
        sg = torch.zeros(sg.shape, dtype=torch.float32, device=sg.device)
    bwd(q, k, v, out, dout, lse, cfg.scale, dlse=dlse, window_left=window[0], window_right=window[1],
        aux_tensors=aux, dq=dq, dk=dk, dv=dv, score_grad=sg, **kwargs)
    if sg is not None and sg is not score_grad:
        score_grad.copy_(sg)


def _expand_bst(bst, batch, heads):
    def expand(t):
        return None if t is None else t.expand(batch, heads, *t.shape[2:])

    return bst._replace(
        mask_block_cnt=expand(bst.mask_block_cnt),
        mask_block_idx=expand(bst.mask_block_idx),
        full_block_cnt=expand(bst.full_block_cnt),
        full_block_idx=expand(bst.full_block_idx),
    )
