# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Machine-readable capability manifest for the flex-attention compatibility layer.

Kept out of :mod:`flex_attention_interface` so that the public API module holds
signatures and control flow only. Re-exported from there as ``SUPPORT_STATUS``.
"""

from typing import Any, Dict

SUPPORT_STATUS: Dict[str, Any] = {
    "supported_now": {
        "mask": ["full", "causal", "sliding_window_causal", "document_causal"],
        "score_mod": ["none", "alibi_detected", "alibi_explicit"],
        "gqa_mqa": True,
        "return_lse": True,
        # A block_mask recognised (exactly) as same_doc(q,kv) & (q>=kv) is routed
        # through the varlen backend (block-diagonal cu_seqlens) rather than a dense
        # causal call. Requires S <= _MASK_PROBE_LIMIT and no bias / return_lse (use
        # flex_attention_varlen for those); recognition is exact so it never misfires.
        "document_causal_dense_recognition": "block_mask_same_doc_and_causal_routed_to_varlen",
        # Document packing is no longer capped at the 512 probe grid: when the probe is
        # truncated, boundaries are read off mask_mod's diagonal/sub-diagonal over the
        # full sequence and the pattern is verified *exactly* (chunked, no sampling) up
        # to _DOC_EXACT_VERIFY_LIMIT; beyond that we decline instead of guessing.
        "document_causal_beyond_probe": "boundaries_from_mask_mod_with_exact_chunked_verification",
        # Sliding-window-causal with a window LARGER than the 512 probe grid (e.g.
        # W=1024/2048/4096 on long sequences) is located by a binary search on the
        # last query row + exact per-row verification (see _locate_left_window), so a
        # big window on S>512 is now classified instead of raising NotImplementedError.
        "sliding_window_large": "window_gt_probe_located_by_binary_search_on_last_row",
    },
    # Classification (block_mask probe) and score_mod (ALiBi/soft-cap) detection are
    # memoised by object identity (weakref) so reusing the same block_mask / score_mod
    # across layers & steps skips the ~1-3 ms per-call probe. Pure speedup; identical
    # results. clear_classification_cache() resets it (tests / cold-cache benchmarks).
    "classification_cache": "memoised_by_block_mask_and_score_mod_object_identity_weakref",
    # Turbo-extension explicit args (superset of the torch signature; both default
    # None so a torch-style call is unchanged and remains a drop-in replacement).
    "turbo_extension_args": {
        # Explicit per-head fp32 slopes -> flash_attn_func(alibi_slopes=...);
        # bypasses the score_mod detector and is live on this build.
        "alibi_slopes": "live_explicit_bypasses_score_mod_detection",
        # Interface is in place but gated: softcap>0 raises NotImplementedError
        # (aiter dense fwd/bwd lack the param); 0/None means disabled (no-op).
        "softcap": "interface_ready_but_gated_positive_softcap_raises",
        # Attention dropout probability (0<=p<1) -> flash_attn_func(dropout_p=...);
        # live on this build. 0.0 (default) disables it (drop-in, no-op).
        "dropout_p": "live_explicit_passthrough_0_disables",
        # Per-query-head attention-sink logits (1D fp32, len==Hq) ->
        # flash_attn_func(sink=...); live on this build. None (default) disables it.
        # Sink kernel path requires head_dim_qk==head_dim_v and power-of-two head dim.
        "sink": "live_explicit_passthrough_none_disables",
        # Additive logits bias -> flash_attn_func(bias=...); live on this build.
        # aiter dense needs a single [Sq,Skv] bias in q's dtype (fp16/bf16) shared
        # across batch/heads (fp32 -> NaN, per-head 4D -> rejected by kernel). Verified
        # fwd+bwd correct. None (default) disables it.
        "bias": "live_explicit_passthrough_needs_Sq_Skv_qdtype_none_disables",
    },
    "unsupported_paths": {
        "arbitrary_score_mod": "path_b_codegen_stub_only",
        "arbitrary_mask_mod": "path_b_codegen_stub_only",
        # Recognised (via _detect_softcap) or requested explicitly, but blocked at
        # the kernel layer: the aiter dense fwd/bwd on this build expose no softcap
        # parameter, so a soft-cap hard-errors instead of silently ignoring the cap.
        "softcap": "detected_or_explicit_but_blocked_aiter_dense_kernel_has_no_softcap_param",
    },
    # Explicit variable-length / document-packing entry point (THD layout). A
    # superset-free thin wrapper around ``flash_attn_varlen_func``: the caller
    # supplies cu_seqlens directly, so there is no mask/score_mod probing here.
    "varlen": {
        "entry": "flex_attention_varlen",
        "layout": "thd_[total_tokens,H,D]",
        "supported": [
            "full",
            "causal (document-internal, block-diagonal via cu_seqlens)",
            "sliding_window_causal (per-segment window_size)",
            "gqa_mqa",
            "alibi_explicit",
            "dropout_p",
            "sink",
            "return_lse",
        ],
        "document_masking": "explicit_cu_seqlens_block_diagonal_plus_causal_true",
        "unsupported": [
            "arbitrary_score_mod_no_such_arg",
            "softcap_gt_0_gated",
            "bias",
        ],
    },
    # The empirically resolved ALiBi sign for this build; see the module docstring.
    # tests/pytorch/ops/test_flex_score_mod.py re-measures it against the real
    # kernel so a build with the opposite convention fails loudly.
    "alibi_sign_convention": "+slope*(kv-q)",
    "alibi_sign_self_check": "tests/pytorch/ops/test_flex_score_mod.py",
    # Layout-native entry: [B,S,H,D] in and out, skipping the 4 transpose+contiguous
    # copies per forward (and their backward mirrors) that the torch-layout entry needs.
    "bshd_native_entry": "flex_attention_bshd",
    # A recognised variant is routed through choose_backend before dispatch.
    # Default policy is "turbo" for everything; register_backend_override lets a
    # tuner steer specific shapes/kinds to the (currently stub) custom hook.
    "backend_routing": {
        "selector": "choose_backend",
        "default": "turbo",
        "backends": ["turbo", "custom"],
        "override_api": ["register_backend_override", "clear_backend_overrides"],
    },
}
