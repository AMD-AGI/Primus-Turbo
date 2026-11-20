###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Machine-readable capability manifest for the flex-attention compatibility layer.

Kept out of :mod:`flex_attention_interface` so that the public API module holds
signatures and control flow only. Re-exported from there as ``SUPPORT_STATUS``.
"""

from typing import Any, Dict

SUPPORT_STATUS: Dict[str, Any] = {
    "supported_now": {
        "mask": [
            "full",
            "causal",
            "sliding_window_causal",
            "sliding_window_bidirectional",
            "document_causal",
            "document_bidirectional",
            "document_windowed",
        ],
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
        # A non-causal band ``-R <= q - kv <= L`` (the local-attention shape used by
        # image / video models) maps exactly onto window_size=(L, R) with causal=False:
        # the csrc/CK entry forwards both edges to the forward and the backward alike.
        # The band is reconstructed from its two extreme diagonals and required to match
        # the probe exactly, so a merely band-shaped-but-holey mask is still refused.
        # Excluded with a sink: that backward takes sliding_window=window_size_left
        # only, so the right edge would be silently dropped (see
        # bidirectional_window_with_sink under not_supported_yet).
        "sliding_window_bidirectional": "band_reconstructed_and_verified_then_window_size_left_right",
        # Packing is no longer assumed to be autoregressive. The varlen kernels apply
        # `causal` and `window_size` *within* each segment on top of cu_seqlens, so all
        # four combinations of {causal, bidirectional} x {unwindowed, windowed} lower
        # onto the same block-diagonal call and differ only in the two flags handed
        # down. Bidirectional packing is the shape diffusion / encoder training uses:
        # several samples concatenated into one sequence with no causal term at all.
        # The within-document pattern is recovered from the mask and required to
        # reconstruct it exactly. This is no longer capped at the probe grid: past it,
        # boundaries come from mask_mod's diagonal/sub-diagonal, the window edges are
        # read off two whole rows of the longest document, and every candidate is
        # rebuilt and compared bit for bit (chunked, no sampling) up to
        # _DOC_EXACT_VERIFY_LIMIT.
        "document_bidirectional_and_windowed": "same_cu_seqlens_with_causal_and_window_size_carried_into_varlen",
        # The corollary, and the reason the above had to grow: inside the probe grid a
        # packed-plus-windowed mask is *byte-identical* to a plain window whenever the
        # first document outlives the probe. The window paths therefore no longer trust
        # the corner -- they rebuild the window over the whole sequence and, if it does
        # not hold, hand the mask to the document locator before refusing. Accepting
        # the corner used to drop every document boundary silently (measured on MI355:
        # 0.688 max abs error against the fp32 reference, vs 0.004 bf16 noise).
        "packed_window_not_flattened_into_plain_window": "window_reverified_over_whole_sequence_then_document_locator",
        # A probe corner that is entirely visible no longer short-circuits to "full":
        # bidirectional packing whose first document exceeds the probe, and bands wider
        # than the probe, look identical there. Fullness is re-verified against mask_mod
        # over the whole sequence, and an unverifiable mask raises instead of silently
        # running dense attention.
        "all_visible_probe_corner_reverified": "full_confirmed_over_whole_sequence_or_raise",
        # A band whose edges lie *outside* the probed corner is now recovered instead of
        # refused. The corner cannot show an edge it does not contain (|q-kv| <= 640 on
        # S=1024 fills a 512x512 corner completely, and L=640/R=64 shows only the right
        # edge), so each edge is binary-searched on the row where it is not clipped --
        # the last query row for the left edge, the first for the right -- and the band
        # is then rebuilt over the whole sequence and compared bit for bit. An edge that
        # reaches the end of the sequence is unbounded and is reported as -1, the
        # kernel's own convention. A band that drifts with position still raises.
        "band_edges_beyond_probe": "edges_binary_searched_on_unclipped_rows_then_verified_exactly",
    },
    # Classification (block_mask probe) and score_mod (ALiBi/soft-cap) detection are
    # memoised by object identity (weakref) so reusing the same block_mask / score_mod
    # across layers & steps skips the cold probe. Pure speedup; identical results.
    # That probe is not the "~1-3 ms" this comment used to claim: measured on MI355
    # (gfx950) it is 0.6-1.8 ms for causal and sliding-window shapes but 4.6 ms at
    # S=1024 rising to 19.2 ms at S=8192 for document packing, whose recognition
    # verifies the reconstruction exactly over the whole sequence. At S=8192 that is
    # ~10x the attention kernel it precedes. A warm hit is ~0.3 us.
    # NOTE: the primary key is object *identity* (weakref). A factory that rebuilds
    # mask_mod every step (natural for THD, whose boundaries change every step) would
    # never hit that, so there is a second, content-addressed level: _fn_fingerprint
    # keys on the function's code object plus its closure cell values, and only when
    # every cell holds a hashable scalar -- unfingerprintable mask_mods fall back to
    # identity and simply pay the cold cost. Bounded at 256 entries; overflow clears.
    # clear_classification_cache() resets both levels (tests / cold-cache benchmarks).
    "classification_cache": "identity_weakref_plus_mask_mod_closure_fingerprint_fallback",
    # Turbo-extension explicit args (superset of the torch signature; both default
    # None so a torch-style call is unchanged and remains a drop-in replacement).
    "turbo_extension_args": {
        # Explicit per-head fp32 slopes -> flash_attn_func(alibi_slopes=...);
        # bypasses the score_mod detector and is live on this build.
        "alibi_slopes": "live_explicit_bypasses_score_mod_detection",
        # Interface is in place but gated: softcap>0 raises NotImplementedError
        # (no trainable aiter fwd+bwd pair implements it -- see unsupported_paths
        # for the surveyed layer-by-layer state); 0/None means disabled (no-op).
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
        # Backward deterministic-accumulation flag -> flash_attn_func(deterministic=...)
        # / flash_attn_varlen_func(deterministic=...); live on this build. False
        # (default) is the historical behaviour. This is a passthrough of the backend's
        # own flag, not an independent bit-reproducibility guarantee: the layer used to
        # hard-code False, so a caller asking for determinism silently did not get it.
        "deterministic": "live_explicit_passthrough_false_is_historical_default",
        # fp8 attention -> flash_attn_fp8_func (aiter Triton kernel) instead of
        # flash_attn_func (aiter CK kernel). fp8=True uses the backend default config
        # (BLOCKWISE, block_size=64); fp8_config overrides it and implies fp8=True.
        # False/None (default) leaves the bf16/fp16 path untouched. The Triton kernel
        # supports strictly less than the CK one, so sink / bias / sliding window /
        # dropout_p>0 / deterministic=True / return_lse / document-causal packing /
        # non-bf16 dtypes are rejected explicitly rather than silently dropped.
        "fp8": "live_explicit_routes_to_flash_attn_fp8_func_reduced_feature_set_gated",
        "fp8_config": "live_explicit_implies_fp8_none_means_backend_default_blockwise_64",
    },
    "unsupported_paths": {
        "arbitrary_score_mod": "path_b_codegen_stub_only",
        "arbitrary_mask_mod": "path_b_codegen_stub_only",
        # Recognised (via _detect_softcap) or requested explicitly, but blocked at
        # the kernel layer, so a soft-cap hard-errors instead of silently ignoring it.
        # The block is NOT one missing python argument -- surveyed on this build:
        #   * CK ck_tile FORWARD has it (block/variants.hpp LogitsSoftCapParams; tanh
        #     is the compile-time default, softsign selectable), and aiter's dense
        #     binding passes a literal `0.0, // logits_soft_cap` at
        #     csrc/py_itfs_ck/mha_fwd_kernels.cu. So dense fwd is binding-only work.
        #   * CK ck_tile BACKWARD has none: fmha_bwd_kernel.hpp contains no soft cap,
        #     and neither do aiter's mha_bwd_kernels.cu / mha_varlen_bwd_kernels.cu.
        #     dS needs the tanh derivative; that math does not exist yet.
        #   * aiter's varlen python fwd takes logits_soft_cap, but
        #     FlashAttnVarlenFunc.backward never passes it and returns None for its
        #     gradient slot -- capped forward, uncapped gradients.
        #   * asm/v3 is disqualified by `logits_soft_cap == 0.0` in the fwd chooser;
        #     the triton kernels that do implement it are inference-only
        #     (unified_attention, mha_v3, whose backward raises NotImplementedError).
        # Enabling forward alone would train capped logits against uncapped gradients
        # -- the exact silent-drop failure this layer exists to prevent. So the gate
        # stays until a trainable fwd+bwd pair supports it.
        "softcap": "detected_or_explicit_but_blocked_no_trainable_aiter_fwd_bwd_pair_implements_softcap",
        # fp8 lands on the Triton kernel family, which has no sink parameter, asserts
        # bias is None and window_size == (-1,-1), has no dropout_p in its backward,
        # never reads `deterministic`, emits a [B,H,2*Sq] LSE, and has no varlen entry.
        "fp8_with_sink_bias_window_dropout_deterministic_lse_or_packing": "explicitly_rejected_triton_fp8_kernel_cannot_honour_them",
        # A bidirectional band is exact on the csrc/CK route but not on the sink route:
        # that backward calls triton_flash_attn_onekernel_backward with
        # sliding_window=window_size_left, dropping the right edge, so it would
        # differentiate a wider mask than the forward computed. Refused, not approximated.
        "bidirectional_window_with_sink": "sink_backward_takes_left_window_only_right_edge_would_be_dropped",
        # Same shape of gap on the dense route: FlashAttnFunc.forward itself asserts
        # deterministic and sink are never on together (no deterministic dQ accumulation
        # in the sink backward). Caught here so the error names the real culprit.
        "deterministic_with_sink_on_dense_route": "backend_asserts_no_deterministic_dq_accumulation_for_sink",
        # Packing + sink: this is a *wrapper* gap, not a kernel gap, and the difference
        # matters to whoever goes to close it. On this build aiter's varlen kernels DO
        # carry a sink on both sides (_flash_attn_varlen_forward has sink_size/sink_ptr,
        # _flash_attn_varlen_backward has sink/d_sink); what lacks it is Primus-Turbo's
        # own flash_attn_varlen_func wrapper, which never exposes the parameter. The
        # other varlen backend that carries one (FlyDSL) requires equal segment lengths.
        # So ragged documents with a sink raise at the flex layer, naming the constraint,
        # rather than reaching a "no compatible backend" error deeper down -- and the fix
        # is to forward sink through the Turbo wrapper, not to write a kernel.
        "sink_with_ragged_document_packing": "turbo_varlen_wrapper_does_not_forward_aiters_sink_and_flydsl_needs_uniform_seglens",
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
            "deterministic",
            "return_lse",
        ],
        "document_masking": "explicit_cu_seqlens_block_diagonal_plus_causal_true",
        "unsupported": [
            "arbitrary_score_mod_no_such_arg",
            "softcap_gt_0_gated",
            "bias",
            # No fp8 varlen entry exists in Primus-Turbo (flash_attn_fp8_func is
            # dense-only), so packed fp8 has nothing to lower onto.
            "fp8",
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
