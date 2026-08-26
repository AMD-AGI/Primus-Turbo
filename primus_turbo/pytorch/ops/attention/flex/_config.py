###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tunable constants and tolerances shared by the flex compat layer.

Pure data: no imports beyond ``torch`` for the dtype set, and no dependency on any
other ``flex`` module. Everything here is a knob that governs *how hard* the
classifier looks at a ``mask_mod`` / ``score_mod`` before giving up.
"""

import torch

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


# Upper bound on the probe grid edge. 512 comfortably covers the sliding-window
# sizes we can express directly; larger windows on longer sequences are located by
# a binary search on the last query row (see _locate_left_window) rather than
# silently mis-classified.
_MASK_PROBE_LIMIT = 512


# Document packing longer than the probe grid is recovered from mask_mod directly
# (see _locate_document_segments). Verification there is *exact*, which costs O(S^2)
# bool comparisons, so it is bounded: beyond this sequence length we decline to
# classify rather than downgrade to sampled verification.
_DOC_EXACT_VERIFY_LIMIT = 16384


# Rows per vectorised mask_mod call during that exact verification (keeps peak
# memory at chunk*S bools instead of S*S).
_DOC_VERIFY_CHUNK = 256


_ALIBI_TOL = 5e-3


# The ALiBi sign convention this layer assumes: Turbo's positive ``alibi_slopes``
# behaves like flex's ``+slope*(kv-q)``. Empirically resolved on rocm/primus:v26.5;
# ``check_alibi_sign_convention()`` re-validates it on any other build.
_ASSUMED_ALIBI_SIGN = 1.0


# Relative tolerance for recognising a logits soft-cap (cap*tanh(score/cap)).
_SOFTCAP_TOL = 1e-2
