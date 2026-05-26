###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""User-facing DeepEP MoE API.

Public surface (8 symbols) — everything else is internal:

  setup            One-call DeepEP bootstrap, required before first
                   moe_dispatch / moe_combine.
  moe_dispatch     All-to-all token dispatch to selected experts.
  moe_combine      All-to-all expert-output combine (inverse of dispatch).
  Config           Tuning knobs (num_sms, chunked send/recv tokens).
  set_ep_group     Pin EP group without a ``jax.sharding.Mesh`` (advanced).
  get_ep_size      Query the EP-group size frozen by ``setup()``.
  is_internode     Query whether the EP group spans multiple nodes
                   (frozen at ``setup()`` time; used by handle-interpretation
                   branches in framework code).
  reset_runtime    Reset all DeepEP state (tests / multi-config jobs).

See ``setup`` docstring for the full bootstrap contract; see
``moe_dispatch`` for the in-graph API.
"""

from .moe_dispatch_combine import (
    Config,
    get_ep_size,
    is_internode,
    moe_combine,
    moe_dispatch,
    reset_runtime,
    set_ep_group,
    setup,
)

__all__ = [
    "Config",
    "get_ep_size",
    "is_internode",
    "moe_combine",
    "moe_dispatch",
    "reset_runtime",
    "set_ep_group",
    "setup",
]
