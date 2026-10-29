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

"""``@autotune(configs=..., key=...)`` on a ``@flyc.jit`` function: flydsl's own autotuner (in its
EP-safe form from ``mega/tune_utils.py``) made safe to call first under CUDA-graph capture.
Independent of ``PRIMUS_TURBO_AUTO_TUNE`` and of the builder ops' engine."""

import torch

from primus_turbo.flydsl.mega.tune_utils import Autotuner as _NativeAutotuner
from primus_turbo.flydsl.mega.tune_utils import Config

__all__ = ["Config", "Autotuner", "autotune"]


class Autotuner(_NativeAutotuner):
    """Timing synchronizes the device, which a capturing stream refuses. Under capture a key not
    tuned yet runs ``configs[0]`` (after the prune) and caches nothing, so the first call outside
    capture tunes it."""

    def __call__(self, *args, **kwargs):
        if not torch.cuda.is_current_stream_capturing():
            return super().__call__(*args, **kwargs)
        cfg = self.cache.get(self._make_key(args, kwargs))
        if cfg is None:
            cfg = self._prune(self.configs, args, kwargs)[0]
        return self._run_config(cfg, args, kwargs)


def autotune(
    configs,
    key=None,
    warmup=5,
    rep=25,
    prune_configs_by=None,
    reset_to_zero=None,
    pre_hook=None,
    post_hook=None,
    do_bench=None,
):
    """Decorator: wrap a ``@flyc.jit`` function in :class:`Autotuner` (``mega.tune_utils.autotune``'s
    signature)."""

    def decorator(fn):
        return Autotuner(
            fn,
            configs,
            key,
            warmup,
            rep,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            pre_hook=pre_hook,
            post_hook=post_hook,
            do_bench_fn=do_bench,
        )

    return decorator
