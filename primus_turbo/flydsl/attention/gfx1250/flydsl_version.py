###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""flydsl release the gfx1250 attention kernels need.

The kernels use FlyDSL APIs that earlier releases lack (``fx.ceildiv``, integer ``fx.max`` /
``fx.min``, ``fx.to_llvm_ptr``, ``fx.gpu.shuffle_xor``, the async LDS-to-global store) and
gfx1250 ops whose API is not stable across releases (the Tensor Data Mover atom and wait,
``s_wait_dscnt``, ``ds_load_tr16_b128``). FLYDSL_REQUIREMENT admits the 0.3.4.x patch releases
from 0.3.4.1 on; the generated code has been verified on 0.3.4.1.
``flydsl_unavailable_reason`` lets the attention gate decline, so the dispatcher falls back to
another backend, when the importable flydsl is missing or another release. Every module of this
package that imports flydsl calls ``require_flydsl`` first, so a direct import fails with an
ImportError that says why. This module imports nothing but the standard library at import time.
"""

import re
from typing import Optional

FLYDSL_REQUIREMENT = ">=0.3.4.1,<0.3.5"
_MIN, _END = (0, 3, 4, 1), (0, 3, 5)
# version string -> satisfies FLYDSL_REQUIREMENT. A plain dict, not functools.lru_cache: the gate
# runs inside torch.compile'd regions, and Dynamo warns on (and ignores) lru_cache wrappers.
_CHECKED = {}


def _version_ok(version: str) -> bool:
    ok = _CHECKED.get(version)
    if ok is None:
        ok = _CHECKED[version] = _satisfies(version)
    return ok


def _satisfies(version: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:  # no packaging: compare the release numbers only
        m = re.match(r"(\d+(?:\.\d+)*)", version)
        if m is None:
            return False
        release = tuple(int(x) for x in m.group(1).split("."))
        return _MIN <= release < _END
    try:
        # prereleases=True: a dev build after 0.3.4.1 is fine; PEP 440 still keeps
        # 0.3.4.1.dev0 / 0.3.4.1rc1 (before 0.3.4.1) and 0.3.5 pre-releases out, and ignores
        # local labels (+g1234567).
        return SpecifierSet(FLYDSL_REQUIREMENT).contains(Version(version), prereleases=True)
    except InvalidVersion:
        return False


def flydsl_unavailable_reason() -> Optional[str]:
    """None when the importable flydsl satisfies FLYDSL_REQUIREMENT, else why it does not."""
    try:
        import flydsl
    except ImportError as exc:
        return (
            f"the gfx1250 attention kernels need flydsl{FLYDSL_REQUIREMENT}; flydsl is not importable ({exc})"
        )
    version = getattr(flydsl, "__version__", None)
    if not isinstance(version, str) or not _version_ok(version):
        return f"the gfx1250 attention kernels need flydsl{FLYDSL_REQUIREMENT}, found flydsl {version}"
    return None


def require_flydsl() -> None:
    """Raise ImportError unless the importable flydsl satisfies FLYDSL_REQUIREMENT."""
    reason = flydsl_unavailable_reason()
    if reason is not None:
        raise ImportError(reason)
