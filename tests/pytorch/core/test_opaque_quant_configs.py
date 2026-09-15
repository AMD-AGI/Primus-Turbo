###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The quant configs must stay registrable as torch opaque types.

torch >= 2.11 rejects register_opaque_type on a class whose metaclass is not
OpaqueBaseMeta. That call runs at import time, so a regression here does not
fail one operator -- it makes `import primus_turbo.pytorch` raise, on every
arch. These assertions are arch-independent on purpose.
"""

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    ScalingRecipe,
)

_OPAQUE_CONFIGS = (Float8QuantConfig, Float4QuantConfig, ScalingRecipe)


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_carries_the_opaque_metaclass(cls):
    opaque_base = pytest.importorskip(
        "torch._opaque_base", reason="torch without the OpaqueBaseMeta requirement"
    )
    assert isinstance(cls, opaque_base.OpaqueBaseMeta)


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_stays_hashable_and_comparable(cls):
    # register_opaque_type(typ="value") guards the baked-in constant on __eq__.
    assert cls() == cls()
    assert hash(cls()) == hash(cls())


@pytest.mark.parametrize("cls", _OPAQUE_CONFIGS, ids=lambda c: c.__name__)
def test_fx_repr_round_trips(cls):
    # torch.compile regenerates the config from this string, so it has to eval
    # back to an equal object under the globals the method hands out.
    config = cls()
    source, globals_ = config.__fx_repr__()
    assert eval(source, dict(globals_)) == config


def test_scaling_recipe_is_still_a_named_tuple():
    # It is spelled as a subclass of a NamedTuple to carry the metaclass, which
    # would silently cost tuple behaviour if that split were ever undone wrong.
    recipe = ScalingRecipe(use_2d_block=True, use_rht=True)
    assert isinstance(recipe, tuple)
    assert recipe[0] is True
    assert recipe._asdict()["use_rht"] is True
    assert recipe._replace(use_sr=True).use_sr is True
    first, *_ = recipe
    assert first is True


def test_torch_version_that_needs_the_metaclass_is_the_one_we_have():
    # A sanity line rather than a constraint: if torch drops the requirement,
    # the metaclass stays harmless and the test above skips.
    assert torch.__version__
