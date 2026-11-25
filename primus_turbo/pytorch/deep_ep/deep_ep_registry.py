###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel registry of DeepEP (see ``primus_turbo.registry``). Data only."""

KERNELS = (
    {
        "name": "deep_ep.flydsl",
        # One source for every architecture, parameterized by primus_turbo/flydsl/deep_ep/arch.py, so
        # the package carries no architecture tag; gfx1250 is the architecture that runs it
        # (FLYDSL_DEEPEP_ARCHS there; gfx942 / gfx950 use the C++ DeepEP). No dtype: it moves rows of
        # any format.
        "summary": "FlyDSL intranode DeepEP: dispatch and combine between the GPUs of one node",
        "arch": ("gfx1250",),
        "backend": "flydsl",
        "kernel": "primus_turbo.flydsl.deep_ep",
        "impl": "primus_turbo.pytorch.deep_ep.flydsl_intranode",
        "gates": ("primus_turbo.flydsl.deep_ep.arch:default_intranode_backend",),
        "tests": (
            "tests/pytorch/deep_ep/test_intranode.py",
            "tests/pytorch/deep_ep/test_module_layering.py",
        ),
    },
)
