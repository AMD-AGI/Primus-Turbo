###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

###############################################################################

import functools
import math

import scipy
import torch


def get_random_sign_vector(device: int) -> torch.Tensor:
    """Hard-coded random signs for Hadamard transform.

    https://xkcd.com/221/

    """

    # fmt: off
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=device,
    )
    # fmt: on


def get_hadamard_matrix(hadamard_dimension: int, device: int) -> torch.Tensor:
    """Construct a 32x32 Hadamard matrix."""
    assert hadamard_dimension == 32, "Only hadamard dimension 32 is supported."
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    # fmt: off
    return (
        torch.tensor(scipy.linalg.hadamard(hadamard_dimension), dtype=torch.float32, device=device)
        * hadamard_scale
    )
    # fmt: on


@functools.lru_cache(maxsize=None)
def get_rht_matrix(dtype: torch.dtype, device: int) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    hadamard_dimension = 32
    signs = get_random_sign_vector(device=device)
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=torch.float32, device=device)
    rht_matrix = sign_matrix @ get_hadamard_matrix(hadamard_dimension, device=device)

    return rht_matrix.to(dtype=dtype)
