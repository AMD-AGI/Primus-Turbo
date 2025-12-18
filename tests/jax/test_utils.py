###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.core.low_precision import is_fp8_dtype


def skip_if_lt_x_gpu(min_gpus):
    """Decorator to skip test if less than min_gpus are available."""
    reason = f"Test requires at least {min_gpus} GPUs, but only {jax.local_device_count()} available"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if jax.local_device_count() < min_gpus:
                pytest.skip(reason)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_tolerances(dtype):
    """Get relative and absolute tolerances for different dtypes."""
    if dtype == jnp.float32:
        return dict(rtol=1e-4, atol=1e-4)
    elif dtype == jnp.float16:
        return dict(rtol=1e-2, atol=1e-2)
    elif dtype == jnp.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    elif is_fp8_dtype(dtype):
        return dict(rtol=1e-1, atol=1e-1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


###################################################################


# Relative Error
# Note: x is ref
def relative_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute relative error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.linalg.norm(x - y) / jnp.linalg.norm(x))


# MSE Error
def mean_squared_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute mean squared error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.mean((x - y) ** 2))


# Max Abs Error
def max_abs_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute maximum absolute error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.max(jnp.abs(x - y)))


# Cosine Similarity
def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray):
    """Compute cosine similarity between two arrays."""
    x, y = x.flatten().astype(jnp.float32), y.flatten().astype(jnp.float32)
    dot_product = jnp.dot(x, y)
    norm_x = jnp.linalg.norm(x)
    norm_y = jnp.linalg.norm(y)
    return float(dot_product / (norm_x * norm_y))


# Symmetric Similarity
def symmetric_similarity_diff(x: jnp.ndarray, y: jnp.ndarray):
    """Compute symmetric similarity difference between two arrays."""
    x, y = x.astype(jnp.float64), y.astype(jnp.float64)
    denominator = jnp.sum(x * x + y * y)
    sim = 2 * jnp.sum(x * y) / denominator
    return float(1 - sim)


# SNR
# Note: x is ref
def compute_snr(x: jnp.ndarray, y: jnp.ndarray):
    """Compute Signal-to-Noise Ratio in dB. x is reference."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    signal_power = jnp.linalg.norm(x) ** 2
    noise_power = jnp.linalg.norm(x - y) ** 2
    snr = 10 * jnp.log10(signal_power / (noise_power + 1e-12))
    return float(snr)


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert two arrays are close. Auto-converts to float32 for numpy compatibility."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
