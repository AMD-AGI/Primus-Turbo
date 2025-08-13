###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.lax.normalization import rmsnorm


def rmsnorm_ref(x, gamma, eps):
    norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / norm * gamma


@pytest.mark.parametrize("shape", [(1024, 4096)])
@pytest.mark.parametrize("dtype", [jnp.float32])
def test_rmsnorm_lax(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype)
    gamma = jax.random.normal(key, (shape[-1],), dtype)
    eps = 1e-6

    #######################################
    # Fwd
    y = rmsnorm(x, gamma, eps)
    y_ref = rmsnorm_ref(x, gamma, eps)

    print("y ", y)
    print("y_ref ", y_ref)
    np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)

    #######################################
    # Backward w.r.t both x and gamma
    def loss_fn(x, gamma):
        return jnp.sum(rmsnorm(x, gamma, eps))

    def loss_fn_ref(x, gamma):
        return jnp.sum(rmsnorm_ref(x, gamma, eps))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_x, grad_gamma = grad_fn(x, gamma)
    grad_x_ref, grad_gamma_ref = grad_fn_ref(x, gamma)

    print("grad_x ", grad_x)
    print("grad_x_ref ", grad_x_ref)
    np.testing.assert_allclose(grad_x, grad_x_ref, rtol=1e-4, atol=1e-4)

    print("grad_gamma ", grad_gamma)
    print("grad_gamma_ref ", grad_gamma_ref)
    np.testing.assert_allclose(grad_gamma, grad_gamma_ref, rtol=1e-4, atol=1e-4)
