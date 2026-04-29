# Description

Fix DeepEP MoE combine hang by increasing chunked recv token buffer sizes, migrating the `Config` class from a Python NamedTuple to C++ pybind11 binding (exposing buffer size hint methods), and correcting a byte-width calculation bug in intranode dispatch.

Fixes # (issue)

## Type of change

- [ ] Documentation change (change only to the documentation, either a fix or a new content)
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Infra/Build change
- [x] Code refactoring

## Changes

- Fix byte-width calculation in `IntranodeDispatch` contiguous check (`element_count()` → `ByteWidth(element_type())`)
- Expose `Config` class and `get_low_latency_rdma_size_hint` from C++ via pybind11, replacing the Python NamedTuple
- Increase `num_max_nvl_chunked_recv_tokens` and `num_max_rdma_chunked_recv_tokens` to 512 in both dispatch and combine config maps to avoid combine hang
- Export `Config` and `set_default_num_sms` from `primus_turbo.jax.lax.moe`
- Fix `get_combine_config` docstring (was incorrectly saying "dispatch config")

# Checklist:

- [x] The functionality is complete
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
