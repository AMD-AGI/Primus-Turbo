# `deep_ep` elastic (V2) device kernels -- placeholder

Reserved for the upstream DeepEP V2 ("elastic") kernels, so they can sit beside
`../legacy/` the way they do upstream rather than displacing it.

Upstream source: `deepseek-ai/DeepEP` `csrc/kernels/elastic/`
(`api.hpp`, `barrier.hpp`, `combine.hpp`, `dispatch.hpp`, `engram.hpp`,
`pp_send_recv.hpp`), plus `csrc/kernels/backend/` (`nvshmem.cu`, `nccl.cu`,
`cuda_driver.cu`, `symmetric.hpp`) and the JIT runtime under `csrc/jit/`.

Not copied yet: elastic is header-only templates driven by a JIT compiler and
the NCCL Device API, so bringing it over pulls in `csrc/jit/` and
`csrc/kernels/backend/` as well. That is a separate step from the V1 port.
