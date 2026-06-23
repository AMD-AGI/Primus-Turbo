###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)

_SUPPORTED_DTYPES = (torch.bfloat16,)


def _flydsl_kernel():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.dispatch_grouped_gemm_bf16_kernel import (
        dispatch_grouped_gemm_bf16,
    )

    return dispatch_grouped_gemm_bf16


class DispatchGroupedGEMMFlyDSLBackend(KernelBackend):
    """FlyDSL fused dispatch + grouped BF16 GEMM (NT only)."""

    @staticmethod
    def can_handle(
        x: torch.Tensor,
        weight: torch.Tensor,
        pool: torch.Tensor,
        layout: str,
        **kwargs,
    ) -> bool:
        supported = True
        supported &= x.dim() == 2 and weight.dim() == 3 and pool.dim() == 2
        supported &= x.dtype in _SUPPORTED_DTYPES and weight.dtype in _SUPPORTED_DTYPES
        supported &= layout in ("nt", "nn")  # pool is M-major -> nt (fwd) / nn (dgrad)
        # K (contraction) = hidden: NT weight [G,N,K]; NN weight [G,K,N]
        k_idx = 2 if layout == "nt" else 1
        supported &= weight.shape[k_idx] == x.shape[1]
        return supported

    @staticmethod
    def execute(
        x: torch.Tensor,
        comm_dest: torch.Tensor,
        comm_start: torch.Tensor,
        comm_count: torch.Tensor,
        comm_src_offset: torch.Tensor,
        comm_src_tokens: torch.Tensor,
        pool: torch.Tensor,
        pool_ptrs: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        tile_to_group: torch.Tensor,
        scoreboard: torch.Tensor,
        scoreboard_ptrs: torch.Tensor,
        expected_count: torch.Tensor,
        num_tile_blocks: torch.Tensor,
        num_comm: int,
        layout: str,
        BM: int,
        BN: int,
        GROUP_M: int,
        num_dispatch_cu: int,
        nt_vmcnt: int,
        autotune: bool,
        **kwargs,
    ) -> torch.Tensor:
        kernel = _flydsl_kernel()
        # DeepEP-style dispatch handle (flat tuple); the GEMM ignores routing -> first 5 suffice
        plan = (comm_dest, comm_start, comm_count, comm_src_offset, comm_src_tokens)
        return kernel(
            x,
            plan,
            pool,
            pool_ptrs,
            weight,
            output,
            tile_to_group,
            scoreboard,
            scoreboard_ptrs,
            expected_count,
            num_tile_blocks,
            layout=layout,
            BM=int(BM),
            BN=int(BN),
            GROUP_M=int(GROUP_M),
            num_dispatch_cu=int(num_dispatch_cu),
            nt_vmcnt=int(nt_vmcnt),
            autotune=bool(autotune),
            autotune_reset=scoreboard.zero_,
        )


_DISPATCH_GROUPED_GEMM_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(DispatchGroupedGEMMFlyDSLBackend, autotune=False),
}


class DispatchGroupedGEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _DISPATCH_GROUPED_GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, x, weight, pool, BM, BN, GROUP_M, num_comm, layout, **kwargs):
        G, N, K = weight.shape
        pool_capacity = pool.shape[0]
        return (G, N, K, pool_capacity, BM, BN, GROUP_M, int(num_comm), x.dtype, layout)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper(
    "primus_turbo::dispatch_grouped_gemm_impl",
    mutates_args=("pool", "scoreboard"),
    device_types="cuda",
)
def dispatch_grouped_gemm_impl(
    x: torch.Tensor,
    comm_dest: torch.Tensor,
    comm_start: torch.Tensor,
    comm_count: torch.Tensor,
    comm_src_offset: torch.Tensor,
    comm_src_tokens: torch.Tensor,
    pool: torch.Tensor,
    pool_ptrs: torch.Tensor,
    weight: torch.Tensor,
    tile_to_group: torch.Tensor,
    scoreboard: torch.Tensor,
    scoreboard_ptrs: torch.Tensor,
    expected_count: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    num_comm: int,
    default_backend: int,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    num_dispatch_cu: int = 16,
    nt_vmcnt: int = 3,
    autotune: bool = False,
) -> torch.Tensor:
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM (NT). Returns C [pool_cap, N].

    ``pool``/``scoreboard`` are symmetric-memory buffers mutated in place; the
    caller MUST zero ``scoreboard`` before each call. Cross-rank writes via
    ``pool_ptrs``/``scoreboard_ptrs`` are raw-pointer and untracked by autograd.
    """
    default_backend_enum = BackendType(default_backend)

    # C freshly allocated; pool/scoreboard are mutated symmetric-mem buffers.
    # NT weight [G,N,K] -> N=shape[1]; NN (dgrad) weight [G,K,N] -> N=shape[2].
    N = weight.shape[1] if layout == "nt" else weight.shape[2]
    output = torch.empty((pool.shape[0], N), device=x.device, dtype=x.dtype)

    # kernel autotune: explicit flag OR global auto-tune, never under capture
    do_autotune = autotune or GlobalBackendManager.auto_tune_enabled()
    do_autotune = do_autotune and not AutoKernelDispatcher._is_graph_capturing()

    kwargs = dict(
        x=x,
        comm_dest=comm_dest,
        comm_start=comm_start,
        comm_count=comm_count,
        comm_src_offset=comm_src_offset,
        comm_src_tokens=comm_src_tokens,
        pool=pool,
        pool_ptrs=pool_ptrs,
        weight=weight,
        output=output,
        tile_to_group=tile_to_group,
        scoreboard=scoreboard,
        scoreboard_ptrs=scoreboard_ptrs,
        expected_count=expected_count,
        num_tile_blocks=num_tile_blocks,
        num_comm=num_comm,
        layout=layout,
        BM=BM,
        BN=BN,
        GROUP_M=GROUP_M,
        num_dispatch_cu=num_dispatch_cu,
        nt_vmcnt=nt_vmcnt,
        autotune=do_autotune,
    )

    DispatchGroupedGEMMKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)
    return output


@dispatch_grouped_gemm_impl.register_fake
def dispatch_grouped_gemm_impl_meta(
    x: torch.Tensor,
    comm_dest: torch.Tensor,
    comm_start: torch.Tensor,
    comm_count: torch.Tensor,
    comm_src_offset: torch.Tensor,
    comm_src_tokens: torch.Tensor,
    pool: torch.Tensor,
    pool_ptrs: torch.Tensor,
    weight: torch.Tensor,
    tile_to_group: torch.Tensor,
    scoreboard: torch.Tensor,
    scoreboard_ptrs: torch.Tensor,
    expected_count: torch.Tensor,
    num_tile_blocks: torch.Tensor,
    num_comm: int,
    default_backend: int,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    num_dispatch_cu: int = 16,
    nt_vmcnt: int = 3,
    autotune: bool = False,
) -> torch.Tensor:
    assert x.dim() == 2, f"x must be 2D, got {x.shape}"
    assert weight.dim() == 3, f"weight must be 3D [G,N,K], got {weight.shape}"
    assert pool.dim() == 2, f"pool must be 2D [pool_cap,K], got {pool.shape}"
    assert x.dtype in _SUPPORTED_DTYPES, f"x must be bf16, got {x.dtype}"
    assert weight.dtype in _SUPPORTED_DTYPES, f"weight must be bf16, got {weight.dtype}"
    assert layout in ("nt", "nn"), "Only layout='nt' (fwd) / 'nn' (dgrad) supported."
    k_idx = 2 if layout == "nt" else 1
    assert weight.shape[k_idx] == x.shape[1], "weight K must match hidden size."

    N = weight.shape[1] if layout == "nt" else weight.shape[2]
    return torch.empty((pool.shape[0], N), device=x.device, dtype=x.dtype)
