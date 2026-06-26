###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused cross-rank dispatch PUSH + grouped BF16 GEMM (NT) as a torch custom op.

Thin wrapper over the FlyDSL kernel ``dispatch_grouped_gemm_bf16``. The kernel now
fetches the active symmetric workspace (pool GEMM-A operand + peer pool / scoreboard
delta tables) internally via ``get_symm_buffer_for_mega_moe()`` and reads the comm
plan + ``tile_to_expert`` / ``expected_count`` from the bundled ``handle`` -- so the
long buffer-list ABI is gone. ``pool`` is kept only to size the output ([pool_cap, N])
and key the dispatcher / declare the in-place mutation; the kernel reads it from the
active symm buffer, not from this arg.

The public wrapper owns the prologue + symm fetch: in forward (``handle=None``) it builds
the active symm workspace and runs ``dispatch_prologue``, returning ``(l1_out, handle)``
where ``handle`` (a :class:`MegaDispatchHandle`) carries the plan + tile tables + the live
symm buffer; the backward NN dgrad re-feeds that handle to reuse the plan.
"""

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


def _flydsl_prologue_and_symm():
    """Lazy import — keep this module importable when FlyDSL is absent."""
    from primus_turbo.flydsl.mega.dispatch_prologue_kernel import dispatch_prologue
    from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_for_mega_moe

    return dispatch_prologue, get_symm_buffer_for_mega_moe


class MegaDispatchHandle:
    """Carries the prologue plan + tile tables + the active symm buffer.

    Built by the forward dispatch (which runs the fused prologue and fetches the active
    symm workspace); consumed by the combine GEMM and re-fed to the backward dispatch,
    which reuses the SAME plan instead of re-running the prologue. ``plan`` is the device
    comm-ABI tuple, ``symm`` is the live :class:`SymmBuffer`."""

    __slots__ = ("plan", "tile_to_expert", "tile_expected", "symm")

    def __init__(self, plan, tile_to_expert, tile_expected, symm):
        self.plan = plan
        self.tile_to_expert = tile_to_expert
        self.tile_expected = tile_expected
        self.symm = symm


class DispatchGroupedGEMMFlyDSLBackend(KernelBackend):
    """FlyDSL fused dispatch + grouped BF16 GEMM (nt fwd / nn dgrad)."""

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
        handle: list,
        weight: torch.Tensor,
        pool: torch.Tensor,
        layout: str,
        BM: int,
        BN: int,
        GROUP_M: int,
        num_dispatch_cu: int,
        autotune: bool,
        **kwargs,
    ) -> torch.Tensor:
        kernel = _flydsl_kernel()
        # group=None -> kernel uses the active symm buffer; tile_to_expert / expected
        # ride the handle tail (handle[-2] / handle[-1]).
        return kernel(
            x,
            weight,
            None,
            handle=handle,
            layout=layout,
            BM=int(BM),
            BN=int(BN),
            GROUP_M=int(GROUP_M),
            num_dispatch_cu=int(num_dispatch_cu),
            autotune=bool(autotune),
        )


_DISPATCH_GROUPED_GEMM_BACKENDS = {
    # autotune is kernel-internal; skip framework-level backend profiling
    BackendType.FLYDSL: BackendEntry(DispatchGroupedGEMMFlyDSLBackend, autotune=False),
}


class DispatchGroupedGEMMKernelDispatcher(AutoKernelDispatcher):
    _backends = _DISPATCH_GROUPED_GEMM_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(cls, x, weight, pool, BM, BN, GROUP_M, layout, **kwargs):
        G, N, K = weight.shape
        pool_capacity = pool.shape[0]
        return (G, N, K, pool_capacity, BM, BN, GROUP_M, x.dtype, layout)


_torch_custom_op_wrapper = torch.library.custom_op


@_torch_custom_op_wrapper(
    "primus_turbo::dispatch_grouped_gemm",
    mutates_args=("pool",),
    device_types="cuda",
)
def _dispatch_grouped_gemm(
    x: torch.Tensor,
    handle: list[torch.Tensor],
    weight: torch.Tensor,
    pool: torch.Tensor,
    default_backend: int,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    num_dispatch_cu: int = 16,
    autotune: bool = False,
) -> torch.Tensor:
    """Custom-op core: fused cross-rank dispatch PUSH + grouped BF16 GEMM. Returns C [pool_cap, N].

    Returns only the GEMM output; ``tile_to_expert`` / ``expected_count`` ride the input
    ``handle`` (they are inputs, so a custom op cannot return them without aliasing). The
    public :func:`dispatch_grouped_gemm_impl` wrapper re-bundles them into the output handle.
    """
    default_backend_enum = BackendType(default_backend)

    # kernel autotune: explicit flag OR global auto-tune, never under capture
    do_autotune = autotune or GlobalBackendManager.auto_tune_enabled()
    do_autotune = do_autotune and not AutoKernelDispatcher._is_graph_capturing()

    kwargs = dict(
        x=x,
        handle=handle,
        weight=weight,
        pool=pool,
        layout=layout,
        BM=BM,
        BN=BN,
        GROUP_M=GROUP_M,
        num_dispatch_cu=num_dispatch_cu,
        autotune=do_autotune,
    )

    return DispatchGroupedGEMMKernelDispatcher.dispatch(default_backend_enum, None, **kwargs)


@_dispatch_grouped_gemm.register_fake
def _dispatch_grouped_gemm_meta(
    x: torch.Tensor,
    handle: list[torch.Tensor],
    weight: torch.Tensor,
    pool: torch.Tensor,
    default_backend: int,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    num_dispatch_cu: int = 16,
    autotune: bool = False,
) -> torch.Tensor:
    assert x.dim() == 2, f"x must be 2D, got {x.shape}"
    assert weight.dim() == 3, f"weight must be 3D, got {weight.shape}"
    assert pool.dim() == 2, f"pool must be 2D [pool_cap,K], got {pool.shape}"
    assert x.dtype in _SUPPORTED_DTYPES, f"x must be bf16, got {x.dtype}"
    assert weight.dtype in _SUPPORTED_DTYPES, f"weight must be bf16, got {weight.dtype}"
    assert layout in ("nt", "nn"), "Only layout='nt' (fwd) / 'nn' (dgrad) supported."
    k_idx = 2 if layout == "nt" else 1
    assert weight.shape[k_idx] == x.shape[1], "weight K must match hidden size."

    N = weight.shape[1] if layout == "nt" else weight.shape[2]
    return torch.empty((pool.shape[0], N), device=x.device, dtype=x.dtype)


def dispatch_grouped_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    default_backend: int,
    *,
    group=None,
    topk_idx: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    handle: MegaDispatchHandle | None = None,
    layout: str = "nt",
    BM: int = 256,
    BN: int = 256,
    GROUP_M: int = 4,
    num_dispatch_cu: int = 16,
    pool_mult: int = 2,
    autotune: bool = False,
) -> tuple[torch.Tensor, MegaDispatchHandle]:
    """Fused cross-rank dispatch PUSH + grouped BF16 GEMM (nt/nn), prologue included.

    Two modes:
      * forward (``handle=None``): fetch/create the active symm workspace via
        ``get_symm_buffer_for_mega_moe(group, ...)`` (sized from the tensor shapes), run the
        fused ``dispatch_prologue`` to build the comm plan + ``tile_to_expert`` /
        ``expected_count``, then dispatch PUSH + grouped GEMM. ``group`` + ``topk_idx``
        (and optionally ``topk_weights``) are required; ``layout`` must be ``nt``.
      * backward / reuse (``handle`` given): skip the prologue and reuse the handle's plan
        (the backward NN dgrad rides the forward's plan).

    Returns ``(l1_out, handle)``: ``l1_out`` is C [pool_cap, N]; ``handle`` is a
    :class:`MegaDispatchHandle` carrying the plan + tile tables + the live symm buffer
    (feed it back for the backward dispatch / read ``handle.symm`` downstream). The symm
    workspace is fetched internally by the kernel; cross-rank writes are raw-pointer and
    untracked by autograd.
    """
    if handle is None:
        # forward: build the active symm workspace + run the fused prologue
        dispatch_prologue, get_symm_buffer_for_mega_moe = _flydsl_prologue_and_symm()
        assert (
            group is not None and topk_idx is not None
        ), "forward dispatch (handle=None) requires group + topk_idx"
        assert layout == "nt", "handle=None auto-prologue is forward-only (nt); pass handle for nn"
        experts_per_rank = weight.shape[0]
        num_tokens, hidden = x.shape
        num_topk = topk_idx.shape[-1]
        symm = get_symm_buffer_for_mega_moe(
            group,
            num_experts=experts_per_rank * group.size(),
            num_max_tokens_per_rank=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=weight.shape[1] // 2,  # w1 [epr, 2I, H]
            block_m=BM,
            block_n=BN,
            pool_mult=pool_mult,
        )
        plan, tile_to_expert, tile_expected, *_ = dispatch_prologue(
            topk_idx,
            topk_weights,
            sym_layout=symm.make_sym_layout(),
            num_tokens=num_tokens,
            num_topk=num_topk,
            num_experts=symm.num_experts,
            world_size=symm.world,
            rank=symm.rank,
            experts_per_rank=experts_per_rank,
            block_m=BM,
            pool_capacity=symm.pool_capacity,
        )
        handle = MegaDispatchHandle(list(plan), tile_to_expert, tile_expected, symm)

    symm = handle.symm
    # device ABI: comm plan + tile_to_expert / expected at the tail (kernel reads handle[-2:])
    in_handle = [*handle.plan, handle.tile_to_expert, handle.tile_expected]
    l1_out = _dispatch_grouped_gemm(
        x,
        in_handle,
        weight,
        symm.pool,
        default_backend,
        layout=layout,
        BM=BM,
        BN=BN,
        GROUP_M=GROUP_M,
        num_dispatch_cu=num_dispatch_cu,
        autotune=autotune,
    )
    return l1_out, handle
