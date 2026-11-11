###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backends of the Ulysses fused GEMM + all-to-all ops in ``ops/attention/ulysses_fused_a2a.py``.

Laid out like ``kernels/moe/moe_dispatch_combine_impl.py``: one class per backend, each owning its
communication state, a name registry, and ``_resolve_backend_name()``, which reads the code setting
(``GlobalBackendManager.set_ulysses_a2a_backend``), then ``PRIMUS_TURBO_ULYSSES_A2A_BACKEND``, and
defaults to ``FLYDSL``. There is no auto-tune: every rank of the group has to run the same backend.

Only the forward pass is dispatched. The backward pass always runs the FlyDSL kernels, which need
nothing but what every forward saves: ``(x, w_qkv)`` for QKV, and ``(w_o, o_gathered)`` for o_proj,
where ``o_gathered`` is the gathered o_proj input ``[s/n * b, h * d]`` in the output's token order.
"""

import glob
import os
from dataclasses import asdict
from typing import Dict, Optional, Protocol, Tuple, Type

import torch
import torch.distributed as dist

import primus_turbo.common.envs as envs
from primus_turbo.flydsl.ulysses.a2a_gemm_kernel import (
    a2a_gemm_bank_bytes,
    ulysses_a2a_gemm,
)
from primus_turbo.flydsl.ulysses.gemm_a2a_kernel import (
    gemm_a2a_bank_bytes,
    ulysses_gemm_a2a,
)
from primus_turbo.flydsl.ulysses.workspace import (
    REGION_O_A2A_GEMM,
    REGION_O_GEMM_A2A,
    REGION_QKV_A2A_GEMM,
    REGION_QKV_GEMM_A2A,
    get_ulysses_workspace,
)
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)

__all__ = [
    "register_ulysses_a2a_backend",
    "ulysses_qkv_gemm_a2a_fwd_impl",
    "ulysses_qkv_gemm_a2a_bwd_impl",
    "ulysses_a2a_o_gemm_fwd_impl",
    "ulysses_a2a_o_gemm_bwd_impl",
]


def _launch_kwargs(tuning, **defaults):
    if tuning is None:
        return defaults
    return {**defaults, **{k: v for k, v in asdict(tuning).items() if v is not None}}


def _rank_major_concat_rows(hq, hkv, d, n, device):
    """Row order of a [q; k; v] weight that makes each rank's q, k, v heads one contiguous block."""
    q, k, v = (torch.arange(h * d, device=device).view(n, h // n * d) for h in (hq, hkv, hkv))
    return torch.cat((q, k + hq * d, v + (hq + hkv) * d), dim=1).reshape(-1)


_PERMUTED_WEIGHTS = {}


def _rank_major_weight(w, hq, hkv, d, n):
    """``w[perm]`` for a concat-layout weight, rebuilt only when the weight changes."""
    key = (w.data_ptr(), hq, hkv, d, n)
    cached = _PERMUTED_WEIGHTS.get(key)
    if cached is not None and cached[0] == w._version:
        return cached[1]
    permuted = w.detach().index_select(0, _rank_major_concat_rows(hq, hkv, d, n, w.device))
    _PERMUTED_WEIGHTS[key] = (w._version, permuted)
    return permuted


def _qkv_rank_block(hq, hkv, d, n, weight_layout):
    """(G, nq, nk, nv) of one rank's QKV column block, see ``gemm_a2a_kernel``."""
    if weight_layout == "interleaved":
        return hkv // n, hq // hkv, 1, 1
    return 1, hq // n, hkv // n, hkv // n


def _qkv_weight(w_qkv, hq, hkv, d, n, weight_layout):
    """The QKV weight with each rank's columns contiguous (rank-major), as both backends need it."""
    w = w_qkv if weight_layout == "interleaved" else _rank_major_weight(w_qkv, hq, hkv, d, n)
    return w.contiguous()


def _o_shapes(o, w_o, n, qkv_format):
    if qkv_format == "sbhd":
        seq, b, heads, d = o.shape
    else:
        b, seq, heads, d = o.shape
    s_local = seq // n
    N, K = w_o.shape
    assert K == n * heads * d, f"w_o has {K} input features, expected {n} * {heads * d}"
    y_shape = (s_local, b, N) if qkv_format == "sbhd" else (b, s_local, N)
    return seq, b, heads, d, s_local, y_shape


# =========================================================================
# Backend interface
# =========================================================================


class UlyssesA2ABackend(Protocol):
    """Structural interface of a Ulysses fused-forward backend (see the module docstring)."""

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        ...

    def qkv_gemm_a2a(self, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning):
        """Return local-head q, k, v over the full sequence (fresh tensors)."""
        ...

    def a2a_o_gemm(self, o, w_o, residual, group, qkv_format, save_o, tuning):
        """Return ``(y, o_gathered)``; ``o_gathered`` is None unless ``save_o``."""
        ...


# =========================================================================
# FLYDSL: in-tree FlyDSL kernels, data moved by the CUs over XGMI peer memory
# =========================================================================


class FlyDSLUlyssesA2ABackend:
    """In-tree FlyDSL GemmA2A / A2AGemm (always available)."""

    @staticmethod
    def is_available() -> bool:
        return True

    def qkv_gemm_a2a(self, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning):
        n = group.size()
        if qkv_format == "sbhd":
            s_local, b, hidden = x.shape
        else:
            b, s_local, hidden = x.shape
        a = x.reshape(s_local * b, hidden)
        if not a.is_contiguous():
            a = a.contiguous()
        w = _qkv_weight(w_qkv, hq, hkv, d, n, weight_layout)
        M, N = a.shape[0], w.shape[0]
        assert N == (hq + 2 * hkv) * d and w.shape[1] == hidden

        seq = s_local * n
        shape = (seq, b) if qkv_format == "sbhd" else (b, seq)
        q, k, v = (torch.empty(seq * b, h // n, d, dtype=x.dtype, device=x.device) for h in (hq, hkv, hkv))
        workspace = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
        G, nq, nk, nv = _qkv_rank_block(hq, hkv, d, n, weight_layout)
        ulysses_gemm_a2a(
            a,
            w,
            (q, k, v),
            workspace,
            layout="nt",
            head_dim=d,
            groups=G,
            nq=nq,
            nk=nk,
            nv=nv,
            seq_local=s_local,
            qkv_format=qkv_format,
            flag_region=REGION_QKV_GEMM_A2A,
            **_launch_kwargs(tuning, num_pull_cu=24, num_tail_pull=8),
        )
        return tuple(t.view(*shape, *t.shape[1:]) for t in (q, k, v))

    def a2a_o_gemm(self, o, w_o, residual, group, qkv_format, save_o, tuning):
        n = group.size()
        seq, b, heads, d, s_local, y_shape = _o_shapes(o, w_o, n, qkv_format)
        M, K_r = s_local * b, heads * d
        N, K = w_o.shape
        if residual is not None:
            assert tuple(residual.shape) == y_shape, f"residual {tuple(residual.shape)} != {y_shape}"
            residual = residual.reshape(M, N).contiguous()
        y = torch.empty(M, N, dtype=o.dtype, device=o.device)
        o_gathered = torch.empty(M, K, dtype=o.dtype, device=o.device) if save_o else None
        workspace = get_ulysses_workspace(group, a2a_gemm_bank_bytes(M, K))
        ulysses_a2a_gemm(
            o.reshape(seq * b, K_r).contiguous(),
            w_o.contiguous(),
            residual,
            y,
            o_gathered,
            workspace,
            seq_local=s_local,
            qkv_format=qkv_format,
            flag_region=REGION_O_A2A_GEMM,
            **_launch_kwargs(tuning),
        )
        return y.view(y_shape), o_gathered


# =========================================================================
# TENSILELITE: TensileLite GemmA2A / A2AGemm, data moved by SDMA (attn-a2a-demo)
# =========================================================================

_TL_CHANNELS = 2


def _tl_module():
    from attn_demo import _aa

    return _aa


def _tl_library(root: Optional[str], side: str):
    """``(dat, co_dir, co)`` of a TensileLite kernel library built for gfx950."""
    if not root:
        raise ValueError(
            f"TENSILELITE Ulysses backend: PRIMUS_TURBO_ULYSSES_TENSILELITE_LIB_{side} is not set "
            "(path to a TensileCreateLibrary output tuned for these shapes)"
        )
    lib_dir = os.path.join(root, "library", "gfx950")
    co = glob.glob(os.path.join(lib_dir, "TensileLibrary_*Contraction*.co"))
    if len(co) != 1:
        raise ValueError(f"{lib_dir}: expected one Contraction code object, found {co}")
    return os.path.join(lib_dir, "TensileLibrary_lazy_gfx950.dat"), lib_dir, co[0]


class _TLComm:
    """SDMA queues to every peer of one group, and IPC exchange over it."""

    def __init__(self, aa, group):
        self.aa, self.group = aa, group
        self.rank, self.world = group.rank(), group.size()
        devices = [None] * self.world
        dist.all_gather_object(devices, torch.cuda.current_device(), group=group)
        self.devices = devices
        self.ctx = aa.CommContext(self.rank, self.world, devices, -1)
        self.queues = [self.ctx.queue_fields(j) for j in range(self.world)]

    def exchange(self, t):
        handles = [None] * self.world
        dist.all_gather_object(handles, self.aa.ipc_handle(t), group=self.group)
        return [t.data_ptr() if j == self.rank else self.aa.ipc_open(h) for j, h in enumerate(handles)]


class _TLHop:
    """One FusedHop shape with its flags, counter and IPC buffers (adapted from the demo's hops.py)."""

    def __init__(self, comm, mode, m, n, k, extent, lib, beta):
        aa = comm.aa
        self.comm, self.mode, self.m, self.n, self.k = comm, mode, m, n, k
        try:
            self.hop = aa.FusedHop.for_selection(
                comm.rank, comm.world, mode, m, n, k, extent, *lib, beta=beta
            )
        except RuntimeError as e:
            raise ValueError(
                f"TENSILELITE Ulysses backend has no tuned {mode} kernel for m={m} n={n} k={k} "
                f"world={comm.world} extent={extent}: {e}"
            ) from e
        self.flags = aa.alloc(_TL_CHANNELS * aa.FLAG_BLOCK_BYTES, True)
        self.flags.zero_()
        self.counter = aa.alloc(self.hop.counter_bytes(), False)
        self.counter.zero_()
        self.flag_ptrs = comm.exchange(self.flags)
        self.bound = None
        self.launches = 0
        world = comm.world
        if mode == "GemmA2A":
            self.d = torch.empty(n, m, dtype=torch.bfloat16, device="cuda")
            slice_bytes = world * n * (extent // world) * 2
            self._recv = [aa.alloc(slice_bytes, False) for _ in range(_TL_CHANNELS)]
            self._recv_ptrs = [comm.exchange(r) for r in self._recv]
            self.shard = extent // world
        else:
            shard = k // world
            self._x = aa.alloc(world * n * shard * 2, False)
            self.x = self._x.view(torch.bfloat16).view(world, n, shard)
            self._x_ptrs = comm.exchange(self._x)
            self.b = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
            self.d = torch.empty(n, m, dtype=torch.bfloat16, device="cuda")
            self._arrival = aa.alloc(aa.ARRIVAL_BYTES, True)
            self._arrival.zero_()
            self._arrival_ptrs = comm.exchange(self._arrival)
        torch.cuda.synchronize()

    def _bind(self, recv_ptrs, a, b, c=0):
        fb = self.comm.aa.FLAG_BLOCK_BYTES
        for ch in range(_TL_CHANNELS):
            flags = [p + ch * fb for p in self.flag_ptrs]
            rp = recv_ptrs[ch] if self.mode == "GemmA2A" else recv_ptrs
            self.hop.bind(
                ch, flags, rp, self.comm.queues, a, b, self.d.data_ptr(), self.counter.data_ptr(), c
            )

    def _next_channel(self):
        ch = self.launches % _TL_CHANNELS
        self.launches += 1
        return ch

    def run_gemm_a2a(self, w, a):
        """``D = a @ w^T`` with each rank's column block pushed to it; returns ``[world, n, shard]``."""
        key = (w.data_ptr(), a.data_ptr())
        if key != self.bound:
            self._bind(self._recv_ptrs, w.data_ptr(), a.data_ptr())
            self.bound = key
        ch = self._next_channel()
        self.hop.launch(ch, torch.cuda.current_stream().cuda_stream)
        return self._recv[ch].view(torch.bfloat16).view(self.comm.world, self.n, self.shard)

    def run_a2a_gemm(self, w, c):
        """``D = gather(x) @ w^T (+ c)`` after every rank has written its ``x``."""
        key = (w.data_ptr(), 0 if c is None else c.data_ptr())
        if key != self.bound:
            self._bind(self._x_ptrs, w.data_ptr(), self.b.data_ptr(), key[1])
            self.bound = key
        stream = torch.cuda.current_stream().cuda_stream
        self.comm.aa.boundary_barrier(self._arrival_ptrs, self.comm.rank, self.comm.world, stream)
        ch = self._next_channel()
        self.hop.launch(ch, stream)
        return self.d


class TensileLiteUlyssesA2ABackend:
    """TensileLite GemmA2A / A2AGemm, forward only, with SDMA moving the data (attn-a2a-demo).

    Needs the ``attn_demo._aa`` extension (built against a TensileLite host library with
    ``TENSILELITE_ENABLE_SDMA=ON``) and kernel libraries tuned for the exact shapes, named by
    ``PRIMUS_TURBO_ULYSSES_TENSILELITE_LIB_PRE`` (GemmA2A) and ``..._LIB_POST`` (A2AGemm). A shape
    without a tuned kernel raises ``ValueError``. Every rank must see every peer GPU of its group.
    """

    def __init__(self):
        self._aa = _tl_module()
        self._comms: Dict[int, _TLComm] = {}
        self._hops: Dict[tuple, _TLHop] = {}

    @staticmethod
    def is_available() -> bool:
        try:
            _tl_module()
            return True
        except ImportError:
            return False

    def _hop(self, group, mode, m, n, k, extent, beta):
        key = (id(group), mode, m, n, k, extent, beta)
        hop = self._hops.get(key)
        if hop is None:
            comm = self._comms.get(id(group))
            if comm is None:
                comm = self._comms[id(group)] = _TLComm(self._aa, group)
            side = "PRE" if mode == "GemmA2A" else "POST"
            root = getattr(envs, f"PRIMUS_TURBO_ULYSSES_TENSILELITE_LIB_{side}")
            hop = self._hops[key] = _TLHop(comm, mode, m, n, k, extent, _tl_library(root, side), beta)
        return hop

    def qkv_gemm_a2a(self, x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning):
        n = group.size()
        if qkv_format == "sbhd":
            s_local, b, hidden = x.shape
        else:
            b, s_local, hidden = x.shape
        a = x.reshape(s_local * b, hidden).contiguous()
        w = _qkv_weight(w_qkv, hq, hkv, d, n, weight_layout)
        N = w.shape[0]
        recv = self._hop(group, "GemmA2A", N, a.shape[0], hidden, N, 0.0).run_gemm_a2a(w, a)
        G, nq, nk, nv = _qkv_rank_block(hq, hkv, d, n, weight_layout)
        # recv[j] holds source rank j's tokens, which are the j-th contiguous chunk of the sequence
        if qkv_format == "sbhd":
            src = recv.view(n, s_local, b, G, nq + nk + nv, d)
            out_shape, split = (n * s_local, b), (n, s_local, b)
        else:
            src = recv.view(n, b, s_local, G, nq + nk + nv, d).transpose(0, 1)
            out_shape, split = (b, n * s_local), (b, n, s_local)
        outs = []
        for lo, cnt in ((0, nq), (nq, nk), (nq + nk, nv)):
            # fresh tensors: the receive slices are reused two calls later
            t = torch.empty(*out_shape, G * cnt, d, dtype=x.dtype, device=x.device)
            t.view(*split, G, cnt, d).copy_(src[..., lo : lo + cnt, :])
            outs.append(t)
        return tuple(outs)

    def a2a_o_gemm(self, o, w_o, residual, group, qkv_format, save_o, tuning):
        n = group.size()
        seq, b, heads, d, s_local, y_shape = _o_shapes(o, w_o, n, qkv_format)
        M, K_r = s_local * b, heads * d
        N, K = w_o.shape
        hop = self._hop(group, "A2AGemm", N, M, K, K, 1.0 if residual is not None else 0.0)
        # x[j] holds the local heads of rank j's tokens, in rank j's row order
        if qkv_format == "sbhd":
            hop.x.view(n, s_local, b, K_r).copy_(o.reshape(n, s_local, b, K_r))
        else:
            hop.x.view(n, b, s_local, K_r).copy_(o.reshape(b, n, s_local, K_r).transpose(0, 1))
        c = None
        if residual is not None:
            assert tuple(residual.shape) == y_shape, f"residual {tuple(residual.shape)} != {y_shape}"
            c = residual.reshape(M, N).contiguous()
        y = hop.run_a2a_gemm(w_o.contiguous(), c).clone()
        o_gathered = None
        if save_o:
            # slot s of the A2AGemm B buffer holds the shard of source rank (rank + s) % n; slot 0,
            # this rank's own shard, is never written (the hop reads it from x in place)
            o_gathered = torch.empty(M, K, dtype=o.dtype, device=o.device)
            og = o_gathered.view(M, n, K_r)
            slots = hop.b.view(n, M, K_r)
            r = group.rank()
            og[:, r:].copy_(slots[: n - r].transpose(0, 1))
            og[:, :r].copy_(slots[n - r :].transpose(0, 1))
            og[:, r].copy_(hop.x[r])
        return y.view(y_shape), o_gathered


# =========================================================================
# Backward (FlyDSL for every backend)
# =========================================================================


def ulysses_qkv_gemm_a2a_bwd_impl(
    x, w_qkv, dq, dk, dv, hq, hkv, d, group, qkv_format, weight_layout, s_local, b, need_dx, need_dw, tuning
):
    """A2AGemm: the local heads' grads over the full sequence are pushed into the sequence owner's
    recv bank in the (rank-major) weight row order; ``dx = dqkv @ w_qkv`` and ``dw = dqkv^T @ x``
    consume them as they land, in the same kernel (shapes it cannot tile copy ``dqkv`` out for a
    separate wgrad)."""
    n = group.size()
    M, hidden = s_local * b, w_qkv.shape[1]
    K = (hq + 2 * hkv) * d
    concat = weight_layout != "interleaved"
    w = _qkv_weight(w_qkv, hq, hkv, d, n, weight_layout)
    parts = tuple(t.reshape(s_local * n * b, -1, d).contiguous() for t in (dq, dk, dv))
    dx = torch.empty(M, hidden, dtype=x.dtype, device=x.device)
    # dw tiles inside the kernel: whole 256-row m-tiles, one source's columns per 256-row dw tile
    fused_wgrad = need_dw and M % 256 == 0 and (K // n) % 256 == 0 and hidden % 256 == 0
    x2d = x.reshape(M, hidden).contiguous() if need_dw else None
    dqkv = torch.empty(M, K, dtype=x.dtype, device=x.device) if need_dw and not fused_wgrad else None
    dw_rank_major = torch.empty(K, hidden, dtype=x.dtype, device=x.device) if fused_wgrad else None
    workspace = get_ulysses_workspace(group, a2a_gemm_bank_bytes(M, K))
    G = _qkv_rank_block(hq, hkv, d, n, weight_layout)[0]
    ulysses_a2a_gemm(
        parts,
        w,
        None,
        dx,
        dqkv,
        workspace,
        seq_local=s_local,
        qkv_format=qkv_format,
        flag_region=REGION_QKV_A2A_GEMM,
        layout="nn",
        groups=G,
        wgrad=(x2d, dw_rank_major) if fused_wgrad else None,
        **_launch_kwargs(tuning),
    )
    dw = None
    if need_dw:
        if not fused_wgrad:
            dw_rank_major = dqkv.t() @ x2d
        if concat:
            dw = torch.empty_like(dw_rank_major)
            dw.index_copy_(0, _rank_major_concat_rows(hq, hkv, d, n, x.device), dw_rank_major)
        else:
            dw = dw_rank_major
    return (dx.view_as(x) if need_dx else None), dw


def _o_gemm_dgrad(dy2d, w_o, group, heads, d, s_local, b, qkv_format, wgrad=None, tuning=None):
    """``dO = (dY @ w_o)`` scattered back to the local heads over the full sequence (GemmA2A);
    ``wgrad=(o_gathered, dw)`` also computes ``dw = dY^T @ o_gathered`` in the same kernel."""
    n = group.size()
    M, K = dy2d.shape[0], w_o.shape[1]
    seq = s_local * n
    do = torch.empty(seq * b, heads, d, dtype=dy2d.dtype, device=dy2d.device)
    workspace = get_ulysses_workspace(group, gemm_a2a_bank_bytes(M, K, 256))
    ulysses_gemm_a2a(
        dy2d,
        w_o.contiguous(),
        (do, do, do),
        workspace,
        layout="nn",
        head_dim=d,
        groups=1,
        nq=heads,
        nk=0,
        nv=0,
        seq_local=s_local,
        qkv_format=qkv_format,
        flag_region=REGION_O_GEMM_A2A,
        wgrad=None if wgrad is None else (dy2d, *wgrad),
        **_launch_kwargs(tuning),
    )
    shape = (seq, b) if qkv_format == "sbhd" else (b, seq)
    return do.view(*shape, heads, d)


def ulysses_a2a_o_gemm_bwd_impl(
    dy2d, w_o, o_gathered, group, heads, d, s_local, b, qkv_format, need_do, need_dw, tuning=None
):
    """``(dO, dW_o)``: dO by GemmA2A, ``dW_o = dY^T @ O_gathered``."""
    do = dw = None
    hidden, K = w_o.shape
    dgrad_args = (dy2d, w_o, group, heads, d, s_local, b, qkv_format)
    if need_do and need_dw and hidden % 256 == 0 and K % 256 == 0:
        # wgrad does not depend on the exchange: its tiles fill the GemmA2A launch behind the GEMM tiles
        dw = torch.empty_like(w_o)
        do = _o_gemm_dgrad(*dgrad_args, wgrad=(o_gathered, dw), tuning=tuning)
    elif need_do and need_dw:
        do = _o_gemm_dgrad(*dgrad_args, tuning=tuning)
        dw = dy2d.t() @ o_gathered
    elif need_do:
        do = _o_gemm_dgrad(*dgrad_args, tuning=tuning)
    elif need_dw:
        dw = dy2d.t() @ o_gathered
    return do, dw


# =========================================================================
# Backend registry and selection
# =========================================================================

_BACKEND_REGISTRY: Dict[str, Type[UlyssesA2ABackend]] = {
    "FLYDSL": FlyDSLUlyssesA2ABackend,
    "TENSILELITE": TensileLiteUlyssesA2ABackend,
}

_backend_instances: Dict[str, UlyssesA2ABackend] = {}

_BACKEND_TYPE_TO_NAME: Dict[BackendType, str] = {
    BackendType.FLYDSL: "FLYDSL",
}


def register_ulysses_a2a_backend(name: str, cls: Type[UlyssesA2ABackend]) -> None:
    """Register a new Ulysses fused-forward backend class."""
    _BACKEND_REGISTRY[name] = cls


def _get_backend_instance(name: str) -> UlyssesA2ABackend:
    """Lazily create and cache a backend singleton."""
    if name not in _backend_instances:
        if name not in _BACKEND_REGISTRY:
            raise ValueError(
                f"Unknown Ulysses A2A backend '{name}'. Available: {list(_BACKEND_REGISTRY.keys())}"
            )
        cls = _BACKEND_REGISTRY[name]
        if not cls.is_available():
            raise RuntimeError(
                f"Ulysses A2A backend '{name}' is registered but its dependencies are not installed."
            )
        _backend_instances[name] = cls()
    return _backend_instances[name]


def _resolve_backend_name() -> str:
    """Determine which Ulysses fused-forward backend to use.

    Priority (high -> low):
      1. ``GlobalBackendManager`` code-level setting (via ``set_ulysses_a2a_backend``)
      2. ``PRIMUS_TURBO_ULYSSES_A2A_BACKEND`` env var (supports names beyond ``BackendType``)
      3. Default: ``FLYDSL``
    """
    user_backend_choice = GlobalBackendManager.get_ulysses_a2a_backend(PrecisionType.BF16_FP16_FP32)
    if user_backend_choice is not None and user_backend_choice.backend is not None:
        user_backend = user_backend_choice.backend
        return _BACKEND_TYPE_TO_NAME.get(user_backend, user_backend.name)

    env_val = envs.PRIMUS_TURBO_ULYSSES_A2A_BACKEND
    if env_val is not None:
        return env_val.strip().upper()

    return "FLYDSL"


def ulysses_qkv_gemm_a2a_fwd_impl(
    x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backend = _get_backend_instance(_resolve_backend_name())
    return backend.qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout, tuning)


def ulysses_a2a_o_gemm_fwd_impl(
    o, w_o, residual, group, qkv_format, save_o, tuning=None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    backend = _get_backend_instance(_resolve_backend_name())
    return backend.a2a_o_gemm(o, w_o, residual, group, qkv_format, save_o, tuning)
