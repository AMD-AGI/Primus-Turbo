###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``put + signal`` on SDMA engines, and the all-gather / all-to-all / reduce-scatter built from it.

Layer L1 of ``docs/design/collective_gemm_mi455.html``. One primitive: ``put`` copies a local range
into this call's receive bank on a peer, and once it has landed the SDMA engine itself writes the
epoch into one of the peer's flags. A call is a session on the caller's stream::

    epoch = comm.begin()
    comm.put(dst_rank, dst_offset, src_ptr, nbytes, flag)   # any number, any peers, chunked or not
    comm.release()             # optional: puts so far read their sources as of this stream point
    comm.wait_flag(src, flag)  # optional: stream waits for one peer's flag (start on that chunk)
    comm.wait(num_flags)       # stream waits: own puts done, every peer's flags 0..num_flags-1 arrived

Each put submits, from the host:

* a data copy on an SDMA engine, gated on an HSA signal that a one-wave kernel on the caller's
  stream opens at the next ``release`` / ``wait_flag`` / ``wait``, so the copy starts only after
  the stream has produced the data;
* an 8-byte copy of the epoch (which that kernel also stores) into the peer's flag, on the same
  engine and dependent on the data copy's completion signal, so data is visible before its flag.

The batch-copy API's signal-after-copy would fold the flag into the data copy, but on gfx1250 with
ROCm 10.2 the runtime accepts it and the engine never performs the write (and stalls that queue).

``wait`` and ``wait_flag`` open the pending gate and spin in the same kernel: ``wait`` for this
rank's flag writes (its sources are free) and for flags ``0 .. num_flags-1`` from every peer (its
bank is complete), ``wait_flag`` for one peer's flag. A fused consumer can instead poll
``flags()`` itself to start on each chunk as it arrives.

Flags: each receiver holds, per bank, a block of ``max_chunks`` flags per source rank, written only
by that source, so a flag has one writer as long as a rank puts each ``(dst_rank, flag)`` at most
once per call. The collectives use flag ``c`` for chunk ``c``.

The host never waits on the GPU except to recycle a slot of signals still in flight from
``_NUM_SLOTS`` calls ago. Not capturable in a CUDA graph (the host submits SDMA work per call).

Protocol (as ``fused_collective_gemm.workspace``): call ``e`` (from 1) uses bank ``e & 1`` and
flag value ``e``; waiters spin until ``flag >= e``. A rank's call ``e`` needs every peer's
call-``e`` data, which a peer only sends once its stream reached call ``e``, i.e. after it consumed
its bank from call ``e - 2``, so two banks need no extra handshake -- provided each rank consumes
a bank on the calling stream before its next-but-one call.

Remote writes land in this GPU's HBM without passing its L2, so the banks and flags are uncached
by default; a cached bank can serve stale L2 lines to a later reader.
"""

from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.pytorch.core.pyhsa_runtime_wrapper import get_hsa_runtime_lib
from primus_turbo.pytorch.core.symm_mem import (
    _SPIN_TIMEOUT_CYCLES,
    SymmetricMemory,
    _read_realtime,
    _trap,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability

# signal-pad bytes below the flags belong to SymmetricMemory.barrier()'s counters
_FLAG_OFFSET = 4096
# calls whose SDMA work may be in flight at once before the host waits to recycle signals
_NUM_SLOTS = 8
_BANK_ALIGN = 256
_CHUNK_ALIGN = 256


@triton.jit
def _open_gate(ctrl, gate_idx, epoch_word, epoch):
    gate = tl.load(ctrl + gate_idx).to(tl.pointer_type(tl.int64))
    tl.store(epoch_word, epoch)
    # system-scope release writes back L2, so the SDMA engine reads what this stream produced
    tl.atomic_xchg(gate, 0, sem="release", scope="sys")


@triton.jit
def _spin_until(ptr, value, EQ: tl.constexpr, GFX12: tl.constexpr, TIMEOUT: tl.constexpr):
    start = _read_realtime(GFX12)
    if EQ:
        while tl.atomic_add(ptr, 0, sem="acquire", scope="sys") != value:
            if _read_realtime(GFX12) - start > TIMEOUT:
                _trap()
    else:
        while tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < value:
            if _read_realtime(GFX12) - start > TIMEOUT:
                _trap()


@triton.jit(do_not_specialize=["gate_idx", "epoch"])
def _release_kernel(ctrl, gate_idx, epoch_word, epoch):
    _open_gate(ctrl, gate_idx, epoch_word, epoch)


@triton.jit(do_not_specialize=["gate_idx", "epoch", "num_puts", "src", "flag0", "num_flags"])
def _sync_kernel(
    ctrl,
    gate_idx,
    epoch_word,
    epoch,
    num_puts,
    flags,
    rank,
    world,
    src,
    flag0,
    num_flags,
    flags_per_rank,
    max_puts,
    GFX12: tl.constexpr,
    TIMEOUT: tl.constexpr,
):
    # ctrl = [gate g for g < max_puts] + [flag-copy completion of put i for i < max_puts]
    if gate_idx >= 0:
        _open_gate(ctrl, gate_idx, epoch_word, epoch)
    for i in range(num_puts):
        done = tl.load(ctrl + max_puts + i).to(tl.pointer_type(tl.int64))
        _spin_until(done, 0, True, GFX12, TIMEOUT)
    # src < 0: every peer
    for peer in range(world):
        if peer != rank and (src < 0 or peer == src):
            for c in range(flag0, flag0 + num_flags):
                _spin_until(flags + peer * flags_per_rank + c, epoch, False, GFX12, TIMEOUT)


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class _Slot:
    """HSA signals of one call: the stream gates (one per release point), and per put the data copy
    and flag copy completions; plus the epoch word the flag copies send."""

    def __init__(self, hsa, max_puts: int):
        self.max_puts = max_puts
        self.gates = [hsa.signal_create(1) for _ in range(max_puts)]
        self.copied = [hsa.signal_create(1) for _ in range(max_puts)]
        self.flagged = [hsa.signal_create(1) for _ in range(max_puts)]
        self.epoch_word = torch.zeros(1, dtype=torch.int64, device="cuda")
        ptrs = [hsa.signal_value_pointer(s) for s in self.gates + self.flagged]
        self.ctrl = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
        self.event: Optional[torch.cuda.Event] = None
        self.used = 0
        self.gates_used = 0
        # puts were submitted behind gates[gates_used], which no kernel has been enqueued to open yet
        self.pending = False

    def recycle(self, hsa) -> None:
        # the event follows the wait kernel, which saw every flag copy (hence every data copy) complete
        if self.event is not None:
            self.event.synchronize()
            self.event = None
        for g in self.gates[: self.gates_used]:
            hsa.signal_store(g, 1)
        for i in range(self.used):
            hsa.signal_store(self.copied[i], 1)
            hsa.signal_store(self.flagged[i], 1)
        self.used = 0
        self.gates_used = 0
        self.pending = False

    def take_pending_gate(self) -> int:
        """Index of the gate to open now (-1 if no put waits on one); later puts use the next gate."""
        if not self.pending:
            return -1
        self.pending = False
        self.gates_used += 1
        return self.gates_used - 1

    def destroy(self, hsa) -> None:
        for s in self.gates + self.copied + self.flagged:
            hsa.signal_destroy(s)


class SdmaCommunicator:
    """SDMA ``put + signal`` and collectives over a group whose ranks share one scale-up domain (one vpod).

    ``max_bytes_per_rank`` bounds the block one rank contributes per peer (the all-gather input,
    or one ``inp[i]`` of all-to-all / reduce-scatter); the bank holds ``world`` such blocks.
    ``max_chunks`` is the number of flags per source rank, i.e. the most puts a rank makes to one
    peer in a call. The caller must set the CUDA device first.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_bytes_per_rank: int,
        max_chunks: int = 4,
        uncached_bank: bool = True,
        backend: Optional[str] = None,
    ):
        if max_chunks < 1:
            raise ValueError(f"max_chunks must be >= 1, got {max_chunks}")
        self.group = group
        self.rank = group.rank()
        self.world_size = group.size()
        self.device = torch.cuda.current_device()
        self.max_chunks = max_chunks
        self.bank_bytes = _align(self.world_size * max_bytes_per_rank, _BANK_ALIGN)
        self.symm = SymmetricMemory(
            group,
            2 * self.bank_bytes,
            signal_pad_size=_FLAG_OFFSET + 2 * self.world_size * max_chunks * 8,
            uncached_signal_pad=True,
            uncached_buffer=uncached_bank,
            backend=backend,
        )
        self.hsa = get_hsa_runtime_lib()
        self.agent = self.hsa.gpu_agent(self.device)
        mask = self.hsa.copy_engine_mask(self.agent, self.agent)
        self.engines = [i for i in range(32) if mask >> i & 1]
        if not self.engines:
            raise RuntimeError(f"no SDMA engine available on device {self.device}")
        self._max_puts = (self.world_size - 1) * max_chunks
        self._slots = [_Slot(self.hsa, self._max_puts) for _ in range(_NUM_SLOTS)]
        self._local_flags = self.symm.get_signal_pad(
            self.rank, (2, self.world_size, max_chunks), torch.int64, storage_offset=_FLAG_OFFSET // 8
        )
        self._gfx12 = get_device_compute_capability()[0] >= 12
        self._epoch = 0
        self._slot: Optional[_Slot] = None
        self._puts_to = [0] * self.world_size
        self.is_destroyed = False

    # ------------------------------------------------------------------ primitive

    def begin(self) -> int:
        """Open call ``epoch`` (returned). Enqueues nothing: a put reads its source as of the stream
        point of the next ``release`` / ``wait_flag`` / ``wait``. Raises while a call is open or under
        CUDA graph capture."""
        if self.is_destroyed:
            raise RuntimeError("SdmaCommunicator used after destroy()")
        if self._slot is not None:
            raise RuntimeError(f"call {self._epoch} is still open: wait() before the next begin()")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SDMA collectives submit host-side work per call and cannot be graph-captured")
        self._epoch += 1
        slot = self._slots[self._epoch % _NUM_SLOTS]
        slot.recycle(self.hsa)
        self._slot = slot
        self._puts_to = [0] * self.world_size
        return self._epoch

    def bank(self, shape: tuple, dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
        """This rank's receive bank of the open call, as ``shape``/``dtype`` from byte ``offset``. Valid
        until this rank's next-but-one call."""
        self._require_open()
        nbytes = dtype.itemsize
        for d in shape:
            nbytes *= d
        if offset % dtype.itemsize or offset + nbytes > self.bank_bytes:
            raise ValueError(
                f"bank view [{offset}, {offset + nbytes}) misaligned or past {self.bank_bytes} bytes"
            )
        bank = self._epoch & 1
        return self.symm.get_buffer(
            self.rank, shape, dtype, storage_offset=(bank * self.bank_bytes + offset) // dtype.itemsize
        )

    def flags(self) -> torch.Tensor:
        """``[world, max_chunks]`` int64 flags of the open call on this rank: ``[src, f] >= epoch`` once
        rank ``src``'s put with flag ``f`` has landed. For consumers that poll instead of ``wait()``."""
        self._require_open()
        return self._local_flags[self._epoch & 1]

    def put(
        self,
        dst_rank: int,
        dst_offset: int,
        src_ptr: int,
        nbytes: int,
        flag: int,
        engine: Optional[int] = None,
    ) -> None:
        """Copy ``nbytes`` at ``src_ptr`` (this GPU's memory, alive and unchanged until ``wait``) to byte
        ``dst_offset`` of ``dst_rank``'s bank, then set ``dst_rank``'s ``flags()[self.rank, flag]`` to
        the epoch. The copy starts once the stream reaches the next ``release`` / ``wait_flag`` /
        ``wait``. ``engine`` indexes ``self.engines``; by default the i-th put to the k-th next peer
        uses ``engines[(k - 1 + i * (world - 1)) % len(engines)]``, spreading peers and chunks."""
        slot = self._require_open()
        if not 0 <= dst_rank < self.world_size or dst_rank == self.rank:
            raise ValueError(f"dst_rank must be a peer in [0, {self.world_size}) other than {self.rank}")
        if not 0 <= flag < self.max_chunks:
            raise ValueError(f"flag must be in [0, {self.max_chunks}), got {flag}")
        if nbytes < 0 or dst_offset < 0 or dst_offset + nbytes > self.bank_bytes:
            raise ValueError(
                f"put [{dst_offset}, {dst_offset + nbytes}) past the {self.bank_bytes}-byte bank"
            )
        if slot.used == self._max_puts:
            raise RuntimeError(f"more than {self._max_puts} puts in one call")
        k = (dst_rank - self.rank) % self.world_size
        if engine is None:
            engine = (k - 1 + self._puts_to[dst_rank] * (self.world_size - 1)) % len(self.engines)
        self._puts_to[dst_rank] += 1
        hsa, bank, i = self.hsa, self._epoch & 1, slot.used
        slot.used += 1
        gate = slot.gates[slot.gates_used]
        slot.pending = True
        # the flag copy reads the epoch word, which the gate kernel stores before opening the gate
        flag_deps = [gate]
        if nbytes:
            hsa.copy_on_engine(
                self.symm.buffer_ptrs[dst_rank] + bank * self.bank_bytes + dst_offset,
                self.agent,
                src_ptr,
                self.agent,
                nbytes,
                deps=[gate],
                completion=slot.copied[i],
                engine=self.engines[engine],
            )
            flag_deps = [slot.copied[i]]
        flag_word = _FLAG_OFFSET + ((bank * self.world_size + self.rank) * self.max_chunks + flag) * 8
        hsa.copy_on_engine(
            self.symm.signal_pad_ptrs[dst_rank] + flag_word,
            self.agent,
            slot.epoch_word.data_ptr(),
            self.agent,
            8,
            deps=flag_deps,
            completion=slot.flagged[i],
            engine=self.engines[engine],
        )

    def release(self) -> None:
        """Let the puts submitted since the last release start once the stream reaches this point
        (they read their sources as of here). A no-op without such puts."""
        slot = self._require_open()
        gate_idx = slot.take_pending_gate()
        if gate_idx >= 0:
            _release_kernel[(1,)](slot.ctrl, gate_idx, slot.epoch_word, self._epoch)

    def wait_flag(self, src_rank: int, flag: int) -> None:
        """Release pending puts, then make the stream wait for ``flags()[src_rank, flag]``: rank
        ``src_rank``'s put with that flag has landed in this rank's bank."""
        if not 0 <= src_rank < self.world_size or src_rank == self.rank:
            raise ValueError(f"src_rank must be a peer in [0, {self.world_size}) other than {self.rank}")
        if not 0 <= flag < self.max_chunks:
            raise ValueError(f"flag must be in [0, {self.max_chunks}), got {flag}")
        self._sync(0, src_rank, flag, 1)

    def wait(self, num_flags: int) -> None:
        """Close the call: release pending puts, then the stream waits until this rank's puts have
        landed and flags ``0 .. num_flags-1`` from every peer have arrived (``0``: own puts only)."""
        slot = self._require_open()
        if not 0 <= num_flags <= self.max_chunks:
            raise ValueError(f"num_flags must be in [0, {self.max_chunks}], got {num_flags}")
        self._sync(slot.used, -1, 0, num_flags)
        slot.event = torch.cuda.Event()
        slot.event.record(torch.cuda.current_stream())
        self._slot = None

    def _sync(self, num_puts: int, src: int, flag0: int, num_flags: int) -> None:
        slot = self._require_open()
        _sync_kernel[(1,)](
            slot.ctrl,
            slot.take_pending_gate(),
            slot.epoch_word,
            self._epoch,
            num_puts,
            self._local_flags[self._epoch & 1],
            self.rank,
            self.world_size,
            src,
            flag0,
            num_flags,
            self.max_chunks,
            slot.max_puts,
            GFX12=self._gfx12,
            TIMEOUT=_SPIN_TIMEOUT_CYCLES,
        )

    # ------------------------------------------------------------------ collectives

    def all_gather(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None, num_chunks: int = 1
    ) -> torch.Tensor:
        """``[world, *inp.shape]``, row ``r`` = rank ``r``'s ``inp``, each peer's block sent in up to
        ``num_chunks`` puts. Without ``out`` the result is a view of the bank, valid until this rank's
        next-but-one call."""
        inp = self._check(inp)
        nbytes = inp.numel() * inp.element_size()
        chunks = self._chunks(nbytes, num_chunks)
        self.begin()
        view = self.bank((self.world_size, *inp.shape), inp.dtype)
        view[self.rank].copy_(inp)
        for c, (off, size) in enumerate(chunks):
            for peer in self._peers():
                self.put(peer, self.rank * nbytes + off, inp.data_ptr() + off, size, c)
        self.wait(len(chunks))
        return self._finish(view, out)

    def all_to_all(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None, num_chunks: int = 1
    ) -> torch.Tensor:
        """``inp[i]`` goes to rank ``i``; returns ``[world, *inp.shape[1:]]`` whose row ``r`` came from
        rank ``r``."""
        inp = self._check(inp, split=True)
        nbytes = inp[0].numel() * inp.element_size()
        chunks = self._chunks(nbytes, num_chunks)
        self.begin()
        view = self.bank(tuple(inp.shape), inp.dtype)
        view[self.rank].copy_(inp[self.rank])
        for c, (off, size) in enumerate(chunks):
            for peer in self._peers():
                self.put(peer, self.rank * nbytes + off, inp.data_ptr() + peer * nbytes + off, size, c)
        self.wait(len(chunks))
        return self._finish(view, out)

    def reduce_scatter(
        self, inp: torch.Tensor, out: Optional[torch.Tensor] = None, num_chunks: int = 1
    ) -> torch.Tensor:
        """Sum over ranks of their ``inp[self.rank]``: an all-to-all, then a local reduce over the ranks in
        a fixed order (fp32 accumulation for narrower floats), so the result is bitwise reproducible."""
        rows = self.all_to_all(inp, num_chunks=num_chunks)
        if rows.dtype in (torch.bfloat16, torch.float16):
            red = rows.sum(0, dtype=torch.float32).to(rows.dtype)
        else:
            red = rows.sum(0)
        if out is None:
            return red
        out.copy_(red)
        return out

    # ------------------------------------------------------------------ internals

    def _require_open(self) -> _Slot:
        if self._slot is None:
            raise RuntimeError("no open call: begin() first")
        return self._slot

    def _peers(self) -> list[int]:
        # rank r sends to r+1, r+2, ... so at each step every rank has a distinct destination
        return [(self.rank + k) % self.world_size for k in range(1, self.world_size)]

    def _chunks(self, nbytes: int, num_chunks: int) -> list[tuple[int, int]]:
        """Byte ranges ``(offset, size)`` covering ``nbytes`` in at most ``num_chunks`` pieces of
        ``_CHUNK_ALIGN``-aligned size; every rank derives the same split from the same ``nbytes``."""
        if not 1 <= num_chunks <= self.max_chunks:
            raise ValueError(f"num_chunks must be in [1, {self.max_chunks}], got {num_chunks}")
        if self.world_size * nbytes > self.bank_bytes:
            raise ValueError(
                f"{nbytes} bytes per rank exceed this communicator's bank "
                f"({self.bank_bytes // self.world_size} bytes per rank)"
            )
        if nbytes == 0:
            return [(0, 0)]
        size = _align(-(-nbytes // num_chunks), _CHUNK_ALIGN)
        return [(off, min(size, nbytes - off)) for off in range(0, nbytes, size)]

    def _check(self, inp: torch.Tensor, split: bool = False) -> torch.Tensor:
        if not inp.is_cuda or inp.device.index != self.device:
            raise ValueError(f"input must be on cuda:{self.device}, got {inp.device}")
        if split and (inp.dim() == 0 or inp.shape[0] != self.world_size):
            raise ValueError(f"input needs a leading dim of {self.world_size} (one block per rank)")
        return inp.contiguous()

    @staticmethod
    def _finish(view: torch.Tensor, out: Optional[torch.Tensor]) -> torch.Tensor:
        if out is None:
            return view
        out.copy_(view.reshape(out.shape))
        return out

    def destroy(self) -> None:
        """Release signals and symmetric memory; collective, call once every rank's work has drained."""
        if self.is_destroyed:
            return
        self.is_destroyed = True
        torch.cuda.synchronize()
        for slot in self._slots:
            slot.destroy(self.hsa)
        self._slots = []
        dist.barrier(group=self.group)
        self.symm.destroy()


__all__ = ["SdmaCommunicator"]
