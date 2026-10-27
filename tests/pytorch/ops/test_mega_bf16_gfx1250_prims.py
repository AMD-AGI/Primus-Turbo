###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU runtime checks of the memory primitives the BF16 Mega MoE EP path is built on.

Everything runs through the exact helpers ``ep_intranode`` uses -- ``buffer_load`` /
``buffer_store`` with the arch-aware cache policies, ``copy_warp``, ``wait_mem_all``,
``atomic_add`` / ``ld`` at system scope and ``spin_until_flag_reaches`` -- on one scratch
allocation that a two-rank ``SymBuffer`` maps onto itself, so ``map()`` resolves a real offset.

* out-of-range buffer loads read 0 and out-of-range stores are dropped, at every cache policy the
  EP path uses: ``copy_warp`` parks a ragged tail's idle lanes one row past ``num_records``, and
  absent pool rows get ``num_records`` 0.
* ``copy_warp`` moves hidden 2880 (ragged tail) and 7168 rows; on wave32 each 64-lane logical warp
  spans two hardware waves.
* a ping-pong flag handoff between producer and consumer blocks placed on different XCDs, the
  publish / wait sequence of ``dispatch_bf16_tile`` repeated for many rounds.
"""

import functools

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401  (pytorch before the flydsl kernels)

flyc = pytest.importorskip("flydsl.compiler")
fx = pytest.importorskip("flydsl.expr")
from flydsl.expr import arith, range_constexpr  # noqa: E402

from primus_turbo.flydsl.mega.gfx1250.ep_intranode import (  # noqa: E402
    spin_until_flag_reaches,
)
from primus_turbo.flydsl.mega.gfx1250.symm_buffer import SymBuffer  # noqa: E402
from primus_turbo.flydsl.mega.gfx1250.prims import (  # noqa: E402
    CPOL_DEVICE_STREAM,
    CPOL_SYSTEM_STREAM,
)
from primus_turbo.flydsl.utils.buffer_ops import (  # noqa: E402
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.mega.gfx1250.prims import (  # noqa: E402
    atomic_add,
    cast,
    copy_warp,
    read_clock,
    wait_mem_all,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# One scratch allocation; "rank 1" is its upper half, so SymBuffer.map(addr, 1) == addr + _HALF_BYTES.
_HALF_WORDS = 1 << 22
_HALF_BYTES = _HALF_WORDS * 4
_SENTINEL = 0x7EADBEEF

_CPOLS = {
    "none": lambda: 0,
    "device": lambda: CPOL_DEVICE_STREAM,
    "system": lambda: CPOL_SYSTEM_STREAM,
}


def _scratch():
    buf = torch.full((2 * _HALF_WORDS,), _SENTINEL, dtype=torch.int32, device="cuda")
    base = buf.data_ptr()
    return buf, SymBuffer([base, base + _HALF_BYTES], 0, device=buf.device)


def _i64(word_offset):
    return fx.Int64(word_offset * 4)


# ── out-of-range buffer loads and stores ─────────────────────────────────────

_OOB_THREADS = 256
_OOB_VALID = 200  # words in range; lanes 200..255 are past num_records
_OOB_VALID4 = 100  # b128 vectors in range; lanes 100..255 are past num_records
# word offsets of each region inside the scratch half
_OOB_SRC, _OOB_LOAD, _OOB_STORE = 0, 1024, 2048
_OOB_SRC4, _OOB_LOAD4, _OOB_STORE4 = 4096, 8192, 12288


@functools.lru_cache(maxsize=None)
def _oob_launcher(cpol_name: str):
    @flyc.kernel(known_block_size=[_OOB_THREADS, 1, 1])
    def oob_kernel(sym: SymBuffer):
        cpol = _CPOLS[cpol_name]()
        t = fx.thread_idx.x
        base = sym.get_base_ptr()
        full = 64 * 1024

        def res(word_offset, nbytes):
            return create_buffer_resource_from_addr(base + _i64(word_offset), num_records_bytes=nbytes)

        # b32: lanes past num_records must read 0, and their stores must leave the sentinel
        v = buffer_load(res(_OOB_SRC, _OOB_VALID * 4), t, vec_width=1, dtype=fx.T.i32, cache_modifier=cpol)
        buffer_store(v, res(_OOB_LOAD, full), t)
        buffer_store(t + fx.Int32(1), res(_OOB_STORE, _OOB_VALID * 4), t, cache_modifier=cpol)

        # b128, the width copy_warp moves
        w = t * fx.Int32(4)
        v4 = buffer_load(res(_OOB_SRC4, _OOB_VALID4 * 16), w, vec_width=4, dtype=fx.T.i32, cache_modifier=cpol)
        buffer_store(v4, res(_OOB_LOAD4, full), w)
        src4 = buffer_load(res(_OOB_SRC4, full), w, vec_width=4, dtype=fx.T.i32)
        buffer_store(src4, res(_OOB_STORE4, _OOB_VALID4 * 16), w, cache_modifier=cpol)

    @flyc.jit
    def launch(sym, stream: fx.Stream):
        oob_kernel(sym).launch(grid=(1, 1, 1), block=(_OOB_THREADS, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("cpol_name", list(_CPOLS))
def test_buffer_oob_reads_zero_and_drops_stores(cpol_name):
    buf, sym = _scratch()
    n = _OOB_THREADS
    src = torch.arange(1000, 1000 + n, dtype=torch.int32, device="cuda")
    src4 = torch.arange(5000, 5000 + 4 * n, dtype=torch.int32, device="cuda")
    buf[_OOB_SRC : _OOB_SRC + n] = src
    buf[_OOB_SRC4 : _OOB_SRC4 + 4 * n] = src4
    _oob_launcher(cpol_name)(sym, torch.cuda.current_stream())
    torch.cuda.synchronize()

    zero = torch.zeros((), dtype=torch.int32, device="cuda")
    sentinel = torch.full((), _SENTINEL, dtype=torch.int32, device="cuda")
    lane = torch.arange(n, device="cuda")
    lane4 = torch.arange(4 * n, device="cuda")
    exp_load = torch.where(lane < _OOB_VALID, src, zero)
    exp_store = torch.where(lane < _OOB_VALID, (lane + 1).to(torch.int32), sentinel)
    exp_load4 = torch.where(lane4 < 4 * _OOB_VALID4, src4, zero)
    exp_store4 = torch.where(lane4 < 4 * _OOB_VALID4, src4, sentinel)
    torch.testing.assert_close(buf[_OOB_LOAD : _OOB_LOAD + n], exp_load, rtol=0, atol=0)
    torch.testing.assert_close(buf[_OOB_STORE : _OOB_STORE + n], exp_store, rtol=0, atol=0)
    torch.testing.assert_close(buf[_OOB_LOAD4 : _OOB_LOAD4 + 4 * n], exp_load4, rtol=0, atol=0)
    torch.testing.assert_close(buf[_OOB_STORE4 : _OOB_STORE4 + 4 * n], exp_store4, rtol=0, atol=0)


# ── copy_warp, ragged and whole rows ─────────────────────────────────────────

_COPY_THREADS = 512  # 8 logical 64-lane warps, one row each
_COPY_ROWS = _COPY_THREADS // 64
_COPY_ROW_WORDS = 4096  # row stride; the gap past each row's bytes must keep the sentinel
_COPY_SRC, _COPY_DST = 0, _COPY_ROWS * _COPY_ROW_WORDS


@functools.lru_cache(maxsize=None)
def _copy_launcher(hidden: int):
    nbytes = hidden * 2

    @flyc.kernel(known_block_size=[_COPY_THREADS, 1, 1])
    def copy_kernel(sym: SymBuffer):
        base = sym.get_base_ptr()
        row = cast(fx.thread_idx.x // fx.Int32(64), fx.T.i64) * _i64(_COPY_ROW_WORDS)
        src = base + _i64(_COPY_SRC) + row
        dst = sym.map(base, fx.Int32(1)) + _i64(_COPY_DST) + row
        copy_warp(
            dst,
            src,
            nbytes,
            load_cache_modifier=CPOL_DEVICE_STREAM,
            store_cache_modifier=CPOL_SYSTEM_STREAM,
        )

    @flyc.jit
    def launch(sym, stream: fx.Stream):
        copy_kernel(sym).launch(grid=(1, 1, 1), block=(_COPY_THREADS, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("hidden", [2880, 7168, 4096 + 8])
def test_copy_warp_rows(hidden):
    buf, sym = _scratch()
    words = hidden // 2
    src = torch.randint(-(2**31), 2**31 - 1, (_COPY_ROWS, _COPY_ROW_WORDS), dtype=torch.int32, device="cuda")
    buf[_COPY_SRC : _COPY_SRC + src.numel()] = src.flatten()
    _copy_launcher(hidden)(sym, torch.cuda.current_stream())
    torch.cuda.synchronize()

    dst = buf[_HALF_WORDS + _COPY_DST : _HALF_WORDS + _COPY_DST + src.numel()].view(_COPY_ROWS, _COPY_ROW_WORDS)
    torch.testing.assert_close(dst[:, :words], src[:, :words], rtol=0, atol=0)
    assert bool((dst[:, words:] == _SENTINEL).all()), "copy_warp wrote past the row (a parked tail store landed)"


# ── flag handoff across XCDs ─────────────────────────────────────────────────

_FLAG_THREADS = 256
# 60 pairs: consumer of pair p is block 60 + p, 60 % 8 == 4 blocks away -> another XCD under the
# round-robin block-to-XCD placement.
_FLAG_PAIRS = 60
_FLAG_WORDS_PER_THREAD = 16
_FLAG_BLOCK_WORDS = _FLAG_THREADS * _FLAG_WORDS_PER_THREAD
# word offsets inside the scratch half
_FLAG_PAYLOAD = 0
_FLAG_READY = _FLAG_PAIRS * _FLAG_BLOCK_WORDS  # one i64 per pair, producer -> consumer
_FLAG_ACK = _FLAG_READY + 2 * 64  # one i64 per pair, consumer -> producer
_FLAG_ERR = _FLAG_ACK + 2 * 64  # one i32 mismatch counter per pair
_FLAG_CYCLES = _FLAG_ERR + 64  # one i64 realtime-cycle total per pair


def _payload(rnd, pair, word):
    return (rnd << 20) | (pair << 14) | word


@functools.lru_cache(maxsize=None)
def _flag_launcher(rounds: int):
    @flyc.kernel(known_block_size=[_FLAG_THREADS, 1, 1])
    def flag_kernel(sym: SymBuffer):
        t = fx.thread_idx.x
        block, _b, _c = fx.block_idx
        local = sym.get_base_ptr()
        peer = sym.map(local, fx.Int32(1))
        is_producer = block < fx.Int32(_FLAG_PAIRS)
        pair = arith.select(is_producer, block, block - fx.Int32(_FLAG_PAIRS))
        payload_res = create_buffer_resource_from_addr(
            peer + _i64(_FLAG_PAYLOAD) + cast(pair, fx.T.i64) * _i64(_FLAG_BLOCK_WORDS),
            num_records_bytes=_FLAG_BLOCK_WORDS * 4,
        )
        ready = peer + _i64(_FLAG_READY)
        ack = peer + _i64(_FLAG_ACK)
        first_word = t * fx.Int32(_FLAG_WORDS_PER_THREAD)
        tag = pair * fx.Int32(1 << 14) + first_word
        if is_producer:
            for r in range(fx.Int32(rounds)):
                spin_until_flag_reaches(ack, pair, cast(r, fx.T.i64), "sys", 0, "ack")
                for k in range_constexpr(_FLAG_WORDS_PER_THREAD):
                    value = (r + fx.Int32(1)) * fx.Int32(1 << 20) + tag + fx.Int32(k)
                    buffer_store(value, payload_res, first_word + fx.Int32(k), cache_modifier=CPOL_SYSTEM_STREAM)
                wait_mem_all()
                fx.gpu.barrier()
                if t == fx.Int32(0):
                    atomic_add(ready, pair, fx.Int64(1), scope="sys")
        else:
            errors = fx.Int32(0)
            start = read_clock()
            for r in range(fx.Int32(rounds)):
                spin_until_flag_reaches(ready, pair, cast(r + fx.Int32(1), fx.T.i64), "sys", 0, "ready")
                for k in range_constexpr(_FLAG_WORDS_PER_THREAD):
                    got = buffer_load(
                        payload_res,
                        first_word + fx.Int32(k),
                        vec_width=1,
                        dtype=fx.T.i32,
                        cache_modifier=CPOL_SYSTEM_STREAM,
                    )
                    want = (r + fx.Int32(1)) * fx.Int32(1 << 20) + tag + fx.Int32(k)
                    errors = errors + arith.select(got != want, fx.Int32(1), fx.Int32(0))
                wait_mem_all()
                fx.gpu.barrier()
                if t == fx.Int32(0):
                    atomic_add(ack, pair, fx.Int64(1), scope="sys")
            elapsed = read_clock() - start
            atomic_add(local + _i64(_FLAG_ERR), pair, errors, scope="sys")
            if t == fx.Int32(0):
                atomic_add(local + _i64(_FLAG_CYCLES), pair, cast(elapsed, fx.T.i64), scope="sys")

    @flyc.jit
    def launch(sym, stream: fx.Stream):
        flag_kernel(sym).launch(grid=(2 * _FLAG_PAIRS, 1, 1), block=(_FLAG_THREADS, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("rounds", [1, 2000])
def test_flag_handoff_across_xcds(rounds):
    buf, sym = _scratch()
    buf.zero_()
    _flag_launcher(rounds)(sym, torch.cuda.current_stream())
    torch.cuda.synchronize()

    hi = buf[_HALF_WORDS:]
    ready = hi[_FLAG_READY : _FLAG_READY + 2 * _FLAG_PAIRS].view(torch.int64)
    ack = hi[_FLAG_ACK : _FLAG_ACK + 2 * _FLAG_PAIRS].view(torch.int64)
    errors = buf[_FLAG_ERR : _FLAG_ERR + _FLAG_PAIRS]
    cycles = buf[_FLAG_CYCLES : _FLAG_CYCLES + 2 * _FLAG_PAIRS].view(torch.int64)
    assert bool((ready == rounds).all()) and bool((ack == rounds).all())
    assert int(errors.sum()) == 0, f"stale payload words seen after the flag: {errors.tolist()}"
    assert bool((cycles > 0).all()), "read_clock did not advance"

    payload = hi[_FLAG_PAYLOAD : _FLAG_PAYLOAD + _FLAG_PAIRS * _FLAG_BLOCK_WORDS].view(_FLAG_PAIRS, -1).cpu()
    pair = torch.arange(_FLAG_PAIRS).view(-1, 1)
    word = torch.arange(_FLAG_BLOCK_WORDS).view(1, -1)
    torch.testing.assert_close(payload, _payload(rounds, pair, word).to(torch.int32), rtol=0, atol=0)
