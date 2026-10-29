# FlyDSL Intranode DeepEP on MI455X (gfx1250)

## Overview

DeepEP's intranode dispatch / combine for MI455X (`gfx1250`), written in **FlyDSL**. Two backends
run on gfx1250:

- **`flydsl` (V1)**: the FlyDSL port of the DeepEP intranode kernels, the same ring protocol as the
  C++ (HIP) DeepEP, ported from gfx942 / gfx950 to gfx1250.
- **`flydsl_v2` (V2)**: a gfx1250-only backend that replaces the rings with fixed per-token slots
  and moves rows with TDM. It is the default on gfx1250.

On gfx942 / gfx950 nothing changes: the default stays `cpp`, and the V1 kernels compile to the same
ISA as before.

### Key points

- **No C++ dependency** — the FlyDSL path runs without the C++ `deep_ep` module, which is not loaded
  on gfx1250.
- **One set of kernels, per-arch traits** — the gfx1250 differences (wave32, split wait counters,
  cache-policy bits, copy-piece sizes, watchdog) live in one `ArchTraits` table; the gfx9 values are
  the constants from before.
- **Fixed slots instead of rings (V2)** — every token has a fixed slot on each rank it goes to, so
  the sender writes without waiting for the receiver, and the combine reduces rows as they land.
- **Fast** — on the DeepSeek-V3 shape at EP2, `flydsl_v2` notify + dispatch takes 183 µs and
  combine 150 µs: 3.0× and 3.3× faster than the RCCL all-to-all of Megatron's alltoall dispatcher.
- **Drop-in** — same `Buffer` API and outputs bit-identical to V1; `moe_dispatch` /
  `DeepEPTokenDispatcher` size the buffer from the dispatched tokens.

How to choose, size and tune a backend: [the DeepEP README](../primus_turbo/pytorch/deep_ep/README.md#intranode-backend).

## Background

MI455X had no intranode DeepEP:

- The C++ (HIP) DeepEP assumes gfx9 (wave64, a unified `s_waitcnt`, gfx9 cache-policy bits) and does
  not run correctly on gfx1250 without changes; porting it is a separate change.
- The FlyDSL V1 kernels were written for gfx942 / gfx950 and make the same assumptions.
- V1's rings are latency bound: each (channel, rank) pair is a ring with a head / tail handshake per
  chunk, and the receiver copies from the ring into `recv_x` in a second pass. On MI455X, V1 needs
  224 CUs to reach 410 / 226 µs (dispatch / combine) on the DeepSeek-V3 shape at EP2, while the
  link moves the same bytes in under 100 µs.

What differs on gfx1250, and how it is handled:

| | gfx942 / gfx950 | gfx1250 | Handling |
|---|---|---|---|
| Wave size | 64 | 32 | `WARP_SIZE` from `ArchTraits`; the notify butterfly starts at `WARP_SIZE / 2`; `copy_warp` covers `WARP_SIZE` lanes |
| Memory waits | unified `s_waitcnt` | split load / store / ds / km counters | `wait_mem_all()` waits on each counter |
| Clock | `s_memrealtime` | none | `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` (100 MHz, as on gfx9) |
| Cache policy | `0x13` (sc0 \| sc1 \| nt) | TH in bits 2:0, SCOPE in 4:3; `0x13` decodes as TH_LU + SCOPE_DEV | `0x19` (TH_NT \| SCOPE_SYS) for loads and stores |
| Store acknowledgement | a store to uncached memory is acknowledged once it reaches memory | a store is acknowledged at the first cache level that covers its scope | payload stores carry `SCOPE_SYS` (see "Memory ordering") |
| Store counting | vmcnt counts loads and stores, so copy pieces are 1 / 2 / 8 KiB by channel count | loads and stores counted separately; pieces limited only by VGPRs | 8 KiB pieces |
| Combine ring lower bound | `recv >= reduce warps + send - 1`, 15 reduce warps | 31 reduce warps | checked on the host |
| Spinning queue at teardown | recoverable | MES can become unrecoverable when the driver preempts or destroys a spinning queue | the watchdog traps (see "Watchdog") |

## Core Design

### 1. Backends and the host side

- **Selection**: `Buffer(intranode_backend=...)`, then `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`,
  then the default: `flydsl_v2` on gfx1250; elsewhere `cpp` when the C++ DeepEP is built, otherwise
  `flydsl`. An explicit choice is never replaced; `flydsl_v2` off gfx1250 raises and names the
  alternatives.
- **No C++ `deep_ep`** (`primus_turbo/pytorch/deep_ep/_compat.py`): the C++ `Config` / `EventHandle`
  are used when the module can be imported, Python versions otherwise (`PyConfig` with
  `get_nvl_buffer_size_hint`, `PyEventHandle` over `torch.cuda.Event`). On gfx1250 the C++ module is
  not loaded even if `_C` was built with it (decided with FlyDSL's `get_rocm_arch()`, which does not
  initialize the GPU).
- **`Buffer`** with a FlyDSL backend constructs no C++ `Buffer`; the comm stream is
  `torch.cuda.Stream(priority=-1)`, and internode / low-latency calls raise a clear error.
  `Buffer.destroy()` is collective: every rank waits for the others before it frees its heap.

### 2. `ArchTraits`

`primus_turbo/flydsl/deep_ep/arch.py` holds every per-arch choice:

```python
class ArchTraits(NamedTuple):
    name: str
    wave_size: int
    cpol_sys_load: int            # system-scope streaming load
    cpol_sys_store: int           # payload stores into a peer's heap: 0 on gfx9, SCOPE_SYS on gfx12
    copy_piece_table: tuple       # (op, ((max_channels, piece_bytes), ...))
    combine_loads_in_flight: int  # 16 B loads in flight per lane in the combine reducer
    spin_timeout_cycles: int      # read_clock cycles (100 MHz)
    trap_on_timeout: bool

TRAITS = get_arch_traits()        # FLYDSL_GPU_ARCH or the current device; unknown arches get the gfx9 values
```

| Field | gfx942 / gfx950 | gfx1250 |
|---|---|---|
| `wave_size` | 64 | 32 |
| `cpol_sys_load` | `0x13` | `0x19` |
| `cpol_sys_store` | 0 | `0x19` |
| `copy_piece_table` | measured on MI355X | 8 KiB |
| `combine_loads_in_flight` | 32 | 32 |
| `spin_timeout_cycles` / `trap_on_timeout` | 3e9 (30 s) / False | 3e10 (300 s) / True |

- `WARP_SIZE`, `SYS_LOAD`, `SYS_STORE` and the timeout helpers in `common.py` read `TRAITS`, as do
  the copy pieces and the combine's `steps_per_pass` in V1 and V2. The helpers whose code differs on
  gfx1250 (`read_clock()`, `wait_mem_all()`, `copy_warp()`) are DeepEP's own in `common.py`, so
  `primus_turbo/flydsl/utils` is unchanged.
- `TRAITS` is a module-level constant rather than an argument to every builder: a process compiles
  for one target, and FlyDSL's cache key includes the target.
- It must stay a NamedTuple (or another builtin container): FlyDSL's persistent cache key records
  module-level globals by value only when they are scalars or builtin containers, and any other
  object only by its type name, so changing its values would not invalidate stale kernels.

### 3. Memory ordering: the cheap fence with `SCOPE_SYS` stores

V1 uses the C++ intranode protocol: one ring per (channel, destination rank) on the receiver's heap
(`SymmetricMemory(uncached_buffer=True)`, i.e. `hipDeviceMallocUncached`). The sender writes the
payload and publishes the tail; the receiver sees the tail, reads the payload and publishes the
head.

- **Full fence**: the tail is a system-scope release store, polled with acquire loads. On gfx1250
  every publish is a full cache write-back (`global_wb scope:SCOPE_SYS`) and every poll a full
  invalidate, and the cost spills over to concurrent GEMMs. On gfx950 the full fence makes dispatch
  23–30% slower, combine 35% slower and a concurrent GEMM 12% slower.
- **Cheap fence** (kept): each sending warp waits for its own stores to complete; after a group
  barrier one lane publishes the tail with a relaxed system store; the receiver reads with
  system-scope loads, which bypass every GPU cache.

The cheap fence rests on one premise: once a store is complete, the peer can see it. On gfx942 /
gfx950 a store to uncached memory is acknowledged only after it reaches memory, so plain payload
stores are enough. On gfx125x a store is acknowledged at the first cache level that covers its scope
(LLVM AMDGPUUsage, GFX125x memory model): `s_wait_storecnt 0` can return before a default-scope
payload reaches the peer, while the peer already sees the new tail. So on gfx1250 every payload
store carries `SCOPE_SYS` (`buffer_store ... th:TH_STORE_NT scope:SCOPE_SYS`), and waiting for
completion means waiting for system-scope visibility. On HIP V1, a `global_wb` before each publish
instead is 3–6× slower.

V2 follows the same rules: payload and TDM stores carry `SYS_STORE`, a warp waits for its stores
(`tensor_wait` / `wait_mem_all`) before it publishes a count or sets a flag, and all reads use
`SYS_LOAD`. notify and `barrier_block` already are release fence + system-scope atomics + acquire
fence; only their waits take the gfx12 form.

### 4. V2: fixed slots

`primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py`:

- **Dispatch**: a token's slot on rank `d` is this rank's offset in `d`'s receive order (from the
  full rank prefix matrix computed by notify) plus its index in the token stream this rank sends to
  `d`, the same order as V1's receiver, so `recv_x` and every per-token output match V1. Each block
  owns one channel (a contiguous range of tokens); each warp reads a token's row once and writes it
  to every destination rank. The block then tells each rank how many rows this channel sent, and a
  rank that has every count copies the rows from its heap into the output.
- **Combine**: the rows of `x` and their top-k weights are pushed back into a compact region on the
  source rank, laid out as [destination rank][index in the stream sent to it], so the slot of (rank
  `r`, token `t`) is `base_r + send_head[t, r]`; this rank's own rows stay in `x`. Only the rows a
  rank actually received are pushed, also under a `num_worst_tokens` allocation. Once a row has
  landed, the pushing warp sets its flag to the epoch of this combine. Reduce warps take 4 tokens at
  a time, in token order, from an atomic queue and add them as their flags arrive (bias first, then
  ranks in ascending order, the summation order of HIP / V1), so push and reduce overlap.
- **TDM**: rows go through LDS (one load, then one store per destination rank); wide rows (e.g.
  H=7168) that do not fit are moved in pieces.
- **Between two combines** a rank may push into a peer's region only after the peer has finished
  the previous reduce: a dispatch in between gives that barrier through its notify, otherwise the
  host launches a `launch_barrier`.

For callers:

- Outputs are bit-identical to V1. Handles use V2's own format (`send_head` holds stream indices)
  and cannot be mixed with `flydsl` / `cpp` handles.
- `num_sms` blocks are `num_sms` channels (need not be even); the chunk fields of `Config` are not
  used.
- **No CUDA-graph capture**: every call advances host state that its launch takes (the epochs of the
  counts and the flags, the base of the reduce queue), which a replay would reuse, so dispatch and
  combine raise under capture. Use `flydsl` for CUDA graphs.
- Limits: channels ≤ 1024; tokens per channel < 2^19; an 11-bit dispatch epoch with an epoch tag on
  the counts; a 4 MiB combine flag region (T × ranks ≤ 1 Mi rows); LDS per block ≤ 320 KiB.

### 5. Buffer sizing

The V2 heap scales with the token count, unlike V1's rings:

- heap ≥ `DATA_OFFSET` (about 4.25 MiB) + 2 × max(dispatch region, combine region), where the
  dispatch region is received tokens × (H bytes + 4 + top-k × (index bytes + 4) + scales × 4), and
  the combine region is this rank's tokens × ranks × align128(H bytes + top-k × 4).
- DeepSeek-V3 at EP4 (T 8192, H 7168 bf16, top-8) needs about 0.95 GB; GPT-OSS at EP4 (T 32768, H
  2880, top-4) about 1.55 GB, against about 128 MB for V1's rings at 32 CUs.
- `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank, hidden_bytes, num_ranks, num_topk)`
  gives the V2 size (0 for the ring backends). `init_buffer`, behind `moe_dispatch` /
  `DeepEPTokenDispatcher`, takes the larger of it and the ring hint.
- As Megatron does with DeepEP V2's `ElasticBuffer`, the token count comes from the first dispatch,
  and a larger dispatch reallocates the heap after a device sync and a barrier.
  `set_buffer_global_config(num_max_tokens_per_rank=...)` allocates up front instead.
- Every allocation all-reduces the group's largest token count and top-k, because peers write into
  this rank's heap; ranks may dispatch different numbers of tokens. Token counts that change per step
  without a known maximum (THD, dynamic CP) need `num_max_tokens_per_rank` set to the maximum, or
  `flydsl`.

### 6. Watchdog

Every spin loop has a timeout on the realtime clock. On gfx9 it prints the location and keeps
waiting after 30 s. On gfx1250 it prints and executes `s_trap 2` after 300 s, the C++ DeepEP's
`NUM_TIMEOUT_CYCLES`:

- A queue still spinning when the driver preempts or destroys it (queue preemption timeout 9 s) can
  leave MES unrecoverable on MI455X, so that the GPU reset fails and the node needs a reboot. A
  trapped queue has no waves left: the process exits within seconds and the GPU stays usable.
- 300 s rather than less, because a slow peer is not an error: a rank behind its peers from a data
  loader stall, a first compilation or a host stall must not kill the job.
- The cost: a process killed while its kernel is still deadlocked, before the trap, can still leave
  MES unrecoverable. Let a hung job reach the trap instead of killing it.

The combine ring also has a deadlock-free lower bound: a reduce warp advances its head only after
it finishes a token, so the published head lags by up to `reduce warps` positions while the sender
needs a whole chunk free. `combine_min_recv_buffer_tokens` checks `recv >= reduce warps + send - 1`
on the host and rejects configs below it.

## Performance

### Test Configuration

- **Device:** 2 × MI455X (`gfx1250`) of one node (EP2)
- **Shape:** Primus's MI455X DeepSeek-V3 proxy: 8192 tokens per rank (micro-batch 2 × sequence
  4096), hidden 7168 bf16, top-8 of 128 experts, routing limited to the top 4 of 8 expert groups,
  random router logits
- **Stack:** `amdprimus/amdprimus:tas-20261019`, torch 2.14.0 with ROCm 10.2, FlyDSL 0.3.4.1, RCCL
  2.32.3
- **Timing:** a 512 MB L2 flush before every iteration, the slowest rank of each iteration, median
  of 30 iterations, mean of 3 rounds
- **Baseline:** the RCCL all-to-all. "defaults" is what Megatron runs; "tuned" registers the buffers
  with `ncclCommRegister` (1.15–1.24× faster; nothing else in a 42-variant rccl-tests sweep moves it
  by more than 1%)

### DeepEP vs RCCL all-to-all

Notify + dispatch / combine, µs, at the same number of workgroups:

| | Token rows received per GPU | MB received per GPU | 32 CUs | 64 CUs | 128 CUs (RCCL's default) |
|---|---|---|---|---|---|
| RCCL all-to-all, defaults (Megatron alltoall dispatcher: one copy per expert) | 32,956 | 472.5 | 3420 / 3372 | 1778 / 1733 | 652 / 603 |
| RCCL all-to-all, tuned (Megatron alltoall dispatcher: one copy per expert) | 32,956 | 472.5 | 2421 / 2375 | 1285 / 1239 | 544 / 491 |
| RCCL all-to-all, tuned (deduplicated control: one copy per GPU) | 8,071 | 115.7 | 660 / 613 | 381 / 333 | 215 / 164 |
| `flydsl_v2` (one copy per GPU) | 8,071 | 115.7 | 331 / 346 | 216 / 195 | 183 / 150 |

- Megatron's alltoall dispatcher sends a token once per expert (about 4 of its 8 experts sit on the
  other GPU); DeepEP sends it once per GPU and copies it per expert on the receiver, so it moves 4.1×
  fewer bytes. The deduplicated row is a control: tuned RCCL moving exactly the bytes DeepEP moves.
- CUs are workgroups per kernel: `flydsl_v2` runs one 1024-thread workgroup per CU; RCCL runs
  256-thread workgroups, 128 at most at EP2, fewer with `NCCL_MAX_P2P_NCHANNELS` (checked with
  rocprofv3).
- notify + dispatch includes DeepEP's notify and its host wait; the RCCL rows include the all_gather
  of the counts and the device-to-host copy of the splits that Megatron does before its all-to-all.
- With the receive counts known (a cached handle, as in the backward pass) `flydsl_v2` takes 124 µs
  at 128 CUs against tuned RCCL's 492 µs; without the host wait (`num_worst_tokens`), 146 µs at 160
  CUs. It saturates at 160 CUs (within 5% of 256): 179 / 146 µs.
- V1 (`flydsl`) on the same input: 1691 / 1067 µs at 32 CUs, 410 / 226 µs at its saturation point
  of 224 CUs.

### Distance to the link

Per GPU and direction with both directions busy, receive counts known:

| | µs | GB/s | of the link |
|---|---:|---:|---:|
| `flydsl_v2` 160 CUs, cached dispatch | 123 | 943 | 61% |
| `flydsl_v2` 160 CUs, combine | 146 | 795 | 52% |
| RCCL all-to-all as Megatron does it, defaults | 602 | 785 | 51% |
| RCCL all-to-all as Megatron does it, tuned | 492 | 960 | 62% |
| RCCL all-to-all, deduplicated, defaults | 195 | 593 | 38% |
| RCCL all-to-all, deduplicated, tuned | 164 | 704 | 46% |
| link: copy engines, 470 MB per direction | | 1542 | 100% |

- A copy kernel on V2's own data path (TDM through LDS into the uncached heap, without slots,
  counts or flags) reaches 1491 GB/s at 256 blocks and 1462 GB/s at 160.
- Fitting time = fixed + bytes / bandwidth over 256 to 32768 tokens per rank at 160 CUs: the
  `flydsl_v2` dispatch streams at 1.28–1.32 TB/s and the combine at 1.24 TB/s, plus a fixed 36 µs
  (cached dispatch) and 59 µs (combine), against 15 µs for the bare copy kernel. At 8192 tokens the
  fixed part is 30–40% of the time. Tuned RCCL has a fixed 48–58 µs and streams at up to 1.07–1.12
  TB/s.
- The notify and its host wait add 56 µs to the dispatch, about 33 µs of it the host wait.

### V1 and V2 at EP2 and EP4

4 × MI455X, `Buffer` interface with `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1`, T=8192, H=7168 bf16,
random top-8 (64 experts at EP2, 256 at EP4), mean of 20 iterations on rank 0 and of 3 passes, ms,
dispatch (with the host wait for the notify counts) / combine. The RCCL column is an evenly split
`all_to_all_single` of T × top-k rows at RCCL's defaults.

| | SM | V1, gfx9 settings | V1, gfx1250 settings | V2 | RCCL a2a |
|---|---|---|---|---|---|
| EP2 | 32 | 2.13 / 2.22 | 1.76 / 1.07 | 0.33 / 0.35 | 0.60 |
| EP2 | 64 | 1.23 / 1.31 | 0.98 / 0.57 | 0.21 / 0.20 | |
| EP2 | 160 | 0.55 / 0.59 | 0.50 / 0.30 | 0.18 / 0.15 | |
| EP4 | 32 | 4.55 / 6.81 | 3.83 / 2.51 | 0.58 / 0.78 | 0.94 |
| EP4 | 64 | 2.41 / 2.01 | 2.13 / 1.37 | 0.40 / 0.42 | |
| EP4 | 160 | 1.11 / 0.86 | 0.98 / 0.64 | 0.36 / 0.34 | |

GPT-OSS shape (EP4, T 32768, H 2880, top-4, 160 SMs): V2 dispatch 353 µs, combine 336 µs.

### V1 settings on gfx1250

"gfx9 settings" above are the MI355X copy-piece table and the upstream chunks (EP2 24/10, EP4 6/9);
"gfx1250 settings", the defaults on gfx1250, are 8 KiB pieces and chunks EP2 24/48, EP4 6/24.

- **Copy pieces** (dispatch + combine total, ms, chunks EP2 24/10 and EP4 6/48; V2 uses TDM
  instead):

  | | MI355X table | 2 KiB | 4 KiB | 8 KiB |
  |---|---|---|---|---|
  | EP2 32 / 64 / 160 SM | 4.27 / 2.53 / 1.13 | 4.15 / 2.33 / 1.08 | 3.97 / 2.28 / 1.07 | **3.82 / 2.17 / 1.05** |
  | EP4 32 / 64 / 160 SM | 7.63 / 4.46 / 1.90 | 7.26 / 4.17 / 1.84 | 6.88 / 3.82 / 1.83 | **6.40 / 3.51 / 1.68** |

  8 KiB is 16 steps on wave32 at 64 VGPRs per lane; anything larger exceeds 128 VGPRs.
- **Combine chunks** (8 KiB pieces): EP2 24/10 → 24/48 takes 3.86 / 2.19 / 1.07 to 2.83 / 1.55 /
  0.80 ms (the combine halves); EP4 6/9 → 6/24 takes 10.27 / 3.67 / 1.64 to 6.35 / 3.50 / 1.62 ms.
  `Buffer.get_dispatch_config` / `get_combine_config(num_ranks, backend=None)` return the defaults
  of the backend in use.
- **Combine loads in flight** (`combine_loads_in_flight`, V1 and V2): 16, 12 and 8 remove the VGPR
  spills of 32 but are slower (V2 EP2 at 32 SMs: 0.352 ms at 32, 0.38–0.39 ms otherwise), so 32 is
  kept.

## Implementation Map

| Component | File |
| --- | --- |
| Per-arch traits | `primus_turbo/flydsl/deep_ep/arch.py` |
| gfx1250-aware helpers (`read_clock`, `wait_mem_all`, `copy_warp`, timeouts) | `primus_turbo/flydsl/deep_ep/common.py` |
| V1 kernels (notify, dispatch, combine) | `primus_turbo/flydsl/deep_ep/intranode_kernel.py` |
| Dispatch layout kernel | `primus_turbo/flydsl/deep_ep/layout_kernel.py` |
| V2 kernels | `primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py` |
| Launcher cache | `primus_turbo/flydsl/deep_ep/launch_cache.py` |
| Host runtime (V1 / V2) | `primus_turbo/pytorch/deep_ep/flydsl_intranode.py` |
| `Buffer`, backend selection, configs | `primus_turbo/pytorch/deep_ep/buffer.py` |
| No-C++ fallbacks (`PyConfig`, `PyEventHandle`) | `primus_turbo/pytorch/deep_ep/_compat.py` |
| Buffer sizing in `moe_dispatch` / `DeepEPTokenDispatcher` | `primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py` |

## Acknowledgements

- [**DeepEP**](https://github.com/deepseek-ai/DeepEP) (DeepSeek, MIT License) — the V1 kernels port
  DeepEP's intranode protocol, and V2 keeps its API and output order.
