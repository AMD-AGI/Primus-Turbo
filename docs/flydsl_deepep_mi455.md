# FlyDSL intranode DeepEP on MI455X (gfx1250): V1 port and V2

> This document describes the FlyDSL intranode DeepEP backends on MI455X (gfx1250): the port of FlyDSL V1, which keeps the cheap fence and makes the payload stores `SCOPE_SYS`, the V2 backend, and the validation and tuning data measured on MI455X (4 GPUs) on 2026-10-27. It builds on Xiaoming's MI455X design document (branch `feat/flydsl-deepep-gfx1250`). Paths are relative to the repository root.

## 0. Status

- **Backends**: `Buffer(..., intranode_backend=...)` or the environment variable `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`, one of `cpp` (HIP V1), `flydsl` (FlyDSL V1) and `flydsl_v2` (FlyDSL V2, gfx1250 only). The C++ V2 backend, a separate change, uses `cpp_v2`; the two value lists are merged when it lands.
  - gfx1250: the C++ DeepEP is not loaded (§5), the default is `flydsl`, and selecting `cpp` raises an error.
  - Other architectures: the C++ DeepEP is loaded and the default is `cpp`; the runtime refuses `flydsl_v2` there (§6).
- **Changes**:

  | Area | Content |
  |---|---|
  | FlyDSL V1 | Intranode backend for gfx942 / gfx950 (Xiaoming) |
  | Host side | The FlyDSL path does not depend on the C++ DeepEP (`_compat.py`); the C++ DeepEP is not loaded on gfx1250 (§5) |
  | V1 on gfx1250 | wave32, gfx12 split-counter waits, realtime clock, `SCOPE_SYS` loads and stores (§2, §4); the gfx1250 `read_clock`, `wait_mem_all` and `copy_warp` live in `deep_ep/common.py`, and `primus_turbo/flydsl/utils` is unchanged (§3) |
  | V2 | The `flydsl_v2` backend for gfx1250: fixed slots instead of rings, per-row flags and a reduce queue in the combine, LDS pieces for wide rows (§6) |
  | `ArchTraits` | The per-arch choices in `arch.py`, a NamedTuple (§3) |
  | Safety | Lower-bound check on the combine ring depth; `s_trap` after a 5 s spin timeout on gfx1250; the V2 combine no longer reads the top-k weights of an empty `x` (§6, §7) |
  | V1 tuning on gfx1250 | 8 KiB copy pieces; backend-aware `get_*_config` with larger combine chunks (§8) |
  | FlyDSL API | `fx.as_ir_value` instead of the private `buffer_ops._unwrap_value`, which #581 removed |
  | Buffer sizing | `init_buffer` sizes the `flydsl_v2` heap from the dispatched tokens (§6) |

- **Validation** (MI455X, 4 GPUs, §7): `test_intranode.py` passes on 2 and 4 GPUs; the V1 and V2 stress tests pass; the combine ring lower-bound probe passes. **No regression run on gfx950** (§1).
- **Performance** (DSv3 shape, §8): at EP4 with 160 SMs, V2 dispatch / combine take 0.36 / 0.34 ms, V1 0.98 / 0.64 ms, and RCCL `all_to_all_single` 0.94 ms.
- **Not done**: end-to-end training with V2; 8 GPUs.

## 1. Goals and non-goals

> The C++ DeepEP on MI455X is out of scope here (HIP V1 and C++ V2 are a separate change). So there is no `cpp` backend to compare against on MI455X and `test_flydsl_matches_cpp` is skipped; correctness relies on the DeepEP reference flow, the layout test, stress tests and probes, and V2 output is compared bit for bit with V1.

**Goals**
1. FlyDSL V1 is functionally correct on gfx1250 and tuned to reasonable performance.
2. FlyDSL V2 is functionally correct on gfx1250, with output bit-identical to V1.
3. The FlyDSL path does not depend on the C++ `deep_ep` module.
4. Generated code for gfx942 / gfx950 is unchanged: the gfx9 trait values match the values before these changes (§3), but this is **not verified on gfx950**.

**Non-goals**
- internode, rocSHMEM, low-latency.
- Porting the C++ DeepEP to MI455X (a separate change). When it lands, it must remove the "do not load the C++ DeepEP on gfx1250" check from `_compat.py` and restore the `cpp` tests.
- Fusion with GEMM (Mega MoE).

## 2. Hardware assumptions of the gfx950 implementation (vs gfx1250)

| # | Assumption on gfx950 | gfx1250 | How it is handled here |
|---|---|---|---|
| A1 | wave64 | wave32 | `WARP_SIZE` comes from `ArchTraits`; the notify xor butterfly starts at `WARP_SIZE / 2` (an offset of 32 on wave32 lowers to a wave64 permlane and the JIT fails outright); `copy_warp` in `common.py` copies over `WARP_SIZE` lanes |
| A2 | ballot is i64 | only the low 32 bits are valid | combine's `_wave_any` still uses the i64 ballot from `attn_helper`; on wave32 the high 32 bits are 0, so the test is unaffected (UT and stress pass) |
| A3 | unified `s_waitcnt` | split into load / store / ds / km counters; no `s_memrealtime` | `wait_mem_all()` in `common.py` uses `s_wait_dscnt / kmcnt / loadcnt / storecnt 0`; `read_clock()` uses `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` (100 MHz, same as gfx9) |
| A4 | cache policy `0x13` (sc0 \| sc1 \| nt) | TH in bits 2:0, SCOPE in bits 4:3; `0x13` decodes as TH_LU + SCOPE_DEV | loads and stores both use `0x19` (TH_NT \| SCOPE_SYS) |
| A5 | cheap fence: on the uncached heap, a store is acknowledged only once it reaches memory | a store is acknowledged at the first cache level that covers its scope | keep the cheap fence; payload stores carry `SCOPE_SYS` (§4) |
| A6 | copy pieces of 1 / 2 / 8 KiB by channel count (gfx9's vmcnt also counts stores) | loads and stores are counted separately; the piece size is limited only by VGPRs | 8 KiB (§8) |
| A7 | buffer resource: out-of-bounds loads return 0, out-of-bounds stores are dropped | confirmed on gfx1250 by Xiaoming's probe P1 for `num_records=0` and for fully out-of-bounds 16 B accesses; a 16 B access that straddles the bound is not checked per dword | relied on; the host ensures `hidden_bytes % 16 == 0`, so accesses are aligned |
| A8 | in-group sync spins on an LDS counter | hardware named barriers | not used |
| A10 | the host depends on the C++ `deep_ep` (`EventHandle`, `Config`, a C++ `Buffer` with a 0-byte heap) | — | `_compat.py` (§5) |
| A11 | number of ranks ∈ {2, 4, 8} | unchanged | tested on 2 and 4 GPUs (the node has only 4) |
| A12 | 1024-thread blocks, at most 128 VGPRs per thread | same | VGPRs / spills checked in the ISA (§8) |
| A13 | chunk and CU counts chosen for MI300 / MI355 | — | chunk defaults per backend (§8) |
| A14 | combine ring lower bound `recv >= reduce warps + send - 1` (15 reduce warps on wave64) | 31 on wave32 | host-side check (§7) |
| A15 | spin timeout of 30 s; print and keep waiting | MES enters an unrecoverable state when the driver preempts or destroys a queue that is spinning | `s_trap 2` after 5 s on gfx1250 (§7) |

## 3. `ArchTraits`

`primus_turbo/flydsl/deep_ep/arch.py`:

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

- **Usage**: `WARP_SIZE`, `SYS_LOAD`, `SYS_STORE` and `timed_out()` / `timeout_hit()` in `common.py` all read `TRAITS`; so do V1's copy pieces and the combine's `steps_per_pass`, and the matching places in V2.
- **`primus_turbo/flydsl/utils` is unchanged**: the helpers whose code differs on gfx1250, `read_clock()`, `wait_mem_all()` and `copy_warp()` (over `WARP_SIZE` lanes instead of 64), are DeepEP's own in `common.py` and pick their code from `TRAITS`; the launcher cache is `deep_ep/launch_cache.py`. The Mega MoE gfx1250 kernels (#582) likewise keep theirs in `flydsl/mega/gfx1250/prims.py`.
- **Difference from Xiaoming's design**: he passes `traits` as an argument to every builder and helper; here a module-level `TRAITS` is used, so most call sites stay unchanged. A process compiles for one target only, and FlyDSL's cache key includes the target, so a module-level constant is enough.
- **It must be a NamedTuple or another builtin container type**, see §9: FlyDSL's cache key records only scalars and builtin containers by value.
- **gfx1250 values**:

  | Field | gfx942 / gfx950 | gfx1250 | Reason |
  |---|---|---|---|
  | `wave_size` | 64 | 32 | |
  | `cpol_sys_load` | `0x13` | `0x19` | A4 |
  | `cpol_sys_store` | 0 | `0x19` | §4 |
  | `copy_piece_table` | table measured on MI355X | 8 KiB | §8 |
  | `combine_loads_in_flight` | 32 | 32 | smaller values remove the spills but are not faster (§8) |
  | `spin_timeout_cycles` / `trap_on_timeout` | 3e9 (30 s) / False | 5e8 (5 s) / True | §7 |

## 4. Memory ordering: the cheap fence and `SCOPE_SYS` stores on gfx1250

**Protocol** (V1, same as the C++ intranode path): one ring per (channel, destination rank), placed on the receiver's heap (`SymmetricMemory(uncached_buffer=True)`, i.e. `hipDeviceMallocUncached`). The sender writes the payload, then publishes the tail; the receiver sees the new tail, reads the payload and publishes the head when done; the sender reads the head to find free slots.

**Full fence vs cheap fence**
- Full fence: the tail is published with a system-scope release store and polled with acquire loads. On gfx1250 every publish does a full cache write-back (`global_wb scope:SCOPE_SYS`) and every polling iteration a full invalidate (`global_inv scope:SCOPE_SYS`), and the cost spills over to GEMMs running concurrently on the same GPU. Measured on gfx950: the full fence makes dispatch 23–30% slower, combine 35% slower and a concurrent GEMM 12% slower (comment in `csrc/include/primus_turbo/deep_ep/configs.h`).
- Cheap fence: each sending warp waits for its own stores to complete; after a group barrier, one lane publishes the tail with a relaxed system store; the receiver reads the payload with system-scope loads (which bypass every GPU cache). There is no full write-back or invalidate.

**Why payload stores carry `SCOPE_SYS` on gfx1250**

The cheap fence rests on one premise: once a store is complete, the peer can see it.
- On gfx942 / gfx950, a store to uncached memory is acknowledged only after it reaches memory, so the premise holds and the payload can use plain stores (`cpol_sys_store = 0`).
- On gfx125x, a store is acknowledged at the first cache level that covers its scope (LLVM AMDGPUUsage, GFX125x memory model). A default-scope store may be acknowledged before it reaches the peer's memory: when `s_wait_storecnt 0` returns, the payload is not necessarily visible yet, while the peer may already see the new tail.
- So every payload store carries `SCOPE_SYS` (`buffer_store ... th:TH_STORE_NT scope:SCOPE_SYS`, and system-scope relaxed scalar stores). Waiting for store completion then means waiting for system-scope visibility, and publishing the tail afterwards is safe.

**Evidence**
- The "phase 2 fast path" in §4 of Xiaoming's design document is exactly this scheme: the payload carries a SCOPE_SYS cpol, and the tail is published with a relaxed store after `s_wait_storecnt 0`.
- The same comparison was done on HIP V1: variant A (cheap fence, default-scope payload) is faster than variant B (payload with `SCOPE_SYS`) in most configurations, e.g. random routing at 64 SMs: dispatch 2244 vs 2731 µs. This is consistent with A's stores being acknowledged early. Variant C (each warp does a `global_wb` before publishing) is 3–6× slower. HIP variant B passed pytest and 12 / 12 stress runs.
- Xiaoming's experimental cheap fence (default-scope payload) passed his stress test (10 configurations × 1000 iterations), but a stress test cannot prove that no window exists, so it is not adopted here.
- The cost of `SCOPE_SYS` stores on FlyDSL has not been measured on its own (the comparison with Xiaoming's experimental version also differs in copy pieces).

**V2** follows the same ordering rules: payload and TDM stores carry `SYS_STORE`; a warp waits for its stores to complete (`tensor_wait` / `wait_mem_all`) before it publishes a row count or sets a flag; all reads use `SYS_LOAD`.

notify and `barrier_block` already are release fence + system-scope atomics + acquire fence; only their waits were changed to the gfx12 form.

## 5. Host side: no dependency on the C++ `deep_ep`

From Xiaoming's gfx1250 branch:
- `primus_turbo/pytorch/deep_ep/_compat.py`: uses the C++ `Config` / `EventHandle` when the C++ `deep_ep` can be imported, and Python versions otherwise (`PyConfig` with `get_nvl_buffer_size_hint`, `PyEventHandle` wrapping `torch.cuda.Event`). On gfx1250 the C++ `deep_ep` is not loaded even if `_C` was built with it (decided with FlyDSL's `get_rocm_arch()`, which does not initialize the GPU).
- `Buffer`: with a FlyDSL backend no C++ `Buffer` is constructed; the comm stream is `torch.cuda.Stream(priority=-1)`; `runtime` is a `_NoCppRuntime`, so internode / low-latency calls raise a clear error.
- Default backend: `cpp` when the C++ can be loaded, otherwise `flydsl`.
- Tests: without the C++, the `cpp` parameter and `test_flydsl_matches_cpp` are skipped.

## 6. V2 (`flydsl_v2`)

**Design** (`primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py`)
- In V1 each (channel, rank) pair is a ring with one head / tail handshake per chunk, and the receiver then copies from the ring into `recv_x`. V2 drops the rings: every token has a fixed slot on each rank it goes to, and the sender writes there directly without waiting for the receiver.
- dispatch: a token's slot on rank `d` = this rank's offset in `d`'s receive order (from the rank prefix matrix computed by notify; V2 needs the whole matrix, `full_rank_matrix`) + its index in the token stream this rank sends to `d`. This is the same order as V1's receiver, so `recv_x` and every per-token output match V1. Each block owns one channel (a contiguous range of tokens), and each warp reads a token's row once and writes it to every destination rank. After the rows are written, the block tells each rank how many rows this channel sent; once a rank has the counts, it copies the rows from its own heap into the output.
- combine: the rows of `x` are pushed, together with their top-k weights, back into a compact region on the source rank laid out as [destination rank][index in the stream the source rank sent to it], so the slot of (rank `r`, token `t`) is `base_r + send_head[t, r]`; this rank's own rows stay in `x` and are not pushed. Once a row has landed, the pushing warp sets that slot's flag to the epoch of this combine. The reduce takes 4 tokens at a time, in token order, from an atomic queue and adds them as their flags arrive (bias first, then ranks in ascending order, the same summation order as HIP / V1), so push and reduce overlap.
- Rows go through LDS via TDM (one load, then one store per destination rank). The kernel also has a VGPR-copy path for other targets, but it has never run, so the runtime accepts `flydsl_v2` on gfx1250 only and the tests run it only there. Wide rows (e.g. H=7168) that do not fit in LDS are moved in pieces.
- Between two combines: a rank may push into a peer's combine region only after the peer has finished the previous reduce. With a dispatch in between, the dispatch's notify is that barrier; otherwise the host launches a `launch_barrier`.

**Differences for callers**
- Outputs are bit-identical to V1; handles use V2's own format (`send_head` holds stream indices) and cannot be mixed with V1 / `cpp` handles. Using them in pairs within one `Buffer` is fine.
- `num_sms` blocks are `num_sms` channels, which need not be even; the chunk fields of `Config` are not used.
- **The heap size scales with the token count**: heap ≥ `DATA_OFFSET` (about 4.25 MiB) + 2 × max(dispatch region, combine region). Dispatch region = received tokens × (H bytes + 4 + top-k × (index bytes + 4) + scales × 4); combine region = this rank's tokens × number of ranks × align128(H bytes + top-k × 4). For example, DSv3 EP4 (T 8192, H 7168 bf16, top-8) needs about 0.95 GB and GPT-OSS EP4 (T 32768, H 2880, top-4) about 1.55 GB. `Config.get_nvl_buffer_size_hint` sizes V1's rings (about 128 MB for GPT-OSS EP4 at 32 CUs), which is not enough for V2.
- **Who sizes the heap**: `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank, hidden_bytes, num_ranks, num_topk)` gives the V2 size (0 for the ring backends), and `init_buffer`, behind `moe_dispatch` / `DeepEPTokenDispatcher`, takes the larger of it and the ring hint. As Megatron does with DeepEP V2's `ElasticBuffer`, the token count comes from the first dispatch, and a larger dispatch later reallocates the heap, after a device sync and a barrier since peers may still be writing into the old one; `set_buffer_global_config(num_max_tokens_per_rank=...)` allocates up front instead. Every rank must dispatch the same number of tokens, as DeepEP V2 requires; in Megatron / Primus that is S / TP × B on every rank.
- Limits: channels ≤ 1024; tokens per channel < 2^19; an 11-bit dispatch epoch (wraps after 2047; the counts carry an epoch tag); a 4 MiB combine flag region, i.e. T × number of ranks ≤ 1 Mi rows; LDS per block ≤ 320 KiB.

**Bug fixed**: to read the top-k weights, the reduce first loaded from every rank with raw global loads and then discarded the ranks the token was not routed to. When such a rank was this rank, the address came from the combine's `topk_weights`; when a rank received 0 tokens in that call, that was an empty tensor with a null address, so the load accessed illegal memory (with tokens it could also read one row past the end). Now a rank the token is not routed to is read from a slot on this rank's heap instead. Regression test `test_rank_receives_nothing`: rank 0 receives no tokens.

## 7. Correctness

**Unit tests** (`tests/pytorch/deep_ep/test_intranode.py`, added to the gfx1250 allow list in `tests/conftest.py`)
- `test_intranode`: the DeepEP reference flow (bf16 / fp8 + scales, with and without top-k, sync / async, `previous_event`, `num_worst_tokens`, cached dispatch, combine), once each for `flydsl` and `flydsl_v2` (`flydsl_v2` on gfx1250 only).
- `test_dispatch_layout`: the layout against a torch reference.
- `test_rank_receives_nothing`: a rank receives 0 tokens. `flydsl_v2` fails without the fix in §6.
- `test_destroy_and_recreate`: `Buffer.destroy()` (a collective: every rank waits for the others before freeing its heap) and a new buffer on the same group, for `flydsl` and `flydsl_v2`.
- `test_flydsl_matches_cpp`: bit-for-bit comparison with the C++ backend; skipped on gfx1250.
- Result: 8 tests pass and 2 are skipped, on 2 GPUs and on 4 GPUs.

**Stress test** (ported from Xiaoming's `stress.py`, with V2 added)
- 10 configurations (num_sms, send, recv, T, H): (2, 1, 2, 37, 256), (2, 2, 4, 513, 7168), (8, 1, 2, 1000, 2048), (8, 4, 8, 4096, 7168), (8, 8, 16, 2049, 4096), (24, 6, 256, 4096, 7168), (24, 3, 7, 777, 1024), (8, 8, 256, 1, 7168), (8, 4, 8, 1000, 2880), (24, 6, 256, 4096, 2880). Dispatch uses small rings to force frequent wrap-around and back-pressure, and the combine ring is at its lower bound; 30% of the top-k entries are set to -1; every iteration checks each row against its source token through `src_idx` and checks the combine result.
- V1: 500 iterations on 2 GPUs and 200 on 4 GPUs pass. V2 (with that fix): 200 iterations on 2 GPUs and 100 on 4 GPUs pass.
- V2 before the fix: on 2 GPUs, iteration 94 of the T=1 configuration failed (in that iteration both ranks sent their token only to rank 1).

**Combine ring lower bound**
- A reduce warp advances its head only after it finishes a token, so the published head lags the oldest in-flight token by at most `reduce warps` positions, while the sender also needs a whole chunk of free slots. The deadlock-free condition is `recv >= reduce warps + send - 1`, with 31 reduce warps on wave32. The C++ version has the same logic but does not check it.
- The host checks it (`combine_min_recv_buffer_tokens`) and raises an error for a config below the bound.
- Probe: every token sent to every rank, a single channel, the ring exactly at the bound (send 1 / 8 / 24, recv 31 / 38 / 54): passes. Xiaoming showed that one slot below the bound always times out (trap). Dispatch has no such constraint (the receiver consumes whole chunks).

**Spin timeout**
- Xiaoming hit this on MI455X: when the combine deadlocked, the watchdog only printed and kept spinning, the driver failed to reset the queue (`MES failed to respond`), the GPU reset did not complete either, and in the end the machine had to be rebooted. The driver's queue preemption timeout is 9 s. So on gfx1250 the timeout is 5 s, after which the location is printed and `s_trap 2` is executed; he verified that the trap aborts the queue cleanly, the process exits within 3 s, and the GPU stays usable.
- Whether skew between ranks caused by first-time compilation can trigger it by mistake: the 4-GPU UT with an empty FlyDSL cache (40 s) did not trigger it. In training, ranks that drift apart by more than 5 s (e.g. first compilation without a cache, or uneven host stalls) will trap; keep this in mind.

**Host heap corruption at exit (a HIP runtime bug, not DeepEP)**
- Seen in V2 stress runs (glibc `corrupted double-linked list` after `STRESS PASS`, or a SIGSEGV at exit on 4 GPUs); Xiaoming also hit it once with his V1 port. It happens inside `dist.destroy_process_group()`, in RCCL's `ncclCommDestroy` → `hipMemUnmap`, and just as often without any DeepEP code (a few percent of 2-rank runs; every run under glibc `malloc.perturb`). The V2 fix in §6 did not fix it; the clean runs after it were luck.
- Cause: a use after free in the HIP runtime's VMM import path. rocm-systems [#11606](https://github.com/ROCm/rocm-systems/pull/11606) (2026-10-09) made the last release of an imported handle free memory that other mappings still use; [#13562](https://github.com/ROCm/rocm-systems/pull/13562) (2026-10-19) fixed it. The ROCm SDK used here (`10.2.0a20261011`) has the bug. RCCL 2.32 imports peer VMM handles for its cuMem P2P transport and its symmetric windows and unmaps them when a communicator is destroyed.
- Workarounds until the SDK has #13562: `NCCL_CUMEM_ENABLE=0`, or skip `destroy_process_group()` at exit.

**Not covered**: gfx950 regression; 8 GPUs; fp8 + scales, `num_worst_tokens` and cached dispatch in the stress test (the UT covers them).

## 8. Performance (MI455X, 4 GPUs)

**Method**: `Buffer` interface, `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1`, per rank T=8192, H=7168 bf16, random top-8 (64 experts in total at EP2, 256 at EP4); mean of 20 iterations on rank 0, in ms. Dispatch includes the host wait for the notify counts (`num_worst_tokens` is not set). The RCCL row is an evenly split `all_to_all_single` of T × top-k rows, the same number of rows Megatron's alltoall sends. A single pass can vary by up to 20%, so every number below is the mean of 3 passes, and every comparison uses a fresh FlyDSL cache directory (§9).

**Results** (dispatch / combine)

| | SM | FlyDSL V1 (before tuning) | FlyDSL V1 (after tuning) | FlyDSL V2 | RCCL a2a |
|---|---|---|---|---|---|
| EP2 | 32 | 2.13 / 2.22 | 1.76 / 1.07 | 0.33 / 0.35 | 0.60 |
| EP2 | 64 | 1.23 / 1.31 | 0.98 / 0.57 | 0.21 / 0.20 | |
| EP2 | 160 | 0.55 / 0.59 | 0.50 / 0.30 | 0.18 / 0.15 | |
| EP4 | 32 | 4.55 / 6.81 | 3.83 / 2.51 | 0.58 / 0.78 | 0.94 |
| EP4 | 64 | 2.41 / 2.01 | 2.13 / 1.37 | 0.40 / 0.42 | |
| EP4 | 160 | 1.11 / 0.86 | 0.98 / 0.64 | 0.36 / 0.34 | |

- "Before tuning": the MI355X copy-piece table and the upstream default chunks (EP2 24/10, EP4 6/9). "After tuning": 8 KiB pieces, chunks EP2 24/48 and EP4 6/24 (the defaults here).
- Xiaoming's EP2 numbers with the same script: with the default full fence, 3.04 / 4.60 at 32 SMs and 2.75 / 4.17 at 64 SMs; with the experimental cheap fence (default-scope payload, 2 KiB pieces), 1.72 / 2.12 at 32 SMs and 1.00 / 1.22 at 64 SMs.
- GPT-OSS shape (EP4, T 32768, H 2880, top-4, 160 SMs): V2 dispatch 353 µs, combine 336 µs; HIP V1 (with the gfx1250 fixes) 1068 / 978 µs.

**Tuning log**
- **Copy pieces** (V1; V2 uses TDM on gfx1250 and does not use this table): dispatch + combine total, chunks EP2 24/10 and EP4 6/48:

  | | MI355X table | 2 KiB | 4 KiB | 8 KiB |
  |---|---|---|---|---|
  | EP2 32 / 64 / 160 SM | 4.27 / 2.53 / 1.13 | 4.15 / 2.33 / 1.08 | 3.97 / 2.28 / 1.07 | **3.82 / 2.17 / 1.05** |
  | EP4 32 / 64 / 160 SM | 7.63 / 4.46 / 1.90 | 7.26 / 4.17 / 1.84 | 6.88 / 3.82 / 1.83 | **6.40 / 3.51 / 1.68** |

  8 KiB is 16 steps on wave32 at 64 VGPRs per lane; anything larger exceeds the 128-VGPR limit.
- **Combine loads in flight L** (V1 and V2 reducers, `steps_per_pass = L // (ranks + biases)`): L=16, 12 and 8 all remove the spills (at L=32 the EP4 V1 / V2 combine spills 59 / 75 VGPRs) but are not faster, e.g. V2 EP2 at 32 SMs: 0.352 ms at L=32 and 0.38–0.39 ms at L=16 / 12 / 8; V1 EP4 at 32 SMs: 6.37 ms at L=32 and 6.97–7.43 ms otherwise. 32 is kept. Xiaoming chose 12 to avoid spills; here that is slightly slower overall.
- **chunk** (8 KiB pieces, dispatch + combine total): EP2 from 24/10 to 24/48, 3.86 / 2.19 / 1.07 → 2.83 / 1.55 / 0.80 (the combine halves; 24/96 is another 2–7% faster, close to the noise); EP4 from 6/9 to 6/24, 10.27 / 3.67 / 1.64 → 6.35 / 3.50 / 1.62 (combine 24 / 48 / 96 are within the noise; dispatch 6 and 24 are about 4% apart).
- **Where the defaults live**: `Buffer.get_dispatch_config` / `get_combine_config(num_ranks, backend=None)`; without `backend` they use `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND` or the default backend, and `Buffer` passes its own backend. FlyDSL combine chunks on gfx1250: 48 at EP2 and 24 at EP4; dispatch unchanged (24 / 6). The optimum for HIP V1 on gfx1250 is different; its PR adds the `cpp` values under the same parameter.

**Registers** (after tuning, kernels compiled at 32 SMs, VGPRs / VGPR spills): V1 dispatch 112 / 0; V1 combine 128 / 22 (EP2) and 128 / 61 (EP4); V2 dispatch 50 / 0 (plus 28 SGPR spills at EP4); V2 combine 126 / 8 (EP2) and 128 / 75 (EP4); notify, layout and cached notify have no VGPR spills.

## 9. FlyDSL compile cache

- FlyDSL stores compiled kernels in `~/.flydsl/cache` by default (`FLYDSL_RUNTIME_CACHE_DIR` changes the location). The persistent cache key contains: a fingerprint of FlyDSL's own sources, the source of the jit function, the sources of the same-directory functions it references, the scalars in its closure, and a snapshot of the module-level globals those functions read.
- The globals snapshot **records only scalars and builtin containers (tuple / list / dict / set) by value**; any other object is recorded only by its type name.
- `ArchTraits` started out as a frozen dataclass, so changing values in `arch.py` did not change the cache key and stale kernels kept being used. Measured (EP2, 32 SMs, V1 dispatch): the default cache held a kernel compiled earlier with 1 KiB pieces, 2.143 ms; a fresh cache, 1.724 ms; after the switch to a NamedTuple, with the default cache (recompiled), 1.725 ms.
- Rules:
  - Module-level values that a kernel builder reads must be scalars or builtin containers, or be written directly in the function's source; `ArchTraits` must stay a NamedTuple.
  - For comparison experiments, use a new `FLYDSL_RUNTIME_CACHE_DIR` for each variant. After changing something a kernel depends on that the cache key cannot see, clear `~/.flydsl/cache` (Xiaoming also hit "the change has no effect" on gfx950).

## 10. Risks and pitfalls

- **Memory ordering**: errors show up as sporadic wrong data or hangs and do not necessarily reproduce every time. Run the stress test after any change to the publish path.
- **A while loop whose loop variable is the return value of an LDS atomic hangs** (reproduced by Xiaoming on gfx950). V2's reduce queue also drives a while loop with the return value of an atomic (lane 0 does a global atomic add and broadcasts the result to the warp through LDS); it works now, but take care when changing it.
- **Uniform values must be in SGPRs**: addresses and flags that come from vector loads need a `readfirstlane`, otherwise every buffer access becomes a waterfall loop.
- **In V1's `rank_prefix_matrix` only column `rank` is meaningful**; the other columns are leftovers in the heap (the same in the C++ version). V2 fills the whole matrix.
- **With `num_worst_tokens`, the rows beyond the actual receive count** are uninitialized memory (only `recv_topk_idx` is filled with -1); truncate them before comparing.
- **`common.py` has its own `read_clock` and `copy_warp`** beside the `prims` ones (on gfx9 they emit the same code): a fix to the `prims` versions may need to be made here too.
- **No gfx950 regression run**: the gfx9 values match the values before these changes, but nothing has been run on gfx950.
- **Merging with the C++ V2 change**: both add `intranode_backend` and `get_intranode_backend` to `Buffer` and read the same environment variable; when merging, unify the values as `cpp` / `cpp_v2` / `flydsl` / `flydsl_v2` and drop the check that keeps the C++ DeepEP unloaded on gfx1250.

## 11. Reproducing

- Unit tests (inside the container; `HIP_VISIBLE_DEVICES` sets the number of GPUs):

  ```bash
  HIP_VISIBLE_DEVICES=0,1 python tests/pytorch/deep_ep/test_intranode.py -v
  HIP_VISIBLE_DEVICES=0,1,2,3 python tests/pytorch/deep_ep/test_intranode.py -v
  ```

- The stress test, the combine ring lower-bound probe and the DSv3-shape benchmark behind §7 and §8 are not part of this repository; their parameters and results are given in §7 and §8.
- A benchmark that calls the ops the way Primus does (`turbo.ops.moe_dispatch` + `moe_combine`, forward and backward, `async_finish=True`, `allocate_on_comm_stream=True`) and end-to-end training have not been run yet. §9 of Xiaoming's document has ready-made scripts and the DSv3 proxy command, as well as his DSv3 EP2 end-to-end result on MI455X (FlyDSL V1 with the default full fence 1318 ms/iter, RCCL all-to-all 1133 ms/iter, nearly identical lm loss).

## 12. To do

- End-to-end runs with V2 (DSv3 proxy, GPT-OSS 20B / 120B).
- V2 with a different token count per rank (THD, dynamic CP): an all-reduce of the maximum before sizing, as Megatron's HybridEP path offers.
- A benchmark that calls the ops the way Primus does.
- gfx950 regression; 8 GPUs.
- Whether the VGPR spills of the V1 / V2 combine and the SGPR spills of the V2 dispatch are worth addressing.
- Merge with the HIP V1 / C++ V2 change (backend values, the C++ check on gfx1250, the `cpp` chunk defaults on gfx1250, a token size hint for `cpp_v2`).
