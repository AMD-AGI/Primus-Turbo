# FlyDSL intranode DeepEP on MI455X (gfx1250): V1 port and V2

> This document describes the FlyDSL intranode DeepEP backends on MI455X (gfx1250): the port of FlyDSL V1, which keeps the cheap fence and makes the payload stores `SCOPE_SYS`, the V2 backend, and the validation, tuning and performance data measured on MI455X on 2026-10-27 and 2026-10-28. It builds on Xiaoming's MI455X design document (branch `feat/flydsl-deepep-gfx1250`). Paths are relative to the repository root.

## 0. Status

- **Backends**: `Buffer(..., intranode_backend=...)` or the environment variable `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`, one of `cpp` (HIP V1), `flydsl` (FlyDSL V1) and `flydsl_v2` (FlyDSL V2; tuned for gfx1250, experimental elsewhere). The C++ V2 backend, a separate change, uses `cpp_v2`; the two value lists are merged when it lands.
  - gfx1250: the C++ DeepEP is not loaded (§5), the default is `flydsl_v2`, and selecting `cpp` raises an error. Use `flydsl` for CUDA graphs or a different token count per rank (§6).
  - Other architectures: the C++ DeepEP is loaded and the default is `cpp`; `flydsl_v2` runs there with a register copy instead of TDM and is experimental (§6, `flydsl_deepep_v2_unified.md`).
  - How to choose, size and tune a backend: [the DeepEP README](../primus_turbo/pytorch/deep_ep/README.md#intranode-backend).
- **Changes**:

  | Area | Content |
  |---|---|
  | FlyDSL V1 | Intranode backend for gfx942 / gfx950 (Xiaoming) |
  | Host side | The FlyDSL path does not depend on the C++ DeepEP (`_compat.py`); the C++ DeepEP is not loaded on gfx1250 (§5) |
  | V1 on gfx1250 | wave32, gfx12 split-counter waits, realtime clock, `SCOPE_SYS` loads and stores (§2, §4); the gfx1250 `read_clock`, `wait_mem_all` and `copy_warp` live in `deep_ep/common.py`, and `primus_turbo/flydsl/utils` is unchanged (§3) |
  | V2 | The `flydsl_v2` backend for gfx1250: fixed slots instead of rings, per-row flags and a reduce queue in the combine, LDS pieces for wide rows (§6) |
  | `ArchTraits` | The per-arch choices in `arch.py`, a NamedTuple (§3) |
  | Safety | Lower-bound check on the combine ring depth; `s_trap` after a 5 s spin timeout on gfx1250; the V2 combine no longer reads the top-k weights of an empty `x`; `Buffer.destroy()` frees the heap only once every rank is done; `flydsl_v2` refuses CUDA-graph capture (§6, §7) |
  | V1 tuning on gfx1250 | 8 KiB copy pieces; backend-aware `get_*_config` with larger combine chunks (§8) |
  | FlyDSL API | `fx.as_ir_value` instead of the private `buffer_ops._unwrap_value`, which #581 removed |
  | Buffer sizing | `init_buffer` sizes the `flydsl_v2` heap from the dispatched tokens (§6) |

- **Validation** (§7): `test_intranode.py` on MI455X (2 and 4 GPUs) and in the gfx942 CI; the V1 and V2 stress tests and the combine ring lower-bound probe pass on MI455X. On gfx942 / gfx950 the V1 kernels compile to the same ISA as before these changes; nothing has been run on gfx950 (§1).
- **Performance** (DeepSeek-V3 proxy shape, EP2, §8): at 128 CUs, the workgroup count of RCCL's all-to-all, `flydsl_v2` takes 183 µs for notify + dispatch and 150 µs for combine: 3.0× and 3.3× faster than the tuned RCCL all-to-all of Megatron's alltoall dispatcher, and 1.17× and 1.09× faster than a tuned all-to-all that moves only the bytes DeepEP moves. At 160 CUs its dispatch with known receive counts moves 943 GB/s per GPU, 61% of the measured 1.54 TB/s link.
- **Not done**: end-to-end training with V2; EP4 numbers with the §8 method; 8 GPUs.

## 1. Goals and non-goals

> The C++ DeepEP on MI455X is out of scope here (HIP V1 and C++ V2 are a separate change). So there is no `cpp` backend to compare against on MI455X and `test_flydsl_matches_cpp` is skipped; correctness relies on the DeepEP reference flow, the layout test, stress tests and probes, and V2 output is compared bit for bit with V1.

**Goals**
1. FlyDSL V1 is functionally correct on gfx1250 and tuned to reasonable performance.
2. FlyDSL V2 is functionally correct on gfx1250, with output bit-identical to V1.
3. The FlyDSL path does not depend on the C++ `deep_ep` module.
4. Generated code for gfx942 / gfx950 is unchanged: the gfx9 trait values are the constants from before these changes (§3), and all six V1 kernels compile to byte-identical gfx942 and gfx950 ISA before and after them (2 and 4 ranks). Nothing has been run on gfx950; the gfx942 CI runs `cpp` and `flydsl`.

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
- Default backend (`default_intranode_backend`): `flydsl_v2` on gfx1250; elsewhere `cpp` when the C++ can be loaded, otherwise `flydsl`. An explicit choice is never replaced: `flydsl_v2` off gfx1250 raises and names the alternatives.
- Tests: without the C++, the `cpp` parameter and `test_flydsl_matches_cpp` are skipped.

## 6. V2 (`flydsl_v2`)

**Design** (`primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py`)
- In V1 each (channel, rank) pair is a ring with one head / tail handshake per chunk, and the receiver then copies from the ring into `recv_x`. V2 drops the rings: every token has a fixed slot on each rank it goes to, and the sender writes there directly without waiting for the receiver.
- dispatch: a token's slot on rank `d` = this rank's offset in `d`'s receive order (from the rank prefix matrix computed by notify; V2 needs the whole matrix, `full_rank_matrix`) + its index in the token stream this rank sends to `d`. This is the same order as V1's receiver, so `recv_x` and every per-token output match V1. Each block owns one channel (a contiguous range of tokens), and each warp reads a token's row once and writes it to every destination rank. After the rows are written, the block tells each rank how many rows this channel sent; once a rank has the counts, it copies the rows from its own heap into the output.
- combine: the rows of `x` are pushed, together with their top-k weights, back into a compact region on the source rank laid out as [destination rank][index in the stream the source rank sent to it], so the slot of (rank `r`, token `t`) is `base_r + send_head[t, r]`; this rank's own rows stay in `x` and are not pushed. Once a row has landed, the pushing warp sets that slot's flag to the epoch of this combine. The reduce takes 4 tokens at a time, in token order, from an atomic queue and adds them as their flags arrive (bias first, then ranks in ascending order, the same summation order as HIP / V1), so push and reduce overlap.
- Rows go through LDS via TDM (one load, then one store per destination rank). Other targets, which have no TDM, take a register (VGPR) copy instead; it passes the tests on gfx1250 with TDM turned off and is experimental elsewhere (`flydsl_deepep_v2_unified.md`). Wide rows (e.g. H=7168) that do not fit in LDS are moved in pieces.
- Between two combines: a rank may push into a peer's combine region only after the peer has finished the previous reduce. With a dispatch in between, the dispatch's notify is that barrier; otherwise the host launches a `launch_barrier`.

**Differences for callers**
- Outputs are bit-identical to V1; handles use V2's own format (`send_head` holds stream indices) and cannot be mixed with V1 / `cpp` handles. Using them in pairs within one `Buffer` is fine.
- `num_sms` blocks are `num_sms` channels, which need not be even; the chunk fields of `Config` are not used.
- **No CUDA-graph capture**: every call advances host state that its launch takes (the epoch of the dispatch counts and the combine flags, the base of the reduce queue); a graph replay would rerun the old values and take the previous call's flags as fresh, so dispatch and combine raise under capture. `cpp` and `flydsl` keep no such host state.
- **The heap size scales with the token count**: heap ≥ `DATA_OFFSET` (about 4.25 MiB) + 2 × max(dispatch region, combine region). Dispatch region = received tokens × (H bytes + 4 + top-k × (index bytes + 4) + scales × 4); combine region = this rank's tokens × number of ranks × align128(H bytes + top-k × 4). For example, DSv3 EP4 (T 8192, H 7168 bf16, top-8) needs about 0.95 GB and GPT-OSS EP4 (T 32768, H 2880, top-4) about 1.55 GB. `Config.get_nvl_buffer_size_hint` sizes V1's rings (about 128 MB for GPT-OSS EP4 at 32 CUs), which is not enough for V2.
- **Who sizes the heap**: `Buffer.get_token_buffer_size_hint(num_max_tokens_per_rank, hidden_bytes, num_ranks, num_topk)` gives the V2 size (0 for the ring backends), and `init_buffer`, behind `moe_dispatch` / `DeepEPTokenDispatcher`, takes the larger of it and the ring hint. As Megatron does with DeepEP V2's `ElasticBuffer`, the token count comes from the first dispatch, and a larger dispatch later reallocates the heap, after a device sync and a barrier since peers may still be writing into the old one; `set_buffer_global_config(num_max_tokens_per_rank=...)` allocates up front instead. Every rank must dispatch the same number of tokens, as DeepEP V2 requires; in Megatron / Primus that is S / TP × B on every rank.
- Limits: channels ≤ 1024; tokens per channel < 2^19; an 11-bit dispatch epoch (wraps after 2047; the counts carry an epoch tag); a 4 MiB combine flag region, i.e. T × number of ranks ≤ 1 Mi rows; LDS per block ≤ 320 KiB.

**Bug fixed**: to read the top-k weights, the reduce first loaded from every rank with raw global loads and then discarded the ranks the token was not routed to. When such a rank was this rank, the address came from the combine's `topk_weights`; when a rank received 0 tokens in that call, that was an empty tensor with a null address, so the load accessed illegal memory (with tokens it could also read one row past the end). Now a rank the token is not routed to is read from a slot on this rank's heap instead. Regression test `test_rank_receives_nothing`: rank 0 receives no tokens.

## 7. Correctness

**Unit tests** (`tests/pytorch/deep_ep/test_intranode.py`, added to the gfx1250 allow list in `tests/conftest.py`)
- `test_intranode`: the DeepEP reference flow (bf16 / fp8 + scales, with and without top-k, sync / async, `previous_event`, `num_worst_tokens`, cached dispatch, combine), once each for `flydsl` and `flydsl_v2`.
- `test_dispatch_layout`: the layout against a torch reference.
- `test_rank_receives_nothing`: a rank receives 0 tokens. `flydsl_v2` fails without the fix in §6.
- `test_destroy_and_recreate`: `Buffer.destroy()` (a collective: every rank waits for the others before freeing its heap) and a new buffer on the same group, for `flydsl` and `flydsl_v2`.
- `test_flydsl_v2_refuses_graph_capture`: `flydsl_v2` dispatch and combine raise under CUDA-graph capture and the buffer keeps working.
- `test_flydsl_matches_cpp`: bit-for-bit comparison with the C++ backend; skipped on gfx1250.
- `test_moe_ops_size_the_v2_buffer` (gfx1250): `moe_dispatch` / `moe_combine` on `flydsl_v2` with no preset buffer, 4096 then 12288 tokens; the buffer grows. `TokenBufferSizeHintTestCase` (host only): the size hint against the V2 heap layout.
- Result on MI455X: 12 tests pass and 2 are skipped (`test_flydsl_matches_cpp`), on 2 GPUs and on 4 GPUs. With the ROCm SDK used here a test process can still crash after its test, inside `dist.destroy_process_group()` (the HIP bug below; 4 of 7 four-GPU runs of `test_moe_ops_size_the_v2_buffer`, 1 of 10 two-GPU runs of `test_rank_receives_nothing`); with `NCCL_CUMEM_ENABLE=0` none of 15 runs did. gfx942 CI: every `cpp` and `flydsl` test passes, `test_destroy_and_recreate` and `test_flydsl_matches_cpp` included.

**Stress test** (ported from Xiaoming's `stress.py`, with V2 added; run on an earlier revision of this branch, before its rebase onto `main`)
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

**Not covered**: runs on gfx950 (its generated code is unchanged, §1); 8 GPUs; fp8 + scales, `num_worst_tokens` and cached dispatch in the stress test (the UT covers them).

## 8. Performance (MI455X)

**DeepSeek-V3 proxy shape, EP2** (2 × MI455X of one node)
- Shape: Primus's MI455X DeepSeek-V3 proxy (`examples/megatron/configs/MI455X/deepseek_v3-BF16-pretrain.yaml` in Primus): 8192 tokens per rank (micro-batch 2 × sequence 4096), hidden 7168 bf16, top-8 of 128 experts, routing limited to the top 4 of 8 expert groups, random router logits.
- Stack: image `amdprimus/amdprimus:tas-20261019`, torch 2.14.0 with ROCm 10.2, FlyDSL 0.3.4.1, RCCL 2.32.3 (the image's build).
- Timing: a 512 MB L2 flush before every iteration, the slowest rank of each iteration, median of 30 iterations, mean of 3 rounds; every DeepEP output and every RCCL round trip checked.
- RCCL: "defaults" is what Megatron runs; "tuned" registers the buffers with `ncclCommRegister`, which makes the all-to-all 1.15–1.24× faster. Nothing else in the RCCL team's rccl-tests sweep (symmetric windows, hipGraph launch, `NCCL_CTA_POLICY`, `RCCL_FORCE_CE`; 42 variants) moves it by more than 1%, except the copy-engine path (symmetric buffers with `NCCL_CTA_POLICY=2`, eager launch), which is slower.

DeepEP against RCCL all-to-all at the same number of workgroups (µs, notify + dispatch / combine):

| | Token rows received per GPU | MB received per GPU | 32 CUs | 64 CUs | 128 CUs (RCCL's default) |
|---|---|---|---|---|---|
| RCCL all-to-all, defaults (Megatron alltoall dispatcher: one copy of a token per expert) | 32,956 | 472.5 | 3420 / 3372 | 1778 / 1733 | 652 / 603 |
| RCCL all-to-all, tuned (Megatron alltoall dispatcher: one copy of a token per expert) | 32,956 | 472.5 | 2421 / 2375 | 1285 / 1239 | 544 / 491 |
| RCCL all-to-all, tuned (deduplicated control: one copy of a token per GPU) | 8,071 | 115.7 | 660 / 613 | 381 / 333 | 215 / 164 |
| `flydsl_v2` (one copy of a token per GPU) | 8,071 | 115.7 | 331 / 346 | 216 / 195 | 183 / 150 |

- Megatron's alltoall dispatcher sends one copy of a token for each of its 8 experts, about 4 of which sit on the other GPU. DeepEP sends a token once per GPU and makes the per-expert copies on the receiver, so it moves 4.1× fewer bytes. The deduplicated row is a control, not a dispatcher: tuned RCCL moving exactly the bytes DeepEP moves.
- Token rows / MB received: the hidden rows (14,336 B each) a GPU receives from the other GPU.
- CUs: workgroups per kernel. `flydsl_v2` runs 1024-thread workgroups, one per CU. RCCL runs 256-thread workgroups; 128 is its default at EP2 and the most it launches, and 32 / 64 are set with `NCCL_MAX_P2P_NCHANNELS`. The counts were checked with rocprofv3.
- notify + dispatch: DeepEP's notify (the receive counts exchanged on the GPU) and its host wait, then the dispatch; RCCL after the all_gather of the counts and the device-to-host copy of the splits that Megatron does before its all-to-all. This is the forward dispatch.
- With the receive counts known (a cached handle; the backward pass) `flydsl_v2` takes 124 µs at 128 CUs against tuned RCCL's 492 µs. Without the host wait (`num_worst_tokens`) it takes 146 µs at 160 CUs. It saturates at 160 CUs (within 5% of 256): 179 / 146 µs.
- V1 (`flydsl`) on the same input: 1691 / 1067 µs at 32 CUs, 410 / 226 µs at its saturation point of 224 CUs.

Distance to the link (GB/s = MB a GPU receives from the other / time, per GPU and direction with both directions busy; receive counts known):

| | µs | GB/s | of the link |
|---|---:|---:|---:|
| `flydsl_v2` 160 CUs, cached dispatch | 123 | 943 | 61% |
| `flydsl_v2` 160 CUs, combine | 146 | 795 | 52% |
| RCCL all-to-all as Megatron does it, defaults | 602 | 785 | 51% |
| RCCL all-to-all as Megatron does it, tuned | 492 | 960 | 62% |
| RCCL all-to-all, deduplicated, defaults | 195 | 593 | 38% |
| RCCL all-to-all, deduplicated, tuned | 164 | 704 | 46% |
| link: copy engines, 470 MB per direction | | 1542 | 100% |

- The link: hipMemcpy on the copy engines reaches 1542 GB/s per GPU with both directions busy (1574 GB/s one way). A copy kernel on V2's own data path (TDM through LDS into the uncached heap, without slots, counts or flags) reaches 1491 GB/s at 256 blocks and 1462 GB/s at 160.
- Where the rest goes (a fit of time = fixed + bytes / bandwidth over 256 to 32768 tokens per rank, 160 CUs): the `flydsl_v2` dispatch streams at 1.28–1.32 TB/s and the combine at 1.24 TB/s, plus a fixed 36 µs (cached dispatch) and 59 µs (combine), against 15 µs for the bare copy kernel. At 8192 tokens the fixed part is 30–40% of the time. Tuned RCCL has a fixed 58 µs (Megatron's rows) and 48 µs (deduplicated) and streams at up to 1.07–1.12 TB/s.
- The notify and its host wait add 56 µs to the dispatch (179 against 123 µs), about 33 µs of it the host wait.

**V1 tuning, and V1 / V2 at EP2 and EP4** (an earlier method, 4 × MI455X)

**Method**: `Buffer` interface, `PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1`, per rank T=8192, H=7168 bf16, random top-8 (64 experts in total at EP2, 256 at EP4); mean of 20 iterations on rank 0, in ms. Dispatch includes the host wait for the notify counts (`num_worst_tokens` is not set). The RCCL row is an evenly split `all_to_all_single` of T × top-k rows with RCCL's defaults, the same number of rows Megatron's alltoall sends. A single pass can vary by up to 20%, so every number below is the mean of 3 passes, and every comparison uses a fresh FlyDSL cache directory (§9).

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
- **Nothing has run on gfx950**: the V1 kernels compile to the same gfx950 ISA as before these changes (§1), and the gfx942 CI covers `cpp` and `flydsl`.
- **Merging with the C++ V2 change**: both add `intranode_backend` and `get_intranode_backend` to `Buffer` and read the same environment variable; when merging, unify the values as `cpp` / `cpp_v2` / `flydsl` / `flydsl_v2` and drop the check that keeps the C++ DeepEP unloaded on gfx1250.

## 11. Reproducing

- Unit tests (inside the container; `HIP_VISIBLE_DEVICES` sets the number of GPUs; with ROCm SDK `10.2.0a20261011` add `NCCL_CUMEM_ENABLE=0` to avoid the crash at exit in §7):

  ```bash
  HIP_VISIBLE_DEVICES=0,1 python tests/pytorch/deep_ep/test_intranode.py -v
  HIP_VISIBLE_DEVICES=0,1,2,3 python tests/pytorch/deep_ep/test_intranode.py -v
  ```

- The stress test, the combine ring lower-bound probe and the DSv3-shape benchmark behind §7 and §8 are not part of this repository; their parameters and results are given in §7 and §8.
- A benchmark that calls the ops the way Primus does (`turbo.ops.moe_dispatch` + `moe_combine`, forward and backward, `async_finish=True`, `allocate_on_comm_stream=True`) has not been run yet; end-to-end training at EP4 (DSv3 proxy, GPT-OSS 20B) is in the description of PR #583. §9 of Xiaoming's document has ready-made scripts and the DSv3 proxy command, as well as his DSv3 EP2 end-to-end result on MI455X (FlyDSL V1 with the default full fence 1318 ms/iter, RCCL all-to-all 1133 ms/iter, nearly identical lm loss).

## 12. To do

- End-to-end runs with real (unforced) routing and GPT-OSS 120B.
- `flydsl` under CUDA-graph capture on MI455X, the backend to use there.
- V2 with a different token count per rank (THD, dynamic CP): an all-reduce of the maximum before sizing, as Megatron's HybridEP path offers.
- A benchmark that calls the ops the way Primus does.
- EP4 numbers with the §8 method; runs on gfx950; 8 GPUs.
- Whether the VGPR spills of the V1 / V2 combine and the SGPR spills of the V2 dispatch are worth addressing.
- Merge with the HIP V1 / C++ V2 change (backend values, the C++ check on gfx1250, the `cpp` chunk defaults on gfx1250, a token size hint for `cpp_v2`).
