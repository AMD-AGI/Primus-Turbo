# FlyDSL DeepEP V2 as one intranode backend for MI455X and MI355X: design

Status: **bring-up branch, not part of PR #583.** `flydsl_v2` is tested and tuned on MI455X (gfx1250). This
design makes the same backend and the same kernel source run on MI355X (gfx950) and MI300X (gfx942), with the
code path chosen per GPU, and keeps the MI455X behavior and generated code unchanged. MI355X results and the
steps that are still open are in [`flydsl_deepep_v2_mi355.md`](flydsl_deepep_v2_mi355.md).

## Goal

- One backend name, `flydsl_v2`, and one kernel source for gfx1250, gfx950 and gfx942.
- Every per-GPU difference is decided in one place, from what the GPU is: how a row moves, the LDS
  budget, the copy piece size, the wave size, the memory ordering. No environment variable is needed
  to run V2 on any of them.
- The default backend of each GPU comes from one table, so making `flydsl_v2` the MI355X default is a
  one-line change once the data supports it.

Not in scope: internode and low-latency (always `cpp`), changes to V2's protocol, CUDA-graph capture of V2,
token counts that change per step without a known maximum (the heap holds the largest dispatch of the group).

## Layer 1: what the GPU is (`ArchTraits`)

`primus_turbo/flydsl/deep_ep/arch.py` already holds one `ArchTraits` NamedTuple per target (wave size,
cache policies, copy pieces, the combine's loads in flight, the spin watchdog). It gains:

| Field | gfx1250 | gfx950 | gfx942 | Used by |
| --- | --- | --- | --- | --- |
| `has_tdm` | True | False | False | V2's row mover |
| `max_lds_bytes` (LDS a block can allocate) | 320 KiB | 160 KiB | 64 KiB | V2's TDM row buffers; the LDS check of every V2 kernel |
| `v2_copy_piece_bytes` (register copy: bytes a warp loads before it stores them) | 8 KiB | 8 KiB | 8 KiB | V2's register copy |

gfx942 and gfx950 get their own entries (their LDS differs); every other field keeps the shared gfx9 value.
An unknown target gets the gfx942 values (no TDM, 64 KiB LDS). Being a NamedTuple keeps every field in
FlyDSL's compile cache key (`flydsl_deepep_mi455.md` §9).

`PRIMUS_TURBO_DEEPEP_ARCH_TRAITS="field=value,..."` (read at import) replaces the tuning fields of the
target's traits, `v2_copy_piece_bytes`, `combine_loads_in_flight`, `spin_timeout_cycles` and
`trap_on_timeout`, so other values can be measured without editing `arch.py`; it refuses the fields that
describe the hardware. Each value is part of the compile cache key, but give each its own
`FLYDSL_RUNTIME_CACHE_DIR` anyway when comparing.

## Layer 2: the code path inside V2 (`intranode_v2_kernel.py`)

Chosen when a kernel is built, from `TRAITS`:

| Decision | gfx1250 | gfx950 / gfx942 |
| --- | --- | --- |
| Row mover | TDM through LDS: one load of a row, one store per destination | register copy: `_fanout_row` (dispatch: each piece loaded once for every destination), `_copy_row` (receive, combine push) |
| LDS | per-warp TDM row buffers sized from `max_lds_bytes`; wide rows in pieces | only the small index arrays (EP8: about 33 KiB); checked against `max_lds_bytes` |
| Copy piece | TDM: the row buffer; register copy: `v2_copy_piece_bytes` | `v2_copy_piece_bytes` |
| Wave size | 32: 32-bit ballots, `mbcnt_lo` | 64: 64-bit ballots, `mbcnt_lo` + `mbcnt_hi` (already in `_ballot` / `_popc` / `_lanes_below`) |
| Payload stores | `SCOPE_SYS` (`cpol_sys_store`); a store is acknowledged at the first cache its scope covers | plain (`cpol_sys_store = 0`): a store to the uncached heap is acknowledged from memory |
| Before a count or flag is published | wait for the TDM stores, then `s_wait_*cnt 0` (`wait_mem_all`) | `s_waitcnt 0` (`wait_mem_all`) |

`PRIMUS_TURBO_DEEPEP_V2_TDM=0` (read at import) forces the register copy where TDM exists, for A/B
measurements and bring-up only; give each setting its own `FLYDSL_RUNTIME_CACHE_DIR`.
`flydsl_intranode.v2_code_path()` reports the path a process uses (`"tdm"` or `"register copy"`).

The per-arch values only replace constants that were already there (`IS_GFX1250`, `MAX_LDS_BYTES`,
`FANOUT_PIECE_BYTES`) with the same values, so the generated code does not change on any target; the
starting value of `v2_copy_piece_bytes` on gfx9 is the current 8 KiB and is the first knob to tune on MI355X.

## Layer 3: which backend a `Buffer` uses (`flydsl_intranode.py`, `buffer.py`)

`get_intranode_backend(backend)`: the explicit argument, else `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`, else
the default of this GPU from one table:

| Target | Default | `flydsl_v2` |
| --- | --- | --- |
| gfx1250 (MI455X) | `flydsl_v2` | supported: tested and tuned |
| gfx950 (MI355X) | `cpp` | experimental: runs, not yet validated or tuned |
| gfx942 (MI300X) | `cpp` | experimental |
| other | `cpp` (`flydsl` without the C++ DeepEP) | experimental |

- `flydsl_v2` is accepted on every target (no `PRIMUS_TURBO_DEEPEP_V2_ANY_ARCH`). On an experimental target
  the runtime warns once per process, naming the target, the code path and this document.
- An explicit choice is never replaced, except by the structural fallbacks that already exist: internode,
  low-latency, `num_ranks` outside {2, 4, 8} or `num_nvl_bytes == 0` take `cpp`.
- Features V2 does not have still fail loudly with the way out: CUDA-graph capture raises and names
  `flydsl`; `get_token_buffer_size_hint` / `moe_dispatch` size the heap for the same token count on
  every rank.
- Making `flydsl_v2` the MI355X default: set the gfx950 row to `flydsl_v2` / supported, once the gates
  below pass.

## Gates before `flydsl_v2` becomes the MI355X default

1. **wave64 and gfx9 ordering, on CI**: `test_intranode.py` with its `flydsl_v2` tests on the gfx942
   runner (wave64, no TDM, gfx9 memory model). With this branch the tests run there without any switch.
   Passed at EP8 on 8× MI300X (see the validation table).
2. **Correctness on MI355X**: `test_intranode.py` on 2 and 8 GPUs; the V1 / V2 stress test.
3. **Kernel performance against `cpp`** (`benchmark/ops/deep_ep_v2/bench_ops.py`, DeepSeek-V3 shape, EP2
   and EP8, at the CU counts training uses): `flydsl_v2` at least on par with `cpp`. Tune
   `v2_copy_piece_bytes` and `combine_loads_in_flight` (gfx950 entry of `arch.py`) first.
4. **End to end on MI355X** (DeepSeek-V3 proxy, EP8) against `cpp`, as in PR #583's MI355X section.

## Validation on this branch

| Check | Where | How | Result |
| --- | --- | --- | --- |
| gfx1250 TDM path unchanged | MI455X, compile only | all 9 kernels at EP2 / EP4, ISA against `461d9caa` | 18 / 18 byte-identical |
| gfx1250 register copy unchanged | MI455X, compile only | the same with `PRIMUS_TURBO_DEEPEP_V2_TDM=0` | 18 / 18 byte-identical |
| gfx950 / gfx942 code unchanged and within limits | MI455X, compile only | ISA against `461d9caa`; LDS of every V2 kernel checked against `max_lds_bytes` when it is built; EP8 | 36 / 36 byte-identical; EP8 compiles on all four; wave64 code (`v_mbcnt_hi`, 64-bit exec), no TDM instructions |
| Both paths run | 2× MI455X | `test_intranode.py` with TDM and with `PRIMUS_TURBO_DEEPEP_V2_TDM=0` (`NCCL_CUMEM_ENABLE=0`, #584); a short benchmark of both | 20 tests each: 18 pass, 2 skipped (`cpp`); dispatch / combine at 32 / 128 / 256 CUs: TDM 332 / 347, 182 / 151, 175 / 136 µs, register copy 758 / 604, 261 / 170, 185 / 139 µs, outputs checked. V2 stress test with the register copy (10 configurations with tiny rings, 30% dropped top-k slots, every row checked): 100 repetitions each, all pass |
| Tuning values | MI455X, compile only | gfx950 at EP2 / EP4 with `v2_copy_piece_bytes` 1 / 2 / 4 KiB and `combine_loads_in_flight` 16 / 12 / 8 | all compile; with the variable unset, 72 / 72 ISA files byte-identical to `461d9caa`. On gfx950 `combine_loads_in_flight=16` removes the combine spills (59 / 48 VGPRs at EP2 / EP4 → 0); 1 KiB pieces take the dispatch from 90 to 56 VGPRs |
| Tuning values run | 2× MI455X, register copy | the same values (and 2 KiB with 16), `test_intranode.py` + V2 stress (30 repetitions) + a short benchmark | all pass. On gfx1250 smaller pieces are slower (64 / 128 / 256 CUs dispatch 430 / 265 / 189 µs at 8 KiB, 543 / 315 / 204 µs at 1 KiB) and the loads in flight do not matter; gfx9's counters differ, so this does not settle gfx950 |
| Combine fix (`41655cf2`: push only the received rows) | MI455X, compile only | the 9 kernels at EP2 / EP4 on gfx1250 (TDM and register copy), gfx950, gfx942, against the commit before | all compile; only `combine_v2` changes, with the same VGPRs, VGPR spills, scratch and LDS everywhere (1–2 SGPRs fewer) |
| wave64 at run time | gfx942 CI, 8× MI300X | gate 1: the CI workflow on `f1eb24bf` (run 40720916625), `--dist-only` step with all 8 GPUs, so EP8 with the register copy | passed on the first attempt: every `flydsl_v2` test (round trip at 4096 tokens, dispatch layout, a rank that receives nothing, destroy and recreate, CUDA-graph refusal, `moe_dispatch` buffer sizing) and the host-only selection / traits tests; `test_moe_ops_v2_buffer_with_uneven_tokens` was still skipped off gfx1250 there (fixed in `b00e402b`) |
| wave64 at run time, with the combine fix | gfx942 CI, 8× MI300X | the same on `7effe850` (run 40741179998) | passed on the first attempt: 20 / 20 multi-GPU tests, now with `test_moe_ops_v2_buffer_with_uneven_tokens` and `test_combine_ignores_the_worst_case_tail` (`num_worst_tokens` dispatch, NaN in the rows past the received ones) for `cpp`, `flydsl` and `flydsl_v2`; host-only tests pass |
| The branch after the PR #583 merge (`f1eb24bf`) and the combine fix | 2× MI455X | `test_intranode.py` with TDM and with `PRIMUS_TURBO_DEEPEP_V2_TDM=0` | not run yet (the rows above on MI455X are from before the merge); compile-only checks of the fix above |
| MI355X | MI355X host | gates 2–4 | not run yet |

## Risks

- The wave64 branches and the gfx9 ordering of V2 have run on MI300X (gate 1, EP8, correctness tests
  only), not yet on MI355X (gate 2) or under the stress test on gfx9. Read for wave64, without finding a
  problem: `_ballot` (64-bit mask), `_popc` (64-bit count truncated),
  `_lanes_below` (`mbcnt_lo` + `mbcnt_hi`); the register copy's ragged last step; the reduce queue, whose
  claim uses `REDUCE_GRAB_TOKENS × num_ranks` lanes (at most 32; the upper lanes idle on wave64), with
  `readlane` and `lane % num_ranks`; the combine's columns, which step by `WARP_SIZE` and leave the
  lanes past a row to the buffer bounds (loads 0, stores dropped); the flags and counts, published after
  `s_waitcnt 0` on the uncached heap like V1's tails, which already run on gfx9.
- Without TDM, V2 needs many CUs to reach the link: on MI455X the register copy is 1.4× / 1.1× slower
  than TDM at 128 CUs and within 6% at 256. On MI355X `cpp` may stay ahead at the CU counts training can
  spare; then the gfx950 default stays `cpp`.
- Registers on gfx9 (compiled at 64 channels): the V2 combine spills 59 / 48 VGPRs at EP2 / EP4 (224 / 180
  bytes of scratch per lane; none at EP8), against 8 / 75 on gfx1250 with TDM, and the EP8 dispatch spills
  97 SGPRs (to VGPRs, no scratch). A smaller `combine_loads_in_flight` removes the combine spills; on
  gfx1250 it was not faster (`flydsl_deepep_mi455.md` §8), on gfx950 it has to be measured.
- EP8 has run with V2 only through the gfx942 CI (register copy, correctness tests); EP8 with TDM has not
  (MI455X nodes have 4 GPUs). The heap holds the batch: about 1.9 GB for the
  DeepSeek-V3 shape at EP8 (`Buffer.get_token_buffer_size_hint`).
