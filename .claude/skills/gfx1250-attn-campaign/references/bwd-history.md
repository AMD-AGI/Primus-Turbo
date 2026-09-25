# FlyDSL attention backward on gfx1250 -- history and state (r0-r23)

Read this before touching the backward. Everything here was re-verified against files on disk
on 2026-09-25. Marks: ✅ verified on disk / ⚠ conditional or stale-prone / ❌ dead (do not rebuild).

Path shorthands used below:

| alias | path |
|---|---|
| `$JOB` | `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934` |
| `$JC` | `$JOB/job_context` |
| `$OUT` | `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output` |
| `$HINT` | `$OUT/0923__flydsl/hint.md` (4244 lines; **h54 at line 4060 is the one-page index -- read it first**) |

## 0. The operator in one table

| item | value | evidence |
|---|---|---|
| op | FA backward, bf16, causal bottom-right, BSHD, GQA 4, d=128 | `$JC/op/ut/common.py:11-22` ✅ |
| scored shapes | fast (1,1024,1024,8,2) / proxy (1,4096,4096,32,8) / **prod (4,8192,8192,32,8)** -- prod is the only ranking shape | `common.py`, h7 ✅ |
| FLOP (prod) | bwd 5.498229e12, fwd 2.199292e12 | `$OUT/0925__flydsl/PARITY-STRATEGY.md` §3 ✅ |
| bar (aiter ASM bwd) | **~7.68 ms / 716 TF/s** (median 7.6766 ms over n=224 raw / 7.6769 over n=146 dedup, sd 0.47%; the earlier "n=177, 7.6721" was retracted at `$HINT`:3052). **Never cite 10.160 ms / 541 TF/s** -- a retracted 09-15 shim | h32, `$OUT/0924__flydsl/DAY-SUMMARY.md` ✅ |
| champion | **round 20**, prod **511.42 TF/s** (516.87 re-measured in r22) = **0.72x** bar | `$JC/state.yaml best_round: 20`; `$JOB/rounds/020/1-opt/act.yaml` ✅ |
| same-session prod floor | **0.24-0.66%** (this operator; r23 0.24, h56). Do not use the corpus's 1.57% -- that was HipKittens' GEMM ladder | h38-CORRECTION, DAY-SUMMARY ✅ |
| job settings | `max_rounds: 40`, `min_gain: 0.0` (let r20 promote at gain 1.0016 -- raise it above the floor), `fast_rounds: 24`, `fast_per_deep: 4` | `$JC/gfx1250-flydsl-attn-bwd_final.yaml:222-232` ✅ |

## 1. Code locations

| what | path | note |
|---|---|---|
| champion source | `$JC/op/current/{kernels.py (1170 l), impl.py (233 l), _env.py}` | byte-identical to `$JOB/rounds/020/op` (only `__pycache__` differs) ✅ |
| flydsl pin | `op/current/_env.py:18,30-33` prepends `/home/lihuzhan/.local/flydsl032` and **asserts 0.3.2** | ✅. A 0.3.4.1 copy (`/home/lihuzhan/.local/flydsl0341`) exists; `$OUT/0925__flydsl/bwd341/{op032,op0341,op_clean}` appeared 2026-09-25 11:36 with `kernels.py` identical to the champion and `_env.py` re-pinned to 0.3.4.1 -- **status unknown, no result recorded** ⚠ |
| kernels | `k_delta_bshd` :111, `_dkdv_impl` :166 (`_body` :346, `qloop_mask` :505, `qloop_full` :516), `k_dkdv`/`_sp` :603/:612, `_dq_impl` :661, `k_dq`/`_sp` :998/:1007, `k_redsp` :1069, `k_redsp_q` :1130 | all `known_block_size=[32,1,1]` = **one wave32 per WG** ✅ |
| tile constants | `BLOCK_KV = 32` (:53, k_dkdv), `BLOCK_Q = 64` (:647, k_dq), padded LDS rows `S_ROW_B` 80 B / `X_ROW_B` 272 B (:54-59) | 256 B stride = 64-way bank collision on gfx1250 ✅ |
| split-K rules | `impl.py:166-168` dkdv `while wgs*nsp < 2048 and nsp < 16`; `impl.py:98 _NSP_Q_CAP = 8`, `:209-211` nsp_q | fast takes split kernels; proxy/prod take nsp_q=1 (h33) ✅ |
| harness | `$JC/op/{validation.py, benchmark.py, drive.py, refcache/, ut/}` | `validation.py:199 DQ_STABILITY_DB = 70.0` ✅ |
| ledger | `$JC/{state.yaml, progress.md, note.md}`, `$JC/findings/{facts,dead_ends,pool,route}.md` | `findings/` is also mirrored in `$OUT/0924__flydsl/{facts,dead_ends}.md` ✅ |
| per-round | `$JOB/rounds/NNN/{1-opt/act.yaml, op/, _scratch/}`; deep r17 uses `1-profiling..4-reflect` | `act.yaml` `arms:` has every arm's prod TF/s ✅ |
| summaries | `$OUT/0922_summary/FLYDSL-ATTENTION.md`, `$OUT/0923__flydsl/CAMPAIGN-FINAL.md` (r0-r12), `$OUT/0924__flydsl/DAY-SUMMARY.md` (r13-r22), `$OUT/0925__flydsl/{PARITY-STRATEGY,AITER-5GEMM-STUDY,facts}.md` | CAMPAIGN-FINAL's "0.693x of aiter" uses the pre-h32 bar ⚠ |

## 2. Round history

`gain` = the verdict number in `$JC/state.yaml` (this round's code vs best-ever code, same session).
prod TF/s is the round's own sweep; **never compare prod down the column** (cross-session drift ~1%).

### r0-r12 (the 8.72x free-win phase; `$OUT/0923__flydsl/CAMPAIGN-FINAL.md`, `state.yaml`)

| r | gain | prod TF/s | verdict | what (fact id) |
|--:|--:|--:|---|---|
| 0 | -- | 57.2 | base | from-scratch FlyDSL build (`op/current/PROVENANCE.md`: no recipe matched d=128/GQA4/gfx1250) |
| 1 | 1.342 | 92.6 | ✅ | causal tile skipping (g01), longest-first dispatch (g03), BLOCK_KV 16->32 (g06) |
| 2 | 1.921 | 143.2 | ✅ | address clamps, dedupe Q/dO loads (g09) |
| 3 | 1.297 | 207.5 | ✅ hand-closed | k_dq BLOCK_Q 16->32->64 (g10,g13), bf16 stores (g11) -- wedge round |
| 4 | 1.512 | 331.0 | ✅ | LDS padding kills 64-way bank conflict (g16) |
| 5 | 1.154 | 335.4 | ✅ | host launcher cache (g17); prod ~flat |
| 6 | 1.049 | 354.3 | ✅ | mask-free/masked loop split (g19), GQA loop transpose (g20) |
| 7 | 1.093 | 382.7 | ✅ | k_dkdv Q/dO prefetch 1 iter ahead (**g21, worth ~+21% today** per g60 -17.27%, `kernels.py:522-527`) |
| 8 | 1.030 | 429.4 | ✅ hand-closed | delete single-wave barriers' conservative drain (g23/g24) -- wedge round |
| 9 | 1.119 | 481.0 | ✅ hand-closed | carried-state merge (g25) + k_dq K/V prefetch (g26) |
| 10 | 0.959 | 430.6 | ❌ | instruction-level scheduling (g27/g28/g30-g32 -> `dead_ends.md`) |
| 11 | 0.929 | 420.2 | ❌ | `s_set_vgpr_msb` bank-tax family (g33/g35/g36/g37) -- 4th wedge |
| 12 | 1.091 | 499.1 | ✅ | k_dq longest-first (**g40** is the win; g39 LDS-segment split is a null) |

### r13-r23 (efficiency phase; `$JOB/rounds/0NN/1-opt/act.yaml`, `$OUT/0924__flydsl/DAY-SUMMARY.md`)

| r | gain | prod TF/s | verdict | what; decisive numbers |
|--:|--:|--:|---|---|
| 13 | 1.370 | 497.4 | ✅ | **g42** deterministic q-axis split-K for k_dkdv (fp32 partial + fixed-order sum; fast-driven). g43 XCD remap null ❌ |
| 14 | 1.111 | 499.1 | ✅ | g44 nsp rule (smallest 2^k with wgs*nsp >= 2048, cap 16) + g45 `k_redsp` fixed-order reduce kernel |
| 15 | 1.004 | 475.1 | ❌ failed | g46 hand graded waits (dropped), g47 `sched_group_barrier` 0.969 |
| 16 | 1.005 | 502.0 | ✅ | g51 LSE/delta loads into g21's prefetch + one `sched_barrier(0)` |
| 17 | 1.159 | 507.8 | ✅ deep | g52 nsp_q split-K for k_dq, only where its grid underfills (fast) |
| 18 | 0.931 | 469.6 | ❌ | g56 LSE/delta via `global_load_async_to_lds_b32` -6.84%; g54 fused-dQ workspace, g55 occupancy priced dead on paper |
| 19 | 0.994 | 505.3 | ❌ | g59 WMMA A-reuse +0.08% (null); g60 delete g21 prefetch **-17.27%** |
| 20 | 1.0016 | **511.42** | ✅ **champion** | g62 k_dkdv Q/dO prefetch depth 1->2 (+1.65%, VGPR 724->912) merged with g61 chain split (null +0.27%); merge +2.00% vs same-sweep champion 501.41, VGPR 904. Probes P1 -6.98%, P2 -18.37%, P3 -8.19% |
| 21 | 0.872 | 412.1 | ❌ failed | g63 k_dq prefetch 1->2 **-19.33%**; g66 spread k_dq loads **-21.93%**; g65 ISA-identical, closed offline |
| 22 | 0.788 | 361.7 | ❌ failed | g68 spread k_dkdv loads **-30.0%** (pre-registered 20-30%); g67 fence form 2 0.892 |
| 23 | -- | 502.4 | ⚠ unsettled | g71 removes g62's 2nd prefetch level: **-1.74% at 7x the 0.24% floor** -> g62 is a real mechanism (h56). `rounds/023/1-opt` is partial; `resume` moves it aside |

Round-ledger traps: r20 wrote shape keys `prod_b4_s8192_hq32_hkv8_d128`, forking `champions`
(`prod: 19` is stale, the long key `: 20` is live) -- **write `fast/proxy/prod` only** (h35 ✅ `state.yaml`).
r23 left a scoring session running and contaminated validation (h56) -- **a round leaves the card idle**.

## 3. The gap: 7 GEMMs vs 5 (h40, re-stated in h54 §1)

```
gap (1.40 champion / 1.385 r22) = 1.386 structure x 1.00-1.01 everything else
```

| | value | evidence |
|---|--:|---|
| our WMMA per (b,hq) | 32,896 x 64 + 16,512 x 96 = **3,690,496** (1.4076x algorithmic) | h40, counted from our ISA ✅ |
| aiter WMMA per (b,hq) | 8,320 x 320 = **2,662,400** (1.0155x) | h40; `/tmp/aiter_dis/...a32_pssk.s` (tmp, may be gone) ⚠ |
| issued-matrix rate, ours / aiter | 719.9 (727.6 @ r22) / **727.1 TF/s** | h40 ✅ -- **our scheduling already equals hand ASM** |
| the 2 extra GEMMs | S and dP **recomputed in `k_dq`**, a separate kernel because dQ cannot be accumulated in one pass without atomics | h28, h40 ✅ |
| aiter's way | KV-outer, 4 waves, BLOCK_KV=128 (ts_qo 32 / ts_kv 128, WG takes a near+far kv pair), dQ via **514 `buffer_atomic_add_f32 scope:SCOPE_DEV`**, 26 `s_barrier_signal`, 40 TDM ops, **0 `buffer_load`**, 1024 VGPR, LDS 327680 | `AITER-5GEMM-STUDY.md:11,31`; `PARITY-STRATEGY.md` §0(1)(2) ✅ |
| why aiter has 0 `buffer_load` | its S/P GEMM takes B **from LDS**; ours from registers, so our staging is dual-use and TDM removes nothing | h49 ✅ |

**Consequence:** any efficiency arm has a ceiling of **~1%**. On a bitwise contract the ceiling
is ~517 TF/s = 0.72x (h40). Parity needs the 5-GEMM structure, i.e. dQ fused into k_dkdv.

Deterministic fusion is ❌ in every geometry (h40 table, `PARITY-STRATEGY.md` R0; reduction priced at
the only measured streaming-reduce rate 3.02 TB/s): KV-outer BLOCK_KV=128/512, Q-outer 256/512 all
fail bandwidth or registers (BLOCK_Q=512 needs 768 acc + ~648 non-acc = 1416 > 1024 VGPR).
Also ❌: bf16 dS materialisation (17.18 GB = 5.7 ms vs 3.07 ms saved), 8-wave WGs (512 VGPR cap),
single-wave fusion (h30: 1032 > 1024), "accumulators resident in LDS" (WMMA C/D are VGPR).

## 4. Closed axes -- do not build these again (h54 §2, `PARITY-STRATEGY.md` §4)

| axis | kill | status |
|---|---|---|
| compute / matrix ILP | g61 +0.27% inside 0.29% floor | ❌ |
| issue roof | g60 deleted 115 issue slots, **-17.27%** | ❌ |
| LDS port pressure | P3 deleted all 80 LDS ops, WMMA same, **-8.19%** | ❌ |
| Q/dO LDS round trip | P1 **-6.98%** | ❌ |
| HBM/bandwidth reading | g68 pure reorder -30%, g71 -1.74% with zero byte change; h8 retracted in r9 | ❌ |
| occupancy / BLOCK_KV single-wave | g55 census; g14 982 VGPR spill 0 ran 2.76x slower | ❌ |
| prefetch depth & position | g62 +1.65% (the only win), g63 -19.33%, g66 -21.93%, g68 -30.0%, P70 -10.86% | ❌ except g62 ✅ |
| `sched_barrier` / `sched_group_barrier` | 0 wins, 5 losses; `sched_barrier(0)` is a **boundary, not a clamp** | ❌ |
| WMMA operand reuse | g59 56/128 `matrix_a_reuse`, +0.08% | ❌ |
| XCD locality | g43 null (one device-wide L2) | ❌ |
| source reordering | g22, g65 ISA-identical | ❌ |
| static ISA metrics as ranking | wrong 4 times (g47, g49, g56, 5-queue model) | ❌ |
| `waves_per_eu` / `maxnreg` | corpus -32% / -5x | ❌ (h17) |
| cluster multicast / TDM fill | g18; h49 dual-use staging | ❌ on this kernel shape |
| power/clock | real (prod window 998-1029 MHz vs 1100 idle) but hits both A/B arms equally | ⚠ not a gap term (DAY-SUMMARY retraction) |

The one positive model: **load-to-use cover to the next full `s_wait_loadcnt 0x0`** explained every
1-wave reorder (DAY-SUMMARY) -- but it was falsified in the 4-wave body (S1, h50). ⚠

## 5. The 4-wave BLOCK_KV=128 barrier mystery (h44-h53)

Motivation: the 5-GEMM design needs dS to cross waves -> needs a real multi-wave WG with barriers.
All artifacts under `$OUT/0925__flydsl/`.

| step | hint | result | artifact |
|---|---|---|---|
| G0 atomic scope | h41->h42 | `BufferAtomicAdd` emits **SCOPE_CU** (silent lost updates across 8 XCDs); `UniversalAtomicAdd(Float32, SyncScope.Agent)` on a plain global ptr gives `global_atomic_add_f32 ... scope:SCOPE_DEV` ✅ | `g0-scope/`, `gate-change/atomic_probe*.py` |
| G1a barriers @1 wave | h43 | restoring both `fx.barrier()` is free (VGPR 904, body -5 instr); LLVM deletes `s_barrier` at 1 wave | `g1a-barriers/` |
| G1b 4-wave skeleton | h44 | built, **UT 15/15**, SQNR identical (52.56/52.60/52.71); VGPR 882, LDS 82944, `s_barrier` 8 | `g1b-4wave/4wave.patch`, `screen4w.py` |
| G2 measure | h45 | prod **337.02 vs 507.34 = 0.664x** | `g1b-4wave/meas` |
| diag 1+2 | h45->h46 | "replicated staging 4x drain" and "1 WG/CU no cover" both refuted by per-body census (40 st / 32 ld same; 4 waves/CU both) ❌ | -- |
| TDM | h47-h49 | lowers, retires on TENSORcnt, can pad 272 B -- but **not a candidate** (dual-use staging) ❌ | `tdm-screen/`, `staging-split-verdict/VERDICT.md` |
| S1 diag 3 | h49->h50 | move prefetch after barrier-2: cover 389->529 (+36%), prod **0.995x** -- cover model refuted here ❌ | `s1-cover/` |
| barrier-free probe | h51 | wrong-by-construction but same work: **495.56 vs 342.10 (1.449x)**; champion 521.86 -> **ceiling 0.950x even with free barriers** | `barrier-price/` |
| S2 diag 4 | h52->h53 | split `s_barrier_signal`/`s_barrier_wait` (from `flydsl._mlir.dialects.rocdl`), gap 1->233 (aiter median 18): prod **332.74 vs 337.56 (0.986)** ❌ | `s2-barrier-gap/` |

**State:** the barriers are the whole ~45% (1.449x), four mechanisms refuted, cause **unknown**.
h53/h54 rule: **no fifth armchair mechanism** -- only an ISA census or a pre-registered experiment.
Even solved, the 4-wave build without fusion tops out at 0.950x of the champion (h51).

## 6. Contract change 2026-09-25 (h41, h54 §5) ✅

| tensor | gate |
|---|---|
| `dk`, `dv` | **200-run bitwise** (unchanged; aiter's dk/dv are bitwise too) |
| `dq` | fp32 atomics allowed; **run-to-run SQNR >= 70 dB** (`validation.py:199`). Calibrated from aiter: 113.0 dB fast / 98.0 dB prod over 6 runs (`gate-change/atomic_spread.py`) |
| all | correctness >= 50 dB vs `op/eager` (champion: 52.5-52.8 dB) |

Spec text updated in place (`_final.yaml:135-142`), `spec_version` stays v000 so refcache survives.
Champion passes unchanged (`gate-change/champion-passes-new-gate.txt`). Relaxation is **latent** --
nothing uses atomics yet. Fixed-order split-K was always legal (shipped since r13).
⚠ The fwd spec still carries the old "split-k or atomic fails" sentence (h55) -- harmless there.

## 7. Open probes (both unmeasured -- grep of `$HINT`, `$OUT/0925__flydsl` and transcripts finds no result)

| probe | question | spec | cost/safety |
|---|---|---|---|
| **dQ-GEMM cost in k_dkdv** (PARITY R1/§5) | can the 1-wave k_dkdv body absorb **+25% WMMA** (16 extra `v_wmma` per 32x32 tile, dS already in regs `kernels.py:445-452`, K loop-invariant) for <= +3.0% time? | pre-registered: **<= +3.0%** parity feasible (703-719 TF/s); **+3.0-7.6%** 0.90-0.98x; **> +7.6%** fusion cannot reach parity, stop at 0.72x | 1 benchmark slot; no atomics/barriers/TDM -> no known wedge path; unattended-safe |
| **fp32 SCOPE_DEV atomic throughput** (AITER-5GEMM §5 G3) | does ~16.25 GiB of dQ atomics (BLOCK_KV=128) fit inside the fused kernel's budget? | micro-kernel, atomics only, BLOCK_KV in {128,64,32} = 16.25/32.25/64.25 GiB; aiter existence proof: 17.45 GB inside 7.236 ms | 1 slot; must predicate every atomic (no descriptor clamp on `global_atomic_*`, h42) |

## 8. Known unapplied defects (h33; line numbers re-checked against r20 on 2026-09-25) ✅ none applied

| # | defect | where (r20 source) | fix | risk if ignored |
|---|---|---|---|---|
| a | floor/ceil mismatch: assert only `sq % 32`, kernel uses `ceil(Sq/BLOCK_Q)` (**:706-707**, not :667-668 as DAY-SUMMARY says) while launcher passes `sq // BLOCK_Q` (`impl.py:223`) | `impl.py:115` | `assert sq % _k.BLOCK_Q == 0` -- ISA byte-identical | at `sq % 64 == 32`: 262,144 B OOB writes, dQ tile 0 never computed. Scored shapes safe |
| b | int32 overflow of dQ workspace size `nsp*B_*Sq*Hq*(D*4)` | `kernels.py:724` | compute in Int64 | nsp_q >= 8 at prod -> `num_records = 0`, all dQ writes silently dropped |
| c | unclamped k_dkdv prologue prefetch `_ldqd(_qt0 ...)` / `_ldqd(qp_start + nmaskp ...)`, plus carried **`kk = ii + 2`** (g62 made it depth 2; h33 saw `jj = ii + 1`) | `kernels.py:573-574`, `:583-584`, `:529` | clamp to `nqt2-1` (k_dq idiom `:898-899`, `:915-916`) | live OOB **read** (~261 KB at prod), held back only by k_dkdv's REAL descriptor extents |

Also: **never convert k_dkdv descriptors to the fake `1<<30` style** (it is what clamps defect c);
and **never "fix" k_dq's seven fake descriptors naively** -- `k_dq` has no `B_`, the `_dkdv_impl`
formula would clamp away 75% of dQ at prod (h33). gfx1250 V# `num_records` is in **128-byte units**.

## 9. API facts that cost builds (h54 §4) -- partly missing from the user skills

- `rocdl.s_waitcnt` **raises** on gfx1250 (split counters) -> bare `fx.barrier()` (h43).
- fp32 atomic: `fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent)` on a plain global pointer; not via `make_buffer_tensor` (MLIR assert, process abort) (h42).
- `s_barrier_signal`/`s_barrier_wait`: import from `flydsl._mlir.dialects.rocdl`; not exported by `fx.rocdl` (h52).
- TDM: use **aiter's** `tdm_ops_gfx1250` shim; pass `pad_interval`/`pad_amount` explicitly (h47, h48).
- 4-wave: `lane = fx.Int32(fx.thread_idx.x)` becomes 0..127 -> silent wrong fragments; use `fx.lane_id()` + separate `wave` (`AITER-5GEMM-STUDY.md:103`).
- `$JOB/rounds/002/_scratch/screen.py` hardcodes `block=(32,1,1)`; use `g1b-4wave/screen4w.py` for 4-wave.
- Card safety: **one benchmark process per shape** -- multi-shape-per-process was the r17/r18 wedge trigger (h31, `rounds/016/_scratch/meas2` vs `meas3`). Check `rocm-smi --showpids` before trusting a measurement (h56).

## 10. Where the existing user skills are stale

All three were last written 2026-09-17, before rounds 13-23.

| skill | stale on | use instead |
|---|---|---|
| `~/.claude/skills/flydsl-gfx1250` | §4 names only flydsl 0.3.2 (0.3.4.1 now installed at `~/.local/flydsl0341`); §6 says "no FlyDSL cache env var" (`FLYDSL_RUNTIME_CACHE_DIR` selects the JIT cache); no gfx1250 API traps (§9 above); §5 presents TDM as generally good (not for this bwd, h49) | §9 here, h54 §4 |
| `~/.claude/skills/gfx1250-card-safety` | no "one shape per process" rule (h31), no RW=0x1/0x5 vs RW=0x0/0x3 fault split (h33, DAY-SUMMARY "The wedge") | h31, `$OUT/0924__flydsl/wedge-rootcause/` |
| `~/.claude/skills/gpu-kernel-campaign` | general discipline still valid; lacks "instrument must assert its subject ran" (pw.sh) and "clean-run count needs a base rate" (h54 §3 traps 3-4) | h54 §3 |

## 11. Recommended next step

Priority is set by h54 §6 / h40: **the forward is worth ~0.83 ms/step, the backward residual ~0.05**.
The forward job (`$OUT/0923__flydsl/fwd-job/`, h55) runs first and **cannot share the card** with
the backward job. When the backward resumes:

1. **Zero card time, first commit:** apply h33 defects a, b, c (section 8). a and c are ISA-neutral
   for live values; b changes only the byte-count path -- confirm with COMPILE_ONLY ISA diff.
2. **Fix the ledger before resuming:** raise `min_gain` above the 0.40-0.66% floor (0.0 promoted r20
   on 1.0016); let `resume` move `rounds/023/1-opt` aside; keep shape keys `fast/proxy/prod`.
3. **Spend the first card slot on the dQ-GEMM cost probe** (section 7), pre-registered thresholds
   committed before measuring (PARITY rule 6). It decides whether fusion can reach parity at all.
4. Only if it lands <= +7.6%: the fp32 SCOPE_DEV atomic throughput micro-kernel (section 7).
5. Only if both pass: attack the 4-wave 45% with an **ISA/instrument census**, not a fifth mechanism
   (h53). If it stays unexplained, report the backward at **0.72x as the contract-bound ceiling**.
6. Do **not** spend slots on any section-4 axis; any efficiency arm is capped at ~1%.

Do not launch any GPU work while another job or benchmark holds the single gfx1250 card.
