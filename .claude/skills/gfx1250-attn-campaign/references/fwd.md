# Forward: the op-evolve job, its spec, the fixes, and the decision index

Paths: `PT` = `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`, `OE` =
`/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`, `FJ` = `OE/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644`.
Background: `PT/output/0923__flydsl/hint.md` h55 (the fwd premise) and `op-evolve-ops.md` §10 (spec audit).

## 1. Where the forward lives

| what | path |
|---|---|
| spec (source of truth, committed `0febef39`) | `PT/output/0925__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml` (+ `NOTES.md`, **stale 09-23 prose**) |
| staged spec the loop was launched from | `OE/jobs/gfx1250-flydsl-attn-fwd.yaml` (copied back to the 0925 dir at launch) |
| running job | `FJ/` (created 2026-09-25 11:46:44; `job_context/state.yaml`, `job_context/gfx1250-flydsl-attn-fwd_final.yaml`) |
| loop log | `OE/LOG.fwd` (launched with `setsid nohup ... op-evolve run --config jobs/gfx1250-flydsl-attn-fwd.yaml >> LOG.fwd`; loop PID = PGID **68769** on the 2026-09-25 boot -- stop with `op-evolve stop --job gfx1250-flydsl-attn-fwd-20260925-114644`, or `kill -TERM -68769`; re-check the PID after any reboot) |
| baseline op | `PT/output/0925__flydsl/fwd-job/op_baseline/` = byte-identical to `fwd341/op0341` (vendored aiter FlyDSL fwd, `_env.py` asserts 0.3.4 + `flydsl0341`) |
| API-cleaned copy (ISA-identical, not the baseline) | `PT/output/0925__flydsl/fwd341/op_clean/` -- see `flydsl-api.md` |
| same-process A/B template vs ASM | `PT/output/0925__flydsl/fwd341/fwdab.py` (+ `ab_prod.sh`, `ab_prod.jsonl`) |
| ASM-vs-FlyDSL ISA diff | `PT/output/0925__flydsl/fwd-isa/` (no REPORT.md; hint f4 carries the table) |
| human hint (decision index below) | `PT/output/0925__flydsl/fwd-hint/hint.md` |
| watcher | `PT/output/0925__flydsl/mon/watch.sh <job>` (uses unsudo'd `dmesg` and `rocm-smi`; prefer the `op-evolve-ops.md` §8 recipe) |

## 2. Spec state and fixes (verified 2026-09-25 against `FJ/job_context/*_final.yaml`)

| item | state |
|---|---|
| `op.precision_gate` | FIXED: split out of the `refcache: >` block; parses as a string (the 0923 copy loaded as `None`) |
| schedule | `fast_rounds 4, fast_per_deep 4, max_rounds 30`, max_timeout 120h -> deep rounds 5, 10, 15, ... |
| `min_gain` | **0.007** (just above the 0.24-0.66% floor). Not a `tune` knob: hand-edit every occurrence in final.yaml, back up, plain `resume` (never `resume --config`, it bumps spec_version and re-runs ~58 min setup) |
| models | setup/profiler `claude-opus-5-5 xhigh`, planner `max`, reviewer codex `gpt-5.6-sol` |
| flydsl | `runtime.python_path` has `~/.local/flydsl0341`; header comments still say 0.3.2 (stale prose) |
| beat | "aiter prebuilt gfx1250 ASM forward, `fmha_fwd_with_sink_asm` ... must be beaten by 0%", `beat_margin_pct: 0` |
| **verify at setup review** (10-min auto-approve window) | (a) `grep -rn attn_bwd FJ/job_context/op/` empty -- the bwd template hardcodes `getattr(mod,"attn_bwd")` at `ut/common.py:104`; (b) FLOP counter called with `backward=False` (bwd `benchmark.py:72-76` has True -> 2.5x inflated TF/s); (c) whether `validation.py` reads the margin or hardcodes it like bwd; (d) `flydsl.__file__` is `flydsl0341`; (e) a deliberate-edit proof that the vendored copy is the one imported; (f) round-0 lse SQNR |
| setup status | `job_setup` done in 256 s but logged "no readable block ... round 0 will carry no throughput"; `op_setup` was running at 11:51 |
| **hint.md NOT DELIVERED** | `FJ/job_context/hint.md` does not exist. And `op_evolve/core/hints.py` parses **only** table rows whose first cell is `h<N>` and sections `## h<N>`; this file's `L#`/`f#` ids parse to **zero hints**, and no prompt names hint.md, so the rounds would never see it. Fix: prepend an `| id | type | title | status |` table with e.g. `h1 must standing "read the fwd decision index every round"`, `h2 must note "round 1: grid layer first (f1)"`, retitle key sections `## h1 -- ...`, then copy to `FJ/job_context/hint.md` (applied at each round head, no restart). Keep the PT copy git-tracked |
| deep_loop-trim patch | **committed** in `OE` as `58b2134`; its CAMPAIGN CORRECTIONS block in the 3 deep `_preamble.md` is **bwd-specific** (k_dkdv 740 VGPR, k_dq 960, ...); rewrite for fwd before round 5 (`op-evolve-ops.md` §7) |
| one card | bwd job must stay stopped while fwd runs; check `ls /sys/class/kfd/kfd/proc` before trusting numbers |

Numbers: bar ASM 1.5724 ms / 1398.67 TF/s; FlyDSL baseline 2.4005 ms / 916.16 TF/s (0.655x, gap 1.5266x);
2026-09-25 re-measure fly/asm 0.657-0.666, 0.3.2 vs 0.3.4.1 perf-neutral, agree_db 49.93. FLOP prod 2.199292e12.

## 3. Decision index (verbatim from `fwd-hint/hint.md`; `fN` sections are in that file)


Status: ❌ measured-dead · ⚠ conditional · 🔬 predicted-only (ISA reading, no card number) · ✅ open-untried

| # | lever | status | one-line reason | ptr |
|---|---|---|---|---|
| L1 | `O_VARIANT` v3 -> v1 | ⚠ | +1.90% over 101 reps, two sessions, on flydsl **0.3.2**; re-verify on 0.3.4.1 (V1 paths hit `ptr_load` breakage) | f3 |
| L2 | Grid: longest-first causal q-tile order | ✅ | FlyDSL dispatches light->heavy; KYLE +2.34% fwd (gfx950); BWD g40 (k_dq longest-first, round 12) +4.91% prod | f5 |
| L3 | Grid: in-WG pairing t with N-1-t (ASM does this) | 🔬 | equal work per WG + epilogue overlap; est. 2-5% | f5 |
| L4 | Grid: XCD-major (b,kv_head) remap | ⚠ | KYLE +2.81% (dkdv, gfx950); BWD g43 **null** on gfx1250 (one device-wide L2). Cheap; bijection check mandatory | f5 |
| L5 | Grid: causal-aligned q-tile origin | ⚠ | KYLE +0.96% (gfx950), re-verify | f5 |
| L6 | QK(j) / softmax(j-1) software pipelining, 2 S register sets | 🔬 | #1 ISA difference, est. 20-35%; never tried on fwd | f6 |
| L7 | Real LO/HI ping-pong (named barrier + `s_setprio`) | 🔬 | `_named_barrier_pair` is a no-op in the ISA today | f6 |
| L8 | Split barrier (signal early / wait late, work in gap) | ⚠ | ASM does it; BWD S2 measured **null** (gap 1->233 bought nothing, 4 barrier mechanisms refuted) -- bwd-measured, fwd may differ | f7 |
| L9 | `sched_barrier` / `sched_group_barrier` patterns | ⚠ | BWD: 0 wins 5 losses -- bwd-measured, fwd may differ; only as part of L6, never alone | f6 |
| L10 | `n_block` 64 -> 128/256 alone | ❌ | SWEEP: 3.2x / 9.9x slower, VGPR spill (2x584 > 1024). gfx1250, 0.3.2 | f8 |
| L11 | joint (BLOCK_M, n_block) retile | ❌ | SWEEP: shipped point is a local optimum (5 points). gfx1250 | f8 |
| L12 | 4 waves / 1 wave per SIMD | ⚠ | **roadblock, not a result**: V2/V3 managers raise on `num_waves != 8`. Never record "4 waves is slower" | f8 |
| L13 | TDM prefetch depth 3 (`N_KV_PP=3`, `tensor_wait(2)`) | ⚠ | ISA est. 3-10%; BWD prefetch depth/position >2 all lost (-10..-30%) -- bwd-measured, fwd may differ; needs `MIN_KV_BLK_BYTES` lowered | f9 |
| L14 | Deep KV-loop unroll (x2, ASM-style) | ✅ | never tried on fwd; ASM unrolls 2x. Tail must be predicated (spill risk) | f9 |
| L15 | Packed FP32 softmax (`v_pk_fma_f32` exp arg) | 🔬 | ISA est. 2-4% | f10 |
| L16 | `v_pk_add_f32` row-sum | ❌ | KYLE dead on gfx950 -- gfx950-measured, re-verify only paired with L17 | f10 |
| L17 | Per-lane partial row-sum, one cross-lane reduce at end | 🔬 | ISA: FlyDSL permlanes every tile, ASM only in epilogue; est. 1-3% | f10 |
| L18 | WMMA ones-row row-sum | ⚠ | KYLE +1.96% (gfx950 MFMA), re-verify | f10 |
| L19 | raw `v_exp` log2 path | ⚠ | KYLE +0.6% gfx950; FlyDSL already uses exp2 with scale folded into Q | f10 |
| L20 | Rescale: drop branch (`ENABLE_DEFER_RESCALE=False` or select) | 🔬 | branch splits basic blocks; est. 1-3%; lazy-rescale alone dead on gfx950 | f11 |
| L21 | Fixed max `_FMAX0` (no running max) | ⚠ | KYLE +13.5% fwd gfx950; **requires adversarial large-logit test**, see f16 | f11 |
| L22 | lgkmcnt/dscnt drain later; delete hand drain before barrier | ⚠ | KYLE +1.42% / +0.54% (gfx950); gfx1250 has split counters, `rocdl.s_waitcnt` raises | f12 |
| L23 | Delete `sched_barrier` pair around `s_barrier` | ⚠ | KYLE +0.96% gfx950 | f12 |
| L24 | `llvm_options` max-memory-clause + post-misched | ⚠ | KYLE +0.6-0.7% gfx950 | f12 |
| L25 | GQA sharer merge | ❌/⚠ | KYLE +2.04% was via clock; FlyDSL fwd already packs 4 q-heads per KV (its advantage over ASM) | f12 |
| L26 | 4-deep LDS ring + unroll2 | ⚠ | KYLE +0.34% / -2.2% elsewhere | f9 |
| L27 | WMMA operand reuse bits | ❌ | BWD g59 +0.08% (null) -- bwd-measured | f13 |
| L28 | Occupancy forcing / `s_sleep` phase offsets | ❌ | KYLE dead gfx950; BWD single-wave occupancy closed | f13 |
| L29 | Epilogue: TDM O-store overlapping next Q tile | 🔬 | est. <=1-2%; free with L3 | f5 |
| L30 | Static ISA metrics as ranking signal | ❌ | BWD: wrong four times. Rank by card numbers only | f13 |

Round-1 order (f1): grid layer first -- L2 longest-first, L4 XCD remap (offline bijection check is a gate),
L5 -- each its own arm; L1 `O_VARIANT` v1 if it compiles on 0.3.4.1 (V1 `ptr_load` breakage, `flydsl-api.md` §3);
body (L6 pipelining) only after grid arms are measured. Every KYLE/BWD row is a prior, not a result.
Correctness gates (f16): adversarial large-logit test for any max-tracking change; lse natural log; o/lse
bitwise deterministic (no atomics, no split-k); log agree_dB vs ASM (49.93).

## 4. Progress log (fwd job `gfx1250-flydsl-attn-fwd-20260925-114644`)

Hints live in `job_context/hint.md` as `hN` (converted from `fwd-hint/hint.md`'s `fN`; h18 = decision index, h20 = re-land r1.i1.g01).

| round | mode | outcome | fast / proxy / prod TF/s | vs champion (prod) | note |
|---|---|---|---|---|---|
| 0 | setup | baseline | 47.3 / 549.8 / 935.0 | -- | setup took ~15 min, not 58 |
| 1 | fast | rejected | 44.2 / 813.4 / 1015.4 | 1.086 | r1.i1.g01 longest-first dispatch: proxy 1.48x, prod +8.6%. Rejected only by operator-side gate bugs (below). O_VARIANT v1 did NOT reproduce (proxy -3/-4%). |
| 2 | fast | **accepted** | 44.5 / 778.5 / 1018.0 | 1.089 | r1.i1.g01 re-landed (h20). Champion copy: `output/0925__flydsl/fwd-job/champion/`. prod = 0.73x ASM. |

### Operator fixes applied during the fwd job (each cost a round -- check them on any new job)
- **op-evolve route parser** (`core/route.py`): row type was matched on the whole row, so "must be predicated" / "== idea r1..." in free-text cells flipped must/idea and crashed round 1 (`RouteError`). Fixed to read the type cell only (op-evolve branch `lhz/gfx1250`).
- **Precision floor 50 dB was unpassable**: the unmodified baseline reads o 49.82-49.99 dB on 7 edge cases. Set `op.precision_sqnr_db: 49` in the resolved yaml (hand edit, no spec bump).
- **Fast shape vetoed a prod win**: band = max(0.95, 1-min_gain) = 0.993 per shape; fast (s=1024) is launch-bound and its median moves with arm position after ASM. Added `evolve.shape_band: {fast: 0.90}` (new op-evolve feature, `core/acceptance.py`).
- **Deep preambles** had bwd-specific hints hard-coded; replaced with an instruction to read the job's own hint.md.
- **Opus 5.5 agents** need `claude-agent-sdk>=0.2.159` in the op-evolve `.venv` (0.2.152 bundles a CLI too old for `claude-opus-5-5`).
- **Stopping the loop**: find the pid with `ps -eo pid,cmd | grep "[o]p-evolve resume"` -- `pgrep -f` matches your own shell and `kill -TERM -<pgid>` then kills it.
