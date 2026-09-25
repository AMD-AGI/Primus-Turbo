# Kyle's campaign learnings, filtered for the gfx1250 attention campaign

Kyle (a colleague) ran FlyDSL auto-optimization campaigns that shipped: attention fwd/bwd on
**gfx950** (hd64 dense, gpt-oss D64/D128, dsv4 sparse-MLA), grouped/dense GEMM on gfx950, and
**GEMM only** on gfx1250. None of his attention numbers were measured on gfx1250. This file keeps
what transfers, what our own gfx1250 data already refuted, and where each claim comes from.

Status legend: ✅ verified (measured on gfx1250, or checked in source today) · ⚠ conditional /
gfx950-only, re-verify before spending a round · ❌ dead on gfx1250 (measured).

## 0. Source shorthand (every claim below cites one of these)

| Tag | Where | How to read it |
|---|---|---|
| `KO:<path>` | `/home/lihuzhan/code/2026_0911__kyle_skill/myskill`, branch `origin/optimizer`, dir `optimizer/` | `git show origin/optimizer:optimizer/<path>` (Chinese). Start with `00-decision-index.md` (lever -> verdict, grep it) |
| `KC:<path>` | same repo, branch `origin/conductor_455` | `git show origin/conductor_455:gfx1250-gemm/SKILL.md` (his gfx1250 GEMM manual) |
| `CE:<file>` | `/home/lihuzhan/code/2026_0903__kyle_flydsl/campaign_example/` | Real harness (`flydsl_campaign.py` 2356 L, `cursor_campaign.py` 2560 L), launcher, goal, bench, `campaign-howto-myskill.md` (English walkthrough) |
| `H hNN` | `output/0923__flydsl/hint.md` (our corpus; **h54 at line 4060 is the index**) | repo-relative to `Primus-Turbo/` |
| `BJ:` | `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/` | bwd job; `findings/facts.md`, `findings/dead_ends.md` |
| `FWD:` | `output/0925__flydsl/fwd341/op0341/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` | aiter's gfx1250 FlyDSL fwd = our fwd baseline (same file as `0923__flydsl/fwd-job/op_baseline/`, line numbers differ by ~30) |
| Kyle code | Primus-Turbo `origin/dev/kyle/flydsl-attn-bwd-nd` (`_bench_campaign_{fwd64,d64,nd,...}.py`, `primus_turbo/flydsl/utils/attn_helper.py`) | gfx950 wave64/MFMA - read for structure, never port (flydsl-gfx1250 skill §2) |

## 1. Harness design (CE, `KO:methodology/16-campaign-harness.md`)

### 1.1 Loop shape
| Element | Kyle's implementation | Evidence |
|---|---|---|
| Phases | **ANALYZE** (collect data, read KB, rewrite `goal.md`; no kernel edits survive) -> **OPTIMIZE** (exactly one change per round) -> **REVIEW** (code-review pass, squash all kept commits into one, human author) | CE:campaign-howto §0; `flydsl_campaign.py` defaults `--analyze-min 3 --analyze-max 5 --review-rounds 2 --rounds 30` (:2168-2171) |
| REPLAN | Auto-inserted when `--stall-window` 5 rounds give < `--stall-threshold` 2% cumulative | `flydsl_campaign.py` :2211, default 0.02; REPLAN_PROMPT :883 |
| Manager/supervisor agent | Runs every round, emits `PHASE_ACTION` in {CONTINUE_ANALYZE, START_OPTIMIZE, CONTINUE_OPTIMIZE, REPLAN, GO_REVIEW} + next directive. **Hard-forbidden to give negative/ceiling verdicts** (`NO_CEILING_RULE`, :729; banned substrings :75) | `flydsl_campaign.py` :956-975 |
| Human channel | `say.sh "..."` appends to `inbox.md`, consumed atomically by the manager next round | :1024, :1394-1411 |
| In-turn bench | Agent gets `bash bench.sh` = **the same ruler the orchestrator scores with**, plus `remote.sh` for ISA dumps / rocprofv3 | KO:16 §3 |
| Keep rule | `--repeat 3`, **median** (help text / log label "best"/"MAX" are stale; impl is median). Gain >= `--keep-threshold` 0.005 -> KEPT (commit) | :2168-2215; KO:16 §1, §6.10 |
| KEPT_WIP | Correct but below threshold -> working copy **kept** so a rewrite can go down-then-up; after `--revert-patience` (default 5) rounds without new best -> revert to best | :1705-1723 |
| Crash handling | `--max-round-crashes 3`; crashed round saves `rounds/round-NN/crash_worktree.patch` (+ truncated `diff.txt`) then reverts | :1209-1221; KO:16 §6.18 |

### 1.2 Failure modes he paid for (all ✅ in his runs; transfer to any loop incl. op-evolve)
- **Sub-threshold wins evaporate.** Seven rounds each +0.12..+0.42% never crossed 0.5%, then
  `revert_worktree` wiped 73+/55- lines at once (KO:16 §6.20). A crash reverted to the last
  *commit* and silently dropped two KEPT_WIP rounds (+0.23/+0.30%) while memory.md still listed
  them (KO:16 §6.12). Rule: at streak ~5/8 steer the agent to consolidate; commit real A/B wins by
  hand; at round start grep the target file for the previous round's marker.
- **Round died in bench, not agent = never scored, not refuted.** Re-apply from
  `crash_worktree.patch` (not `diff.txt`, which truncates) and rescore; his r14 rerun was the
  campaign's biggest single win, +1.04% (KO:16 §6.18).
- **Infra down looks like "nothing left"**: 3 crashes with rsync/ssh exit 255 -> `campaign done
  +0.00% kept 0` (KO:16 §6.16). Agent-API timeouts logged as "no change produced", burned 10 rounds
  (KO:16 §6.14). Count `CRASHED` / `failed` lines, not best_score.
- **Liveness**: CPU/GPU idle is normal. The only in-round signal is the edited kernel file's mtime;
  30 min of silence on all three signals before calling it hung (KO:16 §4).
- **Orphans**: killing the orchestrator leaves agent children (PPID 1) editing the tree for hours
  (one ran ~7 h). `pgrep -f` matches your own shell. Print PIDs, kill by explicit PID
  (KO:16 §6.1-6.2). Same lesson in our `gpu-kernel-campaign` skill §6 and H h56.
- **Budget flags are per round**, not per campaign; bound total spend with `--rounds`. Measured
  cost (opus xhigh): optimize round $13-20, replan $9.4, manager $0.3-0.5 (KO:16 §5b).
  `effort=max` was worse than `xhigh` (3 rounds, net 0; one hit the 2 h timeout) (KO:16 §6.4).
- **Squash picks the wrong base / sweeps in probes**: one squash deleted `attn_helper.py`
  (-2691 lines); a KEPT commit carried 936 lines of `_an_*` probes. `git diff --stat <base>..<head>`
  must show only target files (KO:16 §6.6, §6.17). The harness deletes untracked repo files ->
  keep probes outside the repo (CE:campaign-howto §5).
- **Machine change needs base/best re-measured**, even for in-process ratios: absolute 3.5%
  (1085.8 vs 1123.9), ratio +0.45% (0.95908 vs 0.96341) - enough to cross a 0.5% keep threshold
  (KO:16 §6.5).

### 1.3 Mapping onto our op-evolve jobs
| Kyle knob | op-evolve analogue | Note |
|---|---|---|
| `--keep-threshold` 0.005 | `evolve.min_gain` | fwd spec has **0.005** (`0923__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml:609`) while `NOTES.md:152,182` argues **0.015**; current plan is 0.01. Not tunable via `op-evolve tune` (NOTES:94-96) ✅ |
| KEPT_WIP / revert patience | none known | ⚠ If op-evolve discards below-`min_gain` positives, record them in `pool.md`/hint so a later round can stack them - Kyle lost the most to exactly this |
| supervisor "no ceiling" rule | deep-round planner + hint.md | ⚠ Tension: h40/h54 legitimately close axes by measurement. Keep **evidence-backed dead-end lists**; forbid unmeasured "at the ceiling" verdicts |
| ANALYZE dumps the reference | stage-2 scout | ✅ Kyle: "dump the reference kernel's ISA in round 1" was worth ~15 rounds (CE:campaign-howto §4). Our equivalent: `0923__flydsl/STAGE2-FWD-SCOUT.md`, `0925__flydsl/fwd-isa/`, `0925__flydsl/AITER-5GEMM-STUDY.md` |
| goal dead-end list | `dead_ends.md` + h54 §2 | Must include the **base branch's** history, not just this kernel's (KO:16 §2b). For fwd, copy h54 §2 rows that are not bwd-specific |

## 2. Ruler calibration and noise

| Lesson | Kyle evidence | gfx1250 status |
|---|---|---|
| Score a **ratio vs the reference timed in the same process, interleaved** -> DVFS-immune | CE:`_bench_gptoss_down_padk_native.py:25` (`ratio = t_padN/t_padK`), KO:16 §3.6 | ✅ Our fwd A/B already does this: `output/0925__flydsl/fwd341/ab_prod.log` fly/asm TF ratio 0.657-0.666 at sclk ~1040-1049 MHz, flydsl 0.3.2 vs 0.3.4.1 indistinguishable |
| Freeze the denominator; forbid edits to reference + shared helpers; verify reference absolute time stays in its baseline band at acceptance | KO:16 §3.6 (shared `ceildiv` helper edit drifted the baseline mid-campaign) | ✅ applies: the ASM bar is a prebuilt `.co`, but shared bench/util code is editable |
| Warm-up, never `sleep`; median; warm-cache baseline (first cold shape measured 1.8x slow) | KO:`pitfalls/02-measurement-noise.md` l.10-15; CE:campaign-howto §3.3 | ✅ same discipline in `gpu-kernel-campaign` skill |
| Effects < 0.5% need >= 11 paired trials in >= 2 independent sessions; 8/9 wins at +0.49% later flipped sign | KO:pitfalls/02:194 | ✅ matches ours: same-session floor **0.40-0.66%** (H h54 §3); never borrow another kernel's floor - the 1.57% GEMM-ladder floor misapplied to attention cost a round (H h38-CORRECTION, h54 trap 2) |
| A 5-iteration warm-up already sits at the power-wall steady state (COLD vs HOT min-of-40 differ 0.63%) | KO:00-decision-index:55 | ⚠ gfx950 1400 W wall. Our card is VR-capped ~1100 MHz and dips to 967 (flydsl-gfx1250 skill §8) - always log a clock witness |
| Don't discount cycle levers by a fixed "realization rate" (32% was a cross-kernel artifact; controlled pair gave 91%) | KO:00-decision-index:32 | ⚠ principle transfers; number does not |
| Log precision: harness prints `round(score,1)` -> a ratio stuck near 0.97 reads "1.0" every round; x100 the ratio | KO:16 §6.22 | ✅ fwd ratio ~0.66 would print "0.7" all campaign - emit a pct key |
| Probe that reports success may not have run | - | ✅ ours (H h54 trap 3, `pw.sh` -> /dev/null) |
| Subtractive probes are **upper bounds**; deduct skipped work (0.8% phantom gain from skipping remainder bodies) | KO:00-decision-index:12, 244 | ✅ applies; our g61 "+8.7%" corpus number -> NULL (H h54 trap 2) |

## 3. Bench contract (KO:16 §3, CE:campaign-howto §3, `_bench_gptoss_down_padk_native.py`)

1. Last stdout line = JSON with `ok` + score key; `ok=false` never kept (bench :22, :366-387).
2. **Gates live in the bench**: SNR >= threshold **and** bitwise determinism across **multiple
   cold processes**. SNR alone shipped a race: 127 of 47M elements wrong on 25% of cold runs
   (KO:16 §3.2). Our fwd gate: 200-run bitwise on `o`/`lse` + SQNR 50 dB (fwd yaml `precision_sqnr_db: 50`).
   ⚠ **Known defect**: `precision_gate:` at fwd yaml:401 is empty -> loads as `None` (the text sits
   under `refcache:`). Fix before launch.
3. **Anti-cheat in the bench**, not the prompt: Kyle gates the reference arm against a frozen
   baseline (`padn_ok = t_padn <= PADN_BASELINE_MS * PADN_TOL`, bench :111-113, :339-349; goal
   `ANTI-CHEAT` "within +3%"). Goal also forbids editing the bench/scoring path.
4. Mirror the **deployed** config: builder defaults vs production `block_kv`/`waves_per_eu`
   differed by 1.3% on the same code (KO:16 §3.5).
5. Score one shape and the campaign trades other shapes silently: GQA merge +1.8% full-causal /
   **-12.2% SWA** under a 3:1 SWA shipping mix (KO:16 §3.4). Keep fast/proxy as sentinels but
   rank at prod (H h7, h24 "a candidate that loses at prod does not ship").
6. Acceptance: call **every dispatch branch's real entry** + AST use-before-assign check (a NameError
   swallowed by `except: continue` broke NT for 10 green rounds, KO:16 §7.5); run the same probe on
   base and HEAD; `test -f` paths used in "zero diff" checks (KO:16 §7.6-7.7).
7. Clear the JIT cache per candidate (harness does `rm -rf /root/.flydsl/cache`; comgr too, KO:16
   §6.30). Here: private `FLYDSL_RUNTIME_CACHE_DIR` per agent/job (fwd341 session finding).

## 4. Attention techniques: measured gain on gfx950, status on gfx1250

Scoring shape here: b4 s8192 hq32 hkv8 d128 bf16 causal, BSHD; fwd FLOP 2.199292e12. Fwd bar:
aiter ASM 1.5724 ms / 1398.67 TF/s vs FlyDSL baseline 2.4005 ms / 916.16 TF/s, gap 1.53x, no
structural term (H h55). The fwd baseline facts that decide transfer:
- 8 waves x wave32 = 256 threads, BLOCK_M 256 over the packed `(seq, q_head_in_group)` plane,
  `q_head_in_group` fastest (FWD:112, :248-272); grid `(ceil(S*G/256), Hkv, B)` = (128, 8, 4) (FWD:1786-1790).
- Allocates the **whole 320 KiB LDS** (FWD:645; `LDS=327680` in `0923__flydsl/STAGE2-FWD-SWEEP.md:134`)
  -> **1 WG/CU**, although VGPR 224-232 would allow 4 waves/SIMD (ladder `waves/SIMD = 1024/VGPR`, BJ:findings/facts.md r12.i0.g38).
- Softmax: online max + FAv4 deferred rescale (`RESCALE_THRESHOLD = 8.0`, FWD:165-166), raw `rocdl.exp2`
  with fused `fma(s, log2e, -m*log2e)` (FWD:447-552), VALU tree row-sum + `permlanex16` peer (FWD:555-574).

### 4.1 Grid / dispatch layer (Kyle: "+7% here vs ~1% in the body", KO:methodology/15 §0.5, index:209)
| Lever | gfx950 result | gfx1250 status |
|---|---|---|
| **Longest-first (LPT) dispatch** of causal q-tiles | bwd dq +2.50%, fwd +2.34% (needed a loop-carry split to stay under the VGPR cliff 162->170) (index:157, :216); tent-shaped profiles: walk out from the midpoint, -2.7% wall (methodology/15 l.46-51) | ✅ **Transferred twice on our bwd**: k_dkdv +2.60/+2.68% (r1.i3.g03), k_dq +4.91/+4.98% (~14% of k_dq), bitwise identical, 2-line change (BJ:findings/facts.md r1.i3.g03, r12.i2.g40). ⚠ **fwd: open** - `block_id x` ascends with seq (FWD:263), and x is the fastest grid axis, so the baseline dispatches **shortest-first**. Beware the "hardware already dispatches heaviest-first" argument: it was written once and measured wrong (STAGE2-FWD-SCOUT.md:265) |
| **XCD remap** (`xcd = bid % 8`; co-locate WGs sharing K/V in one XCD's L2) | dkdv +2.81%, dq +0.54%, L2 hit 86.5->94.8%; inner axis must differ per kernel (wrong choice -2.5% vs +2.4%); verify bijectivity offline (index:156) | ❌ **g43 NULL on gfx1250** (k_dq per-XCD K/V footprint cut 4x, six readings 0.9928-1.0065 in a 1.8% floor, H h21). `num_xcc = 8` per KFD. h54 states "one device-wide L2" - ⚠ that topology claim is unproven (h21 left it open); the null is what is measured |
| **Causal-aligned origin** (put tile padding on the shortest tile) | +0.96% (kv-block visits 7481->7396) (index:159) | ❌ **no-op at our shape**: 8192*4/256 = 128 tiles exactly, no padding. Relevant only for ragged S |
| **GQA sharer merge** (one CTA serves several q-heads sharing K/V) | hd64 fwd +2.04%, TCP_TCC_READ_REQ -49.5%, gain was all clock (sclk +8.7% at the power wall) (index:201); **bwd -2.7%** because merge forced 1 WG/CU (KO:pitfalls/13 l.419-437) | ✅ **already in the fwd baseline**: 4 q-heads packed per tile (FWD:248-272). Do not rebuild. Bwd: its preconditions (no drop to 1 WG/CU) must be checked first |
| Second co-resident WG | 8-wave CTA forced to 1 WG/CU: -4.3% / -9.6% / -9.9% (index:40) | ⚠ **lead for fwd**: baseline is LDS-locked at 1 WG/CU (above). Live footprint is 2x128 KiB because of `MIN_KV_BLK_BYTES = 64 KiB` floor (FWD:150) though K/V tiles are 17-18 KiB at n_block 64 (STAGE2-FWD-SCOUT). Unmeasured; card-safety rules apply |

### 4.2 Softmax / math
| Lever | gfx950 result | gfx1250 status |
|---|---|---|
| **Fixed reference max** (`_FMAX0`: drop per-tile max reduce, O rescale, `s-m`) | hd64 fwd **+13.5%** (954.5->1084), SNR 51.26 dB, gated `causal and window_left<0 and not splitk` (index:215; pitfalls/13 l.476-480); dsv4 +13% (methodology/15 l.21) | ⚠ **highest-value untried fwd lever, but a numerics risk**: Kyle's case was D=64 with Q prescale on his workload's score range. At d128, exp2 of an unreferenced score overflows fp32 above ~128 in log2 units; the baseline's deferred rescale (threshold 8.0) already removes most rescale work. Needs an input-range argument + SQNR on adversarial inputs, not just the prod refcache |
| **WMMA row-sum**: row-sum as a matrix op with a ones operand | ones-A `v_mfma_f32_16x16x32_bf16`: **+1.96%** (1137.1->1159.9), `v_add_f32` 68->0, `permlane32_swap` 2->0, vgpr 162->150 (index:231). Must be the 16x16x32 atom; 32x32x16 is net loss on paper. Moving row-sum back to VALU/dot2: -6.98..-8.05% (index:218) | ⚠ **open for fwd**: baseline sums on VALU + `permlanex16` (FWD:555-574). gfx1250 atom is `WMMA 16x16x32 bf16`; lane layout differs (wave32), so the ones-mask `(lane%4==0)&&...` must be re-derived. Row-sum MFMA was also a register/scheduling anchor on gfx950 - measure, do not infer |
| **Raw `v_exp` in log2 space** (intrinsic, not `math.exp2`) | `math.exp2` expands to 72 `v_ldexp` + 144 `v_cndmask` + 72 `v_cmp` per trip; intrinsic cut dq time 0.79% (dkdv +1.06% slower - not universal); fixing the masked diagonal branch alone +0.6% (KO:pitfalls/13 l.71-80) | ✅ **already done** in both fwd (FWD:447-448, `rocdl.exp2`) and bwd champion (`op/current/kernels.py:106-107`). Grep new code for `fx.exp2`/`math.exp2` in masked branches |
| Fold `-log2e*lse` into the GEMM1 C-init (bwd), including masked tiles | gpt-oss D64 fused bwd time -0.98%, masked loop 6338->5402 instr (index:180) | ⚠ open for bwd: champion computes `(masked - lse_q) * LOG2E` per element (`op/current/kernels.py:441, :846`) |
| Replace `v_exp` by same-count `v_mul` (dodge trans rate) | **-2.4%**; 0 exp = +7.8%. Cost is phase (trans pipe co-issues with matrix), not rate (index:54) | ❌ dead family on gfx950; do not re-try without a gfx1250 reason |
| `iglp_opt(2)` to hide the exp chain in fwd | 0..-6.5% (index:61) | ❌ gfx950 |

### 4.3 Waits, barriers, scheduling (the most gfx1250-sensitive group)
| Lever | gfx950 result | gfx1250 status |
|---|---|---|
| **Sink a full drain to the last safe point** (the buffer is only overwritten 2 barriers later) | hd64 fwd **+1.42%** (1116.5->1132.4); backend then emits incremental waits interleaved with MFMA. Needed a `not STAGGER` gate: 8-wave rel_l2 0.0027 -> 0.0112/0.0294 was the race detector SNR missed (pitfalls/13 l.485-497) | ⚠ **concrete fwd target**: `_pv_gemm` drains the whole V transpose burst with `rocdl.s_wait_dscnt(0)` before any WMMA (FWD:605); `_drain_barrier` does `tensor_wait(0)` before the barrier (FWD:1053-1060). On gfx1250 `rocdl.s_waitcnt` raises - only `s_wait_dscnt/asynccnt/tensorcnt` exist (H h43, h54 §4) |
| **Delete hand-written lgkm drains before a rendezvous** (backend models `s_barrier`) | +0.54% hot loop, +0.19..0.32% more in prologue/epilogue; safety criterion is the WAR margin (>= 2 barriers), not visibility (index:31, :42) | ⚠ same target as above. Our bwd precedent: deleting a stale full drain was round 8's +12% (`0923__flydsl/CAMPAIGN-FINAL.md` round-8 row, 382.7->429.4 TF/s) |
| **Remove the `sched_barrier(0)` pair around `s_barrier`** | +0.96% (cycles -1.85%); **both together only** - front alone ISA-identical, back alone -0.13% (index:36). Cluster-boundary fences are NOT removable (-0.21..-0.50%) (index:34) | ⚠ **exact pattern exists in fwd**: `sched_barrier(0); gpu.barrier(); sched_barrier(0)` at FWD:1058-1060. But on our bwd `sched_barrier` edits went **0 wins / 5 losses** and "`sched_barrier(0)` is a BOUNDARY, not a clamp" (H h54 §2) - price with an ISA region census first |
| Price the vm drain at a barrier (`vmcnt(0)` -> `vmcnt(N)`) | <= +0.17% (index:38) | ⚠ method transfers (counter names differ) |
| Reduce barrier count / widen barrier gap | hd64 dualwave fwd: deleting both hot-loop `s_barrier` = +0.10% (index:57); merging two WGs into one 16-wave domain -3.1% (index:67) | ❌ on our bwd: split barrier gap 1 -> 233 instructions bought nothing (S2, H h53); four barrier mechanisms refuted (h53). Split-barrier API itself works (H h52) |
| **`llvm_options` sched strategy** (`compile_hints["llvm_options"]`, scoped save/restore) | fwd `amdgpu-sched-strategy=max-memory-clause` + `enable-post-misched` **+0.6..0.7%**, main-loop `s_waitcnt` 94->66; `iterative-ilp` fastest but SNR=nan (reorders hand fences) (index:235). **bwd: -0.59% / -0.13% / max-ilp -0.85%** - hand-scheduled kernels only lose (pitfalls/13 l.410-418) | ⚠ **untried on gfx1250** (no hit in our corpus). Baseline sets `amdgpu-expert-scheduling-mode` + `waves_per_eu=2`, with `"amdgpu-sched-strategy": "coexec"` commented out "revisit after named barrier" (FWD:1823-1827). Hook exists in 0.3.4.1 (`flydsl/compiler/jit_function.py:760`). Compile-only screen first; re-run SNR/det after any strategy change |
| Instruction count as a ranking signal | Kyle: -351 instr = 0.0% while a zero-instruction strategy change = +0.85% (methodology/15 l.82) | ✅ same on gfx1250: static ISA metrics anti-correlated four times; deleting 115 issue slots = -17.27% (H h54 §2-3) |

### 4.4 LDS ring and loop structure
| Lever | gfx950 result | gfx1250 status |
|---|---|---|
| 4-deep K/V LDS ring + 2x unrolled body -> 1 barrier / 2 kv-tiles | +0.34% on one fwd (index:240); **-2.2%** on hd64 dualwave fwd where the barrier was the ping-pong (index:57) | ⚠ kernel-dependent. Fwd baseline is `N_KV_PP = 2` (FWD:147); prefetch-depth edits on our bwd were -10..-30% (H h54 §2) |
| **Remainder must be predicated, not a tail loop** | tail loop = 3rd body instance -> spill 2->35, **-19%**; fix: step 2 bodies, second wrapped in `scf_if_dispatch(j+1 < t_end, ...)` (WG-uniform) -> spill 0 (index:241) | ✅ principle holds (RA pressure scales with body instances). Keep the condition workgroup-uniform - non-uniform causal bounds hang the card (H h12 "hang trap") |
| Spill cost scale | 5 dword -0.3%, 35 dword -19% (index:242) | ❌ worse here: a spilling gfx1250 build **hangs after first launch** (BJ:findings/dead_ends.md:29-34). `spill > 0` is a kill (H h3) |

## 5. Kyle's gfx1250 GEMM facts (KC:gfx1250-gemm/SKILL.md), checked against flydsl 0.3.4.1

| Fact | Status today |
|---|---|
| Wave32; `get_warp_size()` returns 64 -> hardcode 32 | ⚠ **stale in 0.3.4.1**: `get_warp_size` matches `gfx12*` -> 32 and `wave64` feature is false (`~/.local/flydsl0341/flydsl/runtime/device.py:102-114`, `compiler/backends/rocm.py:75`). But `is_rdna_arch('gfx1250')` is still **False** (device.py:82-99), so buffer rsrc RDNA flags (bit 24, OOB_SELECT) are still not set (`expr/rocdl/universal.py:254-260`). gfx1250 V# `num_records` is in 128-byte units (H h33) |
| WMMA not MFMA: bf16 16x16x32; fp8/fp4 16x16x128 | ✅ (flydsl-gfx1250 skill §1-2) |
| LDS 320 KB (5 x 64 KB) vs gfx950 160 KB | ✅ aiter `_LDS_CAPACITY_BYTES["gfx1250"] = 320*1024`; fwd allocates all of it (FWD:645) |
| TDM: `s_wait_tensorcnt`; update only `dgroup0` in the K-loop; **`update_tensor_descriptor_2d_addr_lo` is carry-unsafe** past 4 GB -> host hangs in `amdgpu_mes_reg_write_reg_wait`, reboot | ✅ `update_tensor_descriptor_2d_addr64` exists in 0.3.4.1 (`expr/rocdl/tdm_ops.py:1060`). Our own: TDM lowers and retires on TENSORcnt (H h47); pass `pad_interval/pad_amount` or get a silent 64-way bank conflict (h48, h54 §4); ❌ TDM is **not** a candidate for the bwd (staging is dual-use, removes no `buffer_load`, H h49) |
| Split barrier: `pipeline_fence_signal` (= `tensor_wait(N)` + `s_barrier_signal -1`) / `pipeline_fence_wait` (`s_barrier_wait -1`), WMMA in between | ✅ API callable (`FlyDSL/kernels/gemm/gemm_common_gfx1250.py:54-89`; H h52). ❌ as a perf lever on our bwd (S2 null, H h53) |
| Cluster multicast needs HIP cluster-launch support, else the cluster barrier **deadlocks** | ⚠ untested here; treat as a card-safety item |
| Eng-sample sclk cap ~1100 MHz (DPM 500/1100, loaded ~1034); "GEMM perf proportional to sclk" | ✅ consistent with our card (1100 MHz VR cap, dips to 967; ab_prod.log sclk 1038-1049) |
| hipBLASLt on gfx1250 is a stub (best of 2 solutions 371.7 TF vs FlyDSL ~1000 TF bf16 8192^3) | ⚠ his June-2026 stack; do not use library defaults as a baseline |
| `wave_specialized_tdm` ~5% faster but NaN; `amdgpu-inst-prefetch-distance` unknown to some LLVMs | ⚠ his GEMM kernel only |
| Source-built LLVM with assertions ON compiles kernels ~10x slower | ⚠ only if we ever build FlyDSL from source; we use the 0.3.4.1 wheel |

## 6. Code style for anything that ships (`KO:methodology/10-code-style.md`, `KO:pitfalls/10-code-style-deploy.md`)
- Comment blocks <= 3 lines, **docstrings count** (methodology/10:21); whole-line comment density
  <= 4% of new lines (:25); English comments.
- No benchmark numbers, dB, dates, `debug/tmp` notes in source - they go in commit messages
  (pitfalls/10:14); gate with `rg -i 'debug|tmp|verified|print\(|TODO'` = 0 hits.
- Production knobs hardcoded; experimental env switches removed before merge (pitfalls/10:19).
- Probe scripts never `git add` (pitfalls/10:15); one squashed commit, human author, no push
  unless asked (CE:campaign-howto §8). Primus-Turbo uses ruff (line-length 110, methodology/10:72);
  `ruff-format` changes bytes -> re-measure after the hook rewrites (KO:16 §6.21).
- Our repo rule overrides: English in code/commits, Chinese in reports (user memory).

## 7. Transfer checklist - read before citing any Kyle number

1. **Every attention number above is gfx950 wave64/MFMA** except rows marked ✅ with a gfx1250 source.
   Our corpus has been misled five times by out-of-context numbers (H h54 trap 2; PARITY-STRATEGY.md:40).
2. **Occupancy regime decides transfer.** Kyle's bwd wins assume occ-2 with free cross-WG overlap
   (methodology/15 l.32); his fwd wins assume >= 2 WG/CU. Our fwd is 1 WG/CU (LDS), our bwd
   kernels run 1 wave/SIMD at 740/960 VGPR (BJ:findings/facts.md r12.i0.g38).
3. **Already in the fwd baseline** (do not rebuild): GQA packing, raw exp2 + fused fma, deferred
   rescale, TDM loaders, named barrier pairs.
4. **Open fwd leads, cheapest first** (none measured on gfx1250): LPT dispatch (2-line grid edit,
   ✅ on our bwd twice) · `O_VARIANT` v1 +1.90% (measured, H h55) · drain placement at FWD:605 /
   :1053-1060 · `llvm_options` strategy screen · WMMA row-sum · LDS floor -> 2 WG/CU · fixed max
   (numerics review first).
5. **Dead here regardless of gfx950**: XCD/L2 locality (g43), prefetch-depth games, barrier-gap
   widening, instruction-count ranking, forcing occupancy via `waves_per_eu` (hangs when it spills).

## 8. Where the sibling skills are stale (as of 2026-09-25)
| Skill | Stale item | Correction |
|---|---|---|
| `~/.claude/skills/flydsl-gfx1250` §3 | Wave-size trap measured on flydsl 0.2.4 | 0.3.4.1 returns warp size 32 for gfx1250; `is_rdna_arch` still False -> buffer-rsrc flags still CDNA (see §5) |
| same, §4 | "aiter needs 0.3.2" | 0.3.4.1 at `/home/lihuzhan/.local/flydsl0341` runs the fwd identically (fwd341/ab_prod.log) |
| same, §5 | Lists `s_wait_dscnt` idioms but not that `rocdl.s_waitcnt` **raises** on gfx1250 | H h43, h54 §4 |
| same, §0 | op-evolve attention corpus "all gfx950" - correct; now also Kyle's myskill | this file |
| `~/.claude/skills/gpu-kernel-campaign` | No bench-contract/anti-cheat section, no KEPT_WIP loss mode | §1.2, §3 here |
| `~/.claude/skills/gfx1250-card-safety` | Not stale for Kyle items; add: TDM `addr_lo` carry overflow and cluster barrier without HIP cluster launch both hang | KC §4.1, §0.4 |
