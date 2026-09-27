| h21 | advise note | L6 QK(i+1)/softmax(i) software pipeline -- compiled, gated, 508 VGPR / 0 spill (proto `pipeline`) | open |
| h22 | must note | L15+L17+L20 softmax arithmetic -- QK->PV serial span -34%, 0 spill (proto `softmax`) | open |
| h23 | advise note | L13+L14 TDM prefetch depth 3 and clean-loop unroll x2 -- 4 compiled arms (proto `unroll`) | open |
| h24 | advise note | L8 barrier per 2 KV tiles + split signal/wait -- p22s, 439 VGPR (proto `barriers`) | open |
| h25 | advise note | L3+L29 in-WG q-tile pairing -- O-store overlap only, dispatch gain is zero (proto `pairing`) | open |
| h26 | standing note | Closed levers from the 2026-09-27 compile-only prototypes | open |

---

Common to h21-h25 (read once):

- All five prototypes are **compile-only** (flydsl 0.3.4.1, container fa-repro, 2026-09-27). **Nothing was
  measured on the card.** Every percentage below is a prior (h13 L30, h15): rank by card numbers only.
- Root: `P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__flydsl/proto/<name>`.
  Each has `base/` (a copy of `op/current`), `op/` (the prototype) and a unified diff. Only
  `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` differs; `impl.py` is unchanged.
  On 2026-09-27, `base/flydsl_fwd` == `job_context/op/current/flydsl_fwd` for all five; the diffs match `op/`.
- Champion ISA reference: prod md5 `23e8ba8d3892`, 445 VGPR, 0 spill, LDS 327680.
- **Round start recipe (same for every entry):**
  1. `diff -r -x __pycache__ $P/base/flydsl_fwd <round op>/flydsl_fwd` must be empty. If the champion has
     moved, do not copy. Apply the diff instead: `patch -p1 -d <round op> --dry-run < $P/<diff>`, then
     without `--dry-run` (the diff paths are `base/flydsl_fwd/...`, and -p1 strips `base/`). If it does
     not apply cleanly, port it by hand, then do step 3.
  2. `cp $P/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py <round op>/flydsl_fwd/`.
  3. Set the toggle for the arm (listed per entry). **Before any card run**, compile-only the prod,
     fast, non-causal and causal gqa=1 configs and check `.vgpr_spill_count == 0` and scratch 0 (h14).
     Compare the prod md5 with the number given here; if it differs, find out why before measuring.
  4. The OFF arm (toggle False) is byte-identical to the champion ISA. Use it as the control in the
     same session (palindromic order, n >= 101, sclk recorded, h15).
- The prototypes are separate trees and **do not stack by toggling**. Stack only after one lands, by
  porting the next diff onto the new champion and re-running step 3 (VGPR budget is 512).

### h21 -- L6 QK(i+1)/softmax(i) software pipeline -- compiled, gated, 508 VGPR / 0 spill (proto `pipeline`)

- **Path:** `$P=.../proto/pipeline`; diff `pipeline.patch`; review `REPORT.md`; configuration matrix
  `review/run.sh` (`GQA`, `CAUSAL`, `D`, `DT`); LDS drain check `review/dscnt_cfg.py`.
- **What it changes:** S is double-buffered in registers. Iteration i runs QK(i+1) and softmax(i) in one
  region, so the softmax VALU/TRANS of tile i sits under QK(i+1)'s 32 WMMAs. The LDS ring is re-phased
  so V lags K by one tile: a slot holds [K(i+1) | V(i)]. It is still 2 slots with one WG barrier per
  tile. The last tile's softmax+PV is peeled after the loop.
  Toggle `PIPELINE_QK_SOFTMAX` (:192).
  A compile-time gate `_PIPE` (~:757) enables it only for `USE_TDM_LOADER and mask_right and not mask_left
  and gqa_ratio in PIPE_GQA_OK (2,4) and qk_hdim==128 and v_hdim==128`. Every other config compiles to
  the champion body byte for byte.
- **ISA evidence (prod = fast, same ISA):**
  - Resources: VGPR 508/512, SGPR 96, spill 0/0, scratch 0, LDS 327680, 6310 instructions.
  - Clean-LO loop, 694 instructions: WMMA 64, v_exp 66, v_nop 25, other VALU 173, ds_load 64, TDM 2,
    barrier 2, s_wait_dscnt 15, SALU 173, v_mov 39 (32 are S copies), s_set_vgpr_msb 103.
    Clean-HI is 751 (v_nop 54). Masked loops are 950/977.
  - v_exp within 8 instructions after a QK WMMA: 59/66 (clean LO), 54/66 (clean HI), 0/66 (masked).
  - thd causal: 508/0. causal gqa2: 508/0. fp16 causal gqa4: 511/0 (not scored). Non-causal, gqa1 and
    win_sink are byte-identical to the champion.
  - Every `s_barrier_signal` has 0 LDS ops outstanding on every CFG path (dscnt_cfg.py), with
    `s_wait_tensorcnt 0` before each barrier. LO/HI both do 1+n_iter WG barriers, same as the champion.
- **Expected effect:** prod +3 to +12%, point estimate about +6%. Round 3 put exp2 on the critical
  path (13.4%), and about 250 softmax VALU/TRANS per clean tile that overlapped no WMMA are now mostly
  hidden.
  Partly offsetting: +32 v_mov_b64 and about +70 s_set_vgpr_msb per iteration; late V LDS reads on
  LO warps; no overlap in the masked loops; PV(i) still serial after softmax(i).
  fast: flat or slightly negative. Could be near zero.
- **Risks:**
  - 4 VGPRs of headroom. Stacking any other lever on this body needs the full `review/run.sh`
    6-config matrix (gqa 1/2/4 x causal 0/1) first.
  - Before the review gate, non-causal (all gqa) compiled to 512 VGPR / 92 spill / 372 B scratch and
    causal gqa1 to 512 / 85 / 344 B. **Never remove or widen `_PIPE`.**
  - Expected bitwise equal to the champion (the per-tile op sequence is unchanged). Gate:
    - bitwise vs champion on toy (gqa2 takes the pipeline), short sequences n_iter=1/2,
      unequal_seqlen*, short_q, kv_len % 64 != 0;
    - the same shape repeated 80 times (mode-2 history);
    - the 200-run determinism gate.
  - `var/sgbA`, `var/sgbB` and `var/vearly` predate the gate, so **do not copy them**. Their settings are
    toggles on `op/`: `PIPE_SGB_ROUNDS=32` (sgbA), `PIPE_SGB_ROUNDS=32, PIPE_SGB_DSRD=1` (sgbB).
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arm A = `PIPELINE_QK_SOFTMAX=True` (as shipped, `PIPE_SGB_ROUNDS=0`). Control = the same file with
     `False`. Measure prod/proxy/fast.
  3. Only if A wins: optional arm B = A + `PIPE_SGB_ROUNDS=32`. Compile-only first; if the ISA equals A's,
     it is not an arm (h6). sched_group_barrier went 0/5 on bwd, so a low prior.

### h22 -- L15+L17+L20 softmax arithmetic -- QK->PV serial span -34%, 0 spill (proto `softmax`)

- **Path:** `$P=.../proto/softmax`; diff `softmax_proto.diff`; review `REPORT.md`; tools `tools/dyn.py`,
  `tools/crit.py`, `tools/cpu_algebra_check.py` (max rel 4.0e-6).
- **What it changes (toggles at :179-185, read at trace time):**
  - `SOFTMAX_PK_EXP` (L15): the exp2 argument is one v8 fma per sub-tile, emitted as `v_pk_fma_f32`
    instead of `v_fmamk`.
  - `SOFTMAX_LANE_ROWSUM` (L17): per-lane partial row-sum carried across tiles, with one cross-lane
    `peer()` in the epilogue instead of a permlane every tile.
  - `SOFTMAX_PK_ROWSUM`: the in-lane sum as a packed v8 tree (`False` = scalar tree).
  - `SOFTMAX_BRANCHFREE_RESCALE` (L20): no ballot and no branch; `o *= corr` every tile.
  - `SOFTMAX_PROTO` sets all four together.
- **ISA evidence (default PK+LANE(pk)+BF; prod = fast, md5 `6b286ce7`):**
  - Resources: VGPR 446, SGPR 98, spill 0/0, scratch 0, 4070 instructions.
  - LO-clean iteration: 537 instructions in one basic block. WMMA 64, v_exp 66, VALU 278, v_pk_fma 33,
    v_pk_add 32, v_pk_mul 65 (unconditional rescale), permlane 2, v_nop 24, vgpr_msb 47, 3 branches.
    The base is 551 (622 when a rescale fires).
  - QK->PV serial span 337 -> 222 (-34%). LO-masked 496 -> 347 (-30%). Whole-kernel permlanex16 16 -> 10.
  - Ablation, all 0 spill (LO-clean total / serial span):

    | arm | total | serial span |
    | --- | --- | --- |
    | PK only | 477 | 263 |
    | LANE only | 527 | 240 |
    | BF only | 627 | 233 |
    | PK+LANE+defer | 506 | 218 |
    | PK+BF | 583 | 193 |
    | PK+LANE(scalar)+defer | 535 | 220 |
    | PK+LANE(scalar)+BF | 606 | 222 |

  - Masked `v_cndmask ..., 0xff800000` counts equal base: prod 126, non-causal 128, win_sink 256.
- **Expected effect:** prod +2 to +5%. Round 3 showed softmax VALU on the critical path, but 2 waves per
  SIMD hide part of the shorter span.
  BF vs defer: at n_block=64, BF does not shorten the span (222 vs 218) and adds 31 VALU per iteration,
  because it runs 64 `v_pk_mul_f32` every 64 KV (ASM rescales once per 256 KV). BF's only upside is the
  single basic block, with rescale interleaved into PV. Only the card can say.
- **Risks:**
  - Non-causal SGPR spill 2 -> 4 and window+sink 6 -> 35. Both go to VGPR lanes with 0 scratch; neither
    path is scored.
  - PV span has 17 WAR `v_nop`; `s_set_vgpr_msb` 32 -> 47.
  - The L15 fma must stay **without** `fastmath=fast`. With `ninf`, LLVM may fold away the causal -inf
    mask. The review already fixed this with byte-identical ISA; keep it fixed when porting.
  - Any arm that keeps defer (`SOFTMAX_BRANCHFREE_RESCALE=False`) needs the h16 adversarial large-logit
    test for o and lse. o/lse SQNR >= 49 dB on all 3 shapes.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Three separate arms against control `SOFTMAX_PROTO=False` (champion bytes):
     - C1 = default, all True (md5 6b286ce7);
     - C2 = PK+LANE+defer (`SOFTMAX_BRANCHFREE_RESCALE=False`, `ENABLE_DEFER_RESCALE=True`);
     - C3 = PK only (`SOFTMAX_LANE_ROWSUM=False`, `SOFTMAX_BRANCHFREE_RESCALE=False`).
  3. C2 vs C1 is the BF question; C3 isolates L15.
  4. Land the best one. If the winner keeps defer, the large-logit gate is mandatory.

### h23 -- L13+L14 TDM prefetch depth 3 and clean-loop unroll x2 -- 4 compiled arms (proto `unroll`)

- **Path:** `$P=.../proto/unroll`; diff `proto.diff`; logs `logs/`, `review/`; tools `tools/`
  (`compile_isa.py`, `loops.py`, `loopops.py`).
- **What it changes:**
  - `N_KV_PP` (:150; 2 or 3), L13: with 3, tile t+2 is issued at tile t, and the top-of-tile wait only
    drains tile t (`s_wait_tensorcnt 0x2`; 0x0 only on the last tile).
  - `KV_UNROLL` (:164; 1 or 2), L14: the clean loop runs two tiles per `scf.for` iteration. The in-pair
    ring rotation is SSA renaming. An odd clean count is absorbed by the masked (predicated) loop,
    with no tail loop.
  - A local `n_kv_pp` falls back to 2 when 3 slots do not fit (qk_hdim=256, n_block=256) or with the V1
    loader. Slot size is 54272 B (0x1a800).
- **ISA evidence** (all 0 spill, 0 scratch; VGPR 445, LDS 327680; prod = fast):

  | arm | SGPR | kernel instr | per tile | v_nop | SALU | branch | wait |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | U1PP2 (both off) | = champion bytes | | | | | | 0x0 |
  | U2PP2 | 95 | 5556 | 557 | 48.5 | 40 | 4 | 0x0, 0 swap moves |
  | U1PP3 | 104 | 4559 | 573 | 37 | 49 | 7 | 0x2 |
  | U2PP3 (default) | 104 | 5852 | 576.5 | 42 | 51 | 6 | 0x2, 6 moves per pair |

  thd: 445/105, 0 spill. win_sink: 453 VGPR, with SGPR spill down from the champion's 6 to 0.
- **Expected effect:**
  - **U1PP3 vs U1PP2 is the core number of the round.** If the champion really stalls on
    `s_wait_tensorcnt 0` on the card: +2 to 8%. If one tile of compute already hides TDM latency:
    -0.5 to -1.5% (about +15 SALU/move/branch per tile). The static ISA cannot tell which.
  - U2 alone: null (-1 to +1%). It is scaffolding for L6 two-tile interleave and for "N_KV_PP=4 + one
    barrier per pair".
  - U2 hands odd remainders to the masked loop: prod about -0.2%, fast -1 to -2% (inside shape_band 0.90).
- **Risks:**
  - BWD: every prefetch depth > 2 lost 10-30%. That staging was dual-use; fwd TDM is not, so it is only
    a prior.
  - qk_hdim=192 with U2PP3: 8 VGPR spill, 14 SGPR spill, scratch 36. The champion there is 0/8/0, and
    U1PP3 and U2PP2 alone are 0 VGPR spill at d192. **If U2 ships, gate it to qk_hdim==128.**
  - Non-causal SGPR spill 2 -> 6, to VGPR lanes.
  - The CPU schedule proof (`prove_schedule.py`) covers only bshd causal, mask_left=False, sq==skv.
    Add sq!=skv and kv_len % 64 != 0 shapes to the gate.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arms in one session:
     - A = `N_KV_PP=2, KV_UNROLL=1` (control, champion bytes);
     - C = `N_KV_PP=3, KV_UNROLL=1`;
     - D = `N_KV_PP=3, KV_UNROLL=2`;
     - optional B = `N_KV_PP=2, KV_UNROLL=2`.
  3. Report C vs A first. D only earns its place if D > C by more than the floor.

### h24 -- L8 barrier per 2 KV tiles + split signal/wait -- p22s, 439 VGPR (proto `barriers`)

- **Path:** `$P=.../proto/barriers`; diff `proto.diff`; review `REPORT.md`; tools `tools/sync_model.py`
  (2484 configs, 0 failures), `tools/isa_loops.py`, extra ISA in `review/isa/`.
- **What it changes (:167-175):**
  - `KV_TILES_PER_BARRIER` (G) = 2: one WG barrier per 128 KV instead of per 64. The SIMD partner waves
    (i / i+4) may drift up to about 1 tile apart, and TDM lead grows from about 1 to 1.9 tiles.
  - `KV_BARRIER_GROUPS` (NG) = 2.
  - `SPLIT_BARRIER` = True: `s_barrier_signal` goes right after the last tile's PV `s_wait_dscnt(0)`, and
    `s_barrier_wait` at the next group top, so 32 PV WMMAs sit in the gap.
  - `KV_SLOT_ALIGN=1024` replaces the 64 KB slot floor.
  - `BARRIER_FENCE=True` puts WG fences around the raw intrinsics.
  - `BARRIER_PROTO=False` gives champion bytes.
  - **Remove the `PROTO_ENV` env override (:177ff) and hard-code the winner** in the op-evolve copy.
- **ISA evidence** (default p22s = G2 NG2 split; prod = fast, md5 `a12d1565`):
  - Resources: VGPR 439, SGPR 98, spill 0/0, scratch 0. LDS 327680 is still allocated; the ring uses 140 KB.
  - Per 256 KV: WMMA/v_exp/ds_load/TDM same as the champion. 2 barrier pairs. signal->wait gap 82-84
    instructions, with 32 PV WMMAs. tensorcnt waits 2x (0x0). dscnt 36 (champion 44). SALU 216,
    branch 28, v_nop 96, VALU 740.
  - Other variants, all 0 spill:

    | variant | VGPR/SGPR | barriers | tensorcnt | SALU | branch |
    | --- | --- | --- | --- | --- | --- |
    | p12s | 439/100 | 4 | 0x0 | 184 | 24 |
    | p13s | 439/98 | 4 | 0x2 | 240 | 32 |
    | p22n | 438/98 | 2 | 0x0 | 184 | 22 |
    | p23s | 439/106 | 2 | 0x4 (= ASM) | 270 | 36 |
    | p23n | 439/105 | 2 | 0x4 | 240 | 30 |

  - thd VGPR 445 -> 439. win_sink 455 -> 449, SGPR spill 4 -> 1.
- **Expected effect:** prod 0 to +4%, median +1%, low confidence. Barriers are 0 for 4 on bwd (h7), so
  this is a prior only. The mechanism is halved rendezvous count and partner-wave phase freedom.
  The split gap (bwd S2 null) and deep tensorcnt (p23s; bwd depth > 2 lost) have low priors.
  Cost: +48 to 100 SALU and +12 to 20 branches per 256 KV.
- **Risks:**
  - p23s and p13s fail at qk_hdim=256 (compile-time assert, ring 2x3x67584 B > LDS). If either wins,
    NG must drop to 2 at d256. p22s compiles at d128, d192 and d256.
  - More SGPR spill on unscored paths (non-causal 2 -> 6, d192 8 -> 9). It goes to VGPR lanes with no
    scratch.
  - Add a **d192** shape to the correctness gate: it is the only path with ops_per_tile=3.
  - The raw split intrinsics have no LLVM memory semantics. Re-check that no ds op crosses the gap in
    any port (`tools/isa_loops.py`).
- **How the round starts:**
  1. Copy per the common recipe, then hard-code the arm constants and delete `PROTO_ENV`.
  2. Arms vs `BARRIER_PROTO=False`:
     - p22s (default);
     - p22n (`SPLIT_BARRIER=False`), which separates barrier count from the split gap;
     - p23s only if p22s wins (`KV_BARRIER_GROUPS=3`, with the d256 fallback).
  3. Do this after h21/h22: the gap needs independent work to fill (h7).

### h25 -- L3+L29 in-WG q-tile pairing -- O-store overlap only, dispatch gain is zero (proto `pairing`)

- **Path:** `$P=.../proto/pairing`; diff `pairing.diff`; CPU dispatch/bijection model
  `bijection_check.py` (`bijection_check.out`: ALL OK); ISA `isa/HASHES.txt`, `review/`.
- **What it changes:**
  - `PAIR_Q_TILES` (:192): the launcher halves grid.x, and WG rank p runs light q-tile p, then heavy N-1-p.
    Tile 1's async O store (LDS -> VRAM) stays in flight while tile 2's Q/K/V TDM prologue issues;
    its O ring moves to the LDS tail.
  - The kernel infers paired mode at runtime from grid.x < N, so one binary serves both grids.
  - `PAIR_POLICY="exact"` pairs only when n_qt is even and the paired grid is whole 256-WG rounds:
    prod (2048 WGs) and proxy (256) pair, fast (32) does not.
  - `PAIR_PIN_LANE=True` stops LICM from hoisting about 25 lane values into v256+ (+100 msb per iteration).
  - Gated to qk_hdim==128.
- **ISA evidence** (prod = fast, md5 `bccf82e8d5b3`):
  - Resources: VGPR 447, SGPR 107, VGPR spill 0, scratch 0, 4853 instructions.
  - SGPR spill 5, to v256 lanes: 5 writelanes at entry and 5 readlanes per pass, none in KV loops.
  - Clean loops 620/620 with loop-top sync identical to the champion. The right-mask loop is 910/899
    (msb 162/158), once per pass.
  - One more barrier per pass and one `s_wait_asynccnt`.
  - Off -> `23e8ba8d3892` (champion); thd, non-causal and win_sink unchanged.
- **Expected effect:** prod +0 to +1.5%, most likely near min_gain (0.5-0.7%), possibly null.
  - Dispatch model: the champion's longest-first order already hits the lower bound (prod makespan
    1064 = 1064, proxy 69 = 69), so pairing gains **nothing on balance**.
  - The only upside is fixed-cost overlap: half as many WG launches, and the O-store drain overlapping
    the next tile's prologue (1 WG/CU because LDS is full).
  - fast runs the new code unpaired, so it should be neutral; it is the sentinel.
- **Risks:**
  - Before review, d256 failed to compile and d192 spilled to scratch. Both are now gated (hdim==128).
    **Never set `PAIR_POLICY="overfill"`**: the model shows odd n_qt or partial rounds losing 5-23%.
  - Only prod and proxy exercise the paired path on the card, and both have sq==skv, even n_qt and no
    tail rows. sq!=skv paired runs are covered by the CPU bijection check only.
  - Bitwise equality with the champion is likely but not guaranteed. A tile-localized large diff means
    a sync bug; uniform ULP diffs mean codegen. The 200-run determinism gate is the race check.
- **How the round starts:**
  1. Copy per the common recipe.
  2. Arm = `PAIR_Q_TILES=True`; control = `False` (champion bytes). Measure prod, proxy and fast.
  3. Lowest expected gain of the five: run it last or in a round's spare slot.

### h26 -- Closed levers from the 2026-09-27 compile-only prototypes

None of the five prototypes is dead. These sub-levers are closed by compile-only or CPU-model evidence;
do not spend a round rebuilding them:

- **L6 on non-causal / gqa=1 / win_sink / qk_hdim 192-256** at the current body:
  - non-causal: 512 VGPR, 92 spill, 372 B scratch;
  - causal gqa1: 512, 85 spill, 344 B scratch.
  These are gated to champion bytes in h21. Reopen only after something frees 60+ VGPRs.
- **L3 pairing as a load-balance lever:** the CPU dispatch model shows longest-first (h20, landed round 2)
  already at the makespan lower bound on prod and proxy. Pairing odd n_qt or partial dispatch rounds
  loses 5-23% (`mha_odd` 0.80x, `even_frac` 0.81x under overfill). Only "exact" policy, and only for
  overlap (h25).
- **L14 unroll x2 as a stand-alone speed lever:** it keeps the per-tile wait, barrier and
  `sched_barrier(0)`, and only saves 4 moves and loop overhead for about +10 v_nop. Expected null. Keep
  it only as scaffolding (h23).
- **U2 together with PP3 at qk_hdim=192:** 8 VGPR spill and 36 B scratch (h23). Gate U2 to d128.
- **Deep tensorcnt (NG=3 / PP3) at qk_hdim=256:** 3 slots do not fit 320 KB LDS (compile-time assert).
  Always fall back to 2 there.
- **L20 branch-free rescale at n_block=64 as a VALU saving:** it adds 31 VALU per iteration and does not
  shorten the serial span (222 vs 218 with defer). If h22's C1 loses to C2 on the card, close BF for
  good.
- **L15 with `fastmath=fast` on the exp-argument fma:** it licenses LLVM to delete the causal -inf mask.
  Closed as a correctness hazard.
