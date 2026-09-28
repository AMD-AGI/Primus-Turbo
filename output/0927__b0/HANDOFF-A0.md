# HANDOFF B0 -> A0: gfx1250 FlyDSL attention campaign (written 2026-09-28 ~11:05 UTC on B0)

For a NEW Claude session on **A0 = heliosr-1b114-c07-1** (one gfx1250 card, container `fa-repro`, user lihuzhan,
same paths). B0 = ctheliosp-1b112-a37-1 ran the campaign 2026-09-27/28 on 4 cards (fa-g0..fa-g3, GPU1 wedged 09-28).
Abbreviations: `PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`, `OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`,
`B=PT/output/0927__b0`, `FJ=OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927`, `BJ=OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934`.
Read order on A0: this file -> `PT/.claude/skills/gfx1250-attn-campaign/SKILL.md` (A0-side state as of 09-27, partly
superseded here) -> `B/fwd/progress.md`, `B/bwd/progress.md` -> `B/e2e/RESULT-final.md` -> `B/profile/REPORT.md` §5.
`B/STOPPED.md` is the day-1 stop (09-27 15:44) and is stale; B0 resumed on day 2.

## 1. TL;DR state

Scoring shape: prod b4 s8192 hq32 hkv8 d128 bf16 causal BSHD. FLOP fwd 2.199292e12, bwd 5.498229e12.

| | fwd champion | bwd champion |
|---|---|---|
| job / round | `FJ` round 16 (refactor h44) = **r13ns** (r13 with `SPEC_STALE_MAX=False` in m32x8 + m32x2) | `BJ` round 29 = **r19h + h70 u2n** (k_dq kv-loop unroll x2) |
| live tree | `FJ/job_context/op/current` (unchanged since 09-28 08:23) | `BJ/job_context/op/current` (unchanged since 09-28 08:41) |
| md5 | `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py 370769c94c0f951655f66e10c39947d8`, `..._m32x2.py b09721562d696d5c1edc7eba29857a88`, `..._m16x8.py 1af89c64d2d3b3fcf28a280054a30c16`, `impl.py 35f4247ad75b0907b83d467b3f8bf33f`, `_env.py fe602c25556302ce80e0fff8b3c168ec` (full list `B/e2e/arms/SNAPSHOT2.md5`, `B/fwd-nospec/arms/fwd_r13ns.md5`) | `kernels.py 37f37052eb739579555d6bdf62a829cc`, `impl.py 08cb8533d75e82198fabb9514fd18ba5`, `_env.py 7f33871b977f16ae03b95dd89a44483e` (flydsl 0.3.2 pin; e2e copy with 0.3.4.1 pin `7df61bba26309ffc14b84a432fb451a7`, ISA identical, `B/e2e/recon/r29isa/isa_compare.txt`) |
| ruler 1: randn, blocked harness (job's own) | 1443-1455 TF/s (1.51-1.52 ms), **93-95% of ASM** (ASM ~1.43 ms). r13ns reads 4.8% slower than r13 here by design (h44) | **656.5 TF/s (8.375 ms), 80.4% of in-process beat 817.0 TF/s** (`B/bwd/progress.md:15`) |
| ruler 2: real q/k/v dumps, blocked | r13ns/ASM time **1.073-1.093** (`B/fwd-nospec/REPORT.md:17-19`) | not separately measured (bwd fly/asm stable 1.30-1.38 across inputs with r20, `B/profile/REPORT.md` §0.5) |
| ruler 3: real dumps right after a GEMM burst (training clock ~1270-1350 MHz) | r13ns/ASM **1.253-1.288** (1.62-1.64 vs 1.27-1.30 ms) | r29 vs r19h **-3.2%** whole op, k_dq -7.8% (`B/lab-kdq/REPORT.md` round 2 §1) |
| e2e Llama-3.1-8B 32L, per step (`B/e2e/RESULT-final.md` §0) | fwd 48.0-49.3 vs ASM 38.8-39.3 ms (~+10 ms) | bwd 285.4-290.0 vs ASM 233.1-236.7 ms (~+52 ms, the big term) |

- **e2e: FlyDSL/ASM step-time ratio 1.0323 / 1.0328** (ABBA and BAAB processes; adjacent-pair method 1.0324/1.0333),
  i.e. **3.2% per step** (~51 ms, tps 20,091 vs 20,747). History: r6+r20 5.0% -> r13ns+r20 4.0% -> r16+r29 3.2%.
- A0-side numbers in SKILL.md §1 (fwd r11 1111 TF/s, bwd r20 511 TF/s) are old-ruler, A0-clock numbers: do not
  compare with the above. Absolute B0 numbers are not comparable with A0 either; only same-process ratios transfer.
- A0's own fwd job `OE/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644` (paused r13, champion A0 r11) is
  **superseded** by `FJ` (a fork of it at r4 that also stacked A0 r11's levers, h34/h39). Do not resume the A0 fwd job.
- In flight on B0 at write time (11:00 UTC): `FJ` round 19 (fast, opt measuring) and `BJ` round 31 (fast). Both loops
  (`op-evolve resume` PIDs 2028638 fwd, 3467401 bwd) were still running. The operator stops them before copying
  (`op-evolve stop --job <job>`); whatever round is open is redone on A0 by a plain `resume`.
- `BJ/job_context/state.yaml`: `best_round: 29`, champions `prod: 29`, but `fast: 30`, `proxy: 30` = best-ever records
  of the rejected round 30 (op-evolve suggestion #1 defect: records from unpromoted arms). A future candidate must beat
  those per-shape records; if one is blocked only by them, use a refactor hint (as h39/h44/h68 did).

## 2. What A0 must bring over

**Git** (`PT`): B0's work branch is **`dev/lhz/flydsl-attn-b0`** (pushed, HEAD `64d59269`, 24 commits ahead of
A0's `dev/lhz/flydsl-attn`; A0's branch has 1 commit `d7ee84d5` not in b0; `git merge-tree` says the merge is clean).
On A0: `git fetch origin && git checkout dev/lhz/flydsl-attn-b0` (or merge `origin/dev/lhz/flydsl-attn` into it); keep
pushing there. Local branch `b0/0927` is B0-only history: never push it. The OE repo (`lhz/gfx1250`, HEAD `4e0ad31`)
has no B0 commits; all B0 op-evolve changes are hand edits inside the job dirs (below) and patches in `B/patches/`.

**Champion trees ARE in git** (added after this handoff was drafted, commit `5188c9da`): `B/champions/fwd_r16_r13ns/` and `B/champions/bwd_r29_r19h_u2n/` are byte copies of both jobs' `op/current` (md5 list `B/champions/MD5SUMS`; bwd kernels.py `37f37052...` = the job's r29). So a fresh job or an e2e run on A0 can start from git alone; the rsync below is only needed to keep the op-evolve job history/state/refcache.

**NOT in git** (B0 local disk only; `B/.gitignore` ignores every `arms/`, `*.pt`, `traces/`):

| item | B0 path | size | why A0 needs it |
|---|---|---|---|
| fwd job (whole) | `FJ/` | 1.4 GB (refcache 295 MB, rounds 1.1 GB) | champion tree, state.yaml, patched harness, hint.md, round history |
| bwd job (whole) | `BJ/` | 3.0 GB (refcache 1.2 GB, sha-refreshed) | same; **A0 has a stale same-name copy (paused mid r23) -- move it aside, do not merge** |
| loop logs | `OE/LOG.fwd-b0`, `OE/LOG.bwd` | small | history only |
| champion copies for e2e | `B/e2e/arms/{fwd_r16,bwd_r29_0341,bwd_r29_032,bwd_r20_0341,fwd_r6,asm,ref}` + `SNAPSHOT2.md5` | 2.2 MB | `E2E_FLY_TREES` targets |
| fwd-nospec arms | `B/fwd-nospec/arms/` (fwd_r13, fwd_r13ns, .md5, .diff) | 1.7 MB | real-dump A/B (`tools/ab_all.sh`) |
| lab trees | `B/lab-kdq/oe/` (arm trees; `OP` symlink -> `oe/artifacts/job/job_context/op`), `B/lab-bwd-r19/oe/` | 1.2 GB / 160 MB | only if re-running those labs |
| real training dumps | `/home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt` | 2.3 GB | ruler 2/3 (h45/h47); [B,S,H,D] bf16, step 43 layers 0/1/2/8/16/31 |

Committed and sufficient to rebuild without B0: `B/fwd/r9_BG/`, `B/fwd/champion_r6/`, `B/lab-bwd-r19/oe/.../op/r19h/`
(r19h), `B/lab-kdq/u2n.{kernels,impl}.diff`, `B/lab-kdq/oe/.../op/u2n/` (lab u2n tree, kernels md5 1fda1828 -- **not** the
job's r29 kernels.py 37f37052; the job re-applied it). **The exact champions r13ns (fwd r16) and r29 are NOT committed
anywhere**; r13ns = round 13 tree + the one-line `SPEC_STALE_MAX` flip, and round 13's tree is also only in `FJ/rounds/013/op`.
Copy them, do not reconstruct them.

Commands, run on A0 (assumes ssh A0 -> B0 works; `devsync` has `peer_sync = false`, so test `ssh ctheliosp-1b112-a37-1.mnb.dcgpu true` first):
```bash
B0=ctheliosp-1b112-a37-1.mnb.dcgpu; OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve
# 0) on B0 first: op-evolve stop both jobs; confirm no `op-evolve resume` process remains
# 1) keep A0's stale bwd job aside (never merge)
mv $OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934 $OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934.a0-pre-b0
# 2) job dirs (exclude core dumps and pycache; keep refcache)
for J in gfx1250-flydsl-attn-fwd-b0-20260927 gfx1250-flydsl-attn-bwd-20260917-115934; do
  rsync -a --exclude='__pycache__/' --exclude='core' --exclude='core.*' $B0:$OE/artifacts/$J/ $OE/artifacts/$J/; done
rsync -a $B0:$OE/LOG.fwd-b0 $B0:$OE/LOG.bwd $OE/   # LOG.bwd: A0 has its own -- rename one first if you want both
# 3) ignored trees under output/0927__b0 (minimal set) and the dumps
P=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0
rsync -a --exclude='__pycache__/' $B0:$P/e2e/arms/ $P/e2e/arms/
rsync -a --exclude='__pycache__/' $B0:$P/fwd-nospec/arms/ $P/fwd-nospec/arms/
rsync -a $B0:/home/lihuzhan/_prof_dump/ /home/lihuzhan/_prof_dump/
# optional: everything else ignored under output/0927__b0 except traces (~4.4 GB total with the labs)
# rsync -a --exclude='traces/' --exclude='__pycache__/' $B0:$P/ $P/
# 4) verify
(cd $OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op/current && md5sum -c $P/fwd-nospec/arms/fwd_r13ns.md5 2>/dev/null | grep -c OK)  # or compare with the md5 table in §1
md5sum $OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/current/kernels.py   # 37f37052...
```
If ssh B0 is not possible: on B0 `~/code/lhz-devsync/bin/devsync backup --snapshot` packs OE `artifacts/` (repo extras
`op-evolve_state`, **excluding refcache/**) and the ignored PT outputs; `refcache` is only in the `--full` tier; `_prof_dump`
is excluded (`*_dump/`) -- tar it by hand. On A0 `devsync restore --dry-run --host ctheliosp-1b112-a37-1 --items ...` first
(skill `xmachine-sync`); never let a restore overwrite A0's job dirs blindly. Without refcache the gate recomputes the fp32
reference on the card (the 09-22 power-cycle trigger): copy it, or re-build and re-stamp provenance (h69).

## 3. Operator changes inside the job dirs (all come along with a whole-dir copy; patches in `B/patches/`, chains verified to reproduce the live files)

| change | files | patch | A0 action |
|---|---|---|---|
| **Blocked timing** (`--block 9 --lead 4`, palindromic rounds; `--block 1` = old interleave) | `{FJ,BJ}/job_context/op/benchmark.py` (fwd `:17-18,189-191`; bwd `:24-32,206-208`); backups `.bak.pre-blocked` | `fwd_benchmark_blocked.diff`, `bwd_benchmark_blocked.diff` (apply after the sclk diff) | keep |
| **sclk witness** via `OE_PHYS_GPU` (reads `/sys/class/drm/renderD{128+8N}/device/pp_dpm_sclk`) | benchmark.py fwd `:61-64`, bwd `:71-74` | `*_benchmark_sclk.diff` | on A0 leave `OE_PHYS_GPU` **unset** -> falls back to rocm-smi (single card, correct) |
| **Validation target per scored shape** (proxy AND prod each >= beat; geomean reported only; user-approved 09-28) | `{FJ,BJ}/job_context/op/validation.py` (fwd ~`:131-141`, bwd ~`:309-316`); `.bak.pre-scored-shapes` | `*_validation_scored_shapes.diff` | keep (stops false `target_met`) |
| `GATE_DB 50 -> 49` (baseline itself reads 49.82-49.99 dB) | `FJ/job_context/op/gates.py:21` | `fwd_gates_49db.diff` | keep |
| `evolve.shape_band.proxy: 0.98` (multi-arm proxy spread +-6%) | `FJ/job_context/gfx1250-flydsl-attn-fwd_final.yaml:736-738` | (in yaml; `.bak.pre-proxy-band`) | keep |
| bwd `min_gain 0.0 -> 0.007` | `BJ/.../gfx1250-flydsl-attn-bwd_final.yaml:224` | `bwd_final_yaml.diff` | keep |
| **reviewer codex -> claude** (codex 401: `~/.op_evolve_openai` absent on B0) | fwd final.yaml `:199`, bwd `:109`; `.bak.pre-reviewer-claude` | -- | A0 has `~/.op_evolve_openai`: may restore `{sdk: codex, model: gpt-5.6-sol, effort: xhigh}` (check the account first with `OE/tools/check_llm_account.py`) |
| **container / host / gpu_id** | fwd final.yaml `:784, 803, 809, 870` (`fa-g2`, host B0); bwd `:247, 264, 268` (`fa-g3`) | `*_final_yaml.diff` (day-1 values fa-g0/fa-g1) | **must edit** before resume: `container: fa-repro`, `host: heliosr-1b114-c07-1`, `gpu_id: 0` comment "the only GPU". Hand-edit final.yaml + plain `resume` (never `--config`) |
| bwd refcache provenance `common_sha 988c14ca -> d441e55a` (ut/common.py changed, inputs/refs did not) | `BJ/job_context/op/refcache/*.pt`; backup `refcache.bak.pre-sha-update/` | -- (h69) | comes with the copy; do NOT use A0's old refcache with B0's ut/ |
| bwd state close of r23 by hand | `BJ/job_context/state.yaml` | `bwd_state_close_r23.diff` | already in the copy |
| hint.md | `{FJ,BJ}/job_context/hint.md` = `B/fwd-hint.md` (h1-h47, md5 378516d7) / `B/bwd-hint.md` (h1-h73, md5 e94d35c3) | -- | append the A0 host hints below |

**Hints that name B0 cards and must be superseded on A0** (append new `## hN` sections; the parser only reads `h<N>`):
fwd h29 (`B/fwd-hint.md:603`, GPU0/fa-g0), h38 (`:725`, GPU2/fa-g2); bwd h62 (`B/bwd-hint.md:4431`, GPU1/fa-g1),
h65 (`:4483`, GPU3/fa-g3). Proposed text (fwd `## h49`, bwd `## h74`):
```
## h49 -- HOST MOVE: A0 (heliosr-1b114-c07-1), one card, container fa-repro
From round <N> this job runs on A0: a single gfx1250 card, container `fa-repro` (logical device 0 = the only card).
This supersedes h29 and h38 (and h62/h65 in the bwd job): no fa-gN containers, no OE_PHYS_GPU (unset -> the sclk
witness falls back to rocm-smi, which is correct on one card), idle/contention checks = the whole node
(`ls /sys/class/kfd/kfd/proc`, `rocm-smi --showpids`). The fwd and bwd jobs share this card and NEVER run at the same
time. A0 absolute numbers are not comparable with B0's: re-measure champion and beat in the same process. Everything
else stands: blocked ruler (h40/h66), fresh FlyDSL JIT cache per process (h46/h72), real-dump + GEMM-burst ruler (h47).
```

## 4. Measurement rules learned on B0 (each cost a round or a false result)

1. **Blocked ruler, never call-by-call interleave** (`B/ruler/REPORT.md`, `B/ruler/bwd/REPORT.md`): interleaving let each
   call inherit the previous arm's power/clock state -> fwd FlyDSL-vs-ASM off by ~25%, bwd ~3.4%; three false wins
   (r6 +5.7% -> real +4.6%; L12 +6.4% and L21 +6.3% -> dead). Blocked A/A: fwd +-0.19%, bwd +-0.05%. Keep an A/A copy of
   the champion in every ranking process; claims < 0.5% are noise.
2. **Real-dump ruler after a GEMM burst** (training operating point): `B/fwd-nospec/tools/ab.py AB_COND=gb` (10 x bf16
   32768x4096x14336 before every timed call, ~25 ms/burst, 5 calls x 4 palindromic rounds, 3 rotated orders);
   `AB_COND=blk` = real dumps under the blocked method. bwd equivalent: `B/lab-kdq/tools/kbench.py` (gb mode).
   The burst **must use the IMAGE hipBLASLt library** `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`:
   the fwd tree's `_env.py` re-points `HIPBLASLT_TENSILE_LIBPATH` to `~/.local/hipblaslt-gfx1250` (GEMM ~80 TF/s, clock stays
   ~2155 MHz, ruler void); ab.py restores the image lib after loading arms (`ab.py:18-22`). Verify sclk ~1270-1350 MHz.
   NB A0 idles/throttles at ~1.0-1.05 GHz under load (`references/env-and-pitfalls.md` §1c) -- re-establish on A0 what
   the blocked and burst conditions read before trusting either; the gap may already look like the training one.
3. **No randn-only wins** (h45, h47): any data-dependent mechanism is judged on the 6 real dumps; a candidate worth
   shipping wins >= 2% on real-dump-after-burst with randn inside its band (operator promotes by refactor if the randn
   geomean blocks it). Fwd r18's g64 was randn +0.9%, real +0.2% -> null.
4. **Speculation is dead** (h43-h45): stale-max/guessed-max softmax recomputes 13-25% of tiles on real data (score std
   21-53 vs randn ~1). Never re-introduce.
5. **Fresh `FLYDSL_RUNTIME_CACHE_DIR` per process** (h46/h72): FlyDSL 0.3.4.1 JIT key (`jit_function.py:580`) ignores
   module-level constants; after r16's promotion `/tmp/flycache` served r13's binary and an A/A copy read 6.6% apart.
   The job harness does NOT enforce this (no `FLYDSL_RUNTIME_CACHE_DIR` in benchmark.py/validation.py): clear
   `/tmp/flycache` in fa-repro after every promotion/refactor, and use `mktemp -d /tmp/flycache.XXXX` in hand runs.
6. One shape per process, compile-only first (0 spill / 0 scratch), bounds-prove new index math on CPU (SKILL §4).

**E2E recipe** (`B/e2e/run_e2e.sh`, `B/e2e/tools/final_e2e.sh`, `final_trees.sh`; RESULT-final.md §1-2):
```bash
E=$PT/output/0927__b0/e2e; . $E/tools/final_trees.sh   # FLY_TREES_JSON -> arms/fwd_r16 + arms/bwd_r29_0341
E2E_FLYCACHE=/tmp/flycache_<tag>_$(date +%H%M%S) E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1 \
  E2E_ENV="-e E2E_FLY_TREES=$FLY_TREES_JSON" bash $E/run_e2e.sh train <tag> "asm,fly,asm,fly;asm,fly,fly,asm" 62 10
# second process with "fly,asm,fly,asm;fly,asm,asm,fly"; analyse with tools/steady_arms.py + tools/trace_breakdown.py
```
A0 adaptations to `run_e2e.sh` (B0-specific lines): `:27 LOCK=/tmp/b0-gpu0.lock` (rename ok), **`:28 CT=fa-g0 -> fa-repro`**,
`:41-43` dmesg filter drops `0002:04:00.0` (B0's dead GPU1; harmless on A0), `:87-131` clocks from `/sys/class/drm/card0`
(confirm A0's card index: `ls /sys/class/drm/card*/device/pp_dpm_sclk`). Same edits in `B/fwd-nospec/tools/run_op.sh:12-21`,
`B/profile/tools/run_op.sh`, `B/lab-kdq/tools/run1.sh:7` (also drop `-e OE_PHYS_GPU=0` there). Dependencies to confirm on A0:
`/home/lihuzhan/code/2026_0828__primus/Primus`, `$PT/../wt-bakeoff`, `~/.local/flydsl0341`, `/home/lihuzhan/code/aiter-src`.
nkfix: use `B/gemm/nkfix_b0.py` (installed by the shim when `E2E_NKFIX=1`); A0's `output/0915__opt/bin/nkfix.py` is broken at
HEAD (`_MM`, `stats`, ... undefined; `B/gemm/REPORT.md` §1). Without nkfix every bwd GEMM lands on MT32x16x32 (~10x slower step).
Always two processes (ABBA + BAAB), a fresh E2E_FLYCACHE each, and check `/proc/<pid>/environ` has `E2E_FLY_TREES`.

## 5. Ranked next steps

Gap per step (e2e): bwd ~52 ms (k_dkdv 185.7-188.0 ms/step, k_dqg 96.0-98.3, k_delta 3.8; ASM 233-237 incl. GQA sum 3.9),
fwd ~10 ms. So **bwd first on A0** (reverses SKILL.md §1 "forward first", which predates the e2e).

1. **bwd k_dkdv** (two thirds of fly bwd): clock-sensitive 1.24x (1350 vs 2350 MHz) vs ASM main kernel 1.16x
   (`B/profile/REPORT.md` §5 #5, ~12 ms if matched). Closed by measurement (h71, h73): unroll x2 +7.3%, VF/fma softmax null,
   sched_barrier losses, FUSED5 (dQ fused with atomics) 0.39x dead (h64). Open: what makes it clock-sensitive (waits/
   barrier stalls vs issue) -- counters at the low-clock operating point, not static ISA (SKILL rule 8).
2. **bwd k_dq**: still ~97 ms/step; lab-kdq found it IS clock-sensitive (4.05 ms at ~1300 MHz vs 2.98 at ~1950), contrary to
   profile §0.5. Closed: 2 waves/SIMD (+41%), GQA-grouped WG, epilogue via LDS (null), sched_barrier between halves.
   `k_delta` fused into k_dq prologue null; into k_dkdv prologue unmeasured (~3.7 ms ceiling).
3. **fwd clock sensitivity** (~10 ms): r13ns prod m32x8 hot loop 4130 instructions; target v_nop/s_delay, SALU bookkeeping,
   redundant waits, barriers per KV tile (h47 item 2); judge on ruler 3.
4. **Second ruler inside op-evolve** (suggestions #9/#11/#12/#16 in `B/OP-EVOLVE-SUGGESTIONS.md`): scored real-dump +
   GEMM-burst condition in validation/acceptance; and **`evolve.gain_weights`** (prod 1.0, proxy 0.25, fast 0; exists in
   A0's op-evolve per `references/fwd.md:111`, not in B0's `OE/op_evolve/core/acceptance.py`) -- **both need user approval**
   (framework/acceptance change). Until then promote real-data winners by refactor hint.
5. **nkfix NaN bound**: B0 0 events in 7 x 32L runs (per-run 95% bound 35%); ~13 clean runs needed to exclude A0's
   historical 21%/run; run with `NKFIX_CHECK=2` (`B/gemm/REPORT.md` §4). A0 is where the NaNs were seen -- watch step <= 10.
6. **hipBLASLt NN bf16 tuning request** (root fix for nkfix's ~100 ms/step transpose+check): `PT/output/0915__opt/VENDOR-REPORT-hipblaslt.md`.
   Cheap interim: `NKFIX_CHECK=0` (-33 ms/step, also raises attention clock), transpose shared x once for q/k/v.
7. Harness hygiene: per-process JIT cache dir in benchmark.py (§4.5); op-evolve suggestions #1 (champions only from promoted
   rounds), #13 (target per scored shape -- applied as hand patch), #14 (refcache sha mismatch should fail, not recompute).

## 6. Pitfalls (short; details in SKILL.md §4 and `references/env-and-pitfalls.md`)

- **Wedge = human AC power cycle.** On B0 GPU1 wedged 09-28 (MES failed to respond). On any fault: stop GPU loops,
  report dmesg lines, switch to CPU work (memory: unattended work authorized -- do not wait). Never `modprobe -r amdgpu`,
  never PC sampling. After reboot: `sudo modprobe amdgpu`, `sudo sysctl -w kernel.dmesg_restrict=0`, `docker start fa-repro`,
  `pip uninstall -y primus_turbo` inside if it came back, and check git objects (power cuts truncated them twice).
- **One card, one GPU client**: on A0 the fwd job, bwd job, e2e and labs are mutually exclusive; check
  `ls /sys/class/kfd/kfd/proc` before trusting a number. The GEMM burst ruler is short and allowed; no sustained burn loops.
- **Never `op-evolve resume --config`** (bumps spec_version, ~58 min setup, wipes hand edits). Hand-edit `*_final.yaml`,
  back it up, plain `resume` with `setsid nohup ... >> LOG 2>&1 &`; stop with `op-evolve stop --job`; kill by explicit PID only.
- **False `target_met`**: bwd r27 ended the job on a geomean carried by fast (ASM slow there, 1.66x) while prod was 0.78x.
  Fixed by the validation patch (§3). A job in `finished: target_met` resumes with a plain `resume` (BJ did at 06:43).
- **JIT cache** stale-binary hazard (§4.5). Two arms with byte-identical time/VGPR that should differ = key miss.
- **Pre-commit 2 MB guard**: B0 had a local hook `PT/.git/hooks/pre-commit` refusing staged files > 2 MB (origin is
  AMD-AGI/Primus-Turbo); `.git/hooks` is not versioned -- copy it to A0. Never commit `arms/`, traces, `.pt`, refcache.
  Never push `b0/0927`. Push after every round (memory rule).
- BLAS env must be assigned inside the exec (`TORCH_BLAS_PREFER_HIPBLASLT=1`, image `HIPBLASLT_TENSILE_LIBPATH`); use
  `bash -c`, not `-lc` (`/etc/profile.d/zz-gfx1250.sh` forces PREFER=0).
- Never `import primus_turbo` in job processes; `_env.py` pins flydsl by `sys.path` + asserts `__file__`
  (fwd: 0.3.4.1 `~/.local/flydsl0341`; bwd job: 0.3.2 `~/.local/flydsl032`; bwd e2e copy: 0.3.4.1, ISA-identical).
- Reporting to the user: Chinese, one summary table per round (SKILL.md "汇报规范"); commits/hints in English.

## 7. Proposed addition to `PT/.claude/skills/gfx1250-attn-campaign/SKILL.md` §3 (operator applies; insert at the top of §3, line 44)

```
- **2026-09-28: the campaign moved B0 -> A0. Read `output/0927__b0/HANDOFF-A0.md` first**; it supersedes the fwd/bwd
  bullets below and §1's numbers. Champions: fwd = job `gfx1250-flydsl-attn-fwd-b0-20260927` round 16 (r13ns,
  speculation off), bwd = job `gfx1250-flydsl-attn-bwd-20260917-115934` round 29 (r19h + u2n). Both job dirs were
  copied from B0 (the A0 bwd dir is kept as `*.a0-pre-b0`); the A0 fwd job `...-20260925-114644` is retired.
  e2e Llama-3.1-8B: FlyDSL/ASM 1.032 per step (bwd ~52 ms, fwd ~10 ms of the ~51-62 ms gap) -> bwd first.
  Rulers: blocked timing (h40/h66), real q/k/v dumps after a GEMM burst (h47), fresh JIT cache per process (h46/h72).
  Before resuming either job on A0: final.yaml container `fa-repro` / host / gpu_id, append host hints h49 (fwd) / h74 (bwd).
```

## 8. Late update (2026-09-28 ~12:00 UTC)
- **fwd job hard-stopped in round 20 (deep, act step).** The act's `adv_m32x8.py` faulted GPU2 inside the hipBLASLt fp32
  `forward_reference` GEMM (TCP permission fault, no MES hang); the loop was killed before it re-ran the script. New hint
  **h50** forbids fp32 references on the card. On resume the round-20 act is redone from scratch.
- **fwd hint h48 (must refactor) is pending**: adopt `rounds/019/op` (+1.5% real-dump, +1.7% randn vs champion; blocked only
  by r18's unshipped best-ever record). It applies at the head of the next round after round 20.
- Host hints for A0 are therefore numbered **h49 (fwd)** / h74 (bwd) (section 3).

## 9. Final state at B0 shutdown (2026-09-28 12:45 UTC)
- **Both loops stopped** (no op-evolve process, no KFD process on B0):
  - fwd `gfx1250-flydsl-attn-fwd-b0-20260927`: stopped in **round 20 (deep), act step 01_implement** (hard stop after the
    GPU2 fault, see section 8). best_round **16** (r13ns). Pending hint: h48 (must refactor, adopt rounds/019/op).
  - bwd `gfx1250-flydsl-attn-bwd-20260917-115934`: stopped at the **start of round 33 (fast, opt)** (`.stop` flag + TERM;
    `run_loop` removes a leftover `.stop` on the next resume). best_round **29** (r19h + u2n). Rounds 30-32 rejected
    (0.9967... null g89; 0.9915; 0.9861): bwd has 3 non-improving rounds -> start A0 with the ranked next steps in section 5.
- Champions unchanged since the `champions/` snapshot (commit 5188c9da): fwd r16 = `champions/fwd_r16_r13ns/`,
  bwd r29 = `champions/bwd_r29_r19h_u2n/` (kernels.py md5 37f37052...).
- Final e2e (fwd r13ns + bwd r29 vs ASM): per-step fly/asm 1.0323 / 1.0328 (`e2e/RESULT-final.md`).
- Day report (Chinese, HTML): `output/0927__b0/REPORT-0928.html`.
- B0 GPU1 is still wedged (MES failed to respond) and needs an AC cycle; containers fa-g0/fa-g2/fa-g3 are left running
  (idle), fa-repro is parked as `fa-repro-parked`.
