---
name: gfx1250-attn-campaign
description: Entry point for the gfx1250 (MI455X) flash-attention optimization campaign - FlyDSL forward/backward kernels scored against aiter's prebuilt ASM kernels. Use when starting or continuing FlyDSL or ASM attention fwd/bwd optimization on gfx1250, driving, resuming, retuning or monitoring the op-evolve jobs (gfx1250-flydsl-attn-fwd / -bwd), quoting or re-measuring the ASM baselines/bars, writing hint.md entries, or deciding which lever to try next. Gives current numbers, where code and ledgers live, the job state, the rules that have each cost a round or a power cycle, and a map of detailed references.
---

# gfx1250 attention campaign (FlyDSL vs aiter ASM)

Shape that scores: **b4 s8192 hq32 hkv8 d128 bf16, causal bottom-right, BSHD** (Llama-3.1-8B, GQA 4).
FLOP from `op-evolve/tools/op_flops.py`: fwd 2.199292e12, bwd 5.498229e12. One gfx1250 card.

## 1. Targets and current numbers (as of 2026-09-25)

| | bar (aiter ASM) | FlyDSL now | ratio | source |
|---|---|---|---|---|
| **fwd** | **1.5724 ms / 1398.67 TF/s** (n=51) | baseline (vendored aiter FlyDSL `m32x8`) 2.4005 ms / ~916 TF/s | 0.655x, gap 1.53x | `output/0924__flydsl/bar-census/fwd_anchor.json` |
| **bwd** | **7.6766 ms / ~716 TF/s** (census median) | champion **r20: 511.42 TF/s** (516.87 re-measured r22) | 0.72x | `output/0924__flydsl/DAY-SUMMARY.md` |

- Never cite **10.160 ms / 541 TF/s** for the bwd bar: a retracted autograd-shim artefact.
- fwd gap is pure efficiency (no structural term); bwd gap = 1.386 structure (7 GEMMs vs aiter's 5) x ~1.01.
  bwd efficiency arms are capped at ~1%; parity needs dQ fused into k_dkdv. Forward is worth ~0.83 ms/step,
  the bwd residual ~0.05 -> **forward first**.
- Same-session noise floor 0.24-0.66%; cross-session drift ~1.5%. Only same-process ratios compare.

## 2. Where the code lives

| what | path |
|---|---|
| op-evolve repo (CLI `OE/.venv/bin/op-evolve`, run from `OE`) | `OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve` |
| bwd job (ledger `job_context/state.yaml`, champion `job_context/op/current` = `rounds/020/op`) | `OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/` |
| fwd job | `OE/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/`; log `OE/LOG.fwd` |
| fwd spec + baseline op | `output/0925__flydsl/fwd-job/{gfx1250-flydsl-attn-fwd.yaml,op_baseline/}` |
| fwd hint (decision index L1-L30) | `output/0925__flydsl/fwd-hint/hint.md` |
| bwd hint corpus h1-h56 (**h54 ~line 4060 = index**) | `output/0923__flydsl/hint.md` (job reads its copy in `job_context/hint.md`) |
| 0.3.4.1 staged trees (fwd, bwd) | `output/0925__flydsl/{fwd341,bwd341}/{op032,op0341,op_clean}` |
| fwd A/B vs ASM (same process) | `output/0925__flydsl/fwd341/fwdab.py` |
| ASM bwd launcher | `primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py` (load by file path) |
| aiter source (not installed; put on `sys.path`) | `/home/lihuzhan/code/aiter-src` (`6963ae9`) |
| flydsl | 0.3.4.1 `~/.local/flydsl0341` (fwd, new work), 0.3.2 `~/.local/flydsl032` (bwd champion pin), 0.2.4 image default |
| FlyDSL source + skills | `/home/lihuzhan/code/2026_0925__flydsl/FlyDSL` |
| container | `fa-repro`, python `/opt/venv/bin/python3`; never restart / `pip install` into it |

(Paths starting `output/` or `primus_turbo/` are relative to the Primus-Turbo root.)

## 3. Current state and next step

- **fwd job: being launched** (2026-09-25 11:46) on flydsl 0.3.4.1 (0.3.4.1 vs 0.3.2 A/B: perf-neutral).
  Spec fixes in: `precision_gate` split out, deep every 5th round (fast_rounds 4 / fast_per_deep 4),
  `min_gain 0.007`, `max_rounds 30`, opus-5-5 agents. At 11:51 `job_setup` done, `op_setup` running.
  **Open before round 1** (details `references/fwd.md` §2):
  1. hint not delivered: `job_context/hint.md` is missing, and its `L#/f#` ids parse to zero hints
     (`core/hints.py` reads only `h<N>` rows). Add an `hN` table, copy it in.
  2. Setup review: no `attn_bwd` left in `op/`, FLOP `backward=False`, margin read not hardcoded, `flydsl0341` imported.
  3. Rewrite the deep_loop-trim patch's bwd-specific CAMPAIGN CORRECTIONS before round 5 (now committed in `OE` as `58b2134`;
     edit the 3 deep `_preamble.md` there, keep a fwd variant as a patch under `output/`).
  Loop PID/PGID 68769 (this boot): stop with `op-evolve stop --job gfx1250-flydsl-attn-fwd-20260925-114644`.
  Round-1 plan: grid layer first (L2 longest-first, L4 XCD remap with offline bijection check, L5), L1 `O_VARIANT` v1.
- **bwd job: paused mid round 23** (host reboot 10:10). r23's `act.yaml` is a complete loser (g71, gain ~0.953);
  `resume` would move `1-opt` aside and redo it -- or close it by hand (back up state.yaml).
  Do not resume while fwd runs. When it resumes: apply h33 defects a/b/c (compile-only), raise `min_gain` above the
  floor, spend the first card slot on the pre-registered **dQ-GEMM cost probe** (`references/bwd-history.md` §7, §11).

## 4. Fifteen rules that each cost a round or a power cycle

1. **One card, one GPU client.** fwd and bwd jobs never overlap; before trusting any number `ls /sys/class/kfd/kfd/proc`
   / `rocm-smi --showpids` must show only expected PIDs (r23's orphan contaminated validation, h56).
2. **A wedge = a human AC power cycle.** On a wedge: stop GPU loops, notify with dmesg evidence, switch to CPU work;
   never `modprobe -r amdgpu`, never re-probe. Budget in startups (skill `gfx1250-card-safety`).
3. **One shape per benchmark process** (~44% fault rate with multi-shape processes, h31).
4. **Compile-only first**: `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<arm>`
   (both arch vars; without them it silently targets gfx942). Any spill / scratch > 0 is a kill: it hangs the card.
5. **Bounds-prove every new index expression** on CPU; OOB reads caused wedge class C. Remainders predicated, never tail loops.
6. Never rocprofv3 **PC sampling** (wedged MES 3/3). `--kernel-trace` gives 0 rows; ATT captures nothing for FlyDSL JIT.
7. **Same-session, palindromic A/B with a clock witness**; prod ranks, fast/proxy are sentinels (a fast win hid a prod loss in r15).
8. **Static ISA metrics are not a ranking signal** (wrong 4 times; deleting 115 issue slots was -17%). Screen with them, rank on card.
9. **Assert the subject ran** (echo per-shape RC; `pw.sh` -> /dev/null faked "throttling ruled out"). Claims under ~0.5% are noise.
10. **Kill by explicit PID only**; `pkill -f`/`pgrep -f` self-match. Launch loops with `setsid`, stop with `op-evolve stop`.
11. **Never `resume --config`** (bumps spec_version -> ~58 min setup re-run, wipes hand patches). `min_gain` = hand-edit final.yaml.
12. **Never `import primus_turbo`** in job processes (pulls a FlyDSL tree needing removed `flydsl.expr.buffer_ops`);
    `_env.py` prepends the flydsl dir and asserts version **and** `__file__`.
13. gfx1250 API traps: `rocdl.s_waitcnt` raises (use `fx.barrier()`/`s_wait_dscnt`); `BufferAtomicAdd` is SCOPE_CU
    (use `UniversalAtomicAdd(Float32, SyncScope.Agent)`); V# `num_records` in 128 B units; `is_rdna_arch` misses gfx1250;
    aiter fwd managers raise on `num_waves != 8` (a roadblock, never "4 waves is slower").
14. **A number from another arch/kernel is a prior, not a result** (Kyle's gfx950 wins; the 1.57% HipKittens floor). Closed bwd
    axes (h54 §2: prefetch depth, sched_barrier, XCD, LDS ports, occupancy) stay closed unless new evidence.
15. **Push after every round**; power cuts truncated git objects twice. op-evolve edits live in another repo: save as patches under `output/`.
    BLAS env must be *assigned* (`HIPBLASLT_TENSILE_LIBPATH`, `TORCH_BLAS_PREFER_HIPBLASLT=1`) -- `setdefault` is a no-op here.

Operator working rules (never stop to ask, keep the loop running during side work, hand-over protocol,
whose containers may be stopped): `references/env-and-pitfalls.md` §11.

## 5. References (`references/`)

| file | one line |
|---|---|
| `baselines.md` | The two ASM bars (numbers to cite, n conventions), FLOP convention, the `.co` objects and launch paths, the vendor GQA OOB bug, `benchmark.py` method, why 10.160 ms is dead, the historical ladder and B0 numbers. |
| `bwd-history.md` | bwd rounds r0-r23 with gains, champion r20 code map, the 7-vs-5-GEMM gap, closed axes, the 4-wave barrier mystery, the dq contract change, open probes, unapplied h33 defects, next steps. |
| `fwd.md` | fwd job paths, spec state and the fixes still needed (hint delivery, setup review), and the L1-L30 decision index copied from `fwd-hint/hint.md`. |
| `env-and-pitfalls.md` | Machine/container/flydsl versions, BLAS env, wedge classes A/B/C and card rules, profiler rules, measurement discipline, git recovery, JIT cache, COMPILE_ONLY recipe, FlyDSL gfx1250 traps, stale user-skill spots, operator lessons from the transcripts (§11). |
| `op-evolve-ops.md` | Driving op-evolve: commands, schedule semantics, mid-job spec changes, acceptance math, artifact layout, hint.md format and the deep-round patch, monitoring recipe, bwd/fwd job state and launch checklist. |
| `flydsl-api.md` | FlyDSL 0.3.4.1 API-stability audit of fwd and bwd trees: stable idioms to use, required unstable gfx1250 ops, found breakage (V1 loader, wrapper bug), proposals needing A/B, compile-only ISA-diff recipe. |
| `kyle-learnings.md` | Kyle's gfx950 campaign harness lessons and attention levers, each marked transferred / open / dead on gfx1250, with the fwd lead list and a transfer checklist. |

Related user skills: `gfx1250-card-safety` (read before any GPU run), `flydsl-gfx1250` (idioms; stale on 0.3.4.1,
cache env, s_waitcnt), `gpu-kernel-campaign` (generic discipline; its 10.160 ms example is the shim).
The latter two carry dated 2026-09-25 correction notes pointing back here.
