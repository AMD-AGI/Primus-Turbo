# op-evolve operations for the gfx1250 attention jobs

How to drive, watch, retune and repair the two op-evolve jobs in this campaign without
reading old transcripts. Verified 2026-09-25 against the files named in each row.
Legend: ✅ verified on disk / in code · ⚠ conditional or stale-prone · ❌ dead, do not use.

Abbreviations: `OE` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve` (the repo),
`BWD` = `OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934`, `PT` = Primus-Turbo repo root,
`HINT` = `PT/output/0923__flydsl/hint.md`.

Related skills (user-level). Link to them rather than copying from them:
- `~/.claude/skills/gfx1250-card-safety/SKILL.md`: what wedges the card, the recovery ladder, watchdogs.
- `~/.claude/skills/gpu-kernel-campaign/SKILL.md` §6 (self-matching patterns), §7 (monitors that stay silent until something breaks), §16 (bind a watchdog to its own run).
- `~/.claude/skills/flydsl-gfx1250/SKILL.md`. ⚠ STALE in two places: §4 says aiter needs 0.3.2, but 0.3.4.1 now works (see §9 below). §6 says "no FlyDSL cache env var", but `FLYDSL_RUNTIME_CACHE_DIR` exists (`~/.local/flydsl0341/flydsl/compiler/jit_function.py:1276`).

---

## 1. Where things are

| what | path | status |
|---|---|---|
| repo (editable install) | `OE`; `op_evolve.__file__` resolves to `OE/op_evolve/__init__.py` | ✅ prompt edits in the tree take effect immediately |
| CLI | `OE/.venv/bin/op-evolve`. It is **not** on the global PATH (`which op-evolve` prints nothing) | ✅ |
| framework code worth knowing | `OE/op_evolve/core/{schedule,acceptance,tune,loop,gate,hints,job}.py`, `OE/op_evolve/cli.py` | ✅ |
| bwd job | `BWD/` (the ledger is `job_context/state.yaml`) | ✅ |
| bwd resolved spec | `BWD/job_context/gfx1250-flydsl-attn-bwd_final.yaml` (plus the `.bak.*` hand-edit backups) | ✅ |
| bwd original input | `OE/jobs/gfx1250-flydsl-attn-bwd.yaml` + `.HINTS.md` (untracked) | ✅ history only |
| fwd spec, current | `PT/output/0925__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml` + `op_baseline/` | ✅ written 09-25 11:37 |
| fwd spec, staged copy | `OE/jobs/gfx1250-flydsl-attn-fwd.yaml`. Byte-identical to the 0925 copy (both `claude-opus-5-5`, checked 2026-09-25 after launch); the running fwd job loaded it | ✅ |
| fwd spec, old | `PT/output/0923__flydsl/fwd-job/` (has the `precision_gate` bug, uses flydsl 0.3.2, `min_gain 0.005`) | ❌ superseded |
| hint corpus (git-tracked) | `HINT`, 4244 lines, h1..h56; **h54 at line 4060 is the index, so read it first** | ✅ |
| hint file the job actually reads | `BWD/job_context/hint.md`. A separate inode, byte-identical to `HINT` as of 09:31 | ⚠ keep both in sync by hand |
| deep-round prompt patch | `PT/output/0924__flydsl/deep-enablement/deep_loop-trim.patch` (814 lines) | ✅ **committed** in `OE` as `58b2134` (2026-09-25 11:38; `git apply --reverse --check` passes). The commit message says the bwd corrections moved to hint.md, but the three `_preamble.md` files **still inline the bwd CAMPAIGN CORRECTIONS** |

---

## 2. Commands (always run from `OE`)

`ARTIFACTS_ROOT = Path("artifacts")` is **relative** (`core/job.py:14`). If you run from another
directory, `run` creates the job there, and `status`/`resume` then cannot find it. ✅

```bash
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve
export PATH="$PWD/.venv/bin:$PATH"
J=gfx1250-flydsl-attn-bwd-20260917-115934

op-evolve status --job $J                       # ledger + "last: <event>"
op-evolve stop   --job $J                       # writes $JOBDIR/.stop; halts at next MODULE boundary
op-evolve stop   --job $J --force               # also SIGTERMs; interrupted module is redone
op-evolve tune   --job $J --fast-rounds 24 --fast-per-deep 4 --max-rounds 40

# launch / resume: detached, own process group, log NOT in /tmp (a reboot wipes it)
LOG=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/oe-$J.log
setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve resume --job $J >> "$LOG" 2>&1 < /dev/null &
setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve run --config jobs/gfx1250-flydsl-attn-fwd.yaml >> "$LOG" 2>&1 < /dev/null &
```

| rule | evidence |
|---|---|
| Use `setsid`. A bare `kill <pid>` on the loop orphans its agent child, and that child goes on to corrupt `job_context`. Stop with `op-evolve stop`, or with `kill -TERM -<PGID>` (note the minus) | `OE/tools/supervise_job.sh:18-21` ✅ |
| The 09-25 08:35 launch used plain `nohup .venv/bin/op-evolve resume … > /tmp/r24_resume.log` **without setsid**. The host rebooted at 10:10:51 (`uptime -s`), so that log is gone | transcripts; `uptime -s` ✅ |
| Before resuming, delete any leftover `.stop`. `request_stop` writes it and `run_loop` unlinks it at start (`core/loop.py`, around the `stop_flag` lines). Path is `BWD/.stop` (`core/job.py:108`) | ✅ |
| Agents run with `--permission-mode bypassPermissions` and `--effort xhigh`. Nothing will ask you for approval | `ps` of the live claude child ✅ |
| Setup cost is **~58 min of card time**: `job_setup` 618.3 s + `op_setup` 2850.0 s | `BWD/job_context/state.yaml` `setup:` ✅ |
| Review gates **auto-approve after 10 min** (`review.auto_approve_after: 10m`). To review, you have to be there in that window | bwd final.yaml:319; fwd spec `review:`; `core/review.py:30-53` ✅ |
| Preflight tools cost **no card time**: `python3 tools/check_llm_account.py --config <yaml>` and `tools/check_runner.py`. ⚠ They leave stray `OE/artifacts/job_context/env/` and `OE/job_context/env/` directories behind. Harmless | `OE/README.md`; `ls` ✅ |

### supervise_job.sh: ⚠ do not use it as-is
- `VENV="/home/xiewen12/workspace/.venv-op-evolve"` is hardcoded to **another user's** venv (`tools/supervise_job.sh:39`). The script falls back to `command -v op-evolve`, so it only works when `PATH` already contains `OE/.venv/bin`.
- It was dropped after 09-21. The last lines of `OE/artifacts/supervisor.log` (09-21 17:47) show it looping "GPU not visible to rocminfo … waiting 900s" on a wedged card. It never restarts into a dead card, but it also never ends on its own.
- If you use it: `setsid nohup env PATH="$PWD/.venv/bin:$PATH" tools/supervise_job.sh --job $J >/dev/null 2>&1 &`. It restarts on loop death with backoff 60→900 s, `MAX_RESTARTS=60`, and stops only on a `finished` event.

---

## 3. Schedule semantics (`core/schedule.py:25-67`) ✅

The **deep round opens each cycle**. It does not close it.

| set | rounds |
|---|---|
| neither | every round fast |
| `fast_rounds: N` only | 1..N fast, then every round deep |
| `fast_per_deep: K` only | D, K×F, D, K×F, … from round 1 |
| both | 1..N fast, then D + K×F repeating: `deep iff (n-N-1) % (K+1) == 0` |
| `fast_rounds: 0` | all deep |

- **A deep round every 5th round** (5, 10, 15, …) means `fast_rounds: 4, fast_per_deep: 4`. Check: (5-4-1)%5=0 and (10-4-1)%5=0. The 0925 fwd spec already sets this ✅.
- bwd now: `fast_rounds: 24, fast_per_deep: 4`. The next deep round is **25**, then 30 and 35; `max_rounds: 40`.
- `--fast-rounds` counts from round 1, not from the current round. `tune` prints a preview of the next 12 rounds, so read that before trusting the change.
- Cost, measured on bwd: fast rounds took 2250–4446 s (r1–r22). Deep r17 took **12,670 s (3.5 h)**: profiling 6916, plan 2487, act 2669, reflect 599. README says ~2.4 h; on this card it runs longer.

---

## 4. Changing the spec mid-job

| lever | how | effect | evidence |
|---|---|---|---|
| `fast_rounds`, `fast_per_deep`, `max_rounds`, `max_timeout`, `--target-tflops` | `op-evolve tune` | Rewrites every occurrence in final.yaml, re-parses it, **does not touch state.yaml**, no spec bump. A running loop only sees the change at the next `resume` | `core/tune.py:1-22,70-110` ✅ |
| `--beat-margin X` | `tune` | Rewrites only the `beaten by N%` inside the `beat:` line, and the `margin:` key if one exists. It raises `TuneError` if the beat line has no `beaten by N%` (regex `(beaten by\s+)(\d+(?:\.\d+)?)(\s*%)`, `tune.py:32`). ⚠ It does **not** change `beat_margin_pct`, and **bwd's `validation.py:45` hardcodes `BEAT_MARGIN = 0.0`**, so on bwd `--beat-margin` changes the prose and **not the gate** | `tune.py:112-160`; `BWD/job_context/op/validation.py:45,309` ✅ |
| `min_gain` | **not a tune option** (`cli.py:74-86`) | Hand-edit **every** occurrence in final.yaml, back the file up first (`cp … .bak.<why>`), then do a plain `resume` **without `--config`**. That is how bwd was hand-edited three times and stayed at `spec_version v000` | `BWD/job_context/*.bak.*`; state `spec_version: v000` ✅ |
| anything via `resume --config edited.yaml` | ❌ **NEVER** | `apply_spec` bumps the version (`core/job.py:174-190`). Setup counts as done only if `record.spec_version == version` (`modules/setup/run.py:151`), so **both setup stages re-run (~58 min)** and regenerate `op/{baseline,eager,ut,benchmark.py,validation.py}`, which wipes every hand patch | ✅ |

---

## 5. Scoring and acceptance (`core/acceptance.py`) ✅

- **Done** = `op/validation.py` exits 0, run by the framework (`core/gate.py:61-62`). That ends the job **only if** the round's score is ≥ 0.99 (`_TARGET_MET_SCORE`, `core/loop.py:221,244-255`). A disagreement is recorded and the job keeps going. bwd `validation.py` exits 0 or 2 ("no partial credit"), and every bwd round so far has exited 2 (`speed FAIL`).
- **Kept (accepted)** is decided by arithmetic over `act.yaml`, which comes from same-session re-measurements. The score does not decide it:
  1. `gain` = mean over shapes of `this_round_tflops / champion_tflops`. It must be **> 1 + min_gain**.
  2. Every shape below target must stay ≥ `band × champion`, where `band = max(0.95, 1-min_gain)` if `min_gain` is set, else 0.95.
  3. A shape that was at target must not fall below it.
  4. Every `no_regression` guard must hold its floor.
- `score` = mean(tflops/target), **uncapped**. It is reported only. Champions are tracked **per shape** and updated even on rejected rounds.
- ⚠ The mean is arithmetic over fast/proxy/prod, so a big `fast` win can pay for a `prod` loss. That happened on bwd r15, which was reverted by hand (`state.yaml.bak.before-r15-revert`). Campaign rule: `prod` ranks candidates, and fast/proxy are only sentinels (HINT h7).
- The same-session noise floor is **0.24–0.66% per shape** (h54 §3 trap 2, h56). This is why fwd uses `min_gain` 0.007 and not 0.0.

---

## 6. What a job leaves behind

```
artifacts/<name>-<id>/
  job_context/state.yaml          # READ FIRST: rounds, score, gain, accepted, champions, lifecycle
  job_context/<name>_final.yaml   # the spec the loop loaded (tune edits this)
  job_context/hint.md             # the ONLY human input
  job_context/findings/{facts,pool,route,dead_ends}.md   # the run's memory (route.md rendered from hints each round)
  job_context/op/{baseline,beat,current,eager,ut,refcache}/ benchmark.py validation.py
  job_context/history/{v000.yaml,changelog.md,review/}   job_context/logs/*.log
  rounds/NNN/{op/, timing.yaml, gate.log, findings_snapshot/}
     fast: 1-opt/{act.yaml,opt.md,raw/} 2-reflect/        deep: 1-profiling 2-plan 3-act 4-reflect
     *.stale-<stamp>/  = an unfinished attempt moved aside on resume (core/loop.py:734-757)
```
- Throughput is **not** in state.yaml headers. It lives per shape in each round's `act.yaml`. state.yaml copies `tflops`/`targets`/`incumbents` per round.
- The champion's code is `job_context/op/current`, which equals `rounds/<best_round>/op`. Verify with `diff -r`.
- ⚠ **Core dumps pile up** in `job_context/op/`, owned by root because they come from the container: `core` 255 MB (09-25 09:30), `core.host.val3-20260924` 1.2 GB, `core.gpu.stale-20260921` 34 MB, plus `op/.runs/r2_gpu*/core*`. BWD is 9.4 GB total. Remove them by explicit path via `docker exec fa-repro rm <path>`. The disk is 78% full (767 GB free).

---

## 7. hint.md: the only human input (`core/hints.py` docstring) ✅

- The format is a table `| id | type | title | status |` plus a `## hN -- title` section for each entry. `type` holds a level from `must|advise` and a kind from `standing|note|refactor`. `status: open` means pending, and Python owns the status column after that.
- `standing` is rendered into route.md's constraints table. `note` is a queue item that the round must place (checked by the gate). `refactor` runs at the head of a round, at most one per round.
- It is applied **at the head of each round, without stopping the job** (`core/loop.py:434+`, `_apply_hints`). To add a hint: append it to `HINT` (git-tracked), `cp` it to `BWD/job_context/hint.md`, and commit in PT.
- ⚠ **Deep rounds do not see hint prose.** `hints` is imported by `fast_loop/run.py` and `deep_loop/reflect/run.py` only. Deep `profiling`/`plan`/`act` never import it; plan sees only route.md's rendered rows. The workaround is the **deep_loop-trim patch** (`PT/output/0924__flydsl/deep-enablement/deep_loop-trim.patch`):
  - 3 `_preamble.md` files (act, plan, profiling) get an inlined "CAMPAIGN CORRECTIONS" block. ⚠ That block is **bwd-specific** (k_dkdv 740 VGPR, k_dq 960, TDM has no target in k_dkdv, …). **It is wrong for the fwd job. Rewrite it for fwd before the fwd's first deep round (round 5).**
  - The profiling prompts 00–07 and 3 schemas get **machine-safety** trims: no `pip install` into `/opt/venv`, no container restart, the ATT decoder pinned at `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib`, no `rocprof-compute`. These are job-agnostic, so keep them for fwd.
  - Since 2026-09-25 11:38 the patch is **committed** in `OE` (`58b2134`), so a checkout of that branch keeps it. Verify with `git -C $OE apply --reverse --check <patch>` (rc 0 = applied). Rewriting CAMPAIGN CORRECTIONS for fwd means editing the 3 `_preamble.md` in `OE` -- and it then applies to the bwd job too if it resumes; save the fwd variant as a separate patch under `PT/output/` and swap per job.

---

## 8. Monitoring recipe (attended or unattended)

The rule is to stay silent while things are healthy (user memory `e2e-monitoring-style.md`; gpu-kernel-campaign §7).

**A. Wedge and exit watch.** A `Monitor` polling every 30 s, `timeout_ms` 1800000, re-armed when it expires. It emits only on a problem:
```bash
J=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/<job-dir>
SIG='MES.*failed to respond|failed to suspend all gangs|might be in unrecoverable|wait for reset ack|GPU reset begin|ring gfx timeout'
base=$(sudo -n dmesg 2>/dev/null | grep -cE "$SIG"); warned=0
while true; do
  n=$(sudo -n dmesg 2>/dev/null | grep -cE "$SIG")
  [ "$n" -gt "$base" ] && { echo "WEDGE: $(sudo -n dmesg | grep -E "$SIG" | tail -2 | tr '\n' ' ')"; base=$n; }
  timeout 20 docker exec fa-repro true >/dev/null 2>&1 || echo "WEDGE?: docker exec no answer in 20s"
  pgrep -f '[o]p-evolve (run|resume)' >/dev/null || { echo "LOOP GONE: $(tail -3 $J/job_context/state.yaml | tr '\n' ' ')"; break; }
  k=$(ls /sys/class/kfd/kfd/proc | wc -l)
  [ "$k" -gt 2 ] && [ $warned = 0 ] && { echo "CARD CONTENTION: $k KFD procs (h56)"; warned=1; }
  [ "$k" -le 2 ] && warned=0
  sleep 30
done
```
- Use the `[o]p-evolve` bracket pattern. The old monitors used `pgrep -f 'op-evolve resume --job gfx1250'`, which **matches the monitor's own bash command line**, so it can never report the loop gone (gpu-kernel-campaign §6).
- Compare the dmesg count against a baseline taken when the watch starts, not against 0, because old incidents are still in the ring buffer.
- Read `/sys/class/kfd/kfd/proc`: a directory read is safe on a sick card. `rocm-smi` is not (gfx1250-card-safety §4). More than 2 procs caught round 23's orphaned `meas2` benchmark (HINT h56).

**B. Progress digest.** `CronCreate` every ~15 min at an off-minute, e.g. `"7,22,37,52 * * * *"`. It reports only a new round, an accepted round, or a `finished` event. It reads `state.yaml` (the last row's `round, mode, progress, running, score, gain, accepted` and `best_round`) plus `rounds/NNN/timing.yaml`. Cron jobs expire after 7 days and exist only for the session.

**On a wedge**: stop the GPU loops, leave the driver alone, tell the user (only an AC-cycle recovers the card), and switch to CPU work (user memories `gpu-recovery-needs-the-user.md`, `unattended-work-authorized.md`). After the AC-cycle: `sudo modprobe amdgpu`, `docker start fa-repro`, then the check ladder in gfx1250-card-safety §5.

---

## 9. bwd job: `gfx1250-flydsl-attn-bwd-20260917-115934` state

| field | value | evidence |
|---|---|---|
| shape (prod) | b4 s8192 hq32 hkv8 d128 bf16 causal-BR BSHD; FLOP 5,498,229,227,520 | state.yaml `flops.prod` ✅ |
| spec | v000, max_rounds 40, max_timeout 48h, min_gain 0.0, fast_rounds 24, fast_per_deep 4, beat "…beaten by 0%" | final.yaml:202,222-232 ✅ |
| champion | **round 20** (`best_round: 20`); per-shape champions fast 17, proxy 17, prod 19, `prod_b4…` 20 | state.yaml ✅ |
| prod | baseline 57.2 → r20 **511.4 TF/s**; bar **7.68 ms / 716 TF/s** (h32, `0924__flydsl/DAY-SUMMARY.md:12`) → ~0.71x | ✅ |
| rounds | 1–22 settled; 10, 11, 18, 19 rejected; 15, 21, 22 `failed` (FastError); 17 is the only deep round | state.yaml ✅ |
| **round 23** | **Paused mid-round.** The row shows `running: opt since 08:35:57` with `progress: []`. `rounds/023/1-opt/act.yaml` (09:28) is a **complete measurement of a loser**: `r23.i1.g71`, prod vs_champion 0.9826, proxy 0.9964, fast 0.88, so gain ≈ 0.953 and it would be rejected. The row was never closed, and the loop was gone after the host rebooted at 10:10:51 | state.yaml; timing.yaml; HINT h56 ✅ |
| on resume | The loop moves `1-opt` aside to `1-opt.stale-<stamp>` and **re-runs round 23's opt from scratch** (`core/loop.py:749-757`). A `1-opt.stale-20260924T143001` from an earlier kill already exists. To keep r23's measurement instead, close the round by hand as was done for r3/r8/r9/r11 (see their `note:`). Back up state.yaml first | ⚠ operator choice |
| next deep | round 25 | §3 ✅ |
| gap analysis | gap = 1.386 (structure) × 1.00–1.01. Efficiency axes are closed, see **h54** (HINT:4060) | ✅ |
| gate | split: dk/dv bitwise over 200 runs; dq ≥ 70 dB run-to-run (h54 §5) | ✅ |

Pre-resume checklist (the 09-25 session ran it as a 9-item check):
1. `spec_version` still v000.
2. `diff -r op/current rounds/020/op` is empty.
3. No `.stop` file.
4. No `op-evolve` process and **0 KFD procs**.
5. hint.md is in sync with `HINT`.
6. The patch is still applied (§7).
7. **The forward job is not running.** There is one card and the two jobs must never overlap (h55, fwd NOTES §0).

---

## 10. fwd job: spec and the fixes it still needs

Spec: `PT/output/0925__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml` (staged copy at `OE/jobs/`).
Background is in HINT **h55**: the gap is 1.53x, efficiency is 100%, and determinism does not bind it.
The baseline is *aiter's mature* FlyDSL forward, so expect single-digit-% rounds.

| item | 0925 value | status |
|---|---|---|
| shape (prod) FLOP | 2,199,291,691,008 | ✅ `0924__flydsl/bar-census/fwd_anchor.json` |
| bar (ASM fwd, `fmha_fwd_with_sink_asm`) | 1.5724 ms / 1398.67 TF/s; FlyDSL baseline 2.4005 ms / 916.16 (sclk 1011→989) | ✅ same file |
| `op.precision_gate` | Fixed. It now parses as a string ("REWRITTEN for the forward …"). In the 0923 copy the key was empty and its text had been merged into the `refcache: >` block, so it loaded as `None` | ✅ checked with `yaml.safe_load` |
| `evolve.schedule` | fast_rounds 4, fast_per_deep 4, so the deep rounds are 5, 10, 15, … | ✅ |
| `max_rounds` / `max_timeout` | 30 / 120h. ⚠ The comment still says "= 172800 s" and "12 rounds" (stale, harmless) | ⚠ |
| `min_gain` | **0.007**, settled: the job launched with it (final.yaml:692). NOTES §四.1 argued 0.015, the 0923 copy had 0.005. Change later only by hand-editing final.yaml (§4) | ✅ |
| flydsl | `runtime.python_path` has `~/.local/flydsl0341`. `op_baseline/_env.py` asserts `0.3.4` and `flydsl0341` in `__file__`. ⚠ Header comments (spec lines 59, 74-79, 330, 699) still say 0.3.2 is required | ⚠ prose stale |
| 0.3.4.1 vs 0.3.2 | prod A/B, palindromic, 3+3 runs × 51 iters: fly/asm median **0.6640 vs 0.6636**, no regression; fly-vs-ASM agreement 49.93 dB on both | ✅ `0925__flydsl/fwd341/ab_prod.jsonl` |
| agent model | `claude-opus-5-5` (setup/profiler xhigh, planner max) in both copies; reviewer codex `gpt-5.6-sol` | ✅ |
| `op/ut/common.py` loader | The bwd template hardcodes `getattr(mod, "attn_bwd")` (bwd `op/ut/common.py:104,106`). The spec tells setup to rename it to `attn_fwd` in all 4 impls, with a gate: `grep -rn attn_bwd op/` must be empty. **Check this in the setup report** | ⚠ verify at review |
| FLOP counter | It must be `backward=False`. bwd's `benchmark.py:72-76` has `True`, which would inflate the fwd numbers 2.5x without any visible sign | ⚠ verify at review |
| beat margin | The `beat:` line contains "beaten by 0%" and `beat_margin_pct: 0`. Check whether setup's `validation.py` **reads** the margin or hardcodes it the way bwd does (§4) | ⚠ verify at review |
| free win in hand | `O_VARIANT` v1 is 1.90% faster than the shipped v3 (h55). A good round-1 candidate | ✅ |
| NOTES.md | `0925__flydsl/fwd-job/NOTES.md` is **still the 09-23 text**. It says bwd is "running round 13", uses 0923 paths, and quotes min_gain 0.015. Read it for the reasoning, not for the numbers | ⚠ stale |

Launch order (**done** 2026-09-25 11:46 -> `OE/artifacts/gfx1250-flydsl-attn-fwd-20260925-114644`; step 2 and the hint delivery in `fwd.md` §2 are still open; kept as the checklist for any relaunch):
1. bwd is stopped and 0 KFD procs.
2. Rewrite the patch's CAMPAIGN CORRECTIONS for fwd (§7).
3. Settle `min_gain`.
4. `cp` the 0925 spec to `OE/jobs/`.
5. `setsid nohup … op-evolve run` from `OE` (§2).
6. Arm the watches (§8).
7. Review the two setup reports within 10 min each: attn_fwd rename, `backward=False`, a deliberate-edit proof that the vendored copy is the one imported, `flydsl.__file__`, and the lse SQNR for round 0.
8. Budget ~58 min of setup before round 1.

---

## 11. Traps already paid for

| trap | cost | evidence |
|---|---|---|
| A round leaves a scoring process running, and it contends with the framework's own validation | round 23, ~2 min of contaminated measurement | h56; commit `6029b430` |
| `pkill`/`pgrep -f` self-match | killed the operator's own shell twice | h56 |
| A card wedge mid-round means the round has to be closed by hand | r3, r8, r9, r11 | state.yaml `note:` ✅ |
| A probe that reports success but never ran (`pw.sh` → /dev/null) | a false "throttling ruled out" reached facts.md | h54 §3.3; `0925__flydsl/facts.md` retraction |
| A static ISA metric used as a ranking signal | wrong 4 times | h54 §2–3 |
| The fast-shape mean hid a prod regression | r15 reverted by hand | `state.yaml.bak.before-r15-revert` |
