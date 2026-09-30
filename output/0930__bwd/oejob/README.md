# bwd op-evolve job on A0: prepared to resume from s6 (2026-09-30, NOT launched)

Job: `OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934` (A0's own copy, stopped in round 23 on 09-25).
`OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`, `J=$OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934`,
`PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`, `O=$PT/output/0930__bwd/oejob` (this directory).
Host A0 = heliosr-1b114-c07-1, one card, container `fa-repro`. B0 (rounds 24-32, champion r29) is unreachable
(ssh host key changed; not bypassed), so A0 resumes at **round 24** from its own round-23 state.

## What happens on resume

1. Round 24 is **deep** (schedule fast_rounds 23 / fast_per_deep 4 -> deep 24, 29, 34, 39, 44; fast otherwise; max 48).
2. At the head of round 24, before any module, the loop runs hint **h75 (`must refactor`)**: the planner agent copies
   `$PT/output/0930__bwd/armsrc/s6/{_env.py,__init__.py,impl.py,kernels.py}` byte for byte into `rounds/024/op`
   (md5s are in the hint), the framework runs `validation.py` itself (correctness + determinism must pass; speed
   FAIL is expected), then `_promote_refactor` makes `op/current` = s6, `best_round` = 24 and repoints every
   per-shape champion to round 24 (`core/loop.py:_promote_refactor`). This is the framework's own adoption path
   (as B0's h68 did for r19h); `op/current` and `state.yaml` champions were NOT edited by hand.
3. Deep profiling then runs on s6 with the patched prompts: step 4 captures ATT for `k_dkdv` / `k_dqg` at prod.
4. Plan / act / reflect, then fast rounds 25-28 (they read h74-h82 through route.md).

## Changes (every file, backup, reason)

Job dir (`J`; all backups sit next to the file; diffs in `$O/patches/`):

| file | backup | change | reason |
|---|---|---|---|
| whole job dir | `$J.a0-stale-0930/` (rsync -a, 4.0 G, excludes the root-only `core*` dumps) | none | pristine pre-edit copy of the stale A0 job |
| `$J/.pid` | moved to `$J/.pid.stale-0930` | stale PID 169563 (not running) | `tune` warned "a loop is running" from it |
| `job_context/op/benchmark.py` | `.bak.pre-a0-0930` | B0 `bwd_benchmark_sclk.diff` + `bwd_benchmark_blocked.diff` (both applied clean; result byte-identical to B0's ruler copy `0927__b0/ruler/bwd/.../op/benchmark.py`) | blocked timing lead 4 + block 9 (h66); `OE_PHYS_GPU` unset on A0 -> sclk via rocm-smi |
| `job_context/op/validation.py` | `.bak.pre-a0-0930` | (a) B0 `bwd_validation_scored_shapes.diff`: proxy AND prod each >= target, fast reported only (h69); (b) `BEAT_MARGIN 0.0 -> 20.0`; (c) `load_reference` accepts the recorded `common_sha 988c14caed5d9a80` for fast/proxy/prod (`_CACHED_COMMON`, as in `0927__b0/lab-kdq/OP/lab_validate.py`); (d) a cache miss is now a correctness FAIL, never an fp32 recompute | (b) s6 already beats ASM, so at 0% the first passing round ends the job as `target_met`; (c) A0's refcache was never re-stamped: `ut/common.py` changed only the unscored `toy` shape (diffed vs the 0917 copy, whose sha is 988c14ca), so without this the gate recomputes the fp32 reference ON THE CARD; (d) h50/h74 |
| `job_context/op/refcache_util.py` | `.bak.pre-a0-0930` | fallbacks never touch the card: proxy/prod cache miss raises; fast non-causal and the six UT edge shapes compute on the CPU | benchmark.py, validation determinism and ut/test_correctness.py all go through it |
| `job_context/gfx1250-flydsl-attn-bwd_final.yaml` | `.bak.pre-a0-0930` | hand edit: `min_gain 0.0 -> 0.007`, `beat_margin_pct 0 -> 20` (prose key only); `op-evolve tune --fast-rounds 23 --fast-per-deep 4 --max-rounds 48 --max-timeout 120h --beat-margin 1.2` (no spec bump; `spec_version` stays v000) | B0 min_gain; deep round first; beat line "beaten by 20%" feeds the ledger targets (`loop._margin`) in step with validation.py. container `fa-repro` / host `heliosr-1b114-c07-1` / `gpu_id 0` were already A0's values -- verified, no edit needed |
| `job_context/state.yaml` | `.bak.pre-a0-r23-close-0930` | round 23 closed by hand (progress [opt, reflect], `running` removed, measured figures from `rounds/023/1-opt/act.yaml`, accepted false, gain 0.9515, note), plus a `hand_edit` lifecycle event. Same closure as B0's `bwd_state_close_r23.diff`, written with the framework's own dump settings (`safe_dump(width=100, sort_keys=False)`), rest of the file byte-identical | resume starts round 24 instead of redoing r23; best_round stays 20 until h75 runs |
| `job_context/hint.md` | `.bak.pre-a0-0930` (A0's old h1-h56 copy) | replaced by `$O/hint.md` (md5 7508ee98): B0 corpus `0927__b0/bwd-hint.md` (h1-h73) + new h74-h82; 21 stale rows retired (status `superseded ...`; h4/h9/h13/h62/h65 also lose `standing`, so they stop rendering as constraints) | only outstanding items now: h75 (refactor), h82 (advise note); parser check: 44 rows, standing h2 h3 h5 h7 h15 h17 h61 h64 h66 h69 h71-h74 h76-h81 |

New hints (full text in `$O/hint.md`): h74 host move + what this copy is (B0 rounds absent, refcache, no fp32 on card);
h75 refactor to s6 + why the target is 1.20x; h76 s6 kernel map and levers; h77 ablations (A2 -18.5%, abl_l2, A5 +4%);
h78 closed arms (lse_late, ku2, flip, trorder_b, trim2, TDM w/o carried operands, k_dq U2); h79 w4f parked (4.24 ms
noatom, footprint fix -28.5%, best 6.51); h80 rulers (arm.sh flow, k_dqg only at proxy/prod, fresh cache + new dir per
arm, PMC cycles); h81 ATT recipe as primary diagnostic; h82 ranked next targets.

OE repo (`git -C $OE status` showed `op_evolve/` clean before editing, so the patch IS applied in the working tree,
uncommitted, on branch `lhz/gfx1250`): `$O/deep_att_enable.patch` (458 lines; `git -C $OE apply --reverse --check`
passes). Files: the three deep `_preamble.md` (CAMPAIGN CORRECTIONS rewritten; source text `$O/campaign_corrections_bwd.md`),
`profiling/prompts/04_thread_trace.md` (verified ATT CLI recipe replaces the "ATT captures nothing -> skip" section),
`02_counters.md` (stall question answered by ATT; PC-sampling ban kept and restated), `03_metrics.md` (drops the stale
"PC sampling for stall reasons"), `01_select.md` (use the job's benchmark with `--warmup-seconds 0` under `--pmc`;
profile prod because k_dqg only runs there). Kept: every machine-safety trim (no pip install, no container restart,
pinned decoder path, no rocprof-compute, `--kernel-trace` ban). All deep templates render with their placeholders.

## Pre-launch checklist (all read-only except items 4-5)

```bash
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve; J=$OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934
O=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/oejob
```
1. No loop, no other GPU client: `pgrep -af '[o]p-evolve (run|resume)'` empty; `ls /sys/class/kfd/kfd/proc` empty;
   `docker ps` shows only `fa-repro`; no hand-campaign `run.sh`/`arm.sh` (flock `/tmp/a0-gpu0.lock`) or e2e running.
2. Card alive: `timeout 20 docker exec fa-repro true`; `timeout 20 sudo -n dmesg | tail` has no new MES/GCVM lines.
3. Job state: `grep spec_version $J/job_context/state.yaml` = v000; no `$J/.stop`; `md5sum $J/job_context/hint.md` =
   `7508ee980b0d53ce20a2b35ce8fd1f28`; `git -C $OE apply --reverse --check $O/deep_att_enable.patch` rc 0.
4. **Fresh JIT cache in fa-repro.** The job harness sets no `FLYDSL_RUNTIME_CACHE_DIR`, so its processes use the default
   `/root/.flydsl/cache` (1.7 MB now); `/tmp/flycache` does not exist on A0 (B0's convention). Clear both, plus comgr:
   `docker exec fa-repro bash -c 'mv /root/.flydsl/cache /root/.flydsl/cache.pre-bwd-0930; rm -rf /tmp/flycache /root/.cache/comgr/*'`
   (optional: `rm -rf /tmp/flycache_*`, 170 hand-campaign dirs, 906 MB).
5. `sudo sysctl -w kernel.dmesg_restrict=0` (it is **1** on this boot, so agents' dmesg checks would read empty).
6. Optional, no card time: `cd $OE && .venv/bin/python tools/check_llm_account.py --config $J/job_context/gfx1250-flydsl-attn-bwd_final.yaml`
   (reviewer is still codex `gpt-5.6-sol`; it worked in A0 round 17; `~/.op_evolve_openai` is a symlink to `~/.op_evolve_anthropic`).
7. Nothing else will use the card for the next ~4-5 h (round 24 = refactor ~0.5-1 h + deep ~3.5 h).

## Launch (from OE; detached, own process group, log outside /tmp)

```bash
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve
J=gfx1250-flydsl-attn-bwd-20260917-115934
LOG=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/oejob/oe-$J.log
setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve resume --job $J >> "$LOG" 2>&1 < /dev/null &
```
Plain `resume`, never `--config`. Stop with `op-evolve stop --job $J` (or `kill -TERM -<PGID>`). Watch with the
Monitor recipe in `.claude/skills/gfx1250-attn-campaign/references/op-evolve-ops.md` §8. First things to confirm in the
log: `round 24 (deep)`, `0-refactor: executing operator hint h75`, `0-refactor: correct against op/eager`,
`hints: op/current/ is round 24`, and in `validation_attempt*.log` / gate output `refcache prod: accepted with recorded
common_sha` (never `IGNORED`); `md5sum $J/job_context/op/current/kernels.py` = `5e61678d53260a2fc17c68286298f4b9` after it.

## Decisions the operator may want to revisit

- **Target 1.20x beat** (validation.py `BEAT_MARGIN = 20.0` + final.yaml "beaten by 20%"; change both together, the
  latter with `op-evolve tune --beat-margin X`). Any value <= ~1.03 ends the job at the first passing round.
- `max_rounds 48`, `max_timeout 120h` (elapsed so far 15.5 h).
- `job_context/profiling/beat/` (the ASM beat profile, taken once per job in round 17 when ATT failed) is kept; ASM ATT
  reads already exist in `0930__bwd/probe/p2_asm/`. Move it aside to have round 24 re-profile the beat (costs one more
  profiling pass).
- `findings/*.md` are A0's round-23 state (pre-B0, pre-s6); h74 tells rounds that hint.md wins where they disagree.
- The OE preambles are shared by all jobs: the new CAMPAIGN CORRECTIONS block is bwd-specific. A0's fwd job is retired;
  swap in a fwd variant before any fwd job's deep round.
- Not patched: the fast-loop opt preamble still names `rocprofv3 --stats` as its default survey (0 dispatches on this
  stack); h80 tells fast rounds to use the PMC csv instead.

## Rollback

Per file: copy the `.bak.pre-a0-0930` / `.bak.pre-a0-r23-close-0930` backup over it; OE:
`git -C $OE apply --reverse $O/deep_att_enable.patch`. Whole job: `rsync -a --delete --exclude='core*' $J.a0-stale-0930/ $J/`.
