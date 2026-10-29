# Handoff: one `flydsl_v2` intranode backend for MI455X and MI355X

Work handed over 2026-10-29 (cloud agent bc-f4b4029b, for boyang). Remove this file before the branch becomes
a pull request; the design and its validation stay in `flydsl_deepep_v2_unified.md`.

## State in one paragraph

`flydsl_v2` runs on every GPU and picks its code path from `ArchTraits`: TDM through LDS on gfx1250
(MI455X), a register (VGPR) copy on gfx950 (MI355X) and gfx942 (MI300X). The gfx1250 code is unchanged by
this work (ISA byte-identical). Validated: MI455X with both code paths (unit tests, short benchmark, stress
test), compile-only checks for gfx950 / gfx942, and the gfx942 CI at EP8 on 8× MI300X (wave64 at run time).
Not done: anything on MI355X. The default backend stays `flydsl_v2` on gfx1250 and `cpp` everywhere else
until the MI355X gates below pass.

## Where things are

| What | Where |
| --- | --- |
| Branch | `dev/boyang/flydsl_deepep_v2_mi355` on GitHub (AMD-AGI/Primus-Turbo), head = the commit that adds this file |
| Base | `461d9caa` (V2 without TDM, off gfx1250), on top of PR #583's history before its rewrite |
| Design and validation table | `docs/flydsl_deepep_v2_unified.md` |
| MI355X bring-up notes | `docs/flydsl_deepep_v2_mi355.md` |
| PR #583 (FlyDSL DeepEP on gfx1250) | branch `dev/boyang/flydsl_deepep_gfx1250`, head `826936e5` at handoff, history rewritten on 2026-10-29 |
| Node working copy | `/home/boyan/project/deepep/Primus-Turbo-v2-unified` (local branch `dev/boyang/flydsl_deepep_v2_mi355_wt`, tracks the GitHub branch), container `dev_mi455_deepep_yangbo` |
| Logs | `/home/boyan/project/deepep/ut_logs/v2_unified_*` (each run directory has its script and, for the last ones, a `README.md`) |
| Backup | `/home/boyan/project/deepep/backups/flydsl_v2_unified_20261029/` (git bundle, logs, CI logs, `SHA256SUMS`, `RESTORE.md`) |
| Coordination | `/home/boyan/project/deepep/cursor-work/` (`STATUS.md`, `SYNC_PR583_E2E.md`, `HANDOFF_MI355_V2_UNIFIED.md`) |

## What the branch changes

- `primus_turbo/flydsl/deep_ep/arch.py`: `ArchTraits` per target (wave size, TDM, LDS size, copy piece size,
  loads in flight, watchdog cycles, trap on timeout, system-scope stores). Unknown targets get the gfx942
  values. `PRIMUS_TURBO_DEEPEP_ARCH_TRAITS="field=value,..."` sets the tuning fields only
  (`v2_copy_piece_bytes`, `combine_loads_in_flight`, `spin_timeout_cycles`, `trap_on_timeout`); hardware
  fields are refused.
- `primus_turbo/flydsl/deep_ep/intranode_v2_kernel.py`: one kernel source; `USE_TDM = traits.has_tdm and
  PRIMUS_TURBO_DEEPEP_V2_TDM != "0"`. Without TDM, rows move through registers in pieces of
  `v2_copy_piece_bytes`; wave64 lane math (`_ballot`, `_popc`, `_lanes_below`) is in place.
- `primus_turbo/pytorch/deep_ep/flydsl_intranode.py`: backend choice = the explicit argument, else
  `PRIMUS_TURBO_DEEPEP_INTRANODE_BACKEND`, else `_DEFAULT_BACKENDS` (`{"gfx1250": "flydsl_v2"}`, otherwise
  `cpp`, or `flydsl` without the C++ DeepEP). `flydsl_v2` is accepted everywhere and warns once off
  `_V2_SUPPORTED` (`("gfx1250",)`). Structural fallbacks (internode, low latency, `num_ranks` not in
  {2, 4, 8}, no NVL buffer) take `cpp`; CUDA-graph capture under `flydsl_v2` raises and names `flydsl`.
- `tests/pytorch/deep_ep/test_intranode.py`: the `flydsl_v2` tests run on every GPU; host-only tests for
  the selection table and `ArchTraits`.

Commits from `461d9caa`: `18203443` design, `c35511e4` implementation, `69ef3136` / `b5a70c2e` /
`bfc90ef3` docs, `b133ca92` tuning overrides, `f1eb24bf` merge of PR #583 (`17968eec` 300 s timeout,
`73de5397` buffer sizing), `b00e402b` uneven-token test everywhere, `51d5c39c` + `7effe850` combine fix
(cherry-pick of bc-50a3eeec's `41655cf2`, now `db2c5784` in PR #583), then docs.

## Validation done (details: validation table in `flydsl_deepep_v2_unified.md`)

| Check | Result |
| --- | --- |
| gfx1250 ISA, TDM and register copy, against `461d9caa` | byte-identical (implementation commits) |
| gfx950 / gfx942 compile only, EP2 / EP4 / EP8 | all 9 kernels compile; LDS within `max_lds_bytes`; wave64 code, no TDM |
| MI455X, branch head, GPUs 2,3 | `test_intranode.py` 24 tests per path OK (2 skipped: need the C++ DeepEP); short benchmark outputs checked; TDM times equal to the tree before the merge within 2% |
| gfx942 CI (manual dispatch), 8× MI300X | runs 40720916625 and 40741179998 pass: every `flydsl_v2` test at EP8, including the combine fix's test for all three backends |
| Combine fix | its test passes on gfx1250 (both paths) and gfx942 |

## Open work, in order

1. **Rebase onto the rewritten PR #583** (or onto main once PR #583 is merged). Start a branch from the new
   PR head and cherry-pick, in order: `461d9caa`, `18203443`, `c35511e4`, `69ef3136`, `b5a70c2e`, `bfc90ef3`,
   `b133ca92`, `b00e402b`, then the doc commits. Skip `f1eb24bf`, `17968eec`, `73de5397`, `51d5c39c` and
   `7effe850`: their content is in the PR. Expect conflicts in `buffer.py` (docstring),
   `docs/flydsl_deepep_mi455.md` (rewritten in `826936e5`), `primus_turbo/pytorch/deep_ep/README.md` and
   `test_intranode.py`. Then rerun the compile-only ISA check against the PR head (gfx1250 TDM must stay
   byte-identical), `test_intranode.py` on 2 GPUs with both paths, and the gfx942 CI.
2. **MI355X gates** (the commands are in `cursor-work/HANDOFF_MI355_V2_UNIFIED.md`, steps 1–5; take the code
   from this GitHub branch instead of the old bundle): `test_intranode.py` on 2 and 8 GPUs; the V2 stress test
   (`flydsl_stress.py`, in the backup and in `_sync_to_mi355_v2_unified_20261028_142146/`); `bench_ops.py`
   against `cpp` at EP2 and EP8 with the DeepSeek-V3 shape; tune `v2_copy_piece_bytes` (1 / 2 / 4 KiB) and
   `combine_loads_in_flight` (16 removes the gfx950 combine spills) through
   `PRIMUS_TURBO_DEEPEP_ARCH_TRAITS`.
3. **The MI355X default**: only if `flydsl_v2` is at least on par with `cpp` at the CU counts training uses,
   add `"gfx950": "flydsl_v2"` to `_DEFAULT_BACKENDS` and `"gfx950"` to `_V2_SUPPORTED`, update
   `test_default_per_architecture` and the default tables in the docs, then the DeepSeek-V3 proxy end to end
   at EP8 against `cpp`.
4. Not covered yet: the V2 stress test on gfx9 hardware, EP8 with TDM (MI455X nodes have 4 GPUs).

## How to run (MI455X node, inside `dev_mi455_deepep_yangbo`)

```bash
cd /home/boyan/project/deepep/Primus-Turbo-v2-unified
export PYTHONPATH=$PWD NCCL_CUMEM_ENABLE=0 ROCR_VISIBLE_DEVICES=2,3
# unit tests, TDM, then the register copy (each with its own FlyDSL cache)
FLYDSL_RUNTIME_CACHE_DIR=/tmp/v2_tdm python3 tests/pytorch/deep_ep/test_intranode.py -v
PRIMUS_TURBO_DEEPEP_V2_TDM=0 FLYDSL_RUNTIME_CACHE_DIR=/tmp/v2_reg python3 tests/pytorch/deep_ep/test_intranode.py -v
# short benchmark (DeepSeek-V3 shape, outputs checked)
cd benchmark/ops/deep_ep_v2 && BACKENDS=flydsl_v2 CU_LIST=32,128,256 OUT_JSON=/tmp/bench.json \
  torchrun --nproc_per_node=2 bench_ops.py
```

- Compile-only ISA check, no GPU: `ut_logs/v2_unified_combinefix_isa_20261028_170428/run_isa.sh <tree> <out>`
  (uses `pr583_bench/compile_deepep_kernels.py`, `COMPILE_ONLY=1 ROCR_VISIBLE_DEVICES=""`).
- gfx942 CI: `gh workflow run ci.yaml --repo AMD-AGI/Primus-Turbo --ref <branch>` (ask the user first). A new
  dispatch on the same branch cancels a running one (concurrency group). The `--dist-only` step sees all 8
  GPUs, so the DeepEP tests run at EP8.
- Lint: `pre-commit run --files <files>` inside the container
  (`docker exec -u $(id -u):$(id -g) -e HOME=/tmp/pc_home dev_mi455_deepep_yangbo ...`).
- Push from the node: `git -c credential.helper= -c credential.helper='!gh auth git-credential' push ...`.

## Things to know

- Branch names on GitHub are `<type>/<owner>/<short-description>` (e.g. `dev/boyang/...`), never `cursor/...`.
- GPUs are shared with other windows: claim them in `cursor-work/STATUS.md` first, use GPUs only when the user
  says so (in the daytime only short runs), release and log afterwards.
- `dist.destroy_process_group()` can crash at exit with this ROCm SDK (issue #584, HIP VMM use after free);
  `NCCL_CUMEM_ENABLE=0` avoids it in tests. The benchmarks exit without destroying the group.
- Importing `primus_turbo.pytorch.deep_ep` opens a HIP context: collecting the tests needs a visible GPU.
- The watchdog prints after 30 s on gfx9 and keeps spinning; on gfx1250 it traps after 300 s.
- MI455X benchmark times move by about 10% from day to day with the node's load; compare trees back to back.
