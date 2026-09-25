# Environment, card safety, measurement discipline, FlyDSL traps (gfx1250 / MI455X)

Scope: everything a new session needs to *run anything at all* on this box without
losing a day or a power cycle. Legend: ✅ verified on disk / by measurement,
⚠ conditional or partly stale, ❌ dead / do not do. Every row cites its evidence.

Paths below abbreviate:
`PT` = `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`,
`HINT` = `PT/output/0923__flydsl/hint.md` (h-ids = `## hNN` headers; `h54` at ~line 4060 is the one-page index),
`BWD` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context`,
`FWDSPEC` = `PT/output/0923__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml` (the **superseded 09-23** spec; runtime/safety rows still hold, but for the live fwd job use `PT/output/0925__flydsl/fwd-job/` and `fwd.md` §2).

Companion user skills (link, do not duplicate):
`~/.claude/skills/gfx1250-card-safety` (wedge catalogue, recovery ladder),
`~/.claude/skills/flydsl-gfx1250` (corpus map, wave-size trap, version split),
`~/.claude/skills/gpu-kernel-campaign` (generic measurement discipline). Stale spots are listed in §10.

---

## 1. Machine and container

| item | value | evidence |
|---|---|---|
| host | `heliosr-1b114-c07-1`, **one** gfx1250 card (gpu_id 0, `gpu_pool: []`) | ✅ FWDSPEC `runtime` block |
| container | `fa-repro`, image `fa-tune:deps`, mounts `/home/lihuzhan:/home/lihuzhan` only | ✅ `docker ps` / `docker inspect fa-repro` |
| python | `/opt/venv/bin/python3` (3.12). **No `/opt/rocm` in the container**; ROCm comes from wheels: `ROCM_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel` | ✅ `docker inspect` Env |
| torch / rocm | `torch 2.11.0+rocm7.14.0a20260625`, `rocm-sdk-libraries-gfx1250 7.14.0a20260625`, triton 3.6.0 | ✅ `pip list` in fa-repro |
| rocm-smi | `/opt/venv/bin/rocm-smi` inside the container | ✅ FWDSPEC `observed.rocm_smi` |
| ownership | `owned: false` -- **never** recreate / restart / `pip install` into fa-repro; installs go to `--target` dirs under `$HOME` | ✅ FWDSPEC `runtime.runner.docker` |
| aiter | **not installed**; source checkout at `/home/lihuzhan/code/aiter-src`, put on `sys.path` | ✅ FWDSPEC `python_path`; `BWD/op/current/_env.py` |
| primus_turbo | image shipped an *editable* install whose `.pth` MetaPathFinder beats `PYTHONPATH`; it has been `pip uninstall -y primus_turbo`'d in fa-repro (absent from `pip list` today). **Never `import primus_turbo`** in a job process: its FlyDSL tree imports `flydsl.expr.buffer_ops`, removed in 0.3.x | ✅ pip list; `BWD/op/current/_env.py` docstring; flydsl-gfx1250 skill §4 |
| sudo | `sudo -n true` works on host; job policy is `sudo: false` anyway | ✅ FWDSPEC `sudo:` comment |

### 1a. Three flydsl versions -- pick by `sys.path` prepend, assert by `__file__`

| version | where | who uses it | evidence |
|---|---|---|---|
| 0.2.4 | image `/opt/venv/lib/python3.12/site-packages/flydsl` (what `import flydsl` gets with no prepend) | Primus-Turbo's own gfx950 FlyDSL tree only | ✅ `flydsl.__version__` in fa-repro |
| 0.3.2 | `~/.local/flydsl032` (`pip --target`) | **bwd job** champion r20 (`BWD/op/current/_env.py` asserts `0.3.2` and `"flydsl032" in __file__`) | ✅ dist-info on disk |
| 0.3.4.1 | `~/.local/flydsl0341` (`pip --target`, installed 2026-09-25) | fwd work in `PT/output/0925__flydsl/fwd341/op0341` (asserts `0.3.4` + `"flydsl0341"`) | ✅ dist-info; `fwd341/op0341/_env.py` |

- ✅ 0.3.2 vs 0.3.4.1 on the aiter fwd, same session, palindromic `op032 op0341 op0341 op032 op032 op0341`, prod: fly 2.3378/2.3434/2.3394 ms vs 2.3346/2.3339/2.3324 ms, `agree_db` identical 49.93 -- **no regression, ~0.2% (inside floor)**. `fwd341/ab_prod.log`.
- ✅ bwd tree also staged for 0.3.4.1: `PT/output/0925__flydsl/bwd341/{op032,op0341,op_clean}` (not yet measured -- check before assuming).
- ⚠ Any `pip install --target` must be audited for packages the container already has: a matplotlib install pulled numpy 2.5.3 over torch's 2.4.1 (h25).
- Rule: `_env.py` must `sys.path.insert(0, <flydsl dir>)` **before** `import flydsl`, then assert version **and** `__file__`. A plain site-packages flydsl is shadowable; an editable one is not (flydsl-gfx1250 §4).

### 1b. BLAS env -- ASSIGN, never `setdefault`

| var | value | why | evidence |
|---|---|---|---|
| `HIPBLASLT_TENSILE_LIBPATH` | `/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250` (328 files, host ROCm 10.1.0 copy) | hipBLASLt's default search path lacks the payload -> `HIPBLAS_STATUS_INVALID_VALUE`; `_rocm_sdk_devel/.../library/gfx1250` is incomplete -> **SIGSEGV 139**; `/opt/rocm/lib/hipblaslt` is an empty decoy | ✅ `PT/output/0923__flydsl/STAGE2-S0-PROBE.md` S0-b; FWDSPEC `runtime.env` |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `1` | `/usr/lib/python3.12/sitecustomize.py` does `setdefault(...,"0")` at every interpreter start, so `_env.py`'s own `setdefault` was a no-op for 12 rounds | ✅ file read in fa-repro; `PT/output/0924__flydsl/REFCACHE-PREMISE-GONE.md` |

- ⚠ Both `BWD/op/current/_env.py` and `fwd341/op0341/_env.py` still contain `os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")` -- harmless only because the scored path does no torch GEMM (refcache, §3). Any script that does a torch matmul/reference must set both vars explicitly (`docker exec -e ...` or assignment) and print them from inside the process.
- ❌ `runtime.env:` in an op-evolve spec is a **silent no-op**: op-evolve's docker runner passes no `-e` and `load_spec` drops unknown keys (h25; `op-evolve/op_evolve/runners/docker.py` has no env pass-through). Set env in the op's own code or the exec line.
- `docker exec ... bash -c` is a non-login shell: `/etc/profile.d/*` never runs (FWDSPEC comment).

### 1c. Host driver and kernel log

| fact | evidence |
|---|---|
| `amdgpu` is **blacklisted on the kernel cmdline** (`modprobe.blacklist=amdgpu`); after every boot: `sudo modprobe amdgpu; sleep 6; ls /sys/class/kfd/kfd/proc/ (empty); docker start fa-repro` | ✅ `/proc/cmdline`; gfx1250-card-safety §5 |
| ❌ `modprobe -r amdgpu` on a wedged card took the **whole box off SSH ~2 h** | gfx1250-card-safety §1 #7 |
| `kernel.dmesg_restrict` defaults to **1** (not overridden in `/etc/sysctl.d/10-kernel-hardening.conf`) -> `dmesg` returns one line and every "no fault" grep reads empty output. Fix: `sudo sysctl -w kernel.dmesg_restrict=0`, then confirm `dmesg \| wc -l` is in the thousands. It is 0 on the current boot (someone set it) -- recheck after every reboot | ✅ `/proc/sys/kernel/dmesg_restrict`, sysctl.d; h22 |
| The ring buffer dies with a power cycle; evidence survives in `/var/log/kern.log` (~5 GB: `tail -n 400000`, never grep whole). `journalctl -k -b -1` did **not** have it | h22 |
| sclk: **1100 MHz is idle**; prod windows run ~998-1049 MHz (VR throttle). 2026-09-25 fwd A/B witnessed 1038-1049 | ✅ h39; `fwd341/ab_prod.log` |

---

## 2. Card safety -- wedge classes and what a new session must never do

A wedge = a human AC-cycle (single operator, often away). Budget in **startups**, not minutes
(memory `gpu-recovery-needs-the-user`; gfx1250-card-safety §0).

### 2a. The three classified wedge signatures

| class | first dmesg line(s) | discriminants | seen | evidence |
|---|---|---|---|---|
| **A** memory aperture | Tensile `Cijk_*` kernel -> `GCVM_L2_PROTECTION_FAULT`, sub-4 GB truncated address, `PERMISSION_FAULTS: 0x5 RW: 0x1`, `copy_context_work_handler hogged CPU` early | fp32 Tensile reference path; bypassed by LIBPATH fix + refcache | bwd round 3 | `PT/output/0923__flydsl/wedge4/ANALYSIS.md` |
| **B** TLB / queue | `MES failed to respond to msg=INVALIDATE_TLBS` with **no preceding fault** -> `failed to suspend all gangs` -> `Failed to detect hung queues` -> `GPU reset begin` | no memory fault at all; ~5 s window between `suspend all gangs` and unrecoverable | bwd round 8, round 11 `d_s4` (2026-09-23 09:21) | wedge4/ANALYSIS.md; h22 |
| **C** (h22 "third class") candidate OOB read storm | `GC_UTCL2` fault storm, `Faulty UTCL2 client ID: TCP`, `RW: 0x0` (read), `PERMISSION_FAULTS: 0x3`, high address, `MORE_FAULTS: 0x1`, several XCDs, `IH ring buffer overflow` -> MES unrecoverable | a candidate kernel's own load past a buffer end | round 14 scratch arms (2026-09-23 13:35) | h22 |
| (not a wedge) per-process queue reset | `GCVM_L2_PROTECTION_FAULT ... PERMISSION_FAULTS 0x5 RW 0x1` then `Queues reset on process ...`, no `GPU reset begin` | process dies, card fine -- verify with KFD empty + one 4096^3 GEMM | 2026-09-17 | gfx1250-card-safety §2a |

Unrecoverable markers: `wait for reset ack`, `ring gfx timeout`, `GPU reset begin`,
`MES might be in unrecoverable state`. `MES ... ring buffer is full` alone is routine backpressure.

### 2b. Rules

| rule | status | evidence |
|---|---|---|
| **One shape per process.** `--shapes fast,proxy,prod` in one process faulted; same binary split per shape rc=0 x3. Measured ~44%/process fault rate before the fix. `validation.py` now runs one process per shape and aborts on first rc!=0 | ✅ | h31; `PT/output/0924__flydsl/wedge-rootcause/one-shape-per-process.patch` |
| **refcache**: the fp32 reference is precomputed (`BWD/op/refcache/{fast,proxy,prod}.pt`, prod 1.08 GB) and loaded via `BWD/op/refcache_util.py`, kept **outside** the sha256-hashed `ut/common.py` / `eager/impl.py` | ✅ | `BWD/op/refcache_util.py:1-15` |
| ⚠ refcache's original premise ("prod reference faults the card") is **gone**: with LIBPATH set, prod reference ran clean 1.77 s, bitwise stable across processes. Keep refcache anyway -- intermittent non-fatal page faults at small addresses still occur | ⚠ | `REFCACHE-PREMISE-GONE.md`; FWDSPEC `op.refcache` |
| **Spill -> hang.** `private_segment_fixed_size > 0` or any `scratch_` op is a **kill, not a cost**: a spilling build hangs after first launch | ✅ | h3 |
| Screen every candidate `COMPILE_ONLY` first (§6); bounds-check every new index expression (block size / split count / grid mapping rewrites address arithmetic -> class C) | ✅ | h22 consequence 1 |
| New kernel: toy shape first, own process, `AMD_SERIALIZE_KERNEL=3`, no in-process autotune | ✅ | gfx1250-card-safety §2 |
| `poison_allocator` does not cover prod (largest poison block 64 MiB vs 256 MiB request); fixed version ships as `BWD/op/poison_util.py` | ⚠ | h22 consequence 2; `BWD/op/` listing |
| Never time a path known to write OOB "as a floor" | ❌ | gfx1250-card-safety §1 #9a |
| Liveness probe that needs no root and no GPU call: `timeout 20 docker exec fa-repro true` -- stops answering first when the card wedges | ✅ | h22 end |
| On a wedged card `rocm-smi`, `ps -eo ...wchan`, `pgrep`, `torch.cuda.device_count()`, `docker stop` all hang; only `timeout 20 sudo -n dmesg \| tail` and bounded `/sys` reads are safe | ✅ | gfx1250-card-safety §4 |
| On a wedge: stop all GPU loops, notify user with dmesg evidence, switch to CPU work; do not wait, do not probe again | ✅ | memories `unattended-work-authorized`, `gpu-recovery-needs-the-user` |
| After a fault, look for a fresh `core.gpu` in `BWD/op/` **before** the power cycle (stale ones renamed to `core.gpu.stale-20260921`, `core.host.val3-20260924`) | ✅ | h31; `BWD/op/` listing |

### 2c. Profiler rules

| tool | status | evidence |
|---|---|---|
| rocprofv3 **PC sampling** | ❌ never -- wedged MES 2026-09-11, 3 attempts 3 faults | FWDSPEC safety block; h4 |
| rocprofv3 `--pmc` | ✅ works, ~2 min per counter group; 51 counters defined, only 9 trustworthy (h20 whitelist); `SQ_VALU_WMMA_FLOP_*` read 0 | STAGE2-S0-PROBE.md S0-c; h20 |
| `--kernel-trace` / `--runtime-trace` / `--hip-trace` | ❌ produce a .db with **0 dispatch rows** (458 symbols registered) -- silently analyses nothing | S0-c |
| rocprofv3 VGPR column | ⚠ **half** the ISA count; read descriptors / `21_final_isa.s`, not derived percentages | FWDSPEC safety block |
| ATT (thread trace) | ⚠ safe (4 runs, zero faults) but **captures nothing for FlyDSL JIT kernels** (only `*_code_object_id_*.out`); `--att-buffer-size` is bytes (`64` aborts and looks like a hang) | h25 |
| rocprof-compute | ❌ not used: gfx1250 panels map to near-empty counters, and it overrides the counter table | h25 |
| static ISA metrics as a ranking signal | ❌ wrong four times; use them to screen (spill/VGPR/LDS), not to rank | h26, h54 §2 |

---

## 3. Measurement discipline

| rule | evidence |
|---|---|
| **Same-session only.** Compare arms inside one sweep; cross-session drift ~1.5% on the fwd | h38-CORRECTION; `fwd-job/NOTES.md` §4.1 |
| **This operator's floor is 0.24-0.66%** same-session same-code (r19 0.61, r20 0.66, r22 0.40, r23 0.24). The 1.57% "floor" is a HipKittens GEMM ladder number -- do not use it here | h38-CORRECTION; h56; h54 trap 2 |
| **Palindromic arm order** (ABBA / ABBAAB), so drift and droop hit every arm equally; A/B *deltas* survive the 9% clock droop, absolute TF/s do not | h39; `fwd341/ab_prod.sh` |
| **Burn-in arm, discarded**, before the timed arms (e.g. `timed(asm, 5)` in `fwd341/fwdab.py`), plus a warmup window; fwd spec protocol is 101 reps with 8 s warmup | `fwdab.py`; FWDSPEC safety block |
| 256 MiB L2 flush per rep, CUDA events, median | `fwdab.py`; flydsl-gfx1250 §8 |
| **sclk witness** at start/end of every timed block (`rocm-smi --showclocks`), recorded with the number | h39; `fwdab.py sclk()` |
| **Assert the subject ran.** `pw.sh` sent its subject to `/dev/null`, returned `wait`'s rc, produced 60 idle rows and a false "throttling ruled out". Echo `RC_<shape>` per shape; never take `wait`'s rc | h39; h54 trap 3 |
| **Correctness before speed**, all tensors, NaN-prefilled outputs, `isfinite` coverage. fwd: `o` and `lse` separately >= 50 dB; bwd: split gate (dk/dv bitwise x200, dq >= 70 dB run-to-run) | FWDSPEC `op.precision`; h54 §5 |
| **`rocm-smi --showpids` before trusting any number** -- must list exactly the expected PIDs. Round 23 left an abandoned `meas2` session contending with the framework's own validation | h56 |
| **Kill by explicit PID, never `pkill -f`** (self-matches the killer's shell; did so twice). Check `ps -eo pid,ppid,args \| awk '$2==1'` for orphans | h56; gfx1250-card-safety §6 |
| A round owns the card and must leave it idle; kill an abandoned session in the same step that abandons it | h56 |
| A delegated "CPU-only" agent will take the GPU unless forbidden; check KFD holders before/after (`ls /sys/class/kfd/kfd/proc/`) | gfx1250-card-safety §1 #9 |
| A clean run count is not evidence unless the base rate says it should have failed (6 clean at 44%/process fault rate: P=0.49) | h31; h54 trap 4 |
| Contention can fake an accuracy failure (42-48 dB vs 53.7 clean) -- re-run failures in an exclusive window | gfx1250-card-safety §7 |
| A number from another kernel/arch/context is not a property of this mechanism (misled 5x) | h54 trap 2 |
| The bwd job and the fwd job **cannot run concurrently** (one card) | h55; FWDSPEC `gpu_pool` |

---

## 4. Git and power cuts

- ✅ 2026-09-16 cycle: 6 loose objects truncated to 0 bytes, branch ref on one of them, `fatal: bad object HEAD` (gfx1250-card-safety §5).
- ✅ 2026-09-22 power loss: **255** objects truncated, commit `70efc407` lost (memory `push-after-every-round`).
- Recovery: quarantine zero-byte objects -> `git update-ref <branch> <last pushed>` -> `rm .git/index && git reset` -> `git fsck --no-dangling`. Check after **every** AC-cycle.
- Rule: push at the end of every round. Unpushed = one power cut from gone.
- op-evolve changes live in someone else's repo; save them as patches under `PT/output/...` (e.g. `PT/output/0924__flydsl/deep-enablement/deep_loop-trim.patch`, h25) so a `git checkout` there cannot erase them.

---

## 5. JIT cache

| fact | status | evidence |
|---|---|---|
| Default disk cache `~/.flydsl/cache` -> `/root/.flydsl/cache` inside fa-repro (root); `FLYDSL_RUNTIME_CACHE_DIR` overrides; `FLYDSL_RUNTIME_ENABLE_CACHE=0` disables disk layer (in-memory stays) | ✅ | `~/.local/flydsl0341/flydsl/utils/env.py:295`; FlyDSL `CLAUDE.md` env table |
| comgr has its own cache `~/.cache/comgr` | ✅ | bwd round 18/19 `act.yaml` notes (`rm -rf /root/.flydsl/* ~/.cache/comgr/*`) |
| **Cache key can miss changes**: module-level helper-class methods, env-gated `const_expr(env)` branches, C++ passes, helpers outside the traced closure. A stale hit returns the *old correct* SQNR -- the most dangerous false negative | ⚠ recorded on another FlyDSL project (gfx950 / older flydsl); 0.3.x now tracks globals of same-dir user functions (`compiler/jit_function.py:_discover_global_refs`) but not everything | `/tmp/kyle_opt/optimizer/pitfalls/07-flydsl-frontend-tracer.md` (ephemeral /tmp copy) |
| Practice: **one cache dir per arm** (`-e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<arm>`), or `rm -rf /root/.flydsl/* ~/.cache/comgr/*` + arm `__pycache__` before a round's first build | ✅ used | `fwd341/ab_prod.sh`; bwd round 18/19 notes |
| Gold check for staleness: break the kernel on purpose (e.g. scale x2) without clearing; if SQNR stays good the cache is stale. Two arms that should differ giving byte-identical time/VGPR = key miss | ⚠ | pitfalls/07 |
| `@flyc.jit` re-derives its key every launch: flat **0.266 ms CPU per call** (51% of `fast` latency) -- cache the compiled launcher (bwd `r5.i1.g17`) | ✅ | `BWD/op/current/impl.py` comment (r5.i1.g17) |
| Arm identity = directory copy; knob screening must fail loudly on a knob that is not a module-level constant, and derived constants (`NKT` from `KV_STEP`) must follow | ✅ | `BWD/../rounds/001/_scratch/arm*` are whole-dir copies; knob-screen error text in session transcripts |

---

## 6. COMPILE_ONLY recipe (zero card time)

```bash
docker exec -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
  -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<name> [-e FLYDSL_DUMP_IR=1 -e FLYDSL_DUMP_DIR=<d>] \
  fa-repro bash -c "/opt/venv/bin/python3 <screen.py> --impl <dir> --dump-dir <d> --json <j>"
```

- **Both** `ARCH` (compile backend) and `FLYDSL_GPU_ARCH` (buffer-descriptor path via `get_rocm_arch()`) -- one alone gives a mixed-target ISA that compiles silently. Assert the resolved arch == `gfx1250`: with no device and no env, it silently answers **gfx942** (flydsl-gfx1250 §3a; `PT/output/0921__flydsl/bin/compile_only_driver.py` docstring).
- `ARCH` is a common env name -- an outer script setting it retargets the compiler.
- Launch wrappers take `stream: fx.Stream`; `None` is rejected, so screens declare their own stream-free `@flyc.jit` wrappers; torch CPU tensors bind fine (compile_only_driver.py docstring).
- ~**4 min per build** on CPU (bwd kernels). Harnesses: `BWD/../rounds/002/_scratch/screen.py` (bwd; hardcodes `block=(32,1,1)` for dkdv, h54 §4), `PT/output/0921__flydsl/bin/compile_only_driver.py`.
- Read `<kernel>/NN_final_isa.s` (stage number varies with the pass count: `21_` for the bwd on 0.3.2, `22_` for the fwd on 0.3.4.1 -- glob `*_final_isa.s`): `.vgpr_count`, `.vgpr_spill_count`, `private_segment_fixed_size`, LDS, WMMA count. WMMA count prices density only; it is blind to trip-count changes (h3).

---

## 7. FlyDSL API traps on gfx1250

| trap | fix | evidence |
|---|---|---|
| `rocdl.s_waitcnt` **raises** on gfx1250 (split counters) | bare `fx.barrier()`; backend derives the dscnt wait | h43 (`universal.py:39-55` message) |
| `BufferAtomicAdd` compiles but emits **SCOPE_CU** -> silent lost updates across 8 XCDs | `fx.UniversalAtomicAdd(fx.Float32, rocdl.SyncScope.Agent)` on a plain global pointer -> `global_atomic_add_f32 ... scope:SCOPE_DEV` (`SyncScope` in `flydsl/expr/rocdl/enum.py:17-24`, both 0.3.2 and 0.3.4.1) | h42; h54 §4 |
| Atomics through a descriptor with flat 1 GiB `num_records` land clamped OOB adds on a *live* element -- deterministic, invisible to both gates | give real extents (g07) and predicate every atomic | `PT/output/0925__flydsl/AITER-5GEMM-STUDY.md:183` |
| `fx.barrier()` = `s_barrier_signal -1` + `s_barrier_wait -1` back to back | split it: `from flydsl._mlir.dialects import rocdl as _mrocdl; _mrocdl.s_barrier_signal(-1); ...; _mrocdl.s_barrier_wait(-1)` (not in `fx.rocdl` exports; precedent `expr/rocdl/cluster.py:35`) | h52 (note h53: widening the gap bought nothing) |
| TDM: FlyDSL's own `tdm_ops` cannot take a Fly shared view | use aiter's `aiter-src/aiter/ops/flydsl/kernels/tdm_ops_gfx1250.py`; pass `pad_interval` / `pad_amount` explicitly, else 256 B row stride = 64-way bank conflict (champion uses `X_ROW_B = D*2+16 = 272`) | h48; h54 §4 |
| TDM takes a raw pointer -- buffer `num_records` bounds are **dropped** | TDM address arithmetic must be bounds-proved on CPU | `PT/output/0925__flydsl/tdm-plan/PLAN.md:166,247` |
| TDM gather without `addr64` carry-safe update hangs at large sizes | ❌ dead end | gfx1250-card-safety §1 #12 |
| V# `num_records` is in **units of 128 B** (`>> 7`); byte counts not a multiple of 128 are truncated down | keep byte counts 128-aligned | h33 "NEW ISA FACT" (~line 2611) |
| int32 `num_records` arithmetic can wrap to **0** at prod (`nsp*B*Sq*Hq*D*4`) -> every write discarded | compute in i64 | `PT/output/0924__flydsl/DAY-SUMMARY.md:110-118` |
| Never give `k_dkdv` flat (1<<30) descriptors -- its unclamped prefetch reads up to 982,272 B past Q/dO | ❌ | DAY-SUMMARY.md:117-118 |
| Don't "fix" `k_dq`'s fake descriptors inside an optimisation round (no `B_` kernarg -> 1/4 size, silent loss of `bat>=1`) | infra round only | h31 |
| `is_rdna_arch()` still misses gfx1250 in 0.3.2 **and** 0.3.4.1 (`runtime/device.py:82-99`) -> buffer descriptors lack bit 24 / `OOB_SELECT` (`expr/rocdl/universal.py:254-260`). Wave size is **fixed** in 0.3.x (`get_warp_size` matches `gfx12*` -> 32, `device.py:102-114`) | don't rely on descriptor OOB semantics; use explicit predicates (h2) | ✅ source read today |
| aiter fwd buffer managers **raise `NotImplementedError` when `num_waves != 8`** (`fmha_b16_buffer_managers.py:1016,1167,1253,1589,1749`; `_DEFAULT_NUM_WAVES = 8` at :66) | record as a roadblock, **never** as "4 waves is slower" | ✅ vendored copy `fwd341/op0341/flydsl_fwd/`; h55 |
| A literal Python `if` becomes `scf.if`; `for v in vec` can hang the tracer; `const_expr(lane==0)` is wrong (lane is a runtime SSA value) | runtime `if` / `.select` | pitfalls/07; `PT/agent/skills/kernel-optimize/knowledge/backend/flydsl/programming-model.md:84-88` |
| `sched_barrier(0)` is a scheduling **boundary**, not a clamp -- fencing a clump loosens it | ❌ 0 wins / 5 losses | h38; h54 §2 |

---

## 8. Shapes and constants you will need

| item | value | evidence |
|---|---|---|
| prod shape | b4 s8192 hq32 hkv8 d128 bf16, causal (block-causal), BSHD | FWDSPEC; `fwdab.py SHAPES` |
| sentinel shapes (fwd harness) | fast (1,1024,8,2), proxy (1,4096,32,8) | `fwd341/fwdab.py` |
| fwd FLOP | `2*b*hq*d*2*s(s+1)/2` = **2.199292e12** at prod (STAGE2-FWD-SWEEP used ~2.233e12 -> 947.8 vs 916 TF/s for the same time; quote **ms**) | `fwdab.py`; FWDSPEC note |
| bwd FLOP | 5.498229e12 | bwd benchmark (see `references/baselines.md`) |
| fwd gap | aiter FlyDSL 2.4005 ms vs aiter ASM 1.5724 ms, 1.5266x, same session 2026-09-24; 2026-09-25 re-measure 2.33 vs 1.55 ms | h55; `fwd341/ab_prod.log` |
| aiter ASM fwd entry | `aiter.ops.mha.fmha_fwd_with_sink_asm(q,k,v,scale,True,True)` | `fwdab.py` |
| aiter FlyDSL fwd LSE | natural log, not log2 | flydsl-gfx1250 §8 |

---

## 9. op-evolve spec / scheduling traps

⚠ The first three rows describe the **superseded 0923 FWDSPEC**. The 0925 spec that the live job
loaded has `precision_gate` fixed, `min_gain 0.007`, `fast_rounds 4 / fast_per_deep 4 / max_rounds 30` (`fwd.md` §2).

| trap | evidence |
|---|---|
| FWDSPEC `op.precision_gate` loads as **None** (its text got merged into the `refcache: >` block scalar) -- fix before launch | ✅ yaml.safe_load today |
| FWDSPEC `evolve.min_gain: 0.005` vs NOTES.md's argued `0.015`; `min_gain` is **not** a `tune` knob -- hand-edit the resolved yaml | ✅ yaml load; `fwd-job/NOTES.md:94-96,152,182-196` |
| FWDSPEC schedule is `fast_rounds 6 / fast_per_deep 3 / max_rounds 12` as written | ✅ yaml load |
| FWDSPEC `determinism_gate` still says split-k/atomics fail the gate -- unenforceable (no detector) and superseded for bwd | h55 |
| Deep rounds: stock `01_select` mandated `--kernel-trace` (0 rows); the trimmed deep_loop = `PT/output/0924__flydsl/deep-enablement/deep_loop-trim.patch`, **committed in op-evolve as `58b2134` (2026-09-25 11:38)** -- its preambles still carry the bwd-only CAMPAIGN CORRECTIONS block despite the commit message | h25; `git -C OE apply --reverse --check <patch>` passes |
| `auto_approve_after: 10m` -- nothing may block on the operator | FWDSPEC `review` |

---

## 10. Where the user skills are stale (as of 2026-09-25)

flydsl-gfx1250 (§1 bwd row, §3, §4, §6, §8) and gpu-kernel-campaign §11 now carry dated *Update/Correction 2026-09-25*
notes pointing here; the original text was kept. gfx1250-card-safety and HINT h4 were not edited.

| skill / section | stale claim | current truth |
|---|---|---|
| flydsl-gfx1250 §3, gfx1250-card-safety §1 #10 | wave64 build under wave32 dispatch via `is_rdna_arch` (0.2.4 line refs `device.py:76`) | 0.3.2/0.3.4.1 `get_warp_size` returns 32 for `gfx12*`; only the buffer-descriptor flags still use `is_rdna_arch` (§7) |
| flydsl-gfx1250 §4 | "aiter needs 0.3.2" | 0.3.4.1 also builds/runs aiter fwd, perf-neutral (§1a) |
| flydsl-gfx1250 §6 | "no FlyDSL cache env var", "cold compile never measured" | `FLYDSL_RUNTIME_CACHE_DIR` exists; COMPILE_ONLY build ~4 min (§5-6) |
| flydsl-gfx1250 §8 | "one card VR-throttled to 1100 MHz" | 1100 is idle; loaded windows ~1000-1050 (h39) |
| HINT h4 | "rocprofv3 dead end", "never schedule a deep round" | `--pmc` works, deep rounds enabled since round 17 (h25, S0-c); PC sampling ban stands |
| gfx1250-card-safety | no wedge-class taxonomy, no one-shape-per-process, no `dmesg_restrict`, no `docker exec true` probe | §1c, §2 here |
| memory `unattended-work-authorized` + h54 | -- still current | -- |

---

## 11. Working with the operator (from user corrections in the 09-15 .. 09-25 transcripts)

Sessions `b596bddb` (09-15), `c4b79aa6` (09-17/21), `a9a96fef` (09-21..25). Each row is something the
user had to say, usually more than once.

| lesson | what the user said / what happened |
|---|---|
| **Do not stop, do not ask.** After each round, pick the next step from your own analysis and start it; when unsure take your own recommendation. Stopping to ask was the single most repeated complaint ("不要停不要停不要停", "不要停下来问我", 09-17, 09-21, 09-23, 09-25) | memory `unattended-work-authorized`; op-evolve was adopted (09-17) precisely because sessions kept stopping early |
| **Keep the op-evolve loop running while doing side research.** 09-25 08:27: "为啥没有round信息了？可以继续推进op-evolve的后续轮次" -- side studies (TDM, barrier gap) had left the loop idle | check `ps`/state.yaml `running:` whenever you start a side stream |
| **Verify stream status before reporting it.** 09-23: "另外两路不是已经结束了吗？确认下" -- a stale "still running" was reported | read the process table / result file, not your memory |
| **Say where every bar number comes from.** 09-24: "bar 711.22 这个数据是哪里来的？" and the 10.160 vs 7.68 dispute | cite a file + n + date for any bar; `baselines.md` §0 |
| **A human AC cycle, not your driver reload, recovered the card.** 09-15: "不是你通过驱动重载解决的，好好记录下本次你是怎么把机器搞挂的" | never claim a recovery you did not do; write the wedge cause down |
| **High-risk work last.** e2e training runs repeatedly wedged the card (09-15); the 09-17 plan put e2e after an op-level win. When e2e runs, check it every ~3 min instead of blocking on it | gfx1250-card-safety §1 #2 (compile in e2e) |
| **"No GPU" in a request means no GPU for every sub-agent too.** 09-24 10:27 asked for a GPU-free search so round 18 was not disturbed | pass the prohibition into each delegated prompt (§3 row "delegated CPU-only agent") |
| **Other users' containers may hold the card.** 09-24 08:34 `jolly_easley` (user linxwang) was on the GPU; the user said: containers of linxwang may be stopped directly. Anyone else's: ask | `docker ps` + KFD PIDs before trusting numbers |
| **Hand-over protocol.** When the user says the node goes to someone else: finish/stop rounds (`op-evolve stop`), write the day summary, push, `docker stop fa-repro`. After such a hand-over, a wedge / reboot / stopped container means **do not auto-restart** anything (09-21 15:42) | `0924__flydsl/DAY-SUMMARY.md` header is the template |
| **Deep-round cadence was set by the user**: deep on rounds divisible by 5 (bwd: 17 was deep, 20 skipped, then 25/30/35) | op-evolve-ops §3 |
| Push after every round; reports and chat in Chinese, code/commits in English | memories `push-after-every-round`, `language-split` |
