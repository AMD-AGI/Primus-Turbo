# B0 4-card campaign -- rules for every lab agent (read fully before any command)

Host B0 = ctheliosp-1b112-a37-1, 4x gfx1250 in ONE XGMI hive. A wedge on any card can take down all four and
costs the user a physical AC power cycle. Four streams run at once:

| card | container | owner (DAY 2, 2026-09-28) | kfd gpu_id | sysfs |
|---|---|---|---|---|
| GPU 0 | fa-g0 | end-to-end training workflow + auxiliary tests (share via flock /tmp/b0-gpu0.lock) | 34992 | /sys/class/drm/card0 |
| GPU 1 | -- | **WEDGED (MES failed to respond). No container. Never touch.** | 30548 | /sys/class/drm/card8 |
| GPU 2 | fa-g2 | op-evolve fwd job `gfx1250-flydsl-attn-fwd-b0-20260927` | 57865 | /sys/class/drm/card16 |
| GPU 3 | fa-g3 | op-evolve bwd job `gfx1250-flydsl-attn-bwd-20260917-115934` | 51359 | /sys/class/drm/card24 |

1. **Use only your own container.** Every GPU command is `flock /tmp/b0-gpuN.lock docker exec fa-gN ...` with
   N = your card. Never `fa-repro` (parked), never another fa-g*, never python/torch on the host, never
   `docker run`, never set HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES (inside fa-gN the only device is logical 0).
   Compile-only work needs no card: run it in your container WITHOUT flock (it does not touch the GPU) with
   `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_<arm>`.
2. **Never touch** the op-evolve job directories under `~/code/2026_0910__op-evolve/op-evolve/artifacts/` except
   to READ them (copy trees out; the harness `job_context/op/benchmark.py` may be run with `--arm-path`, cwd set to
   that `op/` dir, it writes nothing there when given `--json <your dir>`). Never write the jobs' hint.md -- report
   proposed hints in your result; the operator delivers them.
3. **Card safety** (skill `gfx1250-card-safety`, campaign SKILL.md section 4): one shape per benchmark process;
   compile-only first; any `.vgpr_spill_count > 0` or scratch > 0 never goes on the card; bounds-prove every new
   index expression on CPU before its first card run; never rocprofv3 PC sampling; kill only by explicit PID.
   After each card process: `timeout 20 sudo -n dmesg | tail -20` must show no new amdgpu fault. On any fault or a
   hung process: STOP all card work, do not retry, report the dmesg lines.
4. **Measurement**: same-process palindromic A/B vs the champion, n >= 101 at prod, arm orders rotated across
   >= 3 processes; a FlyDSL arm measured after the ASM `beat` arm pays an I-cache penalty (A0 r6) -- keep
   candidate and champion in the same position relative to beat, or leave beat out when ranking. Claims under
   0.5% are noise. Neighbour cards are loaded by design; only same-process ratios are evidence.
5. **Git**: write only under `output/0927__b0/<your lab>/`. Never `git push`, never `git pull`, never commit --
   the operator commits.
6. **Never run sustained GEMM / burn loops** on any card: a neighbour GEMM burn slowed attention on other cards 3-11x (`interference.md`). Attention workloads on other cards are harmless.
7. Report in your final answer: per arm the ratio vs champion (each process), VGPR/spill, dB vs reference, and a
   verdict (win / null / loss / blocked) with the evidence file paths.
