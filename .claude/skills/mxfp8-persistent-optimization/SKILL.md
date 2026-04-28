# SKILL: Grouped MXFP8 Persistent Kernel Optimization

**Branch:** `dev/kyle_mxfp8_gg`
**Repo:** `/workspace/code/Primus-Turbo`
**HW:** AMD MI355X / `gfx950`. Other architectures must not be assumed.
**GPU pool (allowed):** 4,5,6,7. The host is shared; you must never use 0–3.
**Last updated:** 2026-04-28 (post-squash + wrapper cleanup)

This skill is loaded by `scripts/auto_optimize.py` at the start of every round. Read it once, then make a focused single-step optimization.

---

## 1. What you're optimizing

Two MXFP8 grouped GEMM kernels on `gfx950`, both persistent (variable-K is the wgrad direction):

| Direction | Kernel header (header-only, instantiated from .cu) |
| --- | --- |
| Forward / dgrad | `csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_kernel.h` |
| Wgrad (variable-K) | `csrc/kernels/grouped_gemm/turbo/turbo_grouped_gemm_mxfp8_wgrad_kernel.h` |
| Single GEMM (used by wgrad inner core too) | `csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h` |

The launchers and workspace plumbing live in:

- `csrc/kernels/grouped_gemm/turbo_grouped_gemm.cu` — workspace setup, launch config, scale preshuffle launch.
- `csrc/kernels/gemm/turbo_gemm.cu` — single GEMM launcher (rare to need on this branch).

`*_hip.h` siblings are HIP-portable variants kept in sync; only edit one and mirror manually if shared.

Layout: NT (`trans_a=False, trans_b=True`). Tile 256×256×128. Constraints `total_M % 16 == 0, N % 16 == 0, K % 128 == 0, K >= 384`.

---

## 2. Modification scope (HARD RULE)

### Allowed

1. **Kernel headers** — `csrc/kernels/grouped_gemm/turbo/*.h`, `csrc/kernels/gemm/turbo/*.h`. Edit freely.
2. **Launchers (internals only)** — `csrc/kernels/grouped_gemm/turbo_grouped_gemm.cu`, `csrc/kernels/gemm/turbo_gemm.cu`. Do **not** change the top-level entry-point signatures the C++ extension exposes; only edit the launch / workspace logic inside.
3. **Wrapper, equivalent micro-tweaks only** — `primus_turbo/pytorch/ops/grouped_gemm_fp8.py`. You may rearrange call order, fold reshapes, drop redundant `.contiguous()`, narrow `.cpu()` syncs, simplify the `ctx.save_for_backward` list, etc. The forward / backward outputs must remain **bitwise** equivalent on the metric shapes, and the file's public function signatures must stay unchanged.

### Forbidden — **any** of these is auto-reverted

- **Host-side caches in any form.** No `dict` / `weakref` / `data_ptr` / `_version` / LRU / TTL / "lazy compute" caching of: `quantize_*` outputs, scale preshuffle, `group_offs`, `grid_x_hint`, B's column-quant tensors, `autograd` intermediate products, etc. This is a **design decision**, not a performance trade-off. The previous wrapper-level caches were intentionally removed in `86371b3 / 7525339` to keep the cross-granularity comparison honest; do not reintroduce them, even with a different name (e.g. "memo", "stash", "reuse buffer", "registry"). If a host computation looks "wasteful to repeat", the right fix is either (a) fuse it into the kernel, or (b) accept the repetition. Never attempt to share state across calls.
- Dispatcher / backend class changes — `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py` (the `GroupedGEMMFP8*Backend` classes, `_backends` dict, and `grouped_gemm_fp8_impl` / `grouped_gemm_fp8_variable_k_impl` signatures stay untouched).
- C++ entry signatures — `csrc/include/primus_turbo/**`, `csrc/pytorch/extensions.h`, `csrc/pytorch/bindings_pytorch.cpp`, `csrc/pytorch/grouped_gemm/turbo_grouped_gemm.cpp`, `csrc/pytorch/gemm/turbo_gemm.cpp`.
- `benchmark/**`, `tests/**`, `scripts/**`, `.claude/**`, `3rdparty/**`.
- The metric script (`scripts/_metric_mxfp8.py`), `STRESS_*` / `SNR_*` thresholds, and `~/.cursor/cli-config.json`.

If an idea genuinely requires moving outside this list, write it down in the round summary as a "future suggestion" and do **not** commit. Do not be clever with the rules.

---

## 3. Performance baseline you must not regress

Score = `int(round(sum_tflops * 10)) - 1000*snr_fail - 100*stress_bad - 2000*exception` from `scripts/_metric_mxfp8.py`. Higher is better.

Most recent reference numbers (post wrapper cleanup, 2026-04-28):

- `_metric_mxfp8.py` baseline score on the current HEAD: **~40k–43k** (the previous 71215 number from `a268fc8` came largely from now-removed Python-side caches; do not chase that absolute number — the goal is to drive **score up from whatever today's baseline measures**).
- Direct kernel microbench (no wrapper, MI355X) — these are the numbers the kernel itself is capable of:
  - `B=16 M=2048 N=4096 K=7168` (DSv3-GateUP): ~1535 TFLOPS (Triton tensorwise: ~1382)
  - `B=16 M=2048 N=7168 K=2048` (DSv3-Down):   ~1236 TFLOPS (Triton tensorwise: ~1067)
  - `B=16 M=4096 N=4096 K=7168`:                ~1660 TFLOPS (Triton tensorwise: ~1609)
- Determinism stress on `(G=4, M=1024, N=2048, K=2048, E4M3)` is the worst-known shape. Target `stress_bad / 100 ≤ 2`.

The metric shapes (defined in `scripts/_metric_mxfp8.py`):

| Name | G | M | N | K | Format |
|---|---:|---:|---:|---:|---|
| DSv3-GateUP-B16 | 16 | 2048 | 4096 | 7168 | E4M3 |
| DSv3-Down-B16 | 16 | 2048 | 7168 | 2048 | E4M3 |
| gpt-oss-Down-B4 | 4 | 2048 | 2880 | 2880 | E4M3 |
| DSv3-GateUP-B4-E5 (fwd-only) | 4 | 2048 | 4096 | 7168 | E5M2 |
| stress shape | 4 | 1024 | 2048 | 2048 | E4M3 |

---

## 4. Recommended next experiments (kernel-only)

Pick one and follow through end-to-end (edit → rebuild → metric → stress → commit).

1. **Inner-loop K-loop drain tightening.** The `wait_vmcnt<0>() + s_barrier` between Phase 1 and Phase 2 is the single biggest inner-loop stall. `wait_vmcnt<3>` was already taken in the squash. `vmcnt<5..6>` are known unsafe. Values **8..16** have **not** been characterized — sweep them while keeping `wait_vmcnt<4>` as the floor when in doubt. Any change must clear `STOP_AFTER_BAD=1 N_ITERS=1000 stress_grouped_mx_bwd_determinism.py` AND raise `_metric_mxfp8.py` score.
2. **LDS footprint reduction to fit 2 CTAs/CU.** Current kernel metadata: `VGPR=256, SGPR=112, LDS_Block_Size=147456`. Halving LDS would let two CTAs share a CU and likely lift `MfmaUtil` (currently ~12–13 %). Look at `phase_mfma_lds_ldg` LDS slabs for both A and B regions and whether the dual c-store buffer can collapse on FWD now that the alternating buffer fix is in.
3. **Scale preshuffle fusion into the persistent kernel.** Right now `csrc/kernels/grouped_gemm/turbo_grouped_gemm.cu` launches `preshuffle_scale_16x4_kernel` separately before the persistent kernel. Folding the preshuffle into the persistent prologue removes one launch + one workspace read+write. Wgrad already fused LHS+RHS preshuffle in the squash; forward still has it separate.
4. **Forward C-store determinism polish.** The non-volatile C-store + alternating `c_tmp` cleared most FWD `out` races, but `(G=4,M=1024,N=2048,K=2048,E4M3)` still leaks ~2–6/100 in the stress probe. Candidates: explicit L1/L2 fence at the end of `store_c_subtile`, splitting `read_c` and `store_c` register pressure on `c_frags`, or coarsening the `wait_vmcnt` batch in the FWD epilogue.
5. **Wgrad C-store / dB race.** Wgrad C-store is intentionally `volatile` because non-volatile traded ~5 % perf for ~50 % more dB stress failures. There is still potential here for someone who can absorb the dB race elsewhere (e.g. atomic-like fence pattern, per-CTA private accumulators). High risk, high reward.

---

## 5. Hard "Not allowed as final solutions"

These were tried and rejected. Don't reintroduce them under any name:

- Restoring the deleted **flat** grouped MXFP8 kernel.
- Any **uniform-group fast path** (assumes equal `M_g`).
- **Host or device sync** as a "race fix" (`hipDeviceSynchronize`, `__threadfence`, etc.).
- **Cache flush / cache invalidate** as a workaround.
- **`s_nop`** padding or any hand timing pad.
- **Majority voting** / repeated execution to mask non-determinism.
- Removing `__launch_bounds__(256, 1)` from either kernel.
- Removing `"memory"` from inline-asm register clobbers.
- Removing or altering `tests/pytorch/ops/test_grouped_gemm_fp8.py`.
- Adding `pytest.skip` / lowering SNR thresholds anywhere.
- `wait_vmcnt<5>` or larger at Phase 1→2 (race comes back at 5+).
- `wait_lgkmcnt<0>`-only at Phase 1→2 with no `vmcnt` drain (race comes back at 15/200 G=8).
- 512 resident CTAs.
- N-fastest grid order.

---

## 6. Build / test / metric commands (read carefully — `.h`-only edits need a touch)

```bash
# Pick an idle GPU from the pool every time
IDLE=$(MXFP8_GPU_POOL=4,5,6,7 python3 -c \
  "from scripts._metric_mxfp8 import _pick_idle_gpu; print(_pick_idle_gpu())")

# Header-only edits do NOT trigger a rebuild on their own. Touch the .cu
# they're included from before re-installing.
touch csrc/kernels/grouped_gemm/turbo_grouped_gemm.cu csrc/kernels/gemm/turbo_gemm.cu
HIP_VISIBLE_DEVICES=$IDLE pip install --no-build-isolation -e . 2>&1 | tail -20
# Incremental rebuild ~3-7 minutes. Full clean would be ~30+; do not invoke.

# Metric (the loop's score; ~10-25s wall)
HIP_VISIBLE_DEVICES=$IDLE python3 scripts/_metric_mxfp8.py 2>&1 | tail -20

# Determinism stress on the worst-known shape
HIP_VISIBLE_DEVICES=$IDLE STOP_AFTER_BAD=1 N_ITERS=100 \
  python3 .claude/probes/stress_grouped_mx_bwd_determinism.py

# Long-run determinism gate before committing a barrier change
HIP_VISIBLE_DEVICES=$IDLE STOP_AFTER_BAD=1 N_ITERS=1000 \
  python3 .claude/probes/stress_grouped_mx_bwd_determinism.py

# Optional: full mx_blockwise pytest sweep (slow; the loop's deep-check runs
# this every 5 rounds anyway)
HIP_VISIBLE_DEVICES=$IDLE \
  pytest tests/pytorch/ops/test_grouped_gemm_fp8.py::test_grouped_gemm_fp8_mx_blockwise \
  --tb=short -q
```

---

## 7. rocprofv3 — quick recipe

For a single forward kernel invocation on `gfx950`:

```bash
HIP_VISIBLE_DEVICES=$IDLE rocprofv3 \
  --counters MfmaUtil LdsBankConflict LdsUtil VGPR SGPR LDS_Block_Size \
  -- python3 - <<'PY'
import torch
from primus_turbo.pytorch.ops.grouped_gemm_fp8 import grouped_gemm_fp8
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig, Format, ScalingGranularity, ScaleDtype,
)
G, M, N, K = 16, 2048, 4096, 7168
a = torch.randn(G*M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda")
gl = torch.full((G,), M, dtype=torch.int64, device="cuda")
cfg = Float8QuantConfig(format=Format.E4M3,
                       granularity=ScalingGranularity.MX_BLOCKWISE,
                       block_size=32, scale_dtype=ScaleDtype.E8M0)
for _ in range(3):
    out = grouped_gemm_fp8(a, b, gl, config=cfg)
torch.cuda.synchronize()
PY
```

Reference numbers from the squash baseline (forward kernel, MI355X):

- `MfmaUtil`: ~12–13 % (counter saturates at 2^27 over the full dispatch sample, so true per-tile utilization is higher).
- `LdsBankConflict`: 0.
- `LdsUtil`: low.
- ISA stats: `s_waitcnt vmcnt(0)` ≈ 11; `s_barrier` ≈ 24.

---

## 8. Per-round Definition-of-Done

- `scripts/_metric_mxfp8.py` score does not regress. Re-measure at least 2× to discount cross-tenant noise.
- `stress_grouped_mx_bwd_determinism.py` `bad / 100 ≤ 2` per shape.
- If the round changes anything in the inner-loop barrier set, also clear `N_ITERS=1000` once before committing.
- One **focused** `feat:` / `fix:` / `perf:` commit on the local branch only. Never push.
- Working tree must be clean after the round; if you couldn't make the cut, `git restore .` everything you wrote and write an "abandoned" note in the round summary.

If you cannot identify a concrete next step from this skill — write that conclusion in the round summary and exit cleanly. Do **not** invent work that violates §2 to fill the round.
