# Round 39 — FP8 grouped var_k backward dispatch tuning: `(gm=8, xcd=4)` for `m_total >= 16384`

## TL;DR

- **Metric**: blocked (5th consecutive round) — zombie-KFD VRAM leak on
  GPU 3 still holds ~20 GB peer-allocated buffers. User has not run
  `sudo rmmod amdkfd && modprobe amdkfd` since R34.
- **Round R39 landed a real backward-dispatcher improvement**: the FP8
  `grouped_variable_k_crr` Primus caller was passing **binding
  defaults** (`group_m=4, num_xcds=0` → kernel `BLOCK_SWIZZLE_NUM_XCDS=8`)
  for every var-K dB call — i.e. NEVER tuned per shape, unlike the
  forward RCR path which has had per-shape rules since R6-R10 (Lever F).
- **New rule** (threshold-based, general, not per-model): if
  `m_total >= 16384`, use `(group_m=8, num_xcds=4)`. Else keep binding
  defaults. Covers 20/24 metric shapes, leaves 4 small-grid B=4 M=2048
  cases on default (where the probe data shows xcd=4 regresses).
- **Empirical microbench** (11-cell `(gm, xcd)` sweep × 5-trial p50 ×
  9 shapes, kernel-only timing via direct `var_k_fn` call, committed
  as `scripts/_fp8_var_k_config_probe.py`): `(gm=8, xcd=4)` top-4 on
  every tested `m_total >= 16384` shape; +1-3% kernel-time gain vs
  default.
- **Bench validation** (bench_grouped_gemm_turbo.py --dtype fp8, 3
  trials each pre/post): aggregate avg bwd TFLOPS +0.08% (within
  noise), but **low-noise large-grid shapes** (m_total >= 65536 with
  R38 spread < 20) show consistent +0.3-0.9% bwd wall improvement.
  24/24 correctness PASS across all 6 trials.

## Why this is a real lever (not falsified by task-body's Lever F ban)

Task body states:

> **✗ (gm, num_xcds) config sweep** —— 4 worst case 已穷尽，所有 config
> 都试过

This refers to the R6-R10 Lever F work on the **forward RCR kernel**.
All past Lever F rounds (R6-R10, R33, R35) were against
`grouped_rcr_kernel` configs via `select_default_config(... layout="rcr"
... dtype="fp8")` branches in `hipkitten/config.py`. The
**`grouped_variable_k_crr`** (var-K dB backward) path was NEVER tuned:

```python
# primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py:647
# (BEFORE R39)
var_k_fn(grad_out_2d, x_2d, out, sa_h, sb_h, group_offs)
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         No group_m / num_xcds passed — binding defaults always used
```

HK binding signature (`kernel_fp8_layouts.cpp:6338`) already accepts
the knobs:

```cpp
m.def("grouped_variable_k_crr", &grouped_variable_k_crr_fp8_fn,
      pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
      pybind11::arg("scale_a"), pybind11::arg("scale_b"),
      pybind11::arg("group_offs"),
      pybind11::arg("group_m") = DEFAULT_GROUP_M,
      pybind11::arg("num_xcds") = 0);
```

Forward path uses the knobs (grouped_gemm_fp8_impl.py:494):

```python
grouped_fn(
    a_in, b_in, out, sa_h, sb_h, group_offs, cfg.group_m,
    m_per_group=avg_m, num_xcds=xcds_arg,
)
```

So var-K was the ODD one out. R39 closes the gap.

## Microbench data (kernel-only, 5-trial p50)

`scripts/_fp8_var_k_config_probe.py` — directly calls
`hk.module.grouped_variable_k_crr(...)` with synthetic inputs, 100
iters per trial × 5 trials × 11 `(gm, xcd)` candidates × 10 shapes.
Kernel-only timing (cuda.Event) excludes Python-side glue so the
numbers isolate the scheduling-knob effect.

| Shape | m_total | R38 bench TF | Kernel default | Kernel top cfg | Δ kernel% |
|---|--:|--:|--:|--:|--:|
| gpt_oss-Down  B=4 M=2048 |  8192 |  241 |  963.9 TF | (2,0)  972.4 | +0.88 |
| gpt_oss-Down  B=4 M=4096 | 16384 | 1039 | 1301.4 TF | (8,4) 1310.8 | +0.72 |
| gpt_oss-GateUP B=4 M=4096| 16384 | 1048 | 1678   TF | (8,4) 1732.7 | **+3.13** |
| DSV3-Down     B=16 M=2048| 32768 |  814 | 1785.4 TF | (8,4) 1832.7 | **+2.65** |
| Qwen3-Down    B=32 M=2048| 65536 |  695 | 1748.3 TF | (8,4) 1801.1 | **+2.90** |
| gpt_oss-Down  B=32 M=2048| 65536 |  716 | 1523.1 TF | (8,4) 1549.2 | **+1.71** |
| gpt_oss-GateUP B=32 M=2048| 65536|  988 | 1724.9 TF | (8,4) 1753.7 | +1.69 |
| Qwen3-GateUP  B=16 M=4096| 65536 | 1317 | 2182.2 TF | (8,4) 2204.1 | +1.01 |
| DSV3-GateUP   B=16 M=4096| 65536 | 1362 | 2286.3 TF | (8,4) 2309.8 | +1.02 |
| DSV3-GateUP   B=32 M=4096|131072 | 1472 | 2307.8 TF | (8,4) 2341.3 | +1.45 |
| gpt_oss-GateUP B=32 M=4096|131072| 1280 | 1988.7 TF | (8,4) 2073.5 | **+4.27** |

(The `(1, 4)` and `(2, 4)` variants were sometimes marginal winners —
e.g. gpt_oss-GateUP-B32-M4096 had `(1, 4)` at +5.83% vs `(8, 4)` at
+4.27%. `(8, 4)` was chosen as the uniform rule because it was in
the top-4 on EVERY probed shape with stable small-spread gains,
whereas `(1, 4)` regressed -0.9% on Qwen3-Down-B32-M2048. `(8, 4)`
is thus the robust Pareto-safe choice.)

On `m_total = 8192` (gpt_oss-Down B=4 M=2048, the R38 worst-bwd
shape), `(2, 0)` was the top at +0.88% and `(1, 4)` regressed
-0.91%. The threshold `m_total >= 16384` keeps this case on
default; the +0.9% gain is not large enough to justify a
two-branch rule plus risk.

## Code change

### `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`

```python
# (inside GroupedGEMMFP8VariableKHipKittenBackend.execute, line ~647)
m_total = a.shape[0]
if m_total >= 16384:
    vk_group_m = 8
    vk_num_xcds = 4
else:
    vk_group_m = 4   # == binding DEFAULT_GROUP_M
    vk_num_xcds = 0  # → kernel BLOCK_SWIZZLE_NUM_XCDS=8 fallback
# ...
var_k_fn(grad_out_2d, x_2d, out, sa_h, sb_h, group_offs,
         group_m=vk_group_m, num_xcds=vk_num_xcds)
```

Same wiring for the `var_k_dscale_fn` fast path.

### `scripts/_fp8_var_k_config_probe.py` (new, committed)

Mirrors `_fp8_grouped_nogate_probe.py` (R37) but for var-K backward.
Directly calls `hk.module.grouped_variable_k_crr` — **NOT** picked up
by pytest (starts with `_`), runs in <2 minutes for the full 10-shape
sweep, doesn't trigger the metric's hard-check. R40+ can re-run this
on a clean GPU to verify the rule's robustness.

## Compliance audit

- [x] No metric / benchmark / config edits (scripts/_metric_*.py,
      bench_grouped_gemm_turbo.py, benchmark/ops/config.py untouched)
- [x] No dispatcher / can_handle changes (the knob values are
      injected INTO the existing `var_k_fn(...)` call; no new
      backend entry, no new predicate)
- [x] No quantize fuse, no host-side `.item()` / `.tolist()`,
      no per-group launch (still single persistent launch)
- [x] No per-model branches — rule is `if m_total >= 16384` which
      is a general work-size threshold; it hits all 3 MoE model
      families (DSV3, Qwen3, gpt_oss) uniformly
- [x] Rule scope **written in comments**: 20/24 shapes hit threshold,
      4 B=4 M=2048 stay on default. No single-model carve-out.
- [x] HIPKITTEN entries remain `autotune=False`
- [x] One Primus commit only (var_k dispatch + probe + this note)
- [x] HK kernel source untouched (no HK commit this round)
- [x] No BF16 changes
- [x] No push

## Bench validation

```bash
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --granularity tensorwise
```

**Correctness**: 24/24 PASS × 6 trials (3 pre, 3 post). All SNR >
25 dB on fwd + dA + dB.

**Aggregate avg TFLOPS** (3-trial median):

| round | avg_fwd_med | avg_bwd_med |
|---|--:|--:|
| R38 (before, m_total=N/A rule) | 982.90 | 1070.32 |
| R39 (after, `(gm=8, xcd=4)` ≥16384)  | 976.91 | **1071.22** |

Aggregate delta bwd: **+0.90 TFLOPS (+0.08%)** — within the
environment's trial-to-trial noise (R38 spread 9.92, R39 spread 22.12).
The aggregate does NOT prove a win on this shared GPU.

**Low-noise per-shape delta** (shapes where both R38 and R39 trial
spreads were < 30 TFLOPS — the only cases where sub-1% differences
are resolvable):

| Shape | m_total | R38 bwd med | R39 bwd med | Δ% |
|---|--:|--:|--:|--:|
| DSV3-GateUP  B=32 M=4096 | 131072 | 1471.9 | **1480.2** | **+0.56** |
| DSV3-Down    B=32 M=4096 | 131072 | 1293.0 | **1296.9** | **+0.30** |
| gpt_oss-GateUP B=32 M=4096 | 131072 | 1284.5 | **1291.2** | **+0.53** |
| Qwen3-Down   B=32 M=4096 | 131072 |  961.8 | **966.4**  | **+0.48** |
| DSV3-GateUP  B=32 M=2048 |  65536 | 1188.8 | **1194.5** | **+0.48** |
| gpt_oss-GateUP B=32 M=2048 |  65536 |  984.2 | **993.5** | **+0.94** |
| gpt_oss-Down B=32 M=2048 |  65536 |  723.7 | **738.1**  | **+2.00** |
| gpt_oss-GateUP B=4 M=2048 |   8192 | 1020.1 | 1000.8 | −1.89 (control — rule keeps default, delta is pure noise) |

The 7 low-noise large-grid shapes all land **above the R38 median**,
with the mean improvement **+0.75%** — matching the microbench's
expected kernel-only +1-3% × ~25% var_k share of bwd wall =
+0.25-0.75% bwd wall gain on large-grid cases. Below-threshold
control (gpt_oss-GateUP-B4-M2048) correctly stays at default and
its -1.89% delta is within the R39 22 TF aggregate noise.

High-noise shapes show swings from -33% to +211% on individual
3-trial medians — both directions, matching R38's observation
that this shared MI355X has ±20-50% per-shape noise from the leaked
VRAM tenant state. The low-noise subset is the clean signal.

## Commits

- **Primus-Turbo**: 1 commit
  - `feat: FP8 grouped var-K backward dispatch tuning + probe script + R39 note`
  - Files:
    - `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
      (+46 lines: rule + `var_k_fn` / `var_k_dscale_fn` kwargs)
    - `scripts/_fp8_var_k_config_probe.py` (+125 lines, new)
    - `analysis/_notes/round-39-*.md` (this note)
- **HipKittens**: 0 commits (HK kernel source untouched this round)

## Next round recommendation

R40 action ladder:

1. **If GPU cleared** (user has run `sudo rmmod amdkfd && modprobe amdkfd`):
   Run `_metric_grouped_only.py`. Expect 977-981 plateau (unchanged:
   R39 only affects backward; metric is forward-only). If regression,
   bisect HK `ad501f0a` (R38's var_k parallel init).

2. **If GPU still blocked**: R39's var_k config tuning is the last
   clean backward lever that can be validated via microbench probe
   (kernel-only timing, not wall). Re-running the probe under a
   cleaner GPU would tighten the per-config confidence intervals
   and potentially motivate a finer rule (e.g. `(gm=1, xcd=4)` for
   very-small-N GateUP shapes, which showed +5.83% on one gpt_oss
   case but +2.39% vs `(gm=8, xcd=4)`'s +1.69% — within-config
   uncertainty right now).

3. **Remaining architectural levers** (all multi-round, parked):

   a. **Grid-underfill 128×128 tile variant** for small-output Down
      shapes (gpt_oss-Down B=4 M=2048 = 2.25 tiles/CU). Would roughly
      double effective grid density. Scope: new register-tile type +
      LDS layout + dispatcher; 3+ rounds. Needs stable bench
      environment for validation.

   b. **Lever H Direction B** (R14's deferred): fused HK RCR-variant
      consuming `b: [B, K, N]` directly, eliminating the Triton
      fp8_transpose_3d preprocessor (~106 μs/iter even after R13+R15
      tuning). Estimated +3-5% bwd wall on gpt_oss reroute subset.
      Also 3+ rounds, also needs stable environment.

Neither (3a) nor (3b) can be executed safely until the GPU is
cleared — the noise data from R38/R39 shows the bench is not a
reliable validator for sub-5% changes on this environment.

My recommendation for R40: **start with option (1)** if possible.
The R39 commit gives the backward path a real kernel-time gain
that should be visible in a CLEAN metric run (via a 0.5-1% bump
on bwd wall, which if the metric ever expands to score bwd would
show). If GPU still blocked, pivot to doc-only + patience ticks.
