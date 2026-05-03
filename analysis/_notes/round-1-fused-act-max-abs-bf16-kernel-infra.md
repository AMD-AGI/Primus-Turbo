# Round 1 — FP8 grouped fused-act: HK `max_abs_bf16` kernel infra deposit

**Status**: INFRASTRUCTURE LANDED on HipKittens side; standalone Python wiring
of HK `max_abs` + C++ quantize-with-scale **FALSIFIED** for round 1. The
kernel is correct and faster than torch in isolation, but Python-side
orchestration overhead negates the win when the goal is to replace the
existing C++ `quantize_fp8_tensorwise(a)` (which already does amax+apply
in a tight 3-launch C++ pipeline).

**Score**: 833 (baseline 831 / 834 — within noise band of ±2 on a 24-shape
geomean; un-fused path bit-identical, fused-act path silently unchanged
because the new HK binding is not yet called by Primus-Turbo).

**Files touched**:
- HipKittens `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - New kernel template `max_abs_bf16_kernel<MODE>` (modes: `MODE_AMAX` and
    `MODE_FP8_SCALE`). `MODE_AMAX` writes raw `max(|a[i]|)` to a single fp32
    device scalar via grid-stride `uint4` loads + warp-shuffle reduce +
    cross-block `atomicMax(int*)` on `__float_as_int(local_max)` (valid
    because `|·|` ≥ 0). `MODE_FP8_SCALE` extends with an in-kernel
    finalizer that runs `out = fp8_max / max(eps, amax)` (the
    `quantize_fp8_tensorwise(a, scale=...)` "scale" convention) — last
    block to retire (detected via `atomicAdd(done_counter, 1) == gridDim.x-1`)
    overwrites `out` with the post-transform value, all in a single launch.
  - Two pybind11 bindings: `max_abs_bf16(a, out)` (raw amax) and
    `max_abs_bf16_to_fp8_scale(a, out, done, fp8_max, eps=1e-12)`
    (fused-scale).
- Primus-Turbo `analysis/_notes/round-1-fused-act-max-abs-bf16-kernel-infra.md`
  (this note). No code change on Primus side this round.

**Probe results** (correctness — `/tmp/probe_max_abs.py`):
- All 9 test shapes (incl. tail-only `M=8,K=8`, awkward `M=12345,K=567`,
  and the 5 metric-relevant DSV3 / Qwen / gpt_oss shapes) bit-match
  `a.float().abs().amax()` to relative error 0.

**Probe results** (timing isolation — HK `max_abs_bf16` vs `torch.amax(a.abs())`):
| shape | M, K | HK_us | torch_us | speedup |
|---|---|---:|---:|---:|
| DSV3-GateUP-B16-M2048 | 32768, 7168 | 86 | 796 | 9.3× |
| DSV3-GateUP-B32-M4096 | 131072, 7168 | 323 | 3144 | 9.7× |
| DSV3-Down-B16-M2048   | 32768, 2048 | 28 | 239  | 8.5× |
| Qwen-Down-B16-M2048   | 32768, 1536 | 24 | 178  | 7.5× |
| gpt_oss-B4-M2048      | 8192,  2880 | 17 | 76   | 4.6× |

HK kernel is HBM-bound (theoretical `M*K*2 / 5 TB/s` matches measured to
within 10 %). `torch.amax(a.abs())` allocates an intermediate `|a|` tensor
(extra read+write pass) which is what the 8-10× gap reflects.

**Probe results** (timing wired-in — `/tmp/probe_fused_scale_v2.py`):
End-to-end `quantize_fp8(a)` -equivalent (HK_amax + C++ apply with the
HK-produced scale) vs the C++ default `quantize_fp8(a)`:

| shape | default_us | hk_pre_us | hk_fuse_us | win vs default |
|---|---:|---:|---:|---:|
| DSV3-GateUP-B16-M2048 | 218 | 224 | 222 | -2.1 % |
| DSV3-GateUP-B32-M4096 | 787 | 819 | 818 | -4.0 % |
| DSV3-Down-B16-M2048   |  59 |  68 |  66 | -12.5 % |
| DSV3-Down-B32-M4096   | 244 | 253 | 251 | -3.0 % |
| Qwen-Down-B16-M2048   |  48 |  58 |  55 | -12.5 % |
| gpt_oss-B4-M2048      |  30 |  47 |  37 | -22.6 % |
| gpt_oss-B4-M4096      |  46 |  57 |  53 | -13.3 % |

`hk_fuse_us` (`max_abs_bf16_to_fp8_scale` + `quantize_fp8_tensorwise(scale=...)`)
is **always 2–23 % SLOWER** than the C++ default. Numerics are bit-equal
(or within 1 ULP rounding-boundary) on every shape.

**Why the wiring fails**: launch count.
- C++ default `quantize_fp8(a)` = 3 launches (`reduce_row` 2-stage + `apply`)
  fully inside C++, no Python orchestration.
- HK fused-act path = 4 launches (`scale_buf.zero_()`, `done_buf.zero_()`,
  `max_abs_bf16_to_fp8_scale`, C++ `quantize_fp8_tensorwise(scale=)`) +
  per-launch Python dispatch overhead. The extra 5–10 µs/launch = 10–20 µs
  total, which dominates on small shapes (e.g. gpt_oss-B4-M2048 default = 30 µs,
  HK_fused = 37 µs → 7 µs gap = launch overhead).
- Stream overlap of `quantize(a)` ‖ `quantize(b)` is also FALSIFIED
  (`/tmp/probe_stream_overlap.py`): both kernels are HBM-bandwidth-bound,
  competing for BW gives ≤ 0.5 % delta; not a useful lever in isolation.

**The win path** (deferred to Round 2): build the inline-cvt fused GEMM
kernel `grouped_dscale_rcr_fused_act` per the task body's Phase-1 plan.
It loads BF16 `a` directly inside `load_a_tile`, converts to FP8 via the
existing AMD `__builtin_amdgcn_cvt_pk_fp8_*` builtin, AND writes the FP8
to the (still-required) `a_fp8` staging buffer for backward — fusing the
quantize-apply pass INTO the GEMM read. Net A-side HBM traffic drops
from 6 × M × K bytes (un-fused: 2-pass read for amax + apply, +1 read
inside GEMM) to ~5 × M × K bytes (1-pass amax read + 1 fused load+cvt+
GEMM read, +1 staging-buffer write); ~17 % saving on A traffic, expected
fused-wall score 880-920 per the task body.

**Round 2 plan**:
1. Clone `grouped_rcr_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL>`
   (lines ~2317-2978 of `kernel_fp8_layouts.cpp`) into a new template
   `grouped_rcr_fused_act_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL>`.
   Add an `_gl_bf16 a_bf16` field to `grouped_layout_globals` (or a new
   `grouped_layout_globals_fused_act` struct to avoid disturbing existing
   instantiations); modify `prefill_swizzled_offsets` + `load_a_tile` to
   pull BF16 from `g.a_bf16` and convert to FP8 via the `cvt_pk_*`
   builtin BEFORE the LDS write, while ALSO writing the FP8 to `g.a` for
   backward reuse.
2. Clone `grouped_rcr_dscale_fn` host wrapper into
   `grouped_rcr_fused_act_dscale_fn`. Same `dscale_*` plumbing; adds
   `a_bf16` arg.
3. Wire `_fused_act_grouped_fp8_forward` to:
   - `max_abs_bf16_to_fp8_scale(a, scale_buf, done_buf, FP8_MAX)` — produces
     the dscale value the new kernel reads. Passes `scale_inv = 1 / scale`
     (= amax / FP8_MAX) as the `dscale_a` device pointer.
   - `grouped_rcr_fused_act_dscale(a_bf16=a, a_fp8_out=a_fp8, b_fp8, c=out,
     scale_a=scale_inv, scale_b=b_scale_inv, group_offs, group_m,
     m_per_group, num_xcds)`.
4. Guard with `try / except AttributeError` so the binding's absence falls
   back to the un-fused path (preserves the regression-floor invariant
   from constraint #4 of the task body).
5. Initially gate to RCR + K%128==0 (Phase 1 scope: 16/24 shapes — DSV3
   + Qwen). gpt_oss (K=2880) keeps the un-fused fallback.

**Falsification ledger** (do NOT re-try in Round 2+ without new data):
- (R1-A) Pure-Python `torch.amax(a.abs())` → C++ apply with `scale=`:
  -150 % vs default (torch's intermediate `|a|` alloc + 2-pass dominates).
- (R1-B) Stream overlap of `quantize(a)` ‖ `quantize(b)` on separate
  CUDA streams: both HBM-bound, competing → 0.4-3.5 % regression.
- (R1-C) HK `max_abs_bf16` → C++ `quantize_fp8_tensorwise(a, scale=...)`
  with Python-side `FP8_MAX / amax` divide: -3 to -63 % vs default.
- (R1-D) HK `max_abs_bf16_to_fp8_scale` (fused in-kernel scale post-transform
  to remove the Python divide) → C++ apply: -2 to -23 % vs default. Still
  loses to C++ default by launch-count margin.

**Suggestion for next round**: implement the round-2 plan above. Target
deliverable: replace the Phase-0 fallback inside `_fused_act_grouped_fp8_forward`
with a fused HK kernel call when the binding is available AND the shape is
RCR + K%128==0. Expected fused_act_wall_score: 880-920.
