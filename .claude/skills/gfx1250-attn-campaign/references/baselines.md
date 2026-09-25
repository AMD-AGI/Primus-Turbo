# Baselines: the aiter ASM bars, how they are launched, and which numbers to cite

Scope: the two targets the FlyDSL campaign is scored against (aiter prebuilt gfx1250 ASM
forward and backward), the harness that measures them, and the historical numbers people will
quote at you. Abbreviations: `T` = `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`,
`O` = `$T/output`, `J` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context`,
`H` = `$O/0923__flydsl/hint.md` (copy of `$J/hint.md`; h54 index at line ~4060).

Legend: ✅ verified against a file on disk · ⚠ true with a condition · ❌ dead / retracted.

---

## 0. Cite these, nothing else

| bar | ms | TF/s | status | primary source |
|---|--:|--:|---|---|
| **fwd, aiter ASM** | **1.5724** (median, n=51) | **1398.67** | ✅ | `$O/0924__flydsl/bar-census/fwd_anchor.json` (sclk 1011->989) |
| **bwd, aiter ASM** | **7.6766** (median of census, n=224 raw / 7.6769 n=146 dedup) | **~716** | ✅ | `$O/0924__flydsl/DAY-SUMMARY.md:12`, `H` h32 |
| bwd (retracted) | 10.160 | 541 | ❌ artefact | `$O/0915__opt/status.json:34-38`; see §5 |

- Ratios are only trustworthy **same-process, same window** (`beat_measured_same_run: true` in
  both job specs). A stored bar is a sanity figure, never the gate (`$J/gfx1250-flydsl-attn-bwd_final.yaml:202-212`).
- Always publish `n` with its dedup convention: the same bwd census has been reported as
  n=158, 177 and 224 (`DAY-SUMMARY.md:144-146`); the n=177 / median 7.6721 pair is retracted (`H`:3052) -- never cite it. The archived file
  `$O/0924__flydsl/bar-census/beat_prod_RESULT_lines.txt` holds **158** lines: median 7.6770,
  min 7.6087, max 7.7556, sd 0.47% (recomputed 2026-09-25) ✅. All three conventions agree to 0.01%.
- Where we stand (for context, details belong in the history refs): bwd champion r20
  511.42 TF/s (516.87 re-measured r22) = **0.72x** bar (`DAY-SUMMARY.md:9-15`); fwd FlyDSL
  2.4005 ms / 916.16 TF/s = **0.655x** bar, gap 1.5266x (`fwd_anchor.json`, `H` h55).

---

## 1. Shape and FLOP convention ✅

`b=4 s=8192 hq=32 hkv=8 d=128 bf16`, causal **bottom-right** (sq == skv so identical to
top-left here), layout **BSHD** (`q: (b, s, hq, d)`). Llama-3.1-8B, GQA ratio 4.
Job shape table `$J/op/ut/common.py:13-17`:

| name | (b, sq, skv, hq, hkv, d) | role |
|---|---|---|
| fast | (1, 1024, 1024, 8, 2, 128) | fast iteration; launch-bound (bar ~1.8 ms bwd) |
| proxy | (1, 4096, 4096, 32, 8, 128) | quarter-FLOP proxy |
| prod | (4, 8192, 8192, 32, 8, 128) | **the score** |

Do not rename shapes: it forks the champion ledger (`H` h35).

**FLOP** comes from the shipped tool `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/tools/op_flops.py:136-170`,
imported by `benchmark.py`, never recomputed inline:

```
live  = exact count of unmasked (i,j) / (sq*skv)   # causal s=8192: 8193/16384 = 0.500061
pairs = b * hq * sq * skv * live
fwd   = 4 * d * pairs            # QK^T + PV, 2*d each
bwd   = 2.5 * fwd                # five GEMMs; delta = rowsum(dO*O) and softmax not counted
```

| | FLOP | shorthand |
|---|--:|---|
| fwd prod | **2,199,291,691,008** (2.199292e12) | = `2*b*hq*s^2*d` x (s+1)/s |
| bwd prod | **5,498,229,227,520** (5.498229e12) | = `5*b*hq*s^2*d` x (s+1)/s |
| fwd+bwd | 7,697,520,918,528 | `$O/0924__flydsl/bar-census/fwdbwd_anchor.json` |

- GQA does not change FLOP, only bytes (`op_flops.py:156-158`). bwd `bytes_min` prod = 1,342,177,280.
- ⚠ Older harnesses used ~2.233e12 for fwd (`STAGE2-FWD-SWEEP.md` prints 947.8 TF/s for 2.356 ms);
  the fwd spec standardised on the tool (`$O/0923__flydsl/fwd-job/gfx1250-flydsl-attn-fwd.yaml:395-398`).
  `fwd341/fwdab.py` uses `4*b*hq*d*s(s+1)/2` = identical to the tool ✅.
- "Issued" vs "algorithmic" FLOP: our bwd issues 1.407641x algorithmic (7-GEMM recompute), aiter
  1.015501x. Never compare our issued rate with aiter's algorithmic one (`$O/0925__flydsl/PARITY-STRATEGY.md:75-95`).

---

## 2. The code objects ✅

aiter checkout `/home/lihuzhan/code/aiter-src` at `6963ae9` (verified `git log -1`, 2026-09-25).

**Forward** — `/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_fwd_bf16/` (table: `fmha_fwd_bf16.csv`)

| mask | .co | symbol |
|---|---|---|
| 1 (causal, **used**) | `fmha_bf16_pertokenBf16_hd128_128x256_mask.co` | `_ZN5aiter41fmha_bf16_pertokenBf16_hd128_128x256_maskE` |
| 0 | `fmha_bf16_pertokenBf16_hd128_128x256.co` | `_ZN5aiter36fmha_bf16_pertokenBf16_hd128_128x256E` |

Also hd64 pair; varlen in `fmha_fwd_bf16_varlen/`. Kernarg 132 B; host hardcodes opt=7
(reverse_kv | double_q | remap_xy), Q tile 128 (`$T/tools/gfx1250/asm_bwd_launcher.py:47-62`).

**Backward** — `/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd/` (6 `.co` total vs 124 on gfx950)

| stage | .co | symbol | role |
|---|---|---|---|
| 1 | `bwd_hd128_odo_bf16.co` | `_ZN5aiter23fmha_bwd_hd128_odo_bf16E` | delta = rowsum(dO*O) |
| 2 | `bwd_hd128_bf16_causal_br_a32_pssk.co` | `_ZN5aiter38fmha_bwd_hd128_bf16_causal_br_a32_psskE` | main body; 514 `buffer_atomic_add_f32` into fp32 dq_acc |
| 3 | `bwd_hd128_dq_convert_bf16.co` | `_ZN5aiter30fmha_bwd_hd128_dq_convert_bf16E` | dq fp32 -> bf16 |
| non-causal 2 | `bwd_hd128_bf16_a32_pssk.co` | `_ZN5aiter28fmha_bwd_hd128_bf16_a32_psskE` | mask=0, full x-grid |
| ❌ | `*_pssk_perf.co` (x2) | — | dk/dv bitwise-equal, dq 5.84 dB: unusable (`$O/0922_summary/ASM-ATTENTION.md:150`) |

Constants (`$T/primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py:46-52,176-190`):
TS_KV=128, TS_QO=32, TS_ODO=128, TS_DQ=64, BDX=128 (4 wave32), MASK_X=MASK_Y=0. Two packing
conventions coexist: `dqdkdv` fields 16-B aligned (kernarg 704 B), `odo` packed. Strides are
**bytes**. Resource use: VGPR 1024, LDS 320 KiB, wg 1024 — saturated, do not hand-edit ASM
(`ASM-ATTENTION.md:152`).

---

## 3. Launch paths

### 3.1 Forward — public aiter entry, no hand launcher ✅

```python
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")   # beat arm only; baseline arm must NOT
from aiter.ops.mha import fmha_fwd_with_sink_asm      # aiter/ops/mha.py:589
o, lse = fmha_fwd_with_sink_asm(q, k, v, d**-0.5, True, True)   # is_causal, return_lse; sink=None
```

- `@compile_ops(..., ffi_type="ctypes")`: **first call JIT-builds a C++ module** — do it outside every
  timing window (fwd spec `:514-545`).
- Working same-process A/B template: `$O/0925__flydsl/fwd341/fwdab.py` (one shape per process,
  prints `RESULT {json}` with `ratio_fly_over_asm_tflops`, `agree_db`, `sclk`). Older blueprint:
  `$O/0917__flydsl/bin/stage1_fwd_ab.py`.
- A direct `.co` forward launcher also exists in `$T/tools/gfx1250/asm_bwd_launcher.py:47-62`
  but is experiment-only; nothing scores through it.

### 3.2 Backward — our hand-built launcher (aiter's Python gate never selects gfx1250) ✅

- aiter's `can_impl_fmha_v3_bwd` only admits gfx942/gfx950, so the three `.co` are launched by hand
  (`ASM-ATTENTION.md:29-33`).
- Machinery: `$T/primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py` (`asm_backward`, `:193`;
  `HipModule`); re-exported by `$T/tools/gfx1250/asm_bwd_launcher.py` (185 L, bring-up + ABI self-test)
  and field table `$T/tools/gfx1250/asm_bwd_abi.py` (136 L, no torch/no GPU self-test; it caught a
  hand-copied table missing its last 7 fields incl. mask_x/mask_y).
- **The scored arm** is `$J/op/beat/impl.py` (92 L). It loads `_asm_bwd_kernargs.py` **by file path**
  (importing `primus_turbo.pytorch` pulls a FlyDSL tree that needs `flydsl.expr.buffer_ops`, removed in
  0.3.2), loads the 3 modules **once per process**, allocates scratch **once per shape**, then per call:

```python
dq, dk_q, dv_q = mod.asm_backward(q,k,v,o,do,lse, softmax_scale=s, hip=hip,
                                  dkdv_heads="q", causal=True, scratch=scratch)
dk = dk_q.view(b, skv, hkv, g, d).sum(dim=3).to(k.dtype)   # host GQA reduction, bf16
dv = dv_q.view(b, skv, hkv, g, d).sum(dim=3).to(v.dtype)
```

- Spec line (`$J/gfx1250-flydsl-attn-bwd_final.yaml:202`): *"aiter prebuilt gfx1250 ASM backward via
  tools/gfx1250/asm_bwd_launcher.py, dkdv_heads=q plus the host reduction -- must be beaten by 0%"*.
  Keep the substring `beaten by N%` — `op_evolve/core/tune.py:32` parses it.
- Per-call dispatches = odo + pssk + dq_convert + FillFunctor(`dq_acc.zero_()`) + 2x reduce = **6**,
  confirmed by exact dispatch count (`$J/profiling/beat/kernel.yaml:46-49`). All six are inside the
  timed region; that is the honest price of a correct dk/dv from this kernel.

### 3.3 The vendor GQA OOB bug ✅ (filed: `$O/0917__flydsl/VENDOR-REPORT-aiter-gfx1250.md`)

The pssk grid is `(kv_tiles, nhead_q, batch)` but it indexes the **kv-sized** dk/dv buffer by **q head**:
at ratio 4, four workgroups race on one dk/dv tile / write out of bounds.

| config | dq dB | dk dB | dv dB |
|---|--:|--:|--:|
| ratio 1 | 52.26 | 52.27 | 52.76 |
| ratio 4, dk/dv per kv head (`dkdv_heads="kv"`) | 52.24 | **-0.94** | **-0.65** |
| ratio 4, per q head + host sum | 52.24 | 50.55 | 51.09 |

- s=1024: silent corruption (~-0.3 dB); s=256: process fault, card not harmed (`ASM-ATTENTION.md:52-70`).
- Cost of the workaround: dk/dv 0.125 -> 0.500 GiB, +1.254 GiB launch peak; loses 1.7-1.9 dB on dk/dv.
- ⚠ Time cost has **two numbers, different harnesses**: 0.4823 ms (`gqa_workaround_cost.json`, fp32 sum,
  `H` h32) vs **0.2225 ms/call = 2.88% of bar** (bf16 `reduce_kernel` 111269 ns x2, `$J/profiling/beat/kernel.yaml:119-128`).
  Cite the profile for the current `beat/impl.py`. Removing the workaround would move the bar **down**
  (~7.45 ms), never up (`H` h32).
- ✅ The bar does **not** read stale gradients: scratch NaN-poisoned, zero NaN survived at fast and prod
  (`H` h34; script `$O/0924__flydsl/bar-census/beat_scratch_check.py`).
- Do not copy aiter's `grid.y = nhead_q` into our kernel: it is the source of both the reduce cost and the
  dB loss (`$O/0925__flydsl/AITER-5GEMM-STUDY.md:135`).

---

## 4. Harness: `$J/op/benchmark.py` (195 L) ✅

Measures, never judges; `validation.py` is the only judge. Fixed method (docstring `:9-33`):

| knob | value | why |
|---|---|---|
| statistic | **median** of per-iteration CUDA-event times | never mix with best-of-N |
| iters | 51 (default) | at 20, two copies of the same dir disagreed 7.8% at `fast` |
| warmup | 3.0 s **continuous** load per arm, no sleeps | paced warmup reads up to 40% off |
| order | **palindromic** (even i forward, odd i reversed) | `A B C` repeats convict the first arm |
| inputs | one `make_inputs(shape, seed=0)` shared by every arm | |
| o / lse | `refcache_util.cached_forward` (no fp32 GEMM in the timing subprocess) | the fp32 reference cost a power cycle 2026-09-22 |
| L2 | 256 MiB `flush.zero_()` before each timed call, outside ev0/ev1 | |
| clock | `rocm-smi --showclocks` sclk before/after each shape, printed per line | |
| modules | loaded **once per arm for the whole process** (r15) | per-shape reload -> GC'd `hipModuleUnload` mid-launch -> fault |
| output | `RESULT shape= arm= stat= iters= latency_ms= min_ms= max_ms= tflops= bw_gbs= flop= bytes_min= sclk_start= sclk_end= causal=` | parse these |

```
python3 benchmark.py --arms baseline,beat --shapes fast,proxy,prod [--iters 51] [--json out.json]
python3 benchmark.py --arm-path cand=/abs/rounds/NNN/op --arms beat
```

- ⚠ One shape per process is the safe rule for anything outside `benchmark.py` (`H` h31: the fault binds
  to one shape per process). Card rules: `~/.claude/skills/gfx1250-card-safety`.
- ⚠ Forward-job gotchas, as found in the **superseded 0923 spec** (`$O/0923__flydsl/fwd-job/NOTES.md:105-116`; the live 0925 spec/job state is in `fwd.md` §2 -- precision_gate fixed, min_gain 0.007): the bwd
  `benchmark.py:72-76` hardcodes `backward=True`; `ut/common.py:104` hardcodes `getattr(mod, "attn_bwd")`.
  Fwd spec `op.precision_gate` loads as **None** because its text was merged into the `refcache` string
  (`gfx1250-flydsl-attn-fwd.yaml:400-401`). `min_gain` is 0.005 in the yaml (`:609`) vs 0.015 in NOTES
  (`:152,182-195`); the current plan uses 0.01, and `op-evolve tune` cannot change it — hand-edit the yaml.
- Noise floor for this operator, same session: **0.24-0.66%** at prod (r23 0.24%, h56). The 1.57% figure is HipKittens'
  bf16 GEMM ladder floor, not ours ❌ (`H` h38-CORRECTION, `PARITY-STRATEGY.md:40`).
- Clock: 1100 MHz is the **idle** clock; the prod window runs ~998-1030 MHz (185/264 samples <= 1030).
  Same-run A/B cancels it; absolute TF/s does not (`DAY-SUMMARY.md:57-63`, `H` h39 §1).

---

## 5. Why 10.160 ms / 541 TF/s is dead ❌

| fact | source |
|---|---|
| Origin: 2026-09-15 07:23, n=5 (10.116-10.181), "final/champ" | `$O/0915__opt/status.json:34-38` |
| What it timed: `_AsmFwdAsmBwd` autograd shim, `out.backward()`, calling `asm_backward` with **neither `hip=` nor `scratch=`** (3x `hipModuleLoad` + fresh ~1 GiB scratch per call, fp32 GQA sum) | `$T/tools/gfx1250/tune_attention.py:560-578` |
| Superseded the same day; corrected path 8.13-8.68 ms, never hardware-verified at the time | `$O/0915__opt/E2E-AB.md:290-302` |
| Mislabelled "产品路径" (product path); `asm_backward` has zero product callers | `$O/0915__opt/RESULTS.md:15`, `H` h32 |
| Census of the real bar: 7.67-7.68 ms, **zero records >= 7.8 ms** | `DAY-SUMMARY.md:12`, `H` h32 |
| Attributed overhead split (~+1.15 ms module load, ~+1.42 ms autograd, +0.01 ms alloc) is **prose-only** — cite totals, never the decomposition | `DAY-SUMMARY.md:67-75` |

**Inherited wrong numbers still in circulation** (do not quote):
- 541.1 TF/s, the "0.92x" ratio (real 0.70-0.72x), 11.726 ms/layer and its 4.76x, op-level 1.74x (really ~2.18x vs the 09-15 corrected 8.13 ms path, `E2E-AB.md:297-299`; 17.686/7.6766 = ~2.30x vs today's bar).
- `$O/0922_summary/ASM-ATTENTION.md` §2 row 4 and the `0922_summary` xlsx `attention` sheet.
- ⚠ **Stale skill (annotated 2026-09-25):** `~/.claude/skills/gpu-kernel-campaign/SKILL.md:214` presents "17.686 -> 10.160 ms"
  as a measurement. The lesson there (op win != e2e win) stands; the number is the shim.
- ⚠ The fwd spec's `beat_reference_figure` still mentions 10.160 as "the backward's figure" (`gfx1250-flydsl-attn-fwd.yaml:565-567`) — harmless, it says it does not carry.
- ⚠ The `seqlen >= 2048` product gate was derived from a "0.85 ms fixed floor" that partly was the per-call
  module load; its 9-point re-measure is still owed (`E2E-AB.md:301-302`).

---

## 6. Fresh forward re-measures (2026-09-25, same card, `fwd341/`) ✅

Six prod processes, flydsl 0.3.2 vs 0.3.4.1, ABBA x3, n=52 each (`$O/0925__flydsl/fwd341/ab_prod.jsonl`, driver `ab_prod.sh`):

| | asm ms | fly ms | fly/asm TF/s | sclk |
|---|--:|--:|--:|---|
| range (6 runs) | 1.539-1.553 | 2.332-2.343 | 0.657-0.666 | 1038-1049 |

- FlyDSL 0.3.4.1 vs 0.3.2 on the vendored aiter FlyDSL fwd: **null** (ratio 0.664-0.666 vs 0.657-0.664).
  `agree_db` 49.93 at prod — note that is ASM-vs-FlyDSL agreement, just under 50, not an SQNR vs fp32.
- fast shape (`ab_fast.jsonl`, n=22): asm 0.030 ms, ratio 0.73-0.74 (launch-bound, not informative).
- The ASM fwd reproduces across days: 1.557 (09-23, n=101, sd 1.39%, fwd spec `:569-572`), 1.5724 (09-24),
  1.539-1.553 (09-25). Cross-session drift ~1.5-2%; only same-process ratios are comparable.
- ⚠ In the fwd+bwd anchor the forward read **1.6061** (`fwdbwd_anchor.json`), vs 1.5724 in the fwd-only
  probe — forward agreement across windows is 0-2.1%, not 0.03% (`H` h32 "Correction").
- The script that produced `fwd_anchor.json` / `fwdbwd_anchor.json` lived in container `/tmp` and was
  **not archived**; only the JSON results were copied. Use `fwdab.py` / `benchmark.py` to re-measure.

---

## 7. The ladder (history, A0 = `heliosr-1b114-c07-1`, VR-capped 1100 MHz)

From `$O/0915__opt/RESULTS.md:9-15` and `$O/0922_summary/ASM-ATTENTION.md` §2 (one layer, fwd+bwd, prod shape).
Sources' ms are CUDA-event medians via `tools/gfx1250/tune_attention.py` (iters 20).

| # | fwd impl | bwd impl | fwd ms | bwd ms | total ms | status |
|---|---|---|--:|--:|--:|---|
| 1 | Triton (turbo stock `1cb2e183`) | Triton fused (stock) | 10.651 | 45.134 | 55.785 | ✅ n=3 |
| 2 | Triton (forced) | Triton fused (vendored, 316 TF/s) | 4.166 | 17.835 | 22.001 | ✅ n=2 |
| 3 | aiter ASM | Triton fused | 1.561 | 17.686 | 19.239 | ✅ current product default |
| 4 | aiter ASM | aiter ASM (shim) | 1.572 | **10.160** | 11.726 | ❌ bwd is the shim artefact |
| 4' | aiter ASM | aiter ASM (`beat/impl.py`) | 1.5724 | **7.6766** | ~9.25 | ✅ fwd and bwd from separate probes; same-window pair: 1.6061 + 7.7636 = 9.3698 (`fwdbwd_anchor.json`) |

- Ladder is in the xlsx sheet `attention` of `$O/0922_summary/mi455x-gfx1250-kernel-optimization.xlsx`;
  `$O/0915__opt/*.xlsx` has the B0/A0 version.
- ASM fwd alone is worth 2.74x (4.253 -> 1.549 ms) (`RESULTS.md:42`).
- E2E (Llama-3.1-8B, b4 s8192, 1 GPU): ASM bwd -0.28% on 09-15 (GEMM was 94% of step), **+12.72%** on
  09-17 after the GEMM fix (corrected from +14.40% after NaN runs dropped). ASM bwd stays **default-OFF**
  (`PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD`; fwd kill switch `PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD`) (`ASM-ATTENTION.md:83,128-146`).

---

## 8. B0 machine numbers ⚠

B0 = `ctheliosp-1b112-a37-1`, 4x gfx1250, **2133-2244 MHz under load** (quoted from
`$O/0914__campaign/RESULTS.md` header, not sampled). Numbers from raw `$O/0914__campaign/ledgers/*.jsonl`,
ASM-fwd records only, filtered to the quiet window (`$O/0915__opt/JIRA-TRACE-ANALYSIS.md:120-160`):

| cluster | n | fwd | bwd | total |
|---|--:|--:|--:|--:|
| quiet (bwd < 9.2) | 8 | **1.406** | **8.835** | **10.236** (sd 0.25%) |
| contended | 10 | 1.414 | 9.787 | 11.202 |

- Headline claim: B0 clocks 2.13-2.24x A0, but same code is only **1.118x / 1.150x / 1.146x** faster
  (fwd/bwd/total) -> gfx1250 attention is nearly clock-insensitive (memory / fixed latency bound).
- ✅ The **fwd** half holds: 1.572 (A0) vs 1.406 (B0) = 1.118x, both clean paths.
- ⚠ The **bwd/total** ratios are computed against the 10.160 shim, so 1.150x / 1.146x are contaminated.
  B0's 8.835 was also a 09-14 `tune_attention.py` measurement and is > A0's clean 7.68 bar, so it very
  likely carries the same per-call overhead. **Do not use B0 bwd as a hardware ceiling**, and do not
  derive "A0 11.726 ≈ 10.2 ms on a healthy card" from it.
- On B0 ASM bwd showed no edge over fused Triton (8.840 vs 8.848) — same caveat applies.
- External reference rows (JIRA MI455 Flex 3.309/9.638, MI355 aiter 3.797/5.381) are in the same file
  `:55-62,173-181`; their harnesses are unknown (`?` clock) — cite as context only.
- Triton-side B0 sweep result: fwd `num_warps=2` + bwd `waves_per_eu` unset = +4.28% (`ASM-ATTENTION.md:100-103`).

---

## 9. Linked skills and what is stale in them

| skill | use for | stale |
|---|---|---|
| `~/.claude/skills/gfx1250-card-safety` | wedge preconditions, recovery ladder, one-shape-per-process | — (read before any GPU run) |
| `~/.claude/skills/flydsl-gfx1250` | FlyDSL idioms; §8 fwd perf (ASM 1.569 / FlyDSL 2.373 ms, n=20) | consistent with bar (n=20 though); predates flydsl 0.3.4.1 |
| `~/.claude/skills/gpu-kernel-campaign` | measurement discipline, same-run A/B | §11 (`SKILL.md:214`) quotes 10.160 / 1.74x; a correction note pointing here was added 2026-09-25 ⚠ |

Deeper evidence: `H` h32 (bar census + retraction), h34 (stale-gradient check), h54 (one-page index),
h55 (fwd job premise: baseline is a mature vendor kernel, gap 1.5266x is pure efficiency).
