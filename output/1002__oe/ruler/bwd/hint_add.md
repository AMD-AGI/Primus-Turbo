| h83 | must standing | `fast` is scored on the MIN and weighted 0 (gain_weights prod 1 / proxy 0.25 / fast 0); round 24's accept was fast-median noise | open |
| h84 | must standing | RULER 2026-10-02: proxy/prod are SCORED after a GEMM burst (gb ruler: 10 bf16 GEMMs before every timed call; ranks FlyDSL arms like the e2e); blocked figures are reported as blk_*; champions reset to the promoted round; every TF/s recorded before this hint is blocked-ruler and not comparable | open |

## h84 -- RULER: proxy and prod are scored after a GEMM burst (gb ruler), close to the training operating point

Operator, A0, 2026-10-02. Design: `PT/output/1002__oe/RULER.md`; evidence: `PT/output/1002__e2e/RESULT-realab.md` and `RESULT-e2e.md`
(`PT` = /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo).

The blocked ruler (h66) times every arm at the card's steady clock (~1.45-1.6 GHz on A0) with nothing in front of
the call; inside Llama-3.1-8B training every attention backward follows bf16 GEMMs. Same-process op A/B on A0
(6 layers of real training q/k/v) next to the e2e run of the same trees (per-layer CUDA events), 2026-10-02:

| time ratio | blocked (h66) | gb (this ruler) | e2e (training) |
|---|---|---|---|
| s6 / r29 (FlyDSL vs FlyDSL = what acceptance ranks) | 0.806 | **0.774** | 0.754 |
| r29 / ASM | 1.206 | **1.341** | 1.290 |
| s6 / ASM | **0.972** | 1.038 | 0.974 |

gb ranks FlyDSL arms the way training does (the blocked ruler under-rated s6's real gain over r29 by 5 points). Its
one miss is s6 vs ASM: training (step clock ~1.5 GHz median) agrees with the blocked figure there, so under gb
the beat ratio of s6 reads ~6% pessimistic (~0.96-0.98x instead of ~1.03x). In training s6 already beats ASM by
2.6% per step (fly vs ASM step 0.999 overall; the remaining per-step gap is in the forward).

What changed (`op/benchmark.py` + new `op/gbruler.py`; default `--ruler auto`):
- `fast`: unchanged -- blocked, scored on the MIN (h83).
- `proxy`, `prod`: the blocked loop still runs and is reported as `blk_latency_ms` / `blk_tflops`; then the gb loop
  runs and IS the scored figure (`latency_ms`, `tflops`, `ruler=gb`): before EVERY timed call
  `torch.cuda.synchronize()` + 10 bf16 GEMMs 32768x4096x14336 with the IMAGE hipBLASLt library (~21.5 ms), then the
  call with no sync between; palindromic rounds of 5 calls per arm (60 timed calls per arm at `--iters 51`); median.
  Witness per row: `gb_sclk` (call-window clock, expect ~1260-1390 MHz), `gb_burst_ms` (expect ~21-23 ms),
  `gb_iqr_pct`. A burst median above 80 ms (the image library is not in use) VOIDS the ruler: exit 4,
  `SPEED RULER VOID`, no RESULT row. `--ruler blk` gives the pre-10-02 behaviour; `--ruler gb|both` force either.
- `--aa LABEL` adds a byte copy of an arm as `LABEL_aa` in the same rounds; its row carries `aa_ratio`.
- `op/validation.py` is unchanged: it now gates proxy and prod at >= 1.20x beat on the gb figure (s6 reads ~0.96-
  0.98x there, ~6% pessimistic vs training, see above). Correctness and determinism are untouched.

Rules from now on:
1. Rank on proxy/prod `tflops` (gb). `blk_*` is context: a candidate that wins blocked but not gb is not a win.
2. Every ranking process: candidate + incumbent + `--aa <incumbent label>` + beat, then a second process in
   reversed arm order (h80). A difference smaller than max(0.5%, 2 x |1 - aa_ratio|) is noise.
3. The incumbent is `job_context/op/current`. Re-measure it in the same process; never use a ledger figure.
4. When this ruler was installed the operator reset every per-shape champion to the promoted round (== op/current;
   on 2026-10-02 r25's prod/proxy records had been chosen by the blocked ruler and r25 was never promoted). Every TF/s in
   state.yaml / progress.md / findings / hints recorded before this hint is BLOCKED-ruler: for FlyDSL arms ~4-8%
   above the gb figure, for ASM ~1%. Never compare a gb figure with them.
5. Cost: ~28.6 ms per gb timed call at prod (burst 21.5 + call 5.6 + host 1.5), ~23.4 ms at proxy: a 4-arm prod
   process takes ~7 s longer than before. Profiling runs (PMC / ATT): `--ruler blk` for cycle counts as before, or
   `--ruler gb --gb-iters 5` to see a kernel at the training clock (the GEMMs then appear in the trace too).
6. What moves the gb figure: work that costs cycles at every clock and power -- VALU/SALU/v_nop/wait trims, fewer
   barriers, less issued work (h82). A change that only pays at a high clock does not.
