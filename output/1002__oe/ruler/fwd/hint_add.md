| h51 | must standing | RULER 2026-10-02: proxy/prod are SCORED after a GEMM burst (gb ruler: 10 bf16 GEMMs before every timed call; ranks FlyDSL arms like the e2e); blocked figures are reported as blk_*; first gb round is the one after h48 resets the champions (round 21); every TF/s recorded before it is blocked-ruler (and B0 before round 20) | open |

## h51 -- RULER: proxy and prod are scored after a GEMM burst (gb ruler), close to the training operating point

Operator, A0, 2026-10-02. Design: `PT/output/1002__oe/RULER.md`; evidence: `PT/output/1002__e2e/RESULT-realab.md` and `RESULT-e2e.md`
(`PT` = /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo). Installed at the boundary between round 20 (redone on A0
under the blocked ruler, h49) and round 21's first measurement, i.e. after h48 has repointed the champions.

The blocked ruler (h40) times every arm at the card's steady clock (~1.45-1.6 GHz on A0) with nothing in front of
the call; inside Llama-3.1-8B training every attention forward follows bf16 GEMMs and FlyDSL is clock-sensitive
where ASM is not (h42). Same-process op A/B on A0 next to the e2e run of the same trees (per-layer CUDA events),
2026-10-02:

| time ratio, r16 (= r13ns) / ASM | blocked (h40) | gb (this ruler) | e2e (training) |
|---|---|---|---|
| 6 real layers (geomean) | 1.081 | 1.349 | **1.263** (43.9 vs 34.7 ms/step) |
| randn (this job's input) | 1.080 | 1.343 | |

So the blocked ruler shows an 8% gap where training pays 26%; gb reads 35% (it over-weights the clock a little:
the e2e step runs at ~1.5 GHz median, gb calls at ~1.28 GHz). The forward is now the whole per-step gap of the
FlyDSL arm (fly vs ASM step 0.999: bwd -4.5 ms, fwd +9.2 ms per step). On the bwd side the same e2e confirms that
gb ranks FlyDSL arms against each other like training does (s6/r29: e2e 0.754, gb 0.774, blocked 0.806); for the
fwd, B0 showed the same for r13ns vs r13 (h44). h47 asked for "after a GEMM burst" by hand; it is now built into
`op/benchmark.py` (+ new `op/gbruler.py`), default `--ruler auto`:
- `fast`: unchanged (blocked, median; shape_band 0.90).
- `proxy`, `prod`: the blocked loop still runs and is reported as `blk_latency_ms` / `blk_tflops`; then the gb loop
  runs and IS the scored figure (`latency_ms`, `tflops`, `ruler=gb`): before EVERY timed call
  `torch.cuda.synchronize()` + 10 bf16 GEMMs 32768x4096x14336 with the IMAGE hipBLASLt library (~21.5 ms), then the
  call with no sync between; palindromic rounds of 5 calls per arm (110 timed calls per arm at `--iters 101`);
  median. The burst initialises hipBLASLt before any arm is loaded and the env is re-assigned after, because this
  job's `_env.py` ASSIGNS the host library (with it the burst runs ~80 TF/s and the clock never drops). Witness per
  row: `gb_sclk` (call-window clock, expect ~1260-1300 MHz), `gb_burst_ms` (expect ~21-23 ms), `gb_iqr_pct`. A burst
  median above 80 ms VOIDS the ruler: exit 4, `SPEED RULER VOID`, no RESULT row. `--ruler blk` = the old behaviour.
- `--aa LABEL` adds a byte copy of an arm as `LABEL_aa` in the same rounds; its row carries `aa_ratio`.
- `evolve.gain_weights` is {prod: 1.0, proxy: 0.25, fast: 0.0} (as this job had on A0 before the B0 fork, and as the
  bwd job has): prod ranks.
- `op/validation.py` is unchanged: proxy and prod must each reach >= 1.0x beat on the gb figure (r16 reads ~0.74x).

Rules from the first gb round on:
1. Rank on proxy/prod `tflops` (gb). `blk_*` is context: a candidate that wins blocked but not gb is not a win.
2. Every ranking process: candidate + incumbent + `--aa <incumbent label>` + beat, then a second process in reversed
   arm order. A difference smaller than max(0.7%, 2 x |1 - aa_ratio|) is noise (gb per-call IQR is ~2-3% at prod).
3. The incumbent is `job_context/op/current` -- measure it as `--arms current`. After a refactor round whose own
   candidate is rejected, `rounds/<best_round>/op` holds that rejected candidate, NOT op/current: rounds 17-19
   re-measured `rounds/016/op` (g61) as "champion_round 16" while op/current was r13ns. Report `champion_tflops`
   from op/current.
4. h48 repointed every champion to round 21 (= op/current after h48: the r19 tree, or round 20's tree if round 20
   was accepted -- see h48's A0 amendment). Every TF/s in state.yaml / progress.md / findings / hints before the
   first gb round is blocked-ruler (and B0 before round 20): never compare a gb figure with them.
5. Cost: ~25 ms per gb timed call at prod (burst 21.5 + call 1.5 + host ~2): a 4-arm prod process takes ~11 s
   longer than before. Profiling runs: `--ruler blk` for cycle counts, or `--ruler gb --gb-iters 5` to see the
   kernel at the training clock (the GEMMs then appear in the trace too).
6. What moves the gb figure: issued work and power -- the m32x8 hot-loop instruction count (h47 item 2), v_nop /
   s_delay, SALU bookkeeping, redundant waits, barriers per KV tile. Data-dependent candidates still also need the
   real q/k/v dumps (h45).
