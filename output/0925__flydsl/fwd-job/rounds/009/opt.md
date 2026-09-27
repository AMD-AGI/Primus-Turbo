# Round 9 (fast) -- opt: reading, instruments, choice, route

Written as the round went. Container `fa-repro`, GPU0 (single device). Before the first run:
`rocm-smi --showpids` "No KFD PIDs", `--showuse` 0%, no python in the container. A dmesg monitor was armed
for amdgpu fault/hang/reset/MES lines for the whole round.

## 1. Reading

- findings: `facts.md`, `dead_ends.md`, `pool.md`, `route.md` (operator tables h2-h27), `rounds/008/1-opt/opt.md`,
  `rounds/008/2-reflect/{reflect.md,reflect.yaml}`, `rounds/008/gate.log`, `job_context/progress.md`, `note.md`,
  `rounds/006/1-opt/opt.md` s3 (the penalty instruments and their scripts).
- Round-5 profile: not re-opened beyond what facts quotes. Its counter surface is 13 live counters on this card
  (`knowledge/arch/gfx1250/profiling-surface.md`); the only one that separates the two states that matter this
  round is `SQC_ICACHE_MISSES`, which round 6 already took. Nothing else in that directory bears on it.
- **What decides acceptance now.** The gain the ledger records is the geomean over fast/proxy/prod of the
  candidate against the re-measured incumbent, with beat in the same process (palindromic order). Round 8's
  nodelay was +0.7% at prod and still lost the round (gain 0.9746) on fast 0.961 / proxy 0.956. Round 6 measured
  the beat-preceded penalty at fast +30-34 us (1.8x), proxy +47 us (+31%), prod +34 us (+1.7%). The same
  arithmetic run the other way: removing that penalty would be worth far more than any body lever this job has
  priced (softmax, barriers, flags are all <= 5%). Round 8's reflect says the same. So this round spends its
  instrument on the penalty, not on another body census.
- State: pool is effectively empty (g03/g04/g16 closed awaiting retirement, g15 an expected loss alone, g19 blocked
  on a depth-3 ring priced at +3.3% instr/tile, g21 lost). Last four rounds not accepted. Sources (c)/(d) are
  therefore mandatory; see s3.

## 2. Look

### 2a. Survey

`rocprofv3 --stats` not re-run: rounds 5-8 recorded that it writes no files on this stack, and `op/current` is
byte-identical to the round-5 tree (one kernel per call). The question this round is not "which kernel" but
"what does the predecessor do to it", which `--stats` cannot answer.

### 2b. r9.i1.g24 -- instrument: which predecessor puts current into the slow state? (`raw/foreign/`)

Round 6 showed the penalty needs *running* beat (importing it is not enough) and that I-cache misses go 0 -> 808
per dispatch. It never asked whether beat is special. If any other kernel image does the same, the mechanism
is generic I-cache eviction and the lever is current's cold-start cost; if only beat does it, the mechanism is
something beat leaves behind (VGPR mode / 1024-VGPR waves, a queue or SPI config, clocks) and code-side levers
are pointless.

Pattern (fast, 101 reps, 30 warm-up pairs): `sync; <pre>; _sleep(1e5); e0; current(causal); e1`. Arms:
`none`; `beat`; `self_nc` (current's own non-causal kernel, a different ~same-size FlyDSL image);
`g13` (round 5's rejected candidate, 5906 instructions, 1.39x current's image); `g13x2` (g13 causal +
non-causal, two foreign FlyDSL images, ~2.8x); `beat_cur` (beat, then one current, then the timed current).
Two passes.

**Prediction before running:** `self_nc` near none (the two images together fit), `g13x2` penalised (it is larger
than beat's 79 KB? -- no: ~2 x 47 KB = 94 KB, larger), `beat_cur` near none (one call rewarms). If `g13x2` stays
at none while beat penalises, the eviction story is wrong.

(The bracketed "-- no:" in the prediction is left as written: I first mis-sized g13x2 and corrected it in the
same sentence, before the run.)

**Result** (`raw/foreign/out.txt`, two passes, fast current device time median / p10 / p90 us, rc=0):

| pre | pass 1 | pass 2 |
| --- | --- | --- |
| none | 39.3 / 39.1 / 41.7 | 41.0 / 40.6 / 41.5 |
| beat | **76.4** / 45.9 / 77.6 | **80.3** / 44.6 / 81.0 |
| self_nc (own non-causal image) | 39.7 | 40.9 |
| g13 (1.39x image) | 40.0 | 40.1 |
| g13x2 (two foreign images, ~94 KB) | **50.7** / 47.2 / 51.6 | **50.0** / 46.1 / 50.9 |
| beat_cur (beat, one current, then timed current) | **72.2** / 71.0 / 73.2 | **68.0** / 66.6 / 69.3 |
| none (closing) | 39.9 | 39.6 |

Reading:
- Reproduced: beat before current costs +37-39 us at fast (1.9-2.0x).
- **One intermediate current call does not re-warm.** `beat_cur` is as slow as `beat`, and more consistently
  so (its p10 is 67-71 us where `beat`'s is 45). Under the round-6 story (beat's image evicts current's from the
  I-cache, current refetches) the intermediate call would have refilled the cache on the WGPs current uses. It
  did not help. That is the first measurement against the I-cache story.
- A foreign FlyDSL image larger than beat's (g13x2, ~94 KB vs 79 KB) costs +10 us, a quarter of beat's. g13 alone
  (47 KB) and the own non-causal image cost nothing. So code eviction exists but is not the size of beat's penalty.
- My prediction held for self_nc and g13x2's direction, and failed for beat_cur. The failure is the useful part.

Two stories remain: (i) I-cache eviction plus WG placement that moves between dispatches (the intermediate call
warms other WGPs), (ii) a power/current-limit state beat leaves behind that decays with time (round 6 saw a decay
with gap that eviction cannot produce). Instrument 2 separates them with predecessors that share current's code.

### 2c. beat's kernel descriptor (`raw/kd/`)

`fmha_bf16_pertokenBf16_hd128_128x256_mask.co`: RSRC3 = 0x00000ff0, i.e. INST_PREF_SIZE = 255 (the field max),
LDS 327680 (same as current). Current's descriptor sets `instprefsize(image)` = 216 per round 3. Nothing in the
descriptor that current lacks, as far as RSRC3 goes; RSRC1 = 0xc00c003f.

### 2d. r9.i1.g24 instrument 2 -- predecessors that share current's code but carry power (`raw/power/`)

**Prediction:** if the penalty is a power/current-limit state, `cur_prod` (2 ms of current's own dense WMMA,
same code image) and `beat_prod` should both penalise, `cur_prod` at least as much as beat at fast. If it is
code, `cur_*` arms stay at none. I expected power, because of instrument 1's `beat_cur` and round 6's decay.

| pre | pass 1 med / p10 | pass 2 med / p10 |
| --- | --- | --- |
| none | 40.6 / 39.8 | 39.8 / 39.1 |
| beat (fast) | **72.5** / 45.9 | **68.0** / 50.2 |
| cur_x3 (3 current fast calls) | 39.2 | 39.1 |
| nc_x2 (2 non-causal current calls) | 38.9 | 39.3 |
| cur_proxy (current at proxy, ~150 us) | 38.6 | 39.3 |
| cur_prod (current at prod, ~2 ms dense WMMA) | 39.1 | 38.9 |
| beat_proxy | **67.6** / 51.1 | **71.0** / 50.9 |
| beat_prod (~1.5 ms) | **70.3** / 50.4 | **68.1** / 50.7 |
| none | 40.1 | 40.3 |

**My prediction was wrong. It is not power.** Two milliseconds of current's own dense WMMA before the timed
call cost nothing (38.9-39.1 us). Beat costs the same +28-33 us whether it runs for 32 us or 1.5 ms. The penalty
is a property of *beat having run*, not of how much work or power ran. The amount of beat work does not matter.

### 2e. instrument 3 -- does a full-grid current call clear it, and do I-cache misses follow it? (`raw/evict/`)

`beat_curprod` = beat, then current at prod (4096 WGs, every WGP runs current's code many times), then the timed
fast call. `beat_sl2m` = beat, then `_sleep(2e6)`. PMC (`SQC_ICACHE_MISSES`, `SQ_BUSY_CYCLES`) on the timed dispatch.
**Prediction:** if eviction, `beat_curprod` is at none and its misses are ~0; if a beat-left state, it stays slow.

**Result** (`raw/evict/out.txt`, `raw/evict/pmc_summary.txt`; timing two passes, PMC one pass of 31 timed dispatches
under rocprofv3, which serialises and inflates absolute times):

| pre | fast us pass 1 / pass 2 (med) | `SQC_ICACHE_MISSES` / timed dispatch | `SQ_BUSY_CYCLES` |
| --- | --- | --- | --- |
| none | 40.1 / 40.0 | **0** | 12.10 M |
| beat | 72.9 / 73.0 | 808 | 20.76 M |
| beat_cur | 75.5 / 69.3 | **5744** | 19.87 M |
| beat_curprod | **38.7 / 38.8** | **1** | 12.37 M |
| beat_sl2m (beat; 2 ms sleep) | **83.5 / 84.0** | 404 | 24.57 M |

Reading:
- **A full-grid current call clears the state completely** (38.7 us, 1 miss). Two milliseconds of *time* does
  not; it makes it slightly worse (84 us). Round 6's "decays with gap" does not reproduce under a
  controlled sleep. The state is spatial (per WGP), not temporal. It is cleared by running current's code on
  every WGP and by nothing else tried.
- **The miss count does not price the penalty.** 404, 808 and 5744 misses all cost +30-44 us; 0-1 misses cost
  nothing. The relation is binary. That is what one expects if the kernel's time is the slowest WG's time
  (fast is one dispatch wave of 32 WGs) and every WG that starts on a WGP without current's code pays the same
  large cold-start cost. It then does not matter how many WGs were cold, only whether any were. The
  intermediate 32-WG call in `beat_cur` warms some WGPs and the timed call lands on others (5744 misses says the
  timed call found many cold WGPs; the WG-to-WGP placement evidently moves between dispatches).
- This also explains the other two shapes: at proxy and prod a WGP is cold only for its first WG, so they each
  pay roughly one cold start on the critical path (+47 us and +34 us in round 6, the same order as fast's +33).
- **What remains unexplained is the size of one cold start: ~30-40 us.** Current's image is 27.6 KB. A cold
  start that cost the fetch of every line serially at ~140 ns would give ~30 us, but round 6's g13 build
  (1.39x image) paid only +3 us more, so the serial-per-line model is not confirmed either. A per-WG
  timestamp instrument inside the kernel would say where the 30 us goes (see pool g25).
- Why g13x2 costs +10 and beat +33: both evict, but on different sets of WGPs; not resolved here.

### 2f. instrument 4 -- does beat pay a cold start of its own? (`raw/beatcold/`)

Timed beat (fast), with the predecessor varied. Two passes, median us:

| pre | pass 1 | pass 2 |
| --- | --- | --- |
| none (beat after beat) | 34.3 | 34.3 |
| cur | 30.2 | 30.3 |
| g13x2 | 31.4 | 31.3 |
| cur_prod | 30.0 | 30.0 |

**Beat pays no cold-code cost.** After foreign code it is *faster* by about 4 us than back-to-back beat. The cold
start is specific to current.

### 2g. instrument 5 -- is current's cold start fixed or loop-proportional? (`raw/seqsweep/`)

The grid is held at 32 WGs (sq/256 x hkv with GQA 4: hkv = 2048/sq) while each WG's KV tiles grow 8x, from 256 to
2048. `pre` is none or beat (fast inputs). Two passes, median us:

| sq (tiles/WG x) | warm | cold (beat before) | delta |
| --- | --- | --- | --- |
| 256 (1x) | 20.9 / 20.4 | 56.6 / 55.4 | +35 |
| 512 (2x) | 27.2 / 27.0 | 69.2 / 68.5 | +42 |
| 1024 (4x) | 39.9 / 45.2 | 73.9 / 73.1 | +34 / +28 |
| 2048 (8x) | 65.9 / 65.2 | 90.7 / 90.6 | +25 |

**Prediction:** if the cost sits in the loop, delta grows with tiles. If it sits in the prologue or first touch,
delta is flat.
**Result:** flat to falling (+25..+42 us) over an 8x range of loop work. It is a fixed, one-time cold start per WG
start, not a per-tile cost. The p10 of the cold arms (31-47 us) is sometimes near warm, which is consistent with
the WG-placement story in 2e.

### 2h. compile census: `amdgpu-loop-prefetch=True` (`raw/cc/lpf`, `isa_lpf`)

Beat has `s_prefetch_inst` in its loop. The flag was added to both `llvm_options` dicts. It compiled with rc=0,
and the ISA is the same as ctl: 4105 instructions, VGPR 456, SGPR 98, the same p2align count, and **no
`s_prefetch_inst`**. The flag is inert on this stack. Closed as a census; no pool entry, no card time. No build
failures this round.

## 3. Mid-round change: h28 and h29 arrived in route.md

While I was writing up, the operator tables in route.md gained two items:
- **h28** (standing): candidate vs champion is measured without beat in the process.
- **h29** (must): re-land round 8's nodelay.

This changes what the round's instrument is worth. The post-beat cold start (s2b-s2g) is no longer part of the
score. I keep the findings because they explain *why* h28 is right. The penalty is a fixed, spatial,
current-specific cold start, and no body change moves it (nodelay only looked worse after beat). But the route no
longer spends an arm on it: g26 goes to the pool at low priority.

## 4. Corpus (sources c/d) and the choice

Consulted this round (explored.consulted):
- `knowledge/optimization/routes/1-metrics-to-techniques.md` ("When no metric is elevated", "Before a candidate
  becomes a plan")
- `knowledge/optimization/techniques/6-gfx1250-cdna5-mechanisms.md` (s0, s3 split barrier, s4 lock_simd, s5 LDS
  segments, s6 reuse hints, machine behaviours)
- `knowledge/backends/flydsl/attention/techniques.md`, `dead-ends.md` (headings; no cold-start or lock_simd entry)
- `knowledge/backends/flydsl/attention/recipes/hd128.md` (headings; gfx950 structure, nothing new for fwd body)
- `knowledge/pitfalls/measurement-traps.md` ("A probe whose carrier has a different regime", cold vs warm)
- `knowledge/arch/gfx1250/profiling-surface.md` (13 live counters, carried from s1)
- flydsl 0.3.4.1 `flydsl/_mlir/dialects/_llvm_ops_gen.py` (`inline_asm` exists: g25 is expressible)
- op/current `flydsl_fwd/fmha_b16_buffer_managers.py:117-138` (ENABLE_SCHED_MODE2 and the DEP_MODE setreg)
- current ISA and beat `.co` disassembly: both write only `hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2`

Decisions:
- **h29 (must) is row 1.** Re-measure `rounds/008/op` per h28 and ship it. Nothing to build.
- **r9.i2.g25 lock_simd is row 2 and the round's idea.** It is the one corpus mechanism that:
  - applies at 2 waves/SIMD (the corpus says it is worth nothing at 1, and beat is at 1, so beat's absence of it
    tells nothing);
  - adds zero loop instructions (r8's rule: loop-lengthening has never won);
  - is one line;
  - is untried here.
- LDS segment separation (s5) was considered and not taken. It was measured at 0 in the corpus at a large tile.
  Current's LDS is already 327680 B and fully used, and no LDS-port evidence exists on this card.
- r7.i3.g19 stays third. g26 (the cold-start instrument) is fourth.
- Pool additions: r9.i2.g25, r9.i3.g26. r9.i1.g24 (this round's instrument) is done and recorded here, not pooled.

**Expected number.** The selected change is row 2, g25 (row 1 is a re-measure). My estimate:
- **Prod: +0.5%** vs op/current, range -1% to +1.5%.
- Fast/proxy: about +0.3%. At fast the kernel is one dispatch wave, and cold-start and launch cost dilute any body
  change.
- Geomean about +0.4% alone.

Stacked on h29's nodelay (prod +0.7%, fast/proxy about 0 without beat), the merged tree would read about **+0.6-0.7%
geomean**, prod about +1.2%. That is at the 0.70% bar, not safely over it.

The reasoning: the corpus's +1.9% is at 4 waves/SIMD on a GEMM whose inner loop is a pure WMMA run. Current has
half the co-resident waves, and only about half its loop is WMMA (the rest is softmax VALU/trans, h4). So I scale
by about 1/4, and I leave the sign open, because the LO/HI ping-pong may depend on the post-WMMA yield to hand the
SIMD to the softmax wave.

Falsified if prod is within the 0.47% floor or negative in 3 rotated sessions.

## 5. Build (route rows 1 and 2)

Three trees, all from op/current:
- `raw/arms/nodelay`: a byte copy of `rounds/008/op` (h29; 2 lines, `amdgpu-enable-delay-alu=False` in both
  `llvm_options` dicts).
- `raw/arms/lock`: op/current plus g25. That is one `llvm_dialect.inline_asm(None, [], "s_setreg_imm32_b32
  hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1", "", has_side_effects=True)` at the top of the bshd kernel body (the
  entry every scored shape uses; the thd entry is untouched).
- `rounds/009/op` (working copy): nodelay + lock, the merge. It is measured beside both arms in the same processes.

I cleared the compile cache (`rm -rf /root/.flydsl/cache` in the container) before the census and again before
the measurement script.

Compile census (`raw/cc/isa_{lock,merged}`, prod bshd, rc=0 both):

| tree | instructions | VGPR | SGPR | SCHED_MODE writes |
| --- | --- | --- | --- | --- |
| ctl (isa_lpf, = op/current) | 4105 | 456 | 100 | `(0,2)=2` |
| lock | 4106 | 456 | 100 | `(0,2)=2` at line 10, `(2,1)=1` at line 61 (prologue) |
| merged | 3936 | 456 | 100 | same as lock |

The census condition from the route holds: exactly one new prologue instruction, no loop change, the final
SCHED_MODE value is 6. No build failures.
