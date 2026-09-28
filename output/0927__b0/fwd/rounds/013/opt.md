# Round 13 opt (fast round, GPU 2 / fa-g2) -- route design

op/current = round 11 (r12 not accepted). Same-session ratios to beat (r12): fast 1.001, proxy 1.019, prod 0.968.

## What I read
- findings: facts.md, dead_ends.md, pool.md, route.md; hint.md index + h16/h36; rounds/012/2-reflect/reflect.md;
  rounds/010/1-profiling/profiling_summary.md (round 10, before round 11; prod ISA byte-identical since, so its prod
  counters still describe the prod kernel).
- h16 still reads "No atomics, no split-k"; hint.md unchanged since 04:04. No ruling on r5.i3.g15 exists.

## Survey decision
- `rocprofv3 --stats` NOT re-run: its kernel-trace hangs/records zero FlyDSL kernels on this box (closed
  instrument r9.i3.g22). Durations come from benchmark.py, identity from --pmc rows. Re-running it would only
  re-prove g22.
- GPU 2 checked idle before the one device-touching step (rocm-smi: 0% use; one UNKNOWN KFD pid, 0 VRAM).

## Instruments taken this round (no op code touched)

### i-census -- the prod hot clean tile, static (raw/prod_hot_tile_census.txt)
Prediction before reading: "there is still >= 10% non-algorithmic VALU/SALU in the clean tile" (h4's v_nop 156 /
SALU 188 per 256 KV). **Wrong.** The speculative clean tile (.LBB0_3 -> .LBB0_11, 409 instructions per 64-key tile
per wave) is: 64 WMMA, 64 v_exp, 32 v_pk_fma (exp argument), 32 v_cvt_pk_bf16, 31 v_pk_add (row sum), 64 ds_load
(K 32, V-tr 32), 32 s_set_vgpr_msb, 17 v_nop, 9 s_wait_dscnt, 8 s_delay_alu, ~30 SALU, 2 permlanex16, 2 TDM.
The VALU is at the algorithm's floor (exp 1/elem, arg 0.5, cvt 0.5, sum 0.5). What is left is issue-only
(msb/nop/delay/SALU, ~90 slots) and on prod's power-bound card that is exactly the class dead_ends says returns
nothing ("cycles through issue density"). **Conclusion: instruction removal at prod is exhausted on the clean tile.**
The one remaining VALU term with a structural alternative is the exp argument fma -> next instrument.

### r13.i1.g50 -- Q pre-scale by scale*log2e so the QK WMMA takes -m*log2e as C (kills 32 pk_fma/tile): priced by emulation, DEAD
Mechanism: with the stale max, m is constant over a tile, so C = broadcast(-m') could be held in 16 VGPRs and
refreshed only on the slow path; S' = Q'K - m' comes out of the WMMA and the per-element fma disappears.
Cost: Q' = bf16(Q * scale * log2e) adds one bf16 rounding to every logit.
Emulation (raw/emu_qprescale.py, .txt; gate's own inputs seed 0, reference = op/eager, CPU fp32):
base-emulation o SQNR 52.2-54.1 dB, prescaled 50.2-51.9 dB: **-2.1..-2.4 dB on every edge shape**; lse 145 -> 80-85 dB.
Adding that error power to the real kernel's worst case (short_q full, 49.82 dB): ~48.5 dB < 49 dB gate. Dead by the
gate before a build. (Prediction was "costs < 1 dB"; wrong by ~2x.)

### Fast premise re-read (arithmetic, not measured)
r12's fit: P = 0.49 us per 64-key tile at ~2.3 GHz ~= 1130 cycles, one wave per SIMD (m32x2, 2 of 4 SIMDs idle).
64 WMMA 16x16x32 bf16 per tile per wave; at ~16 cycles each (inferred from peak, NOT measured) that is ~1024 cycles.
So the per-tile time is most likely the ONE SIMD's WMMA pipe, not VALU latency or TDM. Two consequences:
- in-wave softmax/QK pipelining (h6/L6) could buy at most the ~10% non-WMMA residue at fast;
- the lever is to spread the heaviest q-tile's WMMA over the idle SIMDs. Split-KV (g15) does it but needs the h16
  ruling. New: r13.i2.g51 does it WITHOUT a combine (d-split, redundant QK), bitwise identical to op/current.

## Decision
- Route: musts h28/h34/h37 discharged (as r12). Row 4 = r13.i2.g51, gated by its own throwaway probe
  (m32x2 with half the PV d-tiles skipped, wrong output): if fast does not move >= 8%, stop before the 4-wave build.
  Row 5 = r5.i3.g15, unchanged condition (h16 ruling).
- Expected from g51: fast 13.7 -> ~11.8-12.5 us (x1.10-1.16) if the chain is single-SIMD WMMA-bound and LDS keeps up
  (4 waves read K twice: 1.5x LDS reads per CU per tile). Could be <= x1.0 if LDS bandwidth binds. proxy/prod
  untouched (gated, ISA must stay byte-identical).
- Nothing built this round. No arms.

## Consulted
knowledge/INDEX.md; backends/flydsl/attention/README.md; backends/flydsl/attention/recipes/hd128.md (s5, b5 MFMA row
sum); backends/hipkittens/attention/recipes/gqa_d128.md (section index, s6 headings); optimization/routes/1-metrics-to-techniques.md (A5/A7 and the power-wall paragraph: "fewer bytes at full value, fewer instructions at 20-45%").

---------------------------------------------------------------------------------------------------
# Build and measure (turn 2)

Compile cache cleared before the first build (`/tmp/flycache`, inside fa-g2).

## Route row 4 -- r13.i2.g51 probe gate (raw/probe)
Throwaway arm `arms/probe` = op/current with `_pv_gemm` skipping the PV WMMA for d-tiles 4..7 (wrong output).
ISA (raw/isa, fresh compile): WMMA 320 -> 256 in the fast kernel, so the probe really dropped 16 of 64 WMMA per tile
per q-tile pair (PV half).
fast only, 3 rotated sessions, same process per session, PYTHONHASHSEED=0:

| session | r11 | champ2 (A/A) | probe | probe/r11 |
|---|---|---|---|---|
| s1 (r11 probe champ2) | 156.44 | 155.53 | 169.01 | 1.080 |
| s2 (probe champ2 r11) | 155.53 | 155.09 | 168.74 | 1.085 |
| s3 (champ2 r11 probe) | 155.53 | 155.08 | 169.27 | 1.088 |

A/A (champ2 vs r11) <= 0.6%. Gate was ">= 8% at fast"; passed, narrowly (+8.0..+8.8%). Built g51.
Reading: removing 25% of a tile's WMMA bought ~8% of the kernel -> the per-tile chain is WMMA-bound only in part
(F + 16P with P ~0.49 us; 16 WMMA x ~16 cyc ~ 0.11 us/tile of 0.49 -> ~23% of P, ~8% of total at fast). The
g51 ceiling is therefore about the probe, minus what 4 waves cost (redundant QK/softmax on the idle SIMDs, 2x K
LDS reads).

## g51 build (rounds/013/op, one file: flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py)
- NUM_WAVES 2 -> 4; NUM_ROW_WAVES = 2; D_SPLIT = 2; BLOCK_M = 16*2*NUM_ROW_WAVES = 64 (unchanged).
- row_w = warp % 2 picks the 32 q rows (QK + softmax, redundant across the pair); dh = warp // 2 picks d half.
- Q manager built for 2 row waves; each d-half gets its own private Q copy in slot 1 (no two waves TDM-write the
  same bytes); slot_bytes = max(K|V, 2*Q) (unchanged value at d=128).
- K/V TDM loads split over all 4 waves (managers already accept 4).
- V transpose loads: the per-lane base is offset by dh*64 elements; `_load_v_half` reads 4 d-tiles (same order).
- O accumulators and `_pv_gemm` over 4 d-tiles; OManagerV3 built for v_hdim 64 / 2 row waves, o_base_elems =
  dh*64, own LDS region per d-half.
- LSE stored by dh == 0 only.
- zero-fill path (kv_len == 0): the LSE loop now masks tid < BLOCK_M (128 threads, 64 rows).
- Nothing in the m32x8 file or the shared managers changed, so proxy/prod are untouched by construction
  (`diff -rq job_context/op/current rounds/013/op`: this one file only).
Build failures: none.

ISA of the fast kernel (raw/isa, fresh compile per arm; raw/isa/code_size.txt via llvm-mc):

| arm | next_free_vgpr | spill | LDS | WMMA | instr | code bytes |
|---|---|---|---|---|---|---|
| champ2 (= op/current) | 513 | 0/0 | 160 KB | 320 | 4786 | 29 892 |
| probe | 513 | 0/0 | 160 KB | 256 | 4639 | 28 664 |
| g51 | 348 | 0/0 | 160 KB | 256 | 4779 | 28 064 (< 32 640 cap) |

## Correctness (raw/build1, raw/measure)
- ut/test_correctness.py: PASS, every row; min o SQNR 49.82 dB (short_q full), the same as op/current.
- armcheck (r12's check.py): bitwise equal to op/current on all 22 cases (sq != skv, 300x1000, non-causal,
  custom scale, 1x1, gqa 1/4), fresh outputs.
- validation.py (raw/measure/validation.txt): every correctness row PASS at 49.0 dB (min 49.82, short_q full;
  fast 51.23, proxy 50.89, prod 50.83), determinism 200/200 bitwise. rc = 2 on the speed bar only, as op/current:
  geomean vs in-process beat 0.9859 (fast 1.0474, proxy 0.9476, prod 0.9654).
- verify_r6 adversarial suite (raw/verify, r11's g20_check.py: reference = op/current forced into m32x8):
  A 16/16 ut rows bitwise (14 take m32x2); C 32 kinds x {toy, short_q, gqa4} x {causal, full} = 192/192 bitwise,
  0 non-finite. rc 0.

## Measurement (raw/measure, launched 05:40Z)
Protocol: fa-g2 (physical GPU 2), default user, PYTHONHASHSEED=0, blocked ruler; 3 sessions, each one ranking
process with r11 (incumbent), r10, g51 (= rounds/013/op), champ2 (byte copy of op/current, A/A) in rotated order;
beat alone in its own process after (h31). rocm-smi before/after: GPU use 0% / 2%, no process of ours alive after.
dmesg: no new amdgpu line during the run (only the pre-existing MES messages from ~3.8 h before).

TFLOP/s, mean of the 3 session medians (raw/measure/summary.txt):

| shape | r10 | r11 (incumbent) | champ2 (A/A) | **g51** | beat (own process) | g51 / best ever | g51 / beat |
|---|---|---|---|---|---|---|---|
| fast | 142.46 | 153.61 | 153.97 | **162.27** | 153.75 | **1.0564** (r11) | 1.0554 |
| proxy | 1610.36 | 1605.31 | 1606.60 | 1606.33 | 1582.50 | 0.9975 (r10) | 1.0151 |
| prod | 1492.44 | 1502.37 | 1502.70 | 1502.81 | 1556.58 | 1.0003 (r11) | 0.9655 |

g51/r11 per session: fast 1.0544 / 1.0547 / 1.0602 (3/3); A/A champ2/r11 fast 1.006 / 0.997 / 1.004.
proxy/prod run the untouched m32x8 kernel; their g51 deltas are inside the A/A spread (proxy +-1.5%).
fast latency 13.99 -> 13.25 us.

Acceptance (all three must hold):
1. throughput vs best ever, re-measured here: score (capped mean of min(x/beat,1)) g51 0.9885 vs r11 0.9881, r10
   0.9618; uncapped g51 1.0120 vs r11 0.9929. Holds (the capped gain is small because fast crosses its target and
   caps at 1.0).
2. every below-target shape >= 95% of its own best: prod 1.0003 of r11. Holds.
3. no shape that had reached its target falls below it: proxy 1606.3 >= 1582.5. Holds. fast now reaches its
   target too (162.3 >= 153.75; also above validation's in-process beat 156.4).

## Verdict
r13.i2.g51 **delivered**, left in rounds/013/op. fast x1.056 (3/3), proxy/prod unchanged, output bitwise equal to
op/current.
Expected before building (route row 4 / pool): "fast x1.10-1.16 ... x<=1.0 if LDS bandwidth binds"; the probe then
capped it at ~x1.08. Got x1.056: about 2/3 of the probe's gain survives the 4-wave cost (redundant QK/softmax on
the other SIMDs, K read by 4 waves, 2x Q copies). What that says: fast's per-tile time is only ~1/4 PV WMMA; the
rest (QK WMMA + softmax + LDS) is still one chain per q-row group, and it is now the critical path.
Route row 5 (r5.i3.g15) not executed: still blocked on the h16 ruling (hint.md unchanged). Note for its re-price:
g51 now occupies all 4 SIMDs at fast, so g15's "idle SIMDs" premise is spent; on this body g15 would have to split
the QK+softmax chain that g51 leaves serial.
