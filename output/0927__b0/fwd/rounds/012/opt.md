# Round 12 (fast) -- opt

Runner: container fa-g2 (physical GPU 2, PCI 0003:04:00.0). Every build/measurement ran there, detached
with rc sentinels, rocm-smi idle-checked before and after. dmesg: no new amdgpu messages on 0003:04:00.0
during the round (the last one is 2026-09-27 21:52, before it).

**Card intrusions (not mine, not killed):**
- 04:49:59: the bwd job's `benchmark.py` (pid 2931708, root, gfx1250-flydsl-attn-bwd rounds/027) registered on
  our card. That overlapped the last ~43 s of session s3 and all of raw/fastfit2. Both were re-run on an idle
  card (raw/rerun, a KFD snapshot after every step, 0 foreign pids). They reproduce: fit F 6.88 / P 0.485 vs
  6.81 / 0.486, and s3b is null like s3.
- 05:02:36: an E2E Primus `torchrun` training run (pids 3004956/3005291) started on our card. It overlaps only
  the last ~6 s of s3b (prod tail, which is within A/A). No GPU work of mine ran after that.

## 1. Survey (raw/survey, blocked ruler, beat in its own process)

| shape | champ | champ2 (A/A) | beat | champ/beat |
|---|---|---|---|---|
| fast  | 154.19 | 154.63 | 154.63 | ~1.00 |
| proxy | 1602.08 | 1607.33 | 1536.09 | ~1.04 |
| prod  | 1504.91 | 1504.47 | 1555.85 | 0.967 |

A/A spread 0.3% (fast), 0.3% (proxy), 0.03% (prod).

## 2. Instrument 1 -- what sets fast's time? (raw/fastfit, then raw/fastfit2)

Prediction: fast is bound by the heaviest causal q-tile's KV chain (t ~= F + 16P), which would price
intra-WG split-KV.

- **First pass (raw/fastfit, no L2 flush), contaminated.** The fit was t = 19.5 us + 0.48 us per 64-key tile,
  so F looked like ~70% of fast, and beat (aiter ASM) showed the same ~20 us floor. That floor was the
  host path (section 3), and it is not on the scored ruler.
- **Corrected pass (raw/fastfit2), with the ruler's 256 MB `flush.zero_()` before every timed call:**

| op/current, b1 sq1024 hq8 hkv2 d128 | us |
|---|---|
| non-causal skv 128 / 256 / 512 / 1024 / 2048 / 4096 | 7.92 / 8.67 / 10.66 / 14.58 / 22.30 / 37.92 |
| fit | **F = 6.8 us, P = 0.486 us per 64-key tile**; clean re-run (raw/rerun) F 6.88, P 0.485. Beat: F 8.13, P 0.495 |
| causal fast (scored shape), 128 WGs | 13.70 (ruler: 13.82-13.90) |
| causal, same per-WG work, 256 / 512 / 1024 WGs | 14.28 / 15.97 / 21.81 |

Causal fast ~= non-causal skv 1024 (F + 16P), and doubling the grid costs +4%. **The prediction holds:**
fast is the heaviest q-tile's 16-tile chain (~7.8 of 13.7 us), with half the SIMDs idle. This prices
intra-WG split-KV (16P -> 8P + combine) at fast x1.15-1.20.

## 3. Instrument 2 -- the host path, and why the ruler doesn't see it (raw/hostpath)

Measured in fa-g2, no profiler, 3000-call loops:

| layer | us/call |
|---|---|
| empty event pair / trivial torch kernel (idle GPU) | 3.3 / 5.9 |
| `attn_fwd` total CPU (== back-to-back throughput) | 15.4-17.0 |
| impl wrapper | ~0.8 |
| `flash_attn_batch_m32x8` Python body (current_stream 2.0, 2x torch.empty 2.0, 17 strides, asserts) | ~8.5 |
| FlyDSL `CompiledFunction` (31 ABI slot fills) | 6.4 |
| of which the raw `func_exe(packed)` launch | 1.83 |

I first read benchmark.py's window (`ev0.record(); call(); ev1.record()`) as opening on an idle GPU,
and built section 4's arms on that. **That was wrong.** `timed()` issues `flush.zero_()` (256 MB, tens of
us of memset) just before `ev0.record()`. The host dispatches the whole call while the flush still
occupies the GPU, so ev0 fires after the host is done. The scored fast time is 13.9 us, not 26 us. The
arms were built before I caught this, and the ruler confirms it (section 5).

## 4. Ideas and arms

Candidates weighed (sources: pool/route, survey, instruments, corpus):

| idea | price | decision |
|---|---|---|
| r12.i1.g47 impl-level host fast path (shape-keyed) | instrument 2: ~ -6 us CPU/call | **arm A** |
| r12.i2.g48 FlyDSL launch slab (re-marshal only on change) | instrument 2: <= -4.6 us, alone ~ -1 us | **arm B** |
| intra-WG split-KV for fast (keeps deep-born id r5.i3.g15) | instrument 1 (flushed): fast x1.15-1.20 | not built: h16 "no split-k" needs an operator ruling first; route row 2 |
| h23 TDM PP3 / unroll x2 | latency on the m32x2 body; prod power-bound | not taken this round; route row 3 |
| deferred peer permlane in speculative d-update | ~0.2-0.5% prod, under min_gain | not taken |
| n_block=128 m32x2 | per-tile overhead is 30% of fast at most | not taken |

Arms built alone from op/current (byte copy `arms/champ2`), then the merge (neither lost):

- **A (g47)** `impl.py`: cache `(shapes, dtypes, device, softmax_scale, causal) -> (CompiledFunction,
  out/lse shapes, scalar tail)` after one normal call; a hit does 3 `is_contiguous`, 2 `new_empty`,
  `torch._C._cuda_getCurrentRawStream`, `cf(*args)`. Any miss / non-contiguous input -> original path.
- **B (g48)** `flydsl_fwd/tensor_shim.py::_run_compiled`: own ABI storages built from FlyDSL's slot spec;
  pointer + stream slots refilled every call, descriptor + scalar slots only when (scalars, tensor
  shapes, tensor strides) differ from the previous call. Thread-guarded (other threads -> `cf`),
  falls back to `cf` if FlyDSL's internals differ.
- **M = A + B**: A's hit calls B's slab.

Pre-ruler checks (raw/armcheck, 11 shapes x 2 reps incl. non-causal, sq != skv, hkv == hq, custom scale,
1x1, shape interleaving): all three arms **bitwise equal** to op/current, repeat-call bitwise, fresh
output tensors. CPU us/call at fast: ref 15.3-17.0; A 10.19; B 13.94; M **7.25**.

Predicted ruler (made before I found the flush, so wrong): fast A +30%, B +6%, M +40%; proxy +7-9%; prod +0.4%.
Re-predicted once the flush was found, before the sessions were read: all null.

## 5. Results (raw/measure; validation per arm, then 3 rotated no-beat sessions)

validation.py: A, B and M all pass correctness (every shape >= 49 dB) and determinism (200/200 bitwise).
All three fail only the speed bar (geomean vs in-process beat >= 1.0), which op/current fails too (round 11
score 0.994). Fast candidate/beat is 0.997 / 0.994 / 1.0015.

Ratio vs champ (champ2 = byte copy, the A/A):

| session (order) | shape | champ2 | A | B | M |
|---|---|---|---|---|---|
| s1 (champ champ2 A B M) | fast | 1.0000 | 1.0030 | 1.0000 | 1.0058 |
| | proxy | 1.0000 | 0.9986 | 1.0077 | 1.0137 |
| | prod | 0.9995 | 1.0009 | 0.9993 | 1.0008 |
| s2 (A B M champ champ2) | fast | 1.0029 | 1.0000 | 0.9914 | 0.9943 |
| | proxy | 0.9965 | 1.0023 | 0.9995 | 0.9949 |
| | prod | 0.9979 | 1.0001 | 0.9994 | 1.0003 |
| s3 (M champ2 B champ A) | fast | 1.0043 | 1.0057 | 0.9986 | 1.0087 |
| | proxy | 1.0054 | 0.9972 | 1.0002 | 1.0064 |
| | prod | 1.0009 | 1.0018 | 0.9995 | 0.9998 |

Clean re-run of s3 (raw/rerun/s3b, order M champ2 B champ A): fast champ2 0.9970, A 1.0028, B 0.9970,
M 1.0028; proxy 0.9967 / 0.9932 / 0.9946 / 0.9911; prod 1.0007 / 1.0005 / 0.9988 / 0.9982.

All arms sit within A/A on every shape, and no arm clears min_gain (0.007 on the score). **Nothing ships.**
g47 and g48 are closed as null in pool.md. They remain valid latency savings for an idle-GPU caller,
which the ruler does not score.

## 6. Route and expected gain

findings/route.md has 13 rows, one id each:
- rows 1-3: the three musts (h37, h34, h28), discharged;
- row 4: r5.i3.g15 intra-WG split-KV, gated on an h16 ruling (fast x1.15-1.20, about +0.05-0.065 on the score);
- row 5: h23 on the m32x2 fast body, then h9 through it;
- rows 7-13: the remaining advise items, each with its own condition.

Expected gain from the arms in section 5: 0.

## 7. Build and measure the route (step 5-6)

Rows 1-3 need no build. Row 4's condition does not hold: hint.md has no h16 ruling (last modified 04:04,
before this round), so it was not built. **Row 5 (h23) is this round's build**, ledger id r12.i3.g49.
It is not a new pool entry; the round's two pool ids are g47/g48.

**Port (h27: port the diff, do not copy files).** proto `unroll`'s `proto.diff` (round-2 base, m32x8), applied
to `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py` only. The m32x8 body (proxy/prod) is untouched, so its
ISA is byte-identical. 13 of 14 hunks applied (offsets only; hunk 3 needed fuzz 3 and was checked by hand).
Hunk 1 (constants) was rewritten by hand. The one real adaptation: the proto's `lds_cap` is the full
320 KB, but m32x2 allocates 160 KB for 2 WG/CU (L12), so `lds_cap = min(163840, ...)`, and the old
`N_KV_PP * slot_bytes` assertion was replaced by the proto's `n_kv_pp` one. 3 slots fit in 160 KB.
Compile cache cleared before the first build (`/tmp/flycache/*`, raw/h23/cache.txt).

Arms (h23's protocol: C vs control first; D only if D > C):
- **C** = `N_KV_PP=3, KV_UNROLL=1`: `arms/h23C`, **left in `rounds/012/op`**;
- **D** = `N_KV_PP=3, KV_UNROLL=2`: `arms/h23D`.

ISA of the fast kernel (raw/h23isa, dumped per arm from a fresh compile):

| arm | tensorcnt waits | VGPR | SGPR | spill | LDS | instr | WMMA | code bytes |
|---|---|---|---|---|---|---|---|---|
| champ (r11) | 8x 0x0 | 454 | 99 | 0/0 | 160 KB | 4321 | 320 | 29 892 |
| C (PP3) | 8x 0x0, 6x 0x2 | 454 | 103 | 0/0 | 160 KB | 4495 | 320 | 30 824 |
| D (PP3+U2) | 10x 0x0, 8x 0x2 | 454 | 100 | 0/0 | 160 KB | 6131 | 512 | **42 540** (> 32 640 cap) |

Correctness (raw/h23):
- armcheck: C and D are **bitwise equal** to op/current on 22 cases. These include sq != skv, skv % 64 != 0
  (300x1000), non-causal, custom scale and 1x1.
- validation.py: C and D pass every correctness row (min 49.82 dB, short_q full, the same as the
  baseline) and determinism 200/200. Speed fails, as for the incumbent.
- ut/test_correctness.py: PASS (C and D).

Measure (raw/h23measure): benchmark.py, blocked ruler, 3 rotated sessions on the idle card (0% use; the
KFD pids rocm-smi lists are host-wide and sit on fa-g0's card 0001:04). Arms: r11 (the incumbent, which
op/current copies), r10, W (= C), D, champ2 (A/A). Beat ran in its own process afterwards (h28/h31).
No dmesg lines on 0003:04 throughout.

| shape | W (C) | D | r11 | r10 | champ2 | beat | W/champ | D/champ | W/beat |
|---|---|---|---|---|---|---|---|---|---|
| fast | 151.4 | 148.4 | **154.8** | 142.4 | 154.2 | 154.6 | 0.978 | 0.959 | 0.979 |
| proxy | 1594.5 | 1598.4 | 1594.8 | **1595.1** | 1602.1 | 1564.8 | 1.000 | 1.002 | 1.019 |
| prod | 1502.6 | 1500.5 | **1502.8** | 1493.6 | 1499.8 | 1552.7 | 1.000 | 0.999 | 0.968 |

(mean TF/s of 3 sessions. The fast losses repeat in every session: C 0.975/0.984/0.975, D 0.952/0.965/0.959; champ2 0.999/0.999/0.991.)

**Verdict: lost.** The round's score is 0.9885 uncapped, against r11's 0.9960 in the same session.

**Mechanism.** At fast the m32x2 KV chain is latency-bound, but not on TDM. Depth 2 already hides the
TDM load behind one tile of compute (h23's "one tile of compute already hides TDM latency" branch). So
PP3 buys nothing and adds 174 instructions (+4 SGPR, the 3-slot ring rotation, the runtime-chosen
`s_wait_tensorcnt 0x2/0x0` branch) to each of the heaviest WG's 16 tiles: -2.2%, above h23's
predicted -0.5 to -1.5%. U2 then pushes the kernel to 42.5 KB, past the 32 640 B instruction-prefetch
window, and loses another 2%. That is the same code-size penalty g16 and g18 saw on m32x8. On this
structure, the fast body's per-tile cost is instruction count on a latency chain, not TDM wait. The
lever that remains is row 4 (halve the chain), not deeper prefetch.
## Corpus consulted

- knowledge/optimization/routes/1-metrics-to-techniques.md
- knowledge/backends/flydsl/attention/README.md, knowledge/backends/flydsl/attention/techniques.md (headers)
- knowledge/backends/flydsl/attention/recipes/hd128.md
- knowledge/optimization/techniques/6-gfx1250-cdna5-mechanisms.md (TDM, cluster multicast, split barrier)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md (headers)
- knowledge/pitfalls/compiler-and-toolchain.md
- knowledge/profiling/0-kernel-profiling.md
- knowledge/backends/flydsl/official-skills-map.md
- listings of knowledge/{arch/gfx1250, ops/attention, optimization/techniques, pitfalls, backends/*/attention/recipes}

**Knowledge gap:** nothing in the corpus says whether a per-call event ruler sees host dispatch. Here it
does not: the flush memset issued before ev0 hides ~15 us of Python and ABI marshal. A probe without that
flush overstates small-shape time by ~2x (fast: 26.7 vs 13.7 us), so price levers with the ruler's flush.
