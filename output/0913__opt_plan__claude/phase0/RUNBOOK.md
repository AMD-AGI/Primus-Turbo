# Runbook — what to run the moment a healthy gfx1250 is available

Everything here is Phase 1 and Phase 2 of the plan. Phase 0 is done (see
`PHASE0-STATUS.md`). Commands are given in order; each step says what would make you stop.

All paths are relative to the Primus-Turbo checkout, branch
`gfx1250-attn-dispatch-and-tuning`.

---

## Step 0 — prove the card is healthy, before anything else

```bash
# NOT rocm-smi: a wedged card leaves tasks in amdgpu_info_ioctl, which is what
# rocm-smi calls, so it hangs instead of reporting.
dmesg | grep -E 'MES\(|GPU Hang|wait for reset ack|Memory access fault' | tail -20

# clock ceiling — this is the number the whole target ladder depends on
cat /sys/class/drm/card*/device/pp_dpm_sclk
dmesg | grep -i 'throttl'
```

**Stop if** the DPM table has only 500/1100 MHz or the VR throttle warning is present.
You can still run — the relative comparisons remain valid — but every absolute number
must be labelled with the clock state, and the target ladder must use the throttled roof.
See `PLATFORM-ESCALATION.md`.

Also confirm the card is idle. There is no GPU lock in this workflow: a shared card does
not raise, it records a low number, and that number becomes the champion.

---

## Step 1 — smoke test the harness (seconds)

```bash
python3 tools/gfx1250/tune_attention.py --shape smoke
```

Expect one JSON line with `"ok": true`, `"is_gfx1250": true`, and four SQNR values in the
52–56 dB band. **Stop if** any tensor is below 50 dB on the *default* config — that means
the reference or the plumbing is wrong, not the kernel.

Then prove the dispatch fix works — this is the whole point of the Phase 0 code change:

```bash
pytest tests/pytorch/ops/test_attention.py -m gfx1250 -v
```

Expect the gating tests, the two new dispatch-regression tests, the tuning-spec tests and
the forward+backward accuracy test to run rather than skip. Before Phase 0 the entire
suite was skipped on this arch.

---

## Step 2 — the round-0 bake-off, in ONE image and ONE session

This settles two open contradictions. Do not skip it and do not split it across sessions.

```bash
# turbo's Triton backend, shipped config, production shape
python3 tools/gfx1250/tune_attention.py --shape llama31-8b \
    --json out/round0-turbo-triton.json

# the shape commit c1325c7e reported 220.6 TFLOP/s on
python3 tools/gfx1250/tune_attention.py --shape llama31-8b-b2 \
    --json out/round0-turbo-triton-b2.json
```

Then the flex anchor and (if aiter is installed in the image) aiter's Triton MHA, from the
Primus-side bench:

```bash
cd /home/lihuzhan/code/2026_0828__primus/Primus/benchmark/kernel/attention
python3 bench_attention.py --backends flex,turbo:TRITON,sdpa 2>&1 | tee out/round0-bench.log
```

**The two questions this answers:**

1. **Is the 1.675× discrepancy the throttle?** `c1325c7e` claims 220.6 TFLOP/s at b=2
   s=8192; the in-tree bench measured 131.7 for the same kernel at b=4, same s/H/D/dtype.
   Batch alone should not move TFLOP/s, and 220.6/131.7 = 1.675 ≈ the 1.65 throttle ratio.
   If the b=2 run now reproduces 131.7-ish on a throttled card and ~220 on a healthy one,
   that is confirmed — **and every absolute target in the plan gets rewritten by 1.65×.**
   This is the highest-value single measurement in the campaign.
2. **Are the two seeds comparable?** aiter's 34.111 ms and turbo's 58.424 ms were taken on
   *different container images* and have never been cross-checked. Whichever is faster in
   one image is the seed.

Also worth one pass here, because it changes the round schedule if it works:

```bash
# does the newer SDK expose memory counters? "defined" is not "returns data" --
# this image already has 13 counters that are accepted and read exactly zero.
rocprofv3 --pmc GL2C_EA_RDREQ_64B SQ_WAVES -- python3 tools/gfx1250/tune_attention.py --shape smoke
```

If memory counters come back live, bound analysis becomes possible and `fast_per_deep`
should be lowered from 5. **Never run PC sampling** — three attempts, three GPU faults,
one requiring a reboot.

---

## Step 3 — R1–R3: reproduce the two known integers

The calibrating prior, from aiter at this shape: forward `num_stages` 1→2 was 2.06× alone,
backward `num_warps` 4→2 was 1.32× alone, together 1.401×, at identical SQNR.

```bash
python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 --baseline \
    --axis fwd:num_stages=1,2,3 --ledger out/r1-fwd-stages.jsonl

python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
    --axis bwd:num_warps=1,2,4,8 --ledger out/r2-bwd-warps.jsonl

# they interact -- the one two-knob forward combination tried on aiter was worse than
# num_stages=2 alone, so do not assume separability
python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
    --axis fwd:num_stages=1,2,3 --axis bwd:num_warps=1,2,4 --ledger out/r3-cross.jsonl
```

Confirm the winner on the production shape before it becomes champion:

```bash
python3 tools/gfx1250/tune_attention.py --shape llama31-8b --tune "<winning spec>"
```

**`num_stages=2` winning is a latency-hiding result**, so re-check it if the throttle is
lifted — the balance shifts at 1700 MHz.

---

## Step 4 — R4–R20: `_bwd_kernel_dkdv`, which is 61.5% of the time

This is the campaign. It leads the second-worst kernel by 3×.

```bash
python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
    --axis bwd:num_warps=1,2,4,8 --axis bwd:num_stages=1,2,3 \
    --ledger out/r4-dkdv-sched.jsonl

python3 tools/gfx1250/sweep_attention.py --shape llama31-8b-s4096 \
    --axis bwd:waves_per_eu=0,1,2,3 --ledger out/r5-dkdv-occupancy.jsonl
```

Guardrails on range selection, measured on aiter at this shape — the penalties are large
and the ranges are not symmetric around the default:
`BLK_SLICE_FACTOR=4` → 151.6 ms; `waves_per_eu=4` → 120.6 ms; `BLOCK_M1=16` → 91.1 ms;
`BLOCK_M2=64, BLOCK_N2=128` → compilation error.

Beyond the scheduling knobs, the structural candidates are in
`docs/gfx1250-attention-tuning.md` §6, in expected-value order. The top three:

1. **Fold `log2(e)` into the softmax scale.** The kernel is on `exp2` but pays a
   per-element `* RCP_LN2` across the whole score tile. At 64×64 that is 4,096 VALU ops
   per tile removed.
2. **Raise the WMMA:VALU ratio.** Softmax is the likely bottleneck at D=128, but by
   ~1.2–1.4× on a cycle-weighted count, not an order of magnitude. (Do not repeat the
   "~32 exps per WMMA" figure — on wave32 a 32-element row segment is *one* `v_exp_f32`,
   so counting elements against instructions overstates it by the wave width.)
3. **XCD workgroup remap** — ~10 lines of `tl.program_id` math, `NUM_XCDS = 8` confirmed
   from node topology. Keep the `num_wgs % 8 == 0` guard: it fails silently.

**Price `sequence_parallel`.** It is hardcoded `True` in `attention_triton_impl.py` and
has never been measured against `False`. Given dK/dV is key-outer and GQA divides the
worker count by G=4, the query-axis split is exactly the knob the structure argues for.

---

## Step 5 — R21–R30 `_bwd_kernel_dq` (20.1%), R31–R36 `attn_fwd` (17.0%)

Same shape of sweep, lower priority. Note `_bwd_preprocess_use_o` is 0.1% of the time and
has no autotune decorator at all — leave it alone; it is not worth a round.

---

## Step 6 — R37–R40: combine, confirm, and check it transfers

```bash
# full production shape, best combined spec, repeated
for i in 1 2 3; do
  python3 tools/gfx1250/tune_attention.py --shape llama31-8b --tune "<best>" \
      --json out/final-$i.json
done
```

Then the transfer check, which is not optional. A microbenchmark win that depends on a
config cache keyed on activation tensors has a **real-training hit rate of zero** — Q/K/V/dO
are fresh every iteration. Run the real thing:

```bash
# Primus, MI455X repro config, with use_turbo_attention turned back on
# examples/torchtitan/configs/MI455X/repro_l8b_bf16_mbs4_seq8k_v5.yaml
```

Two things to capture from that run beyond tps:

- **How many flex kernel launches per step, and their shapes.** This is the one clean way
  to settle what the JIRA's 501 ms cell aggregates, and hence the 15.656 ms/call figure
  that nobody has been able to reproduce by any mask.
- **Whether `aiter` is still required to import.** On gfx1250 the Triton backend should
  now be reachable without it (Phase 0 gated AITER off this arch), but the *varlen* path
  still routes to AITER and is unchanged.

---

## Recording

`sweep_attention.py` appends a JSONL ledger and fsyncs after every candidate, so a wedge
costs you the current candidate and nothing else. Re-running the same command resumes.

`--report` prints the ranking, lists fast-but-incorrect candidates separately (they can
never win), and says explicitly when a winner sits at the edge of a swept range — which
means the range was the constraint, not the hardware.

Keep every ledger. The point of the campaign is a defensible chain from "shipped config"
to "champion", and a number without its ledger row is not evidence.
