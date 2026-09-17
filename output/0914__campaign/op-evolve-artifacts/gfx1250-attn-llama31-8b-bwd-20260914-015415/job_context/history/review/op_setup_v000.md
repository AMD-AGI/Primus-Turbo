# Op Setup -- gfx1250-attn-llama31-8b-bwd (20260914-015415)

**SUCCESS** -- every artifact the evolve loop needs is built, verified by running
it, and `op/validation.py` returns a correct and expected FAIL at round 0.

---

## 1. Which branch step 1 took -- check this first

**Branch: DEVELOP ONE. No knowledge-corpus best-practice file matched.**

The lookup path the brief specifies is
`knowledge/backends/<op.backend>/<op.type>/`. `op.backend` is `triton`, and
**there is no `knowledge/backends/triton/` directory at all** --
`knowledge/backends/README.md` does not list `triton` among the backends it can
describe. The primary lookup came back empty, not merely unmatched.

Every attention recipe in the corpus was then read anyway, in case one matched
on `config:` despite the backend miss. There are five, and **all five are
`arch: gfx950`** where this job is `gfx1250`. Under `knowledge/INDEX.md`'s
precedence rule ("measured for your arch and op" outranks "general technique"),
an arch miss disqualifies on its own.

`config:` fields that differed, per candidate (`=` means agrees; fields only one
side states are not compared):

| card | backend | arch | head_dim | dtype | causal | direction | heads_q_per_kv | format |
|---|---|---|---|---|---|---|---|---|
| **this job** | triton | gfx1250 | 128 | bf16/fp32-acc | bottom-right | fwd+bwd | 4 | bshd |
| aiter `fmha_v3_bwd_hd128_bf16` | **aiter** | **gfx950** | = | = | = | **bwd only** | (unstated) | (unstated) |
| hipkittens `gqa_d128` | **hipkittens** | **gfx950** | = | = | **top-left** | = | = | = |
| hipkittens `gqa_d64` | **hipkittens** | **gfx950** | **64** | = | **top-left** | = | = | = |
| flydsl `hd128` | **flydsl** | **gfx950** | = | = | (unstated) | = | **needs pow2 in [8,256]** | **needs thd** |
| flydsl `hd64` | **flydsl** | **gfx950** | **64** | = | (unstated) | = | **needs pow2 in [8,256]** | **needs thd** |

The closest on `op.config` alone is the aiter card (head_dim, dtype and causal
convention all agree) but it is a different backend, a different arch and
backward-only. `hipkittens/gqa_d128` matches the dtype quartet, head_dim, GQA
ratio and memory format exactly but is **top-left** causal -- a different
masking convention, which is a different kernel and not a tuning delta. Both
FlyDSL cards hard-refuse this job's `heads_q_per_kv: 4` and its `bshd` format.

So the baseline was developed. It was **not** built from nothing: it stands on
the seed the job spec itself names in `op.reference.impl`, which `INDEX.md`
ranks above the corpus ("the job's own `job_context/` outranks the corpus").

**Source:** `https://github.com/AMD-AGI/Primus-Turbo`, host checkout
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`, **commit `d2f75576`**
(upstream aiter commit `ffa945f93115d21d6396e9fb16023e833b660764`, MIT).
Full detail, file list with sha256, and the three tuned knob deltas
(`BLOCK_N1` 128->256, `BLOCK_M2` 128->256, `BLK_SLICE_FACTOR` 2->1) are in
`op/baseline/PROVENANCE.md`.

Nothing failed to build or run. The referenced implementation works on this
machine; the develop-one branch was taken because the *corpus* had no match, not
because something refused to run.

---

## 2. What was created

| path | what it is | how it was verified |
|---|---|---|
| `op/baseline/impl.py` | Primus-Turbo Triton fused attention fwd + fused bwd behind a `torch.autograd.Function`, exposed as `attention(q,k,v,causal,softmax_scale)` | Ran. 52.2-53.7 dB on all four tensors on the spec shape; `fingerprint()` prints the live tile config and kernel file hashes on every benchmark run |
| `op/baseline/vendor/` | 6 byte-identical kernel/support files + 7 empty `__init__.py` + `LICENSE`, arranged to shadow the installed package | sha256 of every file recorded in `PROVENANCE.md`; import guard in `impl.py` raises if an installed `primus_turbo` was imported first |
| `op/baseline/PROVENANCE.md` | branch, source, commit, files, config deltas, the API deviation, the known defect | -- |
| `op/eager/impl.py` | plain-torch precision reference: explicit per-(batch, kv-head) loop, GQA group as a visible einsum contraction, bottom-right mask | It is the reference; verified by the fact that four independent implementations (baseline fwd, baseline bwd, flex fwd, flex bwd) all land 52-54 dB against it, which is the bf16 round-trip floor |
| `op/beat/impl.py` | `torch.nn.attention.flex_attention`, `enable_gqa=True`, bottom-right BlockMask, `num_warps=4` | Ran; `library_version()` prints torch/triton versions on every run |
| `op/beat/PROVENANCE.md` | library, versions, the `num_warps` sweep, block-vs-dense caveat | -- |
| `op/beat/warps_sweep.py` | the script that produced the `num_warps` 8-vs-4 numbers PROVENANCE cites | kept next to the claim so the decision is re-checkable |
| `op/ut/shapes.py` | 1 spec shape + 3 edge shapes (gated) + 1 diagnostic shape | -- |
| `op/ut/correctness.py` | SQNR in fp64, whole tensor, no sampling, separately for out/dq/dk/dv | Run through the runner: **4/4 gated shapes pass, rc=0** |
| `op/benchmark.py` | measures and prints, decides nothing | Run through the runner on 5 shapes x 4 arms in one session, rc=0 |
| `op/validation.py` | the sole success criterion | Run through the runner: rc=1 on `op/baseline`, rc=1 on the missing `op/current` |
| `op/runjob.sh` | detached run helper: `setsid`, sentinel `rc` file, keyed run dirs | Every measurement and every UT run in this stage went through it |

Not created, deliberately: `op/current/`, `rounds/000/`, and `state.yaml`.

### Shape coverage

| shape | B | Sq | Skv | why |
|---|---|---|---|---|
| `b4_s8192_hq32_hkv8_d128` | 4 | 8192 | 8192 | the shape in `op.shape` |
| `sq_lt_skv` | 4 | 1024 | 2048 | **the two sequence axes differ**, as required |
| `min_seqlen` | 4 | 512 | 512 | smallest shape that still takes the fused path |
| `ragged_tail` | 4 | 1000 | 1000 | not a multiple of any tile size; exercises the partial diagonal block |
| `sq_gt_skv` | 4 | 2048 | 1024 | **diagnostic only, not gated** -- see §6 |

All keep heads_q 32 / heads_kv 8 / head_dim 128 and `batch*heads_q >= 32` with
`seqlen_kv >= 512`, so the baseline's fused backward path -- the one being
optimised -- is always the path measured.

---

## 3. Measurements

GPU 1 (pinned, `runtime.gpu_id`), median of 30, 3 s of continuous warmup,
backward via `torch.autograd.grad`. TFLOP/s uses `tools/op_flops.py`'s
`backward=True` count, which is **2.5x forward and is BACKWARD ONLY**, not a
fwd+bwd total. Arms run palindromically in one session.

| shape | arm | fwd ms | bwd ms | bwd TFLOP/s | bwd GB/s | vs baseline | SQNR out/dq/dk/dv | gated |
|---|---|---|---|---|---|---|---|---|
| `b4_s8192_hq32_hkv8_d128` | baseline | 4.9698 | 10.0164 | 548.9 | 134.0 | 1.000x | 53.67 / 52.24 / 52.31 / 52.71 | **pass** |
| | current | *(does not exist)* | | | | | | |
| | beat | 3.7086 | 11.6762 | 470.9 | 114.9 | 0.858x | -- | -- |
| `sq_lt_skv` | baseline | 0.3066 | 0.8070 | 319.4 | 249.5 | 1.000x | 53.01 / 52.50 / 52.58 / 52.57 | **pass** |
| | beat | 0.1998 | 0.7136 | 361.3 | 282.2 | 1.131x | -- | -- |
| `min_seqlen` | baseline | 0.2944 | 0.1801 | 119.5 | 465.7 | 1.000x | 54.09 / 52.04 / 52.08 / 52.77 | **pass** |
| | beat | 0.0677 | 0.1432 | 150.4 | 586.2 | 1.258x | -- | -- |
| `ragged_tail` | baseline | 0.2774 | 0.3143 | 261.0 | 521.4 | 1.000x | 53.99 / 52.09 / 52.15 / 52.75 | **pass** |
| | beat | 0.1444 | 0.6716 | 122.1 | 244.0 | 0.468x | -- | -- |
| `sq_gt_skv` | baseline | 0.2748 | 0.3963 | 217.0 | 762.1 | 1.000x | 53.97 / **-15.95** / 52.10 / 52.72 | diagnostic |
| | beat | 4.4845 | 7.6540 | 11.2 | 39.5 | 0.052x | -- | -- |

(Each figure is the mean of that arm's two palindrome positions.)

**Headline: baseline beats the target arm by 1.166x on the backward, against a
required 1.50x.**

### Statistic, iteration count, and the round-0 agreement check

* **Statistic: median. Iterations: 30. Warmup: 3.0 wall-clock seconds of
  continuous load.** Fixed in `benchmark.py` before any number was recorded, and
  not changed since. Median and best-of-N are not interchangeable and mixing
  them across a comparison invents differences, so one was chosen and kept.
* `knowledge/pitfalls/measurement-traps.md` was read in full **before** the
  first number was recorded, not after.
* **The round-0 `current == baseline` agreement check was satisfied by
  substitution, and here is the substitution.** `op/current/` does not exist --
  the framework creates it, and the brief forbids me to. So the check was made
  against the two `baseline` positions inside the palindrome, which are separate
  measurements minutes apart in the same session. They agree to **0.20%**
  (10.0263 vs 10.0064 ms) in the benchmark session, and to **0.08%** (10.0274 vs
  10.0359 ms) inside `validation.py`'s own run, which reports it as
  `vs baseline 1.001x`. The measurement is stable.
* Spread (p90-p10 over the median) on the gated shape: **baseline 0.67-0.82%,
  beat 2.49-3.60%**.

---

## 4. What `validation.py` checks

**One sentence:** the candidate passes iff every one of out, dq, dk and dv is at
least 50 dB SQNR against `op/eager/` on the spec shape and all three edge
shapes, **and** its backward on the spec shape is at least 1.50x faster than
`op/beat/` measured in the same run.

It takes one optional argument, the implementation directory, defaulting to
`op/current/`. It calls `benchmark.py` for every number -- there is no second
timing loop anywhere in this job. It imports `tools/op_flops.py` through
`benchmark.py` rather than copying it. It depends on nothing outside
`job_context` except that shipped `tools/`.

It cannot be satisfied by editing it because it contains no expected numbers to
edit: the bar is re-measured every run. The only literals are the 50 dB gate and
the 1.50x margin, both straight from the spec, and changing either is a visible
change to the criterion rather than a tweak to a measurement.

**One judgement call, stated rather than buried:** the speed gate is on the
**backward** only. The forward, the fwd+bwd total, bandwidth, ratio-to-baseline,
spread and the diagnostic shape are all measured and printed but not gated.
Gating the total would quietly move the goalposts -- `op/beat/`'s forward is
*faster* than the baseline's on the spec shape, so a total-based gate would
demand the candidate also win a forward race the spec never asked for. The
reasoning is written into `validation.py`'s docstring so a later round that
disagrees has to edit the argument, not discover the omission.

### Step 9 result (run once through the runner, as instructed)

```
$ python op/validation.py              ->  rc=1   "no impl.py under .../op/current"
$ python op/validation.py op/baseline  ->  rc=1   VALIDATION FAILED: speed
    b4_s8192_hq32_hkv8_d128: backward is 1.166x op/beat/, needs 1.50x
                             -- short by 22.3% of the target
```

Both failures are correct and expected. There is real work for the loop to do:
**the baseline must get ~28.7% faster on the backward to clear the target.**

---

## 5. Is there a live anchor?

**Yes, and it agrees.** The campaign that produced the seed recorded, on this
same node, the same kernel at **10.270 ms** backward and flex_attention at
**11.536 ms** (`Primus-Turbo/output/0914__repro__c07/FINAL-TABLE.md`). This
stage measured 10.016 ms and 11.676 ms independently, with its own harness and
its own FLOP accounting. Agreement to within 2.5% on two arms measured months
and one harness apart is a genuine cross-check, not a coincidence.

The SQNR figures are a second live anchor: the seed's own
`FUSED_MHA_PROVENANCE.md` records the four-tensor SQNR it was characterised at,
and this stage reproduced those numbers exactly against an eager reference
written independently here.

---

## 6. Inferred, assumed, or weak -- read this before trusting a number

**Things I decided rather than was told:**

1. **FLOP basis.** `tools/op_flops.py`'s `backward=True` returns 2.5x forward
   and is backward-only; the job file's TFLOP/s anchors are on the 3.5x fwd+bwd
   basis. The two differ by **1.400x**. I time the backward and divide by the
   backward-only count, which is self-consistent and reproduces the seed's
   24.415 ms figure. **Any TFLOP/s number in this report is on the
   backward-only basis and is 1.400x larger than the same kernel quoted on the
   job file's basis.** This is a real trap for anyone comparing the two.
2. **The speed gate is the backward** (§4).
3. **`op/current/` was substituted for by the palindrome's two baseline
   positions** (§3).
4. **`num_warps=4` for the beat arm.** Inductor's default is 8, which measures
   **30.639 ms** against 4's **11.654 ms** -- **2.63x apart**. Had I inherited
   the default, the baseline would already beat the bar by 3.06x at round 0 and
   the target would have meant nothing. This one was nearly a straw man and is
   worth knowing about.

**Known weakness in the baseline:** on `sq_gt_skv` (Sq=2048 > Skv=1024) the
baseline's **dq is -15.95 dB**. The error is localised entirely to the fully
masked query rows, whose gradient is exactly zero: restricted to live rows, dq
is 52.05 dB, while the dead rows come back with `|dq|` up to **5.25**. The seed
does not zero a fully-masked query row. This is **reported but not gated**,
because it is unreachable at `Sq == Skv` and every shape in `op.shape` has
`Sq == Skv` -- gating it would red every round for a pre-existing condition no
round can cause or fix. It is carried as a diagnostic shape so it is printed on
every run and cannot be forgotten. If a later round introduces a path where
`Sq != Skv`, this becomes live. Rationale is written into `op/ut/shapes.py`.

**Deviation from `op.reference.api`:** the mandated
`GlobalBackendManager.set_attn_backend(BackendType.TRITON, ...)` **does not
exist in the pinned image** -- `primus_turbo.pytorch.core.backend_manager` is
absent at the installed commit `f857e429`. Verified by import, not inferred. The
baseline pins by vendoring instead, so no dispatcher is reachable at all. That
is a stronger pin than the one the spec asked for, but it is a deviation and it
is on the record.

**The GPU is not fenced, and this cost real time.** The spec's `runtime.gpu_id:
1` is a convention the framework does not enforce. Another tenant's TorchTitan
Llama-3.1-8B pretrain saturated GPU 0 for the middle of this stage. Because
`runjob.sh` initially issued `docker exec` with no `HIP_VISIBLE_DEVICES`,
everything landed on device 0 and the backward read **2064% spread**. After
pinning to GPU 1 the same measurement reads **0.56%**. This confirms Job Setup's
open question `gpu-not-fenced-by-the-framework`. **Every number in this report
was taken after the pin.** A later round that sees wild spread should check
`HIP_VISIBLE_DEVICES` before it checks its kernel. Related: the KFD PID table
and the memory table disagreed about which card the trainer held; the memory
table was right and the PID table's GPU column is not trustworthy on this node.

**No roofline check is possible on gfx1250**, which `measurement-traps.md`
otherwise asks for. The ISA document publishes **no per-datatype peak**, and the
corpus's `arch/gfx1250/gfx1250.md` states its data "supports porting decisions
and **not** roofline arithmetic". Only 13 of 51 hardware counters return
non-zero, **no counter reports bytes at any level**, and WMMA FLOP counters read
zero -- so the achieved-bandwidth figures above are `op_flops.py`'s *minimum*
byte counts divided by time, which is a lower bound on traffic, not a measured
one. Treat the GB/s column as an arithmetic consequence of the FLOP model, not
as evidence about the memory system.

**One bug I introduced and fixed during this stage:** `correctness.check_shape`
required its caller to have installed the eager reference first, so the first
`validation.py` run died with `ModuleNotFoundError: No module named
'eager_ref'`. Fixed by making `check_shape` install it itself; the UTs were then
**re-run through the runner** and pass, and the validation numbers above come
from after the fix.


---

## Review verdict

- 2026-09-14T02:35:48 -- approved (non_interactive)


---

## Review verdict

- 2026-09-14T03:40:35 -- approved (non_interactive)
