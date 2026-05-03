---
name: avoid-benchmark-overfit
description: Operational checklist that helps the agent decide whether a candidate optimization will transfer to real LLM training, or only inflates the benchmark score. Read this skill before any round that proposes a wrapper-level cache, memoization, or `id`-keyed lookup. Read it again whenever a benchmark gain looks suspiciously larger than what the kernel change can structurally explain.
---

# Avoid Benchmark Over-Fitting in Kernel Optimization

This skill is the operational counterpart to the hard rule
[`../../rules/no_benchmark_overfitting.mdc`](../../../rules/no_benchmark_overfitting.mdc).

The hard rule states **what is forbidden**. This skill states **how to detect
it during ANALYZE / OPTIMIZE / VALIDATE**.

When to invoke this skill:

- ANALYZE phase, before proposing any wrapper-level change that involves
  `dict`, `OrderedDict`, `weakref`, `lru_cache`, `id(...)`, or any "skip work
  if we have seen this tensor before" idea.
- VALIDATE phase, when a round shows benchmark gain larger than ~1% per
  individual operation that the kernel change should produce.
- Any time the benchmark report shows + step TFLOPS but the same change makes
  no structural difference to a kernel hot path.

---

## Step 1 — Classify the proposed change

Place the candidate change into exactly one of these buckets.

| Bucket | Examples | Verdict |
|---|---|---|
| K1. Kernel-internal | autotune config, mfma variant, tile shape, `EVEN_K`, swizzle, pipelining, scale preshuffle, `tl.dot_scaled` | Always allowed |
| K2. Layout duplication | NT-only kernel, NN-only kernel, TN-only kernel | Always allowed |
| K3. Single-launch fusion | `quant_fp8_blockwise_dual_impl`, fused dequant+matmul, fused softmax+drop | Always allowed |
| K4. Same-call data flow | `ctx.save_for_backward` carrying a forward-side intermediate to backward | Always allowed |
| W1. Weight quant cache | cache keyed on `id(weight) + version + shape` | Allowed but limited; see Step 4 |
| W2. Activation quant cache | cache keyed on `id(activation)`, `id(grad_out)`, or `id(activation_scale)` | **FORBIDDEN** |
| W3. Generic `lru_cache` over wrapper code | any memoization wrapping `forward` / `backward` whose key includes a tensor `id(...)` other than a weight | **FORBIDDEN** |

If the bucket is K1–K4, the round is on the safe lane. Continue with normal
ANALYZE / OPTIMIZE / VALIDATE.

If the bucket is W2 or W3, **stop**. Refuse to land the round. The benchmark
gain you are about to chase will not exist in real training.

If the bucket is W1, follow Step 4 to bound the legitimate gain.

---

## Step 2 — The `id(...)` audit

Before writing a single line of cache code, run this audit on the proposed
key tuple.

For every component of the cache key, ask:

> Across one real LLM training step (`forward(batch_t); loss.backward();
> optim.step()`) and the next step's `forward(batch_{t+1})`, is this value
> stable for the **same logical role**?

| Key component | Stable across iterations in real training? |
|---|---|
| `id(weight)` (a module parameter) | **Yes**, until the parameter is re-assigned (rare) |
| `getattr(weight, "_version", 0)` | Bumps once per `optim.step()`, so the (id, version) pair is stable inside one fwd+bwd pair only |
| `id(activation)` | **No** — autograd / dataloader allocates fresh tensors |
| `id(grad_out)` | **No** — autograd internals allocate fresh tensors |
| `id(activation_scale)` | **No** — produced fresh by the quant kernel each call |
| `tuple(t.shape)`, `tuple(t.stride())`, `t.dtype` | Yes, but these alone do not authorize a cache (multiple tensors share them) |

If the cache key relies on any of the "No" rows for its hit rate, the cache
will miss in real training. Reject the round at ANALYZE — do not even start
implementing.

---

## Step 3 — Pen-and-paper hit-rate trace

For any cache that survived Step 2, trace its hit rate on paper across a
hypothetical 4-step real training loop:

```
step 1: a1, w, grad_out_1   ->  fwd(a1, w),   bwd(grad_out_1)
step 2: a2, w, grad_out_2   ->  fwd(a2, w),   bwd(grad_out_2)
step 3: a3, w, grad_out_3   ->  fwd(a3, w),   bwd(grad_out_3)
step 4: a4, w, grad_out_4   ->  fwd(a4, w),   bwd(grad_out_4)
```

Count cache hits for the proposed cache against this trace.

Expected outcomes:

- **W1 weight cache** with `(id(w), w._version)` key: 1 hit per step (the
  backward of step `t` reuses the forward-time entry). After `optim.step()`
  the version bumps and step `t+1`'s forward re-quantizes. Net: +1
  quantization saved per step.
- **W2 activation cache** with `(id(a), a._version)` key: 0 hits (`a` differs
  across steps).
- **W2 grad_out cache** with `(id(grad_out), grad_out._version)` key: 0 hits.

Compare this against the benchmark hit rate (which is 99/100 by construction)
to estimate the gap.

If the real-training hit rate is below 50%, the round MUST be rolled back
even if the benchmark accepts it.

---

## Step 4 — Bounding the legitimate weight-cache gain

Weight cache (W1) is the only wrapper-level cache that survives. Its real
training gain is bounded by:

```
gain_per_step = quant_time(weight) / step_time
```

Some realistic numbers for `gemm_fp8_blockwise` on MI355X:

- `quant_fp8_blockwise_for_weight_impl` for an `[N, K] = [4096, 4096]`
  weight: ~30 µs
- combined fwd + bwd of the same blockwise GEMM at this shape: ~1.5–2 ms
- per-step gain bound: 30 / 2000 ≈ **+1.5%**

If your benchmark reports a +5% combined-step gain and your only mechanical
change is the weight cache, the extra +3.5% comes from the cache hitting on
forward repeats inside the benchmark's 100-iter inner loop. **That part will
not transfer to real training.** Report the realistic per-step gain in the
round summary, not the benchmark number.

---

## Step 5 — Required round summary section

Any round that touches a wrapper-level cache MUST include a section in
`<campaign_dir>/rounds/round-N/summary.md`:

```markdown
## Real-training transfer check

- Bucket: <K1 / K2 / K3 / K4 / W1 / W2 / W3>
- Cache key (if any): <(id(t), t._version, …) or "no cache">
- id(...) audit (Step 2):
  - <key component>: stable across iters? <yes/no>
  - <key component>: stable across iters? <yes/no>
- 4-step pen-and-paper trace hit rate: <0/4 | 1/4 | 4/4 | 0/8 | 1/8 | …>
- Benchmark gain on this round: +X.XX% combined-step
- Estimated real-training gain: +Y.YY% combined-step (with derivation)
- Decision: ACCEPT-as-real / ACCEPT-with-asterisk / REJECT-as-overfit
```

`ACCEPT-as-real` is reserved for K1–K4. `ACCEPT-with-asterisk` is reserved
for W1 with the bounded gain explicitly documented. `REJECT-as-overfit` is
the only valid outcome for W2 / W3.

---

## Step 6 — Tips file hygiene

When the round's reusable lesson is appended to
`agent/historical_experience/<gpu>/<op>/<backend>/tips.md`:

- Never describe a W2 / W3 cache as "high-leverage". The lesson there is
  always: "this pattern is benchmark over-fit; do not retry."
- For W1, describe the per-step real-training gain bound, not the benchmark
  number, so a future agent does not chase the inflated number.

If a previously-recorded tip celebrates a `id(activation)` or `id(grad_out)`
cache, **delete the tip**. Treat it as a known anti-pattern, not as historical
knowledge.

---

## Worked example: the cache rollback that motivated this rule

In an earlier campaign for `gemm_fp8_blockwise / Triton / gfx950`, several
caches were added to `primus_turbo/pytorch/ops/gemm_fp8.py` and
`primus_turbo/triton/gemm/gemm_fp8_kernel.py`:

- `_blockwise_act_row_cache`, key includes `id(a)` — **W2**
- `_blockwise_act_col_cache`, key includes `id(a)` — **W2**
- `_blockwise_grad_out_cache`, key includes `id(grad_out)` — **W2**
- `_scale_t_cache`, key includes `id(scale)` (scales of activations) — **W2 / W3**
- `_blockwise_weight_cache`, key includes `id(b)` for module weight — **W1**

Reported benchmark gain: roughly +6% combined-step geomean, **of which
about +3–4% came from the W2 caches** hitting the benchmark's 100-iter inner
loop and would not transfer to real training. The weight cache (W1) and the
allowed kernel-level changes accounted for the remaining +2–3%, which is the
actual real-training gain.

The W2 caches were rolled back. The legitimate kernel-level changes
(autotune extension, NT/NN kernel duplication, dual quant fusion, ctx-saved
col output, `EVEN_K` fast path) were kept. New W2-like patterns must be
rejected at ANALYZE going forward.

---

## Checklist (use during VALIDATE)

Before recording a round as ACCEPT, confirm all of the following:

- [ ] No new `dict` / `OrderedDict` / `lru_cache` keyed on `id(t)` for any
      `t` that is not a module weight
- [ ] No new code path uses `weakref.ref(activation)` or
      `weakref.ref(grad_out)`
- [ ] If the round adds a weight cache: the `_version` invalidation is
      tested (mutate a clone of the weight in-place, expect cache miss)
- [ ] The summary's `Real-training transfer check` section is filled in and
      explicitly states the estimated real-training gain
- [ ] The reusable tip (if any) does not advertise an `id(activation)` /
      `id(grad_out)` pattern as a useful technique

A round that fails any of these checks is INVALID and must be rolled back,
even if its benchmark aggregate score improved.
