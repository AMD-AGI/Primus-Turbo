# Related Work Template

Use this as the default structure for `<campaign_dir>/related_work.md`.

## Survey Objective

- Target operator:
- Target backend:
- Target GPU:
- Date:
- Campaign:

## Search Scope

- Project-local implementations reviewed:
- AMD / ROCm docs reviewed:
- External repos or papers reviewed:
- Competitor implementations or performance reports reviewed:

## Relevant Implementations

| Source | Repo / Doc | Backend / Lang | Hardware context | Reported performance | Why it matters |
|--------|-------------|----------------|------------------|----------------------|----------------|
| Example | FlashAttention | CUDA / Triton | H100 | 1.2 TB/s | Useful layout / pipelining idea |

## AMD / ROCm Guidance

- Key ROCm or ISA constraints that affect this campaign
- Hardware-specific opportunities on the target GPU
- Any validation or datatype caveats

## Competitor or Alternate Baselines

- Useful NVIDIA / CUDA / other-stack implementation ideas
- Reported performance ceilings worth comparing against
- Important caveats when the comparison is not apples-to-apples

## Transferable Ideas

- Idea 1:
  - Source:
  - Why it may transfer:
  - Main risk:
- Idea 2:
  - Source:
  - Why it may transfer:
  - Main risk:

## Non-Transferable or Misleading Items

- Techniques that depend on incompatible hardware, library fusion, hidden preprocessing, or different datatypes
- Techniques whose reported gain comes from the benchmark loop reusing the same Python tensor object (e.g. `id(activation)`-keyed caches), and that would miss in real LLM training

## Real-training Transfer Audit

Tag every entry from the two sections above using the buckets defined in
[`../../rules/iteration_rules.mdc`](../../rules/iteration_rules.mdc) Rule 11
(operational guidance is in `SKILL.md` under "Avoiding Benchmark Over-Fitting"):

- `K1` kernel-internal (autotune / mfma / tile / `EVEN_K` / `tl.dot_scaled` / pipeline depth)
- `K2` layout-duplicated kernels
- `K3` single-launch fusion
- `K4` same-call data flow (e.g. `ctx.save_for_backward`)
- `W1` weight-quant cache keyed on `id(weight) + _version + shape`
- `W2` activation / grad_out / activation-scale cache keyed on `id(activation-like tensor)`
- `W3` generic wrapper memoization whose key contains a non-weight tensor `id`

| Idea | Source | Bucket | Reported gain | Estimated real-training gain | Verdict |
|------|--------|--------|---------------|------------------------------|---------|
| (example) `tl.dot_scaled` MFMA path | rocm-triton blog | K1 | +12% combined-step | +10–12% combined-step (structural) | Promote to shortlist |
| (example) `activation_quant_cache` | vendor blog post | W2 | +15% combined-step in benchmark | +0% in real training (zero cache hits per step) | Drop, do not promote |
| (example) `weight_quant_cache` keyed on `id(w) + _version` | vendor blog post | W1 | +8% combined-step in benchmark | +1–2% combined-step (bounded by `quant_time(w) / step_time`) | Promote with bounded expectation |

Rules for this table:

- A `W2` or `W3` row MUST end with `Drop, do not promote`. It is recorded here so the agent does not chase it later, but it is not a candidate hypothesis.
- A `W1` row MUST carry a derivation of the per-step gain bound (`quant_time(weight) / step_time` for the project's representative shape) instead of the headline benchmark number.
- A `K1`–`K4` row may use the headline number directly; structural kernel changes are expected to transfer.

## Initial Hypothesis Shortlist

Promote only `K1`–`K4` ideas, plus `W1` ideas that survived the bound check
above. `W2` / `W3` ideas are not allowed on this list.

1. Hypothesis:
   - Bucket:
   - Why first:
   - Evidence from survey:
   - Expected real-training gain (per real LLM step, not per benchmark iter):
2. Hypothesis:
   - Bucket:
   - Why next:
   - Evidence from survey:
   - Expected real-training gain:
3. Hypothesis:
   - Bucket:
   - Why fallback:
   - Evidence from survey:
   - Expected real-training gain:

## Temporary Survey Assets

- Temp repo path(s): `agent/tmp/<campaign_name>/related-work/repos/`
- Notes or downloaded artifacts:
- Anything worth preserving into the main repo:

## Bottom Line

- Best existing implementation found:
- Most relevant idea to try locally first:
- Biggest risk or uncertainty before BASELINE:
- Confirmed bucket of the most-relevant idea (must be `K1`–`K4` or bounded `W1`):
