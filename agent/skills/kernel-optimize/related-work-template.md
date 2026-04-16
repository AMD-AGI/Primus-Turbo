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

## Initial Hypothesis Shortlist

1. Hypothesis:
   - Why first:
   - Evidence from survey:
2. Hypothesis:
   - Why next:
   - Evidence from survey:
3. Hypothesis:
   - Why fallback:
   - Evidence from survey:

## Temporary Survey Assets

- Temp repo path(s): `agent/tmp/<campaign_name>/related-work/repos/`
- Notes or downloaded artifacts:
- Anything worth preserving into the main repo:

## Bottom Line

- Best existing implementation found:
- Most relevant idea to try locally first:
- Biggest risk or uncertainty before BASELINE:
