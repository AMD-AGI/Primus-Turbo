# Round 17 — Dead `_avg_group_m` removal (FP8 grouped, post-R11 inlining)

## Context (entering this round)

- Wall metric `_metric_grouped_fused_wall.py` score: **1000** (capped, 14
  consecutive rounds), geomean 1.3881.
- All 7 below-target shapes are R12-R15 wide-sweep falsified or pre-R6
  wide-sweep verified — no remaining dispatch-tuning lever per R16
  consolidated summary.
- Patience 14/30 with 16 rounds of buffer remaining.

## Lever picked this round

Pure code cleanup — removing the dead `_avg_group_m` function from
`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`:

```python
# (Removed) lines 121-130 in pre-R17 file:
def _avg_group_m(a_total_rows: int, bs: int) -> int:
    """Return ``a_total_rows // bs`` (>=1) for cfg selection only.

    Host端禁止 uniform 判断 / 禁止 per-group fallback —— ``m`` 仅用于
    select_default_config 选 cfg，kernel 内部 ``group_offs`` device-side
    O(G) scan 处理任意 group_lens 的 correctness。
    """
    if bs <= 0:
        return max(a_total_rows, 1)
    return max(a_total_rows // bs, 1)
```

The function became dead in R11 (host-overhead trim commit) when the
sole call site inside `GroupedGEMMFP8HipKittenBackend.execute()` was
inlined as `avg_m = max(m_total // bs, 1) if bs > 0 else max(m_total,
1)` for a 0.10 µs / call host saving (round-11 commit message). The
def itself was preserved for one round in case any other site relied
on it.

## Audit performed before removal

```
$ rg "_avg_group_m" primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py
121:def _avg_group_m(...):  # def
542: #   (c) ``_avg_group_m`` inlined ...  # comment, R11 history
565: # Mirror ``_avg_group_m`` semantics ...  # comment, references inline
```

```
$ rg "_avg_group_m" --type py | grep -v "grouped_gemm_fp8_impl"
analysis/_notes/round-85-bf16-grouped-var-k-ki-spec-FALSIFIED-vgpr-spill-pattern.md  # doc, not code
analysis/_notes/round-5-fused-act-python-overhead-floor-confirmed.md  # doc
analysis/_notes/round-66-bf16-grouped-bwd-cfg-audit-CLOSED.md  # doc
analysis/_notes/round-11-hk-host-overhead-breakdown.md  # doc
primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py  # bf16 file, INDEPENDENT def at line 32
```

The BF16 grouped GEMM file (`grouped_gemm_impl.py`) has its **own**
`_avg_group_m` def at line 32 which is still called at line 452 — that
one is alive and unrelated.

No `from ... import _avg_group_m` or attribute access via module name
`grouped_gemm_fp8_impl._avg_group_m` exists anywhere in the repo.
The removal is safe.

## Comments preserved

The two comment references to `_avg_group_m` at lines 530-531 ("(c)
``_avg_group_m`` inlined — single ``//`` arithmetic, no function call
frame (~0.10 µs)") and 553-554 ("Mirror ``_avg_group_m`` semantics
(max(., 1) clamp for the degenerate ``bs <= 0`` and ``m_total < bs``
paths)") are **kept as-is** as R11 historical breadcrumbs explaining
why the inline arithmetic exists in `execute()`. Future agents tracing
the R11 inline decision via `git blame` will still find the original
def in the R11 commit's parent SHA.

## Verification

- Metric run before edit: score=1000, geomean=1.3881, 0/24 correctness fail.
- Metric run after edit: score=1000, geomean=1.3897, 0/24 correctness fail.
- Both within run-to-run noise floor (~0.005 absolute geomean).
- The metric runs every shape's forward + dA + dB through correctness
  check vs torch-native ref (per-shape SNR > 25 dB on all three outputs);
  removing the dead def cannot affect numerical correctness because the
  def was never called. PASS confirms no other regression introduced.

## Verdict

This is a **maintenance round** doing real-but-tiny code cleanup. The
dispatcher itself is unchanged from the R10/R11 wins (gpt_oss-Down-B4
var-K dB) and 13 confirmed/falsified rules covering all other 24-shape
suite cells (per R16 consolidated inventory).

## Suggested next round

Continuing the maintenance hold. Patience now 15/30 with 15 rounds of
buffer remaining. Possible activities:

1. **Other dead-code or comment-cleanup opportunities**: there are
   several `if False:` / `elif False:` historical-context blocks in
   `grouped_gemm_fp8_impl.py` (e.g., the R38 var-K dB carve-out at line
   ~1042) and `select_default_config` (e.g., R28 confused-shape rule).
   These are intentional documentation breadcrumbs and should NOT be
   removed.

2. **Zero-commit round**: simply run the metric, confirm score=1000 and
   correctness 0/24, write a 5-line summary doc to track the chain.

3. **Pivot to task-scope expansion** (requires user approval): R7
   architectural ceiling could be re-attacked with a new HK kernel
   primitive (e.g., `ds_read_b128 + register-cvt` instead of DTR + cvt).
   Multi-round HipKittens C++ work; out of scope per current task body.

Recommended: option 2 for the next 5-10 rounds, then re-evaluate.
