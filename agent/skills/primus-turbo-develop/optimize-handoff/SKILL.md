---
name: optimize-handoff
description: Primus-Turbo handoff to the autonomous kernel-optimize loop — collect the prerequisites (kernel path, focused test/bench commands, scoring metric, execution mode, quick-validation harness) a kernel campaign needs and pass them on. Use when pushing a Primus-Turbo kernel toward the hardware limit, not just spot-checking perf.
---

# Optimization Handoff (kernel-optimize)

When the goal is to push a kernel toward the hardware limit (not just spot-check perf), this file is the interface between Primus-Turbo and the autonomous optimization loop in [`../../kernel-optimize/SKILL.md`](../../kernel-optimize/SKILL.md). It supplies the prerequisite information that loop requires. Read the hub [`../SKILL.md`](../SKILL.md) first.

## Process

1. **Verify the environment** — `pip show primus_turbo`; reinstall editable if not (hub `SKILL.md` → Build).
2. **Read `kernel-optimize/SKILL.md` "Prerequisite Information"** to learn what the loop needs.
3. **Collect the prerequisites** for the target op (tables below + the code map / commands in the hub `SKILL.md`).
4. **Hand off** to `kernel-optimize/SKILL.md`; it runs DEFINE_TARGET → PREPARE_ENVIRONMENT → BASELINE → (ANALYZE → OPTIMIZE → VALIDATE)* → REPORT.
5. **Acceptance** — back in the project: run the **full** suite `pytest tests/pytorch/ -v`, review the report, confirm a clean diff/commit.

Before the loop starts, read [`../../../rules/iteration_rules.mdc`](../../../rules/iteration_rules.mdc): one hypothesis per round, correctness gate before perf, accept/rollback lineage, and every accepted gain must transfer to a real training step (no `id(...)`-keyed activation/grad_out caches, no uniform-distribution-only GroupGemm shortcuts).

## Prerequisite information (per op)

| Requirement | Where to get it |
|-------------|-----------------|
| Kernel source file path | hub `SKILL.md` → Code Map |
| Focused test command | hub `SKILL.md` → Test & Bench; e.g. `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"` |
| Focused benchmark command | [`../verify-performance/SKILL.md`](../verify-performance/SKILL.md); e.g. `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise` |
| Benchmark output format / metrics | [`../verify-performance/SKILL.md`](../verify-performance/SKILL.md) (`Forward/Backward TFLOPS`, `Check` gate) |
| Quick validation harness | "Quick validation" below |
| Scoring rules | "Optimization scoring" below |
| `execution_mode` + rebuild | "Optimization environment" below |

## Optimization environment

| Backend / language | Recommended `execution_mode` | Rebuild after change |
|--------------------|------------------------------|----------------------|
| Triton (Python) | `repo-mode` | none (editable install) |
| HIP / CK (C++), small/param changes | `repo-mode` | `GPU_ARCHS=<arch> pip install --no-build-isolation -e . -v` |
| HIP / CK (C++), new kernel / heavy trial-and-error | `workspace-mode` | per the minimal stack in `kernel-optimize/SKILL.md` |

## Optimization scoring

The `Check` column is a hard gate — any `FAIL`/`ERROR` shape rejects the candidate.

| Campaign type | `primary_metric` (aggregate = geometric mean across PASS shapes) |
|---------------|------------------------------------------------------------------|
| Compute-bound, forward only | `Forward TFLOPS` |
| Compute-bound, forward + backward (training) | `Combined Step TFLOPS` (forward/backward kept as diagnostics) |
| Memory-bound (elementwise, quant) | `Forward GB/s` / `Backward GB/s` |

Combined-step derivation (preferred for fwd+bwd GEMM-class ops) is in [`../verify-performance/SKILL.md`](../verify-performance/SKILL.md). Confirm `primary_metric` with the user at DEFINE_TARGET.

## Quick validation

**Two tools, separate roles** — do not compare absolute numbers across them (they use different timing engines):
- **`quick_test_bench.py`** (this folder, copied into the campaign dir) is the **authoritative source for round-to-round per-shape regression gating**. BASELINE (round-1) and every VALIDATE round run THIS script and compare its `--summary-csv` output. It uses a hand-rolled `time.perf_counter` loop.
- the **full `bench_<op>_turbo.py`** (see [`../verify-performance/SKILL.md`](../verify-performance/SKILL.md), `torch.utils.benchmark.Timer`) is used only to (i) pick `representative_shapes` from the full shape set and (ii) do final acceptance after the campaign.

Both validation levels run via `quick_test_bench.py`, differing only in `SHAPES`:
- **full validation** (BASELINE + final acceptance) → `SHAPES` = all `target_shapes`.
- **quick validation** (each VALIDATE round) → `SHAPES` = the 3–5 `representative_shapes` subset.

`representative_shapes`: only `Check=PASS` shapes from the BASELINE run, covering small (launch overhead) + large (compute/memory) scales, preferring high-variance shapes. Record them in `manifest.yaml: representative_shapes`.

**Harness:** the template [`quick_test_bench.py`](quick_test_bench.py) is the canonical reference. During PREPARE_ENVIRONMENT, copy it into `<campaign_dir>/quick_test_bench.py` with `SHAPES` empty and `<target_backend>` set; fill `SHAPES` with all `target_shapes` for BASELINE, then narrow it to `representative_shapes` for the VALIDATE rounds. Adapt the op/config for your target; keep the `--summary-csv` writer and the column schema (see the schema-scope note below for non-GEMM ops).

**Measurement-consistency contract:** BASELINE (round-1) and every VALIDATE read the same authoritative CSV via `--summary-csv`, parsed identically across rounds by the **same CSV reader** — the optional `mcp__turbo__parse_bench_csv` MCP helper if it is configured, otherwise a direct `csv.DictReader` keyed on the columns below. The specific reader does not matter; freezing the schema does. The schema MUST match these columns verbatim (any drift silently disables the per-shape regression gate):

```
label,B,M,N,K,Check,Forward TFLOPS,Forward TFLOPS_stddev,Backward TFLOPS,Backward TFLOPS_stddev,Forward Time (ms),Backward Time (ms),out_snr,da_snr,db_snr
```

`Check` uses `PASS` / `FAIL`; `*_stddev` are absolute stddev in the metric's unit. Record the invocation in `manifest.yaml` as `quick_command: "python <campaign_dir>/quick_test_bench.py"`; BASELINE and every VALIDATE append `--summary-csv <round_dir>/artifacts/benchmark.csv`.

**Schema scope:** the columns above are **GEMM-class specific** (`M,N,K` dims, `Forward/Backward TFLOPS`, `out_snr/da_snr/db_snr`). For a non-GEMM op, keep `label`, `Check`, and the `*_stddev` convention, but replace the GEMM-only columns with the ones your `primary_metric` needs — e.g. a memory-bound op (per "Optimization scoring") uses `Forward GB/s` / `Backward GB/s` instead of TFLOPS, and a single-output op replaces `out_snr/da_snr/db_snr` with its own correctness metric. Whichever schema you pick, freeze it at BASELINE, keep it byte-identical across all rounds, and update whichever CSV reader you use (the `mcp__turbo__parse_bench_csv` mapping if it is configured, else your direct `csv.DictReader`) to match — the per-shape regression gate keys on exact column names.
