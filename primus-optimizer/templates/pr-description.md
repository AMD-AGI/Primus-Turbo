## Summary

Optimize {operator} ({backend} backend) on {hw} through {N} rounds
of iterative optimization, achieving **+{geomean}% geomean improvement**
({baseline_tflops} -> {optimized_tflops} TFLOPS).

### Key Changes

- **Round 1 -- {strategy_1}**: {one-line description} (+{pct_1}%)
- **Round 3 -- {strategy_3}**: {one-line description} (+{pct_3}%)
- ...

### Optimization Rationale

{2-3 paragraph technical explanation covering:
  - What bottleneck was identified (e.g., CU underutilization at small M)
  - What approach was taken and why (e.g., tile size downgrade trades
    per-tile compute density for higher CU occupancy)
  - Why this is safe (accuracy verification results)}

## Performance Results

| Shape (M,N,K) | Baseline (TFLOPS) | Optimized (TFLOPS) | Change |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

**Geomean**: +{X.X}% | **Peak**: +{X.X}% @ {shape} | **Regressions**: None

## Accuracy Verification

- {dtype} accuracy: all {N} test cases passed (SNR > {threshold} dB)
- Reproduce: `{accuracy_command}`

## Reproduce Benchmarks

```bash
# Baseline
{baseline_command}

# Optimized
{optimized_command}
```

## Test Plan

- [ ] Accuracy check: `{accuracy_command}`
- [ ] Performance benchmark: `{benchmark_command}`
- [ ] Regression check: `python tools/check_regression.py --baseline {path}`
