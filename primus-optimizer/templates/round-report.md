# Round {N}: {Optimization Strategy Name}

## Optimization Rationale

### Problem Analysis
{Specific bottleneck identified from profiling data, e.g.:
"The current persistent kernel with BLOCK_M=128 only assigns 1-2 tiles per CU
in decode scenarios where M<512, leaving >60% of 304 CUs idle,
resulting in compute utilization of only 9.8%"}

### Optimization Strategy
{Technical principle of the strategy, e.g.:
"Reduce BLOCK_M from 128 to 64, doubling tile count. For M=256, N=7168,
tiles increase from 2x56=112 to 4x56=224, CU utilization from 36.8% to 73.7%.
The tradeoff is lower per-tile compute density, but for small M the CU
utilization gain far outweighs this cost."}

### Theoretical Expectation
{Expected improvement and basis, e.g.: "Expected +30-50% based on linear
extrapolation of CU utilization increase from 36.8% to 73.7%"}

## Code Changes

### Modified Files
- `primus_turbo/triton/gemm/gemm_kernel.py:L142-L158`

### Diff Summary
{Key diff, not exhaustive}

### Commit
- Branch: `opt/{operator}-{backend}-{session_id}`
- Commit: `{hash}` - `{message}`

## Reproduction Commands

### Accuracy Verification
```bash
# Generates accuracy.log
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/accuracy/eval_gemm_accuracy.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} \
    2>&1 | tee agent_docs/{hw}/{operator}-{backend}/round-{N}/accuracy.log
```

### Baseline Performance
```bash
# Generates baseline.json (run from pre-modification state)
git stash  # or git checkout HEAD~1
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/baseline.json
git stash pop  # or git checkout {worktree_branch}
```

### Optimized Performance
```bash
# Generates optimized.json (run from post-modification state)
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/optimized.json
```

## Results

| Shape (M,N,K) | Baseline (TFLOPS) | Optimized (TFLOPS) | Change |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

- **Geomean improvement**: +X.X%
- **Peak improvement**: +X.X% @ shape (M,N,K)
- **Regression detection**: No regressions / {shape} regressed X%

## Conclusion
{Success/failure, whether to adopt, next direction}
