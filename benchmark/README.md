# 📊 Primus-Turbo Benchmarks

This document presents performance benchmarks for **Primus-Turbo**.


**Work in Progress...**

## DeepEP

### 1. benchmark intranode

```bash
python benchmark/ops/training/deep_ep/test_intranode.py
```


### 2. benchmark internode
You should use slurm or any other tools to run the following:
```bash
export NNODES=
export NODE_RANK=
export MASTER_ADDR=
export MASTER_PORT=

torchrun --nproc_per_node 1 --nnodes "${NNODES}" -node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"  benchmark/ops/training/deep_ep/test_internode.py
```

## MegaMoE

Intra-node EP benchmarks of the fused BF16 MegaMoE kernels. See [MegaMoE](../docs/mega_moe.md) for the design.

### 1. gfx950 (MI355X)

`--mode` selects the fused kernel: `dispatch_grouped_gemm` (dispatch + grouped GEMM), `grouped_gemm_combine` (grouped GEMM + combine), or `activation` (the gated-activation kernels).

```bash
python benchmark/ops/training/bench_mega_moe.py --mode dispatch_grouped_gemm --models DeepSeek-V3 --num-processes 8
python benchmark/ops/training/bench_mega_moe.py --mode grouped_gemm_combine --models DeepSeek-V3 --num-processes 8
```

### 2. gfx1250 (MI455X)

Times the six GEMM stages of one forward + backward step, then the whole step through `fused_mega_moe`.

```bash
python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4
python benchmark/ops/training/bench_mega_moe_gfx1250.py --num-processes 4 --num-experts 128 --skew 1 -o out.csv
```
