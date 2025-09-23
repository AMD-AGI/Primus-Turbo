# 📊 Primus-Turbo Benchmarks

This document presents performance benchmarks for **Primus-Turbo**.


**Work in Progress...**

## DeepEP

### 1. benchmark intranode

```bash
python benchmark/ops/deep_ep/bench_intranode.py
```


### 2. benchmark internode
You should use slurm or any other tools to run the following:
```bash
torchrun --nproc-per-node 1 --nnodes "$NNODES" --node_rank "$NODE_RANK" benchmark/ops/deep_ep/bench_internode.py
```
