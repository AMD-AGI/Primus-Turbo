# 📊 Primus-Turbo Benchmarks

This document presents performance benchmarks for **Primus-Turbo**.


**Work in Progress...**

## DeepEP

### 1. benchmark intranode

```bash
cd benchmark/ops/training && python -m deep_ep.legacy.test_intranode
```


### 2. benchmark internode
You should use slurm or any other tools to run the following:
```bash
cd benchmark/ops/training

export NNODES=
export NODE_RANK=
export MASTER_ADDR=
export MASTER_PORT=

torchrun --nproc_per_node 1 --nnodes "${NNODES}" -node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}"  -m deep_ep.legacy.test_internode
```
