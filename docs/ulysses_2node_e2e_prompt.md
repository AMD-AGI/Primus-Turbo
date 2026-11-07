# Ulysses GemmA2A / A2AGemm：Qwen3-30B-A3B 2 节点端到端训练测试 prompt

> 把下面整段交给 agent 作为任务描述，在 2 台 8×MI355X 节点上执行。单节点结果和实现说明见 `docs/ulysses_gemm_a2a.md`。

---

在 2 个节点（每节点 8 张 MI355X）上跑 Qwen3-30B-A3B 的 Megatron 预训练，对比 Ulysses CP 的三种实现的每步耗时，确认单节点上测到的融合收益在多节点下仍然成立。全程用中文汇报，不用 CUDA graph，结束后清理遗留进程。

## 代码

| 仓库 | 分支 | 用到的内容 |
|---|---|---|
| Primus-Turbo（`git@github.com:AMD-AGI/Primus-Turbo.git`） | `feat/ulysses-gemm-a2a` | `ulysses_qkv_gemm_a2a` / `ulysses_a2a_o_gemm`（`primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`，kernel 在 `primus_turbo/flydsl/ulysses/`） |
| Primus（`git@github.com:AMD-AGI/Primus.git`） | `feat/ulysses-fused-a2a` | patch `primus/backends/megatron/patches/ulysses_fused_a2a_patches.py`，配置 `examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml` |

两个节点上的代码必须是同一个 commit。

## 环境（两个节点都要做）

- 容器镜像：`rocm/primus:v26.8`（单节点测试用的就是它）。
- Turbo：在容器里编译这个分支（`GPU_ARCHS=gfx950 pip install --no-build-isolation -e . -v`，或按仓库 README）。
- FlyDSL：融合 kernel 需要 `flydsl==0.3.4.1`，镜像自带的是 0.2.4。单节点上的装法是 `pip install --no-deps --target <dir> flydsl==0.3.4.1`，然后把 `<dir>` 放到 Turbo 前面。
- **坑**：Primus 的 `runner/helpers/envs/base_env.sh` 会把 site-packages 插到 PYTHONPATH 最前面，于是训练进程可能加载镜像里装好的旧 `primus_turbo` / `flydsl`，而不是这个分支。单节点上的解决办法是在一个目录里放下面这个 `sitecustomize.py`，并把这个目录加进 PYTHONPATH（路径按实际修改）：

  ```python
  import sys

  for p in ("/path/to/Primus-Turbo", "/path/to/flydsl0341"):
      while p in sys.path:
          sys.path.remove(p)
      sys.path.insert(0, p)

  _sys_sc = "/usr/lib/python3.12/sitecustomize.py"
  with open(_sys_sc) as f:
      exec(compile(f.read(), _sys_sc, "exec"))
  ```

  如果用 `pip install -e` 替换掉了镜像里的 `primus_turbo`、flydsl 也升级到了 0.3.4.1，就不需要这一步。**无论用哪种方式，都要在训练进程的环境下确认** `primus_turbo.__file__` 指向这个分支，并且 `from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_qkv_gemm_a2a` 能成功；baseline 也必须用同一份 Turbo，否则对比无效（单节点上旧 Turbo 的 baseline 慢了 5%）。

## 并行配置

- TP=1，PP=1，**CP=4（Ulysses，`cp_comm_type: a2a`）**，EP=8，16 卡下 DP=4。
- CP 组是连续的 4 个 rank，在节点内，走 XGMI；跨节点的只有 DP 梯度规约（`overlap_grad_reduce`）和参数 gather。EP=8 也在节点内。
- `micro_batch_size=1`，**`global_batch_size=32`**：每个 DP rank 8 个 micro-batch，和单节点（GBS 16，DP 2）一致，方便对比。
- 融合 patch 只在 TP=1、CP>1 且 CP 整除每节点 GPU 数时生效，否则打印 `keeping the stock path` 警告并保持原路径。

## 要跑的 6 组

序列长度 16K 和 32K，各三种实现：

| 标签 | 额外环境变量 | 含义 |
|---|---|---|
| `base` | 无 | 原路径：zigzag CP 切分 + Turbo `flash_attn_usp_func`（a2a 在 attention 内部，QKV / o_proj GEMM 在外面） |
| `serial` | `PRIMUS_ULYSSES_FUSED_A2A=1 PRIMUS_ULYSSES_FUSED_A2A_IMPL=serial` | 消融：contiguous 切分 + 本地 flash-attn，但 GEMM 和 RCCL all-to-all 串行 |
| `fused` | `PRIMUS_ULYSSES_FUSED_A2A=1` | GemmA2A（QKV GEMM + a2a）/ A2AGemm（a2a + o_proj GEMM）融合 |

每个节点上的启动方式（`NODE_RANK` 分别为 0 / 1，`MASTER_ADDR` 是 0 号节点；网卡相关的 NCCL / RCCL 环境变量按集群惯例设置，可参考 `runner/use_ainic.yaml`）：

```bash
export NNODES=2 GPUS_PER_NODE=8 NODE_RANK=<0|1> MASTER_ADDR=<node0> MASTER_PORT=29500
export PYTHONPATH=<sitecustomize 目录>:<flydsl0341 目录>:<Primus-Turbo 目录>${PYTHONPATH:+:$PYTHONPATH}
export PRIMUS_SEQ_LENGTH=16384 PRIMUS_MAX_POSITION_EMBEDDINGS=16384   # 32K 时两个都改成 32768
export PRIMUS_TRAIN_ITERS=15 PRIMUS_GBS=32 PRIMUS_EXP_NAME=<标签><长度>
# 加上上表里的额外环境变量
cd <Primus 目录>
./runner/primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml > <标签><长度>.log 2>&1
```

6 组依次跑，同一对节点，中间不要有别的任务。

## 检查和指标

- fused / serial 的日志里必须有 `[Patch] ✓ Applied: megatron.ulysses_fused_a2a`；base 里不能有。
- 每步耗时取 `elapsed time per iteration (ms)`，前 2 步是 warmup，取第 3 步之后的平均；同时记录 `TFLOP/s/GPU` 和 `tokens/s/GPU`。
- loss 健全性：mock 数据下三种实现最后一步的 `lm loss` 应在 1e-3 量级内一致，没有 NaN 和跳过的步。
- 收益拆分：`fused` 对 `base` 是总收益；`fused` 对 `serial` 是 GEMM / a2a 重叠本身的收益；`serial` 对 `base` 是路径重构（contiguous 切分 + 本地 flash-attn）的收益。
- 如果 fused 相对单节点的收益明显缩水，用 rocprofv3 抓 fused 和 base 各一步的 trace，看 attention 段的耗时是否变了，还是跨节点的 DP 通信 / 等待把差距盖住了。

## 单节点参考（1×8 MI355X，CP=4，DP=2，EP=8，GBS=16，ms/iter）

| 长度 | base | serial | fused | fused 对 base |
|---|---|---|---|---|
| 16K | 3147 | 3094 | 2897 | -7.9% |
| 32K | 6583 | 6534 | 6103 | -7.3% |

## 汇报

一张表：6 组的 ms/iter、TFLOP/s/GPU、最后一步 loss，以及和单节点的对比；再用几句话说明收益在多节点下是否保持，没保持的话原因是什么。日志和使用的 commit 一起附上。
