# B：四个臂的静态定价，以及静态模型的适用边界被划清了

2026-09-21 · **零 GPU 发射** · 四个臂全部带 `arch=gfx1250 / warp=32 / flydsl 0.3.2` 见证

## 起因

`armAB`（causal 跳过 + 32 深收缩的合并臂）在 0917 被打好补丁后**从未计时**，
作业就被停了。计划把「给它定价」排成 Phase B 的第一项，因为这不需要碰卡。

## k_dkdv（占 prod 运行时 67%）

| arm | wmma | tr16 | exp | instr | vgpr | LDS | spill | dscnt |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| baseline | 24 | 18 | 8 | 898 | 255 | 9216 | **0** | 17 |
| armA（causal 跳过） | **24** | 18 | 8 | 904 | 255 | 9216 | **0** | 17 |
| armB（32 深收缩） | **32** | 36 | 16 | 1212 | 335 | **18432** | **0** | 15 |
| armAB | **32** | 36 | 16 | 1217 | 335 | **18432** | **0** | 15 |

`k_dq` 四个臂几乎不动（wmma 恒为 24，LDS 恒为 8192），只有 armA/armAB 的指令数 +21，
即跳过逻辑本身。**优化全部发生在 k_dkdv，符合它占 67% 的账。**

## 这张表说了三件事

### 1. 静态 WMMA 计数**看不见** armA，但 armA 是四个臂里最快的

armA 的 WMMA 计数与 baseline **完全相同**（24），指令数只多 6 条。
然而它在 prod 上实测 **1.64×**（95.736 → 58.272 ms）。

原因不难理解，但必须写下来当作纪律：
**causal 跳过改的是循环跑多少次，不是每次循环做多少活。**
而我们的静态筛数的是**每次迭代**的指令。它对 trip count 是瞎的。

> **这划清了那个「WMMA 计数把运行时预测到 4% 以内」的结论的适用边界。**
> 它成立的条件是**结构不变、只比密度**。round 1 用它预测 `dkdv ≈ 3× dq` 成立，
> 因为那两个内核的 trip count 都由同一组形状参数决定。
> 拿它去给一个改变 trip count 的改动定价，会得到「没有变化」——而真实收益是 1.64×。
>
> 这不是模型失效，是**模型被用错了地方**。
> 计划 §九.5 写的是「偏差 > 10% 即对该结构失效」；这里把它说得更准：
> **静态筛给每次迭代的工作量定价，不给迭代次数定价。**
> 改 trip count 的候选（causal 跳过、tile 配对、split-K）**必须上卡测**。

### 2. 静态 WMMA 计数**准确地**给 armB 定了价

armB 的 WMMA 从 24 涨到 **32**，乍看是变差了。不是。
armB 每次迭代处理 **32 个 query 而不是 16 个**。所以按单位工作量算：

| | 每次迭代的 query | WMMA | WMMA / query-16 |
|---|--:|--:|--:|
| baseline | 16 | 24 | 1.50 |
| armB | 32 | 32 | **1.00** |

即 **1.5×** 的矩阵指令密度改善——和文档里记的「每 (16kv × 32q) 的 WMMA 数 48 → 32」
（48/32 = 1.5）**精确一致**。静态模型在这里是对的。

**但实测只有 1.13×，不是 1.5×。** 差额在哪，这张表也给了候选答案：
**LDS 从 9216 翻倍到 18432 字节**，vgpr 从 255 涨到 335。
这正是 gfx950 上「放大 KV block 实测慢 3–4×，机理是占用率塌」那条死路的同一族机理。
**armB 把省下来的矩阵指令，又从占用率上还回去了一部分。**

### 3. armAB 干净，可以安全地上卡测

armAB 的静态画像 = armB 的代价结构（32 wmma / 18432 LDS / 335 vgpr）
+ armA 的控制流（instr 1217 vs armB 的 1212，多出来的 5 条正是跳过逻辑，
与 armA 相对 baseline 多出的 6 条同源）。

**`spill = 0`，`private_segment_fixed_size = 0`，LDS 18432 在预算内**——
过了静态硬门，不会撞上「会 spill 的构建在 gfx1250 上第一次发射后挂起」那条。

**所以 armAB 是一个可以安全发射的候选**，这就是这次静态筛要回答的问题。

### 一个不能当结论用的估计

如果两者独立相乘：1.64 × 1.13 = **1.85×**，即 prod 约 51.7 ms / 106 TF/s。
**但没有理由相信它们独立**：armAB 继承了 armB 那份翻倍的 LDS，
而 armA 的收益来自少跑迭代——迭代少了之后，占用率损失的相对权重会变。
**这个数字写在这里只是为了让实测有个可以证伪的靶子，不是预测。**

## 未解决的：armAB 的正确性仍然没有任何证据

`benchAB.json` 每一行只有 timing，**没有 SQNR，没有 determinism**，
而且 armAB 连 timing 都没有。计划 §九.1 的那条通道原样成立：
一个在对角线 tile 边界 off-by-one 的 causal 跳过会**同时变快并且仍然过 50 dB**（余量只有 2.5 dB）。

**上卡后的第一件事仍然是跑 `validation.py` 加 armA-vs-baseline 的逐元素 bitwise 比对**，
不是计时。

## 复现

```bash
R=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
S=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/rounds/001/_scratch
for arm in armA armB armAB; do
  docker run --rm --network none --user "$(id -u):$(id -g)" \
    -v /home/lihuzhan:/home/lihuzhan -v /tmp/flyhome:/tmp/flyhome \
    -e HOME=/tmp/flyhome -e TMPDIR=/tmp/flyhome -e PYTHONDONTWRITEBYTECODE=1 \
    -e COMPILE_ONLY=1 -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
    -e FLYDSL_ROCM_AGENT_TIMEOUT=10 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
    --entrypoint python3 fa-tune:deps \
    $R/output/0921__flydsl/bin/compile_only_driver.py \
    --impl $S/$arm --dump-dir $R/output/0921__flydsl/isa/$arm \
    --json $R/output/0921__flydsl/isa/$arm.json
done
```

**`--user "$(id -u):$(id -g)"` 是必须的。** 不加它容器以 root 身份写出产物目录，
宿主这边之后就写不进去了——op-evolve 为同一个原因在每条命令前加 `PYTHONDONTWRITEBYTECODE=1`。
命令里**没有** `--device /dev/kfd`、**没有** `--device /dev/dri`，这是有意的。
