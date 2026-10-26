# Mega MoE gfx1250：TDM gather 偶发数据损坏（排查记录）

- 状态：**修复进行中**。根因已缩小到 TDM gather 的硬件行为；候选修复（A 改用 per-lane async 加载）在复现器上 0 错误，但 fwd 慢 4-8%，还在优化性能。
- 分支：`feat/mega_moe/gfx1250-bf16`，kernel 基线 `1149c02b`（EP2 campaign round 9）
- 机器 / 容器：MI455X（gfx1250）x4，`xiaompen_mega_gfx1250_ep4`
- 复现和分析脚本：`tools/tdm_gather_bug/`（原始位置 `/home/xiaompen/mega_scratch/ep4c/`，`stress.sh` 里写死了这个路径）
- 当前未提交的 kernel 改动：`tools/tdm_gather_bug/wip_async_a.patch`（带实验用环境变量开关，不是最终形态）

## 1. 问题

EP4 下 Mega MoE BF16 的 fused `dispatch_gemm`（L1 fwd 和 b_nn 的 L2 dgrad）偶发**静默错误结果**，没有 fault，也不是每次都出现：

- 端到端：`uniform` routing 的一次 e2e 跑出过 Inf（mega gain 测量时）。
- 整步 stress（`stress_ep.py`，fwd+bwd）：约 6 / 26000 步有错。
- 只跑 fused fwd `dispatch_gemm` 的复现器：约 1 / 500 次 launch 有错（`1149c02b`）。

### 错误特征（dump 后最小二乘反解 A 数据得到）

- 每次只坏 **一个 256x256 tile**（一个 N-block），坏行以 **16 行块**为单位，即一个 gather op（16 行 x 16 位索引）或者一个 WMMA M 子块的大小。
- 两类表现，常在同一次事件里同时出现：
  1. **A 块读到旧数据**：某个 16 行块在某个 k-tile `kt` 读到的是同一行 `kt-2` 的数据（拟合残差约 0.002），也就是那一轮 LDS 槽没被写；而相邻槽（LDS 地址 +0x1100 = 16 行 x 272 B）里混进了前一个槽本该有的 `kt` 数据，有时跨 wave。读这个块的两个 wave（wave_n 0/1）都错，说明 LDS 里的数据本身就是错的。
  2. **单个 wave 的 B 片段错**：只有一个 wave 的某 16 列（一个 `wn` 片段）错，同样读这些 B 行的其他 wave 是对的。
- 坏的行总在同一个 wave_m 组（共享同一组 A 行的两个 wave）里；20 次里 19 次 `kt >= 28`（K 的后半段）。
- 坏 tile 绝大多数是**满的 tile**（256 行有效），并且和带填充行的 tile 属于同一个 expert，几乎同时运行在相邻的 WGP 上。

### 根因（当前理解）

不是通信，也不是软件同步协议的问题（每个 wave 都先 `s_wait_tensorcnt 0`，再过 barrier，然后才读下一级 stage）。
是 **TDM gather（`tensor_load_to_lds` gather 模式）的硬件问题**：当同时有"几乎不访存"的 gather op 在跑时，
**相邻 WG** 的 TDM gather 写入会被弄错，表现为没写、写偏一个槽或者写晚。这类快速 op 包括：
- 越界的填充行（索引 -1，在 16 位里是 0xFFFF）；
- 全是同一行的重复；
- 热点零行；
- extent 为 0 的 dead tile op。

真实路由数据里的热点 token 行也会触发，只是概率更低。所以只调整索引或数据布局不能根治。

## 2. 复现

脚本都在 `tools/tdm_gather_bug/`，用 `stress.sh` 在 4 卡上跑 torchrun，日志在 `ep4c/logs/`：

```bash
# stress.sh COMMIT|repo ITERS [args]；SCRIPT 选脚本，TAG 区分日志，CACHE 指定独立的 FlyDSL cache
cd /home/xiaompen/mega_scratch/ep4c
TAG=x SCRIPT=stress_dg.py ./stress.sh repo 20000            # fused fwd dispatch_gemm，bitwise 对比
TAG=x SCRIPT=stress_dg.py ./stress.sh repo 20000 --b_nn     # fused b_nn（dgrad）
# 不带通信：先用 fused 填好对称内存里的 token buffer，然后只反复跑独立的 gather GEMM
TAG=x SCRIPT=stress_dg.py ./stress.sh repo 20000 --standalone --no_barrier --skew_us 0 [--copy_a] [--no_dead] [--remap_pad]
# 纯 GEMM，单卡、不需要通信（4 个进程各用一张卡）：随机索引，可以加填充行
TAG=x SCRIPT=stress_gemm.py ./stress.sh repo 20000 --pad 70 [--pad_idx 0] [--zero_rows 16] [--sorted]
```

- `stress_dg.py --dump DIR`：保存出错 launch 的 `out/ref/valid/handle`（每个 rank 最多 2 个，每个 1-2 GB，文件属 root，要用 docker exec 删）。
- 分析：
  - `analyze_valid.py`：坏 tile 和有效行数的关系；
  - `analyze_cols.py`：坏行的哪些 16 列块错了，用来区分是 LDS 数据错还是单个 wave 读错；
  - `analyze_dg4.py`：贪心地找出每个坏行的 (k-tile, 数据来源) 替换；
  - `analyze_dg5.py` / `analyze_dg6.py`：反解实际用到的 A 数据，按 16 元素块追溯出处。

## 3. 已经做过的尝试

如无特别说明，每行都是 4 卡、每卡 20000 次 launch 的出错 launch 数。

| # | 改动 / 实验 | 结果 | 结论 |
|---|---|---|---|
| 1 | flag 之后加 acquire fence（`global_inv SCOPE_SYS`） | 整步 1 / 20000 | 不是通信可见性问题，已回退 |
| 2 | 每条 TDM op 后加 `s_wait_xcnt 0`，描述符 SGPR 作为 asm 输入（防止 WAR 改写） | fwd 约 5 / 20000（4、7、5），b_nn 3 / 10000 | 降了约 10 倍，没消除 |
| 3 | 2 的基础上改 32 位索引（每条 op 8 行） | fwd 2 / 60000，b_nn 1 / 20000 | 再降约 7 倍，没消除 |
| 4 | B 改由 wave 5 加载（原来是 wave 1） | 5 | 无影响，坏槽分布有变化 |
| 5 | `s_wait_tensorcnt` 和 barrier 之间加 `s_sleep 10`（640 周期） | 8 | 不是"数据只是写晚了一点" |
| 6 | TDM `early_timeout=False` | 3 | 无影响 |
| 7 | 两个 gather op 的描述符一直保持到 `s_wait_tensorcnt` 之后（ISA 已确认不被改写） | 12 | "描述符被晚读"不成立 |
| 8 | 独立 gather GEMM 跑在对称内存的 token buffer 上，没有任何通信 | 13 | 不需要通信也会触发 |
| 9 | 8 + A 拷到普通显存 | 8 | 和内存类型无关 |
| 10 | 8 + 去掉 dead tile | 7 | 和 dead tile 无关 |
| 11 | 纯 GEMM（`stress_gemm`），随机索引，无填充 / 排序 | 0 / 80000、0 / 80000 | 正常的 gather 不出错 |
| 12 | 纯 GEMM，每个 expert 的最后一个 tile 填 70 行 -1 | 10 / 160000 | **越界填充是触发源之一** |
| 13 | 纯 GEMM，填充行改指向第 0 行（重复） | 2 / 160000 | 重复读同一行也会触发 |
| 14 | 纯 GEMM，填充行改指向 16 个不同的零行（在界内） | 4 / 160000 | 热点行同样触发，不只是越界 |
| 15 | 在 12 里统计坏 tile 的位置 | 坏的大多是满 tile，和填充 tile 同 expert | **跨 WG 干扰**，WG 内部串行化救不了 |
| 16 | 真实路由的独立 GEMM，填充改指向分散的合法行（另加去掉 dead tile） | 6 / 40000（9 / 40000） | 真实数据里的热点 token 也会触发 |
| 17 | **A 的 gather 改用 per-lane `global_load_async_to_lds_b128`**（VMEM，ASYNCcnt），B 仍用 TDM 稠密加载 | 纯 GEMM 加填充 0 / 80000（和 torch 对拍通过），fused fwd 0 / 20000 | **候选修复** |

### 候选修复（#17）的性能，和 `1149c02b` 交替 A/B（`tools/ab.sh async`）

| shape | base fwd ms | async fwd ms | base step TF | async step TF |
|---|---|---|---|---|
| dsv3_t8192_random r1 | 4.159 | 4.316（+3.8%） | 1498.7 | 1470.1（-1.9%） |
| dsv3_t8192_random r2 | 4.179 | 4.365（+4.5%） | 1430.7（bwd 有噪声） | 1464.8 |
| dsv3_t4096_random r1 | 2.301 | 2.487（+8.1%） | 1278.3 | 1255.2（-1.8%） |
| dsv3_t4096_random r2 | 2.331 | 2.470（+6.0%） | 1276.6 | 1248.1（-2.2%） |

fwd 稳定变慢 4-8%；bwd 的差异在噪声范围内。

#17 的实现要点（见 patch）：
- 每个 lane 负责 16 B 块，每轮覆盖 16 行，每个 k-tile 发 16 条 SADDR 形式的异步加载；
- 行偏移用 32 位，K 不变，只算一次，多占 16 个 VGPR，总共 507 个，没有 spill；
- 无效行（填充或 dead tile）读 A 的第 0 行，写到它自己的填充行位置；docstring 里说明过，填充行的输出下游不会读；
- 本轮发出的加载在 `split_fence` 里用 `s_wait_asynccnt 0` 等待。

## 4. 接下来的计划

1. **把 #17 的 fwd 性能拉回基线**，每次只改一个变量：
   - 16 条异步加载不要集中在 ks=0 一次发完，分散到各个 ks 子步里，和 MMA 交错；
   - 减少地址计算的 VALU；
   - 试 cache policy（比如 NT 或 streaming）；
   - 必要时调整 lane 和行的映射，比如一个 wave 连续读几行。
2. **验证**（修复版）：
   - `stress_dg` fwd 和 `--b_nn` 各至少 40000 次、0 次出错；
   - 纯 GEMM 加填充至少 160000 次、0 次出错；
   - 整步 `stress_ep` 20000 步；
   - pytest：EP1、EP2、EP4（`run_round.sh N full`）；
   - e2e：even、uniform、real 三种 routing。
3. **清理代码**：
   - 删掉 TDM gather 路径、实验开关（`MEGA_*`）和 hold 相关代码；
   - 更新 `gemm_nt_tile.py` 的 docstring；
   - launcher 加断言 `a_rows * lda * 2 < 2^31`（32 位行偏移的前提）。
4. **排查其他 TDM 使用者**：
   - grouped wgrad（variable_k，EP2 round 9 引入的 TDM gather）有没有同样的风险；
   - dead tile 发的 extent 为 0 的稠密 TDM（B 加载）是否也会成为触发源，必要时改成界内的正常加载。
5. 验证通过后 commit + push 修复，并把这份文档更新为结论版。然后补记 EP4 campaign round 5，继续优化，每个 accept 都 commit + push。
