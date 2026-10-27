# Mega MoE gfx1250：TDM gather 偶发数据损坏（排查记录）

- 状态（2026-10-27 更新）：**规避方案已合入（默认开启）**。A 的 gather 改用 per-lane async 加载。第 6 节合入版本的"无效行读 `pos` 行"在 GPT-OSS staged-bias 测试上回归（读到 dispatch 还没写完的行，WGP 缓存里留下旧数据），已修复，见第 7 节。修复后 pytest EP1 / EP2 / EP4 全过，纯 GEMM、fused fwd / b_nn / churn、整步 `stress_ep` 压测全 0 错误；纯 GEMM 比 TDM 慢约 0.8%（无填充）/ 2.7%（每 expert 128 行填充）。对比 `1149c02b`：kernel 级 step TF 慢约 1.9%，e2e 单步慢 0.8-1.0%，loss 跟住。**还没做**：第 5 节第 5 条的清理（删 TDM gather 路径和 `MEGA_*` 开关）。根因判断为 gfx1250 TDM gather 的硬件行为（见第 4 节）。
- 下面第 1-5 节是上午暂停时的记录（当时 kernel 保持 `1149c02b`，规避方案在 `wip_async_a.patch` 里）；`MEGA_TDM_GATHER=1` 仍可切回 TDM gather 复现。
- 分支：`feat/mega_moe/gfx1250-bf16`，kernel 基线 `1149c02b`（EP2 campaign round 9）
- 机器 / 容器：MI455X（gfx1250）x4，`xiaompen_mega_gfx1250_ep4`
- 复现和分析脚本：`tools/tdm_gather_bug/`（原始位置 `/home/xiaompen/mega_scratch/ep4c/`）。`stress.sh` 默认用脚本所在目录和本仓库，路径、容器和 GPU 都可以用环境变量覆盖，见第 2 节"换机器继续"
- 规避方案和实验开关：`tools/tdm_gather_bug/wip_async_a.patch`（基于 `1149c02b`，`git apply` 即可；没有合入）。开关：
  - `MEGA_TDM_GATHER=1`：A 走原来的 TDM gather，默认走 async 加载；
  - `MEGA_A_SPREAD=N`：async 加载分到 N 个 ks 子步发出，默认 0，即在 ks=0 一次发完；
  - 其余 `MEGA_IDX_BITS` / `MEGA_B_WAVE` / `MEGA_ET` / `MEGA_SLEEP` / `MEGA_HOLD` 对应第 3 节的 #2-#7。
  - 实验 #2 的 `s_wait_xcnt` + 描述符 hold 也在这个 patch 里。

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

### 换机器继续

需要 4 张 gfx1250（纯 GEMM 复现每个进程只用一张卡），环境和 `amdprimus/amdprimus:tas-20261019` 镜像一致。

```bash
git clone -b feat/mega_moe/gfx1250-bf16 git@github.com:AMD-AGI/Primus-Turbo.git && cd Primus-Turbo   # 按常规方式 build
T=agent/workspace/mega_moe_flydsl_gfx1250_ep4_20261026/tools/tdm_gather_bug
# 规避方案已合入，不用再打 wip_async_a.patch（会冲突）；MEGA_TDM_GATHER=1 切回 TDM gather 复现
cd $T
# 在容器里直接跑：CONTAINER= ；在宿主机上跑：CONTAINER=<容器名>。日志在 $T/logs/
CONTAINER= TAG=tdm MEGA_TDM_GATHER=1 CACHE=/tmp/fly_tdm SCRIPT=stress_gemm.py ./stress.sh repo 20000 --pad 70   # 复现，预期 4 卡合计约 5 次出错
CONTAINER= TAG=async CACHE=/tmp/fly_async SCRIPT=stress_gemm.py ./stress.sh repo 20000 --pad 70                 # 合入的 async 版本，预期 0 次
CONTAINER= TAG=bench SCRIPT=stress_gemm.py ./stress.sh repo 1 --bench 200                    # 纯 GEMM 计时
```

- 其他可覆盖的变量：`TURBO`（仓库根目录）、`CACHE`（FlyDSL cache，换 kernel 变体时用独立目录）、`GPUS`、`NPROC`、`LIMIT`（超时秒数，由 `capped.sh` 强制）。
- `stress.sh COMMIT ...`（不是 `repo`）要求 `$V/src_COMMIT` 下有对应 commit 的源码树。做法：把仓库的 `primus_turbo` 用 `cp -as` 做成符号链接副本，再把和 HEAD 不同的文件换成该 commit 的版本。
- 整步压测是 `stress_ep.py`，也就是不设 `SCRIPT` 时的默认脚本。

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
| 18 | 17 的基础上，把 16 条 async 加载分到 2 / 4 个 ks 子步发出（`MEGA_A_SPREAD`），和 MMA 交错 | 纯 GEMM 单次 2166 / 2380 us（17 是 2112 us） | 更慢，开销来自加载指令和 MMA 抢发射槽 |
| 19-28 | #17 的性能优化（下午），逐项见第 6 节 | 纯 GEMM 2112 → 2034 us | 已合入 |

### 候选修复（#17）的性能，和 `1149c02b` 交替 A/B（`tools/ab.sh async`）

| shape | base fwd ms | async fwd ms | base step TF | async step TF |
|---|---|---|---|---|
| dsv3_t8192_random r1 | 4.159 | 4.316（+3.8%） | 1498.7 | 1470.1（-1.9%） |
| dsv3_t8192_random r2 | 4.179 | 4.365（+4.5%） | 1430.7（bwd 有噪声） | 1464.8 |
| dsv3_t4096_random r1 | 2.301 | 2.487（+8.1%） | 1278.3 | 1255.2（-1.8%） |
| dsv3_t4096_random r2 | 2.331 | 2.470（+6.0%） | 1276.6 | 1248.1（-2.2%） |

fwd 稳定变慢 4-8%；bwd 的差异在噪声范围内。

纯 GEMM 计时（`stress_gemm.py --bench 200`，每卡单次 launch 的中位数，4 卡）：

| 版本 | us |
|---|---|
| TDM gather（`MEGA_TDM_GATHER=1`，有 bug） | 1966-2008，约 1987 |
| #17 async，ks=0 一次发完 | 2095-2139，约 2112（+6.3%） |
| #18 分到 2 个 ks 子步 | 约 2166 |
| #18 分到 4 个 ks 子步 | 约 2380 |

#17 的实现要点（见 patch）：
- 每个 lane 负责 16 B 块，每轮覆盖 16 行，每个 k-tile 发 16 条 SADDR 形式的异步加载；
- 行偏移用 32 位，K 不变，只算一次，多占 16 个 VGPR，总共 507 个，没有 spill；
- 无效行（填充或 dead tile）读 A 的第 0 行，写到它自己的填充行位置；docstring 里说明过，填充行的输出下游不会读；
- 本轮发出的加载在 `split_fence` 里用 `s_wait_asynccnt 0` 等待。

## 4. 结论（2026-10-26）

**大概率是硬件问题**，出在 gfx1250 TDM 引擎的 gather 模式里，但还不能百分之百定论。

支持硬件问题的证据：
- 软件同步协议是对的。同一份代码、同样的同步，gather 分散的普通行时 16 万次 launch 0 错误（#11）。
- 错误由**别的 WG 的数据**决定（#12-#15）。两个 WG 在不同 WGP 上，LDS 互相隔离，软件上没有路径能让一个 WG 的数据模式改写另一个 WG 的 LDS，能把它们连起来的只有共享的 TDM 引擎和返回路径。
- 软件侧的改动只能降低概率（#1-#10）；只有把 A 的 TDM gather 换成 VMEM 加载才归零（#17），而 LDS 布局、同步方式、数据都不变。
- 错误形态（某个 16 行块没写、数据落在 +0x1100 的相邻槽）像 DMA 写到了错误的 LDS 地址。

还不能排除的：
- gather 描述符是手工拼的（`_gather_static_groups` / `_gather_dgroup0`），没走 `tdm_ops` 的标准构造，某个字段（保留位、cluster / multicast 相关位）设错也可能造成跨 WG 写入。
- 存在未公开的 TDM gather 使用约束或 errata，比如要求某个 wait、不允许越界或重复的索引。
- 和这批芯片的 stepping、固件或驱动配置有关。

决定：先保留 `1149c02b` 的最优性能，不合入规避方案。风险是 EP4 训练中偶发静默错误或 Inf；在问题解决前，正确性要求高的场景可以打上 `wip_async_a.patch`，代价是 step 慢约 2%。

## 5. 恢复修复时的计划

1. **确认是不是硬件问题**：
   - 把单卡最小复现 `stress_gemm.py --pad 70`（10 / 160000 对 0 / 160000）交给硬件或编译器团队，问有没有已知的 TDM gather errata 和推荐的规避方式；
   - 把手工拼的 gather 描述符和 `tdm_ops` 标准构造的结果逐位对比，排除字段设错。
2. **低成本缓解**（只降概率，不根治）：#2 的 `s_wait_xcnt` + 描述符 hold 加 #3 的 32 位索引，fwd 出错率合计降约 60 倍（约 1 / 500 降到 2 / 60000）。性能影响没单独测过，要先 A/B。
3. **如果要用 #17 根治，先把性能追回来**，每次只改一个变量：
   - 去掉每条加载前算 LDS 地址的 VALU（用指令的立即数偏移，或预先算好）；
   - 只让一部分 wave 发加载，减少和 MMA 抢发射槽；
   - 试 cache policy（NT 或 streaming）；
   - 调整 lane 和行的映射，比如一个 wave 连续读几行。
   - 分散到 ks 子步已经试过，更慢（#18）。
4. **验证**（修复版）：
   - `stress_dg` fwd 和 `--b_nn` 各至少 40000 次、0 次出错；
   - 纯 GEMM 加填充至少 160000 次、0 次出错；
   - 整步 `stress_ep` 20000 步；
   - pytest：EP1、EP2、EP4（`run_round.sh N full`）；
   - e2e：even、uniform、real 三种 routing。
5. **清理代码**：
   - 删掉 TDM gather 路径、实验开关（`MEGA_*`）和 hold 相关代码；
   - 更新 `gemm_nt_tile.py` 的 docstring；
   - launcher 加断言 `a_rows * lda * 2 < 2^31`（32 位行偏移的前提）。
6. **排查其他 TDM 使用者**：
   - grouped wgrad（variable_k，EP2 round 9 引入的 TDM gather）有没有同样的风险；
   - dead tile 发的 extent 为 0 的稠密 TDM（B 加载）是否也会成为触发源，必要时改成界内的正常加载。

## 6. 修复（2026-10-26 下午）：async A 加载的性能优化和合入

### 实现（已合入 `gemm_nt_tile.py`，默认开启）

- 每个 lane 负责一行里的一个 16 B 块（16 个 lane 读一行 256 B），一条指令覆盖 16 行，每个 k-tile 每个 lane 发 16 条 `global_load_async_to_lds_b128`；`split_fence` 和 prologue 里用 `s_wait_asynccnt 0` 等。
- **k 偏移折进 32 位行偏移**（每条加载一个 `v_add_nc_u32`），基址是循环不变的 SGPR。这样 zext 留在循环里，ISel 选出 SADDR 形式。#17 的 64 位偏移被 LICM 提到循环外，变成 VADDR64 形式：每条加载多一个 `v_add_nc_u64`，偏移多占 16 个 VGPR，还有一串 `s_wait_alu depctr` 停顿。
- **LDS 步长放进指令立即数偏移**：IOFFSET 同时加到 global 和 LDS 地址上，所以第 i 条的 LDS 步长 `i*16*272` 先从它的行偏移里减掉；基址减去 `A_BIAS = tile_m * LDS_ROW`，保证 32 位无符号偏移不为负。所有加载共用一个 LDS 地址寄存器。
- ~~**无效行（填充 / dead tile）读 tile 内位置 `pos` 对应的行**（越界时夹到 0），不再全部读第 0 行。所有 WG 同时读同一行会集中打到一个内存 channel（camping）。~~ 这会读到 dispatch 还没写完的行，造成回归，第 7 节已改掉。
- 不加 per-load 分支：分支会让加载串行化。
- B 恢复成 `1149c02b` 的写法（不加 `s_wait_xcnt` hold；#7 已否定"描述符被晚读"）。
- launcher 断言：`a_rows > 0`、`a_rows * lda * 2 + tile_m * LDS_ROW < 2^32`（`check_gather_rows`，standalone 和 fused 都调用）。
- `MEGA_TDM_GATHER=1`（trace 时读取）切回 TDM gather，用于复现；换开关时要用独立的 FlyDSL cache（`CACHE=`）。

### 逐项实验（纯 GEMM `stress_gemm.py --bench 200`，4 卡中位数）

同一时段 TDM gather 约 1983-2002 us（无填充）、约 1960 us（`--pad 128`）。

| # | 改动 | us（无填充） | us（`--pad 128`） | 结论 |
|---|---|---|---|---|
| 17 | 原 async（VADDR64，无效行读第 0 行） | 约 2109 | 约 2148 | 起点 |
| 19 | 用空 inline asm "洗"偏移，强制 SADDR | 约 3930 | - | 寄存器溢出到 scratch，放弃 |
| 20 | k 折进 32 位偏移（SADDR） | 约 2038 | - | **采用** |
| 21 | 8 / 4 个 lane 读一行（每 lane 少碰几行，多块用 IOFFSET） | 约 2310 / 2365 | - | 更慢：一行 256 B 连续最好 |
| 22 | 20 + LDS 步长放进 IOFFSET | 约 2039 | 约 2148 | 持平，指令更少，**采用** |
| 23 | 诊断：不加载 A（只有计算 + B） | 约 1740 | - | A 加载的下限开销 |
| 24 | 诊断：所有行读第 0 行 | 约 2610 | - | 同一地址热点很慢 |
| 25 | 22 + 无效行读不同的合法行（`pos`） | 约 2040 | 约 2038 | 填充下快 5%，**采用** |
| 26 | 按行 exec 分支跳过无效行 / wave 级 ballot 标量分支跳过 | 约 2150 / 2178 | 约 2135 / 2150 | 分支让加载串行（每条都要等上一条读完源寄存器），放弃 |
| 27 | B 的 TDM 先于 A 发出；再加 A 加载和 WMMA 交错（`sched_group_barrier`） | 约 2050 / 2144 | - | 更慢，放弃 |
| 28 | 用 i32 GEP 代替 zext+add | 约 2109 | - | InstCombine 显式 sext 后又被提出循环，回到 VADDR64，放弃 |

实验版代码（全部 `MEGA_A_*` 开关）存为 `tools/tdm_gather_bug/async_a_experiments.patch`（基于 `1149c02b`），只作记录。

合入版本，同一时段对比：

| 版本 | us（无填充） | us（`--pad 128`） |
|---|---|---|
| TDM gather（`MEGA_TDM_GATHER=1`，有 bug） | 约 2002 | 约 1965 |
| 合入的 async 版本 | 约 2034（+1.6%） | 约 2034（+3.5%） |

剩下的差距：TDM 不读填充行，async 仍要读（有填充时差距更大）；每条加载要一个 VALU，而 VGPR 已用到 502，只剩约 3 个临时寄存器轮转，`vm_vsrc` 等待没法完全消掉。

### 验证（合入版本）

| 测试 | 结果 | TDM gather 同条件 |
|---|---|---|
| 纯 GEMM `--pad 70`，4 卡各 40000 次 | **0 / 160000** | 10 / 160000 |
| fused fwd `stress_dg`（EP4，T=8192） | **0 / 40000** | 约 1 / 500 |
| fused b_nn `stress_dg --b_nn` | **0 / 40000** | 有错 |
| 整步 `stress_ep` 20000 步 | 未完成（因还机器中断） | 约 6 / 26000 |

`stress_dg` 结束时 rank 0 偶尔以 -11 退出，发生在打印结果之后的 teardown，上午的日志里也有，和本次改动无关。

### 换机器后继续

```bash
cd agent/workspace/mega_moe_flydsl_gfx1250_ep4_20261026/tools/tdm_gather_bug
CONTAINER= TAG=fix LIMIT=2400 ./stress.sh repo 20000                          # 1. 整步 stress_ep，约 7 分钟
CONTAINER= TAG=fix SCRIPT=stress_gemm.py ./stress.sh repo 1 --bench 200        #    纯 GEMM 计时（对照加 MEGA_TDM_GATHER=1 和独立 CACHE）
```

1. 整步 `stress_ep` 20000 步，0 错误；
2. pytest：EP1、EP2、EP4（`run_round.sh N full`）；
3. e2e：even、uniform、real 三种 routing，和 `1149c02b` 对比 step 时间（预期慢 1% 左右，以实测为准），更新 HTML 报告；
4. 第 5 节第 6 条：排查 grouped wgrad（variable_k）的 TDM gather 是否有同样风险。

## 7. 新机器上的验证和 staged-bias 回归的修复（2026-10-27）

新机器同样是 MI455X x4，容器用 `amdprimus/amdprimus:tas-20261019`，名字沿用 `xiaompen_mega_gfx1250_ep4`。开机时 `amdgpu` 在 kernel cmdline 里被列进黑名单，要先 `sudo modprobe amdgpu`。仓库没编译，压测和 pytest 用的 `_C` 先用符号链接指到镜像里预编译的 `.so`；Mega 路径不调 C++ op。e2e 要用 C++ rmsnorm，所以在 `/home/xiaompen/mega_scratch/turbo_build` 里另做了 in-place 编译。

这台机器的绝对时间比上一台慢约 7%（TDM 纯 GEMM 约 2150 us，上一台约 2002 us），下面的对比都是同一时段交替跑的。

### 新机器上复现和第 6 节合入版本的压测

| 测试 | async（第 6 节合入版本） | TDM gather（`MEGA_TDM_GATHER=1`） |
|---|---|---|
| 纯 GEMM `--pad 70` | 0 / 160000 | 3 / 80000，错误特征和之前一致 |
| 整步 `stress_ep` 20000 步 | 0（y、dx、dtw bitwise 一致；dW 有 1 ulp 量级的差，来自 pool 顺序） | 之前约 6 / 26000 |

### grouped wgrad（variable_k）的 TDM gather（第 5 节第 6 条）

新脚本 `stress_wgrad.py`：直接调 `grouped_gemm_bf16_variable_k` 做 dW1（`b_index` gather）和 dW2（`--dw2`，`a_index` gather），pool 按 256 行补齐、`masked_k` 给真实行数，每次和第一次的结果 bitwise 比较。

| 测试 | 结果 |
|---|---|
| dW1，4 卡各 40000 次 | 0 / 160000 |
| dW2，4 卡各 40000 次 | 0 / 160000 |
| dW1，大量重复行（`--tokens` 小，热点行），4 卡各 100000 次 | 0 / 400000 |

wgrad 的 gather 没有越界行（`masked_k` 截掉了填充），每个 expert 尾部 k-tile 重复读最后一行。压测没出错，风险低，但不能证明没有（grouped kernel 头部注释记录的已知缺陷约 1 / 130000 次调用）。暂不改。

### 回归：GPT-OSS staged-bias 测试失败

第 6 节合入版本跑 pytest 时，`test_staged_bias_gpt_oss`（EP4）、EP2 的 1 个和 EP1 的 1 个（`test_fused_mega_moe_staged_bias_ep1[1024-2880-2880-32-4]`）失败，其余通过。三个失败都是 GPT-OSS shape（H=2880，`tile_k=64`）。`1149c02b` 和 `MEGA_TDM_GATHER=1` 都通过。

排查（`probe_staged.py` 打印每轮 SNR，`probe_l1.py` 逐个 pool 行对照 `token_buffer[idx] @ w1[e].T`）：
- 只在第 2 轮及以后出错，第 1 轮总是对的；
- 坏行的源行号全部 < 256；
- 只有部分 WG 错（N-block 12-14、tile 9-15）；
- 部分 k-tile 的数据能用同一行上一轮的数据解释。

原因：第 6 节的"无效行读 `pos` 行"让 dead tile（不等任何 expert 的 flag）和填充行去读 token buffer 的第 0-255 行，这些行可能正在被 dispatch 写入。读到的旧数据留在 WGP 的 vector cache 里，之后同一个 WGP 上负责这些行的 live tile 过了 flag 再读，命中了旧数据。TDM gather 从不读这些行（越界行不访存，dead tile 的 extent 为 0），所以没有这个问题。

修复（`gemm_nt_tile.py`）：只读确定已经写完的行。
- dead tile 完全不发 A 的加载：只在 prologue 里加一个 wave 一致的 `live` 分支，主循环不变（dead tile 本来就只跑 prologue）。
- 填充行读本 tile 自己的行（这个 expert 的 token，在它的 flag 之前已写完）：先取这个 lane 第 0 条加载的行，那一行也是填充时再取 tile 的第一行。如果所有填充行都读 tile 的第一行，最多 128 行的加载会打到同一个地址，`--pad 128` 下慢 6%。

纯 GEMM 计时（`stress_gemm.py --bench 200`，4 卡平均，两轮交替）：

| 版本 | us（无填充） | us（`--pad 128`） |
|---|---|---|
| TDM gather（有 bug） | 约 2150 | 约 2107 |
| 填充行全读 tile 第一行 | 约 2169（+0.9%） | 约 2243（+6.4%） |
| **采用：填充行读本 lane 第 0 条加载的行** | 约 2167（+0.8%） | 约 2163（+2.7%） |

（第 6 节合入版本在这台机器上没有和 TDM 同时段计时；上一台机器上它比 TDM 慢 1.6% / 3.5%。）

### 修复后的验证

| 测试 | 结果 |
|---|---|
| `probe_l1.py --reps 6`（staged-bias GPT-OSS 的 L1 fwd，逐行对照） | 每轮 0 坏行 |
| `probe_staged.py --reps 4` | y 50.8、dx 49.4、d topk_w 55.6、dW1 50.6、dW2 52.6 dB，每轮一致，和 TDM 路径相同 |
| pytest EP4 / EP2 / EP1 | EP4 4 / 4、EP2 5 / 5、EP1 19 / 19，全过 |
| 纯 GEMM `--pad 70`，4 卡各 40000 次 | 0 / 160000 |
| fused fwd `stress_dg`，20000 次 | 0 / 20000 |
| fused b_nn `stress_dg --b_nn`，20000 次 | 0 / 20000 |
| fused fwd 加 churn `stress_dg --churn`，20000 次 | 0 / 20000 |
| 整步 `stress_ep` 20000 步 | 0 / 20000（y、dx、dtw bitwise 一致；dW 1 ulp 量级，同上） |

（"填充行全读 tile 第一行"的版本也跑完了同一套：pytest 全过，纯 GEMM 0 / 160000，fused fwd / b_nn / churn 各 0 / 20000，`stress_ep` 0 / 20000。）

### step 级 A/B 和 e2e（修复后，对比 `1149c02b`）

kernel 级：`tools/ab.sh fix3 2 <3 个 shape>`（`NEW_CACHE` 用新目录），campaign bench `quick_test_bench.py`，两边交替各跑 2 轮，日志在 `/home/xiaompen/mega_scratch/ep4c/logs/ab_fix3_*`。`ab.sh` 加了 `TURBO_SRC`，保证每边导入自己的源码树（日志第一行 `primus_turbo from ...` 可确认）。

| shape | 轮 | base fwd ms | 修复版 fwd ms | base fwd+bwd ms | 修复版 fwd+bwd ms | base step TF | 修复版 step TF |
|---|---|---|---|---|---|---|---|
| dsv3_t8192_random | r1 | 4.072 | 4.237（+4.1%） | 11.775 | 12.018（+2.1%） | 1470.6 | 1440.9（−2.0%） |
| dsv3_t8192_random | r2 | 3.928 | 4.028（+2.5%） | 11.812 | 11.956（+1.2%） | 1466.0 | 1448.4（−1.2%） |
| dsv3_t8192_skew1 | r1 | 9.470 | 9.706（+2.5%） | 27.635 | 28.442（+2.9%） | 626.6 | 608.9（−2.8%） |
| dsv3_t8192_skew1 | r2 | 9.512 | 9.737（+2.4%） | 27.695 | 28.025（+1.2%） | 625.3 | 617.9（−1.2%） |
| dsv3_t4096_random | r1 | 2.235 | 2.288（+2.4%） | 6.750 | 6.900（+2.2%） | 1282.7 | 1255.0（−2.2%） |
| dsv3_t4096_random | r2 | 2.198 | 2.263（+3.0%） | 6.738 | 6.867（+1.9%） | 1285.0 | 1260.9（−1.9%） |

step TF 平均慢约 1.9%，和第 3 节 #17 的 A/B 一样；第 6 节把纯 GEMM 的差距从 6% 降到 1% 以内，但没有传到 fused kernel 上。原因还没查（候选：fused kernel 里 dispatch 和 GEMM 共用 VMEM 发射，async 加载要占 VMEM，TDM 不占）。

e2e：DSV3 proxy EP4（`e2e_suite.sh fix3`，容器内 root 运行），`tdm` = `1149c02b` 源码树，`async` = 本仓库，每个 routing 两边交替，各用独立的 FlyDSL cache。两份源码树的 `_C` / `libprimus_turbo_kernels.so` 已换成 `turbo_build` 里编译的版本（rmsnorm 是 C++ op）。日志 `/home/xiaompen/mega_scratch/e2e_fix3_{tdm,async}_{even,uniform,real}.log`。

| routing | 步数 | `1149c02b` ms/iter | 修复版 ms/iter | 变化 | TFLOP/s/GPU | 最后一步 loss | 逐步 loss 最大相对偏差 |
|---|---|---|---|---|---|---|---|
| even | 10 | 1050.5 | 1061.2 | +1.0% | 1346.4 / 1332.8 | 5.563013 / 5.562970 | 9.4e-6 |
| uniform | 10 | 1093.0 | 1103.8 | +1.0% | 1294.1 / 1281.5 | 8.791574 / 8.791613 | 1.5e-5 |
| 真实路由 | 50 | 1098.6 | 1107.8 | +0.8% | 1287.5 / 1276.8 | 0.1011702 / 0.1012327 | 1.8e-3 |

ms/iter 和 TFLOP/s 是 Primus 日志最后一步打印的平均值（从第 3 步起算）。所有项 0 个 nan iteration，loss 没有 nan / inf，grad norm 一致。真实路由的偏差正负交替、没有趋势，loss 降到 0.1 左右后相对值才变大，10-26 基线和 Mega 之间也有 1.0e-3。峰值显存两边相同（154.5-155.0 GB）。

bitwise 压测（`stress_ep`、`stress_dg`，都是 DSv3 shape，H=7168、`tile_k=128`）在第 6 节合入版本上都是 0 错误，没抓到这个回归；只有 GPT-OSS shape（H=2880、`tile_k=64`）的测试暴露出来。DSv3 shape 为什么不出错还没确认，所以压测之外 pytest 也必须跑。

### 新脚本（`tools/tdm_gather_bug/`）

- `run_tests.sh TAG`：`run_round.sh` 的 pytest（EP4 每个测试一个进程，EP2 用 GPU 0,1，EP1 用 GPU 0），每次用新的 FlyDSL cache，日志在 `logs/test_TAG_*.log`。
- `stress_wgrad.py`：见上。
- `probe_staged.py` / `probe_l1.py`：见上。`probe_tilek.py`：单独的 gather GEMM 按 (K, tile_k) 和 torch 对拍。
- `e2e_suite.sh TAG [ARMS] [ROUTINGS]`：在容器里以 root 跑 Primus 的 DeepSeek-V3 proxy（`--use_turbo_mega_moe True`），默认对比 `tdm=1149c02b` 和 `async=本仓库`，routing 为 even、uniform（10 步）和 real（50 步）。
