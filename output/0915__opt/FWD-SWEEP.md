# Triton 前向配置扫描：3.7% 是真的，但报告把名次排错了

日期 2026-09-16 · 形状 `llama31-8b-s4096` · 反向固定为 ASM（`PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1`）

## 为什么扫前向

反向换成 ASM 后降到 2.740 ms，**前向占到算子总时间的 34%**。而且实测发现
**ASM 前向在这个形状上根本没生效**：`--asm-fwd auto` 与 `--asm-fwd off` 分别是
1.426 / 1.416 ms，关掉反而略快 —— 跑的一直是 Triton 前向。它的配置空间和两 kernel 反向
一样从没扫过。

## 扫描结果：73 点，72 ok，0 个 SQNR 不过，1 个把卡搞挂

## 一个必须先纠正的排序错误

`sweep_attention.py --report` **按 total 排序**，但本轮只扫了前向旋钮。于是表里
**bwd 一列在 2.549–2.848 ms 之间摆动**——只扫前向时它本该是常数，这 10% 是噪声。
按 total 排，等于让反向噪声决定前向名次：报告点名的第一名（`num_warps=4,wpe=0,ns=2,PLV=0`，
fwd 1.396）**并不是前向最快的那个**（`num_warps=2,wpe=2,ns=1,PLV=1`，fwd 1.379）。

**只扫了某一半的旋钮，就只能按那一半的时间排名。**

## 按 `fwd_ms` 重排后复现（交替 4 轮）

| 配置 | fwd ms | sd | vs 默认 |
|---|--:|--:|--:|
| `num_warps=2, waves_per_eu=2, num_stages=1, PRE_LOAD_V=1` | **1.371** | 0.36% | **−3.73%** |
| `num_warps=4, waves_per_eu=0, num_stages=2, PRE_LOAD_V=0` | 1.388 | 0.51% | −2.58% |
| 出厂默认 | 1.424 | 0.32% | — |

分离度 **15.8× sem**，效应真实。曲面同样非单调：最慢的 `num_warps=1,wpe=2,ns=2,PLV=1`
是 8.437 ms，是最快的 **6.1 倍**。

## 但要如实说清楚它有多小

省下 0.053 ms。算子总时间 4.16 → 4.11 ms（**1.3%**），attention 占 e2e 单步约 11.5%，
所以折到端到端约 **0.15%**。

**不改出厂默认。** 只在一个形状上扫过，而默认值要服务所有形状；记为该形状的推荐配置。
这与 0915 的教训一致：一个优化的价值取决于它在当前瓶颈结构里的位置，而前向现在不是瓶颈。

## 代价：一个候选把卡搞挂了

`fwd:num_warps=8, waves_per_eu=0, num_stages=2, PRE_LOAD_V=0` 运行期间：
`MES(7,0) failed to respond to msg=REMOVE_QUEUE` → `wait for reset ack` →
内核线程卡死在 `svm_migrate_to_ram` / `svm_range_restore_work`，需要人工 AC-cycle。

**本轮新加的每候选超时没能拦住它**，而且拦不住是合理的：超时防的是"候选自己挂死"
（子进程不返回），这次是**驱动在候选执行期间整体倒下**，进程已经消失、KFD 里留了个条目。
两者从外面看很像，但可处置性完全不同。该候选已作为 `returncode: "card-wedge"` 写入 ledger
让续跑跳过，**不重试**——故障值得重试，AC-cycle 不值得。

## 顺带发现的两个工具问题

1. 报告对布尔轴也会说 *"PRE_LOAD_V=0 is at the low end of the swept range, extend it"*。
   无害，但"赢家在区间边缘 = 区间太小"这条提示在别处是**有承重作用**的
   （上一次 campaign 正是在这里留下 +18%），不该在只有两个取值的轴上狼来了。
2. ledger 与日志由容器内的 root 写出，宿主机改不动，需要 `chown` 后才能编辑。

## 当前算子状态（proxy 形状 `llama31-8b-s4096`）

| | ms | 说明 |
|---|--:|---|
| 前向 | 1.371 | Triton，本轮调优；ASM 前向未生效 |
| 反向 | 2.740 | ASM（aiter 预编译 `.co`） |
| 合计 | **4.11** | 出厂 Triton 融合路径为 7.07 |

前向与反向都已接近各自路径能给的上限，**attention 这条线的配置面基本关闭**。
下一个值得投的是 GEMM（占 e2e 单步 46.1%）。
