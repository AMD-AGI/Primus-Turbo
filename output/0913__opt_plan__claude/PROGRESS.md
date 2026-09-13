# 持续优化任务队列 — 唤醒后读这个文件，不要重新推导

**规则：每次被唤醒，从 TODO 顶部取第一条没被认领的任务做掉，然后更新本文件。**
**GPU 忙不是停下的理由 —— GPU 跑测量时，CPU 侧永远有代码/文档/下一批可以推进。**
**永远不要以「等待」结束回合。等待前先确认：有没有一件不依赖那个等待的事可以做？**

最后更新：2026-09-13 12:58 UTC

---

## 当前状态

| 项目 | 值 |
|---|---|
| 生产形状 | b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal |
| 起点 | 59.642 ms / 129.0 TFLOP/s |
| **当前树内** | **24.349 ms / 316.1 TFLOP/s**（2.45×） |
| 对照锚点 torch flex | 31.337 ms / 245.6 |
| 参考上界（纯 aiter） | 21.672 ms / 355.1 |
| gfx1250 测试数 | 44 passed |
| 已提交 | f8c45dee, df20f1d6, 49257416, 5eba2cf4, 1cb2e183, 03a76f61 |

冠军配置：融合反向 `BLOCK_M1=32, BLOCK_N1=256, BLOCK_M2=256, BLOCK_N2=32,
BLK_SLICE_FACTOR=1`，前向 `num_stages=2`。已接进 dispatcher，按 `seqlen >= 2048` 门控。

---

## TODO（按优先级，从上往下做）

- [x] **T1. 提高扫描吞吐。** 加了 `--shapes a,b,c`，同一进程内多形状测量。
      实测 3 个形状 23 s → 18 s（**1.28×，不是预期的数倍**）——说明启动开销没有想象中主导，
      测量本身占大头。收益随每进程形状数增长。`--tune`/`--fused-tune` 仍需独立进程
      （导入期读取）。
- [>] **T2.（进行中） 用 wq3 的密集数据校准门控阈值。** 现在 `_MIN_SEQLEN_FOR_FUSED = 2048` 是从
      s=1024/4096 两点外推的。wq3 有 s=1024/2048/4096/8192/16384 × N1∈{32,64,128,256}，
      能定出真正的交叉点。可能需要按 seqlen 分档选 N1，而不是单一阈值。
- [ ] **T3. 前向那 0.9 ms。** turbo 4.15 vs aiter 3.26，两个都是 Triton。配置已扫遍
      （num_stages 2/3/4、num_warps 1/2/4/8、PRE_LOAD_V）都不动 → **是结构差异**。
      下一步：profiler 对比两个前向的内核构成，判断要不要也 vendor 前向。
- [ ] **T4. vendored 与纯 aiter 的 1.8 ms 差**（反向 20.2 vs 18.4）。怀疑是 packed LSE
      gather：turbo 前向写 `[B,Hq,2*Sq]` 交错，融合内核要稠密 `[B,Hq,Sq]`。
      验证方法：profiler 看 gather 的 elementwise 耗时；若坐实，考虑让前向直接写稠密 LSE。
- [ ] **T5. 非因果 / varlen / sink 覆盖。** 融合内核支持 sink，但 dsink 没接。
      varlen 在 gfx1250 上仍无 Triton 路径。
- [ ] **T6. HipKittens udna1 路线。** `3rdparty/hipkittens/include/udna1/` 已是完整
      gfx1250 移植（72 头文件，WMMA 16×16×32 / TDM 描述符 / split waitcnt 都接好），
      只差 attention 内核。仅在 Triton 撞到天花板后才启动。
- [ ] **T7. 端到端。** 被缺失的 gfx1250 BLAS 阻塞（稠密 GEMM 只有 27.4 TFLOP/s）。
      需要平台侧给一个 BLAS 能用的镜像。非代码问题。

## DONE

- [x] dispatch 三个 bug + tuning 开关；36 个测试跑起来（`f8c45dee`）
- [x] 配置调优 1.66×（前向 num_stages 1→2，反向 num_warps 4→2）
- [x] LSE ABI 解耦，逐位验证中性（`5eba2cf4`）
- [x] 证伪「非对称 tile」假设；定位真因是内核融合
- [x] vendor 融合反向，316.1 TFLOP/s（`1cb2e183`）
- [x] 接 dispatcher + shape gate + 8 个测试，44 passed（`03a76f61`）
- [x] 确定性：出货路径 400 次，零失配
- [x] 进度文档 `progress.html`

---

## 不要重做（已被实测推翻）

- turbo 的 dkdv 非对称 tile —— 64×64 本来最优，aiter 的 32×128 在这里慢 59%
- `sequence_parallel=False` —— 只算 1/128 的梯度，纯计时会报成巨大胜利
- XCD remap —— grid 本来就 8 对齐，改了会更慢
- 跳过非对角块的因果掩码 —— 慢 9%（标量分支破坏流水）
- 收紧 dkdv 的 `lo` / 折叠 `log_p_scale` —— 性能中性
- FlyDSL —— 非 gfx950 上 build 时 raise，gfx1250 构建不装它，且 G=4 被拒

## 必须保留的纪律

1. **四张量 SQNR 门**（out/dq/dk/dv 分别）。实测出现过 dk/dv 被毁而 out/dq 完好。
2. **在 launch 处验证配置**，不在源头。aiter 把 tile 编进 kernel name。
   扫描结果「平坦」= 覆盖没生效。
3. **配对约束** `BLOCK_N1==BLOCK_M2`、`BLOCK_M1==BLOCK_N2`，破坏会「更快但 dq 只算一半」。
4. **健康检查用 dmesg**，不用 rocm-smi（卡挂时它自己会 hang）。
5. **禁止 PC sampling**（三次尝试三次 GPU fault，一次需要重启）。
6. 端点获胜 = 范围不够，要扩。
