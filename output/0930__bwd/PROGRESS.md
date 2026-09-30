# bwd 实验进度（A0，blocked 尺子，prod b4 s8192 hq32 hkv8，randn；同进程 r29 + r29_aa + ASM）

bwd FLOP 5.498229e12；判定阈值 0.5%（同进程 A/A 0.01–0.07%）。每个 arm：compile-only → fast 校验（AMD_SERIALIZE_KERNEL=3）→ prod 校验（≥50 dB，dk/dv 逐位，dq run-to-run）→ prod A/B。

| # | arm | 时间（UTC） | 结果 | prod ms（arm / 同进程 r29 / ASM） | 相对 r29 | 占 ASM | 结论 |
|---|---|---|---|---|---|---|---|
| base | r29 | 08:05 | — | 6.579 / — / 5.502 | — | 83.6% | 基线，A/A 0.01% |
| a01 | dqg_u2off（`DQ_U2=False`） | 08:40 | **接受**（暂定基底 c1） | 6.504 / 6.559 / 5.476 | **−0.85%** | 84.2% | A0 新固件上 k_dq 展开 ×2 是负收益，关掉 |
| a02 | dkdv_ku2（`KV_U2=True`） | 08:55 | 否 | 7.099 / 6.563 / 5.489 | +8.17% | 77.3% | 与 B0 +7.3% 一致；ATT：v_mov 轮转消失，但 `s_wait_loadcnt` stall 从 0.50M 涨到 1.73M（预取提前量缩短，全局加载延迟暴露）→ 瓶颈是全局加载的提前量，指向 TDM/更深预取 |
| c1 复核 | c1 反序进程（asm, c1, r29） | 09:00 | 确认 | 6.510 / 6.562 / 5.483 | −0.80% | 84.2% | 两个顺序都 −0.8% |
| a03 | 消融 A1–A5（输出错误，只看上限） | 09:25 | 信息 | 见下 | | | **A2 去掉 k_dkdv 的 Q/dO 全局加载：−18.5%（5.370 ms，快于 ASM 5.495）**；A3 nosm −2.0%；A4 k_dqg 去 K/V 全局加载 −1.7%；A1 去 P/dS LDS 往返 −1.4%；A5 每轮 +16 WMMA +4.0% |

**消融结论**：k_dkdv 的 Q/dO 全局加载路径是主杠杆（上限超过 ASM）；flip（上限 1.4%）降级；dQ 融合每轮多 16 条 WMMA 的纯矩阵代价约 +4%（不含原子操作）。
| a04 | dkdv_lse_late / divfree / trorder / trorder_b（逐位等价） | 09:50 | **trorder 接受**；divfree 留作叠加 | r29 6.588；lse_late 6.856；divfree 6.572；trorder **6.530**；trorder_b 6.616；ASM 5.503 | +4.06% / −0.25% / **−0.89%** / +0.41% | trorder 84.3% | lse_late：loadcnt 等待被挪到 phase K 入口，更深的 0xc 等待反而更慢；trorder：最后 DS 段分 4 区交错，dscnt 等待变浅 |
| a05 | dqg_tailpf / mem_soffset_q / dqg_dkdv_streams / dkdv_flip | 10:20 | **tailpf、streams、soffset 接受**（待叠加） | r29 6.602；tailpf **6.382**；soffset 6.559；streams 6.543；flip 6.572；ASM 5.503 | **−3.33%** / −0.65% / −0.89% / −0.46% | tailpf 86.2% | tailpf：U2 保留、body 2 的 ii+2 预取按 kt 半段提前发出；streams：k_dqg 放到 side stream 与 k_dkdv 并发；flip 上限本来就小（A1 −1.4%） |
| t01 | dkdv_tdm 首次上卡（toy / gqa4_small / unequal_seqlen_2，各自独立进程，SERIAL） | 10:40 | 通过 | — | — | — | 三个形状 dq/dk/dv 与 c1 **逐位一致**，dmesg 干净；576 VGPR（c1 729），热循环 0 条 buffer_load_b128 / 0 条 v_mov_b64 |
| a06 | dkdv_tdm（TDM 3 级 LDS ring 装 Q/dO，基于 c1） | 11:00 | 否 | tdm 6.628 / r29 6.577 / c1 6.517 / ASM 5.481 | 相对 c1 +1.7% | 82.7% | 逐位等价；ATT：loadcnt 等待消失，但 32 条 ring 回读 ds_load_b128 被逐条 dscnt 等待卡住；PMC k_dkdv cycle +13.5%（时钟升到 1939 MHz 抵掉一部分） |
| a07 | TDM_DEPTH=2 | 11:15 | 否（持平） | tdm2 6.503 / tdm3 6.601 / c1 6.493 / c1_aa 6.498 | 相对 c1 +0.15% | — | 深 ring 更慢；TDM 本身只是去掉了加载等待，没有带来净收益 |
| a08 | 消融 abl_l2（tdm2 源地址固定在 tile 0，永远命中 L2） | 11:25 | 信息 | abl_l2 6.508 / r29 6.574 | 相对 r29 −1.0%（≈ tdm2） | — | **L2 带宽不是限制**；A2 的 −18.5% 来自"B 操作数已在寄存器里"→ 下一步：tdm v3 在本轮就把下一轮的 B 操作数从 ring 读进寄存器 |
| **a09** | **s1 = tailpf + trorder + streams**（patch 合并，逐位等价） | 11:45 | **接受（新冠军候选）** | **s1 6.329** / r29 6.575 / tailpf 6.369 / ASM 5.500 | **−3.74%** | **86.9%** | ATT：k_dqg 20.1 cyc/WMMA（WMMA 占 57%，基本健康）；k_dkdv 24.7 cyc/WMMA，VALU 占 37%（~4.6 VALU/WMMA，其中 Q/dO v_mov 轮转 + v_nop 约 130 条/轮）→ tdm v3 同时去掉这部分 |
