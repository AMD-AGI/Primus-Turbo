# bwd 实验进度（A0，blocked 尺子，prod b4 s8192 hq32 hkv8，randn；同进程 r29 + r29_aa + ASM）

bwd FLOP 5.498229e12；判定阈值 0.5%（同进程 A/A 0.01–0.07%）。每个 arm：compile-only → fast 校验（AMD_SERIALIZE_KERNEL=3）→ prod 校验（≥50 dB，dk/dv 逐位，dq run-to-run）→ prod A/B。

| # | arm | 时间（UTC） | 结果 | prod ms（arm / 同进程 r29 / ASM） | 相对 r29 | 占 ASM | 结论 |
|---|---|---|---|---|---|---|---|
| base | r29 | 08:05 | — | 6.579 / — / 5.502 | — | 83.6% | 基线，A/A 0.01% |
| a01 | dqg_u2off（`DQ_U2=False`） | 08:40 | **接受**（暂定基底 c1） | 6.504 / 6.559 / 5.476 | **−0.85%** | 84.2% | A0 新固件上 k_dq 展开 ×2 是负收益，关掉 |
| a02 | dkdv_ku2（`KV_U2=True`） | 08:55 | 否 | 7.099 / 6.563 / 5.489 | +8.17% | 77.3% | 与 B0 +7.3% 一致；ATT：v_mov 轮转消失，但 `s_wait_loadcnt` stall 从 0.50M 涨到 1.73M（预取提前量缩短，全局加载延迟暴露）→ 瓶颈是全局加载的提前量，指向 TDM/更深预取 |
