# GPU0 健康检查 — 2026-09-28 01:55–02:03 UTC，fa-g0，flock /tmp/b0-gpu0.lock

**结论：GPU0 健康。** 用与昨天完全相同的 arm 组合（r4 + baseline + beat）复测，r4 与 ASM 都落在昨天的区间内；dmesg 对 0001:04:00.0 无新增。

| 进程 | harness | arm 组合 | r4 ms | baseline ms | beat(ASM) ms | r4/beat | sclk 起/止 |
|---|---|---|---|---|---|---|---|
| 昨天 fwd_S1/S2/P1/P2/Q1g0/A1/B1（GPU0，7 次） | 0925 job | r4,baseline,beat | 1.482–1.503 | 1.750–1.789 | 1.167–1.188 | 1.26–1.27 | ~1000/1260 |
| A03 | 0925 job | r4,baseline,beat | **1.5025** | 1.7906 | **1.1926** | 1.260 | 994/1259 |
| A04 | b0 job | r4,baseline,beat | **1.5075** | 1.7741 | **1.1923** | 1.264 | 991/1257 |
| A01 | b0 job | **r4,beat（任务指定的两 arm）** | 1.6562 | — | 1.2256 | 1.351 | 994/1157 |
| A02 | b0 job | r4,beat | 1.6522 | — | 1.2270 | 1.347 | 993/1155 |

- 两个 job 的 harness 在 r4/beat 上读数一致（A03 vs A04），r4 代码两 job 相同（flydsl_fwd md5 a1e806c9）。
- **注意（标尺问题，不是卡的问题）**：只少了一个 baseline arm，r4 就从 1.50 变成 1.65 ms（+10%），ASM 从 1.19 变成 1.23（+3%），
  r4/beat 从 1.26 变成 1.35；同时 sclk_end 从 ~1258 降到 ~1156 MHz。同一份代码的读数取决于同进程里还有哪些 arm。
  这正是 Part B 的问题，详见 REPORT.md。
- 运行期间 GPU0 的 KFD 上只有本进程（A02 期间每 5 s 采样，`runs0/A02.kfd`）。
- 证据：`runs0/A0{1,2,3,4}.{log,json,dmesg}`；昨天的对照 `output/0927__b0/probe/fwd_*.log`。
