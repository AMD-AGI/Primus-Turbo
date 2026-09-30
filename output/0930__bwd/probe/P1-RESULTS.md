# P1：新固件（VBIOS 700E / SMU 125.12 / dkms 7.1.0-2411946）下 deep 轮分析工具能力矩阵（2026-09-30，A0）

每项一个独立卡进程（`tools/run.sh`，锁 + KFD 空检查 + dmesg 分类），全部 rc=0，dmesg 无 GPU 错误行。

| 工具 | 旧状态（skill env-and-pitfalls §2c） | 09-30 结果 | 证据 |
|---|---|---|---|
| `rocprofv3 -L` | — | 110 项：51 个硬件计数器（与旧表相同，无 SQ_WAIT_*/LDS 计数器）；**PC sampling 报告一个配置 `SAMPLE_INTERVAL_SCLK_CYCLES` 32–65504** | `runs/p1a_listavail.log` |
| `--kernel-trace` | 0 行 | **仍然 0 行**（同进程 `--memory-copy-trace` 正常出行）→ rocprofv3 1.3.2 的问题，不是固件；PMC csv 自带 kernel 名 + 时间戳，可替代 | `runs/p1b*`, `probe/p1b2/` |
| `--pmc` | 可用 | 可用（roofline 已验证） | `0930__roofline/runs/pmc` |
| **ATT（thread trace）** | 对 FlyDSL JIT 什么也抓不到 | **可用**：FlyDSL k_dkdv / k_dqg 和 ASM `.co` 都得到逐指令 Hitcount/Latency/Stall/Idle（`stats_ui_output_*.csv`）+ wave 时间线 json；需要 `--att-library-path /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib`、`--att-target-cu 1`、`--kernel-include-regex`（注意 shell 引号，`k_d[kq]` 可用） | `probe/p1d/`, `probe/p2_r29/`, `probe/p2_asm/` |
| rocprof-compute | 未用 | 容器内未安装（`which` 为空）；按规则不 pip install → 不可用 | — |
| PC sampling | 3/3 挂 MES（旧固件） | **不可用（工具层拒绝，不涉及卡）**：`-L` 列出 `SAMPLE_INTERVAL_SCLK_CYCLES` 32–65504，但 stochastic/cycles（间隔 65504 和 16384）、host_trap/cycles（16384）三次都在启动时报 `Given PC sampling configuration is not supported on any of the agents`，没有开始采样；dmesg 干净。按 D2 不再继续试 | `runs/p1f*_pcsampling.log` |

## prod ATT 首读（CU 1 上的 wave，`tools/attsum.py`）

| kernel | latency/WMMA | 主要 stall |
|---|--:|---|
| fly k_dkdv（r29） | 30.4 | `s_wait_dscnt 0x6` 9.5%、`s_wait_loadcnt 0x23` 6.8%；VALU 占 latency 30.5%（每 WMMA ~4.9 条 VALU） |
| fly k_dqg（r29） | 21.7 | `s_wait_loadcnt 0x20` 4.8%、`s_wait_xcnt 0x2` 2.6% |
| ASM 主 kernel | 20.7 | `s_wait_tensorcnt`（TDM）~19%、4-wave `s_barrier_wait` ~5%；dq 用 `buffer_atomic_add_f32` |
