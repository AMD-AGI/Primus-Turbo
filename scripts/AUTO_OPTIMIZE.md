# auto_optimize.py 操作手册（dev/kyle_mxfp8_gg）

每轮**新开** cursor-agent（不 resume），用 `git history + skill + prompt
history` 三件套做跨轮延续。详见 `scripts/auto_optimize.py`。

驱动指标（loop metric，每轮跑一次）= `scripts/_metric_mxfp8.py`，10-20 s
跑完一组 DeepSeek-V3 + gpt_oss_20B forward+backward TFLOPS + 一段确定性
stress，打印 1 行 score 到 stdout，越大越好。

深度验收（deep check，每 5 轮跑一次）= `pytest tests/pytorch/ops/
test_grouped_gemm_fp8.py::test_grouped_gemm_fp8_mx_blockwise`。失败时本
轮 `improved=False`、`best_metric` 不更新（即便 cheap metric 上升），用
来挡住"优化骗过 metric 但破坏 SNR 契约"。

最终验收（DoD，人工偶尔抽查）= benchmark 16/16 PASS + 长程 stress
(`N_ITERS=1000`) BAD 不显著上升。

---

## GPU 池

本节点和别的租户共用，**只允许使用 GPU 4-7**。脚本默认 `--gpu-pool=4,5,6,7`，
会把 `MXFP8_GPU_POOL` 传给所有子进程：

- **metric**: `_metric_mxfp8.py` 读 `MXFP8_GPU_POOL` 做 auto-pick（取池里第一张
  KFD VRAM <100 MiB 的卡），不显式设 `HIP_VISIBLE_DEVICES`，避免 cuda
  默认拿到忙卡。
- **deep-check (pytest)**: 直接设 `HIP_VISIBLE_DEVICES=$gpu_pool`，把 4 张卡
  喂进去，pytest 单进程自然拿 cuda:0=池里第一张。
- **cursor-agent**: 只传 `MXFP8_GPU_POOL`，由 agent 在每条命令里自己
  pick，prompt 里给了 one-liner。

需要换池就 `--gpu-pool=2,3` / `MXFP8_GPU_POOL=2,3 ...`。

---

## 启动（后台、SSH 断开也不掉）

```bash
cd /workspace/code/Primus-Turbo
mkdir -p auto_optimize_logs
nohup python3 -u scripts/auto_optimize.py \
    --rounds 20 --patience 5 \
    > auto_optimize_logs/run_$(date +%Y%m%d_%H%M%S).console.log 2>&1 &
echo $! > auto_optimize_logs/run.pid
disown
```

- `-u` 关 Python 缓冲
- `nohup + disown` 让 SSH 断开后进程继续跑
- `run.pid` 记主进程 PID 方便后续 stop

启动后第一次会跑 baseline metric（10-20 s）。看到
`[metric_mxfp8] auto-picked HIP_VISIBLE_DEVICES=...` 然后 `[metric] = <int>`
就是正常起步了。

---

## 看 log

每次启动会创建 `auto_optimize_logs/<timestamp>/`，结构：

```
auto_optimize_logs/
└── 20260427_092700/
    ├── summary.json           # 整体 trajectory，每轮 rewrite
    ├── round_001/
    │   ├── prompt.md          # 那一轮喂给 cursor-agent 的完整 prompt
    │   └── cursor.log         # cursor-agent 的 stdout/stderr
    ├── round_002/
    │   ...
└── run_*.console.log          # 主进程控制台输出（baseline / banner / EARLY-STOP）
└── run.pid
```

### 实时跟最新一轮 agent 输出

```bash
LATEST=$(ls -td /workspace/code/Primus-Turbo/auto_optimize_logs/*/ | head -1)
LAST_ROUND=$(ls "$LATEST" | grep -E '^round_[0-9]+$' | sort | tail -1)
tail -f "$LATEST/$LAST_ROUND/cursor.log"
```

### 看脚本本身的进度（baseline / 每轮 banner / EARLY-STOP）

```bash
tail -f /workspace/code/Primus-Turbo/auto_optimize_logs/run_*.console.log
```

### 看 trajectory 总览（metric / best / improved 流水）

```bash
jq '.rounds[] | {i:.index, metric, best:.best_so_far, improved, dur:.duration_s}' \
   /workspace/code/Primus-Turbo/auto_optimize_logs/*/summary.json | tail -40
```

### 进程是否还活着

```bash
ps -fp $(cat /workspace/code/Primus-Turbo/auto_optimize_logs/run.pid) 2>/dev/null \
  || echo "stopped"
```

---

## 停止

干净退出（finally 会还原 `~/.cursor/cli-config.json` 的 `maxMode`）：

```bash
kill $(cat /workspace/code/Primus-Turbo/auto_optimize_logs/run.pid)
```

实在不退再用 `kill -9`，但那样 `maxMode=true` 会留下来，需要手动改回。

---

## 手动跑一次 metric（不进 loop）

```bash
# 短日志
HIP_VISIBLE_DEVICES=$(rocm-smi --showuse --showpids \
  | awk '/^GPU\[[0-9]+\][[:space:]]+: GPU use \(%\)/{gsub(/[^0-9]/,"",$1); if ($NF+0==0){print $1; exit}}') \
  python3 scripts/_metric_mxfp8.py 2>&1 | tail -5

# 详细每形状日志
HIP_VISIBLE_DEVICES=... python3 scripts/_metric_mxfp8.py --verbose 2>&1 | tail -25
```

stderr 上看到的 PERF / SNR / STR 行用来定位是哪一项把 score 拉低了。

---

## 调参速查

```bash
# 只跑 1 轮 dry-run，验证 metric / 选 GPU 是否正常（不调 cursor-agent，也不跑 deep check）
python3 scripts/auto_optimize.py --dry-run --rounds 1

# 跑得久一点 / 提高耐心
python3 scripts/auto_optimize.py --rounds 80 --patience 8

# 单轮 cursor-agent 上限调到 60 min
python3 scripts/auto_optimize.py --round-timeout 3600

# 自定义 metric（stdout 最后一行非空行为一个数值，越大越好；默认脚本打印 1 个整数）
python3 scripts/auto_optimize.py \
    --metric-cmd 'python3 scripts/_metric_mxfp8.py' \
    --metric-name mxfp8_score

# 把 deep check 改得更紧：每轮都跑（极慢，~3-6 min/round）
python3 scripts/auto_optimize.py --deep-check-every 1

# 关掉 deep check（agent 还是会被 prompt 提醒，只是脚本不再自动验证）
python3 scripts/auto_optimize.py --deep-check-cmd ''

# 改 deep check 命令（譬如换成 stress）
python3 scripts/auto_optimize.py --deep-check-cmd \
    'HIP_VISIBLE_DEVICES=$(rocm-smi --showuse --showpids | awk ... | head -1) \
     STOP_AFTER_BAD=1 N_ITERS=200 \
     python3 .claude/probes/stress_grouped_mx_bwd_determinism.py'
```

完整选项 `python3 scripts/auto_optimize.py --help`。

---

## Deep check 行为

```
每 5 轮（--deep-check-every，可调）：
  1) 跑 cheap metric 拿到 score
  2) 跑 deep check (pytest mx_blockwise)，写 round_NNN/deep_check.log
  3) 如果 deep check 失败：
       - improved 强制为 False
       - best_metric / best_sha 不更新（即便 cheap metric 创新高）
       - rounds_without_improvement += 1
       - console 上打印一行 WARNING
  4) 如果 deep check 通过：
       - 按 cheap metric 正常判定 improved
```

非 deep-check 轮（即 `i % 5 != 0`）行为不变：只看 cheap metric。

`summary.json` 里每一轮多了三个字段：

```json
{
  "deep_check_ran": true,
  "deep_check_passed": true,
  "deep_check_exit_code": 0
}
```

可以 `jq` 取出来看：

```bash
jq '.rounds[] | select(.deep_check_ran) |
    {i:.index, metric, deep:.deep_check_passed, exit:.deep_check_exit_code}' \
   /workspace/code/Primus-Turbo/auto_optimize_logs/*/summary.json
```

---

## metric 评分细节

`_metric_mxfp8.py` 跑这些 shape：

| 形状 | 用途 | 备注 |
|---|---|---|
| DeepSeek-V3 GateUP B=16 (32768 × 4096 × 7168) E4M3 | fwd+bwd | 主权重的大形状，TFLOPS 占大头 |
| DeepSeek-V3 Down B=16   (32768 × 7168 × 2048) E4M3 | fwd+bwd | K 较小、MoE down-projection |
| gpt_oss_20B Down B=4    (8192 × 2880 × 2880)  E4M3 | fwd+bwd | N=2880 不整除 256，专门测 boundary tile |
| DeepSeek-V3 GateUP B=4  (8192 × 4096 × 7168)  E5M2 | fwd-only | E5M2 sanity，SNR 阈值降到 20 dB |

每个 shape 跑：
1. 一次 fwd+bwd 与 bf16 reference 对齐，算 SNR (`out` / `dA` / `dB`)，低于
   阈值（E4M3 默认 `MXFP8_SNR_E4M3=25`，E5M2 默认 `MXFP8_SNR_E5M2=20`，单位 dB）
   记 1 次 `snr_fail`。
2. 性能：`MXFP8_PERF_WARMUP`（默认 20）次预热后，在 `MXFP8_PERF_TRIALS`（默认 8）
   批上做计时；每批跑 `MXFP8_PERF_BATCH_ITERS`（默认 30）次调用，用 CUDA event
   测延迟，取各批平均延迟的 **最小值** 再换算 TFLOPS（对抗共享节点上的偶发抖动）。

最后跑 1 段确定性 stress：`STRESS_SHAPE`（G=4, M=1024, N=2048, K=2048, E4M3），
重复 `MXFP8_STRESS_ITERS`（默认 100）次 fwd+bwd，与首次运行对比 max-abs，超过
`MXFP8_STRESS_THRESH`（默认 `1.0`）记一次 `stress_bad`。这是 post-volatile-fix
后已知会爆 `out` race 的形状，用来给 loop 一个对 race 敏感的信号。

最终：

```
score = int(round(sum_tflops * 10))
      - 1000 * snr_fail
      -  100 * stress_bad
      - 2000 * exception
```

环境变量可以调整：

```bash
MXFP8_SNR_E4M3=25.0       # 默认值（可通过 env 覆盖）
MXFP8_SNR_E5M2=20.0
MXFP8_PERF_WARMUP=20
MXFP8_PERF_TRIALS=8
MXFP8_PERF_BATCH_ITERS=30
MXFP8_STRESS_ITERS=100
MXFP8_STRESS_THRESH=1.0
MXFP8_SNR_FAIL_PENALTY=1000
MXFP8_STRESS_BAD_PENALTY=100
MXFP8_EXCEPTION_PENALTY=2000
```

---

## 注意

- **GPU 选取**：`_metric_mxfp8.py` 自己用 `rocm-smi --showpids` 选空闲
  GPU；只有当 `HIP_VISIBLE_DEVICES` 已经被 export 时才不抢。**别在
  loop 启动时手动 export `HIP_VISIBLE_DEVICES`**，否则每一轮都被钉死在
  那张卡上、轮换不开。
- **maxMode**：脚本启动时把 `~/.cursor/cli-config.json` 的 `maxMode`
  设为 true，正常退出（含 Ctrl+C / `kill`）会 restore 原值。
- **不要 push**：prompt 已硬性禁止 push，每轮只在本地分支
  `dev/kyle_mxfp8_gg` 上提交。
- **scripts/_metric_mxfp8.py 自身改动**也会被算进 score（脚本会 import
  最新的 `primus_turbo`），所以 agent 修了 kernel 重编译后再跑 metric
  就能拿到新分。
- **跨形状 stress 不在这条快速 metric 里**：要拿可信的 stress 数字，
  cursor-agent 会用 `.claude/probes/stress_grouped_mx_bwd_determinism.py`
  跑 `N_ITERS=200..1000`；stress 数字写在 commit body 里以便人工抽查。
