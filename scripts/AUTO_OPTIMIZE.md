# auto_optimize.py 操作手册

每轮**新开** cursor-agent（不 resume），用 `git history + skill + prompt history` 三件套做跨轮延续。详见 `scripts/auto_optimize.py`。

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

启动后第一次会跑 baseline metric（`run_dod_metric.sh` 跑 4 个 DoD 测试文件，~几分钟）。看到 `[run_dod_metric] idle GPUs=... workers=N` 然后 `[metric] = <int>` 就是正常起步了。

---

## 看 log

每次启动会创建 `auto_optimize_logs/<timestamp>/`，结构：

```
auto_optimize_logs/
└── 20260427_084900/
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

## 调参速查

```bash
# 只跑 1 轮 dry-run，验证 metric / 选 GPU 是否正常（不调 cursor-agent）
python3 scripts/auto_optimize.py --dry-run --rounds 1

# 改 patience 为 8 轮无提升才退
python3 scripts/auto_optimize.py --rounds 80 --patience 8

# 改单轮 cursor-agent 上限到 60 min
python3 scripts/auto_optimize.py --round-timeout 3600

# 自定义 metric（必须打印 1 个浮点数到 stdout，越大越好）
python3 scripts/auto_optimize.py --metric-cmd 'bash scripts/run_dod_metric.sh' \
                                 --metric-name dod_score
```

完整选项 `python3 scripts/auto_optimize.py --help`。

---

## 注意

- **GPU 选取**：`run_dod_metric.sh` 每次跑前用 `rocm-smi --showpids` 选空闲 GPU（任何 PID 占 >100 MiB VRAM 的 GPU 视为 busy），自动 set `HIP_VISIBLE_DEVICES` 并把 `-n` 缩到 idle 数量。**别手动指定 `HIP_VISIBLE_DEVICES`** 否则会覆盖。
- **DoD 硬门槛**：4 个测试文件（`test_gemm.py / test_gemm_fp8.py / test_grouped_gemm.py / test_grouped_gemm_fp8.py`）默认模式 + `--deterministic-only` 都必须 0 failed。score 公式 `passed - 1000 * (failed + errors)`，一旦红了 metric 暴跌，patience 计数会增。
- **maxMode**：脚本启动时把 `~/.cursor/cli-config.json` 的 `maxMode` 设为 true，正常退出（含 Ctrl+C / `kill`）会 restore 原值。
- **不要 push**：prompt 里已硬性禁止 push，每轮只在本地 branch `dev/kyle_hipkitten_bf16` 上提交。
