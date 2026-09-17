# 已发车：`gfx1250-flydsl-attn-bwd`

启动时间 2026-09-17 11:59:34 UTC · 宿主 `heliosr-1b114-c07-1` · 容器 `fa-repro` · 单卡 @1100 MHz

```bash
cd ~/code/2026_0910__op-evolve/op-evolve
export PATH="$PWD/.venv/bin:$PATH"
setsid nohup env PATH="$PATH" tools/supervise_job.sh jobs/gfx1250-flydsl-attn-bwd.yaml >/dev/null 2>&1 &
```

- supervisor PID / **PGID 86137**（单进程组）
- 日志 `~/code/2026_0910__op-evolve/op-evolve/artifacts/supervisor.log`
- 产物 `~/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-<id>/`

## 它要回答的问题

**不是「FlyDSL 能不能超过 ASM」，是「能不能追平」。** `beat` margin 设的是 **0%**。
基线 ~5.8 TFLOP/s，锚 ~540 TFLOP/s，差约 93×，搜索空间未开采。

## 发车前踩到并修好的三处（都是 `check_runner` 自己报出来的）

1. **hipBLASLt**：镜像没有 gfx1250 的 Tensile 库，`TORCH_BLAS_PREFER_HIPBLASLT=0` 必须在
   torch 选 BLAS 后端之前设好。op-evolve 走 `docker exec ... bash -c`（**非 login shell**），
   所以 `/etc/profile.d` 不生效。追加到容器**已有的** `/usr/lib/python3.12/sitecustomize.py`
   —— 放进 venv 的 site-packages **不行**，系统那个在 sys.path 上更靠前，会遮蔽它。
2. **flydsl 0.3.2 从 `/tmp` 挪到 `/home/lihuzhan/.local/flydsl032`**（宿主挂载）。
   `/tmp` 不保证活过容器重启，而 supervisor 会重启。与镜像的 0.2.4 并存，绝不覆盖。
3. **`supervise_job.sh` 里 `VENV` 是另一个用户的硬编码路径**，会回落到 `command -v op-evolve`，
   而我们的不在 PATH 上。它自己的注释警告过：BIN 找不到 → 每次调用 exit 127 →
   **无限重启却什么都不跑**。发车时把 venv 的 bin 前置到 PATH。

## 怎么停

```bash
kill -TERM -86137            # 注意那个减号：杀进程组，不是进程
# 或
.venv/bin/op-evolve stop --job gfx1250-flydsl-attn-bwd-<id> --force
```
**绝不要用裸 `kill <pid>`** —— 会把 agent 子进程孤儿化并弄坏 `job_context`。

## 怎么看

```bash
tail -f artifacts/supervisor.log
.venv/bin/op-evolve status --job gfx1250-flydsl-attn-bwd-<id>
cat artifacts/gfx1250-flydsl-attn-bwd-<id>/job_context/progress.md
```

**读 `progress.md` 时注意**：要横着读每一行（shape 对 `current(retest)`），
不要竖着读吞吐那一列 —— 那些数字取自不同 session，同一份代码之间会漂约 1%。
`gain` 和 `changed` 才是判决依据。

## 期间的纪律

**这张卡现在归这个作业。** 期间不要跑任何交互式 GPU 工作，否则两边的数字都会被污染
（同配置在竞争下曾从 11.2 漂到 17.1 ms，SQNR 还会出现假失败）。
