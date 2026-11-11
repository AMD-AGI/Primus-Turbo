# Handoff prompt: FlyDSL MLA backward (gfx950) 继续优化，从 round 20 开始

把下面整段交给新机器上的 agent。

---

继续优化 Primus-Turbo 里 FlyDSL 写的 DeepSeek-V3 MLA attention 反向（gfx950 / MI355X）。目标是追平并超过
aiter a16 的反向。这是一个已经进行到 round 19 的 kernel-optimize campaign，从 round 20 接着做，用中文和我交流。

## 代码与记录

- 仓库：AMD-AGI/Primus-Turbo，分支 `dev/xiaoming/flydsl_mla_bwd_gfx950`（base 是 `dev/xiaoming/flydsl_mla_fwd_gfx950`
  的 `33beb4f8`：fwd round 20 + bwd round 16；本分支追加 bwd round 19）。
- 要优化的文件（只改这两个）：
  - `primus_turbo/flydsl/attention/gfx950_mla/flash_attn_bwd_mla.py`：kernel builder（odo、KV-outer fused dK/dV
    即 dkdv、a16 dQ）。
  - `primus_turbo/flydsl/attention/gfx950_mla/ops_bwd.py`：host 侧 `mla_bwd_sbhd`，编译期旋钮在 `_dkdv_kwargs`。
- 不要改 fwd 文件（`attn_helper_mla.py`、`flash_attn_fwd_mla.py`、`ops.py`），fwd 另有 campaign；也不要改 Turbo 其他
  kernel、Megatron、Primus。
- Campaign 目录：`agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109/`
  - `manifest.yaml`；`logs/optimize.md`、`logs/performance_trend.md`（两者只能追加）；`rounds/round-N/`
    （`summary.md`、`kernel_snapshot/`、`artifacts/`）；`quick_test_bench.py`（正确性 + 性能 harness）；`dev/*.py`（工具）。
  - 先读 `rounds/round-19/summary.md`、`rounds/round-18/summary.md`、`logs/performance_trend.md`。
- 开始前必读：`.cursor/rules/iteration_rules.mdc`、`.cursor/skills/kernel-optimize/SKILL.md`、
  `agent/historical_experience/gfx950/flash_attn_bwd_mla/flydsl/tips.md`。

## 目标与指标

- 形状（full set，7 个）：DSV3 MLA，Hq = Hkv = 128，d_qk 192，d_v 128，causal，SBHD bf16；
  b1_s4k、b1_s8k、b1_s16k、b1_s32k、b1_s64k、b2_s4k、b2_s16k。quick set：b1_s4k、b1_s16k、b1_s64k、b2_s4k。
- 主指标：Combined Step TFLOPS =（fwd FLOPs + bwd FLOPs）/（fwd ms + bwd ms），对 PASS shape 取几何平均。
  fwd 用的是本分支的 fwd r20，不变。
- 正确性门槛：dQ / dK / dV 对 fp32 参考的 SNR > 40 dB（harness 的 `Check` 列）。
- 当前 r19（n02-29 GPU 0，2026-11-11）：step 925.220（aiter 的 0.929），bwd 849.957（0.9125），fwd 1202.626；
  每个 shape 的 step ratio：4K 0.884 / 0.886（最差），8K–64K 0.938–0.961。
  SNR dQ 44.68–47.02、dK 49.54–49.66、dV 51.08–51.38 dB。
- 基线 round 1：step 738.593、fwd 1182.182、bwd 645.451（trend 表 `vs Baseline` 都相对这一行）。

## 环境（关键）

- 已验证的环境：docker 镜像 `tasimage/primus:pr-1275`，torch 2.14.0+rocm10.1.0、flydsl 0.3.4.1、
  primus_turbo 0.6.1.dev4（`/opt/venv` site-packages）。开始先打印这三个版本。
- `gfx950_mla` 必须配 site-packages 里的 primus_turbo 跑：`attn_helper_mla.py` 调用的
  `buffer_ops._get_buffer_flags()` 在仓库 main 的 `primus_turbo/flydsl/utils/buffer_ops.py` 里不存在。不要在仓库根目录
  运行，不要把仓库根加进 `sys.path`，也不要 `pip install` 这个分支。
- harness 和 dev 工具从 `$MLA_PKG_ROOT/flydsl_mla` 导入包。建一个 shim（下面称 `$ROOT`）：
  `mkdir -p $ROOT && ln -s <worktree>/primus_turbo/flydsl/attention/gfx950_mla $ROOT/flydsl_mla`，所有命令都带
  `MLA_PKG_ROOT=$ROOT`。A/B 用的参考版本要做成独立拷贝（`cp -r .../gfx950_mla rounds/round-N/pkg/flydsl_mla`），
  不能用 symlink。
- 硬编码路径要改：脚本默认 `/perf_apps/xiaoming/...`。
  - `dev/d128_regress.py` 第 7 行写死了 `sys.path.insert(0, "/perf_apps/xiaoming/megaattention_overlap_bench")`，
    改成读 `MLA_PKG_ROOT`。
  - `rounds/round-19/artifacts/run_full_ab.sh` 里的 `C=`、包根路径也要换成本机路径。
- 固定用 GPU 0（`HIP_VISIBLE_DEVICES=0`），整个 campaign 不换卡。不要 kill 别人的进程。

## 第一步：在新机器上 REBASE

1. 编译 r19，检查 dkdv ISA hash 能否复现：
   `MLA_PKG_ROOT=$ROOT FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<d> python dev/isa_dump_bwd.py`，然后
   `grep -v '^\s*;' <d>/*mla_dkdv_kernel*/22_final_isa.s | grep -v '^\s*\.' | md5sum | cut -c1-10`。
   r19 应为 `a87e141e6e`（r18 是 `ae8b324271`）。不同说明工具链不同，记下来。
2. full set 跑 3 次（`python quick_test_bench.py --set full --summary-csv <csv>`，每次约 1 分 40 秒），在
   trend 表追加一行 `19 (<node>) REBASE`。之后的轮次都和同机、同 session 的参考版本比。

## 每轮流程（摘要，细则以 iteration_rules.mdc 为准）

- 一轮只做一个改动，先写假设。
- 正确性：
  - full set 7/7 PASS；
  - `dev/bitwise_r18.py`（`dump <out.pt> 3`，然后 `cmp <ref.pt> <new.pt>`）：只改调度、不改算术顺序的改动，dK/dV 应与参考
    bitwise 一致，dQ 是 bf16 atomic，看 SNR 是否落在参考版本重复运行的范围内；
  - 改到共享 builder 代码时跑 `dev/d128_regress.py`，要求 REGRESS PASS。
- 筛选：
  - ISA 先筛：`dev/isa_sweep.py <configs.py> <out_dir>`，输出 spill 数、scratch 字节数、masked loop 和 hot loop 的
    scratch_load 与 vmcnt(0) 数，以及 hot loop 每次 scratch reload 到第一个保护它的 `s_waitcnt vmcnt(k)` 之间的 MFMA 数
    （cover，`dev/isa_reload_cover.py`）。gfx950 的 vmcnt 按发射顺序完成。
  - 再分进程计时：`dev/sweep_proc.py <configs.py> [rounds]`，`CONFIGS = [("label", {overrides}), ...]`，第一项是参考；
    `"__pkg__"` 指定某个配置的包根，`SWEEP_SHAPES` 环境变量选 shape。
  - 不要在同一个进程里比较多个配置：scratch 会继承，round 9 时这样排错过 25 个百分点。
- 验收：full set 轮换 A/B，新版本 ×3、参考 ×2 交替（模板是 `rounds/round-19/artifacts/run_full_ab.sh`）。新版本取
  3 次的中位数，参考取 2 次的均值。step geomean 提升，且没有 shape 退化 ≥ 5%，才 ACCEPT；否则 ROLLBACK。
  b1_s4k 进程间波动最多 5%，4K 上的小幅下降要用单独的分进程 A/B 确认。
- 每轮都要有：`summary.md`（含 Real-training transfer check）、`kernel_snapshot/`（三个文件：
  `flash_attn_bwd_mla.py`、`ops_bwd.py`、`__init__.py`）、`optimize.md` 和 trend 表各追加一条；有可复用的结论就追加到
  tips.md。

## 现状与下一步方向

- 结构要点：
  - kv tile 192，q_split 1，waves_per_eu 1；dkdv 有 98 个 VGPR spill（scratch 396 B），hot loop 约 611 条指令，
    每个 trip 有 3 次 scratch reload。
  - DS_PAR（round 18）在 GQA = 1 路径上让 dS slot 按 trip 交替，修掉了 deferred GEMM3 和下一次 dS 写之间的 slot-0 WAR
    竞争。这是正确性修复，必须保留。
- 建议的下一步（假设，未验证）：
  1. 把 DS_PAR 多出的 4 个 spill 收回来（94 → 98）：GEMM3 的读地址改成从写地址推出，不再两个都由循环携带。
  2. 4K 是最差的 shape：先量 prologue / masked loop 占多少，再决定方向。
  3. 继续减少 hot loop 的 scratch reload，或者给它们更大的 MFMA cover。
- 已回滚、不要原样重试的方向：
  - r5：在 unmasked q-loop 入口加 vmcnt(0)，-2.6%；
  - r6：降序 q walk 的旋转周期，无变化；
  - r10：GEMM1a 的 K 从 LDS 读，-28% 到 -40%；
  - r11：Q/dO 预取的发射位置；
  - r12：kv tile 的派发顺序，zigzag -21%；
  - r14：LLVM 选项第二轮扫描，无变化；
  - r17：把 deferred GEMM3 挪过 publish barrier，-1.8% 到 -8.3%。
  原因见各轮 summary。
- 旋钮最优值会随 hot loop 结构移动：每做完一次结构性改动，都要重扫 `g2d`、`g3d`、`g3_st_at`、`g1_ks_outer` 和
  两个 LLVM 选项。多个旋钮的收益不能叠加。

## 约束

- 未经我明确指示不要 git commit。需要提交时用 `git -c user.name=xiaoming-amd -c user.email=xiaompen@amd.com`，
  代码用 ruff 0.15.12 格式化，并通过 `tools/check_license.py` 和 `tools/check_env_reads.py`。
- 数字必须来自本次实测或已引用的文献，不要编造硬件规格或加速比；事实和假设分开写。不要把发现写进 `.cursor/skills/`。
  不用 emoji。
- 合并提示：`dev/xiaoming/flydsl_mla_fwd_gfx950` 已经前进到 `99db5b10`（fwd QT4 + persistent job loop），只改了 fwd
  的三个文件，和本分支没有文件冲突。新 fwd 的 `mla_fwd_sbhd` 接口和 LSE 约定（`[B*Sq, Hq]` fp32，自然对数）没变，
  但新 fwd 配 bwd r19 这个组合还没验证过。bwd campaign 一直用 fwd r20 计算 step，换 fwd 要单独记一行 REBASE。
