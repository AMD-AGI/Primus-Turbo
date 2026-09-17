# gfx1250 FlyDSL attention：开发计划

日期 2026-09-17 · 依据 `SURVEY.md` · **机器可用时可直接照此执行**

工作量单位是 **agent session**（一次有上下文的连续工作）与 **GPU run**（一次需要占卡的运行），
不是人日。本机的风险按**启动次数**计价：0916 实测 14 次训练启动里 3 次在启动阶段挂卡（21%），
而跑起来之后零挂卡 —— 所以**能在一个进程里做完的事，不要拆成多次运行**。

---

## 总览

| Stage | 内容 | GPU run | agent session | 前置 |
|---|---|--:|--:|---|
| **0** | 静态可行性（**现在就能做**） | **0** | 1–2 | 无 |
| **1** | 接通并测量 aiter 的 gfx1250 FlyDSL **前向** | 1–2 | 1 | Stage 0 |
| **2** | 决策门：继续 / 停止 | 0 | 0.5 | Stage 1 |
| **3** | gfx1250 FlyDSL **反向** 第一版（odo + dkdv 骨架） | 3–5 | 3–5 | Stage 2 通过 |
| **4** | 反向补全（dq、LSE、GQA 规约）与正确性门 | 3–5 | 2–4 | Stage 3 |
| **5** | 调优与 e2e 验证 | 4–6 | 2–3 | Stage 4 |

**合计（若全程通过）：11–18 GPU run，9–15 agent session。**
Stage 3 起是**新写内核**，这个估算的不确定度大，且 §决策门 存在提前终止。

---

## Stage 0 — 静态可行性（0 GPU，现在就能做）

三件事，全部只读 + 无 GPU 一次性容器（`docker run --rm --entrypoint sh fa-tune:deps`，
**不挂 `--device /dev/kfd`**，不会干扰同事）。

### 0.1 aiter 那份前向能否在 flydsl 0.2.4 下工作 ★最高优先

**问题**：aiter pin `flydsl==0.3.2`（`aiter-src/setup.py:17`），我们容器里是 **0.2.4**。
已知断裂：`T.f8` 在 0.3.0 从 `flydsl.expr.typing` 移除。

**做法**（纯静态，不 import）：
1. 列出 `aiter/ops/flydsl/kernels/fmha_gfx1250/*.py` 与 `fmha_kernels.py` 的全部
   `from flydsl...` / `flydsl.` 引用；
2. 对每一个符号，在容器内 `grep` 0.2.4 的包，判断存在 / 缺失 / 签名不同；
3. 产出一张 **API 差集表**，每行标注：0.2.4 有 / 无 / 需适配。

**产出**：`output/0917__flydsl/API-DELTA.md`
**判据**：差集为空或只需少量 shim → 不必升级；若核心 op 缺失 → 升级 0.3.2 成为前置，
并需评估它对 turbo 自己 FlyDSL 代码（pin 0.2.4）的破坏面。

### 0.2 wave32 误判的防护

`flydsl/runtime/device.py:76` 的 `is_rdna_arch` 把 `gfx1250` 判成 CDNA → wave64
（`gfx1250` 不匹配 `gfx120*`）。aiter 在 `kernels_common.py:52-69` 自己重判。

**做法**：静态审计我们将要调用的每一条路径，标出所有会读到 FlyDSL wave size 的位置；
参照 aiter 的写法准备一个显式 wave32 覆盖。
**产出**：写进 `API-DELTA.md` 的一节。
**为什么必须在 Stage 0 做**：这个错误**不报错、只丢工作**，事后从性能数字上看不出来。

### 0.3 harness 接线准备

- `tools/gfx1250/tune_attention.py:346` 的 `--impl` 只接受 `{turbo, aiter, fused, asm, asmbwd}`，
  需要加 `flydsl`；
- **四张量 SQNR 门（out/dq/dk/dv ≥ 50 dB）可原样复用**，不要另写；
- 参照 `tests/pytorch/ops/test_gemm_gfx1250.py`（518 行）的 gating 惯用法写测试骨架。

**产出**：`--impl flydsl` 的代码改动（未运行）+ 测试骨架。

---

## Stage 1 — 接通并测量 aiter 的 gfx1250 前向（1–2 GPU run）

**这是整个计划的决策依据，也是最便宜的一步。**

**做法**：在**一个进程**里完成，不经训练启动（避开 21% 的启动挂卡窗口）：
1. `PYTHONPATH` 已包含 `/home/lihuzhan/code/aiter-src`（`e2e.sh:66` 实测），
   直接调 `aiter.ops.flydsl.fmha_kernels.flydsl_flash_attn_func`；
2. 形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`（生产形状，不用代理形状 —— 这一步只测一次，
   不需要省时间）；
3. **先过正确性再看速度**：NaN 预填充 + `isfinite` 覆盖检查，再四张量 SQNR；
4. 与我们现有 ASM 前向 **1.572 ms** 交替对拍，n≥3。

**产出**：`output/0917__flydsl/STAGE1-FWD.md`，含逐次原始值、SQNR、覆盖检查结果。

**这一步能同时回答四个问题**：FlyDSL 能不能在本卡上跑起来、0.2.4 够不够、
这套 gfx1250 惯用法的竞争力如何、以及派发管线通不通。

---

## Stage 2 — 决策门

| Stage 1 结果 | 决策 |
|---|---|
| FlyDSL 前向 **≤ 1.572 ms**（不差于现有 ASM） | **继续 Stage 3**。惯用法竞争力得到证明 |
| 在 1.572–2.5 ms 之间 | **继续但降级预期**。反向大概率也只是同量级，按"可维护性"而非"性能"立项 |
| **> 2.5 ms**，或 SQNR 不过，或跑不起来 | **停**。见 `SURVEY.md` §8 的终止条件 |

**无论哪个结果都要写下来** —— 一个说"不做"的调研才是有用的调研。

---

## Stage 3 — gfx1250 FlyDSL 反向：第一版（3–5 GPU run，3–5 session）

**不是移植 CDNA4 那份**（三原语全部 Cannot select，见 SURVEY §2），
**是以 aiter 的 gfx1250 前向为模板新写**。

可复用的 gfx1250 惯用法（全部来自 `fmha_gfx1250/`，**已实测存在**）：
`ds_load_tr16_b128`（12 处）、`make_tdm_atom` + `tdm_ops`（TDM 异步拷贝）、
`s_wait_asynccnt` / `s_wait_dscnt`、`sched_barrier`、WMMA `16x16x32` atom、wave32 布局。

**建议的实现顺序**（按"能独立验证"排，不按数据流排）：

1. **`odo`（`delta = rowsum(dO*O)`）** —— 最简单，且 `delta` 在 torch 里是三行，
   **可以在不发射主内核的情况下单独判对错**。0915 的 ASM bring-up 正是这么做的，
   第一次就过（SQNR 147 dB）。**先做它，用来打通整条工具链。**
2. **`dkdv` 主内核** —— 最贵的一块。**这里要特别注意 GQA**：
   aiter 的 ASM 版就是在这里按 q head 索引写进 kv 尺寸缓冲区造成越界写。
   新实现应当**从一开始就按 q head 分配或做 workgroup 内规约**，不要重蹈。
3. **`dq`** —— 需要 atomic 或 split-reduce，决定要不要 fp32 accumulator。

**每一步的验收都是四张量 SQNR + NaN 预填充覆盖检查**，不是只看 `out`。
0916 抓到过 `out 53.67 ✓ / dq 52.24 ✓ / dk −inf ✗ / dv −inf ✗` 这种只查 output 就会放行的情况。

---

## Stage 4 — 补全与正确性门（3–5 GPU run，2–4 session）

- LSE 的打包/解包（参考 PR #516 的 packed-LSE adapter —— **结构上最像我们的处境**：
  FlyDSL 前向 + AITER 反向）；
- GQA 规约路径；
- 接进 `attention_impl.py` 的派发（新增一个 gfx1250 分支，**不要复用 `_flydsl_common_ok`** ——
  它的 `is_gfx950()` 是对的，不该放宽）；
- 端到端可用性：`--impl flydsl` 全链路跑通。

---

## Stage 5 — 调优与 e2e 验证（4–6 GPU run，2–3 session）

- 算子级调优：tile / warp 数 / TDM buffer 数。**用 `sweep_attention.py`**，它已经有
  每候选超时 + 进程回收（0916 加的），并且 `waves_per_eu` 已验证真的生效；
- **e2e A/B 必须 n≥3 对、交替顺序、钉种子**；
- **每次运行后检查 `nan_steps`** —— `e2e.sh` 现在会自己打印，非零即作废。

---

## 全程必须遵守的纪律（都是踩出来的）

1. **先按正确性给运行评分，再按速度。** 0917 扫描 74 次运行发现 8 次 `loss: nan`，
   而且**每一个 nan 运行都是它那组里最快的** —— 崩坏的计算通常更快，
   被污染的运行看起来像你手里最好的结果。
2. **不要在训练路径上调用 autotune。** 它的 `except: continue` 无法从已损坏的 HIP context 恢复，
   0916 因此挂了一次卡。离线建表，在线只查表。
3. **抽样检测器要报覆盖率。** 每 N 次抽查一次的覆盖率可能只有 14%，"零命中"几乎是噪声。
   用异步累加器 + 每 N 次检查累加器，做到 100% 覆盖、1/N 同步。
4. **看门狗绑定到具体运行**（`E2E_RUN_MARKER`），不要用 `pgrep -f e2e.sh` ——
   它会匹配下一次运行并把它杀掉（0916 发生过）。
5. **风险按启动次数计价**：能在一个进程里做完的事不要拆；长任务用 60 步配置
   （`repro_l8b_turbo_conv_8L_long.yaml`）而不是多次 20 步。

---

## 优先级说明：这件事排在哪

`SURVEY.md` §8 已经说明：**单看"FlyDSL 反向比 AITER 快多少"，预期只有 1.13× 量级，很弱。**
它值得做的理由是结构性的（源码可控、补齐 gfx1250 资产缺口、前向模板已存在）。

因此建议的相对优先级：

| | 项目 | 理由 |
|---|---|---|
| **1** | **定位 nkfix 的 21% nan 率** | 决定当前 7.08× 能否交付。**优先于本项目** |
| **2** | 向 aiter 上报 GQA 越界写 + 申请 gfx1250 `psskddv` 变体 | 成本近乎为零，直接对上反向差距的根源 |
| **3** | **本计划的 Stage 0–2** | 便宜（0–2 GPU run），且能一次性回答"这条路通不通" |
| 4 | 本计划 Stage 3–5 | 仅在 Stage 2 通过后 |

**Stage 0 现在就可以开始，不需要等机器。**
