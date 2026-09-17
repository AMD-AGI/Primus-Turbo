# Stage 1 / Phase B：aiter 的 gfx1250 FlyDSL 前向 —— 实测与决策

日期 2026-09-17 · 机器 `heliosr-1b114-c07-1`（**单卡，VR 限频 1100 MHz**）· 容器 `fa-repro`
脚本 `bin/stage1_fwd_ab.py`（可复跑）· 原始数据 `stage1_fwd.json`

---

## 0. 结论先行

| | 结果 |
|---|---|
| FlyDSL gfx1250 前向能跑起来吗 | **能**，但**必须用 flydsl 0.3.2，0.2.4 不行** |
| 它比现有 ASM 前向快吗 | **不快。慢 1.51×**（2.373 ms vs 1.569 ms，median，n=20） |
| 正确性 | 两条臂都过：FlyDSL 51.63 dB / ASM 53.62 dB，`isfinite` 两边都是满覆盖 |
| LSE 是自然对数还是 log2 | **自然对数**（ratio 1.0000，SQNR 81.85 dB）。与 fused backward 的配对不需要转换 |
| 按 `PLAN.md` 的决策门 | 落在**中间档 1.572–2.5 ms → 继续，但降级预期**：按"可维护性 + 补 varlen 缺口"立项，不按性能 |

**并且发现了一个计划里没有的硬约束**：aiter 的 gfx1250 前向要 0.3.2，而 Primus-Turbo 自己的
FlyDSL 树要 0.2.4，**两者在一个进程里装不下**。见 §4。

---

## 1. 测量

形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`（生产形状）。
两条臂**都来自 aiter**，所以这次测量完全没有 import primus_turbo，也就绕开了 §4 的版本冲突。

### 正确性（先做，fp32 分块参考）

| 臂 | SQNR out | isfinite 覆盖 |
|---|--:|---|
| FlyDSL | **51.63 dB** | 134217728 / 134217728 |
| ASM | **53.62 dB** | 134217728 / 134217728 |

输出张量**预填 NaN**，所以"没写"与"写了 0"可区分；两边都是满覆盖，**没有 wave32 误判导致的丢工作**。
参考是分块 fp32（稠密 `[4,32,8192,8192]` fp32 score 要 34 GB，不可能一次算）。

### 速度（ABAB 交替，CUDA event，每次 256 MiB L2 flush，按秒预热）

| 臂 | median | best | sd | n |
|---|--:|--:|--:|--:|
| **FlyDSL** | **2.3732 ms** | 2.3282 ms | 2.68% | 20 |
| **ASM** | **1.5691 ms** | 1.5552 ms | 1.09% | 20 |

**asm / flydsl = 0.6612 —— ASM 快 1.51×。**

### 测量链的交叉验证

ASM 这次测出 **1.5691 ms**，与 0915 记录的冠军 **1.572 ms** 相差 0.2%。
这不是巧合而是证据：测量链（形状、参考、计时器、flush）与那次是同一套，
所以 FlyDSL 那个 2.373 ms 可以和历史表格直接并列。

### 时钟见证（必读）

```
sclk before: 1100 MHz
sclk after :  967 MHz     <-- 测量窗口内 VR 限频下掉了 12%
sclk (事后): 1100 MHz
```

**ABAB 交替正是为这个准备的**：漂移等量落在两条臂上，所以**比值站得住**。
但**绝对值只在 ≤1100 MHz 下成立**，与满频机（2133–2244 MHz）的数字不可直接比。

---

## 2. 决策

`PLAN.md` 的门：

| 结果 | 决策 |
|---|---|
| ≤ 1.572 ms | 继续 Stage 3 |
| **1.572–2.5 ms** | **继续但降级预期** |
| > 2.5 ms 或不过 | 停 |

**2.373 ms 落在中间档。** 按原文：**"反向大概率也只是同量级，按可维护性而非性能立项"。**

要诚实地说清这意味着什么：SURVEY §8 原来的论证里有一条是"前向模板已经存在且有竞争力"。
**前半句成立，后半句不成立** —— 这套 gfx1250 FlyDSL 惯用法在本卡上比手写 ASM 慢 1.51×。
剩下的立项理由**只有结构性的那三条**：

1. 现有 ASM 反向的 GQA 越界写（我们用 1 GiB scratch + host 规约绕过）；
2. gfx1250 的 ASM 资产只有 gfx950 的 1/25（反向 6 个 `.co` vs 124）；
3. **gfx1250 完全没有 varlen ASM 反向。**

第 3 条现在是最强的那条：它不是"快多少"的问题，是**有没有**的问题。

---

## 3. 一条被撤回的结论：0.2.4 不够用

`API-DELTA.md` 原本的结论是"0.2.4 只缺 4 个工具函数，写 shim 即可，不必升级"。
**这条撤回。** 4 个符号确实只是改名/等价物，shim 也确实让构建往前推进了很远，
但推进之后撞上的不是第五个符号，而是**编译器前端的语义差异**：

```
TypeError: state variable 'result' is list, not an MLIR Value;
           stateful dynamic if requires MLIR-backed values.
  at flydsl/compiler/ast_rewriter.py:666
  from fmha_fwd_prefill_a16w16_m32x8.py:1125  _maybe_rescale
```

0.3.2 的 `ast_rewriter` 接受 list 作为 stateful-if 的状态变量，0.2.4 不接受。
**这不是别名能修的。** 换到 0.3.2 之后，**零 shim**，一次就跑通。

shim 本身保留在 `bin/flydsl_024_shim.py`，作为"做到哪一步才撞墙"的记录，不再是方案。
它顺带留下三条对以后有用的经验：

- `fx.ceildiv` 不能简单别名到 `fx.ceil_div` —— 后者带 `@coerce_int_tuple_args`，返回 `IntTuple`，
  而调用点接着要做 fx 标量运算，失败发生在很远的地方，报一个调用点从未提到的类型。
- `fx.max`/`fx.min` 要用 `arith.max{si,ui}` 并**把结果重新包成操作数的 fx 类型**
  （构造模式抄 `numeric.py` 自己的 `out_type(op(lv, rv))`），而且传进去的必须是
  `as_ir_value()` 的真 `ir.Value`，不是 `_extract_arith` 的 `ArithValue` 包装。
- `create_llvm_ptr` 的替换**必须打三个命名空间**（`kernels_common` 和两个
  `fmha_gfx1250/` 模块都在 import 期绑定了这个名字），并且要验证**消费方**读到的是新值。
  这正是 `API-DELTA.md` §2.3 记录的那个陷阱，在这里实际发生了一次。

---

## 4. 新发现的硬约束：0.2.4 与 0.3.2 装不进同一个进程

| 谁 | 要什么 | 证据 |
|---|---|---|
| aiter 的 gfx1250 FlyDSL 前向 | **0.3.2** | 0.2.4 的 `ast_rewriter` 拒绝它的 stateful-if（§3） |
| Primus-Turbo 自己的 `primus_turbo/flydsl/` 树 | **0.2.4** | `flash_attn_bwd.py:27` `from flydsl.expr import ... buffer_ops ...`；**`buffer_ops` 在 0.3.2 里被整个删除**（全包 `find -name "buffer_ops*"` 无结果） |

实测：把 0.3.2 放上 `PYTHONPATH` 之后，`import primus_turbo` 直接失败：

```
ImportError: cannot import name 'buffer_ops' from 'flydsl.expr'
  primus_turbo/flydsl/attention/flash_attn_bwd.py:27
```

而且这个 import 是**无条件的** —— `primus_turbo.pytorch` → `ops.attention` →
`flash_attn_interface` → `attention_flydsl_impl:20` → `primus_turbo.flydsl.attention.flash_attn_bwd`，
一路没有 try/except 护住。所以在 0.3.2 下，**整个 primus_turbo.pytorch 都 import 不进来**，
不只是 FlyDSL 那部分。

### 对计划的影响

- **Phase B 不受影响**：两条臂都来自 aiter，本次测量根本没 import turbo。
- **`tune_attention.py --impl flydsl` 目前跑不了**：它要同时 import turbo（拿 fused backward）
  和 aiter 的 FlyDSL 前向。这不是 harness 的 bug，是版本冲突。
- **Phase D / Stage 4 的派发接线要重新设计。** 原计划"在 `attention_impl.py` 新增一个 gfx1250 分支"
  隐含假设一个进程里能同时有两者。现在至少要先解决：
  1. 让 `primus_turbo/flydsl/` 在 0.3.2 下**import-safe**（惰性 import，或把 gfx950 那棵树
     的 import 放进已有的 `FLYDSL_AVAILABLE` try/except 里）；**这是最小改动，且独立有价值** ——
     现在 turbo 在 gfx1250 上连 `import primus_turbo.pytorch` 都依赖 flydsl 版本；
  2. 或者把 turbo 的 FlyDSL 树移植到 0.3.2（`buffer_ops` 被删，要找替代 API）；
  3. 或者新写的 gfx1250 反向不进 turbo 的 `primus_turbo/flydsl/` 树，而是作为 aiter 侧的贡献。

**建议先做 (1)**：它是一个小改动，不依赖任何决策，并且把"turbo 在 gfx1250 上能不能 import"
这件事和"用哪个 flydsl 版本"解耦。

---

## 5. 复跑

```bash
docker start fa-repro
docker exec fa-repro bash -lc 'pip install --target /tmp/flydsl032 "flydsl==0.3.2"'   # 一次即可
docker cp output/0917__flydsl/bin/stage1_fwd_ab.py fa-repro:/tmp/
docker exec -e TORCH_BLAS_PREFER_HIPBLASLT=0 fa-repro bash -lc 'cd /tmp && python3 stage1_fwd_ab.py'
```

`--target` 安装 + `sys.path` 前插，**不动容器的 site-packages** —— turbo 自己那条
FlyDSL GEMM 路径（0916 实测 6/6 形状可用）因此不受影响。
