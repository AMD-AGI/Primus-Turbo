# softmax 原型（L15/L17/L20）编译期原型

构建者的完整报告因 harness 限制未落盘，见 workflow 返回值。本文件只含对抗性审查。

## Review（2026-09-27，只编译，未碰 GPU）

**结论：ready-for-card。** 没找到正确性 bug。修了一处潜在的 fast-math 隐患，ISA 不变。

### 1. 读 diff 反驳正确性（均未成立）
- **LDS / barrier / 双缓冲**：diff 没动 LDS 布局、barrier、K/V ring 索引、TDM/ds 地址，也没动 `non_cur_pp` 的 O 暂存。新增的只有寄存器算术，以及 epilogue 里的一条 `v_permlanex16`。这条 permlane 位于三个 `_run_tiles` 之后的直线代码中，控制流 uniform，EXEC 满，读不到非活跃 lane。不存在新的跨 wave 竞争。
- **per-lane d（L17）**：lane l 与 l^16 的 m_prev、m_new 都已 peer 归约，defer 下 ballot 又使 do_rescale 在 wave 内一致，所以两个 lane 的 corr 逐位相同。于是 d(l)+d(l^16) 恰好等于完整行和的 corr 链。
  - epilogue 两个 lane 算的是 a+b 和 b+a，逐位相同，所以 1/d 与 LSE 在 lane 对内一致，也是确定性的。
  - 首个 tile：m=BIG_NEG，d=0，corr=exp2(-大)=0，正确。全 mask 行：d_red=0，由原有的 `d>0` guard 置 O=0，LSE=-inf，与 base 一致。
  - `_QS+1` 的消费者只有 epilogue 的 O 归一化和 LSE 两处，两处都已改为 d_red；`_softmax` 只有一个调用点。没有漏改的路径。
- **sink seed**：1.0 只放在 khalf==0 的 lane，归约后为 1。m_init 在 lane 对内一致。代数正确，但没有数值测试，job 也不走这条路径。
- **PK exp（L15）**：切片 `s_masked[idx:idx+8]` 加上 idx+=8，与 base 的 (kvt,i) 顺序一致。P 的来源不变。
- **packed row-sum 树**：先对 NKV 个 v8 做成对相加（奇数情况已处理），再 v8→v4→v2→标量，8 个分量全部覆盖。求和用的是 f32 的 p，与 base 相同；PV 用 bf16 的 p，也不变。
- **branch-free（L20）**：`do_rescale=None` 时走 caller 原有的无条件 `o*=corr` 分支，与 `ENABLE_DEFER_RESCALE=False` 的旧路径相同。m 每个 tile 取精确最大值，没有溢出风险。
- **GQA / causal / tile 边界**：mask 与 head 映射代码未改。masked 循环里 `v_cndmask …, 0xff800000` 的条数：prod 126/126，win_sink 256/256，non-causal 128/128，与 base 相同，mask 全部保留。

### 2. 已修（原地修改，并重新编译验证）
- **L15 的 v8 fma 原来带 `fastmath=fast`（含 ninf）。** 它的输入 S 在 masked 位置是 -inf，按 LLVM 语义结果是 poison。
  - 更要紧的是：原型之后，mask 的 `select(c, -inf, s)` 的全部 user（fmax 与这个 fma）都带 ninf。LLVM 因此有权把 -inf 那一臂折掉，也就是删掉 causal mask。base 里这个 fma 不带 flag，所以不会出现这种情况。
  - 当前编译器没有这样折叠（cndmask 条数相同），但换个 LLVM 就可能出问题。已改为与 base 一致的无 flag `fmath.fma`。
  - 重编结果：prod md5 仍为 6b286ce7，win_sink 仍为 877ebf21，ISA 逐位不变。`softmax_proto.diff` 已重新生成。

### 3. ISA 声明复核（自己重跑了编译和工具）
- 当前 `op/` 重编 prod 得到 md5 6b286ce7，可复现。资源 VGPR 446、SGPR 98、spill 0/0、scratch 0、LDS 327680，属实。window+sink 的 SGPR spill 6→35，属实。
- 用 `tools/dyn.py` 和 `tools/crit.py` 复核：
  - LO-clean 每次迭代：base 551（rescale 触发时 622），proto 537，PK+LANE+defer 506。
  - QK→PV 串行段：base 337/496，proto 222/347，PK+LANE(scalar)+defer 218，均属实。
  - 全 kernel 的 permlanex16：16 → 10。
- **新增 non-causal 编译**（`tools/compile_isa_review.py nc`，driver 是副本，原 driver 未改）：
  - base、proto、proto+defer 都编译通过，VGPR 432/432/440，scratch 0。
  - SGPR spill 2→4，都 spill 到 VGPR lane，不是正确性问题。
  - LO-clean 每次迭代 544→536，趋势与 causal 一致。
- `tools/cpu_algebra_check.py` 重跑：max rel 4.0e-6，OK。

### 4. 残留风险（不阻塞上卡）
1. BF 在 n_block=64 下可能是净负。按报告第 6 节把 C1、C2、C3 分成独立 arm 测。
2. PV 跨度内有 17 个 v_nop（WAR），`s_set_vgpr_msb` 从 32 增到 47。
3. sink/window 路径的 SGPR spill 从 6 增到 35，非 causal 路径从 2 增到 4。job 不走这些路径，以后要走之前先修。
4. 卡上 gate 必须覆盖：3 个 shape 的 o/lse SQNR ≥ 49 dB，加上 h16 大 logit 测试。C2、C3 保留了 defer，这项必须跑。
