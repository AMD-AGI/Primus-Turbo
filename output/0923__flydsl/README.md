# 2026-09-23

## `hint.md` 的副本

`hint.md` 是 op-evolve 作业里**唯一由人写**的文件，框架在每轮开头应用它。它住在
`artifacts/<job>/job_context/`，而 `artifacts/` 在 op-evolve 仓库里是 gitignore 的——
也就是说这 17 条 hint 只存在于一份本地副本里。2026-09-22 的断电已经演示过这意味着什么。

所以在这里留一份。内容按轮次累积：

| hint | 一句话 |
|---|---|
| h1–h6 | 2026-09-21 写的：armAB 的正确性先于速度、边界安全靠显式谓词、离线筛先于上卡、rocprofv3 是死路、flat fast 不构成反证、spec 里的 5.8 TF/s 是陈数 |
| h7 | prod 是唯一排名依据，fast/proxy 是哨兵 |
| h8 | ~~瓶颈是 HBM 流量~~ **已被 h14/h16 推翻**，保留作记录 |
| h9 | g15 必须排在 BLOCK_KV 之后（后被 h16 整体关闭）|
| h10 | 记账修正 |
| h11 | 「要 deep round」不是合法条件；把大件切成 fast 尺寸 |
| h12 | round 8 只打 g04；朴素 4-wave 一个字节都省不下 |
| h13 | round 8 删掉的 barrier 正是 g04 需要的；先加回来否则静默竞态 |
| h14 | 寄存器阻塞被实测证伪；~200 VGPR 是分配失败不是代价 |
| h15 | **每轮自写脚本的规则**：用 cached_forward、加 --line-buffered、块间读 dmesg、不可替代的测量先做、从仓库复制 bitwise.py |
| h16 | **瓶颈是 issue roof 不是带宽**；g15/g18 关闭、g04 降级、round 10 打 g27 |

## 这一天最重要的两条

**挂卡的因果链定位了。** 三次挂卡全部落在一次 prod fp32 Tensile GEMM 的一步之内。
判别证据是 round 9 自己给的：同轮更重的 `d_s2`（4 臂 × 101 迭代 × 三形状）干净 rc=0，
因为它的脚本里没有 `bitwise.py`；而 `bitwise.py` 在 prod 上跑完约一分钟后，
下一个重派发把卡推倒。**Tensile 先损伤卡状态，下一个重派发压垮它。**

挂卡还有可判别的前兆：升级型故障带**4 GB 以下的截断地址**和 `PERMISSION_FAULTS:0x5, RW:0x1`，
随后出现 `copy_context_work_handler [amdgpu] hogged CPU`，计数器跨捕获递增。见 h15。

**坐标系换了。** issue 效率 η = issued FLOP/(peak × time)：我们 0.682，aiter 0.725——
**已经是它调度质量的 94.0%**。1.464× 的差距里 **1.408× 是「多做 40% 矩阵乘」（7 GEMM 对 5，
由我们自己的 ISA 实测证实），只有 1.040× 是调度**。7-GEMM 形态的天花板是 0.710× beat。
剩下的距离不是欠债，是合同代价——5-GEMM 融合被 200 次逐位 determinism 门结构性封死。
