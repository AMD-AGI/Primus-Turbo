# L8 屏障削减原型 -- 对抗性审查（2026-09-27，compile-only）

构建者报告正文未落盘（harness 拦截），见 op-evolve 调度方的 builder 输出。本节只记审查。

## 结论：ready-for-card

未找到正确性 bug，未改动 op/。

## 逐项核对（读 proto.diff + 源码 + 新编译 ISA）

- base/ 与 op-evolve job_context/op/current 逐文件相同；op/ 只改 fmha_fwd_prefill_a16w16_m32x8.py。
- 重新编译 op 默认（p22s）prod：md5 a12d1565，与 isa/p22s 一致；off == base == 23e8ba8d。
- LDS 读写：tile i 落在槽 i%(G*NG)。组顶 wait 之后才向组 g+NG-1（= 组 g-1 的槽）发 TDM；组 g-1 的全部 K/V 读在其末 tile 的 PV `s_wait_dscnt(0)` 处退休，之后才 signal -> WAR 关闭。组 g 的 TDM 在组 g-1 末 signal 前 `tensor_wait` -> RAW 关闭。奇数 tile 无同步，读的是同组已落地的槽。
- ISA 中每对 signal->wait 之间（沿 CFG 回边）0 条 ds_/tensor_/buffer_ 指令，32 个 PV WMMA；signal 前依次是 `s_wait_dscnt 0x0`、`s_wait_tensorcnt`。prod / noncausal / p23s win_sink 均如此。
- 平衡：n_iter>=1 有保证（mask_right 时 kv_len_wg>=1，kv_len==0 走 zero-fill，start_tile 被 clamp 到 n_tiles-1）。signal 数 = 1（prologue）+ 除最后一组外每组 1 次 = 组数 = wait 数；最后一组无论满不满都不 signal。条件只依赖 t、start_tile、n_tiles（均为 SGPR uniform）。
- 首尾：prologue 的 tile 0 无条件，j>=1 以 start_tile+j<n_tiles 判定，与主循环 t+PF+j<n_tiles 一致。mask_left 路径用 i_loc=t-start_tile；causal 路径 LLVM 把它折成 t（start_tile 为常量 0），ISA 中 `s_bitcmp1_b32 s84,0` 测的是 t 的奇偶，正确。
- Q：Q 位于组 NG-1，prologue 只加载组 0..NG-2；split 模式在 signal 前有 dscnt(0)。
- O 槽：(gl+NG-1)%NG。其上一占用组的读都在 gl 组顶 wait 之前的 signal 前退休；此后所有预取都被谓词关掉。gl=0 时落在 Q 组，同样安全。O manager 无跨 wave 交换，与冠军的"无需屏障"假设相同。
- 深 tensorcnt：ops_per_tile 取自 views 长度（d128=2、d192=3），只在被计数 tile 全部发出时用深值，否则 0，逻辑正确；仍依赖 TDM 按序完成（与 ASM 相同）。
- 算术、mask、GQA 映射、softmax/rescale 路径未改动，o/lse 应与冠军逐位相同。
- sync_model.py 重跑：2484 配置 0 失败。模型与代码的 LO/HI 顺序、谓词、槽映射一一对应。
- TDM pad 写入范围 <= n_rows*row_bytes = get_lds_size_in_byte()；K 17408、V 18432 本身是 1 KB 倍数，去掉 64 KB 地板不会越槽。

## 补充编译（review/isa/，compile-only，fa-repro）

| 路径 | base VGPR/SGPR/sSpill/vSpill/scratch | p22s | p23s |
|---|---|---|---|
| noncausal d128 | 432/107/2/0/0 | 425/107/6/0/0 | 425/107/15/0/0 |
| causal d192 | 510/107/8/0/0 | 500/107/9/0/0 | 508/107/23/0/0 |
| causal d256 | 512/99/0/234/704 | 512/100/0/200/628 | 编译期 assert：ring 2x3x67584 超 LDS |
| win_sink d128 | 455/-/4/0/0 | 449/-/1/0/0（builder） | 457/107/14/0/0 |

## 发现（非阻塞）

1. **p23s / p13s 在 qk_hdim=256 无法编译**（assert 显式失败，不会静默出错）。若 p23s 胜出，写死时必须让 NG 随 hdim 退回 2（或按 hdim 选配置），否则 d256 入口直接挂掉。p22s 三个 hdim 都能编译。
2. SGPR spill 在非 prod 路径上变多（noncausal 2->6/15，d192 8->9/23，p23s win_sink 14）。spill 到 VGPR lane，无 scratch；只影响非排名路径，p23s 偏多。
3. 报告中"代价：PV 前的 dscnt 变成硬性全排空"不成立：冠军的 _pv_gemm 本来就无条件 `s_wait_dscnt(0)`，钩子只是插在它后面（dscnt 条数 44->36 反而减少）。
4. d256 冠军本身就有 VGPR spill 234 / scratch 704，p22s 为 200 / 628，不是回归。
5. 上卡第一步仍按 builder 所列：单 shape 小 grid + 外部超时；正确性 shape 里补 d192（TDM ops_per_tile=3，唯一走 3 op/tile 的路径）。
