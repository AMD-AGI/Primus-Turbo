# L22 / L23 compile-only 构建（GPU 2 / fa-g2，2026-09-27，零上卡时间）

冠军：`op-evolve/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op/current`（fwd round 4），
`base/` 为其逐字拷贝。各臂只改 `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`，diff 见 `L22.diff`、`L23.diff`、`L22_L23.diff`。

## 臂

| 臂 | 改动 |
|---|---|
| `L22/` | 删除 `_pv_gemm` 开头的手写 `rocdl.s_wait_dscnt(0)`（V transpose burst 全排空）。V 值是 SSA 可见的 ds_load 结果，由后端 SIInsertWaitcnts 按 WMMA 操作数分级插 `s_wait_dscnt`（K burst 早已如此）。 |
| `L23/` | 删除 `_drain_barrier` 中 `gpu.barrier()` 前后的 `rocdl.sched_barrier(0)` 对。 |
| `L22_L23/` | 两者合并。 |

说明：本 kernel 里屏障前的"手写 drain"只有 `tdm_ops.tensor_wait(0)`（TDM RAW 必需，不可删）；ISA 中屏障前没有额外的 dscnt drain
（`gpu.barrier` 的 workgroup fence 在此处不产生额外 wait）。所以 L22 在 gfx1250 上唯一可落地的形态是"把 PV 的 dscnt 排空推迟给后端"。
未使用 `rocdl.s_waitcnt`（gfx1250 会 raise）。

## 编译结果（`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`，fa-g2，无 flock，flydsl 0.3.4.1）

配置：prod = causal gqa4（prod/proxy/fast 三个 shape，kernel 按 gqa 特化，三者 ISA md5 相同）；nc = non-causal gqa4 prod；mha = causal gqa1。

| 臂 | 配置 | VGPR | SGPR | vSpill | sSpill | scratch | md5 |
|---|---|---|---|---|---|---|---|
| base | prod/proxy/fast | 456 | 100 | 0 | 0 | 0 | 7b078261 |
| base | nc | 450 | 107 | 0 | 4 | 0 | 02756c67 |
| base | mha | 454 | 105 | 0 | 0 | 0 | 26300e15 |
| L22 | prod/proxy/fast | 456 | 100 | 0 | 0 | 0 | 25a1615c |
| L22 | nc | 450 | 107 | 0 | 4 | 0 | 6af8bf81 |
| L22 | mha | 454 | 105 | 0 | 0 | 0 | d60736a0 |
| L23 | prod/proxy/fast | 456 | 100 | 0 | 0 | 0 | de7e104b |
| L23 | nc | 450 | 107 | 0 | 4 | 0 | 03dd1620 |
| L23 | mha | 454 | 105 | 0 | 0 | 0 | a9a7a452 |
| L22_L23 | prod/proxy/fast | 456 | 100 | 0 | 0 | 0 | 93614790 |
| L22_L23 | nc | 450 | 107 | 0 | 4 | 0 | c887bcaa |
| L22_L23 | mha | 454 | 105 | 0 | 0 | 0 | 5d7a567f |

所有臂 vSpill=0、scratch=0，无一被剔除。nc 的 sSpill=4 为冠军原有（spill 到 VGPR lane，无 scratch），不是回归。
全部日志：`build_<arm>.log`，ISA/IR：`isa/<arm>/<variant>_<shape>/`。

## ISA 差异（静态，仅作筛选，不作排名依据）

- L22：`s_wait_dscnt` 60 -> 88 条；PV 区从"0x1d 后立即 0x0"变成 0x1d/0x1c/0x15/0x14/0xd/0xc/0x5/0x4/0x1/0x0 分级等待，
  前几条 PV WMMA 不再等整个 V burst。VGPR 不变。
- L23：调度器把 5-7 条 VALU（ping-pong 指针 `v_dual_mov`、`s_cmp`）填进 `s_barrier_signal` 与 `s_barrier_wait` 之间（split barrier 的空窗）；
  `s_wait_tensorcnt 0x0` 仍紧贴 signal 之前。
- L22_L23：两者叠加，互不干扰。

## CPU 安全证明

两个杠杆都**不改任何索引表达式、LDS 槽映射、谓词或 TDM 描述符**（diff 只删 3 行 intrinsic 调用 + 注释），因此无新的越界面。
同步正确性用 `tools/barrier_check.py` 与 awk 对 15 份 ISA 逐屏障检查：

1. **WAR（下一 tile 的 TDM 覆盖本 tile 读过的 V/K 槽）**：每个 `s_barrier_signal` 前最近的 `s_wait_dscnt` 为 `0x0`，且其后到 signal 之间 0 条 ds_ 指令（全部 6 个屏障 x 4 臂 x 5 配置）。
   原因：每个 v_values 元素都被本 tile 的 PV WMMA 消费，后端的操作数等待在最后一条 WMMA 前达到 `dscnt 0x0`。
2. **RAW（TDM 落地后才读）**：每个 signal 前 1-2 行都是 `s_wait_tensorcnt 0x0`（L23 与 base 一致）。
3. **屏障窗口**：L23/L22_L23 中 signal..wait 之间搬入的指令全是 VALU/SALU，0 条 ds_/tensor_/buffer_/global_；signal 前 40 行内 0 条 ds_/tensor_ 指令。
   memory 顺序由 `fence syncscope("workgroup") release/acquire`（`gpu.barrier` 降级自带，见 `21_llvm_ir.ll`）保证，sched_barrier 本身无语义。
4. 算术、mask、softmax、GQA 映射未改，o/lse 应与冠军逐位相同（上卡时以 bitwise 对比冠军验证）。

## 上卡建议（本步未执行）

单 shape 进程；先 toy（b1 s256 hq2）起一次，再 prod；`--arm-path cand=<arm> --arm-path champ=<current>`，不带 beat；
每臂 >=3 个进程、轮换顺序、n=101；每进程后 `timeout 20 sudo -n dmesg | tail -20`。先验：KYLE gfx950 L22 +1.42%/+0.54%，L23 +0.96%——仅先验。
