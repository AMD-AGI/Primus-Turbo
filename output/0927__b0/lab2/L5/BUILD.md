# L5 因果对齐的 q-tile 起点：仅编译构建（GPU 2 / fa-g2，未占卡）

日期 2026-09-27。冠军版本是 fwd r4（`job_context/op/current`，只读）。全部工作都是在 fa-g2 里做 COMPILE_ONLY 编译，外加 CPU 证明，**没有 flock，没有 GPU 进程**。

## 结论（先看这里）

- **在全部三个计分形状上，L5 按构造就是 no-op。** fast/proxy/prod 都满足 `sq*gqa % 256 == 0`（16/64/128 个 tile 恰好整除），pad=0，所以 pad_w=0。CPU 证明也确认 L5 的行映射与冠军**逐项相同**（`row map identical=True`），KV 访问数 136/2080/8256 一个不变。上卡最好的结果只能是 null。
- L5 只在 ragged 的 causal 形状上省工作量：gqa=1 能省 1.5–5.2%，gqa=4 且 sq==skv 时省 0，gqa=4 且 sq≠skv 时最多省 1.6%。非 causal 时一点不省；**有限左窗口（win128）时反而更差**（+0.07%~+4.6%）。所以两个 arm 都只在 `mask_right && !mask_left` 时启用 L5。
- 两个 arm 均 spill=0、scratch=0，没有一个被淘汰。
- **新发现（要进 hint）**：L5 的 kernel 生成的 LLVM IR 是确定的（hash 恒为 `7132b9567c`），但 AMDGPU 后端产出的 ISA **呈双峰、不确定**。同一份源码、同一个 dump 路径、固定 PYTHONHASHSEED、关掉 ASLR，结果都一样：12 次编译里 8 次得到 `c7cfbf`（4246 条指令，sgpr 93，vgpr 454），4 次得到 `c8b8d1`（4120 条，sgpr 101，vgpr 456）。冠军 IR 在 8+8+9 次里全部得到同一个 ISA `7b0782`。所以 `rot` 上卡时，每个进程跑到的可能是两个变体里的任意一个，这是测量混杂，必须记下每个进程实际用的是哪个 ISA。

## Arms

| arm | 目录 | 做了什么 | 计分形状上的 ISA | VGPR | SGPR | spill (v/s) | scratch |
|---|---|---|---|---|---|---|---|
| rot | `rot/` | L5 常开（bshd 和 thd 都开，仅 causal 且无左窗口时） | 与冠军不同，双峰：`c7cfbf` / `c8b8d1` | 454 / 456 | 93 / 101 | 0/0 | 0 |
| gate | `gate/` | 同一个 L5 kernel，但 bshd 的 host 端只在 pad_w≠0 时选用它（kernel key 加 `"l5"`）；thd 仍常开 | **与冠军逐字节相同**（IR `6a5ed39f88`，ISA `7b0782`）；L5 路径与 rot 相同 | 456（L5 路径 454/456） | 100 | 0/0 | 0 |

对照目录（不是 arm）：`base/` 是冠军原样拷贝；`off/` 只接了管线、`Q_ORIGIN_ALIGN=False`，在 5 个配置上 IR 与冠军逐字节相同。

### 编译矩阵（`isa_table.txt`、`llvm_ir_hashes.txt`、`isa/*/22_final_isa.s`）

| 配置 | 冠军 | rot | gate | gate 的 L5 路径 |
|---|---|---|---|---|
| prod causal g4 | 456/0/0 `7b0782` | 454/0/0 `c7cfbf`（或 456 `c8b8d1`） | = 冠军 | = rot |
| proxy causal g4 | 同 prod（同一个 kernel） | 本次编译 456 `c8b8d1` | = 冠军 | 454 `c7cfbf` |
| fast causal g4 | 同 prod | 454 `c7cfbf` | = 冠军 | 456 `c8b8d1` |
| prod 非 causal g4 | 450/sgpr-spill 4/0 `02756c` | **= 冠军**（L5 关闭） | = 冠军 | = 冠军 |
| causal g1（b1 s4096 h32/32） | 454/0/0 `26300e` | 452 `984566`（此前也出现过 454 `61c466`） | = 冠军 | 454 `61c466` |
| causal g2（toy） | 456/0/0 `3d19ee` | 454 `985ec5` | = 冠军 | 454 `985ec5` |

prod/proxy/fast 编译出来是同一个 kernel，因为 seq 长度是运行时参数。表里同一列出现不同的 hash，就是上面说的后端双峰。冠军的非 causal kernel 本来就有 `sgpr_spill_count=4`（溢出到 VGPR lane，scratch=0）。这是冠军现状，L5 没碰它。

复现方法：`docker exec fa-g2 bash tools/build_all.sh <arm> [l5]`（不加 flock），然后 `bash tools/isa_table.sh`；双峰统计用 `tools/repeat.sh <arm> N [l5]`，结果在 `repeat_prod_causal.txt`。

## 实现（`apply_l5.py`，可以在冠军的新拷贝上重放，diff 见 `rot.diff` / `gate.diff`）

- 新增 `_q_row_origin`：`tot = gx*256`，`pad = max(tot - q_len*gqa, 0)`，`pad_w = ((pad//32) % 8) * 32`；wave 起始行 `r = bx*256 + w*32 - pad_w`，若 `r<0` 则 `r += tot`（回卷）。WG 有效行界 `lo = max(bx*256 - pad_w, 0)`，`hi = bx*256 + 255 - pad_w`。
- `_packed_tile_indices`、Q 装载（`QManager16bV2.load_q_to_vgpr_part1(packed_row0=)`）、O 存储（`OManager16bV3.store_o_to_vram(warp_base=)`）、LSE（经 `seq_idx`）都用同一个 `r`。kv 范围和 clean 切分的 4 处 `wg_min_seq`/`wg_max_seq` 改用 `lo`/`hi`。
- L5 关闭时按冠军原来的顺序发射 op，因此 IR 逐字节相同（off/gate 已验证）。L5 只接了 Q V2 + O V3 这条路径（有 assert）。
- 另外试过两种写法，都没有保留：`umin(u32 r, u32 r+tot)` 的无比较回卷，仍然双峰（4244/4119）；对 `r` 做 `readfirstlane`，双峰而且更差（4267/4264）。

## CPU 安全证明（`tools/l5_proof.py` → `tools/l5_proof.out`，PROOF_OK）

脚本把 kernel 与 manager 中所有被改动的整数表达式逐字转写成整数模型，冠军和 L5 各跑一遍。共 57806 个配置：gqa∈{1,2,4,8,16,32,64}；sq 取 1..399 全部，再加 511..8192 的边界值；skv 取 {sq, sq+7, sq+100, sq-5（仅非 causal）}；掩码取 causal / 非 causal / win128；布局取 bshd（gx 由 sq 决定）和 thd（gx 大于所需）。每个配置都检查以下六条，全部通过：

- P1：每个 `warp_row0` 都在 [0, gx*256) 内，且是 32 的倍数。所以 wave 不会跨越回卷点，Q loader 的 `gqa|32 或 32|gqa` 前提不变。
- P2：所有 (bx, wave, row) 覆盖的 packed 行恰好是 range(gx*256) 的一个排列。**每个有效 q 行（seq, head）恰好被一个 lane 行计算**，回卷的行一律满足 seq≥q_len。
- P3：Q 的 TDM 只读 `0 ≤ seq < q_len`（seq0≥0；有效行数为 0 时什么也不读）。
- P4：O（V3 的 `valid_rows>0` 门控，外加 last_valid 钳位）和 LSE 只写有效 (seq, head)，且每个恰好写一次。
- P5：**掩码不变**。对每个有效行，在 [start_tile, n_tiles) 上按左 / clean / 右三段各自的掩码之后留下来的 kv 集合，都等于参考带 `[max(0,s+off-wl), min(s+off+wr, kv_len-1)]`（非 causal 时为 `[0, kv_len-1]`），并且落在 `[0, kv_len_wg)` 以内，不会读到零填充的 K 行。冠军和 L5 都等于参考，因此两者相等。
- P6：统计 KV-tile 访问数，结论见上面第二条。

## 如果要上卡（建议，由 operator 决定）

1. 计分形状上 L5 最多是 null：gate 与冠军逐字节相同，不值得占卡；rot 只能测出"多出的索引算术 + 后端双峰"带来的成本，预期是 null 或小幅亏损。
2. 如果还是想测 rot：每个进程开 `FLYDSL_DUMP_IR`，或者事后反汇编该进程实际加载的 binary，记下它是 `c7cfbf` 还是 `c8b8d1`，按变体分开统计比值。
3. 现有的 harness 边缘形状（toy/short_q/gqa4_batch2/mha/unequal_seqlen）全部满足 pad=0，**覆盖不到 L5 路径**。要验证 L5 的正确性，得另加 ragged 形状，每个形状单独一个进程、先跑 toy。例如 b1 s200 hq2 hkv2（g1，pad_w 32）、b1 s200 hq2 hkv1（g2，pad_w 96）、b1 s3000 hq8 hkv8（g1，pad_w 64，访问数 -3.5%）。

## 建议的 hint（交给 operator，本 lab 不写 hint.md）

- L5 在本 job 上**按构造关闭**：三个计分形状都是 pad=0，行映射逐项相同，KV 访问数不变。它只对 ragged causal 且 gqa≤2 的形状有价值（-1.5~-5%），对 win128 有害。
- 注意 gfx1250 后端的双峰 ISA：同一份 LLVM IR 可能编出两套 ISA（L5 上 8/12 对 4/12，差 126 条指令），冠军 IR 则稳定。任何 arm 上卡前，都应先把 prod 编译重复 ≥8 次，确认 ISA 唯一；不唯一时要记录每个进程实际跑的是哪个变体。
