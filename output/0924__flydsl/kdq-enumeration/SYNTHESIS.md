## 1. 有没有越界访问？

**k_dq / k_dq_sp：没有。** 三份独立审计 + 六个对抗 reviewer 全部独立枚举、独立复现，结论一致且逐字节相同：

- `fast` 走 **k_dq_sp**（`impl.py:208-211`，`_NSP_Q_CAP=8` 在 `impl.py:98`：wgs_q = 128/2048/16384 → nsp_q = 8/1/1），proxy 与 prod 走 k_dq。我本人复算确认。
- 写只有两处：`kernels.py:949`（split，fp32 → dqp）和 `kernels.py:953`（非 split，bf16 → dq_o）。全索引空间枚举（grid × 32 lane × qh_[4] × dtile[8] × si[8]，fast 8,388,608 次 / proxy 16,777,216 次 / prod 134,217,728 次，无抽样）证明写集合是**到真实张量的双射**——每个元素恰写一次，slack = 0 B。
- 读（Q/dO `:700-703`、K/V `:705-710`、LSE/DEL `:721-722`）同样 slack = 0。`:876-877` 的 `jj=min(ii+1,n-1)` 与 `:928` 的 `_pf` 两处手写钳位都成立。
- 覆盖面超出任务要求：`ut/common.py:13-25` 全部 9 个 shape、causal 与 non-causal 两种模式，全部 in bounds。

**但同一次 backward launch 里确实有一处按地址越界的读**：`kernels.py:538`（split）与 `:545`（非 split）的无条件前导预取 `_ldqd(_qt0, fx.Int32(0))`，query pair 索引可以超出 `nqt2`。经 audit-3 的 lens 1 修正（两个调用点都把 `gh` 钉成字面量 0，`_ldqd` 在 `:271` 算 `qh = hkv*G + gh`，所以可达 q head ≤ Hq−G，不是 Hq−1）：

| shape | Q/dO 超出末尾 | LSE/DEL |
|---|---|---|
| fast | 982,272 B（`_qt0` 到 46，`nqt2`=32）| **不越界**（audit-3 原报的 1,920 B 是 qh=Hq−1 的反事实）|
| proxy / prod | 261,376 B | 不越界 |

它今天不 fault，因为 `_dkdv_impl` 在 `kernels.py:200-206` 给这些 buffer 的是**真实 extent**。`kernels.py:499-504` 的 `jj = ii + 1` 是第二处未钳位预取（audit-3 漏了，lens 1 补上）。

**任务书的前提有两处错，都可自查：**
1. 引的 dmesg 是 **round 17** 的。round 18 自己的记录是 `rounds/018/1-opt/raw/dmesg_pagefault_screen.txt:11` / `opt.md:496`：`TCP (0x8), PERMISSION_FAULTS: 0x3, **RW: 0x0**` —— 是一次**读**。方向反过来后，越界读那一侧（k_dkdv `:538/:545`）才是形状对得上的嫌疑点，而写侧（本次审计的主目标）被排除得更干净。
2. `opt.md:504` 原文「卡未 wedge，未需断电」。round 18 没有烧掉一次断电，fault 出现在 `verdict: pass` 写完之后的老 screen.py 收尾路径上。

**诚实边界：枚举的 vs 论证的。** 枚举的是 FlyDSL 源码层的整数索引代数（完整 cross-product，仅 K/V 的 kv0 用"仿射且单调 ⇒ 极值定界"论证内部）。**没有**枚举的：LDS 之外的 scratch/寄存器溢出写（同样是 TCP 流量，`:741-748` 的 32 寄存器预取元组使溢出可信，不编译看不到）、host 侧分配器页映射、round 18 进程的实际 runtime 参数。

## 2. 七个假 num_records —— 附一条 ISA 层的新事实

我去翻了出货 ISA，确认了此前三份报告都标为"无法解释"的 128 倍差异，它**不是**表示层噪声：

```
k_dq_0/21_final_isa.s:165   s_mov_b32 s6, 0x800000    ; = (1<<30) >> 7,  s[4:7] 的 word2
k_dq_0/21_final_isa.s:196   s_mov_b32 s7, 0           ; word3 (flags) = 0
k_dq_0/21_final_isa.s:34    s_mov_b32 s14, 0x200000   ; = (1<<28) >> 7
k_dq_sp_0/21_final_isa.s    s_lshl_b32 s2, s4, 9  →  s_ashr_i32 s3, s2, 31  →  s_lshr_b64 s[2:3], s[2:3], 7
```

runtime 路径显式做了 `>> 7`，常量路径被折叠成同一件事。**结论：gfx1250 上 V# 的 num_records 字段以 128 字节为单位**，有效硬件上限 = 源码字节数。于是三份审计的倍率表在硬件层是对的：

| 行 | 张量 | 描述符(B) | 真实 extent fast / proxy / prod (B) | 倍率 | 判定 |
|---|---|---|---|---|---|
| `:675` | Q | 1<<30 | 2,097,152 / 33,554,432 / 268,435,456 | 512× / 32× / 4× | SAFETY |
| `:676` | K | 1<<30 | 524,288 / 8,388,608 / 67,108,864 | 2048× / 128× / 16× | SAFETY |
| `:677` | V | 1<<30 | 同 K | 2048× / 128× / 16× | SAFETY |
| `:678` | dO | 1<<30 | 同 Q | 512× / 32× / 4× | SAFETY |
| `:679` | LSE | 1<<28 | 32,768 / 524,288 / 4,194,304 | 8192× / 512× / 64× | SAFETY |
| `:680` | DEL | 1<<28 | 同 LSE | 8192× / 512× / 64× | SAFETY |
| `:687` | dQ | 1<<30 | — / 33,554,432 / 268,435,456 | — / 32× / 4× | SAFETY |
| `:685` | dqp | nsp·B·Sq·Hq·D·4 | 33,554,432 (fast) | **1.000×** | 唯一真实 |

**七个全部偏大，没有一个偏小 —— 不存在活的静默丢写。** prod 上 dQ 之后的无保护窗口是 805,306,368 B；proxy 是 1,040,187,392 B。枚举证明没有任何索引伸进去。

`>> 7` 还带来一个此前无人记录的约束：**字节数不是 128 的倍数就会被向下截断，末尾那段会被真的钳掉。** `_dkdv_impl` 的 `nl_b = B*Hq*Sq*4`（`:202`）之所以安全，只是因为 impl.py 为 `k_delta_bshd` 加的 `n_rows % ROWS_DELTA(32) == 0` 断言顺带保证了它 —— 一个不相关的断言在替它兜底。

关于 reviewer 提的 `is_rdna_arch("gfx1250") == False`（`flydsl/runtime/device.py:82-99` 只匹配 gfx10/gfx11/gfx120，而 `:102-112` 的 `get_warp_size` 用 `gfx12` 前缀返回 32 —— 同一份"single source of truth"对这块卡给出两个矛盾归属），我确认属实，且 ISA 里 word3 确实是 0（不是 `universal.py:245-248` 算的 0x27000）。但**不能**由此推出"范围检查不存在"：num_records 字段在出货 ISA 里是被认真计算的活字段，且 round 2 的行为证据（换成真实 extent 后，原本 fault 的 32-deep arm 不再 fault）是直接的经验证据。位级语义仍未确立，这条留作未验证项。

## 3. 对 wedge 的优先级：**降低**，但重心要挪位置

一次性 fix 已上线，且本次审计证明 k_dq 侧**没有**可被假描述符放行的杂散索引。所以"修 k_dq 的七个描述符"不是防 wedge 的动作，是还技术债。

真正的发现是**不对称**：现在唯一挡着一条活越界读的，是 `kernels.py:200-206` 里 k_dkdv 的真实 extent。所以优先级应该是——**不要动 k_dkdv 的描述符**，并给 `:538/:545`（以及 `:499-504`）补上和 k_dq `:876-877/:928` 同款的钳位。

**正确的 k_dq 描述符修法必须同时满足四条：**
1. `k_dq` 在 `kernels.py:963-964` 把 `B_` 和 `nsp` 硬编码成 `fx.Int32(1)`，签名 `:959-962` 根本没有 `B_`。直接把 `1<<30` 换成 `B_*Sq*Hq*(D*2)` 会在 prod 上得到 67,108,864 而非 268,435,456 —— **实测 4 个 batch 里 3 个被整块钳掉，75% 的 dQ 静默为未初始化值**（proxy 因 b=1 侥幸无恙，所以 proxy gate 抓不到）。必须同时改 `k_dq` 签名、`launch_dq`（`:978-986`）和 `impl.py:214-217` 把 `b` 传进去。
2. 字节数必须是 128 的倍数（ISA `>>7` 截断）。
3. `:685` 要用 Int64 算（见下）。
4. k_dq 的 ISA 会变 —— `:943-948` 的注释明说地址代数的**顺序**都会移动 9 条指令并打破 byte-identical-prod gate，改描述符必然重新 baseline。

## 4. 另外值得单独开一轮的活缺陷

**(a) `impl.py:217/:223` 的 floor 与 `kernels.py:667-668` 的 ceil 不一致 —— 三份审计一致，且是唯一一条形状能对上 RW=0x1 的路径。**
launcher 传 `nblk = sq // BLOCK_Q`（下取整），kernel 内算 `bid = ceil(Sq/64) - 1 - blockIdx.y`（上取整）。`impl.py:115` 只断言 `sq % 32 == 0`，**没有** `% 64`。我实测：

- `b=1 sq=4128 hq=32` 和 `b=4 sq=8224 hq=32` → nsp_q=1，走 k_dq，经 `:687` 的假 1 GiB 描述符，**发射**（不是丢弃）262,144 B 的越界写，同时 query tile 0 永不派发（dQ 前 64 行保持 `torch.empty` 垃圾）。
- 小的不对齐 shape（sq=1056）走 split 路径，越界落在 dqp 内**别的 split 的 slice** 上 —— 静默算错而非 fault。

9 个 shape 的 sq 碰巧全被 64 整除，所以今天够不到。有 reviewer 扫了 258,048 组合法组合，**所有**越界组合都是 `sq % 64 == 32`，此外零越界 —— 这是这个索引空间里 k_dq 的唯一缺陷。

**(b) `kernels.py:685` 的 int32 溢出。** `nsp * B_ * Sq * Hq * fx.Int32(D*4)` 全程 int32。按 prod 维度实测：nsp=4 → 2,147,483,648 回绕成负；nsp=8/16 → 回绕成 **0**，num_records=0，所有 dQ 写被硬件丢弃且不报错。今天不可达（prod 取 nsp_q=1），但 round 18 当时正在扫 nsp=16（`rounds/018/_scratch/screen_cur.out:32`），`_NSP_Q_CAP` 或 2048 阈值一动就可能踩到。

**(c) `kernels.py:499-504`** qloop_full 的 carried prefetch `jj = ii + 1` 无钳位，是 k_dkdv 的第二处未保护预取（audit-3 只报了 prologue）。

**(d) flydsl 的 gfx1250 归类矛盾**（`device.py:82-99` vs `:102-112`），不是本仓库的 bug，但它决定了本卡所有 buffer 描述符的 flags 字，值得上游一张 issue。

**(e) round 18 的 RW=0x0 读 fault 仍未被本次审计解释。** k_dq/k_dq_sp 的读全部 in bounds；k_dkdv 的越界读被真实描述符钳住是死读。round 18 自己把它归给老 screen.py 收尾路径，且明确标注「未定，不得当作 arm 的缺陷」。不要把它当成已结案。

## 5. 最便宜的下一步 —— 三条全部零 GPU

按性价比排序，都可以在同一次提交里做完：

1. **`impl.py:115` 把 `sq % 32 == 0` 收紧为 `sq % _k.BLOCK_Q == 0`**（或让 launcher 传 ceil）。一行，零 GPU，消掉唯一一条"假描述符 + 发射的越界写"的可达路径，且不影响任何现有 shape（9 个全部已满足），**ISA byte-identical**，不需要重新 baseline。这是全表最高性价比的一条。
2. **`kernels.py:538/:545`（和 `:499-504`）补 `min(_qt0, nqt2-1)` 钳位**，用 k_dq `:876-877/:928` 的现成惯用法。它只会把一个**死值**的地址改小，不改变任何活值，所以数值上必然不变。它关掉的是本次审计找到的唯一一条活越界读 —— 方向恰好对上 round 18 的 RW=0x0。
3. **`kernels.py:685` 改用 Int64 算**，消掉 nsp 扫描时的静默归零。

**暂时不要做**的是"给 k_dq 七个描述符填真实 extent"：它必须先改 `k_dq` 签名与 `launch_dq` 把 `b` 传进来（否则 prod 上 75% 的 dQ 静默作废），而且必然改动 ISA、打破 byte-identical-prod gate，所以它属于一轮独立工作，不该和上面三条混在一起。

如果最终还是要上卡验证，最小实验是：**单进程、单 shape、只跑 `fast` 的正确性 gate（不是 benchmark），约 10 s**，对比钳位前后的 SQNR。但因为 (2) 在数学上不可能改变任何活值，更划算的做法是直接合入，由下一轮的常规三 shape gate 顺带覆盖。

---
本次综合所用的 CPU 脚本（含描述符编码、naive-fix 陷阱量化、int32 溢出表、sq%64 越界量化）：`/tmp/kdq_synth/verify.py`。前序枚举脚本：`/tmp/kdq_bounds_7f31/enum_kdq_writes.py`、`/tmp/lens1_kdq/reaudit_kdq.py`、`/tmp/lens2_kdq/audit.py`、`/tmp/attnaudit/audit.py`、`/tmp/lens2/audit2.py` 与 `/tmp/lens2/sweep2.py`。**不要**使用 `/tmp/kdq_oob_enum*.py` —— 有审计者报告该路径在运行期间被另一进程覆写。全程未触碰 GPU。