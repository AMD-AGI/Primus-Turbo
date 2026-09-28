# L4 -- XCD-major (b, kv_head) grid remap on top of longest-first (compile-only)

基线：fwd round 4 champion（`job_context/op/current`，只读拷贝）。本步无卡时间：全部为 fa-g2 内
`COMPILE_ONLY=1 FLYDSL_DUMP_IR=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`，无 flock，无 HIP launch。

## 改动（唯一改动点 `_lpt_block_id` 的 (y, z) 分量）

`x = gx-1-rank` 不变（每个 WG 的 causal 工作量与 champion 逐 lin 相同 -> longest-first 负载均衡不变），
只重排一个 rank 步内 slot `rem ∈ [0, gy*gz)` 对应哪个 (kv_head, batch) group：`grp = _xcd_group(rem, rank, gyz)`。
模块常量 `XCD_REMAP` 选择：

| arm | XCD_REMAP | 映射 | prod 上每个 XCD 持有的 (b,kvh) group |
|---|---|---|---|
| `ctrl` | off | grp = rem（champion 语义） | kv_head k × 4 batch |
| `bmajor` | bmajor | gyz%8==0 时 grp = (rem%8)*(gyz/8) + rem//8（分支无关整数门控），否则 = rem | batch k//2 的 kv_head 4(k%2)..+3 |
| `spread` | spread | grp = (rem + rank) % gyz（反局部性对照） | 32 个 group 轮转，全部 XCD 都碰到 |

diff：`diffs/bmajor.diff`、`diffs/spread.diff`（各 50 行，只动 `_lpt_block_id` 尾部 + 新 helper）。
生成脚本：`tools/apply_l4.py <arm> <mode>`。

## 关键发现（CPU 模型，`xcd = lin % 8`）

**champion 的 longest-first 已经是 XCD-major。** gyz%8==0 时 `lin%8 == rem%8`，所以每个 (b,kvh) group 只落在
一个 XCD 上；prod 每个 XCD 4 个 group、proxy 1 个——都等于下界 ceil(gyz/8)。fast (gyz=2) 每个 group 分到 4 个
XCD，也是下界。所以 L4 在 spec 形状上无法再降低每个 XCD 的 K/V 足迹：`bmajor` 只把"一个 kv_head 跨 4 个 batch"
换成"一个 batch 的 4 个 kv_head"（Kyle "inner axis must differ per kernel" 的两种选择），足迹相同；
`spread` 把 prod 每 XCD 足迹从 4 升到 32 个 group——它是用来测 L2 局部性在 gfx1250 fwd 上到底值多少的对照。
前提假设 `xcd = lin % 8` 在 gfx1250 上未证（kyle-learnings: num_xcc=8；g43 bwd null）。
预期：bmajor null（|Δ| < 0.5%）；spread 若比 champion 慢，说明局部性有价值且 champion 已拿到；若 null，L4 在 gfx1250 fwd 上 ⚠ -> ❌。

## CPU 安全证明（门禁，`tools/bijection_check.py` -> `bijection_check.out`：ALL OK）

逐整数镜像 `_lpt_block_id` + `_xcd_group`，对每个网格枚举全部硬件 block id：
1. 重映射后 (x,y,z) 在界内且恰好命中一次（双射）；每个 (batch, seq, q_head) 行恰好由一个 WG 产出；
2. 每个 lin 的 x 与 champion 相同 -> 工作序列逐项相同，256-CU 模拟 makespan 相同（prod 1064 / proxy 69 / fast 18）；
3. 所有中间量 < 2^31，全部非负（C 风格 `//` `%` 与 Python 一致）。
覆盖：prod / proxy / fast、各自 gqa=1 版本、job 的全部 edge shape、sq≠skv、sq>skv(nc)、奇数 n_qt、gyz ∈ {1,2,4,6,8,12,16,24,32,40,48,128}，
causal 与 non-causal；另穷举 gx<20, gy<18, gz<7 的 3876 个 (网格, mode) 双射，0 失败。
源码里的 `XCD_REMAP` 常量被脚本回读断言与 arm 名一致。

## Compile-only 结果（`logs/build_all.out`，ISA 在 `isa/<arm>_<shape>_<variant>/`）

形状 prod/proxy/fast 是运行时参数，同一 binary；g1 = gqa 1 (hkv=hq)。

| arm | causal gqa4 (prod=proxy=fast) | nc gqa4 | causal gqa1 | nc gqa1 | spill / scratch |
|---|---|---|---|---|---|
| champion | 456 VGPR, 4028 inst, `7b0782612eef` | 450, sgpr-spill 4 | 454, 3818 | 450, sgpr-spill 3 | 0 / 0 |
| ctrl | **字节相同**（全部 7 个配置 md5 = champion） | = | = | = | 0 / 0 |
| bmajor | 456, 4064 (+36, 全部 SALU prologue), `f09d13c7beb4` | 450, sgpr-spill 4 | 454, 3853 | 450, sgpr-spill 3 | 0 / 0 |
| spread | 456, 4056 (+28, prologue; `% gyz` 运行时除法用 f32 倒数展开), `b5e302e87f48` | 450, sgpr-spill 2 | 454, 3835 | 450, sgpr-spill 3 | 0 / 0 |

sgpr-spill 是写入 VGPR lane（非 scratch），champion 本身就有。无 arm 被丢弃（所有 VGPR spill = 0，scratch = 0）。
新增指令全部在 `_lpt_block_id` 的 prologue 计算里，KV 循环体 opcode 计数不变（见 opcode delta，仅 s_* 与一次性 v_*_f32 除法）。

## 上卡建议（下一步，非本步）

1. 同进程 palindromic A/B：`cand=bmajor` / `cand=spread` vs champ，prod n=101，3 进程轮换顺序，不含 beat。
2. `ctrl` 与 champion 字节相同，可作 A/A 噪声底。
3. fast 上三者映射近似等价（gyz=2），fast 只作 sentinel。

## 文件

- `ctrl/ bmajor/ spread/` -- arm 目录（与 op/current 同布局）
- `tools/apply_l4.py` `tools/bijection_check.py` `tools/compile_fwd_isa.py` `tools/build_all.sh` `tools/isa_stats.sh`
- `bijection_check.out` `logs/` `isa/` `diffs/`
- `isa/` 只保留 `22_final_isa.s`（其余 IR dump 已删，362M -> 小）
