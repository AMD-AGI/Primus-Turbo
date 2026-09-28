# L24 -- 编译选项（llvm_options）扫描，compile-only 构建（2026-09-27，GPU 2 lab）

冠军：fwd round 4（`job_context/op/current`，只读复制）。每个 arm = 冠军树的完整拷贝，**内核源码唯一改动**是
`fmha_fwd_prefill_a16w16_m32x8.py` 两处 `_launch.compile_hints["llvm_options"]` 字典（bshd :1867、thd :1973）
增加/替换一个选项；保留原有 `amdgpu-expert-scheduling-mode: ENABLE_SCHED_MODE2` 与 `waves_per_eu=2`
（`nosm2` 除外，它就是把该项改为 False）。生成脚本 `tools/make_arms.py`，每个 arm 目录有 `L24_ARM.json`。

## FlyDSL 0.3.4.1 能调的后端选项（核实过）

- `compile_hints["llvm_options"]`：`flydsl/compiler/llvm_options.py` 以 context manager 设置进程内 LLVM cl::opt，
  编译结束恢复原值；只支持 bool/int/str。未知名字抛 `RuntimeError: Unknown LLVM option`（已测），所以“静默无效”
  只可能是选项对本内核无作用，不是名字写错。
- JIT cache key 含 compile_hints（`jit_function.py:1302-1311`）且含源码 hash，同进程 cand/champ 不会串 binary。
- 其它 hints：`waves_per_eu`、`maxnreg`、`fast_fp_math`、`unsafe_fp_math`。占用率类（L28）已关闭，未扫。
- `amdgpu-sched-strategy` 可取值（从 libFlyPythonCAPI 字符串表）：`max-ilp`、`max-memory-clause`、`iterative-ilp`、
  `iterative-minreg`、`coexec`。Kyle 的 gfx950 “max-memory-clause + post-misched” 对应 `mmc_nopm`。

## 编译方法

`tools/run.sh <arm> <cfg>`：在 fa-g2 内、不加 flock（不碰 GPU），
`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_L24_<arm>_<cfg>
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1`，`tools/compile_isa.py` 走 `_ensure_bshd_kernel` + `flyc.compile`，
CPU 占位张量。配置：prod(4,8192,32,8)/proxy(1,4096,32,8)/fast(1,1024,8,2) 为 causal gqa4；nc_g4 非因果 gqa4；
c_g1 / nc_g1 为 gqa1 因果/非因果。shape 整数是运行时参数，(causal,gqa) 决定 binary，所以 prod/proxy/fast
的 ISA hash 相同（已验证）。78 次编译结果 `isa/compile_rc.txt`，指标 `isa/stats.json`（`tools/stats.py`），
ISA `isa/<arm>/<cfg>/*/22_final_isa.s`，主循环直方图 `tools/isa_loops.py`。

## 结果（VGPR / vgpr_spill / sgpr_spill / scratch；prod 与 prod 主循环指令数）

| arm | 选项 | prod VGPR | 各配置 vspill | sspill (nc_g4/nc_g1) | scratch | prod 总指令 / s_wait | 与冠军 ISA | 结论 |
|---|---|---|---|---|---|---|---|---|
| base（对照） | 无改动 | 456 (nc 450, c_g1 454) | 0 | 4 / 3 | 0 | 4153 / 100 | 相同 | 对照 |
| **mmc** | sched-strategy=max-memory-clause | 456 | 0 | 4 / 3 | 0 | 4279 / 128 | 不同 | 上卡 |
| **nopm** | enable-post-misched=False | 456 | 0 | 4 / 3 | 0 | 4310 / 172（loop dscnt 10→28） | 不同 | 上卡 |
| **mmc_nopm** | 上两项合并（Kyle 组合） | 456 | 0 | 4 / 3 | 0 | 4454 / 200 | 不同 | 上卡 |
| **ilp** | sched-strategy=max-ilp | 462 | 0 | 2 / 3 | 0 | 4269 / 101（loop v_nop 16→20/38） | 不同 | 上卡 |
| **itilp** | sched-strategy=iterative-ilp | **439** (nc 432) | 0 | **0 / 0** | 0 | 4157 / 97（loop dscnt 10→3） | 不同 | 上卡，优先 |
| **nosm2** | expert-scheduling-mode=False | 456 | 0 | 4 / 3 | 0 | 4124 / 72（loop s_wait_alu 类 3→0） | 不同 | 上卡 |
| **nosink** | disable-machine-sink=True | 456 | 0 | 4 / 6 | 0 | 4149 / 99 | 仅微小差异 | 上卡（预期 null） |
| coexec | sched-strategy=coexec | 452 | 0 | **prod 1** | **prod 16 B** | -- | 不同 | **淘汰**：scratch>0 |
| clause32 | amdgpu-max-memory-clause=32 | 456 | 0 | 4 / 3 | 0 | 4153 | **字节相同** | 淘汰：no-op |
| clause4 | amdgpu-max-memory-clause=4 | 456 | 0 | 4 / 3 | 0 | 4153 | **字节相同** | 淘汰：no-op |
| relaxocc | schedule-relaxed-occupancy=True | 456 | 0 | 4 / 3 | 0 | 4153 | **字节相同** | 淘汰：no-op（wpe=2 固定） |
| postra_bu | misched-postra-direction=bottomup | -- | -- | -- | -- | -- | -- | **blocked**：enum cl::opt，FlyDSL str 设置器抛 `basic_string::_M_construct null not valid` |

注：
- 冠军本身在非因果配置就有 SGPR spill（spill 到 VGPR lane，scratch=0）；规则的硬门槛是 vgpr_spill>0 或
  scratch>0，所以 sspill 只作对比。`nosink` nc_g1 sspill 3→6（仍 scratch=0），`itilp` 全部配置 sspill=0。
- `amdgpu-max-memory-clause` 设置器确实生效（`set_llvm_option_int` 返回旧值 15），但本内核的全局访存是
  TDM `tensor_*`，SIFormMemoryClauses 没有可成 clause 的序列，所以 4/32 都与冠军逐字节相同。
  “max-memory-clause” 在 gfx1250 上唯一有意义的形式是调度策略（`mmc`），不是 clause 长度。
- 各 arm 的主循环 WMMA=64、v_exp=66、ds_load=64、TDM=2、barrier signal/wait 各 1 均与冠军相同；差别只在
  指令顺序、s_wait_dscnt 数、v_nop 数（静态指标不作排名依据，rule 8）。

## 安全证明（CPU，无下标改动）

1. 下标：各 arm 与冠军 `diff -r` 只差 llvm_options 字典的 1-2 行（`L24_ARM.json` 记录），无任何索引/地址表达式
   改动，因此冠军已有的越界证明原样适用，无需新的 bounds proof。
2. 访存集合不变：`tools/stats.py` 对每个 arm × 6 配置比较全部 `buffer_*/global_*/tensor_*/ds_*/s_barrier*/v_wmma*`
   助记符的多重集，**全部与冠军相等**（`stats.json` 的 `memops_equal_base=true`）——选项只重排指令和改变
   等待数，不增删任何访存或同步指令。等待计数由 LLVM SIInsertWaitcnts 按重排后的顺序重新推导。
3. 资源：保留的 7 个 arm 全部 vgpr_spill=0、scratch=0、LDS=327680 与冠军相同。
4. 残余风险是编译器本身的 bug（非本 lab 可证）。建议按 card-safety §2：每个 arm 先 `toy` shape 单独进程、
   `AMD_SERIALIZE_KERNEL=3` 跑一次正确性，再上 prod。

## 上卡建议（本步未占卡）

优先级：`itilp`（VGPR -17、sspill 清零、loop dscnt 10→3，结构上最不同）> `mmc_nopm`（gfx950 +0.6-0.7% 的原组合）
> `nopm` > `mmc` > `nosm2` > `ilp` > `nosink`（几乎同 ISA，预期 null，可作 A/A 噪声对照）。
每个 arm：benchmark.py `--arm-path cand=<arm> --arm-path champ=<op/current> --shapes prod --iters 101`，
不带 beat，≥3 进程轮换顺序，每进程后 dmesg 检查。
