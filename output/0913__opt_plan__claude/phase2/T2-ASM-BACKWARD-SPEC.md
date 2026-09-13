# T2：aiter 预编译 gfx1250 ASM 反向 —— 完整调用契约（离线得出，未上卡）

TODO 里 T2 写的是「一小时」。这份文档把那一小时里本来要花在考古上的部分**提前做完了**，
明天那一小时应该只用来测量。全部结论来自 ELF 元数据和 C++ 源码，**没有用 GPU**，
所以所有性能推断都是结构性的，不是实测。

## 它是真的 gfx1250 手写汇编

```
Machine: EM_AMDGPU   Flags: 0x549, gfx1250, xnack, sramecc
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
```

目录里 6 个 `.co`，我们要的因果变体是 `bwd_hd128_bf16_causal_br_a32_pssk.co`。
CSV（`fmha_bwd_dqdkdv.csv`，表头 `dtype,hdim_q,hdim_v,mask,atomic32,pssk,pddv,mode,bf16_cvt,ts_qo,ts,...`）
里对应行是 `bf16,128,128,2,1,0,0,0,3,32,128` —— hdim 128/128、mask=2（因果，bottom-right）、
atomic32=1、**tile 32×128**。

## 它和我们的 Triton 冠军不在一个量级

| | 手写 ASM | Triton 冠军 |
|---|---:|---:|
| LDS | **327,680 B（整个 CU 的 320 KB）** | 65,536 B |
| VGPR | 1024 | 1024 |
| **scratch spill** | **0 B** | 113 store / 245 load |
| `v_wmma_f32_16x16x32_bf16` | **864** | 448 |
| `ds_load_tr16_b128` | 592 | 32 |
| `s_set_vgpr_msb` | 1,302 | 1,834 |
| **存储体切换 / WMMA** | **1.5** | **4.1** |
| tile | 32×128 | 32×256 |
| 每 workgroup 线程 | 128（4 wave） | 128（4 wave） |

**同样的 tile、同样的 wave 数，但每次发射干 1.9 倍的矩阵活、零溢出、单位矩阵指令的
存储体切换开销只有 1/2.7。** 320 KB LDS 意味着**每个 CU 只能驻留一个 workgroup**——
这是一个 Triton 根本无法表达的结构（它这里只分配了 64 KB）。

这解释了为什么值得测：它不是「同一个内核的另一组参数」，是另一种结构。
**但这只是结构性推断，实测可能完全不同——限频态下 320 KB LDS 的单 workgroup 设计
也可能因为占用率过低而输。**

## 它不是 drop-in：一次反向要发三个内核

`csrc/cpp_itfs/mha_bwd.cu` 显示的流水线：

1. **`bwd_hd128_odo_bf16`** —— 预处理 `delta = rowsum(dO*O)`
   grid `(ceil(Sq/ts_odo), nhead_q, batch)`，block **128**
2. **`bwd_hd128_bf16_causal_br_a32_pssk`** —— 主体，**用 514 条 `buffer_atomic_add_f32`
   往 fp32 的 dq_acc 里累加**
   grid `(ceil(Sk/128), nhead_q, batch)`，因果时 `gdx = (gdx+1)/2`，block **128**
   我们的形状：`(32, 32, 4)` = 4,096 个 workgroup，256 个 CU 上跑 16 轮
3. **`bwd_hd128_dq_convert_bf16`** —— fp32 dq_acc → bf16 dq

**所以 `dq.zero_()` 不是可选项**（PROGRESS.md 里已有这条，现在有了直接证据：
514 条原子加）。而且需要一块 **fp32 的 dq_acc 暂存**，布局 C++ 里写明：
`(1, batch, nhead_q, seqlen_q, hdim_q)` fp32。

## 调用契约（已逐字节确证，并更正了我上一版的一处错误）

`tools/gfx1250/asm_bwd_abi.py` 从 ELF 里直接读出并能生成打包代码。

### 主内核 `dqdkdv`：头文件与预编译二进制**精确吻合**

ELF 元数据给出 44 个字段的逐字段偏移，每字段补齐 16 字节，共 704 B。
`csrc/include/mha_bwd.h` 的 `fmha_bwd_dqdkdv_args` 算出来也是 **704 B / 44 字段，完全一致**。

> **更正**：本文档上一版说「两种打包约定并存，`odo`/`dq_convert` 用紧凑打包」，
> 并且我一度得出「头文件已与二进制漂移」的结论。**后者是错的**——那是我的解析器
> 把一个跨行声明（`unsigned int\n    max_seqlen_dq;`）拆成了两个字段，
> 多算了 4 字节。头文件没有漂移。
> 名字差异（`seqlen_q` vs `seq_len_q`、`ptr_qseq` vs `ptr_seqstart_q`）纯粹是命名，
> **共同字段的顺序完全一致**。

### `dq_convert`：ELF 没有字段元数据，但反汇编把它定死了

头文件的 `fmha_bwd_post_kernel_args` 算出 192 B，而 ELF 声明 208 B，差一个 16 字节槽位。
无法从元数据得知那个槽位是什么。**但这个问题在实践中不存在**——反汇编显示
内核实际读取的 kernarg 偏移只有：

```
0x0 0x10 0x20 0x30 0x40 0x50 0x60 0x70 0x80 0x90
   (2 条 s_load_b64 读两个指针，8 条 s_load_b32 读八个 u32)
```

**它从不读 0xa0 / 0xb0**——那两个是 `ptr_qseq` / `ptr_qseq_padded`，varlen/group 模式才用。
所以批模式下：**分配 208 字节清零，按头文件填前 0xa0 即可**，
而头文件对内核真正读的每一个字段都是对的。

### `odo`：这条验证路走不通，如实记录

`odo` 的反汇编里 **`s_load` 条数为 0** —— 它走 gfx1250 的 **kernarg 预加载到 SGPR**，
参数不经过 `s_load`，所以没法用「内核读了哪些偏移」来验证布局。
它的紧凑布局依据是两点：`pack_fmha_bwd_odo_args()`（`mha_bwd.cu:95`）的转录，
以及 3 指针 + 11×u32 + 2 指针 = **84 B 与 `kernarg_segment_size` 精确相等**。
两者独立吻合，但**不如 `dq_convert` 那样被反汇编直接确证**。
（`use_compact_fmha_bwd_kernel_args()` 在 `mha_bwd.cu:68` 对 gfx1250 返回 true，
但它只作用于 `odo`；`dq_convert` 走 `sizeof(post_args)` 的补齐结构体，不是紧凑打包。）

## 明天怎么测

用 `hipModuleLoad` + `hipModuleLaunchKernel` 直调这三个 `.co`，**不要 patch 装好的 aiter**
（会污染 21.684 ms 那个参考基线）。四张量 SQNR 门照常，dq_acc 必须是 fp32 且先清零。
**odo 的布局是三者中唯一没有被反汇编确证的，如果结果是数值垃圾，先怀疑它。**

---

# Launcher 已写好：`tools/gfx1250/asm_bwd_launcher.py`

`hipModuleLoad` / `hipModuleLaunchKernel` 的 ctypes 封装 + 三个内核的参数打包。
**一行都没在 GPU 上跑过**（卡在写它之前就已经挂了），所以明天第一次运行是 bring-up，不是测量。

离线自测（`--selftest`，不需要 GPU）检查打包尺寸、偏移、字节序和 `.co` 是否存在：

```
  odo      packed 84 B (kernarg 84)      ok
  dqdkdv   packed 704 B (kernarg 704), last field ends at 692  ok
  scalar round-trips as float: 0.08838835  ok
  post     packed 208 B (kernarg 208), bytes past 0x9f all zero: ok
```

**自测立刻抓到了我自己的一个错误**：第一版 `DQDKDV_FIELDS` 是手抄的，
**悄悄漏了最后 7 个字段，其中包括 `mask_x` / `mask_y`**。
自测之所以能抓到，只因为它打印「最后一个字段结束于哪里」——580 而不是 692。
现在这张表**从 ELF 生成，不再手抄**。这正是 `asm_bwd_abi.py` 存在的理由，而我第一次没用它。

## 最后一个未知量也解决了：`mask_x` / `mask_y` 传 0

host 只在 `if (mt == 3)`（generic window）时给这两个字段赋值。我们的 CSV 行是
`mask=2`（因果 bottom-right），走不到那个分支，而 `fmha_bwd_dqdkdv_args`
在 `mha_bwd.cu:621` 是**未初始化声明**——也就是说 aiter 自己在这条路径上传的就是栈垃圾。
内核显然不读它们，否则 aiter 早就坏了。因果行为在 `_causal_br_` 这个内核变体本身里。
**传 0 安全。**

## 剩余风险清单（按可疑程度排序）

1. **`odo` 的紧凑布局** —— 三者中唯一没被反汇编确证的（kernarg 预加载，`s_load` 为 0）。
2. **字节步长** —— 所有 stride 字段是**字节**不是元素（bf16 ×2、fp32 lse/delta ×4）。
   传成元素步长不会报错，会静默读错内存。
3. **`dq_acc` 必须是 fp32 且清零** —— 主内核有 514 条 `buffer_atomic_add_f32`。
4. **grid** —— `(ceil(Sk/128), nhead_q, batch)`，因果时 `gdx=(gdx+1)/2`；
   我们的形状是 `(32, 32, 4)`。block 恒为 128。
