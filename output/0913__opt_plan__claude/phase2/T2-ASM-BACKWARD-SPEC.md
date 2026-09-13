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

## 调用契约

`tools/gfx1250/asm_bwd_abi.py` 从 ELF 里直接读出并能生成打包代码
（`--emit-packer` 给出 44 行 `struct.pack_into`）。

**这里有个坑：两种打包约定并存。**
- 主内核 `dqdkdv`：ELF 元数据里有逐字段偏移，**每个字段补齐到 16 字节**，共 704 B。
- `odo` / `dq_convert`：**ELF 里没有任何字段元数据**，host 用**紧凑打包**——
  `use_compact_fmha_bwd_kernel_args()`（`mha_bwd.cu:68`）对 gfx1250 精确返回 true。
  odo 的布局已从 `pack_fmha_bwd_odo_args()`（`mha_bwd.cu:95`）转录进工具，
  3 指针 + 11×u32 + 2 指针 = **84 B，与 `kernarg_segment_size` 精确吻合**，布局确证。

**唯一剩下的未知：`dq_convert` 的 208 B 布局。** 它的打包函数不在 `mha_bwd.cu` 里，
还没找到。这是明天上卡前要补的最后一块。

## 明天怎么测

用 `hipModuleLoad` + `hipModuleLaunchKernel` 直调这三个 `.co`，**不要 patch 装好的 aiter**
（会污染 21.684 ms 那个参考基线）。四张量 SQNR 门照常，dq 必须先清零。
