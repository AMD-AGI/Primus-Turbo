# T7：gfx1250 预编译 ASM **前向** —— 顾虑已排除，而且比 T2 便宜得多

上一轮我把这个记成 T7 时附了一个前提：**名字里的 `pertokenBf16` 可能要求
per-token 量化 scale，核实前不算可用项。** 现在核实完了：

## 不需要任何量化 scale

host 侧的参数结构（`csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu`，
`static_assert(sizeof(KernelArgs) == 0x84)`，**132 B 与 ELF 的
`kernarg_segment_size` 精确相等**，所以这个文件就是它的启动器）只含：

```
d_addr q_addr k_addr v_addr lse_addr scalar q_seq_len
q_seqs q_ts q_hs q_bas gqa k_seqs k_hs k_bas opt lse
kv_seq_len q_head_num v_seqs v_hs v_bas d_seqs d_hs d_bas lse_hs sink_addr
```

**没有一个 scale 张量。** `pertokenBf16` 是命名约定，不是量化要求。**顾虑排除。**

## 而且它已经有 Python 入口——不需要像 T2 那样手写 launcher

`aiter/ops/mha.py:589` 的 `fmha_fwd_with_sink_asm(q, k, v, softmax_scale,
is_causal, return_lse, sink=None, out=None) -> (out, lse)`。

三件事恰好对上，这是它价值高的真正原因：

1. **布局是 bshd `[batch, seq, head, dim]`** —— 正是 Primus-Turbo 的布局，不需要转置。
   注释明确说只要求 `stride(-1) == 1`，非连续的 bshd 视图也接受。
2. **`is_causal` 是一个参数** —— 因果变体 `..._mask.co` 已存在。
3. **`return_lse` 返回 `[batch, q_head_num, q_seq_len]` fp32** ——
   **正是我们融合反向和 `asm_backward()` 需要的那个平铺 LSE 布局。**

也就是说：ASM 前向 + ASM 反向可以直接串成完整一趟，中间的 LSE 不需要任何重排。

## 资源画像和反向同一路数

```
LDS 327,680 B（整个 CU 的 320 KB）   VGPR 1024   wave32   scratch 0 B   kernarg 132 B
```

和反向一样是「每 CU 一个 workgroup、零溢出」的手写结构。

## 上卡前必须先确认的三件事（不要假设）

1. **`@compile_ops` 会在首次调用时 JIT 编译一个 C++ 模块**，需要可用的 hipcc 构建环境。
   这是它唯一比 T2 更脆的地方——T2 是纯 ctypes 直调 `.co`，不依赖构建。
2. **LSE 的底数约定未确认**。我们这条路上 turbo 和 aiter 都用自然对数
   （已在 `attention_fused_bwd_impl` 的注释里确认过），**但这个 ASM 前向写的是什么底数没查**。
   底数错了不会报错，只会让反向的数值全错。
3. **LSE 缓冲区总是被写**，即使 `return_lse=False`（注释明说内容未定义）。
   不要把 `return_lse=False` 当成省一次写。

## 为什么值得做

前向占我们 ~17% 的时间，且**一直没动过**——唯一一次尝试得出「0.9 ms 差距」，
但因为两种测法不一致（profiler 下只差 0.18 ms）而被判未确立，写进了「不要重做」清单。
这条路径绕开了那个僵局：它不是「再测一次 Triton 前向」，是换一个实现。
