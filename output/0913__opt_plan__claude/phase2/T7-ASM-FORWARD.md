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

---

# 三个待确认项的离线结论

## (b) LSE 底数：**自然对数**，已确认

`op_tests/test_fmha_fwd_with_sink_asm.py` 里的参考实现（第 81 行）：

```python
lse = torch.log(denom) + max_total        # 自然对数
```

而第 237-243 行把它和内核返回的 lse **直接** `checkAllclose`（rtol/atol 1e-2），
中间没有任何底数转换。所以这个 ASM 前向写的是自然对数 LSE，
**与 turbo / aiter 在我们这条路上的约定一致**，可以直接喂给融合反向。

## 我们的形状被支持，且会选中因果那个内核

host 侧的检查只有：bf16、4 维、`stride(-1)==1`、`q_head_num % kv_head_num == 0`、
`head_dim ∈ {64,128}`、`v_head_dim == qk_head_dim`。**没有 seqlen 的整除限制。**
我们的 `b4 s8192 hq32 hkv8 d128`（gqa=4）全部满足。

内核选择（`get_heuristic_kernel_fmha_fwd_bf16`）只按 `dtype/hdim_q/hdim_v/mask` 匹配，
所以 `is_causal=True` → `bf16,128,128,mask=1` → `fmha_bf16_pertokenBf16_hd128_128x256_mask.co`。
（注意测试只覆盖了 `hk=4`（gqa=8），我们是 gqa=4——约束上没问题，但**没有被测过**。）

## (a) JIT 构建：风险比想象的小，但有一个**新的、更麻烦的阻塞**

`optCompilerConfig.json:1341` 显示这个模块只有**一个源文件**
（`py_itfs_cu/asm_fmha_fwd_with_sink.cu`）且 `-DENABLE_CK=0`，不牵扯 CK。
镜像里 `hipcc` 存在（`/opt/venv/bin/hipcc`），但模块**没有预编译**
（只有 `module_aiter_core.so` 是现成的）。所以首次调用会编一个文件，量级是秒不是分钟。

**但是**：尝试离线预编译时发现了一个会在明天同样发作的问题——

```
import aiter.ops.mha
  -> aiter/ops/triton/gluon/pa_decode_gluon.py:12
  -> ModuleNotFoundError: No module named 'jax'
```

`aiter.ops.mha` 的导入链会拉进一个 gluon 模块，而它无条件 `import jax`。
今天 harness 用的是 `aiter.ops.triton...`，走的是另一条链，所以没碰到。
**明天要用 T7 的 Python 入口，先得解决这个**：装 jax，或者绕开 `aiter.ops.mha`
直接走 ctypes 入口（`module_fmha_fwd_with_sink_asm` 是 `ffi_type="ctypes"`，
本来就不依赖 torch 扩展）。

另外，离线预编译这条路本身走不通：`aiter` 导入时要靠 `rocminfo` 探测架构，
`GPU_ARCHS` 只覆盖 `get_gfx_custom_op_core()`，**`get_gfx_runtime()` 没有覆盖开关**。
（可以用一个打印 `gfx1250` 的 `rocminfo` 桩绕过，已验证能过架构检测这一关。）
