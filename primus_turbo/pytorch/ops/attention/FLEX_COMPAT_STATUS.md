# Primus-Turbo `flex_attention` 兼容层状态

本文档对应 `primus_turbo.pytorch.ops.attention.flex_attention` 的当前能力边界，
用于回答“哪些 torch flex 变体现在能直连 Turbo 高性能内核、哪些会报错、后续如何补”。

入口：

```python
from primus_turbo.pytorch.ops.attention import flex_attention, flex_attention_varlen, create_block_mask
```

签名与 `torch.nn.attention.flex_attention.flex_attention` 对齐，并在其后追加若干
**可选的 Turbo 扩展参数**（超集，默认关闭 `None`/`0.0`，torch 风格调用零改动、仍可 drop-in 替换 torch）：
`flex_attention(query, key, value, score_mod=None, block_mask=None, scale=None,
enable_gqa=False, return_lse=False, kernel_options=None, alibi_slopes=None, softcap=None,
dropout_p=0.0, sink=None, bias=None)`。
内部完成 `bhsd([B,H,S,D]) <-> bshd([B,S,H,D])` 布局互转，并把可识别的变体映射到
`flash_attn_func`（gfx950 上自动选 FlyDSL/AITER）。

### Turbo 扩展参数（超集，默认关闭）

- **`alibi_slopes: Optional[torch.Tensor] = None`（现在就能直连 Turbo）**：显式的每头 ALiBi
  斜率，要求 1D、`length == Hq`（query 头数）、fp32（否则报清晰 `ValueError`），device 自动
  对齐到 q。给了就**跳过 `_detect_alibi_slopes` 自动识别**，直接把该 slopes 透传到
  `flash_attn_func(alibi_slopes=...)`——因此**绕开识别器的保守限制、当前即生效**，可与
  causal / 滑窗 mask 组合（ALiBi 常配 causal）。**冲突处理**：同时给了显式 `alibi_slopes`
  和**非平凡（非恒等）**`score_mod` 视为歧义，抛 `ValueError`（明确二选一）；`score_mod` 为
  `None`/恒等时允许与显式 slopes 共存。
- **`softcap: Optional[float] = None`（接口就位、当前门控报错）**：logits 软上限
  （`cap*tanh(score/cap)`，Gemma2/Grok）。`None` 或 `0/0.0` = 禁用（no-op，不影响现有路径）；
  正数当前**抛 `NotImplementedError`**（接口已就位，但受阻于本 build 的 aiter dense 前/反向
  kernel 缺 softcap 形参，详见下方“softcap 现状”；上游 kernel 支持后本参数即生效）。显式
  `softcap` 与从 `score_mod` 识别到的 soft-cap **统一走同一处报错**（不重复、不冲突），
  **绝不静默丢弃 cap**。
- **`dropout_p: float = 0.0`（现在就能直连 Turbo）**：注意力 dropout 概率，校验 `0 <= p < 1`
  （`0` = 禁用，drop-in 默认），直接透传 `flash_attn_func(dropout_p=...)`。与 flash-attn /
  torch `scaled_dot_product_attention` 一致：`p>0` 即生效（训练态语义，eval 请传 `0`）；可与
  `return_lse` 并存，兼容层恒以 `deterministic=False` 分发，故无 dropout/确定性冲突需拒绝。
  经 GPU 实测：`p=0` 与不传逐字节一致（零回归），`p=0.1` 前向+反向可正常跑通（输出/梯度有限、形状正确）。
- **`sink: Optional[torch.Tensor] = None`（现在就能直连 Turbo）**：注意力 sink（每个 query 头一个
  可学习 logit），要求 1D、`length == Hq`、fp32（否则清晰 `ValueError`），device 自动对齐到 q；
  sink kernel 路径另要求 `head_dim_qk == head_dim_v` 且 head_dim 为 2 的幂（后端约束，见
  `attention_aiter_impl.AttnFwdAiterBackend`）。直接透传 `flash_attn_func(sink=...)`。经 GPU 实测：
  透传结果与直接调 `flash_attn_func(sink=...)` **逐字节一致**（identity），`sink=None` 零回归。
- **`bias: Optional[torch.Tensor] = None`（现在就能直连 Turbo）**：pre-softmax logits 的加性 bias，
  直接透传 `flash_attn_func(bias=...)`。**关键约束（实测取证，见下方“bias 现状”）**：aiter dense
  kernel 只接受**单个 `[Sq, Skv]`** bias（在 batch/head 间共享）、且 dtype 必须与 q 一致（fp16/bf16；
  **fp32 会 NaN**、4D/每头 bias 被 kernel 拒绝报 `bias shape should be [sq, sk]`）。入口接受
  `[Sq,Skv]` 或前导 singleton 的可广播形状（`[1,Sq,Skv]`/`[1,1,Sq,Skv]`），自动 cast 到 q 的 dtype
  并对齐 device；真正的每头/每样本 bias 抛 `ValueError`。经 GPU 实测前向+反向数值正确。

## 已支持（可直接分发到 Turbo）

- **full attention**（`block_mask is None` 或全 True）→ `causal=False`
- **causal**（`q >= kv`）→ `causal=True`
- **sliding-window causal**（`(q >= kv) & (q-kv <= W)`）→ `causal=True, window_size=(W,0)`；**窗口大于
  探测网格（W>512，如 W=1024/2048/4096）在长序列 S>512 上也已支持**：在最后一个 query 行二分定位 W 的
  True→False 翻转点，再对若干抽样行逐位精确校验为标准左窗因果后分类（见下方“大窗口探测”）
- **varlen / document packing（显式入口 `flex_attention_varlen`）**：THD `[total,H,D]` 打包 + `cu_seqlens`
  直连 `flash_attn_varlen_func`（详见下方“varlen / document packing 现状”）
- **document-causal（dense 入口自动识别）**：`block_mask` 被**精确重建校验**为 `same_doc(q,kv) & (q>=kv)`
  时，dense `flex_attention` 自动改走 varlen（块对角 `cu_seqlens`），而非会跨文档的 dense causal（详见下方）
- **GQA/MQA**：依赖 Turbo 原生能力，要求 `Hq % Hkv == 0`，且 `Hq != Hkv` 时调用侧显式传
  `enable_gqa=True`（与 torch flex 语义一致）
- **score_mod=None**
- **ALiBi（自动识别）**：仅当 `score_mod` 被严格验证为 `score + slope[h] * (kv-q)`（对 score
  加性且系数为 1、随 (kv-q) 线性、平移不变、与 batch 无关）时，映射为 `alibi_slopes`；否则不识别
- **ALiBi（显式参数）**：通过 `alibi_slopes=` 直接传入每头斜率（1D/fp32/len==Hq），**跳过自动
  识别、直连 `flash_attn_func`**，用于绕开识别器的保守限制；与自动识别在相同 slopes 下等价
- **return_lse**：透传 Turbo 的 `softmax_lse`（返回 `(out, lse)`）
- **scale**：默认 `1/sqrt(D)`，与 torch flex 对齐；显式 scale 透传给后端
- **dropout（显式参数）**：`dropout_p` 透传 `flash_attn_func(dropout_p=...)`，`0` 禁用（drop-in 默认）；
  GPU 实测 `p=0` 零回归、`p=0.1` 前向+反向通过
- **attention sink（显式参数）**：`sink` 透传 `flash_attn_func(sink=...)`（1D/fp32/len==Hq，
  head_dim 约束见上）；GPU 实测与直接后端调用逐字节一致、`sink=None` 零回归
- **加性 bias（显式参数）**：`bias` 透传 `flash_attn_func(bias=...)`，需 `[Sq,Skv]`/q-dtype（见上）；
  GPU 实测前向+反向数值正确（bf16/fp16、full/causal 均 rel-L2 < 2e-2）

### 性能路由层（`choose_backend`，默认全走 Turbo）

识别成功（mask 分类 + score_mod 映射）后，分发前会经过一层轻量的性能路由
`choose_backend(mask_cfg, *, shape, dtype, has_alibi, has_softcap=False, has_dropout=False,
has_sink=False, has_bias=False) -> {"turbo","custom"}`：

- **默认返回 `"turbo"`**，即所有受支持变体仍直连 `flash_attn_func`，行为与之前逐字节一致。
- 提供注册表 API：`register_backend_override(matcher, backend)` / `clear_backend_overrides()`。
  `matcher(ctx)->bool` 可读取路由上下文（`kind/causal/window_size/shape/dtype/has_alibi/has_softcap/
  has_dropout/has_sink/has_bias/mask_cfg`），命中则强制该 backend（按注册顺序、首个命中生效）。用于让
  tuner 把特定 shape/kind 引到 `_dispatch_custom` 钩子，而无需改动分类器。
- `custom` 分支共用 `_dispatch_custom(...)`（当前仍是抛 `NotImplementedError` 的 stub），
  既是“任意 score_mod”的入口，也是“被显式路由为 custom 的受支持变体”的入口。

### 关键前提 / 已知约束

- **ALiBi 符号约定（build 相关）**：本兼容层假设 Turbo `alibi_slopes`（正斜率）等价于
  flex 的 `+slope*(kv-q)`。该符号在 `rocm/primus:v26.5`（primus_turbo 0.3.2.dev48,
  commit 6ccf00ff）上经实测为 `alibi_sign=+1`（见 `bench/bench_results_ext2.md`：
  `plus_err=1.6e-3` 匹配、`minus_err=1.32` 不匹配）。**换 build 需重新校验该符号**，
  否则可能静默产生错误结果。
- **mask 探测上限（含大窗口定位）**：分类器在 `min(S,512)` 网格上探测 `block_mask.mask_mod`。当探测角
  看似 full-causal 但远角不可见（`mask_mod(S-1,0)=False`）时，会在**最后一个 query 行**二分搜索窗口边界
  `W = 最大 d 使 mask_mod(S-1, S-1-d) 可见`，再对若干抽样行（边界行 + 均匀分布的约 16 行）逐位精确校验其
  为标准左窗因果 `(q>=kv)&(q-kv<=W)`；**仅在校验完全通过时**分类为 `sliding_window_causal, (W,0)`，否则仍抛
  `NotImplementedError`（绝不误判）。full-causal（远角仍可见）仍归 causal。
- **分类/识别缓存（性能，行为不变）**：`block_mask` 的分类与 `score_mod` 的 ALiBi/soft-cap 识别结果按
  **对象身份**（`weakref.WeakKeyDictionary`）缓存——同一 `block_mask`/`score_mod` 被多层/多步复用时跳过
  重探测（最多 512×512 网格 + 检测器），消除每次调用固定 ~1.6–4.1ms 的探测开销（GPU 端到端 wrapper 前向
  开销由 ~1.5–2.3ms 降至 ~0.03–0.27ms，其余为保留的 bhsd↔bshd transpose）。**纯提速**：不同对象重新分类、
  结果与未缓存逐位一致；无法弱引用的对象自动跳过缓存；`clear_classification_cache()` 可复位。
- **不识别即报错**：任何无法映射到上述固定内核的 `score_mod`/`mask_mod` 都会抛
  `NotImplementedError`（自定义快路径 `_dispatch_custom` 目前是 stub），绝不静默降级，
  从而保证“正确性优先”。

## 不支持 / 待补（按优先级）

### P0/P1（路径 A：模式识别映射，小改动高收益）

| 特性 | 原因 | 路径 | 前置 | 难度 |
|---|---|---|---|---|
| ALiBi（超出保守识别器的等价写法） | 当前自动识别只接受严格线性形式，复杂等价写法会被判为不识别 | ✅ 已提供**显式 `alibi_slopes` 参数入口**绕开识别器；如需增强自动识别器另计 | 补齐等价形式与符号回归单测 | 已缓解 |
| softcap（`logits_soft_cap`） | **受阻于 kernel 层**：本 build 的 aiter dense 前/反向 kernel 无 softcap 形参（详见下方“softcap 现状”） | A（需上游 aiter 支持） | 上游 aiter 的 dense `mha_fwd`/`fmha_v3_fwd`/`mha_bwd` 暴露并实现 softcap（fwd+bwd） | 受阻 |
| attention sink | ✅ **已支持**：入口新增显式 `sink` 参数（1D/fp32/len==Hq，head_dim 约束），透传 `flash_attn_func(sink=...)`；GPU 实测与直接后端调用逐字节一致 | A | — | 已完成 |
| dropout | ✅ **已支持**：入口新增显式 `dropout_p`（`0<=p<1`）透传后端；GPU 实测 `p=0` 零回归、`p=0.1` 前向+反向通过 | A | — | 已完成 |
| 加性 bias / relative position bias | ✅ **已支持**：经 AITER dense，需**形状 `[Sq,Skv]`、dtype 同 q（bf16/fp16）**（fp32→NaN、4D 每头被 kernel 拒绝）；入口自动适配形状/精度后透传（详见下方“bias 现状”） | A | — | 已完成 |

### softcap 现状（P0，已调查：识别到但受阻于 kernel 层）

softcap（logits 软上限，Gemma2/Grok）：`score = cap * tanh(score / cap)`（cap>0；0/None=禁用）。

**Python 侧已就位**：

- 自动识别：`_detect_softcap(score_mod)` 严格识别纯 softcap（仅依赖 score、与 b/h/q/kv 无关、
  `f(0)=0`、尾部饱和到 cap、全网格拟合 `cap*tanh(s/cap)` 且奇对称）。
- 显式参数：入口新增 `softcap: Optional[float] = None`，`None`/`0` 禁用，正数请求 softcap。
- **单一启用点**：显式 `softcap>0` 与识别到的 soft-cap 汇入 `effective_softcap`，一并把
  `has_softcap=True` 传入 `choose_backend`，随后在 `flex_attention` 内的**同一处** `if
  effective_softcap > 0.0:` 显式抛 `NotImplementedError`（**绝不静默丢弃 cap**）。该处标注了
  `# TODO(softcap)`：上游 aiter dense fwd+bwd 支持后，删除此拦截并把 `effective_softcap`
  thread 到 `flash_attn_func(softcap=...)` 即可**一行切换启用**。

**为何必须显式识别（修复一处静默错误风险）**：ALiBi 识别器 `_detect_alibi_slopes` 只在
`score=0` 处探测，而 `cap*tanh(0)=0`，因此它会把一个纯 softcap 误判为“零斜率 ALiBi（no-op）”，
进而落到 `alibi_slopes=None` → 直连 Turbo **忽略 cap**，产生静默错误结果。对典型 cap（20–50）
该误判确会发生（`|f(1)-1| < 5e-3` 容差内）。`_detect_softcap` 先行拦截，把静默错误变为显式报错。
实测：`bench/softcap_flex_validation.py` 中 active cap=1.0 时 `rel_l2(no-cap vs cap)=0.576`
（丢弃 cap 会严重错误）；cap=30 在 ~N(0,1) logits 上 gap=0.0048（大 cap 近似 no-op，但仍须遵从）。

**受阻点（kernel 层，实测 aiter 签名，rocm/primus:v26.5）**：

- dense 前向 `aiter.ops.mha._flash_attn_forward`（兼容层经 `attention_aiter_forward_impl` →
  `AttnFwdAiterBackend` 调用）签名**无** `logits_soft_cap`/`softcap`：
  `(q,k,v,dropout_p,softmax_scale,causal,window_size_left,window_size_right,sink_size,bias,
  alibi_slopes,q_descale,k_descale,v_descale,return_lse,return_softmax,how_v3_bf16_cvt=1,
  cu_seqlens_q=None,cu_seqlens_kv=None,sink_ptr=None,out=None)`。底层 `mha_fwd`/`fmha_v3_fwd`
  运行时类型提示同样无 softcap 形参。
- dense 反向 `aiter.ops.mha._flash_attn_backward` 为 `torch_compile_guard` 包装
  （`(*args, **kwargs)` 透传到 `torch.ops.aiter.<name>`），当前调用未传 softcap；底层
  `mha_bwd`/`fmha_v3_bwd` 亦为无 softcap 形参的包装。**softcap 会改变梯度，反向缺形参 =
  无法正确训练。**
- varlen 前向 `_flash_attn_varlen_forward` **有** `logits_soft_cap: float = 0.0`（且有
  `ret = ret and logits_soft_cap == 0.0` 门控），但 varlen 反向 `_flash_attn_varlen_backward`
  **无** softcap 形参 → 即便走 varlen 也无法训练；且兼容层入口目前仅 dense。
- FlyDSL：本 build 的安装包**不含** `attention_flydsl_impl`（`ModuleNotFoundError`），不可用。
- aiter Triton dense 前向 `aiter.ops.triton.attention.mha._flash_attn_forward`：内部对
  softcap **硬编码 `softcap=0.0`** 且 Python 包装未暴露该形参；Triton 反向
  `flash_attn_onekernel_backward` 无 softcap 形参。

**结论**：在不改 C 扩展/不重编译 aiter 的前提下（本任务约束），dense 前+反向都无法接出
softcap。故 softcap 在兼容层标记为“**识别到但受阻于 kernel 层**”。

**可选推进路径（需上游改动，超出本任务范围）**：

1. 上游 aiter 的 dense `mha_fwd`/`fmha_v3_fwd`/`mha_bwd`（及对应 CK/汇编 kernel）增加并实现
   `logits_soft_cap`（fwd+bwd 一致），随后本兼容层把 `softcap` 从 `flash_attn_func` 一路 thread
   到 `attention_aiter_forward/backward_impl` 并解除拦截即可（Python 侧改动已经预演清楚）。
2. 暴露 aiter Triton dense 路径的内部 `softcap`（当前硬编码 0.0），但需同时补 Triton 反向的
   softcap 支持，且 Triton 路径当前仅在 sink 分支启用。
3. 用 Triton epilogue 近似（自写 `cap*tanh` 前/反向）——工作量与风险高，非最小改动。

### bias 现状（P0，已调查并修复：形状/精度问题，非 kernel 死路）

加性 bias（作用于 pre-softmax logits：`score = q·kᵀ/√d + bias`）。之前报告的 “NaN” 经取证
**不是 kernel 死路，而是形状/精度用错**。在 `rocm/primus:v26.5`（gfx950/MI355X）上用小 shape
（B=2,H=2,S=64,D=64）逐一试 `flash_attn_func(q,k,v, bias=...)`，与 fp32 手工加性 bias 参考
（`softmax(qk/√d + bias) @ v`）对比，结论如下：

| bias dtype / 形状 | 现象 |
|---|---|
| bf16 `[B,H,Sq,Skv]` / `[1,H,Sq,Skv]` / `[1,1,Sq,Skv]` / `[B,1,Sq,Skv]`（4D） | `RuntimeError: bias shape should be [sq, sk]`（kernel 只收 2D） |
| fp32 `[B,H,Sq,Skv]`（4D） | 同上 `RuntimeError` |
| **fp32 `[Sq,Skv]`（2D）** | **输出 NaN**（这就是此前“NaN”的真因：2D 但用了 fp32） |
| **bf16 `[Sq,Skv]`（2D）** | ✅ **正确**：前向 rel-L2 = 2.1e-3（< 2e-2）；反向 dQ/dK/dV/dBias 均有限、rel-L2 ≈ 2.5e-3 |

**根因**：aiter dense（`aiter.ops.mha._flash_attn_forward` → 底层 `mha_fwd`/`fmha_v3_fwd`）的 bias
形参**只接受单个 `[Sq, Skv]`** 矩阵（在 batch/head 间共享）、且 dtype 必须与 q 一致（fp16/bf16）。
底层 `mha_fwd(..., bias)` 与 `mha_bwd(..., dbias, bias)` 均有 bias/dbias 形参（实测 aiter 运行时
type hints 确认），故**前向+反向都支持**。

**修复（最小改动，仅在 flex 入口侧适配，不动 `flash_attn_interface.py`/`attention_aiter_impl.py`）**：
入口新增显式 `bias` 参数，`_validate_and_adapt_bias` 把用户传入的 bias 适配为 kernel 期望的
`[Sq,Skv]`（接受 `[Sq,Skv]` 或前导 singleton 的 `[1,Sq,Skv]`/`[1,1,Sq,Skv]`；真正的每头/每样本
bias 抛清晰 `ValueError`），并 cast 到 q 的 dtype、对齐 device，再透传 `flash_attn_func(bias=...)`。

**验证（经 flex 入口，端到端 fwd+bwd，见 `bench/_investigate_bias.py` 与 `bench/_run_taskB.py`）**：
bf16 与 fp16 × full 与 causal 共 4 组，前向 rel-L2 ∈ [2.6e-4, 2.3e-3]、dQ rel-L2 ∈ [3.2e-4, 2.7e-3]
（均 < 2e-2），且 gap(vs no-bias) ≈ 0.41–0.49（确认 bias 实质生效、非静默 no-op）。

**已知限制**：仅支持在 batch/head 间**共享的单个 `[Sq,Skv]`** bias（相对位置 bias / 共享加性 mask
的常见形态）；每头/每样本 bias 属任意 score_mod，需 codegen 路径（P3）。

### varlen / document packing 现状（P2，已支持）

**（主交付）显式 varlen 入口 `flex_attention_varlen`**：

```python
from primus_turbo.pytorch.ops.attention import flex_attention_varlen
out = flex_attention_varlen(
    query, key, value,               # THD 打包 [total_tokens, H, D]（与 Turbo varlen 一致，无需 transpose）
    cu_seqlens_q, cu_seqlens_k,       # int32 [num_seqs+1] 前缀和，位于 q 的 device
    max_seqlen_q, max_seqlen_k,       # 各自最长段长度（正 int）
    *, causal=False, window_size=(-1, -1), scale=None,
    alibi_slopes=None, dropout_p=0.0, sink=None, softcap=None, return_lse=False,
)
```

- **直连后端**：薄封装，直接映射到 `flash_attn_varlen_func`（THD 进 THD 出，无布局互转）。文档内因果用
  `causal=True`（标准块对角 + 段内 causal，要求逐段 `q_len == k_len`）；`window_size=(W,0)` 为逐段左窗。
- **cu_seqlens 校验（`_validate_cu_seqlens`）**：int32、1D、首元素 0、单调非递减、末元素 == `total`
  （q 用 `query.shape[0]`、k 用 `key.shape[0]`）、`len(cu_q)==len(cu_k)`、与 q 同 device、
  `max_seqlen >= 最长段`；`causal=True` 另要求 `cu_seqlens_q == cu_seqlens_k`（bottom-right 对齐，段长
  不一致会静默错位）。任何不合法 → 清晰 `ValueError`。
- **复用 dense 校验器**：`dropout_p`（`0<=p<1`）、`sink`（1D/fp32/len==Hq，head_dim 约束）、`alibi_slopes`
  （1D/fp32/len==Hq）与 dense 入口同一套校验；GQA/MQA 原生支持（`Hq % Hkv == 0`，无需额外 flag）。
- **不支持项报错（风格与 dense 一致）**：无 `score_mod` 形参（任意 score_mod 走不到这里）；`softcap>0`
  统一 `NotImplementedError`（varlen 反向 kernel 亦缺 softcap 形参，见上文 softcap 现状），绝不静默丢弃 cap。
- **跨 build 兼容**：`sink` 仅在调用方实际提供时才透传（本 build 较旧的 varlen kernel 无 `sink` 形参；默认
  `sink=None` 不传即 no-op，对新旧后端都可用）；`bias` 本入口不暴露，交后端默认。
- **GPU 实测（rocm/primus:v26.5, gfx950, total=512, docs=[128,128,256], H=8, D=128）**：causal/full/SWA(W=64)/
  GQA(Hq8/Hkv2)/ALiBi 前向 rel-L2 ∈ [2.4e-4, 2.3e-3]，causal 反向 dQ/dK/dV ≈ 2.5e-3~2.9e-3（均 < 2e-2）；
  `return_lse` 返回 `(out, lse)`（lse `[H, total]`）；`softcap>0`/非法 cu_seqlens/causal 段长不一致均正确报错。

**（次交付）dense 入口自动识别 document mask**：

- `flex_attention` 分类器在 `min(S,512)` 网格上探测 `block_mask.mask_mod`；当模式落在 causal 之内但既非
  纯 causal 也非单窗 causal 时，`_detect_document_causal_segments` 会尝试把它识别为 `same_doc(q,kv) & (q>=kv)`：
  从次对角线读出文档边界 → 还原每段长度 → **重建完整块对角因果 mask 并逐位精确比对**，完全一致才判定为
  `document_causal`（`_classify_block_mask` 返回 `{"kind":"document_causal","doc_seglens":[...]}`）。
- 命中后，dense 入口把 bhsd `[B,H,S,D]` 打包为 THD、按 `doc_seglens`（跨 batch 复制）构造 `cu_seqlens`、以
  `causal=True` 走 `flash_attn_varlen_func`，再解包回 bhsd（`_dispatch_document_varlen`）。**绝不用会跨文档的
  dense causal**。ALiBi（显式或识别）/dropout/sink 一并透传；`bias` 与 `return_lse` 在此路径显式报错
  （packed `[Sq,Skv]` bias 无法对齐、packed LSE 与 `[B,H,S]` 不对齐——需 LSE/bias 请改用显式 `flex_attention_varlen`）。
- **正确性优先，绝不误判**：仅在 `S <= 512`（探测未截断）且**重建逐位相等**时才路由；`S>512`、含窗、非方阵、
  任何空洞/偏差都会返回 `None` → 回退到原有 `NotImplementedError`（保持现状，绝不静默出错）。
- **GPU 实测**：dense 入口带 document `block_mask`，B=1 / B=2（cu 复制）/ 含显式 ALiBi 三组，rel-L2 ≈ 1.6e-3~2.0e-3
  （均 < 2e-2），与 masked fp32 参考一致。

**已知限制**：document 识别要求 `S <= _MASK_PROBE_LIMIT(512)`（更长序列请用显式 `flex_attention_varlen`）；
dense document 路径不支持 `bias`/`return_lse`（改用显式 varlen 入口）；`softcap>0` 全线受阻（见 softcap 现状）。

### P2（路径 A：中等工作量）

| 特性 | 原因 | 路径 | 前置 | 难度 |
|---|---|---|---|---|
| varlen / document packing | ✅ **已支持**：新增显式入口 `flex_attention_varlen`（THD 打包 + `cu_seqlens` 直连 `flash_attn_varlen_func`）；GPU 实测 causal/full/SWA/GQA/ALiBi 前向 rel-L2≈1.6e-3~2.3e-3、反向 dQ/dK/dV≈2.5e-3~2.9e-3（均 < 2e-2） | A | — | 已完成 |
| document masking | ✅ **已支持**：dense `flex_attention` 精确识别 `same_doc(q,kv) & (q>=kv)` 块对角因果 → 还原每段长度 → 构造 `cu_seqlens` → 改走 varlen；GPU 实测 B=1/B=2/含 ALiBi 均 rel-L2≈1.6e-3~2.0e-3 | A | 依赖 varlen 包装 | 已完成 |
| prefixLM（部分） | 现分类器仅支持 full/causal/单窗 causal/文档块对角因果 | A | 补充可判定模板与后端映射 | 中 |
| 更多 head_dim / dtype 覆盖 | 受后端约束与 dtype guard 限制（当前仅 fp16/bf16） | A | 后端能力验证（FlyDSL 仅 D∈{64,128}） | 中 |

### P3（路径 B：通用 codegen，高难）

| 特性 | 原因 | 路径 | 前置 | 难度 |
|---|---|---|---|---|
| 任意 score_mod | 需把运行时函数编译为高性能 kernel | B（codegen + 自动反向） | IR 设计、算子模板、autograd 方案 | 高 |
| 任意 mask_mod / 通用块稀疏 | 需要通用稀疏布局与调度 | B | mask IR + 稀疏计划器 + kernel 族 | 高 |
| 任意 score_mod + mask_mod 组合 | 组合爆炸，需统一 codegen 管线 | B | 前两项完成后再做组合优化 | 很高 |

`flex_attention` 内已预留 `_dispatch_custom(...)` 钩子作为路径 B 的接入点，目前仅抛
`NotImplementedError`（不实现真实 kernel）。

### P4（暂不规划）

| 特性 | 原因 | 路径 | 前置 | 难度 |
|---|---|---|---|---|
| FP8 + 任意 mod | 训练稳定性与量化标定复杂 | B（远期） | P3 成熟后再评估 | 很高 |
| paged attention | 需要 KV paging 数据结构与调度体系 | 独立路线（非 A/B） | 缓存管理与服务侧协议 | 很高 |

## 测试

- 纯逻辑单测（CPU 即可）：`tests/pytorch/ops/test_flex_attention_dispatch_logic.py`（共 **186 例**，95 原有
  + 44 varlen + 15 document + 32 大窗口/缓存；零回归）。varlen 部分覆盖 `_validate_cu_seqlens`（int32/1D/首 0/单调/末元素/段数一致/
  device/max_seqlen>=最长段/causal 段长一致 的正反例）、`_validate_qkv_varlen`（3D/dtype/head 整除/head_dim）、
  `_validate_window_size`、`_validate_max_seqlen`，及 `flex_attention_varlen` 端到端（mock 后端）：THD 直通不 transpose、
  causal/window/scale/dropout/alibi/sink 透传、softcap>0 报错、非法 cu 分发前报错、`return_lse` 返回 tuple、GQA、
  非因果交叉注意力。document 部分覆盖 `_detect_document_causal_segments`（多文档识别、单 causal/SWA/截断/非方阵/带洞
  的拒绝）、`_classify_block_mask` 返回 `document_causal`、以及 dense 入口端到端路由到 varlen（cu_seqlens 正确、B=2 复制、
  ALiBi 透传、bias/return_lse/超 512 的报错）。大窗口/缓存部分覆盖 `_locate_left_window`（W∈{256,512,1024,2048,4096}
  在 S=8192 上的识别、full-causal/非平移不变/带洞/非方阵 的拒绝）、`_classify_block_mask` 对大窗口的分类（含 S=8192、
  window≥S 归 causal、非标准长 mask 仍报错），以及分类/识别缓存（同对象命中返回同一对象、不同对象/形状重算、
  非弱引用对象跳过缓存、`_cached_detect_alibi/softcap` 命中计数、`clear_classification_cache` 复位、行为与未缓存一致）。
- 旧的原始 95 例：
  覆盖 full/causal/SWA/随机/带状/head 依赖/batch 依赖 的分类、ALiBi 识别器的正反例、
  `_detect_softcap` 的正反例（含 cap=20/30/50 识别、identity/线性/常量/ALiBi/硬 clamp/
  alibi+softcap 组合的拒绝、以及“softcap 不被误判为零斜率 ALiBi”的回归护栏），
  `choose_backend`/注册表（默认 turbo、override 命中 custom、clear 复位、首个命中优先、
  ctx 字段含 `has_dropout/has_sink/has_bias`、参数校验、matcher 异常包装、override 可按
  dropout/sink 命中），以及**显式扩展参数**：`alibi_slopes` 校验器
  （1D/长度/fp32/非张量的正反例）、`softcap` 归一化器（None/0 禁用、正数保留、负数/NaN 报错）、
  `_validate_dropout_p`（0/合法值、>=1/负数/NaN/非数报错）、`_validate_explicit_sink`
  （1D/len==Hq/fp32/head_dim 相等且 2 的幂 的正反例）、`_validate_and_adapt_bias`
  （`[Sq,Skv]`/前导 singleton 广播/矩形形状 通过，每头 4D/每样本 3D/末两维不符/非张量/非浮点 报错、
  dtype 适配到 q），`_is_identity_score_mod` 探测，及端到端分派（mock 掉 `flash_attn_func` 在 CPU 上跑）：
  显式 slopes 被透传且跳过自动识别、显式与自动识别 slopes 等价、与 causal 组合、非平凡 score_mod
  冲突→`ValueError`、恒等 score_mod 允许、非法长度→`ValueError`、显式 `softcap>0`→
  `NotImplementedError`（且未触达后端）、`softcap=0/None` 无影响、`dropout_p` 默认/正数透传与越界报错、
  `sink` 透传与非法形状报错、`bias` 透传/适配与每头 bias 报错、以及“无扩展参数即与原路径一致（含
  dropout_p=0/sink=None/bias=None）”。
- GPU varlen / document 验证（容器内）：`bench/flex_attention_varlen_validation.py`（经
  `bench/_run_varlen_all.py` 先把 workspace `flex_attention.py` 覆盖到已安装包、再从 /tmp 跑纯逻辑单测 +
  本脚本）。覆盖 `flex_attention_varlen` 的 causal/full/SWA/GQA/ALiBi 前向 + causal 反向 dQ/dK/dV、`return_lse`、
  softcap/非法 cu/段长不一致 报错；以及 dense 入口 document 路由（B=1/B=2/ALiBi）对齐 masked fp32 参考。
  实测全部 rel-L2 < 2e-2（详见上方“varlen / document packing 现状”）。
  注意：容器已安装的是较旧 build，其包 `__init__.py` 缺 `sparse_mla_interface`、varlen kernel 缺 `sink` 形参，
  故 runner **只覆盖 `flex_attention.py`**（不覆盖 `__init__.py`），入口对 `sink` 采用“提供才透传”的兼容策略。
- GPU 入口开销 + 大窗口验证（容器内）：`bench/bench_flex_entry_overhead.py`（三方对比 entry/direct/torch flex）
  + `bench/_bench_classify_cache.py`（分类冷/热缓存微基准）。实测：**分类缓存**把冷（重探测）1.6–4.1ms 降到热
  （命中）~0.3–0.4µs（~5000–12000×）；端到端 wrapper 前向开销由基线 ~1.5–2.3ms 降至 ~0.03–0.27ms（其余为保留的
  transpose）。**SWA(W=1024) 在 S=2048/4096/8192 由基线的 `NotImplementedError` 变为全部跑通**，entry vs direct
  的 out rel-L2=0、dQ~1e-6~1e-5（≪2e-2）。
- GPU 冒烟（容器内）：`bench/smoke_flex_attention_turbo.py`
  覆盖 causal / full / SWA(W=128) / GQA / ALiBi 数值对齐（rel-L2 < 2e-2）与报错路径；
  并新增**显式 `alibi_slopes`**（causal，与手工参考及自动识别路径三方数值一致）、**显式
  `softcap=30`→`NotImplementedError`**、以及**dropout**（`p=0` 零回归、`p=0.1` 前向+反向通过）、
  **sink**（透传与直接 `flash_attn_func(sink=...)` 逐字节一致、`sink=None` 零回归）、dropout 越界
  与 sink 非法形状的报错路径。
- GPU bias 验证（容器内）：`bench/_investigate_bias.py`（形状/精度取证：4D→RuntimeError、fp32
  `[Sq,Skv]`→NaN、bf16 `[Sq,Skv]`→正确，并测 dQ/dK/dV/dBias）与 `bench/_run_taskB.py`
  （经 flex 入口端到端 fwd+dQ，bf16/fp16 × full/causal 共 4 组，均 rel-L2 < 2e-2）。
- GPU softcap 验证（容器内）：`bench/softcap_flex_validation.py`
  验证 softcap `score_mod` 显式报错（非静默丢弃）、cap 的数值实质性、以及 `score_mod=None`
  causal 无回归。aiter 签名取证脚本：`bench/_investigate_softcap.py`、`bench/_investigate_softcap2.py`。
