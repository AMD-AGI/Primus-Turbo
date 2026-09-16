# FlyDSL 的 gfx1250 GEMM：dgrad 输、wgrad 赢，混合方案值约 10%

日期 2026-09-16 · 分支 `tmp/flydsl-gemm`（`feat/gemm/gfx1250-flydsl-gemm` 合入 `dev/lhz/attn`，零冲突）

## 先纠正一个自己犯的错

第一次测出来 "fly NN 612 / fly tuned 619"，几乎没差别，差点写成"这条路调优没用"。
实际是**调优根本没生效**：`autotune()` 返回的是元组
`((tile_m, tile_n, tile_k), m_warp, n_warp, num_buffers)`，我按 NamedTuple 去 `_asdict()`，
`hasattr` 为假就落到了 `{}`。改成显式解包后，调优给出 **+33%**——与作者在 commit 里说的
"手挑的默认值曾差到 46%" 一致。

这正是 skill §13 那条：**你的参数解析接受的键，不等于后端真的读到了它**。
一个"没效果"的结论，先要证明覆盖生效了。

## dgrad（NN 布局，`mm(dout[T,out], W[out,in])`）

| 形状 M×N×K | torch NN | nkfix TN 绕路 | FlyDSL NN | FlyDSL 调优 |
|---|--:|--:|--:|--:|
| o_proj 32768×4096×4096 | 76.6 | 1406.9 | 614.0 | 819.9 |
| mlp_up 32768×14336×4096 | 76.0 | 1265.2 | 650.7 | 822.6 |
| mlp_dn 32768×4096×14336 | 75.5 | 1626.6 | 620.5 | 848.8 |
| qkv 32768×6144×4096 | 76.4 | 1238.1 | 620.1 | 820.8 |

单位 TF/s。FlyDSL 是**对 hipBLASLt 缺陷的真修复**（10.8×），但**仍比 nkfix 的绕路慢 1.5–2×**。
而 dgrad 这一半 nkfix 只需要拷贝权重（12.5 ms/步，见 `BOTTLENECK-SHIFT.md`），代价很低。

**dgrad 保持 nkfix。**

## wgrad（TN 布局，`mm(dout.T[out,T], x[T,in])`）—— 这里结论反过来

这一半才是关键，因为 nkfix 在这里每步要付 **137 ms** 的拷贝，且两个操作数
（dout 转置、激活）**都不可缓存**。下表的 `nkfix+copy` 列**含拷贝**，即训练真正付的代价：

| 形状 M×K×N | torch | nkfix+拷贝 | FlyDSL TN | FlyDSL 调优 |
|---|--:|--:|--:|--:|
| o_proj 4096×32768×4096 | 56.3 | 441.7 | 559.5 | **794.2** |
| mlp_up 14336×32768×4096 | 61.4 | 517.0 | 570.9 | **715.8** |
| mlp_dn 4096×32768×14336 | 58.8 | 506.3 | 560.8 | **726.6** |
| qkv 6144×32768×4096 | 61.5 | 470.6 | 568.1 | **625.9** |

**FlyDSL 快 1.4–1.8×**，因为它**直接消费转置视图，不需要那两次拷贝**。
四个形状 SQNR 均为 **345.2 dB**（数值上等同精确，不是勉强过门）。

## 混合方案的预期收益（估算，未端到端验证）

按 8 层配置每步的 wgrad 调用数与 FLOPs：

| 调用 | 每步次数 | FLOPs/步 |
|---|--:|--:|
| o_proj | 16 | 1.76e13 |
| kv | 16 | 4.40e12 |
| mlp | 8 | 3.08e13 |
| mlp2 | 8 | 3.08e13 |
| lm_head | 1 | 3.44e13 |
| **合计** | | **1.18e14** |

- nkfix 有效吞吐 ~480 TF/s → **约 245 ms/步**
- FlyDSL 调优 ~700 TF/s → **约 168 ms/步**
- **节省约 77 ms**，占当前单步 760 ms 的 **约 10%**

**这是估算，不是测量**，三条必须标明的前提：
1. lm_head 的 M=128256 **没有实测**，它一个就占 29% 的 wgrad FLOPs，FlyDSL 在这种极端瘦长
   形状上的表现未知；
2. 有效吞吐取的是四个形状的粗略均值，各形状差异有 25%；
3. 端到端还要算上 dispatch 开销 —— FlyDSL 的 backend 注册**对我们的训练无效**
   （profile 实证走 `aten::mm` 342 次、`aten::linear` 114 次，**零** `primus_turbo::gemm`），
   所以接入方式只能是像 nkfix 那样在 `TorchDispatchMode` 里直接调。

## 下一步

1. 补测 lm_head 形状（M=128256），它是估算里最大的未知数；
2. 在 `nkfix.py` 里加第三条规则：wgrad 命中时改调 FlyDSL TN，而不是做两次拷贝再走 hipBLASLt；
3. 8 层 e2e A/B 验证，n≥3 交替。

**注意不要把这条路和 FlyDSL 注意力混为一谈**：后者七个以上 builder 硬断言 gfx950
（依赖 CDNA4 的 `ds_read_tr16_b64`），是逐 kernel 的移植，见 `output/0914__campaign/HANDOFF.md`。
GEMM 这条是现成可用的 gfx1250 WMMA 实现，107/107 测试通过。
