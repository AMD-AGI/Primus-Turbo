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

## 混合方案的预期收益：58 ms / 7.6%（我先前估的 77 ms 偏高）

第一版估算用四个形状的**平均**吞吐外推，得到 77 ms。补测 lm_head 之后必须下修 ——
**它占 wgrad FLOPs 的 29%，而 FlyDSL 在这个形状上只赢 6.8%，不是其它形状的 1.4–1.8×**：

```
lm_head wgrad  M=128256 K=32768 N=4096   A = 7.83 GiB
  torch          50.3 TF/s
  nkfix+copy    732.1 TF/s      <- 这里 nkfix 本来就跑得好
  fly TN        561.6 TF/s
  fly tuned     782.2 TF/s      SQNR 345.2 dB   tile=(256,128,32) mw=4 nw=2 nb=2
```

nkfix 在 lm_head 上反常地好（732 vs 其它形状的 442–517）。合理的解释是这个形状的
算术强度足够高，7.83 GiB 的那次拷贝被 GEMM 本身的时间摊薄了 —— 但**这只是解释，没有验证**。

逐形状重算（`时间 = FLOPs / 吞吐`）：

| 调用 | 每步 FLOPs | nkfix ms | FlyDSL ms | 省 |
|---|--:|--:|--:|--:|
| o_proj ×16 | 1.76e13 | 39.8 | 22.2 | 17.7 |
| kv ×16 | 4.40e12 | 9.3 | 7.0 | 2.3 |
| mlp ×8 | 3.08e13 | 59.5 | 43.0 | 16.5 |
| mlp2 ×8 | 3.08e13 | 60.8 | 42.4 | 18.4 |
| lm_head ×1 | 3.44e13 | 47.0 | 44.0 | 3.0 |
| **合计** | 1.18e14 | **216.6** | **158.6** | **58.0** |

**约省 58 ms，占单步 760 ms 的 7.6%。**

仍然是**估算**，剩下两条前提：
1. `kv`（M=1024）用 `qkv`（M=6144）的实测值代入，两者 M 差 6 倍；
2. 端到端还要算 dispatch 开销 —— FlyDSL 的 backend 注册**对我们的训练无效**
   （profile 实证走 `aten::mm` 342 次、`aten::linear` 114 次，**零** `primus_turbo::gemm`），
   接入只能像 nkfix 那样在 `TorchDispatchMode` 里直接调。

## 下一步

1. 补测 lm_head 形状（M=128256），它是估算里最大的未知数；
2. 在 `nkfix.py` 里加第三条规则：wgrad 命中时改调 FlyDSL TN，而不是做两次拷贝再走 hipBLASLt；
3. 8 层 e2e A/B 验证，n≥3 交替。

**注意不要把这条路和 FlyDSL 注意力混为一谈**：后者七个以上 builder 硬断言 gfx950
（依赖 CDNA4 的 `ds_read_tr16_b64`），是逐 kernel 的移植，见 `output/0914__campaign/HANDOFF.md`。
GEMM 这条是现成可用的 gfx1250 WMMA 实现，107/107 测试通过。
