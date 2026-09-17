# 反馈给 `feat/gemm/gfx1250-flydsl-gemm` 作者的两个问题

来源：2026-09-16，在 gfx1250（MI455X-class，单卡）上把这个分支接进 llama3.1-8b 训练的
wgrad 路径时发现。**两个问题都不影响该分支自己的 107/107 测试**，它们只在
"从训练路径上在线调用"这种用法下才暴露。

先说结论：**这个 GEMM 本身是好的。** bf16 TN 上调优后 626–794 TF/s，六个形状 SQNR 全部
345.2 dB；对照之下这台机器的 hipBLASLt 在同类调用上是 50–76 TF/s（那是另一个缺陷，
见 `VENDOR-REPORT-hipblaslt.md`）。下面两条是可用性问题，不是性能问题。

---

## 1. 调优缓存的 key 漏掉 M，而给出的理由在 TN 布局下正好反过来

`gemm_gfx1250_kernel.py`：

```python
# Tuned configs, keyed on (kind, layout, N, K). M is left out: it is the token
# count, varies per step, and moves throughput far less than N and K do.
_TUNED: dict = {}
```

理由对 `LAYOUT_NT`（Linear 前向）和 `LAYOUT_NN`（dgrad）成立。但 `_resolve` 里 TN 的定义是

```python
else:  # LAYOUT_TN: A[K,M]^T @ B[K,N] -- wgrad
    (Kb, M), (K2, N) = a.shape, b.shape
```

**TN 里 M 是 out_features，K 才是 token 数** —— 正好与注释相反。

而且 `feasible_configs(kind, layout, M, N, K)` 的筛选条件里**完全没有用到 M**
（只筛 `N % tile_n` 和 `K % tile_k`），所以换一个 M 之后没有任何一处会复核配置是否合适。

**实测证据**（llama3.1-8b 8 层的 wgrad 形状，各自独立调优）：

| M | N | K | 调优结果 tile | m_warp/n_warp |
|--:|--:|--:|---|---|
| 4096 | 4096 | 32768 | [128, 256, 32] | 2 / 4 |
| 1024 | 4096 | 32768 | [128, 128, 64] | 4 / 4 |
| 6144 | 4096 | 32768 | [128, 128, 64] | 2 / 4 |
| 14336 | 4096 | 32768 | [256, 128, 32] | 2 / 4 |
| 128256 | 4096 | 32768 | [256, 128, 32] | 4 / 2 |

**五个形状共用同一个 `(N, K)`，却有四个不同的最优配置。** 现在的 key 会把它们全部折叠成
最先调的那一个。

建议：TN 的 key 带上 M；或者按布局区分，NT/NN 保持现状、TN 用完整形状。

## 2. `autotune` 的 `except Exception: continue` 无法从已损坏的 HIP context 恢复

```python
try:
    for _ in range(3): call()
    torch.cuda.synchronize()          # 发射失败在这里抛出
    ...
except Exception:
    continue  # a config that will not compile is not a candidate
```

注释说的是"编不过的配置不算候选"，这对**编译期**失败是对的。但同一个 `except` 也会接住
**运行期的发射失败**（`hipErrorLaunchFailure` / "unspecified launch failure"），
而 HIP context 一旦进入这个状态就**不可恢复** —— 捕获 Python 异常不能让它复活，
循环会继续对着一个已经死掉的 context 跑完剩下的几十个候选。

后果是错误出现的位置离肇事点很远：真正的失败会在之后某个**无关的** GEMM 上报出来。
我们这边的表现是训练在 step 2 抛 `unspecified launch failure`，随后
`MES might be in unrecoverable state` → `GPU reset begin!` 未完成，需要人工 AC-cycle。

建议：把发射失败与编译失败分开处理 —— 前者应当**中止整个调优并向上抛**，而不是 `continue`；
或者至少在 `except` 里检查 context 是否还活着（例如一次小的 `torch.cuda.synchronize()`），
活不下来就立刻放弃。

（我们这边的规避办法是不在在线路径上调用 `autotune`：离线建表、每形状一个子进程、
带硬超时与逐形状的 SQNR 门，训练只查表。但那是绕开，不是修复。）

---

附带一条不属于这个分支、但相邻的观察：`setup.py:548-554` 在 gfx1250 构建时跳过安装 flydsl，
TODO 里写的理由是 "Triton 3.7.0 and flydsl 0.2.4 does not support gfx1250"。
就 flydsl 而言这一半已被这个分支自己证伪；就 Triton 而言也已被树里的 gfx1250 Triton 后端证伪。
容器里装着的 flydsl 0.2.4 带有 `MmaOpGFX1250_WMMAType`，这个分支在它上面工作正常。

---

## 附：建议的最小修复

两处都很小，且都不改变现有 NT/NN 调用者的行为。

**（1）TN 的调优缓存键补上 M。**

```python
# autotune()
key = (cfg.kind, layout, cfg.N, cfg.K)
```

改为在 TN 下带上 M（NT/NN 维持原样，那里注释是对的）：

```python
key = ((cfg.kind, layout, cfg.M, cfg.N, cfg.K) if layout == "tn"
       else (cfg.kind, layout, cfg.N, cfg.K))
```

并把 `_TUNED` 上方那段注释限定为 NT/NN —— 否则下一个人读到的仍是
"M 是 token 数"，而在 TN 下它不是。

**（2）`autotune` 区分"编不过"与"发射失败"。**

现在两者共用同一个 `except Exception: continue`。前者 continue 是对的，
后者之后 HIP context 已经不可恢复，continue 只会让剩下几十个候选对着一个死掉的 context
跑完，错误最终在**无关的**调用上报出来。最小改法是在 `except` 里判断一次：

```python
except Exception as exc:
    if "launch failure" in str(exc) or "unspecified" in str(exc):
        raise                     # context is gone; continuing cannot help
    continue                      # genuinely just "this config will not compile"
```

（字符串匹配不优雅，但比现状好；更干净的做法是调用方在 autotune 外层包一次
context 存活检查。我们这边的规避是根本不在在线路径上调 `autotune`，
改为离线建表 —— 但那是绕开，不是修复。）

## 我们这边可直接复用的产物

`output/0915__opt/bin/flydsl_table.py` —— 离线建表脚本：每形状一个子进程（一个形状 fault
只损失那个形状）、硬超时、逐形状 SQNR ≥ 50 dB 门。它产出的表也就是上面那张"五个共用
`(N,K)` 的形状调出四个不同最优配置"的证据。
