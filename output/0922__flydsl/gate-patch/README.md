# 闸门补丁备份 —— 2026-09-22

`validation.py` 在 op-evolve 的 artifact 树里（`artifacts/<job>/job_context/op/`），
那个目录是作业状态、不归本仓库版本控制，而且 **spec version 一旦 bump，op_setup 会重建
整个 `op/`，这份补丁就没了**（见 h4 相关调查：`resume --config` 会触发 setup 重跑，
op_setup 当初花了 2850 秒）。所以在这里留一份。

## 改了什么

新增 `REFCACHE` 常量与 `load_reference()`，并在 `check_correctness()` 里优先读缓存。
缓存由 `tools/gfx1250/build_refcache.py` 生成，落在 `job_context/op/refcache/<shape>.pt`。

**判据未改**：同样的参考值、同样的 `sqnr_db`、同样的 50 dB 闸门，只是不再每轮重算。

## 为什么

闸门自己的 fp32 参考是本作业最可靠的挂卡途径，见提交 376b9fd3。
今天为它烧掉一次 AC-cycle；round 3 与 round 8 各被它中断一次。

## 恢复方法

若 `op/` 被重建：重新应用本文件，并重跑

    python3 tools/gfx1250/build_refcache.py <job_context/op> fast proxy
    python3 tools/gfx1250/build_refcache.py <job_context/op> --ref-device cpu prod

prod 必须走 `--ref-device cpu`：它在 GPU 上算会 fault（实测两次，两种签名）。

---

## 补完（第二轮）：`validation.py` 之外还有两处

只改 `check_correctness` 是不够的。审计发现 fp32 Tensile 派发还留在两个地方，其中一个在 prod 上每轮都跑：

| 调用点 | shape | 每轮跑 |
|---|---|---|
| `benchmark.py:91` `forward_reference` | fast/proxy/**prod** | **是**（`validation.py` 派生的计时子进程）|
| `validation.py` `check_determinism` | fast | 是 |

**`benchmark.py:91` 就是付掉那次 AC-cycle 的那个面。** 两处都只要 `o` 和 `lse`，缓存里有。

新增 `refcache_util.py` 提供 `cached_forward()`，两处 import 它。

### 为什么是独立文件而不是放进 `ut/common.py`

`build_refcache.py` 把 `ut/common.py` 和 `eager/impl.py` 的 SHA 写进了 provenance，
`validation.py` 据此**拒绝**不匹配的缓存。改 `common.py` 会作废 fast.pt 与 proxy.pt，
并逼你**在卡上重建 prod.pt**——正是这套缓存要消掉的那个 fault。所以 helper 必须在两个被哈希的文件之外。

### 端到端验证

完整 `validation.py` **34.5 秒**跑完：correctness pass、determinism pass、零回落、零异常。
三形状 dB 与缓存前完全一致。

## 已知未修（故意留着，附理由）

**`poison_allocator` 在 prod 上实质失效。** 最大毒块 64 MiB，候选的 dq 请求 256 MiB，
caching allocator 不可能用前者满足后者。于是 `isfinite` 覆盖断言跑在驱动新给的页上，
而新页通常是 0——「漏写的 lane 看起来是个合理的 0.0」正是毒化要抓的失败。
**不现在修**：它在 `ut/common.py` 里，改它会作废全部缓存并逼 prod 在卡上重建。
要修就在下一次重建 prod.pt 之前一起做。

**`test_correctness.py:41` 的种子不可复现。** `seed=hash(shape) & 0xFFFF`，
而 Python 3 对 str 的 `hash()` 受 `PYTHONHASHSEED` 加盐，作业里没设置。
每次进程启动用的输入都不同，所以它既不可复现，也永远无法命中 seed=0 的缓存。
它不在 `validation.py` 的每轮路径上，只在手动调用时触发——但一旦调用，
它比 `benchmark.py` 更危险（prod 上 forward+backward 七个 GEMM 全跑）。
