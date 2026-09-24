# refcache 的前提没有了（2026-09-24 实测）

## 起因

昨天修好 hipBLASLt（真因是 `HIPBLASLT_TENSILE_LIBPATH` 从未被设，而
`/usr/lib/python3.12/sitecustomize.py:18` 又把 `TORCH_BLAS_PREFER_HIPBLASLT` 钉死成 0，
使 `_env.py` 的 `setdefault` 十二轮来一直是空操作）之后，
有一个问题可以用实测代替争论：**那条「会挂卡」的 fp32 参考，现在还会挂吗。**

## 实测

环境：`HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250`，
`TORCH_BLAS_PREFER_HIPBLASLT=1`。逐级放大，每级查 KFD 与 `docker exec` 响应。

| 形状 | `forward_reference` | `eager_attn_bwd` | 合计 | 结果 |
|---|--:|--:|--:|---|
| fast | 0.17 s | 0.01 s | 0.18 s | 干净 |
| proxy | 0.20 s | — | — | 干净 |
| **prod** | **0.88 s** | **0.89 s** | **1.77 s** | **干净** |

prod 就是 2026-09-22 烧掉一次断电周期的那个调用。它现在 **1.77 秒**跑完，卡全程健康。

**顺带证伪一个可能的替代解释**：不是显存压力。`forward_reference` 是分块的
（`q_chunk=1024`，`common.py:53`），每块 score 只有 `[1024,8192]` fp32 = 32 MiB，
而卡有 432 GiB 显存。当初的 fault 就是 Tensile 库加载失败那条路径。

## 逐位确定性

这是「能不能拆掉 refcache」的关键——如果 hipBLASLt 每次挑不同算法，参考会漂移。

```
fast  同进程 rep0/rep1:  o,lse,dq,dk,dv 五个哈希全部相同
prod  跨进程 两次:       o=7b0249a1cd828ba0  lse=6fa4e0af2ed03308
                        dq=bc919da8bcdb4dbe  dk=2aee8288433b8bcb  dv=d3f40069856de21c
                        —— 两个独立进程完全一致
```

**逐位确定，跨进程成立。**

## 结论

refcache 省下的是**每次 validation 1.8 秒**。它的成本是：

- `op/refcache/{fast,proxy,prod}.pt` 约 1.2 GB
- `op/refcache_util.py` 101 行
- `validation.py` 里的 `load_reference()` 与五项 provenance 校验
- `tools/gfx1250/build_refcache.py` 与 `--ref-device cpu` 重建流程
- **以及最要命的一条：它把 `ut/common.py` 锁死了**

最后一条是实际代价。`build_refcache.py` 把 `ut/common.py` 的 SHA-256 写进每个缓存的
provenance，所以**任何对 `ut/common.py` 的改动都会作废全部缓存**。
这正是 `poison_allocator` 的修复一直被推到「以后某个维护窗口」的原因——
而那个缺陷已经在 2026-09-23 参与了一次越界没被抓住。

**为了省 1.8 秒，锁死了测量工具链里最该能改的那个文件。**

## 尚未验证

- 只在当前这块卡、当前这个容器上测过。换机器要重测。
- fault 的「可复现性」没有反向对照——**没有**故意在旧配置下重跑来确认它还会挂。
  那等于故意重现一次挂卡。所以严格说，证据是「新配置干净」，
  不是「旧配置必挂、新配置必不挂」。
