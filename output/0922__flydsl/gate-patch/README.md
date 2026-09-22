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
