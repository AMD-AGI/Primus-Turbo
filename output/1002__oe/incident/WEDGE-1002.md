# A0 挂卡（2026-10-02 11:40:59 UTC）

## 现象（dmesg，`dmesg_1140.txt`）
11:40:59 `MES(0, 0) failed to respond to msg=REMOVE_QUEUE` → SUSPEND 失败 → `queue id 0x1 at pasid 405030 is reset` → RESET 失败 →
`MES might be in unrecoverable state, issue a GPU reset` → **`GPU reset begin!. Source: 3`** → ADD_QUEUE 失败 → 各 legacy queue REMOVE_QUEUE 失败。
按 skill gfx1250-card-safety §2a 判定为**不可恢复**（有 GPU reset begin），需要人工 AC 断电。

## 触发者
bwd op-evolve job `gfx1250-flydsl-attn-bwd-20260917-115934` round 27（fast）的 opt agent，11:38:55 写出并运行 `rounds/027/1-opt/raw/build_cmd.sh`（副本在本目录）：
- 用 `build_probe.py` 在 **prod 形状**上直接调用两个**从未上过卡**的 w4f 融合 kernel 变体（`rounds/027/_scratch/arms/A_g74`、`B_g82`；4-wave WG + split barrier + dQ 原子写），
  目的只是"dump ISA 读 vgpr_count"——而 compile-only 就能拿到这个数字，根本不需要上卡；
- **没有 toy-first、没有单独进程的串行化（AMD_SERIALIZE_KERNEL=3）、没有锁/KFD 包装**；
- `HIPBLASLT_TENSILE_LIBPATH` 指向**宿主机库 `~/.local/hipblaslt-gfx1250`**（09-28 A0 挂卡/NaN 的嫌疑库）。

## 当时状态
- e2e、realab 都已结束；只有 bwd job 在用卡；fwd job 未启动；fwd 修复 agent 只做 CPU 工作。
- 11:39:57 已写 `.stop`（未 kill 任何进程，避免制造更多 D 状态进程）。

## 断电后的恢复步骤（用户 AC 断电后由 Claude 执行）
1. `sudo modprobe amdgpu`（内核参数 blacklist 了 amdgpu）；`sudo sysctl -w kernel.dmesg_restrict=0`
2. 检查 `pp_dpm_sclk` / VBIOS / dkms 是否又变了（REPORT 的参数表）
3. `ls /sys/class/kfd/kfd/proc` 为空；`docker start fa-repro`；一次 4096³ GEMM 健康检查
4. `git fsck`（断电可能截断 git 对象）；Primus-Turbo 已推送到 `5696ae53` 之后的提交
5. bwd job：round 27 会在下次 resume 时重做；**先加 hint h85（见下）再 resume**

## 防再犯
- bwd job 新增 standing hint（h85）：任何从未上过卡的 kernel/arm 必须先在 toy 形状、单独进程、`AMD_SERIALIZE_KERNEL=3`、经锁/KFD 包装、用 image hipBLASLt 库跑通，才能上 prod；
  只为读 VGPR/ISA 的探针一律 compile-only（`COMPILE_ONLY=1 HIP_VISIBLE_DEVICES=-1`）；禁止 `~/.local/hipblaslt-gfx1250`。
- fwd job 的 hint 和 CAMPAIGN CORRECTIONS 同样加入（已交给 fwd 修复 agent）。
