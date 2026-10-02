# fwd op-evolve job：从 B0 备份恢复到 A0，并准备好续跑（2026-10-02，**未启动**）

缩写：`PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo`，`OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`，
`J=$OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927`（fwd job），`BJ=$OE/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934`（bwd job），
`F=$PT/output/1002__oe/fwdjob`（本次产物）。本次没有用卡，没有启动或恢复任何 job，没有碰 `/tmp/a0-gpu0.lock`，没有在 host 上 import torch。

## 0. 结论

| 项 | 状态 |
|---|---|
| 恢复 | `J` 已从 `/home/lihuzhan/code/0928__bak_b0.tar.gz` → `0928__bak_b0/b0_opevolve_jobs.tar.gz` 解出：sha256 与两份 SHA256SUMS 一致，5976 项，1.4 GB，`tar --compare` 无内容差异。只解了 fwd 目录，`BJ` 和 OE 里其它已有文件都没有动 |
| job 状态 | `best_round 16`（r13ns，与 git 里的 `output/0927__b0/champions/fwd_r16_r13ns` 逐字节相同）；round 20（deep）停在 act/01_implement，resume 时会重做 act；h48（refactor，采用 `rounds/019/op`）会在 round 21 开头执行；`spec_version v000` |
| 已改（都有 `.bak.pre-a0-1002` 备份，diff 在 `$F/patches/`） | `final.yaml`：容器/host/gpu_id 改成 A0 的值；`hint.md`：新增 h49，h29/h38 标为 superseded，h48 补了 A0 条件；`op/eager/impl.py`：CUDA 输入一律转到 CPU 计算；`op/refcache_util.py`：fp32 参考不再在卡上计算；`state.yaml`：删除 round 20 里两个只存在于 B0 的 Claude session id |
| OE deep prompt | 现在 OE 工作树里是 **bwd 版**（`0930__bwd/oejob/oe_uncommitted_0930.diff`）。fwd 版做成了补丁 `$F/oe_deep_fwd_1002.diff`，配有切换脚本 `$F/swap_deep_prompts.sh`，都**没有应用** |
| **现在能否启动** | **不能**。派发任务时说卡上在跑 e2e，但 10:36 UTC 已经没有 drive.sh/torchrun 进程了；10:21:42 UTC 有其他会话恢复了 bwd job（PID 319624，round 26 fast，已经在用卡）。`bash $F/prelaunch_check.sh --no-card` 目前报 NOT READY：4 项 FAIL，分别是 bwd loop、它的 `.pid`、KFD 进程、prompt 仍是 bwd 版。这是预期结果 |

启动顺序：等 bwd job 停下 → `swap_deep_prompts.sh fwd` → 把 fa-repro 里的 FlyDSL cache 挪开 → `prelaunch_check.sh` 全部 PASS → resume（命令见 §5–§6）。

## 1. 恢复过程与校验

```bash
cd /home/lihuzhan/code
# 1) 用流式方式计算内层包的 sha256，不落盘：9fe9ccc756236e54add119d93d5fd83f5a58b1f117ad5ee176e09e361ada144a，
#    与包内 0928__bak_b0/SHA256SUMS 以及 git 里的 output/0928__bak_b0/SHA256SUMS 一致（两份文件逐字节相同）
nice -n 19 ionice -c3 tar -xzOf 0928__bak_b0.tar.gz 0928__bak_b0/b0_opevolve_jobs.tar.gz | sha256sum
# 2) 只解 fwd 目录；--keep-old-files 遇到已存在的文件会报错，而不是覆盖；解压时再 tee 一次 sha256（结果相同）
nice -n 19 ionice -c3 tar -xzOf 0928__bak_b0.tar.gz 0928__bak_b0/b0_opevolve_jobs.tar.gz \
  | tee >(sha256sum) | tar -xzf - -C $OE --keep-old-files artifacts/gfx1250-flydsl-attn-fwd-b0-20260927
# 3) tar -d 逐文件比对：只有 Uid/Gid（2349 项原属 root，现在属 lihuzhan）和 69 项 Mode（容器写的 777 目录按 umask 解出）不同，
#    没有 Contents/Size/Mtime 差异
```
- 记录在 `$F/restore/`：`pass{1,2}_inner_sha256.txt`、`inner_listing.txt`（内层包全部 11072 项）、`pass3_compare.txt`、`outer_meta/`（包内的 SHA256SUMS、MANIFEST、make_bak.log）。
- 包里还有 `LOG.fwd-b0`、`LOG.fwd`、`LOG.bwd`、`jobs/gfx1250-flydsl-attn-fwd.yaml`。它们解到了 `$F/restore/b0_extra/`，没有放回 OE 根目录。其中的 jobs yaml 与 OE 里现有的那份逐字节相同。
- 冠军核对（`$F/checks/champion_check.txt`）：`J/job_context/op/current` 与 git 里的 `output/0927__b0/champions/fwd_r16_r13ns/` 对 MD5SUMS 11/11 OK，`diff -rq` 为空。`rounds/019/op`（h48 的源）只有 m32x8 一个文件不同（`e4485ba0...`，`RESCALE_THRESHOLD` 8.0 → 24.0，另有 g64 d-tile），其余文件与 r13ns 相同（`$F/checks/round019_op_md5.txt`）。
- `op-evolve status --job gfx1250-flydsl-attn-fwd-b0-20260927` 能找到这个 job，命令只读，执行前后 state.yaml 不变（`$F/checks/op_evolve_status_fwd.txt`）。
- 恢复时 `BJ` 的 mtime 仍是 09-30 14:34，没有被改。B0 时期的 OE 是 4e0ad31，A0 是 ee16d5b，后者多了 `evolve.gain_weights`。fwd 的 yaml 没有设置这个键，所以在 A0 上验收仍然是三个 shape 的算术平均，与 B0 相同。

## 2. 改动清单（`J/job_context/` 下）

| 文件 | 备份 | 改了什么 | 为什么 | diff |
|---|---|---|---|---|
| `gfx1250-flydsl-attn-fwd_final.yaml` | `.bak.pre-a0-1002` | `:784` 容器 `fa-g2 → fa-repro`；`:803` gpu_id 仍是 0，只改注释为"节点上唯一的 GPU"；`:809` host 改为 `heliosr-1b114-c07-1`；`:869-870` observed.job_setup 恢复成 A0 的值（09-25 的 setup 本来就是在 A0 上跑的） | HANDOFF §3：手工编辑，然后用普通 `resume`，**不用 `--config`**。用 yaml 解析前后对比，变化的只有 `runtime.runner.docker.container`、`runtime.host` 和 `observed.job_setup_2026_09_25.{host,container}` 三处。OE `load_spec` 能通过 | `fwd_final_yaml_a0.diff` |
| `hint.md` | `.bak.pre-a0-1002` | 新增 **h49**（must standing）：HANDOFF §3 原文（`<N>`=20），另加 A0 的具体事实，包括 round 20 要重做 act、fp32 参考不上卡、FlyDSL cache 在 `/root/.flydsl/cache`、真实 dump 和 burst 尺子在 A0 上的路径、A0 当前的读数。**h29、h38** 去掉 `standing`，status 改为 `superseded by h49`（做法与 bwd 09-30 退役旧 hint 相同）。**h48** 末尾追加 "A0 amendment"，见下 | h29/h38 写死了 B0 的 fa-g0/fa-g2；如果还标 standing，每轮都会被渲染进 route.md 的约束表 | `fwd_hint_a0.diff` |
| `op/eager/impl.py` | `.bak.pre-a0-1002` | `forward_reference` 第一句：只要输入在 CUDA 上，就先 `.cpu()` 算完，再把结果搬回原设备。数学部分一个字没改 | B0 round 20 的 `adv_m32x8.py` 就是直接 import 这个函数，在卡上跑 fp32 GEMM 时出了故障（h50）。加了这一句，round 自己写的脚本也没法再把 fp32 参考放到卡上 | `fwd_eager_guard_a0.diff` |
| `op/refcache_util.py` | `.bak.pre-a0-1002` | `reference()`：fast/proxy/prod 的 causal 用 refcache；其它情况（7 个 edge shape、任何 non-causal、fast 缓存未命中）都在 CPU 上计算；proxy/prod 未命中直接报错。`build()` 也改为在 CPU 上算。只放行一种 provenance 变化：缓存记录的 eager_sha 是 `6bc438c3…` **而且**磁盘上现在的是 `5584c5be…`（即加了 guard 之后的版本）。其它任何漂移（common_sha、eager 再改、dims）仍会使缓存失效 | 与 bwd 09-30 的 `refcache_util` 补丁一致（`0930__bwd/oejob/patches/bwd_refcache_util_a0.diff`）。原来每次 validation 都要在卡上算 13 个 fp32 参考，B0 r19 的 `gate.log` 里 7 个 edge shape 都是 `ref=eager` | `fwd_refcache_util_a0.diff` |
| `state.yaml` | `.bak.pre-a0-1002` | 删除 round 20 行的 `sessions`（planner `d694338b…`、reviewer `82966052…`），在 lifecycle 里加一条 `hand_edit`。其余内容逐字节不变 | act 会 resume planner 的 session，reflect 会 resume reviewer 的 session。这两个 session 只在 B0 的 `~/.claude/projects` 里，A0 上没有，也不在备份包里。resume 失败时 OE **不会**自动回退：`_ask` 拿到 `reply.ok=False` 就抛 ActError，结果 round 20 被记为 failed。删掉 session id 之后，会走框架自己写好的分支："the planner's session could not be resumed; working from the files" 和 "no reviewer session; running a fresh agent given the files" | `fwd_state_sessions_a0.diff` |

改动前后的 md5 见 `$F/patches/MD5-before-after.txt`；`prelaunch_check.sh` 也会核对改动后的 md5。改过的 hint.md 另存了一份在 `$F/hint.md`（md5 `d464f93a…`），供 git 跟踪。

**h48 的 A0 补充说明**：h48 要到 round 21 开头才执行，那时 round 20 的 act 已经在 A0 上重做完了，而 round 20 的 plan 本身就是在 `rounds/019/op` 上做 g65 fold。所以补充说明让 refactor agent 先看 `best_round`：
- 如果仍是 16：照原文逐字节拷 `rounds/019/op`，补充里附了 md5。
- 如果是 20（round 20 被接受，op/current 已经包含 019）：不要拷，保持工作副本不变，跑 gate，回报 no-op refactor。这样促升只会清掉 r18/r19 那些从未晋升的 best-ever 记录。

两种情况下都要先清空 `/root/.flydsl/cache` 再跑 gate。原因是 019 改的是模块级常量 `RESCALE_THRESHOLD`，而 JIT key 不包含模块级常量（h46）；gate 本身不设 cache 目录。

**没有改**：
- `validation.py`：B0 的"proxy 和 prod 各自 ≥ bar"补丁已经在里面；margin 从 spec 读取，为 0%。
- `benchmark.py`：fa-repro 没有设置 `OE_PHYS_GPU`，sclk 会回退到 rocm-smi。
- `gates.py`、`ut/common.py`：改了会让 provenance 失效。
- reviewer 仍用 B0 时改成的 claude。
- `min_gain 0.007`、`shape_band`、`max_rounds 30`、schedule：都不变，deep 轮是 20、25、30。

## 3. refcache 来源核对，以及"fp32 参考不上卡"

- 用 `$F/tools/refcache_prov.py` 读取 provenance。这个工具只用 stdlib 读 torch 的 zip，不需要 torch。三个 `.pt` 记录的都是 `seed 0`、dims 与 `ut/common.py` 一致、`eager_sha 6bc438c3784c3f67`、`common_sha e6f623771c679b7a`、`torch 2.11.0+rocm7.14.0a20260625`、`gfx1250`。用改动前的 eager 文件核对，结果 **ALL MATCH**。这批缓存是 09-25 setup 时在 **A0** 上生成的，与退役的 A0 fwd job（`…-20260925-114644`）里的 refcache 逐字节相同。B0 之后没有改过 `eager/impl.py` 和 `ut/common.py`。记录：`$F/checks/refcache_prov_{pre_guard,now}.txt`。`now` 那份显示 eager_sha 不一致，这是预期的，由上面说的固定放行规则处理。
- `$F/tools/test_refcache_util.py`：用假的 torch 和 common 驱动**真实的** `refcache_util.py`，CPU 上跑完 24 项全部 PASS（`$F/checks/test_refcache_util.txt`），覆盖：
  - 三个 spec shape 都从缓存取；
  - 13 个 edge/non-causal 组合都是 CPU 输入；
  - proxy/prod 的 non-causal、common_sha 漂移、未知 eager_sha、eager 再次改动，都被拒绝；
  - guard 是 `forward_reference` 的第一条语句。
- 仍有的风险：如果 round 自己写的脚本不经过 eager，而是直接在 CUDA 上做 fp32 matmul，代码层面拦不住。这一点只能靠 h50、h49（standing，每轮都会渲染进 route.md）和 fwd 版 CAMPAIGN CORRECTIONS 第 9 条来约束。

## 4. OE deep-round prompt：现状、fwd 版和切换脚本

- **现状**：`git -C $OE diff` 与 `0930__bwd/oejob/oe_uncommitted_0930.diff`（md5 `d55b862d…`）逐字节相同，也就是 bwd 版。需要注意 **HEAD 本身也是 bwd 专用的**，里面是 09-25 那份 corrections（k_dkdv 740 VGPR 等）。日志可以证实，A0 的 r5/r10 和 B0 的 r15/r20 这些 fwd deep 轮，用的都是 bwd 专用的那段 corrections。
- **fwd 版**：`$F/oe_deep_fwd_1002.diff`，相对 HEAD，涉及 7 个文件，md5 `4b787f48…`。由 `$F/oe_prompts/build_fwd_variant.py` 从 `oe_prompts/{head,bwd}/` 加 `campaign_corrections_fwd.md` 生成。每处替换都断言锚点只出现一次；生成后还检查两点：没有引入新的 placeholder，也没有残留 bwd 专用内容。`git apply --cached --check` 对 HEAD 通过（只检查，不写）。
  - 三个 `_preamble.md` 里放 fwd 的 CAMPAIGN CORRECTIONS，共 9 条：
    1. 冠军 r13ns 的结构，以及 round 20 重做的说明；
    2. 两把尺子在 A0 上的读数：1.08 对 1.349；
    3. PMC cycle 构成和 M8 消融；
    4. 静态 ISA 指标不能用来排名；
    5. 本 spec 的验收规则；
    6. 尺子的使用规则，包括 image hipBLASLt 和 `/root/.flydsl/cache`；
    7. speculation 已证实无效；
    8. 新固件下 ATT 可用，fwd kernel 还没抓过，需要确认 capture 真的生效；PC sampling 仍然禁止；
    9. 卡上的规则：fp32 参考不上卡、compile-only、一次只能有一个 GPU 客户端。
  - `01_select` 到 `04_thread_trace`：保留 bwd 版里与具体 job 无关的 ATT 启用内容，删掉 bwd 专用的部分（k_dqg、`"k_d[kq]"`、s6 的 ATT 记录、h26/h78）。换成 fwd 的内容：kernel 名、regex `"fmha_fwd_prefill"` 和 ASM 的 `aiter::fmha_bf16_pertokenBf16_hd128_128x256_mask`，fast 走的是 m32x2，以及 roofline 报告的路径。
- **切换**：`bash $F/swap_deep_prompts.sh status|fwd|bwd|head`。只用 `git apply` 和 `git apply -R`。遇到下列情况会拒绝执行：
  - 有 op-evolve loop 在跑（按进程映像和各 job 的 `.pid` 判断，不会因为命令行里出现 "op-evolve resume" 字样而误判）；
  - 7 个文件既不是 head/bwd/fwd 中的任何一种，或者已经 staged；
  - 补丁的 md5 与脚本里记录的不一致。

  切换前会把当前的 7 个文件打成 tar，存到 `$F/oe_prompt_backups/`。在 OE 的 scratch clone 上测过：bwd→fwd→bwd→head→fwd→bwd 共 5 次结果逐字节正确，3 种拒绝情况也都正确（`$F/checks/swap_test.log`）。
- **什么时候切**：fwd resume **之前**切到 `fwd`，因为 resume 后第一件事就是 round 20 的 deep act，会读 act preamble。fwd job 停下、bwd 要 resume 之前切回 `bwd`。fast 轮不读这些 prompt，但 bwd 的下一个 deep 是 round 29，所以要切回来。

## 5. 启动前清单（按顺序执行；只有第 2–4 步会改动东西）

```bash
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve
J=$OE/artifacts/gfx1250-flydsl-attn-fwd-b0-20260927
F=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__oe/fwdjob
# 1. 卡上只能有一个 GPU 客户端。bwd job 什么时候停，由启动它的人决定：
#    (cd $OE && .venv/bin/op-evolve stop --job gfx1250-flydsl-attn-bwd-20260917-115934)，会在模块边界退出。
#    同时确认没有 e2e（drive.sh/torchrun）、没有 realab 或手工 run.sh，KFD 为空。
# 2. deep prompt 切到 fwd 版（有 loop 在跑时会拒绝）
bash $F/swap_deep_prompts.sh fwd
# 3. 换一个干净的 JIT cache。job 本身不设 FLYDSL_RUNTIME_CACHE_DIR，所以用的是 fa-repro 里的 /root/.flydsl/cache（bwd job 也用这个）
docker exec fa-repro bash -c 'S=$(date -u +%m%d_%H%M%S); [ -d /root/.flydsl/cache ] && mv /root/.flydsl/cache /root/.flydsl/cache.pre-fwd-$S; rm -rf /tmp/flycache /root/.cache/comgr/*'
# 4. 让 agent 能读到 dmesg（重启后会恢复成 1）
sudo sysctl -w kernel.dmesg_restrict=0
# 5. 全部只读检查：必须输出 "RESULT: READY"
bash $F/prelaunch_check.sh
# 6. 可选，不用卡：检查 LLM 账号（会在 OE 下留下无害的 job_context/env 目录）
cd $OE && .venv/bin/python tools/check_llm_account.py --config $J/job_context/gfx1250-flydsl-attn-fwd_final.yaml
```
`prelaunch_check.sh` 检查的项目：
- 没有 op-evolve loop，也没有存活的 `.pid`；
- 没有 e2e/realab/手工 campaign 进程；
- lock 没有人持有；
- KFD 为空；
- `spec_version v000`；
- round 20 停在 act/01_implement，而且不带 session；
- 4 个改动文件和两个冠军 kernel 的 md5 正确（state.yaml 只检查 session 是否已删掉）；
- hint 解析正确，即 h48 待执行且带补充、h49 是 standing、h29 和 h38 已退役；
- refcache provenance 正确，CPU 回退测试通过；
- prompt 是 fwd 版；
- `docker exec` 有响应；
- fa-repro 里 FlyDSL cache 为空，`dmesg_restrict` 为 0（这两项不满足只报 WARN）；
- 打印 `pp_dpm_sclk` 和最近的 amdgpu dmesg。

## 6. 启动命令，以及启动后应该看到的输出

```bash
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve
J=gfx1250-flydsl-attn-fwd-b0-20260927
LOG=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/1002__oe/fwdjob/oe-$J.log
setsid nohup env PATH="$PWD/.venv/bin:$PATH" op-evolve resume --job $J >> "$LOG" 2>&1 < /dev/null &
```
只用普通 `resume`，**永远不要加 `--config`**。停止时用 `op-evolve stop --job $J`，或者 `kill -TERM -<PGID>`。监控按 skill 里的 `references/op-evolve-ops.md` §8：每 30 s 轮询一次，正常时不输出任何东西。每轮结束后发中文汇总表。

日志里依次应该出现：
- `round 20 (deep):`
- `hints: refactor h48 deferred to the next round -- a module has already run`
- `3-act: an unfinished attempt was moved to 3-act.stale-…`
- `3-act: the planner's session could not be resumed; working from the files`：这是预期的，见 §2 对 state.yaml 的说明。
- `rounds/020/gate.log` 里：fast/proxy/prod 是 `ref=refcache`，edge shape 是 `ref=cpu (no cache)` 或 `ref=cpu (non-causal)`。**不应出现** `ref=eager`，也不应出现 `refused`。
- `4-reflect: no reviewer session; running a fresh agent given the files`
- round 21：`0-refactor: executing operator hint h48`，然后是 `0-refactor: correct against op/eager`，再然后是 `hints: op/current/ is round 21`。这之后 `op/current` 的 m32x8 md5 应当是 `e4485ba0…`（如果 round 20 没被接受）。

时间预算：剩下的是 round 20 到 30，共 11 轮。
- round 20 的 act 加 reflect：约 0.5–1 h；
- fast 轮：B0 上每轮 32–55 min；
- deep 轮 25 和 30：现在 ATT 打开了，每轮可能超过 2 h；
- 合计约 10–15 h 独占这张卡。

已用时间 15.5 h，max_timeout 为 120 h。

## 7. 留给操作者决定的事（本次都没做）

1. **尺子**：job 的验收和 validation 用的是 randn 加 blocked 的尺子，A0 上 r16/ASM 读 1.08；训练工作点（真实 dump、GEMM burst 之后）读 1.349。也就是说，job 正在用一把看不到大部分训练差距的尺子做优化。h47 和 fwd 版 corrections 都要求每个候选同时报告 burst 尺子的结果。如果要把 burst 尺子放进验收或 validation，属于框架层的改动，需要用户批准（HANDOFF §5.4）。
2. **`evolve.gain_weights {prod 1, proxy 0.25, fast 0}`**：bwd 09-30 已经加上（D7/h83），A0 的 fwd job 在 round 10 也用过。A0 的 OE 支持这个键，只需要在 final.yaml 的 `evolve:` 下手工加一行。没有加，要等用户批准。bwd 那边配套的"fast 按 min 计分"也一样没有加；fwd 现在靠 `shape_band fast 0.90` 处理 fast 的噪声。
3. **目标 margin 0%**：r13ns 在 randn 尺子上约为 ASM 的 0.93。如果以后的冠军在 randn 上 proxy 和 prod 都超过 ASM，job 会以 `target_met` 结束。要提高的话用 `op-evolve tune --beat-margin X`，validation 会从 spec 里读这个值。
4. **轮数**：`max_rounds 30`，只剩 11 轮。可以用 `op-evolve tune --max-rounds N`，不会提升 spec_version。
5. **beat profile**：`job_context/profiling/beat/` 是 A0 round 5（09-27）在**旧固件**、1100 MHz 封顶的时期抓的。如果想让 round 25 在新固件上重抓，就把它挪开，代价是多一次 profiling。
6. **h30**（B0 时期与 A0 fork、GPU-2 lab 的分工）仍然是 standing。A0 的 fwd job 已退役，GPU-2 lab 也不存在了，可以考虑退役 h30。
7. **先跑 fwd 还是 bwd**：HANDOFF 说先跑 bwd，依据是当时 e2e 里 bwd 差约 52 ms/步。今天 realab 的推算是：s6 相对 ASM 约 +7 ms/步，fwd 约 +13 ms/步。按这个数，fwd 已经是 e2e 上更大的一项。

## 8. 回滚

- 单个文件：`cp -p X.bak.pre-a0-1002 X`，适用于 `J/job_context/` 下的 5 个文件。
- 整个恢复的 job：`rm -rf $J`。这是新建的目录，没有覆盖任何东西；需要的话按 §1 重新解压。
- OE prompt：`bash $F/swap_deep_prompts.sh bwd`（或 `head`）；切换前的 tar 在 `$F/oe_prompt_backups/`。

## 9. 本目录文件

`$F/`：
- `patches/`：5 个 diff 和 `MD5-before-after.txt`；
- `oe_deep_fwd_1002.diff`；
- `swap_deep_prompts.sh`；
- `prelaunch_check.sh`；
- `hint.md`：改动后的副本；
- `oe_prompts/{head,bwd,fwd}/`：7 个文件的三个版本，以及 `campaign_corrections_fwd.md`、`build_fwd_variant.py`、`files.txt`；
- `tools/`：`refcache_prov.py`、`test_refcache_util.py`；
- `checks/`：所有核对的输出；
- `restore/`：解压的记录和 B0 日志。

这些文件都还没有提交。最大的是 `restore/inner_listing.txt`，1.8 MB。

同一目录下的 `_extract/` 和 `ruler/` 是其它并行会话的产物，不属于本工作。
