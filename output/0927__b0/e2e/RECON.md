# B0 e2e 侦察：Llama-3.1-8B 端到端怎么跑、三个 attention arm 怎么接入（2026-09-28）

只做了阅读，外加两次 import-only 检查（都在 `flock /tmp/b0-gpu0.lock` 下跑，每次 < 3 s，没有 kernel launch，
跑完 dmesg 里 0001:04:00.0 没有新行）。日志：`recon/import_check{,_turbo}.log`。

## 0. 结论（先看这里）

1. **Harness**：沿用 0915 的 `primus-cli direct -- train pretrain`，config 以 `repro_l8b_turbo_conv.yaml` 为基础
   （32 层、MBS=GBS=4、seq 8192、AC none、compile 关、`debug.seed 1234`、`converters: ["primus_turbo"]`），
   容器换成 `fa-g0`，整次训练持有 `/tmp/b0-gpu0.lock`。命令见 §4。旧的 `0915__opt/bin/e2e.sh` 在 B0 上有
   **5 处不能照搬**（§4.2），其中两处会让测量静默失真。
2. **分两个进程，而不是三个**：
   - **P1 = arm 1（turbo baseline）**：真正的 Primus-Turbo，取 `wt-bakeoff @1cb2e183`，镜像自带的 flydsl 0.2.4。
   - **P2 = arm 2（aiter ASM）+ arm 3（FlyDSL 冠军）同进程按步交替**：flydsl 0.3.4.1 + fwd r6 树 + bwd r20 树的
     0.3.4.1 版（`output/0925__flydsl/bwd341/op0341`）+ `aiter.ops.mha` + 按文件路径加载的 `_asm_bwd_kernargs.py`，
     **不 import 真正的 primus_turbo**：PYTHONPATH 最前面放一个 shim 包 `primus_turbo`，只提供 converter 需要的
     `TurboAttention` 和 `low_precision` 两个名字。已实测这套组合能在一个进程里共存（§3.2）。
   - arm 1 不能并进 P2：`wt-bakeoff` 的 primus_turbo 在 flydsl 0.3.4.1 下 import 直接失败（§3.3，已实测）。
3. **为什么 P2 要同进程交替**：arm 2 和 arm 3 的差距只有约 **80 ms/step**（op 级 32 层合计 245 vs 325 ms），
   而 A0 上 e2e 的**跨进程**噪声是 3.9% 且三模态（`0915__opt/E2E-AB.md` 第四轮，n=9/臂）。只要单步超过约 2 s，
   这个差距就落在跨进程噪声里。同进程 ABBA 按步交替可以消掉跨进程模态，也少一次启动（启动次数就是卡挂的风险，
   历史约 1/5）。以后 fwd/bwd 有 >5% 提升时，重跑 P2，schedule 改成 `asm / fly_prev / fly_new` 三路交替即可。
4. **最大的未知是 GEMM**，不是 attention：见 §6.1。B0 上 hipBLASLt 用 `/home/lihuzhan/.local/hipblaslt-gfx1250`
   （宿主 ROCm 10.1.0）这份库跑 e2e GEMM 形状的吞吐**从来没测过**。如果它像 0914 的镜像库那样只有约 115 TF/s，
   单步会到 10 s 以上，三个 arm 之间 attention 的差异在 tps 上基本看不出来（arm 2 vs 3 约 0.6%）。
   第一次 P1 运行的 trace 里的 GEMM ms 会直接回答这个问题。
5. **两个必须先跟 operator 对齐的风险**：(a) e2e 本质上是一张卡上**持续的 bf16 GEMM 负载**，而
   `0927__b0/interference.md` B1 实测：邻卡 GEMM burn 会让被测卡的 attention **慢 3-11 倍，比值还会反转**。
   GPU2/3 上的 op-evolve 在 e2e 期间测出来的结果可能被污染（§6.2）。(b) 32 层配置显存在 88% 左右，
   arm 2 的常驻 scratch 会再多 +1 GiB（§6.3）。

## 1. 目标与口径（JIRA 截图）

| 项 | 值 | 来源 |
|---|---|---|
| 模型 / 精度 | Llama 3.1 8B，BF16 | jira_2 Config |
| 并行 | 1 GPU，MBS=GBS=4，seq 8192，fsdp0/tp1/pp1/cp1，AC=none | jira_2；jira_1 表头写的是 `len4k`，二者不一致，按任务以 8192 为准 |
| 指标 | tokens/GPU/s（稳态），外加单步拆分：GEMM ms、FA path ms（含 elementwise/cast 副作用）、其他 | jira_2 Results："FA path = flex 501 + elem 245 + softmax 9" |
| JIRA MI455X（eager，flex） | **19,795 tps**，GEMM 655 ms，FA path 755 ms | jira_2 |
| JIRA MI455X（compile+autotune） | 27,891 tps；attention kernel 414.3 ms/step（fwd 105.9 / bwd 308.4） | jira_1；`0915__opt/JIRA-TRACE-ANALYSIS.md` |
| JIRA MI355X（AITER） | 21,351 tps；attention 294 ms/step | jira_1/2 |

compile 保持关闭（对应 JIRA 的 MI455X eager 那一栏；另外 card-safety 规则 #2：inductor autotune 会挂卡），
所以直接对标的是 **19,795 tps / FA path 755 ms**。

## 2. Primus / TorchTitan 的 attention 路径

- Primus：`/home/lihuzhan/code/2026_0828__primus/Primus` @ `e7968675`（B0 上 2026-09-27 刚 clone，**B0 上还没跑过
  任何 e2e**）。torchtitan 0.2.2 = `Primus/third_party/torchtitan`（镜像里是 editable 安装，路径与 A0 相同）。
- 两层替换，缺一不可：
  1. patch `torchtitan.primus_turbo.turbo_attention`（`primus/backends/torchtitan/patches/turbo/attention_patches.py`）：
     条件是 `enable_primus_turbo && use_turbo_attention`，把 llama3 的 `Attention` 类换成 Primus 子类
     （`primus/backends/torchtitan/models/llama3/model/model.py`）。子类的 forward 跳过 `repeat_kv` 和 transpose，
     直接 `self.inner_attention(xq, xk, xv)`，布局 **BSHD** `[4, 8192, 32|8, 128]`，GQA 由 kernel 自己处理，
     不传 `enable_gqa`，也不传 mask。
  2. converter `primus_turbo`（`primus_turbo_extensions/primus_turbo_converter.py`）：把 `FlexAttentionWrapper` /
     `ScaledDotProductAttentionWrapper` 实例换成 `primus_turbo.pytorch.modules.TurboAttention(causal=True)`。
     它会 import `primus_turbo.pytorch.modules.TurboAttention` 和 `primus_turbo.pytorch.core.low_precision`
     的 `Float8QuantConfig` / `ScalingGranularity` —— **shim 要提供的就是这三个名字**。
- **flavor 必须是 `8B_flex`**（config 注释：默认的 sdpa 在 gfx1250 上会回退到 MATH，显存冲到 442 GB 后 SIGBUS）。
  **`converters` 不能设成 `[]`**：那样会退回 flex，而 flex 内部必定走 `torch.compile`，0915 这样挂过一次卡。
  也就是说 **本机没有安全的"flex 基线"**，arm 1 就是基线。
- 注意 `8B_flex` 的 `attn_mask_type=block_causal`（按文档切分的 mask），而 turbo 路径忽略 mask、做纯 causal。
  三个 arm 口径一致，但和 JIRA 的 flex 不是同一种 mask。
- **为什么 JIRA 里是 Skipped**：patch 的条件没满足。JIRA 用的 Primus 0730 / torchtitan 0.1.0 那套 MI455X 配置里
  `use_turbo_attention` 是 False（默认值 `config_extension.py:22` 就是 False），所以日志里是
  `[Patch] ⊘ Skipped: torchtitan.primus_turbo.turbo_attention (condition not met)`，走 Triton flex。
  同样的日志 0914 在 B0 上复现过（`0914__campaign/RESULTS.md` §10，`compile_mt` 那次）。当时就算 patch 打上了，
  primus_turbo 的 dense attention 也会路由到 aiter CK，而 CK 只支持 CDNA。
- **现在 gfx1250 上的产品默认**：Primus 的 preset `primus/configs/modules/torchtitan/pre_trainer.yaml:154-157` 已经是
  `enable_primus_turbo: true / use_turbo_attention: true`。Primus-Turbo 在 1cb2e183 和当前 HEAD（`05a326a1`）上，
  gfx1250 的 dense attention 都**只会**走 Triton：`DenseAttnFwdAiterBackend` 在 gfx1250 上拒绝；FlyDSL 被
  `_flydsl_common_ok` 卡死在 `is_gfx950()`；HipKittens / Gluon 只支持 gfx950。所以"turbo baseline"就是产品默认。
  - 不过 HEAD 的 Triton attention 和 1cb2e183 **不是同一份代码**：HEAD 删掉了 vendored fused bwd，
    `triton/attention/attention_kernel.py` 改了 223 行。任务定义的 baseline 是 1cb2e183（op 级 5.318 / 25.034 ms），
    所以 P1 用 `wt-bakeoff`。如果还想知道"今天的产品 HEAD"是什么水平，可以加一个 P1'（PYTHONPATH 换成主 checkout），
    多一次启动。
  - 0915 文档里的开关 `PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD` / `PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD` 在**现在任何一个
    checkout 的产品代码里都不存在了**（只有 `tools/gfx1250/tune_attention.py` 还在设置它们）。ASM 路径不能再靠
    环境变量打开，必须用 shim。

## 3. 三个 arm 怎么接入

### 3.1 arm 1 — turbo baseline（P1）

- PYTHONPATH：`/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff:/home/lihuzhan/code/aiter-src`
  （`primus_turbo/pytorch/_C*.so` 和 `lib/libprimus_turbo_kernels.so` 已经在 wt-bakeoff 里编好）。
  镜像里没有安装 primus_turbo，所以不会被遮蔽。flydsl 用镜像的 0.2.4（它的 FlyDSL 树在 gfx1250 上本来就不会被调用）。
- 不写任何 shim，走的就是产品路径：converter → `TurboAttention` → `flash_attn_func` → Triton fwd/bwd。
- FA path 归属：trace 里按 CPU op 的祖先关系归类（forward 看 `TurboAttention`/`FlashAttnFunc` 范围，backward 看
  `autograd::engine::evaluate_function: FlashAttnFuncBackward`）。第一份 trace 出来后要先确认这些名字真的出现。

### 3.2 arm 2 + arm 3 — 一个进程（P2），shim `primus_turbo`

要写的只有 `e2e/shim/primus_turbo/`（英文代码）：

```
primus_turbo/__init__.py                  # docstring only; asserts it is the shim (never the real package)
primus_turbo/pytorch/__init__.py
primus_turbo/pytorch/core/__init__.py
primus_turbo/pytorch/core/low_precision.py   # Float8QuantConfig / ScalingGranularity stubs (converter needs the names only; float8 off)
primus_turbo/pytorch/modules/__init__.py     # TurboAttention(nn.Module): forward(q, k, v, bias=None) -> o  (BSHD)
e2e/arms/fwd_r6/        <- copy of output/0927__b0/fwd/champion_r6      (flydsl 0.3.4.1, identical to fwd job op/current today)
e2e/arms/bwd_r20_0341/  <- copy of output/0925__flydsl/bwd341/op0341     (r20 kernels.py/impl.py byte-identical; only _env.py re-pinned to 0.3.4.1)
e2e/arms/asm/           <- copy of primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py + op/beat/impl.py logic
```

`TurboAttention` 是一个 `torch.autograd.Function` 包装，fwd/bwd 各自可以选 `asm` 或 `fly`，
由环境变量 `E2E_ATTN_SCHEDULE` 按训练步切换（fwd 调用次数 // 32 层 = step；ctx 记住 fwd 用的是哪个 arm，
bwd 跟着走）：

| arm | fwd | bwd |
|---|---|---|
| asm | `aiter.ops.mha.fmha_fwd_with_sink_asm(q,k,v,scale,True,True)` → `(o, lse[B,Hq,Sq] fp32 自然对数)` | `asm_backward(q,k,v,o,do,lse, hip=<进程级单例>, scratch=<一次分配、复用>, dkdv_heads="q", causal=True)` + 主机端 GQA 求和 `dk_q.view(b,s,hkv,g,d).sum(3)`，与 bwd job 的 `op/beat/impl.py` 完全一致 |
| fly | `fwd_r6/impl.py: attn_fwd(q,k,v,scale,True)` → `(o, lse[B,Hq,Sq] 自然对数)` | `bwd_r20_0341/impl.py: attn_bwd(do,q,k,v,o,lse,scale,True)` |

- 两种 lse 的约定相同（`[B,Hq,Sq]` fp32、自然对数、bottom-right causal，Sq=Skv 时等价于普通 causal），所以 fwd 和 bwd
  可以交叉组合（asm fwd + fly bwd 等）。以后想拆开看 fwd、bwd 各自在 e2e 里值多少，不用再改代码。
- 形状就是两个 job 打分用的 prod 形状（b4 s8192 hq32 hkv8 d128 bf16 causal BSHD）：这些 kernel 每天都在这个形状上
  上卡跑，没有新的 index 表达式。
- 进程级的前置步骤（shim import 时执行，早于任何 `import flydsl`）：`sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")`。
  **原因**：Primus 的 `runner/helpers/envs/base_env.sh:115-118` 会把 `${PRIMUS_IMPORT_ROOT}:${site_packages}` **插到
  PYTHONPATH 前面**，所以只靠 PYTHONPATH，`import flydsl` 会拿到镜像的 0.2.4。每棵树的 `_env.assert_environment()`
  都会检查版本和 `__file__`，在第一次调用时执行。
- **一个进程只有一个 flydsl 版本**：bwd r20 固定在 0.3.2，但 `0927__flydsl/bwd/BARRIER-CENSUS.md` §0 显示 champ 在
  0.3.2 和 0.3.4.1 下的 ISA 逐指令相同。我另外比对了 `census/j032_champ.json` 和 `j0341_champ.json`：三个 kernel
  （`k_delta_bshd_0`、`k_dkdv_0`、`k_dq_0`）在 prod 形状下的全部 98 个字段（ISA 字节数、VGPR、LDS、各类指令计数、
  scratch=0）都相同，只有 witness 里的 flydsl 版本和路径不同。`bwd341/op0341` 和 bwd job 的 `op/current` 相比，
  只有 `_env.py` 不同（`PROVENANCE.md` 仅 op/current 里有）。结论：**用 bwd341/op0341 在 0.3.4.1 下跑，是干净的做法**。
  建议第一次上卡前，在 fa-g0 里先不加锁跑一次 compile-only（`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250`），
  预热 `FLYDSL_RUNTIME_CACHE_DIR`，同时再确认一次 spill=0、scratch=0。
- **已实测的共存**（`recon/import_check.log`，fa-g0，2.9 s）：flydsl 0.3.4.1（来自 `~/.local/flydsl0341`）、fwd r6 树、
  bwd341 树、`aiter.ops.mha.fmha_fwd_with_sink_asm`（加载的是预编译的 `aiter-src/aiter/jit/module_fmha_fwd_with_sink_asm.so`，
  没有触发 JIT 构建）、按路径加载的 kernargs（`ASM_DIR=aiter-src/hsa/gfx1250/fmha_v3_bwd`，三个 `.co` 都在）、torchtitan
  llama3 和 Primus 的 `Attention` 子类都能一起 import，**`sys.modules` 里没有任何 `primus_turbo*`**，flydsl 始终是 0.3.4.1。
  另外注意：`aiter` 在 import 时就会初始化 CUDA（`torch.cuda.is_initialized()` 返回 True）。
- **Primus 配置要另外关掉三个 flag**（三个 arm 都关，保证口径一致）：preset 里默认
  `use_turbo_float8_linear: true` 和 `use_turbo_mx_linear: true`。它们各自的 patch 会在启动时 import
  `primus_turbo.pytorch.core.float8`、`primus_turbo.pytorch.modules.linear_fp8` 等 shim 里没有的模块，P2 会在启动阶段
  直接 ImportError。这两个 patch 只是注册 FP8/MX converter，对 BF16 训练没有影响，所以在 e2e config 里设
  `use_turbo_float8_linear: false`、`use_turbo_mx_linear: false`、`use_turbo_async_tp: false`。
  shim 的 `__init__` 遇到任何没提供的子模块，都要带着模块名响亮地报错。
- 一次性的交叉正确性检查（建议做）：step 1 第 0 层，同一组 q/k/v/do 把 asm 和 fly 两边都跑一遍，打印 o/dq/dk/dv
  之间的 SQNR（只在这一次，不计时）。此外 loss 曲线在 arm 切换点不应该出现台阶。这是免费的正确性哨兵。

### 3.3 已证伪的做法：三个 arm 放进同一个进程

`recon/import_check_turbo.log`：在 flydsl 0.3.4.1 下 import `wt-bakeoff` 的 `primus_turbo.pytorch`，会在
`pytorch/ops/attention/__init__.py → sparse_mla_interface → sparse_mla_impl.py:25 → primus_turbo/flydsl/attention/sparse_mla_bwd.py:26`
处抛 `ImportError: cannot import name 'buffer_ops' from 'flydsl.expr'`。1cb2e183 上这个 import 没有 try 保护
（HEAD 已经加了保护，但 HEAD 的 Triton attention 又不是 baseline 那一份）。所以 arm 1 只能单独一个进程（P1），
这和 campaign SKILL 规则 12 一致。

## 4. Harness 命令

### 4.1 单次运行（`e2e/bin/run_e2e.sh <tag> <arm-set> <config>`，待写）

```bash
PRIMUS=/home/lihuzhan/code/2026_0828__primus/Primus
E2E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e
# P1: ARM_PYTHONPATH=/home/lihuzhan/code/2026_0903__turbo/wt-bakeoff:/home/lihuzhan/code/aiter-src
# P2: ARM_PYTHONPATH=$E2E/shim:/home/lihuzhan/.local/flydsl0341:/home/lihuzhan/code/aiter-src
flock /tmp/b0-gpu0.lock timeout 2700 docker exec \
  -e GPUS_PER_NODE=1 -e NNODES=1 -e NODE_RANK=0 -e PRIMUS_GPU_MODEL=MI455X \
  -e MASTER_PORT=$((20000 + RANDOM % 20000)) -e PRIMUS_EXP_NAME=$TAG -e E2E_RUN_MARKER=$TAG \
  -e TRITON_CACHE_DIR=/tmp/triton_cache_e2e \
  -e ARCH=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_e2e \
  -e E2E_ATTN_SCHEDULE="$SCHEDULE" \
  fa-g0 bash -c "ulimit -c 0; \
    export TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250; \
    export PYTHONPATH=$ARM_PYTHONPATH; \
    cd $PRIMUS && exec timeout --foreground -k 20 2600 bash runner/primus-cli direct \
      --log_file /home/lihuzhan/_dbg_l8b/$TAG/launcher.log -- train pretrain --config $E2E/configs/$CFG" \
  > $E2E/logs/e2e.$TAG.log 2>&1
timeout 20 sudo -n dmesg | tail -20      # ignore 0002:04:00.0 (GPU1); any new 0001:04:00.0 line = stop everything
```

### 4.2 相对 `0915__opt/bin/e2e.sh` 必须改的地方

1. **`bash -lc` → `bash -c`，并且在命令字符串里显式 export BLAS 变量**。容器里的 `/etc/profile.d/zz-gfx1250.sh`
   会 `export TORCH_BLAS_PREFER_HIPBLASLT=0`。login shell 会执行它，把 `docker exec -e` 传进来的值覆盖掉，
   结果静默退回 rocBLAS（0914 实测 rocBLAS 比 hipBLASLt 慢约 8×）。另外 `/usr/lib/python3.12/sitecustomize.py`
   里有一个 `setdefault(...,"0")`，它只在变量没设置时才生效。训练进程里要打印 BLAS env，并在第一次 matmul 后打印
   `torch.backends.cuda.preferred_blas_library()` 作为证据（shim 的 import 就是一个自然的挂点；P1 可以在 config 的 log 里 grep）。
   注意：fwd 树的 `_env.py` 会**赋值** `PREFER=1` 和宿主库路径，和这里一致；bwd 树的 `_env.py` 是 `setdefault("0")`，
   在已经设置的情况下是 no-op。所以 P2 里 BLAS 不会被树中途翻转，但仍然以进程内打印为准。
2. **不要全局 reap/等待 KFD**：旧脚本会遍历 `/sys/class/kfd/kfd/proc` 杀 torchrun，并一直等到 KFD 进程 ≤1。
   B0 上 GPU2/3 的 op-evolve 一直持有 KFD，这个循环永远等不完，杀进程的 pattern 也可能误伤别人。改成只看
   `docker top fa-g0`（本容器里残留的 torchrun / `primus/cli/main.py`），按 PID 用 `docker exec fa-g0 kill <pid>` 处理。
3. **时钟采样不要用 `rocm-smi`**：GPU1 已挂，rocm-smi 会遍历所有卡，可能挂住或打扰别人。改读
   `/sys/class/drm/card0/device/pp_dpm_sclk`（带 `*` 的那一档）和 card0 的 hwmon 功率/温度，每 5 s 一次。
   card0 = GPU0，见 LAB-RULES 表。
4. 不要设置 `HIP_VISIBLE_DEVICES`（LAB-RULES 规则 1），容器名 `fa-repro` → `fa-g0`。Primus 的 base_env 自己会
   export `HIP_VISIBLE_DEVICES=0`，在 fa-g0 里只有逻辑 0，没有影响。
5. 保留原来的 nan 检查（loss 出现 nan 的运行作废）和 `watch_e2e.sh`（每 30 s 一次，没事不出声）。
   `watch_e2e.sh` 里 dmesg 的 grep 要**排除 `0002:04:00.0`**：GPU1 每秒都在刷 `MES failed to respond`，不排除会一直误报。

### 4.3 config（放在 `e2e/configs/`，从 `repro_l8b_turbo_conv.yaml` 复制一份再改）

- 继承：`flavor: 8B_flex`、`hf_assets_path: /home/lihuzhan/_hfassets/llama31_8B`（B0 上存在）、`converters: ["primus_turbo"]`、
  `debug.seed: 1234`、`compile.enable: false`、AC none、mock data、MBS=GBS=4、seq 8192、`metrics.log_freq: 1`。
- 修改：`primus_turbo.use_turbo_float8_linear/use_turbo_mx_linear/use_turbo_async_tp: false`（§3.2）；
  步数 P1 30 步 / P2 44 步；`profiling.enable_profiling: true`，`profile_freq` 设成让每个 arm 至少有一个被 profile 的
  稳态步（被 profile 的步不计入 tps）。
- 未验证：Primus 能不能接受放在 repo 外的绝对 config 路径（preset 是按包根目录解析的，看起来可以）。如果不行，
  在第一步前的 config 解析阶段就会报错，不占 GPU；退路是放到 `examples/torchtitan/configs/MI455X/`
  （那个目录本来就是我们自己的、没被 git 跟踪的）。

## 5. 测量设计

- **tps**：torchtitan 每步都会打 `tps:`（log_freq 1）。稳态窗口取 step ≥ 8，去掉被 profile 的步，报中位数和
  (max-min)/median（`0915__opt/bin/steady.py`，按 arm 分组稍改一下）。A0 的经验是：step 4-8 和 step 8-20 的中位数
  差 ≤ 0.22%，所以 warm-up 取 5-7 步就够了。
- **P2 schedule**：step 1-4 = asm, fly, asm, fly（两边的 JIT 编译和 scratch 分配都在这里完成），之后按每 4 步一块
  做 ABBA 回文：`A B B A A B B A …`。按块内同一 arm 的中位数配对比较，不跨进程比较。一次 P2 出 arm 2 vs arm 3
  的比值和各自的 tps。
- **单步拆分**（JIRA 的口径）：用 kineto trace 抓稳态的一步。GEMM = `Cijk_*` / hipblaslt / `*gemm*` kernel；
  **FA path = 从 attention 的 CPU 范围（P2 在 shim 里用 `record_function("e2e::attn_fwd/bwd")`；P1 用 autograd 节点名）
  发出的所有 GPU kernel**，所以 ASM 的 `odo`/`dq_convert`、`dq_acc.zero_()`、GQA 求和、`lse.float()`，FlyDSL 的
  `delta`/`redsp`，Triton 的预处理，以及任何 `.contiguous()`/cast 都会算进去，和 JIRA "elem 245 ms 是 flex 副作用"
  的算法一致；其余 = other。要报的三个量：arm 1 → arm 2 → arm 3 的 FA path ms/step，以及 other 有没有变化
  （这里用来判断"有没有带来别的开销"）。起点可以用 `0915__opt/bin/trace_attn.py`，但它是按 kernel 名归类的，
  会漏掉副作用 kernel，要换成按 CPU 祖先归类。
- **什么情况下再跑 e2e**：fwd 或 bwd 的 champion 相对上次 e2e 用的版本提升超过 5% 时，快照新树到 `e2e/arms/…_rN`，
  跑一次 P2，schedule 为 `asm / fly_prev / fly_new`（或者只换有变化的那一半）。P1 不用重跑。

## 6. 风险

### 6.1 GEMM 吞吐决定能不能看到 attention（最大的未知）

- 已知数字：B0（0914）用镜像里的 hipBLASLt 库 113-120 TF/s，flex eager 只有 2,394 tps（完全被 GEMM 卡住）；
  A0 用修过路径的镜像库，32L 是 2,027 tps。现在 `~/.local/hipblaslt-gfx1250`（宿主 ROCm 10.1.0，328 个文件）
  只验证过"能跑、不 fault"（`0923__flydsl/STAGE2-S0-PROBE.md` S0-b），**吞吐没测过**。
- JIRA MI455X 的 GEMM 是 655 ms/step，按 32768 token 算相当于约 2 PF/s 的有效吞吐。如果我们这边的 GEMM 慢一个数量级，
  那么 arm 1 → arm 2 的差距（约 0.7 s/step）还能看出来，但 arm 2 vs arm 3（约 80 ms）会被稀释到 1% 以下。
  这时 trace 里的 FA path ms 仍然成立，tps 只能作为旁证。
- 不在计划里的备选方案，需要用户点头：compile + `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`（0914 在 B0 上跑到
  9,599/13,204 tps，但 card-safety 规则 #2 记录它挂过卡）；nkfix（A0 上 21% 的运行 loss 变 nan，p=0.004，不能交付）。

### 6.2 e2e 的 GEMM 负载会干扰 GPU2/3 上的 op-evolve

- `interference.md` B1：其余 3 张卡跑持续 bf16 GEMM 时，被测卡 fwd 从 1.49 ms 变成 16.8 ms，bwd 的 FlyDSL/ASM 比值
  从 0.74 变成 1.55（反转）。只有一张卡（e2e）时影响有多大没测过，但方向是明确的。LAB-RULES 规则 6 也禁止
  持续的 GEMM 负载，而用户这次明确要求在 GPU0 上跑 e2e，两者冲突，需要 operator 裁决。
- 建议：(a) e2e 只在两个 job 都处于非上卡阶段（LLM 的 opt/review 阶段）时启动；或者 (b) 至少事后检测：
  e2e 时间窗内完成的 round，看 `act.yaml` 里 beat（ASM）的 ms。fwd 正常在 1.14-1.19 ms、bwd 在 6.41-6.58 ms 的带内；
  偏离超过 2% 就重测那一轮。beat 是固定的 kernel，天然就是干扰计。
- 反过来，邻卡的 attention 负载对 e2e 基本无害（attention 之间的干扰已测过，约 0.5-1%）。

### 6.3 显存和启动风险

- 32L 峰值约 88%（A0：380 GiB）。A0 上 +1.38 GiB 常驻 scratch 那次在 88.30% 时 SIGBUS，还泄漏了 KFD 上下文，需要
  AC-cycle。后来又有一次 SIGBUS 发生在显存只有 6.93% 时，所以"显存导致 SIGBUS"的归因被削弱，但没有排除。
  P2 会同时持有 asm 的 scratch（常数 1 GiB：dq_acc fp32 0.5 + 按 q head 的 dk/dv 各 0.25）。建议在 step 1 读
  torchtitan 的 `memory:`，超过 88.5% 就在下一步边界主动停（watchdog 加一条阈值规则）。
- 启动次数就是风险（card-safety：A0 上 14 次启动挂了 3 次，按启动次数计价）。本计划首轮只需要 **2 次启动**（P1、P2）。
  每次运行后都查 dmesg；0001:04:00.0 上出现任何新行，就停掉全部卡上的工作并上报。
- 观察到的情况（需要上报）：今天 dmesg 在 69813 s 有 `amdgpu: process pid 2015598 DQM create queue type 0 failed. ret -110`，
  69825 s 有 `python3[2015598] general protection fault ... libamdhip64.so.7`（没有 PCI 标签，不是我们的进程，
  发生在我的检查之前）。随后 0002:04:00.0（GPU1）持续刷 `MES(0,0) failed to respond`。另外本次侦察期间 fa-g0 里
  有别的 operator 在跑 ruler 的 benchmark（`ruler/runs0/A03.json`），flock 正常起作用。

## 7. 先验数字与预测（per step，32 层）

| arm | fwd ms/层 | bwd ms/层 | attention ms/step | 来源 |
|---|--:|--:|--:|---|
| 1 turbo stock（1cb2e183） | 5.318 | 25.034 | **971** | `2026_0927__bak/PERF-A0-vs-B0.md` |
| 2 aiter ASM | 1.144 | 6.504 | **245** + 副作用 | 同上（harness beat） |
| 3 FlyDSL r6 + r20 | ≈1.443（prod 1524 TF/s） | 8.716 | **325** | `STOPPED.md` fwd r6、PERF 表 bwd r20 |

- arm 1 → 3 预计节省约 **650 ms/step**，arm 3 比 arm 2 慢约 **80 ms/step**（fwd r6 仍是 ASM 的约 0.79×，bwd r20 约 0.75×）。
- op 级测的是纯 kernel；e2e 里还要加上自动求导的管路（A0：ASM bwd 在 harness 里是 10.10 ms，独立 bring-up 是 8.68 ms，
  差 1.4 ms/层，来自 ctx 存取和 `do.contiguous()`）。arm 2 的副作用（dq_acc 清零 536 MB、GQA 求和、dq_convert）
  在 trace 里能直接看到；arm 3 的 bwd 直接输出 bf16，不需要主机端 GQA 求和，副作用应该更少。
  **e2e 里 arm 2 与 arm 3 的差距有可能小于 op 级的 80 ms**，这正是这次 e2e 要回答的问题。
- tps 的换算：如果 B0 的单步是 JIRA 量级（约 1.7 s），每省 80 ms 约等于 +5% tps；如果单步 10 s，只有约 +0.8%。

## 8. 首轮执行顺序（建议）

1. 写 shim、arm 副本、config、`run_e2e.sh`、分析脚本（全部在 CPU 上）；在 fa-g0 里对 bwd_r20_0341 和 fwd_r6
   跑 compile-only，预热缓存（不加锁）。
2. 先和 operator 对齐 §6.2（e2e 启动的时间窗）。
3. **P1**（arm 1，30 步，profile 一步）：得到 baseline 的 tps、GEMM ms、FA path ms。§6.1 的问题在这里得到回答。
4. **P2**（arm 2/3 交替，44 步，profile 两步）：得到同进程的 arm 2 vs arm 3 比值，以及三个 arm 的单步拆分表。
5. 用中文表格汇报（tps、相对 arm 1、FA path ms、GEMM ms、other ms、显存峰值、sclk）。
