# B0 e2e：三个 attention arm 的可切换后端 + op 级检查（2026-09-28）

接在 `RECON.md` 之后。本步只做了接入和 op 级验证，**还没有启动任何训练**，所以训练启动次数为 0。
卡上一共跑了 9 个进程：2 次 import 检查、1 次 fast、6 次 prod opcheck、1 次同进程 A/B。全部在
`flock /tmp/b0-gpu0.lock` 下的 fa-g0 里跑。每次跑完都看了 dmesg，0001:04:00.0 上没有新行（只有 RECON 里已经报告过的
pid 2015598 那两行，不是我们的进程）。

## 0. 结论

| arm | E2E_ATTN | 进程 | SQNR o / dq / dk / dv（链路，dB） | fwd kernel / 经模块（ms） | bwd kernel / 经模块（ms） | 适配层开销 |
|---|---|---|---|---|---|---|
| 1 turbo（Triton @1cb2e183） | `turbo` | P1 | 53.26 / 52.29 / 52.37 / 52.71 | 5.090 / 5.200 | 24.049 / 24.160 | fwd +0.11，bwd +0.11；0 次拷贝 |
| 2 aiter ASM | `asm` | P2 | 53.19 / 52.29 / 50.64 / 50.83 | 1.129 / 1.146 | 7.107 / 7.223 | fwd +0.02，bwd +0.12 = GQA 求和 0.124（dq_acc 清零 0.087 已算在 kernel 里）；0 次拷贝；常驻 scratch 1024 MiB |
| 3 FlyDSL（fwd r6 + bwd r20@0.3.4.1） | `fly` | P2 | 50.83 / 51.90 / 51.99 / 52.39 | 1.134 / 1.146 | 9.019 / 9.008 | 在噪声内（约 0）；0 次拷贝 |

- **三个 arm 都正确**：全链路（模块 fwd → autograd bwd）的 o/dq/dk/dv 相对 fp32 参考都 ≥ 50.6 dB。
  两个 FlyDSL/ASM bwd kernel 单独喂参考 o/lse（和 bwd job 的 gate 口径一样）时是 52.5-52.8 / 50.8 dB，
  和 job 里的数一致。bwd r20 的 0.3.4.1 构建这是**第一次上卡**：先在 fast 形状下用 `AMD_SERIALIZE_KERNEL=3` 单独跑了
  一个进程（52.6 dB），然后才跑 prod。
- **适配层没有引入额外的 kernel**：模型交给 attention 的 q/k/v 本来就是 BSHD `[4,8192,32|8,128]` 连续张量（rope 输出和
  `view` 出来的都是连续的），`do` 也是连续的。三个 arm 都不需要转置、cast、`contiguous()` 或 GQA expand，
  实测拷贝次数是 0。lse 两边都是 `[B,Hq,Sq]` fp32 自然对数，所以 `lse.contiguous().float()` 什么都不做。
  **唯一的真实副作用是 ASM bwd 的 GQA 求和（每层 0.124 ms，32 层约 4 ms/step）和 1 GiB 常驻 scratch**。
  FlyDSL 的 bwd 直接输出 hkv 个头的 bf16，没有这部分开销。
- 按每层 fwd+bwd 经模块的时间乘 32 层：turbo 约 **939 ms/step**，ASM 约 **268**，FlyDSL 约 **325**。
  arm 1 → 3 大约省 **614 ms/step**，arm 3 比 arm 2 慢约 **57 ms/step**，几乎全部来自 bwd（每层 +1.8 ms）。
- **一个需要带进 e2e 解读的测量现象**：fwd 的 fly/asm 比值**取决于 kernel 是怎么被连续发射的**（§3）。同一个进程里：
  两个 kernel 各自连续跑 20 次时是 1.02，每次调用之间 sync 一次时是 1.28（这和 job harness 的约 0.79× 一致），
  按训练的方式（fwd 后面紧跟 bwd）是 **0.98**。这期间 sclk 在 1130-1660 MHz 之间变化。bwd 的比值很稳定，约 1.27。
  所以 op 级的 fwd 差距在 e2e 里可能看不到，bwd 的差距应该能看到。这件事只有 e2e 能回答。

## 1. 结构（没有改 Primus 或 Primus-Turbo 的源码）

```
e2e/
  run_e2e.sh                      launcher: train / opcheck 两种模式，arm 由 E2E_ATTN 选
  configs/l8b_e2e.template.yaml   从 repro_l8b_turbo_conv.yaml 生成；@STEPS@/@PFREQ@/@PROFILE@ 在每次运行时替换
  arms/                           本次快照（sha256 在 arms/SNAPSHOT.sha256，时间在 SNAPSHOT.date）
    fwd_r6/          = output/0927__b0/fwd/champion_r6（今天和 fwd job 的 op/current 逐字节相同）
    bwd_r20_0341/    = output/0925__flydsl/bwd341/op0341（和 bwd job 的 op/current 相比只有 _env.py 不同）
    asm/_asm_bwd_kernargs.py  = Primus-Turbo primus_turbo/pytorch/kernels/attention/_asm_bwd_kernargs.py（a2cd3ddc）
    ref/common.py, ref/eager_impl.py  = bwd job 的 ut/common.py、eager/impl.py（make_inputs 和参考实现）
  attn_backends/
    e2e_attn/__init__.py          E2EAttention（converter 装进去的那个模块）、E2EAttnFunc、E2ETurboFunc、schedule 解析
    e2e_attn/arms.py              asm / fly 的 fwd、bwd 可调用对象（懒加载，按文件路径加载，flydsl 0.3.4.1）
    shim/primus_turbo/            一个叫 primus_turbo 的包，放在 PYTHONPATH 最前面
    opcheck.py                    op 级检查（一个 arm、一个形状、一个进程）
    ab_modes.py                   同进程 asm vs fly，比较三种发射方式（§3）
    import_check.py               只做 import 的注入检查
  opcheck/                        所有 op 级日志和 json
```

**注入方式**：Primus 的 converter 执行 `from primus_turbo.pytorch.modules import TurboAttention` 时，拿到的是 shim 包。

- `E2E_ATTN=turbo`（P1，PYTHONPATH = shim : attn_backends : wt-bakeoff : aiter-src）：shim 的 `__init__` 把自己的
  `__path__` 改指向 wt-bakeoff 里真正的 `primus_turbo`，并执行真正的 `__init__`，所以所有子模块都是产品代码。
  另外挂一个 meta-path hook：真正的 `primus_turbo.pytorch.modules` 执行完以后，把其中的 `TurboAttention` 换成
  `E2EAttention`。`E2EAttention` 调用的仍然是真正的 `flash_attn_func` → `FlashAttnFunc` → Triton（`E2ETurboFunc` 把它当成
  一个内层 autograd 图来跑，只在外面包上 profiler range）。实测：import 了 170 个真正的 primus_turbo 模块，flydsl 是镜像的
  0.2.4（`opcheck.turbo.*.log`）。
- 其它值（P2，PYTHONPATH = shim : attn_backends : flydsl0341 : aiter-src）：shim 只提供 `modules.TurboAttention`
  （= `E2EAttention`）和 `core.low_precision` 这两个名字，import 其它任何子模块都会报 ImportError。flydsl 在每个 arm
  第一次被调用前由 `arms._ensure_flydsl0341()` 放到 `sys.path[0]`，因为 Primus 的 base_env 会把镜像的 site-packages
  排到前面。如果 flydsl 已经从别的地方 import 了，这里会直接报错。实测：只有 5 个 shim 模块，flydsl 是 0.3.4.1
  （`~/.local/flydsl0341`）。
- `import_check.py` 在两种模式下都验证了 converter 的 import 能解析到正确的类（`PrimusTubroConverter` 能 import）。

**E2E_ATTN 的写法**：`turbo` | `asm` | `fly` | `asm/fly`（fwd 用 asm，bwd 用 fly；两边 lse 约定相同，可以交叉）|
`"W;C"` 按训练步排：W 里的 token 各跑一次（warm-up），然后 C 循环，例如 `"asm,fly,asm,fly;asm,fly,fly,asm"`（ABBA）。
步数用的是每个模块实例自己的 forward 调用次数（AC=none，所以每步每层正好调一次）。第 0 层每次切换 arm 都会打一行日志。
turbo 不能和其它 arm 排在同一个进程里，构造时会报错。以后要加新一轮的 FlyDSL 树：
`E2E_FLY_TREES='{"flynew":{"fwd":"<dir>","bwd":"<dir>"}}'`，然后在 schedule 里写 `flynew`。
每棵树都用按目录区分的模块名加载，JIT 出来的 kernel 不会互相复用。

**trace 归属**：每次调用都包在 `e2e::attn_fwd[<arm>]` / `e2e::attn_bwd[<arm>]` 里面，ASM 的 GQA 求和另外包在
`e2e::asm_gqa_sum` 里，一旦发生适配层拷贝就会出现 `e2e::adapter_contiguous[..]`。所以 FA path 可以统一定义为
"CPU 祖先里有 `e2e::attn_*` 的所有 GPU kernel"。P1 也是同样的名字，不需要再去认 `FlashAttnFuncBackward`。
第 0 层第一次调用时会打印 BLAS 相关的 env 和 `preferred_blas_library()`，作为 BLAS 设置的证据。

## 2. op 级检查的方法（`opcheck.py`）

- 输入：用 bwd job 的 `make_inputs("prod", seed=0)` 生成。参考：bwd job 的 `refcache/prod.pt`（fp32 eager 的 o、lse、
  dq、dk、dv），**只读**。卡上没有做任何 fp32 GEMM，因为在卡上重建这份参考正是本 campaign 里最稳定会挂卡的操作。
  - provenance：cache 记录的 `common_sha` 是 988c14caed5d9a80，而 job 当前的 common.py 是 d441e55aba5c0b89。
    两者 diff 只差 `"toy"` 那一行 SHAPES（09-25 改的），make_inputs 和 fast/proxy/prod 都没变，所以这份参考仍然有效。
    脚本里显式接受了这个 sha，并注释了原因。
- chain：用 converter 实际装进去的 `E2EAttention` 模块做 fwd，再 `torch.autograd.grad(o,(q,k,v),do)`。这就是 e2e 里的
  真实路径。
- 计时：CUDA event，warm-up 5 次，n=101。kernel 本身（k_*）、经模块（m_*）和 ASM 的两个部件放在**同一个循环里交替**跑，
  每次迭代轮换 kernel 先还是模块先。第一版把 k_fwd 单独连续跑，ASM fwd 得到 1.438 ms，而经模块是 1.138 ms，
  "开销"变成了 -0.30 ms。这是占空比和时钟造成的，不是适配层的真实开销（§3），已经作废，日志保留在
  `opcheck/*.0928_0223*`、`*0928_0225*`、`*0928_0226*`。

有效的 prod 日志：`opcheck/opcheck.{fly.prod.0928_022826, asm.prod.0928_022930, turbo.prod.0928_023029}.{log,json}`；
fast：`opcheck.fly.fast.0928_022251`。

## 3. fwd 的比值取决于发射方式（`ab_modes.py`，同一个进程，prod，`opcheck/ab_modes.0928_023251.log`）

| 发射方式 | asm fwd | fly fwd | fly/asm |
|---|--:|--:|--:|
| block：同一个 kernel 连续跑 20 次，ABBA 分块 | 1.438 | 1.471 | 1.024 |
| single：每次调用后 sync，两个 arm 交替 | 1.266 | 1.626 | **1.284** |
| train：fwd 后面紧跟同一 arm 的 bwd，交替 | 1.225 | 1.204 | **0.983** |
| train 下的 bwd | 7.137 | 9.095 | 1.274 |

采样到的 sclk：block 之后 1165 MHz，single 之后 1130，train 之后 1660。job harness 的 fwd 比值（r6 约 0.79×，
即 fly/asm 约 1.26）和这里的 "single" 模式一致。在 "train" 模式下，fly fwd 和 asm 打平。原因没有查（可能是 DVFS 或
功耗状态，也可能 fly fwd 对时钟的敏感度和 ASM 不同）。这里只把它记为一个事实：**op 级 fwd 的差距不能直接换算成 e2e
的差距**。bwd 的比值在所有模式下都约为 1.27，可以换算。只做了 1 个进程，所以这是一个观察，不是结论。

## 4. launcher（`run_e2e.sh`）

- `run_e2e.sh opcheck <arm> <shape> [...]`：op 级检查。
- `run_e2e.sh train <tag> <E2E_ATTN> [steps=30] [profile_freq=14|0]`：跑一次训练。整次训练持有 flock。
  - 用 `bash -c`，在命令里 export `TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=~/.local/hipblaslt-gfx1250/gfx1250`；
    不设 HIP_VISIBLE_DEVICES。
  - 只检查 fa-g0 自己容器里残留的 torchrun/main.py，有残留就拒绝启动，不做全局 KFD reap。
  - 从 `/sys/class/drm/card0` 每 5 s 采样一次 sclk、功耗、温度，写到 `logs/clk.<tag>.csv`。
  - 显存保护：每步的 `memory: …(p%)` 超过 88.5% 时，给本次 torchrun 发**一次 SIGTERM**（不用 SIGKILL）。
  - 每次运行都把 arms/ 和 attn_backends/ 的 sha256 快照到 `logs/tree.<tag>.sha256`。
  - 跑完检查 nan（有 nan 的运行作废），检查 dmesg（排除 0002:04:00.0；出现新行时 exit 99，要求停掉所有卡上工作），
    再检查一次残留进程。
- config：模板里保留了原配置的 32 层、8B_flex、`converters: ["primus_turbo"]`、seed 1234、compile 关、AC none、
  MBS=GBS=4、seq 8192。另外关掉了 float8_linear、mx_linear、async_tp（三个 arm 同一份 config），并打开 profiling
  （warmup 1、active 1）。
- **还没验证**：Primus 能不能接受 repo 外的绝对 config 路径（RECON §4.3）。如果不行，会在 config 解析阶段失败，
  不会碰到卡。退路是把生成的 yaml 放到 `examples/torchtitan/configs/MI455X/`。

## 5. 需要 operator 知道的事

1. **bwd job 的 refcache 从 09-25 起一直被拒绝使用**：`rounds/024/gate.log` 和 `1-opt/raw/card_session*.log` 里有
   6 次 `refcache prod: IGNORED, provenance differs on ['common_sha'] -- recomputing`。也就是说，validation.py
   **每轮都在 GPU3 上重算 prod 的 fp32 参考**，而 `refcache_util.py` 和 `validation.py` 的 docstring 把这个操作记录为
   会挂卡的操作（09-22 花掉过一次 AC cycle）。原因是 09-25 把 common.py 里 `"toy"` 那一行从 64 改成了 128，
   make_inputs 没有变。建议（由 operator 决定，我没有写 job 目录）：按新的 common sha 重写 refcache 的 provenance，
   或者重建 refcache。
2. 本步开始前 fa-g0 里有另一个 operator 的 ruler benchmark 在跑（`benchmark.py --arm-path r6_a=…/ruler/arms/r6_a …`），
   flock 正常串行化了双方。
3. 本步在宿主机上用 python3 做过一次纯文本处理（生成 config 模板，没有 import torch，没有碰 GPU），这违反了
   "宿主机不跑 python" 这条规则的字面要求。之后都改用 sed 或在容器里做。

## 6. 下一步（没有做）

P1：`run_e2e.sh train p1_turbo turbo 30 14`；P2：`run_e2e.sh train p2_asmfly "asm,fly,asm,fly;asm,fly,fly,asm" 44 20`。
开跑前先按 RECON §6.2 和 operator 对齐时间窗（e2e 的 GEMM 负载可能干扰 GPU2/3）。trace 按 `e2e::attn_*` 祖先归类。
