# 行覆盖率 95%：缺口分析与模块补测

## 口径与目标

基线为 2026-11-06 的 `primus-turbo-coverage-postfix-20261106` 报告：

- 行覆盖率 **63.42% = 7,312 / 11,529**，缺失 4,217 行。
- 分支覆盖率 51.30%；coverage.py 显示的综合 61.12% 不是行覆盖率。
- 全包 95% 至少需要 10,953 行命中，即较基线新增 **3,641 行**。
- CI 合并 PyTorch 与 JAX 两组覆盖数据。本次基线只执行 PyTorch，JAX 的 **1,237 行全部未命中**；即使其他代码全部命中，上限也只有 **89.27%**。
- 只看 `primus_turbo/pytorch`，基线是 7,029/9,939 = **70.72%**。这和全包目标是不同口径，不能混用。
- `pyproject.toml` 排除 Triton、FlyDSL、Gluon JIT 实现；Python 包装层计入。C++/HIP 不由该报告衡量。本轮不修改这些分母或排除规则。
- 基线全量运行还包含后来撤回的 Attention 测试修改。新增测试的结果单独保存；累计增量只对与基线逐字节一致的生产源码合并，不能视为当前版本重新跑过全量。

## 本轮验证结果

2026-11-06，gfx950 / MI350X，`rocm/primus:v26.8`，生产源码基于 `3514ad7`：

- 新增 **16 个测试文件、260 个参数化测例**：单卡/host 242 项，双卡 18 项，全部通过。
- 另有原有全局变量检查 1 项、grouped MLP 采集对照 39 项通过；不重复计入新增测例数。
- 新增测试命中原缺口 **536 行**：基线加新增测试为 **68.07%（7,848/11,529）**。
- 既有 grouped MLP 测试在调用线程采集 autograd 后，再补记 **130 行**。
- 三部分累计 **69.20%（7,978/11,529）**；到全包 95% 仍差 **2,975 行**，尚未达标。
- 已对报告中的 **145 个生产文件**验证基线与当前内容逐字节一致；所有分母/排除规则保持不变。上述是累计覆盖，**不是本轮重新执行完整 PyTorch/JAX 全量的结果**。

| 代表模块 | 原行覆盖率 | 累计行覆盖率 |
|---|---:|---:|
| 环境配置 | 73.33% | 100% |
| 日志 | 81.69% | 100% |
| Stream 包装层 | 25.00% | 100% |
| QuantizedTensor | 77.35% | 95.58% |
| DeepEP 兼容层 | 33.33% | 96.49% |
| FP8 AllToAll | 0% | 95.95% |
| grouped GEMM 调优辅助 | 53.76% | 100% |
| grouped GEMM FP8 实现 | 48.12% | 89.11% |
| RMSNorm 实现 | 50.88% | 100% |

## 已补充的测试

测试按职责分别放置，正常进入 `pytest tests/pytorch` 收集。没有修改生产代码、gfx1250 专用文件、Attention 参考实现或现有数值容差。

| 模块 | 新增文件 | 验证的行为 |
|---|---|---|
| 公共配置 | `tests/pytorch/common/test_envs.py` | 未设置/空值、非法整数、列表解析、上下界、缓存失效、禁用/恢复缓存、未知变量 |
| 日志 | `tests/pytorch/common/test_logger.py` | 日志级别解析、格式参数传递、rank 过滤不消耗 once、按级别去重、分布式未初始化/异常回退 |
| AITER 依赖 | `tests/pytorch/common/test_aiter_utils.py` | 本地版本后缀、非法版本、缺失发行包、版本不符仅警告一次、导入错误保留 cause 和安装提示 |
| Stream | `tests/pytorch/core/test_stream_contract.py` | 设备规范化、CU mask 默认值/截断、非法设备、未初始化 runtime、资源释放和失败警告 |
| QuantizedTensor | `tests/pytorch/core/test_quantized_tensor_layout.py` | FP4 packing、MX scale 对齐、view/reshape 保留存储、ATen view、布局变换反向梯度、非方阵权重缓存数值 |
| 对称内存 | `tests/pytorch/core/test_symm_mem_contract.py` | peer 和容量边界、字节偏移、IPC 打开失败后的清理、幂等释放、异常上下文退出、图捕获禁止扩容、workspace 复用 |
| 调优缓存 | `tests/pytorch/core/test_tune_cache.py` | LRU 访问/更新顺序、淘汰警告、清空、缺失查询 |
| grouped GEMM 调优 | `tests/pytorch/kernels/test_grouped_gemm_tuning.py` | 只在 profiling 使用均衡组、原始数据不变、异常/不支持的 backend 不成为赢家、缓存复用、variable-K 的 trans_c |
| grouped GEMM FP8 调度 | `tests/pytorch/kernels/test_grouped_gemm_fp8_dispatch.py` | tight output 限制、CK 未编译、累加输出的 dtype/连续性、MX 架构/布局/对齐、scaling 与累加组合 |
| grouped GEMM FP8 Meta | `tests/pytorch/kernels/test_grouped_gemm_fp8_meta.py` | 通过注册的 custom op 验证 Meta 路径，logical M/N、trans_c、无效 dtype/rank/layout、GLU/dGLU 激活与尺度形状 |
| grouped GEMM Meta | `tests/pytorch/kernels/test_grouped_gemm_meta.py` | BF16/FP16、转置、variable-K 输出维度、不支持布局拒绝 |
| GEMM FP4 Meta | `tests/pytorch/kernels/test_gemm_fp4_meta.py` | 布局与输出转置、K 不匹配、原位累加形状、GLU/dGLU 两个方向的 packing/padding |
| RMSNorm | `tests/pytorch/kernels/test_rmsnorm_backward.py` | 真实 GPU multi-row/grid-stride backward，zero-centered、残差两个输出的梯度、大小 partial reduction，对照 PyTorch autograd |
| DeepEP 兼容层 | `tests/pytorch/deep_ep/test_compat_config.py` | chunk 参数边界、RDMA 对齐、buffer hint 容量、非法 rank/SM、真实 CUDA event 的记录和等待 |
| FP8 AllToAll | `tests/pytorch/ops/test_fp8_alltoall_small.py`、`tests/pytorch/core/test_fp8_alltoall_validation.py` | 真实双卡 NCCL，等长/不等长切分，前后向独立量化，E4M3/HYBRID，反向恢复原切分、空输入/空输出、无效配置提前拒绝 |

### 测试边界

- Stream 与对称内存的 host 契约测试 mock 驱动/通信边界，断言参数、偏移和清理；不将其解读成真实 HIP stream/IPC 协议验证。既有 Stream runtime 并发挂起问题未解除。
- Meta 与 `can_handle` 测试只验证编译/调度契约，不替代 GPU 数值或性能测试。
- AllToAll 双卡测试使用 rank 不同、amax 相同的数据，把传输/切分/反向与跨 rank scale 协议分开验证。不同 rank 的 scale、量化状态下的零 token、不对称极端负载仍需专门补测；已补不量化时的空输入/空输出。
- Block/MX weight cache 改变量化轴，`dequantize()` 恢复原逻辑矩阵；tensor/rowwise cache 显式转置逻辑矩阵。非方阵测试区分这两种契约。

## 重要发现：现有 dist 测试未被默认收集

pytest 默认 `norecursedirs` 含 `dist`。旧报告的 JUnit 未包含 `tests/pytorch/dist/test_fp8_alltoall.py`，对应实现的 74 行为零覆盖。

新增小规模测试放在 `tests/pytorch/ops`，避免该收集盲区，并沿用 MultiProcessTestCase，使 CI 的 xdist 单卡组跳过它、`--dist-only` 组执行它。现有 dist 目录与默认收集规则未在本轮调整：迁移/启用旧测试前，应先修正其超大 shape 和等分维度条件，并验证 GPU 数量变化下的有效性。

## 采集盲区：autograd 的原生工作线程

新增 AllToAll 用例调用真实 `actual.backward()` 并验证 `x.grad`，但默认 coverage.py 线程跟踪仍将 Python `FP8AllToAll.backward` 的 28 行全部记为未覆盖。对照实测：用 `torch.autograd.set_multithreading_enabled(False)` 将 Python backward 回调放在调用线程，保持算子与梯度断言不变，同样 16 项通过，行覆盖从 43/74 提升到 68/74（补记 25 行；其余 3 行是非法格式或单卡/空梯度路径）。随后新增的边界测试让累计覆盖进一步到 71/74。

本轮的小规模 AllToAll 与 QuantizedTensor 布局反向测试显式使用该上下文。普通多线程路径已有对应运行结果；该上下文不是线程安全验证，也不应默默改变所有 CI 测试的执行方式。类似的历史 backward 缺口需要先区分“没执行”和“没采集”，再决定是否新增测试。

## 按模块继续推进到 95%

以下是基线缺口，优先级按风险和可执行性安排，而不是只按测试条数。

| 优先级 | 模块 | 基线未覆盖行 | 后续需要补的测试 |
|---|---|---:|---|
| P0 | JAX 全部 | 1,237 | 先在具有 JAX ROCm 和 Turbo JAX 扩展的环境运行现有 `tests/jax` 并按 CI 合并。再补 plain/fancy batching 的 vmap、SPMD axis、int64 索引保持；DeepEP mode lock、EP 子组、setup/reset；primitive 的 abstract_eval、lowering、JVP/VJP 与错误参数。不能把未运行等同于没有测试。 |
| P1 | grouped GEMM FP8 | 262 | 在新增 Meta/门控测试之外，显式选择 CK/HipBLASLt/Triton/FlyDSL；tensor/row/block/MX，variable-K、trans_c、beta=1/out，tight M/N，GLU/dGLU backward 的数值和重入。只在对应环境支持时执行。 |
| P1 | DeepEP FlyDSL intranode | 223 | dispatch/combine 的 EP2/EP4、空路由和零 token、重复 expert、不同 rank token 数、async/event、cached handle、BF16/FP8、buffer 容量边界；需匹配该 backend 的硬件和构建。 |
| P1 | Attention USP 接口 | 106 | Ulysses/ring/组合并行，causal/window/GQA，前反向；只对需要的输入开梯度；多 rank 序列切分，FP8 scale 和 context 保存。 |
| P1 | Attention 常规接口 | 95 | dense/varlen，sink-only 梯度、bias、deterministic、训练/推理、不同后端与存储布局；错误形状/窗口/不支持后端。架构专用路径在对应节点验证，gfx950 不执行 gfx1250 数值用例。 |
| P1 | 对称内存 | 84 | 真实多 rank IPC rendezvous、跨 rank ring write/read、signal pad barrier、uncached 两种分配；失败注入覆盖分配中途退出，验证无泄漏。脚本入口演示代码另行审查，不靠直接调用内部行数凑指标。 |
| P1 | QuantizedTensor | 82 | 在新增布局和权重测试上补 grouped blockwise、grouped FP4、padding 的双轴裁剪、transpose 重量化、序列化/编译、无效配置。逐例验证 payload/scale 和逻辑值。 |
| P1 | DeepEP Buffer | 76 | backend 选择、通信流模式、配置优先级、dispatch/combine handle 生命周期、异常、组内子集与 teardown。 |
| P1 | FP8 AllToAll | 74 | 已加真实双卡 18 项及 host 参数验证；继续覆盖 world_size=1 的 autograd、量化的零 token、不同 rank 动态 scale、更多 rank 和输入 dtype。 |
| P2 | GEMM FP4 | 76 | 已加 Meta；继续补显式 backend、output accumulation、layout/scale 合法性、图捕获与缓存、融合 dGLU 数值。 |
| P2 | 量化 kernels | 68 | blockwise/grouped axes，with_trans 双输出、2D block、随机舍入与 RHT 合法组合、padding/ragged/零组、scale dtype；用独立参考确认数值和布局。 |
| P2 | grouped GEMM FP4 | 66 | variable-K 的不同 layout、out 累加、ragged group、preshuffle、gfx942 不支持路径、融合 dGLU；FP4 硬件路径需 gfx950 或支持架构。 |
| P2 | grouped MLP FP8/FP4、GEMM FP8、MoE ops | 每文件约 20–63 | 只训练部分参数、wgrad cache/accumulation、train/eval、checkpoint、冻结权重、可选 bias/probs、非整齐 shape；与 torch reference 比较 forward 和各输入梯度。 |
| P2 | RMSNorm | 56 | 本轮新增真实反向和 reduction 测试；后续关注 BF16/FP16、非连续 dy、超大 batch、compile/graph 与性能回归。 |
| P2 | 公共配置、日志、调优缓存 | 每文件约 10–43 | 本轮已补边界和缓存行为；保持环境/单例隔离，并在已有算子测试后验证没有全局状态泄漏。 |
| P3 | 架构专用、可选 backend、错误路径 | 分布于各模块 | 以支持矩阵规划 gfx942/gfx950/gfx1250 和可选库环境。当前 skip 规则应保留；可用 host 门控测试验证拒绝行为，但不能冒充该硬件的数值验证。 |

达到 95% 的验收必须同时满足：使用同一生产源码快照、PyTorch + JAX（及所需架构）覆盖数据按 CI 合并、明确行/分支口径、测试结果独立报告。不要通过删除难测代码的统计范围、降低容差或仅增加形状枚举来宣称达标。
