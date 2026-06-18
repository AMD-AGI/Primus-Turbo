###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-Turbo dense FP8 GEMM kernel (FlyDSL): NT, NN and TN layouts.
256x256 tile, BLOCK_K=128, 8-wave (wave_m=2 x wave_n=4), mfma_f32_16x16x128_f8f6f4,
per-tensor scale, bf16/fp16 out, arbitrary K via native K-tail (TT unsupported).
Primitives are imported from flydsl.utils.gemm_helper as module globals."""

# =============================================================================
# 【整体说明 / 学习版】
# 本文件是面向 AMD MI355X（gfx950 / CDNA4 架构）的 FP8 稠密 GEMM 内核，使用
# FlyDSL（一套基于 MLIR 的张量内核 DSL + 编译器）编写。它是 Primus-Turbo 中
# gemm_fp8_kernel.py 的「带详细中文注释的学习副本」，目的是把内核里每一个
# 设计决策的来龙去脉讲清楚。
#
# 几个最关键的事实，先建立全局认知：
#   1. 三种 layout（由两个矩阵是否转置决定）：
#        - NT：A=[M,K] 行主序，B_T=[N,K] 行主序（即 [K,N] 的转置存储），C=A@B。
#        - NN：A=[M,K]，B=[K,N]，C=A@B（反向传播 dgrad 常用）。
#        - TN：A=[K,M]，B=[K,N]，C=A^T@B。
#        - TT：不支持。
#   2. 每个 workgroup（WG）负责输出 C 的一个 256x256 tile，K 维以 BLOCK_K=128
#      为步长循环累加。
#   3. WG 内 512 个线程 = 8 个 wavefront（每个 wavefront 64 lane）。8 个 wave
#      排布成 wave_m∈{0,1} × wave_n∈{0,1,2,3}（即 2×4）。
#   4. 计算指令是 CDNA4 的 scaled MFMA：mfma_f32_16x16x128_f8f6f4，单条指令完成
#      16x16 输出、K=128 的 fp8 乘加。
#   5. 数据通路：global → LDS（共享内存，G2S，buffer_load_lds）→ 寄存器（S2R，
#      ds_read）→ MFMA。采用 cur/next 双缓冲做软件流水，预取领先 2 个 K 迭代。
#   6. 量化：per-tensor（整张量一个标量 scale），输出 bf16 或 fp16。
#
# 后文每个函数/代码段都会用「设计思路」标注解释为什么这么写。
# =============================================================================

import functools

import torch

# isort: off
# 说明：本内核依赖的所有底层「原语」（loader、MFMA 封装、swizzle、store 等）都
# 集中在 flydsl/utils/gemm_helper.py 里（没有引入 3rdparty/FlyDSL 子模块，唯一的
# FlyDSL 依赖就是编译器本身 flydsl）。这里采用 `from ... import *` 风格逐个导入到
# 「模块全局」的原因：@flyc.kernel 在 trace（构图）时要求内核体里用到的依赖必须能
# 在模块全局命名空间里找到，因此不能写成局部 import 或属性访问。
from primus_turbo.flydsl.utils.gemm_helper import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    S2RLoaderTr,
    StoreCPerTensor,
    asm_mma_do,
    ceildiv,
    compute_global_swizzle,
    compute_global_swizzle_nn,
    make_fp8_buffer_tensor,
    make_value_attrs,
    mask_a_tail,
    wait_barrier,
    xcd_remap_pid,
)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import range_constexpr, rocdl

# isort: on


# 用 lru_cache 缓存编译结果：每个不同的 (K, BLOCK_M, BLOCK_N, GROUP_M, ...) 组合
# 只会触发一次 FlyDSL 构图 + 编译。编译很贵，而真实训练里 K 维基本固定，因此命中率高。
@functools.lru_cache(maxsize=256)
def _compile_dense_nt(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 1,  # L2 复用的 super-block 重排宽度；=1 等价行主序，详见下文 swizzle
    waves_per_eu: int = 2,  # 每个 EU(SIMD) 上的 wave 占用上限，影响 occupancy
    agpr_alloc: int = 0,  # AGPR 分配策略：0=编译器自决；N>0=固定 N 个；-N=最多 N 个
    nt_vmcnt: int = 3,  # 迭代尾部的 s_waitcnt vmcnt(N) 排空：N=3 时确定性最佳（规避 gfx950 上 G2S buffer_load_lds 与 ds_read 之间的 LDS 读写竞态），开销<=1.1%；N>=4 会出现竞态，N<3 开销更大；-1 关闭
    num_xcd: int = 8,  # XCD 感知的 PID 重映射：把落在同一个 XCD 的 WG 聚成连续的逻辑 tile，提升每个 XCD 内的 L2 复用（MI355X 有 8 个 XCD）；=1 关闭
    cbsz: int = 0,  # 源操作数 A 的 fp8 格式：0=E4M3, 1=E5M2
    blgp: int = 0,  # 源操作数 B 的 fp8 格式：0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor 输出类型：True->fp16，否则 bf16
):
    """Build & cache the (K, BLOCK_M, BLOCK_N, GROUP_M)-specialised NT launch.

    GROUP_M is the super-block tile-id swizzle width for L2 reuse (WGs advance
    block_m first within each GROUP_M x n_blocks band; 1 = row-major). The main
    K-loop barriers are all load-bearing (each guards a compiler-reorder race).
    """
    BLOCK_K = 128
    # tile 形状约束：M 维至少 128 且 128 对齐，N 维至少 256 且 256 对齐。这与下面
    # 把 256x256 拆成 2x2 个 128x128 象限、以及 MFMA 16x16 的对齐要求是绑定的。
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert GROUP_M >= 1

    # 【设计思路：原生 K-tail（处理任意 K，不要求 128 整除）】
    # K_ITERS = ceil(K/128)，最后一个迭代长度为 K_TAIL（=0 表示恰好整除）。
    # 对 NT 来说 A 是 K 连续存储的，最后一块里 K 列下标 >= K_TAIL 的那部分是无效数据，
    # 会在 Epilog 2 用 mask_a_tail 把 A 侧这些字节清零（a_k=0 则乘加贡献为 0，与 B 无关）。
    # 而 G2S 读取越界的部分由 buffer descriptor(SRD) 的 num_records 边界硬件钳到 0。
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    # 至少 2 个 K 迭代：因为软件流水的 prelude 预取了 k=0、k=1，主循环跑 K_ITERS-2 次，
    # 后面再接 2 个 epilog，所以 K 必须 >= 129。
    assert K_ITERS >= 2, f"K_ITERS={K_ITERS} too small; need K >= 129 (ceil(K/128) >= 2)"

    # MFMA 是 16x16x128。一个 wave 在 M 方向覆盖 N_TILES_A 个 16 行=BLOCK_M/64 个，
    # 在 N 方向覆盖 N_TILES_B 个 16 列对（每 128 列一组）=BLOCK_N/128 个。
    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B  # 单个象限内每个 wave 持有的累加器分片数
    assert N_ACCUMS > 0

    # 【设计思路：把 256x256 输出 tile 切成 2x2 个 128x128 象限】
    # LDS 里 A、B 各自只缓存「半个 M」和「半个 N」（128），分别命名 cur0/cur1。
    # 这样四个象限 c00/c01/c10/c11 = (上M,左N)/(上M,右N)/(下M,左N)/(下M,右N)，
    # 既减小单个 LDS buffer 体积，又能让 4 次 MFMA 调用与预取细粒度交织。
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    # 一次 G2S 把 LDS_BLOCK(128) 行/列按 64 一组分若干 step 搬运。
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    # 每个 LDS 半块的字节容量（fp8 每元素 1 字节）。
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    # 【设计思路：cur/next 双缓冲共享内存布局】
    # A、B 各 4 个 buffer：cur_0/cur_1（当前 K 块的上下半 / 左右半）与
    # next_0/next_1（下一 K 块的预取目标）。计算当前块时同步预取下一块，
    # 迭代尾部用 Python 变量交换把 next 变成 cur（trace 期完成，零运行开销）。
    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    # known_block_size=[512,1,1]：512 线程 = 8 wave，编译期固定，利于资源分配。
    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # NT 语义：A 是 [M, K] 行主序、K 连续；
        #          B_T 是 [N, K] 行主序、K 连续（即 [K, N] 的转置存储）；
        #          输出 C 是 [M, N] 行主序，bf16。
        # 关键点：A、B_T 的 K 维都是连续的，所以两侧都能直接按 K 连续做 G2S，
        # 不需要转置载入（这正是 NT 是最简单/最快 layout 的原因）。
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        # 申请共享内存并取出 8 个 buffer 的句柄。
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        # lane_id：wavefront 内 0..63；wave_id：WG 内 0..7。
        # wave_m∈{0,1}、wave_n∈{0,1,2,3}，即 8 个 wave 排成 2×4 网格。
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # 【设计思路：GROUP_M super-block tile 重排，提升 L2 复用】
        # 朴素行主序遍历 tile 会让相邻 WG 共享的 A/B 行/列在 L2 里被频繁挤出。
        # 这里把 tile id 重排成「以 GROUP_M 行为一带、带内先沿 M 走再沿 N 走」，
        # 让同一带内的 WG 复用相同的 B 列块、相邻 M 复用相同的 A 行块。
        # group_size_m 用 arith.select（=整数 min）把最后一带不足 GROUP_M 的情况钳住，
        # 因此任意 GROUP_M>=1 都是正确的双射。
        num_pid_m = ceildiv(c_m, BLOCK_M)
        # 先做 XCD 感知重映射：硬件按 block_idx 轮转分发 WG 到 8 个 XCD，这里反向重排，
        # 使连续逻辑 tile 落在同一 XCD，把该 XCD 的 L2 复用留在本 XCD 内。
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

        # 本 WG 负责的 A 上半/下半、B 左半/右半在全局张量里的元素起始偏移。
        # A=[M,K]：行 (block_m*256) 起，每行跨 K 个元素；上半起点再 +LDS_BLOCK_M 行。
        # B_T=[N,K]：同理按 N 行偏移。
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        # 把传入的 i8 扁平张量重解释成 fp8 的 buffer tensor（带硬件 SRD 越界钳位）。
        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # 【设计思路：global swizzle，消除 LDS bank 冲突】
        # compute_global_swizzle 在写 LDS 时对地址做 XOR swizzle（swizzle_128），
        # 使得后续 ds_read 读 LDS 时各 lane 落在不同 bank，避免 bank conflict。
        # NT 的 A、B 都是 K 连续，用同一套 swizzle 即可。
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)

        # MFMA 封装。默认 E4M3；若 A 或 B 是 E5M2（cbsz/blgp），按每操作数 fp8 格式
        # 重建 scaled MFMA atom（指令家族与寄存器分片布局不变，因此 loader 不用改）。
        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        # G2S（global→LDS）与 S2R（LDS→寄存器）loader。NT 两侧都 K 连续，所以
        # A、B 都用普通 S2RLoader（直接 ds_read，无需转置）。
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        # 四个象限的累加器，初值清零。每个是 N_ACCUMS 个 v4f32 分片。
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # 【设计思路：软件流水的 prelude（填充流水线）】
        # 先把 k=0 装进 cur，k=1 装进 next（a_next1 留到主循环第一次迭代再懒加载，
        # 以错开发射、缓解寄存器/发射压力）。这样进入主循环时已有两级数据在飞。
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        # wave_m==1 的一半 wave 先到一次 barrier，与下面的 wait_barrier 配合，
        # 保证 k=0 的 G2S 写 LDS 在被 ds_read 之前已完成（生产者-消费者同步）。
        if wave_m == 1:
            rocdl.s_barrier()

        # wait_barrier(count) = s_waitcnt vmcnt(count) + s_barrier：
        # 允许 count 个 vector-memory（G2S）请求仍在飞，其余必须落地，再过 barrier。
        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 【设计思路：主 K 循环——计算与预取细粒度交织】
        # 每次迭代做四件事并交错排布：
        #   1) S2R 把当前 K 块的 {a0,b0,b1,a1} 读进寄存器；
        #   2) 4 次 MFMA（c00→c01→c10→c11），覆盖 2x2 个象限；
        #   3) 预取 k+1 的 a_next1（补齐 prelude 里懒加载的那块）；
        #   4) 预取 k+2 的 a_cur0/b_cur0/b_cur1（写进交换后即将变成 cur 的 buffer）。
        # 每次 MFMA 前后都包一对 s_barrier，且 MFMA 期间用 s_setprio(1) 提优先级、
        # 结束 s_setprio(0) 复位。
        #   - s_barrier：既同步 LDS 生产者/消费者，也阻止编译器把 G2S/ds_read 跨 MFMA 重排
        #     （否则会读到未写完的 LDS）。注释里说所有 barrier 都是「load-bearing」，
        #     即每一个都在挡一个具体的重排竞态，不能删。
        #   - s_setprio：MFMA 阶段抬高本 wave 发射优先级，避免别的 wave 抢占发射槽、
        #     让 MFMA 流水持续喂满。
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            # 【设计思路：迭代尾部的 vmcnt 排空，修一个 gfx950 硬件竞态】
            # 在 gfx950 上 G2S 的 buffer_load_lds 与 ds_read 之间存在 LDS 读写时序冒险：
            # 若不在迭代尾部强制把在飞的 G2S 排到只剩 nt_vmcnt 个，下一迭代的 ds_read
            # 可能读到尚未写完的 LDS。实测 nt_vmcnt=3 时结果确定（det=0）且开销<=1.1%；
            # >=4 会出现竞态，<3 反而更慢；-1 关闭此修复。
            if nt_vmcnt >= 0:
                _llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string=f"s_waitcnt vmcnt({nt_vmcnt})",
                    constraints="",
                    has_side_effects=True,
                )  # end-of-iter G2S drain (race fix)
            # 双缓冲交换：把 next 改名成 cur（仅是 Python 引用交换，trace 期完成，
            # 不产生任何运行时指令）。下一迭代的 cur 就是这一迭代刚预取的数据。
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # 【设计思路：Epilog 1（倒数第二个 K 迭代，k=K_ITERS-2）】
        # 主循环为了流水把预取领先了 2 个迭代，所以最后两块要单独「排空流水线」。
        # Epilog 1 处理倒数第二块，且只需预取最后一块的 a_next1（其余 next 已无更多块）。
        # 注意中间那行 a_g2s.load(a_next1, A1+(k+1)*BLOCK_K) 是「stale-a1 流水线修复」：
        # 没有它，Epilog 2 里的 a1_frag 会读到更早 K 迭代的旧数据，导致每个输出 tile
        # 的下半（c10/c11）丢掉最后一个 K 块的贡献，结果错误。
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # 【设计思路：Epilog 2（最后一个 K 迭代，k=K_ITERS-1）——K-tail 块】
        # 这是最后一块，可能不满 128。A 是 K 连续的，最后一块里 K 列 >= K_TAIL 的字节
        # 是无效数据，用 mask_a_tail 把 A 侧这些字节按位清零（a_k=0 则该 K 项乘加为 0，
        # 与 B 无关，从而正确地「跳过」尾部无效列）。K_TAIL==0（整除）时是空操作。
        # wait_barrier(0)：此处必须把所有在飞 G2S 全部排空（vmcnt 0），因为后面不再预取。
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # 【设计思路：缩放 + 写回】
        # 累加器是 f32，写回前乘以 per-tensor 的 a_scale*b_scale，再转 bf16/fp16。
        # 每个 wave 在象限内的行/列基址由 wave_m/wave_n 决定（每 wave 覆盖 64 行×32 列）；
        # 四个象限分别落到 (0,0)/(0,N半)/(M半,0)/(M半,N半)。
        # StoreCPerTensor 内部对越界列做 OOB 钳位，因此 M/N 非整除 tile 也安全。
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    # JIT 启动包装：网格大小 = M 方向 tile 数 × N 方向 tile 数；每个 WG 512 线程。
    # value_attrs 把 waves_per_eu / AGPR 分配等编译期属性传给后端。
    @flyc.jit
    def launch_dense_nt(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nt(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nt


# ──────────────────────────────────────────────────────────────────────
# 【NN layout】A=[M,K]、B=[K,N]，C=A@B。与 NT 的核心差异：B 不再是 K 连续，而是
# N 连续（B 的每一行是 K 行、列是 N）。MFMA 需要 B 以 K-major 喂入，因此 B 侧改用
# ds_read_b64_tr_b8「转置载入」（S2RLoaderTr）把 LDS 里的数据在读出时转置到 MFMA
# 所需的寄存器字节布局。A 侧仍与 NT 相同（K 连续，普通 S2RLoader）。
# ──────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def _compile_dense_nn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    num_xcd: int = 8,  # XCD 感知 PID 重映射（MI355X=8 XCD）；=1 关闭。见 xcd_remap_pid。
    waves_per_eu: int = 2,
    agpr_alloc: int = 0,
    # 【设计思路：把 ds_read_b64_tr_b8 写成 inline asm】
    # 默认 intrinsic 路径下，后端会在 ds_read 后自动插入 vmcnt(0) 排空（保守同步），
    # 每个 K 迭代都付出代价。改用 inline asm 发射转置读，可让后端跳过这个自动排空，
    # 改由 vmcnt_hint 提供更精确的 LDS 同步。代价是需要固定 AGPR（agpr_alloc>0）。
    b_inline_asm_load: bool = False,
    vmcnt_hint: int = 2,
    cbsz: int = 0,  # 源操作数 A 的 fp8 格式：0=E4M3, 1=E5M2
    blgp: int = 0,  # 源操作数 B 的 fp8 格式：0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor 输出类型：True->fp16，否则 bf16
):
    """NN-layout fp8 dense kernel. A [M, K], B [K, N], C [M, N].

    ``agpr_alloc`` / ``waves_per_eu`` mirror the NT kernel's knobs; see
    ``make_value_attrs`` for ``agpr_alloc`` encoding (N>0 = exact N AGPRs,
    -N = up to N, 0 = compiler default)."""
    # inline-asm 的操作数约束需要明确的 AGPR 数量，与编译器自决（agpr_alloc=0）冲突，
    # 因此开启 b_inline_asm_load 时必须把 AGPR 固定到非零（例如 32）。
    if b_inline_asm_load and agpr_alloc == 0:
        raise ValueError(
            "b_inline_asm_load=True requires agpr_alloc > 0 (a compiler-decided "
            "AGPR count conflicts with the inline-asm operand constraints); "
            "pin AGPR to a nonzero value such as 32."
        )
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # Odd-K native K-tail: ceil iters; final iter masked on A (see NT note).
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    K_TAIL = K % BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K  # same byte count as NT, different layout

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # 【细节】提前「物化」thread_idx.x：S2RLoaderTr 会在 range_constexpr 循环里
        # 懒用 thread_idx.x，若不在循环外先求值，trace 出的 ds_read_tr8_b64 载入顺序
        # 会错乱。这行 str(...) 强制把它的 expr 节点先建出来，固定求值时机。
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type

        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # Super-block tile swizzle for L2 reuse; group_size_m clamps the last
        # band so any GROUP_M >= 1 is correct (same as NT).
        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        num_pid_in_group = GROUP_M * n_blocks
        group_id = pid // num_pid_in_group
        pid_in_group = pid % num_pid_in_group
        first_pid_m = group_id * GROUP_M
        remaining_m = num_pid_m - first_pid_m
        group_size_m = arith.select(remaining_m < GROUP_M, remaining_m, fx.Int32(GROUP_M))
        block_m = first_pid_m + (pid_in_group % group_size_m)
        block_n = pid_in_group // group_size_m

        # A：与 NT 完全相同（A=[M,K]，K 连续，按行偏移）。
        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K

        # 【设计思路：B 的偏移是 NN 特有的】
        # B=[K,N] 行主序，N 是连续维。本 WG 取 BLOCK_K 行 K × BLOCK_N 列 N，
        # 按 N 切成左右两半 LDS_BLOCK_N。这里 B0/B1 只给出 N 方向的列起点；
        # K 方向推进在下面的 load 里体现：每个 K 迭代偏移 BLOCK_K*c_n 个元素
        # （因为 B 的一行是 c_n 个元素宽，前进 BLOCK_K 行就是 BLOCK_K*c_n）。
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # A 用 K 连续的 swizzle；B 用 N 连续（NN 专用）的 swizzle_nn，
        # 它按 [K_inner, N_out] 行主序、N_out 元素步长计算每 lane 的加载偏移。
        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if cbsz or blgp:
            # E5M2 / 混合精度：按每操作数 fp8 格式重建 MFMA atom（cbsz->srcA，
            # blgp->srcB）。指令家族与寄存器分片布局和默认 e4m3 相同，loader 不用改。
            _ea = fx.Float8E5M2 if cbsz else fx.Float8E4M3FN
            _eb = fx.Float8E5M2 if blgp else fx.Float8E4M3FN
            mfma.atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, _ea, _eb))

        # 关键差异：B 侧用 S2RLoaderTr（转置载入），因为 B 在 LDS 里是 N-major 而
        # MFMA 要 K-major；ds_read_b64_tr_b8 在读 LDS 时完成转置。A 侧仍是普通 loader。
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoaderTr(wave_n, N_TILES_B, 32, inline_asm=b_inline_asm_load, vmcnt_hint=vmcnt_hint)
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude（与 NT 同构）。唯一差别：B 的 K 推进是 BLOCK_K*c_n（B 行宽 c_n）。
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 主循环：结构与 NT 一致（S2R → 4 次 MFMA 交织预取），每个 K 迭代发 7 个 barrier
        # （每次 MFMA 前后各一个）。所有 barrier 都是 load-bearing，删任意一个都可能引入
        # 编译器重排竞态。NN 不需要 NT 的 nt_vmcnt 尾部排空（B 走转置 loader，
        # 同步语义由 vmcnt_hint / 自动 vmcnt(0) 处理）。
        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1（倒数第二块）。同样保留 stale-a1 修复那行 a_g2s.load(a_next1,...)，
        # 否则 Epilog 2 的下半象限会丢最后一块的贡献（与 NT 同理）。
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)  # stale-a1 fix
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 —— K-tail 块。A 是 K 连续，用 mask_a_tail 把 K 列 >= K_TAIL 的字节清零。
        a0_frag = a_s2r.load(a_cur0)
        a0_frag = mask_a_tail(a0_frag, lane_id, K_TAIL)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        a1_frag = mask_a_tail(a1_frag, lane_id, K_TAIL)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_nn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_nn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_nn


# ──────────────────────────────────────────────────────────────────────
# 【TN layout】A=[K,M]、B=[K,N]，C=A^T@B。这是反向传播里 wgrad 的典型形态。
# 与 NT/NN 最大的不同：A 和 B 都是 K 行主序（K 是外层、M/N 是连续维），两侧都需要
# 转置载入；又因为 MFMA 的 A、B 操作数寄存器字节布局相同，同一个 S2RLoaderTr 可以
# 同时喂两侧。此外 TN 走「inplace AGPR 累加」：累加器直接驻留在 AGPR，无溢出，
# 且省掉了 A 侧每个 K 迭代的 vmcnt(0) 排空。
# ──────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=128)
def _compile_dense_tn(
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    GROUP_M: int = 4,
    waves_per_eu: int = 2,
    vmcnt_hint: int = 3,
    group_n: int = 0,  # 0 = 一维 GROUP_M 重排；>0 = 二维 band（带宽 group_n）
    num_xcd: int = 8,  # XCD 感知 PID 重映射（MI355X=8 XCD）；=1 关闭。见 xcd_remap_pid。
    cbsz: int = 0,  # 源操作数 A 的 fp8 格式：0=E4M3, 1=E5M2
    blgp: int = 0,  # 源操作数 B 的 fp8 格式：0=E4M3, 1=E5M2
    out_fp16: bool = False,  # StoreCPerTensor 输出类型：True->fp16，否则 bf16
):
    """TN-layout fp8 dense kernel: A [K, M], B [K, N], C [M, N] = A^T @ B.
    Both A and B are K-row strided, so both go through the wave-coop
    ds_read_b64_tr_b8 transpose load (the mfma A and B operand register byte
    layouts are identical, so the same S2RLoaderTr feeds both operands).
    Inline-asm tr8 on both operands + asm-inplace MFMA (=a,v,v,0; D aliases C in
    AGPR -> accumulators spill-free, no per-K-iter A-side vmcnt(0) drain)."""
    # 【设计思路：TN 固定走 inplace-AGPR 路径】
    # _a_inline/_b_inline：A、B 转置读都用 inline asm（跳过后端自动 vmcnt(0)）。
    # _asm_mma_mode="2"：MFMA 用约束 "=a,v,v,0"——输出 D 与输入 C 同寄存器（AGPR），
    #   累加器整段驻留 AGPR，避免占用并溢出 VGPR；同时不需要 A 侧每迭代的 vmcnt(0) 排空。
    # agpr_alloc=128：固定分配 128 个 AGPR 来容纳累加器（与 inplace 约束匹配）。
    _a_inline = True
    _b_inline = True
    _asm_mma_mode = "2"  # asm-inplace MFMA (accum in AGPR)
    _inplace = True
    agpr_alloc = 128
    BLOCK_K = 128
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0

    # 【设计思路：TN 的 K-tail 不需要 mask】
    # TN 的 A=[K,M]、B=[K,N] 都是 K 行主序，最后一块里无效的 K 行是「整行越界」，
    # 直接被 buffer descriptor 的 num_records 边界硬件钳到 0。这点与 NT/NN 不同——
    # 那两者 A 是 K 连续（K 列在一行内），无效 K 列不会越界，必须显式 mask_a_tail。
    K_ITERS = (K + BLOCK_K - 1) // BLOCK_K
    assert K_ITERS >= 2

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    # 【设计思路：A 侧也走转置载入，强制 2 个 G2S round】
    # TN 的 A 用 wave-coop 的 tr8 转置读，其 K_log 跨度是 [0,128)，需要 2 个 G2S
    # round（=16K 的 LDS 槽）才能放下 K=128。当 BLOCK_M=128 时自然 N_LDS_STEPS_A=1
    # （只有 8K 槽），不足以匹配转置读对 K=128 的预期，因此这里 max(..., 2) 强制 2 轮。
    N_LDS_STEPS_A = max(LDS_BLOCK_M // 64, 2)  # ≥ 2 for tr8 K=128
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)
    # 【设计思路：bank-spread 的 LDS chunk 步长 1056=1024+32】
    # 转置读 ds_read_b64_tr_b8 若各 wave 的 chunk 基址都按 1024 对齐，会落到同一组
    # LDS bank 产生 bank conflict。把每 wave 的 chunk 步长设成 1056（多 32 字节）
    # 使各 wave 基址错开 bank，消除转置读的 bank 冲突。G2S 写端与 S2R 读端必须用
    # 同一个值，否则地址对不上。
    _LDS_CS = 1056
    # a_lds_size：N 轮 × 8 wave × chunk_stride，按步长对齐补齐。
    a_lds_size = max(LDS_BLOCK_M * BLOCK_K, 2 * 8 * 1024) // 1024 * _LDS_CS
    b_lds_size = (LDS_BLOCK_N * BLOCK_K) // 1024 * _LDS_CS

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    # 【设计思路：TN 用纯 Python 辅助函数做 tile 重排】
    # tile-id -> (block_m, block_n)，在 trace 期按 Python 选择的分支建图。
    #   GN==0：一维 GROUP_M super-row 重排（block_m 在内，与 NT/NN 同套路）。
    #   GN>0 ：二维 band——把 N 切成宽 GN 的带，带内再用 GROUP_M。目的是让 A、B 两块
    #          都常驻 L2（A 复用同带的 M 行、B 复用同带的 N 列）。
    # 用普通 Python if（而不是 kernel 里的 if）的原因见下方 block_m/block_n 处注释。
    # 无论哪个分支，映射都是 [0,total_pids) 上的双射。
    def _tn_block_mn(pid, num_pid_m, n_blocks, GM, GN):
        """Tile-id -> (block_m, block_n), resolved at trace time. GN==0: 1D
        GROUP_M super-row swizzle (block_m inner). GN>0: 2D band — N split into
        width-GN bands with GROUP_M inside each, keeping both A and B slabs
        L2-resident. Always a bijection."""
        if GN > 0:
            band_tiles = num_pid_m * GN
            band = pid // band_tiles
            pid_in_band = pid % band_tiles
            band_n0 = band * GN
            rem_n = n_blocks - band_n0
            band_w = arith.select(rem_n < GN, rem_n, fx.Int32(GN))
            nig = GM * band_w
            gid = pid_in_band // nig
            pig = pid_in_band % nig
            fpm = gid * GM
            rem_m = num_pid_m - fpm
            gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
            return fpm + (pig % gsm), band_n0 + (pig // gsm)
        nig = GM * n_blocks
        gid = pid // nig
        pig = pid % nig
        fpm = gid * GM
        rem_m = num_pid_m - fpm
        gsm = arith.select(rem_m < GM, rem_m, fx.Int32(GM))
        return fpm + (pig % gsm), pig // gsm

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        _ = str(fx.thread_idx.x)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        n_blocks = ceildiv(c_n, BLOCK_N)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4

        num_pid_m = ceildiv(c_m, BLOCK_M)
        pid = xcd_remap_pid(fx.block_idx.x, num_pid_m * n_blocks, num_xcd)
        # 用普通 Python 辅助函数做重排，而不是 kernel 里的 if：@flyc.kernel 会把每个
        # if 分支包成独立的子函数，分支内定义的变量在分支外不可见（与 prelude 那条
        # 物化 thread_idx.x 的注意点同源）。辅助函数在 trace 期按 Python 选定的一条
        # 路径（一维 GROUP_M 或二维 band）建出 expr 图。
        block_m, block_n = _tn_block_mn(pid, num_pid_m, n_blocks, GROUP_M, group_n)

        # TN 的 A 存为 [K, M] 行主序：每前进一个 K 行，元素偏移 +M（即 +c_m）。
        # 这里 A0/A1 只给 M 方向的列起点，K 推进在下面 load 里乘 BLOCK_K*c_m。
        A0_gl_offset = block_m * BLOCK_M + 0
        A1_gl_offset = block_m * BLOCK_M + LDS_BLOCK_M

        # B 与 NN 相同：存为 [K, N] 行主序，K 推进乘 BLOCK_K*c_n。
        B0_gl_offset = block_n * BLOCK_N + 0
        B1_gl_offset = block_n * BLOCK_N + LDS_BLOCK_N

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        # A、B 都是 K 行主序，所以两侧都用 NN 风格的 K-strided global swizzle
        # （A 用 c_m 作为连续维步长，B 用 c_n）。
        gl_off_a = compute_global_swizzle_nn(lane_id, wave_id, c_m, N_LDS_ROUNDS)
        gl_off_b = compute_global_swizzle_nn(lane_id, wave_id, c_n, N_LDS_ROUNDS)

        # 【设计思路：把 MFMA 替换成 inline-asm 的 inplace 版本】
        # 默认 mfma._do_mma 走 dialect 调用，结果写新寄存器。这里覆写成 asm_mma_do，
        # 用约束 "=a,v,v,0" 让输出 D 复用输入 C 的 AGPR（累加器原地更新），既不占 VGPR
        # 也不会溢出，且能去掉 A 侧每迭代的 vmcnt(0) 排空。cbsz/blgp 同时透传给 asm。
        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        if _inplace:
            _mm = _asm_mma_mode
            mfma._do_mma = lambda _a, _b, _c, _m=_mm: asm_mma_do(_a, _b, _c, mode=_m, cbsz=cbsz, blgp=blgp)

        # G2S 用 bank-spread 步长 _LDS_CS 写 LDS；A、B 两侧 S2R 都用 S2RLoaderTr 转置读，
        # 且 chunk_stride 必须与 G2S 写端一致。注意 a_s2r 的 tile_stride 是 LDS_BLOCK_M//2，
        # b_s2r 是 32——分别对应 A（沿 M）和 B（沿 N）在转置坐标上的每 wave 覆盖步长。
        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id, chunk_stride=_LDS_CS)
        a_s2r = S2RLoaderTr(
            wave_m,
            N_TILES_A,
            LDS_BLOCK_M // 2,
            inline_asm=_a_inline,
            vmcnt_hint=vmcnt_hint,
            chunk_stride=_LDS_CS,
        )
        b_s2r = S2RLoaderTr(
            wave_n, N_TILES_B, 32, inline_asm=_b_inline, vmcnt_hint=vmcnt_hint, chunk_stride=_LDS_CS
        )
        _out_ty = fx.Float16 if out_fp16 else fx.BFloat16
        store_c = StoreCPerTensor(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _out_ty)

        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        # Prelude（A 的 K 推进乘 BLOCK_K*c_m，B 乘 BLOCK_K*c_n）。
        b_g2s.load(b_cur0, B0_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K * c_m)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * BLOCK_K * c_n)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K * c_m)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * BLOCK_K * c_n)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K * c_m)
        b_g2s.load(b_next1, B1_gl_offset + 1 * BLOCK_K * c_n)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        # 主循环结构与 NT/NN 同：A 上/下半 × {b0,b1} 四次 MFMA 交织预取，每象限一个
        # s_barrier，7 个 barrier 全是 load-bearing（某些 GROUP_M 下删任意一个会在
        # MFMA 重排层引入竞态，靠长时间确定性回归发现）。
        for k in range_constexpr(K_ITERS - 2):
            # 【设计思路：b0 用 drain=False 省一次 lgkmcnt 排空】
            # b0 的转置读会被紧随其后的 a0 load 自带的 lgkmcnt(0) 覆盖（在 c00 消费 b0
            # 之前就已排空），所以 b0 loader 自己尾部的排空是冗余的，可省。b1 仍保留
            # drain——c01 消费 b1 之前没有别的排空来覆盖它。
            b0_frag = b_s2r.load(b_cur0, drain=False)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b1_frag = b_s2r.load(b_cur1)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * BLOCK_K * c_n)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K * c_m)
            rocdl.s_barrier()
            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * BLOCK_K * c_n)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)
            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 1（倒数第二个 K 块，排空流水）。
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b0_frag = b_s2r.load(b_next0)
        a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K * c_m)  # stale-a1 修复（同 NT/NN）
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Epilog 2 —— 最后一个 K 块。注意这里没有 mask_a_tail：TN 的无效 K 行整行越界，
        # 由 buffer SRD 边界钳到 0，无需软件 mask（见上文 K-tail 设计说明）。
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)
        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        b1_frag = b_s2r.load(b_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()
        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset
        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_dense_tn(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_dense_tn(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs=make_value_attrs(waves_per_eu, agpr_alloc, "512,512"),
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_dense_tn


# 历史记录：曾有一个 NN 4-wave 版本，实测始终慢于 8-wave（该 layout 下硬件每个
# SIMD 只能容纳 1 个 wave），已删除。


# ──────────────────────────────────────────────────────────────────────
# 【主机侧：编译缓存 + per-shape autotune + 入口分发】
# 上面三个 _compile_dense_* 返回的是「未编译的 launch 描述」（lru_cache 按 K/tile
# 参数缓存）。下面这层再按运行时的 (shape, dtype) 缓存「已编译可调用对象」，并在
# 首次遇到某个形状时微基准测试若干候选配置，缓存最快的一个。
# ──────────────────────────────────────────────────────────────────────


_COMPILED_DENSE_CACHE: dict = {}


def _get_compiled_dense(launch, args):
    """Cache compiled launcher by (shape, dtype, int-arg) tuple."""
    key_parts = [id(launch)]
    for a in args:
        if isinstance(a, torch.Tensor):
            key_parts.append((tuple(a.shape), a.dtype))
        elif isinstance(a, int):
            key_parts.append(a)
        else:
            key_parts.append(type(a).__name__)
    key = tuple(key_parts)
    cached = _COMPILED_DENSE_CACHE.get(key)
    if cached is None:
        cached = flyc.compile(launch, *args)
        _COMPILED_DENSE_CACHE[key] = cached
    return cached


def _as_i8_flat(t: torch.Tensor) -> torch.Tensor:
    # 零拷贝的扁平字节视图。每次调用都重算（不按 id() 缓存）：一个被释放的 tensor，
    # 它的 id 和 data_ptr 都可能被复用，若缓存了「id->视图」，复用后若 numel 不同
    # 就会按错误长度别名到新内存。view 操作约 1us 且不分配内存，重算成本可忽略。
    if t.element_size() == 1 and t.dtype != torch.int8:  # fp8：先按 int8 重解释再展平
        return t.contiguous().view(torch.int8).view(-1)
    return t.contiguous().view(-1)


def _scalar_scale(scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensorwise scalar -> length-1 fp32 buffer (no broadcast). The kernel reads
    the single value and applies it per-tensor, so only an fp32/device cast is
    needed (no per-row/col vector to materialize)."""
    # per-tensor 量化：scale 必须是标量。内核只读这一个值并整张量统一应用，
    # 因此只需转成 fp32 并搬到目标 device，不需要展开成 per-row/col 向量。
    assert scale.numel() == 1, f"per-tensor expects scalar, got {scale.shape}"
    return scale.to(dtype=torch.float32, device=device).reshape(1)


# 【设计思路：per-shape autotune 候选集】
# 每个元组是 (BLOCK_M, GROUP_M, num_xcd, AGPR)。GROUP_M 和 num_xcd 已按 L2 复用的
# 解析最优固定，只对 BLOCK_M、AGPR 做实测筛选（这些影响 occupancy/计算，热缓存基准
# 能稳定测出）。NN 路径 AGPR 必须非零（inline-asm 的 ds_read_b64_tr_b8 需要固定 AGPR）。
_NN_CANDIDATES = [
    (256, 4, 8, 32),
    (256, 4, 8, 64),
    (128, 4, 8, 48),
]
_NN_AUTOTUNE_CACHE: dict = {}


def _autotune_nn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NN candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NN_AUTOTUNE_CACHE:
        return _NN_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NN_CANDIDATES:
        # M 非整除（M % bm != 0）也安全：最后一个不完整 M-tile 一侧由 c_m 钳住
        # （StoreCPerTensor 的越界钳位），另一侧由 global SRD 钳住（A 的 G2S 读越界
        # 硬件清零），因此不需要「只测能整除的 tile」这类过滤。
        try:
            # inline-asm ds_read_b64_tr_b8 on by default (drops the per-K-iter
            # compiler-auto vmcnt(0) drains).
            launch = _compile_dense_nn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                num_xcd=xcd,
                agpr_alloc=ag,
                b_inline_asm_load=True,
                vmcnt_hint=2,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NN autotune found no working cfg for ({M},{N},{K})")
    _NN_AUTOTUNE_CACHE[key] = best
    return best


# NT per-shape autotune candidates (BLOCK_M, GROUP_M, num_xcd, AGPR). GROUP_M
# and num_xcd are fixed at the analytic L2 optimum; only BLOCK_M and AGPR are
# benched (occupancy/compute effects the hot-cache bench measures reliably).
_NT_CANDIDATES = [
    (256, 4, 8, 64),
    (256, 4, 8, 32),
    (128, 4, 8, 48),
    (128, 4, 8, 32),
]
_NT_AUTOTUNE_CACHE: dict = {}


def _autotune_nt_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench NT candidates, cache best (launch, cfg) by (M,N,K).

    Runtime micro-benches each (BM, GROUP_M, num_xcd, AG) candidate,
    finite-checks the output, times 2-warmup + 20-iter, and caches the
    fastest by shape.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _NT_AUTOTUNE_CACHE:
        return _NT_AUTOTUNE_CACHE[key]
    out_view = args[2]
    best_us = float("inf")
    best = None
    for bm, gm, xcd, ag in _NT_CANDIDATES:
        # odd-M (M % bm != 0) is fine: the partial last M-tile is
        # bounded by c_m (StoreCPerTensor clamp) and the global SRD (HW OOB
        # clamp on the A G2S load), so no even-tiling filter is needed.
        try:
            launch = _compile_dense_nt(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=gm,
                agpr_alloc=ag,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, gm, xcd, ag))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"NT autotune found no working cfg for ({M},{N},{K})")
    _NT_AUTOTUNE_CACHE[key] = best
    return best


# TN 分发：只有一个 inplace-A 内核（两侧都 inline-asm tr8 转置读 + asm_mma=2，累加器
# 别名进 AGPR、无溢出、省掉 A 侧每迭代 vmcnt(0) 排空）。GROUP_M=4 + XCD 重映射与 NT/NN
# 相同；每个形状只实测 num_xcd 的 8 vs 1（L2 常驻的形状选 1，HBM 流式的大形状选 8）。


_TN_AUTOTUNE_CACHE: dict = {}


def _autotune_tn_dispatch(args, M, N, K, cbsz=0, blgp=0, out_fp16=False):
    """First-call bench TN candidates, cache best (launch, cfg) by (M,N,K).

    1D GROUP_M=4 with num_xcd 8 vs 1 (XCD-aware PID remap); large
    (HBM-streaming) shapes expose the per-XCD L2 reuse on the hot bench,
    L2-resident shapes pick num_xcd=1.
    """
    import torch as _torch

    key = (M, N, K, cbsz, blgp, out_fp16)
    if key in _TN_AUTOTUNE_CACHE:
        return _TN_AUTOTUNE_CACHE[key]
    # 【设计思路：TN 按 occupancy 选 BLOCK_M】
    # BLOCK_M=BLOCK_N=256 时 tile 数 = ceil(M/256)*ceil(N/256)。当 tile 数小于
    # NUM_CUS（256 个 CU）时，网格填不满所有 CU，于是改用 BLOCK_M=128 把 M 方向
    # tile 数翻倍、提高 occupancy；当 tile 数已足够多时，更小 block 的每 tile 额外
    # 开销反而占主导，所以维持 256。
    NUM_CUS = 256
    tiles_256 = ((M + 255) // 256) * ((N + 255) // 256)
    bm = 128 if tiles_256 < NUM_CUS else 256
    out_view = args[2]
    best_us = float("inf")
    best = None
    for xcd in (8, 1):
        try:
            launch = _compile_dense_tn(
                K=K,
                BLOCK_M=bm,
                BLOCK_N=256,
                GROUP_M=4,
                vmcnt_hint=3,
                group_n=0,
                num_xcd=xcd,
                cbsz=cbsz,
                blgp=blgp,
                out_fp16=out_fp16,
            )
            c = _get_compiled_dense(launch, args)
            c(*args)
            _torch.cuda.synchronize()
            sample = out_view.view(-1)[:1024].float()
            if not _torch.isfinite(sample).all().item():
                continue
            for _ in range(2):
                c(*args)
            _torch.cuda.synchronize()
            e0 = _torch.cuda.Event(enable_timing=True)
            e1 = _torch.cuda.Event(enable_timing=True)
            _torch.cuda.synchronize()
            e0.record()
            for _ in range(20):
                c(*args)
            e1.record()
            _torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / 20
            if us < best_us:
                best_us = us
                best = (launch, (bm, 4, 0, xcd))
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"TN autotune found no working cfg for ({M},{N},{K})")
    _TN_AUTOTUNE_CACHE[key] = best
    return best


def gemm_fp8_tensorwise_flydsl_kernel(
    a: torch.Tensor,
    a_scale_inv: torch.Tensor,
    b: torch.Tensor,
    b_scale_inv: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
    trans_c: bool = False,
) -> torch.Tensor:
    """Dense FP8 GEMM, per-tensor scaling. Inputs E4M3/E5M2/hybrid, out bf16/fp16,
    arbitrary K (native K-tail). Dispatch by (trans_a, trans_b): NT (F,T), NN
    (F,F, dgrad), TN (T,F) run native; TT (T,T) unsupported. trans_c=True returns
    out.t().contiguous()."""
    # 【入口分发函数】这是对外暴露的 Python 接口。按 (trans_a, trans_b) 路由到
    # NT/NN/TN 三个原生内核；TT 不支持。trans_c=True 时把输出再转置一次返回。
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"FlyDSL wrapper emits bf16 or fp16. Got {out_dtype}.")
    assert a.dim() == 2 and b.dim() == 2
    # 每个操作数的 fp8 格式 -> MFMA 的 cbsz(srcA)/blgp(srcB)：0=E4M3，1=E5M2。
    # 支持 A、B 各自独立选 E4M3 或 E5M2（含混合）。
    cbsz = 1 if a.dtype == torch.float8_e5m2 else 0
    blgp = 1 if b.dtype == torch.float8_e5m2 else 0
    # 输出 fp16 还是 bf16（都从 f32 累加器转换而来）。
    out_fp16 = out_dtype == torch.float16

    if trans_a and (not trans_b):
        # TN 原生：A [K, M]，B [K, N]，数学上 C = A^T @ B。
        # （反向传播 wgrad 的典型形态。）
        K_a, M = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"TN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # TN：per-shape autotune 选最优配置，按 (M,N,K) 缓存。注意 args 要在 autotune
        # 之前就构造好（autotune 直接拿它做基准测试）。
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_tn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
        if trans_c:
            return out.t().contiguous()
        return out

    # 其余 layout 分发。
    if (not trans_a) and (not trans_b):
        # NN 原生：A [M, K]，B [K, N]（反向传播 dgrad 常用）。
        M, K_a = a.shape
        K_b, N = b.shape
        assert K_a == K_b, f"NN K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NN: per-shape runtime autotune over the candidate tiles, caches by
        # (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nn_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    elif (not trans_a) and trans_b:
        # NT 原生：A [M, K]，B [N, K]（[K, N] 的转置存储）。最快的 layout。
        M, K_a = a.shape
        N, K_b = b.shape
        assert K_a == K_b, f"NT K mismatch: a {a.shape}, b {b.shape}"
        K = K_a
        a_scale_v = _scalar_scale(a_scale_inv, a.device)
        b_scale_v = _scalar_scale(b_scale_inv, a.device)
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)
        # NT: per-shape runtime autotune over the 8w/v3 candidate tiles, caches
        # by (M,N,K). Build args before autotune (it benches against them).
        args = (
            _as_i8_flat(a),
            _as_i8_flat(b),
            out.contiguous().view(-1),
            a_scale_v,
            b_scale_v,
            M,
            N,
            torch.cuda.current_stream(),
        )
        launch, _cfg = _autotune_nt_dispatch(args, M, N, K, cbsz, blgp, out_fp16)
        _get_compiled_dense(launch, args)(*args)
    else:
        raise NotImplementedError(
            f"FlyDSL fp8 GEMM does not support the TT layout " f"(trans_a={trans_a}, trans_b={trans_b})."
        )
    if trans_c:
        return out.t().contiguous()
    return out
