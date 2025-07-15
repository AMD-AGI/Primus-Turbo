import triton
import triton.language as tl


@triton.jit
def compute_m_num_tiles_indptr(
    m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    group_id = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            group_id = bs

    idx_start = tl.load(m_num_tiles_indptr + group_id)

    m_range_start = tl.load(seg_indptr + group_id) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + group_id + 1), m_range_start + BLOCK_SIZE_M)
    return m_range_start, m_range_end, group_id


@triton.jit
def grouped_gemm_fp8_blockwise_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    batch_size,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    seg_indptr,
    m_num_tiles_indptr,
    transA: tl.constexpr,
    transB: tl.constexpr,
    SCALE_GROUP_SIZE_M: tl.constexpr,
    SCALE_GROUP_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c_ptr.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, group_id = compute_m_range(
        pid_m, batch_size, seg_indptr, m_num_tiles_indptr, BLOCK_SIZE_M
    )
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bsn = (n_range_start + offs_bn) // SCALE_GROUP_SIZE_N

    #
    if not transA:
        a_ptr = a_ptr + m_range_start * K
        a_ptrs = a_ptr + offs_am[:, None] * K + offs_k[None, :]
        a_ptrs_stride = BLOCK_SIZE_K

        a_s_ptr = a_s_ptr + m_range_start * tl.cdiv(K, SCALE_GROUP_SIZE_K)
        a_s_ptrs = a_s_ptr + offs_am[:, None] * tl.cdiv(K, SCALE_GROUP_SIZE_K)
        a_s_ptrs_stride = 1
    else:
        a_ptr = a_ptr + m_range_start
        a_ptrs = a_ptr + offs_k[:, None] * M + offs_am[None, :]
        a_ptrs_stride = BLOCK_SIZE_K * M

        a_s_ptr = a_s_ptr + m_range_start
        a_s_ptrs = a_s_ptr + offs_am[None, :]
        a_s_ptrs_stride = M

    if not transB:
        b_ptr = b_ptr + group_id * N * K + n_range_start
        b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn[None, :]
        b_ptrs_stride = BLOCK_SIZE_K * N

        b_s_ptr = b_s_ptr + group_id * tl.cdiv(N, SCALE_GROUP_SIZE_N) * tl.cdiv(K, SCALE_GROUP_SIZE_K)
        b_s_ptrs = b_s_ptr + offs_bsn[None, :]
        b_s_ptrs_stride = tl.cdiv(N, SCALE_GROUP_SIZE_N)
    else:
        b_ptr = b_ptr + group_id * N * K + n_range_start * K
        b_ptrs = b_ptr + offs_bn[:, None] * K + offs_k[None, :]
        b_ptrs_stride = BLOCK_SIZE_K

        b_s_ptr = b_s_ptr + group_id * tl.cdiv(N, SCALE_GROUP_SIZE_N) * tl.cdiv(K, SCALE_GROUP_SIZE_K)
        b_s_ptrs = b_s_ptr + offs_bsn[:, None] * tl.cdiv(K, SCALE_GROUP_SIZE_K)
        b_s_ptrs_stride = 1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kid in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K
        a_tile = tl.load(a_ptrs, mask_k[None, :] if not transA else mask_k[:, None], other=0.0)
        b_tile = tl.load(b_ptrs, mask_k[:, None] if not transB else mask_k[None, :], other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)

        if transA:
            a_tile = a_tile.T

        if transB:
            b_tile = b_tile.T

        accumulator += tl.dot(a_tile, b_tile) * a_s_tile * b_s_tile

        a_ptrs += a_ptrs_stride
        b_ptrs += b_ptrs_stride
        a_s_ptrs += a_s_ptrs_stride
        b_s_ptrs += b_s_ptrs_stride
        offs_k += BLOCK_SIZE_K

    c_tile = accumulator.to(c_dtype)
    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptrs, c_tile, mask=c_mask)


@triton.jit
def grouped_gemm_variable_k_fp8_blockwise_tn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    batch_size,
    M: tl.constexpr,
    N: tl.constexpr,
    K,
    seg_indptr,
    scales_seg_indptr,
    SCALE_GROUP_SIZE_M: tl.constexpr,
    SCALE_GROUP_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    k_range_start = tl.load(seg_indptr + pid_b)
    k_range_end = tl.load(seg_indptr + pid_b + 1)
    k_local = k_range_end - k_range_start
    if k_local == 0:
        return
    scales_k_range_start = tl.load(scales_seg_indptr + pid_b)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a_ptr + k_range_start * M
    a_ptrs = a_ptr + offs_k[:, None] * M + offs_am[None, :]
    a_ptrs_stride = BLOCK_SIZE_K * M
    
    b_ptr = b_ptr + k_range_start * N
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_bn[None, :]
    b_ptrs_stride = BLOCK_SIZE_K * N

    a_s_ptr = a_s_ptr + scales_k_range_start * tl.cdiv(M, SCALE_GROUP_SIZE_M)
    b_s_ptr = b_s_ptr + scales_k_range_start * tl.cdiv(N, SCALE_GROUP_SIZE_N)
    a_s_ptrs = a_s_ptr + offs_am[None, :]
    b_s_ptrs = b_s_ptr + offs_bn[None, :]
    a_s_ptrs_stride = tl.cdiv(M, SCALE_GROUP_SIZE_M)
    b_s_ptrs_stride = tl.cdiv(N, SCALE_GROUP_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kid in range(0, tl.cdiv(k_local, BLOCK_SIZE_K)):
        a_tile = tl.load(a_ptrs, mask=offs_k[:, None] < k_local, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_k[:, None] < k_local, other=0.0)
        a_s_tile = tl.load(a_s_ptrs)
        b_s_tile = tl.load(b_s_ptrs)

        accumulator += tl.dot(a_tile.T, b_tile) * a_s_tile.T * b_s_tile
        a_ptrs += a_ptrs_stride
        b_ptrs += b_ptrs_stride
        a_s_ptrs += a_s_ptrs_stride
        b_s_ptrs += b_s_ptrs_stride
        offs_k += BLOCK_SIZE_K

    c_tile = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_b * M * N + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_tile, mask=c_mask)
