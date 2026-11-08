/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hipBLASLt grouped GEMM for JAX with device-side group sizes.
//
// One hipblaslt_ext::GroupedGemm launch covers every group. Its per-group problem sizes and pointers
// (UserArguments) are written on the device from group_lens by a small kernel, so the FFI handler never
// reads group sizes on the host and never synchronizes the stream. Host-side blocking inside an FFI
// handler deadlocks against collectives when one process drives several GPUs (single-controller JAX).
//
// hipBLASLt's heuristic ranks tiny-tile kernels first for grouped problems (~25 TFLOPS on MI325X), so
// the algorithm is chosen by timing candidates in TuneHipblasltGroupedGemm, which the Python wrapper
// calls at trace time (outside any multi-device program, where synchronizing is safe). The execution
// path only looks the choice up and falls back to the heuristic when nothing was tuned.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "../extensions.h"

namespace primus_turbo::jax {

namespace {

using UserArgs = hipblaslt_ext::UserArguments;

constexpr int    kMaxGroups          = 1024;
constexpr size_t kAlgoWorkspaceBytes = 1 << 20;
// Zeroed UserArguments entries after the last group: some kernels read past the last group's entry.
constexpr int64_t kArgsPad = 16;

// Row-major problem mapped onto column-major hipBLASLt by passing b as hipBLASLt's A and a as
// hipBLASLt's B (same mapping as the PyTorch hipBLASLt grouped GEMM):
//   transA=false: c[M, N] rows of group g = a[rows of g, K] x b[g]    -> hip m=N, n=len, k=K
//   transA=true : c[g] [K_a, N] = a[rows of g, K_a]^T x b[rows of g, N] -> hip m=N, n=K_a, k=len
struct Dims {
    bool    transA = false, transB = false;
    int     dtype  = 0; // 0: bf16, 1: fp16, 2: fp8 e4m3 inputs (per-tensor scales) with bf16 output
    int64_t G = 0, M = 0;
    int64_t a_cols = 0;           // K (fwd) or K_a (transA)
    int64_t b_d1 = 0, b_d2 = 0;   // b dims after the group dim (fwd), or b [M, N] (transA)
    int64_t c_rows = 0;           // hip m = last dim of c

    bool               fp8() const { return dtype == 2; }
    hipDataType        type() const { return dtype == 1 ? HIP_R_16F : HIP_R_16BF; }      // output type
    hipDataType        in_type() const { return fp8() ? HIP_R_8F_E4M3 : type(); }         // a / b type
    int64_t            in_es() const { return fp8() ? 1 : 2; }
    hipblasOperation_t opA() const { return transB ? HIPBLAS_OP_T : HIPBLAS_OP_N; }
    hipblasOperation_t opB() const { return transA ? HIPBLAS_OP_T : HIPBLAS_OP_N; }
    int64_t a_bytes() const { return M * a_cols * in_es(); }
    int64_t b_bytes() const { return (transA ? b_d1 * b_d2 : G * b_d1 * b_d2) * in_es(); }
    int64_t c_bytes() const { return (transA ? G * a_cols * c_rows : M * c_rows) * 2; }
};

using TuneKey = std::tuple<bool, bool, int, int64_t, int64_t, int64_t, int64_t>;
TuneKey tune_key(const Dims &d) {
    return {d.transA, d.transB, d.dtype, d.G, d.a_cols, d.transA ? 0 : d.b_d1 * d.b_d2, d.c_rows};
}
std::mutex              g_tune_mu;
std::map<TuneKey, int>  g_tuned_index;

hipblasLtHandle_t get_handle(int device) {
    static std::mutex                       mu;
    static std::map<int, hipblasLtHandle_t> handles;
    std::lock_guard<std::mutex>             lock(mu);
    auto                                    it = handles.find(device);
    if (it != handles.end())
        return it->second;
    hipblasLtHandle_t handle = nullptr;
    PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handle));
    handles.emplace(device, handle);
    return handle;
}

std::unique_ptr<hipblaslt_ext::GroupedGemm> make_gemm(hipblasLtHandle_t handle, const Dims &d) {
    return std::make_unique<hipblaslt_ext::GroupedGemm>(handle, d.opA(), d.opB(), d.in_type(), d.in_type(),
                                                        d.type(), d.type(), HIPBLAS_COMPUTE_32F);
}

// Set every group to `rows` (hip n, or k for transA). Group pointers follow the real layout when the
// buffers are big enough (tuning), which only matters for timing; at run time they come from UserArgs.
hipblasStatus_t set_problem(hipblaslt_ext::GroupedGemm &gemm, const Dims &d, int64_t rows,
                            const void *a, const void *b, void *c, bool real_layout) {
    const int64_t        G = d.G;
    std::vector<int64_t> m(G, d.c_rows), n(G, d.transA ? d.a_cols : rows), k(G, d.transA ? rows : d.a_cols);
    std::vector<int64_t> batch(G, 1), lda(G, d.b_d2), ldb(G, d.a_cols), ldc(G, d.c_rows), ldd(G, d.c_rows);
    std::vector<int64_t> zero(G, 0);
    std::vector<hipblaslt_ext::GemmEpilogue> epilogue(G);
    if (d.fp8())
        for (auto &e : epilogue) {
            e.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
            e.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        }
    std::vector<hipblaslt_ext::GemmInputs>   inputs(G);
    static const float                       one = 1.0f, beta0 = 0.0f;
    for (int64_t g = 0; g < G; ++g) {
        const int64_t a_off = real_layout ? g * rows * d.a_cols * d.in_es() : 0;
        const int64_t b_off = real_layout ? (d.transA ? g * rows * d.b_d2 : g * d.b_d1 * d.b_d2) * d.in_es() : 0;
        const int64_t c_off = real_layout ? (d.transA ? g * d.a_cols * d.c_rows : g * rows * d.c_rows) * 2 : 0;
        inputs[g].setA(static_cast<const char *>(b) + b_off);
        inputs[g].setB(static_cast<const char *>(a) + a_off);
        inputs[g].setC(static_cast<char *>(c) + c_off);
        inputs[g].setD(static_cast<char *>(c) + c_off);
        inputs[g].setAlpha(&one);
        inputs[g].setBeta(&beta0);
        if (d.fp8()) { // placeholders (any valid device pointer); the real scale pointers go into the UserArgs
            inputs[g].setScaleA(a);
            inputs[g].setScaleB(a);
        }
    }
    hipblaslt_ext::GemmProblemType problem(d.opA(), d.opB(), d.in_type(), d.in_type(), d.type(), d.type(),
                                           HIPBLAS_COMPUTE_32F);
    return gemm.setProblem(m, n, k, batch, lda, ldb, ldc, ldd, zero, zero, zero, zero, epilogue, inputs,
                           problem);
}

// Whether `algo` handles arbitrary group sizes up to `max_rows` with the plan's workspace.
bool supports_any_size(hipblaslt_ext::GroupedGemm &gemm, const Dims &d, int64_t max_rows,
                       hipblasLtMatmulAlgo_t &algo, const void *a, const void *b, void *c) {
    for (int64_t rows : {int64_t(1), int64_t(7), max_rows / 2 + 1, max_rows}) {
        size_t ws = 0;
        if (set_problem(gemm, d, std::max<int64_t>(rows, 1), a, b, c, false) != HIPBLAS_STATUS_SUCCESS ||
            gemm.isAlgoSupported(algo, ws) != HIPBLAS_STATUS_SUCCESS || ws > kAlgoWorkspaceBytes)
            return false;
    }
    return true;
}

// Per-group size of the host-side problem passed to setProblem before initialize. The launch grid is the
// tile count of this problem, and work-groups are assigned to groups dynamically from the device-side sizes,
// so the grid only has to cover the real tiles: sum_g ceil(len_g / tile) <= ceil(M / tile) + G. Giving every
// group ceil(M / G) + tile rows (`tile` = the kernel's macro tile along the group-size dimension) covers that
// for any routing skew. PRIMUS_TURBO_HB_GG_HOST_ROWS=max (M rows per group) also does, at the cost of G times
// more idle work-groups.
int64_t host_rows(const Dims &d, int64_t group_tile) {
    const char *env = std::getenv("PRIMUS_TURBO_HB_GG_HOST_ROWS");
    if ((env && std::string(env) == "max") || group_tile <= 0)
        return d.M;
    return std::min<int64_t>(d.M, (d.M + d.G - 1) / d.G + group_tile);
}

// Macro tile along hipBLASLt's n (the group-size dimension for transA=false) from the kernel name
// "..._MT<m>x<n>x<k>_...", or 0 if it can't be parsed.
int64_t group_tile_of(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t &algo) {
    static const std::regex mt_re("_MT([0-9]+)x([0-9]+)x");
    std::smatch             m;
    const std::string       name = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
    return std::regex_search(name, m, mt_re) ? std::stoll(m[2]) : 0;
}

struct Plan {
    std::unique_ptr<hipblaslt_ext::GroupedGemm> gemm;
    UserArgs                                   *host_template = nullptr; // pinned
    void                                       *algo_ws       = nullptr;
    std::mutex                                  mu;
    ~Plan() {
        if (host_template)
            (void) hipHostFree(host_template);
        if (algo_ws)
            (void) hipFree(algo_ws);
    }
};

Plan *get_plan(int device, hipblasLtHandle_t handle, const Dims &d, const void *a, const void *b, void *c) {
    using PlanKey = std::tuple<int, TuneKey, int64_t, int>;
    static std::mutex                               mu;
    static std::map<PlanKey, std::unique_ptr<Plan>> plans;
    // PRIMUS_TURBO_HB_GG_PICK_{NN,NT,TN}=k: use the k-th supported heuristic candidate instead of the first
    // (kernel sweeps; read at every call so one process can try many).
    const char *pick_env = std::getenv(d.transA ? "PRIMUS_TURBO_HB_GG_PICK_TN" : d.transB ? "PRIMUS_TURBO_HB_GG_PICK_NT" : "PRIMUS_TURBO_HB_GG_PICK_NN");
    const int   pick     = pick_env ? std::atoi(pick_env) : 0;
    PlanKey                     key{device, tune_key(d), d.M, pick};
    std::lock_guard<std::mutex> lock(mu);
    auto                        it = plans.find(key);
    if (it != plans.end())
        return it->second.get();

    auto plan  = std::make_unique<Plan>();
    plan->gemm = make_gemm(handle, d);
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&plan->algo_ws, kAlgoWorkspaceBytes));

    const int64_t         max_rows = d.M; // a group can hold every row of a
    hipblasLtMatmulAlgo_t algo;
    bool                  have = false;
    int                   tuned = -1;
    {
        std::lock_guard<std::mutex> tl(g_tune_mu);
        auto                        t = g_tuned_index.find(tune_key(d));
        if (t != g_tuned_index.end())
            tuned = t->second;
    }
    if (tuned >= 0) {
        std::vector<int>                              idx{tuned};
        std::vector<hipblasLtMatmulHeuristicResult_t> res;
        if (hipblaslt_ext::getAlgosFromIndex(handle, idx, res) == HIPBLAS_STATUS_SUCCESS && !res.empty() &&
            supports_any_size(*plan->gemm, d, max_rows, res[0].algo, a, b, c)) {
            algo = res[0].algo;
            have = true;
        }
    }
    if (!have) {
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kAlgoWorkspaceBytes);
        std::vector<hipblasLtMatmulHeuristicResult_t> cand;
        PRIMUS_TURBO_CHECK_HIPBLAS(set_problem(*plan->gemm, d, std::max<int64_t>(1, d.M / d.G) | 1, a, b, c, false));
        PRIMUS_TURBO_CHECK_HIPBLAS(plan->gemm->algoGetHeuristic(pick_env ? 256 : 16, pref, cand));
        int seen = 0;
        for (auto &h : cand) {
            if (supports_any_size(*plan->gemm, d, max_rows, h.algo, a, b, c)) {
                if (seen++ < pick)
                    continue;
                algo = h.algo;
                have = true;
                break;
            }
        }
        PRIMUS_TURBO_CHECK(have, "hipBLASLt grouped GEMM: no algorithm supports arbitrary group sizes");
    }
    // The launch grid comes from the host-side sizes; see host_rows.
    const int64_t rows = host_rows(d, group_tile_of(handle, algo));
    PRIMUS_TURBO_CHECK_HIPBLAS(set_problem(*plan->gemm, d, rows, a, b, c, false));
    PRIMUS_TURBO_CHECK_HIPBLAS(plan->gemm->initialize(algo, plan->algo_ws, true));
    if (std::getenv("PRIMUS_TURBO_HB_GG_VERBOSE")) {
        fprintf(stderr, "[primus_turbo] hipblaslt grouped gemm plan dev=%d G=%ld M=%ld transA=%d transB=%d "
                "m=%ld fixed=%ld tuned=%d host_rows=%ld: %s\n", device, d.G, d.M, d.transA, d.transB, d.c_rows,
                d.a_cols, tuned, rows, hipblaslt_ext::getKernelNameFromAlgo(handle, algo).c_str());
    }
    PRIMUS_TURBO_CHECK_HIP(
        hipHostMalloc(reinterpret_cast<void **>(&plan->host_template), d.G * sizeof(UserArgs)));
    PRIMUS_TURBO_CHECK_HIPBLAS(plan->gemm->getDefaultValueForDeviceUserArguments(plan->host_template));

    Plan *raw = plan.get();
    plans.emplace(key, std::move(plan));
    return raw;
}

// Patch sizes and pointers of every group from the device-resident group sizes (one block).
template <typename LenT>
__global__ void fill_user_args_kernel(UserArgs *args, const LenT *group_lens, int64_t G, bool transA,
                                      const char *a, const char *b, char *c, int64_t a_cols,
                                      int64_t b_group_elems, int64_t b_cols, int64_t c_rows,
                                      int64_t c_group_elems, int64_t in_bytes, int64_t out_bytes,
                                      const void *a_scale, const void *b_scale) {
    __shared__ int64_t offs[kMaxGroups + 1];
    if (threadIdx.x == 0) {
        int64_t acc = 0;
        for (int64_t g = 0; g < G; ++g) {
            offs[g] = acc;
            acc += static_cast<int64_t>(group_lens[g]);
        }
        offs[G] = acc;
    }
    __syncthreads();
    for (int64_t g = threadIdx.x; g < G; g += blockDim.x) {
        const int64_t len = static_cast<int64_t>(group_lens[g]);
        const int64_t off = offs[g];
        UserArgs     &u   = args[g];
        if (!transA) {
            u.m = static_cast<uint32_t>(len == 0 ? 0 : c_rows);
            u.n = static_cast<uint32_t>(len);
            u.k = static_cast<uint32_t>(a_cols);
            u.a = const_cast<char *>(b) + g * b_group_elems * in_bytes;
            u.b = const_cast<char *>(a) + off * a_cols * in_bytes;
            u.c = c + off * c_rows * out_bytes;
        } else {
            u.m = static_cast<uint32_t>(len == 0 ? 0 : c_rows);
            u.n = static_cast<uint32_t>(len == 0 ? 0 : a_cols);
            u.k = static_cast<uint32_t>(len);
            u.a = const_cast<char *>(b) + off * b_cols * in_bytes;
            u.b = const_cast<char *>(a) + off * a_cols * in_bytes;
            u.c = c + g * c_group_elems * out_bytes;
        }
        u.d = u.c;
        if (a_scale != nullptr) { // fp8: hipBLASLt's A is our b, hipBLASLt's B is our a
            u.scaleA = const_cast<void *>(b_scale);
            u.scaleB = const_cast<void *>(a_scale);
        }
    }
}

// Zero rows [sum(group_lens), M) of c [M, N] (transA=false), or the c[g] slices of empty groups (transA).
template <typename LenT>
__global__ void zero_unwritten_kernel(uint16_t *c, const LenT *group_lens, int64_t G, bool transA,
                                      int64_t M, int64_t row_elems, int64_t group_elems) {
    if (!transA) {
        int64_t total = 0;
        for (int64_t g = 0; g < G; ++g)
            total += static_cast<int64_t>(group_lens[g]);
        const int64_t begin = total * row_elems, end = M * row_elems;
        for (int64_t i = begin + blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < end;
             i += static_cast<int64_t>(gridDim.x) * blockDim.x)
            c[i] = 0;
    } else {
        const int64_t g = blockIdx.y;
        if (group_lens[g] != 0)
            return;
        uint16_t *slice = c + g * group_elems;
        for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < group_elems;
             i += static_cast<int64_t>(gridDim.x) * blockDim.x)
            slice[i] = 0;
    }
}

// Pseudo-random bf16/fp16-sized values in about [-0.1, 0.1] (tuning only). Constant data runs at lower power
// and ranks kernels differently from real training data.
__global__ void fill_random_kernel(uint16_t *p, size_t n, uint32_t seed, bool bf16) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        x ^= x >> 13;
        x *= 0x5bd1e995u;
        x ^= x >> 15;
        const float v = ((x & 0xffff) / 65535.0f - 0.5f) * 0.2f;
        p[i] = bf16 ? static_cast<uint16_t>(__float_as_uint(v) >> 16) : __half_as_ushort(__float2half(v));
    }
}

// Random e4m3 bytes with magnitudes in about [0.06, 16] (tuning only).
__global__ void fill_random_fp8_kernel(uint8_t *p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t x = static_cast<uint32_t>(i) * 2654435761u ^ seed;
        x ^= x >> 13;
        x *= 0x5bd1e995u;
        x ^= x >> 15;
        p[i] = static_cast<uint8_t>((x & 0x80) | (0x28 + ((x >> 8) % 0x30)));
    }
}

// Enqueue fill + zero kernels and the grouped GEMM on `stream`.
template <typename LenT>
void launch(Plan *plan, const Dims &d, hipStream_t stream, UserArgs *dev_args, const LenT *gl,
            const void *a, const void *b, void *c, const void *a_scale = nullptr, const void *b_scale = nullptr) {
    (void) hipGetLastError(); // clear sticky errors left by candidate probing (hipBLASLt isAlgoSupported/allocations)
    PRIMUS_TURBO_CHECK_HIP(hipMemcpyAsync(dev_args, plan->host_template, d.G * sizeof(UserArgs),
                                          hipMemcpyHostToDevice, stream));
    PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(dev_args + d.G, 0, kArgsPad * sizeof(UserArgs), stream));
    const int64_t b_group_elems = d.transA ? 0 : d.b_d1 * d.b_d2;
    const int64_t c_group_elems = d.transA ? d.a_cols * d.c_rows : 0;
    const int     fill_threads  = static_cast<int>(std::min<int64_t>(256, d.G));
    fill_user_args_kernel<LenT><<<1, fill_threads, 0, stream>>>(
        dev_args, gl, d.G, d.transA, static_cast<const char *>(a), static_cast<const char *>(b),
        static_cast<char *>(c), d.a_cols, b_group_elems, d.b_d2, d.c_rows, c_group_elems, d.in_es(), 2,
        a_scale, b_scale);
    const dim3 zero_grid = d.transA ? dim3(64, static_cast<unsigned>(d.G)) : dim3(1024);
    zero_unwritten_kernel<LenT><<<zero_grid, 256, 0, stream>>>(static_cast<uint16_t *>(c), gl, d.G,
                                                              d.transA, d.M, d.c_rows, c_group_elems);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
    PRIMUS_TURBO_CHECK_HIPBLAS(plan->gemm->run(dev_args, stream));
}

std::vector<int64_t> dims_of(const ffi::AnyBuffer &buf) {
    auto d = buf.dimensions();
    return std::vector<int64_t>(d.begin(), d.end());
}

} // namespace

int64_t GetHipblasltGroupedGemmWorkspaceSize(int64_t group_num) {
    return (group_num + kArgsPad) * static_cast<int64_t>(sizeof(UserArgs));
}

// Time the grouped GEMM algorithms that handle arbitrary group sizes on the current device and remember
// the fastest for this problem signature. Synchronizes the device: call it outside multi-device programs
// (the Python wrapper does so at trace time). Returns (algo index, ms, kernel name); index -1 if none.
std::tuple<int, double, std::string> TuneHipblasltGroupedGemm(int64_t G, int64_t M, int64_t a_cols,
                                                              int64_t b_d1, int64_t b_d2, int64_t c_rows,
                                                              bool transA, bool transB, int dtype,
                                                              int max_candidates) {
    Dims d;
    d.transA = transA, d.transB = transB, d.dtype = dtype, d.G = G, d.a_cols = a_cols;
    d.b_d1 = b_d1, d.b_d2 = b_d2, d.c_rows = c_rows;
    {
        std::lock_guard<std::mutex> tl(g_tune_mu);
        if (g_tuned_index.count(tune_key(d)))
            return {g_tuned_index[tune_key(d)], 0.0, "cached"};
    }
    // Tune at the real size up to a cap (PRIMUS_TURBO_HB_GG_TUNE_MAX_ROWS, default 131072), halving it while
    // the buffers don't fit; the choice is then validated against the real problem in get_plan.
    const char   *cap_env = std::getenv("PRIMUS_TURBO_HB_GG_TUNE_MAX_ROWS");
    int64_t       m_tune  = std::max<int64_t>(G, std::min<int64_t>(M, cap_env ? std::atoll(cap_env) : 131072) / G * G);

    int device = 0;
    PRIMUS_TURBO_CHECK_HIP(hipGetDevice(&device));
    hipblasLtHandle_t handle = get_handle(device);
    // Some candidate kernels read slightly past the end of their operands. Inside a JAX process these
    // buffers sit at the top of the device address space (XLA preallocated the rest), where such reads
    // fault, so give every operand slack.
    constexpr size_t kSlack = 8 << 20;
    void *a = nullptr, *b = nullptr, *c = nullptr, *ws = nullptr, *dev_args = nullptr;
    for (;; m_tune = std::max<int64_t>(G, m_tune / 2 / G * G)) {
        d.M = m_tune;
        if (transA)
            d.b_d1 = m_tune;
        if (hipMalloc(&a, d.a_bytes() + kSlack) == hipSuccess &&
            hipMalloc(&b, d.b_bytes() + kSlack) == hipSuccess &&
            hipMalloc(&c, d.c_bytes() + kSlack) == hipSuccess)
            break;
        for (void **p : {&a, &b, &c})
            if (*p) {
                (void) hipFree(*p);
                *p = nullptr;
            }
        (void) hipGetLastError();
        PRIMUS_TURBO_CHECK(m_tune > G * 256, "hipBLASLt grouped GEMM tuning: out of device memory");
    }
    const int64_t rows = m_tune / G;
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&ws, kAlgoWorkspaceBytes + kSlack));
    PRIMUS_TURBO_CHECK_HIP(hipMalloc(&dev_args, (G + kArgsPad) * sizeof(UserArgs) + kSlack));
    PRIMUS_TURBO_CHECK_HIP(hipMemset(dev_args, 0, (G + kArgsPad) * sizeof(UserArgs)));
    if (d.fp8()) {
        fill_random_fp8_kernel<<<1024, 256>>>(static_cast<uint8_t *>(a), d.a_bytes(), 1);
        fill_random_fp8_kernel<<<1024, 256>>>(static_cast<uint8_t *>(b), d.b_bytes(), 2);
    } else {
        fill_random_kernel<<<1024, 256>>>(static_cast<uint16_t *>(a), d.a_bytes() / 2, 1, d.dtype == 0);
        fill_random_kernel<<<1024, 256>>>(static_cast<uint16_t *>(b), d.b_bytes() / 2, 2, d.dtype == 0);
    }
    PRIMUS_TURBO_CHECK_HIP(hipDeviceSynchronize());
    hipStream_t stream;
    hipEvent_t  e0, e1;
    PRIMUS_TURBO_CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&e0));
    PRIMUS_TURBO_CHECK_HIP(hipEventCreate(&e1));

    auto                                          gemm = make_gemm(handle, d);
    // Candidates: the heuristic's list for this problem. (getAllAlgos does not return the grouped "GG" solutions of
    // the Tensile library, which is why the tuner used to find at most the generic ones.)
    std::vector<hipblasLtMatmulHeuristicResult_t> all;
    {
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(kAlgoWorkspaceBytes);
        PRIMUS_TURBO_CHECK_HIPBLAS(set_problem(*gemm, d, std::max<int64_t>(1, rows) | 1, a, b, c, false));
        PRIMUS_TURBO_CHECK_HIPBLAS(gemm->algoGetHeuristic(1024, pref, all));
    }

    // Rank by macro-tile area (kernel name MT<m>x<n>, largest first) and keep the first max_candidates that
    // handle any group size. Checking support is the slow part, so stop once enough are found.
    static const std::regex                       mt_re("_MT([0-9]+)x([0-9]+)x");
    std::vector<std::pair<int64_t, size_t>>        ranked; // (-tile area, index into all)
    for (size_t i = 0; i < all.size(); ++i) {
        std::smatch m;
        std::string name = hipblaslt_ext::getKernelNameFromAlgo(handle, all[i].algo);
        int64_t     area = std::regex_search(name, m, mt_re) ? std::stoll(m[1]) * std::stoll(m[2]) : 0;
        ranked.emplace_back(-area, i);
    }
    std::sort(ranked.begin(), ranked.end());
    std::vector<std::pair<int64_t, size_t>> pool;
    std::vector<UserArgs>                   host_args(G);
    for (auto &r : ranked) {
        if (max_candidates > 0 && pool.size() >= static_cast<size_t>(max_candidates))
            break;
        if (supports_any_size(*gemm, d, m_tune, all[r.second].algo, a, b, c))
            pool.push_back(r);
    }

    const bool verbose = std::getenv("PRIMUS_TURBO_HB_GG_VERBOSE") != nullptr;
    if (verbose)
        fprintf(stderr, "[primus_turbo] tune: %zu heuristic candidates, %zu support any group size (m_tune=%ld rows=%ld)\n",
                all.size(), pool.size(), (long) m_tune, (long) rows);
    auto time_algo = [&](hipblasLtMatmulAlgo_t &algo, int iters) -> double {
        auto s1 = set_problem(*gemm, d, rows, a, b, c, true);
        auto s2 = s1 == HIPBLAS_STATUS_SUCCESS ? gemm->initialize(algo, ws, true) : s1;
        auto s3 = s2 == HIPBLAS_STATUS_SUCCESS ? gemm->getDefaultValueForDeviceUserArguments(host_args.data()) : s2;
        if (s3 != HIPBLAS_STATUS_SUCCESS) {
            if (verbose)
                fprintf(stderr, "[primus_turbo] tune: candidate rejected set_problem=%d initialize=%d args=%d\n", (int) s1, (int) s2, (int) s3);
            return -1;
        }
        PRIMUS_TURBO_CHECK_HIP(hipMemcpyAsync(dev_args, host_args.data(), G * sizeof(UserArgs),
                                              hipMemcpyHostToDevice, stream));
        if (gemm->run(dev_args, stream) != HIPBLAS_STATUS_SUCCESS)
            return -1;
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(e0, stream));
        for (int it = 0; it < iters; ++it)
            PRIMUS_TURBO_CHECK_HIPBLAS(gemm->run(dev_args, stream));
        PRIMUS_TURBO_CHECK_HIP(hipEventRecord(e1, stream));
        PRIMUS_TURBO_CHECK_HIP(hipEventSynchronize(e1));
        float ms = 0;
        PRIMUS_TURBO_CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
        return ms / iters;
    };
    std::vector<std::pair<double, size_t>> timed;
    for (auto &p : pool) {
        double ms = time_algo(all[p.second].algo, 3);
        if (ms > 0)
            timed.emplace_back(ms, p.second);
    }
    std::sort(timed.begin(), timed.end());
    int         best_idx = -1;
    double      best_ms  = 0;
    std::string best_name;
    for (size_t i = 0; i < std::min<size_t>(8, timed.size()); ++i) {
        double ms = time_algo(all[timed[i].second].algo, 10);
        if (ms > 0 && (best_idx < 0 || ms < best_ms)) {
            best_ms   = ms;
            best_idx  = hipblaslt_ext::getIndexFromAlgo(all[timed[i].second].algo);
            best_name = hipblaslt_ext::getKernelNameFromAlgo(handle, all[timed[i].second].algo);
        }
    }
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
    (void) hipStreamDestroy(stream);
    for (void *p : {a, b, c, ws, dev_args})
        (void) hipFree(p);
    if (best_idx >= 0) {
        std::lock_guard<std::mutex> tl(g_tune_mu);
        g_tuned_index[tune_key(d)] = best_idx;
    }
    return {best_idx, best_ms, best_name};
}

// Record an algorithm index (e.g. tuned in another process) for a problem signature.
void RegisterHipblasltGroupedGemmAlgo(int64_t G, int64_t a_cols, int64_t b_d1, int64_t b_d2, int64_t c_rows,
                                      bool transA, bool transB, int dtype, int algo_index) {
    Dims d;
    d.transA = transA, d.transB = transB, d.dtype = dtype, d.G = G, d.a_cols = a_cols;
    d.b_d1 = b_d1, d.b_d2 = b_d2, d.c_rows = c_rows;
    std::lock_guard<std::mutex> tl(g_tune_mu);
    g_tuned_index[tune_key(d)] = algo_index;
}

// a: [M, K]; b: [G, K, N] / [G, N, K] (transB) or [M, N] (transA); c: [M, N] or [G, K_a, N] (transA).
ffi::Error HipblasltGroupedGemmFFI(hipStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                   ffi::AnyBuffer group_lens, ffi::Result<ffi::AnyBuffer> c,
                                   ffi::Result<ffi::AnyBuffer> workspace, bool transA,
                                   bool transB) {
    const auto dtype = a.element_type();
    if (dtype != b.element_type() || (dtype != ffi::BF16 && dtype != ffi::F16) ||
        c->element_type() != dtype) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hipblaslt_grouped_gemm needs matching fp16/bf16 a, b and c");
    }
    if (transA && transB) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hipblaslt_grouped_gemm: transA and transB together are not supported");
    }
    const auto gl_type = group_lens.element_type();
    if (gl_type != ffi::S32 && gl_type != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "group_lens must be int32 or int64");
    }
    const std::vector<int64_t> a_shape = dims_of(a), b_shape = dims_of(b), c_shape = dims_of(*c);
    const int64_t              G       = static_cast<int64_t>(group_lens.element_count());
    if (G < 1 || G > kMaxGroups || a_shape.size() != 2 ||
        (transA ? (b_shape.size() != 2 || c_shape.size() != 3 || b_shape[0] != a_shape[0])
                : (b_shape.size() != 3 || c_shape.size() != 2 || b_shape[0] != G))) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "hipblaslt_grouped_gemm: bad shapes");
    }
    if (workspace->size_bytes() < static_cast<size_t>(GetHipblasltGroupedGemmWorkspaceSize(G))) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "hipblaslt_grouped_gemm: workspace too small");
    }

    Dims d;
    d.transA = transA;
    d.transB = transB;
    d.dtype  = dtype == ffi::BF16 ? 0 : 1;
    d.G      = G;
    d.M      = a_shape[0];
    d.a_cols = a_shape[1];
    d.b_d1   = transA ? b_shape[0] : b_shape[1];
    d.b_d2   = transA ? b_shape[1] : b_shape[2];
    d.c_rows = c_shape.back();

    int device = 0;
    PRIMUS_TURBO_CHECK_HIP(hipStreamGetDevice(stream, &device));
    Plan *plan = get_plan(device, get_handle(device), d, a.untyped_data(), b.untyped_data(), c->untyped_data());

    std::lock_guard<std::mutex> lock(plan->mu);
    auto *dev_args = static_cast<UserArgs *>(workspace->untyped_data());
    if (gl_type == ffi::S64)
        launch(plan, d, stream, dev_args, static_cast<const int64_t *>(group_lens.untyped_data()),
               a.untyped_data(), b.untyped_data(), c->untyped_data());
    else
        launch(plan, d, stream, dev_args, static_cast<const int32_t *>(group_lens.untyped_data()),
               a.untyped_data(), b.untyped_data(), c->untyped_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(HipblasltGroupedGemmHandler, HipblasltGroupedGemmFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // a
                                  .Arg<ffi::AnyBuffer>()                   // b
                                  .Arg<ffi::AnyBuffer>()                   // group_lens
                                  .Ret<ffi::AnyBuffer>()                   // c
                                  .Ret<ffi::AnyBuffer>()                   // workspace
                                  .Attr<bool>("transA")
                                  .Attr<bool>("transB"));

// FP8 (e4m3, per-tensor scales): a [M, K] e4m3, b like the bf16 handler but e4m3, a_scale / b_scale f32[1]
// (dequantisation factors, applied by the kernel epilogue), c bf16.
ffi::Error HipblasltGroupedGemmFp8FFI(hipStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                      ffi::AnyBuffer a_scale, ffi::AnyBuffer b_scale,
                                      ffi::AnyBuffer group_lens, ffi::Result<ffi::AnyBuffer> c,
                                      ffi::Result<ffi::AnyBuffer> workspace, bool transA, bool transB) {
    if (a.element_type() != ffi::F8E4M3FN || b.element_type() != ffi::F8E4M3FN ||
        c->element_type() != ffi::BF16 || a_scale.element_type() != ffi::F32 ||
        b_scale.element_type() != ffi::F32 || a_scale.element_count() != 1 || b_scale.element_count() != 1) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hipblaslt_grouped_gemm_fp8 needs e4m3 a, b, f32[1] scales and a bf16 c");
    }
    if (transA && transB) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "hipblaslt_grouped_gemm_fp8: transA and transB together are not supported");
    }
    const auto gl_type = group_lens.element_type();
    if (gl_type != ffi::S32 && gl_type != ffi::S64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "group_lens must be int32 or int64");
    }
    const std::vector<int64_t> a_shape = dims_of(a), b_shape = dims_of(b), c_shape = dims_of(*c);
    const int64_t              G       = static_cast<int64_t>(group_lens.element_count());
    if (G < 1 || G > kMaxGroups || a_shape.size() != 2 ||
        (transA ? (b_shape.size() != 2 || c_shape.size() != 3 || b_shape[0] != a_shape[0])
                : (b_shape.size() != 3 || c_shape.size() != 2 || b_shape[0] != G))) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "hipblaslt_grouped_gemm_fp8: bad shapes");
    }
    if (workspace->size_bytes() < static_cast<size_t>(GetHipblasltGroupedGemmWorkspaceSize(G))) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "hipblaslt_grouped_gemm_fp8: workspace too small");
    }
    Dims d;
    d.transA = transA;
    d.transB = transB;
    d.dtype  = 2;
    d.G      = G;
    d.M      = a_shape[0];
    d.a_cols = a_shape[1];
    d.b_d1   = transA ? b_shape[0] : b_shape[1];
    d.b_d2   = transA ? b_shape[1] : b_shape[2];
    d.c_rows = c_shape.back();

    int device = 0;
    PRIMUS_TURBO_CHECK_HIP(hipStreamGetDevice(stream, &device));
    Plan *plan = get_plan(device, get_handle(device), d, a.untyped_data(), b.untyped_data(), c->untyped_data());

    std::lock_guard<std::mutex> lock(plan->mu);
    auto *dev_args = static_cast<UserArgs *>(workspace->untyped_data());
    if (gl_type == ffi::S64)
        launch(plan, d, stream, dev_args, static_cast<const int64_t *>(group_lens.untyped_data()),
               a.untyped_data(), b.untyped_data(), c->untyped_data(), a_scale.untyped_data(), b_scale.untyped_data());
    else
        launch(plan, d, stream, dev_args, static_cast<const int32_t *>(group_lens.untyped_data()),
               a.untyped_data(), b.untyped_data(), c->untyped_data(), a_scale.untyped_data(), b_scale.untyped_data());
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(HipblasltGroupedGemmFp8Handler, HipblasltGroupedGemmFp8FFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                   // a (e4m3)
                                  .Arg<ffi::AnyBuffer>()                   // b (e4m3)
                                  .Arg<ffi::AnyBuffer>()                   // a_scale f32[1]
                                  .Arg<ffi::AnyBuffer>()                   // b_scale f32[1]
                                  .Arg<ffi::AnyBuffer>()                   // group_lens
                                  .Ret<ffi::AnyBuffer>()                   // c (bf16)
                                  .Ret<ffi::AnyBuffer>()                   // workspace
                                  .Attr<bool>("transA")
                                  .Attr<bool>("transB"));

} // namespace primus_turbo::jax
