#pragma once

#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <type_traits>

#ifndef __grid_constant__
#define __grid_constant__
#endif

#include "primus_turbo/device/lds_swizzle.cuh"
#include "primus_turbo/device/memory.cuh"
#include "primus_turbo/device/mfma.cuh"
#include "primus_turbo/device/register.cuh"
#include "primus_turbo/dtype.h"

#include "../layout/mega_moe.cuh"
#include "../layout/sym_buffer.cuh"
#include "../scheduler/mega_moe.cuh"
#include "prims.cuh"

namespace primus_turbo {
namespace mega_moe {

enum class MegaMoEArch : uint32_t {
    Unknown = 0,
    Gfx942  = 942,
    Gfx950  = 950,
};

// ABLATION (default off): when set, the MFMA matrix instruction in the k-loop is
// replaced by a trivial consume of the A/B/SF operands.  The global->LDS load
// chain (HBM traffic) and the whole pipeline are otherwise UNCHANGED, so the run
// measures the loader's achievable HBM bandwidth with the compute removed.
// Output is garbage -> always run with `--num-correctness-tests 0`.
#ifndef MEGA_MOE_NO_MFMA
#define MEGA_MOE_NO_MFMA 0
#endif

// PROBE (default off): validate route-A premise -- if the FP4 weight (B) were
// pre-swizzled in HBM and loaded global->VGPR, its ~64KB LDS pool would free up
// for a deeper A pipeline.  This probe drops B from the LDS budget (so more A
// stages fit) and skips the B load entirely.  Output is garbage; pair with
// MEGA_MOE_LOADS_ONLY=1 + --num-correctness-tests 0 and sweep KNUMSTAGES to
// measure the pure A-pipeline-depth effect at fixed (B-free) traffic.
#ifndef MEGA_MOE_BDIRECT_PROBE
#define MEGA_MOE_BDIRECT_PROBE 0
#endif

// OPT (default off, under test): nontemporal (streaming) store for the L1
// epilogue fp8-pool write.  The default cached store on a partial cache line
// triggers write-allocate (read-modify-write the 128B line); a nontemporal
// store bypasses write-allocate, eliminating the RMW read storm.  The L1 fp8
// store is ~31% of the kernel (latency-bound, width-independent), so if it is
// write-allocate-miss-bound this should cut it materially.
#ifndef MEGA_MOE_EPI_NT
#define MEGA_MOE_EPI_NT 0
#endif

// PROBE (default off): store a CONSTANT byte in the L1 epilogue, skipping the
// clamp + SwiGLU + fp8-convert chain but KEEPING the store + addressing.  This
// cleanly separates store-memory cost (stays if still slow) from feeding-compute
// cost (recovered if fast).  Garbage output -> --num-correctness-tests 0.
#ifndef MEGA_MOE_EPI_CONST
#define MEGA_MOE_EPI_CONST 0
#endif

// PROBE (default off): redirect the L1 epilogue fp8 store to a plain __device__
// LOCAL scratch buffer (same spread-out address pattern via a power-of-2 mask)
// instead of the symmetric-memory pool.  If the store is materially faster here,
// the symm/IPC mapping (not the access pattern) is the slow path and the fix is
// to move the L1->L2 intermediate pool into a local hipMalloc allocation.
// Garbage output -> --num-correctness-tests 0.  Pair with NO_MFMA.
#ifndef MEGA_MOE_EPI_LOCAL
#define MEGA_MOE_EPI_LOCAL 0
#endif
#if MEGA_MOE_EPI_LOCAL
// 64 MiB local scratch; mask keeps writes spread across it like the real pool.
#define MEGA_MOE_EPI_LOCAL_BYTES (64u * 1024u * 1024u)
__device__ uint8_t g_mega_moe_epi_scratch[MEGA_MOE_EPI_LOCAL_BYTES];
#endif
// Sub-probe of EPI_LOCAL: store a dword (4B) instead of a byte to test whether
// the cost is sub-dword (1B) store RMW serialization.
#ifndef MEGA_MOE_EPI_W4
#define MEGA_MOE_EPI_W4 0
#endif
// Sub-probe of EPI_LOCAL: store every element to a single cache-HOT location
// (~zero memory traffic, same store-instruction count + loop).  Fast => the
// 2.3ms is memory traffic; still slow => store-instruction-issue / loop bound.
#ifndef MEGA_MOE_EPI_HOT
#define MEGA_MOE_EPI_HOT 0
#endif

// OPT (default off, under test): DEFERRED EPILOGUE software-pipeline.  The L1 fp8
// epilogue store is ~27% of the kernel NOT because the store is slow, but because
// it sits between block N's k-loop and block N+1's loads, breaking the inter-block
// load pipeline (exposing load latency).  This defers block N's epilogue to run
// during block N+1's PROLOGUE loads (overlapping the latency).  acc/acc_gate
// persist across blocks; the deferred epilogue reads them before kloop_main resets.
// Deadlock-safe: defers only WITHIN a phase and FLUSHES pending before any L2
// arrival-wait (an L2 block's cross-SM l2_arrival needs the L1 epilogue signals).
// Requires the non-coalesce store path (COALESCE_EPI reuses the B-pool LDS the
// prologue just loaded).
// RESULT (default off, NOT promoted): correct (gate-3 PASS, no deadlock) but ~0%
// E2E (9067 vs 9069us).  Overlapping the epilogue with the next block's PROLOGUE
// loads recovers nothing -> the ~2.9ms epilogue is serial work that load-overlap
// cannot hide; real hiding needs overlap with the MAIN k-loop / MFMA (warp
// specialization or per-k-iter interleaving).  Kept as gated scaffolding.
#ifndef MEGA_MOE_DEFER_EPI
#define MEGA_MOE_DEFER_EPI 0
#endif

// PROBE/OPT (default off): force run_epilogue to NOT inline, so its registers do
// not bloat the k-loop's register allocation / instruction scheduling.  The store
// pattern is fast in isolation (~2670 GB/s standalone); the full-kernel epilogue
// cost (~2.5ms) is a codegen interaction (VGPR 180->239 when inlined).  If
// non-inlining drops main-kernel VGPR + speeds the k-loop -> codegen confirmed.
#ifndef MEGA_MOE_EPI_NOINLINE
#define MEGA_MOE_EPI_NOINLINE 0
#endif

// OPT (PROMOTED, default on, 7th win): K-major weight-SF (SFB) layout.  The
// n-major weight-SF made the per-k_block SFB load scatter (each lane a 4B load at
// stride K/32).  K-major ([E][k_block][N]) makes BLOCK_N consecutive columns
// contiguous -> coalesced.  +4.2% E2E (9150->8769us), gate-3 PASS cos_sim
// 0.999956.  Requires the matching offline transpose (test, same flag).  Reference
// uses the untransformed SF -> gate-3 unaffected.  Set MEGA_MOE_SFB_KMAJOR=0 to revert.
#ifndef MEGA_MOE_SFB_KMAJOR
#define MEGA_MOE_SFB_KMAJOR 1
#endif

// Dispatch token-pull warp_copy unroll (in-flight XGMI loads/lane).  Default 5;
// 7 int4/lane so 8 = all in one batch.  Tunable to probe the latency-bound pull.
#ifndef MEGA_MOE_PULL_UNROLL
#define MEGA_MOE_PULL_UNROLL 5u
#endif

// PROBE (default off): swap the dispatch token copy direction (local-read +
// remote-WRITE instead of remote-read + local-write) to measure XGMI write vs
// read bandwidth for the dispatch volume -- validates the push-dispatch idea
// before the full protocol rewrite.  Garbage output -> --num-correctness-tests 0.
#ifndef MEGA_MOE_PULL_AS_PUSH
#define MEGA_MOE_PULL_AS_PUSH 0
#endif

// OPT (default off, under test): PUSH dispatch.  Source ranks WRITE their tokens
// into the dst pools (remote write, ~6x faster than the pull's remote read --
// validated -662us/-7.5% via PULL_AS_PUSH) + remote notify, replacing the
// dst-driven pull.  Needs a cross-rank nvlink_barrier after the push (vs the
// pull's intra-rank grid_sync) since writes target remote pools.
#ifndef MEGA_MOE_DISPATCH_PUSH
#define MEGA_MOE_DISPATCH_PUSH 0
#endif
#ifndef MEGA_MOE_PUSH_GRIDSYNC_PROBE
#define MEGA_MOE_PUSH_GRIDSYNC_PROBE 0
#endif
#ifndef MEGA_MOE_PUSH_SKIP_META
#define MEGA_MOE_PUSH_SKIP_META 0
#endif

// OPT (default off, under test): PUSH approach-2.  The pull does resolution +
// SF/metadata (cheap LOCAL) but writes the dst pool slot back to each SRC's
// mapping (combine_buffer reused); the src then PUSHES only the big token data
// (fast remote write) + notify.  Avoids approach-1's SF-remote-scatter loss;
// targets the validated ~662us token-data-write win.
#ifndef MEGA_MOE_DISPATCH_PUSH2
#define MEGA_MOE_DISPATCH_PUSH2 0
#endif
#if MEGA_MOE_DEFER_EPI && MEGA_MOE_COALESCE_EPI
#error "MEGA_MOE_DEFER_EPI requires the non-coalesce store path (COALESCE_EPI=0)"
#endif
#if MEGA_MOE_AGPR_ACC &&                                                                            \
    (MEGA_MOE_COALESCE_EPI || MEGA_MOE_EPI_BATCH_STORE || MEGA_MOE_EPI_LDS_BULK ||                  \
     MEGA_MOE_EPI_NT || MEGA_MOE_EPI_CONST || MEGA_MOE_EPI_LOCAL || MEGA_MOE_EPI_W4 ||              \
     MEGA_MOE_EPI_HOT || MEGA_MOE_DEFER_EPI)
#error "MEGA_MOE_AGPR_ACC requires the plain default scalar epilogue (no COALESCE/BATCH/LDS_BULK/NT/CONST/LOCAL/W4/HOT/DEFER)"
#endif

// OPT (PROMOTED, default on): cheaper L1-epilogue quant -- fmed3f clamp (1 op
// vs fmaxf+fminf) and a direct HW cvt_pk_fp8 that skips the cast's redundant
// saturate branch (swiglu is already clamped to +/-kActivationClamp << fp8 range).
// Targets the ~0.9ms epilogue compute; ~0.5% E2E, gate-3 PASS cos_sim 0.999963
// (identical to baseline).  Set MEGA_MOE_FAST_QUANT=0 to revert.
#ifndef MEGA_MOE_FAST_QUANT
#define MEGA_MOE_FAST_QUANT 1
#endif

// ABLATION (default off): turn ONLY the GEMM A/B/SF global->LDS loads into
// no-ops.  Every barrier, arrival-spin, epilogue write, combine, and counter
// reset is left intact, so the persistent bench loop stays healthy (no
// early-return, no broken counters / deadlock).  Combined with
// MEGA_MOE_NO_MFMA=1 the whole GEMM phase becomes ~free, so the measured
// kernel time isolates dispatch + combine + synchronization.  Output is
// garbage -> always run with `--num-correctness-tests 0`.
#ifndef MEGA_MOE_SKIP_GEMM_LOADS
#define MEGA_MOE_SKIP_GEMM_LOADS 0
#endif

// ABLATION (default off): skip ONLY the combine top-k reduce loop (per-rank
// combine-buffer reads + reduction + BF16 y write).  The pre-combine
// nvlink_barrier and the next-launch counter resets are kept, so the bench
// loop stays healthy.  Full - this = combine read/reduce/write contribution.
// Garbage output -> run with `--num-correctness-tests 0`.
#ifndef MEGA_MOE_SKIP_COMBINE
#define MEGA_MOE_SKIP_COMBINE 0
#endif

// ABLATION (default off): skip ONLY the cross-rank dispatch DATA pull (the
// warp_copy_int4 token copy + SF copy over XGMI).  The routing, count
// exchanges, metadata writes, l1_arrival_count signal, and all barriers are
// kept, so the compute loop still proceeds (on garbage) and nothing hangs.
// Full(SKIP_GEMM+SKIP_COMBINE) - this = dispatch-pull XGMI-bandwidth cost,
// isolating it from arrival-spin / barrier / epilogue-store sync.
// Garbage output -> run with `--num-correctness-tests 0`.
#ifndef MEGA_MOE_SKIP_DISPATCH_PULL
#define MEGA_MOE_SKIP_DISPATCH_PULL 0
#endif

// ABLATION (default off): skip ONLY the epilogue global STORES -- the L1 FP8
// requant write into the L2 pool and the L2 BF16 ``write_combine`` write into
// the remote combine buffer.  Every arrival-spin, the L2-arrival release
// signal, all barriers, dispatch, MFMA, and combine stay intact (the bench
// loop stays healthy -- the L2 consumer still proceeds on the kept arrival
// signal, reading garbage).  Full - this = the epilogue store-traffic
// contribution, used to split the ~13ms "arrival-spin + epilogue store"
// residual into spin-idle vs store cost.  Garbage output -> always run with
// `--num-correctness-tests 0`.
#ifndef MEGA_MOE_SKIP_EPI
#define MEGA_MOE_SKIP_EPI 0
#endif

// Optimization (default off, under test): coalesce the L2 epilogue combine
// stores.  The MFMA output layout gives lane l the columns n = (l&15) + sub_n*16
// (strided by 16) for its row, so the scalar path emits kSubTilesN separate
// 32-byte XGMI transactions per row.  With COALESCE_EPI the warp transposes each
// row through LDS (reusing the loader's now-free B-pool smem) so each lane writes
// a contiguous uint4 (8 BF16) -> ~256-byte coalesced remote bursts per row.
// Numerically identical (same values, wider stores) -> gate-3 must still PASS.
#ifndef MEGA_MOE_COALESCE_EPI
#define MEGA_MOE_COALESCE_EPI 0
#endif

// EPI_LDS_BULK (default off): the CUDA-SM100-inspired epilogue.  CUDA stages the
// fp8 epilogue output in smem (STSM) then bulk-stores smem->global via TMA, so it
// NEVER does the scattered per-element global stores that stall the AMD epilogue.
// AMD's COALESCE_EPI already LDS-stages + emits contiguous uint2 stores, but it
// guarded the per-warp LDS exchange with BLOCK-WIDE __syncthreads (8 warps), which
// serialized the epilogue and ate the coalescing win (-> neutral).  The xpose
// buffer is PER-WARP, so only intra-wave LDS write-completion ordering is needed:
// replace the two __syncthreads with wait_lgkmcnt<0> (a wave-local s_waitcnt, no
// block barrier).  Implies COALESCE_EPI.  gate-3 must still PASS (same values).
#ifndef MEGA_MOE_EPI_LDS_BULK
#define MEGA_MOE_EPI_LDS_BULK 0
#endif

// MFMA_BPF (PROMOTED 2026-06-04, default ON, +1.2%): B-vector prefetch in the
// do_burst MFMA loop -- hoist all kSubTilesN B-subtile LDS reads into registers
// before issuing MFMAs, decoupling the ds_read from its consuming MFMA
// (kSubTilesM==1 leaves no A-reuse to hide the read, so the compiler stalled each
// MFMA on its own immediately-preceding ds_read).  Pure reorder -> gate-3 safe
// (cos_sim 0.999951).  Measured 738->744 TF / 8804->8700us.  Set
// MEGA_MOE_MFMA_BPF=0 to revert.
#ifndef MEGA_MOE_MFMA_BPF
#define MEGA_MOE_MFMA_BPF 1
#endif

// MFMA_BPF2 (default off, EXPERIMENTAL): cross-operand B prefetch for L1 -- hoist
// BOTH gate-B and up-B subtiles before issuing MFMAs so up's LDS reads overlap the
// gate MFMA compute.  2x b_vec VGPR shadows (spill risk on the 256-VGPR kernel).
// Assumes kSubTilesM==1.  Pure reorder -> gate-3 safe.
#ifndef MEGA_MOE_MFMA_BPF2
#define MEGA_MOE_MFMA_BPF2 0
#endif

// MFMA_PRIO (default off, EXPERIMENTAL): raise the compute wave's hardware issue
// priority (s_setprio 1) around the MFMA burst, then drop it (s_setprio 0), so the
// MFMA-issuing wave is favored over the 2nd wave's load issue on the shared SIMD.
// VGPR-free.  gate-3 safe (scheduling hint only).
#ifndef MEGA_MOE_MFMA_PRIO
#define MEGA_MOE_MFMA_PRIO 0
#endif

// MFMA_SCHED (PROMOTED 2026-06-04, default ON, +2% over BPF): the 4WARP_BAK
// PP_SCHED pattern ported to run_operand -- a 2-deep B-prefetch software pipeline
// (only 2 b_vec live, VGPR-light vs BPF's full-kSubTilesN hoist) with
// __builtin_amdgcn_sched_barrier(0) locking the read<->MFMA interleave and
// s_setprio(1/0) bracketing each MFMA so the matrix unit is favored.  Pure schedule
// control -> gate-3 safe (cos_sim 0.999957).  Measured 745->760 TF / 8724->8530us.
// Overrides BPF inside run_operand (the #elif chain: SCHED > BPF > plain), so
// MEGA_MOE_MFMA_SCHED=0 falls back to BPF.  Set =0 to revert.
#ifndef MEGA_MOE_MFMA_SCHED
#define MEGA_MOE_MFMA_SCHED 1
#endif

// MFMA_SCHED2 (default off, EXPERIMENTAL): unify L1's gate+up into ONE locked
// 2*kSubTilesN-long sched pipeline (vs SCHED's two separate per-operand pipelines)
// so the up-operand's first B prefetches during gate's last MFMA -- removes the
// operand-boundary cold start.  Same sched_barrier(0)+setprio lock, VGPR-light.
// Assumes kSubTilesM==1.  Pure schedule control -> gate-3 safe.  L2 still uses SCHED.
#ifndef MEGA_MOE_MFMA_SCHED2
#define MEGA_MOE_MFMA_SCHED2 0
#endif

// MFMA_NOPRIO (default off, ABLATION): strip the s_setprio(1/0) from the SCHED path,
// keeping the sched_barrier(0) interleave-lock.  Isolates whether the priority hint
// or the schedule-lock is the active ingredient of the SCHED win.
#ifndef MEGA_MOE_MFMA_NOPRIO
#define MEGA_MOE_MFMA_NOPRIO 0
#endif

// EPI_BATCH_STORE (default off, EXPERIMENTAL): store-side analog of MFMA_BPF -- in
// the default L1 epilogue, compute ALL quant bytes into a small register array
// first, THEN issue all scattered fp8 stores back-to-back, decoupling the
// SwiGLU+quant VALU from the store issue so the stores pipeline maximally (the
// epilogue cost is a per-store issue stall).  ~8 VGPR for qbuf (live shape).
// Uses the FAST_QUANT math -> gate-3 safe.  Default store path only.
#ifndef MEGA_MOE_EPI_BATCH_STORE
#define MEGA_MOE_EPI_BATCH_STORE 0
#endif

// EPI_CONTIG_PROBE (default off, correctness-breaking): write the L1 fp8 to a
// contiguous-by-token [hidden_block][token][128] layout to test whether the current
// [token][hidden] layout's 3072B token write-stride (DRAM row thrash) is the
// epilogue store wall.  L2 reads the old layout -> garbage; --num-correctness-tests 0.
#ifndef MEGA_MOE_EPI_CONTIG_PROBE
#define MEGA_MOE_EPI_CONTIG_PROBE 0
#endif

// KLOOP_SCHED (default off, EXPERIMENTAL): sched_barrier(0) after the k-loop's
// next-stage prefetch, before the MFMA bursts -- locks the buffer_load_lds issue
// ahead of the MFMAs so the async loads overlap the MFMA maximally.  Load-side
// analog of MFMA_SCHED.  VGPR-free, gate-3 safe (schedule only).
#ifndef MEGA_MOE_KLOOP_SCHED
#define MEGA_MOE_KLOOP_SCHED 0
#endif

// MFMA_SCHED3 (default off, EXPERIMENTAL): SCHED but 2-AHEAD B-prefetch (3 b_vec
// live) instead of 1-ahead, to test if deeper prefetch hides more ds-read latency.
// Overrides SCHED in run_operand when set.  VGPR-light, gate-3 safe.
#ifndef MEGA_MOE_MFMA_SCHED3
#define MEGA_MOE_MFMA_SCHED3 0
#endif

// EPI_W16 (default off, correctness-breaking probe): 16-byte (uint4) store per
// element in the L1 epilogue -- the width the dispatch pull uses to write the pool
// FAST.  Tests if 16B is the fast-store threshold that 1/4/8B missed.  Garbage y;
// --num-correctness-tests 0.
#ifndef MEGA_MOE_EPI_W16
#define MEGA_MOE_EPI_W16 0
#endif
#if MEGA_MOE_EPI_LDS_BULK
#undef MEGA_MOE_COALESCE_EPI
#define MEGA_MOE_COALESCE_EPI 1
#endif

// ABLATION (default off): skip ONLY the L2 combine output store (keep ALL of the
// L1 epilogue: SwiGLU compute + fp8 pool write + l2_arrival signal).  Full - this
// = the L2-output-store contribution; combined with SKIP_EPI (both) it isolates
// the L1 (compute + pool write) vs L2 split.  Garbage y -> --num-correctness-tests 0.
#ifndef MEGA_MOE_SKIP_EPI_L2
#define MEGA_MOE_SKIP_EPI_L2 0
#endif

// ABLATION (default off): in the L1 epilogue, replace the SwiGLU transcendental
// (__expf + reciprocal-divide) with a trivial gate*up, KEEPING the clamps, the
// fp8 quant-convert, and the pool store.  Full - this = the __expf/div cost,
// isolating it from the fp8-convert cost within the ~2.58ms L1 epilogue compute.
// Garbage intermediate -> --num-correctness-tests 0.
#ifndef MEGA_MOE_SKIP_SILU
#define MEGA_MOE_SKIP_SILU 0
#endif

// Cache-policy (cpol/aux) for the GEMM activation (A) and weight (B)
// global->LDS buffer_load.  On gfx950: 0 = normal temporal (cacheable in
// L2/MALL, the byte-identical default), 2 = NT non-temporal/streaming (low
// cache retention).  Experiment: mark high-volume/low-reuse A as NT (2) so it
// does not evict the high-reuse weight tiles, keeping B (default 0) resident in
// L2/MALL across the num_m_blocks re-reads.  Read-only loads -> policy affects
// performance only, never correctness.
#ifndef MEGA_MOE_A_CPOL
#define MEGA_MOE_A_CPOL 0
#endif
#ifndef MEGA_MOE_B_CPOL
#define MEGA_MOE_B_CPOL 0
#endif

// ABLATION (default off): isolate the pure GEMM global-load path to measure its
// achievable HBM bandwidth UNOBSTRUCTED by dispatch+sync.  Skips the per-block
// producer arrival-spins (the cross-SM sync waits) and the epilogue global
// STORES (L2-pool write + combine write + l2 arrival signal).  Pair with
// MEGA_MOE_NO_MFMA=1 + MEGA_MOE_SKIP_DISPATCH_PULL=1 + MEGA_MOE_SKIP_COMBINE=1
// so only the A/B/SF buffer_loads stream from HBM: FETCH_SIZE/time = loader's
// peak HBM BW.  Reads garbage pool data -> run with --num-correctness-tests 0.
#ifndef MEGA_MOE_LOADS_ONLY
#define MEGA_MOE_LOADS_ONLY 0
#endif

// #1 shifted-LDG / partial-drain: the k-loop currently fully drains the loader
// with wait_vmcnt<0> every 2 k_blocks, exposing the full load latency at each
// iteration boundary (the 22%-of-peak loader wall).  The SOTA turbo GEMM ends
// each K-iter with a PARTIAL wait_vmcnt<12>, keeping that many loads in flight
// across the barrier (proves partial vmcnt is correct on gfx950).  This knob
// sets the keep-count; 0 = the conservative full drain (default, unchanged).
// NOTE: a non-zero value leaves loads in flight while the burst reads their LDS
// -> only correctness-safe once the loop is reordered to prefetch-before-drain;
// used first as a LOADS_ONLY perf probe (garbage output) to size the MLP win.
#ifndef MEGA_MOE_VMCNT_KEEP
#define MEGA_MOE_VMCNT_KEEP 0
#endif

// De-risk probe for the loader-BW ceiling: route the GEMM A/B global loads
// through VGPR (global_load -> VGPR -> ds_write, the full vmem pipe) instead of
// buffer_load_lds (the narrower LDS-DMA path).  If the raw loader BW (measured
// with MEGA_MOE_LOADS_ONLY) rises above the ~22%-of-peak buffer_load_lds
// ceiling, the primitive is the limiter and the full AGPR+VGPR-staged port is
// worth it.  Default 0 = buffer_load_lds (unchanged).  Garbage numerics (the
// crude single-buffered store overwrites the same LDS slot) -> probe only,
// pair with MEGA_MOE_LOADS_ONLY + --num-correctness-tests 0.
#ifndef MEGA_MOE_VGPR_LOAD
#define MEGA_MOE_VGPR_LOAD 0
#endif

// Probe: drop ALL explicit waitcnt in the compute k-loop (wait_vmcnt +
// wait_lgkmcnt) AND the per-block __syncthreads (correctness-irrelevant under
// MEGA_MOE_LOADS_ONLY).  Tells whether the ~22% raw loader BW is throttled by
// the per-block sync/drain (BW rises when removed) or is the access pattern's
// hard ceiling (BW unchanged).  Default 0.
#ifndef MEGA_MOE_NO_KLOOP_SYNC
#define MEGA_MOE_NO_KLOOP_SYNC 0
#endif

// Probe: skip the MFMA bursts (do_burst) entirely.  The buffer_load_lds still
// side-effect into LDS (not DCE'd), so the loop becomes PURE load issue with no
// LDS reads -> the compiler emits almost no s_waitcnt.  Combined with
// MEGA_MOE_NO_KLOOP_SYNC + MEGA_MOE_LOADS_ONLY this measures the absolute
// pure-load-issue throughput (hardware vmcnt auto-backpressure only), isolating
// the memory system from ALL software waits.  Default 0.
#ifndef MEGA_MOE_NO_BURST
#define MEGA_MOE_NO_BURST 0
#endif

// Occupancy lever (root-caused 2026-06-04): the loader only reaches ~22% HBM
// BW because the kernel runs 1 block/CU (LDS-bound at 140KB) -> too few
// outstanding loads.  Microbench: 2 blocks/CU -> 65% (3x).  MEGA_MOE_2STAGE
// switches the k-loop from the 4-stage / 2-burst-per-barrier overlap to a
// simple 2-stage double buffer (process 1 k_block/iter, stages k&1).  Combined
// with kNumStages=2 this halves loader LDS so 2 blocks fit per CU; the 2nd
// resident block supplies the MLP that the deeper 4-stage prefetch gave at
// occupancy 1.  Default 0 = the 4-stage scheme (unchanged).
#ifndef MEGA_MOE_2STAGE
#define MEGA_MOE_2STAGE 0
#endif

// 2 blocks/CU occupancy build (implies 2STAGE + kNumStages=2 + kNumSMs=512 set
// in jit_launch.cu).  Halves loader LDS (kSmemBytes 140->76KB so 2 fit per CU)
// and raises __launch_bounds__ min-blocks to 2.  Cooperative-launch caveat: the
// kernel uses grid_sync, so ALL grid blocks must be co-resident -> 2 blocks/CU
// MUST actually fit (LDS 76KB*2<=159KB ok; launch_bounds(,2) forces the compiler
// to fit VGPR for 2, spilling if needed) or the grid_sync deadlocks.
#ifndef MEGA_MOE_2BLK
#define MEGA_MOE_2BLK 0
#endif

// Optimization (default 0): replace the PER-TOKEN release on l1_arrival_count in
// the dispatch pull (each forces an s_waitcnt vmcnt(0) drain of the XGMI copy
// stores -> serializes the pulls, ~5ms / 37% of the kernel) with a RELAXED
// increment + ONE phase-boundary release-fence + grid_sync before compute.
// Trades the fine-grained dispatch<->compute overlap for removing ~49k/rank
// per-token drains.  Numerically correct (the grid_sync globally orders all
// dispatch writes before any compute read) -> gate-3 must still PASS.
// PROMOTED (default 1): +13.5% (453->519 TF / 14.35->12.5ms), gate-3 23/23 PASS
// cos_sim 0.999991.  Set MEGA_MOE_DISP_BARRIER=0 to revert to per-token release.
#ifndef MEGA_MOE_DISP_BARRIER
#define MEGA_MOE_DISP_BARRIER 1
#endif

// Optimization (default 0, under test): batch the L2-arrival release.  The L1
// epilogue signals red_add_rel_gpu(l2_arrival) per warp PER gate-N-block (24x per
// pool block), each draining that warp's fp8 pool stores.  But a pool block's 24
// gate-blocks are processed CONSECUTIVELY by a warp and the L2 consumer needs the
// FULL intermediate (all 192 signals) anyway -- so do RELAXED increments for the
// first 23 gate-blocks and ONE release on the last (which drains all 24 at once).
// 24x fewer epilogue drains.  PROMOTED (default 1, user-accepted): +21% on top of
// DISP_BARRIER (519->~630 TF / 12.5->10.3 ms).  NOTE: NOT byte-identical -- cos_sim
// 0.99997 / rel_rmse ~0.007 / max|diff| ~5000 (gate-3 passes with margin; accepted
// trade for the perf).  The per-gate-block release granularity can't be fully
// batched cleanly on gfx950 (see notes); this relaxed variant is the accepted form.
// Set MEGA_MOE_EPI_BATCH_REL=0 to revert to the byte-identical per-token release.
#ifndef MEGA_MOE_EPI_BATCH_REL
#define MEGA_MOE_EPI_BATCH_REL 1
#endif

// Ablation: no-op issue_sf (skip the SFA/SFB __ldg->ds_write, the lgkmcnt path
// interleaved with the A/B buffer_load_lds vmcnt path).  Isolates whether the
// SF load path serializes the loader's MLP.  Garbage -> LOADS_ONLY probe only.
#ifndef MEGA_MOE_SKIP_SF
#define MEGA_MOE_SKIP_SF 0
#endif

// Optimization: load SF (SFA/SFB) via buffer_load_lds<4> (global->LDS, vmcnt
// async path) instead of __ldg->ds_write (vmcnt load + lgkmcnt store, which
// serializes against the A/B buffer_load_lds).  Same per-lane gather addressing
// and identical LDS destination layout, so the MFMA SF byte-read is unchanged
// (gate-3 must still PASS — the SF byte addressing is the landmine).  Localized
// as ~27% of the loader time.  PROMOTED 2026-06-04: default 1 (gate-3 PASS
// cos_sim=0.999991 twice, +3% = 304->313 TF / 21.32->20.73ms on EP8 V4Pro).
#ifndef MEGA_MOE_SF_BLD
#define MEGA_MOE_SF_BLD 1
#endif

// A/B loader: pass the per-k_block K-offset as a SCALAR buffer soffset
// (readfirstlane) instead of baking it into the per-lane VGPR ldg_offset, so the
// inner load loop issues back-to-back with minimal per-load address ALU (turbo's
// precompute_base_soff pattern).  Address is identical (voffset+soffset) ->
// gate-3 safe.  Targets the A/B issue-pattern loss (~45->77% loader).  Default 0.
#ifndef MEGA_MOE_SCALAR_SOFF
#define MEGA_MOE_SCALAR_SOFF 0
#endif

// PROBE: repeat each tile's k-loop N times (re-issuing the same loads) to
// lengthen the per-tile continuous load stream.  If LOADS_ONLY BW rises with N,
// the A/B gap is per-tile stream length / boundary ramp-up (=> cross-tile
// pipelining helps); if flat, it's intrinsic to the per-k_block load pattern.
// Garbage numerics -> LOADS_ONLY probe only.  Default 1 (no change).
#ifndef MEGA_MOE_KLOOP_REPEAT
#define MEGA_MOE_KLOOP_REPEAT 1
#endif

// Cross-CU / cross-rank rendezvous.  Block-local sync is a plain
// ``__syncthreads()`` (every barrier spans the whole block); ``grid_sync`` adds
// the cross-CU handshake and ``nvlink_barrier`` the cross-rank signal.
namespace comm {

// Grid-wide sync: atomic count + relaxed spin + a single acquire fence.
template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0>
__device__ __forceinline__ void grid_sync(const layout::Workspace &workspace,
                                          uint32_t leader_thread_idx, uint32_t sm_idx,
                                          uint32_t thread_idx) {
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    __syncthreads();
    if (thread_idx == leader_thread_idx) {
        auto          *count_ptr = workspace.get_grid_sync_count_ptr(kGridSyncIndex);
        const uint32_t old_value =
            prims::atomic_add_rel(count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1u)) : 1u);
        uint32_t new_value;
        // Cheap-fence-ONCE spin: relaxed ld_volatile loop (atomic loads bypass
        // L1 so the producer's released count write is observed), then a SINGLE
        // acquire fence (one cache invalidate, not one per iteration).
        do {
            new_value = prims::ld_volatile(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0u);
        prims::acquire_fence_agent();
    }
    __syncthreads();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kGridSyncIndex, uint32_t kTag>
__device__ __forceinline__ void
nvlink_barrier(const layout::Workspace &workspace, const layout::SymBuffer<kNumRanks> &sym_buffer,
               uint32_t leader_thread_idx, uint32_t sm_idx, uint32_t thread_idx,
               bool sync_prologue = true, bool sync_epilogue = true) {
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, leader_thread_idx, sm_idx, thread_idx);

    if constexpr (kNumRanks > 1) {
        if (sm_idx == 0) {
            auto *counter_ptr = workspace.get_nvl_barrier_counter_ptr();

            const auto status       = prims::ld_volatile(counter_ptr) & 3u;
            const auto signal_phase = status & 1u;
            const auto signal_sign  = status >> 1u;
            auto      *signal_ptr   = workspace.get_nvl_barrier_signal_ptr(signal_phase);

            if (thread_idx >= leader_thread_idx && thread_idx < leader_thread_idx + kNumRanks)
                prims::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx - leader_thread_idx),
                                       signal_sign ? -1 : 1);
            // sm_idx==0 is uniform across the block, so this __syncthreads() is
            // reached by all threads of the block (all-or-none).
            __syncthreads();

            if (thread_idx == leader_thread_idx) {
                prims::red_add(reinterpret_cast<int *>(counter_ptr), 1);
                const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
                while (prims::ld_acquire_sys(signal_ptr) != target)
                    __builtin_amdgcn_s_sleep(1);
            }
        }
    } else {
        (void) sym_buffer;
    }

    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex>(workspace, leader_thread_idx, sm_idx, thread_idx);
}

} // namespace comm

template <typename Adtype, typename Bdtype>
__device__ __forceinline__ dtype::float32x4 mfma_scaled(dtype::int32x8 a, dtype::int32x8 b,
                                                        dtype::float32x4 c, uint32_t scale_a,
                                                        uint32_t scale_b) {
#if MEGA_MOE_NO_MFMA
    // ABLATION: no MFMA -- XOR-fold every loaded operand (both A uint4 halves,
    // the B uint4, and SFA/SFB) into acc so the loads are not DCE'd.  Output is
    // garbage -> always run with `--num-correctness-tests 0`.
    const int fold = a[0] ^ a[4] ^ b[0] ^ b[3] ^ static_cast<int>(scale_a) ^
                     static_cast<int>(scale_b);
    c[0] += static_cast<float>(fold);
    return c;
#else
    return device::mfma_scale_f32_16x16x128_f8f6f4<Adtype, Bdtype>::run(a, b, c, scale_a, scale_b);
#endif
}

#if MEGA_MOE_AGPR_ACC
// ── Pinned-AGPR (turbo-style) scaffolding ──────────────────────────────────
// Compile-time unroll so run_pinned_acc_agpr / read_agpr receive constexpr
// register indices ("n" immediates).  kSubTilesM/N are constexpr but the
// #pragma-unroll loop variables are not usable as template args, so the pinned
// burst and the epilogue AGPR read-back drive the (sub_m,sub_n) index through
// this instead.
template <int I, int N, typename F> __device__ __forceinline__ void mega_static_for(F &&f) {
    if constexpr (I < N) {
        f.template operator()<I>();
        mega_static_for<I + 1, N>(static_cast<F &&>(f));
    }
}
// Pinned VGPR operand window (top of the 256-VGPR file) for the Linear2 burst:
// A frag = 8 VGPR (FP8 e4m3, cbsz=0), B frag = 4 VGPR (FP4 e2m1, blgp=4), one
// scale VGPR each.  Reserved via reserve_vgpr_range so the compiler will not
// reuse them between the set_vgpr stage and the pinned MFMA.
inline constexpr int kPinA0  = 224; // v[224:231] A frag 0 (8)
inline constexpr int kPinA1  = 232; // v[232:239] A frag 1 (8)
inline constexpr int kPinB0  = 240; // v[240:243] B frag buf0 (4)
inline constexpr int kPinB1  = 244; // v[244:247] B frag buf1 (4) -- double buffer
inline constexpr int kPinSA0 = 248; // v[248] A0 scale
inline constexpr int kPinSA1 = 249; // v[249] A1 scale
inline constexpr int kPinSB0 = 250; // v[250] B scale buf0
inline constexpr int kPinSB1 = 251; // v[251] B scale buf1
// AGPR accumulator bases (per loader wave, 16 subtiles x 4 = 64 AGPR each):
//   up   acc -> a[0:63]   (Linear2 uses only this range)
//   gate acc -> a[64:127] (Linear1 only)
inline constexpr int kAccUp   = 0;
inline constexpr int kAccGate = 64;
// Read ONE subtile's float32x4 accumulator from a compile-time AGPR BASE + a
// RUNTIME subtile index (0..15), via a switch so only ONE v_accvgpr_read runs
// (~4 VGPR live).  Avoids hoisting all 16 subtiles (128 VGPR) at the epilogue
// top, which kept the kernel pinned at 256 VGPR with no room for the pin window.
// Assumes kSubTilesPerWave == 16 (the 4-warp AGPR config).
template <int BASE> __device__ __forceinline__ dtype::float32x4 read_acc_sub16(uint32_t sub) {
    switch (sub) {
    case 0:  return device::read_agpr<dtype::float32x4, BASE + 0>();
    case 1:  return device::read_agpr<dtype::float32x4, BASE + 4>();
    case 2:  return device::read_agpr<dtype::float32x4, BASE + 8>();
    case 3:  return device::read_agpr<dtype::float32x4, BASE + 12>();
    case 4:  return device::read_agpr<dtype::float32x4, BASE + 16>();
    case 5:  return device::read_agpr<dtype::float32x4, BASE + 20>();
    case 6:  return device::read_agpr<dtype::float32x4, BASE + 24>();
    case 7:  return device::read_agpr<dtype::float32x4, BASE + 28>();
    case 8:  return device::read_agpr<dtype::float32x4, BASE + 32>();
    case 9:  return device::read_agpr<dtype::float32x4, BASE + 36>();
    case 10: return device::read_agpr<dtype::float32x4, BASE + 40>();
    case 11: return device::read_agpr<dtype::float32x4, BASE + 44>();
    case 12: return device::read_agpr<dtype::float32x4, BASE + 48>();
    case 13: return device::read_agpr<dtype::float32x4, BASE + 52>();
    case 14: return device::read_agpr<dtype::float32x4, BASE + 56>();
    default: return device::read_agpr<dtype::float32x4, BASE + 60>();
    }
}
#endif

template <uint32_t kNumMaxTokensPerRank, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumExperts, uint32_t kNumTopk, uint32_t kNumExpertsPerWave, uint32_t BLOCK_M,
          uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M,
          uint32_t SF_BLOCK_N, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          uint32_t kNumStages, uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath, uint32_t L1_SHAPE_N = kIntermediateHidden * 2u,
          uint32_t L1_SHAPE_K = kHidden, uint32_t L2_SHAPE_N = kHidden,
          uint32_t L2_SHAPE_K  = kIntermediateHidden,
          uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
          uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks>
__global__ __launch_bounds__(kNumThreads, MEGA_MOE_2BLK ? 2 : 1) void gfx950_fp8_fp4_mega_moe_kernel(
    void *y, int *cumulative_local_expert_recv_stats, const uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer, void *l1_weights,
    void *l1_weights_sf, void *l2_weights, void *l2_weights_sf) {
#if defined(__gfx950__)
    using namespace prims;

    const uint32_t sm_idx     = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx   = get_warp_idx();
    const uint32_t lane_idx   = get_lane_idx();

    constexpr uint32_t kNumDispatchWarps       = kNumDispatchThreads / kWarpSize;
    constexpr uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / kWarpSize;
    // Epilogue role folded into the MFMA warps (combine runs there post-compute),
    // so kNumEpilogueThreads is 0 in the live config; no separate warp count.

    // Token and buffer layouts
    // NOTES: activations are FP8 (1 B/elem); the L2 activation is the SwiGLU
    // intermediate, also FP8
    constexpr auto fp8_token_layout  = layout::Data(kHidden);
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(__hip_bfloat16));
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);
    constexpr auto fp8_sf_layout                 = layout::Data(kHidden / 32u);
    constexpr auto fp8_intermediate_sf_layout    = layout::Data(kIntermediateHidden / 32u);
    constexpr auto input_topk_idx_layout         = layout::Data(kNumTopk * sizeof(int64_t), false);
    constexpr auto input_topk_weights_layout     = layout::Data(kNumTopk * sizeof(float), false);
    constexpr auto l1_topk_weights_layout        = layout::Data(sizeof(float), false);

    constexpr uint64_t kInterBufAlign = 256u;
    auto               align256       = [](void *p) -> void                     *{
        const auto v = reinterpret_cast<uintptr_t>(p);
        return reinterpret_cast<void *>((v + kInterBufAlign - 1u) &
                                        ~uintptr_t(kInterBufAlign - 1u));
    };

    const auto workspace = layout::Workspace(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
                                             kNumMaxTokensPerRank, kNumTopk);

    const auto input_token_buffer    = layout::Buffer(fp8_token_layout, 1, kNumMaxTokensPerRank,
                                                      align256(workspace.get_end_ptr()));
    const auto input_sf_buffer       = layout::Buffer(fp8_sf_layout, 1, kNumMaxTokensPerRank,
                                                      align256(input_token_buffer.get_end_ptr()));
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, kNumMaxTokensPerRank, align256(input_sf_buffer.get_end_ptr()));
    const auto input_topk_weights_buffer =
        layout::Buffer(input_topk_weights_layout, 1, kNumMaxTokensPerRank,
                       align256(input_topk_idx_buffer.get_end_ptr()));

    const auto l1_token_buffer = layout::Buffer(fp8_token_layout, 1, kNumMaxPoolTokens,
                                                align256(input_topk_weights_buffer.get_end_ptr()));
    const auto l1_sf_buffer    = layout::Buffer(fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
                                                align256(l1_token_buffer.get_end_ptr()));
    const auto l1_topk_weights_buffer = layout::Buffer(l1_topk_weights_layout, 1, kNumMaxPoolTokens,
                                                       align256(l1_sf_buffer.get_end_ptr()));

    const auto l2_token_buffer = layout::Buffer(fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
                                                align256(l1_topk_weights_buffer.get_end_ptr()));
    const auto l2_sf_buffer = layout::Buffer(fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
                                             align256(l2_token_buffer.get_end_ptr()));
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank, align256(l2_sf_buffer.get_end_ptr()));

    constexpr uint32_t kGranK                 = 32u;
    constexpr uint32_t kNumUTCCPAlignedElems  = 128u;
    auto               transform_sf_token_idx = [](const uint32_t &token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M + (idx & ~127u) + (idx & 31u) * 4u +
               ((idx >> 5) & 3u);
    };

    // MMA tile shape constraints.  Activations are FP8 (e4m3, 1 B/elem),
    // weights are FP4 (e2m1, packed 2/byte); the K-major 16x16x128 MFMA below
    // depends on these.
    constexpr uint32_t LAYOUT_AD_M = 128u;
    static_assert(BLOCK_M % 16u == 0u, "Invalid block M");
    static_assert(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    static_assert(BLOCK_K == 128u, "Invalid block K");

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_expert_count = reinterpret_cast<uint32_t *>(smem_buffer);

    constexpr uint32_t kMfmaM       = 16u;
    constexpr uint32_t kMfmaN       = 16u;
    constexpr uint32_t kMfmaK       = 128u;
    constexpr uint32_t kInnerKIters = BLOCK_K / kMfmaK;
    static_assert(BLOCK_K % kMfmaK == 0u, "BLOCK_K must be a multiple of MFMA K");

    static_assert(BLOCK_M % kNumMMANonEpilogueWarps == 0u,
                  "BLOCK_M must be evenly partitionable across loader waves");
    constexpr uint32_t kRowsPerLoaderWave = BLOCK_M / kNumMMANonEpilogueWarps;
    constexpr uint32_t kColsPerLoaderWave = BLOCK_N;
    static_assert(kRowsPerLoaderWave % kMfmaM == 0u,
                  "loader wave row partition must be a multiple of MFMA M");
    static_assert(kColsPerLoaderWave % kMfmaN == 0u,
                  "loader wave col partition must be a multiple of MFMA N");
    constexpr uint32_t kSubTilesM       = kRowsPerLoaderWave / kMfmaM;
    constexpr uint32_t kSubTilesN       = kColsPerLoaderWave / kMfmaN;
    constexpr uint32_t kSubTilesPerWave = kSubTilesM * kSubTilesN;

    constexpr uint32_t kBPackBytesPerRow = BLOCK_K / 2u;
    constexpr uint32_t kATileBytes       = kRowsPerLoaderWave * BLOCK_K;
    constexpr uint32_t kBTileBytes       = kColsPerLoaderWave * kBPackBytesPerRow;
    constexpr uint32_t kLaneLoadBytes    = 16u;
    constexpr uint32_t kBytesPerLoadCall = kWarpSize * kLaneLoadBytes;
    static_assert(kATileBytes % kBytesPerLoadCall == 0u,
                  "A tile must be a multiple of one cooperative load call");
    static_assert(kBTileBytes % kBytesPerLoadCall == 0u,
                  "B tile must be a multiple of one cooperative load call");
    constexpr uint32_t kATileLoadsPerWave = kATileBytes / kBytesPerLoadCall;
    constexpr uint32_t kBTileLoadsPerWave = kBTileBytes / kBytesPerLoadCall;

    // Scale factors are staged into the loader's LDS shadow: SFA per-warp (one
    // UE8M0 dword per loader-wave row) rides each stage's A SF region; SFB shared
    // (one dword per BLOCK_N column) gets a pool after the B pool.
    // NOTES: L1 fuses gate+up into one A x two-B MFMA stream (CK Is2B), so the B
    // and SFB pools hold kMaxBOperands operands; Linear2 uses only operand 0
    constexpr uint32_t kMaxBOperands    = 2u;
    constexpr uint32_t kSFAStageBytes   = kRowsPerLoaderWave * sizeof(uint32_t);
    constexpr uint32_t kSFBStageBytes   = kMaxBOperands * BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t kLoaderASFBytes  = kSFAStageBytes;
    constexpr uint32_t kLoaderSinkBytes = kSubTilesPerWave * sizeof(dtype::float32x4);

    constexpr uint32_t kStagedABytesPerStage = kATileBytes + kLoaderASFBytes;
#if MEGA_MOE_BDIRECT_PROBE
    // Route-A probe: B lives in VGPR (not LDS), so it costs 0 LDS budget.
    constexpr uint32_t kStagedBBytesPerStage = 0u;
#else
    constexpr uint32_t kStagedBBytesPerStage = kMaxBOperands * kBTileBytes;
#endif
    constexpr uint32_t kLoaderBaseBytes =
        ((sizeof(uint32_t) * kNumExperts + 16u + 1023u) / 1024u) * 1024u;

    static constexpr uint32_t kSmemBytesBudget = 140u * 1024u;
    static_assert(kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderSinkBytes < kSmemBytesBudget,
                  "loader sink + base already exceed kSmemBytes budget");
    // Per-stage LDS cost: per-warp A (+SFA) tiles, shared B tile, and the
    // shared SFB pool slot.
    constexpr uint32_t kPerStageLdsBytes =
        kNumMMANonEpilogueWarps * kStagedABytesPerStage + kStagedBBytesPerStage + kSFBStageBytes;
    constexpr uint32_t kMaxLoaderStagesByLds =
        (kSmemBytesBudget - kLoaderBaseBytes - kNumMMANonEpilogueWarps * kLoaderSinkBytes) /
        kPerStageLdsBytes;
    static_assert(kMaxLoaderStagesByLds >= 1u,
                  "loader staged carve-out too large for one stage even - shrink BLOCK_K?");
    constexpr uint32_t kLoaderStages =
        (kNumStages < kMaxLoaderStagesByLds) ? kNumStages : kMaxLoaderStagesByLds;

    constexpr uint32_t kLoaderAWaveBytes = kLoaderStages * kStagedABytesPerStage + kLoaderSinkBytes;
    constexpr uint32_t kLoaderBPoolBaseBytes =
        kLoaderBaseBytes + kNumMMANonEpilogueWarps * kLoaderAWaveBytes;
    constexpr uint32_t kLoaderBPoolBytes = kLoaderStages * kStagedBBytesPerStage;
    static_assert(kLoaderBPoolBaseBytes + kLoaderBPoolBytes <= kSmemBytesBudget,
                  "loader tile carve-out (per-warp A + shared B) exceeds kSmemBytes budget");

    // Shared SFB pool: kLoaderStages slots of kSFBStageBytes, appended after the
    // shared-B pool.  SFA lives inside each per-warp A stage (kATileBytes..).
    constexpr uint32_t kLoaderSFBPoolBaseBytes = kLoaderBPoolBaseBytes + kLoaderBPoolBytes;
    constexpr uint32_t kLoaderSFBPoolBytes     = kLoaderStages * kSFBStageBytes;
    static_assert(kLoaderSFBPoolBaseBytes + kLoaderSFBPoolBytes <= kSmemBytesBudget,
                  "shared SFB pool exceeds kSmemBytes budget");

    constexpr uint32_t kLoadsPerStage = kATileLoadsPerWave + kMaxBOperands * kBTileLoadsPerWave;

    static_assert(kLoadsPerStage < 63u,
                  "kLoadsPerStage exceeds vmcnt range - reduce per-stage loads");

    if (num_tokens == 0u) {
        (void) l1_weights;
        (void) l1_weights_sf;
        (void) l2_weights;
        (void) l2_weights_sf;
        (void) sym_buffer;
        (void) cumulative_local_expert_recv_stats;
        (void) y;
        return;
    }

    // Grid-sync / NVLink-barrier leader threads (combine is folded onto the MMA
    // warps, so its barriers are led by the first MMA thread)
    constexpr uint32_t kDispLeader = 0u;
    constexpr uint32_t kEpiLeader  = kNumDispatchWarps * kWarpSize;

    if (warp_idx == 0u && elect_one()) {
#pragma unroll
        for (uint32_t i = 0; i < kNumExperts; ++i)
            smem_expert_count[i] = 0u;
    }
    __syncthreads();

    auto scheduler = sched::MegaMoEScheduler<BLOCK_M, BLOCK_N, BLOCK_K, L1_SHAPE_N, L1_SHAPE_K,
                                             L2_SHAPE_N, L2_SHAPE_K, kNumExpertsPerRank,
                                             kNumExpertsPerWave, kNumSMs, kNumRanks>(workspace);

    // Warp roles: all warps run dispatch (Phase 1), then the same warps run the
    // MMA loader + compute + epilogue + combine (Phase 2).  The SM100 dedicated
    // dispatch / TMA-load / epilogue warp roles are folded together here because
    // there is no TMA and combine reuses the MMA warps (kNumDispatchThreads ==
    // kNumEpilogueThreads == 0).

    // Dispatch: count + route the topk indices, then pull each owned token from
    // its remote rank into the local L1 pool.  All warps participate.
    {
        constexpr uint32_t kDispWarps   = kNumMMANonEpilogueWarps;
        constexpr uint32_t kDispThreads = kNumThreads;

        constexpr uint32_t kNumTokensPerWarp = kWarpSize / kNumTopk;
        constexpr uint32_t kNumGlobalWarps   = kNumSMs * kDispWarps;
        static_assert(kNumTokensPerWarp * kNumTopk <= kWarpSize,
                      "kNumTopk does not divide wave size");

        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk;
        const auto         read_topk_idx     = [&](const auto &process) {

#pragma unroll
            for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
                 i += kNumSMs * kDispWarps * kNumTokensPerWarp) {

                int expert_idx = -1;
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(__ldg(
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        read_topk_idx([&](const uint32_t &token_topk_idx, const int &expert_idx) {
            atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        __syncthreads();

        for (uint32_t i = thread_idx; i < kNumExperts; i += kDispThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i]      = static_cast<uint32_t>(
                atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        __syncthreads();

        for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
             i += kNumGlobalWarps * kNumTokensPerWarp) {
            const uint32_t slot = i + (lane_idx / kNumTopk);
            if (slot < num_tokens && lane_idx < kNumTokensPerWarp * kNumTopk) {
                const uint32_t topk_idx_in_token = lane_idx % kNumTopk;
                const int e = static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                                     slot * kNumTopk + topk_idx_in_token));
                if (e >= 0) {
                    const uint32_t dst_rank_idx = e / kNumExpertsPerRank;
                    const uint32_t dst_slot_idx = atomic_add_block(smem_expert_count + e, 1u);
                    auto          *dst_ptr      = workspace.get_src_token_topk_idx_ptr(
                        e % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
                    *sym_buffer.map(dst_ptr, dst_rank_idx) = slot * kNumTopk + topk_idx_in_token;
                }
            }
        }

        comm::grid_sync<kNumSMs, 0>(workspace, kDispLeader, sm_idx, thread_idx);

        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kDispThreads) {
                const uint32_t dst_rank         = i / kNumExpertsPerRank;
                const uint32_t dst_local_expert = i % kNumExpertsPerRank;
                const uint64_t expert_status    = *workspace.get_expert_send_count_ptr(i);

                // Cross-rank recv_count write.  RELAXED, SYSTEM scope: the
                // happens-before for the pull loop's read comes from the
                // nvlink_barrier below, not this store; SYSTEM scope is required
                // so the write reaches the remote agent.
                __hip_atomic_store(
                    reinterpret_cast<unsigned int *>(sym_buffer.map(
                        workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert),
                        dst_rank)),
                    static_cast<unsigned int>(expert_status & 0xffffffffull), __ATOMIC_RELAXED,
                    __HIP_MEMORY_SCOPE_SYSTEM);
                atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert),
                                   dst_rank),
                    expert_status);
            }
        }
        __syncthreads();

        comm::nvlink_barrier<kNumRanks, kNumSMs, 0, 1>(workspace, sym_buffer, kDispLeader, sm_idx,
                                                       thread_idx, false, true);

        scheduler.fetch_expert_recv_count();

        constexpr uint32_t kNumRanksPerLane = (kNumRanks + kWarpSize - 1u) / kWarpSize;

        int      current_expert_idx                  = -1;
        uint32_t expert_start_idx                    = 0u;
        uint32_t expert_end_idx                      = 0u;
        uint32_t expert_pool_block_offset            = 0u;
        uint32_t stored_rank_count[kNumRanksPerLane] = {};

        // One token's pull, extracted so the PINGPONG path can interleave it with
        // GEMM blocks (dispatch<->mma role-switch).  Returns false once this warp's
        // expert stream is exhausted.  Loop-carried expert state is captured above.
        auto pull_one_token = [&](uint32_t token_idx) -> bool {
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                    break;
                expert_pool_block_offset +=
                    (expert_end_idx - expert_start_idx + BLOCK_M - 1u) / BLOCK_M;
                expert_start_idx = expert_end_idx;
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
            if (current_expert_idx >= static_cast<int>(kNumExpertsPerRank))
                return false;

            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    const uint32_t j = i * kWarpSize + lane_idx;
                    if (j < kNumRanks) {
                        const auto raw = __hip_atomic_load(
                            reinterpret_cast<const unsigned long long *>(
                                workspace.get_expert_recv_count_ptr(
                                    j, static_cast<uint32_t>(current_expert_idx))),
                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                        stored_rank_count[i] = static_cast<uint32_t>(raw);
                    } else {
                        stored_rank_count[i] = 0u;
                    }
                }
            }

            const uint32_t token_idx_in_expert = token_idx - expert_start_idx;
            uint32_t       remaining[kNumRanksPerLane];
#pragma unroll
            for (uint32_t i = 0u; i < kNumRanksPerLane; ++i)
                remaining[i] = stored_rank_count[i];

            uint32_t current_rank_in_expert_idx = 0u;
            uint32_t token_idx_in_rank          = 0u;
            uint32_t slot_idx                   = token_idx_in_expert;
            uint32_t offset                     = 0u;
            while (true) {

                uint32_t num_actives_in_lane = 0u;
                uint32_t min_in_lane         = 0xffffffffu;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] > 0u) {
                        ++num_actives_in_lane;
                        if (remaining[i] < min_in_lane)
                            min_in_lane = remaining[i];
                    }
                }
                const uint32_t num_active_ranks = reduce_add(num_actives_in_lane);
                const uint32_t length           = reduce_min(min_in_lane);

                const uint32_t num_round_tokens = length * num_active_ranks;
                if (slot_idx < num_round_tokens) {

                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t       num_seen_ranks    = 0u;
#pragma unroll
                    for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                        const uint64_t mask             = ballot(remaining[i] > 0u);
                        const uint32_t num_active_lanes = popcnt(mask);
                        if (slot_idx_in_round >= num_seen_ranks &&
                            slot_idx_in_round < num_seen_ranks + num_active_lanes) {
                            current_rank_in_expert_idx =
                                i * kWarpSize +
                                nth_set_bit(mask, slot_idx_in_round - num_seen_ranks + 1u);
                        }
                        num_seen_ranks += num_active_lanes;
                    }
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                slot_idx -= num_round_tokens;
                offset += length;
#pragma unroll
                for (uint32_t i = 0u; i < kNumRanksPerLane; ++i) {
                    if (remaining[i] >= length)
                        remaining[i] -= length;
                    else
                        remaining[i] = 0u;
                }
            }

            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                static_cast<uint32_t>(current_expert_idx), current_rank_in_expert_idx,
                token_idx_in_rank);
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            const uint32_t src_topk_idx  = src_token_topk_idx % kNumTopk;

            const uint32_t pool_token_idx =
                expert_pool_block_offset * BLOCK_M + token_idx_in_expert;

            auto *src_token_ptr = sym_buffer.map(
                input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint8_t>(),
                current_rank_in_expert_idx);
            auto *dst_token_ptr =
                l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr<uint8_t>();
            // Pull the dispatched token (remote-global -> local-pool global).
            // NOTES: no TMA on AMD -- a cooperative unrolled warp copy streams the
            // kHidden FP8 bytes directly (mirrors UNROLLED_WARP_COPY in deep_ep)
            static_assert(kHidden % 16u == 0u, "token bytes must be int4-aligned");
#if MEGA_MOE_DISPATCH_PUSH2
            // Approach-2: instead of READING the token data (slow remote read), write
            // the resolved dst pool slot back to the SRC's mapping (combine_buffer
            // reused as uint32, only used post-dispatch).  The src then PUSHES the
            // token data (fast remote write) in phase B.  SF/metadata stay here
            // (cheap local), avoiding approach-1's SF-remote-scatter loss.
            if (elect_one())
                sym_buffer.map(combine_token_buffer.get_base_ptr<uint32_t>(),
                               current_rank_in_expert_idx)[src_token_topk_idx] = pool_token_idx;
            (void) dst_token_ptr;
            (void) src_token_ptr;
#elif !MEGA_MOE_SKIP_DISPATCH_PULL
            // kHidden/16/64 = 7 int4/lane; unroll 8 puts all in flight in ONE
            // batch -> max XGMI latency hiding (dispatch phase has VGPR headroom).
#if MEGA_MOE_PULL_AS_PUSH
            // PROBE: swap direction (local-read + REMOTE-WRITE) to measure XGMI
            // write vs read BW for the dispatch volume.  Garbage output (writes
            // into the remote input region) -> --num-correctness-tests 0 only.
            warp_copy_int4<MEGA_MOE_PULL_UNROLL>(src_token_ptr, dst_token_ptr, kHidden / 16u,
                                                 lane_idx);
#else
            warp_copy_int4<MEGA_MOE_PULL_UNROLL>(dst_token_ptr, src_token_ptr, kHidden / 16u,
                                                 lane_idx);
#endif
#else
            (void) dst_token_ptr;
            (void) src_token_ptr;
#endif

            constexpr uint32_t kNumSFUint32  = kHidden / 128u;
            const auto         remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            auto      *local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            const auto sf_pool_token_idx =
                expert_pool_block_offset * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert);
#if !MEGA_MOE_SKIP_DISPATCH_PULL
#pragma unroll
            for (uint32_t i = 0u; i < (kNumSFUint32 + 31u) / 32u; ++i) {
                const uint32_t j = i * 32u + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
#else
            (void) remote_sf_ptr;
            (void) local_sf_ptr;
            (void) sf_pool_token_idx;
#endif

            if (elect_one()) {
                const float weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() =
                    weight;

                *workspace.get_token_src_metadata_ptr(pool_token_idx) = {
                    current_rank_in_expert_idx, src_token_idx, src_topk_idx};

#if MEGA_MOE_DISPATCH_PUSH2
                // Approach-2: notify happens in phase B, AFTER the src pushes the
                // token data (so the compute's l1_arrival wait sees ready data).
#elif MEGA_MOE_DISP_BARRIER
                // Relaxed increment: the per-token XGMI store drain is replaced by
                // ONE phase-boundary release-fence + grid_sync below.
                red_add(reinterpret_cast<int *>(workspace.get_l1_arrival_count_ptr(
                            expert_pool_block_offset + token_idx_in_expert / BLOCK_M)),
                        1);
#else
                red_add_rel(workspace.get_l1_arrival_count_ptr(expert_pool_block_offset +
                                                               token_idx_in_expert / BLOCK_M),
                            1u);
#endif
            }
            return true;
        }; // pull_one_token

#if MEGA_MOE_DISPATCH_PUSH
        (void) pull_one_token;
        // ---- PUSH dispatch: source ranks WRITE their tokens into the dst pools
        // (remote write + remote notify), replacing the dst-driven pull. ----
        // Precompute per-global-expert: this rank's dst pool block + slot prefix.
        uint32_t *push_pool_blk   = reinterpret_cast<uint32_t *>(smem_buffer) + kNumExperts;
        uint32_t *push_src_prefix = reinterpret_cast<uint32_t *>(smem_buffer) + 2u * kNumExperts;
        uint32_t *push_recount    = reinterpret_cast<uint32_t *>(smem_buffer) + 3u * kNumExperts;
        uint32_t *push_recv_sum   = reinterpret_cast<uint32_t *>(smem_buffer) + 4u * kNumExperts;
        for (uint32_t e = thread_idx; e < kNumExperts; e += kNumThreads) {
            push_recount[e]    = 0u;
            push_src_prefix[e] = 0u;
            push_recv_sum[e]   = 0u;
        }
        __syncthreads();
        // Recover this SM's per-expert GLOBAL slot base via recount-and-subtract
        // (atomic_add_block alone collides across SMs since an expert spans SMs).
        read_topk_idx([&](const uint32_t &, const int &e) { atomicAdd_block(push_recount + e, 1); });
        // COALESCED bulk-load of the recv_count table (consecutive threads -> consecutive
        // addresses per owner rank) -> per-expert recv_sum + this rank's cross-rank
        // src_prefix.  Replaces ~10K scattered remote READS (the slow XGMI direction).
        for (uint32_t R = 0u; R < kNumRanks; ++R)
            for (uint32_t idx = thread_idx; idx < kNumRanks * kNumExpertsPerRank;
                 idx += kNumThreads) {
                const uint32_t r   = idx / kNumExpertsPerRank;
                const uint32_t el  = idx % kNumExpertsPerRank;
                const uint32_t cnt = static_cast<uint32_t>(
                    *sym_buffer.map(workspace.get_expert_recv_count_ptr(r, el), R));
                const uint32_t e = R * kNumExpertsPerRank + el;
                atomicAdd_block(push_recv_sum + e, cnt);
                if (r < sym_buffer.rank_idx)
                    atomicAdd_block(push_src_prefix + e, cnt);
            }
        __syncthreads();
        // Local prefix sums (smem only, fast).
        for (uint32_t e = thread_idx; e < kNumExperts; e += kNumThreads) {
            const uint32_t base_e = (e / kNumExpertsPerRank) * kNumExpertsPerRank;
            uint32_t       pblk   = 0u;
            for (uint32_t ep = base_e; ep < e; ++ep)
                pblk += (push_recv_sum[ep] + BLOCK_M - 1u) / BLOCK_M;
            push_pool_blk[e]     = pblk;
            smem_expert_count[e] = smem_expert_count[e] - push_recount[e]; // per-SM slot base
        }
        __syncthreads();

        constexpr uint32_t kNumSFUint32 = kHidden / 128u;
        // Same per-SM token distribution as read_topk_idx / the routing slot pass
        // (kNumTokensPerWarp-chunk stride) so the routing-derived per-SM base lines
        // up with the push's token set.  Each warp copies its tokens cooperatively.
        for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
             i += kNumGlobalWarps * kNumTokensPerWarp) {
          for (uint32_t t = 0u; t < kNumTokensPerWarp; ++t) {
            const uint32_t token_idx = i + t;
            if (token_idx >= num_tokens)
                break;
#pragma unroll
            for (uint32_t k = 0u; k < kNumTopk; ++k) {
                int      e_w = -1;
                uint32_t dr_w = 0u, pidx_w = 0u, sfidx_w = 0u, dblk_w = 0u;
                if (elect_one()) {
                    const int e = static_cast<int>(__ldg(
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + k));
                    if (e >= 0) {
                        const uint32_t dr   = static_cast<uint32_t>(e) / kNumExpertsPerRank;
                        const uint32_t slot = push_src_prefix[e] +
                                              atomic_add_block(smem_expert_count + e, 1u);
                        const uint32_t pblk = push_pool_blk[e];
                        e_w     = e;
                        dr_w    = dr;
                        pidx_w  = pblk * BLOCK_M + slot;
                        sfidx_w = pblk * SF_BLOCK_M + transform_sf_token_idx(slot);
                        dblk_w  = pblk + slot / BLOCK_M;
                    }
                }
                e_w = __shfl(e_w, 0);
                if (e_w < 0)
                    continue;
                dr_w    = __shfl(dr_w, 0);
                pidx_w  = __shfl(pidx_w, 0);
                sfidx_w = __shfl(sfidx_w, 0);
                dblk_w  = __shfl(dblk_w, 0);

#if !MEGA_MOE_SKIP_DISPATCH_PULL
                // Token data: local read -> REMOTE write into the dst pool.
                auto *src_tok =
                    input_token_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>();
                auto *dst_tok = sym_buffer.map(
                    l1_token_buffer.get_data_buffer(pidx_w).get_base_ptr<uint8_t>(), dr_w);
                warp_copy_int4<MEGA_MOE_PULL_UNROLL>(dst_tok, src_tok, kHidden / 16u, lane_idx);

                // SF: local read -> REMOTE column-major write into the dst SF pool.
                auto *src_sf = input_sf_buffer.get_data_buffer(token_idx).get_base_ptr<uint32_t>();
                auto *dst_sf = sym_buffer.map(l1_sf_buffer.get_base_ptr<uint32_t>(), dr_w);
#pragma unroll
                for (uint32_t ii = 0u; ii < (kNumSFUint32 + 31u) / 32u; ++ii) {
                    const uint32_t j = ii * 32u + lane_idx;
                    if (j < kNumSFUint32)
                        dst_sf[j * kNumPaddedSFPoolTokens + sfidx_w] = src_sf[j];
                }
#endif
                if (elect_one()) {
#if !MEGA_MOE_PUSH_SKIP_META
                    const float w = *(input_topk_weights_buffer.get_base_ptr<float>() +
                                      token_idx * kNumTopk + k);
                    *sym_buffer.map(
                        l1_topk_weights_buffer.get_data_buffer(pidx_w).get_base_ptr<float>(), dr_w) =
                        w;
                    *sym_buffer.map(workspace.get_token_src_metadata_ptr(pidx_w), dr_w) = {
                        sym_buffer.rank_idx, token_idx, k};
#endif
                    // Notify (relaxed; the cross-rank nvlink_barrier below publishes).
                    red_add(reinterpret_cast<int *>(
                                sym_buffer.map(workspace.get_l1_arrival_count_ptr(dblk_w), dr_w)),
                            1);
                }
            } // for k
          } // for t
        } // for i (token chunk)
#else
        for (uint32_t token_idx = sm_idx * kDispWarps + warp_idx;; token_idx += kNumGlobalWarps)
            if (!pull_one_token(token_idx))
                break;
#endif

#if MEGA_MOE_DISPATCH_PUSH2
        // Phase A above (pull resolution wrote SF/metadata LOCAL + the dst pool slot
        // back to each SRC's mapping).  Barrier so all cross-rank mapping writes are
        // visible, then phase B: each rank reads its mapping (local) and PUSHES only
        // the big token data to the dst pool (fast remote write) + notify.
        prims::release_fence_agent();
        comm::nvlink_barrier<kNumRanks, kNumSMs, 1, 2>(workspace, sym_buffer, kDispLeader, sm_idx,
                                                       thread_idx, true, true);
        const uint32_t *push2_map = combine_token_buffer.get_base_ptr<uint32_t>();
        for (uint32_t i = (sm_idx * kDispWarps + warp_idx) * kNumTokensPerWarp; i < num_tokens;
             i += kNumGlobalWarps * kNumTokensPerWarp) {
          for (uint32_t t = 0u; t < kNumTokensPerWarp; ++t) {
            const uint32_t token_idx = i + t;
            if (token_idx >= num_tokens)
                break;
#pragma unroll
            for (uint32_t k = 0u; k < kNumTopk; ++k) {
                int      e_w = -1;
                uint32_t dr_w = 0u, pidx_w = 0u, dblk_w = 0u;
                if (elect_one()) {
                    const int e = static_cast<int>(__ldg(
                        input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + k));
                    if (e >= 0) {
                        e_w    = e;
                        dr_w   = static_cast<uint32_t>(e) / kNumExpertsPerRank;
                        pidx_w = push2_map[token_idx * kNumTopk + k];
                        dblk_w = pidx_w / BLOCK_M;
                    }
                }
                e_w = __shfl(e_w, 0);
                if (e_w < 0)
                    continue;
                dr_w   = __shfl(dr_w, 0);
                pidx_w = __shfl(pidx_w, 0);
                dblk_w = __shfl(dblk_w, 0);
                auto *src_tok =
                    input_token_buffer.get_data_buffer(token_idx).get_base_ptr<uint8_t>();
                auto *dst_tok = sym_buffer.map(
                    l1_token_buffer.get_data_buffer(pidx_w).get_base_ptr<uint8_t>(), dr_w);
                warp_copy_int4<MEGA_MOE_PULL_UNROLL>(dst_tok, src_tok, kHidden / 16u, lane_idx);
                if (elect_one())
                    red_add(reinterpret_cast<int *>(
                                sym_buffer.map(workspace.get_l1_arrival_count_ptr(dblk_w), dr_w)),
                            1);
            }
          }
        }
#endif

        // NOTES: no cross-role rendezvous before compute -- the same warps run
        // both phases, and the per-block l1_arrival_count spin in the compute
        // loop provides the cross-SM producer->consumer ordering.  recv-stats and
        // the next-launch counter reset are done in the combine tail below.
    }

#if MEGA_MOE_DISPATCH_PUSH || MEGA_MOE_DISPATCH_PUSH2
    // PUSH writes target REMOTE pools -> a cross-rank barrier is required so every
    // rank's pushes (token data [+ approach-1: SF/metadata] + l1_arrival) are
    // globally visible before any rank's compute reads its pool.
    prims::release_fence_agent();
#if MEGA_MOE_PUSH_GRIDSYNC_PROBE
    // PROBE (incorrect for cross-rank, perf-only): use the cheaper intra-rank
    // grid_sync to isolate the nvlink_barrier cost.  --num-correctness-tests 0.
    comm::grid_sync<kNumSMs, 0>(workspace, 0u, sm_idx, thread_idx);
#else
    comm::nvlink_barrier<kNumRanks, kNumSMs, 0, 1>(workspace, sym_buffer, kDispLeader, sm_idx,
                                                   thread_idx, true, true);
#endif
#elif MEGA_MOE_DISP_BARRIER
    // One drain of every wave's relaxed dispatch stores into L2, then a grid-wide
    // barrier so ALL dispatch writes are globally visible before ANY compute read
    // (replaces the per-token release ordering above).
    prims::release_fence_agent();
    comm::grid_sync<kNumSMs, 0>(workspace, 0u, sm_idx, thread_idx);
#endif

    // Compute: persistently schedule over blocks.  Per block, wait the producer
    // arrival (signaled by dispatch above, possibly on another SM), cooperatively
    // load A/B into staged LDS, run the overlapped MFMA k-loop, then the epilogue
    // (L1: SwiGLU + FP8 requant into the L2 pool; L2: BF16 write into the remote
    // combine buffer).
    if (warp_idx >= kNumDispatchWarps && warp_idx < kNumDispatchWarps + kNumMMANonEpilogueWarps) {

        const uint32_t loader_warp_local = warp_idx - kNumDispatchWarps;

        const uint32_t wave_base_byte = kLoaderBaseBytes + loader_warp_local * kLoaderAWaveBytes;

        const uint32_t a_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + wave_base_byte));
        const uint32_t b_lds_stage0 =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_buffer + kLoaderBPoolBaseBytes));

        device::BufferSRD srd_l1_a(l1_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l2_a(l2_token_buffer.get_base_ptr<void>());
        device::BufferSRD srd_l1_b(l1_weights);
        device::BufferSRD srd_l2_b(l2_weights);

        // VGPR-resident MFMA accumulators, also read by the epilogue.  Under Is2B
        // both are live across the whole L1 k-loop: gate -> acc_gate, up -> acc.
        dtype::float32x4 acc[kSubTilesPerWave]      = {};
        dtype::float32x4 acc_gate[kSubTilesPerWave] = {};
#if MEGA_MOE_AGPR_ACC
        // Pinned-AGPR: reserve BOTH accumulator ranges (up a[0:63] + gate a[64:127]
        // = a[0:2*4*kSub-1]) and the operand-staging VGPR window so the compiler
        // keeps them free across the k-loop + pinned MFMA.  Removing acc[]/acc_gate[]
        // from VGPR (all uses redirected below) drops the compiler-managed footprint
        // below the v[242:255] pin window so the one-time reserve actually holds.
        static_assert(kSubTilesPerWave == 16u,
                      "MEGA_MOE_AGPR_ACC assumes the 4-warp config (kSubTilesPerWave==16)");
        device::reserve_agpr_range<0, 2 * static_cast<int>(kSubTilesPerWave) * 4 - 1>();
        device::reserve_vgpr_range<kPinA0, kPinSB1>();
#endif

#if MEGA_MOE_DEFER_EPI
        // Deferred-epilogue pipeline state: the block whose epilogue is pending
        // (to be run during the NEXT same-phase block's prologue).
        bool             have_pending = false;
        sched::BlockPhase pend_phase  = sched::BlockPhase::Linear1;
        uint32_t         pend_pbi = 0u, pend_nbi = 0u, pend_vm = 0u;
#endif

        scheduler.for_each_block([&](sched::BlockPhase phase, uint32_t local_expert_idx,
                                     uint32_t num_k_blocks, uint32_t m_block_idx,
                                     uint32_t n_block_idx) {
#if !MEGA_MOE_DEFER_EPI
            // Terminal sentinel: nothing to do for the non-pipelined path.
            if (phase == sched::BlockPhase::None)
                return;
#endif
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            // Cross-SM producer-arrival wait.  All kNumMMANonEpilogueWarps loader
            // warps cooperate on the SAME block and wait on the SAME per-block
            // arrival counter, so having all 512 threads spin on one global
            // address is pure L2 contention.  Elect ONE thread to poll; its single
            // acquire fence invalidates the CU-wide vector L1, and the
            // __syncthreads broadcasts the release ordering to the whole block.
            // Wrapped in a lambda so DEFER_EPI can move it AFTER the pending-epilogue
            // flush (an L2 arrival-wait must see the deferred L1 signals first).
            auto do_arrival_wait = [&]() {
#if !MEGA_MOE_LOADS_ONLY
                if (thread_idx == 0u) {
                    if (phase == sched::BlockPhase::Linear1) {
                        const auto    *ptr      = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                        const uint32_t expected = scheduler.template get_valid_m<false>();
                        while (ld_volatile(ptr) != expected)
                            __builtin_amdgcn_s_sleep(1);
                    } else {
                        const auto        *ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        constexpr uint32_t kL1GateNBlocksWait = (L1_SHAPE_N / BLOCK_N) / 2u;
                        const uint64_t expected = static_cast<uint64_t>(kNumMMANonEpilogueWarps) *
                                                  static_cast<uint64_t>(kL1GateNBlocksWait);
                        while (ld_volatile(ptr) != expected)
                            __builtin_amdgcn_s_sleep(1);
                    }
                    acquire_fence_agent();
                }
                __syncthreads();
#endif
            };
#if !MEGA_MOE_DEFER_EPI
            do_arrival_wait();
#endif

            const auto &srd_a = (phase == sched::BlockPhase::Linear1) ? srd_l1_a : srd_l2_a;
            const auto &srd_b = (phase == sched::BlockPhase::Linear1) ? srd_l1_b : srd_l2_b;

            // A-row byte stride: FP8 full-width (1 B/elem) for both phases.
            const uint32_t a_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? kHidden : kIntermediateHidden;
            const uint32_t b_row_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? (L1_SHAPE_K / 2u) : (L2_SHAPE_K / 2u);
            const uint32_t b_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                       ? (L1_SHAPE_N * L1_SHAPE_K) / 2u
                                                       : (L2_SHAPE_N * L2_SHAPE_K) / 2u;

            const uint32_t a_wave_row0 =
                pool_block_idx * BLOCK_M + loader_warp_local * kRowsPerLoaderWave;
            const uint32_t a_tile_base_bytes = a_wave_row0 * a_row_stride_bytes;

            uint32_t b_tile_base_bytes = local_expert_idx * b_expert_stride_bytes +
                                         (n_block_idx * BLOCK_N) * b_row_stride_bytes;

            constexpr uint32_t kScaleOne = 0x7f7f7f7fu;

            const auto    *sfa_pool_base = (phase == sched::BlockPhase::Linear1)
                                               ? l1_sf_buffer.get_base_ptr<uint32_t>()
                                               : l2_sf_buffer.get_base_ptr<uint32_t>();
            const uint8_t *sfb_weights_base =
                (phase == sched::BlockPhase::Linear1)
                    ? reinterpret_cast<const uint8_t *>(l1_weights_sf)
                    : reinterpret_cast<const uint8_t *>(l2_weights_sf);
            const uint32_t sfa_pool_token_idx_base = pool_block_idx * SF_BLOCK_M;
            const uint32_t sfb_n_stride_bytes =
                (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_K / kGranK : L2_SHAPE_K / kGranK;
            const uint32_t sfb_expert_stride_bytes = (phase == sched::BlockPhase::Linear1)
                                                         ? (L1_SHAPE_N * L1_SHAPE_K) / kGranK
                                                         : (L2_SHAPE_N * L2_SHAPE_K) / kGranK;

            uint32_t sfb_n_global_base = n_block_idx * BLOCK_N;

#if MEGA_MOE_SF_BLD
            // SRDs for the SF pools (buffer_load_lds path); built once per block.
            device::BufferSRD srd_sfa(reinterpret_cast<const void *>(sfa_pool_base));
            device::BufferSRD srd_sfb(reinterpret_cast<const void *>(sfb_weights_base));
#endif

            // Stage A and B tiles for one (stage, k_block): direct global->LDS
            // (buffer_load to LDS, vmcnt-only, no ds_write), kept separate from SF
            // so the prefetch can stay in flight across the wait below.
            const auto issue_ab = [&](uint32_t stage, uint32_t k_block) {
                const uint32_t b_k_offset_bytes = k_block * (BLOCK_K / 2u);
                const uint32_t a_lds            = a_lds_stage0 + stage * kStagedABytesPerStage;
                const uint32_t b_lds            = b_lds_stage0 + stage * kStagedBBytesPerStage;
                {
                    const uint32_t a_k_offset_bytes = k_block * BLOCK_K;
#pragma unroll
                    for (uint32_t c = 0u; c < kATileLoadsPerWave; ++c) {
                        const uint32_t m_in_wave       = c * 8u + lane_idx / 8u;
                        const uint32_t k_chunk_in_lane = (lane_idx % 8u) ^ ((lane_idx / 8u) & 7u);
                        const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
#if MEGA_MOE_SCALAR_SOFF
                        const uint32_t ldg_offset = a_tile_base_bytes +
                                                    m_in_wave * a_row_stride_bytes + k_byte_in_tile;
                        const int32_t a_soff =
                            __builtin_amdgcn_readfirstlane(static_cast<int32_t>(a_k_offset_bytes));
#else
                        const uint32_t ldg_offset = a_tile_base_bytes +
                                                    m_in_wave * a_row_stride_bytes +
                                                    a_k_offset_bytes + k_byte_in_tile;
                        const int32_t a_soff = 0;
#endif
                        const uint32_t lds_offset =
                            device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                0u, m_in_wave, k_byte_in_tile);
#if !MEGA_MOE_SKIP_GEMM_LOADS
#if MEGA_MOE_VGPR_LOAD
                        device::load_gmem_to_smem_via_vgpr_srd<16, MEGA_MOE_A_CPOL>(
                            srd_a, ldg_offset, a_lds + lds_offset, a_soff);
#else
                        device::load_gmem_to_smem_srd<16, MEGA_MOE_A_CPOL>(srd_a, ldg_offset,
                                                                           a_lds + lds_offset, a_soff);
#endif
#else
                        (void) srd_a;
                        (void) ldg_offset;
                        (void) lds_offset;
#endif
                    }
                }
                // Balanced shared-B load: loader warps round-robin the B chunks
                // (warp w loads chunk c where c % nwarps == w) into the shared-B
                // LDS slot, published to all warps by the __syncthreads() below.
                // NOTES: Is2B stages both gate-B (operand 0) and up-B (operand 1,
                // shifted kIntermediateHidden columns) so one A read drives both
#if !MEGA_MOE_BDIRECT_PROBE
                {
                    const uint32_t num_b_ops =
                        (phase == sched::BlockPhase::Linear1) ? kMaxBOperands : 1u;
                    for (uint32_t b_op = 0u; b_op < num_b_ops; ++b_op) {
                        const uint32_t b_op_base =
                            b_tile_base_bytes + b_op * kIntermediateHidden * b_row_stride_bytes;
                        const uint32_t b_op_lds = b_lds + b_op * kBTileBytes;
#pragma unroll
                        for (uint32_t c = 0u; c < kBTileLoadsPerWave; ++c) {
                            if (c % kNumMMANonEpilogueWarps != loader_warp_local)
                                continue;
                            const uint32_t n_in_wave       = c * 16u + lane_idx / 4u;
                            const uint32_t k_chunk_in_lane = lane_idx % 4u;
                            const uint32_t k_byte_in_tile  = k_chunk_in_lane * kLaneLoadBytes;
#if MEGA_MOE_SCALAR_SOFF
                            const uint32_t ldg_offset =
                                b_op_base + n_in_wave * b_row_stride_bytes + k_byte_in_tile;
                            const int32_t b_soff = __builtin_amdgcn_readfirstlane(
                                static_cast<int32_t>(b_k_offset_bytes));
#else
                            const uint32_t ldg_offset = b_op_base + n_in_wave * b_row_stride_bytes +
                                                        b_k_offset_bytes + k_byte_in_tile;
                            const int32_t b_soff = 0;
#endif
                            const uint32_t lds_offset = device::b_tile_smem_byte_offset_rowmajor<
                                kColsPerLoaderWave, BLOCK_K / 2u>(0u, n_in_wave, k_byte_in_tile);
#if !MEGA_MOE_SKIP_GEMM_LOADS
#if MEGA_MOE_VGPR_LOAD
                            device::load_gmem_to_smem_via_vgpr_srd<16, MEGA_MOE_B_CPOL>(
                                srd_b, ldg_offset, b_op_lds + lds_offset, b_soff);
#else
                            device::load_gmem_to_smem_srd<16, MEGA_MOE_B_CPOL>(
                                srd_b, ldg_offset, b_op_lds + lds_offset, b_soff);
#endif
#else
                            (void) srd_b;
                            (void) ldg_offset;
                            (void) lds_offset;
#endif
                        }
                    }
                }
#else
                (void) srd_b;
#endif
            }; // issue_ab

            // Stage scale factors for one (stage, k_block) via __ldg -> ds_write.
            // SFA: per-warp, one UE8M0 dword per loader-wave row (Linear1 only;
            // Linear2 uses scale 1.0).  SFB: shared, one dword per BLOCK_N column,
            // round-robined across loader warps.
            const auto issue_sf = [&](uint32_t stage, uint32_t k_block) {
#if MEGA_MOE_SKIP_SF
                (void) stage;
                (void) k_block;
                return;
#endif
                {
                    if (phase == sched::BlockPhase::Linear1) {
                        uint32_t *sfa_lds = reinterpret_cast<uint32_t *>(
                            smem_buffer + wave_base_byte + stage * kStagedABytesPerStage +
                            kATileBytes);
#pragma unroll
                        for (uint32_t r = lane_idx; r < kRowsPerLoaderWave; r += kWarpSize) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave + r;
                            const uint32_t sf_token_idx =
                                sfa_pool_token_idx_base + transform_sf_token_idx(m_in_block);
#if MEGA_MOE_SF_BLD
                            // global->LDS direct (vmcnt), same dest sfa_lds[r].
                            device::load_gmem_to_smem_srd<4>(
                                srd_sfa,
                                (k_block * kNumPaddedSFPoolTokens + sf_token_idx) *
                                    static_cast<uint32_t>(sizeof(uint32_t)),
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sfa_lds + r)), 0);
#elif !MEGA_MOE_SKIP_GEMM_LOADS
                            sfa_lds[r] = __ldg(sfa_pool_base + k_block * kNumPaddedSFPoolTokens +
                                               sf_token_idx);
#else
                            (void) sfa_lds;
                            (void) sfa_pool_base;
                            (void) sf_token_idx;
#endif
                        }
                    }
                    uint32_t *sfb_lds = reinterpret_cast<uint32_t *>(
                        smem_buffer + kLoaderSFBPoolBaseBytes + stage * kSFBStageBytes);
                    const uint32_t sfb_k_byte0 = (k_block * BLOCK_K) / kGranK;
                    const uint32_t num_sfb_ops =
                        (phase == sched::BlockPhase::Linear1) ? kMaxBOperands : 1u;
                    for (uint32_t b_op = 0u; b_op < num_sfb_ops; ++b_op) {
                        const uint32_t sfb_op_n_base =
                            sfb_n_global_base + b_op * kIntermediateHidden;
                        uint32_t *sfb_lds_op = sfb_lds + b_op * BLOCK_N;
#pragma unroll
                        for (uint32_t c = loader_warp_local * kWarpSize + lane_idx; c < BLOCK_N;
                             c += kNumMMANonEpilogueWarps * kWarpSize) {
                            const uint32_t n_global = sfb_op_n_base + c;
#if MEGA_MOE_SFB_KMAJOR
                            // K-major weight-SF: [E][k_block][N] -> consecutive n
                            // (lanes) are CONTIGUOUS -> coalesced 4B load (vs the
                            // n-major scatter at stride K/32).  Offline transpose
                            // matches (test, gated MEGA_MOE_SFB_KMAJOR).
                            const uint32_t sfb_N =
                                (phase == sched::BlockPhase::Linear1) ? L1_SHAPE_N : L2_SHAPE_N;
                            const uint32_t off = local_expert_idx * sfb_expert_stride_bytes +
                                                 k_block * (sfb_N * 4u) + n_global * 4u;
#else
                            const uint32_t off = local_expert_idx * sfb_expert_stride_bytes +
                                                 n_global * sfb_n_stride_bytes + sfb_k_byte0;
#endif
#if MEGA_MOE_SF_BLD
                            // global->LDS direct (vmcnt), same dest sfb_lds_op[c].
                            device::load_gmem_to_smem_srd<4>(
                                srd_sfb, off,
                                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sfb_lds_op + c)),
                                0);
#elif !MEGA_MOE_SKIP_GEMM_LOADS
                            sfb_lds_op[c] =
                                __ldg(reinterpret_cast<const uint32_t *>(sfb_weights_base + off));
#else
                            (void) sfb_lds_op;
                            (void) sfb_weights_base;
                            (void) off;
#endif
                        }
                    }
                }
            };

            // Run the full k-loop for the current N-tile, accumulating into
            // acc[] (the live VGPR MFMA accumulators).
            // Split into prologue (issue the first loads) + main (reset acc, MFMA
            // loop) so the block loop can run a DEFERRED epilogue between them,
            // overlapping the prologue load latency.  Default calls both back-to-back;
            // moving acc-reset after the issue is independent -> neutral.
            auto kloop_prologue = [&]() {
                if (num_k_blocks > 0u) {
                    issue_ab(0u, 0u);
                    issue_sf(0u, 0u);
                }
#if !MEGA_MOE_2STAGE
                if (num_k_blocks > 1u) {
                    issue_ab(1u % kLoaderStages, 1u);
                    issue_sf(1u % kLoaderStages, 1u);
                }
#endif
            };

            auto kloop_main = [&]() {
#if MEGA_MOE_AGPR_ACC
                // Zero the AGPR accumulator range(s): up a[0:63] always; gate
                // a[64:127] additionally for Linear1.  acc[]/acc_gate[] VGPR unused.
                if (phase == sched::BlockPhase::Linear1)
                    device::zero_agpr_range<kAccUp,
                                            kAccGate + static_cast<int>(kSubTilesPerWave) * 4 - 1>();
                else
                    device::zero_agpr_range<kAccUp, static_cast<int>(kSubTilesPerWave) * 4 - 1>();
#else
#pragma unroll
                for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                    acc[s] = dtype::float32x4{};
                if (phase == sched::BlockPhase::Linear1) {
#pragma unroll
                    for (uint32_t s = 0u; s < kSubTilesPerWave; ++s)
                        acc_gate[s] = dtype::float32x4{};
                }
#endif

                // One k_block's MFMA burst: read A/SFA/B/SFB from this stage's LDS
                // slot and accumulate kSubTilesM x kSubTilesN MFMAs into acc[].  A
                // lambda so the overlap loop can issue two back-to-back per barrier.
                auto do_burst = [&](uint32_t k_block, uint32_t this_stage) {
                    const uint32_t a_stage_byte =
                        wave_base_byte + this_stage * kStagedABytesPerStage;
                    const uint32_t b_stage_byte =
                        kLoaderBPoolBaseBytes + this_stage * kStagedBBytesPerStage;
                    // SF read shadows for this stage: SFA per-warp (in the A
                    // stage SF region), SFB shared (in the SFB pool).
                    const uint32_t *sfa_lds_u32 = reinterpret_cast<const uint32_t *>(
                        smem_buffer + a_stage_byte + kATileBytes);
                    const uint32_t *sfb_lds_u32 = reinterpret_cast<const uint32_t *>(
                        smem_buffer + kLoaderSFBPoolBaseBytes + this_stage * kSFBStageBytes);

                    auto read_int32x8 = [&](uint32_t base_byte, uint32_t off_lo, uint32_t off_hi) {
                        dtype::int32x8 v;
                        const auto     lo =
                            *reinterpret_cast<const uint4 *>(smem_buffer + base_byte + off_lo);
                        const auto hi =
                            *reinterpret_cast<const uint4 *>(smem_buffer + base_byte + off_hi);
                        reinterpret_cast<uint4 *>(&v)[0] = lo;
                        reinterpret_cast<uint4 *>(&v)[1] = hi;
                        return v;
                    };

                    auto read_fp4_b = [&](uint32_t base_byte, uint32_t off) {
                        dtype::int32x8 v;
#pragma unroll
                        for (uint32_t i = 0u; i < 8u; ++i)
                            v[i] = 0;
                        reinterpret_cast<uint4 *>(&v)[0] =
                            *reinterpret_cast<const uint4 *>(smem_buffer + base_byte + off);
                        return v;
                    };

#pragma unroll
                    for (uint32_t k_inner = 0u; k_inner < kInnerKIters; ++k_inner) {
                        const uint32_t m_in_subtile = lane_idx & 15u;
                        const uint32_t kb_in_lane   = (lane_idx >> 4u) & 3u;
                        const uint32_t k_lo         = kb_in_lane * 16u;
                        const uint32_t a_win_hi     = k_lo + 64u;
                        const uint32_t perm_kblk    = kb_in_lane;

#if MEGA_MOE_AGPR_ACC
                        // ds_read both A frags DIRECTLY into pinned VGPRs (no a_vec
                        // C++ array, no v_mov): A0->v[kPinA0:+7], A1->v[kPinA1:+7];
                        // two ds_read_b128 per frag (k_lo and k_lo+64, swizzled).
                        // SFA scales are computed -> set_vgpr.  Staged ONCE per
                        // k_inner, shared by gate+up and all sub_n.
                        static_assert(kSubTilesM == 2u, "AGPR A-staging assumes kSubTilesM==2");
                        {
                            const uint32_t a_base = static_cast<uint32_t>(
                                reinterpret_cast<uintptr_t>(smem_buffer + a_stage_byte));
                            const uint32_t m0 = 0u * kMfmaM + m_in_subtile;
                            const uint32_t m1 = 1u * kMfmaM + m_in_subtile;
                            device::ds_read_pinned<16, kPinA0 + 0, 0>(
                                a_base + device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                             0u, m0, k_lo));
                            device::ds_read_pinned<16, kPinA0 + 4, 0>(
                                a_base + device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                             0u, m0, a_win_hi));
                            device::ds_read_pinned<16, kPinA1 + 0, 0>(
                                a_base + device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                             0u, m1, k_lo));
                            device::ds_read_pinned<16, kPinA1 + 4, 0>(
                                a_base + device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                             0u, m1, a_win_hi));
                            if (phase == sched::BlockPhase::Linear2) {
                                device::set_vgpr<kPinSA0>(kScaleOne);
                                device::set_vgpr<kPinSA1>(kScaleOne);
                            } else {
                                const uint32_t d0 = sfa_lds_u32[0u * kMfmaM + m_in_subtile];
                                const uint32_t d1 = sfa_lds_u32[1u * kMfmaM + m_in_subtile];
                                device::set_vgpr<kPinSA0>(0x7f7f7f00u |
                                                          ((d0 >> (perm_kblk * 8u)) & 0xffu));
                                device::set_vgpr<kPinSA1>(0x7f7f7f00u |
                                                          ((d1 >> (perm_kblk * 8u)) & 0xffu));
                            }
                            device::wait_lgkmcnt<0>(); // A frags landed in pinned VGPRs
                        }
#else
                        dtype::int32x8 a_vec[kSubTilesM];
                        uint32_t       sa_arr[kSubTilesM];
#pragma unroll
                        for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                            const uint32_t m_in_wave = sub_m * kMfmaM + m_in_subtile;
                            {
                                const uint32_t off_lo =
                                    device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                        0u, m_in_wave, k_lo);
                                const uint32_t off_hi =
                                    device::a_tile_smem_byte_offset<kRowsPerLoaderWave, BLOCK_K>(
                                        0u, m_in_wave, a_win_hi);
                                a_vec[sub_m] = read_int32x8(a_stage_byte, off_lo, off_hi);
                            }

                            if (phase == sched::BlockPhase::Linear2) {
                                sa_arr[sub_m] = kScaleOne;
                            } else {
                                const uint32_t sfa_dword_db =
                                    sfa_lds_u32[sub_m * kMfmaM + m_in_subtile];
                                const uint32_t sfa_byte =
                                    (sfa_dword_db >> (perm_kblk * 8u)) & 0xffu;
                                sa_arr[sub_m] = 0x7f7f7f00u | sfa_byte;
                            }
                        }
#endif

                        auto b_off = [&](uint32_t sub_n) {
                            const uint32_t n_in_wave = sub_n * kMfmaN + m_in_subtile;
                            return device::b_tile_smem_byte_offset_rowmajor<kColsPerLoaderWave,
                                                                            BLOCK_K / 2u>(
                                0u, n_in_wave, kb_in_lane * 16u);
                        };
                        // One B operand's MFMA pass: read its B subtile + SFB byte
                        // and accumulate into acc_tgt[].  A (a_vec/sa_arr) is read
                        // ONCE above and shared by both operands -- the Is2B halving.
                        // (Builtin/VGPR-acc path only; under AGPR_ACC a_vec/sa_arr do
                        // not exist -- the pinned run_op_agpr below is used instead.)
#if !MEGA_MOE_AGPR_ACC
                        auto run_operand = [&](uint32_t b_lds_off, uint32_t sfb_u32_off,
                                               dtype::float32x4 *acc_tgt) {
                            uint32_t sb_arr[kSubTilesN];
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sfb_dword =
                                    sfb_lds_u32[sfb_u32_off + sub_n * kMfmaN + m_in_subtile];
                                sb_arr[sub_n] =
                                    0x7f7f7f00u | ((sfb_dword >> (kb_in_lane * 8u)) & 0xffu);
                            }
#if MEGA_MOE_MFMA_SCHED3
                            // SCHED but 2-AHEAD prefetch (3 b_vec live).
                            dtype::int32x8 b0 = read_fp4_b(b_stage_byte + b_lds_off, b_off(0u));
                            dtype::int32x8 b1 =
                                (kSubTilesN > 1u)
                                    ? read_fp4_b(b_stage_byte + b_lds_off, b_off(1u % kSubTilesN))
                                    : b0;
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const dtype::int32x8 b_cur = b0;
                                b0                         = b1;
                                const uint32_t nxt         = sub_n + 2u;
                                if (nxt < kSubTilesN)
                                    b1 = read_fp4_b(b_stage_byte + b_lds_off, b_off(nxt));
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
#pragma unroll
                                for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                                    const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                    acc_tgt[sub] = mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                        a_vec[sub_m], b_cur, acc_tgt[sub], sa_arr[sub_m],
                                        sb_arr[sub_n]);
                                }
                                __builtin_amdgcn_s_setprio(0);
                                __builtin_amdgcn_sched_barrier(0);
                            }
#elif MEGA_MOE_MFMA_SCHED
                            // PP_SCHED (4WARP_BAK port): 2-deep B-prefetch software
                            // pipeline -- read b[sub_n+1] ahead of MFMA[sub_n], lock the
                            // interleave with sched_barrier(0), bracket the MFMA with
                            // s_setprio(1/0).  Only 2 b_vec live (VGPR-light).
                            dtype::int32x8 b_buf = read_fp4_b(b_stage_byte + b_lds_off, b_off(0u));
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const dtype::int32x8 b_cur = b_buf;
                                if (sub_n + 1u < kSubTilesN)
                                    b_buf = read_fp4_b(b_stage_byte + b_lds_off, b_off(sub_n + 1u));
                                __builtin_amdgcn_sched_barrier(0);
#if !MEGA_MOE_MFMA_NOPRIO
                                __builtin_amdgcn_s_setprio(1);
#endif
#pragma unroll
                                for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                                    const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                    acc_tgt[sub] = mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                        a_vec[sub_m], b_cur, acc_tgt[sub], sa_arr[sub_m],
                                        sb_arr[sub_n]);
                                }
#if !MEGA_MOE_MFMA_NOPRIO
                                __builtin_amdgcn_s_setprio(0);
#endif
                                __builtin_amdgcn_sched_barrier(0);
                            }
#elif MEGA_MOE_MFMA_BPF
                            // B-prefetch: hoist ALL kSubTilesN B-subtile LDS reads (fp4
                            // = one uint4/lane) into registers BEFORE issuing any MFMA,
                            // so the ds_reads overlap MFMA execution instead of each
                            // MFMA stalling on its own immediately-preceding ds_read
                            // (kSubTilesM==1 -> 1 MFMA/b_vec, no reuse to hide it).
                            // Pure reorder, identical math -> gate-3 safe.  Costs
                            // kSubTilesN*4 VGPR for the b_vec shadows (risk: spill).
                            uint4 b_pf[kSubTilesN];
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n)
                                b_pf[sub_n] = *reinterpret_cast<const uint4 *>(
                                    smem_buffer + b_stage_byte + b_lds_off + b_off(sub_n));
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                dtype::int32x8 b_vec;
#pragma unroll
                                for (uint32_t i = 0u; i < 8u; ++i)
                                    b_vec[i] = 0;
                                reinterpret_cast<uint4 *>(&b_vec)[0] = b_pf[sub_n];
#pragma unroll
                                for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                                    const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                    acc_tgt[sub] = mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                        a_vec[sub_m], b_vec, acc_tgt[sub], sa_arr[sub_m],
                                        sb_arr[sub_n]);
                                }
                            }
#else
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const dtype::int32x8 b_vec =
                                    read_fp4_b(b_stage_byte + b_lds_off, b_off(sub_n));
#pragma unroll
                                for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
                                    const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                    acc_tgt[sub] = mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                        a_vec[sub_m], b_vec, acc_tgt[sub], sa_arr[sub_m],
                                        sb_arr[sub_n]);
                                }
                            }
#endif
                        };
#endif // !MEGA_MOE_AGPR_ACC (run_operand)
#if MEGA_MOE_AGPR_ACC
                        // Pinned-AGPR operand burst: accumulate one B operand into the
                        // AGPR range based at ACC_BASE.  A (a_vec) + scales (sa_arr) are
                        // read above; B + SFB read per sub_n.  Operands staged into the
                        // reserved pinned VGPR window via set_vgpr (the A LDS swizzle
                        // precludes turbo's compile-time-offset ds_read_pinned), then
                        // run_pinned_acc_agpr issues the MFMA with acc resident in AGPR.
                        using AgprMfma = device::mfma_scale_f32_16x16x128_f8f6f4<
                            __hip_fp8_e4m3, dtype::float4x2_e2m1>;
                        auto run_op_agpr = [&]<int ACC_BASE>(uint32_t b_lds_off,
                                                             uint32_t sfb_u32_off) {
                            const uint32_t b_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
                                smem_buffer + b_stage_byte + b_lds_off));
                            // Double-buffered B: prefetch B[SN+1] (ds_read runs during
                            // MFMA[SN]); partial wait_lgkmcnt<1> drains the consumed B[SN]
                            // (LDS reads complete in issue order) and leaves the prefetch
                            // in flight, so the ~25cyc ds_read hides behind the 2 MFMAs.
                            // A frags pre-staged once in kPinA0/kPinA1.
                            auto issb = [&](int SN, int buf) {
                                const uint32_t d =
                                    sfb_lds_u32[sfb_u32_off + static_cast<uint32_t>(SN) * kMfmaN +
                                                m_in_subtile];
                                const uint32_t sb = 0x7f7f7f00u | ((d >> (kb_in_lane * 8u)) & 0xffu);
                                if (buf == 0) {
                                    device::ds_read_pinned<16, kPinB0, 0>(
                                        b_base + b_off(static_cast<uint32_t>(SN)));
                                    device::set_vgpr<kPinSB0>(sb);
                                } else {
                                    device::ds_read_pinned<16, kPinB1, 0>(
                                        b_base + b_off(static_cast<uint32_t>(SN)));
                                    device::set_vgpr<kPinSB1>(sb);
                                }
                            };
                            issb(0, 0); // prologue: B[0] -> buf0
                            mega_static_for<0, static_cast<int>(kSubTilesN)>([&]<int SN>() {
                                constexpr int kN = static_cast<int>(kSubTilesN);
                                if constexpr (SN + 1 < kN) {
                                    issb(SN + 1, (SN + 1) & 1); // prefetch next (overlaps MFMA)
                                    device::wait_lgkmcnt<1>();   // B[SN] done; prefetch in flight
                                } else {
                                    device::wait_lgkmcnt<0>();   // last: drain
                                }
                                if constexpr ((SN & 1) == 0) {
                                    AgprMfma::run_pinned_acc_agpr<
                                        kPinA0, kPinB0, ACC_BASE + (0 * kN + SN) * 4, kPinSA0,
                                        kPinSB0>();
                                    AgprMfma::run_pinned_acc_agpr<
                                        kPinA1, kPinB0, ACC_BASE + (1 * kN + SN) * 4, kPinSA1,
                                        kPinSB0>();
                                } else {
                                    AgprMfma::run_pinned_acc_agpr<
                                        kPinA0, kPinB1, ACC_BASE + (0 * kN + SN) * 4, kPinSA0,
                                        kPinSB1>();
                                    AgprMfma::run_pinned_acc_agpr<
                                        kPinA1, kPinB1, ACC_BASE + (1 * kN + SN) * 4, kPinSA1,
                                        kPinSB1>();
                                }
                            });
                        };
#endif
#if MEGA_MOE_MFMA_PRIO
                        asm volatile("s_setprio 1");
#endif
                        if (phase == sched::BlockPhase::Linear1) {
#if MEGA_MOE_AGPR_ACC
                            run_op_agpr.template operator()<kAccGate>(0u, 0u);          // gate
                            run_op_agpr.template operator()<kAccUp>(kBTileBytes, BLOCK_N); // up
#else
#if MEGA_MOE_MFMA_SCHED2
                            static_assert(kSubTilesM == 1u, "SCHED2 assumes kSubTilesM==1");
                            // Unified gate+up locked pipeline: 2*kSubTilesN slots, the
                            // up operand's B prefetched during gate's tail (no cold start
                            // at the operand boundary).  sched_barrier(0)+setprio lock.
                            uint32_t sbg[kSubTilesN], sbu[kSubTilesN];
#pragma unroll
                            for (uint32_t sn = 0u; sn < kSubTilesN; ++sn) {
                                const uint32_t dg = sfb_lds_u32[0u + sn * kMfmaN + m_in_subtile];
                                const uint32_t du =
                                    sfb_lds_u32[BLOCK_N + sn * kMfmaN + m_in_subtile];
                                sbg[sn] = 0x7f7f7f00u | ((dg >> (kb_in_lane * 8u)) & 0xffu);
                                sbu[sn] = 0x7f7f7f00u | ((du >> (kb_in_lane * 8u)) & 0xffu);
                            }
                            constexpr uint32_t kL1Slots = 2u * kSubTilesN;
                            auto rd = [&](uint32_t slot) {
                                const bool     up = slot >= kSubTilesN;
                                const uint32_t sn = up ? slot - kSubTilesN : slot;
                                return read_fp4_b(b_stage_byte + (up ? kBTileBytes : 0u),
                                                  b_off(sn));
                            };
                            dtype::int32x8 b_buf = rd(0u);
#pragma unroll
                            for (uint32_t slot = 0u; slot < kL1Slots; ++slot) {
                                const dtype::int32x8 b_cur = b_buf;
                                if (slot + 1u < kL1Slots)
                                    b_buf = rd(slot + 1u);
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                const bool     up = slot >= kSubTilesN;
                                const uint32_t sn = up ? slot - kSubTilesN : slot;
                                dtype::float32x4 *acc_tgt = up ? acc : acc_gate;
                                const uint32_t   sb       = up ? sbu[sn] : sbg[sn];
                                acc_tgt[sn] = mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                    a_vec[0], b_cur, acc_tgt[sn], sa_arr[0], sb);
                                __builtin_amdgcn_s_setprio(0);
                                __builtin_amdgcn_sched_barrier(0);
                            }
#elif MEGA_MOE_MFMA_BPF2
                            static_assert(kSubTilesM == 1u, "BPF2 assumes kSubTilesM==1");
                            // CROSS-OPERAND prefetch: hoist BOTH gate-B and up-B (+SFB)
                            // for all kSubTilesN into registers, then issue gate MFMAs
                            // then up MFMAs -- so the up-operand's LDS reads overlap the
                            // gate MFMA compute (vs run_operand-twice, where up's reads
                            // stall after gate's last MFMA).  2x the b_vec VGPR shadows
                            // (spill risk).  Pure reorder -> gate-3 safe.
                            uint4    bg_pf[kSubTilesN], bu_pf[kSubTilesN];
                            uint32_t sbg[kSubTilesN], sbu[kSubTilesN];
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                bg_pf[sub_n] = *reinterpret_cast<const uint4 *>(
                                    smem_buffer + b_stage_byte + 0u + b_off(sub_n));
                                bu_pf[sub_n] = *reinterpret_cast<const uint4 *>(
                                    smem_buffer + b_stage_byte + kBTileBytes + b_off(sub_n));
                                const uint32_t dg =
                                    sfb_lds_u32[0u + sub_n * kMfmaN + m_in_subtile];
                                const uint32_t du =
                                    sfb_lds_u32[BLOCK_N + sub_n * kMfmaN + m_in_subtile];
                                sbg[sub_n] = 0x7f7f7f00u | ((dg >> (kb_in_lane * 8u)) & 0xffu);
                                sbu[sub_n] = 0x7f7f7f00u | ((du >> (kb_in_lane * 8u)) & 0xffu);
                            }
                            auto issue = [&](uint4 *bpf, uint32_t *sb, dtype::float32x4 *acc_tgt) {
#pragma unroll
                                for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                    dtype::int32x8 b_vec;
#pragma unroll
                                    for (uint32_t i = 0u; i < 8u; ++i)
                                        b_vec[i] = 0;
                                    reinterpret_cast<uint4 *>(&b_vec)[0] = bpf[sub_n];
                                    acc_tgt[sub_n] =
                                        mfma_scaled<__hip_fp8_e4m3, dtype::float4x2_e2m1>(
                                            a_vec[0], b_vec, acc_tgt[sub_n], sa_arr[0], sb[sub_n]);
                                }
                            };
                            issue(bg_pf, sbg, acc_gate);
                            issue(bu_pf, sbu, acc);
#else
                            // gate -> acc_gate (operand 0), up -> acc (operand 1).
                            run_operand(0u, 0u, acc_gate);
                            run_operand(kBTileBytes, BLOCK_N, acc);
#endif
#endif // MEGA_MOE_AGPR_ACC (Linear1 dispatch select)
                        } else {
#if MEGA_MOE_AGPR_ACC
                            run_op_agpr.template operator()<kAccUp>(0u, 0u); // L2 -> a[0:63]
#else
                            run_operand(0u, 0u, acc);
#endif
                        }
#if MEGA_MOE_MFMA_PRIO
                        asm volatile("s_setprio 0");
#endif
                    }
                }; // do_burst

                // k_block overlap: load two B-stages (k, k+1), drain with a
                // single full wait_vmcnt<0>, publish with ONE __syncthreads(), then
                // issue BOTH MFMA bursts back-to-back so the matrix pipe stays fed.
                // NOTES: the two bursts read stages s0=k%S, s1=(k+1)%S while the
                // loader fills (k+2)%S, (k+3)%S -- 4 distinct in-flight stages, so
                // requires kLoaderStages>=4.  A partial wait_vmcnt<K> is unsafe on
                // CDNA4 (no vmem completion-order guarantee) and the 2-wave/SIMD
                // occupancy already hides the full-drain latency.
#if MEGA_MOE_2STAGE
                // 2-stage double buffer: stage 0 preloaded by the prologue above;
                // each iter drains stage k&1, prefetches k+1 into the other slot
                // (in flight during the burst), then bursts k.
                static_assert(kLoaderStages >= 2u, "2-stage needs kLoaderStages>=2");
                for (uint32_t k_block = 0u; k_block < num_k_blocks; ++k_block) {
                    const uint32_t s = k_block & 1u;
#if !MEGA_MOE_NO_KLOOP_SYNC
                    device::wait_vmcnt<MEGA_MOE_VMCNT_KEEP>();
                    device::wait_lgkmcnt<0>();
                    __syncthreads();
#endif
                    if (k_block + 1u < num_k_blocks) {
                        issue_ab((k_block + 1u) & 1u, k_block + 1u);
                        issue_sf((k_block + 1u) & 1u, k_block + 1u);
                    }
#if !MEGA_MOE_NO_BURST
                    do_burst(k_block, s);
#else
                    (void) s;
#endif
                }
#else
                static_assert(kLoaderStages >= 4u,
                              "k_block-overlap needs kLoaderStages>=4 (reads s0,s1 "
                              "while loading s0+2,s1+2 -> 4 distinct in-flight stages)");

                // stage-0/1 prologue loads issued by kloop_prologue() before this.

                for (uint32_t _rep = 0u; _rep < MEGA_MOE_KLOOP_REPEAT; ++_rep)
                for (uint32_t k_block = 0u; k_block < num_k_blocks; k_block += 2u) {
                    const uint32_t s0 = k_block % kLoaderStages;
                    const uint32_t s1 = (k_block + 1u) % kLoaderStages;

                    // Both stage-k and stage-(k+1) loads have landed.
#if !MEGA_MOE_NO_KLOOP_SYNC
                    device::wait_vmcnt<MEGA_MOE_VMCNT_KEEP>();
                    device::wait_lgkmcnt<0>();

                    // ONE rendezvous publishes BOTH stages' shared-B to all warps.
                    __syncthreads();
#endif

                    // Prefetch the NEXT pair of stages (k+2, k+3) AFTER the wait.
                    if (k_block + 2u < num_k_blocks) {
                        issue_ab((k_block + 2u) % kLoaderStages, k_block + 2u);
                        issue_sf((k_block + 2u) % kLoaderStages, k_block + 2u);
                    }
                    if (k_block + 3u < num_k_blocks) {
                        issue_ab((k_block + 3u) % kLoaderStages, k_block + 3u);
                        issue_sf((k_block + 3u) % kLoaderStages, k_block + 3u);
                    }
#if MEGA_MOE_KLOOP_SCHED
                    // Lock the next-stage buffer_load_lds ISSUE before the MFMA bursts
                    // so the async loads are all in flight during the MFMAs (max
                    // overlap), instead of the compiler interleaving the load issue
                    // late among the MFMAs.  Load-side analog of MFMA_SCHED.  VGPR-free.
                    __builtin_amdgcn_sched_barrier(0);
#endif

                    // Two MFMA bursts back-to-back, NO barrier between.
#if !MEGA_MOE_NO_BURST
                    do_burst(k_block, s0);
                    if (k_block + 1u < num_k_blocks) // odd-tail: skip 2nd burst
                        do_burst(k_block + 1u, s1);
#else
                    (void) s0;
                    (void) s1;
#endif
                }
#endif
            };

            constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
            static_assert(kNumL1BlockNs >= 2u && kNumL1BlockNs % 2u == 0u,
                          "L1 N must split evenly into gate||up halves");
            static_assert(kIntermediateHidden % BLOCK_N == 0u,
                          "BLOCK_N must divide kIntermediateHidden for SwiGLU pairing");
            constexpr uint32_t kL1GateNBlocks = kNumL1BlockNs / 2u;

            // (Block orchestration -- skip-check, kloop, epilogue -- is placed AFTER
            // run_epilogue's definition below so the DEFER path can call it.)

            // Epilogue extracted into a lambda so the block loop can DEFER it (run
            // block N's epilogue during block N+1's prologue loads).  Params carry
            // the (possibly deferred) block's state since the scheduler advances.
            auto run_epilogue = [&](sched::BlockPhase phase, uint32_t pool_block_idx,
                                    uint32_t n_block_idx, uint32_t valid_m)
#if MEGA_MOE_EPI_NOINLINE
                __attribute__((noinline))
#endif
            {
#if !MEGA_MOE_LOADS_ONLY
                const uint32_t n_lane     = lane_idx & 15u;
                const uint32_t m_out_base = ((lane_idx >> 4u) & 3u) * 4u;

                if (phase == sched::BlockPhase::Linear1) {

                    auto *l2_pool_base = l2_token_buffer.get_base_ptr<uint8_t>();
#if MEGA_MOE_COALESCE_EPI
                    // L1 epilogue, COALESCED: SwiGLU+quant each column, scatter the fp8
                    // bytes into a per-warp LDS row (reusing the loader's free B pool),
                    // then each lane writes 8 CONTIGUOUS fp8 (uint2) to the local pool
                    // instead of kSubTilesN strided 1-byte writes.  Pool layout
                    // (row-major per token) is unchanged -> L2 GEMM A read identical.
                    static_assert(kIntermediateHidden % BLOCK_N == 0u, "BLOCK_N | interm");
                    static_assert(BLOCK_N % 8u == 0u, "uint2 (8 fp8) must tile BLOCK_N");
                    constexpr uint32_t kEpiGroups = kWarpSize / kMfmaN;
                    auto *xpose = smem_buffer + kLoaderBPoolBaseBytes +
                                  loader_warp_local * kEpiGroups * BLOCK_N;
                    const uint32_t grp = (lane_idx >> 4u) & (kEpiGroups - 1u);
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t i = 0u; i < 4u; ++i) {
                            uint8_t *row_lds = xpose + grp * BLOCK_N;
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub = sub_m * kSubTilesN + sub_n;
                                const float gate =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, acc_gate[sub][i]));
                                const float up =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, acc[sub][i]));
                                const float          swiglu = (gate / (1.0f + __expf(-gate))) * up;
                                const __hip_fp8_e4m3 quant  = static_cast<__hip_fp8_e4m3>(swiglu);
                                row_lds[sub_n * kMfmaN + n_lane] =
                                    reinterpret_cast<const uint8_t &>(quant);
                            }
#if MEGA_MOE_EPI_LDS_BULK
                            // Per-warp xpose: only the wave's own ds_writes must land
                            // before the wave's ds_reads below (lockstep wave64), so a
                            // wave-local s_waitcnt replaces the block-wide barrier.
                            primus_turbo::device::wait_lgkmcnt<0>();
#else
                            __syncthreads();
#endif
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_out_base + i;
                            if (m_in_block < valid_m) {
                                const uint32_t pool_token_idx = pool_block_idx * BLOCK_M + m_in_block;
                                auto *dst_row = l2_pool_base +
                                                pool_token_idx * fp8_intermediate_token_layout.num_bytes;
                                const uint32_t col = n_lane * 8u;
#if !MEGA_MOE_SKIP_EPI
                                *reinterpret_cast<uint2 *>(dst_row + n_block_idx * BLOCK_N + col) =
                                    *reinterpret_cast<const uint2 *>(row_lds + col);
#else
                                (void) dst_row;
                                (void) col;
#endif
                            }
#if MEGA_MOE_EPI_LDS_BULK
                            // Next i overwrites row_lds; ensure this wave's ds_read for
                            // the store above completed first (wave-local, no barrier).
                            primus_turbo::device::wait_lgkmcnt<0>();
#else
                            __syncthreads();
#endif
                        }
                    }
#elif MEGA_MOE_EPI_BATCH_STORE
                    // Compute ALL quant bytes first (VALU), then issue all stores.
                    uint8_t qbuf[kSubTilesM * kSubTilesN * 4u];
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m)
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub = sub_m * kSubTilesN + sub_n;
#pragma unroll
                            for (uint32_t i = 0u; i < 4u; ++i) {
                                const float gate = __builtin_amdgcn_fmed3f(
                                    acc_gate[sub][i], -kActivationClamp, kActivationClamp);
                                const float up = __builtin_amdgcn_fmed3f(
                                    acc[sub][i], -kActivationClamp, kActivationClamp);
                                const float silu_gate = gate / (1.0f + __expf(-gate));
                                const float swiglu    = silu_gate * up;
                                const uint32_t quant_pk =
                                    __builtin_amdgcn_cvt_pk_fp8_f32(swiglu, swiglu, 0, false);
                                qbuf[sub * 4u + i] = static_cast<uint8_t>(quant_pk & 0xffu);
                            }
                        }
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m)
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub              = sub_m * kSubTilesN + sub_n;
                            const uint32_t n_in_wave        = sub_n * kMfmaN + n_lane;
                            const uint32_t intermediate_col = n_block_idx * BLOCK_N + n_in_wave;
                            if (intermediate_col >= kIntermediateHidden)
                                continue;
#pragma unroll
                            for (uint32_t i = 0u; i < 4u; ++i) {
                                const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                            sub_m * kMfmaM + m_out_base + i;
                                if (m_in_block >= valid_m)
                                    continue;
                                const uint32_t pool_token_idx =
                                    pool_block_idx * BLOCK_M + m_in_block;
                                auto *dst_row =
                                    l2_pool_base +
                                    pool_token_idx * fp8_intermediate_token_layout.num_bytes;
#if !MEGA_MOE_SKIP_EPI
                                dst_row[intermediate_col] = qbuf[sub * 4u + i];
#else
                                (void) dst_row;
#endif
                            }
                        }
#else
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                            const uint32_t sub              = sub_m * kSubTilesN + sub_n;
                            const uint32_t n_in_wave        = sub_n * kMfmaN + n_lane;
                            const uint32_t intermediate_col = n_block_idx * BLOCK_N + n_in_wave;
                            if (intermediate_col >= kIntermediateHidden)
                                continue;
#if MEGA_MOE_AGPR_ACC
                            // Stream this subtile's gate/up acc from AGPR (4 VGPR each,
                            // live only across the i-loop) instead of a 128-VGPR hoist.
                            const dtype::float32x4 g_sub = read_acc_sub16<kAccGate>(sub);
                            const dtype::float32x4 u_sub = read_acc_sub16<kAccUp>(sub);
#endif
#pragma unroll
                            for (uint32_t i = 0u; i < 4u; ++i) {
                                const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                            sub_m * kMfmaM + m_out_base + i;
                                if (m_in_block >= valid_m)
                                    continue;
                                const uint32_t pool_token_idx =
                                    pool_block_idx * BLOCK_M + m_in_block;
                                auto *dst_row =
                                    l2_pool_base +
                                    pool_token_idx * fp8_intermediate_token_layout.num_bytes;

#if MEGA_MOE_AGPR_ACC
                                const float gate_raw = g_sub[i];
                                const float up_raw   = u_sub[i];
#else
                                const float gate_raw = acc_gate[sub][i];
                                const float up_raw   = acc[sub][i];
#endif
#if MEGA_MOE_FAST_QUANT
                                // fmed3f = median(x,lo,hi) = clamp in ONE op vs
                                // fmaxf(fminf()) = two.  Numerically identical.
                                const float gate = __builtin_amdgcn_fmed3f(
                                    gate_raw, -kActivationClamp, kActivationClamp);
                                const float up = __builtin_amdgcn_fmed3f(
                                    up_raw, -kActivationClamp, kActivationClamp);
#else
                                const float gate =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, gate_raw));
                                const float up =
                                    fmaxf(-kActivationClamp, fminf(kActivationClamp, up_raw));
#endif

#if MEGA_MOE_SKIP_SILU
                                const float          swiglu = gate * up;
#else
                                const float          silu_gate = gate / (1.0f + __expf(-gate));
                                const float          swiglu    = silu_gate * up;
#endif
#if MEGA_MOE_EPI_CONST
                                const uint8_t quant_byte =
                                    static_cast<uint8_t>(intermediate_col & 0xffu);
#elif MEGA_MOE_FAST_QUANT
                                // swiglu is already clamped to +/-kActivationClamp
                                // (<< fp8 e4m3 range +/-448), so skip the cast's
                                // internal saturate/NAN-INF branch: convert directly
                                // via the HW pack instruction and take byte 0.
                                const uint32_t quant_pk =
                                    __builtin_amdgcn_cvt_pk_fp8_f32(swiglu, swiglu, 0, false);
                                const uint8_t quant_byte = static_cast<uint8_t>(quant_pk & 0xffu);
#else
                                const __hip_fp8_e4m3 quant = static_cast<__hip_fp8_e4m3>(swiglu);
                                const uint8_t quant_byte = reinterpret_cast<const uint8_t &>(quant);
#endif
#if !MEGA_MOE_SKIP_EPI
#if MEGA_MOE_EPI_LOCAL
                                {
                                    const uint64_t goff =
                                        (uint64_t) pool_token_idx *
                                            fp8_intermediate_token_layout.num_bytes +
                                        intermediate_col;
                                    const uint32_t midx =
#if MEGA_MOE_EPI_HOT
                                        // UNIQUE per thread (no cross-thread collision),
                                        // reused every iter -> ~128KB L2-resident, ~0 HBM.
                                        // Fast => store is memory/scatter-traffic bound;
                                        // slow => instruction-issue / stall bound.
                                        (((sm_idx * 512u + loader_warp_local * 64u + lane_idx) * 4u) &
                                         (MEGA_MOE_EPI_LOCAL_BYTES - 1u));
#else
                                        (uint32_t)(goff & (MEGA_MOE_EPI_LOCAL_BYTES - 1u));
#endif
#if MEGA_MOE_EPI_W4
                                    // dword (4B, aligned) store: 4x fewer, no sub-dword RMW.
                                    *reinterpret_cast<uint32_t *>(
                                        &g_mega_moe_epi_scratch[midx & ~3u]) =
                                        0x01010101u * (uint32_t) quant_byte;
#else
                                    g_mega_moe_epi_scratch[midx] = quant_byte;
#endif
                                    (void) dst_row;
                                }
#elif MEGA_MOE_EPI_NT
                                __builtin_nontemporal_store(quant_byte,
                                                            &dst_row[intermediate_col]);
#elif MEGA_MOE_EPI_CONTIG_PROBE
                                // PROBE (correctness-breaking, --num-correctness-tests 0):
                                // write the fp8 to a CONTIGUOUS-by-token layout
                                // [hidden_block][token][128] so consecutive-token writes
                                // are adjacent (no 3072B token stride -> no DRAM row
                                // thrash).  L2 reads the OLD layout -> garbage y.  If this
                                // is much faster, the token-stride is the epilogue wall.
                                {
                                    const uint64_t coff =
                                        (uint64_t) n_block_idx *
                                            ((uint64_t) kNumMaxPoolTokens * 128u) +
                                        (uint64_t) pool_token_idx * 128u + n_in_wave;
                                    l2_pool_base[coff] = quant_byte;
                                    (void) dst_row;
                                }
#elif MEGA_MOE_EPI_W16
                                // PROBE (correctness-breaking): 16-byte (uint4) store per
                                // element -- the width the dispatch pull uses to write the
                                // pool FAST.  Tests if 16B is the fast-store threshold the
                                // 1/4/8B widths missed.  Overwrites 16 cols -> garbage y.
                                *reinterpret_cast<uint4 *>(
                                    &dst_row[intermediate_col & ~15u]) =
                                    make_uint4(0x01010101u * (uint32_t) quant_byte,
                                               0x01010101u * (uint32_t) quant_byte,
                                               0x01010101u * (uint32_t) quant_byte,
                                               0x01010101u * (uint32_t) quant_byte);
#else
                                dst_row[intermediate_col] = quant_byte;
#endif
#else
                                (void) dst_row;
                                (void) quant_byte;
#endif
                            }
                        }
                    }
#endif

                    // Signal L2 arrival (release-ordered): publishes the L2-pool
                    // token writes above before the consumer's gating count read.
                    if (elect_one()) {
                        auto *l2arr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
#if MEGA_MOE_EPI_BATCH_REL
                        // EXPERIMENTAL (default-off, NOT clean): relaxed increments
                        // for gate 0..22 + one release on gate-23.  +21% but cos_sim
                        // 0.99997 (small error).  The per-gate-block release
                        // GRANULARITY is itself the correctness requirement on gfx950
                        // -- batching to one +24 release FAILED gate-3 worse (0.9976),
                        // and __threadfence writeback did not help.  No clean batching
                        // exists; left as a perf-vs-tiny-error opt-in only.
                        if (n_block_idx == kL1GateNBlocks - 1u)
                            red_add_rel_gpu(l2arr, 1ull);
                        else
                            (void) atomic_add(l2arr, 1ull); // relaxed
#else
                        red_add_rel_gpu(l2arr, 1ull);
#endif
                    }
                } else {
#if MEGA_MOE_COALESCE_EPI
                    // L2 epilogue, COALESCED: the MFMA layout scatters a row's
                    // columns across the group's 16 lanes (lane owns n = nl + sub_n*16,
                    // stride 16), so the scalar path emits kSubTilesN separate 32-byte
                    // remote stores per row.  Transpose each row through a per-warp LDS
                    // scratch (reusing the loader's now-free B pool) so each lane reads
                    // 8 CONTIGUOUS columns and writes them as one uint4 -> ~256-byte
                    // coalesced XGMI bursts per row.  Same values -> gate-3 identical.
                    static_assert(kHidden % BLOCK_N == 0u,
                                  "coalesced L2 epilogue needs BLOCK_N | kHidden");
                    static_assert(BLOCK_N % 8u == 0u, "uint4 (8 bf16) must tile BLOCK_N");
                    constexpr uint32_t kEpiGroups = kWarpSize / kMfmaN; // 16 lanes/group
                    auto *xpose = reinterpret_cast<__hip_bfloat16 *>(
                        smem_buffer + kLoaderBPoolBaseBytes +
                        loader_warp_local * kEpiGroups * BLOCK_N *
                            static_cast<uint32_t>(sizeof(__hip_bfloat16)));
                    const uint32_t grp = (lane_idx >> 4u) & (kEpiGroups - 1u);
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t i = 0u; i < 4u; ++i) {
                            __hip_bfloat16 *row_lds = xpose + grp * BLOCK_N;
                            // Scatter this lane's columns into the contiguous LDS row.
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n)
                                row_lds[sub_n * kMfmaN + n_lane] =
                                    __float2bfloat16(acc[sub_m * kSubTilesN + sub_n][i]);
                            __syncthreads();
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_out_base + i;
                            if (m_in_block < valid_m) {
                                const auto meta = *workspace.get_token_src_metadata_ptr(
                                    pool_block_idx * BLOCK_M + m_in_block);
                                auto *dst_local = combine_token_buffer.get_rank_buffer(meta.topk_idx)
                                                      .get_data_buffer(meta.token_idx)
                                                      .get_base_ptr<uint8_t>();
                                auto *dst_remote = sym_buffer.map(dst_local, meta.rank_idx);
#pragma unroll
                                for (uint32_t c = n_lane * 8u; c < BLOCK_N; c += kWarpSize * 8u) {
                                    const uint32_t n_global = n_block_idx * BLOCK_N + c;
#if !MEGA_MOE_SKIP_EPI
                                    const uint4 v = *reinterpret_cast<const uint4 *>(row_lds + c);
                                    *reinterpret_cast<uint4 *>(
                                        dst_remote + n_global * sizeof(__hip_bfloat16)) = v;
#else
                                    (void) dst_remote;
                                    (void) n_global;
#endif
                                }
                            }
                            __syncthreads();
                        }
                    }
#else
                    // L2 epilogue (scalar): write each output element (BF16) into the
                    // remote combine buffer.  The source-token metadata (rank/token/
                    // topk) depends ONLY on m_in_block, so the m-row is the OUTER loop
                    // and the metadata + remote-row base are computed ONCE per row and
                    // reused across all sub_n columns.
#pragma unroll
                    for (uint32_t sub_m = 0u; sub_m < kSubTilesM; ++sub_m) {
#pragma unroll
                        for (uint32_t i = 0u; i < 4u; ++i) {
                            const uint32_t m_in_block = loader_warp_local * kRowsPerLoaderWave +
                                                        sub_m * kMfmaM + m_out_base + i;
                            if (m_in_block >= valid_m)
                                continue;
                            const auto meta = *workspace.get_token_src_metadata_ptr(
                                pool_block_idx * BLOCK_M + m_in_block);
                            auto *dst_local = combine_token_buffer.get_rank_buffer(meta.topk_idx)
                                                  .get_data_buffer(meta.token_idx)
                                                  .get_base_ptr<uint8_t>();
                            auto *dst_remote = sym_buffer.map(dst_local, meta.rank_idx);
#pragma unroll
                            for (uint32_t sub_n = 0u; sub_n < kSubTilesN; ++sub_n) {
                                const uint32_t sub       = sub_m * kSubTilesN + sub_n;
                                const uint32_t n_in_wave = sub_n * kMfmaN + n_lane;
                                const uint32_t n_global  = n_block_idx * BLOCK_N + n_in_wave;
                                if (n_global >= kHidden)
                                    continue;
#if !MEGA_MOE_SKIP_EPI && !MEGA_MOE_SKIP_EPI_L2
#if MEGA_MOE_AGPR_ACC
                                const __hip_bfloat16 bf =
                                    __float2bfloat16(read_acc_sub16<kAccUp>(sub)[i]);
#else
                                const __hip_bfloat16 bf = __float2bfloat16(acc[sub][i]);
#endif
                                *reinterpret_cast<__hip_bfloat16 *>(
                                    dst_remote + n_global * sizeof(__hip_bfloat16)) = bf;
#else
                                (void) dst_remote;
                                (void) n_global;
                                (void) sub;
#endif
                            }
                        }
                    }
#endif
                }
#else
                (void) phase;
                (void) pool_block_idx;
                (void) n_block_idx;
                (void) valid_m;
#endif
            };

            // ---- Block orchestration ----
#if MEGA_MOE_DEFER_EPI
            // Deferred-epilogue software pipeline.  Run block N's epilogue during
            // block N+1's PROLOGUE loads, overlapping the store/load latency that
            // otherwise breaks the inter-block load pipeline (~27%).
            //
            // Terminal sentinel: flush the last pending epilogue (run_epilogue is
            // in scope here, unlike after for_each_block).
            if (phase == sched::BlockPhase::None) {
#if !MEGA_MOE_LOADS_ONLY
                if (have_pending) {
                    run_epilogue(pend_phase, pend_pbi, pend_nbi, pend_vm);
                    have_pending = false;
                }
#endif
                return;
            }
            // Phase change: FLUSH a pending epilogue of a DIFFERENT phase BEFORE the
            // arrival-wait.  An L2 block's arrival-wait needs the (cross-SM) L1
            // epilogue signals; a pending un-run L1 epilogue here would deadlock.
#if !MEGA_MOE_LOADS_ONLY
            if (have_pending && pend_phase != phase) {
                run_epilogue(pend_phase, pend_pbi, pend_nbi, pend_vm);
                have_pending = false;
            }
#endif
            // Is2B skip-blocks (up-halves): no loads/compute/epilogue -> carry pending.
            if (phase == sched::BlockPhase::Linear1 && n_block_idx >= kL1GateNBlocks)
                return;
            do_arrival_wait();
            kloop_prologue(); // issue THIS block's first loads
#if !MEGA_MOE_LOADS_ONLY
            // Run the PREVIOUS same-phase block's epilogue now (reads acc, which
            // still holds its results) while this block's prologue loads are in
            // flight.  kloop_main() below resets/refills acc afterwards.
            if (have_pending) {
                run_epilogue(pend_phase, pend_pbi, pend_nbi, pend_vm);
                have_pending = false;
            }
#endif
            kloop_main();
#if !MEGA_MOE_LOADS_ONLY
            pend_phase   = phase;
            pend_pbi     = pool_block_idx;
            pend_nbi     = n_block_idx;
            pend_vm      = scheduler.template get_valid_m<false>();
            have_pending = true;
#endif
#else  // default (inline epilogue; arrival-wait already done at top)
            if (phase == sched::BlockPhase::Linear1 && n_block_idx >= kL1GateNBlocks)
                return;
            kloop_prologue();
            kloop_main();
#if !MEGA_MOE_LOADS_ONLY
            run_epilogue(phase, pool_block_idx, n_block_idx,
                         scheduler.template get_valid_m<false>());
#endif
#endif
        });

        // Cumulative recv-stats: expert_recv_count_sum is globally visible (the
        // dispatch nvlink_barrier used sync_epilogue=true); each non-zero SM folds
        // its disjoint expert subset into the host-visible stat.
        if (sm_idx != 0u && cumulative_local_expert_recv_stats != nullptr) {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                if (warp_idx == 0u && elect_one()) {
                    const uint32_t num_recv_tokens =
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                    red_add(cumulative_local_expert_recv_stats + i,
                            static_cast<int>(num_recv_tokens));
                }
            }
        }

        // Combine: top-k reduce the per-rank combine buffers into the BF16 output
        // y.  The nvlink_barrier (grid_sync + cross-rank signal) makes every L2
        // write_combine globally visible before the reduction reads it.
        comm::nvlink_barrier<kNumRanks, kNumSMs, 1, 2>(workspace, sym_buffer, kEpiLeader, sm_idx,
                                                       thread_idx, true, true);

        const uint32_t     combine_warp_local = warp_idx - kNumDispatchWarps;
        constexpr uint32_t kNumElemsPerUint4  = sizeof(uint4) / sizeof(__hip_bfloat162);
        constexpr uint32_t kNumUint4PerToken  = (kHidden * sizeof(__hip_bfloat16)) / sizeof(uint4);
        static_assert(
            (kHidden * sizeof(__hip_bfloat16)) % sizeof(uint4) == 0u,
            "hidden * sizeof(bf16) must be a multiple of uint4 for combine vectorization");

#if !MEGA_MOE_SKIP_COMBINE
        for (uint32_t token_idx = sm_idx * kNumMMANonEpilogueWarps + combine_warp_local;
             token_idx < num_tokens; token_idx += kNumSMs * kNumMMANonEpilogueWarps) {

            const int slot =
                lane_idx < kNumTopk
                    ? static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() +
                                             token_idx * kNumTopk + lane_idx))
                    : -1;
            const uint64_t mask = ballot(slot >= 0);

            for (uint32_t i = lane_idx; i < kNumUint4PerToken; i += kWarpSize) {
                const uint32_t off                        = i * sizeof(uint4);
                float2         reduced[kNumElemsPerUint4] = {};

                uint64_t remaining = mask;
                while (remaining) {
                    const uint32_t b = ffs(remaining) - 1u;
                    remaining ^= 1ull << b;
                    auto       *src_ptr = combine_token_buffer.get_rank_buffer(b)
                                              .get_data_buffer(token_idx)
                                              .get_base_ptr<uint8_t>();
                    const uint4 partial = *reinterpret_cast<const uint4 *>(src_ptr + off);
                    const auto *bf16    = reinterpret_cast<const __hip_bfloat162 *>(&partial);
#pragma unroll
                    for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l) {
                        const float2 fp32 = __bfloat1622float2(bf16[l]);
                        reduced[l].x += fp32.x;
                        reduced[l].y += fp32.y;
                    }
                }

                uint4 out;
                auto *bf16_out = reinterpret_cast<__hip_bfloat162 *>(&out);
#pragma unroll
                for (uint32_t l = 0u; l < kNumElemsPerUint4; ++l)
                    bf16_out[l] = __float22bfloat162_rn(reduced[l]);
                auto *dst =
                    reinterpret_cast<uint8_t *>(y) + token_idx * kHidden * sizeof(__hip_bfloat16);
                *reinterpret_cast<uint4 *>(dst + off) = out;
            }
        }
#else
        (void) combine_warp_local;
        (void) kNumUint4PerToken;
#endif

        // Reset workspace counters for the next launch.  All SMs are past the
        // grid_sync above, so every l1_arrival_count / l2_arrival_mask has been
        // consumed and expert_recv_count_sum read; each non-zero SM owns a
        // disjoint expert subset, so no extra rendezvous is needed.
        if (sm_idx == 0u) {
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumThreads)
                *workspace.get_expert_send_count_ptr(i) = 0u;
        } else {
            for (uint32_t i = sm_idx - 1u; i < kNumExpertsPerRank; i += kNumSMs - 1u) {
                const uint32_t num_recv_tokens =
                    static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
                const uint32_t num_recv_m_blocks       = (num_recv_tokens + BLOCK_M - 1u) / BLOCK_M;
                const uint32_t reset_pool_block_offset = scheduler.get_pool_block_offset(i);

                if (warp_idx == 0u && elect_one())
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0u;

                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0u;

                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumThreads) {
                    *workspace.get_l1_arrival_count_ptr(reset_pool_block_offset + j) = 0u;
                    *workspace.get_l2_arrival_mask_ptr(reset_pool_block_offset + j)  = 0ull;
                }
            }
        }
        return;
    }

#else
    PRIMUS_TURBO_DEVICE_CHECK(false);
#endif
}

template <MegaMoEArch kArch, uint32_t kNumMaxTokensPerRank, uint32_t kHidden,
          uint32_t kIntermediateHidden, uint32_t kNumExperts, uint32_t kNumTopk,
          uint32_t kNumExpertsPerWave, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t STORE_BLOCK_M, uint32_t SF_BLOCK_M, uint32_t SF_BLOCK_N,
          uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens, uint32_t kNumStages,
          uint32_t kNumDispatchThreads, uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumRanks,
          float kActivationClamp, bool kFastMath>
hipError_t launch_fp8_fp4_mega_moe_impl(void *y, int *cumulative_local_expert_recv_stats,
                                        const uint32_t                      num_tokens,
                                        const layout::SymBuffer<kNumRanks> &sym_buffer,
                                        const void *l1_weights, const void *l1_weights_sf,
                                        const void *l2_weights, const void *l2_weights_sf,
                                        hipStream_t stream) {
    static_assert(kArch == MegaMoEArch::Gfx950, "Only gfx950 (MI355X) is supported for now");

    constexpr uint32_t kNumThreads =
        kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;

#if MEGA_MOE_2BLK
    constexpr uint32_t kSmemBytes = 76u * 1024u; // 2-stage carve-out ~70KB; 2*76<=159KB -> 2 blocks/CU
#else
    constexpr uint32_t kSmemBytes = 140u * 1024u;
#endif

    const dim3 grid(kNumSMs);
    const dim3 block(kNumThreads);

    auto kernel = gfx950_fp8_fp4_mega_moe_kernel<
        kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk,
        kNumExpertsPerWave, BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M, SF_BLOCK_M, SF_BLOCK_N,
        kNumMaxPoolTokens, kNumPaddedSFPoolTokens, kNumStages, kNumDispatchThreads,
        kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp,
        kFastMath>;

    if constexpr (kSmemBytes > 64u * 1024u) {
        const auto attr_err = hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(kSmemBytes));
        if (attr_err != hipSuccess)
            return attr_err;
    }

#if MEGA_MOE_2BLK
    // 2 blocks/CU (grid=512 on 256 CUs): the software grid_sync requires ALL
    // grid blocks co-resident, which a normal launch does NOT guarantee at
    // grid>maxActive -> deadlock.  Cooperative launch guarantees co-residency
    // (or fails cleanly with hipErrorCooperativeLaunchTooLarge).
    void *l1w = const_cast<void *>(l1_weights), *l1wsf = const_cast<void *>(l1_weights_sf);
    void *l2w = const_cast<void *>(l2_weights), *l2wsf = const_cast<void *>(l2_weights_sf);
    void *kargs[] = {(void *) &y,
                     (void *) &cumulative_local_expert_recv_stats,
                     (void *) &num_tokens,
                     (void *) &sym_buffer,
                     (void *) &l1w,
                     (void *) &l1wsf,
                     (void *) &l2w,
                     (void *) &l2wsf};
    return hipLaunchCooperativeKernel(reinterpret_cast<const void *>(kernel), grid, block, kargs,
                                      kSmemBytes, stream);
#else
    hipLaunchKernelGGL(kernel, grid, block, kSmemBytes, stream, y,
                       cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
                       const_cast<void *>(l1_weights), const_cast<void *>(l1_weights_sf),
                       const_cast<void *>(l2_weights), const_cast<void *>(l2_weights_sf));
    return hipGetLastError();
#endif
}

} // namespace mega_moe
} // namespace primus_turbo
