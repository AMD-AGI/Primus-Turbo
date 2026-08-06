###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP8 perf/correctness for the all-fp8 (MXFP8) mega MoE, ported from the Primus-Turbo
``tests/pytorch/modules/test_mega_moe_mxfp8.py`` and organized as a ``MultiProcessTestCase``.

Uses MegaMoE's VENDORED fp8 stack (``primus_turbo.flydsl.mega.fp8``: its own SymLayout /
scoreboard / two-heap symm / dispatch prologue / GEMM / combine / quant). Ported stage by stage
to align per-stage perf against the source bench (same seed 123+rank RNG + sigmoid(randn) topk
routing + DSv3 EP8 shape), so any gap points at a real code/stack difference, not the harness.

Stages:
  * Stage 1 (this file): ``test_l1_dispatch_fc1_bench`` -- L1 = fused mxfp8 dispatch + fc1.
  # * next: L2 combine / L2 dgrad / dW2 / dW1 / L1 dgrad (added incrementally).

Also holds ``test_staged_vs_fused_parity``: the two-stage (gate-up / gate-down) fp8 op pair must
reproduce the single fused fp8 op, since it is the same kernels in the same order cut at ``l1``.

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python tests/pytorch/modules/test_mega_moe_mxfp8.py
  # or: PYTHONPATH=<repo> pytest tests/pytorch/modules/test_mega_moe_mxfp8.py -k l1 -q -s
"""

import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from primus_turbo.pytorch.core.low_precision import check_mxfp8_support
from primus_turbo.pytorch.kernels.fused_mega_moe.fused_mega_moe_weight_prep_fp8 import (
    prepare_dispatch_weight_fp8,
)

_WORLD = 8
_MXFP8_BLOCK = 32
# DeepSeek-V3 EP8 (the bench scale; the source bench methods use this real shape).
# T overridable via PT_BENCH_T (use 2048 for a fast smoke; 8192 = real DSv3).
_H, _I, _E = 7168, 2048, 256
_T = int(os.environ.get("PT_BENCH_T", "8192"))
_ITERS = int(os.environ.get("PT_BENCH_ITERS", "30"))
_BM = _BN = 256
_H_TILE_TO_EXPERT = 7


def _dequant_mxfp8(q, s_raw, block=_MXFP8_BLOCK):
    """Dequant a rowwise (along last dim) mxfp8 tensor: q(fp8) * 2^(s_raw - 127) -> fp32."""
    *lead, K = q.shape
    qf = q.float().view(*lead, K // block, block)
    scale = torch.exp2(s_raw.view(torch.uint8).float() - 127.0).unsqueeze(-1)
    return (qf * scale).view(*lead, K)


@instantiate_parametrized_tests
class TestMegaMoEMxfp8(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()

    @property
    def world_size(self) -> int:
        return _WORLD

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.device)
        torch.manual_seed(123 + self.rank)  # source RNG seed (per rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank, store=store)

    def _ep_group(self):
        return dist.new_group(list(range(self.world_size)))

    @staticmethod
    def _amax(group, v):
        t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group); return float(t)

    @staticmethod
    def _amin(group, v):
        t = torch.tensor([v], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group); return float(t)

    @staticmethod
    def _bench(fn, group, *, warmup, iters):
        """Per-call CUDA-event latency; each iter is preceded by a cross-rank sync so the ranks
        enter the launch together. Event brackets only ``fn`` -> kernel time."""
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        def _one(s=None, e=None):
            torch.cuda.synchronize(); group.barrier()
            if s is None:
                fn(); return
            s.record(); fn(); e.record()

        for _ in range(warmup):
            _one()
        for i in range(iters):
            _one(ev_s[i], ev_e[i])
        torch.cuda.synchronize()
        return float(np.average([s.elapsed_time(e) for s, e in zip(ev_s, ev_e)][1:]))

    # ───────────────────────── Stage 1: L1 (dispatch + fc1) ─────────────────────────
    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [8])
    def test_l1_dispatch_fc1_bench(self, top_k):
        """L1 = fused mxfp8 dispatch + fc1 on the vendored fp8 stack, DSv3 EP8 scale.

        Correctness: torch dequant grouped-GEMM over the kernel's OWN dispatched pool
        (``symm.pool_fp8``/``pool_scale``) -> cos/rel gate. Latency: token-quant (rowwise mxfp8,
        per-forward) + the fused dispatch+GEMM kernel."""
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.flydsl.mega.fp8 import (
            dispatch_grouped_gemm_mxfp8,
            dispatch_prologue,
            extend_handle,
            get_symm_buffer_for_mega_moe,
            quantize_rowwise_mxfp8_flydsl,
        )
        from primus_turbo.flydsl.mega.fp8 import (
            quantize_grouped_weight_mxfp8_flydsl as quantize_grouped_weight_mxfp8,
        )

        rank, dev = self.rank, self.device
        world, epr = self.world_size, _E // self.world_size
        H, I, E, T, K, BM, BN = _H, _I, _E, _T, top_k, _BM, _BN
        N = 2 * I

        # source RNG sequence: x -> w1 -> gate (seed 123+rank set in _init_process)
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        W1 = torch.randn(epr, N, H, device=dev, dtype=torch.bfloat16) * 0.05
        gate = torch.randn(T, E, device=dev)
        topk_w0, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
        topk_w = (topk_w0 / (topk_w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)

        symm = get_symm_buffer_for_mega_moe(
            group, num_experts=E, num_max_tokens_per_rank=T, num_topk=K, hidden=H,
            intermediate_hidden=I, block_m=BM, block_n=BN, use_mxfp8=True,
        )
        sym_layout = symm.make_sym_layout()
        handle = extend_handle(dispatch_prologue(
            topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
            world_size=world, rank=rank, experts_per_rank=epr, block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        ), symm)
        w1_fp8 = prepare_dispatch_weight_fp8(W1)  # static weight prep (out of the timed step)
        w1q, w1s = w1_fp8[:2]
        num_tile_blocks = symm.meta_scalars[1:2]

        def _l1():  # token quant (inside, bf16-x path) + fused dispatch + mxfp8 NT GEMM
            return dispatch_grouped_gemm_mxfp8(x, w1_fp8, handle, sym_layout, symm, BM=BM, BN=BN)

        def _quant():  # per-forward token quant alone (breakdown)
            xq, xs = quantize_rowwise_mxfp8_flydsl(x)
            return xq, xs.view(torch.float8_e8m0fnu)

        # ── correctness: one L1 step, then torch dequant grouped-GEMM over the dispatched pool ──
        torch.cuda.synchronize(); group.barrier()
        # dispatch/combine gates self-reset on device (epoch) -> no host scoreboard reset.
        out = _l1()
        torch.cuda.synchronize(); group.barrier()
        real_tiles = int(num_tile_blocks[0].item())
        M_eff = real_tiles * BM
        A = _dequant_mxfp8(symm.pool_fp8[:M_eff], symm.pool_scale[:M_eff])   # [M_eff, H]
        Wd = _dequant_mxfp8(w1q, w1s)                                        # [G, N, H]
        row_expert = handle[_H_TILE_TO_EXPERT][:real_tiles].to(torch.long).repeat_interleave(BM)
        ref = torch.empty((M_eff, N), device=dev, dtype=torch.float32)
        for gi in torch.unique(row_expert).tolist():
            m = row_expert == gi
            ref[m] = A[m] @ Wd[gi].t()
        o = out[:M_eff].float()
        cos = float(torch.dot(o.flatten(), ref.flatten()) / (o.norm() * ref.norm() + 1e-12))
        rel = float((o - ref).norm() / (ref.norm() + 1e-12))
        del A, Wd, ref, o, out  # free the large fp32 temporaries before timing (T=8192 -> ~GBs)

        # ── fp8 latency: token quant (local) + fused L1 ──
        t_quant = self._bench(_quant, group, warmup=5, iters=_ITERS)
        t_l1 = self._bench(_l1, group, warmup=5, iters=_ITERS)
        flops = 2.0 * M_eff * N * H
        m_pad = int(handle[10][-1].item())
        symm.destroy()  # free the fp8 symm before building the bf16 stack (no same-process coexistence)
        torch.cuda.synchronize(); group.barrier()

        # ── bf16 reference leg: the shared-stack bf16 L1 (dispatch + fc1, nt) on IDENTICAL inputs.
        # Use handle=None PER CALL (auto-prologue re-inits the cross-rank dispatch state each time;
        # reusing a handle back-to-back races the nt PUSH). So the bf16 number is the FULL per-forward
        # L1 (prologue + dispatch + GEMM), i.e. it INCLUDES the prologue the fp8 fused leg amortizes
        # -> the bf16 ms is a slight over-estimate; the true fp8 win is a touch larger than shown. ──
        from primus_turbo.flydsl.mega import dispatch_grouped_gemm_bf16_flydsl_kernel

        def _bf16():
            return dispatch_grouped_gemm_bf16_flydsl_kernel(
                x, W1, group, handle=None, topk_idx=topk_idx, topk_weights=topk_w, layout="nt", BM=BM, BN=BN,
            )

        t_bf16 = self._bench(_bf16, group, warmup=5, iters=_ITERS)

        cos_m, rel_m = self._amin(group, cos), self._amax(group, rel)
        quant_ms, l1_ms = self._amax(group, t_quant), self._amax(group, t_l1)
        bf16_ms = self._amax(group, t_bf16)
        if rank == 0:
            tf = lambda ms: flops / (ms * 1e-3) / 1e12
            print(f"\n{'='*72}")
            print(f"[Stage1 L1  dispatch+fc1  fp8 vs bf16]  EP{world} T={T} H={H} I={I} E={E} K={K}")
            print(f"{'='*72}")
            print(f"  token_quant  : {quant_ms:8.3f} ms  (rowwise mxfp8, per-forward)")
            print(f"  fp8 fused    : {l1_ms - quant_ms:8.3f} ms | {tf(l1_ms - quant_ms):8.1f} TFLOPS  (= L1 - quant)")
            print(f"  fp8 L1 total : {l1_ms:8.3f} ms | {tf(l1_ms):8.1f} TFLOPS  (M_eff={M_eff}, m_pad={m_pad})")
            print(f"  bf16 L1      : {bf16_ms:8.3f} ms | {tf(bf16_ms):8.1f} TFLOPS")
            print(f"  fp8/bf16     : {l1_ms / bf16_ms:.3f}x  ({'fp8 faster' if l1_ms < bf16_ms else 'fp8 SLOWER'})")
            print(f"  [acc] fp8 vs torch dequant-GEMM: cos={cos_m:.5f} rel={rel_m:.4f}  "
                  f"{'PASS' if cos_m >= 0.99 and rel_m <= 0.05 else 'FAIL'}")
        dist.destroy_process_group()
        self.assertGreaterEqual(cos_m, 0.99, f"L1 cos {cos_m:.5f} < 0.99")
        self.assertLessEqual(rel_m, 0.05, f"L1 rel {rel_m:.4f} > 0.05")

    # ──────────────── two-stage split parity: staged fp8 == fused fp8 ────────────────
    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [8])
    def test_staged_vs_fused_parity(self, top_k):
        """``fused_mega_moe_fp8_stage1/stage2`` vs the single fused ``fused_mega_moe_fp8``.

        The split only moves the autograd boundary to ``l1`` (pre-SwiGLU) -- same kernels, same
        order -- so the two must be numerically interchangeable.

        The gate is calibrated against the op's OWN run-to-run spread rather than a fixed
        tolerance, because neither mega MoE path is bit-reproducible: the combine sums each token's
        top-k contributions cross-rank in arrival order, so bf16 rounding lands differently every
        call (the bf16 op shows the same effect; at ``top_k=1``, with one contribution per token and
        no backward, the fp8 forward IS bit-exact). Measuring that floor from two fused runs and
        requiring staged-vs-fused to stay inside 2x it keeps the test sensitive to a real split bug
        -- any mis-threaded state blows past the floor by orders of magnitude -- without chasing
        reduce order. ``dW1`` / ``dW2`` / ``d(topk_weights)`` barely move run to run, so ``_ABS_TOL``
        is what actually gates them."""
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
            fused_mega_moe_fp8,
            fused_mega_moe_fp8_stage1,
            fused_mega_moe_fp8_stage2,
        )

        dev = self.device
        epr = _E // self.world_size
        H, I, E, T, K = _H, _I, _E, _T, top_k

        # source RNG sequence: x -> w1 -> w2 -> gate (seed 123+rank set in _init_process)
        x0 = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        w1_0 = torch.randn(epr, 2 * I, H, device=dev, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))
        w2_0 = torch.randn(epr, H, I, device=dev, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))
        gate = torch.randn(T, E, device=dev)
        topk_w0, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
        topk_w_0 = (topk_w0 / (topk_w0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32)
        grad_y = torch.randn(T, H, device=dev, dtype=torch.bfloat16)

        def _leaves():
            """Fresh leaves per run so the version-keyed weight-quant caches stay independent."""
            return tuple(t.detach().clone().requires_grad_(True) for t in (x0, topk_w_0, w1_0, w2_0))

        def _run(staged):
            xL, twL, w1L, w2L = _leaves()
            if staged:
                l1, dispatch_weights, handle, state = fused_mega_moe_fp8_stage1(
                    xL, topk_idx, twL, w1L, group
                )
                y = fused_mega_moe_fp8_stage2(
                    l1, dispatch_weights, handle, state, topk_idx, twL, w2L, group
                )
            else:
                y = fused_mega_moe_fp8(group, xL, topk_idx, twL, w1L, w2L)
            y.backward(grad_y)
            # Both paths drive the same process-global symm buffer, so one run must finish its
            # backward before the next forward overwrites the pool.
            torch.cuda.synchronize(); group.barrier()
            return (y.detach(), xL.grad, twL.grad, w1L.grad, w2L.grad)

        def _rel(ref, act):
            r, a = ref.float().flatten(), act.float().flatten()
            return self._amax(group, float((a - r).norm() / (r.norm() + 1e-12)))

        _run(staged=False)  # discard: the first call of the process is a cold-start outlier
        fused_a, fused_b, staged = _run(False), _run(False), _run(True)

        tags = ("y", "dx", "dtopk_weights", "dW1", "dW2")
        abs_tol, floor_slack = 1e-3, 2.0
        stats = []
        for tag, ref, other, act in zip(tags, fused_a, fused_b, staged):
            floor = _rel(ref, other)
            stats.append((tag, floor, _rel(ref, act), max(floor_slack * floor, abs_tol)))

        if self.rank == 0:
            print(f"\n{'='*72}")
            print(f"[staged vs fused fp8 parity]  EP{self.world_size} T={T} H={H} I={I} E={E} K={K}")
            print(f"{'='*72}")
            for tag, floor, cross, limit in stats:
                verdict = "PASS" if cross <= limit else "FAIL"
                print(f"  {tag:<14}: staged_vs_fused={cross:.6f}  fused_self={floor:.6f}  "
                      f"limit={limit:.6f}  {verdict}")
        dist.destroy_process_group()
        for tag, floor, cross, limit in stats:
            self.assertLessEqual(
                cross, limit,
                f"[{tag}] staged-vs-fused rel {cross:.6f} > {limit:.6f} "
                f"(fused self-noise {floor:.6f})",
            )


    @skip_if_lt_x_gpu(_WORLD)
    @parametrize("top_k", [8])
    def test_training_loop_fwd_bwd(self, top_k):
        """Drive the op the way Megatron drives it, which nothing else here does.

        Every other test and bench holds the routing fixed and/or runs forward only. A training
        step differs in three ways that all touch state the op carries between calls:

          * the router emits **different top-k every microbatch**, so the tile count, the pool
            layout and which expert owns which tile all move call to call;
          * forward and backward alternate continuously with **no cross-rank rendezvous** between
            them -- `test_staged_vs_fused_parity` inserts `synchronize() + barrier()` between runs,
            and real training has no such thing, so any epoch/flag scheme that quietly depends on
            it will pass there and hang here;
          * weights are **persistent** and only change on an optimizer step, which is what the
            `w._version`-keyed mxfp8 weight-quant cache is built around, and grads accumulate over
            the microbatches in between.

        Asserts only that every step produces finite y / dx / dW: this is a liveness and
        state-threading test, not a precision one. It prints per step so a hang says which one.
        """
        if not check_mxfp8_support():
            self.skipTest("MXFP8 requires gfx950")
        self._init_process()
        group = self._ep_group()
        from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
            fused_mega_moe_fp8_stage1,
            fused_mega_moe_fp8_stage2,
        )

        dev = self.device
        epr = _E // self.world_size
        H, I, E, T, K = _H, _I, _E, _T, top_k
        num_iters = int(os.environ.get("PT_TRAIN_ITERS", "3"))
        micro_per_iter = int(os.environ.get("PT_TRAIN_MICRO", "4"))

        # persistent bf16 parameters, as MegaMoEWeightModule holds them
        w1 = (torch.randn(epr, 2 * I, H, device=dev, dtype=torch.bfloat16) * (2.0 / math.sqrt(H))).requires_grad_(True)
        w2 = (torch.randn(epr, H, I, device=dev, dtype=torch.bfloat16) * (2.0 / math.sqrt(I))).requires_grad_(True)
        nonfinite = torch.zeros(4, dtype=torch.int64, device=dev)

        for it in range(num_iters):
            for mb in range(micro_per_iter):
                step = it * micro_per_iter + mb
                torch.manual_seed(9000 + self.rank + step * 131)
                x = torch.randn(T, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
                gate = torch.randn(T, E, device=dev)
                tw0, topk_idx = torch.sigmoid(gate).topk(K, dim=-1)
                topk_w = (tw0 / (tw0.sum(-1, keepdim=True) + 1e-20)).to(torch.float32).requires_grad_(True)
                grad_y = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
                # Tokens the loss masks out arrive with an exactly-zero gradient row -- the last
                # position of each sequence always, padding and EOD besides. Random data never
                # produces one, and a zero row is the input that makes a block-scaled quantizer
                # divide by a zero amax, so leaving it out hides a whole class of NaN.
                grad_y[T - 2 :] = 0.0
                grad_y[[551 % T, 2344 % T, 4185 % T]] = 0.0

                # mirrors MegaMoEFP8Experts.forward: w1, stage1, w2, stage2
                l1, dispatch_weights, handle, state = fused_mega_moe_fp8_stage1(
                    x, topk_idx, topk_w, w1, group
                )
                y = fused_mega_moe_fp8_stage2(
                    l1, dispatch_weights, handle, state, topk_idx, topk_w, w2, group
                )
                y.backward(grad_y)  # grads accumulate across the microbatches, as in GA

                # Accumulate the check on device: reading it here would insert a host sync every
                # step, which is itself the thing under test -- training never pauses like that.
                for slot, t in enumerate((y, x.grad, w1.grad, w2.grad)):
                    nonfinite[slot] += (~torch.isfinite(t.float())).sum()
                if self.rank == 0:
                    print(f"  [iter {it} micro {mb}] step {step} launched", flush=True)

            # optimizer step: mutating the weights bumps w._version, retiring the quant cache
            with torch.no_grad():
                for w in (w1, w2):
                    w -= 1e-4 * w.grad
                    w.grad = None

        counts = [int(v) for v in nonfinite]
        dist.destroy_process_group()
        for tag, count in zip(("y", "dx", "dW1", "dW2"), counts):
            self.assertEqual(count, 0, f"rank {self.rank}: {count} non-finite {tag} values")


if __name__ == "__main__":
    run_tests()
