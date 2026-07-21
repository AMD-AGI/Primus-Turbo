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
  # * next: L2 combine / STEP1 / dW2 / dW1 / STEP3 (added incrementally).

Run inside the dev container (8 GPUs):
  PYTHONPATH=<repo> python tests/pytorch/modules/test_mega_moe_mxfp8.py
  # or: PYTHONPATH=<repo> pytest tests/pytorch/modules/test_mega_moe_mxfp8.py -k l1 -q -s
"""

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
    def _bench(fn, group, *, warmup, iters, reset=None):
        """Per-call CUDA-event latency; ``reset`` (e.g. scoreboard.zero_) runs OUTSIDE the timed
        window each iter (L1 needs a cross-rank scoreboard reset between launches, so it cannot be
        timed strictly back-to-back). Event brackets only ``fn`` -> kernel time."""
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

        def _one(s=None, e=None):
            torch.cuda.synchronize(); group.barrier()
            if reset is not None:
                reset()
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
            get_symm_buffer_for_mega_moe,
            quantize_grouped_weight_mxfp8,
            quantize_rowwise_mxfp8_flydsl,
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
        handle = tuple(dispatch_prologue(
            topk_idx, topk_w, sym_layout=sym_layout, num_tokens=T, num_topk=K, num_experts=E,
            world_size=world, rank=rank, experts_per_rank=epr, block_m=BM,
            num_max_pool_tokens=symm.num_max_pool_tokens,
        ))
        w1q, w1s = quantize_grouped_weight_mxfp8(W1)  # static weight quant (out of the timed step)
        num_tile_blocks = symm.meta_scalars[1:2]

        def _l1():  # token quant (inside, bf16-x path) + fused dispatch + mxfp8 NT GEMM
            return dispatch_grouped_gemm_mxfp8(x, None, w1q, w1s, handle, sym_layout, symm, BM=BM, BN=BN)

        def _quant():  # per-forward token quant alone (breakdown)
            xq, xs = quantize_rowwise_mxfp8_flydsl(x)
            return xq, xs.view(torch.float8_e8m0fnu)

        # ── correctness: one L1 step, then torch dequant grouped-GEMM over the dispatched pool ──
        torch.cuda.synchronize(); group.barrier()
        symm.scoreboard.zero_()
        torch.cuda.synchronize(); group.barrier()
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

        # ── fp8 latency: token quant (local) + fused L1 (scoreboard reset outside timed window) ──
        t_quant = self._bench(_quant, group, warmup=5, iters=_ITERS)
        t_l1 = self._bench(_l1, group, warmup=5, iters=_ITERS, reset=lambda: symm.scoreboard.zero_())
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


if __name__ == "__main__":
    run_tests()
