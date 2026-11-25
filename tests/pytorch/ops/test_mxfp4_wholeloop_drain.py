###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MXFP4 whole-K-loop asm must retire its own LDS reads before it hands registers back (AIMA-354).

The loop's last phase still prefetches the next k-block's operands, and its phase barrier waits only down to
``lgkmcnt(elgk)``. The backend cannot see LDS traffic an opaque asm block started, so it treats those
destination VGPRs as free at the block's end; whatever it computes into them next is overwritten when a late
read lands. In the dense dglu-quant kernel that was the epilogue's row index, and a wrong tile of grad_l1 came
out whenever a memory-bound kernel ran next to it (RCCL's copy-engine ReduceScatter in training).

``test_wholeloop_asm_exit_drained`` compiles the kernels and checks the ISA: no LDS read may be in flight when
any K-loop asm block ends. ``test_dglu_quant_bitwise_under_hbm_load`` is the original reproducer (opt-in, about
a minute: PRIMUS_TURBO_STRESS_TESTS=1): without the drain it fails within seconds on MI355X.
"""

import os
import re
import subprocess
import sys
import textwrap
import time

import pytest
import torch


def _gfx950():
    try:
        return torch.cuda.is_available() and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


_SCRIPT = textwrap.dedent(
    """
    import sys, torch
    import primus_turbo.pytorch.ops.mlp_fp4  # noqa: F401
    M = sys.modules["primus_turbo.pytorch.ops.mlp_fp4"]
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import (Float4QuantConfig, ScalingGranularity, ScalingRecipe,
                                                         float4_e2m1fn_x2)
    HB = BackendType.HIPBLASLT.value  # the default backend the env override (flydsl) replaces
    which, T, H, F = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    torch.manual_seed(0)
    dev = "cuda"
    cfg = Float4QuantConfig()
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    w1 = torch.randn(2 * F, H, device=dev, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(H, F, device=dev, dtype=torch.bfloat16) * 0.02
    probs = M._probs_ones(T, dev)
    q = lambda t, sr=False: M.quantize_fp4_with_trans(
        t, float4_e2m1fn_x2, ScalingGranularity.MX_BLOCKWISE, block_size=M.MXFP4_BLOCK_SIZE,
        scaling_recipe=ScalingRecipe(use_sr=sr), scaling_recipe_for_trans=ScalingRecipe(use_sr=sr, use_rht=True),
        scale_rounding_mode=cfg.scale_rounding_mode)
    xq, gq = q(x), q(torch.randn(T, H, device=dev, dtype=torch.bfloat16) * 1e-3)
    w1q, w2q = M._quantize_weight(w1, None, cfg), M._quantize_weight(w2, None, cfg)
    glu = M.gemm_fp4_glu_quant_impl(xq[0], xq[1], w1q[0], w1q[1], probs, torch.bfloat16, False, False,
                                    cfg.scale_rounding_mode, "silu", None)
    if which == "dglu":
        M.gemm_fp4_dglu_quant_impl(gq[0], gq[1], w2q[2], w2q[3], glu[0], probs, torch.bfloat16, False, False,
                                   cfg.scale_rounding_mode, "silu", None)
    elif which == "wgrad":
        out = torch.zeros(H, F, device=dev, dtype=torch.bfloat16)
        M._bgrad_gemm_fp4_impl_wrapper(gq[2], gq[3], False, glu[3], glu[4], True, torch.bfloat16, False,
                                       granularity=cfg.granularity.value, default_backend=HB,
                                       preshuffled=False, inplace_add_to_out=True, out=out)
    elif which == "fwd":
        M.gemm_fp4_impl(glu[1], glu[2], False, w2q[0], w2q[1], True, torch.bfloat16, False,
                        granularity=cfg.granularity.value, default_backend=HB, preshuffled=False)
    torch.cuda.synchronize()
    """
)

# Per counter: every op of that kind, and the ones among them that write registers (LDS-DMA `... lds` loads
# write LDS, not registers, and stay in flight on purpose: the caller drains them before reusing the pools).
_COUNTERS = {
    "lgkmcnt": (re.compile(r"^(ds_\w+|s_load\w*|s_buffer_load\w*)\b"),
                re.compile(r"^(ds_read\w*|ds_load\w*|s_load\w*|s_buffer_load\w*)\b")),
    "vmcnt": (re.compile(r"^(buffer_\w+|global_\w+|flat_\w+)\b"),
              re.compile(r"^(buffer_load\w*|global_load\w*|flat_load\w*)\b(?!.*\blds\b)")),
}  # fmt: skip


def _register_loads_in_flight_at_exit(body, counter):
    """Register-writing loads an asm block can leave outstanding on its fall-through exit.

    A ``s_waitcnt <counter>(N)`` leaves the N most recently issued ops of that counter outstanding (they retire
    in order); every op issued after the block's last such wait is outstanding too.
    """
    any_op, reg_load = _COUNTERS[counter]
    ins = [ln.split(";")[0].strip() for ln in body]
    ins = [s for s in ins if s and not s.endswith(":")]
    last = None
    for i, s in enumerate(ins):
        m = re.match(rf"s_waitcnt\b.*\b{counter}\((\d+)\)", s)
        if m:
            last = (i, int(m.group(1)))
    ops = [i for i, s in enumerate(ins) if any_op.match(s)]
    if last is not None:
        i, n = last
        ops = ([x for x in ops if x < i][-n:] if n else []) + [x for x in ops if x > i]
    return sum(1 for x in ops if reg_load.match(ins[x]))


@pytest.mark.skipif(not _gfx950(), reason="MXFP4 FlyDSL GEMMs need gfx950")
@pytest.mark.parametrize("which", ["dglu", "wgrad", "fwd"])
def test_wholeloop_asm_exit_drained(tmp_path, which):
    """No K-loop asm block of the dense MXFP4 kernels ends with a register-writing load in flight."""
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=str(tmp_path), FLYDSL_RUNTIME_ENABLE_CACHE="0",
               PRIMUS_TURBO_GEMM_BACKEND="flydsl")  # fmt: skip
    # K = 4096 (16 k-blocks) keeps the full loop + peel structure; M, N one or two tiles.
    r = subprocess.run([sys.executable, "-c", _SCRIPT, which, "512", "4096", "512"], env=env, capture_output=True,
                       text=True, timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stderr[-4000:]
    isa = sorted(tmp_path.rglob("*final_isa.s"))
    assert isa, f"no ISA dumped under {tmp_path}"
    blocks = 0
    for f in isa:
        lines = f.read_text().splitlines()
        i = 0
        while i < len(lines):
            if ";;#ASMSTART" in lines[i]:
                j = next(k for k in range(i + 1, len(lines)) if ";;#ASMEND" in lines[k])
                body = lines[i + 1 : j]
                if any("v_mfma_scale_f32_16x16x128_f8f6f4" in ln for ln in body):  # an MXFP4 K-loop
                    blocks += 1
                    for counter, what in (("lgkmcnt", "LDS reads"), ("vmcnt", "register loads")):
                        left = _register_loads_in_flight_at_exit(body, counter)
                        assert left == 0, (
                            f"{f}: K-loop asm at line {i + 1} exits with {left} {what} in flight"
                        )
                i = j
            i += 1
    assert blocks, "no MXFP4 K-loop asm block found in the dumped ISA"


@pytest.mark.skipif(not _gfx950(), reason="MXFP4 FlyDSL GEMMs need gfx950")
@pytest.mark.skipif(
    os.environ.get("PRIMUS_TURBO_STRESS_TESTS") != "1", reason="opt-in: PRIMUS_TURBO_STRESS_TESTS=1"
)
def test_dglu_quant_bitwise_under_hbm_load():
    """The dglu-quant kernel is bitwise deterministic while a memory-bound kernel runs on another stream."""
    import primus_turbo.pytorch.ops.mlp_fp4  # noqa: F401

    M = sys.modules["primus_turbo.pytorch.ops.mlp_fp4"]
    from primus_turbo.pytorch.core.low_precision import (
        Float4QuantConfig,
        ScalingGranularity,
        ScalingRecipe,
        float4_e2m1fn_x2,
    )

    T, H, F = 32768, 4096, 14336  # llama3.1-8B MLPerf, where it was found
    torch.manual_seed(0)
    cfg = Float4QuantConfig()
    q = lambda t: M.quantize_fp4_with_trans(  # noqa: E731
        t, float4_e2m1fn_x2, ScalingGranularity.MX_BLOCKWISE, block_size=M.MXFP4_BLOCK_SIZE,
        scaling_recipe=ScalingRecipe(), scaling_recipe_for_trans=ScalingRecipe(use_rht=True),
        scale_rounding_mode=cfg.scale_rounding_mode)  # fmt: skip
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(2 * F, H, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(H, F, device="cuda", dtype=torch.bfloat16) * 0.02
    probs = M._probs_ones(T, "cuda")
    xq, gq = q(x), q(torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 1e-3)
    w1q, w2q = M._quantize_weight(w1, None, cfg), M._quantize_weight(w2, None, cfg)
    l1 = M.gemm_fp4_glu_quant_impl(xq[0], xq[1], w1q[0], w1q[1], probs, torch.bfloat16, False, False,
                                   cfg.scale_rounding_mode, "silu", None)[0]  # fmt: skip
    run = lambda: M.gemm_fp4_dglu_quant_impl(gq[0], gq[1], w2q[2], w2q[3], l1, probs, torch.bfloat16,  # noqa: E731
                                             False, False, cfg.scale_rounding_mode, "silu", None)  # fmt: skip
    ref = [t.clone() for t in run()]
    side = torch.cuda.Stream()
    a = torch.empty(1 << 28, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    bad, n, t0 = 0, 0, time.time()
    while time.time() - t0 < float(os.environ.get("PRIMUS_TURBO_STRESS_SECONDS", "60")):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(32):
                b.add_(a)
        out = run()
        bad += sum(not torch.equal(o.view(torch.uint8), r.view(torch.uint8)) for o, r in zip(out, ref))
        n += 1
    torch.cuda.synchronize()
    assert bad == 0, f"{bad} corrupted outputs in {n} dglu-quant calls under concurrent HBM load"
