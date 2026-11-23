###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Guards of the fp8_block MegaMoE operators (EP=1 and EP>1) on gfx1250.

Host-only (no GPU): which wgrad kernel is picked for which expert-window bound, that the EP operator takes the whole
vkq family (persistent wgrad, compact column scales, LPT order) and not just "vkp", the vk window bound and the
combine column block. GPU (gfx1250): vk with an expert window longer than the old 4096-token default and the combine
kernel at a hidden size that is not a multiple of 1024.

Listed in ``tests/conftest.py``'s ``_GFX1250_ENABLED_TESTS``, without which it never runs on gfx1250.
"""

import inspect

import pytest
import torch

pytest.importorskip("flydsl", reason="the fp8_block MegaMoE kernels live behind flydsl")
pytest.importorskip("triton")

import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk as FM  # noqa: E402
import primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk_ep as EP  # noqa: E402
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_vk_kernel import flyvkblk_fp8  # noqa: E402
from primus_turbo.triton.quantization import quantization_fp8blk_moe_kernel as QB  # noqa: E402


def _have_gfx1250() -> bool:
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        return bool(torch.cuda.is_available() and is_gfx1250())
    except Exception:
        return False


needs_gfx1250 = pytest.mark.skipif(not _have_gfx1250(), reason="needs a gfx1250 (MI455X) GPU")


# ------------------------------------------------------------------------------------------------ host only
@pytest.mark.parametrize(
    "window, expected",
    [(1, 4096), (4096, 4096), (4097, 8192), (5376, 8192), (8192, 8192), (20000, 8192)],
)
def test_vk_kwin_is_a_power_of_two_capped_at_the_lds_limit(window, expected):
    assert FM._vk_kwin(window) == expected


@pytest.mark.parametrize(
    "mode, T, expected",
    [
        ("vkq", 4096, "vkq"),
        ("vkq", 5120, "vkq"),
        ("vkq", 5121, "vk"),  # window bound 5376 > the vkq LDS arena
        ("vkq", 32768, "vk"),
        ("vk", 1024, "vk"),
    ],
)
def test_wgrad_kernel_selection(monkeypatch, mode, T, expected):
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", mode)
    assert FM._wgrad_kernel(T) == expected


def test_vk_kwin_max_is_required_and_validated():
    assert inspect.signature(flyvkblk_fp8).parameters["kwin_max"].default is inspect.Parameter.empty
    out = torch.empty((1, 256, 256), dtype=torch.bfloat16)
    for bad in (64, 3000, 6144):
        with pytest.raises(ValueError, match="power of two"):
            flyvkblk_fp8(None, None, None, None, None, 256, 256, out, kwin_max=bad)


@pytest.mark.parametrize(
    "H, expected", [(4096, 1024), (1024, 1024), (1536, 512), (2304, 256), (768, 256), (384, 128)]
)
def test_combine_column_block_divides_h(H, expected):
    bh = QB._column_block(H, 1024)
    assert bh == expected and H % bh == 0


def test_combine_column_block_rejects_h_not_multiple_of_128():
    with pytest.raises(ValueError, match="multiple of 128"):
        QB._column_block(192, 1024)


@pytest.mark.parametrize(
    "mode, T_rank, R, vkq", [("vkq", 1024, 2, True), ("vk", 1024, 2, False), ("vkq", 4096, 2, False)]
)
def test_ep_state_takes_the_vkq_family(monkeypatch, mode, T_rank, R, vkq):
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", mode)
    st = EP._EPState(EP.EmulatedEPComm(R), E_l=8, K=4, H=1024, I=512, T=T_rank, dispatch="fp8")
    assert (st.wk == "vkq") == vkq
    assert st.ct == vkq  # compact column scales go with the persistent wgrad


@pytest.mark.parametrize("mode, T_rank", [("vkq", 1024), ("vk", 1024)])
def test_ep_wgrad_runs_the_kernel_of_the_state(monkeypatch, mode, T_rank):
    monkeypatch.setenv("PRIMUS_TURBO_MOE_FP8BLK_WGRAD", mode)
    st = EP._EPState(EP.EmulatedEPComm(2), E_l=2, K=2, H=1024, I=512, T=T_rank, dispatch="fp8")
    rk = EP._Rank()
    rk.poffs, rk.perm = torch.zeros(3, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    seen = {}

    def fake_run_wgrad(wk, lhs, rhs, lhs_s, rhs_s, poffs, n, k, out, beta, perm, window, sq=None):
        seen.update(wk=wk, window=window, perm=perm)

    monkeypatch.setattr(FM, "_run_wgrad", fake_run_wgrad)
    w = torch.empty((2, 1024, 512), dtype=torch.bfloat16)
    EP._wgrad(st, rk, w, None, None, None, None, 1024, 512, accumulate=False)
    assert seen["wk"] == mode and seen["window"] == st.N


# ------------------------------------------------------------------------------------------------ GPU
def _dequant(q, s):
    f = torch.pow(2.0, (s.to(torch.int32) - 127).float()).repeat_interleave(QB.MX, dim=-1)
    return q.float() * f


@needs_gfx1250
def test_vk_wgrad_with_an_expert_window_above_4096_tokens():
    """One expert holds 6144 tokens (> the old default kwin_max of 4096): vk with kwin_max from the window bound
    matches the dequantized fp32 reference."""
    dev = "cuda"
    torch.manual_seed(0)
    C, windows = 256, (6144, 256)
    P = sum(windows)
    a = torch.randn(P, C, device=dev, dtype=torch.bfloat16)
    b = torch.randn(P, C, device=dev, dtype=torch.bfloat16)
    _, _, qa, sa = QB.quant_dual_blk(a, P)
    _, _, qb, sb = QB.quant_dual_blk(b, P)
    poffs = torch.tensor([0, windows[0], P], dtype=torch.int32, device=dev)
    out = torch.empty((2, C, C), device=dev, dtype=torch.bfloat16)
    flyvkblk_fp8(qa, qb, sa, sb, poffs, C, C, out, beta=0.0, kwin_max=FM._vk_kwin(FM._vkp_kwin(max(windows))))
    da, db = _dequant(qa, sa), _dequant(qb, sb)
    lo = 0
    for g, win in enumerate(windows):
        ref = da[:, lo : lo + win].double() @ db[:, lo : lo + win].double().T
        err = (out[g].double() - ref).norm() / ref.norm()
        assert err < 1e-2, (g, win, float(err))
        lo += win


@needs_gfx1250
@pytest.mark.parametrize("H", [1536, 2304])
def test_combine_rows_writes_every_column(H):
    """H not a multiple of 1024: the tail columns used to stay unwritten."""
    dev = "cuda"
    torch.manual_seed(1)
    T, K = 48, 2
    y = torch.randn(T * K + 64, H, device=dev, dtype=torch.bfloat16)
    inv = torch.randperm(y.shape[0], device=dev)[: T * K].to(torch.int32)
    out = QB.combine_rows(y, inv, T, K)
    ref = y[inv.long()].view(T, K, H).float().sum(1).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@needs_gfx1250
@pytest.mark.parametrize("k", [384, 1280])
def test_flygg_rejects_a_k_that_is_not_a_power_of_two(k):
    """K / 32 is the TDM pad interval of the scale planes: other K used to abort the process at compile time."""
    from primus_turbo.flydsl.grouped_gemm.grouped_gemm_fp8blk_nt_kernel import flygg_fp8blk_nt

    m, n, g = 256, 256, 1
    fp8 = torch.float8_e4m3fn
    a = torch.zeros(m, k, device="cuda", dtype=fp8)
    b = torch.zeros(g, n, k, device="cuda", dtype=fp8)
    sa = torch.zeros(m, k // 32, device="cuda", dtype=torch.uint8)
    sb = torch.zeros(g, n, k // 32, device="cuda", dtype=torch.uint8)
    offs = torch.tensor([0, m], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="power of two"):
        flygg_fp8blk_nt(a, b, sa, sb, offs, tile=(256, 256, 128 if k % 256 else 256))
