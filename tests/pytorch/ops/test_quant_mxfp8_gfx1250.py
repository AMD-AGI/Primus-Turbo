###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 MXFP8 quantization kernels (primus_turbo/flydsl/quantization/mxfp8_quant_gfx1250.py) against the recipe
reference tests/pytorch/ref/mxfp8_ref.py, bit for bit (reference on the CPU: IEEE denormals, no device pow). Each
case compiles its own kernel (widths, expert counts, layouts and producers are compile-time), so every feature
appears in a few cases rather than in a full sweep: K / M tails (K % 128, K % 8, M % 128 / 32 / 16), 2-D blocks,
every operand layout, each producer (dense, grouped, GLU / dGLU, gather, MX-row gather), single copies, empty
experts, and inputs that end where their mapping does. The dispatch to these kernels, the layout's inverse and
the operands' storage slack are checked on the host."""

import contextlib
import ctypes

import pytest
import torch

from primus_turbo.pytorch.core.low_precision import ScalingRecipe
from primus_turbo.pytorch.core.utils import is_gfx1250
from primus_turbo.pytorch.kernels.quantization import quantization_impl as QI
from tests.pytorch.ref import mxfp8_ref as R
from tests.pytorch.ref.mxfp8_ref import assert_bytes_equal as _eq
from tests.pytorch.ref.mxfp8_ref import mx_input as _x

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
if is_gfx1250():
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q

# The kernels' layout bits (SHUF_ROW, SHUF_COL = 4, 8: that copy's codes in 16x16 blocks) by the copies' roles in
# the MX GEMM: a gradient's copies are A operands, an activation's column copy is the wgrad's B, a weight's copies
# are both Bs.
GRAD, ACT, WGT = 0, 8, 12
NAMES = ("q_row", "s_row", "q_col", "s_col")


def _flags(lay):
    return dict(pack_row=True, pack_col=True, shuf_row=bool(lay & 4), shuf_col=bool(lay & 8))


def _c128(n):
    return -(-n // 128) * 128


def _offs(lens):
    return torch.nn.functional.pad(torch.cumsum(lens, 0), (1, 0))


def _dense_ref(x, lay, blk_2d=False):
    """The reference copies of x [M, K] as the kernels pad and lay them out: each copy's last dim to 128, which is
    the quantization of the zero-padded x (zero codes and byte-0 scales; NaN codes in the padding of a NaN block)."""
    M, K = x.shape
    xp = torch.nn.functional.pad(x.cpu(), (0, _c128(K) - K, 0, _c128(M) - M))
    qr, sr, qc, sc = (R.quant_2d if blk_2d else R.quant_dual)(xp)
    return R.to_layout((qr[:M], sr[:M], qc[:K], sc[:K]), **_flags(lay))


def _written(out, lay):
    """The part of a grouped 8-tuple a call writes: the rows / tokens up to the padded totals offs_row[G] /
    offs_col[G] (16x16 column codes undone) and the lens / offsets, not the rest of the bounds-sized copies."""
    qr, sr, qc, sc, *lo = out
    Mr, Mc = int(lo[1][-1]), int(lo[3][-1])
    if qr is not None:
        qr, sr = qr[:Mr], sr[: Mr // 32]
    if qc is not None:
        qc, sc = (R.unpreshuffle_b16(qc) if lay & 8 else qc)[:, :Mc], sc[:, : Mc // 128]
    return (qr, sr, qc, sc), lo


def _same(got, ref, lay, what):
    """Two grouped 8-tuples hold the same written bytes; a copy ``got`` leaves out (None) is not compared."""
    (gc, glo), (rc, rlo) = _written(got, lay), _written(ref, lay)
    for a, b, name in zip(gc, rc, NAMES):
        if a is not None:
            _eq(a, b, f"{what}: {name}")
    for a, b in zip(glo, rlo):
        assert torch.equal(a, b), f"{what}: lens / offsets"


# K: 1 / 33 / 300 (K % 8 != 0: the input gets storage past its end), 200 / 576 (K % 128 != 0: the last row's partial
# k-tile by masked loads), 128 / 7168; M: 1, 3, 100, 128, 130, 160, 257, 300 (M % 128 / 32 / 16).
@gpu
@pytest.mark.parametrize(
    "M,K,lay",
    [
        (1, 1, GRAD),
        (3, 200, GRAD),
        (100, 33, GRAD),
        (128, 128, WGT),
        (130, 300, GRAD),
        (160, 336, WGT),
        (257, 576, ACT),
        (300, 7168, ACT),
    ],
)
def test_dense(M, K, lay):
    x = _x(M, K, seed=M * 7 + K)
    for got, ref, what in zip(Q.quant_mxfp8(x, layout=lay), _dense_ref(x, lay), NAMES):
        _eq(got, ref, what)


@gpu
@pytest.mark.parametrize("B,N,K", [(1, 1, 1), (3, 64, 96), (2, 129, 257), (1, 304, 7168)])
def test_2d(B, N, K):
    """Weights: one scale per 32 x 32 block, shared by both copies; a batch of them in one launch."""
    w = _x(B, N, K, seed=B + N + K)
    lay = WGT if N % 16 == K % 16 == 0 else GRAD
    out = Q.quant_mxfp8(w, blk_2d=True, layout=lay)
    for b in range(B):
        for got, ref, what in zip(out, _dense_ref(w[b], lay, blk_2d=True), NAMES):
            _eq(got[b], ref, f"{what}[{b}]")
    if B == 1:
        for got, ref, what in zip(Q.quant_mxfp8(w[0], blk_2d=True, layout=lay), out, NAMES):
            _eq(got, ref[0], f"2-D input: {what}")


@gpu
@pytest.mark.parametrize(
    "M,K,lay,blk_2d", [(257, 200, GRAD, False), (130, 576, ACT, False), (160, 336, WGT, True)]
)
def test_single_copy(M, K, lay, blk_2d):
    """``rows`` / ``cols`` False: the other copy of the dual cast, bit for bit, alone."""
    x = _x(M, K, seed=M + K)
    both = Q.quant_mxfp8(x, blk_2d=blk_2d, layout=lay)
    rows = Q.quant_mxfp8(x, blk_2d=blk_2d, cols=False, layout=lay)
    cols = Q.quant_mxfp8(x, blk_2d=blk_2d, rows=False, layout=lay)
    assert rows[2:] == (None, None) and cols[:2] == (None, None)
    for got, ref, what in zip(rows[:2] + cols[2:], both, NAMES):
        _eq(got, ref, what)


@gpu
def test_special_blocks():
    """Hand-made blocks: zero, denormal, NaN, +-Inf, exact powers, the rule's boundary."""
    vals = [0.0, 2.0**-133, -(2.0**-130), 2.0**-126, 448.0, -448.0, 449.0, 1.9375 * 2**20, -1.875 * 2**-90,
            float("inf"), -float("inf"), float("nan"), 3.0e38, -3.0e38, 1.0, -0.0]  # fmt: skip
    x = torch.zeros(64, 64)
    for i, v in enumerate(vals):
        x[i, :32] = v * torch.linspace(-1, 1, 32) if i % 2 else v  # whole-block value or a ramp ending in it
        x[i, 32 + i] = v
        x[16 + i, 7] = v
    x = x.to(torch.bfloat16).cuda()
    for got, ref, what in zip(Q.quant_mxfp8(x), _dense_ref(x, GRAD), NAMES):
        _eq(got, ref, what)


@gpu
@pytest.mark.parametrize("M,K", [(130, 512), (3, 200)])
def test_wire_rows(M, K):
    """``quant_mxfp8_rows``, MegaMoE's wire format: row-major codes and raw scales, K padded to 128."""
    x = _x(M, K, seed=M + K)
    ref = R.quant_rows(torch.nn.functional.pad(x.cpu(), (0, _c128(K) - K)))
    for got, exp, what in zip(Q.quant_mxfp8_rows(x), ref, NAMES):
        _eq(got, exp, what)


def _lens(case, seed):
    if case in ("random", "many"):  # many: 70 experts, the lookup's multi-chunk path
        g = torch.Generator().manual_seed(seed)
        n, hi = (13, 300) if case == "random" else (70, 60)
        lens = torch.randint(0, hi, (n,), generator=g)
        lens[torch.randint(0, n, (n // 4,), generator=g)] = 0
        return lens
    if case == "skew":  # one hot expert, empties, a single token
        return torch.tensor([1500, 0, 1, 300, 0, 0, 77, 640, 0, 2, 129, 0, 31, 0, 0, 256])
    return torch.tensor(case)


RAGGED = [31, 32, 33, 1, 0, 127, 128, 129]


# (Inf / NaN inputs only for K % 32 == 0: in a partial 32-block the K padding of a NaN block holds NaN codes, which
# the reference, quantizing at K, does not model.)
@gpu
@pytest.mark.parametrize(
    "case,K,lay",
    [
        ([1], 64, ACT),
        ([0, 5, 0], 200, GRAD),
        (RAGGED, 2048, ACT),
        ([256, 256, 256, 256], 200, GRAD),
        ([0, 0, 300], 64, GRAD),
        ([0, 0, 0], 200, GRAD),  # no tokens: only the lens / offsets are written
        ("random", 2048, GRAD),
        ("many", 64, ACT),
    ],
)
def test_grouped(case, K, lay):
    lens = _lens(case, K)
    M = int(lens.sum())
    x = _x(M, K, seed=K + M, special=M > 0 and K % 32 == 0)
    lens_d = lens.cuda()
    out = Q.grouped_quant_mxfp8(x, lens_d, _offs(lens_d), layout=lay)
    R.check_grouped(out, x, lens, **_flags(lay))
    if case == RAGGED:  # one copy alone
        rows = Q.grouped_quant_mxfp8(x, lens_d, _offs(lens_d), cols=False, layout=lay)
        cols = Q.grouped_quant_mxfp8(x, lens_d, _offs(lens_d), rows=False, layout=lay)
        assert rows[2:4] == (None, None) and cols[:2] == (None, None)
        _same(rows, out, lay, "rows only")
        _same(cols, out, lay, "columns only")


# ---- fused producers: GLU / dGLU (gfx950 phase-2 math), gather (permute), gather of MX rows
def _ulp_close(got, ref, what):
    """bf16 within one ulp of the torch math, and nearly all exactly equal. The kernels evaluate the same fp32
    formula with the device's exp2 and division, whose last bits may differ from the host's; a few fp32 ulps move
    the bf16 rounding only for values next to a rounding boundary, by one bf16 ulp. The 99.5 % floor is empirical
    (every case here is above it): it catches a systematic difference, not a derived bound."""
    a, b = got.cpu().view(torch.int16).int(), ref.cpu().view(torch.int16).int()
    near = ((a - b).abs() <= 1) | ((got.cpu().float() - ref.cpu().float()).abs() <= 1e-35)
    assert near.all(), f"{what}: {int((~near).sum())} values more than an ulp off"
    assert (a == b).float().mean() > 0.995, f"{what}: only {float((a == b).float().mean()):.4f} exact"


GLU_CASES = [
    ([0, 5, 0], 128, "silu", None),
    (RAGGED, 2048, "silu", 7.0),
    ("random", 256, "gelu", 7.0),
    ("skew", 2048, "gelu", None),
]


def _glu_inputs(case, I, seed):
    lens = _lens(case, seed)
    M = int(lens.sum())
    g = torch.Generator(device="cuda").manual_seed(M + seed)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    probs[::17] = 0
    return lens, lens.cuda(), _offs(lens.cuda()), l1, dact, probs


@gpu
@pytest.mark.parametrize("case,I,act,clamp", GLU_CASES)
def test_glu(case, I, act, clamp):
    """The activation (taken out by ``act_out``) is the torch math's within an ulp, and what the kernel made of it
    is its quantization bit for bit; without ``act_out`` (the kernel training runs) the outputs are the same."""
    lens, lens_d, offs, l1, _, probs = _glu_inputs(case, I, I)
    act_v = torch.full((l1.shape[0], I), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, act, clamp, act_out=act_v, layout=ACT)
    _ulp_close(act_v, R.glu_ref(l1.cpu(), probs.cpu(), act, clamp), "act")
    R.check_grouped(out, act_v, lens, **_flags(ACT))
    _same(Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, act, clamp, layout=ACT), out, ACT, "no act_out")
    if case == "skew":  # MegaMoE's forward when no backward follows: no column copy
        rows = Q.grouped_glu_quant_mxfp8(l1, probs, lens_d, offs, act, clamp, cols=False)
        assert rows[2:4] == (None, None)
        _same(rows, out, ACT, "rows only")


# the gradient's column copy is an A operand (GRAD); once as a B operand, for the 16x16 stores of both halves
@gpu
@pytest.mark.parametrize("case,I,act,clamp", GLU_CASES)
def test_dglu(case, I, act, clamp):
    lens, lens_d, offs, l1, dact, probs = _glu_inputs(case, I, I + 1)
    lay = ACT if case == "random" else GRAD
    grad_v = torch.full((l1.shape[0], 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gp, *out = Q.grouped_dglu_quant_mxfp8(
        dact, l1, probs, lens_d, offs, act, clamp, grad_out=grad_v, layout=lay
    )
    ref_grad, ref_gp = R.dglu_ref(dact.cpu(), l1.cpu(), probs.cpu(), act, clamp)
    _ulp_close(grad_v, ref_grad, "grad_l1")
    torch.testing.assert_close(gp.cpu(), ref_gp, rtol=1e-4, atol=1e-4 * float(ref_gp.abs().max() + 1))
    R.check_grouped(out, grad_v, lens, **_flags(lay))
    gp2, *plain = Q.grouped_dglu_quant_mxfp8(dact, l1, probs, lens_d, offs, act, clamp, layout=lay)
    _same(plain, out, lay, "no grad_out")
    assert torch.equal(gp2, gp), "no grad_out: grad_probs"


@gpu
@pytest.mark.parametrize(
    "case,H,with_probs,lay",
    [
        ([0, 5, 0], 64, True, ACT),
        (RAGGED, 200, False, GRAD),
        ("random", 7168, True, ACT),
        ("skew", 7168, False, GRAD),
    ],
)
def test_gather(case, H, with_probs, lay):
    lens = _lens(case, H)
    M = int(lens.sum())
    T = max(1, M // 3)
    g = torch.Generator(device="cuda").manual_seed(M + H)
    x = _x(T, H, seed=H, special=H % 32 == 0)
    d2s = torch.randint(0, T, (M,), generator=g, device="cuda", dtype=torch.int32)
    if M > 4:
        d2s[1::9], d2s[3] = -1, T + 5  # rows of no token
    pp = torch.rand(M, generator=g, device="cuda") if with_probs else None
    if pp is not None:
        pp[::7] = 0
    lens_d = lens.cuda()
    out = Q.grouped_gather_quant_mxfp8(x, d2s, lens_d, _offs(lens_d), pp, layout=lay)
    vals = R.gather_rows(x.cpu(), d2s.cpu(), None if pp is None else pp.cpu())
    R.check_grouped(out, vals, lens, **_flags(lay))


# MegaMoE's pool gather of dispatched MX rows: dy (both copies A operands), x (the column copy a B operand), and the
# rows alone
@gpu
@pytest.mark.parametrize("case,H,lay", [([0, 5, 0], 512, GRAD), (RAGGED, 7168, GRAD), ("skew", 7168, ACT)])
def test_gather_fp8(case, H, lay):
    """Codes and scales kept by the row copy, the columns re-quantized from the exact dequantized values."""
    lens = _lens(case, H)
    M = int(lens.sum())
    T = max(1, M // 3)
    x = _x(T, H, seed=H)  # denormal rows too: scale byte 0 with non-zero codes
    q, s = R.quant_rows(x.cpu())
    g = torch.Generator(device="cuda").manual_seed(M + H)
    d2s = torch.randint(-1, T, (M,), generator=g, device="cuda", dtype=torch.int32)
    if M > 4:
        d2s[3] = T + 5
    lens_d = lens.cuda()
    out = Q.grouped_gather_fp8_quant_mxfp8(q.cuda(), s.cuda(), d2s, lens_d, _offs(lens_d), layout=lay)
    R.check_grouped_ref(out, R.quant_grouped_fp8(q, s, d2s.cpu(), lens), lens, **_flags(lay))
    if case == "skew":
        rows = Q.grouped_gather_fp8_quant_mxfp8(q.cuda(), s.cuda(), d2s, lens_d, _offs(lens_d), cols=False)
        assert rows[2:4] == (None, None)
        _same(rows, out, lay, "rows only")


class _HipMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", _HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_uint32),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _HipMemLocation), ("flags", ctypes.c_int)]


class _CudaArray:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


@contextlib.contextmanager
def tensor_at_mapping_end(shape, dtype):
    """A CUDA tensor whose last byte is the last mapped byte of its virtual address range: HIP's virtual memory API
    reserves a page more than it maps, so a kernel that reads past the tensor faults instead of reading a neighbour.
    A regression therefore shows as a GPU page fault that ends the pytest process, not as a failed assertion.
    Skips the test where the API is not available. Unmapped on exit, after a device synchronize; the tensor must
    not be used afterwards."""
    torch.zeros(1, device="cuda")  # the runtime is up, so its library is loaded
    if not torch.version.hip:
        pytest.skip("HIP virtual memory API")
    with open("/proc/self/maps") as maps:
        path = next((line.split()[-1] for line in maps if "libamdhip64" in line), None)
    if path is None:
        pytest.skip("libamdhip64 not loaded")
    hip = ctypes.CDLL(path)
    loc = _HipMemLocation(type=1, id=torch.cuda.current_device())  # hipMemLocationTypeDevice
    prop = _HipMemAllocationProp(type=1, requestedHandleType=0, location=loc)  # pinned, no export handle
    gran = ctypes.c_size_t()
    if hip.hipMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop), 0):
        pytest.skip("hipMemGetAllocationGranularity failed")
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    page = gran.value
    size = max(1, -(-nbytes // page)) * page
    va, handle = ctypes.c_void_p(), ctypes.c_void_p()
    no_flags = ctypes.c_ulonglong(0)
    if hip.hipMemAddressReserve(
        ctypes.byref(va), ctypes.c_size_t(size + page), ctypes.c_size_t(page), None, no_flags
    ):
        pytest.skip("hipMemAddressReserve failed")
    mapped = False
    try:
        if hip.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), no_flags):
            pytest.skip("hipMemCreate failed")
        if hip.hipMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), handle, no_flags):
            pytest.skip("hipMemMap failed")
        mapped = True
        access = _HipMemAccessDesc(location=loc, flags=3)  # read-write
        if hip.hipMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(access), ctypes.c_size_t(1)):
            pytest.skip("hipMemSetAccess failed")
        raw = torch.as_tensor(_CudaArray(va.value + size - nbytes, nbytes), device="cuda")
        yield raw.view(dtype).view(shape)
    finally:
        torch.cuda.synchronize()  # nothing may still use the range
        if mapped:
            hip.hipMemUnmap(va, ctypes.c_size_t(size))
        if handle.value:
            hip.hipMemRelease(handle)
        hip.hipMemAddressFree(va, ctypes.c_size_t(size + page))


@gpu
def test_tail_at_mapping_end():
    """Inputs whose last byte ends their GPU mapping (a read past them faults), with K % 128 != 0: the tile load
    addresses whole 128-column rows, so the last row's partial k-tile must not come in by TDM. Every producer and
    entry, each input in turn; K % 8 != 0 and K < 64 (a row's tile load reaches several rows ahead) go through a
    copy with room. The outputs are those of the same inputs in ordinary storage."""
    M, K, I = 257, 576, 128
    lens = torch.tensor([100, 0, 157], device="cuda")
    offs = _offs(lens)
    x, w, x300 = _x(M, K, seed=1), _x(3, 64, 96, seed=2), _x(130, 300, seed=3)
    x32, x8 = _x(130, 32, seed=6), _x(129, 8, seed=7)
    lens32 = torch.tensor([100, 0, 30], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(4)
    l1 = (torch.randn(M, 2 * I, generator=g, device="cuda") * 3).to(torch.bfloat16)
    dact = torch.randn(M, I, generator=g, device="cuda").to(torch.bfloat16)
    probs = torch.rand(M, generator=g, device="cuda")
    d2s = torch.arange(M, device="cuda", dtype=torch.int32).flip(0)  # the last row first
    q8, s8 = (t.cuda() for t in R.quant_rows(_x(M, 512, seed=5, special=False).cpu()))
    cases = {
        "dual": (lambda a: Q.quant_mxfp8(a, layout=ACT), [x]),
        "rows": (lambda a: Q.quant_mxfp8(a, cols=False), [x]),
        "columns": (lambda a: Q.quant_mxfp8(a, rows=False), [x]),
        "wire rows": (lambda a: Q.quant_mxfp8_rows(a), [x]),
        "2d": (lambda a: Q.quant_mxfp8(a, blk_2d=True, layout=WGT), [w]),
        "K % 8 != 0": (lambda a: Q.quant_mxfp8(a), [x300]),
        "K = 32": (lambda a: Q.quant_mxfp8(a), [x32]),
        "K = 8": (lambda a: Q.quant_mxfp8(a), [x8]),
        "grouped": (lambda a: Q.grouped_quant_mxfp8(a, lens, offs), [x]),
        "grouped, K = 32": (lambda a: Q.grouped_quant_mxfp8(a, lens32, _offs(lens32)), [x32]),
        "glu": (lambda a: Q.grouped_glu_quant_mxfp8(a, probs, lens, offs), [l1]),
        "dglu": (lambda a, b: Q.grouped_dglu_quant_mxfp8(a, b, probs, lens, offs)[1:], [dact, l1]),
        "gather": (lambda a: Q.grouped_gather_quant_mxfp8(a, d2s, lens, offs), [x]),
        "fp8 rows": (lambda a, b: Q.grouped_gather_fp8_quant_mxfp8(a, b, d2s, lens, offs), [q8, s8]),
    }
    for name, (fn, inputs) in cases.items():
        ref = fn(*inputs)
        for i, t in enumerate(inputs):  # each input in turn at the end of its mapping
            with tensor_at_mapping_end(t.shape, t.dtype) as e:
                e.copy_(t)
                got = fn(*[e if j == i else u for j, u in enumerate(inputs)])
                what = f"{name}, input {i} at the mapping end"
                if len(got) == 8:
                    _same(got, ref, GRAD, what)
                else:
                    for a, b, n in zip(got, ref, NAMES):
                        assert (a is None) == (b is None)
                        if a is not None:
                            _eq(a, b, f"{what}: {n}")


@gpu
def test_refused():
    """What the kernels do not write, or cannot keep in bounds, raises ``ValueError`` before anything runs."""
    x, x256 = _x(8, 128, special=False), _x(8, 256, special=False)
    l1, probs = _x(8, 192, special=False), torch.ones(8, device="cuda")
    lens = torch.tensor([8], device="cuda")
    offs = _offs(lens)
    d2s = torch.zeros(8, dtype=torch.int32, device="cuda")
    q8, s8 = (t.cuda() for t in R.quant_rows(x.cpu()))
    bad = {
        "no copy": lambda: Q.quant_mxfp8(x, rows=False, cols=False),
        "unknown layout bits": lambda: Q.quant_mxfp8(x, layout=1),
        "fp16 input": lambda: Q.quant_mxfp8(x.half()),
        "E5M2 codes": lambda: Q.quant_mxfp8(x, torch.float8_e5m2),
        "16x16 codes, rows % 16 != 0": lambda: Q.quant_mxfp8(_x(100, 128, special=False), layout=WGT),
        "16x16 codes in an expert row copy": lambda: Q.grouped_quant_mxfp8(x, lens, offs, layout=WGT),
        "int32 offsets": lambda: Q.grouped_quant_mxfp8(x, lens, offs.int()),
        "offsets on another device": lambda: Q.grouped_quant_mxfp8(x, lens, offs.cpu()),
        "no experts": lambda: Q.grouped_quant_mxfp8(x, lens[:0], offs[:1]),
        "K = 0": lambda: Q.grouped_quant_mxfp8(x[:, :0], lens, offs),
        "GLU width off the 128-column tiles": lambda: Q.grouped_glu_quant_mxfp8(l1, probs, lens, offs),
        "dGLU width": lambda: Q.grouped_dglu_quant_mxfp8(l1[:, :96].contiguous(), l1, probs, lens, offs),
        "probs dtype": lambda: Q.grouped_glu_quant_mxfp8(x256, probs.double(), lens, offs),
        "act_out shape": lambda: Q.grouped_glu_quant_mxfp8(x256, probs, lens, offs, act_out=x[:, :64]),
        "dest2src dtype": lambda: Q.grouped_gather_quant_mxfp8(x, d2s.long(), lens, offs),
        "dest2src shorter than M": lambda: Q.grouped_gather_quant_mxfp8(x, d2s[:4], lens, offs, total_m=8),
        "gather width % 8 != 0": lambda: Q.grouped_gather_quant_mxfp8(
            _x(8, 300, special=False), d2s, lens, offs
        ),
        "MX rows: scales shape": lambda: Q.grouped_gather_fp8_quant_mxfp8(
            q8, s8[:, :2].contiguous(), d2s, lens, offs
        ),
        "MX rows: row_map dtype": lambda: Q.grouped_gather_fp8_quant_mxfp8(q8, s8, d2s.long(), lens, offs),
    }
    for what, call in bad.items():
        try:
            call()
        except ValueError:
            continue
        pytest.fail(f"{what}: no ValueError")


def test_operand_slack(monkeypatch):
    """Every copy an entry allocates keeps the storage past its last byte that the MX GEMM's two-step K loop may
    address (2048 bytes: the next 16 rows of a B operand's 128-wide step); without it the GEMM falls back to its
    one-step kernels. Host-only: the entries allocate, the kernels are not run."""
    monkeypatch.setattr(Q, "_run", lambda *a, **k: None)
    monkeypatch.setattr(Q, "_check", lambda x, *a: x)
    bf = torch.bfloat16
    x, w, l1 = (
        torch.zeros(208, 256, dtype=bf),
        torch.zeros(3, 64, 96, dtype=bf),
        torch.zeros(208, 512, dtype=bf),
    )
    lens = torch.tensor([0, 37, 1, 170])
    offs = _offs(lens)
    probs, d2s = torch.ones(208), torch.zeros(208, dtype=torch.int32)
    q8, s8 = torch.zeros(208, 256, dtype=torch.float8_e4m3fn), torch.zeros(208, 8, dtype=torch.uint8)
    outs = {
        "dual": Q.quant_mxfp8(x, layout=ACT),
        "weights": Q.quant_mxfp8(w, blk_2d=True, layout=WGT),
        "wire rows": Q.quant_mxfp8_rows(x),
        "grouped": Q.grouped_quant_mxfp8(x, lens, offs, layout=ACT)[:4],
        "glu": Q.grouped_glu_quant_mxfp8(l1, probs, lens, offs, layout=ACT)[:4],
        "dglu": Q.grouped_dglu_quant_mxfp8(x, l1, probs, lens, offs)[1:5],
        "gather": Q.grouped_gather_quant_mxfp8(x, d2s, lens, offs, probs)[:4],
        "MX-row gather": Q.grouped_gather_fp8_quant_mxfp8(q8, s8, d2s, lens, offs)[:4],
    }
    for name, ts in outs.items():
        for i, t in enumerate(ts):
            assert t.is_contiguous() and Q._slack(t) >= 2048, f"{name}[{i}]: {Q._slack(t)} bytes past its end"


# ---- the dispatch to these kernels (the recipes' checks and the layout's inverse: host-only)
RAW, MX_A = ScalingRecipe(), ScalingRecipe(shuffle_scale=True)
MX_B = ScalingRecipe(shuffle_scale=True, shuffle_out=True)


@gpu
def test_ops_reach_the_kernels():
    """The quantization ops with the MX GEMM recipes return what the kernel entries return: the dual cast and
    single copies, weights, grouped, and the permute handoff's gather."""
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.ops import quantization as O

    E4, MX, kw = torch.float8_e4m3fn, ScalingGranularity.MX_BLOCKWISE, dict(block_size=32)
    act = dict(scaling_recipe=MX_A, scaling_recipe_for_trans=MX_B, **kw)
    x, w = _x(130, 576, seed=1), _x(2, 64, 96, seed=2)
    both = Q.quant_mxfp8(x, layout=ACT)
    got = O.quantize_fp8_with_trans(x, E4, MX, **act)
    rows = O.quantize_fp8(x, E4, MX, axis=1, scaling_recipe=MX_A, **kw)
    cols = O.quantize_fp8(x, E4, MX, axis=0, scaling_recipe=MX_B, **kw)
    for a, b, what in zip((*got, *rows, *cols), (*both, *both), NAMES * 2):
        _eq(a, b, what)
    mx_w = ScalingRecipe(use_2d_block=True, shuffle_scale=True, shuffle_out=True)
    got = O.quantize_fp8_with_trans(w, E4, MX, scaling_recipe=mx_w, scaling_recipe_for_trans=mx_w, **kw)
    for a, b, what in zip(got, Q.quant_mxfp8(w, blk_2d=True, layout=WGT), NAMES):
        _eq(a, b, f"weights: {what}")
    lens = torch.tensor([0, 37, 1, 92], device="cuda")
    offs = _offs(lens)
    ref = Q.grouped_quant_mxfp8(x, lens, offs, layout=ACT)
    _same(O.grouped_quantize_fp8_with_trans(x, E4, MX, lens, offs, **act), ref, ACT, "grouped")
    q, s, ln, of = O.grouped_quantize_fp8(x, E4, MX, lens, offs, axis=1, scaling_recipe=MX_A, **kw)
    _same((q, s, None, None, ln, of, *ref[6:]), ref, ACT, "grouped rows")
    q, s, ln, of = O.grouped_quantize_fp8(x, E4, MX, lens, offs, axis=0, scaling_recipe=MX_B, **kw)
    _same((None, None, q, s, *ref[4:6], ln, of), ref, ACT, "grouped columns")
    g = torch.Generator(device="cuda").manual_seed(3)
    d2s = torch.randint(0, 130, (130,), generator=g, device="cuda", dtype=torch.int32)
    pp = torch.rand(130, generator=g, device="cuda")
    ref = Q.grouped_gather_quant_mxfp8(x, d2s, lens, offs, pp, layout=ACT)
    got = O.grouped_quantize_fp8_with_trans(
        x, E4, MX, lens, offs, dest2src=d2s, permuted_probs=pp, total_m=130, **act
    )
    _same(got, ref, ACT, "gathered")


def test_gemm_layout_dispatch(monkeypatch):
    """``_gfx1250_mxfp8``: recipes that ask for the MX GEMM layout go to the FlyDSL quantizer with their layout
    bits, raw recipes keep the usual path, and a GEMM layout the quantizer cannot write raises; never off gfx1250."""
    x, E4 = torch.empty(32, 64, dtype=torch.bfloat16), torch.float8_e4m3fn
    assert (GRAD, ACT, WGT) == (0, Q.SHUF_COL, Q.SHUF_ROW | Q.SHUF_COL)
    assert QI._gfx1250_mxfp8(x, E4, RAW, RAW) is None
    for row, col, lay in ((MX_A, MX_A, GRAD), (MX_A, MX_B, ACT), (MX_B, MX_B, WGT)):
        assert QI._gfx1250_mxfp8(x, E4, row, col) == (Q, lay)
    for args in (
        (x.float(), E4, MX_A, MX_A),
        (x, torch.float8_e5m2, MX_A, MX_A),
        (x, E4, MX_A, RAW),
        (x, E4, RAW, MX_B),
        (x, E4, ScalingRecipe(shuffle_out=True), MX_A),
        (x, E4, MX_A, ScalingRecipe(shuffle_scale=True, use_sr=True)),
        (x, E4, MX_A, MX_A, False),
    ):
        with pytest.raises(NotImplementedError, match="MX GEMM layout"):
            QI._gfx1250_mxfp8(*args)
    monkeypatch.setattr(QI, "is_gfx1250", lambda: False)
    assert QI._gfx1250_mxfp8(x, E4, MX_A, MX_B) is None


@pytest.mark.parametrize("shape", [(96, 384), (3, 64, 128), (100, 200)])
def test_gemm_layout_undone(shape):
    """``_gfx1250_unshuffle_mxfp8`` gives dequantization row-major codes and raw scales back (rows % 32 != 0: the
    packed scales' pad rows are dropped); a raw recipe's tensors pass through."""
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(sum(shape))).to(torch.bfloat16)
    x = torch.nn.functional.pad(x, (0, _c128(shape[-1]) - shape[-1]))
    qs = [R.quant_rows(t) for t in x.reshape(-1, *x.shape[-2:])]
    q, s = (torch.stack(t).reshape(*x.shape[:-1], -1) for t in zip(*qs))
    for rec in (MX_A, MX_B) if shape[-2] % 16 == 0 else (MX_A,):
        qg = R.preshuffle_b16(q) if rec.shuffle_out else q
        got = QI._gfx1250_unshuffle_mxfp8(qg, R.pack_scales(s), rec)
        _eq(got[0], q, f"{rec}: codes")
        _eq(got[1], s, f"{rec}: scales")
    for rec in (RAW, None):
        got = QI._gfx1250_unshuffle_mxfp8(q, s, rec)
        assert got[0] is q and got[1] is s
