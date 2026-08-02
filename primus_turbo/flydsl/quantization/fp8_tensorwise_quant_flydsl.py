# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL); Apache-2.0, see LICENSE-APACHE.

# Per-tensor fp8 cast + K-pad to Kp=ceil(K/128)*128 so the grouped-gemm consumer never splits a cache line on K%128!=0 hidden dims.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fm
from flydsl.expr.typing import Vector as Vec

from primus_turbo.flydsl.quantization.mxfp8_quant_flydsl import (
    _WR64,
    _ceil128,
    _sat,
    _warp_reduce_max,
    fp8_params,
    in_elt,
)
from primus_turbo.flydsl.utils.gemm_helper import make_row_band_resource

_CM = 1  # glc: bypass L2 on the fp8 output stores
_OOB = 0x7FFFFFFF  # past-end buffer offset -> HW drops the load (K-pad columns read 0)
_AMAX_VW = 8  # 128-bit (dwordx4) bf16/fp16 vec load per lane
_AMAX_NTH = 64  # one wave/WG; the wave butterfly-reduces its 64 lanes to its own partial (no LDS)
_AMAX_NWARP = _AMAX_NTH // 64
_AMAX_GRID_CAP = 32768


def compile_cast_pad(B, R, K, elt=None, out_fp8="e4m3", Kp=None, bm=128, bk=128, nth=256, cm=_CM):
    """Compile the batched per-tensor fp8 cast-and-K-pad quant for a uniform [B, R, K] input
    (B=1 = the plain 2D activation case; B=G = the batched weight [G, N, K]). Tiles the padded
    [R, Kp] extent as [bm rows x bk K-cols]; real K-cols cast the scaled input, pad K-cols write
    0. Rows past R (the ragged last M-tile) HW-drop via the per-batch band num_records. The scale
    is a single f32 scalar (device [1], the whole-tensor 1/scale) loaded by every thread."""
    if elt is None:
        elt = fx.BFloat16
    _va, _ep_sub, sat_bnd, cvt = fp8_params(out_fp8)
    Kp = _ceil128(K) if Kp is None else Kp
    assert Kp % 128 == 0 and Kp >= K
    assert bk % 128 == 0 and Kp % bk == 0, "bk must be a 128-multiple dividing Kp"
    BMv, BKv, NTHv = bm, bk, nth
    VPR = BKv // 4  # vec4 dwords per row within a tile
    assert (BMv * BKv) % (4 * NTHv) == 0, "tile must be a whole number of vec4/thread"
    ITERS = BMv * BKv // 4 // NTHv
    NBK = Kp // BKv
    NBM = (R + BMv - 1) // BMv  # rows are NOT padded; the last tile is masked by the band
    NPB = NBM * NBK  # tiles per batch
    ELT_BYTES = 2  # bf16/fp16 input
    SCMASK = (K % 4) != 0  # K not vec4-aligned -> per-element K-tail mask
    LMASK = (Kp != K) and not SCMASK  # vec4-aligned K-pad -> single masked base redirect

    @flyc.kernel(known_block_size=[NTHv, 1, 1])
    def kern(X: fx.Tensor, Sc: fx.Tensor, Qr: fx.Tensor):
        I32 = fx.Int32
        F32 = fx.Float32
        BF = elt.ir_type
        FF = fx.Float32.ir_type
        IRI = fx.Int32.ir_type
        z = I32(0)

        src = bo.create_buffer_resource(Sc, max_size=False, num_records_bytes=I32(4))
        q_mul = bo.buffer_load(src, z, vec_width=1, dtype=FF)  # vec_width=1 -> scalar f32

        t = fx.thread_idx.x
        pid = fx.block_idx.x
        batch = pid // I32(NPB)
        pib = pid - batch * I32(NPB)
        br = pib // I32(NBK)
        bk_ = pib - br * I32(NBK)

        rx = make_row_band_resource(
            bo.extract_base_index(X), batch * I32(R), (batch + I32(1)) * I32(R), I32(K), ELT_BYTES
        )
        rq = make_row_band_resource(
            bo.extract_base_index(Qr), batch * I32(R), (batch + I32(1)) * I32(R), I32(Kp), 1
        )
        gbase = (br * I32(BMv)) * I32(K) + bk_ * I32(BKv)  # input element base within batch band

        for ls in range_constexpr(ITERS):
            lin = t + I32(ls * NTHv)
            lr = lin // I32(VPR)
            cv = (lin - lr * I32(VPR)) * I32(4)
            grow = br * I32(BMv) + lr
            gcol = bk_ * I32(BKv) + cv
            if const_expr(SCMASK):
                lanes = []
                for j in range_constexpr(4):
                    gcj = gcol + I32(j)
                    vj = bo.buffer_load(
                        rx, gbase + lr * I32(K) + cv + I32(j), vec_width=1, dtype=BF, mask=(gcj < I32(K))
                    )
                    lanes.append(Vec.from_elements([vj], elt).to(F32)[0])
                qf = [lanes[j] * q_mul for j in range_constexpr(4)]
            else:
                ioff = gbase + lr * I32(K) + cv
                if const_expr(LMASK):
                    ioff = (gcol < I32(K)).select(ioff, I32(_OOB))
                vf = Vec(bo.buffer_load(rx, ioff, vec_width=4, dtype=BF)).to(F32)
                qf = [vf[j] * q_mul for j in range_constexpr(4)]
            word = I32(cvt(IRI, _sat(qf[0], sat_bnd), _sat(qf[1], sat_bnd), z, 0))
            word = I32(cvt(IRI, _sat(qf[2], sat_bnd), _sat(qf[3], sat_bnd), word, 1))
            bo.buffer_store(word, rq, grow * I32(Kp) + gcol, cache_modifier=cm, offset_is_bytes=True)

    @flyc.jit
    def launch(X: fx.Tensor, Sc: fx.Tensor, Qr: fx.Tensor, stream: fx.Stream):
        grid = B * NPB
        kern(X, Sc, Qr).launch(grid=(grid, 1, 1), block=(NTHv, 1, 1), stream=stream)

    return launch


def _amax_cfg(nelem):
    nvec = (nelem + _AMAX_VW - 1) // _AMAX_VW
    G = min(_AMAX_GRID_CAP, max(1, (nvec + _AMAX_NTH - 1) // _AMAX_NTH))
    ITERS = (nvec + G * _AMAX_NTH - 1) // (G * _AMAX_NTH)
    G = (nvec + ITERS * _AMAX_NTH - 1) // (ITERS * _AMAX_NTH)
    return G, ITERS


def compile_amax(nelem, elt=None):
    """Compile a whole-tensor abs-amax reduce over a flat [nelem] bf16/fp16 buffer. Each WG
    grid-strides 128-bit vec loads into a per-lane running max; every 64-lane wave butterfly-reduces
    to its lane 0, which writes one f32 partial (G*NWARP total). The host maxes the partials. OOB
    tail lanes read 0, which the abs-max ignores."""
    if elt is None:
        elt = fx.BFloat16
    G, ITERS = _amax_cfg(nelem)
    NTH, VW, NWARP = _AMAX_NTH, _AMAX_VW, _AMAX_NWARP
    STRIDE = G * NTH * VW
    ELT_BYTES = 2

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def kern(X: fx.Tensor, Pr: fx.Tensor):
        I32 = fx.Int32
        F32 = fx.Float32
        IRI = fx.Int32.ir_type
        BF = elt.ir_type
        z = I32(0)

        rx = bo.create_buffer_resource(X, max_size=False, num_records_bytes=I32(nelem * ELT_BYTES))
        rp = bo.create_buffer_resource(Pr, max_size=False, num_records_bytes=I32(G * NWARP * 4))
        t = fx.thread_idx.x
        pid = fx.block_idx.x
        lane = t & I32(63)
        warp = t >> I32(6)
        base = (pid * I32(NTH) + t) * I32(VW)

        acc = F32(0.0)
        for i in range_constexpr(ITERS):
            v = Vec(bo.buffer_load(rx, base + I32(i * STRIDE), vec_width=VW, dtype=BF)).to(F32)
            for j in range_constexpr(VW):
                a = fm.absf(v[j])
                acc = (acc > a).select(acc, a)
        acc = _warp_reduce_max(acc, lane, _WR64, IRI)
        bo.buffer_store(acc, rp, pid * I32(NWARP) + warp, mask=(lane == z))

    @flyc.jit
    def launch(X: fx.Tensor, Pr: fx.Tensor, stream: fx.Stream):
        kern(X, Pr).launch(grid=(G, 1, 1), block=(NTH, 1, 1), stream=stream)

    return launch


_CAST_PAD_CACHE: dict = {}
_AMAX_CACHE: dict = {}


def _flydsl_amax(x):
    """Whole-tensor abs-amax as a device [1] f32, read at peak HBM BW (the FlyDSL reduce beats
    torch.aminmax ~3x). Falls back to torch for tiny inputs where launch overhead dominates."""
    import flydsl.compiler as _flyc
    import torch

    nelem = x.numel()
    if nelem < (1 << 16):
        return x.abs().amax().float().view(1)
    G, _ = _amax_cfg(nelem)
    Pr = torch.empty(G * _AMAX_NWARP, dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()
    key = (nelem, x.dtype)
    comp = _AMAX_CACHE.get(key)
    if comp is None:
        comp = _flyc.compile(compile_amax(nelem, elt=in_elt(x.dtype)), x, Pr, stream)
        _AMAX_CACHE[key] = comp
    comp(x, Pr, stream)
    return Pr.amax().view(1)


def _q_mul_scale(x, out_dtype):
    """Whole-tensor scalar (1/scale device [1] f32, dequant scale device [1] f32) from the real
    data amax, matching the deployed tensorwise quant: scale = amax / fp8_max, q_mul = 1/scale;
    amax==0 -> scale=1 (identity)."""
    import torch

    fp8_max = 57344.0 if out_dtype == torch.float8_e5m2 else 448.0
    amax = _flydsl_amax(x)
    scale = (amax / fp8_max).clamp(min=torch.finfo(torch.float32).tiny)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    q_mul = (1.0 / scale).to(torch.float32).view(1).contiguous()
    return q_mul, scale.view(1)


def quant_fp8_tensorwise_pad_batched(x_3d, out_dtype, scale=None):
    """Batched per-tensor fp8 cast-and-K-pad for a uniform [B, R, K] input (grouped-gemm weight
    [G, N, K]) in ONE launch. Returns (fp8 [B, R, Kp], dequant_scale [1]) with real data in
    [:, :, :K] and K-pad columns [:, :, K:Kp]=0, Kp=ceil(K/128)*128. ``scale`` (dequant factor,
    device [1] f32) may be passed to reuse a caller-computed whole-tensor amax; else it's reduced
    here. The GEMM consumes fp8 at stride Kp and contracts K=Kp (pad lanes 0*0=0)."""
    import flydsl.compiler as _flyc
    import torch

    assert x_3d.ndim == 3 and x_3d.is_contiguous()
    assert x_3d.is_cuda and x_3d.dtype in (torch.bfloat16, torch.float16)
    assert "float8" in str(out_dtype), f"out_dtype must be an fp8 dtype, got {out_dtype}"
    B, R, K = int(x_3d.shape[0]), int(x_3d.shape[1]), int(x_3d.shape[2])
    Kp = _ceil128(K)
    out_fp8 = "e5m2" if out_dtype == torch.float8_e5m2 else "e4m3"

    if scale is None:
        q_mul, scale = _q_mul_scale(x_3d, out_dtype)
    else:
        q_mul = (1.0 / scale.float()).to(torch.float32).view(1).contiguous()
    Qr = torch.empty(B, R, Kp, dtype=out_dtype, device=x_3d.device)
    stream = torch.cuda.current_stream()

    key = (B, R, K, Kp, x_3d.dtype, out_dtype)
    comp = _CAST_PAD_CACHE.get(key)
    if comp is None:
        launch = compile_cast_pad(B, R, K, elt=in_elt(x_3d.dtype), out_fp8=out_fp8, Kp=Kp)
        comp = _flyc.compile(launch, x_3d, q_mul, Qr, stream)
        _CAST_PAD_CACHE[key] = comp
    comp(x_3d, q_mul, Qr, stream)
    return Qr, scale


def quant_fp8_tensorwise_pad(x, out_dtype, scale=None):
    """Per-tensor fp8 cast-and-K-pad for a 2D [R, K] input (activation). Returns (fp8 [R, Kp],
    dequant_scale [1]); delegates to the B-batched path with B=1."""
    assert x.ndim == 2, f"quant_fp8_tensorwise_pad expects 2D, got {x.ndim}D"
    Qr, scale = quant_fp8_tensorwise_pad_batched(x.unsqueeze(0), out_dtype, scale=scale)
    return Qr[0], scale
