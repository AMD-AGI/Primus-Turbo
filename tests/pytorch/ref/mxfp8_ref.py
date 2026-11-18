###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch reference of the gfx1250 MXFP8 training recipe (tests and benchmarks only).

Recipe: E4M3 data, one E8M0 scale per 32 consecutive elements along each GEMM's reduction dim.
Activations and gradients are quantized twice: rows (blocks along the last dim) and a transposed
column copy (blocks along the first dim, stored ``[C, R]``). Weights use 32x32 blocks, so both
copies hold the same fp8 values.

Scale: the smallest 2^e with amax / 2^e <= 448, so nothing saturates (the MXFP8 attention's rule).
E8M0 byte = exp_field(amax) - 8 + (mantissa(amax) > 0x600000), clamped to >= 0; amax = 0 gives byte 0
(2^-127); a non-finite amax gives 0xFF (NaN), so Inf / NaN inputs come out as NaN (every code of such a
block is 0xFF, what gfx1250's scaled conversion makes of a NaN scale).

Scale layout of the GEMM (aiter's mx32 layout): ``[ceil(R/32), ceil(K/128), 32, 4]`` uint8, where
``s4[g, t, r, j]`` scales row ``32 g + r`` over K block ``4 t + j`` (elements ``128 t + 32 j ..``).
One int32 of it is the WMMA scale operand of one row for one 128-wide K step. Pad bytes are 0x00.
"""

import torch

MX = 32


def _e8m0_bytes(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 byte (int32) of a float32 amax tensor."""
    bits = amax.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    byte = (exp - 8 + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)).clamp_(min=0)
    return torch.where(exp == 0xFF, torch.full_like(byte, 0xFF), byte)


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e as float32 built from its bits, exact (no pow / ldexp, which need not be exact on a GPU); e >= -127."""
    e = e.to(torch.int32)
    bits = torch.where(e > -127, (e + 127) << 23, torch.full_like(e, 0x400000))
    return bits.view(torch.float32)


def _scale_of(byte: torch.Tensor) -> torch.Tensor:
    """2^(byte - 127) as float32 (byte 0xFF -> NaN)."""
    s = _pow2(byte.to(torch.int32).clamp(max=254) - 127)
    return torch.where(byte == 0xFF, torch.full_like(s, float("nan")), s)


def _quant_blocks(xb: torch.Tensor, amax: torch.Tensor):
    """xb [..., 32] float32 with its block amax [...] -> (fp8 [..., 32], E8M0 bytes [...] uint8)."""
    byte = _e8m0_bytes(amax)
    # x * 2^(127 - byte): exact, and a normal float for every finite byte (<= 247), so no denormal
    # arithmetic on any device; non-finite blocks are overwritten below
    v = xb * _pow2(127 - byte.clamp(max=247)).unsqueeze(-1)
    q = v.to(torch.float8_e4m3fn).view(torch.uint8)
    q = torch.where((byte == 0xFF).unsqueeze(-1), torch.full_like(q, 0xFF), q)  # NaN scale: NaN codes
    return q.view(torch.float8_e4m3fn), byte.to(torch.uint8)


def quant_rows(x: torch.Tensor):
    """[R, C] -> (fp8 [R, C], raw E8M0 [R, ceil(C/32)]). C is zero-padded to a multiple of 32 for amax."""
    R, C = x.shape
    Cp = -(-C // MX) * MX
    xf = torch.nn.functional.pad(x.float(), (0, Cp - C)).view(R, Cp // MX, MX)
    q, s = _quant_blocks(xf, xf.abs().amax(-1))
    return q.view(R, Cp)[:, :C].contiguous(), s


def quant_dual(x: torch.Tensor):
    """Activation / gradient: rows of x and rows of x^T (= column blocks of x).

    Returns (q_row [R, C], s_row [R, ceil(C/32)], q_col [C, R], s_col [C, ceil(R/32)])."""
    q_row, s_row = quant_rows(x)
    q_col, s_col = quant_rows(x.t())
    return q_row, s_row, q_col, s_col


def quant_2d(w: torch.Tensor):
    """Weight: one scale per 32x32 block, shared by both copies.

    Returns (q [N, K], s_row [N, ceil(K/32)], q_t [K, N] = q^T, s_t [K, ceil(N/32)])."""
    N, K = w.shape
    Np, Kp = -(-N // MX) * MX, -(-K // MX) * MX
    wf = torch.nn.functional.pad(w.float(), (0, Kp - K, 0, Np - N))
    blk = wf.view(Np // MX, MX, Kp // MX, MX).permute(0, 2, 1, 3)  # [N/32, K/32, 32, 32]
    amax = blk.abs().amax((-1, -2))
    q, s = _quant_blocks(blk.reshape(*amax.shape, MX * MX), amax)
    q = q.view(Np // MX, Kp // MX, MX, MX).permute(0, 2, 1, 3).reshape(Np, Kp)[:N, :K].contiguous()
    s_row = s.repeat_interleave(MX, 0)[:N]  # [N, K/32]
    s_t = s.t().repeat_interleave(MX, 0)[:K]  # [K, N/32]
    return q, s_row, q.t().contiguous(), s_t.contiguous()


PAD_SCALE = 127  # E8M0 byte of a column block that holds no token (padding only): 1.0


def quant_grouped(x: torch.Tensor, group_lens: torch.Tensor):
    """MoE activation / gradient ``x [M, C]`` in expert order, ``group_lens [G]`` tokens per expert.

    Rows: ``quant_rows(x)`` with each expert's rows zero-padded to a multiple of 32 (Turbo's
    ``group_offs_padded_rowwise``, so an expert starts on a 32-row scale block of the packed layout).
    Columns: per expert, its tokens transposed and zero-padded to a multiple of 128, so no
    32-block straddles two experts: ``q_col [C, sum(pad_g)]``, ``s_col [C, sum(pad_g) / 32]``. A pad row's
    blocks and a column block holding no token get ``PAD_SCALE`` (127 = 1.0: Turbo's grouped contract,
    ``grouped_gemm_fp8_impl._alloc_mx_act_buffers``, whose quantizer gives every zero block 1.0); a block
    with tokens is quantized normally (all-zero tokens: byte 0, like any zero block). Codes of padding are 0.

    Returns (q_row, s_row, q_col, s_col, lens_col [G] int64, offs_col [G + 1] int64, lens_row, offs_row)."""
    lens = [int(n) for n in group_lens.tolist()]
    rpads = [-(-n // MX) * MX for n in lens]
    q_row = torch.zeros(sum(rpads), x.shape[1], dtype=torch.uint8, device=x.device)
    s_row = torch.full((sum(rpads), -(-x.shape[1] // MX)), PAD_SCALE, dtype=torch.uint8, device=x.device)
    qr, sr = quant_rows(x)
    o = orow = 0
    for n, pn in zip(lens, rpads):
        q_row[orow : orow + n], s_row[orow : orow + n] = qr[o : o + n].view(torch.uint8), sr[o : o + n]
        o, orow = o + n, orow + pn
    q_row = q_row.view(torch.float8_e4m3fn)
    pads = [-(-n // 128) * 128 for n in lens]
    q_col = torch.zeros(x.shape[1], sum(pads), dtype=torch.uint8, device=x.device)
    s_col = torch.zeros(x.shape[1], sum(pads) // MX, dtype=torch.uint8, device=x.device)
    o = oc = 0
    for n, pn in zip(lens, pads):
        if pn:
            xg = torch.zeros(pn, x.shape[1], dtype=x.dtype, device=x.device)
            xg[:n] = x[o : o + n]
            q, s = quant_rows(xg.t())
            s[:, -(-n // MX) :] = PAD_SCALE
            q_col[:, oc : oc + pn] = q.view(torch.uint8)
            s_col[:, oc // MX : (oc + pn) // MX] = s
        o, oc = o + n, oc + pn
    lens_col, lens_row = (torch.tensor(v, dtype=torch.int64, device=x.device) for v in (pads, rpads))
    offs_col, offs_row = (torch.nn.functional.pad(torch.cumsum(v, 0), (1, 0)) for v in (lens_col, lens_row))
    return q_row, s_row, q_col.view(torch.float8_e4m3fn), s_col, lens_col, offs_col, lens_row, offs_row


_GELU_A2, _GELU_B, _GELU_C2 = 2 * 0.7978845608028654, 0.044715, 6 * 0.7978845608028654 * 0.044715


def _glu_in(l1, clamp_limit):
    """gate / up halves of l1 [M, 2I] in fp32, clamped (min(gate, L), clamp(up, -L, L)), with kept masks."""
    g, u = l1.float().chunk(2, dim=1)
    if not clamp_limit:
        return g, u, None, None
    gc, uc = g.clamp(max=clamp_limit), u.clamp(-clamp_limit, clamp_limit)
    return gc, uc, gc == g, uc == u


def glu_ref(l1, probs, activation="silu", clamp_limit=None):
    """MoE GLU forward before its quantization (bf16 [M, I]): f(gate) * up * probs[row] in fp32 as the
    gfx950 / gfx1250 phase-2 kernels compute it: silu as g / (1 + exp2(-g log2 e)), gelu (tanh form) as
    g * sigmoid(2 sqrt(2/pi) g (1 + 0.044715 g^2)). The kernels' exp2 / reciprocal are the hardware's, so
    their bf16 values may be an ulp off this; their quantization is checked on their own values."""
    g, u, _, _ = _glu_in(l1, clamp_limit)
    gate = (
        g / (1 + torch.exp2(g * -1.4426950408889634))
        if activation == "silu"
        else g * torch.sigmoid(g * _GELU_A2 * (1 + g * g * _GELU_B))
    )
    return ((gate * u) * probs.float()[:, None]).to(torch.bfloat16)


def dglu_ref(grad_act, l1, probs, activation="silu", clamp_limit=None):
    """MoE GLU backward before quantization: (grad_l1 = [dgate | dup] [M, 2I] bf16, grad_probs [M] fp32)
    with d = grad_act, a = f(gate): dgate = d p up f'(gate), dup = d p a (zero where the clamp bit),
    grad_probs = sum_j d a up."""
    g, u, kg, ku = _glu_in(l1, clamp_limit)
    if activation == "silu":
        sg = torch.sigmoid(g)
        a, da = sg * g, sg * (1 + g - sg * g)
    else:
        sg = torch.sigmoid(g * _GELU_A2 * (1 + g * g * _GELU_B))
        a, da = g * sg, sg + g * sg * (1 - sg) * (_GELU_A2 + g * g * _GELU_C2)
    d = grad_act.float()
    dp = d * probs.float()[:, None]
    dg, du = dp * u * da, dp * a
    if kg is not None:
        dg, du = torch.where(kg, dg, 0.0), torch.where(ku, du, 0.0)
    return torch.cat([dg, du], 1).to(torch.bfloat16), (d * a * u).sum(1)


def gather_rows(x, dest2src, permuted_probs=None):
    """The permuted activation: row d = x[dest2src[d]]; zero where dest2src is out of range or the
    row's prob is exactly 0 (the routing padding rule)."""
    idx = dest2src.long()
    ok = (idx >= 0) & (idx < x.shape[0])
    if permuted_probs is not None:
        ok &= permuted_probs != 0
    return torch.where(ok[:, None], x[idx.clamp(0, x.shape[0] - 1)], torch.zeros((), dtype=x.dtype))


def gather_mx_rows(q, s, row_map):
    """MX rows ``q [T, C]`` with raw scales ``s [T, ceil(C/32)]``, row d = row ``row_map[d]``; out of range: zero codes,
    byte-0 scales (a zero row)."""
    idx = row_map.long()
    ok = ((idx >= 0) & (idx < q.shape[0]))[:, None]
    i = idx.clamp(0, q.shape[0] - 1)
    zero = torch.zeros((), dtype=torch.uint8)
    qg = torch.where(ok, q.view(torch.uint8)[i], zero).view(torch.float8_e4m3fn)
    return qg, torch.where(ok, s.view(torch.uint8)[i], zero)


def quant_grouped_fp8(q, s, row_map, group_lens):
    """``quant_grouped`` of gathered MX rows (MegaMoE's pool, ``gather_mx_rows``): the row copy keeps each row's codes
    and scales, the column copy quantizes the dequantized (exact) values."""
    qg, sg = gather_mx_rows(q, s, row_map)
    out = list(quant_grouped(dequant(qg, sg, torch.bfloat16), group_lens))
    q_row, o, orow = out[0].view(torch.uint8), 0, 0
    for n, pn in zip(group_lens.tolist(), out[6].tolist()):
        q_row[orow : orow + n], out[1][orow : orow + n] = qg[o : o + n].view(torch.uint8), sg[o : o + n]
        o, orow = o + n, orow + pn
    return tuple(out)


def pack_scales(s: torch.Tensor) -> torch.Tensor:
    """Raw E8M0 [.., R, ceil(K/32)] -> the GEMM layout [.., ceil(R/32), ceil(K/128), 32, 4], pad bytes 0x00."""
    if s.ndim > 2:
        return torch.stack([pack_scales(t) for t in s])
    s = s.view(torch.uint8)
    R, KB = s.shape
    Rp, KBp = -(-R // 32) * 32, -(-KB // 4) * 4
    sp = torch.zeros(Rp, KBp, dtype=torch.uint8, device=s.device)
    sp[:R, :KB] = s
    return sp.view(Rp // 32, 32, KBp // 4, 4).permute(0, 2, 1, 3).contiguous()


def preshuffle_b16(q: torch.Tensor) -> torch.Tensor:
    """A GEMM B operand [.., N, K] in the 16x16-preshuffled byte order of the gfx1250 MX GEMM
    (``[N/16][K/16][16][16]``: each 16x16 block's 256 bytes contiguous), same shape and dtype."""
    *b, N, K = q.shape
    u = q.contiguous().view(torch.uint8).view(*b, N // 16, 16, K // 16, 16)
    return u.transpose(-3, -2).contiguous().view(*b, N, K).view(q.dtype)


def unpreshuffle_b16(q: torch.Tensor) -> torch.Tensor:
    """Inverse of ``preshuffle_b16``."""
    *b, N, K = q.shape
    u = q.contiguous().view(torch.uint8).view(*b, N // 16, K // 16, 16, 16)
    return u.transpose(-3, -2).contiguous().view(*b, N, K).view(q.dtype)


def dequant(q: torch.Tensor, s: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """fp8 [R, C] with raw E8M0 [R, ceil(C/32)] -> [R, C] in ``dtype``."""
    R, C = q.shape
    scale = _scale_of(s.to(torch.int32)).repeat_interleave(MX, 1)[:, :C]
    return (q.to(torch.float32) * scale).to(dtype)


def gemm_ref(a, sa, b, sb, dtype=torch.float64) -> torch.Tensor:
    """NT MX GEMM reference: dequant(a) @ dequant(b)^T in ``dtype`` (float64 = exact for small K)."""
    return dequant(a, sa, dtype) @ dequant(b, sb, dtype).t()


def snr_db(x: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    noise = (x.double() - ref).pow(2).mean()
    return float(10 * torch.log10(ref.pow(2).mean() / noise.clamp_min(1e-300)))


def int_valued_operands(M: int, N: int, K: int, device="cuda", seed: int = 0):
    """fp8 operands holding small integers and scales near 2^0: products and sums are exact in fp32
    (|C| < 2^24), so a correct kernel matches ``gemm_ref`` bitwise in fp32."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randint(-4, 5, (M, K), generator=g, device=device).to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (N, K), generator=g, device=device).to(torch.float8_e4m3fn)
    kb = -(-K // MX)
    sa = torch.randint(126, 129, (M, kb), generator=g, device=device).to(torch.uint8)
    sb = torch.randint(126, 129, (N, kb), generator=g, device=device).to(torch.uint8)
    return a, sa, b, sb


# ---- test helpers: inputs, byte comparisons, the layouts of the quantizers' outputs


def mx_input(*shape, seed=0, special=True, device="cuda"):
    """bf16 data over a wide range, per-row scaled, with all-zero blocks, denormal blocks, values around the scale
    rule's boundary (+-448 * 2^k: mantissas 1.75 / 1.7578125, and others) and +-Inf / NaN (``special``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    rows = shape[-2]
    x = torch.randn(*shape, generator=g, device=device)
    x = x * torch.exp2(torch.randint(-24, 24, (*shape[:-1], 1), generator=g, device=device).float())
    if special and x.numel() >= 64:
        x[..., ::7, :32] = 0
        x[..., 1::5, :] *= 2.0**-128  # rows of denormals / tiny normals
        f = x.view(-1)
        n = f.numel()

        def at(k):
            return torch.randint(0, n, (k,), generator=g, device=device)

        k = max(1, n // 64)
        mant = torch.tensor([1.75, 1.7578125, 1.875, 1.9375, 1.0, 1.9921875], device=device)
        v = mant[torch.randint(0, 6, (k,), generator=g, device=device)] * 256
        e = torch.randint(-40, 40, (k,), generator=g, device=device).float()
        sgn = torch.randint(0, 2, (k,), generator=g, device=device).float() * 2 - 1
        f[at(k)] = v * torch.exp2(e) * sgn
        if rows > 2:
            f[at(3)] = torch.tensor([float("inf"), -float("inf"), float("nan")], device=device)
    return x.to(torch.bfloat16)


def assert_bytes_equal(a: torch.Tensor, b: torch.Tensor, what: str):
    """``a`` and ``b`` hold the same bytes (and shape); the message names the first difference."""
    a, b = a.contiguous().view(torch.uint8).cpu(), b.contiguous().view(torch.uint8).cpu()
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    bad = (a != b).nonzero()
    assert bad.numel() == 0, (
        f"{what}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}: {a[tuple(bad[0])]} vs {b[tuple(bad[0])]}"
    )


def padded(t: torch.Tensor, shape) -> torch.Tensor:
    """``t``'s bytes (uint8, CPU) in the top-left corner of zeros of ``shape``: the quantizers' 128-padding."""
    out = torch.zeros(shape, dtype=torch.uint8)
    out[tuple(slice(0, d) for d in t.shape)] = t.cpu().view(torch.uint8)
    return out


def to_layout(copies, pack_row=False, shuf_row=False, pack_col=False, shuf_col=False):
    """Reference (q_row, s_row, q_col, s_col) in the gfx1250 GEMM layouts: packed scales (``pack_scales``) and / or
    16x16-preshuffled codes (``preshuffle_b16``), per copy."""
    qr, sr, qc, sc = (t.cpu() for t in copies)
    return (
        preshuffle_b16(qr) if shuf_row else qr,
        pack_scales(sr) if pack_row else sr,
        preshuffle_b16(qc) if shuf_col else qc,
        pack_scales(sc) if pack_col else sc,
    )


def check_grouped(out, vals, lens, pack_row=False, shuf_row=False, pack_col=False, shuf_col=False):
    """A grouped quantizer's 8-tuple against ``quant_grouped`` of the bf16 values it quantized (see
    ``check_grouped_ref``)."""
    ref = quant_grouped(vals.cpu(), lens)
    check_grouped_ref(out, ref, lens, pack_row, shuf_row, pack_col, shuf_col)


def check_grouped_ref(out, ref, lens, pack_row=False, shuf_row=False, pack_col=False, shuf_col=False):
    """A grouped quantizer's 8-tuple against a reference 8-tuple (``quant_grouped``'s order): the allocation bounds
    (rows ceil32(M + 32 G), tokens ceil128(M + 128 G)), the 32-row / 128-token padded lens and offsets of both
    copies, and the rows / columns up to them in the given layout (the row copy's columns padded to the
    quantizer's width: zero codes, byte-0 scales)."""
    q_row, s_row, q_col, s_col, lens_c, offs_c, lens_r, offs_r = ref
    M, G, Mr, Mc = int(lens.sum()), len(lens), int(offs_r[-1]), int(offs_c[-1])
    Kp = out[0].shape[1]
    q_row, s_row = padded(q_row, (Mr, Kp)).view(q_row.dtype), padded(s_row, (Mr, Kp // MX))
    assert (
        out[0].shape[0] == -(-(M + 32 * G) // 32) * 32 and out[2].shape[1] == -(-(M + 128 * G) // 128) * 128
    )
    for got, ref, what in zip(
        out[4:], (lens_r, offs_r, lens_c, offs_c), ("lens_row", "offs_row", "lens_col", "offs_col")
    ):
        assert torch.equal(got.cpu(), ref.cpu()), f"{what}: {got.tolist()} vs {ref.tolist()}"
    qr, sr, _, sc = to_layout((q_row, s_row, q_col, s_col), pack_row, shuf_row, pack_col, False)
    assert_bytes_equal(out[0][:Mr], qr, "q_row")
    assert_bytes_equal(out[1][: Mr // 32] if pack_row else out[1][:Mr], sr, "s_row")
    assert_bytes_equal((unpreshuffle_b16(out[2].cpu()) if shuf_col else out[2])[:, :Mc], q_col, "q_col")
    assert_bytes_equal(out[3][:, : Mc // 128] if pack_col else out[3][:, : Mc // 32], sc, "s_col")
