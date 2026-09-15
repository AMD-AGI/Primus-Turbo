#!/usr/bin/env python3
"""Launch aiter's prebuilt gfx1250 ASM attention backward directly.

There is no Python-side architecture gate for these kernels -- the C++ host
special-cases gfx1250 in five places, the Python wrapper does not -- so the way
to measure them is to load the .co files and launch them by hand.

Deliberately does NOT patch the installed aiter: the 21.684 ms pure-aiter number
is a reference baseline and must stay reproducible.

The call contract was derived offline and is documented, with its one
unconfirmed part, in output/0913__opt_plan__claude/phase2/T2-ASM-BACKWARD-SPEC.md.
Nothing here has been executed -- the card has been wedged since before it was
written -- so treat the first run as a bring-up, not a measurement.

Self-check without a GPU (packing only):

    python tools/gfx1250/asm_bwd_launcher.py --selftest
"""
import argparse
import ctypes
import struct
import sys
from pathlib import Path

ASM_DIR = Path("/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd")

# Byte strides throughout: the host multiplies element strides by the element
# size (mha_bwd.cu:640 "Hs_q = nhead_stride_q * 2" for bf16, "* 4" for the fp32
# lse/delta). Passing element strides silently reads the wrong memory.
BF16 = 2
FP32 = 4

# mask_x / mask_y are assigned by the host only under "if (mt == 3)", the
# generic-window path. Our CSV row is mask=2, bottom-right causal, so the host
# leaves them at whatever was on the stack -- fmha_bwd_dqdkdv_args is declared
# uninitialised at mha_bwd.cu:621. The causal behaviour lives in the
# _causal_br_ kernel variant itself, not in these fields, so passing zero is
# safe and is what this launcher does.
MASK_X = MASK_Y = 0

TS_KV = 128   # dqdkdv tile along k, from fmha_bwd_dqdkdv.csv
TS_QO = 32    # dqdkdv tile along q/o
TS_ODO = 128  # odo preprocess tile, from fmha_bwd_odo.csv
TS_DQ = 64    # dq_convert tile, from fmha_bwd_dq_convert.csv
BDX = 128     # threads per workgroup on gfx1250 (mha_bwd.cu:580) -- 4 wave32s

# From the ELF metadata; every field padded to 16 bytes. GENERATED -- do not
# hand-edit. The first hand-written version of this table silently dropped the
# last seven fields, mask_x and mask_y among them, which the causal variant
# needs; the self-test caught it only because it prints where the last field
# ends. Regenerate with: tools/gfx1250/asm_bwd_abi.py --emit-packer
DQDKDV_FIELDS = [
    ("ptr_dq", 0, 8), ("ptr_dk", 16, 8), ("ptr_dv", 32, 8), ("ptr_q", 48, 8), ("ptr_k", 64, 8),
    ("ptr_v", 80, 8), ("ptr_do", 96, 8), ("ptr_lse", 112, 8), ("ptr_d", 128, 8),
    ("scalar", 144, 4), ("log2e", 160, 4), ("seq_len_q", 176, 4), ("Ts", 192, 4), ("Hs_q", 208, 4),
    ("BAs_q", 224, 4), ("Seqs_q", 240, 4), ("ratio", 256, 4), ("Hs_k", 272, 4), ("BAs_k", 288, 4),
    ("Seqs_k", 304, 4), ("Seqs_dk", 320, 4), ("seq_len_k", 336, 4), ("head_dim_q", 352, 4),
    ("head_dim_v", 368, 4), ("nhead_q", 384, 4), ("Hs_v", 400, 4), ("BAs_v", 416, 4),
    ("Seqs_v", 432, 4), ("Hs_do", 448, 4), ("BAs_do", 464, 4), ("Seqs_do", 480, 4),
    ("Hs_dk", 496, 4), ("BAs_dk", 512, 4), ("Hs_dv", 528, 4), ("BAs_dv", 544, 4),
    ("Seqs_dv", 560, 4), ("Hs_lsed", 576, 4), ("ptr_seqstart_q", 592, 8),
    ("ptr_seqstart_k", 608, 8), ("ptr_seqstart_q_padded", 624, 8),
    ("ptr_seqstart_k_padded", 640, 8), ("max_seq_len_dq", 656, 4), ("mask_x", 672, 4),
    ("mask_y", 688, 4)
]
DQDKDV_SIZE = 704

# The odo kernel is packed COMPACTLY (use_compact_fmha_bwd_kernel_args() returns
# true for exactly gfx1250), transcribed from pack_fmha_bwd_odo_args().
# NOTE: this is the one layout that could not be confirmed against the
# disassembly -- odo uses gfx1250 kernarg preload and issues zero s_load
# instructions, so "which offsets does it read" cannot be asked of it. The
# evidence is the C++ packer plus the fact that the fields sum to exactly its
# 84-byte kernarg_segment_size. If a first run returns numerical garbage,
# suspect this before anything else.
ODO_FIELDS = [
    ("ptr_o", 8), ("ptr_do", 8), ("ptr_d", 8),
    ("Hs_o", 4), ("BAs_o", 4), ("Seqs_o", 4),
    ("Hs_do", 4), ("BAs_do", 4), ("Seqs_do", 4),
    ("Hs_d", 4), ("BAs_d", 4), ("Seqs_d", 4),
    ("seqlen_q", 4), ("head_dim", 4),
    ("ptr_qseq", 8), ("ptr_qseq_padded", 8),
]
ODO_SIZE = 84

# dq_convert: 16-byte-padded struct (fmha_bwd_post_kernel_args). Its ELF
# declares 208 bytes while the header sums to 192, but the disassembly shows the
# kernel loads only offsets 0x00..0x90 and never touches ptr_qseq/ptr_qseq_padded
# at 0xa0/0xb0, which are group-mode only. So 208 zeroed bytes with the first
# 0xa0 filled is correct for batch mode.
POST_FIELDS = [
    ("ptr_dq_acc", 0x00, 8), ("ptr_dq", 0x10, 8),
    ("Hs_dq_acc", 0x20, 4), ("BAs_dq_acc", 0x30, 4), ("Seqs_dq_acc", 0x40, 4),
    ("Hs_dq", 0x50, 4), ("BAs_dq", 0x60, 4), ("Seqs_dq", 0x70, 4),
    ("seqlen_q", 0x80, 4), ("head_dim", 0x90, 4),
]
POST_SIZE = 208


def pack_padded(fields, size, values):
    buf = bytearray(size)
    for name, offset, width in fields:
        v = values[name]
        if width == 8:
            struct.pack_into("<Q", buf, offset, int(v))
        elif isinstance(v, float):
            struct.pack_into("<f", buf, offset, v)
        else:
            struct.pack_into("<I", buf, offset, int(v) & 0xFFFFFFFF)
    return buf


def pack_compact(fields, size, values):
    buf, off = bytearray(size), 0
    for name, width in fields:
        v = values[name]
        if width == 8:
            struct.pack_into("<Q", buf, off, int(v))
        elif isinstance(v, float):
            struct.pack_into("<f", buf, off, v)
        else:
            struct.pack_into("<I", buf, off, int(v) & 0xFFFFFFFF)
        off += width
    assert off == size, f"compact pack produced {off} bytes, kernarg segment is {size}"
    return buf


class HipModule:
    """Minimal hipModuleLoad / hipModuleLaunchKernel wrapper over libamdhip64."""

    HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
    HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
    HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)

    def __init__(self):
        self.hip = ctypes.CDLL("libamdhip64.so")
        self.modules = {}

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what} failed with hipError {rc}")

    def function(self, co_path: Path, symbol: str):
        key = str(co_path)
        if key not in self.modules:
            mod = ctypes.c_void_p()
            self._check(self.hip.hipModuleLoad(ctypes.byref(mod), str(co_path).encode()),
                        f"hipModuleLoad({co_path.name})")
            self.modules[key] = mod
        fn = ctypes.c_void_p()
        self._check(self.hip.hipModuleGetFunction(ctypes.byref(fn), self.modules[key], symbol.encode()),
                    f"hipModuleGetFunction({symbol})")
        return fn

    def launch(self, fn, grid, block, args: bytearray, stream=None, shared=0):
        size = ctypes.c_size_t(len(args))
        buf = (ctypes.c_char * len(args)).from_buffer(args)
        config = (ctypes.c_void_p * 5)(
            self.HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.cast(buf, ctypes.c_void_p),
            self.HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
            self.HIP_LAUNCH_PARAM_END)
        self._check(self.hip.hipModuleLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            ctypes.c_uint(shared), ctypes.c_void_p(stream or 0), None, config),
            "hipModuleLaunchKernel")



SYMBOLS = {
    "odo": "_ZN5aiter23fmha_bwd_hd128_odo_bf16E",
    "dqdkdv": "_ZN5aiter38fmha_bwd_hd128_bf16_causal_br_a32_psskE",
    "post": "_ZN5aiter30fmha_bwd_hd128_dq_convert_bf16E",
}
CO = {
    "odo": "bwd_hd128_odo_bf16.co",
    "dqdkdv": "bwd_hd128_bf16_causal_br_a32_pssk.co",
    "post": "bwd_hd128_dq_convert_bf16.co",
}


def asm_backward(q, k, v, o, do, lse, softmax_scale=None, hip=None, dkdv_heads="kv"):
    """Run the three-kernel ASM backward. Returns (dq, dk, dv).

    q/k/v/o/do are [B, S, H, D] bf16 as Primus-Turbo lays them out; lse is
    [B, Hq, Sq] fp32, or Primus-Turbo's packed [B, Hq, 2*Sq] scratch, which is
    gathered exactly as attention_fused_bwd_impl does rather than duplicating
    the layout knowledge.

    NEVER RUN. Written while the card was wedged; the first execution is a
    bring-up. Risks, in order of suspicion, are in
    output/0913__opt_plan__claude/phase2/T2-ASM-BACKWARD-SPEC.md.
    """
    import math

    import torch

    batch, seqlen_q, nhead_q, head_dim = q.shape
    seqlen_k, nhead_k = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    for name, t in (("q", q), ("k", k), ("v", v), ("o", o), ("do", do)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous; the byte strides below assume it")

    if lse.dim() == 3 and lse.shape[2] == 2 * seqlen_q:
        from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
            _packed_lse_index,
        )
        lse = lse.gather(2, _packed_lse_index(seqlen_q, lse.device)
                         .view(1, 1, -1).expand(batch, nhead_q, -1))
    lse = lse.contiguous().float()

    # 514 buffer_atomic_add_f32 accumulate into this, so it must be fp32 and zeroed.
    dq_acc = torch.zeros((batch, nhead_q, seqlen_q, head_dim), device=q.device, dtype=torch.float32)
    delta = torch.empty((batch, nhead_q, seqlen_q), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    # Under GQA, ratio q heads share one kv head, and the main kernel's grid is
    # (kv_tiles, nhead_q, batch) -- so `ratio` workgroups would write the same dk/dv tile.
    # dkdv_heads="q" gives each q head its own slice, which the caller then reduces.
    # Measured on 0915: with ratio=1 all three tensors come back at ~52 dB, while at ratio=4
    # dq stays correct and dk/dv collapse to about -0.3 dB, which is what an unsynchronised
    # 4-way overwrite of the same tile looks like.
    if dkdv_heads == "q":
        dk = torch.zeros((batch, seqlen_k, nhead_q, head_dim), device=k.device, dtype=k.dtype)
        dv = torch.zeros((batch, seqlen_k, nhead_q, head_dim), device=v.device, dtype=v.dtype)
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

    def bs(t, elem):  # byte strides, in the host's (batch, seq, head) order
        return (t.stride(0) * elem, t.stride(1) * elem, t.stride(2) * elem)

    b_q, s_q, h_q = bs(q, BF16)
    b_k, s_k, h_k = bs(k, BF16)
    b_v, s_v, h_v = bs(v, BF16)
    b_do, s_do, h_do = bs(do, BF16)
    b_o, s_o, h_o = bs(o, BF16)
    b_dk, s_dk, h_dk = bs(dk, BF16)
    b_dv, s_dv, h_dv = bs(dv, BF16)
    h_lsed = seqlen_q * FP32

    hip = hip or HipModule()
    f_odo = hip.function(ASM_DIR / CO["odo"], SYMBOLS["odo"])
    f_main = hip.function(ASM_DIR / CO["dqdkdv"], SYMBOLS["dqdkdv"])
    f_post = hip.function(ASM_DIR / CO["post"], SYMBOLS["post"])
    stream = torch.cuda.current_stream().cuda_stream

    hip.launch(f_odo, ((seqlen_q + TS_ODO - 1) // TS_ODO, nhead_q, batch), (BDX, 1, 1),
               pack_compact(ODO_FIELDS, ODO_SIZE, {
                   "ptr_o": o.data_ptr(), "ptr_do": do.data_ptr(), "ptr_d": delta.data_ptr(),
                   "Hs_o": h_o, "BAs_o": b_o, "Seqs_o": s_o,
                   "Hs_do": h_do, "BAs_do": b_do, "Seqs_do": s_do,
                   "Hs_d": h_lsed, "BAs_d": nhead_q * h_lsed, "Seqs_d": FP32,
                   "seqlen_q": seqlen_q, "head_dim": head_dim,
                   "ptr_qseq": 0, "ptr_qseq_padded": 0}), stream)

    gdx = (seqlen_k + TS_KV - 1) // TS_KV
    gdx = (gdx + 1) // 2  # causal: the host halves it for mask types 1 and 2
    main_args = {name: 0 for name, _, _ in DQDKDV_FIELDS}
    main_args.update({
        "ptr_dq": dq_acc.data_ptr(), "ptr_dk": dk.data_ptr(), "ptr_dv": dv.data_ptr(),
        "ptr_q": q.data_ptr(), "ptr_k": k.data_ptr(), "ptr_v": v.data_ptr(),
        "ptr_do": do.data_ptr(), "ptr_lse": lse.data_ptr(), "ptr_d": delta.data_ptr(),
        "scalar": float(softmax_scale), "log2e": 1.4426950408889634,
        "seq_len_q": seqlen_q, "seq_len_k": seqlen_k,
        "Ts": TS_KV * s_k, "ratio": nhead_q // nhead_k,
        "Hs_q": h_q, "BAs_q": b_q, "Seqs_q": s_q,
        "Hs_k": h_k, "BAs_k": b_k, "Seqs_k": s_k,
        "Hs_v": h_v, "BAs_v": b_v, "Seqs_v": s_v,
        "Hs_do": h_do, "BAs_do": b_do, "Seqs_do": s_do,
        "Hs_dk": h_dk, "BAs_dk": b_dk, "Seqs_dk": s_dk,
        "Hs_dv": h_dv, "BAs_dv": b_dv, "Seqs_dv": s_dv,
        "head_dim_q": head_dim, "head_dim_v": head_dim, "nhead_q": nhead_q,
        "Hs_lsed": h_lsed, "max_seq_len_dq": seqlen_q,
        "mask_x": MASK_X, "mask_y": MASK_Y,
    })
    hip.launch(f_main, (gdx, nhead_q, batch), (BDX, 1, 1),
               pack_padded(DQDKDV_FIELDS, DQDKDV_SIZE, main_args), stream)

    b_dq, s_dq, h_dq = bs(dq, BF16)
    hip.launch(f_post, ((seqlen_q + TS_DQ - 1) // TS_DQ, nhead_q, batch), (BDX, 1, 1),
               pack_padded(POST_FIELDS, POST_SIZE, {
                   "ptr_dq_acc": dq_acc.data_ptr(), "ptr_dq": dq.data_ptr(),
                   "Hs_dq_acc": seqlen_q * head_dim * FP32,
                   "BAs_dq_acc": nhead_q * seqlen_q * head_dim * FP32,
                   "Seqs_dq_acc": head_dim * FP32,
                   "Hs_dq": h_dq, "BAs_dq": b_dq, "Seqs_dq": s_dq,
                   "seqlen_q": seqlen_q, "head_dim": head_dim}), stream)
    return dq, dk, dv



# ---- forward -------------------------------------------------------------
# The prebuilt gfx1250 ASM forward. Launching it here rather than through
# aiter.ops.mha.fmha_fwd_with_sink_asm takes three things off the critical
# path: the missing jax dependency that aiter.ops.mha's import chain pulls in,
# the @compile_ops JIT build, and hipcc. The ABI is fully documented in
# csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu, whose static_assert of 132 bytes
# matches the ELF kernarg_segment_size exactly.
FWD_DIR = Path("/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_fwd_bf16")
FWD_CO = {True: "fmha_bf16_pertokenBf16_hd128_128x256_mask.co",
          False: "fmha_bf16_pertokenBf16_hd128_128x256.co"}
FWD_SYM = {True: "_ZN5aiter41fmha_bf16_pertokenBf16_hd128_128x256_maskE",
           False: "_ZN5aiter36fmha_bf16_pertokenBf16_hd128_128x256E"}

SUB_Q = 128   # Q-tile size (ts_qo), asm_fmha_fwd_with_sink.cu:265
FWD_OPT = 7   # bit0 reverse_kv | bit1 double_q | bit2 remap_xy; the host hardcodes 7

# struct KernelArgs is "#pragma pack(push, 1)" -- genuinely packed, unlike the
# 16-byte-padded backward struct. Offsets are from the .cu comments.
FWD_FIELDS = [
    ("d_addr", 0x00, 8), ("q_addr", 0x08, 8), ("k_addr", 0x10, 8), ("v_addr", 0x18, 8),
    ("lse_addr", 0x20, 8), ("scalar", 0x28, 4), ("q_seq_len", 0x2C, 4),
    ("q_seqs", 0x30, 4), ("q_ts", 0x34, 4), ("q_hs", 0x38, 4), ("q_bas", 0x3C, 4),
    ("gqa", 0x40, 4), ("k_seqs", 0x44, 4), ("k_hs", 0x48, 4), ("k_bas", 0x4C, 4),
    ("opt", 0x50, 4), ("lse", 0x54, 4), ("kv_seq_len", 0x58, 4), ("q_head_num", 0x5C, 4),
    ("v_seqs", 0x60, 4), ("v_hs", 0x64, 4), ("v_bas", 0x68, 4),
    ("d_seqs", 0x6C, 4), ("d_hs", 0x70, 4), ("d_bas", 0x74, 4), ("lse_hs", 0x78, 4),
    ("sink_addr", 0x7C, 8),
]
FWD_SIZE = 0x84  # 132


def asm_forward(q, k, v, softmax_scale=None, causal=True, hip=None):
    """Prebuilt gfx1250 ASM forward. Returns (out, lse) with lse [B, Hq, Sq] fp32.

    lse is NATURAL log, matching what the fused backward and asm_backward()
    expect, so the two chain with no relayout. NEVER RUN.
    """
    import math

    import torch

    batch, seqlen_q, nhead_q, head_dim = q.shape
    seqlen_k, nhead_k = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the ASM forward requires stride(-1) == 1 on q/k/v")

    out = torch.empty((batch, seqlen_q, nhead_q, head_dim), device=q.device, dtype=q.dtype)
    # The kernel always writes LSE, even when the caller does not want it.
    lse = torch.empty((batch, nhead_q, seqlen_q), device=q.device, dtype=torch.float32)

    def bs(t, elem):
        return (t.stride(0) * elem, t.stride(1) * elem, t.stride(2) * elem)

    q_bas, q_seqs, q_hs = bs(q, BF16)
    k_bas, k_seqs, k_hs = bs(k, BF16)
    v_bas, v_seqs, v_hs = bs(v, BF16)
    d_bas, d_seqs, d_hs = bs(out, BF16)

    hip = hip or HipModule()
    fn = hip.function(FWD_DIR / FWD_CO[causal], FWD_SYM[causal])

    q_tiles = (seqlen_q + SUB_Q - 1) // SUB_Q
    gdx = (q_tiles + 1) // 2 if (FWD_OPT & 0x2) else q_tiles   # double_q halves it
    # remap_xy: the host swaps gdx and gdy so bid.x indexes heads.
    grid = (nhead_q, gdx, batch)

    hip.launch(fn, grid, (BDX, 1, 1), pack_padded(FWD_FIELDS, FWD_SIZE, {
        "d_addr": out.data_ptr(), "q_addr": q.data_ptr(), "k_addr": k.data_ptr(),
        "v_addr": v.data_ptr(), "lse_addr": lse.data_ptr(), "sink_addr": 0,
        "scalar": float(softmax_scale), "q_seq_len": seqlen_q, "kv_seq_len": seqlen_k,
        "q_seqs": q_seqs, "q_ts": SUB_Q * q_seqs, "q_hs": q_hs, "q_bas": q_bas,
        "k_seqs": k_seqs, "k_hs": k_hs, "k_bas": k_bas,
        "v_seqs": v_seqs, "v_hs": v_hs, "v_bas": v_bas,
        "d_seqs": d_seqs, "d_hs": d_hs, "d_bas": d_bas,
        "lse_hs": seqlen_q * FP32, "gqa": nhead_q // nhead_k,
        "opt": FWD_OPT, "lse": 1, "q_head_num": nhead_q,
    }), torch.cuda.current_stream().cuda_stream)
    return out, lse


def selftest() -> int:
    """Exercise the packing without a GPU: sizes, offsets, and byte-stride units."""
    ok = True

    odo = pack_compact(ODO_FIELDS, ODO_SIZE, {n: 0x1122334455667788 if w == 8 else 0x11223344
                                              for n, w in ODO_FIELDS})
    print(f"  odo      packed {len(odo)} B (kernarg 84)      {'ok' if len(odo) == 84 else 'BAD'}")
    ok &= len(odo) == 84

    vals = {n: (0x1122334455667788 if w == 8 else 1) for n, _, w in DQDKDV_FIELDS}
    vals["scalar"], vals["log2e"] = 0.08838835, 1.4426950408889634
    dq = pack_padded(DQDKDV_FIELDS, DQDKDV_SIZE, vals)
    last = max(o + w for _, o, w in DQDKDV_FIELDS)
    print(f"  dqdkdv   packed {len(dq)} B (kernarg 704), last field ends at {last}  "
          f"{'ok' if len(dq) == 704 and last <= 704 else 'BAD'}")
    ok &= len(dq) == 704 and last <= 704
    got = struct.unpack_from("<f", dq, 144)[0]
    print(f"  scalar round-trips as float: {got:.8f}  {'ok' if abs(got - 0.08838835) < 1e-7 else 'BAD'}")
    ok &= abs(got - 0.08838835) < 1e-7

    post = pack_padded(POST_FIELDS, POST_SIZE, {n: 1 for n, _, w in POST_FIELDS})
    tail = bytes(post[0xA0:])
    print(f"  post     packed {len(post)} B (kernarg 208), bytes past 0x9f all zero: "
          f"{'ok' if tail == bytes(len(tail)) else 'BAD'}")
    ok &= len(post) == 208 and tail == bytes(len(tail))

    fwd = pack_padded(FWD_FIELDS, FWD_SIZE, {n: 0 for n, _, _ in FWD_FIELDS})
    last = max(o + w for _, o, w in FWD_FIELDS)
    print(f"  forward  packed {len(fwd)} B (kernarg 132), last field ends at {last}  "
          f"{'ok' if len(fwd) == 132 and last == 132 else 'BAD'}")
    ok &= len(fwd) == 132 and last == 132
    for p in (FWD_DIR / FWD_CO[True], FWD_DIR / FWD_CO[False]):
        print(f"  {p.name:<46} {'present' if p.exists() else 'MISSING'}")
        ok &= p.exists()

    for stem in ("bwd_hd128_odo_bf16", "bwd_hd128_bf16_causal_br_a32_pssk",
                 "bwd_hd128_dq_convert_bf16"):
        p = ASM_DIR / f"{stem}.co"
        print(f"  {stem:<38} {'present' if p.exists() else 'MISSING'}")
        ok &= p.exists()

    print("\nselftest:", "pass" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check packing and file presence; needs no GPU")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    print("Nothing to do without --selftest; import this module to launch.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
