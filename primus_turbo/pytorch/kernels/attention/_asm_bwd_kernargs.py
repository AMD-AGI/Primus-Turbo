"""ctypes launch machinery for aiter's prebuilt gfx1250 ASM backward.

Moved here from tools/gfx1250/asm_bwd_launcher.py when the kernels went behind the
dispatcher: product code must not import from tools/. The tools script now re-exports
from this module, so the bring-up harness and the ABI self-test keep working unchanged.

There is no Python arch gate in aiter that reaches these kernels -- can_impl_fmha_v3_bwd
starts from get_gfx() == "gfx942" and the only widening is gfx950, so gfx1250 can never
select them -- which is why they are loaded and launched by hand rather than called.
"""

import argparse
import ctypes
import struct
import sys
from pathlib import Path

def _asm_dir() -> Path:
    """Where aiter keeps its prebuilt gfx1250 objects, derived from the installed aiter
    rather than hardcoded: the launcher carried an absolute /home path, which is fine for a
    bring-up script and not for product code."""
    try:
        import aiter
        cand = Path(aiter.__file__).resolve().parent.parent / "hsa" / "gfx1250" / "fmha_v3_bwd"
        if cand.is_dir():
            return cand
    except Exception:
        pass
    return Path("/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd")


ASM_DIR = _asm_dir()

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

# mask=0 in fmha_bwd_dqdkdv.csv -- same hdim, same 32x128 tile, different mangled symbol.
# The causal object halves the x grid because the host does; the non-causal one must not.
CO_NONCAUSAL = "bwd_hd128_bf16_a32_pssk.co"
SYM_NONCAUSAL = "_ZN5aiter28fmha_bwd_hd128_bf16_a32_psskE"


def asm_backward(q, k, v, o, do, lse, softmax_scale=None, hip=None, dkdv_heads="kv",
                 co_variant="", causal=True, grid_halve=None):
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
    # co_variant="_perf" selects bwd_hd128_bf16_causal_br_a32_pssk_perf.co. It exports the
    # SAME mangled symbol as the shipped one, so the ABI is unchanged and only the file
    # swaps. Neither _perf object is mentioned in any of the campaign documents -- they are
    # an unexplored free variable, not a known-better build.
    if causal:
        main_co, main_sym = CO["dqdkdv"], SYMBOLS["dqdkdv"]
    else:
        main_co, main_sym = CO_NONCAUSAL, SYM_NONCAUSAL
    if co_variant:
        main_co = main_co.replace(".co", f"{co_variant}.co")
    f_main = hip.function(ASM_DIR / main_co, main_sym)
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
    # grid_halve=None follows the host: halve for mask types 1 and 2 only. Overridable
    # because the undocumented _perf build returns dq at 5.84 dB -- partially right rather
    # than garbage, which is what missing contributions look like -- while returning dk and
    # dv BIT-identical to the shipped build. If _perf assigns dq where the shipped one
    # accumulates atomically, a halved grid would drop half of it.
    if grid_halve is None:
        grid_halve = causal
    if grid_halve:
        gdx = (gdx + 1) // 2
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



