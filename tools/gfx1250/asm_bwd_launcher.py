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
TS_ODO = 128  # odo preprocess tile
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
