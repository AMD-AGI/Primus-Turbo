#!/usr/bin/env python3
"""Read the call contract of aiter's prebuilt gfx1250 ASM attention backward.

These .co files are hand-written assembly shipped as gfx1250 ELF objects. There
is no Python-side architecture gate for them (the C++ host special-cases
gfx1250 in five places, the Python wrapper does not), so reaching them means
launching them directly -- which means knowing the kernarg layout exactly. The
layout is in the ELF metadata, so read it rather than transcribing it.

    python tools/gfx1250/asm_bwd_abi.py                  # human-readable
    python tools/gfx1250/asm_bwd_abi.py --emit-packer    # generated struct code

Needs only llvm-readelf; no GPU, no ROCm runtime.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ASM_DIR = Path("/home/lihuzhan/code/aiter-src/hsa/gfx1250/fmha_v3_bwd")
READELF = "/opt/rocm/llvm/bin/llvm-readelf"

# The three launches that make up one backward pass, in order. Established from
# csrc/cpp_itfs/mha_bwd.cu: the ASM path is not a drop-in single kernel.
PIPELINE = ("bwd_hd128_odo_bf16", "bwd_hd128_bf16_causal_br_a32_pssk", "bwd_hd128_dq_convert_bf16")

# Two packing conventions coexist, which is the trap here. The dqdkdv kernel
# publishes per-field offsets in its ELF metadata and pads every field to 16
# bytes. The odo and dq_convert kernels publish no field metadata at all, and
# the host packs them COMPACTLY -- use_compact_fmha_bwd_kernel_args() in
# csrc/cpp_itfs/mha_bwd.cu:68 returns true for exactly gfx1250. Sizes check out:
# odo is 3 pointers + 11 u32 + 2 pointers = 84 bytes, its kernarg_segment_size.
# Transcribed from pack_fmha_bwd_odo_args(), mha_bwd.cu:95.
COMPACT_ODO_FIELDS = (
    ("ptr_o", 8), ("ptr_do", 8), ("ptr_d", 8),
    ("Hs_o", 4), ("BAs_o", 4), ("Seqs_o", 4),
    ("Hs_do", 4), ("BAs_do", 4), ("Seqs_do", 4),
    ("Hs_d", 4), ("BAs_d", 4), ("Seqs_d", 4),
    ("seqlen_q", 4), ("head_dim", 4),
    ("ptr_qseq", 8), ("ptr_qseq_padded", 8),
)


def read_metadata(co: Path) -> dict:
    notes = subprocess.run([READELF, "--notes", str(co)], capture_output=True, text=True).stdout
    kern = {"args": []}
    for key in ("kernarg_segment_size", "group_segment_fixed_size", "private_segment_fixed_size",
                "vgpr_count", "sgpr_count", "wavefront_size", "max_flat_workgroup_size"):
        m = re.search(rf"\.{key}:\s*(\S+)", notes)
        if m:
            kern[key] = int(m.group(1))
    m = re.search(r"\.symbol:\s*(\S+)", notes)
    kern["symbol"] = m.group(1) if m else None
    # Argument records do not use a fixed field order -- the dqdkdv kernel
    # leads with .name, the odo kernel leads with .actual_access -- so parse
    # each record as a block rather than matching a name/offset/size sequence.
    body = notes.split(".args:", 1)[1] if ".args:" in notes else ""
    body = body.split("amdhsa.target", 1)[0]
    for block in re.split(r"\n\s*-\s+(?=\.)", body):
        name = re.search(r"\.name:\s*(\S+)", block)
        offset = re.search(r"\.offset:\s*(\d+)", block)
        size = re.search(r"\.size:\s*(\d+)", block)
        if name and offset and size and not name.group(1).startswith("_ZN"):
            kern["args"].append({"name": name.group(1),
                                 "offset": int(offset.group(1)),
                                 "size": int(size.group(1))})
    kern["args"].sort(key=lambda a: a["offset"])
    return kern


def emit_packer(kern: dict) -> str:
    """Generate the exact struct.pack_into calls for this kernel's kernarg buffer.

    Every field is 16-byte aligned regardless of its size, so a naive packed
    struct of the field types would be wrong; the offsets are authoritative.
    """
    lines = [f"# generated from {kern['symbol']}",
             f"KERNARG_SIZE = {kern['kernarg_segment_size']}",
             "def pack(buf, **kw):",
             "    import struct"]
    for a in kern["args"]:
        fmt = {8: "<Q", 4: "<I"}.get(a["size"])
        if fmt is None:
            lines.append(f"    # UNHANDLED size {a['size']} for {a['name']}")
            continue
        lines.append(f"    struct.pack_into({fmt!r}, buf, {a['offset']}, kw[{a['name']!r}])")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-packer", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = {}
    for stem in PIPELINE:
        co = ASM_DIR / f"{stem}.co"
        if not co.exists():
            print(f"missing: {co}", file=sys.stderr)
            continue
        out[stem] = read_metadata(co)

    if args.json:
        print(json.dumps(out, indent=2))
        return
    if args.emit_packer:
        for stem, kern in out.items():
            print(f"\n# ---- {stem} ----")
            print(emit_packer(kern))
        return

    off = 0
    print("bwd_hd128_odo_bf16 compact layout (from C++, not from ELF):")
    for name, size in COMPACT_ODO_FIELDS:
        print(f"    +{off:<4} {size}B  {name}")
        off += size
    print(f"    total {off} B\n")

    for stem, kern in out.items():
        print(f"{stem}")
        print(f"  symbol   {kern['symbol']}")
        n = len(kern["args"])
        note = "" if n else "   <- no field metadata; layout comes from the C++ packer"
        print(f"  kernarg  {kern['kernarg_segment_size']} B, {n} fields{note}")
        print(f"  lds      {kern['group_segment_fixed_size']} B"
              + ("   <- the entire 320 KB CU LDS: one workgroup per CU"
                 if kern["group_segment_fixed_size"] >= 327680 else ""))
        print(f"  vgpr     {kern['vgpr_count']}   sgpr {kern['sgpr_count']}   "
              f"wave{kern['wavefront_size']}   scratch {kern['private_segment_fixed_size']} B")


if __name__ == "__main__":
    main()
