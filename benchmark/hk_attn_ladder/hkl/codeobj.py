###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read the kernel table of an AMDGPU code object (``.hsaco``) without any ROCm tool.

    kernels(path) -> {name: {"kernarg_size", "args", "lds", "scratch", "vgpr", "sgpr",
                             "vgpr_spill", "sgpr_spill", "max_block", "wavefront_size"}}

The table is the code object's AMDGPU metadata note (NT_AMDGPU_METADATA = 32, owner "AMDGPU"), a
msgpack map (``amdhsa.kernels``); ``args`` is the list of ``{"name", "offset", "size",
"value_kind"}`` in kernarg order. Standard library only (a minimal ELF64 note reader and msgpack
decoder), so hosts without torch / llvm-readelf can check a code object.
"""

import struct

__all__ = ["CodeObjectError", "kernels", "metadata"]

NT_AMDGPU_METADATA = 32


class CodeObjectError(ValueError):
    pass


# ---------------------------------------------------------------- msgpack (decode only)


def _mp(buf, i):
    b = buf[i]
    i += 1
    if b <= 0x7F:
        return b, i
    if b >= 0xE0:
        return b - 0x100, i
    if 0x80 <= b <= 0x8F:
        return _mp_map(buf, i, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return _mp_arr(buf, i, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return buf[i : i + n].decode(), i + n
    if b == 0xC0:
        return None, i
    if b in (0xC2, 0xC3):
        return b == 0xC3, i
    if b in (0xC4, 0xC5, 0xC6):  # bin 8/16/32
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(buf[i : i + w], "big")
        i += w
        return bytes(buf[i : i + n]), i + n
    if b == 0xCA:
        return struct.unpack(">f", buf[i : i + 4])[0], i + 4
    if b == 0xCB:
        return struct.unpack(">d", buf[i : i + 8])[0], i + 8
    if 0xCC <= b <= 0xCF:  # uint 8..64
        w = 1 << (b - 0xCC)
        return int.from_bytes(buf[i : i + w], "big"), i + w
    if 0xD0 <= b <= 0xD3:  # int 8..64
        w = 1 << (b - 0xD0)
        return int.from_bytes(buf[i : i + w], "big", signed=True), i + w
    if b in (0xD9, 0xDA, 0xDB):  # str 8/16/32
        w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[b]
        n = int.from_bytes(buf[i : i + w], "big")
        i += w
        return buf[i : i + n].decode(), i + n
    if b in (0xDC, 0xDD):
        w = 2 if b == 0xDC else 4
        return _mp_arr(buf, i + w, int.from_bytes(buf[i : i + w], "big"))
    if b in (0xDE, 0xDF):
        w = 2 if b == 0xDE else 4
        return _mp_map(buf, i + w, int.from_bytes(buf[i : i + w], "big"))
    raise CodeObjectError(f"msgpack: unsupported type byte 0x{b:02x} at {i - 1}")


def _mp_arr(buf, i, n):
    out = []
    for _ in range(n):
        v, i = _mp(buf, i)
        out.append(v)
    return out, i


def _mp_map(buf, i, n):
    out = {}
    for _ in range(n):
        k, i = _mp(buf, i)
        v, i = _mp(buf, i)
        out[k] = v
    return out, i


# ---------------------------------------------------------------- ELF64 notes


def _notes(data):
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise CodeObjectError("not a little-endian ELF64 file")
    (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", data, 0x3A)
    for s in range(e_shnum):
        sh = e_shoff + s * e_shentsize
        (sh_type,) = struct.unpack_from("<I", data, sh + 4)
        if sh_type != 7:  # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", data, sh + 0x18)
        p, end = off, off + size
        while p + 12 <= end:
            namesz, descsz, ntype = struct.unpack_from("<III", data, p)
            p += 12
            name = data[p : p + namesz].rstrip(b"\0").decode(errors="replace")
            p += (namesz + 3) & ~3
            desc = data[p : p + descsz]
            p += (descsz + 3) & ~3
            yield name, ntype, desc


def metadata(path):
    """The decoded AMDGPU metadata map of a code object."""
    with open(path, "rb") as f:
        data = f.read()
    for name, ntype, desc in _notes(data):
        if name == "AMDGPU" and ntype == NT_AMDGPU_METADATA:
            v, _ = _mp(desc, 0)
            return v
    raise CodeObjectError(f"{path}: no AMDGPU metadata note")


def kernels(path):
    """{kernel name: resource and kernarg summary} of a code object (see the module docstring)."""
    out = {}
    for k in metadata(path).get("amdhsa.kernels", []):
        args = [
            {
                "name": a.get(".name", f"arg{j}"),
                "offset": int(a[".offset"]),
                "size": int(a[".size"]),
                "value_kind": a.get(".value_kind", "?"),
            }
            for j, a in enumerate(k.get(".args", []))
        ]
        out[k[".name"]] = {
            "kernarg_size": int(k.get(".kernarg_segment_size", 0)),
            "args": args,
            "lds": int(k.get(".group_segment_fixed_size", 0)),
            "scratch": int(k.get(".private_segment_fixed_size", 0)),
            "vgpr": int(k.get(".vgpr_count", -1)),
            "sgpr": int(k.get(".sgpr_count", -1)),
            "vgpr_spill": int(k.get(".vgpr_spill_count", 0)),
            "sgpr_spill": int(k.get(".sgpr_spill_count", 0)),
            "max_block": int(k.get(".max_flat_workgroup_size", 0)),
            "wavefront_size": int(k.get(".wavefront_size", 0)),
            "dynamic_stack": bool(k.get(".uses_dynamic_stack", False)),
        }
    return out
