###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit.build -- assemble an edited compiler ``.s`` into a code object exactly as the clang driver does.

The driver's own ``-cc1as`` and ``lld`` commands are read from ``<compile command> -###`` (as hsaco.py does), so an
unedited ``.s`` gives the compiler's code object byte for byte. The notes (``llvm-readelf --notes``) and the listing
(``llvm-objdump -d``) are written next to the code object."""

import hashlib
import re
import shlex
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_vgpr_count(lines, n):
    """Raise the kernel's VGPR allocation to at least n (``.amdhsa_next_free_vgpr`` and ``.vgpr_count``)."""
    out = []
    for l in lines:
        m = re.match(r"(\s*\.amdhsa_next_free_vgpr\s+)(\d+)(.*)", l)
        if m and int(m.group(2)) < n:
            l = f"{m.group(1)}{n}{m.group(3)}"
        m = re.match(r"(\s*\.vgpr_count:\s+)(\d+)(.*)", l)
        if m and int(m.group(2)) < n:
            l = f"{m.group(1)}{n}{m.group(3)}"
        out.append(l)
    return out


def driver_steps(compile_cmd, cwd):
    """(cc1as argv, lld argv) of a device-only compile command (run with -###; nothing is built)."""
    dry = subprocess.run(list(compile_cmd) + ["-###"], cwd=cwd, capture_output=True, text=True).stderr
    steps = [shlex.split(line) for line in dry.splitlines() if line.startswith(' "')]
    cc1as = next(s for s in steps if "-cc1as" in s)
    lld = next(s for s in steps if s[0].endswith("lld"))
    return cc1as, lld


def assemble(s_path, out_path, cc1as, lld, cwd="."):
    """s_path -> out_path (code object) with the driver's commands; writes <out>.notes and <out>.objdump.s. Returns
    {sha256, vgpr_count, sgpr_count, vgpr_spill_count, sgpr_spill_count, private_segment_fixed_size,
    group_segment_fixed_size}."""
    s_path, out_path = Path(s_path).resolve(), Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj = out_path.with_suffix(".o")
    a = list(cc1as)
    a[a.index("-o") + 1] = str(obj)
    a[-1] = str(s_path)
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cc1as failed ({r.returncode}):\n{r.stderr[-4000:]}")
    b = [t for t in lld if t != "-save-temps"]
    b[b.index("-o") + 1] = str(out_path)
    b[b.index("--no-whole-archive") - 1] = str(obj)
    r = subprocess.run(b, cwd=cwd, capture_output=True, text=True)
    obj.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"lld failed ({r.returncode}):\n{r.stderr[-4000:]}")
    llvm_bin = Path(cc1as[0]).parent
    meta = {"sha256": sha256(out_path)}
    notes = subprocess.run(
        [str(llvm_bin / "llvm-readelf"), "--notes", str(out_path)], capture_output=True, text=True, check=True
    ).stdout
    out_path.with_suffix(".notes").write_text(notes)
    for k in (
        "vgpr_count",
        "sgpr_count",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "private_segment_fixed_size",
        "group_segment_fixed_size",
    ):
        m = re.search(r"\.%s:\s+(\d+)" % k, notes)
        if m:
            meta[k] = int(m.group(1))
    dis = subprocess.run(
        [str(llvm_bin / "llvm-objdump"), "-d", "--mcpu=gfx1250", str(out_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    Path(str(out_path)[: -len(out_path.suffix)] + ".objdump.s").write_text(dis)
    return meta
