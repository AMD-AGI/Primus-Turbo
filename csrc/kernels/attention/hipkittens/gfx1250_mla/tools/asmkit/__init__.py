###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit -- the gfx1250 assembly toolkit of the kernel generators (tools/fwdgen, tools/bwdgen).

Modules (standard library only):
  isa         the ``.s`` / ``llvm-objdump`` text model: true VGPR numbers, VGPR and scalar defs / uses, classes;
  expert      the expert scheduling mode in-flight model (va_vdst, vm_vsrc, counters) and the basic blocks;
  emit        the derived lines of a generated block (s_set_vgpr_msb, waits, v_nop) and their re-check;
  build       VGPR allocation, and assembly with the clang driver's own -cc1as / lld commands;
  verify_obj  the code object holds the intended instructions (true registers, MSB immediates, branch targets).
"""

__all__ = ["isa", "expert", "emit", "build", "verify_obj"]
