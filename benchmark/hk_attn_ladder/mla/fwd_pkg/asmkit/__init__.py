###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""asmkit -- gfx1250 assembly backend for generated / rescheduled hot loops spliced into compiler-built kernels.

Modules (standard library only; nothing is imported from torch or the kernel packages):
  isa        the ``.s`` / objdump text model: true VGPR numbers, scalar defs / uses, classes, loops, parse coverage;
  expert     the expert-scheduling-mode in-flight model (va_vdst / vm_vsrc / counters) and the CFG / liveness;
  emit       re-derivation of s_set_vgpr_msb, expert waits, XCNT waits, hazard v_nops and WMMA reuse flags for a
             given instruction order of a loop region (LLVM's rules);
  splice     shell-loop discovery, entry register map, identity rebuild, splicing of generated code, static checks
             (hazards inside the region on every path, and after every exit);
  build      assembly with the clang driver's own -cc1as / lld commands (byte-identical for an unedited .s);
  verify_obj the code object holds the intended instructions (true registers, MSB immediates, branch targets).

Command line (``PYTHONPATH=primus_turbo/hipkittens python -m asmkit -h``): loops, map, identity, generate, build,
verify-obj. The independent verification of transformed code (symbolic equivalence, bounds, preflight stamp) lives
in ``asmkit/verify`` (V track) and must not import this package's modules.
"""

__all__ = ["isa", "expert", "emit", "splice", "build", "verify_obj"]
