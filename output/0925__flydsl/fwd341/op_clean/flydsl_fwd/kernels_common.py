"""Common helpers shared by kernel modules.

Trimmed to what this package uses (``LOG2E``, ``create_llvm_ptr``); the aiter
original also carries arch/dtype lookups, int32 atomics and a compile cache that
nothing here calls -- see ../PROVENANCE.md.
"""

import flydsl.expr as fx
from flydsl.expr import as_ir_value

LOG2E = 1.4426950408889634


# LLVM address-space numbers as fx spaces: Global(1) and Shared, which is 2 in
# fx terms but lowers to !llvm.ptr<3>. to_llvm_ptr resolves it, so the backend's
# number never appears at a call site.
FX_ADDRESS_SPACE = {1: fx.AddressSpace.Global, 3: fx.AddressSpace.Shared}


def create_llvm_ptr(value, address_space=1):
    """Raw LLVM pointer for atomics and intrinsic APIs."""
    # Accept either the LLVM number (1 global / 3 LDS) or an fx.AddressSpace,
    # so a caller cannot silently pass the wrong one.
    space = FX_ADDRESS_SPACE.get(address_space, address_space)
    pt = fx.PointerType.get(fx.Int32.ir_type, address_space=space, alignment=4)
    return as_ir_value(fx.to_llvm_ptr(fx.inttoptr(pt, value)))
