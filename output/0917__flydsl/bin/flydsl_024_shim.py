"""Make aiter's gfx1250 FlyDSL kernels build against flydsl 0.2.4.

aiter pins flydsl==0.3.2. The delta on the gfx1250 forward path is NOT missing
functionality -- it is renames and one helper that 0.2.4 spells differently.

Two-phase, and the order matters:

  install_expr_shims()   BEFORE importing aiter. These names are resolved inside
                         kernel bodies at JIT BUILD time, not at import, so a
                         missing one surfaces as an AttributeError during the
                         first build rather than as an ImportError.

  patch_create_llvm_ptr()  AFTER importing aiter's kernel modules. Two of them do
                         `from ..kernels_common import create_llvm_ptr` at import
                         time, binding the name in their own namespace. Patching
                         only kernels_common leaves those two calling the
                         original -- the patch is accepted, the module attribute
                         reads back correct, and the build still uses the old one.
                         It patches all three and VERIFIES what the consumers see.

Each entry records what it was checked against, so a later flydsl bump can tell a
still-needed shim from a stale one. Checked against flydsl 0.2.4.
"""
import flydsl.expr as fx

installed = []


def install_expr_shims():
    """Call BEFORE `import aiter`."""
    # These must PRESERVE fx typing. aiter keeps doing fx arithmetic on the results
    # (`fx.Int32(n_tiles) - start_tile`, `(qmax_min + fx.Int32(1)) // ...`), so a shim
    # that returns a raw ir.Value or an IntTuple fails later and far away, with a
    # TypeError naming a type the call site never mentions.
    #
    # NOT `fx.ceil_div`: it carries @coerce_int_tuple_args and returns an IntTuple.
    # Arithmetic keeps the operands' fx type, and `//` is already used this way
    # throughout the aiter kernel.
    if not hasattr(fx, "ceildiv"):
        def _ceildiv(a, b):
            return (a + (b - 1)) // b

        fx.ceildiv = _ceildiv
        installed.append("fx.ceildiv -> (a + (b-1)) // b")

    # 0.3.x `fx.max` / `fx.min` are the INTEGER ops. 0.2.4 exposes only float
    # (`maxnumf`) and the atomics on `fx`; the integer ops live in the arith dialect
    # as maxsi/minsi (signed) and maxui/minui (unsigned).
    #
    # `fx.select` is NOT a ternary here -- in 0.2.4 it selects from an IntTuple -- so
    # the result has to be rebuilt as an fx numeric by hand. The construction pattern
    # is numeric.py's own: `out_type(op(extract(lhs), extract(rhs)))`.
    if not hasattr(fx, "max") or not hasattr(fx, "min"):
        from flydsl.expr import arith as _arith
        from flydsl.expr import numeric as _num

        def _mk(signed_op, unsigned_op, label):
            def f(a, b):
                # Either operand may be a plain Python int; adopt the other's fx type.
                if not isinstance(a, _num.Numeric):
                    a = type(b)(a)
                if not isinstance(b, _num.Numeric):
                    b = type(a)(b)
                out_type = type(a)
                signed = bool(getattr(a, "signed", True))
                op = signed_op if signed else unsigned_op
                # The MLIR builders need real ir.Values. numeric.py's own
                # `_extract_arith` hands back an ArithValue wrapper, which the
                # Python operators accept and `arith.maxsi` does not.
                lv = fx.as_ir_value(a)
                rv = fx.as_ir_value(b)
                return out_type(op(lv, rv))

            f.__name__ = label
            return f

        if not hasattr(fx, "max"):
            fx.max = _mk(_arith.maxsi, _arith.maxui, "max")
            installed.append("fx.max -> arith.max{si,ui}, rewrapped as the operand type")
        if not hasattr(fx, "min"):
            fx.min = _mk(_arith.minsi, _arith.minui, "min")
            installed.append("fx.min -> arith.min{si,ui}, rewrapped as the operand type")
    return installed


def _create_llvm_ptr_024(value, address_space=1):
    """aiter's create_llvm_ptr, expressed in 0.2.4 terms.

    aiter builds it as `to_llvm_ptr(inttoptr(PointerType.get(...), value))`, and
    0.2.4 has no `fx.to_llvm_ptr`. It does have the whole operation as one
    function -- `flydsl.expr.buffer_ops.create_llvm_ptr(value, address_space)` --
    which takes the LLVM address-space NUMBER directly and returns an ir.Value.

    aiter's own comment says Shared is 2 in fx terms and lowers to !llvm.ptr<3>,
    and its callers already pass the LLVM numbers (1 global, 3 LDS), so passing
    them straight through is the same mapping, not a new one.
    """
    from flydsl.expr import buffer_ops as _bops

    if not isinstance(address_space, int):
        address_space = {fx.AddressSpace.Global: 1, fx.AddressSpace.Shared: 3}[address_space]
    return _bops.create_llvm_ptr(value, address_space=address_space)


def patch_create_llvm_ptr():
    """Call AFTER aiter's gfx1250 kernel modules are imported. Returns the
    namespaces patched, and raises if any consumer still holds the old binding."""
    import importlib

    targets = [
        "aiter.ops.flydsl.kernels.kernels_common",
        "aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_fwd_prefill_a16w16_m32x8",
        "aiter.ops.flydsl.kernels.fmha_gfx1250.fmha_b16_buffer_managers",
    ]
    patched = []
    for name in targets:
        mod = importlib.import_module(name)
        if getattr(mod, "create_llvm_ptr", None) is not None:
            mod.create_llvm_ptr = _create_llvm_ptr_024
            patched.append(name.rsplit(".", 1)[-1])

    # Verify what the CONSUMER reads, not what we set. A patch that does nothing
    # looks exactly like a patch that worked.
    for name in targets:
        mod = importlib.import_module(name)
        fn = getattr(mod, "create_llvm_ptr", None)
        if fn is not None and fn is not _create_llvm_ptr_024:
            raise RuntimeError(
                f"create_llvm_ptr patch did not take in {name}: still {fn!r}. "
                "That module bound the name at import time."
            )
    installed.append("create_llvm_ptr -> buffer_ops.create_llvm_ptr in " + ", ".join(patched))
    return patched
