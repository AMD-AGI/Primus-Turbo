"""Symbol-existence audit of the flydsl API surface aiter's gfx1250 forward needs.

Scope note, stated up front so the result is not over-read: this checks whether a
SYMBOL EXISTS in the installed flydsl. It does not resolve a compilation target and
does not depend on one, so no gfx1250 assertion applies here. Any audit that DOES
resolve a target must assert the resolved string, because without a device flydsl
either raises or silently falls back to a default.
"""
import importlib, sys

try:
    import flydsl
    ver = getattr(flydsl, "__version__", "?")
except Exception as e:
    print(f"FATAL: import flydsl failed: {type(e).__name__}: {e}")
    sys.exit(2)
print(f"flydsl version: {ver}")
print(f"flydsl __file__: {flydsl.__file__}")

MODULES = [
    "flydsl.compiler", "flydsl.expr", "flydsl.expr.arith", "flydsl.expr.gpu",
    "flydsl.expr.rocdl", "flydsl.expr.rocdl.tdm_ops", "flydsl.expr.math",
    "flydsl.expr.typing", "flydsl.expr.utils.arith", "flydsl.expr.primitive",
    "flydsl._mlir.ir", "flydsl._mlir.dialects.builtin", "flydsl._mlir.dialects.gpu",
    "flydsl._mlir.dialects.llvm", "flydsl.runtime.device",
]
# (module, attribute)
ATTRS = [
    ("flydsl.compiler", a) for a in ["jit", "kernel"]
] + [
    ("flydsl.expr", a) for a in [
        "add_offset","AddressSpace","BFloat16","ceildiv","copy_atom_call","Float16",
        "Float32","Float8E4M3FN","Float8E4M3FNUZ","get_iter","Index","Int32","Int64",
        "inttoptr","log2","make_layout","make_view","max","min","Pointer","PointerType",
        "ptr_load","ptrtoint","range_constexpr","SharedAllocator","Stream","Tensor",
        "to_llvm_ptr","Uint32","Vector","as_ir_value",
    ]
] + [
    ("flydsl.expr.rocdl", a) for a in ["make_tdm_atom", "tdm_ops"]
] + [
    ("flydsl.expr.typing", a) for a in ["T"]
] + [
    ("flydsl.expr.utils.arith", "_to_raw"),
    ("flydsl.expr.primitive", "const_expr"),
    ("flydsl.runtime.device", "get_rocm_arch"),
    ("flydsl.runtime.device", "is_rdna_arch"),
]
T_ATTRS = ["bf16", "f16", "f32", "f8", "i32"]

print("\n--- modules ---")
mods, missing_mod = {}, []
for m in MODULES:
    try:
        mods[m] = importlib.import_module(m); print(f"  OK      {m}")
    except Exception as e:
        missing_mod.append(m); print(f"  MISSING {m}  ({type(e).__name__}: {e})")

print("\n--- attributes ---")
missing_attr = []
for m, a in ATTRS:
    if m not in mods:
        print(f"  SKIP    {m}.{a}  (module missing)"); continue
    if hasattr(mods[m], a):
        print(f"  OK      {m}.{a}")
    else:
        missing_attr.append(f"{m}.{a}"); print(f"  MISSING {m}.{a}")

print("\n--- flydsl.expr.typing.T members ---")
missing_T = []
if "flydsl.expr.typing" in mods and hasattr(mods["flydsl.expr.typing"], "T"):
    T = mods["flydsl.expr.typing"].T
    for a in T_ATTRS:
        if hasattr(T, a): print(f"  OK      T.{a}")
        else: missing_T.append(f"T.{a}"); print(f"  MISSING T.{a}")

print("\n--- gfx1250 primitives in the package ---")
import subprocess, os
root = os.path.dirname(flydsl.__file__)
for pat in ["MmaOpGFX1250","ds_load_tr16_b128","make_tdm_atom","wmma_f32_16x16x32",
            "s_wait_asynccnt","cluster_load_async_to_lds","sched_barrier"]:
    r = subprocess.run(["grep","-rl",pat,root], capture_output=True, text=True)
    n = len([x for x in r.stdout.split() if x])
    print(f"  {'OK     ' if n else 'ABSENT '} {pat}: {n} file(s)")

print("\n=== SUMMARY ===")
print(f"missing modules   : {missing_mod or 'none'}")
print(f"missing attributes: {missing_attr or 'none'}")
print(f"missing T members : {missing_T or 'none'}")
