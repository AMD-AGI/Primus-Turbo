import flydsl.expr as fx
from flydsl._mlir.ir import Context
print("--- sub-attributes ---")
pairs = [("AddressSpace","Global"),("AddressSpace","Shared"),("PointerType","get"),
         ("Vector","filled"),("Vector","from_elements"),("Vector","make_type"),
         ("Int32","ir_type"),("Float8E4M3FN","ir_type"),("Float8E4M3FNUZ","ir_type")]
with Context():
    for obj, attr in pairs:
        o = getattr(fx, obj, None)
        print(f"  {'OK     ' if (o is not None and hasattr(o, attr)) else 'MISSING'} fx.{obj}.{attr}")
print("\n--- rocdl symbols aiter uses ---")
import flydsl.expr.rocdl as rd
for a in ["WMMA","MFMA","sched_barrier","s_wait_asynccnt","s_wait_dscnt","ds_load_tr16_b128",
          "make_tdm_atom","cluster_load_async_to_lds"]:
    print(f"  {'OK     ' if hasattr(rd, a) else 'MISSING'} rocdl.{a}")
print("\n--- tdm_ops members ---")
import flydsl.expr.rocdl.tdm_ops as t
print("  ", [m for m in dir(t) if not m.startswith('_')][:25])
