import flydsl, flydsl.expr as fx, subprocess, os, re
print("flydsl version:", getattr(flydsl, "__version__", "?"))
print("flydsl path   :", os.path.dirname(flydsl.__file__))
root = os.path.dirname(flydsl.__file__)

print("\n--- T members, inside an MLIR Context ---")
from flydsl._mlir.ir import Context
from flydsl.expr.typing import T
with Context():
    for a in ["bf16","f16","f32","f8","i32"]:
        try:
            getattr(T, a); print(f"  OK      T.{a}")
        except AttributeError:
            print(f"  MISSING T.{a}")
        except Exception as e:
            print(f"  ERROR   T.{a}: {type(e).__name__}: {e}")

print("\n--- the four absent from flydsl.expr: where do they actually live? ---")
for name in ["ceildiv","max","min","to_llvm_ptr"]:
    r = subprocess.run(["grep","-rn",f"def {name}\\b","--include=*.py",root],
                       capture_output=True, text=True)
    hits = [l.split(":")[0].replace(root+"/","") + ":" + l.split(":")[1]
            for l in r.stdout.strip().split("\n") if l]
    print(f"  {name}: {hits[:6] if hits else 'NOT DEFINED ANYWHERE in the package'}")

print("\n--- how aiter actually calls them (so we know what a shim must match) ---")
