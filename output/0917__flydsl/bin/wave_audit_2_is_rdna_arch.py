import flydsl, os
root = os.path.dirname(flydsl.__file__)
print("=== runtime/device.py is_rdna_arch ===")
src = open(os.path.join(root, "runtime/device.py")).read().split("\n")
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(src[69:92])))
print("\n=== expr/buffer_ops.py:60-80 ===")
src2 = open(os.path.join(root, "expr/buffer_ops.py")).read().split("\n")
print("\n".join(f"{i+1}: {l}" for i, l in enumerate(src2[59:80], start=60)))
print("\n=== behaviour on the strings that matter ===")
from flydsl.runtime.device import is_rdna_arch
for a in ["gfx1250","gfx1201","gfx1200","gfx950","gfx942","gfx1100"]:
    print(f"  is_rdna_arch({a!r}) = {is_rdna_arch(a)}   -> warp_size {32 if is_rdna_arch(a) else 64}")
