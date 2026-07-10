import re
import primus_turbo.flydsl.attention.flash_attn_bwd_kernel as _k
rocdl = _k.rocdl
names = [n for n in dir(rocdl) if "mfma" in n.lower()]
print("=== all mfma intrinsics ===")
for n in sorted(names):
    print(" ", n)
print("=== f32/xf32/f8/f6 K-width variants ===")
for n in sorted(names):
    if re.search(r"xf32|_f32$|f8|f6|f4|32x32x|16x16x", n):
        print(" ", n)
