import flydsl, os, subprocess
root = os.path.dirname(flydsl.__file__)
r = subprocess.run(["grep","-rn","FLYDSL_GPU_ARCH","--include=*.py",root], capture_output=True, text=True)
print("=== FLYDSL_GPU_ARCH 读取点 ===")
for l in r.stdout.strip().split("\n"):
    if l: print("  ", l.replace(root+"/",""))
print("\n=== get_rocm_arch 实现 ===")
src = open(os.path.join(root,"runtime/device.py")).read().split("\n")
import re
for i,l in enumerate(src):
    if l.startswith("def get_rocm_arch"):
        print("\n".join(f"{j+1}: {src[j]}" for j in range(i, min(i+22, len(src)))))
        break
