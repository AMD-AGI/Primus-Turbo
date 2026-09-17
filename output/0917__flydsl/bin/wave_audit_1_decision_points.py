import flydsl, os, subprocess
root = os.path.dirname(flydsl.__file__)
print("=== flydsl 0.2.4 internal wave-size decision points ===")
for pat in ["is_rdna_arch","wave_size","warp_size","WAVE_SIZE","WARP_SIZE","wavefront"]:
    r = subprocess.run(["grep","-rn",pat,"--include=*.py",root], capture_output=True, text=True)
    lines = [l.replace(root+"/","") for l in r.stdout.strip().split("\n") if l]
    print(f"\n-- {pat}: {len(lines)} hit(s)")
    for l in lines[:12]: print("   ", l[:150])
