import re, sys, shutil
path = sys.argv[1]
src = open(path).read()
if "OE_PHYS_GPU" in src:
    print("already", path); sys.exit()
import os; os.path.exists(path + ".bak.pre-b0-sclk") or shutil.copy(path, path + ".bak.pre-b0-sclk")
old = re.search(r"    \"\"\"Current shader clock, or -1 if it cannot be read\.[^\n]*\"\"\"\n    try:\n", src).group(0)
new = ('    """Current shader clock of the card under test, or -1 if it cannot be read.\n\n'
       '    B0 runs one container per card (fa-gN, OE_PHYS_GPU=N). rocm-smi reads sysfs for every\n'
       '    card and its first sclk line is GPU[0], so read this card\'s pp_dpm_sclk directly.\n'
       '    """\n'
       '    phys = os.environ.get("OE_PHYS_GPU")\n'
       '    if phys is not None:\n'
       '        try:\n'
       '            with open(f"/sys/class/drm/renderD{128 + 8 * int(phys)}/device/pp_dpm_sclk") as f:\n'
       '                for line in f:\n'
       '                    if line.rstrip().endswith("*"):\n'
       '                        return int(line.split(":")[1].strip().split("Mhz")[0])\n'
       '        except Exception:\n'
       '            pass\n'
       '        return -1\n'
       '    try:\n')
assert src.count(old) == 1, path
src = src.replace(old, new)
if not re.search(r"^import os\b", src, re.M):
    src = src.replace("import subprocess", "import os\nimport subprocess", 1)
open(path, "w").write(src)
print("patched", path)
