import sys, pickle, glob, re
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
for f in sorted(glob.glob(sys.argv[1] + "/*/*.pkl")):
    obj = pickle.loads(open(f, "rb").read())
    t = obj._ir_text
    ks = re.findall(r"gpu\.(?:func|binary) @(\w+)", t)
    print(f, len(t), ks[:5], re.findall(r"kernel_\w+|fmha\w*|attn\w*", t)[:5])
