import sys, pickle, glob, re, hashlib, os, time
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
for d in sorted(glob.glob(sys.argv[1] + "/*")):
    print("==", os.path.basename(d))
    for f in sorted(glob.glob(d + "/*.pkl"), key=os.path.getmtime):
        obj = pickle.loads(open(f, "rb").read())
        t = obj._ir_text
        m = re.search(r'gpu\.binary @kernels.*?bin = "(.*?)">', t, re.S)
        binh = hashlib.md5(m.group(1).encode()).hexdigest()[:10] if m else "nobin"
        # host part = ir with binary removed
        host = re.sub(r'bin = ".*?"', 'bin=X', t, flags=re.S)
        print(" ", os.path.basename(f), time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f))), "bin", binh, "len", len(m.group(1)) if m else 0, "host", hashlib.md5(host.encode()).hexdigest()[:10])
