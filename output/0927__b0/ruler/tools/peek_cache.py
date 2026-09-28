import sys, pickle, glob, re, zlib
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
for f in sorted(glob.glob(sys.argv[1] + "/*/*.pkl")):
    raw = open(f, "rb").read()
    try:
        obj = pickle.loads(raw)
    except Exception as e:
        print(f, "unpickle fail", e); continue
    s = repr(type(obj))
    txt = ""
    if isinstance(obj, dict):
        print(f, "dict keys", list(obj.keys())[:10])
        for v in obj.values():
            if isinstance(v, (bytes, str)): txt += v if isinstance(v, str) else v.decode("latin1")
    else:
        st = obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__
        print(f, s, list(st.keys()) if isinstance(st, dict) else type(st))
    names = set(re.findall(r"gpu\.func @(\w+)|llvm\.func @(\w+)", str(obj.__dict__ if hasattr(obj,'__dict__') else obj)))
    print("  names:", list(names)[:6])
