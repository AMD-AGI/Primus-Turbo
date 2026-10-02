"""CPU-only: FlyDSL 0.3.4.1 JIT cache keys (manager_key) of the bwd launchers in two trees loaded in ONE process.
If a key is equal across trees, the two trees share disk-cache entries for that launcher (harmless only if the
generated code is identical too). usage: jitkey_check.py <treeA> <treeB>   (HIP_VISIBLE_DEVICES=-1)"""
import hashlib, importlib.util, os, sys, types
assert os.environ.get("HIP_VISIBLE_DEVICES") == "-1"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
AITER = "/home/lihuzhan/code/aiter-src"
for pkg in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
    m = types.ModuleType(pkg); m.__path__ = [os.path.join(AITER, *pkg.split("."))]; sys.modules[pkg] = m
import flydsl
assert "flydsl0341" in flydsl.__file__
mods = []
for i, d in enumerate(sys.argv[1:3]):
    sp = importlib.util.spec_from_file_location(f"tree{i}_kernels", os.path.join(d, "kernels.py"))
    K = importlib.util.module_from_spec(sp); sys.modules[sp.name] = K; sp.loader.exec_module(K); mods.append(K)
for fn in ("launch_delta", "launch_dkdv", "launch_dkdv_sp", "launch_dqg"):
    ks = []
    for K in mods:
        f = getattr(K, fn); f._ensure_cache_manager(); ks.append(hashlib.sha256(f.manager_key.encode()).hexdigest()[:16])
    print(f"{fn:15s} A={ks[0]} B={ks[1]} {'SAME KEY (shared cache entries)' if ks[0] == ks[1] else 'distinct'}")
