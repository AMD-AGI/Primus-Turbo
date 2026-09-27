"""Compile-only check of k_atom (usage: ATOM_MODE=atom|store|none python compile_probe.py)."""
import os, sys, importlib.util
assert os.environ.get("COMPILE_ONLY") == "1"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
import torch, flydsl, flydsl.compiler as flyc
assert flydsl.__version__ == "0.3.4.1", flydsl.__file__
sp = importlib.util.spec_from_file_location("k_atom", os.path.join(os.path.dirname(__file__), "k_atom.py"))
K = importlib.util.module_from_spec(sp); sp.loader.exec_module(K)
b, sq, hq, hkv, d = 4, 8192, 32, 8, 128
acc = torch.empty((b, sq, hq, d), dtype=torch.float32, device="meta")
kvg = 1
flyc.compile(K.launch_atom, acc, sq, sq, hq, hq // hkv, sq // 32, 0, kvg, b, hkv, sq // (32 * kvg), b, None)
print("COMPILE_OK")
