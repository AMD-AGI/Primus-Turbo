import importlib.util
import sys
from pathlib import Path


def load_impl(d):
    d = Path(d).resolve()
    name = "vr6_impl_" + str(abs(hash(str(d))))
    spec = importlib.util.spec_from_file_location(name, d / "impl.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.attn_fwd
