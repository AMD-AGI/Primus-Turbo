#!/usr/bin/env python3
"""CPU-only check of the fwd job's patched op/refcache_util.py and op/eager/impl.py -- no torch, no GPU.

    python3 test_refcache_util.py <job_context/op>

torch and ut/common.py are replaced by fakes: torch.load returns the REAL provenance of each refcache/*.pt (read
with refcache_prov.py's stub unpickler) and tensors are objects that only know their device. What is checked:
  1. the three caches are accepted on A0 with the guarded eager/impl.py (the pinned pre-guard eager_sha);
  2. every non-cache path computes on the CPU (the fake eager records the device of its inputs);
  3. proxy/prod refuse instead of computing (cache miss, provenance drift, non-causal);
  4. any OTHER drift still invalidates a cache (common_sha, a further eager edit, dims);
  5. the device guard is the first statement of eager/impl.py forward_reference.
"""
import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

# Never write __pycache__/*.pyc next to the job's own modules (refcache_util.py, eager/impl.py) -- the job directory
# is checked file by file before a launch. Same effect as `python3 -B`, for callers that forget it.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refcache_prov  # noqa: E402  (stdlib-only reader of torch zip files)

OP = Path(sys.argv[1]).resolve()
CALLS = []


class T:
    """A tensor that only knows where it lives."""

    def __init__(self, device="cuda:0"):
        self.device = device

    @property
    def is_cuda(self):
        return str(self.device).startswith("cuda")

    def cpu(self):
        return T("cpu")

    def to(self, device):
        return T(device)


def real_blob(shape):
    import zipfile
    with zipfile.ZipFile(OP / "refcache" / f"{shape}.pt") as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        return refcache_prov._Unpickler(z.open(pkl)).load()


TAMPER = {}


def fake_load(path, map_location=None):
    shape = Path(path).stem
    blob = real_blob(shape)
    prov = dict(blob["provenance"])
    prov.update(TAMPER.get(shape, {}))
    return {"provenance": prov, "o": T("cpu"), "lse": T("cpu")}


torch = types.ModuleType("torch")
torch.load = fake_load
sys.modules["torch"] = torch

common = types.ModuleType("common")
common.SHAPES = refcache_prov._shapes(OP / "ut" / "common.py")


def fake_eager(q, k, v, causal=True):
    CALLS.append((q.device, k.device, v.device, causal))
    return T(q.device), T(q.device)


common.load_impl = lambda d: fake_eager
sys.modules["common"] = common

spec = importlib.util.spec_from_file_location("refcache_util_under_test", OP / "refcache_util.py")
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)

fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def call(shape, causal=True):
    CALLS.clear()
    q, k, v = T(), T(), T()
    try:
        o, lse, src = ru.reference(shape, q, k, v, causal=causal)
        return src, list(CALLS), str(o.device)
    except RuntimeError as exc:
        return f"REFUSED: {exc}", list(CALLS), None


sha16 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
print(f"eager/impl.py sha16 now {sha16(OP / 'eager' / 'impl.py')}, ut/common.py {sha16(OP / 'ut' / 'common.py')}")

for s in ("fast", "proxy", "prod"):
    src, calls, dev = call(s)
    check(f"{s} causal served from refcache", src == "refcache" and not calls and dev == "cuda:0", src)
for s in ("toy", "short_q", "gqa4_batch2", "mha", "unequal_seqlen", "unequal_seqlen2"):
    for c in (True, False):
        src, calls, dev = call(s, c)
        check(f"{s} causal={c} on the CPU", src.startswith("cpu (") and calls == [("cpu", "cpu", "cpu", c)]
              and dev == "cuda:0", src)
src, calls, dev = call("sq_gt_skv", False)
check("sq_gt_skv non-causal on the CPU", src == "cpu (non-causal)" and calls == [("cpu", "cpu", "cpu", False)], src)
src, calls, _ = call("fast", False)
check("fast non-causal on the CPU", src == "cpu (non-causal)" and calls == [("cpu", "cpu", "cpu", False)], src)
for s in ("proxy", "prod"):
    src, calls, _ = call(s, False)
    check(f"{s} non-causal refused", src.startswith("REFUSED") and not calls, src[:90])

TAMPER.update(prod={"common_sha": "0000000000000000"})
src, calls, _ = call("prod")
check("prod common_sha drift refused", src.startswith("REFUSED") and "common_sha" in src and not calls, src[:100])
TAMPER.clear(); TAMPER.update(proxy={"eager_sha": "1111111111111111"})
src, calls, _ = call("proxy")
check("proxy unknown eager_sha refused", src.startswith("REFUSED") and "eager_sha" in src and not calls, src[:100])
TAMPER.clear(); TAMPER.update(fast={"dims": (1, 1, 1, 1, 1, 1)})
src, calls, _ = call("fast")
check("fast dims drift -> CPU recompute", src.startswith("cpu (provenance differs: dims") and calls, src)
TAMPER.clear()
real_sha = ru._sha
ru._sha = lambda p: "2222222222222222" if str(p).endswith("eager/impl.py") else real_sha(p)
src, calls, _ = call("prod")
check("a further eager/impl.py edit invalidates prod", src.startswith("REFUSED") and not calls, src[:100])
ru._sha = real_sha

tree = ast.parse((OP / "eager" / "impl.py").read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_reference")
first = fn.body[1] if isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) else fn.body[0]
seg = ast.get_source_segment((OP / "eager" / "impl.py").read_text(), first) or ""
check("eager guard is the first statement of forward_reference",
      isinstance(first, ast.If) and "is_cuda" in seg and ".cpu()" in seg, seg.splitlines()[0] if seg else "")

print("RESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
