"""CPU proof for arm dqg_dkdv_streams (no torch, no GPU).

The arm changes NO kernel and NO index/address/predicate expression. What it changes is
the queue each launch goes to. This proof runs impl.py of r29 and of this arm against a
stub torch/kernels module and checks, for prod / fast / toy shapes, that every kernel is
launched with exactly the same scalar arguments (grid dims, sizes, nkvt, cshift, ...) and
the same tensor shapes/dtypes as r29 -- i.e. every address the kernels form is the one r29
forms -- and that the stream graph is: delta(main) -> fork -> {dK/dV chain on main,
dQ chain on side} -> join (main waits side) before return.
"""
import sys, types, pathlib
HERE = pathlib.Path(__file__).resolve().parent
R29 = HERE.parent / "r29"

class St:
    def __init__(s, name="main", device=None): s.name = "side" if device is not None else name
    def wait_stream(s, o): LOG.append(("wait", s.name, o.name))

class T:
    class _D: index = 0
    device = _D()
    def __init__(s, shape, dtype="bf16"): s.shape = tuple(shape); s.dtype = dtype
    def is_contiguous(s): return True
    def contiguous(s): return s
    def float(s): return T(s.shape, "f32")
    def dim(s): return len(s.shape)
    def record_stream(s, st): LOG.append(("rec", st.name))

LOG = []
MAIN = St("main")
torch = types.ModuleType("torch"); torch.bfloat16 = "bf16"; torch.float32 = "f32"; torch.Tensor = T
torch.empty = lambda shape, device=None, dtype=None: T(shape, dtype)
torch.empty_like = lambda t: T(t.shape, t.dtype)
torch.cuda = types.SimpleNamespace(current_stream=lambda: MAIN, Stream=St)
sys.modules["torch"] = torch

def _norm(a):
    return ("T", a.shape, a.dtype) if isinstance(a, T) else a

def mk(n):
    def f(*a):
        LOG.append((n, tuple(_norm(x) for x in a[:-1]), a[-1].name))
    return f

def load(dirp, kmod):
    src = (dirp / "impl.py").read_text()
    src = src.replace('_env = _sibling("_env")', '_env = types.SimpleNamespace(assert_environment=lambda: None)')
    src = src.replace('_k = _sibling("kernels")', "_k = K")
    src = src.replace("import flydsl.compiler as _flyc", "_flyc = FLYC")
    FLYC = types.SimpleNamespace(compile=lambda f, *a: (f(*a), (lambda *b: f(*b)))[1])
    g = {"types": types, "K": kmod, "FLYC": FLYC, "__name__": "m", "__file__": str(dirp / "impl.py")}
    exec(compile(src, str(dirp / "impl.py"), "exec"), g)
    return g

def kstub(dfuse):
    # constants read from r29 kernels.py text (arm's kernels.py is byte-identical)
    txt = (HERE / "kernels.py").read_text()
    import re
    def c(n): return int(re.search(rf"^{n}\s*=\s*(\d+)", txt, re.M).group(1))
    names = ["D", "BLOCK_Q", "KV_STEP", "BLOCK_KV", "ROWS_DELTA", "DQ_NW", "DQ_BQW", "RED_VEC", "RED_THREADS"]
    ns = {n: c(n) for n in names}
    ns["DQ_DFUSE"] = dfuse
    for n in ("delta", "dkdv", "dkdv_sp", "redsp", "dq", "dq_sp", "redsp_q", "dqg"):
        ns["launch_" + n] = mk(n)
    return types.SimpleNamespace(**ns)

assert (HERE / "kernels.py").read_text() == (R29 / "kernels.py").read_text(), "kernels.py differs from r29"
SHAPES = {"prod": (4, 8192, 32, 8), "fast": (1, 1024, 8, 2), "toy": (1, 128, 2, 1)}
ok = True
for dfuse in (False, True):
    K = kstub(dfuse)
    gr, ga = load(R29, K), load(HERE, K)
    for first in (False, True):
        ga["DQ_SIDE_FIRST"] = first
        for nm, (b, s, hq, hkv) in SHAPES.items():
            runs = []
            for g in (gr, ga):
                LOG.clear()
                q = T((b, s, hq, 128)); k = T((b, s, hkv, 128))
                g["flydsl_attn_bwd"](q, q, k, k, q, T((b, hq, s), "f32"))
                runs.append(list(LOG))
            r, a = runs
            rk = sorted((x[0], x[1]) for x in r if x[0] not in ("wait", "rec"))
            ak = sorted((x[0], x[1]) for x in a if x[0] not in ("wait", "rec"))
            same = rk == ak
            order = [(x[0], x[-1]) if x[0] != "wait" else (f"{x[1]}.wait", x[2]) for x in a if x[0] != "rec"]
            fused = not any(x[0] == "delta" for x in a)
            if not fused:
                # fork after delta, join last, dQ chain on side, dK/dV chain on main
                assert order[0] == ("delta", "main") and order[1] == ("side.wait", "main")
                assert order[-1] == ("main.wait", "side")
                for x in a:
                    if x[0] in ("dq", "dq_sp", "redsp_q", "dqg"): assert x[-1] == "side", x
                    if x[0] in ("dkdv", "dkdv_sp", "redsp", "delta"): assert x[-1] == "main", x
            else:
                assert all(x[-1] == "main" for x in a if x[0] not in ("wait", "rec")), "fused path must stay serial"
                assert not any(x[0] == "wait" for x in a)
            ok &= same
            print(f"dfuse={dfuse!s:5} side_first={first!s:5} {nm:4} args_identical_to_r29={same} "
                  f"order={[f'{n}@{st}' for n, st in order]}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
