#!/usr/bin/env python3
"""End-to-end CPU smoke test of the two READY benchmark.py copies against a FAKE torch (never the real one).

    python3 test/smoke_fake.py            # host python3, stdlib only; no GPU, no torch, no container

Builds a throw-away job_context/op tree per job (ready benchmark.py + gbruler.py, fake ut/common.py,
refcache_util.py, op_flops, three sleeping arms whose import re-points HIPBLASLT_TENSILE_LIBPATH like the fwd
_env.py does) and runs the scenarios that matter for the rollout:
  auto      fast blk-only; prod blk (reported) + gb (scored); A/A copy; witness + header
  blk       --ruler blk: no burst, no env change, rows as before (+ ruler=blk)
  void      a burst slower than --gb-max-burst-ms: exit 4, SPEED RULER VOID on stdout and stderr, no RESULT
  gate      the call validation.py makes (candidate + beat, one shape, --json), rows carry what it reads
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RULER = Path(__file__).resolve().parents[1]
HOST_LIB = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

FAKE_TORCH = {
    "torch/__init__.py": '''
        """FAKE torch for CPU smoke tests of benchmark.py -- not torch."""
        import os, time
        from . import cuda, backends, nn  # noqa: F401
        bfloat16, float32 = "bfloat16", "float32"
        class Tensor:
            def __init__(self, shape, dtype=None):
                self.shape, self.dtype = tuple(shape), dtype
            def to(self, *a, **k): return self
            def __mul__(self, o): return self
            __rmul__ = __mul__
            def zero_(self): return self
            def contiguous(self): return self
        def _shape(shape): return shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        def empty(*shape, device=None, dtype=None): return Tensor(_shape(shape), dtype)
        def randn(*shape, generator=None, device=None, dtype=None): return Tensor(_shape(shape), dtype)
        class Generator:
            def __init__(self, device=None): pass
            def manual_seed(self, s): return self
        ''',
    "torch/cuda/__init__.py": '''
        import time
        class Event:
            def __init__(self, enable_timing=False): self.t = None
            def record(self): self.t = time.perf_counter()
            def synchronize(self): pass
            def elapsed_time(self, other): return (other.t - self.t) * 1e3
        class _P:
            gcnArchName = "gfx1250:fake"
            name = "fake"
        def synchronize(): pass
        def get_device_properties(i): return _P()
        ''',
    "torch/backends/__init__.py": "from . import cuda  # noqa: F401\n",
    "torch/backends/cuda.py": '''
        _cur = ["_BlasBackend.Cublas"]
        def preferred_blas_library(backend=None):
            if backend is not None:
                _cur[0] = "_BlasBackend.Cublaslt" if backend in ("hipblaslt", "cublaslt") else backend
            return _cur[0]
        ''',
    "torch/nn/__init__.py": "from . import functional  # noqa: F401\n",
    "torch/nn/functional.py": '''
        import os, time
        CALLS = [0]
        def linear(x, w):
            CALLS[0] += 1
            time.sleep(float(os.environ.get("FAKE_GEMM_MS", "1.0")) * 1e-3)
            from torch import Tensor
            return Tensor((x.shape[0], w.shape[0]))
        ''',
}

COMMON = {
    "bwd": '''
        import importlib.util, sys
        from pathlib import Path
        import torch
        SHAPES = {"fast": (1, 1024, 1024, 8, 2, 128), "proxy": (1, 4096, 4096, 32, 8, 128),
                  "prod": (4, 8192, 8192, 32, 8, 128)}
        SPEC_SHAPES = ("fast", "proxy", "prod")
        def make_inputs(name, seed=0, device="cuda"):
            b, sq, skv, hq, hkv, d = SHAPES[name]
            return (torch.Tensor((b, sq, hq, d)), torch.Tensor((b, skv, hkv, d)), torch.Tensor((b, skv, hkv, d)),
                    torch.Tensor((b, sq, hq, d)))
        def forward_reference(*a, **k): raise RuntimeError("fp32 reference must not run in benchmark.py")
        def load_impl(impl_dir):
            impl_dir = Path(impl_dir).resolve()
            name = "op_impl_" + str(abs(hash(str(impl_dir))))
            spec = importlib.util.spec_from_file_location(name, impl_dir / "impl.py")
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
            return mod.attn_bwd
        ''',
    "fwd": '''
        import importlib.util, sys
        from pathlib import Path
        import torch
        SHAPES = {"fast": (1, 1024, 1024, 8, 2, 128), "proxy": (1, 4096, 4096, 32, 8, 128),
                  "prod": (4, 8192, 8192, 32, 8, 128)}
        SPEC_SHAPES = ("fast", "proxy", "prod")
        def make_inputs(name, seed=0, device="cuda"):
            b, sq, skv, hq, hkv, d = SHAPES[name]
            return torch.Tensor((b, sq, hq, d)), torch.Tensor((b, skv, hkv, d)), torch.Tensor((b, skv, hkv, d))
        def load_impl(impl_dir):
            impl_dir = Path(impl_dir).resolve()
            name = "op_impl_" + str(abs(hash(str(impl_dir))))
            if name in sys.modules:
                return getattr(sys.modules[name], "attn_fwd")
            spec = importlib.util.spec_from_file_location(name, impl_dir / "impl.py")
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
            return mod.attn_fwd
        ''',
}

IMPL = '''
    import os, time
    from pathlib import Path
    os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "{host}"     # like the fwd trees' _env.py (ASSIGNS the host lib)
    _MS = float((Path(__file__).parent / "ms.txt").read_text())
    _SCALE = {{1024: 0.05, 4096: 0.25, 8192: 1.0}}
    def {fn}(*args, causal=True):
        q = args[1] if len(args) == 7 else args[0]
        time.sleep(_MS * _SCALE[q.shape[1]] * 1e-3)
        return args[:3]
    '''

RUNNER = '''
import runpy, sys
op, img, bench = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, op)
import gbruler
gbruler.IMAGE_BLAS = img          # the test host has no /opt/venv image library
sys.argv = [bench] + sys.argv[4:]
runpy.run_path(bench, run_name="__main__")
'''


def w(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip("\n"))


def build(root: Path, job: str) -> Path:
    op = root / job / "x" / "artifacts" / "job" / "job_context" / "op"
    (op / "ut").mkdir(parents=True)
    (op / "benchmark.py").write_text((RULER / job / "benchmark.py").read_text())
    (op / "gbruler.py").write_text((RULER / "gbruler.py").read_text())
    w(op / "ut" / "common.py", COMMON[job])
    w(op / "refcache_util.py", '''
        def cached_forward(shape, q, k, v, causal=True):
            return q, q
        ''')
    w(root / job / "x" / "tools" / "op_flops.py", '''
        from types import SimpleNamespace
        def attention(batch, heads, seqlen_q, seqlen_kv, head_dim, kv_heads, causal, dtype, backward):
            f = 2.0 * batch * heads * seqlen_q * seqlen_kv * head_dim * (2.5 if backward else 1.0) / 2
            return SimpleNamespace(flop=f, bytes_min=1.0e9)
        ''')
    fn = "attn_bwd" if job == "bwd" else "attn_fwd"
    for arm, ms in (("current", 2.0), ("cand", 1.9), ("beat", 2.2), ("slowb", 2.0)):
        w(op / arm / "impl.py", IMPL.format(host=HOST_LIB, fn=fn))
        (op / arm / "ms.txt").write_text(str(ms))
    # an arm whose first CALL (not its import) slows every later burst: the per-shape void path
    impl = op / "slowb" / "impl.py"
    impl.write_text(impl.read_text().replace("    time.sleep(_MS", "    os.environ['FAKE_GEMM_MS'] = '10'\n    time.sleep(_MS"))
    return op


def run(op: Path, img: Path, sclk: Path, args, gemm_ms="1.0"):
    env = dict(os.environ, PYTHONPATH=str(op.parents[5] / "fake"), FAKE_GEMM_MS=gemm_ms, OE_SCLK_FILE=str(sclk),
               HIPBLASLT_TENSILE_LIBPATH=HOST_LIB, TORCH_BLAS_PREFER_HIPBLASLT="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", RUNNER, str(op), str(img), str(op / "benchmark.py")] + args,
                       capture_output=True, text=True, env=env, timeout=300)
    rows = [dict(kv.split("=", 1) for kv in line.split()[1:])
            for line in p.stdout.splitlines() if line.startswith("RESULT ")]
    return p, rows


def main() -> int:
    bad = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            bad.append(msg)

    with tempfile.TemporaryDirectory(prefix="gbruler_smoke_") as td:
        root = Path(td)
        for rel, text in FAKE_TORCH.items():
            w(root / "fake" / rel, text)
        img = root / "imagelib"
        img.mkdir()
        sclk = root / "freq1_input"
        sclk.write_text("1280000000\n")
        for job in ("bwd", "fwd"):
            op = build(root, job)
            base = ["--arm-path", f"cand={op / 'cand'}", "--arms", "current,beat", "--iters", "10",
                    "--warmup-seconds", "0.05", "--aa", "current"]
            # auto: fast blk, prod blk + gb, A/A
            p, rows = run(op, img, sclk, base + ["--shapes", "fast,prod", "--json", str(root / f"{job}_auto.json")])
            check(p.returncode == 0, f"{job} auto rc=0 (rc={p.returncode}) {p.stderr.strip()[-300:]}")
            if os.environ.get("SMOKE_SHOW"):
                print("\n".join(x for x in p.stdout.splitlines() if x.startswith(("# gb", "# ENV", "# ruler"))
                                 or (x.startswith("RESULT") and "shape=prod" in x and "arm=current " in x)))
            by = {(r["shape"], r["arm"]): r for r in rows}
            check(len(rows) == 8, f"{job} auto 8 rows (2 shapes x cand/current/current_aa/beat), got {len(rows)}")
            check(all(by[("fast", a)]["ruler"] == "blk" for a in ("cand", "current", "beat")), f"{job} fast scored blk")
            check(all(by[("prod", a)]["ruler"] == "gb" for a in ("cand", "current", "beat")), f"{job} prod scored gb")
            pr = by.get(("prod", "current"), {})
            check(int(pr.get("iters", 0)) == 10, f"{job} prod gb timed calls = 10 (2 rounds x 5): {pr.get('iters')}")
            check("blk_latency_ms" in pr and "gb_burst_ms" in pr and pr.get("gb_sclk") == "1280",
                  f"{job} prod row has blk_* and gb witness (gb_sclk {pr.get('gb_sclk')}, burst {pr.get('gb_burst_ms')})")
            check(9.0 <= float(pr.get("gb_burst_ms", 0)) <= 40.0, f"{job} burst ~10 x 1 ms fake GEMM: {pr.get('gb_burst_ms')}")
            check(1.5 <= float(pr.get("latency_ms", 0)) <= 4.0, f"{job} prod current ~2 ms: {pr.get('latency_ms')}")
            check("aa_ratio" in by.get(("prod", "current_aa"), {}), f"{job} A/A row carries aa_ratio "
                  f"{by.get(('prod', 'current_aa'), {}).get('aa_ratio')}")
            check("gb ruler:" in p.stdout and f"{HOST_LIB} -> {img}" in p.stdout,
                  f"{job} header shows the env re-assigned host -> image after loading the arms")
            check(all(" " not in r["order"] for r in rows), f"{job} order field has no spaces")
            js = json.loads((root / f"{job}_auto.json").read_text())
            check(all(k in js[0] for k in ("tflops", "latency_ms", "sclk_start", "sclk_end", "stat", "iters", "arm",
                                           "shape")), f"{job} json rows keep the keys validation.py reads")
            if job == "bwd":
                check(by[("fast", "current")]["stat"] == "min", "bwd fast still scored on the MIN (h83)")
            # blk: the pre-10-02 behaviour
            p, rows = run(op, img, sclk, base + ["--shapes", "prod", "--ruler", "blk"])
            check(p.returncode == 0 and rows and all(r["ruler"] == "blk" and "gb_sclk" not in r for r in rows),
                  f"{job} --ruler blk: rows without gb fields (rc={p.returncode})")
            check("gb ruler:" not in p.stdout and "SPEED RULER" not in p.stdout, f"{job} --ruler blk: no burst")
            if job == "fwd":
                envl = [x for x in p.stdout.splitlines() if x.startswith("# ENV")]
                check(bool(envl) and HOST_LIB in envl[0], f"fwd --ruler blk leaves the host library env: {envl[:1]}")
            # void: burst 10 x 10 ms = 100 ms > 80 ms
            p, rows = run(op, img, sclk, base + ["--shapes", "prod"], gemm_ms="10")
            check(p.returncode == 4 and not rows, f"{job} slow burst -> exit 4, no RESULT (rc={p.returncode})")
            check("SPEED RULER VOID" in p.stdout and "SPEED RULER VOID" in p.stderr and "Traceback" not in p.stderr,
                  f"{job} void message on stdout and stderr, no traceback")
            # a burst that turns slow after the load-time check: void for that shape, exit 4, no RESULT for it
            p, rows = run(op, img, sclk, ["--arm-path", f"s={op / 'slowb'}", "--arms", "beat", "--shapes", "prod",
                                          "--iters", "5", "--warmup-seconds", "0.01"])
            check(p.returncode == 4 and not rows and "SPEED RULER VOID (gb, prod)" in p.stdout,
                  f"{job} burst slow during the loop -> void for prod, exit 4 (rc={p.returncode})")
            # the gate's call (validation.py): candidate + beat, one shape per process, --json
            gate = (["--arm-path", f"candidate={op / 'cand'}", "--arms", "beat"] if job == "bwd" else
                    ["--arm-path", f"candidate={op / 'cand'}", "--arm-path", f"beat={op / 'beat'}"])
            p, rows = run(op, img, sclk, gate + ["--shapes", "proxy", "--iters", "6", "--warmup-seconds", "0.05",
                                                 "--json", str(root / f"{job}_gate.json")])
            js = json.loads((root / f"{job}_gate.json").read_text()) if p.returncode == 0 else []
            check(p.returncode == 0 and {r["arm"] for r in js} == {"candidate", "beat"}
                  and all(r["ruler"] == "gb" for r in js), f"{job} gate call: candidate + beat scored gb at proxy")
    print(f"\n{'ALL OK' if not bad else f'{len(bad)} FAILED'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
