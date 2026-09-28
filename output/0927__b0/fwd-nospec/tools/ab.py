"""Op-level A/B r13 vs r13ns vs ASM on prod shape, REAL training inputs (+ randn reference), one condition per process.
  AB_COND=blk   the job's ruler method (benchmark.py): 8 s continuous warmup per arm, rounds palindromic over arms,
                each round = 4 untimed + 9 timed same-arm calls, 256 MB L2 flush before each timed call (outside the
                events), CUDA-event median of >= 101 timed calls per arm per input set.
  AB_COND=gb    the profile's in-context method: before EVERY timed call a GEMM burst (10 x 32768x4096x14336 bf16,
                profile/tools/replay.py gburst), then one call timed by CUDA events; rounds palindromic, AB_GB_N calls
                per arm per round, AB_GB_ROUNDS rounds.
  AB_ORDER=r13,r13ns,asm   arm order of round 0 (rotate across processes)
  AB_SETS=L00,...,randn    input sets (default: all 6 dumps + randn)
Prints one RESULT line per (set, arm) and per-set ratios; writes AB_JSON."""
import os, sys, glob, json, time
from pathlib import Path
L = Path(__file__).resolve().parents[1]; H = L / "harness"
sys.path.insert(0, str(H / "ut")); sys.path.insert(0, str(H))
import torch  # noqa
import torch.nn.functional as F  # noqa
from common import load_impl  # noqa
BLAS = {k: os.environ[k] for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH")}
COND = os.environ["AB_COND"]; ORDER = os.environ.get("AB_ORDER", "r13,r13ns,asm").split(",")
PATHS = {"r13": L / "arms/fwd_r13", "r13ns": L / "arms/fwd_r13ns", "asm": H / "beat"}
fns = {a: load_impl(PATHS[a]) for a in ORDER}
os.environ.update(BLAS)   # impl _env.py re-points hipBLASLt at the host library; the burst must use the image one (profile/tools/replay.py)
print("BLAS", BLAS, flush=True)
for a in ORDER:
    m = sys.modules[fns[a].__module__]
    print("arm", a, m.__file__, getattr(getattr(m, "_kern", None), "SPEC_STALE_MAX", "-"), flush=True)
dev = "cuda"; bf = torch.bfloat16
SETS = {}
want = os.environ.get("AB_SETS")
for f in sorted(glob.glob("/home/lihuzhan/_prof_dump/qkv_call*.pt")):
    t = torch.load(f); n = f"L{t['call'] % 32:02d}"
    if want and n not in want.split(","):
        continue
    SETS[n] = tuple(t[x].to(dev).contiguous() for x in ("q", "k", "v"))
if not want or "randn" in want.split(","):
    g = torch.Generator(device=dev).manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32).to(bf)
    SETS["randn"] = (r(4, 8192, 32, 128), r(4, 8192, 8, 128), r(4, 8192, 8, 128))
sc = 128 ** -0.5
flush = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
if COND == "gb":
    xg = torch.randn(32768, 4096, device=dev, dtype=bf); wg = torch.randn(14336, 4096, device=dev, dtype=bf) * 0.02
    for _ in range(3): F.linear(xg, wg)
    torch.cuda.synchronize(); _t = time.perf_counter()
    for _ in range(10): F.linear(xg, wg)
    torch.cuda.synchronize(); print(f"gburst 10 GEMMs {(time.perf_counter() - _t) * 1e3:.1f} ms", flush=True)
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)


def call(a, qkv):
    return fns[a](*qkv, softmax_scale=sc, causal=True)


def timed(a, qkv, pre):
    pre()
    ev0.record(); call(a, qkv); ev1.record(); ev1.synchronize()
    return ev0.elapsed_time(ev1)


for a in ORDER:
    call(a, SETS[next(iter(SETS))])
torch.cuda.synchronize()
for a in ORDER:       # continuous load per arm (benchmark.py warmup)
    t_end = time.perf_counter() + float(os.environ.get("AB_WARM", "8"))
    while time.perf_counter() < t_end:
        call(a, SETS[next(iter(SETS))])
    torch.cuda.synchronize()

times = {s: {a: [] for a in ORDER} for s in SETS}
t0 = time.time()
if COND == "blk":
    iters, block, lead = int(os.environ.get("AB_ITERS", "108")), 9, 4
    for sname, qkv in SETS.items():
        for rd in range(-(-iters // block)):
            for a in (ORDER if rd % 2 == 0 else ORDER[::-1]):
                for _ in range(lead):
                    timed(a, qkv, flush.zero_)
                for _ in range(block):
                    times[sname][a].append(timed(a, qkv, flush.zero_))
else:
    n, rounds = int(os.environ.get("AB_GB_N", "5")), int(os.environ.get("AB_GB_ROUNDS", "4"))

    def gb():
        torch.cuda.synchronize()
        for _ in range(10): F.linear(xg, wg)
    for sname, qkv in SETS.items():
        for rd in range(rounds):
            for a in (ORDER if rd % 2 == 0 else ORDER[::-1]):
                for _ in range(n):
                    times[sname][a].append(timed(a, qkv, gb))
print(f"timed loop {time.time() - t0:.1f}s", flush=True)
res = {}
for sname in SETS:
    med = {}
    for a in ORDER:
        ts = sorted(times[sname][a]); k = len(ts)
        med[a] = ts[k // 2] if k % 2 else 0.5 * (ts[k // 2 - 1] + ts[k // 2])
        print(f"RESULT cond={COND} set={sname} arm={a} n={k} median_ms={med[a]:.4f} min={ts[0]:.4f} max={ts[-1]:.4f}", flush=True)
    print(f"RATIO cond={COND} set={sname} order={','.join(ORDER)} r13ns/r13={med['r13ns'] / med['r13']:.4f} "
          f"r13/asm={med['r13'] / med['asm']:.4f} r13ns/asm={med['r13ns'] / med['asm']:.4f}", flush=True)
    res[sname] = {"median_ms": med, "raw": times[sname]}
json.dump({"cond": COND, "order": ORDER, "sets": res}, open(os.environ["AB_JSON"], "w"))
