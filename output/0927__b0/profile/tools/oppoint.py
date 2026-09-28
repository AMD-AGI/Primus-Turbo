"""Attention at the in-training operating point, op level, one process, prod shape only.

Arms (same process): asm (aiter ASM fwd+bwd), fly (fwd r6 + bwd r20), fly11 (fwd r11 + bwd r20).
Conditions, each wrapped in record_function("P::<cond>::<arm>::<rep>") so that every GPU kernel of
the kineto trace can be attributed offline (tools/opana.py):
  blk      4 warm + 9 timed back-to-back calls (fwd block, then bwd block) -- the blocked ruler
  iso      5 x (sleep 150 ms, one call, sync)                             -- cold / idle clocks
  gburst   5 x (10 x 32768x4096x14336 GEMM, then one call, sync)         -- right after a GEMM burst
  eburst   5 x (40 x 2.8 GB elementwise add, then one call, sync)         -- right after a memory burst
  scale    fwd block (2 warm + 5) with q scaled x{1,4,16}: the FlyDSL deferred-rescale branch is data dependent
  layer    NL-layer training emulation: per layer qkv GEMMs -> attn fwd -> o GEMM -> MLP GEMMs, then the
           bwd in reverse with dgrad/wgrad GEMMs in the GOOD layout (what nkfix produces) -> attn bwd
  layerbad same but the bwd GEMMs in the raw autograd layout (pre-fix MT32x16x32 fallback), NLB layers
All GEMM/elementwise here is bounded (a few seconds per condition) -- it is the e2e load, not a burn loop.
Env: OP_ORDER=asm,fly,fly11  OP_REPS=3  OP_NL=16  OP_NLB=3  OP_OUT=<trace.json>
"""
import os, sys, time, json
from pathlib import Path

E2E = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e")
PROF = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/profile")
sys.path.insert(0, str(E2E / "attn_backends"))
sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
os.environ["E2E_FLY_TREES"] = json.dumps({"fly11": {"fwd": str(PROF / "arms" / "fwd_r11"),
                                                     "bwd": str(E2E / "arms" / "bwd_r20_0341")}})
BLAS = {k: os.environ[k] for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH")}
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import record_function as rf  # noqa: E402

ORDER = os.environ.get("OP_ORDER", "asm,fly,fly11").split(",")
REPS = int(os.environ.get("OP_REPS", "3"))
NL = int(os.environ.get("OP_NL", "16"))
NLB = int(os.environ.get("OP_NLB", "3"))
OUT = os.environ["OP_OUT"]
dev = "cuda"
bf = torch.bfloat16
torch.manual_seed(0)
b, s, hq, hkv, d = 4, 8192, 32, 8, 128
M, H, HK, FF = b * s, hq * d, hkv * d, 14336
sc = d ** -0.5


def rn(*sh, std=1.0):
    return (torch.randn(*sh, device=dev, dtype=torch.float32) * std).to(bf)


# ---- GEMM operands (weights ~ N(0, 0.02), activations ~ N(0,1))
x = rn(M, H)
W = {"q": rn(H, H, std=.02), "k": rn(HK, H, std=.02), "v": rn(HK, H, std=.02), "o": rn(H, H, std=.02),
     "w1": rn(FF, H, std=.02), "w3": rn(FF, H, std=.02), "w2": rn(H, FF, std=.02)}
WT = {n: w.t().contiguous() for n, w in W.items()}           # for good-layout dgrad
T = {H: rn(H, M), HK: rn(HK, M), FF: rn(FF, M)}             # transposed activations for wgrad
dm = rn(M, H)
big_a, big_b = rn(M, FF), rn(M, FF)
# initialise hipBLASLt with the IMAGE library before any FlyDSL tree's _env.py rewrites the env
for _ in range(3):
    F.linear(x, W["w1"])
torch.cuda.synchronize()
from e2e_attn import arms  # noqa: E402
FW = {a: arms.get_fwd(a) for a in ORDER}
BW = {a: arms.get_bwd("asm" if a == "asm" else "fly") for a in ORDER}
os.environ.update(BLAS)
print(arms.check_versions(), "BLAS", BLAS, flush=True)

q0 = rn(b, s, hq, d); k0 = rn(b, s, hkv, d); v0 = rn(b, s, hkv, d); do0 = rn(b, s, hq, d)
for a in ORDER:                                                # compile / first launch, untimed
    o, lse = FW[a](q0, k0, v0, sc); BW[a](do0, q0, k0, v0, o, lse, sc)
torch.cuda.synchronize()
OL = {a: FW[a](q0, k0, v0, sc) for a in ORDER}
torch.cuda.synchronize()


def gburst():
    for _ in range(10):
        F.linear(x, W["w1"])


def eburst():
    for _ in range(40):
        torch.add(big_a, big_b, out=big_a)


def layer_step(a, nl, bad):
    fwd, bwd = FW[a], BW[a]
    acts = []
    h = x
    for L in range(nl):
        with rf(f"L::f{L}"):
            xn = h * 1.0
            q = F.linear(xn, W["q"]); k = F.linear(xn, W["k"]); v = F.linear(xn, W["v"])
            q4, k4, v4 = (t.view(b, s, -1, d) * 1.0 for t in (q, k, v))   # rope-like copy, as in training
            o, lse = fwd(q4, k4, v4, sc)
            h2 = h + F.linear(o.view(M, H), W["o"])
            hn = h2 * 1.0
            g = F.silu(F.linear(hn, W["w1"])) * F.linear(hn, W["w3"])
            h = h2 + F.linear(g, W["w2"])
            acts.append((q4, k4, v4, o, lse))
    for L in reversed(range(nl)):
        q4, k4, v4, o, lse = acts[L]
        with rf(f"L::b{L}"):
            if bad:
                dg = torch.mm(dm, W["w2"]); torch.mm(dm.t(), big_b)            # w2 dgrad / wgrad
                da = dg * 1.0
                torch.mm(da, W["w1"]); torch.mm(da, W["w3"])
                torch.mm(da.t(), x); torch.mm(da.t(), x)                        # w1/w3 wgrad
                dattn = torch.mm(dm, W["o"]); torch.mm(dm.t(), x)              # o dgrad / wgrad
            else:
                dg = F.linear(dm, WT["w2"]); F.linear(T[H], T[FF])
                da = dg * 1.0
                F.linear(da, WT["w1"]); F.linear(da, WT["w3"])
                F.linear(T[FF], T[H]); F.linear(T[FF], T[H])
                dattn = F.linear(dm, WT["o"]); F.linear(T[H], T[H])
            do4 = dattn.view(b, s, hq, d)
            dq, dk, dv = bwd(do4, q4, k4, v4, o, lse, sc)
            if bad:
                torch.mm(dq.reshape(M, H), W["q"]); torch.mm(dk.reshape(M, HK), W["k"]); torch.mm(dv.reshape(M, HK), W["v"])
                torch.mm(dq.reshape(M, H).t(), x); torch.mm(dk.reshape(M, HK).t(), x); torch.mm(dv.reshape(M, HK).t(), x)
            else:
                F.linear(dq.reshape(M, H), WT["q"]); F.linear(dk.reshape(M, HK), WT["k"]); F.linear(dv.reshape(M, HK), WT["v"])
                F.linear(T[H], T[H]); F.linear(T[HK], T[H]); F.linear(T[HK], T[H])


def run_cond(cond, a, rep):
    tag = f"P::{cond}::{a}::{rep}"
    fwd, bwd = FW[a], BW[a]
    o, lse = OL[a]
    torch.cuda.synchronize()
    with rf(tag):
        if cond == "blk":
            with rf("K::fwd"):
                for _ in range(13):
                    fwd(q0, k0, v0, sc)
            with rf("K::bwd"):
                for _ in range(13):
                    bwd(do0, q0, k0, v0, o, lse, sc)
        elif cond in ("iso", "gburst", "eburst"):
            for kind in ("fwd", "bwd"):
                for _ in range(5):
                    torch.cuda.synchronize()
                    if cond == "iso":
                        time.sleep(0.15)
                    elif cond == "gburst":
                        gburst()
                    else:
                        eburst()
                    with rf(f"K::{kind}"):
                        if kind == "fwd":
                            fwd(q0, k0, v0, sc)
                        else:
                            bwd(do0, q0, k0, v0, o, lse, sc)
                    torch.cuda.synchronize()
        elif cond == "scale":
            for f in (1, 2, 4, 8, 16):
                qs = (q0.float() * f).to(bf)
                with rf(f"K::fwd_x{f}"):
                    for _ in range(7):
                        fwd(qs, k0, v0, sc)
                torch.cuda.synchronize()
        elif cond == "layer":
            layer_step(a, NL, False)
        elif cond == "layerbad":
            layer_step(a, NLB, True)
    torch.cuda.synchronize()


CONDS = os.environ.get("OP_CONDS", "blk,iso,gburst,eburst,scale,layer,layerbad").split(",")
act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
t0 = time.time()
with torch.profiler.profile(activities=act) as prof:
    for rep in range(REPS):
        order = ORDER if rep % 2 == 0 else ORDER[::-1]
        for cond in CONDS:
            for a in order:
                run_cond(cond, a, rep)
        print(f"rep {rep} done {time.time() - t0:.1f}s", flush=True)
prof.export_chrome_trace(OUT)
print("trace", OUT, "wall", round(time.time() - t0, 1), flush=True)
