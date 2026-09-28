"""Replay REAL training attention inputs (dumped by arms/flyd at step 43, layers 0,1,2,8,16,31) at op level.

Per input set (6 real layers + 'randn' = N(0,1) q/k/v like the rulers): for each arm (asm, fly=r6, fly11=r11)
  blk  fwd 4 warm + 9 timed back-to-back;  iso fwd 5 x (sleep 150 ms, call, sync);  gb fwd 5 x (GEMM burst, call)
  bwd  (asm vs fly r20) blk 2 warm + 5 timed on the same inputs with a fixed random dO
Kernel times come from the kineto trace (tools/opana.py attributes by P::<cond>::<arm>::<rep> where
cond = <kind>_<set>). Also prints, per set, the FlyDSL speculative-softmax trigger statistic computed
from the real scores (bf16 GEMM on the card, batch 0, all heads): fraction of (row, 64-wide KV tile)
steps where the tile max exceeds the stale running max by > 7 nats (r6 SPEC_TRIGGER = e^7), and the
same promoted to 32-row groups (wave-uniform ballot) for two plausible row packings.
Env: RP_ORDER=asm,fly,fly11  RP_REPS=2  RP_OUT=<trace>
"""
import os, sys, time, json, glob
from pathlib import Path

E2E = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e")
PROF = Path("/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/profile")
sys.path.insert(0, str(E2E / "attn_backends")); sys.path.insert(0, "/home/lihuzhan/.local/flydsl0341")
os.environ["E2E_ATTN"] = "asm"
os.environ["E2E_FLY_TREES"] = json.dumps({"fly11": {"fwd": str(PROF / "arms" / "fwd_r11"), "bwd": str(E2E / "arms" / "bwd_r20_0341")}})
BLAS = {k: os.environ[k] for k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_TENSILE_LIBPATH")}
import torch  # noqa
import torch.nn.functional as F  # noqa
from torch.profiler import record_function as rf  # noqa

ORDER = os.environ.get("RP_ORDER", "asm,fly,fly11").split(",")
REPS = int(os.environ.get("RP_REPS", "2")); OUT = os.environ["RP_OUT"]
DDIR = os.environ.get("RP_DIR", "/home/lihuzhan/_prof_dump")
bf = torch.bfloat16; dev = "cuda"
b, s, hq, hkv, d = 4, 8192, 32, 8, 128; sc = d ** -0.5
xg = torch.randn(32768, 4096, device=dev, dtype=bf); wg = torch.randn(14336, 4096, device=dev, dtype=bf) * 0.02
for _ in range(3): F.linear(xg, wg)
torch.cuda.synchronize()
from e2e_attn import arms  # noqa
FW = {a: arms.get_fwd(a) for a in ORDER}
BW = {"asm": arms.get_bwd("asm"), "fly": arms.get_bwd("fly")}
os.environ.update(BLAS)
print(arms.check_versions(), flush=True)

SETS = {}
for f in sorted(glob.glob(f"{DDIR}/qkv_call*.pt")):
    t = torch.load(f)
    L = t["call"] % 32
    SETS[f"L{L:02d}"] = tuple(t[n].to(dev) for n in ("q", "k", "v"))
torch.manual_seed(0)
SETS["randn"] = (torch.randn(b, s, hq, d, device=dev, dtype=bf), torch.randn(b, s, hkv, d, device=dev, dtype=bf),
                 torch.randn(b, s, hkv, d, device=dev, dtype=bf))
do = torch.randn(b, s, hq, d, device=dev, dtype=bf) * 1e-3
print("sets", list(SETS), flush=True)


def stats(name, q, k):
    # per-row stale-max trigger walk over 64-wide KV tiles, batch 0, all q heads
    T = 64; nt = s // T
    trig_rows = 0; steps = 0; grp_a = 0; grp_b = 0; grp_steps = 0; std = []
    for h in range(hq):
        qh = q[0, :, h, :]; kh = k[0, :, h // 4, :]
        S = (qh @ kh.t()).float() * sc                                   # [s, s] natural logits
        std.append(S.std().item())
        idx = torch.arange(s, device=dev)
        S.masked_fill_(idx[None, :] > idx[:, None], float("-inf"))
        tm = S.view(s, nt, T).amax(-1)                                    # [s, nt]
        del S
        m = tm[:, 0].clone()
        trig = torch.zeros(s, nt, dtype=torch.bool, device=dev)
        for j in range(1, nt):
            valid = (j * T) <= idx
            tj = tm[:, j]
            t_ = valid & (tj - m > 7.0)
            trig[:, j] = t_
            m = torch.where(t_, torch.maximum(m, tj), m)
        valid_all = (torch.arange(nt, device=dev)[None, :] * T) <= idx[:, None]
        valid_all[:, 0] = False
        trig_rows += trig.sum().item(); steps += valid_all.sum().item()
        # packing A: 32 consecutive seq rows of one head
        ga = trig.view(s // 32, 32, nt).any(1); va = valid_all.view(s // 32, 32, nt).any(1)
        grp_a += (ga & va).sum().item(); grp_steps += va.sum().item()
        STAT.setdefault(name, {}).setdefault("heads_trig", []).append(round(trig.sum().item() / max(1, valid_all.sum().item()), 5))
    # packing B: 8 seq x 4 heads of one kv group -> approximated by 8-seq groups OR-ed over the 4 heads
    STAT[name].update(row_trigger_frac=trig_rows / steps, wave32_seq_trigger_frac=grp_a / grp_steps,
                      score_std=sum(std) / len(std))


STAT = {}
with torch.no_grad():
    for name, (q, k, v) in SETS.items():
        stats(name, q, k)
        print("stat", name, {kk: vv for kk, vv in STAT[name].items() if kk != "heads_trig"}, flush=True)
torch.cuda.empty_cache()
# first launch / compile
for a in ORDER:
    o, lse = FW[a](*SETS["randn"], sc)
for a in ("asm", "fly"):
    o, lse = FW[a](*SETS["randn"], sc); BW[a](do, *SETS["randn"], o, lse, sc)
torch.cuda.synchronize()


def gburst():
    for _ in range(10): F.linear(xg, wg)


act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
t0 = time.time()
with torch.profiler.profile(activities=act) as prof:
    for rep in range(REPS):
        order = ORDER if rep % 2 == 0 else ORDER[::-1]
        for name, (q, k, v) in SETS.items():
            for a in order:
                torch.cuda.synchronize()
                with rf(f"P::blk_{name}::{a}::{rep}"):
                    for _ in range(13): FW[a](q, k, v, sc)
                torch.cuda.synchronize()
                with rf(f"P::iso_{name}::{a}::{rep}"):
                    for _ in range(5):
                        torch.cuda.synchronize(); time.sleep(0.15); FW[a](q, k, v, sc); torch.cuda.synchronize()
                with rf(f"P::gb_{name}::{a}::{rep}"):
                    for _ in range(5):
                        torch.cuda.synchronize(); gburst(); FW[a](q, k, v, sc); torch.cuda.synchronize()
            for a in (("asm", "fly") if rep % 2 == 0 else ("fly", "asm")):
                o, lse = FW[a](q, k, v, sc); torch.cuda.synchronize()
                with rf(f"P::bblk_{name}::{a}::{rep}"):
                    for _ in range(7): BW[a](do, q, k, v, o, lse, sc)
                torch.cuda.synchronize()
        print(f"rep {rep} {time.time() - t0:.1f}s", flush=True)
prof.export_chrome_trace(OUT)
json.dump(STAT, open(OUT.replace(".json", ".stat.json"), "w"), indent=1)
print("done", time.time() - t0, flush=True)
