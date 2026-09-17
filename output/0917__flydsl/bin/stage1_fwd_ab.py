"""Phase B decision gate: aiter's FlyDSL gfx1250 forward vs aiter's prebuilt ASM forward.

Both arms come from aiter, so primus_turbo is never imported and the flydsl 0.2.4 / 0.3.2
conflict does not arise here.

Discipline, all of it load-bearing on this box:
  - correctness BEFORE speed; NaN-prefill so "never written" != "wrote zero"
  - ABAB interleaving, not AAAA/BBBB: this card is VR-throttled and drifts, and a
    then-B split turns drift into the ratio you report
  - sclk witness around every measurement window
  - chunked fp32 reference: a dense [4,32,8192,8192] fp32 score tensor is 34 GB
"""
import os, sys, time, json, subprocess
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
sys.path.insert(0, "/tmp/flydsl032")
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")

import torch
import flydsl
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func as FLYDSL
from aiter.ops.mha import fmha_fwd_with_sink_asm as ASM

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
DT, CAUSAL = torch.bfloat16, True
SCALE = D ** -0.5
ITERS, WARMUP_S = 20, 3.0

def sclk():
    try:
        out = subprocess.run(["bash","-lc","cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\\*'"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.replace("\n", " ")
    except Exception as e:
        return f"(unavailable: {e})"

print(f"flydsl {flydsl.__version__} @ {flydsl.__file__}")
print(f"arch   {torch.cuda.get_device_properties(0).gcnArchName}")
print(f"shape  b={B} s={S} hq={HQ} hkv={HKV} d={D} {DT} causal={CAUSAL}")
print(f"sclk before: {sclk()}")

torch.manual_seed(0)
q = torch.randn(B, S, HQ, D, device="cuda", dtype=DT)
k = torch.randn(B, S, HKV, D, device="cuda", dtype=DT)
v = torch.randn(B, S, HKV, D, device="cuda", dtype=DT)

def run_flydsl():
    r = FLYDSL(q, k, v, softmax_scale=SCALE, causal=CAUSAL, return_lse=True)
    if r is None:
        raise SystemExit("FlyDSL declined this configuration (returned None)")
    return r[0]

def run_asm():
    o, _ = ASM(q, k, v, SCALE, CAUSAL, True)
    return o

ARMS = {"flydsl": run_flydsl, "asm": run_asm}

# ---------------- correctness first ----------------
def chunked_ref_out(qb=1024):
    rep = HQ // HKV
    out = torch.empty(B, S, HQ, D, device="cuda", dtype=torch.float32)
    for b in range(B):
        for h in range(HQ):
            hk = h // rep
            qh = q[b, :, h, :].float()                       # [S, D]
            kh = k[b, :, hk, :].float(); vh = v[b, :, hk, :].float()
            for s0 in range(0, S, qb):
                s1 = min(s0 + qb, S)
                sc = (qh[s0:s1] @ kh.T) * SCALE              # [qb, S]
                if CAUSAL:
                    idx = torch.arange(s0, s1, device=sc.device).unsqueeze(1)
                    jdx = torch.arange(S, device=sc.device).unsqueeze(0)
                    sc = sc.masked_fill(jdx > idx, float("-inf"))
                out[b, s0:s1, h, :] = sc.softmax(-1) @ vh
    return out

def sqnr_db(ref, got):
    ref, got = ref.float(), got.float()
    return float(10 * torch.log10(ref.pow(2).mean() / (ref - got).pow(2).mean().clamp_min(1e-30)))

print("\n--- correctness (fp32 chunked reference) ---")
t0 = time.time(); ref = chunked_ref_out(); torch.cuda.synchronize()
print(f"reference built in {time.time()-t0:.1f}s")

res = {}
for name, fn in ARMS.items():
    o = fn(); torch.cuda.synchronize()
    fin = int(torch.isfinite(o).sum()); tot = o.numel()
    db = sqnr_db(ref, o)
    res[name] = {"sqnr_out_db": round(db, 2), "isfinite": f"{fin}/{tot}", "full_coverage": fin == tot}
    print(f"  {name:7s} SQNR out {db:6.2f} dB   isfinite {fin}/{tot}" + ("" if fin == tot else "   <-- INCOMPLETE"))
del ref; torch.cuda.empty_cache()

bad = [n for n, r in res.items() if not r["full_coverage"] or r["sqnr_out_db"] < 50.0]
if bad:
    print(f"\nCORRECTNESS FAILED for {bad}; not timing a wrong kernel.")
    json.dump(res, open("/tmp/phaseB.json", "w"), indent=2); sys.exit(2)

# ---------------- speed, ABAB ----------------
print("\n--- timing (ABAB interleaved, CUDA events, L2 flush between reps) ---")
flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
def timed(fn):
    flush.zero_()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)

for name, fn in ARMS.items():                    # warm to a clock state, by seconds not iters
    t0 = time.time()
    while time.time() - t0 < WARMUP_S:
        fn()
    torch.cuda.synchronize()

samples = {n: [] for n in ARMS}
for i in range(ITERS):
    order = list(ARMS.items()) if i % 2 == 0 else list(ARMS.items())[::-1]   # ABBA
    for name, fn in order:
        samples[name].append(timed(fn))

print(f"sclk after: {sclk()}")
import statistics as st
print()
for name in ARMS:
    xs = sorted(samples[name])
    med, best = st.median(xs), xs[0]
    sd = st.pstdev(xs)
    res[name].update(median_ms=round(med, 4), best_ms=round(best, 4),
                     sd_pct=round(100 * sd / med, 2), n=len(xs))
    print(f"  {name:7s} median {med:7.3f} ms   best {best:7.3f} ms   sd {100*sd/med:5.2f}%   n={len(xs)}")

f, a = res["flydsl"]["median_ms"], res["asm"]["median_ms"]
res["ratio_asm_over_flydsl"] = round(a / f, 4)
print(f"\n  asm / flydsl = {a/f:.4f}   ({'FlyDSL faster' if f < a else 'ASM faster'})")
json.dump(res, open("/tmp/phaseB.json", "w"), indent=2)
print("\nwritten /tmp/phaseB.json")
