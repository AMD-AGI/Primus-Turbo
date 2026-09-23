"""Stage 2 gate: does the FlyDSL forward's KV block want to be bigger?

Four arms in ONE session, round-robin interleaved: aiter's prebuilt ASM forward, and
aiter's FlyDSL forward at n_block 64 (its shipping value), 128 and 256.

n_block is a real builder parameter with choices (32,64,128,256) but the host entry never
forwards it, so every shipping launch gets DEFAULT_N_BLOCK=64. We reach it by setting the
module constant, clearing the launch cache, building, and stealing the cached closure --
so all three variants exist at once and can be interleaved rather than run in blocks.

Discipline carried over from stage 1, all of it load-bearing on this box:
  - correctness BEFORE speed; an arm that fails SQNR is not timed
  - round-robin interleaving with a reversed order on odd reps: this card is VR-throttled
    and drifts, and an AAAA/BBBB split turns drift into the ratio you report
  - sclk witness around the measurement window
  - 256 MiB L2 flush between every rep
  - chunked fp32 reference: a dense [4,32,8192,8192] fp32 score tensor would be 34 GB
"""
import os, sys, time, json, subprocess, statistics as st

# [stage2 S0-b] hipBLASLt is usable here once its library is named; the fp32 reference
# below is exactly the GEMM that used to fault.
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
sys.path.insert(0, "/home/lihuzhan/.local/flydsl032")
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")

import torch, flydsl
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_batch_func as FLYDSL
from aiter.ops.mha import fmha_fwd_with_sink_asm as ASM
from aiter.ops.flydsl.kernels.fmha_gfx1250 import fmha_fwd_prefill_a16w16_m32x8 as M

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
DT, CAUSAL, SCALE = torch.bfloat16, True, D ** -0.5
ITERS, WARMUP_S = 20, 3.0
# Arms are (WMMA_ROW_PER_WAVE, n_block) pairs: "R:N,R:N,...".
# n_block alone is a measured dead end -- 128 is 3.2x slower and 256 is 9.9x, because the
# KV width drives the K/V burst and the S accumulator while the Q tile keeps its own
# registers, and 2 waves/SIMD x ~584 VGPRs overruns the 1024-VGPR file into scratch.
# The bar's tile is 128 Q x 256 KV against our 256 x 64, so it buys KV width by halving
# Q rows. R=1 halves BLOCK_M to 128 and with it the O and S accumulators, which is the
# only way the wider KV block has room to land.
ARGV = sys.argv[1] if len(sys.argv) > 1 else "2:64,1:64,1:128,1:256"
CONFIGS = [tuple(int(y) for y in x.split(":")) for x in ARGV.split(",")]

def sclk():
    try:
        return subprocess.run(["bash", "-c",
            "cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\\*'"],
            capture_output=True, text=True, timeout=10).stdout.strip().replace("\n", " ")
    except Exception as e:
        return f"(unavailable: {e})"

print(f"flydsl {flydsl.__version__} @ {flydsl.__file__}")
print(f"arch   {torch.cuda.get_device_properties(0).gcnArchName}")
print(f"shape  b={B} s={S} hq={HQ} hkv={HKV} d={D} {DT} causal={CAUSAL}")
print(f"ships as WMMA_ROW_PER_WAVE={M.WMMA_ROW_PER_WAVE} BLOCK_M={M.BLOCK_M} "
      f"n_block={M.DEFAULT_N_BLOCK}; sweeping (R,n) = {CONFIGS}")
print(f"sclk before: {sclk()}")

torch.manual_seed(0)
q = torch.randn(B, S, HQ, D, device="cuda", dtype=DT)
k = torch.randn(B, S, HKV, D, device="cuda", dtype=DT)
v = torch.randn(B, S, HKV, D, device="cuda", dtype=DT)

def _call_flydsl():
    r = FLYDSL(q, k, v, softmax_scale=SCALE, causal=CAUSAL, return_lse=True)
    if r is None:
        raise RuntimeError("FlyDSL declined this configuration (returned None)")
    return r[0]

# ---- build one cached closure per n_block, then keep them all alive -------------------
# Two things lock n_block down, and missing either one yields a silent NULL EXPERIMENT
# (three "variants" that are byte-identical and time the same to within noise):
#   1. `n_block: int = DEFAULT_N_BLOCK` is a keyword default, BOUND AT DEF TIME. Setting
#      M.DEFAULT_N_BLOCK afterwards changes nothing. Patch __kwdefaults__ instead.
#   2. build_fmha_fwd_prefill_a16w16_m32x8 is @functools.cache'd and _ensure_bshd_kernel
#      passes the same args every time, so it returns the SAME kernel. Clear the cache.
# The build-time assert below is the tripwire: a real rebuild is seconds, a cache hit 0.0s.
_BUILD = M.build_fmha_fwd_prefill_a16w16_m32x8
assert "n_block" in (_BUILD.__wrapped__.__kwdefaults__ or {}), "n_block is not a kwarg default"
_orig_n = _BUILD.__wrapped__.__kwdefaults__["n_block"]
_orig_R, _orig_BM = M.WMMA_ROW_PER_WAVE, M.BLOCK_M
variants, build_err = {}, {}
for R, n in CONFIGS:
    M._launch_fns.clear()
    _BUILD.cache_clear()
    _BUILD.__wrapped__.__kwdefaults__["n_block"] = n
    M.WMMA_ROW_PER_WAVE = R
    M.BLOCK_M = M.WMMA_M * R * M.NUM_WAVES        # BLOCK_M is a derived module global
    try:
        t0 = time.time()
        _call_flydsl(); torch.cuda.synchronize()
        dt = time.time() - t0
        variants[(R, n)] = dict(M._launch_fns)     # steal every entry this build created
        print(f"  R={R} n_block={n:3d} (BLOCK_M={M.BLOCK_M:3d}): built {len(variants[(R,n)])} "
              f"launch fn(s) in {dt:5.1f}s")
    except Exception as e:
        build_err[f"{R}:{n}"] = f"{type(e).__name__}: {e}"
        print(f"  R={R} n_block={n:3d}: BUILD/LAUNCH FAILED -- {build_err[f'{R}:{n}']}")
_BUILD.__wrapped__.__kwdefaults__["n_block"] = _orig_n
M.WMMA_ROW_PER_WAVE, M.BLOCK_M = _orig_R, _orig_BM
if len(variants) > 1 and len({id(list(v.values())[0]) for v in variants.values()}) == 1:
    print("\nEVERY VARIANT IS THE SAME OBJECT -- the sweep reached nothing. Aborting.")
    sys.exit(3)

def flydsl_arm(key):
    def go():
        M._launch_fns.clear(); M._launch_fns.update(variants[key])
        return _call_flydsl()
    return go

ARMS = {"asm": (lambda: ASM(q, k, v, SCALE, CAUSAL, True)[0])}
for key in CONFIGS:
    if key in variants:
        ARMS[f"fly_R{key[0]}n{key[1]}"] = flydsl_arm(key)
if len(ARMS) < 2:
    print("\nnothing to compare against the bar."); sys.exit(2)

# ---- correctness first ---------------------------------------------------------------
def chunked_ref_out(qb=1024):
    rep = HQ // HKV
    out = torch.empty(B, S, HQ, D, device="cuda", dtype=torch.float32)
    for b in range(B):
        for h in range(HQ):
            qh = q[b, :, h, :].float()
            kh = k[b, :, h // rep, :].float(); vh = v[b, :, h // rep, :].float()
            for s0 in range(0, S, qb):
                s1 = min(s0 + qb, S)
                sc = (qh[s0:s1] @ kh.T) * SCALE
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
for name, fn in list(ARMS.items()):
    try:
        o = fn(); torch.cuda.synchronize()
    except Exception as e:
        print(f"  {name:12s} RUN FAILED -- {type(e).__name__}: {e}"); ARMS.pop(name); continue
    fin, tot = int(torch.isfinite(o).sum()), o.numel()
    db = sqnr_db(ref, o)
    res[name] = {"sqnr_out_db": round(db, 2), "isfinite": f"{fin}/{tot}", "full_coverage": fin == tot}
    flag = "" if fin == tot and db >= 50.0 else "   <-- REJECTED"
    print(f"  {name:12s} SQNR out {db:6.2f} dB   isfinite {fin}/{tot}{flag}")
    if flag: ARMS.pop(name)
del ref; torch.cuda.empty_cache()
if "asm" not in ARMS or len(ARMS) < 2:
    print("\nthe bar or every candidate failed correctness; not timing a wrong kernel.")
    json.dump({"res": res, "build_err": build_err}, open("/tmp/fwd_nblock.json", "w"), indent=2); sys.exit(2)

# ---- speed, round-robin --------------------------------------------------------------
print("\n--- timing (round-robin interleaved, CUDA events, 256 MiB L2 flush per rep) ---")
flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
def timed(fn):
    flush.zero_()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)

for fn in ARMS.values():
    t0 = time.time()
    while time.time() - t0 < WARMUP_S: fn()
    torch.cuda.synchronize()

samples = {n: [] for n in ARMS}
for i in range(ITERS):
    order = list(ARMS.items()) if i % 2 == 0 else list(ARMS.items())[::-1]
    for name, fn in order: samples[name].append(timed(fn))

print(f"sclk after: {sclk()}\n")
# forward FLOPs, block-causal: sum over 128-row tiles, matching what the kernel does
tiles = S // 128
flop = 2 * 2 * B * HQ * D * 128 * 128 * (tiles * (tiles + 1) // 2)
for name in ARMS:
    xs = sorted(samples[name]); med = st.median(xs)
    res[name].update(median_ms=round(med, 4), best_ms=round(xs[0], 4),
                     sd_pct=round(100 * st.pstdev(xs) / med, 2), n=len(xs),
                     tflops=round(flop / (med * 1e-3) / 1e12, 1))
    print(f"  {name:12s} median {med:7.3f} ms   best {xs[0]:7.3f} ms   "
          f"sd {100*st.pstdev(xs)/med:5.2f}%   {res[name]['tflops']:7.1f} TF/s   n={len(xs)}")

a = res["asm"]["median_ms"]
print()
for name in ARMS:
    if name == "asm": continue
    print(f"  asm / {name:12s} = {a/res[name]['median_ms']:.4f}"
          f"   ({'candidate faster' if res[name]['median_ms'] < a else 'ASM faster'})")
SHIP = f"fly_R{_orig_R}n{_orig_n}"
base = res.get(SHIP, {}).get("median_ms")
if base:
    print()
    for name in ARMS:
        if name.startswith("fly_") and name != SHIP:
            print(f"  {name} vs {SHIP} (shipping) = {base/res[name]['median_ms']:.4f}x")
res["build_err"] = build_err
json.dump(res, open("/tmp/fwd_nblock.json", "w"), indent=2)
print("\nwritten /tmp/fwd_nblock.json")
