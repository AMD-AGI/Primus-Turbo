"""Price the GQA out-of-bounds workaround: per-q-head dk/dv scratch + host reduction.

The workaround for aiter's gfx1250 ASM backward (it indexes a kv-sized buffer by q head)
is to allocate dk/dv per q HEAD and reduce on the host. That has never been metered on
its own, and it sits on the critical path of the current champion.

Three arms, same process, same tensors, ABAB-interleaved:
  q      dkdv_heads="q"   -- correct kernel launch, WITHOUT the reduction
  q+red  dkdv_heads="q"   -- correct, WITH the reduction the caller must do

dkdv_heads="kv" is NOT timed here. A first version of this script timed it as a "floor"
and it did exactly what our own vendor report says it does: wrote out of bounds and took
a GCVM_L2_PROTECTION_FAULT (PERMISSION_FAULTS 0x5, RW 0x1, client TCP) at s=8192. The
driver reset that process's queues, the card stayed healthy, and the run was lost.
Timing a path that is known to write out of bounds does not measure a floor; it measures
a fault. The allocation delta it was meant to provide is arithmetic, computed below.
"""
import os, sys, time, json, statistics as st, subprocess
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
REPO = "/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo"
sys.path.insert(0, os.path.join(REPO, "tools", "gfx1250"))
sys.path.insert(0, "/home/lihuzhan/code/aiter-src")
import torch
import asm_bwd_launcher as abl

B, S, HQ, HKV, D = 4, 8192, 32, 8, 128
SCALE, REP = D ** -0.5, HQ // HKV
ITERS = 15

def sclk():
    return subprocess.run(["bash","-lc","cat /sys/class/drm/card*/device/pp_dpm_sclk|grep '\\*'"],
                          capture_output=True, text=True, timeout=10).stdout.strip()

torch.manual_seed(0)
q = torch.randn(B, S, HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.bfloat16)
do = torch.randn(B, S, HQ, D, device="cuda", dtype=torch.bfloat16)
o, lse = abl.asm_forward(q, k, v, SCALE, True)
torch.cuda.synchronize()
print(f"arch {torch.cuda.get_device_properties(0).gcnArchName}   sclk before {sclk()}")

def reduce_qheads(dk_, dv_):
    b_, s_, _, d_ = dk_.shape
    dk_r = dk_.view(b_, s_, HKV, REP, d_).float().sum(3).to(k.dtype)
    dv_r = dv_.view(b_, s_, HKV, REP, d_).float().sum(3).to(v.dtype)
    return dk_r, dv_r

def arm_q():
    return abl.asm_backward(q, k, v, o, do, lse, SCALE, dkdv_heads="q")
def arm_q_red():
    dq_, dk_, dv_ = abl.asm_backward(q, k, v, o, do, lse, SCALE, dkdv_heads="q")
    dk_, dv_ = reduce_qheads(dk_, dv_)
    return dq_, dk_, dv_

ARMS = {"q(no reduce)": arm_q, "q+reduce(shipped)": arm_q_red}

# --- resident scratch, measured not computed
print("\n--- scratch footprint ---")
elem = torch.tensor([], dtype=k.dtype).element_size()
want = 2 * B * S * HKV * D * elem                      # what a correct kernel would need
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
dq_, dk_, dv_ = abl.asm_backward(q, k, v, o, do, lse, SCALE, dkdv_heads="q")
torch.cuda.synchronize()
held = dk_.numel()*dk_.element_size() + dv_.numel()*dv_.element_size()
peak = torch.cuda.max_memory_allocated() - base
print(f"  correct kernel would need     {want/2**30:.3f} GiB   (dk/dv at [B,S,Hkv,D])")
print(f"  workaround actually holds     {held/2**30:.3f} GiB   dk.shape={tuple(dk_.shape)}")
print(f"  EXCESS                        {(held-want)/2**30:.3f} GiB   ({held/want:.1f}x)")
print(f"  peak delta over the launch    {peak/2**30:.3f} GiB")
scratch = {"needed_gib": round(want/2**30,4), "held_gib": round(held/2**30,4),
           "excess_gib": round((held-want)/2**30,4), "peak_delta_gib": round(peak/2**30,4)}
del dq_, dk_, dv_
torch.cuda.empty_cache()

# --- timing
flush = torch.empty(256*1024*1024, device="cuda", dtype=torch.uint8)
def timed(fn):
    flush.zero_()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)

for fn in ARMS.values():
    t0 = time.time()
    while time.time() - t0 < 3.0:
        fn()
torch.cuda.synchronize()

samples = {n: [] for n in ARMS}
names = list(ARMS)
for i in range(ITERS):
    order = names if i % 2 == 0 else names[::-1]
    for n in order:
        samples[n].append(timed(ARMS[n]))

print(f"\nsclk after {sclk()}")
res = {}
print("\n--- timing (ABAB, CUDA events, 256 MiB L2 flush) ---")
for n in names:
    xs = sorted(samples[n]); med = st.median(xs)
    res[n] = {"median_ms": round(med,4), "best_ms": round(xs[0],4),
              "sd_pct": round(100*st.pstdev(xs)/med,2), "n": len(xs)}
    print(f"  {n:20s} median {med:7.3f} ms   best {xs[0]:7.3f} ms   sd {res[n]['sd_pct']:5.2f}%")

qm, qrm = (res[n]["median_ms"] for n in names)
print(f"\n  the host reduction costs     {qrm - qm:+7.3f} ms  ({100*(qrm/qm-1):+5.1f}%) of the backward")
res["deltas_ms"] = {"reduce": round(qrm-qm,4)}
res["scratch"] = scratch
json.dump(res, open("/tmp/scratch_cost.json","w"), indent=2)
print("\nwritten /tmp/scratch_cost.json")
