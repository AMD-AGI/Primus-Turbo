import os, sys, json
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import benchmark as B
r = B.measure("beat", B.get_shape("b4_s8192_hq32_hkv8_d128"), iters=20, warmup_s=3.0)
print("[warps] num_warps=%s fwd_ms=%.4f bwd_ms=%.4f bwd_tflops=%.2f bwd_spread=%.2f" % (
    os.environ.get("BEAT_BWD_NUM_WARPS","4"), r["fwd_ms"], r["bwd_ms"], r["bwd_tflops"], r["bwd_spread_pct"]))
