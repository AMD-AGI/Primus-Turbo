import io, sys

OLD_BLOCK = '''# CHANGED round 1 (r1.i2.g2). Triton's AMD backend lowers a transposed dot
# operand either through LDS or, with this knob, inside the thread's own
# registers. The fused backward transposes on nearly every dot, and on
# b4_s8192_hq32_hkv8_d128 the in-thread form measured 9.418 ms against 9.996 ms
# without it. setdefault, so an explicit environment setting still wins.
os.environ.setdefault("TRITON_HIP_USE_IN_THREAD_TRANSPOSE", "1")
'''

NEW_BLOCK = '''# CHANGED round 1 (r1.i2.g2). Triton's AMD backend lowers a transposed dot
# operand either through LDS or, with this knob, inside the thread's own
# registers. The fused backward transposes on nearly every dot, and on
# b4_s8192_hq32_hkv8_d128 the in-thread form measured 9.418 ms against 9.996 ms
# without it.
#
# Scoped around the launch, NOT set at import. The knob is read at compile time
# and is cache-invalidating (`get_cache_invalidating_env_vars()` reports it), and
# `validation.py` loads beat and baseline in this same process, so a process-wide
# setting would change how THEIR kernels compile and corrupt both ratios.
_ITT = "TRITON_HIP_USE_IN_THREAD_TRANSPOSE"
_ITT_AT_LAUNCH = "not launched"


@contextlib.contextmanager
def _in_thread_transpose():
    global _ITT_AT_LAUNCH
    prev = os.environ.get(_ITT)
    os.environ[_ITT] = "1"
    _ITT_AT_LAUNCH = "1"
    try:
        yield
    finally:
        if prev is None:
            del os.environ[_ITT]
        else:
            os.environ[_ITT] = prev
'''

OLD_CALL = '''        dq, dk, dv = dense_fused_backward(
            dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal
        )
'''
NEW_CALL = '''        with _in_thread_transpose():
            dq, dk, dv = dense_fused_backward(
                dout, q, k, v, out, lse, ctx.softmax_scale, ctx.causal
            )
'''

OLD_FP = '''            "in_thread_transpose": os.environ.get("TRITON_HIP_USE_IN_THREAD_TRANSPOSE", "unset"),'''
NEW_FP = '''            "in_thread_transpose": _ITT_AT_LAUNCH,'''

for path in sys.argv[1:]:
    s = io.open(path).read()
    for old, new in ((OLD_BLOCK, NEW_BLOCK), (OLD_CALL, NEW_CALL), (OLD_FP, NEW_FP)):
        assert s.count(old) == 1, (path, old[:40], s.count(old))
        s = s.replace(old, new)
    assert "import contextlib" not in s
    s = s.replace("import os\n", "import contextlib\nimport os\n", 1)
    io.open(path, "w").write(s)
    print("patched", path)
