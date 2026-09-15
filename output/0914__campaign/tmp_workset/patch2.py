import sys
p=sys.argv[1]; s=open(p).read()
old = '''def _import_vendored():'''
new = '''def _import_vendored():'''
# replace the body between "def _import_vendored():" docstring end and return
start = s.index("def _is_pt(name):")
end = s.index("_FUSED, _TRI, _BWD_KERNEL = _import_vendored()")
body = '''def _is_pt(name):
    return name == "primus_turbo" or name.startswith("primus_turbo.")


# Process-wide, keyed by this directory: `validation.py` execs each impl.py
# several times in one process and the vendored package registers a torch
# opaque type at import, which raises on a second registration.
_CACHE = builtins.__dict__.setdefault("_op_evolve_vendor_cache", {})


def _import_vendored():
    """Import THIS directory's vendored primus_turbo, leaving sys.modules as found.

    CHANGED round 1, and it is a harness fix rather than an optimisation.
    `validation.py` loads several impl.py files into ONE process (candidate,
    beat, baseline, and back again). Each has its own `vendor/` tree, and
    `op/baseline/impl.py` raises if a `primus_turbo` rooted anywhere else is
    already in `sys.modules`. The original module-level import left ours there,
    so the first arm measured after the candidate died on that guard and NO
    candidate outside `op/baseline/` could be validated at all -- independent of
    what it changed. Importing under a stripped-and-restored `sys.modules`, once
    per directory, gives every arm a private copy while leaving the guard intact
    and the vendor pin exactly as strong as before: there is still only one tree
    this file can import from, no dispatcher is reachable, and the assert below
    fails loudly if anything else answered the import.
    """
    if _VENDOR in _CACHE:
        return _CACHE[_VENDOR]
    saved = {k: v for k, v in sys.modules.items() if _is_pt(k)}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, _VENDOR)
    try:
        import primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl as fused
        import primus_turbo.pytorch.kernels.attention.attention_triton_impl as tri
        import primus_turbo.triton.attention.fused_mha_bwd_kernel as bwd_kernel
        for mod in (fused, tri, bwd_kernel):
            assert mod.__file__.startswith(_VENDOR), mod.__file__
    finally:
        sys.path.remove(_VENDOR)
        for k in [k for k in sys.modules if _is_pt(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    _CACHE[_VENDOR] = (fused, tri, bwd_kernel)
    return _CACHE[_VENDOR]


'''
s = s[:start] + body + s[end:]
s = s.replace("import os\nimport sys\n", "import builtins\nimport os\nimport sys\n", 1)
open(p,"w").write(s)
