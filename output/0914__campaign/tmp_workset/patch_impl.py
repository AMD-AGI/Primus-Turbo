import sys
p=sys.argv[1]; s=open(p).read()

old_import = '''_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    # insert(0): shadow any primus_turbo installed in the image. The installed
    # copy is an older commit (f857e429) and differs materially; the baseline is
    # frozen, so it must not drift with the image.
    sys.path.insert(0, _VENDOR)

if "primus_turbo" in sys.modules and not getattr(
    sys.modules["primus_turbo"], "__file__", ""
).startswith(_VENDOR):
    raise RuntimeError(
        "an installed primus_turbo was imported before this module; the vendored "
        "baseline would not be the code under measurement"
    )

from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (  # noqa: E402
    dense_fused_backward,
    fused_backward_eligible,
)
from primus_turbo.pytorch.kernels.attention.attention_triton_impl import (  # noqa: E402
    dense_forward,
)
'''

new_import = '''_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")


def _is_pt(name):
    return name == "primus_turbo" or name.startswith("primus_turbo.")


def _import_vendored():
    """Import THIS directory's vendored primus_turbo, leaving sys.modules as found.

    CHANGED round 1, and it is a harness fix rather than an optimisation.
    `validation.py` loads several impl.py files into ONE process (candidate,
    beat, baseline, and back again). Each has its own `vendor/` tree, and
    `op/baseline/impl.py` raises if a `primus_turbo` rooted anywhere else is
    already in `sys.modules`. The original module-level import left ours there,
    so the first arm measured after the candidate died on that guard and NO
    candidate outside `op/baseline/` could be validated at all -- independent of
    what it changed. Stripping and restoring `sys.modules` around the import
    gives every arm a private copy of the package while leaving the guard
    intact and the vendor pin exactly as strong as before: there is still only
    one tree this file can import from, and no dispatcher is reachable.
    """
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
    return fused, tri, bwd_kernel


_FUSED, _TRI, _BWD_KERNEL = _import_vendored()
dense_fused_backward = _FUSED.dense_fused_backward
fused_backward_eligible = _FUSED.fused_backward_eligible
dense_forward = _TRI.dense_forward
'''
assert s.count(old_import) == 1, "import block not found"
s = s.replace(old_import, new_import)

old_fp = '''    import hashlib

    from primus_turbo.pytorch.kernels.attention.attention_fused_bwd_impl import (
        fused_backward_tile,
    )
    from primus_turbo.triton.attention.fused_mha_bwd_kernel import get_fused_bwd_config

'''
new_fp = '''    import hashlib

    fused_backward_tile = _FUSED.fused_backward_tile
    get_fused_bwd_config = _BWD_KERNEL.get_fused_bwd_config

'''
assert s.count(old_fp) == 1, "fingerprint block not found"
s = s.replace(old_fp, new_fp)
open(p, "w").write(s)
