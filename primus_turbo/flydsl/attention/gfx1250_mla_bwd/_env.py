"""sys.path preamble for this kernel tree; import it BEFORE `import flydsl`.

The kernels need flydsl 0.3.4.x. The container image ships 0.2.4 in site-packages, so a
`pip install --target` copy named by $FLYDSL_PATH is prepended to sys.path. Loaded by path
under a directory-unique module name (see impl._sibling), so two trees in one process never
share it. Never import primus_turbo in the same process: its gfx950 FlyDSL tree imports
`flydsl.expr.buffer_ops`, which 0.3.x removed.
"""
import os
import sys

for _p in (os.environ.get("FLYDSL_PATH", ""),):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break


def assert_environment():
    """Assert the two things that silently produce wrong answers if they are wrong."""
    import flydsl
    import torch
    assert flydsl.__version__.startswith("0.3.4"), (
        f"flydsl 0.3.4.x required, got {flydsl.__version__} at {flydsl.__file__}")
    _fp = os.environ.get("FLYDSL_PATH", "")
    assert not _fp or flydsl.__file__.startswith(_fp), (
        f"flydsl resolved to {flydsl.__file__}, not the FLYDSL_PATH install {_fp}")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx1250" in arch, f"these kernels target gfx1250, got {arch}"
    return flydsl.__version__, flydsl.__file__, arch
