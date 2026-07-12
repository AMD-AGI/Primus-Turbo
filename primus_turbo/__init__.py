import os as _os
import sys as _sys

# A script run from inside this dir puts primus_turbo/ itself on sys.path[0], so
# `import triton`/`import flydsl` alias to this package's own subpackages
# (primus_turbo/triton, primus_turbo/flydsl) instead of the real installed ones.
# Strip that entry; no-op for normal (non-script) imports.
_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:] = [_p for _p in _sys.path if _os.path.abspath(_p or ".") != _pkg_dir]
del _os, _sys, _pkg_dir

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0.dev0"

try:
    from ._build_info import __build_time__, __git_commit__
except Exception:
    __git_commit__ = "unknown"
    __build_time__ = "unknown"
