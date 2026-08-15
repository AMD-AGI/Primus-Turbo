###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def _ensure_flydsl_on_path():
    """Fall back to a staged flydsl>=0.2 tree when site-packages only has a stale egg.

    The container ships flydsl 0.2.4 in site-packages, but a stale
    0.1.1.dev409 egg is also registered via easy-install.pth. If the 0.2.x
    install goes missing, the egg silently wins and every kernel import dies at
    `TargetAddressSpace` / `AddressSpace`. Probing for the symbols (not a
    version string) and prepending a known-good staged tree keeps the package
    importable without requiring write access to site-packages.
    """
    import importlib
    import importlib.util
    import os
    import sys

    if "flydsl" in sys.modules:
        return  # already imported; swapping it now would double-load the MLIR extension

    def _usable(pkg_dir):
        # Probe on disk only. Importing flydsl to test it would load the native
        # MLIR extension, and PyGlobals aborts if a second copy is loaded after.
        try:
            with open(os.path.join(pkg_dir, "expr", "typing.py"), "r") as f:
                return "AddressSpace" in f.read()
        except OSError:
            return False

    try:
        spec = importlib.util.find_spec("flydsl")
    except Exception:
        spec = None
    locs = list(getattr(spec, "submodule_search_locations", None) or ())
    if locs and _usable(locs[0]):
        return  # a usable flydsl is already first on sys.path

    for candidate in (
        os.environ.get("FLYDSL_FALLBACK_PATH"),
        "/perf_apps/zhuang12/MegaKernel/.flydsl024",
    ):
        if candidate and _usable(os.path.join(candidate, "flydsl")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            importlib.invalidate_caches()
            return


_ensure_flydsl_on_path()
del _ensure_flydsl_on_path

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0.dev0"

try:
    from ._build_info import __build_time__, __git_commit__
except Exception:
    __git_commit__ = "unknown"
    __build_time__ = "unknown"
