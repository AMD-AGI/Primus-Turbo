try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0.dev0"

try:
    from ._build_info import __build_time__, __git_commit__
except Exception:
    __git_commit__ = "unknown"
    __build_time__ = "unknown"
