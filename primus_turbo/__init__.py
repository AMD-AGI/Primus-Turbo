__version__ = "0.2.0"

try:
    from ._build_info import __commit__
except Exception:
    __commit__ = "unknown"
