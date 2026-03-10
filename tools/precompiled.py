"""
Precompiled wheel support for Primus-Turbo.

Set PRIMUS_TURBO_USE_COMPILED=1 and PRIMUS_TURBO_PRECOMPILED_PATH=/path/to/wheel.whl
to extract prebuilt .so files and skip native extension compilation.
"""

import os
import zipfile
from pathlib import Path


def _strtobool(val: str) -> bool:
    return val.lower() in {"1", "true", "yes", "on"}


def use_precompiled_wheel(
    env_flag: str = "PRIMUS_TURBO_USE_COMPILED",
    path_var: str = "PRIMUS_TURBO_PRECOMPILED_PATH",
) -> bool:
    """
    Extract prebuilt .so files from a wheel into the source tree.
    Returns True so callers can skip building native extensions.
    """
    if not _strtobool(os.getenv(env_flag, "0")):
        return False

    explicit_path = os.getenv(path_var)
    if not explicit_path:
        raise EnvironmentError(
            f"{env_flag}=1 is set but {path_var} is not set. "
            "Provide the path to the precompiled wheel."
        )

    wheel_path = Path(explicit_path).expanduser().resolve()
    if not wheel_path.is_file():
        raise FileNotFoundError(
            f"{env_flag}=1 is set but {path_var} does not point to a file: {wheel_path}"
        )

    repo_root = Path(__file__).resolve().parent.parent
    print(f"[Primus-Turbo] {env_flag}=1: extracting .so files from {wheel_path}")

    extracted = []
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".so"):
                continue
            dest = repo_root / member
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            print(f"[Primus-Turbo]   extracted -> {dest}")
            extracted.append(dest)

    if not extracted:
        raise RuntimeError(
            f"No .so files found in wheel {wheel_path}. Is this a valid primus_turbo wheel?"
        )

    print(
        f"[Primus-Turbo] Precompiled wheel installed ({len(extracted)} .so files). "
        "Skipping native extension build."
    )
    return True
