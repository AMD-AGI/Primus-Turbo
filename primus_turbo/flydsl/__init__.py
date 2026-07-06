import os
import tempfile

# WORKAROUND: flydsl autotune's disk cache is a single JSON overwritten wholesale.
# EP ranks share a filesystem and key on their own `rank`, so concurrent writers
# clobber each other -> some ranks hit / others re-tune -> cross-rank barriers
# desync. Per-PID dirs keep every rank tuning in lockstep.
# Remove once upstream flydsl makes the cache write per-key (merge, not overwrite).
os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"] = os.path.join(
    os.environ.get("FLYDSL_AUTOTUNE_CACHE_DIR", os.path.join(tempfile.gettempdir(), "flydsl_autotune")),
    f"pid_{os.getpid()}",
)
