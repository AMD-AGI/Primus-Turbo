"""quick_test_bench with dkdv builder overrides. Usage: python bench_over.py '<json overrides>' <bench args...>"""

import functools
import json
import sys

sys.path.insert(0, "/perf_apps/xiaoming/Primus-Turbo/agent/workspace/flash_attn_bwd_mla_flydsl_gfx950_20261109")
import quick_test_bench as qb  # noqa: E402

over = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
        for k, v in json.loads(sys.argv[1]).items()}
qb.mla_bwd_sbhd = functools.partial(qb.mla_bwd_sbhd, **over)
sys.argv = [sys.argv[0]] + sys.argv[2:]
print(f"overrides: {over}", flush=True)
sys.exit(qb.main())
