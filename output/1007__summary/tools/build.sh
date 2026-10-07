#!/usr/bin/env bash
# Rebuild SUMMARY.md / SUMMARY.en.md, the two HTML reports and the two round workbooks
# from parts/ and parts/en/ (CPU only, no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -I - <<'PY'
import os
names = ['0_head.md', 'a_mainline.md', 'b_backends.md', 'c_rounds.md', 'd_problems.md', 'z_appendix.md']
for src, dst in (('parts/', 'SUMMARY.md'), ('parts/en/', 'SUMMARY.en.md')):
    paths = [src + n for n in names]
    if not all(os.path.exists(p) for p in paths):
        print('skip', dst)
        continue
    out = '\n\n'.join(open(p, encoding='utf-8').read().strip('\n') for p in paths) + '\n'
    open(dst, 'w', encoding='utf-8').write(out)
    print('wrote', dst)
PY
python3 -I tools/md2html.py SUMMARY.md REPORT-1007.html zh
[ -f SUMMARY.en.md ] && python3 -I tools/md2html.py SUMMARY.en.md REPORT-1007.en.html en
python3 -I tools/build_xlsx.py
