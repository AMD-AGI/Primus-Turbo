#!/usr/bin/env bash
# Rebuild SUMMARY.md, rounds.csv and REPORT-1007.html from parts/ (CPU only, no GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -I - <<'PY'
parts = ['parts/0_head.md', 'parts/a_mainline.md', 'parts/b_backends.md',
         'parts/c_rounds.md', 'parts/d_problems.md', 'parts/z_appendix.md']
out = '\n\n'.join(open(p, encoding='utf-8').read().strip('\n') for p in parts) + '\n'
open('SUMMARY.md', 'w', encoding='utf-8').write(out)
PY
cp parts/rounds.csv rounds.csv
python3 -I tools/md2html.py SUMMARY.md REPORT-1007.html
