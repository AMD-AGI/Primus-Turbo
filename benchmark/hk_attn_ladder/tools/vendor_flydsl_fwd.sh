#!/bin/bash
# vendor_flydsl_fwd.sh -- the FlyDSL MLA package of dev/lhz/dsv3-attn-hk-b0 at b12f5953 (the persistent forward the HK
# MLA forward was measured against on a08-1) into build/vendor/flydsl_b12f5953/, byte for byte from git, for the MLA
# forward ladder's "flydsl_b12f5953" baseline (mla/lab/flydsl_fwd.py). The package's SOURCE_DIGEST also reads
# primus_turbo/flydsl/utils/{buffer_ops,prims}.py next to it, so those come along (identical at both commits).
set -euo pipefail
HERE=$(readlink -f "$(dirname "$0")/..")
C=b12f5953
DST=$HERE/build/vendor/flydsl_$C
cd "$HERE"
git cat-file -e "$C^{commit}"
for f in $(git ls-tree --full-tree --name-only "$C" primus_turbo/flydsl/attention/gfx1250_mla/) \
         primus_turbo/flydsl/utils/buffer_ops.py primus_turbo/flydsl/utils/prims.py; do
  mkdir -p "$DST/$(dirname "$f")"
  git show "$C:$f" > "$DST/$f"
done
find "$DST" -type f | sort | sed "s|$HERE/||"
