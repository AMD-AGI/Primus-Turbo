#!/bin/bash
for v in op032 op0341 op0341 op032 op032 op0341; do
  timeout 900 docker exec -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_$v fa-repro bash -c "cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd341 && /opt/venv/bin/python3 fwdab.py --impl /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd341/$v --shape prod --iters 51 --json /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0925__flydsl/fwd341/ab_prod.jsonl 2>&1 | grep RESULT" || echo "FAIL $v rc=$?"
done
echo DONE
