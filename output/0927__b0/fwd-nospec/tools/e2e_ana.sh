#!/bin/bash
# e2e_ana.sh <tag> <schedule>: steady tps + paired ratio, and per-trace breakdown (container, no card)
E=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/e2e; D=$E/../fwd-nospec; T=$1
docker exec fa-g0 bash -c "cd $E && /opt/venv/bin/python3 tools/steady_arms.py logs/e2e.$T.log '$2' 10" > $D/runs/e2e.$T.steady.txt 2>&1
docker exec fa-g0 bash -c "cd $E && /opt/venv/bin/python3 tools/trace_breakdown.py \$(ls traces/$T/iteration_*/*.json* | sort -V)" > $D/runs/e2e.$T.breakdown.txt 2>&1
grep -E "^arm|paired|median tps" $D/runs/e2e.$T.steady.txt
grep -E "^===|FA path" $D/runs/e2e.$T.breakdown.txt | paste - - | sed -E 's/.*iteration_([0-9]+).*arm ranges: /s\1 /'
