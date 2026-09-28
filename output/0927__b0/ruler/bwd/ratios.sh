#!/bin/bash
# ratios.sh REF TAG...  -- per process: each arm's prod median / REF's median (>1 = slower than REF)
REF=$1; shift
D=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler/bwd/runs
for t in "$@"; do
  grep ^RESULT $D/$t.log | awk -v ref=$REF -v tag=$t '{for(i=1;i<=NF;i++){split($i,a,"=");v[a[1]]=a[2]} arm[NR]=v["arm"]; ms[v["arm"]]=v["latency_ms"]; o=v["order"]; s=v["sclk_start"]"-"v["sclk_end"]}
   END{printf "%s %s sclk %s |", tag, o, s; for(i=1;i<=NR;i++) printf " %s=%.4f(%.3fms)", arm[i], ms[arm[i]]/ms[ref], ms[arm[i]]; print ""}'
done
