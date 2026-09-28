#!/bin/bash
# ratios.sh TAG...  -- per process: each arm's median / current's median (<1 = faster than r20)
D=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-bwd-r19/runs
for t in "$@"; do
  grep ^RESULT $D/$t.log | awk -v tag=$t '{for(i=1;i<=NF;i++){split($i,a,"=");v[a[1]]=a[2]} arm[NR]=v["arm"]; ms[v["arm"]]=v["latency_ms"]; n=v["iters"]; s=v["sclk_start"]"-"v["sclk_end"]}
   END{printf "%s n=%s sclk %s | order", tag, n, s; for(i=1;i<=NR;i++) printf " %s", arm[i]; printf " |"; split("current_copy r019 r19h",X," "); for(j=1;j<=3;j++) printf " %s=%.4f", X[j], ms[X[j]]/ms["current"]; printf " (current %.4f ms)\n", ms["current"]}'
done
