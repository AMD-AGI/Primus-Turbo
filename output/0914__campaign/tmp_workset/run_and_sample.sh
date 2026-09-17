#!/bin/bash
# run_and_sample.sh <label> <repo> <iters> -- <args...>
LABEL=$1; REPO=$2; ITERS=$3; shift 3; [ "$1" = "--" ] && shift
LOG=/tmp/rs_${LABEL}.log; CLK=/tmp/clk_${LABEL}.txt
nohup docker exec -e GPU=2 -e PYTHONPATH=/home/lihuzhan/code/aiter-src fa-repro \
  bash -lc "cd $REPO && python3 tools/gfx1250/tune_attention.py --shape llama31-8b --iters $ITERS --warmup 100 $*" > $LOG 2>&1 &
PID=$!
/tmp/sample_clk.sh 400 > $CLK &
SPID=$!
wait $PID
kill $SPID 2>/dev/null
echo "LABEL=$LABEL"; tail -1 $LOG
