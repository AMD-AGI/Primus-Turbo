#!/bin/bash
# run_round.sh N [full|quick] [TAG]: bench on $GPUS (default 2,3 = EP2) of the repo tree, one torchrun per shape, merged into
# rounds/round-N/artifacts/bench[_TAG].csv. full first runs the focused tests: EP4 one test per process (a
# second symmetric-buffer key inside one EP4 process can hang its rendezvous), EP2 on GPUs 0,1, EP1 on GPU 0,
# and the gfx1250 FlyDSL grouped GEMM tests on GPU 0. Every call compiles into a fresh FlyDSL cache dir.
C=$(cd "$(dirname "$0")" && pwd)
N=$1; LEVEL=${2:-quick}; TAG=${3:+_$3}
A=$C/rounds/round-$N/artifacts
mkdir -p "$A"
R=/home/xiaompen/Turbo
CAP=/home/xiaompen/mega_scratch/capped.sh
CACHE=/home/xiaompen/mega_scratch/l2c/cache/round-$N$TAG-$(date +%s)
EXEC="docker exec -w $R -e PYTHONPATH=$R -e TURBO_SRC=$R -e TRITON_HIP_USE_EXPERT_SCHEDULING=0 -e FLYDSL_RUNTIME_CACHE_DIR=$CACHE"
CT=xiaompen_mega_gfx1250_ep4
EPT=tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe_ep.py
# GPUs 0,1 belong to another user from 2026-10-27 15:30: default GPUS=2,3 (EP4 tests then skipped)
GPUS=${GPUS:-2,3}; NP=$(echo $GPUS | tr ',' ' ' | wc -w); MPORT=${MPORT:-29617}
G1=${GPUS%%,*}; G2=$(echo $GPUS | cut -d, -f1-2)
if [ "$LEVEL" = full ]; then
    if (( NP >= 4 )); then
        for t in test_deepseek_v3 test_ragged_skewed test_one_token test_staged_bias_gpt_oss; do
            $EXEC -e HIP_VISIBLE_DEVICES=$GPUS $CT bash -c "ulimit -c 0; $CAP 900 $A/test_ep4_$t.log \
                python3 -m pytest $EPT -q -p no:cacheprovider -k $t" > /dev/null
            echo "EP4 $t: $(grep -E 'passed|failed|KILLED' $A/test_ep4_$t.log | tail -n 1)"
        done
    else
        echo "EP4 tests: skipped (GPUS=$GPUS)"
    fi
    $EXEC -e HIP_VISIBLE_DEVICES=$G2 $CT bash -c "ulimit -c 0; $CAP 1500 $A/test_ep2.log \
        python3 -m pytest $EPT -q -p no:cacheprovider" > /dev/null
    echo "EP2: $(grep -E 'passed|failed|KILLED' $A/test_ep2.log | tail -n 1)"
    $EXEC -e HIP_VISIBLE_DEVICES=$G1 $CT bash -c "ulimit -c 0; $CAP 1500 $A/test_ep1.log \
        python3 -m pytest tests/pytorch/ops/test_mega_bf16_gfx1250_gemm_combine.py \
        tests/pytorch/ops/test_mega_bf16_gfx1250_dispatch_gemm.py tests/pytorch/ops/test_mega_bf16_gfx1250_fused_moe.py \
        -q -p no:cacheprovider" > /dev/null
    echo "EP1: $(grep -E 'passed|failed|KILLED' $A/test_ep1.log | tail -n 1)"
    $EXEC -e HIP_VISIBLE_DEVICES=$G1 -e PRIMUS_TURBO_GROUPED_GEMM_BACKEND=FLYDSL $CT bash -c "ulimit -c 0; \
        $CAP 1500 $A/test_gg.log python3 -m pytest tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_shapes.py \
        -q -p no:cacheprovider" > /dev/null
    echo "grouped GEMM gfx1250: $(grep -E 'passed|failed|KILLED' $A/test_gg.log | tail -n 1)"
fi
SHAPES="dsv3_t8192_random dsv3_t8192_skew1 dsv3_t4096_random"
for s in $SHAPES; do
    $EXEC -e HIP_VISIBLE_DEVICES=$GPUS -e ONLY_SHAPE=$s $CT bash -c "ulimit -c 0; $CAP 900 \
        $A/quick${TAG}_$s.log torchrun --nproc-per-node $NP --master-port $MPORT $C/quick_test_bench.py \
        --csv $A/bench${TAG}_$s.csv" > /dev/null
    grep -E "^dsv3|KILLED" "$A/quick${TAG}_$s.log" | sed -E 's/snr\S+=\S+ //'
done
python3 - "$A" "$TAG" $SHAPES <<'EOF'
import csv, os, statistics, sys
a, tag, shapes = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for s in shapes:
    f = f"{a}/bench{tag}_{s}.csv"
    rows += list(csv.DictReader(open(f))) if os.path.exists(f) else [dict(label=s, check="MISSING")]
ok = len(rows) == len(shapes) and all(r["check"] == "PASS" for r in rows)
gm = lambda k: statistics.geometric_mean(float(r[k]) for r in rows) if ok else 0.0
print(f"AGGREGATE step {gm('step_tflops'):.3f} TF  fwd {gm('fwd_tflops'):.3f} TF  bwd {gm('bwd_tflops'):.3f} TF"
      f"  ({sum(r['check'] == 'PASS' for r in rows)}/{len(shapes)} PASS)")
with open(f"{a}/bench{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
EOF
