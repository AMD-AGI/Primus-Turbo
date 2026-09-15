set -x
R=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds
# The target is relative: op.target.tflops is null and op.target.beat is
# "flex_attention ... MEASURED IN THE SAME RUN -- must be beaten by 50%".
# So the denominator of `score` has to be measured here, beside the numerator.
for r in beat 001 beat 001; do
  case $r in beat) I=beat;; *) I=$R/$r/op;; esac
  echo "---- $r"
  python -u op/benchmark.py --impl $I --shape all --iters 30 --warmup-s 3.0
done
