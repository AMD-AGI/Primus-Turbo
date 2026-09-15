set -x
echo "=== triton cache BEFORE clear: $(du -sh /root/.triton/cache 2>/dev/null)"
rm -rf /root/.triton/cache
echo "=== cleared: $(ls /root/.triton/cache 2>&1)"
R=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415/rounds
echo "############ UNIT TESTS (cold cache, this is also the first build)"
python -u op/ut/correctness.py $R/001/op
echo "ut_rc=$?"
echo "############ BENCH, back to back, same session, same device"
for pass in 1 2; do
  for r in 001 000 000 001; do
    echo "---- pass=$pass round=$r"
    python -u op/benchmark.py --impl $R/$r/op --shape all --iters 30 --warmup-s 3.0
  done
done
