#!/bin/bash
# campaign.sh <stage> -- the measurement campaign of the HK attention ladders, in the order it must run.
# Every GPU step goes through tools/run_gpu.sh (HKL_GPU = gpu index; container from start_gpu_container.sh).
#   A  CPU only: build every rung of the three ladders (sha256 pins must match) + self-check (no GPU)
#   B  smoke: every rung written for the ladders (Llama fwd f*, bwd b*, MLA m*; the rebuilt MLA sub-rungs r1a..r2f),
#      ONE rung per job, toy shape only, stop at the first failure
#   C  correctness: every rung + baseline, every shape of its problem kind (toy first; prod last)
#   D  timing: the main line of each ladder (+ side rungs), -r 10, JSON + Markdown tables in logs/
#   E  ATT: one thread trace of the hot kernel of every main-line rung (tools/att.sh)
# Stages are separate on purpose: look at each stage's logs before starting the next.
set -euo pipefail
HERE=$(readlink -f "$(dirname "$0")/..")
cd "$HERE"
STAGE=${1:?stage A|B|C|D|E}
RUN=tools/run_gpu.sh
LADDERS="llama/ladder_fwd.json llama/ladder_bwd.json mla/ladder_bwd.json"
case "$STAGE" in
A)
  tools/start_cpu_container.sh
  for l in $LADDERS; do python3 -m hkl.build "$l" --check; done
  docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -w "$HERE" lhz-hk-cpu python3 -m hkl.selfcheck $LADDERS
  ;;
B)
  for r in f00 f01t f02 f03b f04r f05p f06g f07l f08s f09m f10z f01 f03 f04 f05 f06 f07 f08 f09z; do
    $RUN "smoke_fwd_$r" 600 -- python3 -m hkl.test llama/ladder_fwd.json --rung "$r" --baseline --shapes toy toy_full
  done
  for r in b00 b01t b02 b03b b04f b05g b06k b07l b08d b09i b10p b11s b12l b01 b05s b05r b05n b07n b03 b04 b05 b06 b07; do
    $RUN "smoke_bwd_$r" 600 -- python3 -m hkl.test llama/ladder_bwd.json --rung "$r" --baseline --shapes toy toy_full
  done
  for r in m00 m01 m02 m03q m04b m05w m06g m07q m07k m07l m08p m08q m08h m09r m10t; do
    $RUN "smoke_mla_$r" 600 -- python3 -m hkl.test mla/ladder_bwd.json --rung "$r" --baseline --shapes toy
  done
  for r in r1a r1b r1c r1d r1e r1f r1g r1h r1i r1j r1k r1l r2a r2b r2c r2d r2e; do  # never ran in this form
    $RUN "smoke_mla_$r" 600 -- python3 -m hkl.test mla/ladder_bwd.json --rung hk_v0 "$r" --baseline --shapes toy
  done
  ;;
C)
  $RUN test_llama_fwd 3600 -- python3 -m hkl.test llama/ladder_fwd.json --json logs/test_llama_fwd.json
  $RUN test_llama_bwd 3600 -- python3 -m hkl.test llama/ladder_bwd.json --json logs/test_llama_bwd.json
  $RUN test_mla_bwd 3600 -- python3 -m hkl.test mla/ladder_bwd.json --keep-going --json logs/test_mla_bwd.json
  ;;
D)
  for l in $LADDERS; do
    n=$(dirname "$l")_$(basename "$l" .json)
    $RUN "ladder_$n" 5400 -- python3 -m hkl.ladder "$l" --side -r 10 --json "logs/ladder_$n.json" --md "logs/ladder_$n.md"
  done
  ;;
E)
  # per-kernel times (rocprofv3 kernel trace) of the multi-kernel rungs, then one thread trace per hot kernel
  for r in b00 b01t b02 b03b b04f b12l v0 p1 v1_atom2 v11_best; do tools/ktrace.sh llama/ladder_bwd.json "$r"; done
  # hk_v0 .. r4 run FlyDSL's k_delta, whose JIT launch deadlocks under rocprofv3 (e15-02, 2026-10-27): profile them
  # with r7's HIP k_delta (bitwise equal); their dK/dV and dQ kernels are unchanged
  export HKL_MLA_DELTA_DIR=$HERE/build/mla_bwd/r7
  for r in m00 m01 m02 m03q m04b m05w m06g m07q hk_v0 r1 r2 r3 r4 r7; do tools/ktrace.sh mla/ladder_bwd.json "$r"; done
  for r in f00 f01t f02 f03b f04r f05p f06g f07l f08s f09m f10z; do tools/att.sh llama/ladder_fwd.json "$r" "k_fwd_$r"; done
  for r in v0 p1 v1_lsum0 v11_best; do tools/att.sh llama/ladder_fwd.json "$r" 'k_attn_fwd_v'; done
  for r in b00 b01t b02 b03b b04f b05g b06k b07l b08d b09i b10p b11s b12l; do
    HKL_ATT_TAG=dkdv tools/att.sh llama/ladder_bwd.json "$r" "k_bwd_${r}_dkdv"
  done
  for r in b00 b01t b02 b03b; do HKL_ATT_TAG=dq tools/att.sh llama/ladder_bwd.json "$r" "k_bwd_${r}_dq"; done
  for r in v0 p1 v1_atom2 v11_best; do HKL_ATT_TAG=g4 tools/att.sh llama/ladder_bwd.json "$r" 'k_hk_bwd1?_g4'; done
  for r in m00 m01 m02 m03q m04b m05w m06g m07q; do
    HKL_ATT_TAG=dkdv tools/att.sh mla/ladder_bwd.json "$r" "k_mla_${r}_dkdv"
    HKL_ATT_TAG=dq tools/att.sh mla/ladder_bwd.json "$r" "k_mla_${r}_dq"
  done
  for r in hk_v0 r1 r2 r3 r4 r7; do
    HKL_ATT_TAG=dkdv tools/att.sh mla/ladder_bwd.json "$r" 'k_dkdv96ds'
    HKL_ATT_TAG=dq tools/att.sh mla/ladder_bwd.json "$r" 'k_dqds_c'
  done
  ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
