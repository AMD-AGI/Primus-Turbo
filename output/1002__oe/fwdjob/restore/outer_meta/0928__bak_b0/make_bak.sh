#!/bin/bash
# B0 (ctheliosp-1b112-a37-1) end-of-campaign backup, 2026-09-28. Writes tar.gz files next to this script.
set -u
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
PT=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo
OE=/home/lihuzhan/code/2026_0910__op-evolve/op-evolve
PR=/home/lihuzhan/code/2026_0828__primus/Primus
X="--exclude=__pycache__ --exclude=core --exclude=core.[0-9]* --exclude=*.gpucore"
mkdir -p filelists
# op-evolve: local branch lhz/gfx1250 is on no remote -> bundle it
git -C $OE bundle create "$D/op-evolve_lhz-gfx1250.bundle" lhz/gfx1250 2>&1 | tail -1
( tar -C $OE $X -cf - artifacts/gfx1250-flydsl-attn-fwd-b0-20260927 artifacts/gfx1250-flydsl-attn-bwd-20260917-115934 \
    LOG.bwd LOG.fwd LOG.fwd-b0 jobs/gfx1250-flydsl-attn-fwd.yaml | pigz -p 16 > b0_opevolve_jobs.tar.gz; echo "done jobs $?" ) &
( tar -C $PT $X --exclude='output/0927__b0/*/refcache/*.pt' --exclude='output/0927__b0/*/*/refcache/*.pt' \
    -cf - output/0927__b0 | pigz -p 16 > b0_output_0927b0_full.tar.gz; echo "done output $?" ) &
( tar -C /home/lihuzhan -cf - _prof_dump | pigz -p 8 > b0_real_qkv_dumps.tar.gz; echo "done dumps $?" ) &
( tar -C /home/lihuzhan $X -cf - .claude/projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo \
    .claude/projects/-home-lihuzhan/1215a290-cfed-4b25-b0c7-1dafa0d8fa99 .claude/projects/-home-lihuzhan/1215a290-cfed-4b25-b0c7-1dafa0d8fa99.jsonl \
    .claude/plans/attn-fwd-bwd-1-baseline-aiter-peaceful-valley.md | pigz -p 8 > b0_claude_sessions.tar.gz; echo "done sessions $?" ) &
# misc: untracked Primus-Turbo files outside 0927__b0, Primus e2e configs/bench, e2e rank logs, day-1 perf data
git -C $PT ls-files --others --exclude-standard | grep -v '^output/0927__b0/' | grep -v '^output/0928__bak_b0/' | sed 's|^|Primus-Turbo/|' > filelists/misc.txt
for f in ato_test.sh benchmark/kernel/attention examples/torchtitan/configs/MI455X; do echo "Primus/$f"; done >> filelists/misc.txt
( tar -C /home/lihuzhan/code $X -cf - -T filelists/misc.txt 2026_0927__bak/perf_b0 2026_0927__bak/PERF-A0-vs-B0.md \
    -C /home/lihuzhan _dbg_l8b | pigz -p 8 > b0_misc.tar.gz; echo "done misc $?" ) &
wait
ls -la *.tar.gz *.bundle | awk '{print $5, $9}'
sha256sum *.tar.gz *.bundle > SHA256SUMS
echo ALL_DONE
