#!/bin/bash
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
export BENCH=bench3
R=tools/run.sh
stop() { grep -qiE "amdgpu.*(fault|error|reset)|gpu reset|ring .* timeout" runs/$1.dmesg && { echo "FAULT in $1 -- stopping"; exit 3; }; grep -q "rc=0" runs/$1.log || { echo "rc!=0 in $1 -- stopping"; exit 4; }; }
$R a01 "" r6_a r4_a r4_b r6_b; stop a01
$R a02 "" r4_c r6_c r6_d r4_d; stop a02
$R a03 "PYTHONHASHSEED=0" r6_b r4_b r4_a r6_a; stop a03
$R a04 "PYTHONHASHSEED=0" r4_d r6_d r6_c r4_c; stop a04
$R a05 "" r6_c r4_a r4_c r6_a; stop a05
$R a06 "" r4_b r6_d r6_b r4_d; stop a06
$R q03 "" l12_a r6_a r6_b l12_b; stop q03
$R q04 "PYTHONHASHSEED=0" r6_c l12_c l12_d r6_d; stop q04
$R q05 "PYTHONHASHSEED=0" l12_d r6_d r6_c l12_c; stop q05
$R q06 "" r6_b l12_a l12_c r6_d; stop q06
$R q07 "" l12_b r6_a r6_c l12_d; stop q07
SHAPE=proxy $R y01 "" r4_a r6_a l12_a r6_b; stop y01
SHAPE=proxy $R y02 "" r6_b l12_a r6_a r4_a; stop y02
echo BATCH2 DONE
