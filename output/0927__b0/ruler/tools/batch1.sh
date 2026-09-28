#!/bin/bash
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/ruler
R=tools/run.sh
stop() { grep -qiE "amdgpu.*(fault|error|reset)|gpu reset|ring .* timeout" runs/$1.dmesg && { echo "FAULT in $1 -- stopping"; exit 3; }; grep -q "rc=0" runs/$1.log || { echo "rc!=0 in $1 -- stopping"; exit 4; }; }
$R p05 "PYTHONHASHSEED=0" r6_a r4_a r6_b r4_b; stop p05
$R p06 "PYTHONHASHSEED=0" r4_c r6_c r4_d r6_d; stop p06
$R p07 "PYTHONHASHSEED=12345" r6_c r4_c r6_d r4_d; stop p07
$R p08 "" r4_d r4_c r4_b r4_a; stop p08
$R p09 "" r6_a l12_a r6_b l12_b; stop p09
$R p10 "" l12_a r6_a l12_b r6_b; stop p10
$R p11 "PYTHONHASHSEED=0" r6_c l12_c r6_d l12_d; stop p11
$R p12 "PYTHONHASHSEED=0" l12_d r6_d l12_c r6_c; stop p12
$R p13 "" l12_a l12_b l12_c l12_d; stop p13
$R p14 "PYTHONHASHSEED=7" r4_b r6_b l12_b r4_a r6_a l12_a; stop p14
$R p15 "PYTHONHASHSEED=7" l12_a r6_a r4_a l12_b r6_b r4_b; stop p15
SHAPE=proxy $R x01 "" r4_a r6_a l12_a r6_b; stop x01
echo BATCH1 DONE
