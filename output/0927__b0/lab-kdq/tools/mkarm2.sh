#!/bin/bash
# usage: mkarm2.sh NAME [KEY=VALUE ...] -- arm tree = OP/lab2 (= champion r29 + lab switches,
# defaults == champion) with module constants KEY overridden, e.g. DQ_EPI=True KV_U2B=True
OP=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq/OP
N=$1; shift; mkdir -p $OP/$N; cp $OP/lab2/{impl.py,_env.py,__init__.py,kernels.py} $OP/$N/
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; grep -q "^$k = " $OP/$N/kernels.py || { echo "no $k"; exit 1; }
  sed -i -E "s/^$k = [^ #]+/$k = $v/" $OP/$N/kernels.py; done
echo "$N: $(grep -E '^(DQ_|VF_|KV_U2)' $OP/$N/kernels.py | sed -E 's/ +#.*//' | tr '\n' ' ')"
