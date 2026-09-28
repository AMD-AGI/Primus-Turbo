#!/bin/bash
# usage: mkarm.sh NAME NW BQW PF DFUSE [VF_KV VF_Q DQ_U2 KV_U2] (True/False) -- arm tree = lab/ with constants set
OP=/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq/OP
N=$1; mkdir -p $OP/$N; cp $OP/lab/{impl.py,_env.py,__init__.py,kernels.py} $OP/$N/
VFK=${6:-False}; VFQ=${7:-False}; U2=${8:-False}; KU2=${9:-False}; sed -i -e "s/^DQ_U2 = False/DQ_U2 = $U2/" -e "s/^KV_U2 = False /KV_U2 = $KU2 /" $OP/$N/kernels.py; sed -i -e "s/^VF_KV = False /VF_KV = $VFK /" -e "s/^VF_Q = False /VF_Q = $VFQ /" $OP/$N/kernels.py
sed -i -e "s/^DQ_NW = .*/DQ_NW = $2/" -e "s/^DQ_BQW = .*/DQ_BQW = $3/" -e "s/^DQ_PF = .*/DQ_PF = $4/" -e "s/^DQ_DFUSE = .*/DQ_DFUSE = $5/" $OP/$N/kernels.py
grep -n "^DQ_\|^VF_" $OP/$N/kernels.py | tr '\n' ' '; echo
