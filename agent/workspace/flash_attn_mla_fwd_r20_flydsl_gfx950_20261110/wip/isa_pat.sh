#!/bin/bash
# Condensed instruction pattern of the hot loop: wip/isa_pat.sh FILE.s FIRST_LABEL LAST_LABEL
awk -v a="^\\$2:" -v b="^\\$3:" '$0 ~ a {on=1} on {print} $0 ~ b {exit}' $1 | grep -v '^\s*;' | awk '
/^\.LBB/ {printf "\n%s ", $1; n=0; next}
{op=$1; c="o";
 if (op ~ /^v_mfma/) c="M"; else if (op ~ /^ds_read/) c="d"; else if (op ~ /^v_accvgpr_read/) c="r"; else if (op ~ /^v_accvgpr_write/) c="w"; else if (op ~ /^v_accvgpr_mov/) c="m"; else if (op ~ /^v_exp/) c="e"; else if (op ~ /^v_permlane/) c="p"; else if (op ~ /^v_/) c="v"; else if (op ~ /^s_waitcnt/) c="W"; else if (op ~ /^s_nop/) c="n"; else if (op ~ /^s_barrier/) c="B"; else if (op ~ /^buffer_load/) c="L"; else if (op ~ /^scratch/) c="S"; else if (op ~ /^s_cbranch|^s_branch/) c="|" $2 "|";
 printf "%s", c}
END {print ""}'
