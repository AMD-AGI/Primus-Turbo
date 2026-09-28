#!/bin/bash
# usage: isa_stats.sh <dump_dir>  -> "vgpr spill sgpr_spill scratch md5"
S=$(ls $1/*/22_final_isa.s | head -1)
v=$(grep -m1 -E '^\s+\.vgpr_count:' $S | awk '{print $2}')
sp=$(grep -m1 -E '^\s+\.vgpr_spill_count:' $S | awk '{print $2}')
ss=$(grep -m1 -E '^\s+\.sgpr_spill_count:' $S | awk '{print $2}')
sc=$(grep -m1 -E '^\s+\.private_segment_fixed_size:' $S | awk '{print $2}')
h=$(grep -v -E '^\s*(;|\.file|\.ident)' $S | sed 's/_[0-9a-f]\{16,\}//g' | md5sum | cut -c1-12)
n=$(grep -c -E '^\s+[vsgbd][a-z_0-9]+ ' $S)
echo "vgpr=$v vspill=$sp sspill=$ss scratch=$sc insts=$n md5=$h"
