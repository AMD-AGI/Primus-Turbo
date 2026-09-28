#!/bin/bash
# One row per dump: normalized ISA hash, VGPR, SGPR, spill, scratch, instruction count.
cd ${1:-/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec/isa}
for d in */; do f=$(ls $d*/22_final_isa.s | head -1)
 h=$(grep -v '^\s*\.\(file\|ident\)' $f | md5sum | cut -c1-12)
 g(){ grep -E "^\s+\.$1:" $f | awk '{print $2}' | head -1; }
 n=$(grep -cE '^\s+[a-z][a-z0-9_]+' $f)
 echo "${d%/} $h vgpr=$(g vgpr_count) sgpr=$(g sgpr_count) spill=$(g vgpr_spill_count)/$(g sgpr_spill_count) scratch=$(g private_segment_fixed_size) ninst=$n"
done | column -t
