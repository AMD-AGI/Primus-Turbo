import glob, re, sys, json, os
def cen(f):
    src=open(f).read()
    gg=lambda p:(re.search(p,src) or [None,"?"])[1]
    # per-region: split at backward branches is overkill here; global counts
    return dict(vgpr=gg(r"\.vgpr_count:\s*(\d+)"), spill=gg(r"\.vgpr_spill_count:\s*(\d+)"),
        sspill=gg(r"\.sgpr_spill_count:\s*(\d+)"), sgpr=gg(r"\.sgpr_count:\s*(\d+)"),
        scratchB=gg(r"\.private_segment_fixed_size:\s*(\d+)"),
        wmma=len(re.findall("v_wmma",src)), msb=len(re.findall("s_set_vgpr_msb",src)),
        sload=len(re.findall("scratch_load",src)), sstore=len(re.findall("scratch_store",src)),
        ds=len(re.findall(r"\bds_",src)), ninstr=len(re.findall(r"^\s+[a-z]",src,re.M)))
for d in sys.argv[1:]:
    fs=sorted(glob.glob(os.path.join(d,"*","bwd_kernel_causal.amdgcn")))
    for f in fs:
        j=f.replace(".amdgcn",".json")
        shared=json.load(open(j)).get("shared") if os.path.exists(j) else None
        print(os.path.basename(d), json.dumps(cen(f)), "shared="+str(shared))
