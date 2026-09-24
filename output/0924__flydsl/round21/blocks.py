import sys,re
src=open(sys.argv[1]).read()
lines=src.splitlines()
# split into blocks at labels
blocks=[];cur=None
for l in lines:
    m=re.match(r'^(\.LBB[\w\.]+):',l)
    if m:
        cur=[m.group(1),[]];blocks.append(cur)
    elif cur is not None:
        cur[1].append(l)
def c(b,p): return len([x for x in b if re.search(p,x)])
print(f"{'label':16s}{'lines':>6}{'wmma':>6}{'bufld':>7}{'dsst':>6}{'dstr':>6}{'wloadcnt':>9}{'wdscnt':>8}{'vnop':>6}{'setmsb':>7}")
for lab,b in blocks:
    if len(b)<40: continue
    print(f"{lab:16s}{len(b):6d}{c(b,'v_wmma'):6d}{c(b,'buffer_load_b128'):7d}{c(b,'ds_store_b128'):6d}{c(b,'ds_load_tr16'):6d}{c(b,'s_wait_loadcnt'):9d}{c(b,'s_wait_dscnt'):8d}{c(b,'v_nop'):6d}{c(b,'s_set_vgpr_msb'):7d}")
vg=re.search(r'\.vgpr_count:\s*(\d+)',src); sp=re.search(r'\.vgpr_spill_count:\s*(\d+)',src)
print("vgpr",vg.group(1),"spill",sp.group(1),"total_lines",len(lines))
