# mb5.hip: switch-cost patterns between WMMA runs and ds_load runs, 16 WMMA per iteration.
# Pattern (nW, nD, sep): repeat [nW WMMA][sep][nD ds_load_b128] until 16 WMMA; sep = '' or 'v' (one v_add_f32 between).
exec(open('gen_mb4.py').read().split("kerns=[]; names=[]")[0])   # NSET, hdr, clob
def dst(j): r=96+4*(j%NSET); return f"v[{r}:{r+3}]"
kerns=[]; names=[]
src4=open('gen_mb4.py').read()
emit_src=src4.split("def emit(")[1].split("# 16 WMMA per iteration")[0]
exec("def emit(" + emit_src)
pats=[(1,1,''),(1,2,''),(1,4,''),(2,2,''),(4,4,''),(8,8,''),(2,1,''),(4,1,''),(8,1,''),(16,1,''),(1,1,'v'),(1,1,'s'),(4,4,'v'),(1,4,'v')]
for nW,nD,sep in pats:
    lines=[]; j=0; w=0
    while w<16:
        for _ in range(nW): lines.append(f"v_wmma_f32_16x16x32_bf16 %{w%8}, %8, %9, %{w%8}"); w+=1
        if sep=='v': lines.append("v_add_f32 %10, 0, %10")  # addr += 0 : touches addr VGPR, harmless
        if sep=='s': lines.append("s_nop 0")
        for _ in range(nD):
            lines.append(f"ds_load_b128 {dst(j)}, %10 offset:{(j%16)*2048}"); j+=1
            if j%8==0: lines.append("s_wait_dscnt 16")
    emit(f"p{nW}w{nD}d{sep}", lines, j, 16)
host = open('gen_mb3.py').read().split("host = r'''")[1].split("'''")[0]
host = host.replace('printf("MB3 ', 'printf("MB5 ').replace("wps <= 4", "wps <= 2").replace("for (int seg = 0; seg <= 1; seg++)", "for (int seg = 0; seg <= 0; seg++)")
cs=", ".join(f'{{"{n}", k_{n}, {l}, {w}}}' for n,l,w in names)
open("mb5.hip","w").write(hdr+"\n".join(kerns)+host.replace("%CS%",cs))
print(len(names), [(n,l) for n,l,_ in names])
