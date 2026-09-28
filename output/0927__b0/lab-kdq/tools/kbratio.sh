#!/bin/bash
# usage: kbratio.sh LOG... -- per process: arm/r19h median ratio for each (mode, what); then mean over processes
awk -v REF=${REF:-r19h} '/^KB /{for(i=2;i<=NF;i++){split($i,kv,"=");f[kv[1]]=kv[2]}; key=f["mode"]"/"f["what"]; t[FILENAME,key,f["arm"]]=f["median_ms"]; arms[f["arm"]]=1; keys[key]=1; files[FILENAME]=1; ck[FILENAME,key,f["arm"]]=f["sclk_med"]}
END{for(k in keys){printf "== %s\n",k; for(a in arms){line=sprintf("%-8s",a); s=0;n=0; for(F in files){ if((F,k,a) in t && (F,k,REF) in t){r=t[F,k,a]/t[F,k,REF]; line=line sprintf(" %.4f(%.3fms,%s)",r,t[F,k,a],ck[F,k,a]); s+=r;n++}} if(n) printf "%s  mean %.4f\n", line, s/n}}}' "$@"
