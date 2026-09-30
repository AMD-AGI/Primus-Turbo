import csv,sys,re
f=sys.argv[1]; H=int(sys.argv[2])
L=[r for r in csv.DictReader(open(f)) if r['Hitcount']==str(H)]
n=len(L)
isload=lambda s:s.startswith(('buffer_load','global_load'))
isvmem=lambda s:s.startswith(('buffer_','global_'))
loads=[i for i,r in enumerate(L) if isload(r['Instruction'])]
vm=[i for i,r in enumerate(L) if isvmem(r['Instruction'])]
cum=[0];
for r in L: cum.append(cum[-1]+int(r['Latency'])/H)
def back(p,k,lst):
    # k-th most recent (1-based) element of lst strictly before p, cyclic
    seq=[i for i in lst if i<p][::-1]+[i for i in lst if i>=p][::-1]
    wrap=[False]*len([i for i in lst if i<p])+[True]*len([i for i in lst if i>=p])
    return seq[k-1],wrap[k-1]
for p,r in enumerate(L):
    s=r['Instruction']
    m=re.match(r's_wait_(loadcnt|xcnt) (0x[0-9a-f]+|\d+)',s)
    if not m: continue
    N=int(m.group(2),0); lst=loads if m.group(1)=='loadcnt' else vm
    if N>=len(lst): print(f"{r['Vaddr']} {s:22s} N={N} >= #loads/trip {len(lst)}: waits on previous-trip load"); 
    q,w=back(p,N+1,lst)
    dist=(cum[p]-cum[q]) if not w else (cum[n]-cum[q]+cum[p])
    print(f"{r['Vaddr']:>6} {s:22s} stall/it={int(r['Stall'])/H:6.1f} -> load@{L[q]['Vaddr']} {L[q]['Instruction'][:48]:48s} {'(prev trip)' if w else ''} issue->wait ~{dist:6.0f} cyc (latency sum)")
print('loads/trip',len(loads),'vmem/trip',len(vm), 'trip lat sum',cum[n])
print('load positions (vaddr, cum cycles):',[(L[i]['Vaddr'],int(cum[i])) for i in loads])
