import json,glob,sys,statistics as st
d=sys.argv[1]; top=int(sys.argv[2])
code=json.load(open(d+'/code.json'))['code']
va={i:c[5] for i,c in enumerate(code)}
allt=[];tot=[];loopfrac=[]
for f in sorted(glob.glob(d+'/se*_wv*.json')):
    w=json.load(open(f))['wave']; ins=w['instructions']
    ts=[x[0] for x in ins if va.get(x[4])==top]
    dif=[b-a for a,b in zip(ts,ts[1:])]
    allt+=dif
    dur=w['end']-w['begin']; lp=(ts[-1]+ (st.median(dif) if dif else 0) -ts[0]) if ts else 0
    tot.append(dur); loopfrac.append(lp/dur if dur else 0)
    print(f.split('/')[-1], 'simd?',w.get('simd'),'dur',dur,'trips',len(ts),'med trip',st.median(dif) if dif else None,'loop frac %.2f'%(lp/dur))
print('ALL trips med',st.median(allt),'mean',st.mean(allt),'min',min(allt),'max',max(allt))
print('mean wave dur',st.mean(tot),'loop frac',st.mean(loopfrac))
