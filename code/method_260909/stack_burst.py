import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""버스트 설정에서의 쌓기: 같은 장면·노출의 출력들을 평균한 뒤 하이라이트 정합 보정을 얹는다.
정답은 보정 장면에서만 쓴다. 평가 단위는 (장면,노출) 묶음이며 단일 프레임 성적과 구분해 보고한다."""
import json, os, collections, numpy as np
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]; LAMS=[0.0,0.25,0.5,0.75,1.0]
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def linfit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);v=x.var(); b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
grp=collections.defaultdict(list)
for i,r in enumerate(rows): grp[(r["scene"], r["exp"])].append(i)
keys=[k for k in sorted(grp) if len(grp[k])>=2]
print(f"묶음 {len(keys)}개 (2장 이상), 장면 {len(set(k[0] for k in keys))}개, 묶음 크기 중앙값 {int(np.median([len(grp[k]) for k in keys]))}")
MY,MQ,MA,MB,MS,MSINGLE=[],[],[],[],[],[]
for k in keys:
    idx=grp[k]; g=gt_of(k[0]); G=g8(g)
    ym=np.mean([np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32) for i in idx],0)
    MQ.append(np.percentile(ym,QL)); MA.append(float((ym*g).sum()/((ym*ym).sum()+1e-12)))
    MB.append(p8(g8(ym),G)); MS.append(k[0]); MY.append(k)
    MSINGLE.append(float(np.mean([p8(g8(np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0)),G) for i in idx])))
MQ,MA,MB,MS,MSINGLE=np.array(MQ),np.array(MA),np.array(MB),np.array(MS),np.array(MSINGLE)
QG=np.array([np.percentile(gt_of(s),QL) for s in MS])
print(f"단일 프레임 평균 {MSINGLE.mean():.4f} dB → 출력 평균 {MB.mean():.4f} dB ({MB.mean()-MSINGLE.mean():+.4f})")
us=sorted(set(MS.tolist())); perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
gains=np.zeros(len(MB)); yv=np.log(np.maximum(MA,1e-6))
for te in folds:
    tr=np.array([s not in te for s in MS]); trs=[s for s in us if s not in te]; Su=MS[tr]
    best=None
    for kk in range(len(QL)):
        uu=np.log(np.maximum(QG[tr,kk].mean(),1e-6)/np.maximum(MQ[:,kk],1e-4)); pr=np.zeros(tr.sum())
        for s0 in trs:
            m=Su!=s0
            if m.sum()<8: continue
            b,a0=linfit(uu[tr][m],yv[tr][m]); pr[~m]=a0+b*uu[tr][~m]
        r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
        if best is None or r2>best[0]: best=(r2,kk)
    kk=best[1]; uu=np.log(np.maximum(QG[tr,kk].mean(),1e-6)/np.maximum(MQ[:,kk],1e-4))
    b,a0=linfit(uu[tr],yv[tr]); p=np.clip(np.exp(a0+b*uu),0.5,2.0)
    for s in us:
        m=MS==s
        if m.sum(): p[m]=np.exp(np.log(p[m]).mean())
    sub=np.where(tr)[0]
    def cs(v): return float(np.mean([p8(g8(np.mean([np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0) for i in grp[MY[j]]],0)*v[j]),g8(gt_of(MS[j])))-MB[j] for j in sub[::2]]))
    lam=LAMS[int(np.argmax([0.0 if l==0 else cs(1+l*(p-1)) for l in LAMS]))]
    for j in np.where(~tr)[0]:
        ym=np.mean([np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32) for i in grp[MY[j]]],0)
        gains[j]=p8(g8(ym*(1+lam*(p[j]-1))),g8(gt_of(MS[j])))-MB[j]
rng=np.random.RandomState(20260909); sc=np.array([gains[MS==s].mean() for s in us])
boot=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
print(f"출력 평균 위에 보정: {gains.mean():+.4f} dB  장면95% [{np.percentile(boot,2.5):+.4f},{np.percentile(boot,97.5):+.4f}]  개선 장면 {(sc>0).sum()}/{len(us)}")
print(f"\n합계: 단일 {MSINGLE.mean():.4f} → 출력평균 {MB.mean():.4f} → +보정 {MB.mean()+gains.mean():.4f} dB  (총 {MB.mean()+gains.mean()-MSINGLE.mean():+.4f})")
json.dump(dict(single=float(MSINGLE.mean()),burst=float(MB.mean()),burst_calib=float(MB.mean()+gains.mean()),
               calib_gain=float(gains.mean()),ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
               n_groups=len(keys)), open(f"{OUT}/stack_burst.json","w"), indent=1)
