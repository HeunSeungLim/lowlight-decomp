"""LOL 에서 독립 재현: 같은 절차(하이라이트 정합, 장면 단위, 강도는 보정셋 PSNR로 선택)를 15장에 적용.
표본이 작아 leave-one-out 으로 평가한다."""
import os, json, os, sys, numpy as np
R=os.environ.get("LLROOT", "."); OUT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{R}/code")
import diag_lolv1 as DL
QL=[50,75,90,95,98,99,99.5,99.9]; LAMS=[0.0,0.25,0.5,0.75,1.0]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def linfit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);v=x.var(); b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
pairs=list(DL.build_pairs()); ids=[p[0] for p in pairs]; GT={p[0]: p[2] for p in pairs}
print(f"LOL 프레임 {len(pairs)}개")
RES={}
for model in ("retinexformer","snrnet","llformer","gsad","lightendiff","uretinex","zerodcepp"):
    d=f"{R}/numbers/cache_{model}/LOL"
    if not os.path.isdir(d): continue
    Y={}
    for f in ids:
        p=f"{d}/{f}.npy"
        if not os.path.exists(p): Y=None; break
        a=np.load(p); Y[f]=(a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
    if Y is None: print(f"{model}: 캐시 없음"); continue
    QY=np.array([np.percentile(Y[f],QL) for f in ids]); QG=np.array([np.percentile(GT[f].astype(np.float32)/255.0,QL) for f in ids])
    A=np.array([float((Y[f]*(GT[f].astype(np.float32)/255.0)).sum()/((Y[f]**2).sum()+1e-12)) for f in ids])
    BASE=np.array([p8(g8(Y[f]),GT[f]) for f in ids]); yv=np.log(np.maximum(A,1e-6))
    gains=np.zeros(len(ids)); gc=np.zeros(len(ids))
    for t in range(len(ids)):
        tr=np.ones(len(ids),bool); tr[t]=False
        best=None
        for k in range(len(QL)):
            uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)); pr=np.zeros(tr.sum())
            for j in range(tr.sum()):
                m=np.ones(tr.sum(),bool); m[j]=False
                b,a0=linfit(uu[tr][m],yv[tr][m]); pr[j]=a0+b*uu[tr][j]
            r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
            if best is None or r2>best[0]: best=(r2,k)
        k=best[1]; uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))
        b,a0=linfit(uu[tr],yv[tr]); p=float(np.clip(np.exp(a0+b*uu[t]),0.5,2.0)); const=float(np.exp(yv[tr].mean()))
        cs=lambda v: float(np.mean([p8(g8(Y[ids[i]]*v),GT[ids[i]])-BASE[i] for i in np.where(tr)[0]]))
        lam=LAMS[int(np.argmax([0.0 if l==0 else cs(1+l*(np.exp(a0+b*uu[tr]).mean()-1)) for l in LAMS]))]
        gains[t]=p8(g8(Y[ids[t]]*(1+lam*(p-1))),GT[ids[t]])-BASE[t]
        gc[t]=p8(g8(Y[ids[t]]*const),GT[ids[t]])-BASE[t]
    rng=np.random.RandomState(20260909); d_=gains-gc
    b1=np.array([np.mean(rng.choice(gains,len(gains))) for _ in range(20000)]); b2=np.array([np.mean(rng.choice(d_,len(d_))) for _ in range(20000)])
    RES[model]=dict(base=float(BASE.mean()),gain=float(gains.mean()),ci=[float(np.percentile(b1,2.5)),float(np.percentile(b1,97.5))],
                    control=float(gc.mean()),diff=float(d_.mean()),ci_diff=[float(np.percentile(b2,2.5)),float(np.percentile(b2,97.5))],improved=int((gains>0).sum()))
    r=RES[model]; print(f"{model:14s} {r['base']:.4f}→{r['base']+r['gain']:.4f} ({r['gain']:+.4f}) 95%[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  상수대조 {r['control']:+.4f}  차이 {r['diff']:+.4f}  개선 {r['improved']}/15", flush=True)
json.dump(RES,open(f"{OUT}/lol_replicate.json","w"),indent=1)
