"""대조군과 검정: (1) 보정셋에서 맞춘 전역 상수 이득(장면별 예측 없음)과 비교, (2) 장면 단위 짝지은 검정 + Holm 보정."""
import os, json, os, sys, numpy as np
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", ".")
CACHE=os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD=os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]; LAMS=[0.0,0.25,0.5,0.75,1.0]
rows=json.load(open(f"{R}/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def linfit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);v=x.var()
    b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(model,i,r):
    if model=="retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/repro/cache_{model}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
def feats(model):
    f=f"{OUT}/tr_{model}.npz"; z=np.load(f if os.path.exists(f) else f"{OUT}/calib2_cache.npz",allow_pickle=True)
    return z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
rng=np.random.RandomState(20260909); RES={}
for model in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    QY,QG,A,S,BASE=feats(model); us=sorted(set(S.tolist()))
    perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
    gm=np.zeros(len(A)); gc=np.zeros(len(A))
    for te in folds:
        tr=np.array([s not in te for s in S]); trs=[s for s in us if s not in te]
        best=None
        for k in range(len(QL)):
            uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)); yv=np.log(np.maximum(A,1e-6)); pr=np.zeros(tr.sum()); Su=S[tr]
            for s0 in trs:
                m=Su!=s0; b,a0=linfit(uu[tr][m],yv[tr][m]); pr[~m]=a0+b*uu[tr][~m]
            r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
            if best is None or r2>best[0]: best=(r2,k)
        k=best[1]; uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))
        b,a0=linfit(uu[tr],np.log(np.maximum(A[tr],1e-6))); p=np.clip(np.exp(a0+b*uu),0.5,2.0)
        for s in us:
            m=S==s; p[m]=np.exp(np.log(p[m]).mean())
        const=float(np.exp(np.log(np.maximum(A[tr],1e-6)).mean()))          # 대조군: 보정셋 평균 이득 상수
        sub=np.where(tr)[0][::3]
        def cal_score(vals):
            return float(np.mean([p8(g8(load(model,i,rows[i])*vals[i]),g8(gt_of(S[i])))-BASE[i] for i in sub]))
        lam=LAMS[int(np.argmax([0.0 if l==0 else cal_score(1+l*(p-1)) for l in LAMS]))]
        lamc=LAMS[int(np.argmax([0.0 if l==0 else cal_score(np.full(len(A),1+l*(const-1))) for l in LAMS]))]
        for i in np.where(~tr)[0]:
            y=load(model,i,rows[i]); G=g8(gt_of(S[i]))
            gm[i]=p8(g8(y*(1+lam*(p[i]-1))),G)-BASE[i]; gc[i]=p8(g8(y*(1+lamc*(const-1))),G)-BASE[i]
    sm=np.array([gm[S==s].mean() for s in us]); sc=np.array([gc[S==s].mean() for s in us]); d=sm-sc
    bm=np.array([np.mean(rng.choice(sm,len(sm))) for _ in range(20000)])
    bd=np.array([np.mean(rng.choice(d,len(d))) for _ in range(20000)])
    pval=2*min((bd<=0).mean(),(bd>=0).mean())
    RES[model]=dict(base=float(BASE.mean()),method=float(sm.mean()),control_const=float(sc.mean()),diff=float(d.mean()),
                    ci_method=[float(np.percentile(bm,2.5)),float(np.percentile(bm,97.5))],
                    ci_diff=[float(np.percentile(bd,2.5)),float(np.percentile(bd,97.5))],p_raw=float(pval),improved=int((sm>0).sum()))
    r=RES[model]; print(f"{model:14s} 방법 {r['method']:+.4f} / 상수대조 {r['control_const']:+.4f} / 차이 {r['diff']:+.4f} "
                        f"[{r['ci_diff'][0]:+.4f},{r['ci_diff'][1]:+.4f}] p={pval:.4f}", flush=True)
ps=sorted((RES[m]["p_raw"],m) for m in RES); mtot=len(ps)
for rank,(p,m) in enumerate(ps):
    RES[m]["p_holm"]=float(min(1.0,p*(mtot-rank)))
print("\nHolm 보정:", {m: round(RES[m]["p_holm"],4) for m in RES})
json.dump(RES,open(f"{OUT}/control_stats.json","w"),indent=1)
