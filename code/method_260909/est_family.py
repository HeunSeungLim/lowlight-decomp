"""예측기 후보군을 보정셋 안쪽 교차검증으로 고른다(중첩 선택). 후보: 단일 백분위, 백분위 능형회귀, 두 백분위 비.
평가 규약은 동일: 장면 단위 적용, 강도 lambda 는 보정셋 PSNR 로, 대조군은 보정셋 상수 이득."""
import os, json, os, numpy as np
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", ".")
CACHE=os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD=os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]; LAMS=[0.0,0.25,0.5,0.75,1.0]
rows=json.load(open(f"{R}/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
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
def ridge(Z,y,lam):
    Zc=Z-Z.mean(0); yc=y-y.mean()
    w=np.linalg.solve(Zc.T@Zc+lam*np.eye(Z.shape[1])+1e-9*np.eye(Z.shape[1]), Zc.T@yc)
    return w, float(y.mean()-Z.mean(0)@w)
def design(QY,QG,tr,kind,k=None,k2=None):
    if kind=="single": return np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))[:,None]
    if kind=="ratio":  return np.stack([np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)),
                                        np.log(np.maximum(QY[:,k2],1e-4)/np.maximum(QY[:,k],1e-4))],1)
    return np.stack([np.log(np.maximum(QG[tr,j].mean(),1e-6)/np.maximum(QY[:,j],1e-4)) for j in range(len(QL))],1)
rng=np.random.RandomState(20260909); RES={}
for model in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    QY,QG,A,S,BASE=feats(model); us=sorted(set(S.tolist())); yv=np.log(np.maximum(A,1e-6))
    perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
    gm=np.zeros(len(A)); gc=np.zeros(len(A)); chosen=[]
    for te in folds:
        tr=np.array([s not in te for s in S]); trs=[s for s in us if s not in te]; Su=S[tr]
        cands=[("single",k,None,0.0) for k in range(len(QL))] + \
              [("ridge",None,None,l) for l in (0.03,0.3,3.0,30.0)] + \
              [("ratio",5,7,0.03),("ratio",6,7,0.03),("ratio",5,2,0.03)]
        best=None
        for kind,k,k2,lam in cands:
            X=design(QY,QG,tr,kind,k,k2); pr=np.zeros(tr.sum())
            for s0 in trs:
                m=Su!=s0
                if m.sum()<8: continue
                w,b0=ridge(X[tr][m],yv[tr][m],max(lam,1e-6)); pr[~m]=b0+X[tr][~m]@w
            r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
            if best is None or r2>best[0]: best=(r2,kind,k,k2,lam)
        r2,kind,k,k2,lam=best; chosen.append((kind,k,k2,lam,round(r2,3)))
        X=design(QY,QG,tr,kind,k,k2); w,b0=ridge(X[tr],yv[tr],max(lam,1e-6)); p=np.clip(np.exp(b0+X@w),0.5,2.0)
        for s in us:
            m=S==s; p[m]=np.exp(np.log(p[m]).mean())
        const=float(np.exp(yv[tr].mean())); sub=np.where(tr)[0][::3]
        cs=lambda v: float(np.mean([p8(g8(load(model,i,rows[i])*v[i]),g8(gt_of(S[i])))-BASE[i] for i in sub]))
        l1=LAMS[int(np.argmax([0.0 if l==0 else cs(1+l*(p-1)) for l in LAMS]))]
        l2=LAMS[int(np.argmax([0.0 if l==0 else cs(np.full(len(A),1+l*(const-1))) for l in LAMS]))]
        for i in np.where(~tr)[0]:
            y=load(model,i,rows[i]); G=g8(gt_of(S[i]))
            gm[i]=p8(g8(y*(1+l1*(p[i]-1))),G)-BASE[i]; gc[i]=p8(g8(y*(1+l2*(const-1))),G)-BASE[i]
    sm=np.array([gm[S==s].mean() for s in us]); sc=np.array([gc[S==s].mean() for s in us]); d=sm-sc
    bm=np.array([np.mean(rng.choice(sm,len(sm))) for _ in range(20000)]); bd=np.array([np.mean(rng.choice(d,len(d))) for _ in range(20000)])
    pv=2*min((bd<=0).mean(),(bd>=0).mean())
    RES[model]=dict(base=float(BASE.mean()),method=float(sm.mean()),control=float(sc.mean()),diff=float(d.mean()),
                    ci=[float(np.percentile(bm,2.5)),float(np.percentile(bm,97.5))],
                    ci_diff=[float(np.percentile(bd,2.5)),float(np.percentile(bd,97.5))],p_raw=float(pv),
                    improved=int((sm>0).sum()),chosen=[list(map(str,c)) for c in chosen])
    r=RES[model]; print(f"{model:14s} {r['base']:.4f}→{r['base']+r['method']:.4f} ({r['method']:+.4f})  대조 {r['control']:+.4f}  차이 {r['diff']:+.4f} [{r['ci_diff'][0]:+.4f},{r['ci_diff'][1]:+.4f}] p={pv:.4f}  개선 {r['improved']}/50", flush=True)
    print(f"   선택된 예측기: {chosen}", flush=True)
ps=sorted((RES[m]["p_raw"],m) for m in RES)
for rank,(p,m) in enumerate(ps): RES[m]["p_holm"]=float(min(1.0,p*(len(ps)-rank)))
print("\nHolm:", {m: round(RES[m]["p_holm"],4) for m in RES})
json.dump(RES,open(f"{OUT}/est_family.json","w"),indent=1)
