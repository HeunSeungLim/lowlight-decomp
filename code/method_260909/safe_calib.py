import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""안전판: 보정 강도 lambda 를 보정 장면의 실제 PSNR 이득으로 고른다(0 이면 보정 안 함).
평가 장면의 정답은 쓰지 않는다. 장면 5겹 교차검증, 장면 단위 적용."""
import json, os, sys, numpy as np
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]; LAMS=[0.0,0.25,0.5,0.75,1.0]
sys.path.insert(0, f"{R}/code"); from repro_measure import ssim_indep
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def linfit(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); v=x.var()
    b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(model,i,r):
    if model=="retinexformer": return np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{model}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
def feats(model):
    f=f"{OUT}/tr_{model}.npz"
    z=np.load(f if os.path.exists(f) else f"{OUT}/calib2_cache.npz",allow_pickle=True); return z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
rng=np.random.RandomState(20260909); RES={}
for model in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    QY,QG,A,S,BASE=feats(model); us=sorted(set(S.tolist()))
    perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
    gains=np.zeros(len(A)); ss=np.zeros(len(A)); ss0=np.zeros(len(A)); picked=[]
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
        for s in us:                                            # 장면 단위 평균
            m=S==s; p[m]=np.exp(np.log(p[m]).mean())
        # lambda 선택: 보정 장면에서 실제 PSNR 이득이 가장 큰 값
        cal=np.where(tr)[0]; sub=cal[::3]                        # 보정셋 일부만 채점해도 충분
        scores=[]
        for lam in LAMS:
            if lam==0: scores.append(0.0); continue
            d=[]
            for i in sub:
                y=load(model,i,rows[i]); d.append(p8(g8(y*(1+lam*(p[i]-1))),g8(gt_of(S[i])))-BASE[i])
            scores.append(float(np.mean(d)))
        lam=LAMS[int(np.argmax(scores))]; picked.append(lam)
        for i in np.where(~tr)[0]:
            y=load(model,i,rows[i]); G=g8(gt_of(S[i])); yc=g8(y*(1+lam*(p[i]-1)))
            gains[i]=p8(yc,G)-BASE[i]; ss[i]=ssim_indep(yc,G); ss0[i]=ssim_indep(g8(y),G)
    sc=np.array([gains[S==s].mean() for s in us]); boot=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    RES[model]=dict(base=float(BASE.mean()),gain=float(gains.mean()),ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
                    improved=int((sc>0).sum()),lams=picked,ssim0=float(ss0.mean()),ssim=float(ss.mean()),worst=float(sc.min()))
    r=RES[model]; print(f"{model:14s} {r['base']:7.4f} → {r['base']+r['gain']:7.4f} dB ({r['gain']:+.4f})  장면95% [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  개선 {r['improved']}/50  "
                        f"SSIM {r['ssim0']:.4f}→{r['ssim']:.4f}  선택 강도 {picked}", flush=True)
json.dump(RES,open(f"{OUT}/safe_calib.json","w"),indent=1)
