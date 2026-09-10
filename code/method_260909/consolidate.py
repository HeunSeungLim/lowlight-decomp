import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""정리 실험: 장면 단위 신뢰구간, SSIM 동반 보고, 모델별 진단(왜 되고 왜 안 되는가)."""
import json, os, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX = os.environ.get("LLCACHE", "cache")
GTD = (os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; QL = [50,75,90,95,98,99,99.5,99.9]
import sys; sys.path.insert(0, f"{R}/code"); from repro_measure import ssim_indep
rows = json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def linfit(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float); v=x.var()
    b = 0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b, float(y.mean()-b*x.mean())
def gt_of(s, _c={}):
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
    if os.path.exists(f):
        z=np.load(f,allow_pickle=True); return z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
    z=np.load(f"{OUT}/calib2_cache.npz",allow_pickle=True); return z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
rng=np.random.RandomState(20260909); RES={}
for model in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    QY,QG,A,S,BASE=feats(model); us=sorted(set(S.tolist()))
    # 진단: 하이라이트 비율과 필요한 이득의 장면 단위 상관
    k99=QL.index(99); u=np.log(QG[:,k99].mean()/np.maximum(QY[:,k99],1e-4))
    sm=np.array([np.log(A[S==s]).mean() for s in us]); um=np.array([u[S==s].mean() for s in us])
    diag=float(np.corrcoef(um,sm)[0,1])
    gains=np.zeros(len(A)); ss=np.zeros(len(A)); ss0=np.zeros(len(A))
    perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
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
        for s in te:                                       # 장면 단위로 예측 평균
            m=S==s; p[m]=np.exp(np.log(p[m]).mean())
        for i in np.where(~tr)[0]:
            y=load(model,i,rows[i]); g=gt_of(S[i]); G=g8(g)
            gains[i]=p8(g8(y*p[i]),G)-BASE[i]; ss[i]=ssim_indep(g8(y*p[i]),G); ss0[i]=ssim_indep(g8(y),G)
    sc=np.array([gains[S==s].mean() for s in us]); boot=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    RES[model]=dict(base=float(BASE.mean()), gain=float(gains.mean()), ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
                    improved=int((sc>0).sum()), ssim0=float(ss0.mean()), ssim=float(ss.mean()), diag_corr=diag, astar_sd=float(A.std(ddof=1)))
    r=RES[model]; print(f"{model:14s} 기준 {r['base']:7.4f} dB → {r['gain']:+.4f} dB  장면95% [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  개선 {r['improved']}/50  "
                        f"SSIM {r['ssim0']:.4f}→{r['ssim']:.4f}  |  하이라이트-이득 상관 {diag:+.3f}, a* 표준편차 {r['astar_sd']:.3f}", flush=True)
json.dump(RES, open(f"{OUT}/consolidate.json","w"), indent=1)
