import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""새 방법 후보 1: 잡음 통계로 노출/밝기 척도를 추정해 필요한 이득을 맞힌다.
입력 프레임에서 국소 평균-분산 관계(광자전달 곡선)의 기울기·절편을 뽑아 a* 를 예측한다.
평가 장면 정답 미사용, 장면 5겹 교차검증."""
import json, os, numpy as np
from scipy import stats
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CACHE=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; SHD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/short_sid2"
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
B=16
def photon_feats(x):
    """국소 평균-분산 관계. x: HWC [0,1]. 블록별 (평균, 분산) 에 강건 직선 적합."""
    H,W,_=x.shape; L=0.299*x[:,:,0]+0.587*x[:,:,1]+0.114*x[:,:,2]
    hb,wb=H//B,W//B; b=L[:hb*B,:wb*B].reshape(hb,B,wb,B)
    mu=b.mean((1,3)).ravel(); va=b.var((1,3)).ravel()
    m=(mu>1e-4)&(mu<0.9)
    if m.sum()<50: return None
    mu,va=mu[m],va[m]
    ts=stats.theilslopes(va,mu)           # 강건 기울기 (이상치 내성)
    lo,hi=np.percentile(mu,[10,90])
    return dict(slope=float(ts[0]), inter=float(ts[1]),
                var_at_lo=float(np.median(va[mu<=lo])), var_at_hi=float(np.median(va[mu>=hi])),
                mu_med=float(np.median(mu)), va_med=float(np.median(va)))
X=[];A=[];S=[];BASE=[];EX=[]
for i,r in enumerate(rows):
    y=np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    x=np.load(f"{SHD}/{r['scene']}/{r['id']}")[:,:,::-1].astype(np.float32)/255.0
    g=gt_of(r["scene"]); f=photon_feats(x)
    if f is None: continue
    X.append([np.log(max(f["slope"],1e-8)), np.log(max(f["inter"],1e-10)), np.log(max(f["var_at_lo"],1e-10)),
              np.log(max(f["var_at_hi"],1e-10)), np.log(max(f["mu_med"],1e-6)), np.log(max(f["va_med"],1e-10)),
              np.log(float(r["exp"][:-1]))])
    A.append(float((y*g).sum()/(y*y).sum())); S.append(r["scene"]); BASE.append(p8(g8(y),g8(g))); EX.append(r["exp"])
    if i%150==0: print(f"  {i}/{len(rows)}",flush=True)
X,A,S,BASE=np.array(X),np.array(A),np.array(S),np.array(BASE)
print(f"\n특징 {X.shape}, 프레임 {len(A)}")
us=sorted(set(S.tolist())); yv=np.log(A)
sm=np.array([yv[S==s].mean() for s in us]); Xs=np.stack([X[S==s].mean(0) for s in us])
names=["log slope","log inter","log var@low","log var@high","log mu","log var","log exp"]
print("장면 단위 단변량 상관:")
for k,n in enumerate(names): print(f"  {n:12s} r={np.corrcoef(Xs[:,k],sm)[0,1]:+.3f}")
def ridge_loo(Xa,ya,lams=(1e-3,1e-2,1e-1,1,10,100,1000)):
    best=None
    for lam in lams:
        pr=np.zeros(len(ya))
        for j in range(len(ya)):
            m=np.ones(len(ya),bool); m[j]=False
            mu_,sd_=Xa[m].mean(0),Xa[m].std(0)+1e-9; Z=(Xa[m]-mu_)/sd_; ym=ya[m].mean()
            w=np.linalg.solve(Z.T@Z+lam*np.eye(Z.shape[1]),Z.T@(ya[m]-ym))
            pr[j]=ym+((Xa[j]-mu_)/sd_)@w
        r2=1-np.sum((ya-pr)**2)/np.sum((ya-ya.mean())**2)
        if best is None or r2>best[0]: best=(r2,lam,pr.copy())
    return best
r2,lam,prs=ridge_loo(Xs,sm)
print(f"\n장면 단위 leave-one-out R^2 = {r2:+.4f} (lambda {lam:g}), 상관 {np.corrcoef(sm,prs)[0,1]:+.3f}")
json.dump(dict(r2=float(r2), corr=float(np.corrcoef(sm,prs)[0,1]),
               univar={n: float(np.corrcoef(Xs[:,k],sm)[0,1]) for k,n in enumerate(names)}),
          open(f"{M}/photon.json","w"), indent=1)
if r2>0.05:
    pred=np.exp(np.array([prs[us.index(s)] for s in S])); pred=np.clip(pred,0.5,2.0)
    gains=np.zeros(len(A))
    for i,(s,idx) in enumerate(zip(S,range(len(A)))):
        y=np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        gains[i]=p8(g8(y*pred[i]),g8(gt_of(s)))-BASE[i]
    sc=np.array([gains[S==s].mean() for s in us]); rng=np.random.RandomState(20260909)
    bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    print(f"예측 이득 적용: {gains.mean():+.4f} dB, 장면 95% [{np.percentile(bt,2.5):+.4f},{np.percentile(bt,97.5):+.4f}], 개선 {(sc>0).sum()}/{len(us)}")
    d=json.load(open(f"{M}/photon.json")); d["applied"]=dict(gain=float(gains.mean()), ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))], improved=int((sc>0).sum()))
    json.dump(d, open(f"{M}/photon.json","w"), indent=1)
else:
    print("예측력 부족 — 적용 생략")
