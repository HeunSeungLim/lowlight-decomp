import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""국소 하이라이트 정합: 블록별 상위 백분위를 기준 통계에 맞춰 블록 이득을 예측한다.
전역 보정 위에 얹어서 남은 공간 성분(오라클 +1.19dB)을 노린다. 장면 5겹 교차검증."""
import json, os, numpy as np
from scipy.ndimage import zoom
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
z=np.load(f"{OUT}/calib2_cache.npz",allow_pickle=True); QY,QG,A,S,BASE=z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
B=64; H,W=512,960; hb,wb=H//B,W//B
def linfit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);v=x.var(); b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
F=f"{OUT}/local_cache_b{B}.npz"
if os.path.exists(F):
    zz=np.load(F); GY,GG,GB=zz["GY"],zz["GG"],zz["GB"]
else:
    GY,GG,GB=[],[],[]
    for i,r in enumerate(rows):
        y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"])
        yb=y.reshape(hb,B,wb,B,3); gb=g.reshape(hb,B,wb,B,3)
        GY.append(np.percentile(yb,[90,99],axis=(1,3,4)))          # 블록별 상위 백분위 (2,hb,wb)
        GG.append(np.percentile(gb,[90,99],axis=(1,3,4)))
        GB.append((yb*gb).sum((1,3,4))/((yb*yb).sum((1,3,4))+1e-12))
        if i%150==0: print(f"  {i}/{len(rows)}",flush=True)
    GY,GG,GB=np.array(GY),np.array(GG),np.array(GB); np.savez(F,GY=GY,GG=GG,GB=GB)
us=sorted(set(S.tolist())); perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
rng=np.random.RandomState(20260909)
print(f"\n블록 {B}px, 블록 이득/프레임 전역이득 비의 표준편차: {np.std(GB/GB.mean((1,2),keepdims=True)):.4f}")
for qi,qn in ((0,"블록 p90"),(1,"블록 p99")):
    gm=np.zeros(len(A)); gglob=np.zeros(len(A))
    for te in folds:
        tr=np.array([s not in te for s in S]); yv=np.log(np.maximum(A,1e-6))
        kb=None; best=None                                          # 전역 보정(기존 방식)
        for k in range(len(QL)):
            uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)); Su=S[tr]; pr=np.zeros(tr.sum())
            for s0 in [s for s in us if s not in te]:
                m=Su!=s0; b,a0=linfit(uu[tr][m],yv[tr][m]); pr[~m]=a0+b*uu[tr][~m]
            r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
            if best is None or r2>best[0]: best=(r2,k)
        k=best[1]; uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))
        b,a0=linfit(uu[tr],yv[tr]); pg=np.clip(np.exp(a0+b*uu),0.5,2.0)
        for s in us:
            m=S==s; pg[m]=np.exp(np.log(pg[m]).mean())
        # 국소: 블록 이득의 전역 대비 비를 국소 하이라이트 비로 회귀
        T=GG[tr,qi].mean()                                          # 학습 GT 의 블록 상위백분위 평균(스칼라)
        ul=np.log(np.maximum(T,1e-6)/np.maximum(GY[:,qi],1e-4))     # (n,hb,wb)
        tgt=np.log(np.maximum(GB/np.maximum(A[:,None,None],1e-6),1e-6))
        bb,aa=linfit(ul[tr].ravel(),tgt[tr].ravel())
        for i in np.where(~tr)[0]:
            y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); G=g8(gt_of(S[i]))
            loc=np.clip(np.exp(aa+bb*ul[i]),0.7,1.4); Lf=zoom(loc,(B,B),order=1)[:,:,None]
            gm[i]=p8(g8(y*pg[i]*Lf),G)-BASE[i]; gglob[i]=p8(g8(y*pg[i]),G)-BASE[i]
    sm=np.array([gm[S==s].mean() for s in us]); sg=np.array([gglob[S==s].mean() for s in us]); d=sm-sg
    bd=np.array([np.mean(rng.choice(d,len(d))) for _ in range(20000)])
    print(f"{qn}: 전역만 {sg.mean():+.4f} dB / 전역+국소 {sm.mean():+.4f} dB / 국소 기여 {d.mean():+.4f} [{np.percentile(bd,2.5):+.4f},{np.percentile(bd,97.5):+.4f}]  회귀기울기 {bb:+.3f}", flush=True)
