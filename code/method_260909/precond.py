import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""전제의 예측력 검정: 출력 포화도가 규칙의 이득을 예측하는가.
장면을 출력 포화 비율로 5분위로 나눠 이득을 재고, 단조성과 상관을 검정한다.
또 작동 프레임(148장)만의 짝지은 검정과 두 노출 구간에서의 재현을 낸다."""
import json, os, collections, numpy as np
from scipy import stats
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD="/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"; Q=99.9
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
base=[];gain=[];S=[];EX=[];satfrac=[];q999=[]
for i,r in enumerate(rows):
    y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"]); Gu=g8(g)
    qv=float(np.percentile(y,Q)); p=float(np.clip(1.0/max(qv,1e-6),0.5,2.0)); b=p8(g8(y),Gu)
    base.append(b); gain.append(p8(g8(y*p),Gu)-b); S.append(r["scene"]); EX.append(r["exp"])
    satfrac.append(float((y>=0.999).mean())); q999.append(qv)
    if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
base,gain,S,EX,satfrac,q999=map(np.array,(base,gain,S,EX,satfrac,q999))
us=sorted(set(S.tolist())); sc=np.array([gain[S==s].mean() for s in us]); ss=np.array([satfrac[S==s].mean() for s in us])
OUT={}
# 5분위 단조성
qs=np.quantile(ss,[0,.2,.4,.6,.8,1.0]); bins=[]
for k in range(5):
    m=(ss>=qs[k])&(ss<=qs[k+1] if k==4 else ss<qs[k+1])
    bins.append(dict(n=int(m.sum()), sat=float(ss[m].mean()), gain=float(sc[m].mean())))
OUT["quintiles"]=bins
sp=stats.spearmanr(ss,sc); OUT["spearman"]=dict(rho=float(sp.correlation), p=float(sp.pvalue))
print("\n포화 5분위 (낮은 포화 -> 높은 포화):")
for k,bn in enumerate(bins): print(f"  {k+1}: n={bn['n']:2d} 포화 {bn['sat']*100:5.2f}%  이득 {bn['gain']:+.3f} dB")
print(f"Spearman rho={sp.correlation:+.3f} p={sp.pvalue:.4f}")
# 작동 프레임만의 짝지은 검정 (장면 단위)
act=np.abs(1.0/np.maximum(q999,1e-6)-1.0)>1e-4
acts=np.array([s for s in us if act[S==s].any()])
scA=np.array([gain[(S==s)&act].mean() for s in acts])
rng=np.random.RandomState(20260909); bt=np.array([np.mean(rng.choice(scA,len(scA))) for _ in range(20000)])
t=stats.ttest_1samp(scA,0.0); w=stats.wilcoxon(scA)
OUT["acting"]=dict(n_scenes=len(acts), gain=float(scA.mean()), ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],
                   p_t=float(t.pvalue), p_wil=float(w.pvalue))
print(f"\n작동 장면 {len(acts)}개만: {scA.mean():+.3f} dB 95%[{np.percentile(bt,2.5):+.3f},{np.percentile(bt,97.5):+.3f}] t p={t.pvalue:.4f} 부호순위 p={w.pvalue:.4f}")
# 노출별 재현
for e in sorted(set(EX.tolist())):
    m=EX==e; sce=np.array([gain[(S==s)&m].mean() for s in us if m[S==s].any()])
    if len(sce)<5: continue
    bt2=np.array([np.mean(rng.choice(sce,len(sce))) for _ in range(20000)])
    OUT[f"exp_{e}"]=dict(n=len(sce), gain=float(sce.mean()), ci=[float(np.percentile(bt2,2.5)),float(np.percentile(bt2,97.5))])
    print(f"노출 {e}: 장면 {len(sce)}개 {sce.mean():+.3f} dB 95%[{np.percentile(bt2,2.5):+.3f},{np.percentile(bt2,97.5):+.3f}]")
json.dump(OUT, open(f"{M}/precond.json","w"), indent=1)
