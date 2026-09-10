import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""순환 없는 근거만 다시 측정: 작동 장면 안에서의 견고성(상위 제거·부호검정·잭나이프),
같은 장면집합의 전 프레임 평균, 프레임 최악, 노출별 짝지은 t검정."""
import json, os, numpy as np
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
gain=[];S=[];EX=[];act=[]
for i,r in enumerate(rows):
    y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"]); Gu=g8(g)
    q=float(np.percentile(y,Q)); p=float(np.clip(1.0/max(q,1e-6),0.5,2.0))
    gain.append(p8(g8(y*p),Gu)-p8(g8(y),Gu)); S.append(r["scene"]); EX.append(r["exp"]); act.append(abs(p-1.0)>1e-4)
    if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
gain,S,EX,act=np.array(gain),np.array(S),np.array(EX),np.array(act)
us=sorted(set(S.tolist())); acts=[s for s in us if act[S==s].any()]
scA=np.array([gain[(S==s)&act].mean() for s in acts])      # 작동 프레임만
scAall=np.array([gain[S==s].mean() for s in acts])          # 같은 장면 전 프레임
rng=np.random.RandomState(20260909)
def boot(v): 
    b=np.array([np.mean(rng.choice(v,len(v))) for _ in range(20000)]); return [float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
OUT={}
OUT["acting_frames"]=dict(n=len(acts), mean=float(scA.mean()), ci=boot(scA), t=float(stats.ttest_1samp(scA,0).pvalue),
    wil=float(stats.wilcoxon(scA).pvalue), pos=int((scA>0).sum()),
    sign=float(stats.binomtest(int((scA>0).sum()), len(scA), 0.5, alternative="greater").pvalue))
k=np.argsort(-scA)[:3]; rest=np.delete(scA,k)
OUT["drop_top3"]=dict(mean=float(rest.mean()), t=float(stats.ttest_1samp(rest,0).pvalue), ci=boot(rest), n=len(rest))
jk=np.array([np.mean(np.delete(scA,j)) for j in range(len(scA))])
OUT["jackknife"]=[float(jk.min()), float(jk.max())]
OUT["acting_scenes_allframes"]=dict(mean=float(scAall.mean()), ci=boot(scAall))
OUT["worst"]=dict(scene_all50=float(np.array([gain[S==s].mean() for s in us]).min()), scene_acting=float(scA.min()), frame=float(gain.min()))
print(f"\n작동 장면 {len(acts)}개: 작동프레임 평균 {scA.mean():+.3f} CI {OUT['acting_frames']['ci']} t {OUT['acting_frames']['t']:.4f} 부호검정 {OUT['acting_frames']['sign']:.4f} 양수 {OUT['acting_frames']['pos']}/{len(scA)}")
print(f"상위3 제거: {rest.mean():+.3f} t {OUT['drop_top3']['t']:.4f} CI {OUT['drop_top3']['ci']}")
print(f"잭나이프 범위 [{jk.min():+.3f},{jk.max():+.3f}]  같은 장면 전프레임 {scAall.mean():+.3f}")
print(f"최악: 50장면 {OUT['worst']['scene_all50']:+.3f} / 작동장면 {OUT['worst']['scene_acting']:+.3f} / 프레임 {OUT['worst']['frame']:+.3f}")
# 노출별 짝지은 t (장면 단위)
for e in sorted(set(EX.tolist())):
    m=EX==e; sce=np.array([gain[(S==s)&m].mean() for s in us if m[S==s].any()])
    OUT[f"exp_{e}"]=dict(n=len(sce), mean=float(sce.mean()), t=float(stats.ttest_1samp(sce,0).pvalue), ci=boot(sce))
    print(f"노출 {e}: 장면 {len(sce)} 평균 {sce.mean():+.3f} t {OUT[f'exp_{e}']['t']:.4f}")
# 순환 제거 검정: 작동 프레임 안에서 포화도 대 이득
sat=[]
for i,r in enumerate(rows):
    if not act[i]: continue
    y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); sat.append(float((y>=0.999).mean()))
sat=np.array(sat); gA=gain[act]
sp=stats.spearmanr(sat,gA); OUT["noncircular_spearman"]=dict(rho=float(sp.correlation), p=float(sp.pvalue), n=int(act.sum()))
print(f"순환 제거(작동 프레임 내) Spearman rho {sp.correlation:+.3f} p {sp.pvalue:.3f} (n={act.sum()})")
json.dump(OUT, open(f"{M}/robust_acting.json","w"), indent=1)
