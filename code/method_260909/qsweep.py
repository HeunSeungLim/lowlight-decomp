import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""q 민감도와 q=99.9 판(K=1, 데이터 불필요) 확정 측정 + SSIM + 10쌍 Holm."""
import json, os, sys, collections, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))); sys.path.insert(0,f"{R}/code")
from repro_measure import ssim_indep
CODEX=os.environ.get("LLCACHE", "cache")
GTD="/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QLt=[50,75,90,95,98,99,99.5,99.9]
QS=[95,98,99,99.5,99.9]
KTR={q: float(np.percentile(tz["QG"][:,QLt.index(q)],50)) if False else None for q in QS}
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(m,i,r):
    if m=="retinexformer": return np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{m}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
# 학습 분할 기준영상에서 q별 K (테스트 라벨 미사용)
K={q: float(tz["QG"][:,QLt.index(q)].mean()) for q in QS}
SAT={q: float((tz["QG"][:,QLt.index(q)]>=0.999).mean())*100 for q in QS}
print("q별 K:", {q: round(K[q],4) for q in QS}); print("q별 학습기준 포화율(%):", {q: round(SAT[q],1) for q in QS})
rng=np.random.RandomState(20260909); RES={"K":K,"sat":SAT,"models":{}}
for m in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    base=[]; S=[]; G={q:[] for q in QS}; SS={q:[] for q in QS}; SS0=[]
    for i,r in enumerate(rows):
        y=load(m,i,r); g=gt_of(r["scene"]); Gu=g8(g); b=p8(g8(y),Gu); base.append(b); S.append(r["scene"])
        if m=="retinexformer": SS0.append(ssim_indep(g8(y),Gu))
        for q in QS:
            p=float(np.clip(K[q]/max(float(np.percentile(y,q)),1e-6),0.5,2.0)); yc=g8(y*p)
            G[q].append(p8(yc,Gu)-b)
            if m=="retinexformer" and q in (99.5,99.9): SS[q].append(ssim_indep(yc,Gu))
        if m=="retinexformer" and i%200==0: print(f"  {i}/{len(rows)}",flush=True)
    base,S=np.array(base),np.array(S); us=sorted(set(S.tolist())); out={}
    for q in QS:
        gv=np.array(G[q]); sc=np.array([gv[S==s].mean() for s in us]); bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
        pv=2*min((bt<=0).mean(),(bt>=0).mean())
        out[str(q)]=dict(gain=float(gv.mean()), ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))], p_raw=float(pv),
                         improved=int((sc>0).sum()), median=float(np.median(sc)), worst=float(sc.min()),
                         top5=float(np.sort(sc)[-5:].sum()/sc.sum()*100) if sc.sum()!=0 else None)
    RES["models"][m]=dict(base=float(base.mean()), q=out)
    if m=="retinexformer":
        RES["ssim"]=dict(base=float(np.mean(SS0)), q995=float(np.mean(SS[99.5])), q999=float(np.mean(SS[99.9])))
    print(f"{m:14s} " + " | ".join(f"q{q}: {out[str(q)]['gain']:+.3f} p={out[str(q)]['p_raw']:.4f}" for q in QS), flush=True)
json.dump(RES, open(f"{M}/qsweep.json","w"), indent=1)
print("SSIM(retinexformer):", RES.get("ssim"))
