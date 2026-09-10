import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""v51 판정 반영 측정: q=99.95/99.99 이득, LOL 학습기준 전량 포화율, 장면 t검정·Wilcoxon, 집중도."""
import json, os, glob, numpy as np
from scipy import stats
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD="/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
QX=[99.9,99.95,99.99]; G={q:[] for q in QX}; S=[]
for i,r in enumerate(rows):
    y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"]); Gu=g8(g); b=p8(g8(y),Gu)
    for q in QX:
        p=float(np.clip(1.0/max(float(np.percentile(y,q)),1e-6),0.5,2.0)); G[q].append(p8(g8(y*p),Gu)-b)
    S.append(r["scene"])
    if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
S=np.array(S); us=sorted(set(S.tolist())); OUT={"q":{}}
rng=np.random.RandomState(20260909)
for q in QX:
    gv=np.array(G[q]); sc=np.array([gv[S==s].mean() for s in us])
    bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)]); pb=2*min((bt<=0).mean(),(bt>=0).mean())
    t=stats.ttest_1samp(sc,0.0); w=stats.wilcoxon(sc[sc!=0]) if (sc!=0).sum()>5 else None
    OUT["q"][str(q)]=dict(gain=float(gv.mean()), p_boot=float(pb), p_t=float(t.pvalue), p_wilcoxon=float(w.pvalue) if w else None,
        top1=float(np.sort(sc)[-1]/sc.sum()*100), top5=float(np.sort(sc)[-5:].sum()/sc.sum()*100),
        jack=float(np.mean([np.mean(np.delete(sc,k)) for k in range(len(sc))])))
    o=OUT["q"][str(q)]; print(f"q={q}: {o['gain']:+.4f} dB  p(boot)={o['p_boot']:.4f} p(t)={o['p_t']:.4f} p(wil)={o['p_wilcoxon']:.4f}  상위1 {o['top1']:.0f}% 상위5 {o['top5']:.0f}%", flush=True)
# 학습 기준 포화율(q별)
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QLt=[50,75,90,95,98,99,99.5,99.9]
OUT["sat_train_sony_q999"]=float((tz["QG"][:,QLt.index(99.9)]>=0.999).mean())*100
# LOL 전량 포화율
import cv2
fs=sorted(glob.glob("/data/HSL/lowlight_model/data/LOLv1/our485/high/*"))
v=[np.percentile(cv2.imread(f)[:,:,::-1].astype(np.float32)/255.,99.9) for f in fs]
OUT["lol_sat_all"]=float(np.mean(np.array(v)>=0.999))*100; OUT["lol_K_all"]=float(np.mean(v)); OUT["lol_n"]=len(fs)
print(f"LOL 학습기준 {len(fs)}장 전량: q99.9 포화율 {OUT['lol_sat_all']:.1f}%, K={OUT['lol_K_all']:.4f}")
# 회귀판(라벨 40장면) 실제 값
SC=json.load(open(f"{M}/safe_calib.json")); Q0=json.load(open(f"{M}/qualitative.json"))
sg=np.array([Q0["scene_gain"][k] for k in sorted(Q0["scene_gain"])])
OUT["regression_variant"]=dict(gain=SC["retinexformer"]["gain"], worst=float(sg.min()))
print(f"회귀판 실제: {SC['retinexformer']['gain']:+.4f} dB, 최악 장면 {sg.min():+.3f}")
json.dump(OUT, open(f"{M}/fix_v52.json","w"), indent=1)
