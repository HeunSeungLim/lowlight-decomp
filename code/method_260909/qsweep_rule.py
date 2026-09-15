import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""논문이 정의한 규칙(앵커 = 8비트 천장, K=1)으로 q 민감도를 다시 잰다. 규약 혼용 제거."""
import json, os, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CACHE=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
QS=[95,98,99,99.5,99.9,99.95,99.99]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QLt=[50,75,90,95,98,99,99.5,99.9]
SAT={q: float((tz["QG"][:,QLt.index(q)]>=0.999).mean())*100 for q in QS if q in QLt}
G={q:[] for q in QS}; S=[]
for i,r in enumerate(rows):
    y=np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"]); Gu=g8(g); b=p8(g8(y),Gu)
    for q in QS:
        p=float(np.clip(1.0/max(float(np.percentile(y,q)),1e-6),0.5,2.0))   # 규칙 그대로: 천장 앵커
        G[q].append(p8(g8(y*p),Gu)-b)
    S.append(r["scene"])
    if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
S=np.array(S); us=sorted(set(S.tolist())); rng=np.random.RandomState(20260909); OUT={"sat_train":SAT,"rule":{}}
for q in QS:
    gv=np.array(G[q]); sc=np.array([gv[S==s].mean() for s in us]); bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    OUT["rule"][str(q)]=dict(gain=float(gv.mean()), ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],
                             p=float(2*min((bt<=0).mean(),(bt>=0).mean())))
    print(f"q={q}: {gv.mean():+.3f} dB  95%[{np.percentile(bt,2.5):+.3f},{np.percentile(bt,97.5):+.3f}]  포화율 {SAT.get(q,'-')}", flush=True)
json.dump(OUT, open(f"{M}/qsweep_rule.json","w"), indent=1)
