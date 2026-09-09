"""하이라이트 정합 보정을 다른 모델로 전이하고, 수축·보호 장치로 최악 장면을 잡는다. 장면 5겹 교차검증."""
import os, json, os, sys, numpy as np
R = os.environ.get("LLROOT", "."); GTD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; OUT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
rows = json.load(open(f"{R}/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
QL = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(model, i, r):
    if model == "retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a = np.load(f"{R}/repro/cache_{model}/Sony/{r['id']}.npy")
    return (a.transpose(1,2,0) if a.shape[0] == 3 else a).astype(np.float32)
def run(model):
    cf = f"{OUT}/tr_{model}.npz"
    if os.path.exists(cf):
        z = np.load(cf, allow_pickle=True); QY,QG,A,S,BASE = [z[k] for k in ("QY","QG","A","S","BASE")]
    else:
        QY,QG,A,S,BASE = [],[],[],[],[]
        for i, r in enumerate(rows):
            y = load(model, i, r); g = gt_of(r["scene"])
            QY.append(np.percentile(y,QL)); QG.append(np.percentile(g,QL))
            A.append(float((y*g).sum()/((y*y).sum()+1e-12))); S.append(r["scene"]); BASE.append(p8(g8(y), g8(g)))
        QY,QG,A,S,BASE = map(np.array,(QY,QG,A,S,BASE)); np.savez(cf, QY=QY,QG=QG,A=A,S=S,BASE=BASE)
    us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
    folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
    out = {}
    for lam_name, use_shrink in (("보정", False), ("보정+수축", True)):
        pg = np.ones(len(A))
        for te in folds:
            tr = np.array([s not in te for s in S]); trs = [s for s in us if s not in te]
            best = None
            for k in range(len(QL)):
                u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-6)); yv = np.log(np.maximum(A,1e-6))
                inner = np.zeros(tr.sum()); Su = S[tr]
                for s0 in trs:
                    m = Su != s0; b,a0 = np.polyfit(u[tr][m], yv[tr][m], 1); inner[~m] = a0 + b*u[tr][~m]
                r2 = 1 - np.sum((yv[tr]-inner)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
                if best is None or r2 > best[0]: best = (r2, k)
            k = best[1]; u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-6))
            b,a0 = np.polyfit(u[tr], np.log(np.maximum(A[tr],1e-6)), 1); p = np.exp(a0 + b*u)
            lam = 1.0
            if use_shrink:                                     # 학습 장면에서 수축계수 선택
                cand = np.arange(0.1, 1.01, 0.1); err = [np.mean((np.log(np.maximum(A[tr],1e-6)) - np.log(1 + l*(p[tr]-1)))**2) for l in cand]
                lam = float(cand[int(np.argmin(err))])
            pg[~tr] = 1 + lam*(p[~tr]-1)
        pg = np.clip(pg, 0.5, 2.0); gains = np.zeros(len(A))
        for i, r in enumerate(rows):
            y = load(model, i, r); gains[i] = p8(g8(y*pg[i]), g8(gt_of(S[i]))) - BASE[i]
        sc = np.array([gains[S==s].mean() for s in us]); boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
        out[lam_name] = dict(gain=float(gains.mean()), ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
                             improved=int((sc>0).sum()), worst=float(sc.min()), base=float(BASE.mean()))
        print(f"  {model:14s} {lam_name:8s} 기준 {BASE.mean():7.4f} → {gains.mean():+.4f} dB  95% [{out[lam_name]['ci'][0]:+.4f}, {out[lam_name]['ci'][1]:+.4f}]  개선 {out[lam_name]['improved']}/{len(us)}  최악 {sc.min():+.3f}", flush=True)
    return out
res = {}
for m in ("retinexformer", "snrnet", "lightendiff", "zerodcepp"):
    print(f"[{m}]", flush=True); res[m] = run(m)
json.dump(res, open(f"{OUT}/transfer.json","w"), indent=1)
