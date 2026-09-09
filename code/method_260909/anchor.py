"""고정 앵커판: p = K / Q99.5(y). 보정 장면도, 분할 시드도 필요 없다.
K 는 학습 분할의 기준 영상 99.5백분위 평균에서 얻는다(테스트 라벨 미사용). K 민감도도 함께 낸다."""
import os, json, os, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", ".")
CACHE=os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD=os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; QL=[50,75,90,95,98,99,99.5,99.9]; QI=QL.index(99.5)
rows=json.load(open(f"{R}/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True)
K_train=float(tz["QG"][:,QI].mean()); print(f"학습 분할 기준 99.5백분위 평균 K={K_train:.4f} (프레임 {len(tz['QG'])})")
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(model,i,r):
    if model=="retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/repro/cache_{model}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
KS=[0.90,0.92,0.94,0.953,0.96,0.98,1.00,1.02, K_train]
rng=np.random.RandomState(20260909); RES={}
for model in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    S=[]; base=[]; G={k: [] for k in KS}; sat=[]
    for i,r in enumerate(rows):
        y=load(model,i,r); g=gt_of(r["scene"]); Gu=g8(g); q=float(np.percentile(y,99.5))
        b=p8(g8(y),Gu); base.append(b); S.append(r["scene"]); sat.append(float((y>=0.999).mean()))
        for k in KS:
            p=float(np.clip(k/max(q,1e-6),0.5,2.0)); G[k].append(p8(g8(y*p),Gu)-b)
        if model=="retinexformer" and i%200==0: print(f"  {i}/{len(rows)}",flush=True)
    S=np.array(S); base=np.array(base); us=sorted(set(S.tolist()))
    out={}
    for k in KS:
        gv=np.array(G[k]); sc=np.array([gv[S==s].mean() for s in us]); boot=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
        out[f"{k:.4f}"]=dict(gain=float(gv.mean()), ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
                             improved=int((sc>0).sum()), worst=float(sc.min()), median=float(np.median(sc)))
    RES[model]=dict(base=float(base.mean()), K_train=K_train, sweep=out, sat_mean=float(np.mean(sat)))
    b=RES[model]["base"]; kt=out[f"{K_train:.4f}"]
    print(f"{model:14s} 기준 {b:.4f} → K_train {b+kt['gain']:.4f} ({kt['gain']:+.4f}) 95%[{kt['ci'][0]:+.4f},{kt['ci'][1]:+.4f}] 개선 {kt['improved']}/50 최악 {kt['worst']:+.2f}", flush=True)
    print("   K 민감도:", {k: round(v["gain"],3) for k,v in out.items()}, flush=True)
json.dump(RES, open(f"{M}/anchor.json","w"), indent=1)
