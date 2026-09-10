import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""앵커판을 LOL 6모델에 적용. K 는 LOL 학습 분할 기준영상의 99.5백분위 평균(테스트 라벨 미사용)."""
import json, os, sys, glob, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))); sys.path.insert(0,f"{R}/code")
import diag_lolv1 as DL
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
pairs=list(DL.build_pairs()); ids=[p[0] for p in pairs]; GT={p[0]:p[2] for p in pairs}
# LOL 학습 분할 기준영상에서 K
cand=[d for d in ("/data/HSL/enhance_data/lowlight_bench/LOLdataset/our485/high","/data/HSL/lowlight_model/data/LOLv1/our485/high",
                  "/data/HSL/enhance_data/LOLv1/our485/high") if os.path.isdir(d)]
if cand:
    import cv2
    fs=sorted(glob.glob(f"{cand[0]}/*"))[:120]
    K=float(np.mean([np.percentile(cv2.imread(f)[:,:,::-1].astype(np.float32)/255.,99.5) for f in fs])); src=f"{cand[0]} ({len(fs)}장)"
else:
    K=float(np.mean([np.percentile(GT[i].astype(np.float32)/255.,99.5) for i in ids])); src="테스트 기준영상(대체, 라벨 사용)"
print(f"K={K:.4f}  출처: {src}")
rng=np.random.RandomState(20260909); RES={}
for m in ("retinexformer","snrnet","llformer","gsad","lightendiff","uretinex","zerodcepp","cidnet_woperc"):
    d=f"{R}/numbers/cache_{m}/LOL"
    if not os.path.isdir(d): continue
    b=[];g=[]
    ok=True
    for f in ids:
        p=f"{d}/{f}.npy"
        if not os.path.exists(p): ok=False; break
        a=np.load(p); y=(a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32); G=GT[f]
        bb=p8(g8(y),G); pr=float(np.clip(K/max(float(np.percentile(y,99.5)),1e-6),0.5,2.0))
        b.append(bb); g.append(p8(g8(y*pr),G)-bb)
    if not ok: continue
    b,g=np.array(b),np.array(g); boot=np.array([np.mean(rng.choice(g,len(g))) for _ in range(20000)])
    pv=2*min((boot<=0).mean(),(boot>=0).mean())
    RES[m]=dict(base=float(b.mean()),gain=float(g.mean()),ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],p_raw=float(pv),improved=int((g>0).sum()),n=len(g))
    print(f"{m:14s} {b.mean():.4f} → {b.mean()+g.mean():.4f} ({g.mean():+.4f}) 95%[{RES[m]['ci'][0]:+.4f},{RES[m]['ci'][1]:+.4f}] p={pv:.3f} 개선 {RES[m]['improved']}/{len(g)}")
RES["K"]=K; RES["K_source"]=src
json.dump(RES, open(f"{M}/anchor_lol.json","w"), indent=1)
