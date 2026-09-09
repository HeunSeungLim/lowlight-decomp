"""R5'·R7' 용 수치: 작동 집합 내 집중도, 프레임가중 대 장면가중 평균, 무변화 장면 수."""
import os, json, os, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", ".")
CACHE=os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD=os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; Q=99.9
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
gain=[];S=[];act=[];EX=[]
for i,r in enumerate(rows):
    y=np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(r["scene"]); Gu=g8(g)
    p=float(np.clip(1.0/max(float(np.percentile(y,Q)),1e-6),0.5,2.0))
    gain.append(p8(g8(y*p),Gu)-p8(g8(y),Gu)); S.append(r["scene"]); act.append(abs(p-1.0)>1e-4); EX.append(r["exp"])
    if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
gain,S,act,EX=np.array(gain),np.array(S),np.array(act),np.array(EX)
us=sorted(set(S.tolist())); acts=[s for s in us if act[S==s].any()]
scA=np.array([gain[(S==s)&act].mean() for s in acts]); sc=np.array([gain[S==s].mean() for s in us])
OUT=dict(frame_weighted=float(gain.mean()), scene_weighted=float(sc.mean()),
         unchanged_scenes=int((np.abs(sc)<1e-9).sum()), n_acting=len(acts),
         best_share_acting=float(np.sort(scA)[-1]/scA.sum()*100), top3_share_acting=float(np.sort(scA)[-3:].sum()/scA.sum()*100),
         best_share_pooled=float(np.sort(sc)[-1]/sc.sum()*100),
         exp_scenes={e:int(len({s for s,m in zip(S,EX==e) if m})) for e in sorted(set(EX.tolist()))},
         exp_frames={e:int((EX==e).sum()) for e in sorted(set(EX.tolist()))})
print(json.dumps(OUT, indent=1, ensure_ascii=False))
json.dump(OUT, open(f"{M}/shares.json","w"), indent=1)
