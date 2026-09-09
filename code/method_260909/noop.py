"""규칙이 실제로 작동하는 프레임 비율과 그 안에서의 이득."""
import os, json, os, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", ".")
CACHE=os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD=os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; Q=99.9
rows=json.load(open(f"{R}/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
def load(m,i,r):
    if m=="retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/repro/cache_{m}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
OUT={}
for m in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    act=[];gain=[];S=[]
    for i,r in enumerate(rows):
        y=load(m,i,r); g=gt_of(r["scene"]); Gu=g8(g); q=float(np.percentile(y,Q))
        p=float(np.clip(1.0/max(q,1e-6),0.5,2.0)); a=abs(p-1.0)>1e-4
        act.append(a); gain.append(p8(g8(y*p),Gu)-p8(g8(y),Gu)); S.append(r["scene"])
    act,gain,S=np.array(act),np.array(gain),np.array(S); us=sorted(set(S.tolist()))
    sca=np.array([act[S==s].mean() for s in us])
    OUT[m]=dict(act_frames=float(act.mean())*100, act_scenes=float((sca>0.5).sum()),
                gain_acting=float(gain[act].mean()) if act.any() else 0.0,
                gain_all=float(gain.mean()), n_act=int(act.sum()))
    o=OUT[m]; print(f"{m:14s} 작동 프레임 {o['n_act']}/598 ({o['act_frames']:.0f}%), 작동 장면 {o['act_scenes']:.0f}/50, 작동 프레임 내 이득 {o['gain_acting']:+.3f} dB, 전체 {o['gain_all']:+.3f}")
json.dump(OUT, open(f"{M}/noop.json","w"), indent=1)
