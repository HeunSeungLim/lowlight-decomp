"""새 방법 후보 2·3: 천장 앵커를 채널별로, 그리고 블록별(국소 화이트패치)로 내린다.
A) 채널별: y_c / Q999(y_c)   B) 국소: 블록별 Q999 로 이득장 만들고 매끄럽게, 포화 블록은 항등
C) A+B.  정답 미사용. 장면 부트스트랩."""
import os, json, os, numpy as np
from scipy import stats
from scipy.ndimage import zoom, uniform_filter
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
def load(m,i,r):
    if m=="retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{m}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
def glob_anchor(y):
    return float(np.clip(1.0/max(float(np.percentile(y,Q)),1e-6),0.5,2.0))
def chan_anchor(y):
    return np.array([np.clip(1.0/max(float(np.percentile(y[:,:,c],Q)),1e-6),0.5,2.0) for c in range(3)],np.float32)
def local_anchor(y, B=128, cap=1.35, sm=1):
    H,W,_=y.shape; hb,wb=H//B,W//B
    L=y.max(2)[:hb*B,:wb*B].reshape(hb,B,wb,B)
    q=np.percentile(L,Q,axis=(1,3))                     # 블록별 상위 백분위 (채널 최대값 기준)
    g=np.clip(1.0/np.maximum(q,1e-6),1.0,cap)           # 천장에 못 미친 블록만 올린다(내리지 않음)
    g=uniform_filter(g,size=3,mode="nearest") if sm else g
    return np.clip(zoom(g,(B,B),order=1),1.0,cap)[:,:,None]
rng=np.random.RandomState(20260909); RES={}
for m in ("retinexformer","snrnet","lightendiff"):
    base=[];S=[];G={k:[] for k in ("glob","chan","loc","glob+loc","chan+loc")}
    for i,r in enumerate(rows):
        y=load(m,i,r); g=gt_of(r["scene"]); Gu=g8(g); b=p8(g8(y),Gu); base.append(b); S.append(r["scene"])
        pg=glob_anchor(y); pc=chan_anchor(y); pl=local_anchor(y)
        pl=np.pad(pl,((0,y.shape[0]-pl.shape[0]),(0,y.shape[1]-pl.shape[1]),(0,0)),mode="edge") if pl.shape[:2]!=y.shape[:2] else pl
        G["glob"].append(p8(g8(y*pg),Gu)-b); G["chan"].append(p8(g8(y*pc[None,None,:]),Gu)-b)
        G["loc"].append(p8(g8(y*pl),Gu)-b); G["glob+loc"].append(p8(g8(y*pg*pl),Gu)-b)
        G["chan+loc"].append(p8(g8(y*pc[None,None,:]*pl),Gu)-b)
        if m=="retinexformer" and i%150==0: print(f"  {i}/{len(rows)}",flush=True)
    S=np.array(S); us=sorted(set(S.tolist())); RES[m]={"base":float(np.mean(base))}
    for k,v in G.items():
        gv=np.array(v); sc=np.array([gv[S==s].mean() for s in us]); bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
        RES[m][k]=dict(gain=float(gv.mean()), ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],
                       p=float(2*min((bt<=0).mean(),(bt>=0).mean())), improved=int((sc>0).sum()), worst=float(sc.min()))
    print(f"{m:14s} 기준 {RES[m]['base']:.4f}")
    for k in ("glob","chan","loc","glob+loc","chan+loc"):
        d=RES[m][k]; print(f"   {k:10s} {d['gain']:+.4f} dB 95%[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] p={d['p']:.4f} 개선 {d['improved']}/50 최악 {d['worst']:+.2f}", flush=True)
json.dump(RES, open(f"{M}/anchor_ext.json","w"), indent=1)
