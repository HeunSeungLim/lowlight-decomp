import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""최종 규칙 q=99.9, K=1(8비트 상한): Sony 4모델 + LOL 6모델, 10쌍 Holm, SSIM, 버스트, 포화 의존."""
import json, os, sys, collections, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))); sys.path.insert(0,f"{R}/code")
from repro_measure import ssim_indep
import diag_lolv1 as DL
CODEX=os.environ.get("LLCACHE", "cache")
GTD="/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
Q=99.9; rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QLt=[50,75,90,95,98,99,99.5,99.9]
Ksony=float(tz["QG"][:,QLt.index(99.9)].mean()); satS=float((tz["QG"][:,QLt.index(99.9)]>=0.999).mean())*100
print(f"Sony 학습기준 q99.9 평균 K={Ksony:.4f}, 포화율 {satS:.1f}%")
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
def load(m,i,r):
    if m=="retinexformer": return np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{m}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
rng=np.random.RandomState(20260909); RES={"K":1.0,"q":Q,"sat_train_sony":satS,"sony":{},"lol":{}}
grp=collections.defaultdict(list)
for i,r in enumerate(rows): grp[(r["scene"],r["exp"])].append(i)
for m in ("retinexformer","snrnet","lightendiff","zerodcepp"):
    base=[];gain=[];S=[];ss0=[];ss1=[];sat=[]
    for i,r in enumerate(rows):
        y=load(m,i,r); g=gt_of(r["scene"]); Gu=g8(g); b=p8(g8(y),Gu)
        p=float(np.clip(1.0/max(float(np.percentile(y,Q)),1e-6),0.5,2.0)); yc=g8(y*p)
        base.append(b); gain.append(p8(yc,Gu)-b); S.append(r["scene"]); sat.append(float((y>=0.999).mean()))
        ss0.append(ssim_indep(g8(y),Gu)); ss1.append(ssim_indep(yc,Gu))
        if m=="retinexformer" and i%200==0: print(f"  {i}/{len(rows)}",flush=True)
    base,gain,S,sat=map(np.array,(base,gain,S,sat)); us=sorted(set(S.tolist()))
    sc=np.array([gain[S==s].mean() for s in us]); bt=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    pv=2*min((bt<=0).mean(),(bt>=0).mean()); satsc=np.array([sat[S==s].mean() for s in us]); hi=satsc>np.median(satsc)
    RES["sony"][m]=dict(base=float(base.mean()),gain=float(gain.mean()),ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],
        p_raw=float(pv),improved=int((sc>0).sum()),median=float(np.median(sc)),worst=float(sc.min()),
        top5=float(np.sort(sc)[-5:].sum()/sc.sum()*100),ssim0=float(np.mean(ss0)),ssim=float(np.mean(ss1)),
        sat_lo=float(sc[~hi].mean()),sat_hi=float(sc[hi].mean()))
    r0=RES["sony"][m]; print(f"{m:14s} {r0['base']:.4f}→{r0['base']+r0['gain']:.4f} ({r0['gain']:+.4f}) 95%[{r0['ci'][0]:+.3f},{r0['ci'][1]:+.3f}] p={pv:.4f} 개선 {r0['improved']}/50 최악 {r0['worst']:+.2f} SSIM {r0['ssim0']:.4f}→{r0['ssim']:.4f}", flush=True)
# 버스트
keys=[k for k in sorted(grp) if len(grp[k])>=2]; sb=[];sa=[];SS=[]
for k in keys:
    ym=np.mean([load("retinexformer",i,rows[i]) for i in grp[k]],0); g=gt_of(k[0]); Gu=g8(g)
    b=p8(g8(ym),Gu); p=float(np.clip(1.0/max(float(np.percentile(ym,Q)),1e-6),0.5,2.0))
    sb.append(b); sa.append(p8(g8(ym*p),Gu)); SS.append(k[0])
sb,sa,SS=np.array(sb),np.array(sa),np.array(SS)
single=float(np.mean([np.mean([RES["sony"]["retinexformer"]["base"]]) for _ in keys]))
d=np.array([(sa-sb)[SS==s].mean() for s in sorted(set(SS.tolist()))]); bt=np.array([np.mean(rng.choice(d,len(d))) for _ in range(20000)])
RES["burst"]=dict(avg=float(sb.mean()),avg_anc=float(sa.mean()),gain=float((sa-sb).mean()),
                  ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],n=len(keys))
print(f"버스트: 출력평균 {sb.mean():.4f} → 앵커 {sa.mean():.4f} ({(sa-sb).mean():+.4f})")
# LOL
pairs=list(DL.build_pairs()); ids=[p[0] for p in pairs]; GT={p[0]:p[2] for p in pairs}
import glob, cv2
hi_dir="/data/HSL/lowlight_model/data/LOLv1/our485/high"
fs=sorted(glob.glob(f"{hi_dir}/*"))
Klol=float(np.mean([np.percentile(cv2.imread(f)[:,:,::-1].astype(np.float32)/255.,Q) for f in fs]))
satL=float(np.mean([np.percentile(cv2.imread(f)[:,:,::-1].astype(np.float32)/255.,Q)>=0.999 for f in fs]))*100
RES["K_lol"]=Klol; RES["sat_train_lol"]=satL; print(f"LOL 학습기준 q99.9 K={Klol:.4f}, 포화율 {satL:.1f}%")
for m in ("retinexformer","snrnet","llformer","gsad","lightendiff","uretinex","zerodcepp","cidnet_woperc"):
    d0=f"{R}/numbers/cache_{m}/LOL"
    if not os.path.isdir(d0): continue
    b=[];g=[];ok=True
    for f in ids:
        p0=f"{d0}/{f}.npy"
        if not os.path.exists(p0): ok=False; break
        a=np.load(p0); y=(a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32); Gu=GT[f]
        bb=p8(g8(y),Gu); pr=float(np.clip(1.0/max(float(np.percentile(y,Q)),1e-6),0.5,2.0))
        b.append(bb); g.append(p8(g8(y*pr),Gu)-bb)
    if not ok: continue
    b,g=np.array(b),np.array(g); bt=np.array([np.mean(rng.choice(g,len(g))) for _ in range(20000)])
    pv=2*min((bt<=0).mean(),(bt>=0).mean())
    RES["lol"][m]=dict(base=float(b.mean()),gain=float(g.mean()),ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))],p_raw=float(pv),improved=int((g>0).sum()),n=len(g))
    print(f"LOL {m:14s} {b.mean():.4f}→{b.mean()+g.mean():.4f} ({g.mean():+.4f}) p={pv:.4f}")
# 10쌍 Holm
pairs_p=[(f"Sony/{k}", v["p_raw"], v["gain"]) for k,v in RES["sony"].items()]+[(f"LOL/{k}", v["p_raw"], v["gain"]) for k,v in RES["lol"].items()]
pairs_p.sort(key=lambda x: x[1]); mtot=len(pairs_p); holm=[]
_prev=0.0
for i,(nm,p,gn) in enumerate(pairs_p):
    _adj=max(_prev,min(1.0,p*(mtot-i))); _prev=_adj; holm.append((nm, p, _adj, gn))
RES["holm"]=dict(m=mtot, table=[dict(pair=a,p_raw=b,p_holm=c,gain=d) for a,b,c,d in holm],
                 pass_pos=[a for a,b,c,d in holm if c<0.05 and d>0], pass_neg=[a for a,b,c,d in holm if c<0.05 and d<0])
print(f"\nHolm m={mtot}: 통과(이득) {RES['holm']['pass_pos']} / 통과(손해) {RES['holm']['pass_neg']}")
for a,b,c,d in holm: print(f"  {a:22s} p={b:.4f} → {c:.4f}  ({d:+.3f} dB)")
json.dump(RES, open(f"{M}/final_rule.json","w"), indent=1)
