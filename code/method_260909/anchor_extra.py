import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""앵커판 보강 측정: (1) 시드 무관성 확인(분할이 없으므로 시드 자체가 없음을 대조군으로 보임),
(2) 포화 퇴화 진단, (3) 버스트 합성, (4) LOL 재측정, (5) Holm 족을 실제 적용 쌍 10개로."""
import json, os, collections, numpy as np
M=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CACHE=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QL=[50,75,90,95,98,99,99.5,99.9]
K=float(tz["QG"][:,QL.index(99.5)].mean()); KSAT=float((tz["QG"][:,QL.index(99.5)]>=0.999).mean())
print(f"K={K:.4f}, 학습 기준영상 중 99.5백분위 포화 비율 {KSAT*100:.1f}%")
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
def load(m,i,r):
    if m=="retinexformer": return np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{m}/Sony/{r['id']}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
rng=np.random.RandomState(20260909); OUT={}
grp=collections.defaultdict(list)
for i,r in enumerate(rows): grp[(r["scene"],r["exp"])].append(i)
for m in ("retinexformer",):
    base=[];gain=[];S=[];sat=[];q=[]
    for i,r in enumerate(rows):
        y=load(m,i,r); g=gt_of(r["scene"]); G=g8(g); qq=float(np.percentile(y,99.5))
        b=p8(g8(y),G); p=float(np.clip(K/max(qq,1e-6),0.5,2.0))
        base.append(b); gain.append(p8(g8(y*p),G)-b); S.append(r["scene"]); sat.append(float((y>=0.999).mean())); q.append(qq)
        if i%200==0: print(f"  {i}/{len(rows)}",flush=True)
    base,gain,S,sat,q=map(np.array,(base,gain,S,sat,q)); us=sorted(set(S.tolist()))
    sc=np.array([gain[S==s].mean() for s in us]); satsc=np.array([sat[S==s].mean() for s in us])
    boot=np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)]); pv=2*min((boot<=0).mean(),(boot>=0).mean())
    hi=satsc>np.median(satsc)
    OUT["retinexformer"]=dict(base=float(base.mean()),gain=float(gain.mean()),ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))],
        p_raw=float(pv), improved=int((sc>0).sum()), median=float(np.median(sc)), worst=float(sc.min()),
        sat_high_gain=float(sc[hi].mean()), sat_low_gain=float(sc[~hi].mean()), sat_scene_frac=float((satsc>0.2).mean()),
        top5_share=float(np.sort(sc)[-5:].sum()/sc.sum()*100))
    r0=OUT["retinexformer"]; print(f"앵커판: {r0['base']:.4f} → {r0['base']+r0['gain']:.4f} ({r0['gain']:+.4f}) 95%[{r0['ci'][0]:+.4f},{r0['ci'][1]:+.4f}] p={pv:.4f} 개선 {r0['improved']}/50 중앙값 {r0['median']:+.3f}")
    print(f"  포화 상위 절반 장면 {r0['sat_high_gain']:+.3f} vs 하위 {r0['sat_low_gain']:+.3f} dB")
    # 버스트 합성
    keys=[k for k in sorted(grp) if len(grp[k])>=2]; sb=[];sa=[];SS=[]
    for k in keys:
        ym=np.mean([load(m,i,rows[i]) for i in grp[k]],0); g=gt_of(k[0]); G=g8(g)
        b=p8(g8(ym),G); p=float(np.clip(K/max(float(np.percentile(ym,99.5)),1e-6),0.5,2.0))
        sb.append(b); sa.append(p8(g8(ym*p),G)); SS.append(k[0])
    sb,sa,SS=np.array(sb),np.array(sa),np.array(SS)
    single=float(np.mean([np.mean([base[i] for i in grp[k]]) for k in keys]))
    d=np.array([ (sa-sb)[SS==s].mean() for s in sorted(set(SS.tolist()))]); bt=np.array([np.mean(rng.choice(d,len(d))) for _ in range(20000)])
    OUT["burst"]=dict(single=single, avg=float(sb.mean()), avg_cal=float(sa.mean()), gain=float((sa-sb).mean()),
                      ci=[float(np.percentile(bt,2.5)),float(np.percentile(bt,97.5))], n_groups=len(keys))
    print(f"버스트: 단일 {single:.4f} → 출력평균 {sb.mean():.4f} → 앵커 {sa.mean():.4f} ({(sa-sb).mean():+.4f})")
json.dump(OUT, open(f"{M}/anchor_extra.json","w"), indent=1)
