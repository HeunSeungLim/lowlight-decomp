import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""정성 증거: 보정이 크게 도운 장면과 손해 본 장면을 실제 입력·출력·보정본·기준으로 저장."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__)); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX=os.environ.get("LLCACHE", "cache")
GTD="/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"; SHD="/data/HSL/lowlight_model/data/SID_raw/SID/short_sid2"
QL=[50,75,90,95,98,99,99.5,99.9]
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
z=np.load(f"{OUT}/calib2_cache.npz",allow_pickle=True); QY,QG,A,S,BASE=z["QY"],z["QG"],z["A"],z["S"],z["BASE"]
def linfit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);v=x.var(); b=0.0 if v<1e-18 else float(((x-x.mean())*(y-y.mean())).mean()/v); return b,float(y.mean()-b*x.mean())
def gt_of(s,_c={}):
    if s not in _c:
        f=[x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s]=np.load(f"{GTD}/{s}/{f}")[:,:,::-1].astype(np.float32)/255.0
    return _c[s]
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
us=sorted(set(S.tolist())); perm=np.random.RandomState(20260909).permutation(len(us)); folds=[set(np.array(us)[perm[j::5]].tolist()) for j in range(5)]
pred=np.ones(len(A)); K=[]
for te in folds:
    tr=np.array([s not in te for s in S]); trs=[s for s in us if s not in te]; Su=S[tr]; yv=np.log(np.maximum(A,1e-6))
    best=None
    for k in range(len(QL)):
        uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)); pr=np.zeros(tr.sum())
        for s0 in trs:
            m=Su!=s0; b,a0=linfit(uu[tr][m],yv[tr][m]); pr[~m]=a0+b*uu[tr][~m]
        r2=1-np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
        if best is None or r2>best[0]: best=(r2,k)
    k=best[1]; K.append(QL[k]); uu=np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))
    b,a0=linfit(uu[tr],np.log(np.maximum(A[tr],1e-6))); p=np.clip(np.exp(a0+b*uu),0.5,2.0)
    for s in te:
        m=S==s; pred[m]=np.exp(np.log(p[m]).mean())
gains=np.array([p8(g8(np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0)*pred[i]),g8(gt_of(S[i])))-BASE[i] for i in range(len(A))])
sc=np.array([gains[S==s].mean() for s in us])
pick=[us[int(np.argmax(sc))], us[int(np.argsort(sc)[len(sc)//2])], us[int(np.argmin(sc))]]
print("선택 장면(최고/중앙/최악):", pick, [round(float(sc[us.index(s)]),3) for s in pick])
fig,axes=plt.subplots(len(pick),4,figsize=(11,2.4*len(pick)))
for r_,s in enumerate(pick):
    i=int(np.where(S==s)[0][0]); rr=rows[i]
    x=np.load(f"{SHD}/{s}/{rr['id']}")[:,:,::-1].astype(np.float32)/255.0
    y=np.load(f"{CODEX}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g=gt_of(s); yc=np.clip(y*pred[i],0,1)
    for c_,(im,t) in enumerate(((np.clip(x*4,0,1),f"input $\\times$4"),(y,f"model {BASE[i]:.2f} dB"),(yc,f"calibrated {p8(g8(yc),g8(g)):.2f} dB ($\\times${pred[i]:.3f})"),(g,"reference"))):
        ax=axes[r_,c_]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=9)
    axes[r_,0].set_ylabel(f"scene {s}\n{sc[us.index(s)]:+.2f} dB",fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/qualitative.png",dpi=140); print("저장 qualitative.png")
json.dump(dict(scene_gain={s: float(sc[us.index(s)]) for s in us}, picked=pick, percentiles=K,
               overall=float(gains.mean())), open(f"{OUT}/qualitative.json","w"), indent=1)
