"""보정 후보 여러 개를 같은 장면 5겹 교차검증으로 비교한다. 특징 선택·회귀·수축계수는 전부 학습 장면 안에서만 정한다.
후보: (a) 백분위 기반 전역 이득 예측, (b) 색 비율만 정규화, (c) 둘 다."""
import os, json, os, numpy as np
CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; OUT = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.environ.get("LLROOT", ".") + "/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
QL = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
F = f"{OUT}/calib2_cache.npz"
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
if os.path.exists(F):
    z = np.load(F, allow_pickle=True); QY,QG,QYc,QGc,MY,MG,A,Ac,S,BASE = [z[k] for k in ("QY","QG","QYc","QGc","MY","MG","A","Ac","S","BASE")]
else:
    QY,QG,QYc,QGc,MY,MG,A,Ac,S,BASE = [],[],[],[],[],[],[],[],[],[]
    for i, r in enumerate(rows):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g = gt_of(r["scene"])
        QY.append(np.percentile(y,QL)); QG.append(np.percentile(g,QL))
        QYc.append(np.percentile(y,QL,axis=(0,1))); QGc.append(np.percentile(g,QL,axis=(0,1)))
        MY.append(y.mean((0,1))); MG.append(g.mean((0,1)))
        A.append(float((y*g).sum()/(y*y).sum())); Ac.append([float((y[:,:,c]*g[:,:,c]).sum()/(y[:,:,c]**2).sum()) for c in range(3)])
        S.append(r["scene"]); BASE.append(p8(g8(y), g8(g)))
        if i % 200 == 0: print(f"  적재 {i}/{len(rows)}", flush=True)
    QY,QG,QYc,QGc,MY,MG,A,Ac,S,BASE = map(np.array,(QY,QG,QYc,QGc,MY,MG,A,Ac,S,BASE))
    np.savez(F, QY=QY,QG=QG,QYc=QYc,QGc=QGc,MY=MY,MG=MG,A=A,Ac=Ac,S=S,BASE=BASE)
us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
def eval_pred(P, tag):
    P = np.clip(P, 0.5, 2.0); gains = np.zeros(len(A))
    for i in range(len(A)):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        gains[i] = p8(g8(y*P[i][None,None,:]), g8(gt_of(S[i]))) - BASE[i]
    sc = np.array([gains[S==s].mean() for s in us]); boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    print(f"{tag:26s} {gains.mean():+.4f} dB  장면95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  개선 {(sc>0).sum()}/{len(us)}  최악 {sc.min():+.3f}", flush=True)
    return dict(gain=float(gains.mean()), ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))], improved=int((sc>0).sum()), worst=float(sc.min()))
pg = np.ones(len(A)); pc = np.ones((len(A),3)); picked = []
for te in folds:
    tr = np.array([s not in te for s in S]); trs = np.array([s for s in np.array(us) if s not in te])
    # (a) 전역: 백분위 후보 중 학습 장면 안쪽 교차검증으로 가장 잘 맞히는 것 + 수축계수
    best = None
    for k in range(len(QL)):
        u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-6)); yv = np.log(A)
        inner = np.zeros(tr.sum()); Su = S[tr]
        for s0 in trs:
            m = Su != s0
            b, a0 = np.polyfit(u[tr][m], yv[tr][m], 1); inner[~m] = a0 + b*u[tr][~m]
        r2 = 1 - np.sum((yv[tr]-inner)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
        if best is None or r2 > best[0]: best = (r2, k)
    r2, k = best; picked.append((QL[k], round(r2,3)))
    u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-6))
    b, a0 = np.polyfit(u[tr], np.log(A[tr]), 1); pg[~tr] = np.exp(a0 + b*u[~tr])
    # (b) 색 비율만: 학습 GT의 채널 비율 평균에 출력 채널 비율을 맞춘다 (기하평균 1로 정규화)
    rg = MG[tr]/MG[tr].mean(1, keepdims=True); tgt = rg.mean(0)
    ry = MY[~tr]/MY[~tr].mean(1, keepdims=True); c = tgt[None,:]/np.maximum(ry,1e-6)
    pc[~tr] = c/np.exp(np.log(np.maximum(c,1e-6)).mean(1, keepdims=True))
print(f"\n폴드별 선택 백분위·학습 R^2: {picked}")
print(f"예측 상관: 전역 {np.corrcoef(A, pg)[0,1]:+.3f}")
res = {}
res["global"] = eval_pred(pg[:,None].repeat(3,1), "(a) 전역 이득 예측")
res["color"] = eval_pred(pc, "(b) 색 비율만 정규화")
res["both"] = eval_pred(pg[:,None]*pc, "(c) 전역+색")
json.dump(dict(picked=[list(p) for p in picked], res=res), open(f"{OUT}/calib2.json","w"), indent=1)
