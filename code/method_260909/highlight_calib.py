"""하이라이트 정합 보정(HC): 기준 촬영의 상위 백분위는 장면이 달라도 거의 일정하다는 성질을 이용한다.
학습 장면에서 (1) 어느 백분위가 가장 안정적인지, (2) 그 비율에서 이득으로 가는 회귀계수를 배운다.
평가 장면에서는 출력만 보고 이득을 정한다. 전역판과 채널별판을 함께 낸다. 장면 5겹 교차검증."""
import os, json, os, numpy as np
CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; OUT = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.environ.get("LLROOT", ".") + "/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
QL = [90, 95, 98, 99, 99.5, 99.9]
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
QY, QG, QYc, QGc, A, Ac, S, BASE = [], [], [], [], [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g = gt_of(r["scene"])
    QY.append(np.percentile(y, QL)); QG.append(np.percentile(g, QL))
    QYc.append(np.percentile(y, QL, axis=(0,1))); QGc.append(np.percentile(g, QL, axis=(0,1)))
    A.append(float((y*g).sum()/(y*y).sum()))
    Ac.append([float((y[:,:,c]*g[:,:,c]).sum()/(y[:,:,c]*y[:,:,c]).sum()) for c in range(3)])
    S.append(r["scene"]); BASE.append(p8(g8(y), g8(g)))
    if i % 200 == 0: print(f"  적재 {i}/{len(rows)}", flush=True)
QY, QG, QYc, QGc = np.array(QY), np.array(QG), np.array(QYc), np.array(QGc)
A, Ac, S, BASE = np.array(A), np.array(Ac), np.array(S), np.array(BASE)
us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
pred_g, pred_c, chosen = np.zeros(len(A)), np.zeros((len(A),3)), []
for te in folds:
    tr = np.array([s not in te for s in S])
    cv = QG[tr].std(0)/QG[tr].mean(0); k = int(np.argmin(cv)); chosen.append(QL[k])      # 학습 장면에서만 백분위 선택
    T = QG[tr, k].mean(); u = np.log(T/np.maximum(QY[:, k], 1e-6))
    b, a0 = np.polyfit(u[tr], np.log(A[tr]), 1)                                          # 비율 -> 이득 회귀 (학습 장면)
    pred_g[~tr] = np.exp(a0 + b*u[~tr])
    for c in range(3):
        Tc = QGc[tr, k, c].mean(); uc = np.log(Tc/np.maximum(QYc[:, k, c], 1e-6))
        bc, ac0 = np.polyfit(uc[tr], np.log(Ac[tr, c]), 1); pred_c[~tr, c] = np.exp(ac0 + bc*uc[~tr])
print(f"\n폴드별 선택 백분위: {chosen}  (학습 장면 변동계수 최소 기준)")
print(f"예측-실제 상관: 전역 {np.corrcoef(A, pred_g)[0,1]:+.3f} / 채널 {[round(float(np.corrcoef(Ac[:,c], pred_c[:,c])[0,1]),3) for c in range(3)]}")
res = {}
for name, P in (("전역 하이라이트 정합", pred_g[:,None].repeat(3,1)), ("채널별 하이라이트 정합", pred_c)):
    P = np.clip(P, 0.5, 2.0); gains = np.zeros(len(A))
    for i in range(len(A)):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        gains[i] = p8(g8(y*P[i][None,None,:]), g8(gt_of(S[i]))) - BASE[i]
    sc = np.array([gains[S==s].mean() for s in us]); boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    res[name] = dict(gain=float(gains.mean()), ci=[float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))],
                     improved=int((sc>0).sum()), worst_scene=float(sc.min()), best_scene=float(sc.max()))
    print(f"{name}: {gains.mean():+.4f} dB  장면95% [{res[name]['ci'][0]:+.4f}, {res[name]['ci'][1]:+.4f}]  개선 {res[name]['improved']}/{len(us)}  최악 장면 {sc.min():+.3f}", flush=True)
print(f"\n기준 {BASE.mean():.4f} dB | 오라클 전역 +1.341 | 오라클 채널별 +2.05 (논문 값)")
json.dump(dict(folds_percentile=chosen, res=res, corr_global=float(np.corrcoef(A,pred_g)[0,1])), open(f"{OUT}/highlight_calib.json","w"), indent=1)
