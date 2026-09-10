import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""학습 장면의 기준 통계에 출력 밝기를 맞추는 보정. 평가 장면의 정답은 쓰지 않는다.
gain = (학습 GT 통계 평균) / (이 출력의 같은 통계). 장면 5겹 교차검증, 통계 후보별 전량 보고."""
import json, os, numpy as np
CACHE = os.environ.get("LLCACHE", "cache")
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"; OUT = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))), "numbers", "compare_methods.json")))["per_frame"]["Sony"]["retinexformer"]
QS = [("mean", None), ("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99)]
def stat(x, q): return float(x.mean()) if q is None else float(np.percentile(x, q))
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
SY, SG, A, S, BASE = [], [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g = gt_of(r["scene"])
    SY.append([stat(y,q) for _,q in QS]); SG.append([stat(g,q) for _,q in QS])
    A.append(float((y*g).sum()/(y*y).sum())); S.append(r["scene"]); BASE.append(p8(g8(y), g8(g)))
    if i % 200 == 0: print(f"  {i}/{len(rows)}", flush=True)
SY, SG, A, S, BASE = np.array(SY), np.array(SG), np.array(A), np.array(S), np.array(BASE)
us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
print(f"\n장면 단위 GT 통계 변동(변동계수): " + ", ".join(f"{n} {SG[:,k].std()/SG[:,k].mean():.3f}" for k,(n,_) in enumerate(QS)))
res = {}
for k,(n,_) in enumerate(QS):
    pred = np.zeros(len(A))
    for te in folds:
        tr = np.array([s not in te for s in S]); T = SG[tr,k].mean()
        pred[~tr] = T / np.maximum(SY[~tr,k], 1e-6)
    pred = np.clip(pred, 0.5, 2.0)
    gains = np.zeros(len(A))
    for i in range(len(A)):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        gains[i] = p8(g8(y*pred[i]), g8(gt_of(S[i]))) - BASE[i]
    sc = np.array([gains[S==s].mean() for s in us]); boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
    r = np.corrcoef(A, pred)[0,1]
    res[n] = dict(gain=float(gains.mean()), ci=[float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))],
                  improved=int((sc>0).sum()), corr_with_astar=float(r))
    print(f"{n:5s}: {gains.mean():+.4f} dB  95% [{res[n]['ci'][0]:+.4f}, {res[n]['ci'][1]:+.4f}]  개선 {res[n]['improved']}/{len(us)}  예측-a* 상관 {r:+.3f}", flush=True)
print(f"\n참고: 오라클 전역이득 상한 = +1.341 dB")
json.dump(res, open(f"{OUT}/stat_match.json","w"), indent=1)
