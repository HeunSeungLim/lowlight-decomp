import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""최종 형태: 보정용 장면 몇 개로 회귀를 맞추고, 평가 장면에서는 장면 단위로 예측 이득을 평균해 적용한다.
보정 장면 수를 바꿔가며 성능을 재고, 프레임 단위 예측과 비교한다. 정답은 보정 장면에서만 쓴다."""
import json, os, numpy as np
def linfit(x, y):                      # 닫힌형 최소자승 (LAPACK 회피)
    x = np.asarray(x, float); y = np.asarray(y, float); vx = x.var()
    b = 0.0 if vx < 1e-18 else float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    return b, float(y.mean() - b * x.mean())
OUT = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CACHE = os.environ.get("LLCACHE", "cache")
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
z = np.load(f"{OUT}/calib2_cache.npz", allow_pickle=True); QY, QG, A, S, BASE = z["QY"], z["QG"], z["A"], z["S"], z["BASE"]
tz = np.load(f"{OUT}/train_feats.npz", allow_pickle=True)
QL = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
print(f"학습 장면 a*: 평균 {tz['A'].mean():.4f} 표준편차 {tz['A'].std(ddof=1):.4f} (프레임 {len(tz['A'])})")
print(f"테스트 장면 a*: 평균 {A.mean():.4f} 표준편차 {A.std(ddof=1):.4f} (프레임 {len(A)})  → 밝기 보정 실패는 일반화 문제")
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
us = sorted(set(S.tolist())); K = 6                                   # 백분위 99 (인덱스 5) 고정 대신 보정셋에서 선택
rng = np.random.RandomState(20260909)
def fit_predict(cal_scenes, ev_idx, per_scene):
    tr = np.array([s in cal_scenes for s in S]) & np.isfinite(np.log(np.maximum(A, 1e-9)))
    best = None
    for k in range(len(QL)):
        u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4)); yv = np.log(np.maximum(A,1e-6))
        if not np.isfinite(u[tr]).all(): continue
        pr = np.zeros(tr.sum()); Su = S[tr]
        for s0 in cal_scenes:
            m = Su != s0
            if m.sum() < 5: continue
            b,a0 = linfit(u[tr][m], yv[tr][m]); pr[~m] = a0 + b*u[tr][~m]
        r2 = 1 - np.sum((yv[tr]-pr)**2)/np.sum((yv[tr]-yv[tr].mean())**2)
        if best is None or r2 > best[0]: best = (r2, k)
    k = best[1]; u = np.log(np.maximum(QG[tr,k].mean(),1e-6)/np.maximum(QY[:,k],1e-4))
    b,a0 = linfit(u[tr], np.log(np.maximum(A[tr],1e-6))); p = np.exp(a0 + b*u)
    if per_scene:
        for s in set(S[ev_idx].tolist()):
            m = S == s; p[m] = np.exp(np.log(p[m]).mean())
    return np.clip(p, 0.5, 2.0), QL[k]
def score(pred, idx):
    g = np.zeros(len(idx))
    for j, i in enumerate(idx):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        g[j] = p8(g8(y*pred[i]), g8(gt_of(S[i]))) - BASE[i]
    return g
print("\n=== 보정 장면 수와 적용 단위 ===")
res = {}
for ncal in (10, 20, 25, 40):
    for per_scene in (False, True):
        gains, folds_done = [], 0
        for rep in range(4):
            perm = np.random.RandomState(1000+rep).permutation(len(us))
            cal = set(np.array(us)[perm[:ncal]].tolist()); ev = [i for i in range(len(S)) if S[i] not in cal]
            pred, kq = fit_predict(cal, np.array(ev), per_scene)
            gains.append(score(pred, ev)); folds_done += 1
        gm = np.concatenate(gains)
        sc = np.array([gm[i] for i in range(len(gm))])
        boot = np.array([np.mean(rng.choice(sc, len(sc))) for _ in range(5000)])
        tag = "장면단위" if per_scene else "프레임단위"
        res[f"{ncal}_{tag}"] = dict(gain=float(gm.mean()), ci=[float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))])
        print(f"보정장면 {ncal:2d}개 · {tag}: {gm.mean():+.4f} dB  95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  (반복 {folds_done}회)", flush=True)
json.dump(res, open(f"{OUT}/final_calib.json","w"), indent=1)
