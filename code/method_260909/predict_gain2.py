import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""전역 이득 예측 재시도: 로그 특징(비율 구조)과 장면 단위 leave-one-scene-out 로 정규화 세기를 고른다."""
import json, os, numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)); D = np.load(f"{OUT}/feat_cache.npz", allow_pickle=True) if os.path.exists(f"{OUT}/feat_cache.npz") else None
CACHE = os.environ.get("LLCACHE", "cache")
GTD = (os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; SHD = (os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/short_sid2"
rows = json.load(open(os.path.join(os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))), "numbers", "compare_methods.json")))["per_frame"]["Sony"]["retinexformer"]
if D is None:
    def gt_of(s, _c={}):
        if s not in _c:
            f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
            _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
        return _c[s]
    def st(x):
        q = np.percentile(x, [1,5,25,50,75,95,99], axis=(0,1))
        return np.log(np.concatenate([x.mean((0,1)), x.std((0,1)), q.ravel(), [x.mean(), x.std()]]) + 1e-4)
    X, A, S, BASE = [], [], [], []
    p8 = lambda y,g: 10*np.log10(255.0**2/np.mean((np.rint(np.clip(y,0,1)*255).astype(np.float64)-np.rint(g*255).astype(np.float64))**2))
    for i, r in enumerate(rows):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
        x = np.load(f"{SHD}/{r['scene']}/{r['id']}")[:, :, ::-1].astype(np.float32)/255.0; g = gt_of(r["scene"])
        fy, fx = st(y), st(x)
        X.append(np.concatenate([fy, fx, fy-fx, [np.log(float(r["exp"][:-1]))]]))
        A.append(float((y*g).sum()/(y*y).sum())); S.append(r["scene"]); BASE.append(p8(y,g))
        if i % 200 == 0: print(f"  {i}/{len(rows)}", flush=True)
    X, A, S, BASE = np.stack(X), np.array(A), np.array(S), np.array(BASE)
    np.savez(f"{OUT}/feat_cache.npz", X=X, A=A, S=S, BASE=BASE)
else:
    X, A, S, BASE = D["X"], D["A"], D["S"], D["BASE"]
y = np.log(A); us = sorted(set(S.tolist()))
print(f"특징 {X.shape[1]}개 / 프레임 {len(A)} / 장면 {len(us)}")
sm = np.array([y[S==s].mean() for s in us]); Xs = np.stack([X[S==s].mean(0) for s in us])
c = np.array([np.corrcoef(Xs[:,k], sm)[0,1] for k in range(X.shape[1])])
top = np.argsort(-np.abs(c))[:5]; print("장면 단위 단변량 상관 상위:", [f"f{k}:{c[k]:+.2f}" for k in top])
def ridge_cv(Xa, ya, groups):
    lams = 10.0**np.arange(-2, 6); best = None
    for lam in lams:
        pr = np.zeros(len(ya))
        for g0 in sorted(set(groups.tolist())):
            tr = groups != g0; mu, sd = Xa[tr].mean(0), Xa[tr].std(0)+1e-9; Z = (Xa[tr]-mu)/sd; ym = ya[tr].mean()
            w = np.linalg.solve(Z.T@Z + lam*np.eye(Z.shape[1]), Z.T@(ya[tr]-ym))
            pr[~tr] = ym + ((Xa[~tr]-mu)/sd)@w
        r2 = 1 - np.sum((ya-pr)**2)/np.sum((ya-ya.mean())**2)
        if best is None or r2 > best[0]: best = (r2, lam, pr.copy())
    return best
r2, lam, pr_s = ridge_cv(Xs, sm, np.arange(len(us)))
print(f"장면 단위 예측: R^2 = {r2:+.4f} (lambda {lam:g}), 상관 {np.corrcoef(sm, pr_s)[0,1]:+.3f}")
pred = np.exp(np.array([pr_s[us.index(s)] for s in S]))
p8 = lambda a,b: 10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
g8 = lambda x: np.rint(np.clip(x,0,1)*255).astype(np.uint8)
gains = np.zeros(len(A))
for i in range(len(A)):
    yv = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); gv = gt_of(S[i])
    gains[i] = p8(g8(yv*pred[i]), g8(gv)) - BASE[i]
rng = np.random.RandomState(20260909); sc = np.array([gains[S==s].mean() for s in us])
boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
print(f"예측 이득 적용: {gains.mean():+.4f} dB, 장면 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], 개선 장면 {(sc>0).sum()}/{len(us)}")
json.dump(dict(scene_r2=float(r2), gain_db=float(gains.mean()), ci=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]), open(f"{OUT}/predict_gain2.json","w"), indent=1)
