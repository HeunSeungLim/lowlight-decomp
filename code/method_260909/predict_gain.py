"""정답 없이 전역 이득 a* 를 맞힐 수 있는가. 입력·출력 통계만 쓰고 장면 5겹 교차검증으로 학습·평가한다."""
import os, json, os, numpy as np
CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"; SHD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/short_sid2"
OUT = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.environ.get("LLROOT", ".") + "/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32)/255.0
    return _c[s]
def stats(x):
    q = np.percentile(x, [1,5,25,50,75,95,99], axis=(0,1))
    return np.concatenate([x.mean((0,1)), x.std((0,1)), q.ravel(), [x.mean(), x.std()]])
p8 = lambda y,g: 10*np.log10(255.0**2/np.mean((np.rint(np.clip(y,0,1)*255).astype(np.float64)-np.rint(g*255).astype(np.float64))**2))
X, A, S, EX, BASE = [], [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    x = np.load(f"{SHD}/{r['scene']}/{r['id']}")[:, :, ::-1].astype(np.float32)/255.0
    g = gt_of(r["scene"])
    X.append(np.concatenate([stats(y), stats(x), [float(r["exp"][:-1])]])); A.append(float((y*g).sum()/(y*y).sum()))
    S.append(r["scene"]); EX.append(r["exp"]); BASE.append(p8(y, g))
    if i % 150 == 0: print(f"  {i}/{len(rows)}", flush=True)
X, A, S, BASE = np.stack(X), np.array(A), np.array(S), np.array(BASE)
print(f"특징 {X.shape[1]}개, 프레임 {len(A)}개, 장면 {len(set(S.tolist()))}개")
us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
pred = np.zeros(len(A))
for te in folds:
    tr = np.array([s not in te for s in S]); Xt, yt = X[tr], np.log(A[tr])
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9; Z = (Xt-mu)/sd
    best, bl = None, None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000):      # 내부 분할로 정규화 세기 선택
        itr = np.array([s not in list(te)[:0] for s in S[tr]])
        w = np.linalg.solve(Z.T@Z + lam*np.eye(Z.shape[1]), Z.T@(yt-yt.mean()))
        r = np.mean((yt - yt.mean() - Z@w)**2)
        if best is None or r < best: best, bl, bw, bm = r, lam, w, yt.mean()
    Zte = (X[~tr]-mu)/sd; pred[~tr] = np.exp(bm + Zte@bw)
ss_res = np.sum((A-pred)**2); ss_tot = np.sum((A-A.mean())**2)
print(f"\n예측 R^2 (장면 홀드아웃) = {1-ss_res/ss_tot:+.4f}   상관 {np.corrcoef(A,pred)[0,1]:+.3f}")
print(f"a* 실제 평균 {A.mean():.4f} 표준편차 {A.std(ddof=1):.4f} / 예측 표준편차 {pred.std(ddof=1):.4f}")
gains = np.zeros(len(A))
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32)
    gains[i] = p8(y*pred[i], gt_of(S[i])) - BASE[i]
sc = np.array([gains[S==s].mean() for s in us]); boot = np.array([np.mean(rng.choice(sc,len(sc))) for _ in range(20000)])
print(f"예측 이득 적용: {gains.mean():+.4f} dB, 장면 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], 개선 장면 {(sc>0).sum()}/{len(us)}")
json.dump(dict(r2=float(1-ss_res/ss_tot), gain_db=float(gains.mean()),
               ci=[float(np.percentile(boot,2.5)), float(np.percentile(boot,97.5))], scenes_improved=int((sc>0).sum())),
          open(f"{OUT}/predict_gain.json","w"), indent=1)
