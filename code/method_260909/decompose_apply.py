import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""적용 규약을 갈라서 잰다: 전역 오라클 / 고정 공간맵 / 둘 다 / 프레임 오라클 장.
클리핑·8bit 채점과 클리핑 없는 float 채점을 함께 낸다. 공간맵은 장면 5겹 교차검증으로 학습 장면에서만 만든다."""
import json, os, numpy as np
from scipy.ndimage import zoom
CACHE = os.environ.get("LLCACHE", "cache")
GTD = (os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; OUT = os.path.dirname(os.path.abspath(__file__)); B = 16
rows = json.load(open(os.path.join(os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", ".."))), "numbers", "compare_methods.json")))["per_frame"]["Sony"]["retinexformer"]
def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
    return _c[s]
p8 = lambda y, g: 10*np.log10(255.0**2/np.mean((np.rint(np.clip(y,0,1)*255).astype(np.float64)-np.rint(g*255).astype(np.float64))**2))
pf = lambda y, g: 10*np.log10(1.0/np.mean((y.astype(np.float64)-g.astype(np.float64))**2))
H, W = 512, 960; hb, wb = H//B, W//B
Y, G, F, A, S = [], [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1,2,0).astype(np.float32); g = gt_of(r["scene"])
    yb, gb = y.reshape(hb,B,wb,B,3), g.reshape(hb,B,wb,B,3)
    F.append(((yb*gb).sum((1,3,4))/((yb*yb).sum((1,3,4))+1e-12)).astype(np.float32))
    A.append(float((y*g).sum()/(y*y).sum())); S.append(r["scene"])
    if i % 150 == 0: print(f"  적재 {i}/{len(rows)}", flush=True)
F, A, S = np.stack(F), np.array(A), np.array(S)
Fn = F / F.mean((1,2), keepdims=True)
r_common = np.array([np.corrcoef((Fn[i]-1).ravel(), (Fn.mean(0)-1).ravel())[0,1] for i in range(len(Fn))])
print(f"\n프레임 이득장과 전체 평균맵의 상관: 중앙값 {np.median(r_common):.3f}, 하위25% {np.percentile(r_common,25):.3f}")
us = sorted(set(S.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
names = ["기준", "전역 오라클", "고정맵만", "전역 오라클+고정맵", "프레임 오라클장"]
R8 = {n: np.zeros(len(rows)) for n in names}; RF = {n: np.zeros(len(rows)) for n in names}
for te in folds:
    trm = np.array([s not in te for s in S]); M = np.clip(zoom(Fn[trm].mean(0), (B,B), order=1), 0.5, 2.0)[:,:,None]
    for j in np.where(~trm)[0]:
        y = np.load(f"{CACHE}/{j:04d}.npy").transpose(1,2,0).astype(np.float32); g = gt_of(S[j])
        Fo = zoom(F[j], (B,B), order=1)[:,:,None]
        for n, z in zip(names, [y, y*A[j], y*M, y*A[j]*M, y*Fo]):
            R8[n][j] = p8(z, g); RF[n][j] = pf(np.clip(z,0,1), g)
print("\n=== 적용 규약별 PSNR (598프레임, 장면 5겹 교차검증) ===")
print(f"{'조건':22s} {'8bit dB':>9s} {'Δ':>8s}   {'float dB':>9s} {'Δ':>8s}")
for n in names:
    print(f"{n:22s} {R8[n].mean():9.4f} {R8[n].mean()-R8['기준'].mean():+8.4f}   {RF[n].mean():9.4f} {RF[n].mean()-RF['기준'].mean():+8.4f}")
sc = lambda v: np.array([v[S==s].mean() for s in us])
d = sc(R8["전역 오라클+고정맵"]) - sc(R8["전역 오라클"])
boot = np.array([np.mean(rng.choice(d, len(d))) for _ in range(20000)])
print(f"\n전역 오라클 위에 고정맵을 더한 효과: {d.mean():+.4f} dB, 장면 95% [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], 개선 장면 {(d>0).sum()}/{len(us)}")
json.dump({n: float(R8[n].mean()) for n in names} | {"float_" + n: float(RF[n].mean()) for n in names}, open(f"{OUT}/apply_conventions.json","w"), indent=1)
