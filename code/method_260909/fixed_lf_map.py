"""고정 저주파 보정맵: 학습 장면에서 블록 이득장의 평균 모양을 배우고, 처음 보는 장면에 그대로 곱한다.
GT는 학습 장면에서만 쓰고 평가 장면에서는 쓰지 않는다. 장면 단위 5겹 교차검증."""
import os, json, os, sys, numpy as np

CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GT = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"
OUT = os.path.dirname(os.path.abspath(__file__))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 16

rows = json.load(open(os.environ.get("LLROOT", ".") + "/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
def gt_of(scene, _c={}):
    if scene not in _c:
        f = [x for x in sorted(os.listdir(f"{GT}/{scene}")) if x.endswith(".npy")][0]
        _c.clear(); _c[scene] = np.load(f"{GT}/{scene}/{f}")[:, :, ::-1].astype(np.float32) / 255.0   # BGR -> RGB
    return _c[scene]
psnr = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
u8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)

H, W = 512, 960; hb, wb = H // B, W // B
fields, base, scenes, outs = [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float32)
    g = gt_of(r["scene"])
    if i < 3: print(f"  대조 프레임 {i}: 기록 PSNR {r['psnr']:.4f} / 재계산 {psnr(u8(y), u8(g)):.4f}", flush=True)
    yb = y.reshape(hb, B, wb, B, 3); gb = g.reshape(hb, B, wb, B, 3)
    num = (yb * gb).sum((1, 3, 4)); den = (yb * yb).sum((1, 3, 4)) + 1e-12
    fields.append((num / den).astype(np.float32)); base.append(psnr(u8(y), u8(g))); scenes.append(r["scene"]); outs.append(i)
    if i % 100 == 0: print(f"  {i}/{len(rows)}", flush=True)
F = np.stack(fields); base = np.array(base); scenes = np.array(scenes)
Fn = F / F.mean((1, 2), keepdims=True)                      # 전역 수준 제거: 알 수 없는 성분
print(f"\n블록 {B}px, 정규화 이득장: 평균 {Fn.mean():.4f}, 프레임내 표준편차 중앙값 {np.median(Fn.std((1,2))):.4f}")
M_all = Fn.mean(0); print(f"전체 평균맵 범위 [{M_all.min():.4f}, {M_all.max():.4f}], 표준편차 {M_all.std():.4f}")

us = sorted(set(scenes.tolist())); rng = np.random.RandomState(20260909); perm = rng.permutation(len(us))
folds = [set(np.array(us)[perm[k::5]].tolist()) for k in range(5)]
gains, gains_o = np.zeros(len(rows)), np.zeros(len(rows))
from scipy.ndimage import zoom
for k, te in enumerate(folds):
    trm = np.array([s not in te for s in scenes]); M = Fn[trm].mean(0)
    Mf = zoom(M, (B, B), order=1)                            # 블록 격자를 부드럽게 확대
    Mf = np.clip(Mf, 0.5, 2.0)[:, :, None]
    for j in np.where(~trm)[0]:
        y = np.load(f"{CACHE}/{outs[j]:04d}.npy").transpose(1, 2, 0).astype(np.float32); g = gt_of(scenes[j])
        gains[j] = psnr(u8(y * Mf), u8(g)) - base[j]
        Fo = zoom(F[j] / F[j].mean(), (B, B), order=1)[:, :, None]      # 같은 형태의 오라클 상한(그 프레임 자신의 장)
        gains_o[j] = psnr(u8(y * Fo), u8(g)) - base[j]
    print(f"  fold {k+1}: 평가 {(~trm).sum()}프레임  고정맵 {gains[~trm].mean():+.4f} dB  (오라클 {gains_o[~trm].mean():+.4f})", flush=True)
sc_mean = np.array([gains[scenes == s].mean() for s in us])
boot = np.array([np.mean(rng.choice(sc_mean, len(sc_mean))) for _ in range(20000)])
res = dict(block=B, n=len(rows), base_psnr=float(base.mean()), fixed_map_gain=float(gains.mean()),
           oracle_field_gain=float(gains_o.mean()), scene_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           scenes_improved=int((sc_mean > 0).sum()), n_scenes=len(us), map_std=float(M_all.std()))
print(f"\n=== 결과 (블록 {B}px) ===")
print(f"기준 PSNR {base.mean():.4f} dB")
print(f"고정맵 적용 {gains.mean():+.4f} dB   장면 부트스트랩 95% [{res['scene_ci95'][0]:+.4f}, {res['scene_ci95'][1]:+.4f}]   개선 장면 {res['scenes_improved']}/{len(us)}")
print(f"같은 형태의 오라클 상한 {gains_o.mean():+.4f} dB")
json.dump(res, open(f"{OUT}/fixed_lf_map_b{B}.json", "w"), indent=1)
np.save(f"{OUT}/mean_field_b{B}.npy", M_all)
