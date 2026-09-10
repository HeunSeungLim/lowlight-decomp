import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""반증 좌석이 요구한 대조군: 작동집합에서의 오라클 상수배, 배율 순열, 오라클 회수율.

앵커와 같은 캐시·같은 8비트 규칙을 쓰되 통계는 독립 재구현한다.
"""
import json, os, sys, collections, numpy as np
M = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX = os.environ.get("LLCACHE", "cache")
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
Q = 99.9
rows = json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8 = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
g8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)

def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
    return _c[s]

Y, G8, S, base, mult, num, den = [], [], [], [], [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CODEX}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float32)
    g = gt_of(r["scene"]); Gu = g8(g)
    base.append(p8(g8(y), Gu)); S.append(r["scene"])
    mult.append(float(np.clip(1.0 / max(float(np.percentile(y, Q)), 1e-6), 0.5, 2.0)))
    num.append(float(np.sum(y.astype(np.float64) * g.astype(np.float64))))
    den.append(float(np.sum(y.astype(np.float64) ** 2)))
    Y.append(y); G8.append(Gu)
    if i % 100 == 0: print(f"  적재 {i}/{len(rows)}", flush=True)
base = np.array(base); S = np.array(S); mult = np.array(mult)
num = np.array(num); den = np.array(den)
acting = np.abs(mult - 1.0) > 1e-4   # 논문의 작동 판정과 같은 허용오차
print(f"작동 프레임 {int(acting.sum())}/{len(rows)}, 작동 장면 {len(set(S[acting].tolist()))}")

def eval_mult(ms, sel):
    """주어진 배율로 8비트 PSNR 이득(프레임 평균)과 장면 평균."""
    d = np.full(len(rows), np.nan)
    for i in np.where(sel)[0]:
        d[i] = p8(g8(Y[i] * ms[i]), G8[i]) - base[i]
    return d

def scene_mean(d, sel):
    ss = sorted(set(S[sel].tolist()))
    return np.array([np.nanmean(d[(S == s) & sel]) for s in ss]), ss

def boot(v, seed=20260909, B=20000):
    rng = np.random.RandomState(seed)
    bt = np.array([np.mean(rng.choice(v, len(v))) for _ in range(B)])
    return [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]

out = {"n_frames": len(rows), "n_acting_frames": int(acting.sum()),
       "n_acting_scenes": len(set(S[acting].tolist()))}
allsel = np.ones(len(rows), bool)

# 1) 앵커 자신 (기준선 재확인)
for name, sel in (("pooled", allsel), ("acting", acting)):
    d = eval_mult(mult, sel)
    sc, _ = scene_mean(d, sel)
    out[f"anchor_{name}"] = dict(frame_mean=float(np.nanmean(d[sel])), scene_mean=float(sc.mean()),
                                 scene_ci=boot(sc), scenes=len(sc))

# 2) 오라클 상수배: 집합 전체에 하나의 상수. 참조로 맞춘 값이라 상한이다.
for name, sel in (("pooled", allsel), ("acting", acting)):
    c = float(num[sel].sum() / den[sel].sum())
    d = eval_mult(np.full(len(rows), c), sel)
    sc, _ = scene_mean(d, sel)
    out[f"oracle_constant_{name}"] = dict(constant=c, frame_mean=float(np.nanmean(d[sel])),
                                          scene_mean=float(sc.mean()), scene_ci=boot(sc), scenes=len(sc))

# 3) 프레임별 오라클 이득 (전역 이득 오라클, 앵커가 근사하려는 대상)
d = np.full(len(rows), np.nan)
for i in range(len(rows)):
    a = num[i] / den[i]
    d[i] = p8(g8(Y[i] * a), G8[i]) - base[i]
for name, sel in (("pooled", allsel), ("acting", acting)):
    sc, _ = scene_mean(d, sel)
    out[f"oracle_perframe_{name}"] = dict(frame_mean=float(np.nanmean(d[sel])), scene_mean=float(sc.mean()), scenes=len(sc))
    out[f"anchor_recovery_{name}"] = float(out[f"anchor_{name}"]["frame_mean"] / out[f"oracle_perframe_{name}"]["frame_mean"] * 100)

# 3b) 짝지은 비교: 같은 장면에서 앵커 - 오라클상수. 구간이 0을 걸치는지가 아니라 이게 비교다.
c_act = float(num[acting].sum() / den[acting].sum())
d_anchor = eval_mult(mult, acting)
d_const = eval_mult(np.full(len(rows), c_act), acting)
sc_a, ss = scene_mean(d_anchor, acting)
sc_c, _ = scene_mean(d_const, acting)
diff = sc_a - sc_c
out["paired_anchor_minus_constant_acting"] = dict(
    scene_mean=float(diff.mean()), scene_ci=boot(diff), scenes=len(diff),
    positive=int((diff > 0).sum()),
    frame_mean=float(np.nanmean(d_anchor[acting]) - np.nanmean(d_const[acting])))

# 4) 순열 대조: 작동 프레임끼리 배율을 섞는다. 배율이 그 프레임의 것이어야 하는지 검정.
rng = np.random.RandomState(20260909)
perm_means = []
idx = np.where(acting)[0]
for rep in range(20):
    ms = mult.copy(); ms[idx] = mult[idx][rng.permutation(len(idx))]
    dp = eval_mult(ms, acting)
    perm_means.append(float(np.nanmean(dp[acting])))
    if rep == 0: print("  순열 1회 완료", flush=True)
out["permutation_acting"] = dict(mean=float(np.mean(perm_means)), sd=float(np.std(perm_means, ddof=1)),
                                 repeats=len(perm_means), values=[round(v, 4) for v in perm_means])

np.savez(f"{M}/controls_cache.npz", base=base, S=S, mult=mult, num=num, den=den)
json.dump(out, open(f"{M}/controls_v58.json", "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v) if not isinstance(v, (int, float)) else round(v, 4))
