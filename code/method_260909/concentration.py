import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""집중도 민감도: 장면을 하나씩 빼면 풀드 이득이 어떻게 줄어드는가."""
import json, os, numpy as np
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
gain, S = [], []
for i, r in enumerate(rows):
    y = np.load(f"{CODEX}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float32)
    Gu = g8(gt_of(r["scene"]))
    p = float(np.clip(1.0 / max(float(np.percentile(y, Q)), 1e-6), 0.5, 2.0))
    gain.append(p8(g8(y * p), Gu) - p8(g8(y), Gu)); S.append(r["scene"])
    if i % 150 == 0: print(f"  {i}/{len(rows)}", flush=True)
gain, S = np.array(gain), np.array(S)
us = sorted(set(S.tolist()))
sc = np.array([gain[S == s].mean() for s in us])          # 장면 평균
order = np.argsort(-sc)
frame_pooled = float(gain.mean())
out = dict(frame_pooled=frame_pooled, scene_pooled=float(sc.mean()), scenes=len(sc))
for k in (1, 3, 5):
    keep = np.ones(len(sc), bool); keep[order[:k]] = False
    # 프레임 가중 헤드라인에서 그 장면들의 프레임을 제외
    mask = ~np.isin(S, [us[i] for i in order[:k]])
    out[f"drop_top{k}"] = dict(scene_mean=float(sc[keep].mean()), frame_mean=float(gain[mask].mean()),
                               share_of_scene_pooled_pct=float((1 - sc[keep].mean() / sc.mean()) * 100),
                               share_of_frame_pooled_pct=float((1 - gain[mask].mean() / frame_pooled) * 100))
    print(f"상위 {k} 제외: 장면평균 {out[f'drop_top{k}']['scene_mean']:+.3f} "
          f"({out[f'drop_top{k}']['share_of_scene_pooled_pct']:.0f}% 감소), "
          f"프레임평균 {out[f'drop_top{k}']['frame_mean']:+.3f} "
          f"({out[f'drop_top{k}']['share_of_frame_pooled_pct']:.0f}% 감소)")
out["top5_share_of_scene_pooled_pct"] = float(sc[order[:5]].sum() / sc.sum() * 100)
np.savez(f"{M}/concentration_frames.npz", gain=gain, base=None if False else np.array([]), S=S)
json.dump(out, open(f"{M}/concentration.json", "w"), indent=1)
print("상위5가 장면 풀드에서 차지하는 몫", round(out["top5_share_of_scene_pooled_pct"], 1), "%")
