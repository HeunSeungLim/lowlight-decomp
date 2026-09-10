import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""노출 3구간의 프레임평균 이득. 인쇄값이 헤드라인으로 정확히 재합성되는지 확인한다."""
import json, os, collections, numpy as np
M = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
CODEX = os.environ.get("LLCACHE", "cache")
GTD = (os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"
Q = 99.9
rows = json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8 = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
g8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)

def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
    return _c[s]

gain, S, EX = [], [], []
for i, r in enumerate(rows):
    y = np.load(f"{CODEX}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float32)
    Gu = g8(gt_of(r["scene"]))
    p = float(np.clip(1.0 / max(float(np.percentile(y, Q)), 1e-6), 0.5, 2.0))
    gain.append(p8(g8(y * p), Gu) - p8(g8(y), Gu)); S.append(r["scene"]); EX.append(r["exp"])
    if i % 150 == 0: print(f"  {i}/{len(rows)}", flush=True)
gain, S, EX = np.array(gain), np.array(S), np.array(EX)
out = {"headline_frame_mean": float(gain.mean()), "exposures": {}}
recomb = 0.0
for e in sorted(set(EX.tolist())):
    m = EX == e
    sc = sorted(set(S[m].tolist()))
    out["exposures"][str(e)] = dict(frames=int(m.sum()), scenes=len(sc),
                                    frame_mean=float(gain[m].mean()),
                                    scene_mean=float(np.mean([gain[m & (S == s)].mean() for s in sc])))
    recomb += gain[m].sum()
out["recombination_check"] = dict(
    frame_weighted_from_frame_means=float(sum(v["frames"] * v["frame_mean"] for v in out["exposures"].values()) / len(gain)),
    frame_weighted_from_scene_means=float(sum(v["frames"] * v["scene_mean"] for v in out["exposures"].values()) / len(gain)),
    identity_error=float(abs(recomb / len(gain) - gain.mean())))
json.dump(out, open(f"{M}/exposure_frame_means.json", "w"), indent=1)
print(json.dumps(out, indent=1))
