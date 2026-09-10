"""학습 분할 전체(접두 0+2, 181장면)에서 q별 기준영상 포화율. 같은 문단의 100퍼센트와 분모를 맞춘다."""
import json, os, numpy as np
M = os.path.dirname(os.path.abspath(__file__))
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
QS = [95, 98, 99, 99.5, 99.9, 99.95, 99.99]
scenes = sorted(s for s in os.listdir(GTD) if s[0] in "02")
q = np.zeros((len(scenes), len(QS)))
for i, s in enumerate(scenes):
    f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
    g = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
    q[i] = np.percentile(g, QS)
    if i % 40 == 0: print(f"  {i}/{len(scenes)}", flush=True)
out = dict(scenes=len(scenes), split="training split, leading digit 0 or 2",
           q_grid=QS,
           saturated_pct={str(k): float((q[:, j] >= 0.999).mean() * 100) for j, k in enumerate(QS)},
           min_q={str(k): float(q[:, j].min()) for j, k in enumerate(QS)},
           note="Denominator is 181 scenes, the same as the 100 percent figure printed for q=99.9.")
json.dump(out, open(f"{M}/qsat_train181.json", "w"), indent=1)
print(json.dumps(out["saturated_pct"], indent=1))
