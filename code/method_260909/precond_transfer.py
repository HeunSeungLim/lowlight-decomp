"""전제조건 범위 정정: 학습 분할 전체(접두 0 + 2, 181장면)와 테스트(접두 1, 50장면)의 q 포화율."""
import json, os, numpy as np
M = os.path.dirname(os.path.abspath(__file__))
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
Q = 99.9
scenes = sorted(os.listdir(GTD))
split = {"train": [s for s in scenes if s[0] in "02"], "test": [s for s in scenes if s[0] == "1"]}
out = {"q": Q, "split_rule": "SID loader convention: leading digit 0 or 2 = train, 1 = test",
       "gt_per_scene": "one long-exposure reference per scene directory (first .npy)"}
for name, ss in split.items():
    qs = []
    for s in ss:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        g = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
        qs.append(float(np.percentile(g, Q)))
    qs = np.array(qs)
    out[name] = dict(scenes=len(ss), references=len(qs),
                     saturated_pct=float((qs >= 0.999).mean() * 100),
                     min_q=float(qs.min()), n_below=int((qs < 0.999).sum()))
    print(f"{name}: 장면 {len(ss)}, 포화 {out[name]['saturated_pct']:.1f}%, 최소 Q {out[name]['min_q']:.4f}")
# 이전 판본이 쓴 범위(접두 0 만)도 함께 기록해 서술 범위를 추적 가능하게 둔다
p0 = [s for s in scenes if s[0] == "0"]
out["previous_scope_prefix0_only"] = dict(scenes=len(p0),
    note="Earlier measurement used train_feats.npz whose 644 rows are frames from these scenes, not scenes; the field name train_scenes was wrong.")
json.dump(out, open(f"{M}/precond_transfer.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:600])
