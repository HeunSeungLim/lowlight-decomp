"""전프레임 아핀 오라클: 3x3 색행렬 + 편향. pal2026 의 광도항과 같은 범위다.

우리 전역 이득·채널 이득이 그 진부분집합이므로, 아핀이 잔차를 얼마나 더 가져가는지 재서
"당신 잔차가 사실 광도항 아니냐"는 질문에 답한다.
"""
import os, json, os, numpy as np
M = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", ".")
CACHE = os.environ.get("LLCACHE", "numbers/cache_retinexformer_sony")
GTD = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID/long_sid2"
rows = json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
p8 = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
g8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)

def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float32) / 255.0
    return _c[s]

base, gl, ch, af, S = [], [], [], [], []
sq_res, sq_gl, sq_ch, sq_af = 0.0, 0.0, 0.0, 0.0
for i, r in enumerate(rows):
    y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float64)
    g = gt_of(r["scene"]).astype(np.float64); Gu = g8(g)
    b0 = p8(g8(y), Gu); base.append(b0); S.append(r["scene"])
    Y = y.reshape(-1, 3); G = g.reshape(-1, 3)
    # 전역 스칼라 이득
    a = float((Y * G).sum() / max((Y * Y).sum(), 1e-12))
    gl.append(p8(g8((y * a)), Gu) - b0)
    # 채널별 이득
    c = (Y * G).sum(0) / np.maximum((Y * Y).sum(0), 1e-12)
    ch.append(p8(g8(y * c), Gu) - b0)
    # 아핀: 3x3 + 편향
    X = np.concatenate([Y, np.ones((Y.shape[0], 1))], 1)
    W = np.linalg.lstsq(X.T @ X, X.T @ G, rcond=None)[0]
    P = (X @ W).reshape(y.shape)
    af.append(p8(g8(P), Gu) - b0)
    # 제곱오차 몫(클리핑 전, 논문 관례와 같게)
    e0 = ((Y - G) ** 2).sum(); sq_res += e0
    sq_gl += e0 - ((Y * a - G) ** 2).sum()
    sq_ch += e0 - ((Y * c - G) ** 2).sum()
    sq_af += e0 - ((X @ W - G) ** 2).sum()
    if i % 100 == 0: print(f"  {i}/{len(rows)}", flush=True)
base, gl, ch, af, S = map(np.array, (base, gl, ch, af, S))
us = sorted(set(S.tolist()))
out = dict(n=len(rows), base=float(base.mean()),
           gain_global=float(gl.mean()), gain_channel=float(ch.mean()), gain_affine=float(af.mean()),
           psnr_global=float((base + gl).mean()), psnr_channel=float((base + ch).mean()),
           psnr_affine=float((base + af).mean()),
           share_global_pct=float(sq_gl / sq_res * 100), share_channel_pct=float(sq_ch / sq_res * 100),
           share_affine_pct=float(sq_af / sq_res * 100),
           affine_minus_channel_db=float((af - ch).mean()),
           scene_mean_affine_minus_channel=float(np.mean([ (af-ch)[S==s].mean() for s in us])))
json.dump(out, open(f"{M}/affine_oracle.json", "w"), indent=1)
print(json.dumps(out, indent=1))
