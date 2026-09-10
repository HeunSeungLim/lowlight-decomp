import os as _os
_M = _os.environ.get("LLMETHOD", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     "..", "..", "numbers", "method_260909")))
"""보정된 출력의 잔차 분해: 표 1 의 gl./ch./res./Delta16 을 우리 행에도 채운다.

집계는 표 1 과 같다 — 프레임별 dB 를 구한 뒤 프레임 평균. Delta16 은
(16픽셀 블록 이득 - 전역 1x1 블록 이득) 에서 구조없는표적 대조군의 같은 차를 뺀 값이다.
검증: 같은 절차를 보정하지 않은 출력에 돌려 표 1 의 1.19 를 재현하는지 먼저 확인한다.
"""
import json, os, sys, numpy as np, torch
M = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", os.path.abspath(os.path.join(_M, "..", "..")))
for _c in (f"{R}/code", f"{R}/release_v2/code", os.path.join(os.path.dirname(M), "code")):
    if os.path.isdir(_c) and _c not in sys.path: sys.path.insert(0, _c)
from diag_sid_lowfreq import block_index, block_fit
CACHE = os.environ.get("LLCACHE", "cache")
GTD = "/data/HSL/lowlight_model/data/SID_raw/SID/long_sid2"
Q = 99.9
rows = json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]

def gt_of(s, _c={}):
    if s not in _c:
        f = [x for x in sorted(os.listdir(f"{GTD}/{s}")) if x.endswith(".npy")][0]
        _c.clear(); _c[s] = np.load(f"{GTD}/{s}/{f}")[:, :, ::-1].astype(np.float64) / 255.0
    return _c[s]

def run(anchored):
    gen = torch.Generator(device="cpu").manual_seed(20260905)
    per = {k: [] for k in ("g1", "b16", "wg1", "wb16")}
    sq = dict(e0=0.0, gl=0.0, ch=0.0)
    idxc = {}
    for i, r in enumerate(rows):
        y = np.load(f"{CACHE}/{i:04d}.npy").transpose(1, 2, 0).astype(np.float64)
        g = gt_of(r["scene"])
        if anchored:
            p = float(np.clip(1.0 / max(float(np.percentile(y, Q)), 1e-6), 0.5, 2.0))
            y = np.clip(y * p, 0, 1)
        Y, G = y.reshape(-1, 3), g.reshape(-1, 3)
        e0 = ((Y - G) ** 2).sum()
        a = float((Y * G).sum() / max((Y * Y).sum(), 1e-12))
        c = (Y * G).sum(0) / np.maximum((Y * Y).sum(0), 1e-12)
        sq["e0"] += e0
        sq["gl"] += e0 - ((Y * a - G) ** 2).sum()
        sq["ch"] += ((Y * a - G) ** 2).sum() - ((Y * c - G) ** 2).sum()
        t_y = torch.from_numpy(y.transpose(2, 0, 1)); t_g = torch.from_numpy(g.transpose(2, 0, 1))
        H, W = y.shape[:2]
        sd = (t_y - t_g).std(dim=(1, 2), unbiased=False, keepdim=True)
        wn = torch.randn(3, H, W, dtype=torch.float64, generator=gen)
        wn = wn / torch.clamp(wn.std(dim=(1, 2), unbiased=False, keepdim=True), min=1e-30) * sd
        t_w = t_y - wn
        if (H, W) not in idxc:
            idxc[(H, W)] = {B: block_index(H, W, B, torch, "cpu") for B in (0, 16)}
        for B, ka, kb in ((0, "g1", "wg1"), (16, "b16", "wb16")):
            idx, nblk, _, _ = idxc[(H, W)][B]
            out, _, _ = block_fit(t_y, t_g, idx, nblk, "gain", torch)
            per[ka].append(float(((out - t_g) ** 2).mean()) * 255.0 ** 2)
            out, _, _ = block_fit(t_y, t_w, idx, nblk, "gain", torch)
            per[kb].append(float(((out - t_w) ** 2).mean()) * 255.0 ** 2)
        if i % 150 == 0: print(f"  {'anc' if anchored else 'raw'} {i}/{len(rows)}", flush=True)
    pf = lambda v: float(np.mean([10 * np.log10(255.0 ** 2 / x) for x in v]))
    d16_raw = pf(per["b16"]) - pf(per["g1"])
    white16 = pf(per["wb16"]) - pf(per["wg1"])
    return dict(share_global_pct=sq["gl"] / sq["e0"] * 100,
                share_channel_pct=sq["ch"] / sq["e0"] * 100,
                share_residual_pct=(sq["e0"] - sq["gl"] - sq["ch"]) / sq["e0"] * 100,
                d16_raw=d16_raw, white16=white16, d16=d16_raw - white16)

out = dict(n=len(rows), raw=run(False), anchored=run(True),
           table_reference=dict(gl=27.0, ch=8.8, res=64.2, d16=1.19,
                                note="Table 1 Sony Retinexformer row; the raw block must reproduce it"))
json.dump(out, open(f"{M}/decomp_anchored.json", "w"), indent=1)
print(json.dumps(out, indent=1))
