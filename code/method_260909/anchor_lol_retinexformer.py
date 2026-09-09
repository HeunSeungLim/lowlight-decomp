"""LOL 최고 모델(Retinexformer)에 같은 앵커 규칙을 적용해 표 1 LOL 블록의 우리 행을 만든다.

검증: 보정 전 값이 표 1 의 LOL Retinexformer 행(25.15 / 0.845 / 41.4 / 4.1 / 54.5 / 1.17)을
재현하는지 먼저 확인한 뒤 보정본을 보고한다.
"""
import os, json, os, sys, numpy as np, torch
M = os.path.dirname(os.path.abspath(__file__)); R = os.environ.get("LLROOT", ".")
sys.path.insert(0, f"{R}/release_v2/code"); sys.path.insert(0, f"{R}/code")
from diag_sid_lowfreq import block_index, block_fit
from repro_measure import ssim_indep
RF_REPO = os.environ.get("LLDATA", "data") + "/hsl/hsl/Retinexformer"
RF_W = os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/LOL_v1.pth"
LOW = os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/low"
HIGH = os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/high"
Q = 99.9
from PIL import Image
sys.path.insert(0, RF_REPO)
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
import torch.nn.functional as F
net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
net.load_state_dict(torch.load(RF_W, map_location="cpu")["params"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(dev).eval()
p8 = lambda a, b: 10 * np.log10(255.0 ** 2 / np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
g8 = lambda x: np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)
names = sorted(f for f in os.listdir(HIGH) if f.lower().endswith(".png"))
gen = torch.Generator(device="cpu").manual_seed(20260905)
res = {}
for anchored in (False, True):
    per = {k: [] for k in ("g1", "b16", "wg1", "wb16")}
    sq = dict(e0=0.0, gl=0.0, ch=0.0); ps, ss = [], []
    for nm in names:
        lo = np.asarray(Image.open(f"{LOW}/{nm}").convert("RGB")).astype(np.float64) / 255.0
        hi = np.asarray(Image.open(f"{HIGH}/{nm}").convert("RGB")).astype(np.float64) / 255.0
        with torch.no_grad():
            x = torch.from_numpy(lo.transpose(2, 0, 1))[None].float().to(dev)
            h, w = x.shape[2], x.shape[3]; ph, pw = (-h) % 4, (-w) % 4
            if ph or pw: x = F.pad(x, (0, pw, 0, ph), "reflect")
            y = torch.clamp(net(x)[:, :, :h, :w], 0, 1)[0].double().cpu().numpy().transpose(1, 2, 0)
        if anchored:
            m = float(np.clip(1.0 / max(float(np.percentile(y, Q)), 1e-6), 0.5, 2.0))
            y = np.clip(y * m, 0, 1)
        Gu = g8(hi); Yu = g8(y)
        ps.append(p8(Yu, Gu)); ss.append(ssim_indep(Yu, Gu))
        Y, G = y.reshape(-1, 3), hi.reshape(-1, 3)
        e0 = ((Y - G) ** 2).sum()
        a = float((Y * G).sum() / max((Y * Y).sum(), 1e-12))
        c = (Y * G).sum(0) / np.maximum((Y * Y).sum(0), 1e-12)
        sq["e0"] += e0; sq["gl"] += e0 - ((Y * a - G) ** 2).sum()
        sq["ch"] += ((Y * a - G) ** 2).sum() - ((Y * c - G) ** 2).sum()
        t_y = torch.from_numpy(y.transpose(2, 0, 1)); t_g = torch.from_numpy(hi.transpose(2, 0, 1))
        H, W = y.shape[:2]
        sd = (t_y - t_g).std(dim=(1, 2), unbiased=False, keepdim=True)
        wn = torch.randn(3, H, W, dtype=torch.float64, generator=gen)
        wn = wn / torch.clamp(wn.std(dim=(1, 2), unbiased=False, keepdim=True), min=1e-30) * sd
        t_w = t_y - wn
        for B, ka, kb in ((0, "g1", "wg1"), (16, "b16", "wb16")):
            idx, nblk, _, _ = block_index(H, W, B, torch, "cpu")
            out, _, _ = block_fit(t_y, t_g, idx, nblk, "gain", torch)
            per[ka].append(float(((out - t_g) ** 2).mean()) * 255.0 ** 2)
            out, _, _ = block_fit(t_y, t_w, idx, nblk, "gain", torch)
            per[kb].append(float(((out - t_w) ** 2).mean()) * 255.0 ** 2)
    pf = lambda v: float(np.mean([10 * np.log10(255.0 ** 2 / x) for x in v]))
    d16 = (pf(per["b16"]) - pf(per["g1"])) - (pf(per["wb16"]) - pf(per["wg1"]))
    res["anchored" if anchored else "raw"] = dict(
        n=len(names), psnr=float(np.mean(ps)), psnr_sd=float(np.std(ps)), ssim=float(np.mean(ss)),
        share_global_pct=sq["gl"] / sq["e0"] * 100, share_channel_pct=sq["ch"] / sq["e0"] * 100,
        share_residual_pct=(sq["e0"] - sq["gl"] - sq["ch"]) / sq["e0"] * 100, d16=d16)
    r = res["anchored" if anchored else "raw"]
    print(f"{'보정' if anchored else '원본'}: PSNR {r['psnr']:.3f}+-{r['psnr_sd']:.2f} SSIM {r['ssim']:.3f} "
          f"gl {r['share_global_pct']:.1f} ch {r['share_channel_pct']:.1f} res {r['share_residual_pct']:.1f} D16 {r['d16']:.2f}", flush=True)
res["gain_db"] = res["anchored"]["psnr"] - res["raw"]["psnr"]
res["table_reference"] = dict(psnr=25.15, sd=2.87, ssim=0.845, gl=41.4, ch=4.1, res=54.5, d16=1.17)
json.dump(res, open(f"{M}/anchor_lol_retinexformer.json", "w"), indent=1)
print("보정 이득", round(res["gain_db"], 3), "dB")
