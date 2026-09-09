"""Second verified baseline on LOLv1: Retinexformer with the authors' released LOL_v1.pth.

Having two independently reproduced baselines on the same benchmark is what makes any
later comparison trustworthy. Measurement is ours (repro_measure), not the authors'.
Both test settings are reported, because Retinexformer's own README says it does NOT
recommend the GT-mean setting (it uses the ground truth to improve the output) while
CIDNet's headline table uses it -- so the two papers' LOLv1 numbers are not directly
comparable either.
"""
import os, sys, json
sys.path.insert(0, "code")
REPO = "third_party/Retinexformer"
sys.path.insert(0, REPO)
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

LOW = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/low")
HIGH = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/high")
W = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/LOL_v1.pth")

net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
net.load_state_dict(torch.load(W, map_location="cpu")["params"])
net = net.cuda().eval()

rows = []
with torch.inference_mode():
    for f in sorted(os.listdir(HIGH)):
        if not f.lower().endswith(".png"):
            continue
        lq = np.array(Image.open(os.path.join(LOW, f)).convert("RGB")).astype(np.float32) / 255.
        gt = np.array(Image.open(os.path.join(HIGH, f)).convert("RGB"))
        x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
        h, w = x.shape[2], x.shape[3]
        m = 4
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), "reflect")
        y = torch.clamp(net(x)[:, :, :h, :w], 0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255.
        r = dict(name=f)
        for k, pred in [("round", np.clip(np.rint(y), 0, 255).astype(np.uint8)),
                        ("trunc", np.clip(np.floor(y), 0, 255).astype(np.uint8))]:
            r[f"psnr_{k}"] = psnr_indep(pred, gt)
            r[f"ssim_{k}"] = ssim_indep(pred, gt)
            rec = gt_mean_rectify(pred, gt)
            r[f"psnr_{k}_gtmean"] = psnr_indep(rec, gt)
            r[f"ssim_{k}_gtmean"] = ssim_indep(rec, gt)
        rows.append(r)

keys = [k for k in rows[0] if k != "name"]
avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
print(f"\n=== Retinexformer / LOLv1 eval15  n={len(rows)}")
print(f"{'8bit rule':10s} {'PSNR':>8s} {'SSIM':>8s} | {'PSNR gtmean':>12s} {'SSIM gtmean':>12s}")
for k in ["round", "trunc"]:
    print(f"{k:10s} {avg['psnr_'+k]:8.4f} {avg['ssim_'+k]:8.4f} | "
          f"{avg['psnr_'+k+'_gtmean']:12.4f} {avg['ssim_'+k+'_gtmean']:12.4f}")
print("\n보고값: Retinexformer README, GT평균 적용 설정에서 LOL-v1 PSNR 27.18 / SSIM 0.850")
print("        (저자는 이 설정을 권하지 않는다고 명시. GT평균 미적용 수치는 README 이미지 안에 있어 텍스트로 확인 못 함)")
with open("numbers/measure_LOLv1_retinexformer.json", "w") as f:
    json.dump({"avg": avg, "per_image": rows}, f, indent=2)
