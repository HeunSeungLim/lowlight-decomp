"""Reproduction audit on Sony-Total-Dark (SID) with the authors' released SID.pth.

Protocol taken from the authors' eval_SID.py / measure_SID.py:
  - inputs  : Sony_total_dark/test/short/<scene>/*.png   (scene ids start with 1)
  - GT      : the single long-exposure image in Sony_total_dark/test/long/<scene>/
  - forward : plain model(x); unlike LOL, eval_SID.py sets no gated flag
  - average : flat mean over all short images (not per-scene first)
Measurement is our own numpy implementation from repro_measure.py, not their cv2 code.
"""
import os, sys, json, argparse
sys.path.insert(0, (os.environ.get("LLDATA", "data") + "/lowlight_model/HVI-CIDNet"))
sys.path.insert(0, "code")
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from net.CIDNet import CIDNet
from repro_measure import psnr_indep, ssim_indep, ssim_authors, gt_mean_rectify

p = argparse.ArgumentParser()
p.add_argument("--root", default=(os.environ.get("LLDATA", "data") + "/lowlight_model/data/Sony_total_dark"))
p.add_argument("--weights", default=(os.environ.get("LLDATA", "data") + "/lowlight_model/weights/author_all/SID.pth"))
p.add_argument("--tag", default="SID")
p.add_argument("--limit_scenes", type=int, default=0)
a = p.parse_args()

net = CIDNet().cuda()
net.load_state_dict(torch.load(a.weights, map_location="cpu"))
net.eval()
tt = transforms.ToTensor()

short_root = os.path.join(a.root, "test", "short")
long_root = os.path.join(a.root, "test", "long")
scenes = sorted(os.listdir(short_root))
if a.limit_scenes:
    scenes = scenes[:a.limit_scenes]

rows = []
with torch.no_grad():
    for si, sc in enumerate(scenes):
        gtf = sorted(f for f in os.listdir(os.path.join(long_root, sc))
                     if f.lower().endswith((".png", ".jpg")))
        gt = np.array(Image.open(os.path.join(long_root, sc, gtf[0])).convert("RGB"))
        for f in sorted(os.listdir(os.path.join(short_root, sc))):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            x = tt(Image.open(os.path.join(short_root, sc, f)).convert("RGB"))[None].cuda()
            y = torch.clamp(net(x), 0, 1)[0].cpu().numpy()
            flt = np.transpose(y, (1, 2, 0)) * 255.0
            trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)   # authors' ToPILImage rule
            rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)
            r = dict(scene=sc, name=f)
            for k, pred in [("trunc", trc), ("round", rnd)]:
                r[f"psnr_{k}"] = psnr_indep(pred, gt)
                r[f"ssim_{k}"] = ssim_indep(pred, gt)
                rec = gt_mean_rectify(pred, gt)
                r[f"psnr_{k}_gtmean"] = psnr_indep(rec, gt)
                r[f"ssim_{k}_gtmean"] = ssim_indep(rec, gt)
            if len(rows) < 30:
                r["ssim_authors_trunc"] = ssim_authors(trc, gt)
            rows.append(r)
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{len(scenes)} scenes, {len(rows)} images", flush=True)

keys = [k for k in rows[0] if k not in ("scene", "name")]
avg = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
print(f"\n=== {a.tag}  scenes {len(scenes)}  images {len(rows)}")
print(f"{'variant':16s} {'PSNR':>8s} {'SSIM':>8s} | {'PSNR gtmean':>12s} {'SSIM gtmean':>12s}")
for k in ["trunc", "round"]:
    print(f"{k:16s} {avg['psnr_'+k]:8.4f} {avg['ssim_'+k]:8.4f} | "
          f"{avg['psnr_'+k+'_gtmean']:12.4f} {avg['ssim_'+k+'_gtmean']:12.4f}")
if "ssim_authors_trunc" in avg:
    sub = [r for r in rows if "ssim_authors_trunc" in r]
    d = np.mean([r["ssim_trunc"] for r in sub]) - np.mean([r["ssim_authors_trunc"] for r in sub])
    print(f"\nSSIM cross-check on first {len(sub)} images: independent vs authors' cv2 formula "
          f"delta {d:+.2e}")
print("\n참고 보고값: 이 가중치(CIDNet, arXiv 2502.20272 Table 2) Sony-Total-Dark PSNR 22.904 / SSIM 0.676; CIDNet+ 는 별도 모델로 23.482 / 0.691")
print("           Retinexformer README  SID PSNR 24.44 / SSIM 0.680")
with open(f"numbers/measure_{a.tag}.json", "w") as f:
    json.dump({"avg": avg, "n": len(rows), "per_image": rows}, f, indent=2)
