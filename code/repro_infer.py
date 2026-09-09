"""Run the authors' released CIDNet weights on LOLv1 eval15 and save outputs.

Uses the authors' network definition and their eval-time settings (model.trans.gated=True
for LOL, ToTensor with no resize), because the point of this step is to reproduce THEIR
number. The measurement side is deliberately NOT theirs -- see repro_measure.py.
"""
import os, sys, argparse
sys.path.insert(0, (os.environ.get("LLDATA", "data") + "/lowlight_model/HVI-CIDNet"))
import torch, numpy as np
from PIL import Image
from torchvision import transforms
from net.CIDNet import CIDNet

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)      # dir with pytorch_model.bin
p.add_argument("--indir", required=True)
p.add_argument("--outdir", required=True)
p.add_argument("--gated", default="lol", choices=["lol", "v2", "none"])
p.add_argument("--alpha", type=float, default=1.0)
a = p.parse_args()

os.makedirs(a.outdir, exist_ok=True)
net = CIDNet().cuda()
sd = torch.load(os.path.join(a.weights, "pytorch_model.bin"), map_location="cpu")
net.load_state_dict(sd)
net.eval()
if a.gated == "lol":
    net.trans.gated = True
elif a.gated == "v2":
    net.trans.gated2 = True
    net.trans.alpha = a.alpha

tt = transforms.ToTensor()
files = sorted(f for f in os.listdir(a.indir) if f.lower().endswith((".png", ".jpg", ".bmp")))
with torch.no_grad():
    for f in files:
        x = tt(Image.open(os.path.join(a.indir, f)).convert("RGB"))[None].cuda()
        y = torch.clamp(net(x), 0, 1)[0].cpu()
        # the authors save via ToPILImage, which truncates (mul(255).byte()).
        # keep BOTH so the audit can measure how much the 8-bit rounding rule is worth.
        transforms.ToPILImage()(y).save(os.path.join(a.outdir, f))
        np.save(os.path.join(a.outdir, f + ".npy"), y.numpy().astype(np.float32))
print(f"wrote {len(files)} outputs to {a.outdir}")
