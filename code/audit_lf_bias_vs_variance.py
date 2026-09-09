"""audit_*: the precondition for the proposed method, checked with an independent implementation.

If the low-frequency error of the model is a frame-common bias, averaging k frames of the
scene cannot remove it and the proposal is dead. If it is variance, it falls as 1/k. This
is the gate the design document says must be passed before anything else is built, so it
is measured here from scratch: own frame grouping, own low-pass, own energy bookkeeping,
no code reused from the multi-frame measurement that runs in parallel.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "third_party/Retinexformer")
import torch.nn.functional as Fn
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

R = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
W = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth")
KS, CUT = (1, 2, 4, 8), 0.10                     # low-pass cutoff = the paper's 0.10
net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
net.load_state_dict(torch.load(W, map_location="cpu")["params"]); net = net.cuda().eval()


def run(x):
    t = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda().float()
    h, w = t.shape[2:]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw: t = Fn.pad(t, (0, pw, 0, ph), "reflect")
    with torch.inference_mode():
        y = torch.clamp(net(t)[:, :, :h, :w], 0, 1)
    return y[0].cpu().numpy().transpose(1, 2, 0)


def lowpass_mask(H, Wd):
    u = np.fft.fftfreq(H)[:, None]; v = np.fft.fftfreq(Wd)[None, :]
    return (np.sqrt((2 * u) ** 2 + (2 * v) ** 2) <= CUT)


def split_energy(res, mask):
    """squared error split into low- and high-pass parts, Parseval-exact"""
    Fr = np.fft.fft2(res, axes=(0, 1))
    P = np.abs(Fr) ** 2 / (res.shape[0] * res.shape[1])
    lo = P[mask].sum(); hi = P[~mask].sum()
    return lo, hi


groups = []          # (scene, exposure, [frames sorted])
for sc in sorted(s for s in os.listdir(f"{R}/short_sid2") if s.startswith("1")):
    by = {}
    for f in sorted(os.listdir(f"{R}/short_sid2/{sc}")):
        by.setdefault(f.rsplit("_", 1)[1].replace(".npy", ""), []).append(f)
    for exp, fs in by.items():
        if len(fs) >= 2: groups.append((sc, exp, fs))
gt_of = {sc: np.load(f"{R}/long_sid2/{sc}/" + sorted(os.listdir(f"{R}/long_sid2/{sc}"))[0]).astype(np.float64)
         for sc, _, _ in groups}

# same-scene subset for every k: keep only groups that have >= 8 frames so k=1..8 see identical scenes
groups8 = [g for g in groups if len(g[2]) >= 8]
print(f"groups with >=8 frames: {len(groups8)}  (scenes {len(set(g[0] for g in groups8))})")
mask = lowpass_mask(512, 960)
acc = {k: dict(lo=[], hi=[], tot=[], n=0) for k in KS}
by_exp = {}
for sc, exp, fs in groups8:
    gt = gt_of[sc]
    for k in KS:
        x = np.mean([np.load(f"{R}/short_sid2/{sc}/{f}").astype(np.float32) / 255. for f in fs[:k]], 0)
        y = run(x) * 255.
        a = float((y * gt).sum() / (y * y).sum())         # oracle global gain, as in the paper
        res = a * y - gt
        lo, hi = split_energy(res, mask)
        acc[k]["lo"].append(lo); acc[k]["hi"].append(hi); acc[k]["tot"].append((res ** 2).sum()); acc[k]["n"] += 1
        by_exp.setdefault(exp, {}).setdefault(k, []).append((lo, hi))

print(f"\n{'k':>3s} {'n':>4s} {'LF energy':>12s} {'ratio to k=1':>13s} {'1/k':>6s} {'HF energy':>12s} {'ratio':>7s}")
lo1 = np.mean(acc[1]["lo"]); hi1 = np.mean(acc[1]["hi"])
out = {"n_groups": len(groups8), "cutoff": CUT, "rows": [], "by_exposure": {}}
for k in KS:
    lo, hi = np.mean(acc[k]["lo"]), np.mean(acc[k]["hi"])
    print(f"{k:3d} {acc[k]['n']:4d} {lo:12.1f} {lo/lo1:13.3f} {1/k:6.3f} {hi:12.1f} {hi/hi1:7.3f}")
    out["rows"].append(dict(k=k, n=acc[k]["n"], lf=lo, lf_ratio=lo / lo1, hf=hi, hf_ratio=hi / hi1))
for exp in sorted(by_exp):
    out["by_exposure"][exp] = {}
    l1 = np.mean([a for a, _ in by_exp[exp][1]])
    print(f"\n  {exp}: n={len(by_exp[exp][1])}  LF ratio to k=1 by k:",
          " ".join(f"k{k}={np.mean([a for a,_ in by_exp[exp][k]])/l1:.3f}" for k in KS))
    for k in KS:
        out["by_exposure"][exp][k] = float(np.mean([a for a, _ in by_exp[exp][k]]) / l1)
# verdict on a fixed rule, written before the run: variance-dominated if LF at k=8 is below
# 0.5 of k=1 (pure variance would give 0.125), bias-dominated if above 0.8.
r8 = out["rows"][-1]["lf_ratio"]
out["verdict"] = "variance" if r8 < 0.5 else ("bias" if r8 > 0.8 else "mixed")
print(f"\nLF energy at k=8 relative to k=1: {r8:.3f}  ->  {out['verdict']}  (rule: <0.5 variance, >0.8 bias)")
json.dump(out, open("numbers/audit_lf_bias_vs_variance.json", "w"), indent=2)
