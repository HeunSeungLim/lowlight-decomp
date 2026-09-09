"""refute_*: does the single-frame model read the input noise level as an exposure cue?

Averaging k frames made the low-frequency error grow, not shrink. The hypothesis is that
the network, trained only on single noisy frames, uses noise level as evidence of how
dark the scene is; a cleaner input then reads as a brighter scene and the brightness field
comes out wrong. Test: take the k=8 average, re-inject synthetic noise until the input's
noise level matches a single frame, and see whether the low-frequency error returns to
its k=1 value. If it does, the cue hypothesis survives; if it does not, it is refuted and
the growth has another cause.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "code")
from audit_lf_bias_vs_variance import run, lowpass_mask, split_energy, groups8, gt_of, R

rng = np.random.default_rng(0)
mask = lowpass_mask(512, 960)
rows = {"k1": [], "k8": [], "k8+noise_match": [], "k8+noise_half": [], "k1+extra_noise": []}
for sc, exp, fs in groups8:
    gt = gt_of[sc]
    fr = [np.load(f"{R}/short_sid2/{sc}/{f}").astype(np.float32) / 255. for f in fs[:8]]
    x1, x8 = fr[0], np.mean(fr, 0)
    # per-pixel noise std of one frame, from the frame differences of this very group
    d = np.stack(fr[1:]) - fr[0]
    s1 = float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2.0))
    s8 = s1 / np.sqrt(8.0)
    add_match = np.sqrt(max(s1 ** 2 - s8 ** 2, 0.0))          # brings k=8 back to k=1 noise level
    cases = {"k1": x1, "k8": x8,
             "k8+noise_match": np.clip(x8 + rng.normal(0, add_match, x8.shape), 0, 1),
             "k8+noise_half": np.clip(x8 + rng.normal(0, add_match / 2, x8.shape), 0, 1),
             "k1+extra_noise": np.clip(x1 + rng.normal(0, s1, x1.shape), 0, 1)}
    for name, x in cases.items():
        y = run(x.astype(np.float32)) * 255.
        a = float((y * gt).sum() / (y * y).sum())
        lo, hi = split_energy(a * y - gt, mask)
        rows[name].append((lo, hi, a))

base = np.mean([r[0] for r in rows["k1"]])
print(f"{'input':18s} {'LF err / k1':>12s} {'HF err / k1':>12s} {'oracle gain a':>14s}")
out = {}
for name in rows:
    lo = np.mean([r[0] for r in rows[name]]); hi = np.mean([r[1] for r in rows[name]])
    a = np.mean([r[2] for r in rows[name]]); hi1 = np.mean([r[1] for r in rows["k1"]])
    print(f"{name:18s} {lo/base:12.3f} {hi/hi1:12.3f} {a:14.4f}")
    out[name] = dict(lf_ratio=float(lo / base), hf_ratio=float(hi / hi1), gain=float(a))
json.dump(out, open("numbers/refute_noise_cue.json", "w"), indent=2)
print("\n읽는 법: k8+noise_match 의 LF 가 k1 근처로 돌아오면 잡음이 노출 단서라는 가설이 살아남는다.")
print("        k1+extra_noise 의 LF 가 k1 보다 작아지면(더 어둡게 읽음) 같은 방향의 증거다.")
