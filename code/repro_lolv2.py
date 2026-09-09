"""Third benchmark for the reproduction table (Table 4): LOLv2-real and LOLv2-synthetic.

Same recipe as the LOLv1 audit (repro/AUDIT_LOLv1_260905.md):
  - inference uses the authors' network definition and the authors' eval-time settings,
    because the point is to reproduce THEIR number;
  - scoring does NOT use the authors' code: psnr_indep / ssim_indep from repro_measure.py
    (pure numpy) are the reference, the authors' cv2 formula and scikit-image are run only
    as cross-checks;
  - both 8-bit rules (round / truncate) and both GT-mean settings (rect. / no rect.) are
    measured for every weight, with per-image std.

CIDNet eval.py settings for LOLv2 (transcribed, not imported):
  --lol_v2_real : model.trans.gated2 = True, model.trans.alpha = 0.84 (w_perc, "best gt mean"),
                  0.80 (best_PSNR), 0.82 (best_SSIM).  gated2 multiplies the final RGB by alpha.
  --lol_v2_syn  : no gating at all (v2 flag is only set for lol_v2_real).
  The CLI default alpha is 1.0 (== no gating), so real weights are run at both alphas.
Retinexformer test_from_dataset.py: pad to x4 reflect, no self-ensemble for the LOL numbers,
  PSNR on the float output (no quantisation), SSIM on img_as_ubyte (round). GT_mean rectifies
  the float output using cv2 BGR2GRAY applied to RGB data (i.e. weights 0.114R+0.587G+0.299B).
"""
import os, sys, json, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
from PIL import Image
from multiprocessing import Pool

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep, ssim_authors, ssim_skimage, gt_mean_rectify

CID = (os.environ.get("LLDATA", "data") + "/lowlight_model/HVI-CIDNet")
RF = "third_party/Retinexformer"
DATA = (os.environ.get("LLDATA", "data") + "/lowlight_model/data")
WA = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/author_all")
WR = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model")
OUT = "numbers"
OUT_IMG = (os.environ.get("LLDATA", "data") + "/lowlight_model/repro_out")   # per-image PNG/npy (home disk is 96% full)

# HF mirrors (okhater/lolv2-real, okhater/lolv2-synthetic) name the splits Input/GT, not
# Low/Normal as the authors' code expects. Synthetic Input files carry an 'r' prefix that the
# GT files do not have (r00816405t.png <-> 00816405t.png).
BENCH = {
    "real": dict(low=f"{DATA}/LOLv2_real/Test/Input", gt=f"{DATA}/LOLv2_real/Test/GT", low_prefix=""),
    "syn":  dict(low=f"{DATA}/LOLv2_syn/Test/Input",  gt=f"{DATA}/LOLv2_syn/Test/GT",  low_prefix="r"),
}

# tag, bench, model, weight file, gated2, alpha, reported {setting: (psnr, ssim)}
RUNS = [
    ("real_cid_bestPSNR_a0.80", "real", "cidnet", "LOLv2_real_best_PSNR.pth", True, 0.80,
     {"no rect.": (23.9040, 0.8656)}),
    ("real_cid_bestPSNR_a1.00", "real", "cidnet", "LOLv2_real_best_PSNR.pth", True, 1.00, {}),
    ("real_cid_bestSSIM_a0.82", "real", "cidnet", "LOLv2_real_best_SSIM.pth", True, 0.82,
     {"no rect.": (23.8975, 0.8705), "rect.": (28.3926, 0.8873)}),
    ("real_cid_bestSSIM_a1.00", "real", "cidnet", "LOLv2_real_best_SSIM.pth", True, 1.00, {}),
    ("real_cid_wperc_a0.84",    "real", "cidnet", "LOLv2_real_w_perc.pth",    True, 0.84,
     {"rect.": (28.1387, 0.8920)}),
    ("real_cid_wperc_a1.00",    "real", "cidnet", "LOLv2_real_w_perc.pth",    True, 1.00, {}),
    ("syn_cid_woperc",          "syn",  "cidnet", "LOLv2_syn_wo_perc.pth",    False, 1.0,
     {"no rect.": (25.7048, 0.9419), "rect.": (29.5663, 0.9497)}),
    ("syn_cid_wperc",           "syn",  "cidnet", "LOLv2_syn_w_perc.pth",     False, 1.0,
     {"no rect.": (25.1294, 0.9388), "rect.": (29.3666, 0.9500)}),
    ("real_rf",                 "real", "retinexformer", "LOL_v2_real.pth",   None, None,
     {"rect.": (27.71, 0.856)}),
    ("syn_rf",                  "syn",  "retinexformer", "LOL_v2_synthetic.pth", None, None,
     {"rect.": (29.04, 0.939)}),
]
LABEL = {
    "real_cid_bestPSNR_a0.80": "CIDNet (best PSNR, alpha 0.80)",
    "real_cid_bestSSIM_a0.82": "CIDNet (best SSIM, alpha 0.82)",
    "real_cid_wperc_a0.84":    "CIDNet (w_perc, alpha 0.84)",
    "syn_cid_woperc":          "CIDNet (wo_perc)",
    "syn_cid_wperc":           "CIDNet (w_perc)",
    "real_rf":                 "Retinexformer",
    "syn_rf":                  "Retinexformer",
}


def pairs(bench):
    b = BENCH[bench]
    names = sorted(f for f in os.listdir(b["gt"]) if f.lower().endswith(".png"))
    out = []
    for n in names:
        lo = os.path.join(b["low"], b["low_prefix"] + n)
        assert os.path.exists(lo), lo
        out.append((n, lo, os.path.join(b["gt"], n)))
    return out


# ---------------- inference ----------------
def infer_cidnet(weight, gated2, alpha, plist, outdir):
    import torch
    from torchvision import transforms
    sys.path.insert(0, CID)
    from net.CIDNet import CIDNet
    net = CIDNet().cuda()
    net.load_state_dict(torch.load(os.path.join(WA, weight), map_location="cpu"))
    net.eval()
    net.trans.gated = False
    net.trans.gated2 = bool(gated2)
    net.trans.alpha = float(alpha)
    tt = transforms.ToTensor()
    with torch.no_grad():
        for n, lo, _ in plist:
            x = tt(Image.open(lo).convert("RGB"))[None].cuda()
            y = torch.clamp(net(x), 0, 1)[0].cpu()
            transforms.ToPILImage()(y).save(os.path.join(outdir, n))        # authors: truncation
            np.save(os.path.join(outdir, n + ".npy"), y.numpy().astype(np.float32))
    del net
    torch.cuda.empty_cache()


def infer_retinexformer(weight, plist, outdir):
    import torch, torch.nn.functional as F
    sys.path.insert(0, RF)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(os.path.join(WR, weight), map_location="cpu")["params"])
    net = net.cuda().eval()
    with torch.inference_mode():
        for n, lo, _ in plist:
            lq = np.array(Image.open(lo).convert("RGB")).astype(np.float32) / 255.
            x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
            h, w = x.shape[2], x.shape[3]
            ph, pw = (-h) % 4, (-w) % 4
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), "reflect")
            y = torch.clamp(net(x)[:, :, :h, :w], 0, 1)[0].cpu().numpy().astype(np.float32)
            # authors save img_as_ubyte -> round
            Image.fromarray(np.clip(np.rint(y.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)).save(
                os.path.join(outdir, n))
            np.save(os.path.join(outdir, n + ".npy"), y)
    del net
    torch.cuda.empty_cache()


# ---------------- authors' GT-mean variants, transcribed (diagnostic only) ----------------
def gray_cv2_u8(img_u8):
    """cv2.cvtColor(RGB uint8, COLOR_RGB2GRAY): fixed point 4899/9617/1868 >> 14 with rounding"""
    r, g, b = [img_u8[:, :, i].astype(np.int64) for i in range(3)]
    return ((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14).astype(np.uint8)


def rect_cidnet_style(pred_u8, gt_u8):
    mp = gray_cv2_u8(pred_u8).mean()
    mt = gray_cv2_u8(gt_u8).mean()
    return np.clip(pred_u8.astype(np.float64) * (mt / mp), 0, 255)


def rect_rf_style(pred01, gt01):
    """Retinexformer: cv2 BGR2GRAY on RGB float data -> weights reversed (0.114R+0.587G+0.299B)"""
    w = np.array([0.114, 0.587, 0.299])
    mp = (pred01 * w).sum(2).mean()
    mt = (gt01 * w).sum(2).mean()
    return np.clip(pred01 * (mt / mp), 0, 1)


def psnr_float01(a01, b01):
    mse = np.mean((a01.astype(np.float64) - b01.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(1.0 / mse))


def measure_one(args):
    model, n, gtp, outdir = args
    gt = np.array(Image.open(gtp).convert("RGB"))
    png = np.array(Image.open(os.path.join(outdir, n)).convert("RGB"))
    npy = np.load(os.path.join(outdir, n + ".npy"))
    flt01 = np.transpose(npy, (1, 2, 0)).astype(np.float64)
    flt = flt01 * 255.0
    rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)
    trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)
    r = dict(name=n)
    for key, pred in [("round", rnd), ("trunc", trc)]:
        r[f"psnr_{key}"] = psnr_indep(pred, gt)
        r[f"ssim_{key}"] = ssim_indep(pred, gt)
        rec = gt_mean_rectify(pred, gt)                 # ours: float luma on the 8-bit pred
        r[f"psnr_{key}_gtmean"] = psnr_indep(rec, gt)
        r[f"ssim_{key}_gtmean"] = ssim_indep(rec, gt)
    # cross-check of three SSIM implementations on the saved PNG
    r["ssim_saved_png"] = ssim_indep(png, gt)
    r["ssim_authors"] = ssim_authors(png, gt)
    r["ssim_skimage"] = ssim_skimage(png, gt)
    if model == "cidnet":
        r["png_vs_trunc_maxdiff"] = int(np.abs(png.astype(int) - trc.astype(int)).max())
        rec = rect_cidnet_style(trc, gt)               # authors' measure.py rectification
        r["psnr_trunc_gtmean_cidstyle"] = psnr_indep(rec, gt)
        r["ssim_trunc_gtmean_cidstyle"] = ssim_indep(rec, gt)
    else:
        r["png_vs_round_maxdiff"] = int(np.abs(png.astype(int) - rnd.astype(int)).max())
        gt01 = gt.astype(np.float64) / 255.0
        r["psnr_float"] = psnr_float01(flt01, gt01)    # authors' PSNR: unquantised output
        rec01 = rect_rf_style(flt01, gt01)             # authors' GT_mean procedure
        r["psnr_float_gtmean_rfstyle"] = psnr_float01(rec01, gt01)
        r["ssim_round_gtmean_rfstyle"] = ssim_indep(
            np.clip(np.rint(rec01 * 255), 0, 255).astype(np.uint8), gt)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None, help="subset of run tags")
    p.add_argument("--skip-infer", action="store_true")
    p.add_argument("--workers", type=int, default=20)
    a = p.parse_args()

    result = {"data": {}, "runs": {}, "table4_rows": []}
    for b in BENCH:
        pl = pairs(b)
        sz = Image.open(pl[0][2]).size
        result["data"][b] = dict(low=BENCH[b]["low"], gt=BENCH[b]["gt"], n=len(pl),
                                 size_wh=list(sz), low_prefix=BENCH[b]["low_prefix"],
                                 first=pl[0][0], last=pl[-1][0])
        print(f"[data] {b}: {len(pl)} pairs, {sz[0]}x{sz[1]}, {pl[0][0]}..{pl[-1][0]}")

    for tag, bench, model, weight, gated2, alpha, reported in RUNS:
        if a.only and tag not in a.only:
            continue
        outdir = os.path.join(OUT_IMG, "LOLv2_" + tag)
        os.makedirs(outdir, exist_ok=True)
        pl = pairs(bench)
        t0 = time.time()
        if not a.skip_infer:
            if model == "cidnet":
                infer_cidnet(weight, gated2, alpha, pl, outdir)
            else:
                infer_retinexformer(weight, pl, outdir)
        t1 = time.time()
        with Pool(a.workers) as pool:
            rows = pool.map(measure_one, [(model, n, gtp, outdir) for n, _, gtp in pl])
        keys = [k for k in rows[0] if k != "name"]
        avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        std = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}
        meta = dict(bench=bench, model=model, weight=weight, gated2=gated2, alpha=alpha,
                    reported=reported, n=len(rows), outdir=outdir,
                    infer_sec=round(t1 - t0, 1), measure_sec=round(time.time() - t1, 1))
        result["runs"][tag] = dict(meta=meta, avg=avg, std=std, per_image=rows)

        print(f"\n=== {tag}  ({model}, {weight}, gated2={gated2}, alpha={alpha})  n={len(rows)}")
        print(f"{'8bit':8s} {'PSNR':>8s} {'SSIM':>8s} | {'PSNR rect':>10s} {'SSIM rect':>10s}")
        for key in ["round", "trunc"]:
            print(f"{key:8s} {avg['psnr_'+key]:8.4f} {avg['ssim_'+key]:8.4f} | "
                  f"{avg['psnr_'+key+'_gtmean']:10.4f} {avg['ssim_'+key+'_gtmean']:10.4f}")
        if model == "cidnet":
            print(f"authors' rect (cv2 u8 gray) on trunc: {avg['psnr_trunc_gtmean_cidstyle']:.4f} / "
                  f"{avg['ssim_trunc_gtmean_cidstyle']:.4f}   png==trunc maxdiff "
                  f"{max(r['png_vs_trunc_maxdiff'] for r in rows)}")
        else:
            print(f"authors' float PSNR {avg['psnr_float']:.4f}; authors' rect (float, reversed gray) "
                  f"{avg['psnr_float_gtmean_rfstyle']:.4f} / {avg['ssim_round_gtmean_rfstyle']:.4f}   "
                  f"png==round maxdiff {max(r['png_vs_round_maxdiff'] for r in rows)}")
        print(f"SSIM xcheck on saved PNG: indep {avg['ssim_saved_png']:.6f}  "
              f"cv2 delta {avg['ssim_saved_png']-avg['ssim_authors']:+.2e}  "
              f"skimage delta {avg['ssim_saved_png']-avg['ssim_skimage']:+.2e}")
        for setting, (rp, rs) in reported.items():
            rule = "trunc" if model == "cidnet" else "round"
            suf = "_gtmean" if setting == "rect." else ""
            ours_p, ours_s = avg[f"psnr_{rule}{suf}"], avg[f"ssim_{rule}{suf}"]
            print(f"  reported [{setting}] {rp:.4f}/{rs:.4f}  ours({rule}) {ours_p:.4f}/{ours_s:.4f}  "
                  f"dPSNR {ours_p-rp:+.4f}  {'REPRODUCED' if abs(ours_p-rp) <= 0.05 else 'OFF'}")

    # ---- rows for the manuscript's Table 4 (auto-read from this json) ----
    for tag, bench, model, weight, gated2, alpha, reported in RUNS:
        if tag not in result["runs"]:
            continue
        avg, std = result["runs"][tag]["avg"], result["runs"][tag]["std"]
        rule = "trunc" if model == "cidnet" else "round"
        for setting in ["rect.", "no rect."]:
            if setting not in reported:
                continue
            rp, rs = reported[setting]
            suf = "_gtmean" if setting == "rect." else ""
            ours_p, ours_s = avg[f"psnr_{rule}{suf}"], avg[f"ssim_{rule}{suf}"]
            result["table4_rows"].append(dict(
                bench="LOLv2-real" if bench == "real" else "LOLv2-syn",
                setting=f"{LABEL[tag]}, {setting}",
                ours=round(ours_p, 4), reported=rp,
                ours_ssim=round(ours_s, 4), reported_ssim=rs,
                ours_psnr_std=round(std[f"psnr_{rule}{suf}"], 4),
                diff=round(ours_p - rp, 4), reproduced=bool(abs(ours_p - rp) <= 0.05),
                rule8bit=rule, run=tag))

    print("\n=== table4_rows")
    for r in result["table4_rows"]:
        print(f"{r['bench']:11s} {r['setting']:40s} ours {r['ours']:8.4f}/{r['ours_ssim']:.4f}  "
              f"reported {r['reported']:8.4f}/{r['reported_ssim']:.4f}  diff {r['diff']:+.4f}  "
              f"{'ok' if r['reproduced'] else 'OFF'}")

    with open(os.path.join(OUT, "measure_LOLv2.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", os.path.join(OUT, "measure_LOLv2.json"))


if __name__ == "__main__":
    main()
