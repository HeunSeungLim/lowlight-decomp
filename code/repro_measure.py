"""audit_*: independent re-measurement of the reproduction.

Rule (from the depth project's audit_port_check.py): when checking ported/reproduced
results, do NOT reuse the source's measurement code -- reimplement and compare.
So this file builds PSNR/SSIM from scratch with numpy (own Gaussian kernel, own
convolution) and cross-checks against (a) the authors' cv2-based formulas and
(b) scikit-image, which is a third independent implementation.

It also measures the 8-bit conversion rule, because the authors save outputs with
torchvision ToPILImage -> .mul(255).byte(), which TRUNCATES. Rounding is the other
defensible choice and the two disagree by up to 1/255 per pixel.
"""
import os, sys, glob, json, argparse
import numpy as np
from PIL import Image

# ---------- independent implementation ----------
def gauss_kernel_1d(n=11, sigma=1.5):
    x = np.arange(n) - (n - 1) / 2.0
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def conv_valid(img, k1d):
    """separable 'valid' convolution, pure numpy (no cv2, no scipy)"""
    n = len(k1d)
    H, W = img.shape
    out = np.empty((H - n + 1, W), np.float64)
    for i in range(n):
        contrib = img[i:i + H - n + 1, :] * k1d[i]
        out = contrib if i == 0 else out + contrib
    out2 = np.empty((out.shape[0], W - n + 1), np.float64)
    for j in range(n):
        contrib = out[:, j:j + W - n + 1] * k1d[j]
        out2 = contrib if j == 0 else out2 + contrib
    return out2


def ssim_indep(a, b):
    """Wang et al. SSIM, 11x11 gaussian sigma 1.5, per channel, [0,255]."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = gauss_kernel_1d()
    vals = []
    for c in range(a.shape[2]):
        x = a[:, :, c].astype(np.float64)
        y = b[:, :, c].astype(np.float64)
        mx, my = conv_valid(x, k), conv_valid(y, k)
        mxx, myy, mxy = mx * mx, my * my, mx * my
        sx = conv_valid(x * x, k) - mxx
        sy = conv_valid(y * y, k) - myy
        sxy = conv_valid(x * y, k) - mxy
        m = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mxx + myy + C1) * (sx + sy + C2))
        vals.append(m.mean())
    return float(np.mean(vals))


def psnr_indep(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(10.0 * np.log10(255.0 * 255.0 / np.mean(d * d)))


# ---------- the authors' formulas, transcribed for comparison ----------
def ssim_authors(a, b):
    import cv2
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    vals = []
    for c in range(a.shape[2]):
        img1 = a[:, :, c].astype(np.float64)
        img2 = b[:, :, c].astype(np.float64)
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        s2 = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        s12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        vals.append((((2 * mu1_mu2 + C1) * (2 * s12 + C2)) /
                     ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))).mean())
    return float(np.mean(vals))


def ssim_skimage(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255,
                                       gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))


def gt_mean_rectify(pred_u8, gt_u8):
    """the authors' 'GT mean' trick: rescale so grayscale means match"""
    w = np.array([0.299, 0.587, 0.114])
    mp = (pred_u8.astype(np.float64) * w).sum(2).mean()
    mt = (gt_u8.astype(np.float64) * w).sum(2).mean()
    return np.clip(pred_u8.astype(np.float64) * (mt / mp), 0, 255)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()

    rows = []
    for f in sorted(os.listdir(a.gt)):
        if not f.lower().endswith((".png", ".jpg", ".bmp")):
            continue
        gt = np.array(Image.open(os.path.join(a.gt, f)).convert("RGB"))
        png = np.array(Image.open(os.path.join(a.pred, f)).convert("RGB"))
        npy = np.load(os.path.join(a.pred, f + ".npy"))          # float in [0,1], CHW
        flt = np.transpose(npy, (1, 2, 0)) * 255.0
        rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)     # round
        trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)    # truncate (= what they save)
        r = dict(name=f)
        for key, pred in [("saved_png", png), ("round", rnd), ("trunc", trc)]:
            r[f"psnr_{key}"] = psnr_indep(pred, gt)
            r[f"ssim_{key}"] = ssim_indep(pred, gt)
            rec = gt_mean_rectify(pred, gt)
            r[f"psnr_{key}_gtmean"] = psnr_indep(rec, gt)
            r[f"ssim_{key}_gtmean"] = ssim_indep(rec, gt)
        r["ssim_authors"] = ssim_authors(png, gt)
        r["ssim_skimage"] = ssim_skimage(png, gt)
        r["png_vs_trunc_maxdiff"] = int(np.abs(png.astype(int) - trc.astype(int)).max())
        rows.append(r)

    keys = [k for k in rows[0] if k != "name"]
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print(f"\n=== {a.tag}   n={len(rows)}")
    print(f"{'variant':22s} {'PSNR':>8s} {'SSIM':>8s} | {'PSNR gtmean':>12s} {'SSIM gtmean':>12s}")
    for key in ["saved_png", "round", "trunc"]:
        print(f"{key:22s} {avg['psnr_'+key]:8.4f} {avg['ssim_'+key]:8.4f} | "
              f"{avg['psnr_'+key+'_gtmean']:12.4f} {avg['ssim_'+key+'_gtmean']:12.4f}")
    print(f"\nSSIM cross-check on the saved PNG (should agree to ~1e-6):")
    print(f"  independent (this file) {avg['ssim_saved_png']:.6f}")
    print(f"  authors' cv2 formula    {avg['ssim_authors']:.6f}   "
          f"delta {avg['ssim_saved_png']-avg['ssim_authors']:+.2e}")
    print(f"  scikit-image            {avg['ssim_skimage']:.6f}   "
          f"delta {avg['ssim_saved_png']-avg['ssim_skimage']:+.2e}")
    print(f"saved PNG vs truncation of the float output: max abs diff "
          f"{max(r['png_vs_trunc_maxdiff'] for r in rows)} levels")

    with open(f"numbers/measure_{a.tag}.json", "w") as f:
        json.dump({"avg": avg, "per_image": rows}, f, indent=2)


if __name__ == "__main__":
    main()
