"""repro_sid_retinexformer.py

Retinexformer 저자 공개 SID 가중치(SID.pth)를 우리가 가진 SID npy 데이터에 돌려서
저자 보고값(PSNR 24.44 / SSIM 0.680)이 재현되는지 확인한다.

- 데이터 로딩은 저자 로더(basicsr/data/SID_image_dataset.py + util.read_img_seq2)를
  읽고 독립 재구현한다. 저자 로더 코드를 import 하지 않는다.
- 채점도 저자 코드를 그대로 쓰지 않고 repro_measure.psnr_indep / ssim_indep 로 독립 채점한다.
  비교용으로 저자 공식(Enhancement/utils.py 전사본)도 같이 잰다.
- 8bit 변환은 반올림(round)과 버림(trunc) 둘 다 잰다.
- 평균은 short 전체 장수(598)에 대한 평평한 평균.
"""
import os, sys, glob, json, argparse, math
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep, ssim_authors, gt_mean_rectify

REPO = "third_party/Retinexformer"
DATA = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
WEIGHTS = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth")


# ---------- 저자 로더 규약의 독립 재구현 ----------
# SID_image_dataset.Dataset_SIDImage:
#   test phase -> 폴더명 첫 글자가 '1' 인 씬만 사용
#   LQ = 폴더 안 idx 번째 파일, GT = 같은 이름 GT 폴더의 0번째 파일
# util.read_img2: np.load -> (size 주어지면) cv2.resize(img,(960,512)) -> float32/255
# util.read_img_seq2: imgs[:,:,:,[2,1,0]] (BGR->RGB 가정) -> CHW
def build_pairs():
    lq_dirs = sorted(glob.glob(os.path.join(DATA, "short_sid2", "*")))
    gt_dirs = sorted(glob.glob(os.path.join(DATA, "long_sid2", "*")))
    pairs = []
    for ld, gd in zip(lq_dirs, gt_dirs):
        name = os.path.basename(ld)
        if name[0] != "1":          # 저자 코드: "'1' in name[0]" == 첫 글자가 '1'
            continue
        lqs = sorted(glob.glob(os.path.join(ld, "*")))
        gts = sorted(glob.glob(os.path.join(gd, "*")))
        for p in lqs:
            pairs.append((name, p, gts[0]))
    return pairs


def load_npy(path, size=(960, 512), flip_channels=True):
    import cv2
    img = np.load(path)
    if size is not None and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, (size[0], size[1]))
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if flip_channels:
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)     # HWC, [0,1]


# ---------- 저자 채점 공식 전사 (Enhancement/utils.py) ----------
def psnr_authors_float(target, restored):
    mse = np.mean((target - restored) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def score_one(args):
    gt_u8, pred_f = args
    flt = pred_f * 255.0
    rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)
    trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)
    r = {}
    # 독립 채점
    for key, pred in (("round", rnd), ("trunc", trc)):
        r[f"psnr_{key}"] = psnr_indep(pred, gt_u8)
        r[f"ssim_{key}"] = ssim_indep(pred, gt_u8)
        rec = gt_mean_rectify(pred, gt_u8)
        r[f"psnr_{key}_gtmean"] = psnr_indep(rec, gt_u8)
        r[f"ssim_{key}_gtmean"] = ssim_indep(np.rint(rec).astype(np.uint8), gt_u8)
    # 저자 공식 (PSNR: float [0,1] / SSIM: img_as_ubyte == round)
    r["psnr_authors"] = psnr_authors_float(gt_u8.astype(np.float64) / 255.0,
                                           pred_f.astype(np.float64))
    r["ssim_authors"] = ssim_authors(rnd, gt_u8)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="numbers/measure_SID_retinexformer.json")
    ap.add_argument("--no-flip", action="store_true",
                    help="채널 뒤집기 없이(=npy 를 RGB 로 간주) 돌린다")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    import torch
    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    ck = torch.load(WEIGHTS, map_location="cpu")
    net.load_state_dict(ck["params"])
    net = net.cuda().eval()

    pairs = build_pairs()
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"pairs: {len(pairs)}  scenes: {len(set(p[0] for p in pairs))}")

    from multiprocessing import Pool
    pool = Pool(a.workers)
    jobs, meta = [], []
    gt_cache = {}

    import torch.nn.functional as F
    with torch.inference_mode():
        for i, (scene, lqp, gtp) in enumerate(pairs):
            lq = load_npy(lqp, flip_channels=not a.no_flip)
            if gtp not in gt_cache:
                gt_cache[gtp] = load_npy(gtp, flip_channels=not a.no_flip)
            gt = gt_cache[gtp]
            x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
            h, w = x.shape[2], x.shape[3]
            f = 4
            H, W = ((h + f) // f) * f, ((w + f) // f) * f
            padh = H - h if h % f != 0 else 0
            padw = W - w if w % f != 0 else 0
            if padh or padw:
                x = F.pad(x, (0, padw, 0, padh), "reflect")
            y = net(x)[:, :, :h, :w]
            y = torch.clamp(y, 0, 1).cpu().numpy()[0].transpose(1, 2, 0)
            gt_u8 = np.clip(np.rint(gt * 255.0), 0, 255).astype(np.uint8)
            jobs.append(pool.apply_async(score_one, ((gt_u8, y.astype(np.float64)),)))
            meta.append(dict(scene=scene, lq=lqp, gt=gtp))
            if (i + 1) % 50 == 0:
                print(f"  infer {i+1}/{len(pairs)}", flush=True)

    rows = []
    for m, j in zip(meta, jobs):
        r = dict(m); r.update(j.get()); rows.append(r)
    pool.close(); pool.join()

    keys = [k for k in rows[0] if k.startswith(("psnr_", "ssim_"))]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["n_images"] = len(rows)
    summary["n_scenes"] = len(set(r["scene"] for r in rows))
    summary["flip_channels"] = not a.no_flip
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fp:
        json.dump({"summary": summary, "rows": rows}, fp, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
