"""newbase_snrnet.py

SNR-Net (SNR-aware Low-Light Image Enhancement, CVPR 2022) 저자 공개 가중치를
우리 벤치마크 두 개(LOLv1 eval15 15장 / SID Sony sRGB test 598장)에 돌려
출력 캐시를 만들고 저자 보고값이 재현되는지 확인한다.

규약
  - 네트워크(models/archs/low_light_transformer.py)만 저자 저장소에서 import 한다.
    데이터로더(data/dataset_LOLv1.py, data/dataset_SID.py, data/util.py)와
    모델 래퍼(models/Video_base_model4_m.py)는 import 하지 않고 아래에서 독립 재구현한다.
  - 입력 프레임은 우리 파이프라인과 같은 소스에서 받는다.
      LOL  : diag_lolv1.build_pairs()  (PIL RGB uint8, 400x600)
      Sony : compare_methods.sid_pairs() + compare_methods.sid_load()
             (= diag_sid_failure.load_npy, np.load -> /255 -> BGR->RGB, 512x960)
  - 채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.
  - 8bit 변환은 반올림(round)과 버림(trunc) 둘 다, GT-mean 정합 없음/있음 둘 다 잰다.

저자 전처리 재현 근거 (저자 코드 위치)
  data/util.py read_img/read_img2  : cv2.imread(UNCHANGED) 또는 np.load -> float32/255
  data/util.py read_img_seq(2)     : imgs[:,:,:,[2,1,0]] 로 BGR->RGB, HWC->CHW
  data/dataset_LOLv1.py            : phase 'test' 이면 리사이즈 없이 원본 크기
  data/dataset_SID.py              : train_size [960,512] 로 cv2.resize (원본이 이미 512x960)
  두 데이터셋 공통                  : nf = cv2.blur(LQ*255, (5,5))/255
  models/Video_base_model4_m.py    : SNR map
      dark  = 0.299R + 0.587G + 0.114B   (LQ)
      light = 0.299R + 0.587G + 0.114B   (nf)
      noise = |dark - light|
      mask  = light / (noise + 1e-4);  mask /= (max(mask) + 1e-4);  clamp(0,1)
  test_LOLv1_v2_real.py -> model.test4() : LQ/mask 를 400x608 로 bilinear 리사이즈해
      forward 하고 출력을 원본 크기로 되돌린다 (fea 를 4x4 로 unfold 하므로 W 가 16 의
      배수여야 한다. 600 은 아니고 608 은 맞다).
  test.py -> model.test()                : SID 는 512x960 이 이미 16 의 배수라 리사이즈 없음.

출력
  numbers/cache_snrnet/{LOL,Sony}/<id>.npy
  float32, CHW, RGB, [0,1], 입력과 같은 해상도
"""
import os, sys, json, time, argparse
import numpy as np
import cv2

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify

REPO = "third_party/SNR-Aware-Low-Light-Enhance"
WDIR = "third_party/dl/snr/pretrain_model"
W = {"LOL": os.path.join(WDIR, "LOLv1.pth"), "Sony": os.path.join(WDIR, "SID.pth")}
CACHE = "numbers/cache_snrnet"
OUT_JSON = os.environ.get("SNR_OUT", "numbers/newbase_snrnet.json")

# 저자 보고값 (CVPR2022 본문 Table)
PAPER = {"LOL": 24.61, "Sony": 22.87}


# ---------------- 네트워크 ----------------
def build_net(bench, torch):
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import models.archs.low_light_transformer as M
    # options/test/{LOLv1,SID}.yml 의 network_G 는 두 벤치마크가 동일하다
    net = M.low_light_transformer(nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1,
                                  center=None, predeblur=True, HR_in=True, w_TSA=True)
    sd = torch.load(W[bench], map_location="cpu", weights_only=False)
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(clean, strict=True), None
    net = net.cuda().eval()
    return net


# ---------------- 저자 전처리 독립 재구현 ----------------
def snr_mask(lq01_chw, torch):
    """lq01_chw: (1,3,H,W) float32 텐서 (cuda). 저자 test()/test4() 의 mask 계산 전사."""
    lq_hwc = lq01_chw[0].permute(1, 2, 0).cpu().numpy()          # HWC RGB float32
    nf = cv2.blur(lq_hwc * 255.0, (5, 5)) * (1.0 / 255.0)        # dataset_*.py 의 img_nf
    nf = torch.from_numpy(np.ascontiguousarray(nf)).float().permute(2, 0, 1)[None].cuda()
    w = (0.299, 0.587, 0.114)
    dark = lq01_chw[:, 0:1] * w[0] + lq01_chw[:, 1:2] * w[1] + lq01_chw[:, 2:3] * w[2]
    light = nf[:, 0:1] * w[0] + nf[:, 1:2] * w[1] + nf[:, 2:3] * w[2]
    noise = torch.abs(dark - light)
    mask = torch.div(light, noise + 0.0001)
    b, _, h, wd = mask.shape
    mmax = torch.max(mask.view(b, -1), dim=1)[0].view(b, 1, 1, 1).repeat(1, 1, h, wd)
    mask = mask * 1.0 / (mmax + 0.0001)
    return torch.clamp(mask, min=0, max=1.0).float()


def run_frame(net, lq01_hwc, bench, torch):
    """lq01_hwc: HWC float32 RGB [0,1]. 반환: CHW float32 RGB [0,1]."""
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(lq01_hwc.transpose(2, 0, 1)))[None].float().cuda()
    with torch.no_grad():
        mask = snr_mask(x, torch)
        if bench == "LOL":
            # test_LOLv1_v2_real.py -> test4()
            B, C, H, Wd = x.shape
            xr = F.interpolate(x, size=[400, 608], mode="bilinear")
            mr = F.interpolate(mask, size=[400, 608], mode="bilinear")
            y = net(xr, mr)
            y = F.interpolate(y, size=[H, Wd], mode="bilinear")
        else:
            # test.py -> test()  (512x960 은 이미 16 의 배수)
            y = net(x, mask)
        y = torch.clamp(y, 0.0, 1.0)
    return y[0].float().cpu().numpy().astype(np.float32)


# ---------------- 프레임 소스 (우리 파이프라인과 동일) ----------------
def frames(bench, limit=0):
    if bench == "LOL":
        import diag_lolv1 as DL
        pairs = DL.build_pairs()
        if limit:
            pairs = pairs[:limit]
        for f, lq, gt in pairs:
            yield f, lq.astype(np.float32) / 255.0, gt
    else:
        import compare_methods as CM
        pairs = CM.sid_pairs()
        if limit:
            pairs = pairs[:limit]
        gt_cache = {}
        for scene, lqp, gtp in pairs:
            if gtp not in gt_cache:
                gt_cache[gtp] = np.clip(np.rint(CM.sid_load(gtp) * 255.0), 0, 255).astype(np.uint8)
            yield os.path.basename(lqp), CM.sid_load(lqp).astype(np.float32), gt_cache[gtp]


def n_frames(bench):
    if bench == "LOL":
        import diag_lolv1 as DL
        return len(DL.build_pairs())
    import compare_methods as CM
    return len(CM.sid_pairs())


# ---------------- 채점 ----------------
def score(pred_chw, gt_u8, want_ssim):
    flt = pred_chw.transpose(1, 2, 0).astype(np.float64) * 255.0
    r = {}
    for key, p in (("round", np.clip(np.rint(flt), 0, 255).astype(np.uint8)),
                   ("trunc", np.clip(np.floor(flt), 0, 255).astype(np.uint8))):
        r["psnr_" + key] = psnr_indep(p, gt_u8)
        rec = gt_mean_rectify(p, gt_u8)
        r["psnr_%s_gtmean" % key] = psnr_indep(rec, gt_u8)
        if want_ssim and key == "round":
            r["ssim_round"] = ssim_indep(p, gt_u8)
            r["ssim_round_gtmean"] = ssim_indep(np.rint(rec).astype(np.uint8), gt_u8)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="both", choices=["LOL", "Sony", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ssim", action="store_true")
    a = ap.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    benches = ["LOL", "Sony"] if a.bench == "both" else [a.bench]

    out = {}
    for bench in benches:
        os.makedirs(os.path.join(CACHE, bench), exist_ok=True)
        net = build_net(bench, torch)
        rows, t0 = [], time.time()
        total = a.limit or n_frames(bench)
        for i, (fid, lq01, gt_u8) in enumerate(frames(bench, a.limit)):
            pred = run_frame(net, lq01, bench, torch)
            assert pred.shape == (3,) + lq01.shape[:2], (fid, pred.shape)
            np.save(os.path.join(CACHE, bench, fid + ".npy"), pred)
            r = score(pred, gt_u8, not a.no_ssim)
            r["id"] = fid
            rows.append(r)
            if (i + 1) % 25 == 0 or i + 1 == total:
                el = time.time() - t0
                print("[%s] %d/%d  psnr_round(avg)=%.3f  %.1fs (%.2fs/frame)"
                      % (bench, i + 1, total, np.mean([x["psnr_round"] for x in rows]),
                         el, el / (i + 1)), flush=True)
        agg = {k: float(np.mean([x[k] for x in rows]))
               for k in rows[0] if k != "id"}
        agg["n"] = len(rows)
        agg["psnr_round_std"] = float(np.std([x["psnr_round"] for x in rows]))
        agg["paper"] = PAPER[bench]
        agg["weights"] = W[bench]
        out[bench] = {"agg": agg, "per_frame": rows}
        print("[%s] %s" % (bench, json.dumps(agg, ensure_ascii=False)), flush=True)
        del net
        torch.cuda.empty_cache()

    prev = {}
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(open(OUT_JSON))
        except Exception:
            prev = {}
    prev.update(out)
    json.dump(prev, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
