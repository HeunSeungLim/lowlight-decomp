"""newbase_gsad.py -- GSAD (NeurIPS 2023) LOL-v1 추론 캐시.

방법: GSAD / GlobalDiff, "Global Structure-Aware Diffusion Process for
      Low-Light Image Enhancement", NeurIPS 2023.
가중치: 저자 공개 checkpoints/lolv1_gen.pth.

입력 규약은 우리 파이프라인과 동일하게 diag_lolv1.build_pairs() 하나만 쓴다.
저자 데이터로더(data/LoL_dataset.py 의 LOLv1_Dataset)는 import 하지 않고
전처리만 독립 재구현한다. 재구현 대상 (저자 코드 그대로):

  LoL_dataset.LOLv1_Dataset.__getitem__ (split=='val')
    1. cv2.imread -> BGR2RGB (uint8 RGB)            == build_pairs 의 lq
    2. cv2.copyMakeBorder(lr, 8,8,4,4, BORDER_REFLECT)   400x600 -> 416x608
    3. ToTensor()  == uint8/255, float32, CHW
    4. transform_augment(min_max=(-1,1))  ==  x*2-1
  test.py
    5. diffusion.test(continous=False) -> super_resolution -> p_sample_loop
       (val 스케줄: linear, n_timestep=20, linear_start=6e-4, linear_end=8.8e-1)
    6. core.metrics.tensor2img: clamp(-1,1) -> (x+1)/2 -> round(x*255)
    7. lolv1 은 normal_img[8:408, 4:604] 로 패딩 제거 -> 400x600

  저장은 6번의 8bit 양자화 전 float32 CHW RGB [0,1] (.npy).
  (저자 tensor2img 는 반올림. 8bit 규칙 차이는 채점 단계에서 둘 다 낸다.)

  주의: 저자 README/test.py 는 LLFlow·KinD 관례대로 출력 밝기를 GT 평균에
  맞춘 뒤 PSNR 을 잰다. 논문 보고값 27.6dB 은 GT-mean 정합값이다.
  여기서는 정합 없음/있음 둘 다 낸다.

채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import cv2

sys.path.insert(0, "code")
import diag_lolv1 as DL                                   # noqa: E402
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify   # noqa: E402

SCRATCH = "third_party"
REPO = os.path.join(SCRATCH, "GSAD")
WEIGHTS = os.path.join(SCRATCH, "dl/gsad/lolv1_gen.pth")

OUT_DIR = "numbers/cache_gsad/LOL"
OUT_JSON = "numbers/newbase_gsad.json"

# config/lolv1_test.json 그대로
SEED = 4088
UNET = dict(in_channel=6, out_channel=3, inner_channel=64,
            channel_mults=[1, 1, 2, 2, 4], attn_res=[16], res_blocks=2,
            dropout=0, norm_groups=32, image_size=128)
SCHED_TRAIN = dict(schedule="linear", n_timestep=500,
                   linear_start=1e-4, linear_end=2e-2)
SCHED_VAL = dict(schedule="linear", n_timestep=20,
                 linear_start=6e-4, linear_end=8.8e-1)
PAD = (8, 8, 4, 4)      # top, bottom, left, right


def build_model():
    import torch
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from model.ddpm_modules import unet, diffusion

    net = unet.UNet(**UNET)
    g = diffusion.GaussianDiffusion(net, image_size=UNET["image_size"],
                                    channels=6, loss_type="l1",
                                    conditional=True, schedule_opt=SCHED_TRAIN)
    dev = torch.device("cuda")
    # 저자 DDPM.__init__ 순서: train 스케줄로 버퍼 등록 -> state_dict 적재
    g.set_new_noise_schedule(SCHED_TRAIN, dev)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in sd):        # DataParallel 로 저장됨
        sd = {k[7:]: v for k, v in sd.items()}
    g.load_state_dict(sd, strict=True)
    g = g.to(dev).eval()
    # 저자 test.py: 생성 후 train -> val 스케줄로 다시 설정 (val 이 최종)
    g.set_new_noise_schedule(SCHED_VAL, dev)
    return g


def preprocess(lq_u8):
    """uint8 HWC RGB 400x600 -> float32 CHW [-1,1] 416x608 (저자 val 경로)"""
    t, b, l, r = PAD
    p = cv2.copyMakeBorder(lq_u8, t, b, l, r, cv2.BORDER_REFLECT)
    x = p.astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    return x * 2.0 - 1.0


def postprocess(hq, hw):
    """float CHW [-1,1] 416x608 -> float32 CHW RGB [0,1] 400x600"""
    t, b, l, r = PAD
    h, w = hw
    y = np.clip(hq, -1.0, 1.0)
    y = (y + 1.0) / 2.0
    return np.ascontiguousarray(y[:, t:t + h, l:l + w]).astype(np.float32)


def run_set(g, pairs, seed, save_dir=None):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = []
    for i, (f, lq, gt) in enumerate(pairs):
        t = time.time()
        x = torch.from_numpy(preprocess(lq))[None].cuda()
        with torch.no_grad():
            hq = g.super_resolution(x, False)          # (C,H,W), [-1,1]
        out = postprocess(hq.float().cpu().numpy(), lq.shape[:2])
        dt = time.time() - t
        assert out.shape == (3, lq.shape[0], lq.shape[1]), (f, out.shape)
        if save_dir:
            np.save(os.path.join(save_dir, f + ".npy"), out)

        flt = out.transpose(1, 2, 0) * 255.0
        rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)
        trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)
        r = dict(name=f, sec=dt)
        for key, pred in [("round", rnd), ("trunc", trc)]:
            r[f"psnr_{key}"] = psnr_indep(pred, gt)
            r[f"ssim_{key}"] = ssim_indep(pred, gt)
            r[f"psnr_{key}_gtmean"] = psnr_indep(gt_mean_rectify(pred, gt), gt)
            r[f"ssim_{key}_gtmean"] = ssim_indep(
                np.clip(np.rint(gt_mean_rectify(pred, gt)), 0, 255).astype(np.uint8), gt)
        rows.append(r)
        print(f"  {i+1:2d}/{len(pairs)} {f:>10s}  "
              f"PSNR(round)={r['psnr_round']:6.3f}  "
              f"gtmean={r['psnr_round_gtmean']:6.3f}  {dt:.2f}s", flush=True)
    return rows


def summarize(rows):
    keys = [k for k in rows[0] if k.startswith(("psnr_", "ssim_"))]
    return ({k: float(np.mean([r[k] for r in rows])) for k in keys},
            {k + "_sd": float(np.std([r[k] for r in rows], ddof=1)) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-sweep", type=str, default="",
                    help="쉼표로 구분한 추가 시드 (캐시는 안 덮어씀)")
    a = ap.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUT_DIR, exist_ok=True)

    g = build_model()
    nparam = sum(p.numel() for p in g.parameters())
    print(f"[gsad] params={nparam/1e6:.3f}M  steps={SCHED_VAL['n_timestep']}"
          f"  seed={SEED}", flush=True)

    pairs = DL.build_pairs()
    if a.limit:
        pairs = pairs[:a.limit]

    t0 = time.time()
    rows = run_set(g, pairs, SEED, save_dir=OUT_DIR)
    agg, sd = summarize(rows)

    sweep = {}
    for s in [int(x) for x in a.seed_sweep.split(",") if x.strip()]:
        print(f"-- seed sweep {s} (캐시 저장 안 함)", flush=True)
        rs = run_set(g, pairs, s, save_dir=None)
        m, _ = summarize(rs)
        sweep[str(s)] = {k: m[k] for k in
                         ("psnr_round", "psnr_round_gtmean", "ssim_round_gtmean")}

    out = dict(method="GSAD", venue="NeurIPS 2023", n=len(rows),
               weights=WEIGHTS, params_M=nparam / 1e6, cache=OUT_DIR,
               seed=SEED, sampling=SCHED_VAL, total_sec=time.time() - t0,
               mean=agg, sd=sd, seed_sweep=sweep, per_image=rows)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n[gsad] LOL-v1 eval15 (n=%d)" % len(rows))
    for k in ["psnr_round", "psnr_trunc", "psnr_round_gtmean",
              "psnr_trunc_gtmean", "ssim_round", "ssim_round_gtmean"]:
        print(f"  {k:22s} {agg[k]:.4f}")
    if sweep:
        print("  seed sweep:", json.dumps(sweep, indent=1))
    print("  json:", OUT_JSON)


if __name__ == "__main__":
    main()
