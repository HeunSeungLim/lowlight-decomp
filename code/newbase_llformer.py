"""newbase_llformer.py -- LLFormer (AAAI 2023) LOL-v1 추론 캐시.

방법: LLFormer, "Ultra-High-Definition Low-Light Image Enhancement:
      A Benchmark and Transformer-Based Method", AAAI 2023 (Oral).
가중치: 저자 공개 LOL 사전학습 (model_bestPSNR.pth, epoch 2186).

입력 규약은 우리 파이프라인과 동일하게 diag_lolv1.build_pairs() 하나만 쓴다.
저자 데이터로더(test.py 의 glob/natsort/PIL 경로)는 import 하지 않고,
전처리만 독립 재구현한다. 재구현 대상 (저자 test.py 그대로):

  1. PIL RGB -> TF.to_tensor  == uint8/255, float32, CHW           (여기선 lq/255)
  2. mul=16 배수로 우하단 reflect 패딩
       H = ((h+16)//16)*16, padh = H-h  단 h%16==0 이면 0
       400x600 -> padh=0, padw=8 -> 400x608
  3. model(input_) -> clamp(0,1) -> [:, :, :h, :w] 로 원해상도 복원
  4. 저장은 float32 CHW RGB [0,1] (.npy). 저자는 여기서 skimage.img_as_ubyte
     (= 반올림) 로 8bit 저장한다. 8bit 규칙 차이는 채점 단계에서 둘 다 낸다.

채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, "code")
import diag_lolv1 as DL                                   # noqa: E402
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify   # noqa: E402

SCRATCH = "third_party"
REPO = os.path.join(SCRATCH, "LLFormer")
WEIGHTS = os.path.join(SCRATCH, "dl/llformer/model_bestPSNR.pth")

OUT_DIR = "numbers/cache_llformer/LOL"
OUT_JSON = "numbers/newbase_llformer.json"

MUL = 16          # 저자 test.py 의 not_multiple_of
SEED = 20260905   # 결정적 경로(랜덤 없음)지만 관례상 고정


def build_model():
    import torch
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from model.LLFormer import LLFormer

    # 저자 test.py 의 생성 인자 그대로
    net = LLFormer(inp_channels=3, out_channels=3, dim=16,
                   num_blocks=[2, 4, 8, 16], num_refinement_blocks=2,
                   heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                   LayerNorm_type='WithBias', attention=True, skip=False)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=True), None
    net = net.cuda().eval()
    return net, ck.get("epoch", None)


def run_one(net, lq01):
    """lq01: float32 HWC RGB [0,1] -> float32 CHW RGB [0,1], 원해상도"""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(lq01.transpose(2, 0, 1)))[None].cuda()
    h, w = x.shape[2], x.shape[3]
    H = ((h + MUL) // MUL) * MUL
    W = ((w + MUL) // MUL) * MUL
    padh = H - h if h % MUL != 0 else 0
    padw = W - w if w % MUL != 0 else 0
    x = F.pad(x, (0, padw, 0, padh), "reflect")
    with torch.no_grad():
        y = net(x)
    y = torch.clamp(y, 0, 1)[:, :, :h, :w]
    return y[0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    net, epoch = build_model()
    nparam = sum(p.numel() for p in net.parameters())
    print(f"[llformer] params={nparam/1e6:.3f}M  ckpt_epoch={epoch}", flush=True)

    pairs = DL.build_pairs()
    if a.limit:
        pairs = pairs[:a.limit]

    rows = []
    t0 = time.time()
    for i, (f, lq, gt) in enumerate(pairs):
        t = time.time()
        out = run_one(net, lq.astype(np.float32) / 255.0)
        dt = time.time() - t
        assert out.shape == (3, lq.shape[0], lq.shape[1]), (f, out.shape)
        np.save(os.path.join(OUT_DIR, f + ".npy"), out.astype(np.float32))

        flt = out.transpose(1, 2, 0) * 255.0
        rnd = np.clip(np.rint(flt), 0, 255).astype(np.uint8)
        trc = np.clip(np.floor(flt), 0, 255).astype(np.uint8)
        r = dict(name=f, sec=dt)
        for key, pred in [("round", rnd), ("trunc", trc)]:
            r[f"psnr_{key}"] = psnr_indep(pred, gt)
            r[f"ssim_{key}"] = ssim_indep(pred, gt)
            r[f"psnr_{key}_gtmean"] = psnr_indep(gt_mean_rectify(pred, gt), gt)
        rows.append(r)
        print(f"  {i+1:2d}/{len(pairs)} {f:>10s}  "
              f"PSNR(round)={r['psnr_round']:6.3f}  "
              f"gtmean={r['psnr_round_gtmean']:6.3f}  {dt:.2f}s", flush=True)

    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in rows[0] if k.startswith(("psnr_", "ssim_"))}
    agg_sd = {k + "_sd": float(np.std([r[k] for r in rows], ddof=1))
              for k in rows[0] if k.startswith("psnr_")}
    out = dict(method="LLFormer", venue="AAAI 2023", n=len(rows),
               weights=WEIGHTS, ckpt_epoch=epoch, params_M=nparam / 1e6,
               cache=OUT_DIR, seed=SEED, total_sec=time.time() - t0,
               mean=agg, sd=agg_sd, per_image=rows)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n[llformer] LOL-v1 eval15 (n=%d)" % len(rows))
    for k in ["psnr_round", "psnr_trunc", "psnr_round_gtmean",
              "psnr_trunc_gtmean", "ssim_round"]:
        print(f"  {k:22s} {agg[k]:.4f}")
    print("  json:", OUT_JSON)


if __name__ == "__main__":
    main()
