"""newbase_zerodcepp.py — Zero-DCE++ (TPAMI 2022) 저자 공개 가중치 추론 → 출력 캐시.

논문 : Li, Guo, Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep
       Curve Estimation", IEEE TPAMI 44(8), 2022.
코드 : github.com/Li-Chongyi/Zero-DCE_extension (Zero-DCE++)
가중치: 저장소 동봉 snapshots_Zero_DCE++/Epoch99.pth (저자 배포)
학습  : 저자 학습본. 무참조(zero-reference) 학습이라 LOL/SID 의 정답쌍을 쓰지 않았다.
        그래서 LOL 과 Sony(SID sRGB) 둘 다 같은 가중치로 돌린다. 재학습 없음.

규약
  - LOL  : diag_lolv1.build_pairs() 의 PIL RGB uint8 HWC 를 /255 (400x600).
  - Sony : compare_methods.sid_pairs() / sid_load() (HWC float [0,1] RGB 512x960,
           저자 로더의 채널 뒤집기 포함). 우리 파이프라인과 동일.
  - 저자 lowlight_test.py 의 기본값 scale_factor=12 를 그대로 쓴다.

저자 전처리 한 가지가 해상도를 바꾼다
  lowlight_test.py 는 입력을 scale_factor 의 배수로 자른다:
    LOL  400x600 -> 396x600 (아래 4행 버림)
    Sony 512x960 -> 504x960 (아래 8행 버림)
  캐시 규약은 입력 해상도이므로 자른 결과를 그대로 쓸 수 없다. 다음처럼 처리한다:
    1) 저자와 똑같이 자른 입력으로 곡선맵 x_r 을 얻는다 (여기까지 저자 코드 그대로).
    2) x_r 을 아래쪽으로 가장자리 복제(replicate) 확장해 입력 해상도로 만든다.
    3) 저자 enhance() 를 원본 해상도 입력에 적용한다.
  자른 영역 안에서는 저자 출력과 비트 단위로 같다는 것을 매 프레임 확인한다(최대차 0).
  버려지는 4/8행의 영향을 따로 알 수 있게 '자른 영역만'의 PSNR 도 같이 낸다.

측정은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

CODE = "code"
sys.path.insert(0, CODE)
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify   # noqa: E402
import diag_lolv1 as DL                                              # noqa: E402
import compare_methods as CM                                         # noqa: E402

REPO = os.path.join("third_party", "Zero-DCE++")
sys.path.insert(0, REPO)
import model as zdce_model                                           # noqa: E402

WEIGHTS = os.path.join(REPO, "snapshots_Zero_DCE++", "Epoch99.pth")
CACHE = "numbers/cache_zerodcepp"
OUT_JSON = "numbers/newbase_zerodcepp.json"
SCALE_FACTOR = 12          # lowlight_test.py 의 기본값


def build_net(scale_factor=SCALE_FACTOR):
    net = zdce_model.enhance_net_nopool(scale_factor).cuda()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cuda", weights_only=True))
    net.eval()
    return net


def forward_full(net, x, sf=SCALE_FACTOR):
    """x: 1x3xHxW [0,1] cuda -> (원본 해상도 출력, 저자 크롭 출력, 크롭 크기, 최대 불일치)"""
    H, W = x.shape[2], x.shape[3]
    h, w = (H // sf) * sf, (W // sf) * sf
    xc = x[:, :, 0:h, 0:w].contiguous()          # 저자 전처리와 동일한 자르기
    y_crop, x_r = net(xc)                        # 저자 forward 그대로
    if (h, w) != (H, W):
        x_r_full = F.pad(x_r, (0, W - w, 0, H - h), mode="replicate")
    else:
        x_r_full = x_r
    y_full = net.enhance(x, x_r_full)            # 저자 enhance() 를 원본 해상도에 적용
    dev = float((y_full[:, :, 0:h, 0:w] - y_crop).abs().max())
    return y_full, y_crop, (h, w), dev


def measure(pred_u8, gt_u8, r, tag):
    r[f"psnr_{tag}"] = psnr_indep(pred_u8, gt_u8)
    r[f"ssim_{tag}"] = ssim_indep(pred_u8, gt_u8)
    rec = gt_mean_rectify(pred_u8, gt_u8)
    r[f"psnr_{tag}_gtmean"] = psnr_indep(rec, gt_u8)
    r[f"ssim_{tag}_gtmean"] = ssim_indep(rec, gt_u8)


def frames(bench, limit=0):
    if bench == "LOL":
        pairs = DL.build_pairs()
        if limit:
            pairs = pairs[:limit]
        for f, lq, gt in pairs:
            yield f, lq.astype(np.float32) / 255.0, gt
    else:
        pairs = CM.sid_pairs()
        if limit:
            pairs = pairs[:limit]
        gt_cache = {}
        for scene, lqp, gtp in pairs:
            if gtp not in gt_cache:
                gt_cache[gtp] = CM.u8(CM.sid_load(gtp))
            yield os.path.basename(lqp), CM.sid_load(lqp), gt_cache[gtp]


def run_bench(net, bench, limit=0, sf=SCALE_FACTOR):
    out_dir = os.path.join(CACHE, bench)
    os.makedirs(out_dir, exist_ok=True)
    rows, worst_dev, t0 = [], 0.0, time.time()
    with torch.no_grad():
        for i, (fid, lq01, gt) in enumerate(frames(bench, limit)):
            x = torch.from_numpy(np.ascontiguousarray(lq01.transpose(2, 0, 1)))[None].cuda()
            y_full, y_crop, (h, w), dev = forward_full(net, x, sf)
            worst_dev = max(worst_dev, dev)
            assert dev == 0.0, (bench, fid, dev)      # 자른 영역은 저자 출력과 동일해야 한다

            arr = torch.clamp(y_full, 0, 1)[0].cpu().numpy().astype(np.float32)
            assert arr.shape == (3,) + gt.shape[:2], (fid, arr.shape, gt.shape)
            np.save(os.path.join(out_dir, fid + ".npy"), arr)

            flt = arr.transpose(1, 2, 0) * 255.0
            r = dict(name=fid, crop=[h, w])
            measure(np.clip(np.rint(flt), 0, 255).astype(np.uint8), gt, r, "round")
            measure(np.clip(np.floor(flt), 0, 255).astype(np.uint8), gt, r, "trunc")
            # 저자 크롭 영역만 (해상도 복원의 영향을 분리해서 본다)
            fc = np.clip(torch.clamp(y_crop, 0, 1)[0].cpu().numpy(), 0, 1)
            fc = fc.transpose(1, 2, 0) * 255.0
            r["psnr_crop_round"] = psnr_indep(
                np.clip(np.rint(fc), 0, 255).astype(np.uint8), gt[0:h, 0:w])
            rows.append(r)
            if bench == "LOL" or (i + 1) % 100 == 0:
                print(f"  [{bench}] {i+1} {fid:34s} {r['psnr_round']:6.3f} dB "
                      f"(크롭만 {r['psnr_crop_round']:6.3f})", flush=True)

    keys = [k for k in rows[0] if k not in ("name", "crop")]
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    sd = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}
    print(f"\n=== Zero-DCE++ / {bench}  n={len(rows)}  ({time.time()-t0:.1f}s)")
    print(f"{'8bit 규칙':12s} {'PSNR':>8s} {'sd':>6s} {'SSIM':>8s} | "
          f"{'PSNR GT평균':>11s} {'SSIM GT평균':>11s}")
    for k in ["round", "trunc"]:
        print(f"{k:12s} {avg['psnr_'+k]:8.4f} {sd['psnr_'+k]:6.3f} {avg['ssim_'+k]:8.4f} | "
              f"{avg['psnr_'+k+'_gtmean']:11.4f} {avg['ssim_'+k+'_gtmean']:11.4f}")
    print(f"저자 크롭 영역만(반올림) {avg['psnr_crop_round']:.4f} dB  "
          f"(전체 대비 {avg['psnr_crop_round']-avg['psnr_round']:+.4f})")
    print(f"크롭 영역 저자 출력과의 최대 불일치: {worst_dev:g}")
    print(f"캐시: {out_dir}  ({len(os.listdir(out_dir))} 파일)\n")
    return {"n": len(rows), "avg": avg, "sd": sd, "max_dev_vs_author_crop": worst_dev,
            "per_image": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default="LOL,Sony")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale-factor", type=int, default=SCALE_FACTOR)
    a = ap.parse_args()

    net = build_net(a.scale_factor)
    print(f"Zero-DCE++ Epoch99.pth, scale_factor={a.scale_factor}, "
          f"params={sum(p.numel() for p in net.parameters())}", flush=True)

    out = {}
    for b in a.benches.split(","):
        if b:
            out[b] = run_bench(net, b, a.limit, a.scale_factor)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("json:", OUT_JSON)


if __name__ == "__main__":
    main()
