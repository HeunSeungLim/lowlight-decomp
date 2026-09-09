"""newbase_uretinex.py — URetinex-Net (CVPR 2022) 저자 공개 가중치 추론 → 출력 캐시.

논문 : Wu et al., "URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light
       Image Enhancement", CVPR 2022, pp.5901-5910.
코드 : github.com/AndersonYong/URetinex-Net (커밋 7ee2196)
가중치: 저장소에 동봉된 ckpt/{init_low,init_high,unfolding,L_adjust}.pth (저자 배포)
학습  : 저자 배포본은 LOL 학습본. 재학습·미세조정 없음.

규약
  - 입력은 우리 파이프라인과 동일: diag_lolv1.build_pairs() 의 PIL RGB uint8 HWC 를 /255.
    저자 데이터로더(transforms.ToTensor(Image.open(...)))는 import 하지 않고 독립 재구현하되,
    같은 값이 나오는지 15장 전량 화소단위로 대조한다(--audit-loader).
  - 조명 조정비는 저자 test.py 기본값 ratio=5 를 그대로 쓴다(주 설정, 캐시에 저장되는 값).
  - 저자 evaluate.py 는 GT(정상광) 영상의 분해 조도로 ratio 를 프레임마다 만든다.
    논문 보고값은 그 설정에서 나온 것이므로, GT 를 쓰는 그 변형도 같이 재서 진단으로만 남긴다
    (캐시에는 넣지 않는다).
  - 출력은 입력 해상도, [0,1] 클리핑, float32 CHW RGB 로 저장.

측정은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

CODE = "code"
sys.path.insert(0, CODE)
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify   # noqa: E402
import diag_lolv1 as DL                                              # noqa: E402

REPO = os.path.join("third_party", "URetinex-Net")
sys.path.insert(0, REPO)
from network.Math_Module import P as P_module, Q as Q_module         # noqa: E402
from network.decom import Decom                                      # noqa: E402
from network.restoration import HalfDnCNNSE                          # noqa: E402
from network.illumination_enhance import Illumination_Alone          # noqa: E402
from network.illumination_adjustment import Adjust_naive             # noqa: E402

CKPT = os.path.join(REPO, "ckpt")
OUT_DIR = "numbers/cache_uretinex/LOL"
OUT_JSON = "numbers/newbase_uretinex.json"
RATIO_DEFAULT = 5          # test.py 의 --ratio 기본값


# ---------------- 가중치 적재 (utils.py 와 같은 키 구조, torch 2.x 대응) ----------------
def _load(path):
    # 저자 체크포인트는 argparse.Namespace(opts)를 함께 담고 있어 weights_only 로는 못 연다.
    return torch.load(path, map_location="cpu", weights_only=False)


def build_models():
    ck_low = _load(os.path.join(CKPT, "init_low.pth"))
    dec_low = Decom()
    dec_low.load_state_dict(ck_low["state_dict"]["model_R"])

    ck_high = _load(os.path.join(CKPT, "init_high.pth"))
    dec_high = Decom()
    dec_high.load_state_dict(ck_high["state_dict"]["model_R"])

    ck_unf = _load(os.path.join(CKPT, "unfolding.pth"))
    uopts = ck_unf["opts"]
    assert uopts.R_model == "HalfDnCNNSE" and uopts.L_model == "Illumination_Alone", uopts
    model_R = HalfDnCNNSE(uopts)
    model_L = Illumination_Alone(uopts)
    model_R.load_state_dict(ck_unf["state_dict"]["model_R"])
    model_L.load_state_dict(ck_unf["state_dict"]["model_L"])

    ck_adj = _load(os.path.join(CKPT, "L_adjust.pth"))
    assert ck_adj["opts"].A_model == "naive", ck_adj["opts"]
    model_A = Adjust_naive(ck_adj["opts"])
    model_A.load_state_dict(ck_adj["state_dict"]["model_A"])

    nets = dict(dec_low=dec_low, dec_high=dec_high, R=model_R, L=model_L, A=model_A)
    for n in nets.values():
        n.cuda().eval()
        for p in n.parameters():
            p.requires_grad = False
    return nets, uopts


class URetinex:
    """test.py / evaluate.py 의 Inference 와 같은 계산 그래프."""

    def __init__(self):
        self.nets, self.uopts = build_models()
        self.P = P_module()
        self.Q = Q_module()

    def unfolding(self, I):
        for t in range(self.uopts.round):
            if t == 0:
                Pt, Qt = self.nets["dec_low"](I)
            else:
                w_p = self.uopts.gamma + self.uopts.Roffset * t
                w_q = self.uopts.lamda + self.uopts.Loffset * t
                Pt = self.P(I=I, Q=Qt, R=R, gamma=w_p)
                Qt = self.Q(I=I, P=Pt, L=L, lamda=w_q)
            R = self.nets["R"](r=Pt, l=Qt)
            L = self.nets["L"](l=Qt)
        return R, L

    def adjust(self, L, ratio_scalar_or_map):
        alpha = torch.ones_like(L) * ratio_scalar_or_map
        return self.nets["A"](l=L, alpha=alpha)

    def run(self, I, ratio):
        """I: 1x3xHxW [0,1] cuda. ratio: float (test.py) 또는 텐서 (evaluate.py)."""
        R, L = self.unfolding(I)
        return self.adjust(L, ratio) * R, L

    def gt_ratio(self, L_low, I_high):
        """evaluate.py get_ratio: 정상광 영상의 분해 조도로 프레임별 비율을 만든다."""
        _, high_L = self.nets["dec_high"](I_high)
        r = (L_low / (high_L + 0.0001)).mean()
        return torch.ones_like(high_L) * (1.0 / (r + 0.0001))


# ---------------- 입력 규약 대조 (저자 로더를 쓰지 않고 같은 값인지 확인) ----------------
def audit_loader(pairs):
    """우리 규약(np uint8 /255)이 저자 transforms.ToTensor(PIL) 와 같은 텐서인지 대조."""
    from PIL import Image
    from torchvision import transforms
    tt = transforms.ToTensor()
    worst = 0.0
    for f, lq, _ in pairs:
        ours = torch.from_numpy(lq.astype(np.float32) / 255.0).permute(2, 0, 1)
        theirs = tt(Image.open(os.path.join(DL.LOW, f)))
        assert ours.shape == theirs.shape, (f, ours.shape, theirs.shape)
        worst = max(worst, float((ours - theirs).abs().max()))
    return worst


# ---------------- 실행 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=RATIO_DEFAULT)
    ap.add_argument("--ratio-sweep", type=str, default="3,4,5")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = DL.build_pairs()
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"LOL-v1 eval15: {len(pairs)} pairs, shape {pairs[0][1].shape}", flush=True)

    d = audit_loader(pairs)
    print(f"[audit] 우리 입력 규약 vs 저자 ToTensor(PIL): 최대 절대차 {d:.3e}", flush=True)
    assert d == 0.0, "입력 규약이 저자 전처리와 다르다"

    net = URetinex()
    print(f"[unfolding opts] round={net.uopts.round} gamma={net.uopts.gamma} "
          f"Roffset={net.uopts.Roffset} lamda={net.uopts.lamda} Loffset={net.uopts.Loffset} "
          f"concat_L={net.uopts.concat_L}", flush=True)

    sweep = [float(s) for s in a.ratio_sweep.split(",") if s]
    rows, t0 = [], time.time()
    with torch.no_grad():
        for f, lq, gt in pairs:
            x = torch.from_numpy(lq.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
            y, L_low = net.run(x, a.ratio)
            y = torch.clamp(y, 0, 1)[0]
            arr = y.cpu().numpy().astype(np.float32)
            assert arr.shape == (3,) + lq.shape[:2], (f, arr.shape)
            np.save(os.path.join(OUT_DIR, f + ".npy"), arr)

            flt = arr.transpose(1, 2, 0) * 255.0
            r = dict(name=f, ratio=a.ratio)
            for k, pred in [("round", np.clip(np.rint(flt), 0, 255).astype(np.uint8)),
                            ("trunc", np.clip(np.floor(flt), 0, 255).astype(np.uint8))]:
                r[f"psnr_{k}"] = psnr_indep(pred, gt)
                r[f"ssim_{k}"] = ssim_indep(pred, gt)
                rec = gt_mean_rectify(pred, gt)
                r[f"psnr_{k}_gtmean"] = psnr_indep(rec, gt)
                r[f"ssim_{k}_gtmean"] = ssim_indep(rec, gt)

            # 진단 1: 고정 ratio 몇 개
            for rt in sweep:
                if rt == a.ratio:
                    r[f"psnr_ratio{rt:g}"] = r["psnr_round"]
                    continue
                y2 = torch.clamp(net.adjust(L_low, rt) * net.unfolding(x)[0], 0, 1)[0]
                p2 = np.clip(np.rint(y2.cpu().numpy().transpose(1, 2, 0) * 255.0),
                             0, 255).astype(np.uint8)
                r[f"psnr_ratio{rt:g}"] = psnr_indep(p2, gt)

            # 진단 2: evaluate.py 설정 (GT 로 ratio 생성) — 캐시에는 넣지 않는다
            xh = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
            R, L = net.unfolding(x)
            ratio_map = net.gt_ratio(L, xh)
            yg = torch.clamp(net.adjust(L, ratio_map) * R, 0, 1)[0].cpu().numpy()
            fg = yg.transpose(1, 2, 0) * 255.0
            for k, pred in [("round", np.clip(np.rint(fg), 0, 255).astype(np.uint8)),
                            ("trunc", np.clip(np.floor(fg), 0, 255).astype(np.uint8))]:
                r[f"psnr_gtratio_{k}"] = psnr_indep(pred, gt)
                r[f"ssim_gtratio_{k}"] = ssim_indep(pred, gt)
            r["gtratio_mean"] = float(ratio_map.mean())
            rows.append(r)
            print(f"  {f:10s} ratio{a.ratio:g} {r['psnr_round']:6.3f} dB | "
                  f"evaluate.py(GT ratio {r['gtratio_mean']:5.2f}) {r['psnr_gtratio_round']:6.3f} dB",
                  flush=True)

    keys = [k for k in rows[0] if k not in ("name",)]
    avg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    sd = {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}

    print(f"\n=== URetinex-Net / LOL-v1 eval15  n={len(rows)}  ({time.time()-t0:.1f}s)")
    print(f"{'설정':28s} {'PSNR':>8s} {'sd':>6s} {'SSIM':>8s} | {'PSNR GT평균':>11s}")
    print(f"{'test.py ratio=%g (반올림)' % a.ratio:28s} {avg['psnr_round']:8.4f} "
          f"{sd['psnr_round']:6.3f} {avg['ssim_round']:8.4f} | {avg['psnr_round_gtmean']:11.4f}")
    print(f"{'test.py ratio=%g (버림)' % a.ratio:28s} {avg['psnr_trunc']:8.4f} "
          f"{sd['psnr_trunc']:6.3f} {avg['ssim_trunc']:8.4f} | {avg['psnr_trunc_gtmean']:11.4f}")
    print(f"{'evaluate.py GT비율 (반올림)':28s} {avg['psnr_gtratio_round']:8.4f} "
          f"{sd['psnr_gtratio_round']:6.3f} {avg['ssim_gtratio_round']:8.4f} |")
    print(f"{'evaluate.py GT비율 (버림)':28s} {avg['psnr_gtratio_trunc']:8.4f} "
          f"{sd['psnr_gtratio_trunc']:6.3f} {avg['ssim_gtratio_trunc']:8.4f} |")
    print("\n고정 ratio 훑기(반올림 PSNR): " +
          "  ".join(f"{rt:g}={avg['psnr_ratio%g' % rt]:.3f}" for rt in sweep))
    print(f"\n캐시: {OUT_DIR}  ({len(os.listdir(OUT_DIR))} 파일)")

    json.dump({"n": len(rows), "ratio": a.ratio, "avg": avg, "sd": sd,
               "loader_maxdiff_vs_author_totensor": d,
               "unfolding_opts": {k: v for k, v in vars(net.uopts).items()
                                  if isinstance(v, (int, float, str, bool))},
               "per_image": rows}, open(OUT_JSON, "w"), indent=1)
    print("json:", OUT_JSON)


if __name__ == "__main__":
    main()
