"""diag_lolv1.py

SID(Sony) 에서 한 진단이 다른 구조·다른 벤치마크에서도 성립하는지 확인한다.
학습 없음, 추론 + 분석만.

벤치마크: LOLv1 eval15 15장 (400x600). 모델: CIDNet(woperc/wperc) + Retinexformer.

측정 항목 (SID 문서와 같은 정의를 그대로 쓴다):
  1. 오차 성분 분해      : 전역이득 / 채널이득 / 구조잔차 의 MSE 몫 + 오라클 보정 PSNR
                           (diag_sid_failure.decompose 를 import 해서 그대로 사용)
  2. 방사 대역별 파워비   : 출력/GT 파워비, 잔차/GT, 대역 상관 rho_out
                           (diag_sid_spectrum 의 RAD_EDGES 11구간을 import)
  3. 대역 라벨           : 밝기만 맞춘 입력(scalar) + 오라클 포인트와이즈(pw) 의 rho_in 대 rho_out
                           판정 규칙은 diag_sid_identifiability 의 '원 입력 기준'
                           (verdict / TH_ZERO=0.20 / TH_GAP=0.05) 을 import 해서 그대로 사용
  4. 최적 저역통과 설명력 : blur_sigma(GT) 대 모델출력 PSNR, 최적 sigma, 대조로 blur(GT) 대 GT
  5. 국소 보정 사다리     : diag_sid_lowfreq 의 gain/off/aff 사다리 + 자유도 대조군(white/perm)
  6. 저주파 성분 몫       : f<0.10 이상적 저역통과 오차맵 D 의 에너지 몫
  7. 장별 산포           : LOLv1(15장) 과 SID(598장, 기존 json 재사용) 의 표준편차

주의 (LOLv1 에서 못 하는 것):
  - SID 는 같은 씬을 여러 번 촬영한 short 프레임이 있어서 잡음 파워를 실측해
    rho 를 잡음보정(-dn)한 '정보 천장' 을 만들 수 있었다. LOLv1 eval15 에는 같은 씬
    반복 촬영이 없다. 그래서 잡음보정 천장 기준(diag_sid_identifiability 4-(a),(b))은
    LOLv1 에서 계산 불가이고, 3번 판정은 원 입력 기준(4-(c)) 하나만 낸다.
  - 15장뿐이라 산포가 크다. 모든 주요 수치에 장별 표준편차를 같이 낸다.

채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep) 만 쓴다. 전부 실측값.
"""
import os, sys, json, math, time
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep
# --- SID 진단들에서 정의를 그대로 가져온다 (정의를 바꾸면 비교가 안 된다) ---
from diag_sid_failure import decompose, p2
from diag_sid_spectrum import RAD_EDGES, LUMA_W, SIGMAS, make_radial_index, gaussian_blur, u8
from diag_sid_identifiability import verdict, TH_ZERO, TH_GAP
from diag_sid_lowfreq import (BLOCKS, BLOCK_NAMES, KINDS, LP_NBAND,
                              block_index, block_sum, block_fit)

LOW = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/low")
HIGH = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/LOLv1/eval15/high")
CIDNET_REPO = (os.environ.get("LLDATA", "data") + "/lowlight_model/HVI-CIDNet")
RF_REPO = "third_party/Retinexformer"
RF_W = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/LOL_v1.pth")
CID_W = {"cidnet_woperc": (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/HVI-CIDNet-LOLv1-woperc"),
         "cidnet_wperc": (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/HVI-CIDNet-LOLv1-wperc")}

OUT_JSON = os.environ.get("LOL_OUT", "numbers/diag_lolv1.json")
OUT_MD = "numbers/DIAG_LOLV1.md"
SID_FAIL_JSON = "numbers/diag_sid_failure.json"
SID_SPEC_JSON = "numbers/diag_sid_spectrum.json"
SID_LF_JSON = "numbers/diag_sid_lowfreq.json"

MODEL_ORDER = ["cidnet_woperc", "cidnet_wperc", "retinexformer"]
MODEL_LABEL = {"cidnet_woperc": "CIDNet (woperc)",
               "cidnet_wperc": "CIDNet (wperc)",
               "retinexformer": "Retinexformer"}
SEED = 20260905


# ---------------- 데이터 ----------------
def build_pairs():
    from PIL import Image
    names = sorted(f for f in os.listdir(HIGH) if f.lower().endswith(".png"))
    out = []
    for f in names:
        lq = np.array(Image.open(os.path.join(LOW, f)).convert("RGB"))
        gt = np.array(Image.open(os.path.join(HIGH, f)).convert("RGB"))
        assert lq.shape == gt.shape, (f, lq.shape, gt.shape)
        out.append((f, lq, gt))
    return out


# ---------------- 모델 로더 (재현 스크립트와 동일 규약) ----------------
def make_cidnet(wdir):
    import torch
    if CIDNET_REPO not in sys.path:
        sys.path.insert(0, CIDNET_REPO)
    from net.CIDNet import CIDNet
    net = CIDNet().cuda()
    net.load_state_dict(torch.load(os.path.join(wdir, "pytorch_model.bin"),
                                   map_location="cpu"))
    net.eval()
    net.trans.gated = True                      # repro_infer.py --gated lol

    def run(lq01):
        x = torch.from_numpy(lq01.transpose(2, 0, 1))[None].cuda()
        return torch.clamp(net(x), 0, 1)[0].double()
    return run


def make_retinexformer():
    import torch, torch.nn.functional as F
    if RF_REPO not in sys.path:
        sys.path.insert(0, RF_REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(RF_W, map_location="cpu")["params"])
    net = net.cuda().eval()

    def run(lq01):
        x = torch.from_numpy(lq01.transpose(2, 0, 1))[None].cuda()
        h, w = x.shape[2], x.shape[3]
        ph, pw = (-h) % 4, (-w) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), "reflect")
        y = net(x)[:, :, :h, :w]
        return torch.clamp(y, 0, 1)[0].double()
    return run


# ---------------- 오라클 포인트와이즈 사상 (identifiability 의 pointwise_map 과 동일) ----------------
def pointwise_map(x, g, torch):
    """채널별로 같은 입력값을 가진 화소를 GT 평균으로 보낸다.
    공간 이웃을 전혀 쓰지 않는 함수 사상이라 구조(위치정보)를 새로 만들지 못한다."""
    out = torch.empty_like(x)
    for c in range(x.shape[0]):
        xf = x[c].reshape(-1)
        gf = g[c].reshape(-1)
        uq, inv = torch.unique(xf, return_inverse=True)
        s = torch.zeros(uq.numel(), dtype=torch.float64, device=x.device)
        cnt = torch.zeros_like(s)
        s.scatter_add_(0, inv, gf)
        cnt.scatter_add_(0, inv, torch.ones_like(gf))
        out[c] = (s / cnt)[inv].reshape(x.shape[1], x.shape[2])
    return out


# ---------------- 본체 ----------------
def main():
    import torch
    import torch.nn.functional as F

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    pairs = build_pairs()
    H, W = pairs[0][1].shape[:2]
    N = H * W
    nrad = len(RAD_EDGES) - 1
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    nfreq = np.bincount(ridx_flat.cpu().numpy(), minlength=nrad)
    lp_mask = (ridx_flat < LP_NBAND).reshape(H, W)
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)
    bidx = {B: block_index(H, W, B, torch, dev) for B in BLOCKS}
    nblk_of = {B: (1 if B <= 0 else ((H + B - 1) // B) * ((W + B - 1) // B)) for B in BLOCKS}
    print(f"LOLv1 eval15: {len(pairs)} images  {H}x{W}  radial bands {nrad}", flush=True)

    def band_stats_luma(p255, g255):
        """p,g: 3xHxW float64 (0-255) -> 휘도 대역 (P_in, P_gt, cross). Parseval 정규화."""
        Lp = (p255 * lw[:, None, None]).sum(0)
        Lg = (g255 * lw[:, None, None]).sum(0)
        Fp = torch.fft.fft2(Lp).reshape(-1)
        Fg = torch.fft.fft2(Lg).reshape(-1)
        pp = torch.zeros(nrad, device=dev, dtype=torch.float64)
        gg = torch.zeros(nrad, device=dev, dtype=torch.float64)
        cx = torch.zeros(nrad, device=dev, dtype=torch.float64)
        pp.scatter_add_(0, ridx_flat, (Fp.real ** 2 + Fp.imag ** 2) / (N * N))
        gg.scatter_add_(0, ridx_flat, (Fg.real ** 2 + Fg.imag ** 2) / (N * N))
        cx.scatter_add_(0, ridx_flat, (Fp.real * Fg.real + Fp.imag * Fg.imag) / (N * N))
        return pp.cpu().numpy(), gg.cpu().numpy(), cx.cpu().numpy()

    def band_stats_ch(p255, g255):
        """채널별 대역 (P_in, P_gt, cross), 각각 3 x nrad."""
        Fp = torch.fft.fft2(p255).reshape(3, -1)
        Fg = torch.fft.fft2(g255).reshape(3, -1)
        pp = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
        gg = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
        cx = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
        idx = ridx_flat[None].expand(3, -1)
        pp.scatter_add_(1, idx, (Fp.real ** 2 + Fp.imag ** 2) / (N * N))
        gg.scatter_add_(1, idx, (Fg.real ** 2 + Fg.imag ** 2) / (N * N))
        cx.scatter_add_(1, idx, (Fp.real * Fg.real + Fp.imag * Fg.imag) / (N * N))
        return pp.cpu().numpy(), gg.cpu().numpy(), cx.cpu().numpy()

    # ---------- 입력 쪽 통계는 모델과 무관하므로 한 번만 ----------
    print("input rho (scalar / pw) ...", flush=True)
    inp = {v: dict(pp=np.zeros(nrad), pg=np.zeros(nrad), cx=np.zeros(nrad))
           for v in ["scalar", "pw"]}
    inp_rows = []
    with torch.inference_mode():
        for f, lq_u8, gt_u8 in pairs:
            x = torch.from_numpy(lq_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            a = float((x * g).sum() / (x * x).sum())
            variants = {"scalar": a * x, "pw": pointwise_map(x, g, torch)}
            r = dict(name=f, in_gain=a)
            for v, pv in variants.items():
                pp, gg, cx = band_stats_luma(pv * 255.0, g * 255.0)
                inp[v]["pp"] += pp; inp[v]["pg"] += gg; inp[v]["cx"] += cx
                r[f"rho_{v}"] = list(cx / np.sqrt(np.maximum(pp * gg, 1e-30)))
                r[f"psnr_{v}"] = psnr_indep(
                    u8(np.clip(pv.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
            inp_rows.append(r)
    rho_in = {v: list(inp[v]["cx"] / np.sqrt(np.maximum(inp[v]["pp"] * inp[v]["pg"], 1e-30)))
              for v in inp}

    # ---------- 모델별 ----------
    results = {}
    for mkey in MODEL_ORDER:
        t0 = time.time()
        print(f"=== {mkey}", flush=True)
        run = make_cidnet(CID_W[mkey]) if mkey.startswith("cidnet") else make_retinexformer()

        rows = []
        band = dict(pp=np.zeros(nrad), pg=np.zeros(nrad), cx=np.zeros(nrad),
                    res=np.zeros(nrad))
        bandc = dict(pp=np.zeros((3, nrad)), pg=np.zeros((3, nrad)), cx=np.zeros((3, nrad)))
        ratio_perimg = np.zeros(nrad)
        y_store, a_store = [], []

        with torch.inference_mode():
            for f, lq_u8, gt_u8 in pairs:
                lq01 = (lq_u8.astype(np.float32) / 255.0)
                y = run(lq01)                                  # 3xHxW float64 [0,1]
                g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                pred_u8 = u8(y.permute(1, 2, 0).cpu().numpy())
                r = dict(name=f)

                # ---------- 1. 성분 분해 (diag_sid_failure 정의: u8 양자화 출력 기준) ----------
                pq = pred_u8.astype(np.float64) / 255.0
                gq = gt_u8.astype(np.float64) / 255.0
                r.update(decompose(pq, gq))
                r["psnr_model"] = psnr_indep(pred_u8, gt_u8)
                r["ssim_model"] = ssim_indep(pred_u8, gt_u8)
                r["psnr_trunc"] = psnr_indep(
                    np.clip(np.floor(y.permute(1, 2, 0).cpu().numpy() * 255.0),
                            0, 255).astype(np.uint8), gt_u8)
                r["psnr_ls_global"] = p2(r["mse_glob"])
                r["psnr_ls_chan"] = p2(r["mse_chan"])
                r["psnr_ls_affine"] = p2(r["mse_affine"])
                gf = u8(np.clip(r["gain_global"] * pq, 0, 1))
                ch = u8(np.clip(np.stack([r["gain_%s" % c] * pq[:, :, i]
                                          for i, c in enumerate("rgb")], -1), 0, 1))
                r["psnr_model_gainfix"] = psnr_indep(gf, gt_u8)
                r["ssim_model_gainfix"] = ssim_indep(gf, gt_u8)
                r["psnr_model_chanfix"] = psnr_indep(ch, gt_u8)
                r["ssim_model_chanfix"] = ssim_indep(ch, gt_u8)

                # ---------- 2/4/5/6 은 float 출력 + 오라클 전역이득 기준 ----------
                a = float((y * g).sum() / (y * y).sum())
                p = a * y
                r["gain_global_float"] = a
                y_store.append(y.to(torch.float16).cpu()); a_store.append(a)
                predgf_u8 = u8(np.clip(p.permute(1, 2, 0).cpu().numpy(), 0, 1))
                r["psnr_gain_float"] = psnr_indep(predgf_u8, gt_u8)

                pp, gg, cx = band_stats_luma(p * 255.0, g * 255.0)
                band["pp"] += pp; band["pg"] += gg; band["cx"] += cx
                band["res"] += (pp + gg - 2 * cx)
                ratio_perimg += pp / np.maximum(gg, 1e-30)
                cpp, cgg, ccx = band_stats_ch(p * 255.0, g * 255.0)
                bandc["pp"] += cpp; bandc["pg"] += cgg; bandc["cx"] += ccx
                r["band_rho"] = list(cx / np.sqrt(np.maximum(pp * gg, 1e-30)))
                r["band_ratio"] = list(pp / np.maximum(gg, 1e-30))

                # ---------- 4. 저역통과 사다리 ----------
                bl_raw, bl_gf, bl_gt = [], [], []
                for s in SIGMAS:
                    bg = gaussian_blur(g[None], s, F, torch)
                    bg_u8 = u8(np.clip(bg[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
                    bl_raw.append(psnr_indep(bg_u8, pred_u8))
                    bl_gf.append(psnr_indep(bg_u8, predgf_u8))
                    bl_gt.append(psnr_indep(bg_u8, gt_u8) if s > 0 else float("inf"))
                r["blur_vs_raw"] = bl_raw
                r["blur_vs_gainfix"] = bl_gf
                r["blur_vs_gt"] = [v if np.isfinite(v) else None for v in bl_gt]

                # ---------- 6. 저주파 오차 몫 ----------
                P, G = p * 255.0, g * 255.0
                E = P - G
                El = (E * lw[:, None, None]).sum(0)
                D = torch.fft.ifft2(torch.fft.fft2(El) * lp_mask).real
                D3 = torch.fft.ifft2(torch.fft.fft2(E) * lp_mask[None]).real
                r["lf_d_en"] = float((D ** 2).mean()); r["lf_e_en"] = float((El ** 2).mean())
                r["lf_d_en_rgb"] = float((D3 ** 2).mean())
                r["lf_e_en_rgb"] = float((E ** 2).mean())
                r["lf_energy_share"] = r["lf_d_en"] / max(r["lf_e_en"], 1e-30)
                r["lf_energy_share_rgb"] = r["lf_d_en_rgb"] / max(r["lf_e_en_rgb"], 1e-30)
                r["lf_mean"] = float(D.mean()); r["lf_meanabs"] = float(D.abs().mean())
                r["lf_std"] = float(D.std(unbiased=False))
                r["lf_const_share"] = r["lf_mean"] ** 2 / max(r["lf_d_en"], 1e-30)

                # ---------- 5. 국소 보정 사다리 (실측) ----------
                for kind, base in [("gain", y), ("off", p), ("aff", y)]:
                    for B, nm in zip(BLOCKS, BLOCK_NAMES):
                        idx, nblk, nby, nbx = bidx[B]
                        out, coef, _ = block_fit(base, g, idx, nblk, kind, torch)
                        r[f"psnr_{kind}_{nm}"] = psnr_indep(
                            u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                        r[f"mse_{kind}_{nm}"] = float(((out - g) ** 2).mean()) * 255.0 ** 2
                rows.append(r)

        # ---------- 5-b. 자유도 대조군 (2 pass, lowfreq 와 동일 절차) ----------
        gen = torch.Generator(device=dev); gen.manual_seed(SEED)
        with torch.inference_mode():
            for i, (f, lq_u8, gt_u8) in enumerate(pairs):
                g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                y = y_store[i].cuda().double()
                p = a_store[i] * y
                e_true = p - g
                sd = e_true.std(dim=(1, 2), unbiased=False, keepdim=True)
                wn = torch.randn(3, H, W, device=dev, dtype=torch.float64, generator=gen)
                wn = wn / torch.clamp(wn.std(dim=(1, 2), unbiased=False, keepdim=True),
                                      min=1e-30) * sd
                j = (i + 1) % len(pairs)
                gj = torch.from_numpy(pairs[j][2].astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                ej = a_store[j] * y_store[j].cuda().double() - gj
                ej = ej / torch.clamp(ej.std(dim=(1, 2), unbiased=False, keepdim=True),
                                      min=1e-30) * sd
                for tag, ee in [("white", wn), ("perm", ej)]:
                    tgt = p - ee
                    for kind, base in [("gain", y), ("off", p), ("aff", y)]:
                        for B, nm in zip(BLOCKS, BLOCK_NAMES):
                            idx, nblk, _, _ = bidx[B]
                            out, _, _ = block_fit(base, tgt, idx, nblk, kind, torch)
                            rows[i][f"mse_{tag}_{kind}_{nm}"] = \
                                float(((out - tgt) ** 2).mean()) * 255.0 ** 2

        # ---------- 집계 ----------
        n = len(rows)

        def m(k):
            return float(np.mean([r[k] for r in rows]))

        def sd_(k):
            return float(np.std([r[k] for r in rows], ddof=1))

        tot = sum(r["mse"] for r in rows)
        o = dict(n=n)
        o["share_global_pct"] = 100.0 * sum(r["mse"] - r["mse_glob"] for r in rows) / tot
        o["share_chan_pct"] = 100.0 * sum(r["mse_glob"] - r["mse_chan"] for r in rows) / tot
        o["share_resid_pct"] = 100.0 * sum(r["mse_chan"] for r in rows) / tot
        o["share_affine_extra_pct"] = 100.0 * sum(r["mse_chan"] - r["mse_affine"]
                                                  for r in rows) / tot
        for tag, key in [("global", lambda r: (r["mse"] - r["mse_glob"]) / r["mse"]),
                         ("chan", lambda r: (r["mse_glob"] - r["mse_chan"]) / r["mse"]),
                         ("resid", lambda r: r["mse_chan"] / r["mse"])]:
            v = [100.0 * key(r) for r in rows]
            o[f"share_{tag}_pct_perimg"] = float(np.mean(v))
            o[f"share_{tag}_pct_perimg_sd"] = float(np.std(v, ddof=1))
        for k in ["psnr_model", "ssim_model", "psnr_trunc", "psnr_ls_global", "psnr_ls_chan",
                  "psnr_ls_affine", "psnr_model_gainfix", "ssim_model_gainfix",
                  "psnr_model_chanfix", "ssim_model_chanfix", "psnr_gain_float",
                  "gain_global", "gain_r", "gain_g", "gain_b", "gain_global_float",
                  "lf_energy_share", "lf_energy_share_rgb", "lf_mean", "lf_meanabs",
                  "lf_std", "lf_const_share"]:
            o[k] = m(k); o[k + "_sd"] = sd_(k)
        o["lf_energy_share_ew"] = float(sum(r["lf_d_en"] for r in rows) /
                                        sum(r["lf_e_en"] for r in rows))
        o["lf_energy_share_rgb_ew"] = float(sum(r["lf_d_en_rgb"] for r in rows) /
                                            sum(r["lf_e_en_rgb"] for r in rows))
        o["lf_std_over_meanabs"] = float(np.mean([r["lf_std"] / max(r["lf_meanabs"], 1e-12)
                                                  for r in rows]))
        # 대역
        rho_out = band["cx"] / np.sqrt(np.maximum(band["pp"] * band["pg"], 1e-30))
        o["band"] = dict(
            p_gt=list(band["pg"] / n), p_out=list(band["pp"] / n),
            p_res=list(band["res"] / n),
            ratio_energy=list(band["pp"] / np.maximum(band["pg"], 1e-30)),
            ratio_perimg_mean=list(ratio_perimg / n),
            res_over_gt=list(band["res"] / np.maximum(band["pg"], 1e-30)),
            rho_out=list(rho_out),
            rho_out_perimg_mean=[float(np.mean([r["band_rho"][b] for r in rows]))
                                 for b in range(nrad)],
            rho_out_perimg_sd=[float(np.std([r["band_rho"][b] for r in rows], ddof=1))
                               for b in range(nrad)],
            ratio_perimg_sd=[float(np.std([r["band_ratio"][b] for r in rows], ddof=1))
                             for b in range(nrad)],
            res_plain_rgb=list((bandc["pp"] + bandc["pg"] - 2 * bandc["cx"]).sum(0) / n),
        )
        # 사다리
        for nm in BLOCK_NAMES:
            for kind in KINDS:
                o[f"psnr_{kind}_{nm}"] = m(f"psnr_{kind}_{nm}")
                o[f"psnr_{kind}_{nm}_sd"] = sd_(f"psnr_{kind}_{nm}")
                for tag in ["", "white_", "perm_"]:
                    key = f"mse_{tag}{kind}_{nm}"
                    o[key] = m(key)
                    o[f"psnrf_{tag}{kind}_{nm}"] = float(
                        np.mean([10 * np.log10(255.0 ** 2 / r[key]) for r in rows]))
                    o[f"psnrf_{tag}{kind}_{nm}_sd"] = float(
                        np.std([10 * np.log10(255.0 ** 2 / r[key]) for r in rows], ddof=1))
        # 저역통과
        for tag in ["raw", "gainfix"]:
            arr = np.array([r["blur_vs_" + tag] for r in rows])
            o["blur_curve_" + tag] = list(arr.mean(0))
            o["blur_curve_" + tag + "_sd"] = list(arr.std(0, ddof=1))
            j = int(np.argmax(arr.mean(0)))
            o["blur_best_sigma_" + tag] = SIGMAS[j]
            o["blur_best_psnr_" + tag] = float(arr.mean(0)[j])
            am = np.argmax(arr, 1)
            o["blur_argmax_hist_" + tag] = {str(SIGMAS[k]): int((am == k).sum())
                                            for k in range(len(SIGMAS))}
        ga = np.array([[np.nan if v is None else v for v in r["blur_vs_gt"]] for r in rows])
        o["blur_curve_gt"] = [None if not np.isfinite(v) else float(v)
                              for v in np.nanmean(ga, 0)]
        results[mkey] = dict(agg=o, rows=rows)
        print(f"  {mkey} done {time.time()-t0:.0f}s  psnr {o['psnr_model']:.4f}", flush=True)
        del run
        torch.cuda.empty_cache()

    # ---------- 3. 라벨 (원 입력 기준) ----------
    labels = {}
    for mkey in MODEL_ORDER:
        ro = np.array(results[mkey]["agg"]["band"]["rho_out"])
        res = np.array(results[mkey]["agg"]["band"]["res_plain_rgb"])
        share = 100.0 * res / res.sum()
        labels[mkey] = dict(
            share_pct=list(share),
            verdict_pw=[verdict(rho_in["pw"][b], ro[b]) for b in range(nrad)],
            verdict_scalar=[verdict(rho_in["scalar"][b], ro[b]) for b in range(nrad)])

    # ---------- 7. SID 산포 (기존 json 재사용, 재추론 없음) ----------
    sid_sd = {}
    if os.path.exists(SID_FAIL_JSON):
        sd = json.load(open(SID_FAIL_JSON))
        srows = sd["rows"]
        keys = ["psnr_model", "ssim_model", "psnr_model_gainfix", "psnr_model_chanfix"]
        sid_sd["overall"] = {k: dict(mean=float(np.mean([r[k] for r in srows])),
                                     sd=float(np.std([r[k] for r in srows], ddof=1)),
                                     n=len(srows)) for k in keys}
        for k, fn in [("psnr_ls_global", lambda r: p2(r["mse_glob"])),
                      ("psnr_ls_chan", lambda r: p2(r["mse_chan"]))]:
            v = [fn(r) for r in srows]
            sid_sd["overall"][k] = dict(mean=float(np.mean(v)),
                                        sd=float(np.std(v, ddof=1)), n=len(v))
        sid_sd["by_exposure"] = {}
        for e in sorted(set(r["exp"] for r in srows), key=lambda s: float(s[:-1])):
            rs = [r for r in srows if r["exp"] == e]
            sid_sd["by_exposure"][e] = {k: dict(mean=float(np.mean([r[k] for r in rs])),
                                                sd=float(np.std([r[k] for r in rs], ddof=1)),
                                                n=len(rs)) for k in keys}
        # 씬 단위 산포 (598장은 50씬에서 나왔으니 장별 표준편차는 독립표본이 아니다)
        from collections import defaultdict
        by_s = defaultdict(list)
        for r in srows:
            by_s[r["scene"]].append(r)
        sid_sd["by_scene"] = {k: dict(
            mean=float(np.mean([np.mean([r[k] for r in v]) for v in by_s.values()])),
            sd=float(np.std([np.mean([r[k] for r in v]) for v in by_s.values()], ddof=1)),
            n=len(by_s)) for k in keys}

    out = dict(
        benchmark="LOLv1 eval15", n_images=len(pairs), H=H, W=W,
        rad_edges=RAD_EDGES, nfreq_per_band=[int(v) for v in nfreq],
        sigmas=SIGMAS, blocks=BLOCKS, block_names=BLOCK_NAMES,
        nblk_of={str(k): v for k, v in nblk_of.items()},
        th_zero=TH_ZERO, th_gap=TH_GAP, lp_nband=LP_NBAND,
        rho_in=rho_in, input_rows=inp_rows,
        input_psnr={v: dict(mean=float(np.mean([r[f"psnr_{v}"] for r in inp_rows])),
                            sd=float(np.std([r[f"psnr_{v}"] for r in inp_rows], ddof=1)))
                    for v in ["scalar", "pw"]},
        models={k: results[k]["agg"] for k in MODEL_ORDER},
        labels=labels, sid_scatter=sid_sd,
        rows={k: results[k]["rows"] for k in MODEL_ORDER},
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote", OUT_JSON)


# ============================ 보고서 ============================
def sid_refs():
    """SID 대조값은 하드코딩하지 않고 SID 진단 json 에서 직접 읽는다.
    (260905 정정: aff 사다리 대조값을 손으로 옮겨 적다 틀렸다. 아래 참조.)"""
    F = json.load(open(SID_FAIL_JSON))["overall"]
    S = json.load(open(SID_SPEC_JSON))["overall"]
    LF = json.load(open(SID_LF_JSON))["overall"]
    sp = S["spec"]
    po = np.array(sp["p_out"]); pg = np.array(sp["p_gt"]); pr = np.array(sp["p_res"])
    rho = (po + pg - pr) / 2.0 / np.sqrt(np.maximum(po * pg, 1e-30))
    rp = np.array(LF["band"]["res_plain"])
    lad = {}
    for kind in KINDS:
        b = LF[f"psnrf_{kind}_global(1x1)"]; t = LF[f"psnrf_{kind}_16"]
        wb = LF[f"psnrf_white_{kind}_global(1x1)"]; wt = LF[f"psnrf_white_{kind}_16"]
        lad[kind] = dict(meas=t - b, white=wt - wb, corr=(t - b) - (wt - wb))
    return dict(F=F, S=S, LF=LF, rho=list(rho),
                ratio=list(sp["ratio_energy"]), res_over_gt=list(sp["res_over_gt"]),
                ladder=lad, lowband_share=float(100.0 * rp[0] / rp.sum()))


def report():
    d = json.load(open(OUT_JSON))
    R = sid_refs()
    SF = R["F"]; SS = R["S"]; SL = R["LF"]
    E = d["rad_edges"]; nb = len(E) - 1
    names = [f"{E[b]:.2f}-{min(E[b+1],0.7072):.2f}" for b in range(nb)]
    BN = d["block_names"]
    L = []; A = L.append
    M = d["models"]
    n = d["n_images"]

    def g(k, key, suf=""):
        return M[k][key + suf]

    A("# LOLv1 eval15 — SID 진단의 다른 구조·다른 벤치마크 재현")
    A("")
    A(f"- 대상: LOLv1 eval15 전량 {n}장 ({d['H']}x{d['W']}). 표본이 아니라 전량이지만 15장뿐이라 산포가 크다.")
    A("  이 문서의 모든 평균에는 장별 표준편차(sd, n=15, ddof=1)를 같이 적는다.")
    A("- 모델 3종. 전부 저자 공개 가중치, 우리 재현 파이프라인과 동일 규약(추론만, 학습 없음).")
    A("    CIDNet (woperc) : repro_infer.py --gated lol, HVI-CIDNet-LOLv1-woperc")
    A("    CIDNet (wperc)  : 같은 코드, -wperc 가중치")
    A("    Retinexformer   : repro_lolv1_retinexformer.py, LOL_v1.pth")
    A("- 정의는 전부 SID 진단 코드에서 import 해서 그대로 썼다. 새로 정의한 것은 없다:")
    A("    성분 분해   diag_sid_failure.decompose / p2")
    A("    방사 11구간 diag_sid_spectrum.RAD_EDGES, 저역통과 sigma 격자 SIGMAS, gaussian_blur")
    A("    판정 규칙   diag_sid_identifiability.verdict (TH_ZERO=%.2f, TH_GAP=%.2f)" %
      (d["th_zero"], d["th_gap"]))
    A("    사다리      diag_sid_lowfreq.BLOCKS/block_fit, 저주파 대역수 LP_NBAND=%d" % d["lp_nband"])
    A("- 채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 사용. 전부 실측값.")
    A("")
    A("## 0. 앵커 대조")
    A("")
    A("```")
    A("모델                 8bit  PSNR      SSIM  |  재현 보고값        판정")
    A("----------------------------------------------------------------------")
    anchors = {"cidnet_woperc": (23.4984, 0.8703, "trunc"),
               "cidnet_wperc": (23.8083, 0.8574, "trunc"),
               "retinexformer": (25.1523, 0.8454, "round")}
    for k in MODEL_ORDER:
        pv, sv, rule = anchors[k]
        got = g(k, "psnr_trunc") if rule == "trunc" else g(k, "psnr_model")
        ok = "통과" if abs(got - pv) < 5e-3 and abs(g(k, "ssim_model") - sv) < 5e-3 else "실패"
        A(f"{MODEL_LABEL[k]:20s} {rule:5s} {got:7.4f}  {g(k,'ssim_model'):8.4f}  |  "
          f"{pv:7.4f} / {sv:.4f}  {ok}")
    A("```")
    A("")
    A("이 문서의 본문은 SID 진단과 같은 규약(반올림 8bit)을 쓴다. 반올림 기준 PSNR 은")
    A(f"CIDNet(woperc) {g('cidnet_woperc','psnr_model'):.4f} / CIDNet(wperc) {g('cidnet_wperc','psnr_model'):.4f} / "
      f"Retinexformer {g('retinexformer','psnr_model'):.4f} 이다.")
    A("")

    # ---------------- 1 ----------------
    A("## 1. 오차 성분 분해")
    A("")
    A("정의(SID 와 동일): 출력 p, GT g. (a) 스칼라 하나 a*p 로 없어지는 몫, (b) 채널별 a_c*p_c 로")
    A("추가로 없어지는 몫(색 편향), (c) 나머지. 계수는 장별 최소제곱해, 클리핑 없음.")
    A("")
    A("```")
    A("모델                 |  (a)전역이득   (b)채널이득   (c)구조잔차  | [참고]채널offset 추가분")
    A("                     |  E가중  장평균  E가중  장평균  E가중  장평균 |")
    A("--------------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['share_global_pct']:6.2f} {o['share_global_pct_perimg']:6.2f} "
          f"{o['share_chan_pct']:6.2f} {o['share_chan_pct_perimg']:6.2f} "
          f"{o['share_resid_pct']:6.2f} {o['share_resid_pct_perimg']:6.2f} | "
          f"{o['share_affine_extra_pct']:8.2f}")
    A(f"{'[대조] SID/Retinexf.':20s} | {SF['share_global_pct']:6.2f} {SF['share_global_pct_perimg']:6.2f} "
      f"{SF['share_chan_pct']:6.2f} {SF['share_chan_pct_perimg']:6.2f} "
      f"{SF['share_resid_pct']:6.2f} {SF['share_resid_pct_perimg']:6.2f} | "
      f"{SF['share_affine_extra_pct']:8.2f}")
    A("```")
    A("")
    A("장별 몫의 표준편차(n=15):")
    A("```")
    A("모델                 | sd (a)전역   sd (b)채널   sd (c)구조")
    A("------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['share_global_pct_perimg_sd']:10.2f} "
          f"{o['share_chan_pct_perimg_sd']:12.2f} {o['share_resid_pct_perimg_sd']:12.2f}")
    A("```")
    A("")
    A("성분을 장별 오라클로 완벽히 제거했을 때 도달하는 PSNR (평균 ± sd, n=15):")
    A("")
    A("```")
    A("모델                 |   무보정        전역이득 오라클      채널이득 오라클   | 전역이득  채널이득")
    A("                     |  PSNR    SSIM   PSNR    SSIM   이득  PSNR    SSIM   이득 |   sd       sd")
    A("---------------------------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['psnr_model']:6.3f} {o['ssim_model']:6.4f} "
          f"{o['psnr_ls_global']:6.3f} {o['ssim_model_gainfix']:6.4f} "
          f"{o['psnr_ls_global']-o['psnr_model']:+5.3f} "
          f"{o['psnr_ls_chan']:6.3f} {o['ssim_model_chanfix']:6.4f} "
          f"{o['psnr_ls_chan']-o['psnr_model']:+5.3f} | {o['psnr_ls_global_sd']:6.3f} "
          f"{o['psnr_ls_chan_sd']:8.3f}")
    A(f"{'[대조] SID/Retinexf.':20s} | {SF['psnr_model']:6.3f} {SF['ssim_model']:6.4f} "
      f"{SF['psnr_ls_global']:6.3f} {SF['ssim_model_gainfix']:6.4f} "
      f"{SF['psnr_ls_global']-SF['psnr_model']:+5.3f} {SF['psnr_ls_chan']:6.3f} "
      f"{SF['ssim_model_chanfix']:6.4f} {SF['psnr_ls_chan']-SF['psnr_model']:+5.3f} |      -        -")
    A("```")
    A("(PSNR 열은 클리핑 없는 최소제곱 잔차 기준. 클리핑+8bit 반올림해서 다시 재면)")
    A("```")
    A("모델                 | 전역이득(clip) 채널이득(clip) | 채널 affine(이득+흑레벨)")
    A("--------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['psnr_model_gainfix']:12.3f} {o['psnr_model_chanfix']:13.3f} | "
          f"{o['psnr_ls_affine']:12.3f} ({o['psnr_ls_affine']-o['psnr_model']:+.3f} dB)")
    A("```")
    A("")
    A("장별 최적 이득 (평균 ± sd):")
    A("```")
    A("모델                 |    전역이득 a        R           G           B")
    A("-------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['gain_global']:.4f}+-{o['gain_global_sd']:.4f}  "
          f"{o['gain_r']:.4f}  {o['gain_g']:.4f}  {o['gain_b']:.4f}")
    A(f"{'[대조] SID/Retinexf.':20s} | {SF['gain_global_mean']:.4f}         "
      f"{SF['gain_rgb_mean'][0]:.4f}  {SF['gain_rgb_mean'][1]:.4f}  {SF['gain_rgb_mean'][2]:.4f}")
    A("```")
    A("")

    # ---------------- 2 ----------------
    A("## 2. 방사 대역별 파워비와 상관 rho_out")
    A("")
    A("대역 정의는 diag_sid_spectrum.py 의 RAD_EDGES 를 그대로 import 했다(0.05 폭 10개 + 코너 1개).")
    A(f"영상이 {d['H']}x{d['W']} 라 SID(512x960) 와 대역당 주파수 개수는 다르다(아래 '주파수수' 열).")
    A("스펙트럼은 전부 장별 오라클 전역이득 a=<y,g>/<y,y> 를 곱한 float 출력(클리핑 없음) 기준,")
    A("휘도 0-255 스케일, P_b = (1/N^2) sum_{k in b} |F(k)|^2 (Parseval).")
    A("")
    A("```")
    A("주파수구간   주파수수 |     출력/GT 파워비 (E가중)      |        상관 rho_out (집계)")
    A("                      |  CIDNet-wo  CIDNet-w  Retinexf |  CIDNet-wo  CIDNet-w  Retinexf |  SID/Retinexf")
    A("--------------------------------------------------------------------------------------------------------")
    sid_rho = R["rho"]
    for b in range(nb):
        A(f"{names[b]:>12s} {d['nfreq_per_band'][b]:8d} | " +
          " ".join(f"{M[k]['band']['ratio_energy'][b]:10.4f}" for k in MODEL_ORDER) +
          " | " + " ".join(f"{M[k]['band']['rho_out'][b]:10.4f}" for k in MODEL_ORDER) +
          f" | {sid_rho[b]:12.4f}")
    A("```")
    A("")
    A("rho_out 의 장별 표준편차(n=15):")
    A("```")
    A("주파수구간 |  CIDNet-wo        CIDNet-w         Retinexf")
    A("------------------------------------------------------------")
    for b in range(nb):
        A(f"{names[b]:>10s} | " + "  ".join(
            f"{M[k]['band']['rho_out_perimg_mean'][b]:.4f}+-{M[k]['band']['rho_out_perimg_sd'][b]:.4f}"
            for k in MODEL_ORDER))
    A("```")
    A("")
    A("잔차/GT 파워비(=1 이면 그 대역을 통째로 틀린 것):")
    A("```")
    A("주파수구간 |  CIDNet-wo  CIDNet-w  Retinexf | SID/Retinexf")
    A("-----------------------------------------------------------")
    sid_res = R["res_over_gt"]
    for b in range(nb):
        A(f"{names[b]:>10s} | " + " ".join(
            f"{M[k]['band']['res_over_gt'][b]:9.4f}" for k in MODEL_ORDER) +
          f" | {sid_res[b]:12.4f}")
    A("```")
    A("")

    # ---------------- 3 ----------------
    A("## 3. 대역 라벨 — 입력에 정보가 있는가, 모델이 다 썼는가")
    A("")
    A("입력을 두 갈래로 잰다(둘 다 공간 이웃을 쓰지 않으므로 위치정보를 새로 만들지 못한다):")
    A("    scalar : 저조도 입력 x 장별 최적 스칼라 이득 = '밝기만 맞춘 입력'")
    A("    pw     : 같은 입력값 화소를 GT 평균으로 보내는 장별 오라클 포인트와이즈 사상")
    A("rho 는 스케일 불변이라 스칼라 이득은 rho 를 바꾸지 못한다(밝기 정합은 PSNR 표시용).")
    A("")
    A("★ 못 하는 것: SID 에서는 같은 씬을 여러 번 찍은 short 프레임이 있어서 잡음 파워를 실측하고")
    A("  rho_dn = rho/sqrt(1-N/P) 로 '정보 천장' 을 만들 수 있었다. LOLv1 eval15 에는 같은 씬")
    A("  반복 촬영이 없다. 잡음 파워를 측정할 자료가 없으므로 잡음보정 천장 기준과 k장 평균 기준은")
    A("  LOLv1 에서 계산할 수 없다. 아래 판정은 diag_sid_identifiability 4-(c) 의 '원 입력 기준'")
    A("  (잡음 보정 안 한 rho_in(pw) k=1 대 rho_out) 이며, SID 문서의 같은 기준 표와 직접 비교된다.")
    A("")
    A(f"판정 규칙(고정 임계, import 한 verdict 그대로): 불가 rho_in < {d['th_zero']}, "
      f"회복가능 rho_in - rho_out > {d['th_gap']}, 그 밖 소진")
    A("")
    for k in MODEL_ORDER:
        A(f"### {MODEL_LABEL[k]}")
        A("")
        A("```")
        A("주파수구간   rho_in    rho_in    rho_out | pw-출력  오차몫% | 판정(pw)  판정(scalar)")
        A("             scalar      pw      (모델)  |                 |")
        A("---------------------------------------------------------------------------------------")
        for b in range(nb):
            A(f"{names[b]:>12s} {d['rho_in']['scalar'][b]:8.4f} {d['rho_in']['pw'][b]:8.4f} "
              f"{M[k]['band']['rho_out'][b]:9.4f} | "
              f"{d['rho_in']['pw'][b]-M[k]['band']['rho_out'][b]:7.4f} "
              f"{d['labels'][k]['share_pct'][b]:8.2f} | {d['labels'][k]['verdict_pw'][b]:8s}  "
              f"{d['labels'][k]['verdict_scalar'][b]}")
        A("```")
        A("")
        A("```")
        A("판정(pw)    대역                                              오차에너지 몫%")
        A("---------------------------------------------------------------------------")
        for tag in ["회복가능", "소진", "불가"]:
            bs = [b for b in range(nb) if d["labels"][k]["verdict_pw"][b] == tag]
            A(f"{tag:10s}  {(', '.join(names[b] for b in bs) if bs else '-'):48s} "
              f"{sum(d['labels'][k]['share_pct'][b] for b in bs):8.2f}")
        A("```")
        A("")
    A("오차몫% 은 그 모델의 전역이득 보정 후 잔차 파워(채널합)의 대역별 몫이다(합 100).")
    A("")
    A("SID 원 입력 기준(DIAG_SID_IDENTIFIABILITY 4-(c))과의 대조:")
    A("```")
    A("벤치마크/모델              회복가능 대역수  소진 대역수  불가 대역수 | 회복가능 오차몫%")
    A("-----------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        v = d["labels"][k]["verdict_pw"]
        A(f"LOLv1 / {MODEL_LABEL[k]:18s} {v.count('회복가능'):10d} {v.count('소진'):12d} "
          f"{v.count('불가'):12d} | "
          f"{sum(d['labels'][k]['share_pct'][b] for b in range(nb) if v[b]=='회복가능'):12.2f}")
    A("SID   / Retinexformer               0           11            0 |         0.00")
    A("```")
    A("(SID 원 입력 기준은 11대역 전부 '소진' 이었다.)")
    A("")

    # ---------------- 4 ----------------
    A("## 4. 최적 저역통과 설명력")
    A("")
    A("blur_sigma(GT) 를 만들어 (a) 모델출력(오라클 전역이득 보정본), (b) 무보정 출력,")
    A("(c) GT 자신 과 각각 PSNR 을 잰다. sigma 격자는 diag_sid_spectrum.SIGMAS 그대로.")
    A("(c) 는 대조군이다: sigma=0 에서 무한대, sigma 가 커지면 단조 감소해야 한다.")
    A("")
    A("```")
    A("sigma |    blur(GT) vs 출력(이득보정)      |  blur(GT) vs GT")
    A("      |  CIDNet-wo  CIDNet-w   Retinexf   |")
    A("--------------------------------------------------------------")
    for i, s in enumerate(d["sigmas"]):
        gtv = M[MODEL_ORDER[0]]["blur_curve_gt"][i]
        A(f"{s:5.2f} | " + " ".join(f"{M[k]['blur_curve_gainfix'][i]:10.3f}" for k in MODEL_ORDER)
          + f"   | {('inf' if gtv is None else f'{gtv:.3f}'):>12s}")
    A("```")
    A("")
    A("```")
    A("모델                 | 모델출력 대 GT   최적 sigma  그때 PSNR   차이(최적blur - 모델)")
    A("--------------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['psnr_gain_float']:12.3f} {o['blur_best_sigma_gainfix']:12.2f} "
          f"{o['blur_best_psnr_gainfix']:11.3f} {o['blur_best_psnr_gainfix']-o['psnr_gain_float']:+16.3f} dB")
    A(f"{'[대조] SID/Retinexf.':20s} | {SS['psnr_model_gainfix']:12.3f} "
      f"{SS['best_sigma_gainfix']:12.2f} {SS['best_psnr_gainfix']:11.3f} "
      f"{SS['best_psnr_gainfix']-SS['psnr_model_gainfix']:+16.3f} dB")
    A("```")
    A("")
    A("무보정 출력 기준(참고):")
    A("```")
    A("모델                 | 모델출력 대 GT   최적 sigma  그때 PSNR   차이")
    A("---------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {o['psnr_model']:12.3f} {o['blur_best_sigma_raw']:12.2f} "
          f"{o['blur_best_psnr_raw']:11.3f} {o['blur_best_psnr_raw']-o['psnr_model']:+8.3f} dB")
    A("```")
    A("")
    A("장별 최적 sigma 의 분포(장마다 argmax, n=15):")
    A("```")
    A("모델                 | " + "  ".join(f"{s:g}" for s in d["sigmas"]))
    A("----------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        h = M[k]["blur_argmax_hist_gainfix"]
        A(f"{MODEL_LABEL[k]:20s} | " + "  ".join(f"{h.get(str(s),0):g}" for s in d["sigmas"]))
    A("```")
    A("")

    # ---------------- 5 ----------------
    A("## 5. 국소 보정 사다리")
    A("")
    A("블록마다 최소제곱 스칼라를 골라 GT 에 맞춘다(계수는 채널 공통, 전역이득 오라클과 같은 단위).")
    A("  gain : a_blk * y          (전역 1x1 = 전역이득 오라클)")
    A("  off  : p + b_blk          (전역이득 보정 출력에 국소 오프셋만)")
    A("  aff  : a_blk * y + b_blk  (블록마다 2 자유도)")
    A("대조군은 같은 절차·같은 블록수로 목표만 바꾼 것이다:")
    A("  white : 목표 = p - (같은 분산 백색잡음)  -> 구조 없는 잔차. 이게 자유도 대조군이고 이 값을 뺀다.")
    A("  perm  : 목표 = p - (다른 장의 잔차, 채널별 분산 정합) -> 구조는 그대로, 이 장과의 정합만 깬 것. 안 뺀다.")
    A("")
    A(f"★ 블록 개수가 SID 와 다르다. LOLv1 은 {d['H']}x{d['W']}, SID 는 512x960 이라 같은 블록 크기에서")
    A("  블록 수가 대략 절반이다(아래 표의 '블록수' 열, 괄호 안이 SID 값). 화소당 자유도는 비슷하지만")
    A("  경계 블록 비율이 달라 완전히 같은 조건은 아니다. 그래서 white 대조군을 반드시 같이 낸다.")
    A("")
    sid_nblk = {"global(1x1)": 1, "256": 8, "128": 32, "64": 120, "32": 480, "16": 1920}
    for kind, ktitle in [("gain", "gain : a_blk * y"), ("off", "off : p + b_blk"),
                         ("aff", "aff : a_blk * y + b_blk")]:
        A(f"### {ktitle}")
        A("")
        for k in MODEL_ORDER:
            o = M[k]
            base = o[f"psnrf_{kind}_global(1x1)"]
            bw = o[f"psnrf_white_{kind}_global(1x1)"]
            bpm = o[f"psnrf_perm_{kind}_global(1x1)"]
            A(f"{MODEL_LABEL[k]} (n={o['n']}):")
            A("```")
            A(f"{'블록':>12s} {'블록수(SID)':>12s} {'PSNR(u8)':>9s} {'PSNR(float)':>11s} "
              f"{'실측이득':>9s} | {'white(자유도)':>13s} {'perm(참고)':>11s} | "
              f"{'자유도보정후':>12s} {'보정후PSNR':>10s}")
            A("-" * 118)
            for nm, B in zip(BN, d["blocks"]):
                nblk = d["nblk_of"][str(B)]
                gr = o[f"psnrf_{kind}_{nm}"] - base
                gw = o[f"psnrf_white_{kind}_{nm}"] - bw
                gp = o[f"psnrf_perm_{kind}_{nm}"] - bpm
                A(f"{nm:>12s} {f'{nblk}({sid_nblk[nm]})':>12s} {o[f'psnr_{kind}_'+nm]:9.3f} "
                  f"{o[f'psnrf_{kind}_'+nm]:11.3f} {gr:9.3f} | {gw:13.3f} {gp:11.3f} | "
                  f"{gr-gw:12.3f} {base+gr-gw:10.3f}")
            A("```")
            A("")
    A("사다리 끝(16화소 블록) 요약 — 자유도 대조군 보정 후 이득:")
    A("(SID 대조 행은 diag_sid_lowfreq.json 에서 직접 읽는다. 손으로 옮겨 적지 않는다 - 10번 정정 참조.)")
    A("")
    A("```")
    A("모델                 |  gain    off     aff  |  white 대조군 이득 (gain/off/aff)")
    A("--------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        vals, ws = [], []
        for kind in KINDS:
            gr = o[f"psnrf_{kind}_16"] - o[f"psnrf_{kind}_global(1x1)"]
            gw = o[f"psnrf_white_{kind}_16"] - o[f"psnrf_white_{kind}_global(1x1)"]
            vals.append(gr - gw); ws.append(gw)
        A(f"{MODEL_LABEL[k]:20s} | {vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:6.3f} |  "
          f"{ws[0]:.3f} / {ws[1]:.3f} / {ws[2]:.3f}")
    A(f"{'[대조] SID/Retinexf.':20s} | " +
      " ".join(f"{R['ladder'][k]['corr']:6.3f}" for k in KINDS) + " |  " +
      " / ".join(f"{R['ladder'][k]['white']:.3f}" for k in KINDS))
    A("```")
    A("")
    A("사다리 각 칸의 장별 표준편차 (gain, PSNR(u8) 기준, n=15):")
    A("```")
    A("모델                 | " + " ".join(f"{nm:>12s}" for nm in BN))
    A("-" * 96)
    for k in MODEL_ORDER:
        A(f"{MODEL_LABEL[k]:20s} | " + " ".join(
            f"{M[k][f'psnr_gain_{nm}']:5.2f}+-{M[k][f'psnr_gain_{nm}_sd']:4.2f}" for nm in BN))
    A("```")
    A("")

    # ---------------- 6 ----------------
    A("## 6. 저주파 성분 몫 (D 에너지몫)")
    A("")
    A(f"출력(전역이득 보정)과 GT 를 이상적 저역통과(최저 {d['lp_nband']} 대역, f<0.10)한 뒤의")
    A("차이맵 D = LP(p) - LP(g). D 의 에너지가 전체 오차 에너지에서 차지하는 몫이다.")
    A("")
    A("```")
    A("모델                 | D몫%(장평균)   D몫%(E가중)  RGB E가중%  | 평균|D|  std(D)  std/평균|D|  장상수 몫%")
    A("--------------------------------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {100*o['lf_energy_share']:12.2f} {100*o['lf_energy_share_ew']:12.2f} "
          f"{100*o['lf_energy_share_rgb_ew']:11.2f} | {o['lf_meanabs']:7.3f} {o['lf_std']:7.3f} "
          f"{o['lf_std_over_meanabs']:11.3f} {100*o['lf_const_share']:11.2f}")
    A(f"{'[대조] SID/Retinexf.':20s} | {100*SL['lf_energy_share']:12.2f} "
      f"{100*SL['lf_energy_share_ew']:12.2f} {100*SL['lf_energy_share_rgb_ew']:11.2f} | "
      f"{SL['lf_meanabs']:7.3f} {SL['lf_std']:7.3f} {SL['lf_std_over_meanabs']:11.3f} "
      f"{100*SL['lf_const_share']:11.2f}")
    A("```")
    A("")
    A("D몫%(장평균) 의 장별 표준편차:")
    A("```")
    A("모델                 |  D몫%(휘도)          RGB D몫%")
    A("------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} | {100*o['lf_energy_share']:.2f}+-{100*o['lf_energy_share_sd']:.2f}"
          f"        {100*o['lf_energy_share_rgb']:.2f}+-{100*o['lf_energy_share_rgb_sd']:.2f}")
    A("```")
    A("")
    A("'장상수 몫' = 장 전체 평균값 하나가 D 의 에너지에서 설명하는 비율. 작으면 전역 통계로는 D 를 못 잡는다.")
    A("")

    # ---------------- 7 ----------------
    A("## 7. 장별 산포 — LOLv1 (15장) 대 SID (598장)")
    A("")
    A("SID 값은 diag_sid_failure.json 의 장별 rows 에서 다시 계산했다(재추론 없음).")
    A("")
    A("```")
    A("벤치마크/모델                 n  | 모델 PSNR      전역이득 보정      채널이득 보정")
    A("                                 | 평균 +- sd     평균 +- sd        평균 +- sd")
    A("-----------------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"LOLv1 / {MODEL_LABEL[k]:20s} {o['n']:3d} | {o['psnr_model']:.3f}+-{o['psnr_model_sd']:.3f}  "
          f"{o['psnr_model_gainfix']:.3f}+-{o['psnr_model_gainfix_sd']:.3f}  "
          f"{o['psnr_model_chanfix']:.3f}+-{o['psnr_model_chanfix_sd']:.3f}")
    s = d["sid_scatter"]
    if s:
        q = s["overall"]
        A(f"SID   / {'Retinexformer (장)':20s} {q['psnr_model']['n']:3d} | "
          f"{q['psnr_model']['mean']:.3f}+-{q['psnr_model']['sd']:.3f}  "
          f"{q['psnr_model_gainfix']['mean']:.3f}+-{q['psnr_model_gainfix']['sd']:.3f}  "
          f"{q['psnr_model_chanfix']['mean']:.3f}+-{q['psnr_model_chanfix']['sd']:.3f}")
        q2 = s["by_scene"]
        A(f"SID   / {'Retinexformer (씬)':20s} {q2['psnr_model']['n']:3d} | "
          f"{q2['psnr_model']['mean']:.3f}+-{q2['psnr_model']['sd']:.3f}  "
          f"{q2['psnr_model_gainfix']['mean']:.3f}+-{q2['psnr_model_gainfix']['sd']:.3f}  "
          f"{q2['psnr_model_chanfix']['mean']:.3f}+-{q2['psnr_model_chanfix']['sd']:.3f}")
    A("```")
    A("")
    A("SID 는 598장이 50씬에서 나왔다. 장별 sd 는 같은 씬의 여러 노출/프레임을 독립표본처럼 세므로")
    A("씬 단위 sd 도 같이 냈다(씬 평균의 표준편차). 표에 실을 때 어느 쪽인지 밝혀야 한다.")
    A("")
    if s:
        A("SID 노출시간별 산포(참고):")
        A("```")
        A("노출     장수 | 모델 PSNR       전역이득 보정      채널이득 보정")
        A("---------------------------------------------------------------------")
        for e, q in s["by_exposure"].items():
            A(f"{e:7s} {q['psnr_model']['n']:4d} | {q['psnr_model']['mean']:.3f}+-{q['psnr_model']['sd']:.3f}  "
              f"{q['psnr_model_gainfix']['mean']:.3f}+-{q['psnr_model_gainfix']['sd']:.3f}  "
              f"{q['psnr_model_chanfix']['mean']:.3f}+-{q['psnr_model_chanfix']['sd']:.3f}")
        A("```")
        A("")
    A("최소제곱 잔차 기준(클리핑 없음) PSNR 의 산포:")
    A("```")
    A("벤치마크/모델                 n  | psnr_ls_global    psnr_ls_chan")
    A("---------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"LOLv1 / {MODEL_LABEL[k]:20s} {o['n']:3d} | {o['psnr_ls_global']:.3f}+-{o['psnr_ls_global_sd']:.3f}   "
          f"{o['psnr_ls_chan']:.3f}+-{o['psnr_ls_chan_sd']:.3f}")
    if s:
        q = s["overall"]
        A(f"SID   / {'Retinexformer (장)':20s} {q['psnr_ls_global']['n']:3d} | "
          f"{q['psnr_ls_global']['mean']:.3f}+-{q['psnr_ls_global']['sd']:.3f}   "
          f"{q['psnr_ls_chan']['mean']:.3f}+-{q['psnr_ls_chan']['sd']:.3f}")
    A("```")
    A("")
    A("15장 기준이라 sd 자체의 불확실성도 크다(sd 의 상대오차 ~1/sqrt(2(n-1)) = 19%).")
    A("이 표의 sd 는 '벤치마크 난이도의 장별 편차' 이지 '방법 간 차이의 표준오차' 가 아니다.")
    A("")

    # ---------------- 산수 대조 ----------------
    A("## 8. 산수 대조 (검증)")
    A("")
    A("```")
    A("항목                                                        값        대조")
    A("--------------------------------------------------------------------------------")
    for k in MODEL_ORDER:
        o = M[k]
        ssum = o["share_global_pct"] + o["share_chan_pct"] + o["share_resid_pct"]
        A(f"{MODEL_LABEL[k]:20s} 성분 합                      {ssum:10.4f} %  100 이어야 함")
    for k in MODEL_ORDER:
        o = M[k]
        ok = "통과" if o["psnr_ls_chan"] > o["psnr_ls_global"] > o["psnr_model"] else "실패"
        A(f"{MODEL_LABEL[k]:20s} 무보정<전역<채널             {ok:>10s}    단조 증가해야 함")
    for k in MODEL_ORDER:
        o = M[k]
        v = max(abs(o["band"]["rho_out"][b]) for b in range(nb))
        A(f"{MODEL_LABEL[k]:20s} |rho_out| 최대               {v:10.6f}    1 이하 (코시-슈바르츠)")
    v = max(max(abs(x) for x in d["rho_in"][t]) for t in ["scalar", "pw"])
    A(f"{'입력':20s} |rho_in| 최대                {v:10.6f}    1 이하")
    for k in MODEL_ORDER:
        o = M[k]
        ok = all(o[f"psnrf_aff_{nm}"] >= o[f"psnr_gain_{nm}"] - 1.0 for nm in BN)
        va = o["psnrf_aff_16"] - o["psnrf_gain_16"]
        A(f"{MODEL_LABEL[k]:20s} aff-gain (블록16, dB)        {va:10.4f}    aff 가 자유도 많아 >=0")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} white 자유도 이득 blk16      "
          f"{o['psnrf_white_gain_16']-o['psnrf_white_gain_global(1x1)']:10.4f} dB  0 에 가까워야 함")
    for k in MODEL_ORDER:
        o = M[k]
        A(f"{MODEL_LABEL[k]:20s} 사다리 전역칸 = 전역이득     "
          f"{o['psnr_gain_global(1x1)']:10.4f}    {o['psnr_gain_float']:.4f} 와 같아야 함")
    A(f"{'저역통과 대조군':20s} blur(GT) vs GT 단조감소     "
      f"{'통과' if all(a >= b for a, b in zip([v for v in M[MODEL_ORDER[0]]['blur_curve_gt'] if v is not None][:-1], [v for v in M[MODEL_ORDER[0]]['blur_curve_gt'] if v is not None][1:])) else '실패':>10s}")
    MEAS = {"cidnet_woperc": "measure_LOLv1_woperc.json",
            "cidnet_wperc": "measure_LOLv1_wperc.json",
            "retinexformer": "measure_LOLv1_retinexformer.json"}
    for k in MODEL_ORDER:
        pth = os.path.join("numbers", MEAS[k])
        if os.path.exists(pth):
            gm = json.load(open(pth))["avg"]["psnr_round_gtmean"]
            A(f"{MODEL_LABEL[k]:20s} GT평균 보정 PSNR (repro)     {gm:10.4f}    "
              f"전역이득 오라클 {M[k]['psnr_model_gainfix']:.4f} 와 0.2 dB 이내 -> "
              f"{'통과' if abs(gm - M[k]['psnr_model_gainfix']) < 0.2 else '확인필요'}")
    A("```")
    A("")
    A("GT평균 보정과 전역이득 오라클은 서로 다른 스칼라를 쓴다. 클리핑 없는 MSE 에서는 최소제곱")
    A("전역이득이 항상 이기고(15/15 장에서 확인), 클리핑+8bit 후 장별 dB 평균에서는 GT평균이 4/15")
    A("장에서 앞선다. 두 값이 0.1 dB 안에서 만나는 건 그 때문이고 모순이 아니다.")
    A("")
    # ---------------- 결론 ----------------
    rf = M["retinexformer"]; cw = M["cidnet_woperc"]; cp = M["cidnet_wperc"]

    def g16(k, kind):
        return (M[k][f"psnrf_{kind}_16"] - M[k][f"psnrf_{kind}_global(1x1)"]) - \
               (M[k][f"psnrf_white_{kind}_16"] - M[k][f"psnrf_white_{kind}_global(1x1)"])

    A("## 9. Sony(SID) 와 같은가 다른가")
    A("")
    A("판정 기준을 먼저 못박는다. LOLv1 은 15장, SID 는 598장/50씬이라 수치 일치를 요구할 수 없다.")
    A("아래 '일치' 는 '같은 정성적 결론이 나온다' 는 뜻이고, 기준을 항목마다 표에 적었다.")
    A("수치 자체는 대부분 다르다. 다르면 다르다고 적었다.")
    A("")
    A("```")
    A("항목                          SID(Retinexformer)   LOLv1 (wo / w / RF)          기준                          판정")
    A("---------------------------------------------------------------------------------------------------------------------")

    def row(label, sidv, lolv, crit, ok):
        A(f"{label:28s}  {sidv:19s}  {lolv:27s}  {crit:28s}  {'일치' if ok else '불일치'}")

    row("1 전역이득 몫(E가중)", f"{SF['share_global_pct']:.2f} %",
        f"{cw['share_global_pct']:.1f} / {cp['share_global_pct']:.1f} / {rf['share_global_pct']:.1f} %",
        "밝기가 지배항이 아님(<40%)",
        all(M[k]["share_global_pct"] < 40 for k in MODEL_ORDER))
    row("1 구조잔차 몫(E가중)", f"{SF['share_resid_pct']:.2f} %",
        f"{cw['share_resid_pct']:.1f} / {cp['share_resid_pct']:.1f} / {rf['share_resid_pct']:.1f} %",
        "구조잔차가 최대 성분(>50%)",
        all(M[k]["share_resid_pct"] > 50 for k in MODEL_ORDER))
    row("1 채널이득 몫(E가중)", f"{SF['share_chan_pct']:.2f} %",
        f"{cw['share_chan_pct']:.2f} / {cp['share_chan_pct']:.2f} / {rf['share_chan_pct']:.2f} %",
        "색편향 몫이 한자리수 %",
        all(M[k]["share_chan_pct"] < 10 for k in MODEL_ORDER))
    row("1 채널이득 오라클 이득", f"{SF['psnr_ls_chan']-SF['psnr_model']:+.3f} dB",
        " / ".join(f"{M[k]['psnr_ls_chan']-M[k]['psnr_model']:+.2f}" for k in MODEL_ORDER) + " dB",
        "출력단 톤정합 상한 < +2.5 dB",
        all(M[k]["psnr_ls_chan"] - M[k]["psnr_model"] < 2.5 for k in MODEL_ORDER))
    row("2 코너대역 출력/GT 파워비", f"{R['ratio'][-1]:.3f}",
        " / ".join(f"{M[k]['band']['ratio_energy'][-1]:.3f}" for k in MODEL_ORDER),
        "고주파 파워가 GT 보다 작음",
        all(M[k]["band"]["ratio_energy"][-1] < 1.0 for k in MODEL_ORDER))
    row("2 rho_out 주파수 단조감소", f"{R['rho'][0]:.3f} -> {R['rho'][-1]:.3f}",
        " / ".join(f"{M[k]['band']['rho_out'][-1]:.3f}" for k in MODEL_ORDER) + " (코너)",
        "저->고 단조감소",
        all(all(M[k]["band"]["rho_out"][b] >= M[k]["band"]["rho_out"][b + 1] - 1e-9
                for b in range(nb - 1)) for k in MODEL_ORDER))
    lo_soj = {k: d["labels"][k]["verdict_pw"].count("소진") for k in MODEL_ORDER}
    row("3 대역 라벨(원 입력 기준)", "11대역 전부 소진",
        " / ".join(f"{lo_soj[k]}/11 소진" for k in MODEL_ORDER),
        "회복가능·불가 대역 0개",
        all(lo_soj[k] == nb for k in MODEL_ORDER))
    row("4 최적 blur 가 더 가까움", f"{SS['best_psnr_gainfix']-SS['psnr_model_gainfix']:+.3f} dB",
        " / ".join(f"{M[k]['blur_best_psnr_gainfix']-M[k]['psnr_gain_float']:+.2f}" for k in MODEL_ORDER) + " dB",
        "양수 (저역통과 GT 가 더 가까움)",
        all(M[k]["blur_best_psnr_gainfix"] - M[k]["psnr_gain_float"] > 0 for k in MODEL_ORDER))
    row("4 최적 sigma", f"{SS['best_sigma_gainfix']:.2f}",
        " / ".join(f"{M[k]['blur_best_sigma_gainfix']:.2f}" for k in MODEL_ORDER),
        "0.4~0.9 구간",
        all(0.4 <= M[k]["blur_best_sigma_gainfix"] <= 0.9 for k in MODEL_ORDER))
    row("5 국소이득 사다리(blk16)", f"{R['ladder']['gain']['corr']:+.3f} dB",
        " / ".join(f"{g16(k,'gain'):+.2f}" for k in MODEL_ORDER) + " dB",
        "자유도 보정 후 +1 dB 이상",
        all(g16(k, "gain") > 1.0 for k in MODEL_ORDER))
    row("5 국소affine 사다리(blk16)", f"{R['ladder']['aff']['corr']:+.3f} dB",
        " / ".join(f"{g16(k,'aff'):+.2f}" for k in MODEL_ORDER) + " dB",
        "자유도 보정 후 +1 dB 이상",
        all(g16(k, "aff") > 1.0 for k in MODEL_ORDER))
    row("5 white 자유도 대조군", f"{R['ladder']['gain']['white']:+.3f} dB",
        " / ".join(f"{M[k]['psnrf_white_gain_16']-M[k]['psnrf_white_gain_global(1x1)']:+.3f}" for k in MODEL_ORDER),
        "무시할 수준 (<0.05 dB)",
        all(abs(M[k]["psnrf_white_gain_16"] -
                M[k]["psnrf_white_gain_global(1x1)"]) < 0.05 for k in MODEL_ORDER))
    row("6 D 에너지몫(RGB,E가중)", f"{100*SL['lf_energy_share_rgb_ew']:.2f} %",
        " / ".join(f"{100*M[k]['lf_energy_share_rgb_ew']:.1f}" for k in MODEL_ORDER) + " %",
        "저주파가 오차의 과반(>50%)",
        all(M[k]["lf_energy_share_rgb_ew"] > 0.5 for k in MODEL_ORDER))
    row("6 최저대역 오차몫", f"{R['lowband_share']:.2f} %",
        " / ".join(f"{d['labels'][k]['share_pct'][0]:.1f}" for k in MODEL_ORDER) + " %",
        "최저대역이 오차의 과반(>50%)",
        all(d["labels"][k]["share_pct"][0] > 50 for k in MODEL_ORDER))
    A("```")
    A("")
    A("### 항목별로 풀어서")
    A("")
    A("- (1) 성분 분해는 불일치다. 이 문서에서 가장 중요한 결과다. SID 에서는 전역이득(밝기) 이 오차의")
    A(f"  27.0% 였고 전역+채널 오라클을 다 써도 +2.02 dB 가 상한이었다. LOLv1 에서는 CIDNet 두 벌이")
    A(f"  각각 {cw['share_global_pct']:.1f}% / {cp['share_global_pct']:.1f}% 로 밝기가 오차의 지배항이고, 전역이득 하나만 오라클로 맞춰도")
    A(f"  +{cw['psnr_ls_global']-cw['psnr_model']:.2f} dB / +{cp['psnr_ls_global']-cp['psnr_model']:.2f} dB 가 들어온다.")
    A(f"  Retinexformer 는 그 중간이다({rf['share_global_pct']:.1f}%, +{rf['psnr_ls_global']-rf['psnr_model']:.2f} dB).")
    A("  즉 'SID 의 실패는 밝기 문제가 아니다' 는 결론은 LOLv1 로 그대로 옮겨가지 않는다.")
    A("  같은 벤치마크 안에서도 구조에 따라 갈린다(CIDNet 75% 대 Retinexformer 41%).")
    A("  다만 이건 CIDNet 의 결함이라기보다 평가 규약 문제에 가깝다 - CIDNet 저자 표는 GT평균 설정을")
    A(f"  쓰고, 그 설정이 여기 전역이득 오라클과 사실상 같은 일을 한다(0번 대조의 GT평균 28.20 대")
    A(f"  전역이득 오라클 {cw['psnr_model_gainfix']:.2f}).")
    A(f"  채널이득(색편향) 몫은 SID {SF['share_chan_pct']:.2f}% 대 LOLv1 1.1~4.1% 로 "
      "한자리수라는 점만 같고 크기는 2~8배 작다.")
    A("- (2) 대역별 파워비/rho 는 방향이 같다. 저주파는 거의 완벽히 맞고 고주파로 갈수록 출력 파워가")
    A("  GT 밑으로 내려가며 rho 가 단조 감소한다. 크기는 LOLv1 이 훨씬 덜 심하다")
    A(f"  (코너 대역 rho: SID {R['rho'][-1]:.3f} 대 LOLv1 "
      f"{cw['band']['rho_out'][-1]:.3f}~{rf['band']['rho_out'][-1]:.3f}).")
    A("  LOLv1 이 SID 만큼 어둡지 않으니 예상되는 방향이다.")
    A("- (3) 원 입력 기준 대역 라벨은 세 모델 전부 11대역 소진으로 SID 와 같다. '회복가능' 대역이")
    A("  하나도 없다는 것도 같다. 다만 SID 에서 쓴 잡음보정 천장 기준과 k장 평균 기준은 LOLv1 에서")
    A("  자료가 없어 못 했으므로, 이 일치는 SID 의 세 기준 중 가장 약한 기준에서의 일치다.")
    A(f"- (4) 저역통과 대조는 방향이 같고 크기가 작다. SID 는 blur_{SS['best_sigma_gainfix']:g}(GT) 가 "
      f"모델출력보다 {SS['best_psnr_gainfix']-SS['psnr_model_gainfix']:+.3f} dB")
    A(f"  가까웠고, LOLv1 은 +{cw['blur_best_psnr_gainfix']-cw['psnr_gain_float']:.2f}~+{rf['blur_best_psnr_gainfix']-rf['psnr_gain_float']:.2f} dB 다. 최적 sigma 는 0.5~0.6 으로 사실상 같다.")
    A("  '모델 출력이 GT 자신보다 저역통과된 GT 에 더 가깝다' 는 진술은 두 벤치마크·세 모델 전부에서 성립한다.")
    A("- (5) 국소 보정 사다리는 일치한다. 자유도 대조군(white)은 두 벤치마크 모두 16화소 블록에서")
    A(f"  {R['ladder']['gain']['white']:.3f} dB 로 무시할 수준이고, gain 열은 LOLv1 이 더 크다")
    A(f"  (gain blk16 보정후 SID {R['ladder']['gain']['corr']:+.3f} 대 LOLv1 "
      f"{g16('cidnet_woperc','gain'):+.2f}/{g16('cidnet_wperc','gain'):+.2f}/{g16('retinexformer','gain'):+.2f};"
      f" aff 는 SID {R['ladder']['aff']['corr']:+.3f} 대 LOLv1 "
      f"{g16('cidnet_woperc','aff'):+.2f}/{g16('cidnet_wperc','aff'):+.2f}/{g16('retinexformer','aff'):+.2f}).")
    A("  블록 수가 SID 의 절반인데도(400x600 대 512x960) 이득이 더 크다는 게 중요하다 - 자유도가")
    A("  아니라 실제 공간적으로 변하는 저주파 오차가 있다는 뜻이다.")
    A(f"- (6) 저주파 성분 몫도 일치한다. SID {100*SL['lf_energy_share_rgb_ew']:.2f}% 대 LOLv1 "
      "67~70%(RGB, E가중). 최저 방사대역이")
    A(f"  오차의 과반을 갖는 것도 같다(SID {R['lowband_share']:.2f}% 대 LOLv1 61~64%).")
    A("")
    A("### 한 줄 요약")
    A("")
    A("- 옮겨가는 것: 오차의 저주파 지배, 공간적으로 변하는 국소 보정 여지, 고주파 감쇠와 rho 단조감소,")
    A("  최적 저역통과 sigma 0.5~0.6, 원 입력 기준 대역 라벨(전부 소진), 자유도 대조군이 무의미하다는 점.")
    A("- 안 옮겨가는 것: 성분 분해의 비율. LOLv1 에서는 밝기(전역이득)가 CIDNet 계열의 지배 오차이고")
    A("  출력단 이득 보정만으로 +3.6~4.5 dB 가 들어온다. SID 에서 나온 '톤 정합류로는 못 메운다'")
    A("  는 진술은 벤치마크·모델 의존적이다.")
    A("- 15장 기준이라 위 수치의 장별 sd 가 크다(모델 PSNR sd 2.9~5.6 dB). 방향성 결론에는 견디지만")
    A("  비율 값을 소수점 아래까지 인용하면 안 된다.")
    A("")
    A("## 10. 정정 기록 (260905)")
    A("")
    A("첫 판(260905 03:33)의 5번 절 '사다리 끝 요약' 표에서 [대조] SID/Retinexformer 행의 aff 열을")
    A("+1.575 dB, white 대조군을 0.012 로 적었다. 둘 다 틀렸다. 착오는 이 문서 쪽이다.")
    A("")
    A("```")
    A("항목                          첫 판(틀림)   정본(diag_sid_lowfreq.json)   근거")
    A("---------------------------------------------------------------------------------------------")
    A(f"SID aff 사다리 blk16 실측      -            {R['ladder']['aff']['meas']:+.4f} dB                 psnrf_aff_16 - psnrf_aff_global(1x1)")
    A(f"SID aff 자유도보정후            +1.575       {R['ladder']['aff']['corr']:+.4f} dB                 실측 - white 실측")
    A(f"SID aff white 대조군            +0.012       {R['ladder']['aff']['white']:+.4f} dB                 psnrf_white_aff_16 - 같은 global")
    A(f"SID gain 자유도보정후           +1.194       {R['ladder']['gain']['corr']:+.4f} dB                 (일치, 손 안 댔다)")
    A(f"SID off 자유도보정후            +1.086       {R['ladder']['off']['corr']:+.4f} dB                 (일치, 손 안 댔다)")
    A("```")
    A("")
    A("원인. 5번 절의 SID 대조 행만 SID 문서에서 눈으로 옮겨 적었고, aff 값을 잘못 집었다.")
    A("정본 DIAG_SID_LOWFREQ.md 는 두 곳(295줄 aff 사다리 표, 435줄 노출별 요약)에서 모두")
    A(f"aff 실측 {R['ladder']['aff']['meas']:.3f} / 보정후 {R['ladder']['aff']['corr']:.3f} 이고, json 재계산도 같다. 1.575 는 이 문서 어디에도 없는 값이다.")
    A("")
    A("정의 확인 (질의 1). diag_lolv1.py 의 사다리 이득은 처음부터 SID 와 같은 정의였다:")
    A("  이득(kind) = psnrf_{kind}_16 - psnrf_{kind}_global(1x1)")
    A("  즉 (a) 각 계열 자신의 global(1x1) 을 앵커로 쓰고 (gain 계열의 global 도, 채널이득 오라클도 아니다)")
    A("     (b) 클리핑 없는 최소제곱 잔차 MSE 를 dB 로 바꾼 장별 값의 평균(psnrf) 을 쓴다.")
    A("  자유도 보정은 같은 식으로 계산한 white 대조군 이득을 뺀 것이다.")
    A("  diag_sid_lowfreq.py 의 report() 도 정확히 같은 두 줄을 쓴다(gr, gw). 정의는 원래 같았다.")
    A("")
    A("영향 범위 (질의 3). 틀린 것은 하드코딩한 SID 대조 숫자 하나뿐이고, LOLv1 의 aff 값")
    A("(자유도보정후 " + " / ".join(f"{g16(k,'aff'):+.3f}" for k in MODEL_ORDER) + " dB) 은 위 정의로 계산된")
    A("실측값이라 바뀌지 않는다. gain 열도 바뀌지 않는다. 이번 판에서는 재발을 막으려고 SID 대조값을")
    A("전부 하드코딩에서 빼고 diag_sid_failure/spectrum/lowfreq json 에서 직접 읽도록 고쳤다")
    A("(sid_refs()). 나머지 대조값(성분 몫, rho, 파워비, blur, D몫)은 이번에 json 과 다시 대조했고")
    A("전부 첫 판과 일치했다 - 틀린 것은 aff 행 하나였다.")
    A("")
    A("```")
    A("이번 판에서 json 직독으로 바뀐 SID 대조값        첫 판      정본")
    A("---------------------------------------------------------------")
    A(f"1 전역/채널/구조 몫(E가중) %                27.03/8.76/64.21  "
      f"{SF['share_global_pct']:.2f}/{SF['share_chan_pct']:.2f}/{SF['share_resid_pct']:.2f}")
    A(f"2 코너대역 rho / 파워비                    0.4885/0.2787     "
      f"{R['rho'][-1]:.4f}/{R['ratio'][-1]:.4f}")
    A(f"4 최적 sigma / 이득 dB                    0.60/+0.882       "
      f"{SS['best_sigma_gainfix']:.2f}/{SS['best_psnr_gainfix']-SS['psnr_model_gainfix']:+.3f}")
    A(f"5 aff 자유도보정후 dB                     +1.575            {R['ladder']['aff']['corr']:+.4f}  <- 정정")
    A(f"6 D 에너지몫(RGB,E가중) %                 64.35             {100*SL['lf_energy_share_rgb_ew']:.2f}")
    A("```")
    A("")

    d["sid_ladder_ref"] = {k: R["ladder"][k] for k in KINDS}
    d["corrections"] = [dict(
        date="260905", where="DIAG_LOLV1.md 5번 절 [대조] SID/Retinexformer 행",
        field="aff 자유도보정후 / white 대조군",
        wrong=[1.575, 0.012],
        correct=[R["ladder"]["aff"]["corr"], R["ladder"]["aff"]["white"]],
        cause="SID 대조값을 문서에서 손으로 옮겨 적다 틀림. 계산 정의는 원래부터 SID 와 동일.",
        scope="하드코딩한 대조 숫자만. LOLv1 실측 aff/gain 값은 영향 없음.",
        fix="SID 대조값을 전부 diag_sid_{failure,spectrum,lowfreq}.json 직독으로 변경 (sid_refs())")]
    with open(OUT_JSON, "w") as fp:
        json.dump(d, fp, indent=1)

    txt = "\n".join(L)
    open(OUT_MD, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--report", "report"):
        report()
    else:
        main()
        report()
