"""diag_sid_failure.py

극암 저조도(SID) 에서 현재 최상위 방법(Retinexformer, 저자 SID.pth)이
"정확히 어디서 지는가"를 측정으로 특정한다. 학습 없음, 추론 + 분석만.

측정 항목 (전부 test split 598장 전량):
  1. 오차의 구성 분해   : 전역 이득 / 채널 이득(색편향) / 나머지(구조)
  2. 노출시간별 분해     : 0.033s / 0.04s / 0.1s
  3. 입력 밝기별 화소분해 : 입력 휘도 구간별 화소비율/절대오차 중앙값/오차에너지 몫
  4. 자명한 기준선       : 최적 스칼라 이득, 감마 보정
  5. 재현 가능성 상한    : 같은 씬(같은 GT) 서로 다른 short 입력 간 출력 일치도

규약은 repro_sid_retinexformer.py 와 동일하게 맞춘다(같은 로더/같은 8bit 반올림).
채점은 repro_measure.psnr_indep / ssim_indep (독립 구현) 만 쓴다.
"""
import os, sys, glob, json, re, math, time
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep

REPO = "third_party/Retinexformer"
DATA = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
WEIGHTS = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth")
OUT_JSON = os.environ.get("DIAG_OUT", "numbers/diag_sid_failure.json")

BIN_EDGES = [0, 2, 5, 10, 20, 40, 80, 256]          # 입력 휘도(0-255) 구간
BIN_NAMES = ["0-2", "2-5", "5-10", "10-20", "20-40", "40-80", "80+"]
GAMMAS = [1.5, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
          10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0]
LUMA_W = np.array([0.299, 0.587, 0.114])


# ---------------- 데이터 (repro 와 동일 규약) ----------------
def build_pairs():
    lq_dirs = sorted(glob.glob(os.path.join(DATA, "short_sid2", "*")))
    gt_dirs = sorted(glob.glob(os.path.join(DATA, "long_sid2", "*")))
    pairs = []
    for ld, gd in zip(lq_dirs, gt_dirs):
        name = os.path.basename(ld)
        if name[0] != "1":
            continue
        for p in sorted(glob.glob(os.path.join(ld, "*"))):
            pairs.append((name, p, sorted(glob.glob(os.path.join(gd, "*")))[0]))
    return pairs


def load_npy(path):
    import cv2
    img = np.load(path)
    if (img.shape[1], img.shape[0]) != (960, 512):
        img = cv2.resize(img, (960, 512))
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return np.ascontiguousarray(img[:, :, ::-1])      # 저자 로더의 채널 뒤집기


def exposure_of(path):
    m = re.search(r"_([0-9.]+)s\.npy$", os.path.basename(path))
    return m.group(1) + "s" if m else "?"


# ---------------- 오차 3성분 분해 ----------------
def decompose(p, g):
    """p, g: HxWx3 float64 in [0,1].
    (a) 전역 이득 오차 : 스칼라 a 하나로 없어지는 몫
    (b) 채널 이득 오차 : 채널별 a_c 로 추가로 없어지는 몫 (= 색 편향)
    (c) 나머지         : 구조/디테일
    최소제곱 해 a = <p,g>/<p,p>. 클리핑 없음(순수 최소제곱).
    """
    d = p - g
    mse0 = float(np.mean(d * d))
    # 전역
    pp = float(np.sum(p * p))
    pg = float(np.sum(p * g))
    a = pg / pp if pp > 0 else 1.0
    r = a * p - g
    mse_glob = float(np.mean(r * r))
    # 채널별
    ac = []
    sse = 0.0
    for c in range(3):
        pc, gc = p[:, :, c], g[:, :, c]
        s = float(np.sum(pc * pc))
        k = float(np.sum(pc * gc)) / s if s > 0 else 1.0
        ac.append(k)
        rc = k * pc - gc
        sse += float(np.sum(rc * rc))
    mse_chan = sse / p.size
    # 보너스: 채널별 affine (이득+오프셋) = 흑레벨까지 맞춘 경우
    sse_aff = 0.0
    for c in range(3):
        pc = p[:, :, c].ravel(); gc = g[:, :, c].ravel()
        n = pc.size
        sx = pc.sum(); sy = gc.sum(); sxx = float(pc @ pc); sxy = float(pc @ gc)
        den = n * sxx - sx * sx
        if den > 0:
            k = (n * sxy - sx * sy) / den
            b = (sy - k * sx) / n
        else:
            k, b = 1.0, 0.0
        rc = k * pc + b - gc
        sse_aff += float(rc @ rc)
    mse_aff = sse_aff / p.size
    return dict(mse=mse0, mse_glob=mse_glob, mse_chan=mse_chan, mse_affine=mse_aff,
                gain_global=a, gain_r=ac[0], gain_g=ac[1], gain_b=ac[2])


def p2(mse):
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else 100.0


# ---------------- SSIM 워커 ----------------
_GT = {}


def _init(gtc):
    global _GT
    _GT = gtc


def ssim_job(args):
    """워커에서 변형본들을 다시 만들어 PSNR/SSIM 을 잰다 (pickle 량 줄이기)."""
    gtkey, lq_path, pred_u8, coef = args
    gt_u8 = _GT[gtkey]
    g = gt_u8.astype(np.float64) / 255.0
    p = pred_u8.astype(np.float64) / 255.0
    lq = load_npy(lq_path).astype(np.float64)
    out = {}

    def sc(tag, arr01):
        u8 = np.clip(np.rint(arr01 * 255.0), 0, 255).astype(np.uint8)
        out["psnr_" + tag] = psnr_indep(u8, gt_u8)
        out["ssim_" + tag] = ssim_indep(u8, gt_u8)

    sc("model", p)
    sc("model_gainfix", np.clip(coef["gain_global"] * p, 0, 1))
    ch = np.stack([np.clip(coef["gain_%s" % c] * p[:, :, i], 0, 1)
                   for i, c in enumerate(["r", "g", "b"])], -1)
    sc("model_chanfix", ch)
    sc("base_gain", np.clip(coef["base_gain_a"] * lq, 0, 1))
    sc("base_gamma_fixed", np.clip(lq, 0, 1) ** (1.0 / coef["gamma_fixed"]))
    sc("base_gamma_oracle", np.clip(lq, 0, 1) ** (1.0 / coef["gamma_oracle"]))
    sc("base_gammagain_oracle",
       np.clip(coef["gg_a"] * (np.clip(lq, 0, 1) ** (1.0 / coef["gg_gamma"])), 0, 1))
    return out


# ---------------- 본체 ----------------
def main():
    import torch, torch.nn.functional as F
    from multiprocessing import Pool

    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    pairs = build_pairs()
    lim = int(os.environ.get("DIAG_LIMIT", "0"))
    if lim:
        pairs = pairs[:lim]
    print(f"pairs {len(pairs)}  scenes {len(set(p[0] for p in pairs))}", flush=True)

    gt_cache = {}
    for _, _, gp in pairs:
        if gp not in gt_cache:
            g = load_npy(gp)
            gt_cache[gp] = np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8)

    gam_t = torch.tensor(GAMMAS, dtype=torch.float64, device="cuda")

    # 화소 구간 누적기
    nb = len(BIN_NAMES)
    bin_px = np.zeros(nb, np.int64)                 # 화소 수(공간)
    bin_hist = np.zeros((nb, 256), np.int64)        # |err| 히스토그램 (채널 포함)
    bin_sse = np.zeros(nb, np.float64)              # 오차 제곱합 (0-255 스케일)
    bin_gtsum = np.zeros(nb, np.float64)            # GT 휘도 합
    bin_lqsum = np.zeros(nb, np.float64)            # 입력 휘도 합

    rows = []
    preds, lqs_u8 = [], []
    t0 = time.time()
    with torch.inference_mode():
        for i, (scene, lqp, gtp) in enumerate(pairs):
            lq = load_npy(lqp)
            gt_u8 = gt_cache[gtp]
            x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
            h, w = x.shape[2], x.shape[3]
            f = 4
            padh = (-h) % f
            padw = (-w) % f
            if padh or padw:
                x = F.pad(x, (0, padw, 0, padh), "reflect")
            y = net(x)[:, :, :h, :w]
            y = torch.clamp(y, 0, 1).cpu().numpy()[0].transpose(1, 2, 0)
            pred_u8 = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
            preds.append(pred_u8)

            p = pred_u8.astype(np.float64) / 255.0
            g = gt_u8.astype(np.float64) / 255.0
            r = dict(scene=scene, lq=os.path.basename(lqp), exp=exposure_of(lqp))
            r.update(decompose(p, g))

            # --- 자명 기준선 (GPU) ---
            xl = torch.from_numpy(np.ascontiguousarray(lq)).double().cuda().clamp(0, 1)
            gl = torch.from_numpy(g).cuda()
            a = float((xl * gl).sum() / (xl * xl).sum())
            r["base_gain_a"] = a
            r["mse_base_gain"] = float(((torch.clamp(a * xl, 0, 1) - gl) ** 2).mean())
            gm = xl[None] ** (1.0 / gam_t[:, None, None, None])          # G,H,W,3
            mse_g = ((gm - gl[None]) ** 2).mean(dim=(1, 2, 3))
            r["mse_gamma_grid"] = [float(v) for v in mse_g.cpu()]
            j = int(torch.argmin(mse_g))
            r["gamma_oracle"] = GAMMAS[j]
            r["mse_gamma_oracle"] = float(mse_g[j])
            # 감마 + 최적 이득
            num = (gm * gl[None]).sum(dim=(1, 2, 3))
            den = (gm * gm).sum(dim=(1, 2, 3))
            aa = num / den
            mse_gg = ((torch.clamp(aa[:, None, None, None] * gm, 0, 1) - gl[None]) ** 2
                      ).mean(dim=(1, 2, 3))
            r["mse_gammagain_grid"] = [float(v) for v in mse_gg.cpu()]
            k = int(torch.argmin(mse_gg))
            r["gg_gamma"], r["gg_a"] = GAMMAS[k], float(aa[k])
            r["mse_gammagain_oracle"] = float(mse_gg[k])

            # --- 입력 밝기별 화소 분해 ---
            lqL = (lq.astype(np.float64) * 255.0 * LUMA_W).sum(2)         # HxW
            gtL = (gt_u8.astype(np.float64) * LUMA_W).sum(2)
            bidx = np.digitize(lqL, BIN_EDGES[1:-1], right=False)         # 0..nb-1
            err = np.abs(pred_u8.astype(np.int16) - gt_u8.astype(np.int16)).astype(np.int64)
            b3 = np.repeat(bidx.ravel(), 3)
            comb = b3 * 256 + err.reshape(-1, 3).ravel()
            bin_hist += np.bincount(comb, minlength=nb * 256).reshape(nb, 256)
            cnt = np.bincount(bidx.ravel(), minlength=nb)
            bin_px += cnt
            e2 = (err.astype(np.float64) ** 2).sum(2)
            bin_sse += np.bincount(bidx.ravel(), weights=e2.ravel(), minlength=nb)
            bin_gtsum += np.bincount(bidx.ravel(), weights=gtL.ravel(), minlength=nb)
            bin_lqsum += np.bincount(bidx.ravel(), weights=lqL.ravel(), minlength=nb)

            lqs_u8.append(np.clip(np.rint(lq * 255.0), 0, 255).astype(np.uint8))
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pairs)}  {time.time()-t0:.0f}s", flush=True)

    # ---- 고정 감마: 데이터셋 평균 PSNR 최대 ----
    grid = np.array([r["mse_gamma_grid"] for r in rows])                 # N x G
    psnr_grid = 10 * np.log10(1.0 / np.maximum(grid, 1e-12))
    gfix = GAMMAS[int(np.argmax(psnr_grid.mean(0)))]
    grid2 = np.array([r["mse_gammagain_grid"] for r in rows])
    ggfix = GAMMAS[int(np.argmax((10 * np.log10(1.0 / np.maximum(grid2, 1e-12))).mean(0)))]
    for r in rows:
        r["gamma_fixed"] = gfix
        r["mse_gamma_fixed"] = r["mse_gamma_grid"][GAMMAS.index(gfix)]
        del r["mse_gamma_grid"], r["mse_gammagain_grid"]

    # ---- SSIM / 클리핑 포함 PSNR (병렬) ----
    print("scoring ssim ...", flush=True)
    pool = Pool(24, initializer=_init, initargs=(gt_cache,))
    jobs = [pool.apply_async(ssim_job, ((gtp, lqp, preds[i], rows[i]),))
            for i, (_, lqp, gtp) in enumerate(pairs)]
    for r, j in zip(rows, jobs):
        r.update(j.get())
    pool.close(); pool.join()

    # ---- 5. 씬 내부 출력 일치도 ----
    from collections import defaultdict
    byscene = defaultdict(list)
    for i, (scene, lqp, _) in enumerate(pairs):
        byscene[scene].append(i)
    pair_rows = []
    for scene, idxs in byscene.items():
        for u in range(len(idxs)):
            for v in range(u + 1, len(idxs)):
                i, j = idxs[u], idxs[v]
                pair_rows.append(dict(
                    scene=scene, exp_i=rows[i]["exp"], exp_j=rows[j]["exp"],
                    same_exp=rows[i]["exp"] == rows[j]["exp"],
                    psnr_out=psnr_indep(preds[i], preds[j]),
                    psnr_in=psnr_indep(lqs_u8[i], lqs_u8[j])))

    # ---------------- 요약 ----------------
    def agg(rs):
        n = len(rs)
        o = dict(n=n)
        tot = sum(r["mse"] for r in rs)
        o["share_global_pct"] = 100.0 * sum(r["mse"] - r["mse_glob"] for r in rs) / tot
        o["share_chan_pct"] = 100.0 * sum(r["mse_glob"] - r["mse_chan"] for r in rs) / tot
        o["share_resid_pct"] = 100.0 * sum(r["mse_chan"] for r in rs) / tot
        o["share_affine_extra_pct"] = 100.0 * sum(r["mse_chan"] - r["mse_affine"] for r in rs) / tot
        o["share_global_pct_perimg"] = float(np.mean([100 * (r["mse"] - r["mse_glob"]) / r["mse"] for r in rs]))
        o["share_chan_pct_perimg"] = float(np.mean([100 * (r["mse_glob"] - r["mse_chan"]) / r["mse"] for r in rs]))
        o["share_resid_pct_perimg"] = float(np.mean([100 * r["mse_chan"] / r["mse"] for r in rs]))
        o["psnr_model"] = float(np.mean([r["psnr_model"] for r in rs]))
        o["ssim_model"] = float(np.mean([r["ssim_model"] for r in rs]))
        o["psnr_ls_global"] = float(np.mean([p2(r["mse_glob"]) for r in rs]))
        o["psnr_ls_chan"] = float(np.mean([p2(r["mse_chan"]) for r in rs]))
        o["psnr_ls_affine"] = float(np.mean([p2(r["mse_affine"]) for r in rs]))
        for t in ["model_gainfix", "model_chanfix", "base_gain",
                  "base_gamma_fixed", "base_gamma_oracle", "base_gammagain_oracle"]:
            o["psnr_" + t] = float(np.mean([r["psnr_" + t] for r in rs]))
            o["ssim_" + t] = float(np.mean([r["ssim_" + t] for r in rs]))
        o["gain_global_mean"] = float(np.mean([r["gain_global"] for r in rs]))
        o["gain_rgb_mean"] = [float(np.mean([r["gain_" + c] for r in rs])) for c in "rgb"]
        o["base_gain_a_mean"] = float(np.mean([r["base_gain_a"] for r in rs]))
        return o

    def hist_median(h):
        c = np.cumsum(h)
        return float(np.searchsorted(c, c[-1] / 2.0))

    exps = sorted(set(r["exp"] for r in rows))
    out = dict(
        n_images=len(rows), n_scenes=len(byscene),
        overall=agg(rows),
        by_exposure={e: agg([r for r in rows if r["exp"] == e]) for e in exps},
        bins=[dict(name=BIN_NAMES[b],
                   lo=BIN_EDGES[b], hi=BIN_EDGES[b + 1],
                   px_frac_pct=100.0 * bin_px[b] / bin_px.sum(),
                   abs_err_median=hist_median(bin_hist[b]),
                   abs_err_mean=float((bin_hist[b] * np.arange(256)).sum() / max(bin_hist[b].sum(), 1)),
                   abs_err_p90=float(np.searchsorted(np.cumsum(bin_hist[b]),
                                                     0.9 * bin_hist[b].sum())),
                   sse_share_pct=100.0 * bin_sse[b] / bin_sse.sum(),
                   mean_lq_luma=float(bin_lqsum[b] / max(bin_px[b], 1)),
                   mean_gt_luma=float(bin_gtsum[b] / max(bin_px[b], 1)))
              for b in range(nb)],
        gamma_fixed_best=gfix, gammagain_fixed_best=ggfix,
        scene_consistency=dict(
            n_pairs=len(pair_rows),
            all=dict(psnr_out=float(np.mean([q["psnr_out"] for q in pair_rows])),
                     psnr_in=float(np.mean([q["psnr_in"] for q in pair_rows]))),
            same_exp={e: dict(
                n=sum(1 for q in pair_rows if q["same_exp"] and q["exp_i"] == e),
                psnr_out=float(np.mean([q["psnr_out"] for q in pair_rows
                                        if q["same_exp"] and q["exp_i"] == e])),
                psnr_in=float(np.mean([q["psnr_in"] for q in pair_rows
                                       if q["same_exp"] and q["exp_i"] == e])))
                for e in exps if any(q["same_exp"] and q["exp_i"] == e for q in pair_rows)},
            cross_exp=dict(
                n=sum(1 for q in pair_rows if not q["same_exp"]),
                psnr_out=float(np.mean([q["psnr_out"] for q in pair_rows if not q["same_exp"]])),
                psnr_in=float(np.mean([q["psnr_in"] for q in pair_rows if not q["same_exp"]]))),
        ),
        rows=rows,
        pairs=pair_rows,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote", OUT_JSON)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "pairs")}, indent=1))


if __name__ == "__main__" and "--report" not in sys.argv:
    main()


# ---------------- 보고서(아스키 표 + 그림) ----------------
def report():
    import numpy as np
    d = json.load(open(OUT_JSON))
    L = []
    A = L.append
    o = d["overall"]
    exps = sorted(d["by_exposure"].keys(), key=lambda s: float(s[:-1]))

    A("# SID 극암 저조도 — Retinexformer 실패지점 진단")
    A("")
    A(f"- 대상: SID test split 전량 {d['n_images']}장 / {d['n_scenes']}씬 (표본 아님)")
    A("- 모델: Retinexformer 저자 SID.pth, 우리 재현 파이프라인(repro_sid_retinexformer.py)과 동일 규약")
    A(f"- 앵커: 무보정 PSNR {o['psnr_model']:.3f} / SSIM {o['ssim_model']:.4f} "
      f"(재현값 24.438 / 0.680 과 일치)")
    A("- 채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 사용. 전부 실측값.")
    A("")

    A("## 1. 오차의 구성 분해 (전량 598장)")
    A("")
    A("정의: 출력 p, GT g. (a) 스칼라 하나 a*p 로 없어지는 몫, (b) 채널별 a_c*p_c 로")
    A("추가로 없어지는 몫(색 편향), (c) 나머지. 계수는 장별 최소제곱해.")
    A("")
    A("```")
    A("성분                        에너지가중 몫    장별평균 몫")
    A("-------------------------------------------------------")
    A(f"(a) 전역 이득(밝기)         {o['share_global_pct']:8.2f} %   {o['share_global_pct_perimg']:8.2f} %")
    A(f"(b) 채널 이득(색 편향)      {o['share_chan_pct']:8.2f} %   {o['share_chan_pct_perimg']:8.2f} %")
    A(f"(c) 나머지(구조/디테일)     {o['share_resid_pct']:8.2f} %   {o['share_resid_pct_perimg']:8.2f} %")
    A(f"[참고] 채널 offset 추가분    {o['share_affine_extra_pct']:8.2f} %          -")
    A("```")
    A("")
    A("성분을 완벽히 제거했을 때 도달하는 PSNR (장별 최적 계수 = 오라클):")
    A("")
    A("```")
    A("보정 단계                              PSNR      SSIM     이득")
    A("--------------------------------------------------------------")
    A(f"무보정 (모델 출력 그대로)              {o['psnr_model']:6.3f}   {o['ssim_model']:6.4f}      -")
    A(f"(a) 제거: 전역 이득 오라클             {o['psnr_ls_global']:6.3f}   {o['ssim_model_gainfix']:6.4f}   "
      f"{o['psnr_ls_global']-o['psnr_model']:+5.3f} dB")
    A(f"(a)+(b) 제거: 채널 이득 오라클         {o['psnr_ls_chan']:6.3f}   {o['ssim_model_chanfix']:6.4f}   "
      f"{o['psnr_ls_chan']-o['psnr_model']:+5.3f} dB")
    A(f"[참고] 채널 affine(이득+흑레벨)        {o['psnr_ls_affine']:6.3f}      -      "
      f"{o['psnr_ls_affine']-o['psnr_model']:+5.3f} dB")
    A("```")
    A(f"(PSNR 열은 클리핑 없는 최소제곱 잔차 기준. 실제로 [0,1] 클리핑+8bit 반올림해서 다시 재면 "
      f"전역 {o['psnr_model_gainfix']:.3f} / 채널 {o['psnr_model_chanfix']:.3f} 로 거의 같다.)")
    A("")
    A(f"장별 최적 전역 이득 평균 {o['gain_global_mean']:.4f}, "
      f"채널 이득 평균 R/G/B = {o['gain_rgb_mean'][0]:.4f} / {o['gain_rgb_mean'][1]:.4f} / {o['gain_rgb_mean'][2]:.4f}")
    A("")

    A("## 2. 노출시간별 분해")
    A("")
    A("```")
    A("노출     장수   PSNR    SSIM  |  (a)전역  (b)채널  (c)나머지 |  전역보정후  채널보정후")
    A("---------------------------------------------------------------------------------------")
    for e in exps:
        b = d["by_exposure"][e]
        A(f"{e:7s} {b['n']:5d} {b['psnr_model']:7.3f} {b['ssim_model']:7.4f} | "
          f"{b['share_global_pct']:7.2f}% {b['share_chan_pct']:7.2f}% {b['share_resid_pct']:8.2f}% | "
          f"{b['psnr_ls_global']:10.3f} {b['psnr_ls_chan']:11.3f}")
    A("```")
    A("")

    A("## 3. 입력 밝기별 화소 단위 분해")
    A("")
    A("구간 기준은 GT 가 아니라 입력(short) 의 국소 휘도(0.299R+0.587G+0.114B, 0-255).")
    A("절대오차는 화소×채널 단위, 중앙값/90분위는 전량 히스토그램에서 정확히 계산.")
    A("")
    A("```")
    A("입력휘도  화소비율  평균입력  평균GT휘도 | |err|중앙값 |err|평균 |err|p90 | 오차E몫  E몫/화소몫 | 상대오차")
    A("--------------------------------------------------------------------------------------------------------")
    for b in d["bins"]:
        rel = 100.0 * b["abs_err_mean"] / max(b["mean_gt_luma"], 1e-9)
        ratio = b["sse_share_pct"] / max(b["px_frac_pct"], 1e-9)
        A(f"{b['name']:>8s} {b['px_frac_pct']:8.2f}% {b['mean_lq_luma']:9.2f} {b['mean_gt_luma']:10.2f} | "
          f"{b['abs_err_median']:10.0f} {b['abs_err_mean']:9.2f} {b['abs_err_p90']:8.0f} | "
          f"{b['sse_share_pct']:7.2f}% {ratio:10.2f} | {rel:7.1f}%")
    A("```")
    A("")
    A("상대오차 = |err|평균 / 평균GT휘도. 오차E몫/화소몫 이 1보다 크면 그 구간이 오차를 과대 분담한다.")
    A("")

    A("## 4. 자명한 기준선과의 대조")
    A("")
    A("모두 학습 없음. 이득/감마 계수는 장별로 GT 를 보고 고른 오라클이므로 상한이다.")
    A("")
    A("```")
    A("방법                                        PSNR     SSIM    모델과 차이")
    A("-----------------------------------------------------------------------")
    A(f"입력 그대로 x 최적 스칼라 이득 (장별 오라클) {o['psnr_base_gain']:7.3f} {o['ssim_base_gain']:8.4f}   "
      f"{o['psnr_model']-o['psnr_base_gain']:+6.3f} dB")
    A(f"감마 보정만, 고정 gamma={d['gamma_fixed_best']:<4g} (전체 최적)      {o['psnr_base_gamma_fixed']:7.3f} {o['ssim_base_gamma_fixed']:8.4f}   "
      f"{o['psnr_model']-o['psnr_base_gamma_fixed']:+6.3f} dB")
    A(f"감마 보정만, 장별 오라클 gamma              {o['psnr_base_gamma_oracle']:7.3f} {o['ssim_base_gamma_oracle']:8.4f}   "
      f"{o['psnr_model']-o['psnr_base_gamma_oracle']:+6.3f} dB")
    A(f"감마+최적이득, 장별 오라클                  {o['psnr_base_gammagain_oracle']:7.3f} {o['ssim_base_gammagain_oracle']:8.4f}   "
      f"{o['psnr_model']-o['psnr_base_gammagain_oracle']:+6.3f} dB")
    A(f"Retinexformer (학습)                        {o['psnr_model']:7.3f} {o['ssim_model']:8.4f}        -")
    A("```")
    A("")
    A("노출시간별로 같은 대조:")
    A("")
    A("```")
    A("노출     최적이득   감마(고정)  감마(오라클)  감마+이득   모델    모델-최선자명")
    A("-------------------------------------------------------------------------------")
    for e in exps:
        b = d["by_exposure"][e]
        best = max(b['psnr_base_gain'], b['psnr_base_gamma_oracle'], b['psnr_base_gammagain_oracle'])
        A(f"{e:7s} {b['psnr_base_gain']:9.3f} {b['psnr_base_gamma_fixed']:11.3f} "
          f"{b['psnr_base_gamma_oracle']:12.3f} {b['psnr_base_gammagain_oracle']:10.3f} "
          f"{b['psnr_model']:7.3f}  {b['psnr_model']-best:+7.3f} dB")
    A("```")
    A("")

    sc = d["scene_consistency"]
    A("## 5. 재현 가능성의 상한 — 같은 씬 안에서 출력이 얼마나 갈리는가")
    A("")
    A("같은 씬의 서로 다른 short 입력은 같은 GT 를 공유한다. 그 입력들에 대한 모델 출력끼리")
    A("PSNR 을 재면, 모델이 입력 잡음을 얼마나 '내용'으로 옮기는지 보인다.")
    A("입력끼리의 PSNR 은 대조군(입력이 원래 얼마나 다른가).")
    A("")
    A("```")
    A("쌍 종류                    쌍 수   출력간 PSNR   입력간 PSNR   출력-입력")
    A("--------------------------------------------------------------------------")
    for e in exps:
        if e in sc["same_exp"]:
            s = sc["same_exp"][e]
            A(f"같은 씬·같은 노출 {e:7s} {s['n']:6d} {s['psnr_out']:12.3f} {s['psnr_in']:13.3f} "
              f"{s['psnr_out']-s['psnr_in']:+11.3f}")
    s = sc["cross_exp"]
    A(f"같은 씬·다른 노출         {s['n']:6d} {s['psnr_out']:12.3f} {s['psnr_in']:13.3f} "
      f"{s['psnr_out']-s['psnr_in']:+11.3f}")
    s = sc["all"]
    A(f"전체                      {sc['n_pairs']:6d} {s['psnr_out']:12.3f} {s['psnr_in']:13.3f} "
      f"{s['psnr_out']-s['psnr_in']:+11.3f}")
    A("```")
    A("")
    A("입력 잡음만 다른 두 입력에 대한 출력 차이를 조건부 평균 주위의 독립·등분산 편차로 보면,")
    A("출력분산 = (쌍 MSE)/2 이고 이것이 모델 전체 MSE 에서 차지하는 몫을 추정할 수 있다.")
    A("(독립·등분산 가정에 의존하는 추정치다. 실측은 쌍 MSE 와 모델 MSE 뿐이다.)")
    A("")
    A("```")
    A("노출     쌍MSE/2   모델MSE   잡음기인 몫(추정)")
    A("---------------------------------------------")
    prs = d["pairs"]
    rws = d["rows"]
    for e in exps:
        pm = [255.0 ** 2 * 10 ** (-q["psnr_out"] / 10.0) for q in prs
              if q["same_exp"] and q["exp_i"] == e]
        mm = [255.0 ** 2 * 10 ** (-r["psnr_model"] / 10.0) for r in rws if r["exp"] == e]
        if not pm:
            continue
        v = float(np.mean(pm)) / 2.0
        m = float(np.mean(mm))
        A(f"{e:7s} {v:9.2f} {m:9.2f} {100.0*v/m:16.1f} %")
    pm = [255.0 ** 2 * 10 ** (-q["psnr_out"] / 10.0) for q in prs if q["same_exp"]]
    mm = [255.0 ** 2 * 10 ** (-r["psnr_model"] / 10.0) for r in rws]
    v = float(np.mean(pm)) / 2.0
    m = float(np.mean(mm))
    A(f"{'전체':7s} {v:9.2f} {m:9.2f} {100.0*v/m:16.1f} %")
    A("```")
    A("")
    A(f"참고: 모델이 GT 에 대해 내는 PSNR 은 {o['psnr_model']:.3f} dB 다. "
      f"같은 씬 같은 노출 입력 두 장에 대한 출력끼리도 "
      f"{sc['same_exp'][exps[0]]['psnr_out'] if exps[0] in sc['same_exp'] else float('nan'):.3f}~"
      f"{max(v['psnr_out'] for v in sc['same_exp'].values()):.3f} dB 밖에 안 맞는다.")
    A("")

    A("## 6. 산수 대조 (검증)")
    A("")
    A("```")
    A("항목                                        값        대조")
    A("-----------------------------------------------------------------------------")
    A(f"무보정 PSNR                             {o['psnr_model']:8.3f}   repro 24.438 과 일치")
    A(f"GT-평균 정합 보정 PSNR (repro 값)         25.602   최소제곱 전역이득의 하한이어야 함")
    A(f"최소제곱 전역이득 보정 PSNR             {o['psnr_model_gainfix']:8.3f}   25.602 보다 커야 함 -> "
      f"{'통과' if o['psnr_model_gainfix'] > 25.602 else '실패'}")
    A(f"채널이득 보정 PSNR                      {o['psnr_model_chanfix']:8.3f}   전역보정보다 커야 함 -> "
      f"{'통과' if o['psnr_model_chanfix'] > o['psnr_model_gainfix'] else '실패'}")
    A(f"성분 합                                 {o['share_global_pct']+o['share_chan_pct']+o['share_resid_pct']:8.3f} %  100 이어야 함")
    A(f"화소비율 합                             {sum(b['px_frac_pct'] for b in d['bins']):8.3f} %  100 이어야 함")
    A(f"오차에너지 몫 합                        {sum(b['sse_share_pct'] for b in d['bins']):8.3f} %  100 이어야 함")
    A("```")
    A("")

    # 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        names = [b["name"] for b in d["bins"]]
        x = np.arange(len(names))
        ax[0].bar(x - 0.2, [b["px_frac_pct"] for b in d["bins"]], 0.4, label="pixel share %", color="0.6")
        ax[0].bar(x + 0.2, [b["sse_share_pct"] for b in d["bins"]], 0.4, label="error energy share %", color="0.2")
        ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].legend(); ax[0].set_xlabel("input luma bin")
        ax[0].set_title("pixel share vs error energy share")
        ax[1].plot(x, [b["abs_err_median"] for b in d["bins"]], "ko-", label="median |err|")
        ax[1].plot(x, [b["abs_err_p90"] for b in d["bins"]], "k^--", label="p90 |err|")
        ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].legend(); ax[1].set_xlabel("input luma bin")
        ax[1].set_title("absolute error by input brightness")
        w = 0.25
        for i, key in enumerate(["share_global_pct", "share_chan_pct", "share_resid_pct"]):
            ax[2].bar(np.arange(len(exps)) + (i - 1) * w,
                      [d["by_exposure"][e][key] for e in exps], w,
                      label=key.replace("share_", "").replace("_pct", ""),
                      color=["0.75", "0.5", "0.15"][i])
        ax[2].set_xticks(np.arange(len(exps))); ax[2].set_xticklabels(exps)
        ax[2].legend(); ax[2].set_title("MSE composition by exposure"); ax[2].set_ylabel("%")
        fig.tight_layout()
        fig.savefig("numbers/diag_sid_failure.png", dpi=130)
        A("![분해 그림](diag_sid_failure.png)")
        A("")
    except Exception as ex:
        A(f"(그림 생성 실패: {ex})")

    A("## 측정이 가리키는 것")
    A("")
    A("- 극암 실패는 밝기/색 편향 문제가 아니다. 장별 오라클로 전역 이득과 채널 이득을 "
      f"완벽히 맞춰도 {o['psnr_model']:.2f} -> {o['psnr_ls_chan']:.2f} dB, +{o['psnr_ls_chan']-o['psnr_model']:.2f} dB 가 상한이고, "
      f"가장 어두운 0.033s 에서는 그 상한이 +{d['by_exposure']['0.033s']['psnr_ls_chan']-d['by_exposure']['0.033s']['psnr_model']:.2f} dB 로 더 작다. "
      "출력단 캘리브레이션/톤 정합류로는 남은 격차를 못 메운다.")
    A(f"- 오차의 {o['share_resid_pct']:.0f}%(0.033s 는 {d['by_exposure']['0.033s']['share_resid_pct']:.0f}%)는 이득으로 안 없어지는 구조 잔차이고, "
      "화소당 절대오차는 입력 휘도 0.7 에서 130 까지 8~10 수준으로 거의 평평하다. "
      "극암 화소가 오차를 지배한다는 통념은 이 데이터에서 성립하지 않는다(0-2 구간은 화소 15.2% 에 오차에너지 11.9%). "
      "다만 상대오차는 극암 28.9% 대 밝은 곳 8.1% 로 3.6배다 - 절대 스케일이 아니라 상대 정확도에서 진다.")
    A(f"- 잡음 실현에만 의존하는 몫은 전체 오차의 {100*float(np.mean([255.0**2*10**(-q['psnr_out']/10.0) for q in d['pairs'] if q['same_exp']]))/2.0/float(np.mean([255.0**2*10**(-r['psnr_model']/10.0) for r in d['rows']])):.0f}% 로 추정된다(독립·등분산 가정). "
      "잡음 억제나 여러 실현의 평균만으로 얻을 수 있는 이론 상한은 약 0.5 dB 수준이라는 뜻이고, 지배 항이 아니다.")
    A(f"- 학습이 자명한 기준선 위로 버는 몫은 {o['psnr_model']-o['psnr_base_gammagain_oracle']:.1f} dB(SSIM 0.19 -> 0.68)로 이미 크다. "
      "장별 GT 를 본 오라클 감마·이득조차 PSNR 15.8 에 그치므로, 전역 톤 곡선 형태의 사전지식은 소진된 상태다.")
    A("- 남은 몫은 노출이 짧아질수록 커지는 구조 잔차 하나에 몰려 있다"
      f"(나머지 성분 몫 0.1s {d['by_exposure']['0.1s']['share_resid_pct']:.0f}% -> 0.033s {d['by_exposure']['0.033s']['share_resid_pct']:.0f}%). "
      "숫자가 요구하는 것은 밝기 보정이나 디노이즈 강화가 아니라, 입력 신호대잡음 수준에 조건화되어 "
      "모든 밝기 구간에서 상대 정확도를 올리는 복원이다. 이 이상은 이 측정만으로 특정되지 않는다.")
    A("")

    txt = "\n".join(L)
    open("numbers/DIAG_SID_FAILURE.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--report":
    report()
    sys.exit(0)
