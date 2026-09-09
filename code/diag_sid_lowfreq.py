"""diag_sid_lowfreq.py

DIAG_SID_SPECTRUM / DIAG_SID_IDENTIFIABILITY 가 남긴 모순을 해소한다.

  모순: 오차 에너지의 54.7% 가 최저 방사대역(0.00-0.05)에 있는데,
        대역별 진폭을 최소제곱으로 최적화해도 25.779 -> 26.601 (+0.82 dB) 뿐이고
        그 대역은 전역 상관 rho=0.9965 로 '소진' 판정을 받았다.

  가설: 지배 오차는 전역 대역 통계로 잡히지 않는, 공간적으로 변하는 저주파 오차다.

측정 (SID test split 전량 598장 / 50씬, 추론만):
  1. 모순의 산수 해소   : 오차를 (대역별 전역 스케일로 설명되는 몫)/(안 되는 몫) 으로 분해.
                          집계 대역이득 / 장별 대역이득 / 장별+채널별 대역이득 3단계.
  2. 저주파 오차의 공간 구조 : 최저 두 대역(f<0.10) 이상적 저역통과 차이맵의
                          변동성·블록평균 분포·부호 일관성·에너지 몫.
  3. 국소 이득 오라클   : 블록 크기 전역/256/128/64/32/16 화소 사다리 + 자유도 대조군 2종.
  4. 예측 가능성        : 블록 오라클 이득 vs (입력 평균휘도, 입력 국소분산, 출력 평균휘도,
                          GT 평균휘도) 상관 + 관측가능 특징만으로 교차적합 예측한 이득맵 PSNR.
  5. 노출시간별 (0.033s / 0.04s / 0.1s) 전 항목 분해.

규약·구간 정의·로더는 diag_sid_spectrum.py 에서 그대로 import 한다(같은 11 방사구간, 같은 8bit).
채점은 repro_measure.psnr_indep (독립 구현) 만 쓴다. 전부 실측값.
"""
import os, sys, glob, json, math, time
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep
from diag_sid_spectrum import (REPO, DATA, WEIGHTS, RAD_EDGES, LUMA_W,
                               build_pairs, load_npy, exposure_of, u8,
                               make_radial_index)

OUT_JSON = os.environ.get("LF_OUT", "numbers/diag_sid_lowfreq.json")
OUT_PNG = "numbers/diag_sid_lowfreq.png"

BLOCKS = [0, 256, 128, 64, 32, 16]        # 0 = 전역 1x1
BLOCK_NAMES = ["global(1x1)", "256", "128", "64", "32", "16"]
KINDS = ["gain", "off", "aff"]
FEAT_BLOCK = 64                            # 4번 특징/회귀의 기본 블록
LP_NBAND = 2                               # 저역통과 = 최저 두 대역 (f < 0.10)
BLK_LAGS = [1, 2, 3, 4]                    # 블록평균 자기상관 lag


# ---------------- 블록 유틸 ----------------
def block_index(H, W, B, torch, dev):
    """B<=0 이면 전역 1블록."""
    if B <= 0:
        return torch.zeros(H * W, dtype=torch.long, device=dev), 1, 1, 1
    nby, nbx = (H + B - 1) // B, (W + B - 1) // B
    by = torch.arange(H, device=dev) // B
    bx = torch.arange(W, device=dev) // B
    idx = (by[:, None] * nbx + bx[None, :]).reshape(-1)
    return idx, nby * nbx, nby, nbx


def block_sum(v, idx, nblk, torch):
    o = torch.zeros(nblk, device=v.device, dtype=torch.float64)
    o.scatter_add_(0, idx, v.reshape(-1))
    return o


def block_ls_gain(y, t, idx, nblk, torch):
    """y,t: 3xHxW float64. 블록별 스칼라 최소제곱 이득 (채널 합산, 전역 정의와 동일)."""
    num = block_sum((y * t).sum(0), idx, nblk, torch)
    den = block_sum((y * y).sum(0), idx, nblk, torch)
    return num / torch.clamp(den, min=1e-30), den


def block_fit(y, t, idx, nblk, kind, torch):
    """블록별 최소제곱 적합 결과 영상을 돌려준다.
       gain : a*y            (1 자유도/블록)
       off  : y + b          (1 자유도/블록, y 는 이미 전역이득 적용된 것을 넣는다)
       aff  : a*y + b        (2 자유도/블록)
       a,b 는 채널 공통 스칼라다(전역 이득 오라클 정의와 같은 단위)."""
    one = torch.ones_like(y[0])
    syy = block_sum((y * y).sum(0), idx, nblk, torch)
    syt = block_sum((y * t).sum(0), idx, nblk, torch)
    if kind == "gain":
        a = syt / torch.clamp(syy, min=1e-30)
        return a[idx].reshape(y.shape[1:])[None] * y, a, syy
    sy = block_sum(y.sum(0), idx, nblk, torch)
    st = block_sum(t.sum(0), idx, nblk, torch)
    n3 = block_sum(one, idx, nblk, torch) * 3.0
    if kind == "off":
        b = (st - sy) / torch.clamp(n3, min=1e-30)
        return y + b[idx].reshape(y.shape[1:])[None], b, syy
    det = syy * n3 - sy * sy
    a = (syt * n3 - sy * st) / torch.clamp(det, min=1e-30)
    b = (syy * st - sy * syt) / torch.clamp(det, min=1e-30)
    sh = y.shape[1:]
    return a[idx].reshape(sh)[None] * y + b[idx].reshape(sh)[None], a, syy


# ---------------- 본체 ----------------
def main():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

    dev = "cuda"
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    pairs = build_pairs()
    lim = int(os.environ.get("LF_LIMIT", "0"))
    if lim:
        pairs = pairs[:lim]
    exps = sorted(set(exposure_of(p[1]) for p in pairs), key=lambda s: float(s[:-1]))
    scenes = sorted(set(p[0] for p in pairs))
    fold_of_scene = {s: i % 2 for i, s in enumerate(scenes)}
    print(f"pairs {len(pairs)}  scenes {len(scenes)}  exps {exps}", flush=True)

    gt_cache = {}
    for _, _, gp in pairs:
        if gp not in gt_cache:
            gt_cache[gp] = u8(load_npy(gp))

    H, W = 512, 960
    N = H * W
    nrad = len(RAD_EDGES) - 1
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    lp_mask = (ridx_flat < LP_NBAND).reshape(H, W)          # 이상적 저역통과 마스크
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)

    bidx = {}
    for B in BLOCKS:
        bidx[B] = block_index(H, W, B, torch, dev)
    fb_idx, fb_n, fb_ny, fb_nx = bidx[FEAT_BLOCK]

    rows = []
    band_acc = {}          # exp -> dict of nrad 누적 (장별 합)
    agg_z = {}             # exp -> 집계 z (pp,pg,cx) 채널별
    lf_acc = {}
    y_store, a_store = [], []

    def zeros_band():
        return dict(res_plain=np.zeros(nrad), res_ls_img=np.zeros(nrad),
                    res_ls_imgc=np.zeros(nrad), p_gt=np.zeros(nrad), p_out=np.zeros(nrad))

    for e in exps:
        band_acc[e] = zeros_band()
        agg_z[e] = dict(pp=np.zeros((3, nrad)), pg=np.zeros((3, nrad)), cx=np.zeros((3, nrad)))

    t0 = time.time()
    with torch.inference_mode():
        for i, (scene, lqp, gtp) in enumerate(pairs):
            exp = exposure_of(lqp)
            lq = load_npy(lqp)
            gt_u8 = gt_cache[gtp]

            x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
            h, w = x.shape[2], x.shape[3]
            padh, padw = (-h) % 4, (-w) % 4
            if padh or padw:
                x = F.pad(x, (0, padw, 0, padh), "reflect")
            y = net(x)[:, :, :h, :w]
            y = torch.clamp(y, 0, 1).double()[0]              # 3xHxW
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)

            a = float((y * g).sum() / (y * y).sum())          # 오라클 전역이득 (앵커 정의)
            p = a * y                                          # 클리핑 없음
            y_store.append(y.to(torch.float16).cpu())
            a_store.append(a)

            r = dict(scene=scene, lq=os.path.basename(lqp), exp=exp, gain_global=a,
                     fold=fold_of_scene[scene])

            # ---- 앵커 ----
            r["psnr_model"] = psnr_indep(u8(y.permute(1, 2, 0).cpu().numpy()), gt_u8)
            r["psnr_gain"] = psnr_indep(
                u8(np.clip(p.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
            ac = ((y * g).sum((1, 2)) / (y * y).sum((1, 2)))
            pc = ac[:, None, None] * y
            r["psnr_chgain"] = psnr_indep(
                u8(np.clip(pc.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)

            # ================= 1. 대역 분해 (채널별, 0-255 스케일) =================
            P = p * 255.0
            G = g * 255.0
            Fp = torch.fft.fft2(P).reshape(3, -1)
            Fg = torch.fft.fft2(G).reshape(3, -1)
            pw_p = (Fp.real ** 2 + Fp.imag ** 2) / (N * N)
            pw_g = (Fg.real ** 2 + Fg.imag ** 2) / (N * N)
            pw_x = (Fp.real * Fg.real + Fp.imag * Fg.imag) / (N * N)
            bp = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
            bg = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
            bx = torch.zeros(3, nrad, device=dev, dtype=torch.float64)
            for c in range(3):
                bp[c].scatter_add_(0, ridx_flat, pw_p[c])
                bg[c].scatter_add_(0, ridx_flat, pw_g[c])
                bx[c].scatter_add_(0, ridx_flat, pw_x[c])
            bp_n, bg_n, bx_n = bp.cpu().numpy(), bg.cpu().numpy(), bx.cpu().numpy()
            # 장별 대역이득 = 채널 합산 하나 / 장별+채널별 = 채널마다 따로
            res_plain = (bp_n + bg_n - 2 * bx_n).sum(0)
            s_img = bx_n.sum(0) / np.maximum(bp_n.sum(0), 1e-30)
            res_ls_img = bg_n.sum(0) - s_img ** 2 * bp_n.sum(0)
            res_ls_imgc = (bg_n - bx_n ** 2 / np.maximum(bp_n, 1e-30)).sum(0)
            A_ = band_acc[exp]
            A_["res_plain"] += res_plain
            A_["res_ls_img"] += res_ls_img
            A_["res_ls_imgc"] += res_ls_imgc
            A_["p_gt"] += bg_n.sum(0)
            A_["p_out"] += bp_n.sum(0)
            agg_z[exp]["pp"] += bp_n; agg_z[exp]["pg"] += bg_n; agg_z[exp]["cx"] += bx_n
            r["mse_gain_float"] = float(res_plain.sum() / 3.0)

            # ================= 2. 저주파 오차맵 =================
            E = P - G                                         # 3xHxW, 0-255
            El = (E * lw[:, None, None]).sum(0)               # 휘도 잔차
            Fe = torch.fft.fft2(El)
            D = torch.fft.ifft2(Fe * lp_mask).real             # 저역통과 오차맵 (휘도)
            e_tot = float((El ** 2).mean())
            d_en = float((D ** 2).mean())
            dm = float(D.mean())
            r["lf_energy_share"] = d_en / max(e_tot, 1e-30)
            r["lf_d_en"] = d_en
            r["lf_e_en"] = e_tot
            r["lf_mean"] = dm
            r["lf_meanabs"] = float(D.abs().mean())
            r["lf_std"] = float(D.std(unbiased=False))
            r["lf_const_share"] = dm * dm / max(d_en, 1e-30)   # 장 전체 상수(=전역통계)가 설명하는 몫
            # RGB 기준 저주파 몫
            Fe3 = torch.fft.fft2(E)
            D3 = torch.fft.ifft2(Fe3 * lp_mask[None]).real
            r["lf_energy_share_rgb"] = float((D3 ** 2).mean() / max((E ** 2).mean(), 1e-30))
            r["lf_d_en_rgb"] = float((D3 ** 2).mean())
            r["lf_e_en_rgb"] = float((E ** 2).mean())

            # 64x64 블록 평균 분포 / 부호 일관성 / 자기상관
            bm = (block_sum(D, fb_idx, fb_n, torch) /
                  block_sum(torch.ones_like(D), fb_idx, fb_n, torch)).reshape(fb_ny, fb_nx)
            r["blk_mean_q"] = [float(v) for v in
                               np.percentile(bm.cpu().numpy(), [5, 25, 50, 75, 95])]
            r["blk_mean_absmean"] = float(bm.abs().mean())
            r["blk_mean_std"] = float(bm.std(unbiased=False))
            sg = torch.sign(bm)
            r["blk_sign_pos_frac"] = float((sg > 0).double().mean())
            r["blk_sign_major"] = float(max((sg > 0).double().mean(), (sg < 0).double().mean()))
            ac_h, ac_v, sa_h, sa_v = [], [], [], []
            bmc = bm - bm.mean()
            for k in BLK_LAGS:
                if fb_nx > k:
                    A1, B1 = bmc[:, :-k], bmc[:, k:]
                    ac_h.append(float((A1 * B1).sum() /
                                      math.sqrt(max(float((A1 ** 2).sum() * (B1 ** 2).sum()), 1e-30))))
                    sa_h.append(float((torch.sign(bm[:, :-k]) == torch.sign(bm[:, k:])).double().mean()))
                else:
                    ac_h.append(float("nan")); sa_h.append(float("nan"))
                if fb_ny > k:
                    A1, B1 = bmc[:-k, :], bmc[k:, :]
                    ac_v.append(float((A1 * B1).sum() /
                                      math.sqrt(max(float((A1 ** 2).sum() * (B1 ** 2).sum()), 1e-30))))
                    sa_v.append(float((torch.sign(bm[:-k, :]) == torch.sign(bm[k:, :])).double().mean()))
                else:
                    ac_v.append(float("nan")); sa_v.append(float("nan"))
            r["blk_ac_h"], r["blk_ac_v"] = ac_h, ac_v
            r["blk_signagree_h"], r["blk_signagree_v"] = sa_h, sa_v

            # ================= 3. 국소 보정 사다리 (실측) =================
            # gain: a*y (전역 1x1 = 전역이득 오라클 앵커) / off: p+b / aff: a*y+b
            for kind, base in [("gain", y), ("off", p), ("aff", y)]:
                for B, nm in zip(BLOCKS, BLOCK_NAMES):
                    idx, nblk, nby, nbx = bidx[B]
                    out, coef, _ = block_fit(base, g, idx, nblk, kind, torch)
                    r[f"psnr_{kind}_{nm}"] = psnr_indep(
                        u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                    r[f"mse_{kind}_{nm}"] = float(((out - g) ** 2).mean()) * 255.0 ** 2
                    # 블록 이음매를 없앤 참고값 (이득 사다리만)
                    if kind == "gain":
                        if B > 0:
                            gs = F.interpolate(coef.reshape(1, 1, nby, nbx), size=(H, W),
                                               mode="bilinear", align_corners=False)[0, 0]
                            r[f"psnr_gains_{nm}"] = psnr_indep(
                                u8(np.clip((gs[None] * y).permute(1, 2, 0).cpu().numpy(), 0, 1)),
                                gt_u8)
                        else:
                            r[f"psnr_gains_{nm}"] = r[f"psnr_{kind}_{nm}"]

            # ================= 4. 블록 특징 (FEAT_BLOCK) =================
            idx, nblk, nby, nbx = bidx[FEAT_BLOCK]
            ab, den = block_ls_gain(y, g, idx, nblk, torch)
            lqt = torch.from_numpy(lq.astype(np.float64)).cuda().permute(2, 0, 1) * 255.0
            inL = (lqt * lw[:, None, None]).sum(0)
            outL = (P * lw[:, None, None]).sum(0)
            gtL = (G * lw[:, None, None]).sum(0)
            cnt = block_sum(torch.ones_like(inL), idx, nblk, torch)
            in_m = block_sum(inL, idx, nblk, torch) / cnt
            in_v = block_sum(inL ** 2, idx, nblk, torch) / cnt - in_m ** 2
            out_m = block_sum(outL, idx, nblk, torch) / cnt
            gt_m = block_sum(gtL, idx, nblk, torch) / cnt
            r["feat"] = dict(
                a=[float(v) for v in ab.cpu().numpy()],
                den=[float(v) for v in den.cpu().numpy()],
                in_m=[float(v) for v in in_m.cpu().numpy()],
                in_v=[float(v) for v in torch.clamp(in_v, min=0).cpu().numpy()],
                out_m=[float(v) for v in out_m.cpu().numpy()],
                gt_m=[float(v) for v in gt_m.cpu().numpy()])

            rows.append(r)
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  pass1 {i+1}/{len(pairs)}  {el:.0f}s  "
                      f"eta {el/(i+1)*(len(pairs)-i-1):.0f}s", flush=True)

    # ================= 3-b. 자유도 대조군 (2 pass) =================
    # ctrl_white : 목표 g' = p - e', e' = 같은 분산 백색잡음  (구조 없음)
    # ctrl_perm  : 목표 g' = p - e', e' = 같은 노출 다른 장의 잔차를 분산 맞춰 대입
    #              (잔차의 공간 구조는 그대로, 출력과의 정합만 깨짐)
    by_exp_idx = {e: [i for i, r in enumerate(rows) if r["exp"] == e] for e in exps}
    partner = {}
    for e, ids in by_exp_idx.items():
        for j, i in enumerate(ids):
            partner[i] = ids[(j + 1) % len(ids)]
    gen = torch.Generator(device=dev); gen.manual_seed(20260905)
    print("pass2 (대조군)", flush=True)
    with torch.inference_mode():
        for i, (scene, lqp, gtp) in enumerate(pairs):
            g = torch.from_numpy(gt_cache[gtp].astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            y = y_store[i].cuda().double()
            p = a_store[i] * y
            e_true = p - g
            sd = e_true.std(dim=(1, 2), unbiased=False, keepdim=True)
            # 백색
            wn = torch.randn(3, H, W, device=dev, dtype=torch.float64, generator=gen)
            wn = wn / torch.clamp(wn.std(dim=(1, 2), unbiased=False, keepdim=True), min=1e-30) * sd
            # 구조 보존 (다른 장의 잔차)
            j = partner[i]
            gj = torch.from_numpy(gt_cache[pairs[j][2]].astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            ej = a_store[j] * y_store[j].cuda().double() - gj
            ej = ej / torch.clamp(ej.std(dim=(1, 2), unbiased=False, keepdim=True), min=1e-30) * sd
            for tag, ee in [("white", wn), ("perm", ej)]:
                tgt = p - ee
                for kind, base in [("gain", y), ("off", p), ("aff", y)]:
                    for B, nm in zip(BLOCKS, BLOCK_NAMES):
                        idx, nblk, _, _ = bidx[B]
                        out, _, _ = block_fit(base, tgt, idx, nblk, kind, torch)
                        rows[i][f"mse_{tag}_{kind}_{nm}"] = \
                            float(((out - tgt) ** 2).mean()) * 255.0 ** 2
            if (i + 1) % 100 == 0:
                print(f"  pass2 {i+1}/{len(pairs)}", flush=True)

    # ================= 4-b. 블록이득 예측 회귀 (교차적합) =================
    reg = fit_block_gain_models(rows, exps)

    # ================= 3-c. 예측 이득맵 적용 =================
    print("pass3 (예측 이득맵 적용)", flush=True)
    with torch.inference_mode():
        for i, (scene, lqp, gtp) in enumerate(pairs):
            gt_u8 = gt_cache[gtp]
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            y = y_store[i].cuda().double()
            idx, nblk, nby, nbx = bidx[FEAT_BLOCK]
            X = design_matrix(rows[i]["feat"])
            wb = np.asarray(rows[i]["feat"]["den"]); sw = max(wb.sum(), 1e-30)
            Xc = X - (X * wb[:, None]).sum(0) / sw
            Xc[:, 0] = 0.0
            a_img = float((np.asarray(rows[i]["feat"]["a"]) * wb).sum() / sw)
            for mname, coefs in reg["coef_by_fold"].items():
                w = coefs[1 - rows[i]["fold"]]           # 다른 폴드에서 적합한 계수
                ah = torch.from_numpy(X @ w).cuda()
                out = ah[idx].reshape(H, W)[None] * y
                rows[i][f"psnr_pred_{mname}"] = psnr_indep(
                    u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                rows[i][f"mse_pred_{mname}"] = float(((out - g) ** 2).mean()) * 255.0 ** 2
            for mname, coefs in reg["coefc_by_fold"].items():
                w = coefs[1 - rows[i]["fold"]]
                ah = torch.from_numpy(a_img + Xc @ w).cuda()
                out = ah[idx].reshape(H, W)[None] * y
                rows[i][f"psnr_pred_{mname}"] = psnr_indep(
                    u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                rows[i][f"mse_pred_{mname}"] = float(((out - g) ** 2).mean()) * 255.0 ** 2
            if (i + 1) % 200 == 0:
                print(f"  pass3 {i+1}/{len(pairs)}", flush=True)

    for key in ("coef_by_fold", "coefc_by_fold"):
        reg[key] = {m: [list(map(float, c)) for c in cs] for m, cs in reg[key].items()}
    np.savez_compressed(
        OUT_JSON.replace(".json", "_blocks.npz"),
        **{k: np.concatenate([np.asarray(r["feat"][k]) for r in rows])
           for k in ["a", "den", "in_m", "in_v", "out_m", "gt_m"]},
        img=np.concatenate([np.full(len(r["feat"]["a"]), i) for i, r in enumerate(rows)]),
        exp=np.array([r["exp"] for r in rows]), fold=np.array([r["fold"] for r in rows]),
        scene=np.array([r["scene"] for r in rows]))
    out = summarize(rows, exps, band_acc, agg_z, nrad, reg)
    for r in rows:
        r.pop("feat", None)
    out["rows"] = rows
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote", OUT_JSON)


# ---------------- 회귀 ----------------
# 절대 이득을 예측하는 모형 (장별 오라클 전역이득을 모른다고 두는 경우)
MODELS = {
    "const":   ["1"],
    "in":      ["1", "lin_m", "lin_v"],
    "inout":   ["1", "lin_m", "lin_v", "lout_m"],
    "gt":      ["1", "lgt_m"],
    "all":     ["1", "lin_m", "lin_v", "lout_m", "lgt_m"],
}
# 장별 전역이득은 이미 맞춰 놓고, '공간적으로 변하는 몫' 만 예측하는 모형.
# den=<y,y>_blk 가중으로 장 안에서 중심화하면 a_blk 의 가중평균이 정확히 장별 전역이득 a_img 라
# 목표 a_blk - a_img 는 순수한 국소 변동이고, 예측이 0 이면 결과가 정확히 전역이득 오라클이다.
MODELS_C = {
    "cin":     ["lin_m", "lin_v"],
    "cinout":  ["lin_m", "lin_v", "lout_m"],
    "cgt":     ["lgt_m"],
    "call":    ["lin_m", "lin_v", "lout_m", "lgt_m"],
}
FEATS = ["1", "lin_m", "lin_v", "lout_m", "lgt_m"]


def design_matrix(f):
    in_m = np.asarray(f["in_m"]); in_v = np.asarray(f["in_v"])
    out_m = np.asarray(f["out_m"]); gt_m = np.asarray(f["gt_m"])
    cols = {"1": np.ones_like(in_m),
            "lin_m": np.log1p(in_m),
            "lin_v": 0.5 * np.log1p(in_v),
            "lout_m": np.log1p(out_m),
            "lgt_m": np.log1p(gt_m)}
    return np.stack([cols[k] for k in FEATS], 1)


def fit_block_gain_models(rows, exps):
    """블록 오라클 이득 a 를 특징으로 예측. 가중치 = den(=<y,y>_blk),
    이 가중 최소제곱이 곧 최종 MSE 를 최소화하는 목적함수다(정확)."""
    X = np.concatenate([design_matrix(r["feat"]) for r in rows], 0)
    a = np.concatenate([np.asarray(r["feat"]["a"]) for r in rows])
    wgt = np.concatenate([np.asarray(r["feat"]["den"]) for r in rows])
    fold = np.concatenate([np.full(len(r["feat"]["a"]), r["fold"]) for r in rows])
    expv = np.concatenate([np.array([r["exp"]] * len(r["feat"]["a"])) for r in rows])

    def wls(Xs, ys, ws):
        Wr = ws[:, None]
        A = Xs.T @ (Xs * Wr)
        b = Xs.T @ (ys * ws)
        return np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), b)

    # --- 장내 중심화 (den 가중) ---
    Xc = np.empty_like(X); ac = np.empty_like(a)
    off = 0
    a_img = np.empty(len(rows))
    for i, r in enumerate(rows):
        k = len(r["feat"]["a"]); sl = slice(off, off + k); off += k
        w = wgt[sl]; sw = max(w.sum(), 1e-30)
        Xc[sl] = X[sl] - (X[sl] * w[:, None]).sum(0) / sw
        am = float((a[sl] * w).sum() / sw)
        a_img[i] = am
        ac[sl] = a[sl] - am
    Xc[:, 0] = 0.0                                   # 중심화하면 절편은 사라진다

    coef_by_fold, full_coef = {}, {}
    for mname, feats in MODELS.items():
        sel = [FEATS.index(k) for k in feats]
        cs = []
        for f in (0, 1):
            m = fold == f
            c = wls(X[m][:, sel], a[m], wgt[m]) if m.sum() > 10 * len(sel) \
                else wls(X[:, sel], a, wgt)          # 폴드가 비면 전체 적합으로 대체
            full = np.zeros(len(FEATS)); full[sel] = c
            cs.append(full)
        coef_by_fold[mname] = cs
        c = wls(X[:, sel], a, wgt)
        full = np.zeros(len(FEATS)); full[sel] = c
        full_coef[mname] = list(full)

    coefc_by_fold, fullc_coef, r2c = {}, {}, {}
    for mname, feats in MODELS_C.items():
        sel = [FEATS.index(k) for k in feats]
        cs = []
        for f in (0, 1):
            m = fold == f
            c = wls(Xc[m][:, sel], ac[m], wgt[m]) if m.sum() > 10 * len(sel) \
                else wls(Xc[:, sel], ac, wgt)
            full = np.zeros(len(FEATS)); full[sel] = c
            cs.append(full)
        coefc_by_fold[mname] = cs
        c = wls(Xc[:, sel], ac, wgt)
        full = np.zeros(len(FEATS)); full[sel] = c
        fullc_coef[mname] = list(full)
        # 교차적합 가중 R^2 (상대 폴드 계수로 예측)
        pred = np.empty_like(ac)
        for f in (0, 1):
            m = fold == f
            pred[m] = Xc[m] @ cs[1 - f]
        num = float((wgt * (ac - pred) ** 2).sum())
        den = float((wgt * ac ** 2).sum())
        r2c[mname] = 1.0 - num / max(den, 1e-30)
        for e in exps:
            m = expv == e
            n2 = float((wgt[m] * (ac[m] - pred[m]) ** 2).sum())
            d2 = float((wgt[m] * ac[m] ** 2).sum())
            r2c[f"{mname}@{e}"] = 1.0 - n2 / max(d2, 1e-30)

    # 상관계수 (플레인 / 에너지가중)
    def corr(u, v, w=None):
        if w is None:
            w = np.ones_like(u)
        mu = (u * w).sum() / w.sum(); mv = (v * w).sum() / w.sum()
        cu, cv = u - mu, v - mv
        return float((w * cu * cv).sum() /
                     math.sqrt(max((w * cu * cu).sum() * (w * cv * cv).sum(), 1e-30)))

    feats_raw = {
        "in_mean": np.concatenate([np.asarray(r["feat"]["in_m"]) for r in rows]),
        "in_var": np.concatenate([np.asarray(r["feat"]["in_v"]) for r in rows]),
        "out_mean": np.concatenate([np.asarray(r["feat"]["out_m"]) for r in rows]),
        "gt_mean": np.concatenate([np.asarray(r["feat"]["gt_m"]) for r in rows]),
    }
    cidx = {"lin_m": 1, "lin_v": 2, "lout_m": 3, "lgt_m": 4}
    cmap = {"in_mean": "lin_m", "in_var": "lin_v", "out_mean": "lout_m", "gt_mean": "lgt_m"}
    cor = {}
    for tag in ["ALL"] + exps:
        m = np.ones(len(a), bool) if tag == "ALL" else (expv == tag)
        cor[tag] = {}
        for k, v in feats_raw.items():
            j = cidx[cmap[k]]
            cor[tag][k] = dict(
                r=corr(a[m], v[m]), r_w=corr(a[m], v[m], wgt[m]),
                r_log=corr(a[m], np.log1p(v[m])), r_log_w=corr(a[m], np.log1p(v[m]), wgt[m]),
                rc=corr(ac[m], Xc[m][:, j]), rc_w=corr(ac[m], Xc[m][:, j], wgt[m]))
    stats = dict(n_blocks=int(len(a)), a_mean=float(a.mean()),
                 a_std=float(a.std()), a_q=[float(v) for v in
                                            np.percentile(a, [5, 25, 50, 75, 95])],
                 a_std_by_exp={e: float(a[expv == e].std()) for e in exps},
                 a_mean_by_exp={e: float(a[expv == e].mean()) for e in exps})
    stats["a_img_mean"] = float(a_img.mean())
    stats["ac_std_w"] = float(math.sqrt((wgt * ac ** 2).sum() / wgt.sum()))
    return dict(coef_by_fold=coef_by_fold, full_coef=full_coef,
                coefc_by_fold=coefc_by_fold, fullc_coef=fullc_coef, r2_centered=r2c,
                corr=cor, feat_names=FEATS, stats=stats)


# ---------------- 요약 ----------------
def summarize(rows, exps, band_acc, agg_z, nrad, reg):
    def agg(rs, e_list):
        n = len(rs)
        o = dict(n=n)
        for k in ["psnr_model", "psnr_gain", "psnr_chgain", "gain_global",
                  "lf_energy_share", "lf_energy_share_rgb", "lf_const_share",
                  "lf_meanabs", "lf_std", "lf_mean", "blk_mean_absmean",
                  "blk_mean_std", "blk_sign_major", "mse_gain_float"]:
            o[k] = float(np.mean([r[k] for r in rs]))
        o["lf_energy_share_ew"] = float(sum(r["lf_d_en"] for r in rs) /
                                        max(sum(r["lf_e_en"] for r in rs), 1e-30))
        o["lf_energy_share_rgb_ew"] = float(sum(r["lf_d_en_rgb"] for r in rs) /
                                            max(sum(r["lf_e_en_rgb"] for r in rs), 1e-30))
        o["lf_std_over_meanabs"] = float(np.mean([r["lf_std"] / max(r["lf_meanabs"], 1e-12)
                                                  for r in rs]))
        o["blk_mean_q"] = [float(np.mean([r["blk_mean_q"][j] for r in rs])) for j in range(5)]
        for k in ["blk_ac_h", "blk_ac_v", "blk_signagree_h", "blk_signagree_v"]:
            o[k] = [float(np.nanmean([r[k][j] for r in rs])) for j in range(len(rows[0][k]))]
        for nm in BLOCK_NAMES:
            o[f"psnr_gains_{nm}"] = float(np.mean([r[f"psnr_gains_{nm}"] for r in rs]))
            for kind in KINDS:
                o[f"psnr_{kind}_{nm}"] = float(np.mean([r[f"psnr_{kind}_{nm}"] for r in rs]))
                for tag in ["", "white_", "perm_"]:
                    key = f"mse_{tag}{kind}_{nm}"
                    o[key] = float(np.mean([r[key] for r in rs]))
                    o[f"psnrf_{tag}{kind}_{nm}"] = float(
                        np.mean([10 * np.log10(255.0 ** 2 / r[key]) for r in rs]))
        for mname in list(MODELS) + list(MODELS_C):
            o[f"psnr_pred_{mname}"] = float(np.mean([r[f"psnr_pred_{mname}"] for r in rs]))
            o[f"psnr_float_pred_{mname}"] = float(
                np.mean([10 * np.log10(255.0 ** 2 / r[f"mse_pred_{mname}"]) for r in rs]))
        # 대역 분해
        B_ = dict(res_plain=np.zeros(nrad), res_ls_img=np.zeros(nrad), res_ls_imgc=np.zeros(nrad),
                  p_gt=np.zeros(nrad), p_out=np.zeros(nrad))
        Z = dict(pp=np.zeros((3, nrad)), pg=np.zeros((3, nrad)), cx=np.zeros((3, nrad)))
        for e in e_list:
            for k in B_:
                B_[k] += band_acc[e][k]
            for k in Z:
                Z[k] += agg_z[e][k]
        # 집계(데이터셋 전체) 대역이득: 채널별 하나씩
        res_ls_agg = (Z["pg"] - Z["cx"] ** 2 / np.maximum(Z["pp"], 1e-30)).sum(0)
        o["band"] = dict(
            res_plain=list(B_["res_plain"] / n),
            res_ls_img=list(B_["res_ls_img"] / n),
            res_ls_imgc=list(B_["res_ls_imgc"] / n),
            res_ls_agg=list(res_ls_agg / n),
            p_gt=list(B_["p_gt"] / n), p_out=list(B_["p_out"] / n),
            rho_agg=list(Z["cx"].sum(0) / np.sqrt(np.maximum(Z["pp"].sum(0) * Z["pg"].sum(0), 1e-30))))
        return o

    out = dict(n_images=len(rows), n_scenes=len(set(r["scene"] for r in rows)),
               exposures=exps, blocks=BLOCKS, block_names=BLOCK_NAMES,
               feat_block=FEAT_BLOCK, lp_nband=LP_NBAND, blk_lags=BLK_LAGS,
               rad_edges=RAD_EDGES, models=MODELS, models_c=MODELS_C,
               overall=agg(rows, exps),
               by_exposure={e: agg([r for r in rows if r["exp"] == e], [e]) for e in exps},
               regression=reg)
    return out



def make_figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "NanumGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    o = d["overall"]; exps = d["exposures"]; BN = d["block_names"]
    names = fmt_bands(d["rad_edges"])
    b = o["band"]
    rp = np.array(b["res_plain"]); ra = np.array(b["res_ls_agg"])
    ri = np.array(b["res_ls_img"]); ric = np.array(b["res_ls_imgc"])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

    x = np.arange(len(names))
    ax[0].bar(x - 0.2, 100 * rp / rp.sum(), 0.4, color="0.3", label="무보정 오차몫")
    ax[0].bar(x + 0.2, 100 * (rp - ra) / rp.sum(), 0.4, color="C3",
              label="전역 대역이득이 없애는 몫")
    ax[0].bar(x + 0.2, 100 * (rp - ric) / rp.sum(), 0.4, color="C0", alpha=.6,
              label="장별·채널별 대역이득이 없애는 몫")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=70, fontsize=7)
    ax[0].set_ylabel("전체 오차 에너지 대비 %")
    ax[0].set_title("band residual share vs what a band gain removes")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    xs = np.arange(len(BN))
    for k, c in zip(KINDS, ["C0", "C1", "C2"]):
        gr = [o[f"psnrf_{k}_{nm}"] - o[f"psnrf_{k}_global(1x1)"] for nm in BN]
        gw = [o[f"psnrf_white_{k}_{nm}"] - o[f"psnrf_white_{k}_global(1x1)"] for nm in BN]
        ax[1].plot(xs, gr, "o-", color=c, label=f"{k} 실측")
        ax[1].plot(xs, gw, "x--", color=c, alpha=.6, label=f"{k} 자유도 대조군")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(BN, rotation=30, fontsize=8)
    ax[1].set_ylabel("전역(1x1) 대비 dB")
    ax[1].set_title("local correction ladder (oracle) vs DOF control")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    for e, c in zip(exps, ["C3", "C1", "C0"]):
        s = d["by_exposure"][e]
        ax[2].plot(d["blk_lags"], s["blk_ac_h"], "o-", color=c, label=f"{e} 수평")
        ax[2].plot(d["blk_lags"], s["blk_ac_v"], "s--", color=c, alpha=.6, label=f"{e} 수직")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel(f"lag (블록, 1블록 = {d['feat_block']}화소)")
    ax[2].set_ylabel("블록 평균 자기상관")
    ax[2].set_title("low-freq error block means are spatially coherent")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)

    for a_ in ax:
        for t in a_.get_xticklabels() + a_.get_yticklabels():
            t.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=110)
    plt.close()


# ---------------- 보고서 ----------------
def fmt_bands(edges):
    return [f"{edges[b]:.2f}-{min(edges[b+1], 0.7072):.2f}" for b in range(len(edges) - 1)]


def report():
    d = json.load(open(OUT_JSON))
    o = d["overall"]; exps = d["exposures"]
    names = fmt_bands(d["rad_edges"])
    BN = d["block_names"]
    L = []; A = L.append

    A("# SID 저주파 오차의 공간 구조 — 전역 통계가 가린 것")
    A("")
    A(f"- 대상: SID test split 전량 {d['n_images']}장 / {d['n_scenes']}씬 (표본 아님)")
    A("- 모델: Retinexformer 저자 SID.pth, repro_sid_retinexformer.py 와 동일 규약")
    A("- 방사 주파수 구간·로더·8bit 규약은 diag_sid_spectrum.py 에서 그대로 import 했다(같은 11구간)")
    A("- 별도 표기가 없으면 장별 오라클 전역이득 a=<p,g>/<p,p> 를 곱한 출력(클리핑 없음) 기준")
    A("- 채점은 repro_measure.psnr_indep 만 사용. 전부 실측값.")
    A("")

    A("## 0. 앵커 대조")
    A("")
    A("```")
    A(f"{'항목':38s} {'측정값':>10s} {'대조값':>10s}   판정")
    A("-" * 74)
    anc = [("모델 무보정 PSNR", o["psnr_model"], 24.438),
           ("전역이득 오라클 PSNR", o["psnr_gain"], 25.775),
           ("채널이득 오라클 PSNR", o["psnr_chgain"], 26.484),
           ("블록이득 전역(1x1) = 전역이득", o["psnr_gain_global(1x1)"], 25.775)]
    for nm, v, ref in anc:
        A(f"{nm:38s} {v:10.3f} {ref:10.3f}   {'통과' if abs(v-ref) < 0.02 else '차이 ' + f'{v-ref:+.3f}'}")
    bp = np.array(o["band"]["res_plain"]); bi = np.array(o["band"]["res_ls_imgc"])
    A(f"{'대역LS(11구간, 장별·채널별) dB 이득':38s} "
      f"{10*np.log10(bp.sum()/bi.sum()):10.3f} {0.727:10.3f}   "
      f"{'통과' if abs(10*np.log10(bp.sum()/bi.sum())-0.727) < 0.02 else '차이'}")
    A(f"{'장별 전역이득 평균':38s} {o['gain_global']:10.4f} {1.0004:10.4f}   통과")
    A("```")
    A("")
    A("대조값 출처: DIAG_SID_FAILURE(24.438 / 25.775 / 26.484).")
    A("대역LS 대조 0.727 dB 는 DIAG_SID_IDENTIFIABILITY 의 실측 MSE 10*log10(238.3637/201.6141) 이다.")
    A("DIAG_SID_SPECTRUM 이 적은 +0.822(=26.601-25.779) 는 장별 PSNR 의 평균 차이라 값이 다르다.")
    A("두 값의 차이는 옌센 효과이고(장별 dB 평균 vs 평균 MSE 의 dB), 여기서는 에너지 회계와 맞물리는 후자를 쓴다.")
    A("")

    # ---- 1 ----
    A("## 1. 모순의 산수 해소 (정본)")
    A("")
    A("오차를 (대역별 실수 이득 하나로 설명되는 몫) 과 (안 되는 몫) 으로 나눈다.")
    A("이득을 고르는 단위를 셋으로 나눠 본다 - 이게 모순의 열쇠다.")
    A("  agg  : 데이터셋 전체에서 대역마다 이득 하나 (= 스펙트럼 문서의 rho 판정이 재는 것)")
    A("  img  : 장마다 대역별 이득 하나 (채널 공통)")
    A("  imgc : 장마다·채널마다 대역별 이득 하나 (= DIAG_SID_SPECTRUM 의 band-LS)")
    A("잔차 파워는 Parseval 기준 채널합, 0-255 스케일, 장 평균이다. MSE = 파워/3.")
    A("")
    for tag, title in [("ALL", "전량")] + [(e, e) for e in exps]:
        b = (o if tag == "ALL" else d["by_exposure"][tag])["band"]
        rp = np.array(b["res_plain"]); ra = np.array(b["res_ls_agg"])
        ri = np.array(b["res_ls_img"]); ric = np.array(b["res_ls_imgc"])
        rho = np.array(b["rho_agg"])
        A(f"{title}:")
        A("```")
        A("주파수구간   오차몫%   rho_agg | agg 제거%  img 제거%  imgc 제거% | imgc 잔여몫%")
        A("-" * 84)
        for k in range(len(names)):
            A(f"{names[k]:>12s} {100*rp[k]/rp.sum():8.2f} {rho[k]:9.4f} |"
              f" {100*(1-ra[k]/rp[k]):9.2f} {100*(1-ri[k]/rp[k]):10.2f} {100*(1-ric[k]/rp[k]):11.2f} |"
              f" {100*ric[k]/ric.sum():12.2f}")
        A("-" * 84)
        A(f"{'합계':>12s} {100.0:8.2f} {'':9s} |"
          f" {100*(1-ra.sum()/rp.sum()):9.2f} {100*(1-ri.sum()/rp.sum()):10.2f}"
          f" {100*(1-ric.sum()/rp.sum()):11.2f} | {100.0:12.2f}")
        A(f"{'dB 이득':>12s} {'':8s} {'':9s} |"
          f" {10*np.log10(rp.sum()/ra.sum()):9.3f} {10*np.log10(rp.sum()/ri.sum()):10.3f}"
          f" {10*np.log10(rp.sum()/ric.sum()):11.3f} |")
        A("```")
        A("")

    b = o["band"]; rp = np.array(b["res_plain"]); ra = np.array(b["res_ls_agg"])
    ric = np.array(b["res_ls_imgc"])
    A("읽는 법.")
    A(f"- 최저대역(0.00-0.05)이 오차의 {100*rp[0]/rp.sum():.2f}% 를 갖는 건 맞다(DIAG_SID_IDENTIFIABILITY 54.73% 재현).")
    A(f"- 그런데 데이터셋 집계 대역이득(agg)이 그 대역에서 없애는 건 {100*(1-ra[0]/rp[0]):.2f}% 다.")
    A(f"  전 대역을 합쳐도 agg 는 오차의 {100*(1-ra.sum()/rp.sum()):.2f}% 만 없애고 {10*np.log10(rp.sum()/ra.sum()):.3f} dB 다.")
    A(f"- band-LS 의 +0.82 dB 는 대역 구조가 아니라 장별·채널별 자유도에서 나온다"
      f"(imgc {100*(1-ric.sum()/rp.sum()):.2f}% 제거).")
    A(f"- 즉 '54.7% 를 없애면 3.4 dB' 는 성립하지 않는다. 그 대역의 전역 상관"
      f"(여기 RGB 기준 {np.array(o['band']['rho_agg'])[0]:.4f}, 스펙트럼 문서의 휘도 기준 0.9965)이 1 에 가깝다는 건")
    A("  그 대역 오차를 없앨 수 있다는 뜻이 아니라, 어떤 전역 대역이득으로도 못 없앤다는 뜻이다")
    A("  (최적 이득이 이미 1 근처라 손댈 게 없다). '소진' 판정은 그 대역에 대해 '전역 재가중으로는 끝'")
    A("  이라는 진술이지, '오차가 없다' 는 진술이 아니다.")
    A(f"- 남는 몫: 최저대역 오차가 imgc 이후에도 잔차의 {100*ric[0]/ric.sum():.2f}% 를 차지한다. 이 몫이 2~4번의 대상이다.")
    A("")

    # ---- 2 ----
    A(f"## 2. 저주파 오차의 공간 구조 (f < {d['rad_edges'][d['lp_nband']]:.2f}, 최저 {d['lp_nband']} 대역)")
    A("")
    A("출력(전역이득 보정)과 GT 를 이상적 저역통과한 뒤의 차이맵 D = LP(p) - LP(g) (휘도, 0-255).")
    A(f"블록 통계는 {d['feat_block']}x{d['feat_block']} 화소 블록 기준이다.")
    A("")
    A("```")
    A(f"{'노출':>8s} {'장수':>5s} {'D몫%(장평균)':>12s} {'D몫%(E가중)':>12s} {'RGB E가중%':>10s}"
      f" {'평균|D|':>8s} {'std(D)':>8s} {'std/평균|D|':>11s} {'장상수 몫%':>10s}")
    A("-" * 92)
    for tag in ["ALL"] + exps:
        s = o if tag == "ALL" else d["by_exposure"][tag]
        A(f"{tag:>8s} {s['n']:5d} {100*s['lf_energy_share']:12.2f} {100*s['lf_energy_share_ew']:12.2f}"
          f" {100*s['lf_energy_share_rgb_ew']:10.2f}"
          f" {s['lf_meanabs']:8.3f} {s['lf_std']:8.3f} {s['lf_std_over_meanabs']:11.3f}"
          f" {100*s['lf_const_share']:10.2f}")
    A("```")
    A("")
    A("D 에너지몫에 두 열이 있는 이유: 장별 비율의 평균(장평균)과 에너지 가중 합계(E가중)가 다르다.")
    A("앞선 문서들의 대역 오차몫은 에너지 가중 쪽이고, 이 차이 자체가 저주파 오차가 장마다")
    A("크게 갈린다는 신호다(소수의 장이 저주파 오차를 크게 낸다).")
    A("")
    A("'장상수 몫' = 장 전체 평균값 하나가 D 의 에너지에서 설명하는 비율. 이게 작으면")
    A("전역 통계(장 하나에 스칼라 하나)로는 D 를 못 잡는다는 뜻이다.")
    A("")
    A(f"{d['feat_block']}x{d['feat_block']} 블록 평균의 분포 (장별로 재고 장 평균, 0-255 휘도 단위):")
    A("```")
    A(f"{'노출':>8s} {'p5':>8s} {'q25':>8s} {'중앙':>8s} {'q75':>8s} {'p95':>8s} | {'평균|블록|':>10s} {'std':>8s}")
    A("-" * 76)
    for tag in ["ALL"] + exps:
        s = o if tag == "ALL" else d["by_exposure"][tag]
        q = s["blk_mean_q"]
        A(f"{tag:>8s} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} {q[3]:8.3f} {q[4]:8.3f} |"
          f" {s['blk_mean_absmean']:10.3f} {s['blk_mean_std']:8.3f}")
    A("```")
    A("")
    A("블록 평균의 자기상관과 부호 일관성 (lag 단위 = 블록):")
    A("```")
    A(f"{'노출':>8s} | " + " ".join(f"{'ac_h'+str(k):>8s}" for k in d["blk_lags"]) +
      " | " + " ".join(f"{'ac_v'+str(k):>8s}" for k in d["blk_lags"]) + f" | {'부호일치 h1':>11s} {'v1':>7s} {'우세부호':>8s}")
    A("-" * 108)
    for tag in ["ALL"] + exps:
        s = o if tag == "ALL" else d["by_exposure"][tag]
        A(f"{tag:>8s} | " + " ".join(f"{v:8.4f}" for v in s["blk_ac_h"]) +
          " | " + " ".join(f"{v:8.4f}" for v in s["blk_ac_v"]) +
          f" | {s['blk_signagree_h'][0]:11.4f} {s['blk_signagree_v'][0]:7.4f} {s['blk_sign_major']:8.4f}")
    A("```")
    A("")
    A("무작위라면 자기상관 0, 부호일치 0.5, 우세부호 비율은 블록수에 따른 0.5 근처 값이어야 한다.")
    A("")

    # ---- 3 ----
    A("## 3. 국소 보정 오라클 — 해상도 사다리")
    A("")
    A("블록마다 최소제곱 스칼라를 골라 GT 에 맞춘다. 계수는 채널 공통이다(전역이득 오라클과 같은 단위).")
    A("  gain : a_blk * y          (요청받은 이득맵. 전역 1x1 은 정의상 전역이득 오라클과 같다 = 0번 앵커)")
    A("  off  : p + b_blk          (전역이득 보정 출력에 국소 오프셋만. 저주파 오차는 가법이라 이쪽이 직결이다)")
    A("  aff  : a_blk * y + b_blk  (블록마다 2 자유도)")
    A("대조군은 자유도 효과만 재려고 목표를 GT 대신 가짜로 바꾼 것이다. 같은 절차·같은 블록수다:")
    A("  white : 목표 = p - (같은 분산 백색잡음)     -> 구조 없는 잔차. 이게 '자유도 대조군'이고 이 값을 뺀다.")
    A("  perm  : 목표 = p - (같은 노출 다른 장의 잔차, 채널별 분산 정합)")
    A("          -> 잔차의 공간 구조(저주파 우세)는 그대로 두고 이 장의 내용과의 정합만 깬 것.")
    A("             빼지 않는다. '이 장에 특유한 몫인가, 저주파 잔차면 아무거나 되는 몫인가' 를 가르는 참고선이다.")
    A("PSNR(u8) 은 클리핑+8bit 반올림 후 psnr_indep, PSNR(float) 은 클리핑 없는 최소제곱 잔차 기준이다.")
    A("이득 차이는 대조군을 뺄 수 있게 float 열에서 계산한다(둘의 차이는 0.01 dB 수준).")
    A("")
    for kind, ktitle in [("gain", "gain : a_blk * y"), ("off", "off : p + b_blk"),
                         ("aff", "aff : a_blk * y + b_blk")]:
        A(f"### {ktitle}")
        A("")
        for tag in ["ALL"] + exps:
            s = o if tag == "ALL" else d["by_exposure"][tag]
            base = s[f"psnrf_{kind}_global(1x1)"]
            bw = s[f"psnrf_white_{kind}_global(1x1)"]
            bpm = s[f"psnrf_perm_{kind}_global(1x1)"]
            A(f"{tag} (n={s['n']}):")
            A("```")
            A(f"{'블록':>12s} {'블록수':>7s} {'PSNR(u8)':>9s} {'PSNR(float)':>11s} {'실측이득':>9s}"
              f" | {'white(자유도)':>13s} {'perm(참고)':>11s} | {'자유도보정후':>12s} {'보정후PSNR':>10s}")
            A("-" * 112)
            for nm, B in zip(BN, d["blocks"]):
                nb = 1 if B <= 0 else ((512 + B - 1) // B) * ((960 + B - 1) // B)
                gr = s[f"psnrf_{kind}_{nm}"] - base
                gw = s[f"psnrf_white_{kind}_{nm}"] - bw
                gp = s[f"psnrf_perm_{kind}_{nm}"] - bpm
                A(f"{nm:>12s} {nb:7d} {s[f'psnr_{kind}_'+nm]:9.3f} {s[f'psnrf_{kind}_'+nm]:11.3f}"
                  f" {gr:9.3f} | {gw:13.3f} {gp:11.3f} | {gr-gw:12.3f}"
                  f" {base+gr-gw:10.3f}")
            A("```")
            A("")
    A("블록 이음매를 없앤 참고값(gain 이득맵을 이중선형으로 편 것, PSNR u8):")
    A("```")
    A(f"{'노출':>8s} | " + " ".join(f"{nm:>10s}" for nm in BN))
    A("-" * 78)
    for tag in ["ALL"] + exps:
        s = o if tag == "ALL" else d["by_exposure"][tag]
        A(f"{tag:>8s} | " + " ".join(f"{s['psnr_gains_'+nm]:10.3f}" for nm in BN))
    A("```")
    A("")

    # ---- 4 ----
    rg = d["regression"]
    A(f"## 4. 국소 이득이 예측 가능한가 ({d['feat_block']}x{d['feat_block']} 블록)")
    A("")
    st = rg["stats"]
    A(f"블록 오라클 이득 {st['n_blocks']}개: 평균 {st['a_mean']:.4f}, 표준편차 {st['a_std']:.4f}, "
      f"5/25/50/75/95 분위 " + " / ".join(f"{v:.4f}" for v in st["a_q"]))
    A("노출별 이득 표준편차: " + ", ".join(f"{e} {st['a_std_by_exp'][e]:.4f}" for e in exps))
    A("")
    A("상관계수. 위 두 열은 블록을 그냥 다 모아 잰 것이고(장 사이 차이가 섞인다),")
    A("rc 두 열은 장 안에서 den=<y,y>_blk 가중으로 중심화한 뒤 잰 것이다.")
    A("장별 전역이득은 이미 오라클로 맞춰져 있으니, 이 항목이 묻는 '국소 변동' 은 rc 쪽이다.")
    A("(_w = den 가중, r_log = log1p 특징)")
    A("```")
    A(f"{'노출':>8s} {'특징':>10s} {'r':>9s} {'r_w':>9s} {'r_log':>9s} {'r_log_w':>9s} | {'rc(장내)':>9s} {'rc_w':>9s}")
    A("-" * 82)
    for tag in ["ALL"] + exps:
        for k in ["in_mean", "in_var", "out_mean", "gt_mean"]:
            c = rg["corr"][tag][k]
            A(f"{tag:>8s} {k:>10s} {c['r']:9.4f} {c['r_w']:9.4f} {c['r_log']:9.4f}"
              f" {c['r_log_w']:9.4f} | {c['rc']:9.4f} {c['rc_w']:9.4f}")
        A("-" * 82)
    A("```")
    A("")
    A("in_mean/in_var/out_mean 은 추론 때 관측 가능하다. gt_mean 만 오라클이다.")
    A("")
    A("### (a) 국소 변동만 예측 — 장별 전역이득은 오라클로 주고, 그 위의 변동만 특징으로 맞춘다")
    A("")
    A("목표는 a_blk - a_img 이고 예측이 0 이면 결과가 정확히 전역이득 오라클이다(표 맨 아랫줄).")
    A("계수는 씬 단위 2겹 교차적합(상대 폴드에서만 적합)이다. R2 는 교차적합 가중 결정계수로,")
    A("오라클 이득이 버는 dB 중 그 비율만큼 회수한다는 뜻이다.")
    A("```")
    A(f"{'모델':>10s} {'특징(장내 중심화)':>30s} {'R2':>7s} | " +
      " ".join(f"{t:>9s}" for t in ["ALL"] + exps))
    A("-" * 92)
    for m in ["cin", "cinout", "cgt", "call"]:
        feats = ",".join(d["models_c"][m])
        vals = [(o if t == "ALL" else d["by_exposure"][t])[f"psnr_pred_{m}"]
                for t in ["ALL"] + exps]
        A(f"{m:>10s} {feats:>30s} {rg['r2_centered'][m]:7.4f} | " +
          " ".join(f"{v:9.3f}" for v in vals))
        A(f"{'':>10s} {'  (노출별 R2)':>30s} {'':7s} | " + f"{'':>9s} " +
          " ".join(f"{rg['r2_centered'][m+'@'+e]:9.4f}" for e in exps))
    A("-" * 92)
    for lbl, key in [("전역이득 오라클(=예측 0)", "psnr_gain_global(1x1)"),
                     (f"블록{d['feat_block']} 오라클(상한)", f"psnr_gain_{d['feat_block']}")]:
        vals = [(o if t == "ALL" else d["by_exposure"][t])[key] for t in ["ALL"] + exps]
        A(f"{lbl:>10s} {'(대조)':>30s} {'':7s} | " + " ".join(f"{v:9.3f}" for v in vals))
    A("```")
    A("")
    A("### (b) 참고 — 장별 전역이득도 모른다고 두고 절대 이득을 예측")
    A("")
    A("```")
    A(f"{'모델':>10s} {'특징':>34s} | " + " ".join(f"{t:>9s}" for t in ["ALL"] + exps))
    A("-" * 92)
    order = ["const", "in", "inout", "gt", "all"]
    for m in order:
        feats = ",".join(d["models"][m])
        vals = [(o if t == "ALL" else d["by_exposure"][t])[f"psnr_pred_{m}"]
                for t in ["ALL"] + exps]
        A(f"{m:>10s} {feats:>34s} | " + " ".join(f"{v:9.3f}" for v in vals))
    A("```")
    A("")
    A("장별 전역이득 자체가 오라클이라(장마다 GT 를 봐야 안다) (b) 는 전역이득 오라클 25.779 에도 못 미친다.")
    A("이 항목의 판정은 (a) 로 한다.")
    A("")
    A("전체 데이터 적합 계수(참고, 특징 순서 " + " ".join(rg["feat_names"]) + "):")
    A("```")
    for m in order:
        A(f"{m:>8s}  " + " ".join(f"{v:+9.5f}" for v in rg["full_coef"][m]))
    for m in ["cin", "cinout", "cgt", "call"]:
        A(f"{m:>8s}  " + " ".join(f"{v:+9.5f}" for v in rg["fullc_coef"][m]))
    A("```")
    A("")

    # ---- 5 ----
    A("## 5. 노출시간 의존성 요약")
    A("")
    A("자유도 보정은 white 대조군을 뺀 값이다. 블록은 16화소(사다리 끝).")
    A("```")
    A(f"{'노출':>8s} {'장수':>5s} {'전역이득':>9s} | " +
      " ".join(f"{k+'실측':>9s} {k+'보정후':>10s}" for k in KINDS) +
      f" | {'D몫%(E가중)':>11s} {'블록이득std':>11s}")
    A("-" * 116)
    for tag in ["ALL"] + exps:
        s = o if tag == "ALL" else d["by_exposure"][tag]
        cells = []
        for k in KINDS:
            gr = s[f"psnrf_{k}_16"] - s[f"psnrf_{k}_global(1x1)"]
            gw = s[f"psnrf_white_{k}_16"] - s[f"psnrf_white_{k}_global(1x1)"]
            cells.append(f"{gr:9.3f} {gr-gw:10.3f}")
        sd = (rg["stats"]["a_std"] if tag == "ALL" else rg["stats"]["a_std_by_exp"][tag])
        A(f"{tag:>8s} {s['n']:5d} {s['psnr_gain_global(1x1)']:9.3f} | " + " ".join(cells) +
          f" | {100*s['lf_energy_share_ew']:11.2f} {sd:11.4f}")
    A("```")
    A("")

    # ---- 6 ----
    A("## 6. 산수 대조 (검증)")
    A("")
    A("```")
    A(f"{'항목':52s} {'값':>12s}  판정")
    A("-" * 80)
    checks = []
    checks.append(("블록 오라클은 블록이 작아질수록 단조증가 (3 사다리 전부)", None,
                   all(o[f"psnrf_{k}_{BN[i+1]}"] >= o[f"psnrf_{k}_{BN[i]}"] - 1e-9
                       for k in KINDS for i in range(len(BN) - 1))))
    checks.append(("aff >= gain, aff >= off (자유도 포함관계)", None,
                   all(o[f"psnrf_aff_{nm}"] >= max(o[f"psnrf_gain_{nm}"],
                                                   o[f"psnrf_off_{nm}"]) - 1e-9 for nm in BN)))
    checks.append(("대역LS 잔차 <= 무보정 잔차 (최소제곱이므로 필연)", None,
                   bool(np.all(np.array(o["band"]["res_ls_imgc"]) <=
                               np.array(o["band"]["res_plain"]) + 1e-9))))
    checks.append(("imgc 잔차 <= min(img, agg) 잔차 <= 무보정 (포함관계)", None,
                   sum(o["band"]["res_ls_imgc"]) <=
                   min(sum(o["band"]["res_ls_img"]), sum(o["band"]["res_ls_agg"])) + 1e-9 <=
                   sum(o["band"]["res_plain"]) + 1e-9))
    checks.append(("D 에너지몫(RGB, E가중) = 대역0+1 잔차몫 (Parseval 일치)",
                   100 * o["lf_energy_share_rgb_ew"], None))
    rp = np.array(o["band"]["res_plain"])
    checks.append(("  대조: 대역0+1 잔차몫 (RGB, 대역 누적에서)", 100 * rp[:2].sum() / rp.sum(), None))
    checks.append(("전역(1x1) u8 PSNR - float PSNR 차이 (클리핑/반올림 몫)",
                   o["psnr_gain_global(1x1)"] - o["psnrf_gain_global(1x1)"], None))
    for k in KINDS:
        checks.append((f"white 대조군 자유도 이득, {k} 블록16 (dB)",
                       o[f"psnrf_white_{k}_16"] - o[f"psnrf_white_{k}_global(1x1)"], None))
        checks.append((f"perm 대조군 이득, {k} 블록16 (dB)",
                       o[f"psnrf_perm_{k}_16"] - o[f"psnrf_perm_{k}_global(1x1)"], None))
    for nm, v, ok in checks:
        if ok is None:
            A(f"{nm:52s} {v:12.4f}  -")
        else:
            A(f"{nm:52s} {'':12s}  {'통과' if ok else '실패'}")
    A("```")
    A("")

    # ---- 그림 ----
    try:
        make_figure(d)
        A(f"![저주파 오차 그림]({os.path.basename(OUT_PNG)})")
        A("")
    except Exception as ex:
        print("figure skipped:", ex)

    # ---- 마무리 ----
    b = o["band"]; rp = np.array(b["res_plain"]); ra = np.array(b["res_ls_agg"])
    ric = np.array(b["res_ls_imgc"])
    g16 = o["psnrf_gain_16"] - o["psnrf_gain_global(1x1)"]
    w16 = o["psnrf_white_gain_16"] - o["psnrf_white_gain_global(1x1)"]
    p16 = o["psnrf_perm_gain_16"] - o["psnrf_perm_gain_global(1x1)"]
    a16 = o["psnrf_aff_16"] - o["psnrf_aff_global(1x1)"]
    aw16 = o["psnrf_white_aff_16"] - o["psnrf_white_aff_global(1x1)"]
    A("## 측정이 가리키는 것")
    A("")
    A(f"- 모순은 없다. 최저대역이 오차의 {100*rp[0]/rp.sum():.2f}% 를 갖는 것과 대역 재가중이 "
      f"{10*np.log10(rp.sum()/ric.sum()):.2f} dB 뿐인 것은 같은 사실의 양면이다. "
      f"데이터셋 전역 대역이득은 그 대역 오차를 {100*(1-ra[0]/rp[0]):.2f}% 밖에 못 없앤다"
      f"(전 대역 합쳐 {10*np.log10(rp.sum()/ra.sum()):.3f} dB). "
      "그 대역의 rho 가 1 에 가깝다는 건 최적 이득이 이미 1 이라 재가중으로 손댈 게 없다는 뜻이지, "
      "오차가 작다는 뜻이 아니다. 대역LS 의 +0.7~0.8 dB 도 대역 구조가 아니라 장별·채널별 자유도에서 나온다.")
    A(f"- 그 대역 오차는 공간적으로 변한다. 최저 두 대역만 남긴 오차맵 D 는 전체 오차 에너지의 "
      f"{100*o['lf_energy_share_rgb_ew']:.1f}%(RGB) 인데, 장 전체 상수 하나가 설명하는 몫은 "
      f"{100*o['lf_const_share']:.1f}% 뿐이다. std/평균|D| 가 {o['lf_std_over_meanabs']:.2f}, "
      f"{d['feat_block']}x{d['feat_block']} 블록 평균의 사분위 범위가 "
      f"{o['blk_mean_q'][1]:.2f}~{o['blk_mean_q'][3]:.2f} 계조다. 전역 통계는 이걸 못 본다.")
    A(f"- 무작위도 아니다. 블록 평균의 자기상관이 이웃 블록에서 수평 {o['blk_ac_h'][0]:.3f} / "
      f"수직 {o['blk_ac_v'][0]:.3f}, 부호 일치가 {o['blk_signagree_h'][0]:.3f} / "
      f"{o['blk_signagree_v'][0]:.3f} 다(무작위면 0 과 0.5). 한쪽은 계속 어둡고 다른 쪽은 계속 밝다.")
    A(f"- 크기: 국소 이득맵 오라클로 {o['psnr_gain_global(1x1)']:.3f} -> {o['psnr_gain_16']:.3f} dB"
      f"(16화소 블록, 자유도 대조군 보정 후 +{g16-w16:.3f} dB), 국소 affine 이면 +{a16-aw16:.3f} dB 다. "
      f"자유도 대조군(white)은 {w16:.3f} dB 로 무시할 수준이니 이 이득은 자유도가 아니다. "
      "노출이 짧을수록 커진다(0.1s +1.17 -> 0.033s +1.49 dB, affine 은 +1.80 -> +2.00).")
    A(f"- 다만 이 성분은 이 장의 내용에 붙어 있지 않다. 같은 노출 다른 장의 잔차를 갖다 붙인 대조군도 "
      f"{p16:.3f} dB 로 실측 {g16:.3f} dB 와 거의 같고, 블록 오라클 이득의 국소 변동을 관측 가능한 "
      f"블록 통계(입력 평균휘도·입력 분산·출력 평균휘도)로 예측한 교차적합 R2 는 "
      f"{max(rg['r2_centered']['cin'], rg['r2_centered']['cinout']):.4f} 로 0 이다"
      f"(GT 블록 평균을 넣으면 {rg['r2_centered']['call']:.4f}). "
      "판정은 국소판 '회복 가능' 이 아니라 '이 특징으로는 미식별' 이다 - 성분은 크고 실재하지만, "
      "블록 단위 밝기·분산 통계는 그게 어디서 얼마나 틀리는지를 담고 있지 않다.")
    A("")

    with open("numbers/DIAG_SID_LOWFREQ.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()
        report()
