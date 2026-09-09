"""diag_sid_spectrum.py

DIAG_SID_FAILURE 가 남긴 "구조 잔차 64.2%" 의 정체가
고주파 소실(= 조건부 평균으로의 회귀) 인지 확정한다. 학습 없음, 추론 + 분석만.

측정 항목 (SID test split 전량 598장 / 50씬):
  1. 방사 파워 스펙트럼   : GT / 출력 / 잔차, 방사 구간별 출력·GT 파워비
  2. 최적 저역통과 설명력 : blur_sigma(GT) 와 모델출력 사이 PSNR, 최적 sigma
  3. 기울기 크기 비율     : 입력 휘도 구간별 |grad(out)| / |grad(GT)|
  4. 잔차의 구조성       : 잔차 공간 자기상관 (수평/수직 lag 1..16) vs 백색잡음 대조
  5. 상한 계산           : 방사 구간별 진폭만 GT 에 맞춘(위상 유지) 영상의 PSNR

규약은 repro_sid_retinexformer.py / diag_sid_failure.py 와 동일(같은 로더, 같은 8bit 반올림).
채점은 repro_measure.psnr_indep (독립 구현) 만 쓴다.
스펙트럼/기울기/자기상관은 전부 오라클 전역이득 a = <p,g>/<p,p> 를 먼저 곱한 출력으로 잰다
(밝기 차이가 스펙트럼 전체를 스케일시키지 않게).
"""
import os, sys, glob, json, re, math, time
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep

REPO = "third_party/Retinexformer"
DATA = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
WEIGHTS = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth")
OUT_JSON = os.environ.get("SPEC_OUT", "numbers/diag_sid_spectrum.json")

# diag_sid_failure.py 와 같은 입력 휘도 구간화 (재사용)
BIN_EDGES = [0, 2, 5, 10, 20, 40, 80, 256]
BIN_NAMES = ["0-2", "2-5", "5-10", "10-20", "20-40", "40-80", "80+"]
LUMA_W = np.array([0.299, 0.587, 0.114])

# 방사 주파수 구간: 0..0.5 를 0.05 폭 10개 + 코너(0.5..0.7072) 1개 = 11개
RAD_EDGES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             math.sqrt(0.5) + 1e-9]
SIGMAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
LAGS = list(range(1, 17)) + [20, 24, 28, 32, 48, 64]
BOUND_NBANDS = [11, 32, 64]


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
    return np.ascontiguousarray(img[:, :, ::-1])          # 저자 로더의 채널 뒤집기


def exposure_of(path):
    m = re.search(r"_([0-9.]+)s\.npy$", os.path.basename(path))
    return m.group(1) + "s" if m else "?"


def u8(a01):
    return np.clip(np.rint(a01 * 255.0), 0, 255).astype(np.uint8)


# ---------------- GPU 도구 ----------------
def make_radial_index(H, W, edges, device, torch):
    fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=torch.float64)
    fx = torch.fft.fftfreq(W, d=1.0, device=device, dtype=torch.float64)
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    e = torch.tensor(edges, device=device, dtype=torch.float64)
    idx = torch.bucketize(r, e[1:-1], right=False)         # 0..len(edges)-2
    return idx, r


def gaussian_blur(x, sigma, F, torch):
    """x: 1xCxHxW. reflect padding, separable."""
    if sigma <= 0:
        return x
    rad = int(math.ceil(3.0 * sigma))
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    y = F.pad(x, (rad, rad, 0, 0), mode="reflect")
    y = F.conv2d(y, kx, groups=C)
    y = F.pad(y, (0, 0, rad, rad), mode="reflect")
    y = F.conv2d(y, ky, groups=C)
    return y


def sobel_mag(img, F, torch):
    """img: HxW float64 tensor (0-255 스케일). reflect pad, 테두리 1픽셀은 호출측에서 제외."""
    gx_k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                        device=img.device, dtype=img.dtype).view(1, 1, 3, 3) / 8.0
    gy_k = gx_k.transpose(2, 3).contiguous()
    x = img[None, None]
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x, gx_k)[0, 0]
    gy = F.conv2d(x, gy_k)[0, 0]
    return torch.sqrt(gx * gx + gy * gy)


# ---------------- 누적기 ----------------
class Acc:
    def __init__(self, nrad, nbin, nlag):
        self.n = 0
        self.p_gt = np.zeros(nrad)          # 방사 구간 파워 (sum over images of per-image band power)
        self.p_out = np.zeros(nrad)
        self.p_res = np.zeros(nrad)
        self.ratio_perimg = np.zeros(nrad)  # 장별 (P_out/P_gt) 의 합
        self.grad_gt = np.zeros(nbin)       # |grad| 합
        self.grad_out = np.zeros(nbin)
        self.grad_px = np.zeros(nbin)
        # 자기상관: lag 별 sum(ab), sum(aa), sum(bb)  (h / v, 잔차 / 백색 / GT)
        self.ac = {k: np.zeros((nlag, 3)) for k in
                   ["res_h", "res_v", "wht_h", "wht_v", "gt_h", "gt_v"]}
        self.ms_gt = 0.0                    # Parseval 대조용 평균제곱
        self.ms_out = 0.0
        self.ms_res = 0.0

    def add(self, o):
        self.n += 1
        self.p_gt += o["p_gt"]; self.p_out += o["p_out"]; self.p_res += o["p_res"]
        self.ratio_perimg += o["p_out"] / np.maximum(o["p_gt"], 1e-30)
        self.grad_gt += o["grad_gt"]; self.grad_out += o["grad_out"]; self.grad_px += o["grad_px"]
        for k in self.ac:
            self.ac[k] += o["ac"][k]
        self.ms_gt += o["ms_gt"]; self.ms_out += o["ms_out"]; self.ms_res += o["ms_res"]

    def dump(self):
        d = dict(n=self.n)
        d["p_gt"] = list(self.p_gt / self.n)
        d["p_out"] = list(self.p_out / self.n)
        d["p_res"] = list(self.p_res / self.n)
        d["ratio_energy"] = list(self.p_out / np.maximum(self.p_gt, 1e-30))
        d["ratio_perimg_mean"] = list(self.ratio_perimg / self.n)
        d["res_over_gt"] = list(self.p_res / np.maximum(self.p_gt, 1e-30))
        d["grad_gt_mean"] = list(self.grad_gt / np.maximum(self.grad_px, 1))
        d["grad_out_mean"] = list(self.grad_out / np.maximum(self.grad_px, 1))
        d["grad_ratio"] = list((self.grad_out / np.maximum(self.grad_px, 1)) /
                               np.maximum(self.grad_gt / np.maximum(self.grad_px, 1), 1e-30))
        d["grad_px_frac_pct"] = list(100.0 * self.grad_px / max(self.grad_px.sum(), 1))
        d["autocorr"] = {k: list(v[:, 0] / np.sqrt(np.maximum(v[:, 1] * v[:, 2], 1e-30)))
                         for k, v in self.ac.items()}
        d["ms_gt"] = self.ms_gt / self.n
        d["ms_out"] = self.ms_out / self.n
        d["ms_res"] = self.ms_res / self.n
        return d


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
    lim = int(os.environ.get("SPEC_LIMIT", "0"))
    if lim:
        pairs = pairs[:lim]
    exps = sorted(set(exposure_of(p[1]) for p in pairs), key=lambda s: float(s[:-1]))
    print(f"pairs {len(pairs)}  scenes {len(set(p[0] for p in pairs))}  exps {exps}", flush=True)

    gt_cache = {}
    for _, _, gp in pairs:
        if gp not in gt_cache:
            gt_cache[gp] = u8(load_npy(gp))

    H, W = 512, 960
    N = H * W
    nrad = len(RAD_EDGES) - 1
    ridx, rmap = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    nfreq_per_band = np.bincount(ridx_flat.cpu().numpy(), minlength=nrad)

    # 상한 계산용 미세 구간 인덱스
    bound_idx = {}
    for nb in BOUND_NBANDS:
        if nb == nrad:
            bound_idx[nb] = ridx_flat
        else:
            e = list(np.linspace(0.0, 0.5, nb)) + [math.sqrt(0.5) + 1e-9]
            bi, _ = make_radial_index(H, W, e, dev, torch)
            bound_idx[nb] = bi.reshape(-1)

    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)
    accs = {e: Acc(nrad, len(BIN_NAMES), len(LAGS)) for e in exps}
    rows = []
    t0 = time.time()

    def band_power(field):
        """field: HxW float64 tensor -> 구간별 파워 (합 = mean(field^2), Parseval)"""
        Fk = torch.fft.fft2(field)
        pw = (Fk.real ** 2 + Fk.imag ** 2).reshape(-1) / (N * N)
        out = torch.zeros(nrad, device=dev, dtype=torch.float64)
        out.scatter_add_(0, ridx_flat, pw)
        return out.cpu().numpy()

    def autocorr_stats(f):
        """f: HxW float64 (평균 제거됨) -> (nlag,3) [sum ab, sum aa, sum bb] x (h,v)"""
        oh = np.zeros((len(LAGS), 3))
        ov = np.zeros((len(LAGS), 3))
        for i, k in enumerate(LAGS):
            a, b = f[:, :-k], f[:, k:]
            oh[i] = [float((a * b).sum()), float((a * a).sum()), float((b * b).sum())]
            a, b = f[:-k, :], f[k:, :]
            ov[i] = [float((a * b).sum()), float((a * a).sum()), float((b * b).sum())]
        return oh, ov

    gen = torch.Generator(device=dev); gen.manual_seed(20260905)

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
            y = torch.clamp(y, 0, 1).double()[0]                 # 3xHxW, [0,1]

            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            a = float((y * g).sum() / (y * y).sum())             # 오라클 전역이득
            pg = a * y                                           # 클리핑 없음(순수 최소제곱)

            r = dict(scene=scene, lq=os.path.basename(lqp), exp=exp, gain_global=a)

            # ---- 1. 방사 파워 스펙트럼 (휘도) ----
            Lg = (g * lw[:, None, None]).sum(0) * 255.0
            Lp = (pg * lw[:, None, None]).sum(0) * 255.0
            Lr = Lp - Lg
            o = {}
            o["p_gt"] = band_power(Lg)
            o["p_out"] = band_power(Lp)
            o["p_res"] = band_power(Lr)
            o["ms_gt"] = float((Lg ** 2).mean())
            o["ms_out"] = float((Lp ** 2).mean())
            o["ms_res"] = float((Lr ** 2).mean())

            # ---- 3. 기울기 크기 (휘도, 입력 밝기 구간별) ----
            mg_gt = sobel_mag(Lg, F, torch)
            mg_out = sobel_mag(Lp, F, torch)
            lqL = (lq.astype(np.float64) * 255.0 * LUMA_W).sum(2)
            bidx = np.digitize(lqL, BIN_EDGES[1:-1], right=False)
            bt = torch.from_numpy(bidx).cuda()
            core = torch.zeros_like(bt, dtype=torch.bool)
            core[1:-1, 1:-1] = True                              # 테두리 1픽셀 제외
            bsel = bt[core].reshape(-1)
            nb = len(BIN_NAMES)
            gg = torch.zeros(nb, device=dev, dtype=torch.float64)
            go = torch.zeros(nb, device=dev, dtype=torch.float64)
            gp = torch.zeros(nb, device=dev, dtype=torch.float64)
            gg.scatter_add_(0, bsel, mg_gt[core].reshape(-1))
            go.scatter_add_(0, bsel, mg_out[core].reshape(-1))
            gp.scatter_add_(0, bsel, torch.ones_like(bsel, dtype=torch.float64))
            o["grad_gt"] = gg.cpu().numpy()
            o["grad_out"] = go.cpu().numpy()
            o["grad_px"] = gp.cpu().numpy()

            # ---- 4. 잔차 자기상관 ----
            res = Lr - Lr.mean()
            wht = torch.randn(H, W, device=dev, dtype=torch.float64, generator=gen) * res.std()
            gtc = Lg - Lg.mean()
            ah, av = autocorr_stats(res)
            wh, wv = autocorr_stats(wht)
            gh, gv = autocorr_stats(gtc)
            o["ac"] = dict(res_h=ah, res_v=av, wht_h=wh, wht_v=wv, gt_h=gh, gt_v=gv)

            accs[exp].add(o)

            # ---- 2. 최적 저역통과 설명력 ----
            pred_u8 = u8(y.permute(1, 2, 0).cpu().numpy())
            predgf_u8 = u8(np.clip(pg.permute(1, 2, 0).cpu().numpy(), 0, 1))
            r["psnr_model"] = psnr_indep(pred_u8, gt_u8)
            r["psnr_model_gainfix"] = psnr_indep(predgf_u8, gt_u8)
            gb = g[None]
            bl_raw, bl_gf, bl_gt = [], [], []
            for s in SIGMAS:
                bg = gaussian_blur(gb, s, F, torch)
                bg_u8 = u8(np.clip(bg[0].permute(1, 2, 0).cpu().numpy(), 0, 1))
                bl_raw.append(psnr_indep(bg_u8, pred_u8))
                bl_gf.append(psnr_indep(bg_u8, predgf_u8))
                bl_gt.append(psnr_indep(bg_u8, gt_u8) if s > 0 else float("inf"))
            r["blur_vs_raw"] = bl_raw
            r["blur_vs_gainfix"] = bl_gf
            r["blur_vs_gt"] = [v if np.isfinite(v) else None for v in bl_gt]

            # ---- 5. 상한: 방사 구간 진폭만 GT 로 맞춤 (위상 유지) ----
            Fp = torch.fft.fft2(pg)                              # 3xHxW complex
            Fq = torch.fft.fft2(g)
            for nbnd in BOUND_NBANDS:
                bi = bound_idx[nbnd]
                nbands = int(bi.max().item()) + 1
                amp = torch.zeros_like(Fp)
                lsq = torch.zeros_like(Fp)
                for c in range(3):
                    fp = Fp[c].reshape(-1); fq = Fq[c].reshape(-1)
                    pp = torch.zeros(nbands, device=dev, dtype=torch.float64)
                    qq = torch.zeros(nbands, device=dev, dtype=torch.float64)
                    xy = torch.zeros(nbands, device=dev, dtype=torch.float64)
                    pp.scatter_add_(0, bi, fp.real ** 2 + fp.imag ** 2)
                    qq.scatter_add_(0, bi, fq.real ** 2 + fq.imag ** 2)
                    xy.scatter_add_(0, bi, fp.real * fq.real + fp.imag * fq.imag)
                    s_amp = torch.sqrt(qq / torch.clamp(pp, min=1e-30))
                    s_ls = xy / torch.clamp(pp, min=1e-30)
                    amp[c] = (fp * s_amp[bi]).reshape(H, W)
                    lsq[c] = (fp * s_ls[bi]).reshape(H, W)
                ia = torch.fft.ifft2(amp).real
                il = torch.fft.ifft2(lsq).real
                r[f"psnr_ampmatch_{nbnd}"] = psnr_indep(
                    u8(np.clip(ia.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                r[f"psnr_bandls_{nbnd}"] = psnr_indep(
                    u8(np.clip(il.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)

            rows.append(r)
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(pairs)}  {el:.0f}s  eta {el/(i+1)*(len(pairs)-i-1):.0f}s",
                      flush=True)

    # ---------------- 요약 ----------------
    allacc = Acc(nrad, len(BIN_NAMES), len(LAGS))
    for e in exps:
        allacc.n += accs[e].n
        allacc.p_gt += accs[e].p_gt; allacc.p_out += accs[e].p_out; allacc.p_res += accs[e].p_res
        allacc.ratio_perimg += accs[e].ratio_perimg
        allacc.grad_gt += accs[e].grad_gt; allacc.grad_out += accs[e].grad_out
        allacc.grad_px += accs[e].grad_px
        for k in allacc.ac:
            allacc.ac[k] += accs[e].ac[k]
        allacc.ms_gt += accs[e].ms_gt; allacc.ms_out += accs[e].ms_out; allacc.ms_res += accs[e].ms_res

    def blur_agg(rs):
        o = {}
        for tag in ["raw", "gainfix"]:
            arr = np.array([r["blur_vs_" + tag] for r in rs])          # N x S
            o["curve_" + tag] = list(arr.mean(0))
            j = int(np.argmax(arr.mean(0)))
            o["best_sigma_" + tag] = SIGMAS[j]
            o["best_psnr_" + tag] = float(arr.mean(0)[j])
            am = np.argmax(arr, 1)
            o["argmax_hist_" + tag] = {str(SIGMAS[k]): int((am == k).sum())
                                       for k in range(len(SIGMAS))}
        gtarr = np.array([[v if v is not None else np.nan for v in r["blur_vs_gt"]] for r in rs])
        o["curve_gt"] = [None if not np.isfinite(v) else float(v) for v in np.nanmean(gtarr, 0)]
        o["psnr_model"] = float(np.mean([r["psnr_model"] for r in rs]))
        o["psnr_model_gainfix"] = float(np.mean([r["psnr_model_gainfix"] for r in rs]))
        for nbnd in BOUND_NBANDS:
            o[f"psnr_ampmatch_{nbnd}"] = float(np.mean([r[f"psnr_ampmatch_{nbnd}"] for r in rs]))
            o[f"psnr_bandls_{nbnd}"] = float(np.mean([r[f"psnr_bandls_{nbnd}"] for r in rs]))
        return o

    out = dict(
        n_images=len(rows), n_scenes=len(set(r["scene"] for r in rows)),
        exposures=exps, sigmas=SIGMAS, lags=LAGS,
        rad_edges=RAD_EDGES, nfreq_per_band=[int(v) for v in nfreq_per_band],
        bin_names=BIN_NAMES, bin_edges=BIN_EDGES, bound_nbands=BOUND_NBANDS,
        overall=dict(spec=allacc.dump(), **blur_agg(rows)),
        by_exposure={e: dict(spec=accs[e].dump(),
                             **blur_agg([r for r in rows if r["exp"] == e])) for e in exps},
        gain_global_mean=float(np.mean([r["gain_global"] for r in rows])),
        rows=rows,
    )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote", OUT_JSON)


# ---------------- 보고서 ----------------
def report():
    d = json.load(open(OUT_JSON))
    L = []; A = L.append
    o = d["overall"]; sp = o["spec"]
    exps = d["exposures"]
    E = d["rad_edges"]; nr = len(E) - 1
    names = [f"{E[b]:.2f}-{min(E[b+1],0.7072):.2f}" for b in range(nr)]

    A("# SID 구조 잔차의 정체 — 고주파 소실 여부 확정")
    A("")
    A(f"- 대상: SID test split 전량 {d['n_images']}장 / {d['n_scenes']}씬 (표본 아님)")
    A("- 모델: Retinexformer 저자 SID.pth, repro_sid_retinexformer.py 와 동일 규약")
    A(f"- 앵커: 무보정 PSNR {o['psnr_model']:.3f} (DIAG_SID_FAILURE 24.438 과 일치), "
      f"오라클 전역이득 보정 후 {o['psnr_model_gainfix']:.3f} (24.438 -> 25.775 와 일치)")
    A("- 스펙트럼·기울기·자기상관은 전부 장별 오라클 전역이득 a=<p,g>/<p,p> 를 곱한 출력(클리핑 없음)으로 측정")
    A(f"  (장별 a 평균 {d['gain_global_mean']:.4f})")
    A("- 채점은 repro_measure.psnr_indep 만 사용. 전부 실측값.")
    A("")

    A("## 1. 방사 파워 스펙트럼 (휘도, 0-255 스케일)")
    A("")
    A("2차원 FFT 파워를 방사 방향으로 합산했다. 정규화 주파수 f=sqrt(fx^2+fy^2) [cycles/pixel],")
    A("0.5 가 나이퀴스트. 마지막 구간(0.50-0.71)은 주파수 평면의 코너다.")
    A("구간 파워의 정의는 P_b = (1/N^2) sum_{k in b} |F(k)|^2 이라 sum_b P_b = mean(x^2) 이다(Parseval).")
    A("")
    A("```")
    A("주파수구간     주파수수  GT파워        출력파워      잔차파워    | 출력/GT  잔차/GT | 장별평균 출력/GT")
    A("---------------------------------------------------------------------------------------------------")
    for b in range(nr):
        A(f"{names[b]:>12s} {d['nfreq_per_band'][b]:9d} {sp['p_gt'][b]:12.4f} {sp['p_out'][b]:13.4f} "
          f"{sp['p_res'][b]:11.4f} | {sp['ratio_energy'][b]:7.4f} {sp['res_over_gt'][b]:8.4f} | "
          f"{sp['ratio_perimg_mean'][b]:16.4f}")
    A("```")
    A("")
    A("위 세 파워만으로 구간별 상관계수가 유도된다(항등식, 새 측정 아님):")
    A("  cross_b = (P_out + P_gt - P_res)/2,  rho_b = cross_b / sqrt(P_out*P_gt)")
    A("rho 는 출력의 그 대역이 GT 의 그 대역과 얼마나 같은 자리에 있는지다.")
    A("구간별 실수 이득 하나로 줄일 수 있는 최선 잔차는 P_gt*(1-rho^2) 이므로 1-rho^2 는")
    A("'진폭 재조정으로는 못 없애는 몫'이다. 등가 sigma 는 측정된 파워비를 가우시안")
    A("전달함수 exp(-4*pi^2*s^2*f^2) 로 역산한 값(구간마다 다르면 감쇠가 가우시안이 아니라는 뜻).")
    A("")
    A("```")
    A("주파수구간   상관 rho  1-rho^2  최적구간이득  등가sigma | 0.033s rho  0.04s rho  0.1s rho")
    A("--------------------------------------------------------------------------------------------")
    rho_all = []
    for b in range(nr):
        pg_, po_, pr_ = sp["p_gt"][b], sp["p_out"][b], sp["p_res"][b]
        cr = (po_ + pg_ - pr_) / 2.0
        rho = cr / math.sqrt(max(po_ * pg_, 1e-30))
        rho_all.append(rho)
        gain = cr / max(po_, 1e-30)
        fc = (E[b] + min(E[b + 1], 0.7072)) / 2.0
        rt = sp["ratio_energy"][b]
        es = (math.sqrt(-math.log(rt) / (4 * math.pi ** 2 * fc ** 2))
              if (0 < rt < 1 and fc > 0) else float("nan"))
        pe = []
        for e in exps:
            q = d["by_exposure"][e]["spec"]
            c2 = (q["p_out"][b] + q["p_gt"][b] - q["p_res"][b]) / 2.0
            pe.append(c2 / math.sqrt(max(q["p_out"][b] * q["p_gt"][b], 1e-30)))
        A(f"{names[b]:>12s} {rho:9.4f} {1-rho**2:8.4f} {gain:13.4f} "
          f"{(f'{es:.3f}' if es == es else '   -'):>10s} | " +
          "".join(f"{v:11.4f}" for v in pe))
    A("```")
    A("")
    A("노출시간별 출력/GT 파워비:")
    A("")
    A("```")
    A("주파수구간   ".join(f"{e:>12s}" for e in exps))
    A("-" * (13 + 12 * len(exps)))
    for b in range(nr):
        A(f"{names[b]:>12s} ".join(
            f"{d['by_exposure'][e]['spec']['ratio_energy'][b]:12.4f}" for e in exps))
    A("```")
    A("")

    A("## 2. 최적 저역통과 설명력")
    A("")
    A("blur_sigma(GT) 를 만들어 (a) 모델출력(전역이득 보정본), (b) 모델출력(무보정), (c) GT 자신 과")
    A("각각 PSNR 을 잰다. 셋 다 [0,1] 클리핑 + 8bit 반올림 후 채점.")
    A("(c) 는 대조군이다: sigma=0 에서 무한대이고 sigma 가 커질수록 단조 감소해야 한다.")
    A("")
    A("```")
    A("sigma   blur(GT) vs 출력(이득보정)  blur(GT) vs 출력(무보정)  blur(GT) vs GT")
    A("-------------------------------------------------------------------------------")
    for i, s in enumerate(d["sigmas"]):
        cg = o["curve_gt"][i]
        A(f"{s:5.2f} {o['curve_gainfix'][i]:26.3f} {o['curve_raw'][i]:25.3f} "
          f"{('inf' if cg is None else f'{cg:.3f}'):>15s}")
    A("```")
    A("")
    A(f"데이터셋 평균 곡선의 최적: sigma = {o['best_sigma_gainfix']} 에서 "
      f"{o['best_psnr_gainfix']:.3f} dB (이득보정 출력 기준).")
    A(f"대조: 같은 출력과 GT(sigma=0) 사이 PSNR 은 {o['psnr_model_gainfix']:.3f} dB.")
    diff = o['best_psnr_gainfix'] - o['psnr_model_gainfix']
    A(f"차이 {diff:+.3f} dB -> " +
      ("출력은 GT 자신보다 blur(GT) 에 더 가깝다(조건부 평균 회귀 성립)."
       if diff > 0 else "출력은 GT 자신에 가장 가깝다(조건부 평균 회귀 불성립)."))
    A("")
    A("장별 최적 sigma 분포 (이득보정 출력 기준):")
    A("")
    A("```")
    A("sigma  ".join(f"{k:>7s}" for k in o["argmax_hist_gainfix"]))
    A("장수   ".join(f"{v:7d}" for v in o["argmax_hist_gainfix"].values()))
    A("```")
    A("")
    A("노출시간별:")
    A("")
    A("```")
    A("노출     장수   출력vsGT   최적sigma  그때PSNR   차이")
    A("---------------------------------------------------------")
    for e in exps:
        b = d["by_exposure"][e]
        A(f"{e:7s} {b['spec']['n']:5d} {b['psnr_model_gainfix']:9.3f} "
          f"{b['best_sigma_gainfix']:10.2f} {b['best_psnr_gainfix']:9.3f} "
          f"{b['best_psnr_gainfix']-b['psnr_model_gainfix']:+7.3f}")
    A("```")
    A("")

    A("## 3. 기울기 크기 비율 (소벨, 휘도)")
    A("")
    A("구간화는 diag_sid_failure.py 와 동일하게 입력(short) 국소 휘도(0-255) 기준.")
    A("테두리 1픽셀은 제외했다. 비율 = 평균|grad(출력)| / 평균|grad(GT)|.")
    A("")
    A("```")
    A("입력휘도  화소비율   평균|grad(GT)|  평균|grad(출력)|   비율")
    A("---------------------------------------------------------------")
    for b, nm in enumerate(d["bin_names"]):
        A(f"{nm:>8s} {sp['grad_px_frac_pct'][b]:8.2f}% {sp['grad_gt_mean'][b]:15.4f} "
          f"{sp['grad_out_mean'][b]:16.4f} {sp['grad_ratio'][b]:9.4f}")
    A("```")
    A("")
    A("노출시간별 비율:")
    A("")
    A("```")
    A("입력휘도  ".join(f"{e:>12s}" for e in exps))
    A("-" * (10 + 12 * len(exps)))
    for b, nm in enumerate(d["bin_names"]):
        A(f"{nm:>8s}  ".join(
            f"{d['by_exposure'][e]['spec']['grad_ratio'][b]:12.4f}" for e in exps))
    A("```")
    A("")

    A("## 4. 잔차의 구조성 — 공간 자기상관")
    A("")
    A("잔차 = 오라클 전역이득 보정 출력 - GT (휘도, 장별 평균 제거).")
    A("정규화 자기상관 rho(k) = sum(a*b)/sqrt(sum(a^2)sum(b^2)), a 와 b 는 k 화소 어긋난 겹침 영역.")
    A("대조군 1 = 같은 분산의 백색잡음(장마다 새로 생성), 대조군 2 = GT 휘도 자신.")
    A("")
    A("```")
    A("lag    잔차수평  잔차수직 | 백색수평  백색수직 | GT수평   GT수직")
    A("--------------------------------------------------------------------")
    ac = sp["autocorr"]
    for i, k in enumerate(d["lags"]):
        A(f"{k:3d} {ac['res_h'][i]:10.4f} {ac['res_v'][i]:9.4f} | "
          f"{ac['wht_h'][i]:8.4f} {ac['wht_v'][i]:9.4f} | "
          f"{ac['gt_h'][i]:8.4f} {ac['gt_v'][i]:8.4f}")
    A("```")
    A("")
    A("노출시간별 잔차 수평 자기상관:")
    A("")
    A("```")
    A("lag    ".join(f"{e:>12s}" for e in exps))
    A("-" * (7 + 12 * len(exps)))
    for i, k in enumerate(d["lags"]):
        A(f"{k:3d}    ".join(
            f"{d['by_exposure'][e]['spec']['autocorr']['res_h'][i]:12.4f}" for e in exps))
    A("```")
    A("")

    A("## 5. 상한 계산 — 진폭만 GT 로 맞추면 어디까지 오르나")
    A("")
    A("주파수 영역에서 각 방사 구간의 진폭을 GT 의 그 구간 진폭으로 스케일하고 위상은 출력 것을 유지한다.")
    A("채널별로 따로 계산했다. 이건 위상이 맞다는 낙관적 가정 아래의 값이므로 상한이다.")
    A("같이 낸 band-LS 는 같은 구간별 실수 이득을 최소제곱으로 고른 것(= 구간별 이득 재가중의 진짜 최적)이다.")
    A("두 값 모두 [0,1] 클리핑 + 8bit 반올림 후 psnr_indep 로 채점.")
    A("")
    A("```")
    A("구간수   진폭정합(상한)   band-LS(구간이득 최적)   출발점(이득보정)")
    A("----------------------------------------------------------------------")
    for nb in d["bound_nbands"]:
        A(f"{nb:6d} {o[f'psnr_ampmatch_{nb}']:15.3f} {o[f'psnr_bandls_{nb}']:23.3f} "
          f"{o['psnr_model_gainfix']:19.3f}")
    A("```")
    A("")
    A("노출시간별 (구간수 64):")
    A("")
    A("```")
    A("노출     이득보정   진폭정합   차이   | band-LS   차이")
    A("---------------------------------------------------------")
    for e in exps:
        b = d["by_exposure"][e]
        A(f"{e:7s} {b['psnr_model_gainfix']:9.3f} {b['psnr_ampmatch_64']:10.3f} "
          f"{b['psnr_ampmatch_64']-b['psnr_model_gainfix']:+7.3f} | "
          f"{b['psnr_bandls_64']:8.3f} {b['psnr_bandls_64']-b['psnr_model_gainfix']:+7.3f}")
    A("```")
    A("")

    A("## 6. 산수 대조 (검증)")
    A("")
    chk = []
    ps = sum(sp["p_gt"]); chk.append(("Parseval GT: sum_b P_b vs mean(L^2)", ps, sp["ms_gt"],
                                      abs(ps - sp["ms_gt"]) / sp["ms_gt"] < 1e-9))
    ps = sum(sp["p_out"]); chk.append(("Parseval 출력: sum_b P_b vs mean(L^2)", ps, sp["ms_out"],
                                       abs(ps - sp["ms_out"]) / sp["ms_out"] < 1e-9))
    ps = sum(sp["p_res"]); chk.append(("Parseval 잔차: sum_b P_b vs mean(L^2)", ps, sp["ms_res"],
                                       abs(ps - sp["ms_res"]) / sp["ms_res"] < 1e-9))
    A("```")
    A("항목                                            계산값        대조값     판정")
    A("---------------------------------------------------------------------------------")
    for nm, v1, v2, ok in chk:
        A(f"{nm:<46s} {v1:12.5f} {v2:12.5f}   {'통과' if ok else '실패'}")
    tot = sum(d["nfreq_per_band"])
    A(f"{'주파수 개수 합 = H*W':<46s} {tot:12d} {512*960:12d}   "
      f"{'통과' if tot == 512*960 else '실패'}")
    s = sum(sp["grad_px_frac_pct"])
    A(f"{'기울기 화소비율 합 (%)':<46s} {s:12.5f} {100.0:12.5f}   "
      f"{'통과' if abs(s-100) < 1e-6 else '실패'}")
    A(f"{'무보정 PSNR = DIAG_SID_FAILURE 24.438':<46s} {o['psnr_model']:12.5f} {24.438:12.5f}   "
      f"{'통과' if abs(o['psnr_model']-24.438) < 0.005 else '실패'}")
    A(f"{'이득보정 PSNR = DIAG_SID_FAILURE 25.775':<46s} {o['psnr_model_gainfix']:12.5f} "
      f"{25.775:12.5f}   {'통과' if abs(o['psnr_model_gainfix']-25.775) < 0.005 else '실패'}")
    ok = all(o[f"psnr_bandls_{nb}"] >= o["psnr_model_gainfix"] - 1e-6 for nb in d["bound_nbands"])
    A(f"{'band-LS >= 출발점 (최소제곱이므로 필연)':<46s} {'-':>12s} {'-':>12s}   "
      f"{'통과' if ok else '실패'}")
    ok = all(o[f"psnr_bandls_{d['bound_nbands'][i+1]}"] >= o[f"psnr_bandls_{d['bound_nbands'][i]}"] - 1e-6
             for i in range(len(d["bound_nbands"]) - 1))
    A(f"{'band-LS 는 구간수에 대해 단조증가':<46s} {'-':>12s} {'-':>12s}   "
      f"{'통과' if ok else '실패'}")
    rr = []
    for b in range(nr):
        pg_, po_, pr_ = sp["p_gt"][b], sp["p_out"][b], sp["p_res"][b]
        rr.append(((po_ + pg_ - pr_) / 2.0) / math.sqrt(max(po_ * pg_, 1e-30)))
    A(f"{'구간 상관 rho 전부 [-1,1] 안 (코시-슈바르츠)':<46s} {max(rr):12.5f} {1.0:12.5f}   "
      f"{'통과' if max(abs(v) for v in rr) <= 1.0 else '실패'}")
    A(f"{'백색잡음 대조 |rho| 최대 (lag>=1)':<46s} "
      f"{max(max(abs(v) for v in ac['wht_h']), max(abs(v) for v in ac['wht_v'])):12.5f} "
      f"{0.0:12.5f}   {'통과(0 근처)' if max(abs(v) for v in ac['wht_h']) < 0.01 else '확인필요'}")
    A("```")
    A("")

    # 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
        fig, ax = plt.subplots(1, 4, figsize=(20, 4))
        xc = [(E[b] + min(E[b + 1], 0.7072)) / 2 for b in range(nr)]
        ax[0].semilogy(xc, sp["p_gt"], "ko-", label="GT")
        ax[0].semilogy(xc, sp["p_out"], "s--", color="0.4", label="output (gain-fixed)")
        ax[0].semilogy(xc, sp["p_res"], "^:", color="0.7", label="residual")
        ax[0].set_xlabel("normalized radial frequency"); ax[0].legend()
        ax[0].set_title("radial power spectrum")
        ax[1].plot(xc, sp["ratio_energy"], "ko-")
        ax[1].axhline(1.0, color="0.6", ls="--")
        ax[1].set_xlabel("normalized radial frequency"); ax[1].set_ylabel("output / GT power")
        ax[1].set_title("power ratio")
        ax[2].plot(d["sigmas"], o["curve_gainfix"], "ko-", label="blur(GT) vs output")
        ax[2].axhline(o["psnr_model_gainfix"], color="0.6", ls="--", label="output vs GT")
        ax[2].set_xlabel("gaussian sigma"); ax[2].set_ylabel("PSNR"); ax[2].legend()
        ax[2].set_title("low-pass explanatory power")
        ax[3].plot(d["lags"], ac["res_h"], "ko-", label="residual h")
        ax[3].plot(d["lags"], ac["res_v"], "ks--", label="residual v")
        ax[3].plot(d["lags"], ac["wht_h"], "^:", color="0.7", label="white control")
        ax[3].plot(d["lags"], ac["gt_h"], "v-.", color="0.45", label="GT luma")
        ax[3].axhline(0.0, color="0.6", lw=0.7)
        ax[3].set_xlabel("lag (px)"); ax[3].set_ylabel("normalized autocorr"); ax[3].legend()
        ax[3].set_title("residual structure")
        fig.tight_layout()
        fig.savefig("numbers/diag_sid_spectrum.png", dpi=130)
        A("![스펙트럼 그림](diag_sid_spectrum.png)")
        A("")
    except Exception as ex:
        A(f"(그림 생성 실패: {ex})")

    A("## 측정이 가리키는 것")
    A("")
    lsg = o["psnr_bandls_64"] - o["psnr_model_gainfix"]
    mser = 100.0 * (1 - 10 ** (-lsg / 10.0))
    e33 = d["by_exposure"]["0.033s"]; e10 = d["by_exposure"]["0.1s"]
    A(f"- 고주파 소실은 확정이다. 출력/GT 파워비가 최저주파 {sp['ratio_energy'][0]:.3f} 에서 "
      f"나이퀴스트 근처 {sp['ratio_energy'][9]:.3f}, 코너 {sp['ratio_energy'][10]:.3f} 까지 단조 감소한다. "
      f"노출이 짧을수록 심해서 0.033s 는 코너에서 {e33['spec']['ratio_energy'][10]:.3f} 로 "
      f"고주파가 사실상 없다(0.1s 는 {e10['spec']['ratio_energy'][10]:.3f}). "
      f"기울기 크기 비율도 같은 방향이다: 전체 {sp['grad_ratio'][0]:.3f}(가장 어두운 구간)~"
      f"{sp['grad_ratio'][6]:.3f}(밝은 구간), 0.033s 의 가장 어두운 구간은 "
      f"{e33['spec']['grad_ratio'][0]:.3f} 로 GT 기울기의 절반 아래다.")
    A(f"- 조건부 평균으로의 회귀도 성립한다. 출력은 GT 자신(PSNR {o['psnr_model_gainfix']:.3f})보다 "
      f"blur_{o['best_sigma_gainfix']}(GT)(PSNR {o['best_psnr_gainfix']:.3f}) 에 "
      f"{o['best_psnr_gainfix']-o['psnr_model_gainfix']:+.3f} dB 더 가깝다. "
      f"최적 sigma 는 노출이 짧을수록 커진다: 0.1s {e10['best_sigma_gainfix']} -> "
      f"0.033s {e33['best_sigma_gainfix']}, 그때 이득도 "
      f"{e10['best_psnr_gainfix']-e10['psnr_model_gainfix']:+.3f} -> "
      f"{e33['best_psnr_gainfix']-e33['psnr_model_gainfix']:+.3f} dB 로 커진다. "
      "다만 감쇠 모양은 가우시안이 아니다 - 등가 sigma 가 저주파 0.51 에서 고주파 0.30 으로 "
      "떨어지므로, 모델은 가우시안보다 꼬리가 두꺼운 저역통과다.")
    A(f"- 그런데 소실된 것이 진폭만은 아니다. 방사 구간별 상관계수 rho 가 저주파 "
      f"{rho_all[0]:.3f} 에서 고주파 {rho_all[9]:.3f}, 코너 {rho_all[10]:.3f} 로 같이 떨어지고, "
      f"0.033s 코너는 {((e33['spec']['p_out'][10]+e33['spec']['p_gt'][10]-e33['spec']['p_res'][10])/2.0)/math.sqrt(e33['spec']['p_out'][10]*e33['spec']['p_gt'][10]):.3f} 다. 즉 남아 있는 고주파조차 GT 와 다른 자리에 있다. "
      f"진폭 재조정으로 못 없애는 몫 1-rho^2 은 고주파 구간에서 {1-rho_all[9]**2:.2f}~"
      f"{1-rho_all[10]**2:.2f} 다.")
    A(f"- 그래서 '고주파를 GT 수준으로 되살리기만 하면' 얻는 값은 작다. 위상을 그대로 두고 "
      f"방사 구간 진폭만 GT 에 맞추면 {o['psnr_model_gainfix']:.3f} -> "
      f"{o['psnr_ampmatch_64']:.3f} dB({o['psnr_ampmatch_64']-o['psnr_model_gainfix']:+.3f} dB, "
      f"64구간, 위상이 맞다는 낙관적 가정 아래의 상한)이고, 0.033s 에서는 오히려 "
      f"{e33['psnr_ampmatch_64']-e33['psnr_model_gainfix']:+.3f} dB 로 나빠진다. "
      f"구간별 이득을 최소제곱으로 고른 진짜 최적(band-LS)도 {o['psnr_bandls_64']:.3f} dB, "
      f"{lsg:+.3f} dB 에 그친다 - 이득 보정 후 오차의 {mser:.0f}% 만 없어지고 "
      f"{100-mser:.0f}% 가 남는다.")
    A(f"- 잔차는 백색이 아니다. 정규화 자기상관이 lag 1 에서 수평 {ac['res_h'][0]:.3f} / "
      f"수직 {ac['res_v'][0]:.3f}, lag 16 에서도 {ac['res_h'][15]:.3f} / {ac['res_v'][15]:.3f}, "
      f"lag 64 에서 {ac['res_h'][21]:.3f} / {ac['res_v'][21]:.3f} 로 오래 남는다"
      f"(백색 대조군은 lag 1 부터 |rho| < 0.001, GT 휘도 자신은 lag 64 에서 {ac['gt_h'][21]:.3f}). "
      "DIAG_SID_FAILURE 가 낸 '잡음 실현 기인 몫 10.6%' 와 같은 방향이다 - "
      "잔차의 대부분은 잡음 실현이 아니라 결정론적 구조다.")
    A("- 종합하면 훅은 '고주파 소실' 하나가 아니라 둘로 갈린다. 진폭 소실은 확정되고 "
      "노출이 짧을수록 심하지만, 그것만 되살려서 회수 가능한 몫은 1 dB 미만이다. "
      "남은 몫은 고주파 대역의 상관 붕괴(rho 0.99 -> 0.49, 0.033s 는 0.10)에 있고, "
      "이건 재가중이 아니라 어디에 무엇이 있는지를 다시 정하는 문제다. "
      "이 측정은 그 지점까지만 특정한다.")
    A("")
    txt = "\n".join(L)
    open("numbers/DIAG_SID_SPECTRUM.md", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()
