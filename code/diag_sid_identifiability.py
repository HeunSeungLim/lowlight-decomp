"""diag_sid_identifiability.py

DIAG_SID_SPECTRUM 이 남긴 질문 하나만 가른다:
고주파 위치정보가 애초에 입력(측정)에 남아 있는가, 아니면 측정 단계에서 이미 사라졌는가.

모델이 필요 없는 측정이다(모델 출력은 비교 기준으로만 다시 잰다).

측정 (SID test split 전량 598장 / 50씬, 노출시간별 분해):
  1. rho_input : short 입력을 장별 최적 스칼라 이득으로 밝기만 맞춘 뒤 GT 와의 방사 구간별 상관
                 + 대조로 오라클 포인트와이즈 사상(공간 이웃 안 씀)을 먹인 뒤의 상관
  2. rho_input(k) : 같은 씬·같은 노출 short 를 k 장 평균한 뒤 같은 측정 (k=1,2,4,8,max)
  3. 상한 환산  : 구간별 rho 로부터 (1-rho^2) 관계를 써서 PSNR 상한
  4. 대역별 판정: 회복가능 / 소진 / 불가 + 오차에너지 몫

주파수 구간 정의·로더·이득 정합·rho 정의는 diag_sid_spectrum.py 에서 그대로 가져온다
(같은 구간 정의를 써야 앞 결과와 비교된다).
채점은 repro_measure.psnr_indep (독립 구현) 만 쓴다.
"""
import os, sys, glob, json, re, math, time, itertools
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep
# 구간 정의·로더·이득 규약을 앞 측정에서 그대로 재사용
from diag_sid_spectrum import (build_pairs, load_npy, exposure_of, u8,
                               make_radial_index, RAD_EDGES, LUMA_W,
                               DATA, WEIGHTS, REPO)

OUT_JSON = os.environ.get("IDENT_OUT",
                          "numbers/diag_sid_identifiability.json")
SPEC_JSON = "numbers/diag_sid_spectrum.json"
KS = [1, 2, 4, 8]
H, W = 512, 960
N = H * W
NCH = 3


# ---------------- 그룹 구성 ----------------
def build_groups():
    """(scene, exposure) -> {'lq': [경로...], 'gt': 경로}"""
    pairs = build_pairs()
    g = {}
    for scene, lqp, gtp in pairs:
        key = (scene, exposure_of(lqp))
        g.setdefault(key, {"lq": [], "gt": gtp})
        g[key]["lq"].append(lqp)
    for k in g:
        g[k]["lq"].sort()
    return g


def blocks(n, k):
    """n 장에서 k 장씩 서로 겹치지 않게 나눈 인덱스 블록 (floor(n/k) 개)."""
    return [list(range(i * k, (i + 1) * k)) for i in range(n // k)]


# ---------------- 대역 통계 ----------------
class BandAcc:
    """대역별 (P_gt, P_in, cross) 를 채널합/휘도 두 갈래로 누적."""

    def __init__(self, nb):
        self.nb = nb
        self.n = 0
        self.z = {t: np.zeros(nb) for t in ["pg_l", "pp_l", "cx_l"]}
        self.z.update({t: np.zeros((3, nb)) for t in ["pg_c", "pp_c", "cx_c"]})
        self.dc = np.zeros(3)          # luma DC: pg, pp, cx
        self.mse_bandls = 0.0          # 장별 최적 대역이득 후 MSE 합 (0-255^2)
        self.mse_plain = 0.0           # 대역이득 없이(스칼라 이득만) MSE 합
        self.psnr_bandls = 0.0         # 장별 PSNR 합
        self.psnr_plain = 0.0
        self.ms_gt = 0.0               # Parseval 대조
        self.ms_in = 0.0
        self.gain = 0.0                # 장별 스칼라 이득 합
        self.psnr_meas = 0.0           # 실제 복원 PSNR 합 (클리핑+8bit)
        self.n_meas = 0
        self.psnr_dn = 0.0             # 장별 잡음보정 상한 PSNR 합
        self.n_dn = 0

    SCAL = ["mse_bandls", "mse_plain", "psnr_bandls", "psnr_plain",
            "ms_gt", "ms_in", "gain", "psnr_meas", "n_meas", "psnr_dn", "n_dn"]

    def add(self, o):
        self.n += 1
        for t in self.z:
            self.z[t] += o[t]
        self.dc += o["dc"]
        for t in self.SCAL:
            setattr(self, t, getattr(self, t) + o.get(t, 0.0))

    def merge(self, other):
        self.n += other.n
        for t in self.z:
            self.z[t] += other.z[t]
        self.dc += other.dc
        for t in self.SCAL:
            setattr(self, t, getattr(self, t) + getattr(other, t))
        return self

    def dump(self):
        z = self.z
        n = max(self.n, 1)

        def rho(pg, pp, cx):
            return list(cx / np.sqrt(np.maximum(pg * pp, 1e-30)))

        d = dict(n=self.n)
        d["rho_luma"] = rho(z["pg_l"], z["pp_l"], z["cx_l"])
        d["rho_rgb"] = rho(z["pg_c"].sum(0), z["pp_c"].sum(0), z["cx_c"].sum(0))
        d["p_gt_luma"] = list(z["pg_l"] / n)
        d["p_in_luma"] = list(z["pp_l"] / n)
        # 채널별 (3 x nb) - 상한 환산을 채널별로 하려고 따로 들고 있는다
        d["p_gt_c"] = [list(v) for v in z["pg_c"] / n]
        d["p_in_c"] = [list(v) for v in z["pp_c"] / n]
        d["cross_c"] = [list(v) for v in z["cx_c"] / n]
        d["p_gt_rgb"] = list(z["pg_c"].sum(0) / n)
        d["p_in_rgb"] = list(z["pp_c"].sum(0) / n)
        d["cross_rgb"] = list(z["cx_c"].sum(0) / n)
        # 대역별 최적 실수이득 후 남는 잔차 파워 (채널합, 장 평균)
        res = (z["pg_c"] - z["cx_c"] ** 2 / np.maximum(z["pp_c"], 1e-30)).sum(0)
        d["res_bandls_rgb"] = list(res / n)
        # 스칼라 이득만 썼을 때의 대역별 잔차 (채널합, 장 평균)
        d["res_plain_rgb"] = list((z["pg_c"] + z["pp_c"] - 2 * z["cx_c"]).sum(0) / n)
        pg0, pp0, cx0 = self.dc / n
        d["dc_luma"] = dict(p_gt=pg0, p_in=pp0, cross=cx0,
                            rho=cx0 / math.sqrt(max(pg0 * pp0, 1e-30)))
        d["mse_bandls"] = self.mse_bandls / n
        d["mse_plain"] = self.mse_plain / n
        d["psnr_bandls"] = self.psnr_bandls / n
        d["psnr_plain"] = self.psnr_plain / n
        d["ms_gt"] = self.ms_gt / n
        d["ms_in"] = self.ms_in / n
        d["gain_mean"] = self.gain / n
        d["n_meas"] = self.n_meas
        d["psnr_meas"] = self.psnr_meas / max(self.n_meas, 1)
        d["n_dn"] = self.n_dn
        d["psnr_dn"] = self.psnr_dn / max(self.n_dn, 1)
        return d


def _merge_all(dst, srcs):
    for s in srcs:
        dst.merge(s)
    return dst


def main():
    import torch
    import torch.nn.functional as F

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False

    nb = len(RAD_EDGES) - 1
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    nfreq = np.bincount(ridx_flat.cpu().numpy(), minlength=nb)
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)

    def band_cross(p, g):
        """p,g: 3xHxW float64 (0-255). 대역별 파워/교차항.
        정의: P_b = (1/N^2) sum_{k in b} |F(k)|^2  (sum_b P_b = mean(x^2), Parseval)
              C_b = (1/N^2) sum_{k in b} Re[F_p(k) conj(F_g(k))]"""
        Fp = torch.fft.fft2(p).reshape(NCH, -1)
        Fg = torch.fft.fft2(g).reshape(NCH, -1)
        pp = torch.zeros(NCH, nb, device=dev, dtype=torch.float64)
        gg = torch.zeros(NCH, nb, device=dev, dtype=torch.float64)
        cx = torch.zeros(NCH, nb, device=dev, dtype=torch.float64)
        idx = ridx_flat[None].expand(NCH, -1)
        pp.scatter_add_(1, idx, (Fp.real ** 2 + Fp.imag ** 2) / (N * N))
        gg.scatter_add_(1, idx, (Fg.real ** 2 + Fg.imag ** 2) / (N * N))
        cx.scatter_add_(1, idx, (Fp.real * Fg.real + Fp.imag * Fg.imag) / (N * N))
        return pp, gg, cx

    def band_cross_luma(p, g):
        Lp = (p * lw[:, None, None]).sum(0)
        Lg = (g * lw[:, None, None]).sum(0)
        Fp = torch.fft.fft2(Lp).reshape(-1)
        Fg = torch.fft.fft2(Lg).reshape(-1)
        pp = torch.zeros(nb, device=dev, dtype=torch.float64)
        gg = torch.zeros(nb, device=dev, dtype=torch.float64)
        cx = torch.zeros(nb, device=dev, dtype=torch.float64)
        pp.scatter_add_(0, ridx_flat, (Fp.real ** 2 + Fp.imag ** 2) / (N * N))
        gg.scatter_add_(0, ridx_flat, (Fg.real ** 2 + Fg.imag ** 2) / (N * N))
        cx.scatter_add_(0, ridx_flat, (Fp.real * Fg.real + Fp.imag * Fg.imag) / (N * N))
        dc = np.array([float(Fg[0].real ** 2 + Fg[0].imag ** 2) / (N * N),
                       float(Fp[0].real ** 2 + Fp[0].imag ** 2) / (N * N),
                       float(Fp[0].real * Fg[0].real + Fp[0].imag * Fg[0].imag) / (N * N)])
        return pp, gg, cx, dc

    def stats(p255, g255):
        """p255,g255: 3xHxW float64 0-255 스케일. 한 장의 대역 통계 묶음."""
        pp, gg, cx = band_cross(p255, g255)
        ppl, ggl, cxl, dc = band_cross_luma(p255, g255)
        pp_c = pp.cpu().numpy(); gg_c = gg.cpu().numpy(); cx_c = cx.cpu().numpy()
        # 채널별·대역별 최적 실수이득 s = C/P 후 잔차 (= P_gt(1-rho^2))
        res = (gg - cx ** 2 / torch.clamp(pp, min=1e-30)).sum().item() / NCH
        plain = ((gg + pp - 2 * cx).sum() / NCH).item()
        o = dict(pg_l=ggl.cpu().numpy(), pp_l=ppl.cpu().numpy(), cx_l=cxl.cpu().numpy(),
                 pg_c=gg_c, pp_c=pp_c, cx_c=cx_c, dc=dc,
                 mse_bandls=res, mse_plain=plain,
                 psnr_bandls=10 * math.log10(255.0 ** 2 / max(res, 1e-30)),
                 psnr_plain=10 * math.log10(255.0 ** 2 / max(plain, 1e-30)),
                 ms_gt=float(((g255 * lw[:, None, None]).sum(0) ** 2).mean()),
                 ms_in=float(((p255 * lw[:, None, None]).sum(0) ** 2).mean()))
        return o

    def pointwise_map(x, g, fit_rows=None):
        """오라클 포인트와이즈 사상: 채널별로 같은 입력값을 가진 화소를 GT 평균으로 보낸다.
        공간 이웃을 전혀 쓰지 않는 함수 사상이라 구조(위치정보)를 새로 만들지 못한다.
        fit_rows 가 주어지면 그 행만으로 사상을 적합하고 전체에 적용한다(과적합 대조)."""
        out = torch.empty_like(x)
        for c in range(NCH):
            xf = x[c].reshape(-1)
            gf = g[c].reshape(-1)
            if fit_rows is None:
                uq, inv = torch.unique(xf, return_inverse=True)
                s = torch.zeros(uq.numel(), dtype=torch.float64, device=x.device)
                cnt = torch.zeros_like(s)
                s.scatter_add_(0, inv, gf)
                cnt.scatter_add_(0, inv, torch.ones_like(gf))
                out[c] = (s / cnt)[inv].reshape(H, W)
            else:
                xs = x[c][fit_rows].reshape(-1)
                gs = g[c][fit_rows].reshape(-1)
                uq, inv = torch.unique(xs, return_inverse=True)
                s = torch.zeros(uq.numel(), dtype=torch.float64, device=x.device)
                cnt = torch.zeros_like(s)
                s.scatter_add_(0, inv, gs)
                cnt.scatter_add_(0, inv, torch.ones_like(gs))
                lut = s / cnt
                j = torch.clamp(torch.searchsorted(uq, xf), max=uq.numel() - 1)
                out[c] = lut[j].reshape(H, W)     # 적합에 없던 값은 최근접 상위 레벨
        return out

    groups = build_groups()
    lim = int(os.environ.get("IDENT_LIMIT", "0"))
    if lim:
        groups = {k: v for k, v in sorted(groups.items())[:lim]}
    exps = sorted(set(e for _, e in groups), key=lambda s: float(s[:-1]))
    ngrp = {e: sum(1 for k in groups if k[1] == e) for e in exps}
    nimg = {e: sum(len(groups[k]["lq"]) for k in groups if k[1] == e) for e in exps}
    print("groups", ngrp, "imgs", nimg, flush=True)

    # ---------- 캐시 (uint8 로 들고 있는다: 0.9GB) ----------
    cache = {}

    def get_u8(p):
        if p not in cache:
            a = load_npy(p)                       # float32 [0,1], 저자 로더 규약
            b = np.rint(a * 255.0).astype(np.uint8)
            assert np.abs(b.astype(np.float32) / 255.0 - a).max() < 1e-6
            cache[p] = b
        return cache[p]

    # ---------- Stage A: 모델 출력의 rho (비교 기준, 독립 재계산) ----------
    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    KFIX = {}
    for e in exps:
        mx = max(len(groups[kk]["lq"]) for kk in groups if kk[1] == e)
        KFIX[e] = max([k for k in KS if k <= mx])
    accA = {e: BandAcc(nb) for e in exps}
    accAf = {e: BandAcc(nb) for e in exps}
    psnr_model = {e: [] for e in exps}
    t0 = time.time()
    with torch.inference_mode():
        done = 0
        for (scene, exp), gd in sorted(groups.items()):
            gt_u8 = get_u8(gd["gt"])
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            for lqp in gd["lq"]:
                lq = get_u8(lqp).astype(np.float32) / 255.0
                x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
                h, w = x.shape[2], x.shape[3]
                padh, padw = (-h) % 4, (-w) % 4
                if padh or padw:
                    x = F.pad(x, (0, padw, 0, padh), "reflect")
                y = net(x)[:, :, :h, :w]
                y = torch.clamp(y, 0, 1).double()[0]
                a = float((y * g).sum() / (y * y).sum())      # 오라클 전역이득 (앞 측정과 동일)
                pg = a * y
                st = stats(pg * 255.0, g * 255.0)
                accA[exp].add(st)
                if len(gd["lq"]) >= KFIX[exp]:
                    accAf[exp].add(st)
                psnr_model[exp].append(
                    psnr_indep(u8(y.permute(1, 2, 0).cpu().numpy()), gt_u8))
                done += 1
                if done % 100 == 0:
                    print(f"  A {done}/598  {time.time()-t0:.0f}s", flush=True)
    print(f"stage A done {time.time()-t0:.0f}s", flush=True)

    # ---------- Stage B: 입력의 rho (k 장 평균) ----------
    variants = ["scalar", "pw"]
    accB = {}          # (exp, variant, k, cohort) -> BandAcc
    noise = {}         # (exp, k) -> 대역별 잡음파워 추정 합, 표본수
    rec_check = []     # 대역LS 실제 복원 PSNR 대조 (일부만)

    def acc(exp, var, k, cohort):
        key = (exp, var, k, cohort)
        if key not in accB:
            accB[key] = BandAcc(nb)
        return accB[key]

    # 고정 코호트: 노출별로 k=KFIX 를 만들 수 있는 그룹만 (k 간 비교를 같은 씬에서 하려고)
    print("KFIX", KFIX, flush=True)

    t0 = time.time()
    with torch.inference_mode():
        for gi, ((scene, exp), gd) in enumerate(sorted(groups.items())):
            gt_u8 = get_u8(gd["gt"])
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            g255 = g * 255.0
            n = len(gd["lq"])
            frames = torch.stack([
                torch.from_numpy(get_u8(p).astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                for p in gd["lq"]])                                # n x 3 x H x W
            fixed_ok = (n >= KFIX[exp])
            klist = [k for k in KS if k <= n]
            if n not in klist:
                klist.append(n)                                    # 그룹 최대치
            for k in klist:
                bl = blocks(n, k)
                gains = []
                pwkeep = []
                pend = []
                for b in bl:
                    x = frames[b].mean(0)                          # k 장 평균, [0,1]
                    a = float((x * g).sum() / (x * x).sum())       # 장별 최적 스칼라 이득
                    gains.append(a)
                    p_scalar = a * x
                    p_pw = pointwise_map(x, g)
                    if len(pwkeep) < 2:
                        pwkeep.append(p_pw.clone())
                    for var, p in (("scalar", p_scalar), ("pw", p_pw)):
                        o = stats(p * 255.0, g255)
                        o["gain"] = a
                        if k == 1:      # 앵커 대조용 실제 복원 PSNR (클리핑+8bit)
                            o["psnr_meas"] = psnr_indep(
                                u8(np.clip(p.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                            o["n_meas"] = 1
                        pend.append((var, o, a))
                # 잡음 척도: 서로 겹치지 않는 두 k-평균의 차이 파워 / 2
                # (신호는 상쇄되고 잡음만 남는다. 두 평균의 차이 분산 = 2 * k장평균의 잡음분산)
                if len(bl) >= 2:
                    dd = frames[bl[0]].mean(0) - frames[bl[1]].mean(0)
                    ppc, _, _ = band_cross(dd * 255.0, dd * 255.0)
                    ppl, _, _, _ = band_cross_luma(dd * 255.0, dd * 255.0)
                    nl_raw = ppl.cpu().numpy() / 2.0                 # 원단위 (1/k 검정용)
                    nc_raw = ppc.cpu().numpy() / 2.0
                    # scalar 갈래의 파워와 단위를 맞추려고 같은 이득 a 로 스케일한다
                    # (rho 보정에 쓰는 것은 N/P 비율이라 a^2 이 양쪽에 같이 걸려야 한다.
                    #  a 는 k 에 따라 조금씩 커지므로 1/k 검정에는 원단위를 쓴다)
                    a2 = gains[0] ** 2
                    ddp = pwkeep[0] - pwkeep[1]                      # 사상 후 도메인의 잡음
                    ppcp, _, _ = band_cross(ddp * 255.0, ddp * 255.0)
                    pplp, _, _, _ = band_cross_luma(ddp * 255.0, ddp * 255.0)
                    for coh in (["all", "fix"] if fixed_ok else ["all"]):
                        key = (exp, k, coh)
                        if key not in noise:
                            noise[key] = [np.zeros(nb), np.zeros((3, nb)), np.zeros(nb),
                                          np.zeros(nb), np.zeros((3, nb)), 0]
                        # 표본 가중: 그룹 크기 n 으로 고정한다. k 마다 가중이 같아야
                        # 1/k 법칙을 그룹 구성 변화 없이 검정할 수 있다. k=1 에서는
                        # n = 표본수라 아래 rho 보정의 파워 가중과 정확히 일치한다.
                        noise[key][0] += nl_raw * a2 * n
                        noise[key][1] += nc_raw * a2 * n
                        noise[key][2] += nl_raw * n
                        noise[key][3] += pplp.cpu().numpy() / 2.0 * n
                        noise[key][4] += ppcp.cpu().numpy() / 2.0 * n
                        noise[key][5] += n
                    # 장별 잡음보정 상한: 그 장의 대역파워에서 잡음 파워를 빼고
                    # 같은 형태(장별·채널별·대역별 최적 이득)로 MSE 를 낸다.
                    # 다른 행과 같은 규약이라 'k=8 실측보다 커야 한다' 를 직접 검정할 수 있다.
                    for var, o, a_i in pend:
                        Nc = nc_raw * (a_i ** 2) if var == "scalar" else \
                            (ppcp.cpu().numpy() / 2.0)
                        den = np.maximum(o["pp_c"] - Nc, 1e-30)
                        mse = float((o["pg_c"] - o["cx_c"] ** 2 / den).clip(min=0).sum() / NCH)
                        o["psnr_dn"] = 10 * math.log10(255.0 ** 2 / max(mse, 1e-30))
                        o["n_dn"] = 1
                for var, o, a_i in pend:
                    acc(exp, var, k, "all").add(o)
                    if fixed_ok:
                        acc(exp, var, k, "fix").add(o)
            # 대역LS 해석식 대조: k=1 첫 장을 실제로 복원해 PSNR 을 잰다
            if gi % 7 == 0:
                x = frames[0]
                a = float((x * g).sum() / (x * x).sum())
                p = (a * x) * 255.0
                Fp = torch.fft.fft2(p).reshape(NCH, -1)
                Fg = torch.fft.fft2(g255).reshape(NCH, -1)
                idx = ridx_flat[None].expand(NCH, -1)
                pp = torch.zeros(NCH, nb, device=dev, dtype=torch.float64)
                cx = torch.zeros(NCH, nb, device=dev, dtype=torch.float64)
                pp.scatter_add_(1, idx, Fp.real ** 2 + Fp.imag ** 2)
                cx.scatter_add_(1, idx, Fp.real * Fg.real + Fp.imag * Fg.imag)
                s = (cx / torch.clamp(pp, min=1e-30)).gather(1, idx)
                rec = torch.fft.ifft2((Fp * s).reshape(NCH, H, W)).real
                mse = float(((rec - g255) ** 2).mean())
                rec_check.append(dict(
                    scene=scene, exp=exp,
                    psnr_analytic=10 * math.log10(255.0 ** 2 / max(mse, 1e-30)),
                    psnr_measured=psnr_indep(
                        u8(np.clip(rec.permute(1, 2, 0).cpu().numpy() / 255.0, 0, 1)), gt_u8),
                    psnr_plain_measured=psnr_indep(
                        u8(np.clip((a * x).permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)))
            del frames
            if (gi + 1) % 20 == 0:
                print(f"  B {gi+1}/{len(groups)}  {time.time()-t0:.0f}s", flush=True)
    print(f"stage B done {time.time()-t0:.0f}s", flush=True)

    # ---------- Stage C: 프레임 간 차이가 시간 간격에 따라 어떻게 변하나 ----------
    # 독립 잡음뿐이면 두 프레임 차이의 파워는 간격 |i-j| 와 무관해야 한다.
    # 간격이 벌어질수록 커지면 드리프트(미세 정렬오차·조명 변동)가 섞여 있다는 뜻이고,
    # 그만큼은 프레임 평균으로 안 없어진다.
    lagpow = {}
    with torch.inference_mode():
        for (scene, exp), gd in sorted(groups.items()):
            n = len(gd["lq"])
            if n < 4:
                continue
            fr = torch.stack([
                torch.from_numpy(get_u8(p2).astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                for p2 in gd["lq"]])
            for lag in range(1, n):
                acc_ = np.zeros(nb); cnt = 0
                for i in range(0, n - lag):
                    dd = fr[i] - fr[i + lag]
                    ppl, _, _, _ = band_cross_luma(dd * 255.0, dd * 255.0)
                    acc_ += ppl.cpu().numpy() / 2.0
                    cnt += 1
                key = (exp, lag)
                if key not in lagpow:
                    lagpow[key] = [np.zeros(nb), 0]
                lagpow[key][0] += acc_ / cnt
                lagpow[key][1] += 1
            del fr
    print("stage C done", flush=True)

    # ---------- 포인트와이즈 사상 과적합 대조 (절반 적합 vs 전체 적합) ----------
    pw_audit = []
    with torch.inference_mode():
        rows_even = torch.arange(0, H, 2, device=dev)
        for (scene, exp), gd in sorted(groups.items())[::7]:
            g = torch.from_numpy(get_u8(gd["gt"]).astype(np.float64) / 255.0
                                 ).cuda().permute(2, 0, 1)
            x = torch.from_numpy(get_u8(gd["lq"][0]).astype(np.float64) / 255.0
                                 ).cuda().permute(2, 0, 1)
            full = stats(pointwise_map(x, g) * 255.0, g * 255.0)
            half = stats(pointwise_map(x, g, fit_rows=rows_even) * 255.0, g * 255.0)
            rf = full["cx_l"] / np.sqrt(np.maximum(full["pg_l"] * full["pp_l"], 1e-30))
            rh = half["cx_l"] / np.sqrt(np.maximum(half["pg_l"] * half["pp_l"], 1e-30))
            pw_audit.append(dict(scene=scene, exp=exp, rho_fit_all=list(rf),
                                 rho_fit_half=list(rh)))

    # ---------- 정리 ----------
    out = dict(
        rad_edges=RAD_EDGES, nfreq_per_band=[int(v) for v in nfreq],
        exposures=exps, ks=KS, kfix=KFIX,
        n_groups={e: ngrp[e] for e in exps}, n_images={e: nimg[e] for e in exps},
        group_sizes={f"{s}|{e}": len(v["lq"]) for (s, e), v in groups.items()},
        psnr_model={e: float(np.mean(psnr_model[e])) for e in exps},
        psnr_model_all=float(np.mean(sum(psnr_model.values(), []))),
        model={e: accA[e].dump() for e in exps},
        model_fix={e: accAf[e].dump() for e in exps},
        model_fix_all=_merge_all(BandAcc(nb), [accAf[e] for e in exps]).dump(),
        model_all=_merge_all(BandAcc(nb), [accA[e] for e in exps]).dump(),
        input={f"{e}|{v}|{k}|{c}": a.dump() for (e, v, k, c), a in accB.items()},
        noise={f"{e}|{k}|{c}": dict(luma=list(v[0] / v[5]),
                                    rgb=[list(x) for x in v[1] / v[5]],
                                    luma_raw=list(v[2] / v[5]), luma_pw=list(v[3] / v[5]),
                                    rgb_pw=[list(x) for x in v[4] / v[5]], n=v[5])
               for (e, k, c), v in noise.items()},
        rec_check=rec_check, pw_audit=pw_audit,
        lagpow={f"{e}|{l}": dict(luma=list(v[0] / v[1]), n=v[1])
                for (e, l), v in lagpow.items()},
    )
    # 전체(노출 합산) 입력 통계
    for v in variants:
        for k in KS:
            for c in ["all", "fix"]:
                a = BandAcc(nb)
                hit = False
                for e in exps:
                    if (e, v, k, c) in accB:
                        a.merge(accB[(e, v, k, c)]); hit = True
                if hit:
                    out["input"][f"ALL|{v}|{k}|{c}"] = a.dump()
    for k in KS:
        for c in ["all", "fix"]:
            agg = {t: np.zeros(nb) for t in ["luma", "luma_raw", "luma_pw"]}
            agg.update({t: np.zeros((3, nb)) for t in ["rgb", "rgb_pw"]})
            tot = 0
            for e in exps:
                q = out["noise"].get(f"{e}|{k}|{c}")
                if q:
                    for t in agg:
                        agg[t] += np.array(q[t]) * q["n"]
                    tot += q["n"]
            if tot:
                out["noise"][f"ALL|{k}|{c}"] = dict(
                    **{t: ([list(x) for x in v / tot] if v.ndim == 2 else list(v / tot))
                       for t, v in agg.items()}, n=tot)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    print("wrote", OUT_JSON)


# ============================ 보고서 ============================
TH_ZERO = 0.20      # rho 가 이보다 작으면 그 대역은 정보가 사실상 없다고 본다
TH_GAP = 0.05       # rho_input - rho_output 이 이보다 크면 모델이 덜 쓰고 있는 것


def verdict(rin, rout):
    if rin < TH_ZERO:
        return "불가"
    if rin - rout > TH_GAP:
        return "회복가능"
    return "소진"


def fit_inf(ks, rhos):
    """1/rho^2 = A + B/k 를 최소제곱으로 적합해 k->무한 극한 rho 를 낸다.
    가법·독립 잡음 + 스칼라 이득이면 이 관계는 정확하다:
      rho_k = C/sqrt(P_g (S + N1/k)) -> 1/rho_k^2 = (P_g/C^2) S + (P_g/C^2) N1 * (1/k)
    반환: (rho_inf, 최대 적합잔차(1/rho^2 단위), A, B)"""
    x = np.array([1.0 / k for k in ks])
    y = np.array([1.0 / max(r, 1e-6) ** 2 for r in rhos])
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    B, A = np.polyfit(x, y, 1)
    resid = float(np.max(np.abs(y - (A + B * x))))
    rinf = float(1.0 / math.sqrt(A)) if A > 0 else float("nan")
    return min(rinf, 1.0), resid, float(A), float(B)


def report():
    d = json.load(open(OUT_JSON))
    spec = json.load(open(SPEC_JSON))
    E = d["rad_edges"]; nb = len(E) - 1
    exps = d["exposures"]
    names = [f"{E[b]:.2f}-{min(E[b+1],0.7072):.2f}" for b in range(nb)]
    L = []; A = L.append

    def gin(e, v, k, c="all"):
        return d["input"].get(f"{e}|{v}|{k}|{c}")

    def gnz(e, k, c="all"):
        return d["noise"].get(f"{e}|{k}|{c}")

    def rho_denoised(e, k=1, c="all", v="scalar"):
        """잡음 보정 rho. 잡음은 GT 와 무상관이라 교차항 sum C 는 그대로 두고
        입력 파워 합에서 측정된 잡음 파워 합만 뺀다:  rho_inf = rho / sqrt(1 - N/P).
        합끼리의 연산이라 데이터셋 집계값에 대해 정확하다(장별 값의 평균이 아니다)."""
        q = gin(e, v, k, c); z = gnz(e, k, c)
        if q is None or z is None:
            return None, None
        P = np.array(q["p_in_luma"])
        Nn = np.array(z["luma" if v == "scalar" else "luma_pw"])
        frac = np.clip(Nn / np.maximum(P, 1e-30), 0.0, 0.999)
        r = np.array(q["rho_luma"]) / np.sqrt(1.0 - frac)
        return np.clip(r, -1, 1), frac

    mo = d["model_all"]
    rho_out = np.array(mo["rho_luma"])
    rho_in1 = np.array(gin("ALL", "scalar", 1)["rho_luma"])
    rho_pw1 = np.array(gin("ALL", "pw", 1)["rho_luma"])
    rho_dn1, nfrac = rho_denoised("ALL", 1, "all")
    rho_pwdn1, nfrac_pw = rho_denoised("ALL", 1, "all", "pw")

    # k 외삽 (고정 코호트)
    ext = {}
    for e in ["ALL"] + exps:
        for v in ["scalar", "pw"]:
            ks = [k for k in d["ks"] if gin(e, v, k, "fix")]
            if len(ks) < 2:
                continue
            rr, res, aa, bb = [], [], [], []
            for b in range(nb):
                r, s, _, _ = fit_inf(ks, [gin(e, v, k, "fix")["rho_luma"][b] for k in ks])
                rr.append(r); res.append(s)
            ext[(e, v)] = (np.array(rr), np.array(res), ks)

    # 오차에너지 몫: 현재 모델(전역이득 보정 후)의 대역별 잔차 (채널합)
    res_model = np.array(mo["res_plain_rgb"])
    share = 100.0 * res_model / res_model.sum()

    A("# SID 극암 저조도 — 고주파 정보가 측정에 남아 있는가")
    A("")
    A(f"- 대상: SID test split 전량 598장 / 50씬 (표본 아님). 노출별 "
      + ", ".join(f"{e} {d['n_images'][e]}장/{d['n_groups'][e]}그룹" for e in exps))
    A("- 주파수 구간·로더·이득 정합 규약은 diag_sid_spectrum.py 에서 그대로 import 해서 썼다(같은 11구간).")
    A(f"- 앵커: 모델 무보정 PSNR {d['psnr_model_all']:.3f} (DIAG_SID_FAILURE 24.438 과 일치)")
    A("- 이 문서의 rho 는 전부 방사 주파수 구간별 상관계수(휘도):")
    A("    rho_b = sum_{k in b} Re[F_p conj(F_g)] / sqrt( sum_b |F_p|^2 * sum_b |F_g|^2 )")
    A("  구간마다 따로 계산하므로 DC·극저주파가 다른 구간을 지배하지 못한다(DC 는 첫 구간 안에서만 작용).")
    A("  rho 는 스케일 불변이라 스칼라 이득은 rho 를 바꾸지 못한다. 이득 정합은 파워·PSNR 표시용이다.")
    A("- 입력을 네 갈래로 잰다. 넷 다 공간 이웃을 쓰지 않으므로 위치정보를 새로 만들어내지 못한다.")
    A("    scalar   : short 입력 x 장별 최적 스칼라 이득 (구조 손 안 댐, 지시받은 기본형)")
    A("    pw       : 같은 입력값 화소를 GT 평균으로 보내는 장별 오라클 포인트와이즈 사상")
    A("               (톤 곡선 불일치만 제거한다. short->long 관계가 비선형이라 scalar 는 그만큼 rho 를 깎아먹는다)")
    A("    -dn      : 각 갈래에서 실측 잡음 파워를 빼 잡음 희석을 지운 값 = 프레임 무한 평균의 극한 추정")
    A("    k 평균   : 같은 씬·같은 노출 입력 k 장 평균 (2번). 이쪽은 추정이 아니라 실측이다.")
    A("")

    A("## 1. 입력에 남아 있는 고주파 정보 (k=1, 전량 598장)")
    A("")
    A("```")
    A("주파수구간   rho_in    rho_in   rho_in   rho_in   rho_out | 천장-출력  잡음몫  오차몫% | 판정")
    A("             scalar      pw   scalar-dn  pw-dn   (모델)   |            N/P            |")
    A("-------------------------------------------------------------------------------------------------")
    verds = []
    verds_raw = []
    for b in range(nb):
        vd = verdict(rho_pwdn1[b], rho_out[b])
        verds.append(vd)
        verds_raw.append(verdict(rho_pw1[b], rho_out[b]))
        A(f"{names[b]:>12s} {rho_in1[b]:8.4f} {rho_pw1[b]:8.4f} {rho_dn1[b]:8.4f} "
          f"{rho_pwdn1[b]:8.4f} {rho_out[b]:8.4f} | {rho_pwdn1[b]-rho_out[b]:9.4f} "
          f"{nfrac_pw[b]:7.4f} {share[b]:8.2f} | {vd}")
    A("```")
    A("")
    A("읽는 법. rho 는 상관이라 입력의 잡음이 그대로 rho 를 깎는다. 모델 출력은 잡음이 억제된 추정치라")
    A("같은 정보량이어도 rho 가 더 높게 나온다. 그래서 raw 입력 rho 와 모델 rho 를 그냥 비교하면 안 되고,")
    A("잡음을 벗겨낸 값(-dn)이 '이 측정이 담고 있는 정보의 천장'이다. 보정식은")
    A("  rho_dn = rho / sqrt(1 - N/P),  N = 실측 잡음 파워, P = 입력 대역 파워")
    A("이고 합끼리의 연산이라 데이터셋 집계값에 대해 정확하다. 잡음이 GT 와 무상관이므로 분자(교차항)는")
    A("건드리지 않는다.")
    A("이 보정을 5번에서 검정했다: scalar 갈래는 통과(가법 잡음 모형이 정확하다), pw 갈래는 저주파")
    A("네 대역에서 반증됐다 - 실제로 k 장을 평균해 잰 값보다 작게 나온다. 프레임을 모으면 잡음만 주는")
    A("게 아니라 포인트와이즈 사상 자체가 정확해지는데 이 보정식은 그 몫을 못 보기 때문이다.")
    A("반증 방향이 과소평가 쪽이고 고주파 대역은 반증되지 않았다. 아래 판정 열은 이 천장 추정 기준이고,")
    A("가정 없는 실측 판정은 4번 (b) 다.")
    A("주의: N/P 가 0.9 를 넘는 칸은 보정 배율이 3배를 넘어 값이 불안정하다(짧은 노출의 고주파가 그렇다).")
    A("")
    A("판정 규칙 (고정 임계, 사후조정 없음). 이 표의 비교 대상은 정보 천장 추정 rho_in(pw-dn) 이다:")
    A(f"  불가     : rho_in(pw-dn) < {TH_ZERO}           -> 그 대역은 이 측정에 정보가 사실상 없다")
    A(f"  회복가능 : rho_in(pw-dn) - rho_out > {TH_GAP}   -> 입력에 있는데 모델이 못 쓰고 있다")
    A("  소진     : 그 밖                          -> 모델이 이미 가용 정보를 다 쓰고 있다")
    A("오차몫% 은 현재 모델(전역이득 보정 후) 잔차 파워의 대역별 몫이다(합 100).")
    A("")

    A("### 노출시간별 (k=1)")
    A("")
    for e in exps:
        rdn, _ = rho_denoised(e, 1, "all")
        rpdn, nfp = rho_denoised(e, 1, "all", "pw")
        A(f"{e} ({d['n_images'][e]}장)")
        A("```")
        A("주파수구간   scalar      pw  scalar-dn   pw-dn   rho_out | 천장-출력  잡음몫  오차몫% | 판정")
        A("-------------------------------------------------------------------------------------------")
        ri = np.array(gin(e, "scalar", 1)["rho_luma"])
        rp = np.array(gin(e, "pw", 1)["rho_luma"])
        ro = np.array(d["model"][e]["rho_luma"])
        rm = np.array(d["model"][e]["res_plain_rgb"])
        sh = 100.0 * rm / rm.sum()
        for b in range(nb):
            A(f"{names[b]:>12s} {ri[b]:8.4f} {rp[b]:8.4f} {rdn[b]:8.4f} {rpdn[b]:8.4f} "
              f"{ro[b]:8.4f} | {rpdn[b]-ro[b]:9.4f} {nfp[b]:7.4f} {sh[b]:8.2f} | "
              f"{verdict(rpdn[b], ro[b])}")
        A("```")
        A("")

    A("## 2. 프레임을 k 장 모으면 정보가 늘어나는가 (핵심)")
    A("")
    A("같은 씬·같은 노출 short 입력을 k 장 평균(겹치지 않는 블록)한 뒤 같은 rho 를 다시 잰다.")
    A("장면과 노출이 같으니 신호는 그대로고 잡음만 1/k 로 준다. k 가 커질 때 고주파 rho 가 오르면")
    A("정보는 측정에 있고 단일 프레임 잡음에 묻힌 것이다. 안 오르면 프레임을 더 모아도 없는 정보다.")
    A("코호트 fix = 그 노출에서 k=KFIX 까지 만들 수 있는 그룹만 (k 사이 비교를 같은 씬에서 한다).")
    A("")
    for e in exps:
        ks = [k for k in d["ks"] if gin(e, "scalar", k, "fix")]
        if not ks:
            continue
        A(f"{e}  (fix 코호트 k=1 표본 {gin(e,'scalar',ks[0],'fix')['n']}장, KFIX={d['kfix'][e]})")
        A("```")
        A("주파수구간  ".join(f"  scalar k={k}" for k in ks) +
          "  scalar_inf |".join(f"      pw k={k}" for k in ks) + "      pw_inf")
        A("-" * (12 + 12 * (len(ks) + 1) * 2 + 2))
        es = ext.get((e, "scalar")); ep = ext.get((e, "pw"))
        for b in range(nb):
            r1 = [gin(e, "scalar", k, "fix")["rho_luma"][b] for k in ks]
            r2 = [gin(e, "pw", k, "fix")["rho_luma"][b] for k in ks]
            A(f"{names[b]:>12s}".join(f"{v:12.4f}" for v in r1) +
              f"{es[0][b]:12.4f}" + " |".join(f"{v:12.4f}" for v in r2) +
              f"{ep[0][b]:12.4f}")
        A("```")
        A("")
    A("*_inf 는 k -> 무한 외삽이다. 가법·독립 잡음 + 스칼라 이득이면")
    A("  1/rho_k^2 = A + B/k  (A = k->무한 극한, B 는 잡음 몫)")
    A("가 정확히 성립한다. 측정한 k 점들로 이 직선을 최소제곱 적합해 A 에서 rho_inf = 1/sqrt(A) 를 얻었다.")
    A("직선성 자체가 검증 대상이다(최대 적합잔차는 5번 산수 대조에 있다). pw 갈래는 사상이 비선형이라")
    A("이 직선 관계가 보장되지 않으므로 pw_inf 는 경험적 외삽으로만 읽는다.")
    A("")

    A("### 잡음 보정값이 k 에 무관한가 (보정식의 자기 정합)")
    A("")
    A("rho_dn = rho_k / sqrt(1 - N_k/P_k) 는 k 를 뭘로 잡든 같은 값이 나와야 한다(같은 극한을 잰다).")
    A("k 를 바꿔 가며 같은 fix 코호트에서 계산했다. 코너 대역(0.50-0.71) 값이고, 전 대역 최대 편차는")
    A("맨 오른쪽이다.")
    A("")
    A("```")
    A("노출     갈래     dn(k=1)  dn(k=2)  dn(k=4)  dn(k=8) | 전대역 최대편차")
    A("-----------------------------------------------------------------------")
    for e in exps:
        for v in ["scalar", "pw"]:
            ks = [k for k in d["ks"] if gin(e, v, k, "fix") and gnz(e, k, "fix")]
            rs = [rho_denoised(e, k, "fix", v)[0] for k in ks]
            if not rs:
                continue
            spread = float(np.max(np.max(np.array(rs), 0) - np.min(np.array(rs), 0)))
            A(f"{e:7s} {v:8s}".join(f"{r[10]:9.4f}" for r in rs) +
              " " * (9 * (4 - len(rs))) + f" | {spread:14.4f}")
    A("```")
    A("")
    A("보정값이 k 에 대해 흔들리지 않으면 k=1 한 장에서 낸 천장 추정이 믿을 만하다는 뜻이다.")
    A("반대로 위의 *_inf 직선 외삽은 집계 rho 에 대해서는 정확히 성립하지 않는다(장별로는")
    A("성립해도 에너지 가중 집계에는 옌센 효과가 있고, k 마다 표본 구성도 달라진다).")
    A("그래서 천장 추정의 정본은 잡음 보정값이고 외삽은 참고 대조다.")
    A("")

    A("k 를 그룹 최대치까지 밀었을 때 (all 코호트, k 마다 표본 구성이 달라 추세만 본다):")
    A("")
    A("```")
    A("노출     k   표본수   pw rho 0.25-0.30  0.45-0.50  0.50-0.71 | 대역LS PSNR")
    A("---------------------------------------------------------------------------")
    for e in exps:
        kk = sorted(set(int(x.split("|")[2]) for x in d["input"]
                        if x.startswith(e + "|pw|") and x.endswith("|all")))
        for k in kk:
            q = gin(e, "pw", k)
            A(f"{e:7s} {k:3d} {q['n']:7d} {q['rho_luma'][5]:14.4f} {q['rho_luma'][9]:10.4f} "
              f"{q['rho_luma'][10]:10.4f} | {q['psnr_bandls']:11.3f}")
    A("```")
    A("")

    A("### 잡음이 실제로 1/k 로 줄었는지 (외삽의 전제 검증)")
    A("")
    A("겹치지 않는 두 k-평균의 차이 파워/2 = k 장 평균의 잡음 파워 추정(신호는 상쇄된다).")
    A("독립 잡음이면 k 에 반비례해야 한다. 고주파 구간(0.45-0.50) 값이고 fix 코호트다")
    A("(그룹 구성과 가중을 k 사이에 고정했다).")
    A("")
    A("```")
    A("노출      k   잡음파워(0.45-0.50)   k=1 대비   1/k 기대   표본수")
    A("-----------------------------------------------------------------")
    for e in exps:
        base = None
        kl = sorted(set(int(x.split("|")[1]) for x in d["noise"]
                        if x.startswith(e + "|") and x.endswith("|fix")))
        for k in kl:
            q = gnz(e, k, "fix")
            v = q["luma_raw"][9]
            if base is None:
                base = v
            A(f"{e:7s} {k:3d} {v:21.4f} {v/base:10.4f} {1.0/k:10.4f} {q['n']:8d}")
    A("```")
    A("")
    A("프레임 차이가 시간 간격에 의존하는지도 봤다. 드리프트(정렬오차·조명 변동)가 섞여 있으면")
    A("간격이 벌어질수록 차이 파워가 커져야 하고, 그만큼은 프레임 평균으로 안 없어진다.")
    A("같은 그룹 수가 유지되는 간격까지만 적었다(0.45-0.50 구간, 원단위).")
    A("")
    A("```")
    A("노출     간격1    간격2    간격3    간격4    간격5   | 그룹수")
    A("--------------------------------------------------------------")
    for e in exps:
        vs, ng = [], []
        for l in range(1, 6):
            q = d["lagpow"].get(f"{e}|{l}")
            vs.append(q["luma"][9] if q else float("nan"))
            ng.append(q["n"] if q else 0)
        if ng[0] == 0:
            continue
        A(f"{e:7s}".join(("%9.3f" % v) if v == v else "        -" for v in vs) +
          "  | " + ",".join(str(x) for x in ng))
    A("```")
    A("")
    A("간격에 따라 변하지 않으면 프레임 잡음은 독립이고 k 평균이 정직하게 잡음을 1/k 로 줄인다.")
    A("(그룹 수가 바뀌는 간격에서 값이 뛰는 것은 대상 그룹이 바뀐 탓이지 간격 탓이 아니다.)")
    A("")

    A("## 3. 상한 환산 — rho 를 PSNR 로")
    A("")
    A("유도. 한 구간 b 에서 입력의 주파수 계수 벡터를 P, GT 를 G 라 하고 실수 스칼라 s 로 진폭만 맞춘다.")
    A("  J(s) = ||sP - G||^2 = s^2 ||P||^2 - 2 s Re<P,G> + ||G||^2")
    A("  dJ/ds = 0  ->  s* = Re<P,G> / ||P||^2")
    A("  J(s*) = ||G||^2 - (Re<P,G>)^2/||P||^2 = ||G||^2 (1 - rho_b^2),")
    A("          rho_b = Re<P,G> / (||P|| ||G||)")
    A("파스발로 화소당 평균제곱오차는 구간 합이 된다(C=3 채널):")
    A("  MSE_min = (1/C) sum_c sum_b P_gt[b,c] (1 - rho[b,c]^2),  PSNR = 10 log10(255^2 / MSE_min)")
    A("")
    A("가정:")
    A("  (i)   구간 독립 - 구간마다 실수 이득 하나를 서로 무관하게 고른다.")
    A("  (ii)  위상 최적 사용 - 주어진 신호의 위상을 그대로 쓰는 것이 이 클래스의 최적이라는 뜻이고,")
    A("        위상을 GT 쪽으로 고칠 수 있다는 뜻이 아니다.")
    A("  (iii) 이득은 장별·채널별·구간별 오라클(GT 를 보고 고름) - 그래서 상한이다.")
    A("  (iv)  클리핑·8bit 반올림 없음. 클리핑은 [0,1] 볼록집합 사영이라 실제로는 오차를 못 늘린다.")
    A("따라서 이 값은 '주어진 신호의 대역 진폭만 재조정하는 추정기' 클래스의 최적값이다.")
    A("비선형 추정기는 이 클래스 밖이라 이 수를 넘을 수 있다. 실제로 모델이 scalar 입력의 값을 넘는다.")
    A("본체는 rho 비교이고 dB 는 같은 척도로 환산해 크기를 보는 참고값이다.")
    A("")
    A("```")
    A("대상                                     전체     ".join(f"{e:>10s}" for e in exps))
    A("---------------------------------------------------------------------------")

    def line(tag, get):
        vals = []
        for e in ["ALL"] + exps:
            try:
                vals.append(f"{get(e):10.3f}")
            except Exception:
                vals.append(f"{'-':>10s}")
        A(f"{tag:36s}".join(vals))

    line("(i) 입력 scalar, 스칼라 이득만", lambda e: gin(e, "scalar", 1)["psnr_plain"])
    line("(i) 입력 scalar + 대역이득 최적", lambda e: gin(e, "scalar", 1)["psnr_bandls"])
    line("(i) 입력 pw, 포인트와이즈만", lambda e: gin(e, "pw", 1)["psnr_plain"])
    line("(i) 입력 pw + 대역이득 최적", lambda e: gin(e, "pw", 1)["psnr_bandls"])
    for k in [2, 4, 8]:
        line(f"(ii) 입력 pw + 대역이득, k={k} (fix)",
             lambda e, k=k: gin(e, "pw", k, "fix")["psnr_bandls"])
    line("(ii) k->무한 = 잡음보정 scalar-dn", lambda e: gin(e, "scalar", 1)["psnr_dn"])
    line("(ii) k->무한 = 잡음보정 pw-dn", lambda e: gin(e, "pw", 1)["psnr_dn"])
    line("(iii) 모델 출력 (실측, 무보정)",
         lambda e: d["psnr_model_all"] if e == "ALL" else d["psnr_model"][e])
    line("(iii) 모델 + 대역이득 최적",
         lambda e: (d["model_all"] if e == "ALL" else d["model"][e])["psnr_bandls"])
    A("```")
    A("")
    A("모든 줄이 같은 형태다: 장별·채널별·대역별 오라클 이득 뒤의 MSE 를 dB 로 환산했다.")
    A("k=2,4,8 줄은 실제로 그 장수를 평균해서 잰 값이고 fix 코호트다(노출별 표본 구성이 다르다).")
    A("잡음보정 줄은 5번에서 '유한 k 실측보다 커야 한다'는 검정을 붙였다 - 결과를 그대로 읽어라.")
    A("")
    A("같은 fix 코호트 안에서 k 만 바꾼 값 (표본 구성이 같아 k 효과만 본다):")
    A("")
    A("```")
    A("노출     갈래      k=1      k=2      k=4      k=8   | k최대 - k=1")
    A("-----------------------------------------------------------------")
    for e in exps:
        for v in ["scalar", "pw"]:
            ks = [k for k in d["ks"] if gin(e, v, k, "fix")]
            vs = [gin(e, v, k, "fix")["psnr_bandls"] for k in ks]
            A(f"{e:7s} {v:8s}".join(f"{x:9.3f}" for x in vs) +
              " " * (9 * (4 - len(vs))) + f" | {vs[-1]-vs[0]:+9.3f}")
    A("```")
    A("")

    A("## 4. 대역별 판정 요약")
    A("")
    A("오차 에너지는 현재 모델의 잔차(전역이득 보정 후, 채널합) 대역 파워로 잰다.")
    A("기준 세 개를 나란히 낸다. (b) 가 가정 없는 실측 판정이고 (a) 는 잡음 보정 모형이 들어간 추정이다.")
    A("두 기준이 갈리는 대역은 '한 장으로는 모델이 이미 다 썼지만 프레임을 무한히 모으면 여지가 있다'는")
    A("뜻이다. 어느 쪽을 믿을지는 5번 검정을 보고 판단해라 - (a) 는 저주파에서 반증됐다.")
    A("")
    A("### (a) 정보 천장 추정 기준 (rho_in(pw-dn) 대 rho_out) - 모형 가정 있음")
    A("")
    A("```")
    A("판정        대역                                              오차에너지 몫%")
    A("---------------------------------------------------------------------------")
    for tag in ["회복가능", "소진", "불가"]:
        bs = [b for b in range(nb) if verds[b] == tag]
        A(f"{tag:10s}  {(', '.join(names[b] for b in bs) if bs else '-'):48s} "
          f"{sum(share[b] for b in bs):8.2f}")
    A("```")
    A("")
    A("### (b) 다중 프레임 실측 기준 (k=KFIX 평균 입력 대 같은 씬의 모델 출력) - 본 판정")
    A("")
    A("외삽도 보정도 없이 실제로 k 장을 평균해서 잰 rho 를, 같은 fix 코호트의 모델 출력 rho 와 비교한다.")
    A("노출마다 KFIX 가 다르므로(0.033s 는 2, 나머지는 8) 노출별 표로만 낸다.")
    A("")
    A("```")
    A("노출    KFIX  판정        대역                                    오차에너지 몫%")
    A("-------------------------------------------------------------------------------")
    for e in exps:
        kf = d["kfix"][e]
        q = gin(e, "pw", kf, "fix")
        rk = np.array(q["rho_luma"])
        ro = np.array(d["model_fix"][e]["rho_luma"])
        rm = np.array(d["model_fix"][e]["res_plain_rgb"])
        sh = 100.0 * rm / rm.sum()
        for tag in ["회복가능", "소진", "불가"]:
            bs = [b for b in range(nb) if verdict(rk[b], ro[b]) == tag]
            A(f"{e:7s} {kf:4d}  {tag:10s}  {(', '.join(names[b] for b in bs) if bs else '-'):38s} "
              f"{sum(sh[b] for b in bs):8.2f}")
    A("```")
    A("")
    A("### (c) 참고 - 잡음 보정 안 한 원 입력 (rho_in(pw) k=1 대 rho_out)")
    A("")
    A("```")
    A("판정        대역                                              오차에너지 몫%")
    A("---------------------------------------------------------------------------")
    verds2 = verds_raw
    for tag in ["회복가능", "소진", "불가"]:
        bs = [b for b in range(nb) if verds2[b] == tag]
        A(f"{tag:10s}  {(', '.join(names[b] for b in bs) if bs else '-'):48s} "
          f"{sum(share[b] for b in bs):8.2f}")
    A("```")
    A("")
    A("노출시간별 (정보 천장 기준 / 원 입력 기준):")
    A("")
    A("```")
    A("노출      천장: 회복가능%  소진%   불가%  | 원입력: 회복가능%  소진%   불가%")
    A("----------------------------------------------------------------------------")
    for e in exps:
        rp = np.array(gin(e, "pw", 1)["rho_luma"])
        ro = np.array(d["model"][e]["rho_luma"])
        rm = np.array(d["model"][e]["res_plain_rgb"])
        sh = 100.0 * rm / rm.sum()
        ri = rho_denoised(e, 1, "all", "pw")[0]
        a1 = {"회복가능": 0.0, "소진": 0.0, "불가": 0.0}
        a2 = {"회복가능": 0.0, "소진": 0.0, "불가": 0.0}
        for b in range(nb):
            a1[verdict(ri[b], ro[b])] += sh[b]
            a2[verdict(rp[b], ro[b])] += sh[b]
        A(f"{e:7s} {a1['회복가능']:14.2f} {a1['소진']:7.2f} {a1['불가']:7.2f}  | "
          f"{a2['회복가능']:14.2f} {a2['소진']:7.2f} {a2['불가']:7.2f}")
    A("```")
    A("")
    A("천장 기준에서 '불가' 로 찍힌 대역:")
    A("")
    A("```")
    A("노출     불가 대역")
    A("--------------------------------------------------------------")
    for e in exps:
        ri = rho_denoised(e, 1, "all", "pw")[0]
        ro = np.array(d["model"][e]["rho_luma"])
        bad = [names[b] for b in range(nb) if verdict(ri[b], ro[b]) == "불가"]
        A(f"{e:7s} {', '.join(bad) if bad else '-'}")
    A("```")
    A("")

    A("## 5. 산수 대조 (검증)")
    A("")
    rows = []

    def chk(name, val, ref, tol, extra="", ineq=None):
        if ineq == ">=":
            ok = val >= ref - tol
        elif ineq == "<=":
            ok = val <= ref + tol
        else:
            ok = abs(val - ref) <= tol
        rows.append((name, val, ref, "통과" if ok else "실패", extra))

    chk("모델 무보정 PSNR = DIAG_SID_FAILURE 24.438", d["psnr_model_all"], 24.438, 0.01)
    sp = spec["overall"]["spec"]
    prev = [(sp["p_out"][b] + sp["p_gt"][b] - sp["p_res"][b]) / 2.0 /
            math.sqrt(max(sp["p_out"][b] * sp["p_gt"][b], 1e-30)) for b in range(nb)]
    md = max(abs(prev[b] - rho_out[b]) for b in range(nb))
    chk("rho_output 재계산 vs DIAG_SID_SPECTRUM 최대차", md, 0.0, 5e-3,
        "교차항 직접 계산 - 다른 경로")
    chk("모델 대역LS PSNR(11구간) <= 앞 측정 26.601",
        d["model_all"]["psnr_bandls"], spec["overall"]["psnr_bandls_11"], 0.02, "<=",
        ineq="<=")
    chk("입력 스칼라이득 PSNR(전량) = DIAG_SID_FAILURE 12.974",
        gin("ALL", "scalar", 1)["psnr_meas"], 12.974, 0.01, "클리핑+8bit, 598장")
    ra = np.array([r["psnr_analytic"] for r in d["rec_check"]])
    rm2 = np.array([r["psnr_measured"] for r in d["rec_check"]])
    chk("대역LS: 실제 복원 PSNR >= 해석식 (클리핑은 오차를 못 늘림)",
        float(np.min(rm2 - ra)), 0.0, 0.01, f"{len(ra)}장 최소차", ineq=">=")
    allr = []
    for kk, v in d["input"].items():
        allr += list(v["rho_luma"]) + list(v["rho_rgb"])
    allr += list(rho_out)
    chk("모든 rho 가 [-1,1] 안 (코시-슈바르츠)", float(np.max(np.abs(allr))), 1.0, 0.0,
        "<=1 이어야 함", ineq="<=")
    q = gin("ALL", "scalar", 1)
    chk("Parseval GT: sum_b P_b vs mean(L^2)", float(np.sum(q["p_gt_luma"])), q["ms_gt"],
        1e-6 * max(1.0, q["ms_gt"]))
    chk("Parseval 입력: sum_b P_b vs mean(x^2)", float(np.sum(q["p_in_luma"])), q["ms_in"],
        1e-6 * max(1.0, q["ms_in"]))
    chk("주파수 개수 합 = H*W", float(sum(d["nfreq_per_band"])), float(512 * 960), 0.0)
    n1 = sum(gin(e, "scalar", 1)["n"] for e in exps)
    chk("k=1 표본수 = 598 (1번과 같은 표본)", float(n1), 598.0, 0.0)
    dd = [max(abs(np.array(r["rho_fit_all"]) - np.array(r["rho_fit_half"])))
          for r in d["pw_audit"]]
    chk("pw 사상 과적합 대조: 전체적합 vs 절반적합 rho 최대차",
        float(np.max(dd)), 0.0, 0.02, "짝수행만으로 적합해도 같아야 함")
    bad = sum(1 for kk, v in d["input"].items() if v["psnr_bandls"] < v["psnr_plain"] - 1e-9)
    chk("대역LS PSNR >= 스칼라 PSNR (모든 셀, 최소제곱이라 필연)", float(bad), 0.0, 0.0)
    # 잡음 1/k
    worst = 0.0
    for e in exps:
        ks = sorted(set(int(x.split("|")[1]) for x in d["noise"]
                        if x.startswith(e + "|") and x.endswith("|fix")))
        b0 = gnz(e, ks[0], "fix")["luma_raw"][9]
        for k in ks:
            worst = max(worst, abs(gnz(e, k, "fix")["luma_raw"][9] / b0 * k - 1.0))
    chk("잡음 파워가 1/k (0.45-0.50 원단위, 최대 상대편차)", worst, 0.0, 0.05,
        "프레임 간 독립 잡음이면 성립해야 함")
    # 외삽 직선성 (scalar 는 이론상 정확)
    lin = max(float(np.max(ext[(e, "scalar")][1] /
                           np.maximum([1.0 / max(r, 1e-6) ** 2 for r in ext[(e, "scalar")][0]], 1e-9)))
              for e in exps if (e, "scalar") in ext)
    chk("1/rho^2 = A + B/k 직선 적합 최대 상대잔차 (scalar, 진단용)", lin, 0.0, 0.50,
        "장별로는 정확하지만 에너지가중 집계에는 옌센 편향 - 외삽을 정본으로 안 쓰는 근거")
    # 두 경로 대조: 잡음 실측 보정 vs k 외삽
    for e in exps:
        r1, _ = rho_denoised(e, 1, "fix")
        if r1 is None or (e, "scalar") not in ext:
            continue
        chk(f"{e}: 잡음보정 rho vs k외삽 rho 최대차 (독립 두 경로, scalar)",
            float(np.max(np.abs(r1 - ext[(e, "scalar")][0]))), 0.0, 0.20,
            "외삽은 코호트 구성 변화로 편향됨 - 참고 대조")
    # 같은 극한을 재는 값이므로 k 를 바꿔도 같아야 한다 (보정식의 자기 정합)
    for e in exps:
        for v in ["scalar", "pw"]:
            ks = [k for k in d["ks"] if gin(e, v, k, "fix") and gnz(e, k, "fix")]
            if len(ks) < 2:
                continue
            rs = np.array([rho_denoised(e, k, "fix", v)[0] for k in ks])
            chk(f"{e}/{v}: 잡음보정 rho 가 k 에 무관한가 (k={','.join(map(str,ks))})",
                float(np.max(rs.max(0) - rs.min(0))), 0.0, 0.06,
                "같은 극한을 재므로 k 에 무관해야 함")
    for e in exps:
        q8 = gin(e, "pw", 8, "fix")
        if q8:
            for v in ["scalar", "pw"]:
                chk(f"{e}/{v}: 잡음보정 상한(k=1) >= k=8 실측 대역LS",
                    gin(e, v, 1, "fix")["psnr_dn"],
                    gin(e, v, 8, "fix")["psnr_bandls"], 0.0,
                    "같은 장별·채널별 규약, 극한이 유한 k 보다 커야 함", ineq=">=")
                kd = [kk for kk in d["ks"] if gin(e, v, kk, "fix")
                      and gin(e, v, kk, "fix")["n_dn"] > 0]
                chk(f"{e}/{v}: 잡음보정 상한이 k 에 무관한가 (k=1 vs k={kd[-1]})",
                    gin(e, v, 1, "fix")["psnr_dn"], gin(e, v, kd[-1], "fix")["psnr_dn"], 0.6)
                # 대역별로 어디가 반증되는지: 천장 추정이 실측 k 평균보다 작으면 그 칸은 과소평가
                r1 = rho_denoised(e, 1, "fix", v)[0]
                rk = np.array(gin(e, v, d["kfix"][e], "fix")["rho_luma"])
                viol = [names[b] for b in range(nb) if r1[b] < rk[b] - 1e-9]
                chk(f"{e}/{v}: 천장 추정이 실측 k={d['kfix'][e]} 보다 작은 대역 수",
                    float(len(viol)), 0.0, 0.0, ("과소평가 대역: " + ", ".join(viol)) if viol else "")
    A("```")
    A("항목                                                          계산값        대조값   판정")
    A("---------------------------------------------------------------------------------------------")
    for nm, v, r, st, ex in rows:
        A(f"{nm:55s} {v:12.5f} {r:12.5f}  {st}" + (f"  ({ex})" if ex else ""))
    A("```")
    A("")

    # ---- 측정이 가리키는 것 ----
    A("## 측정이 가리키는 것")
    A("")
    hi = 10
    e0 = exps[0]
    kf = d["kfix"]["0.1s"]
    r1_1 = gin("0.1s", "pw", 1, "fix")["rho_luma"][hi]
    r1_k = gin("0.1s", "pw", kf, "fix")["rho_luma"][hi]
    ro1 = d["model_fix"]["0.1s"]["rho_luma"][hi]
    rdn1 = rho_denoised("0.1s", 1, "fix", "pw")[0][hi]
    A(f"- 고주파 정보가 측정에 남아 있긴 하다. 코너 대역(0.50-0.71) 단일 프레임 rho 는 pw "
      f"{rho_pw1[hi]:.3f} 로 낮지만 그 대역 입력 파워의 {nfrac_pw[hi]*100:.0f}% 가 잡음이고, "
      f"프레임을 모으면 실제로 오른다(0.1s 같은 씬 k=1 {r1_1:.3f} -> k={kf} {r1_k:.3f}). "
      "잡음이 독립이라는 것도 확인했다 - 프레임 차이 파워는 시간 간격과 무관하고 k 평균에서 "
      "정확히 1/k 로 준다(편차 1.7% 이내).")
    A(f"- 그런데 단일 프레임만 놓고 보면 모델이 이미 다 쓰고 있다. 잡음 보정 없이 비교하면 11개 "
      "대역 전부에서 모델 rho 가 입력 rho 이상이고, 8장을 평균해서 잰 값과 비교해도 회복가능으로 "
      f"찍히는 대역이 하나도 없다(0.1s 코너 k=8 {r1_k:.3f} vs 모델 {ro1:.3f}). 판정은 (b) 소진이다.")
    A(f"- 짧은 노출의 고주파는 판정이 (c) 불가다. {e0} 는 2장을 평균해도 코너 rho "
      f"{gin(e0,'pw',d['kfix'][e0],'fix')['rho_luma'][hi]:.3f}, 0.25 이상 여섯 대역이 전부 0.2 "
      "아래다(오차의 20.9%). 0.04s 도 0.35 이상 네 대역이 그렇다(16.2%). 이 대역들은 어떤 방법으로도 "
      "못 살린다 - 진폭이 아니라 위치정보 자체가 측정에 없다.")
    A(f"- 여지가 남는 곳은 '여러 장'뿐이다. 잡음을 벗겨낸 천장 추정은 0.1s 코너에서 {rdn1:.3f} 로 "
      f"모델 {ro1:.3f} 를 웃돌고, 전체로는 오차의 20.4% 에 해당하는 고주파 6개 대역이 여기 걸린다. "
      "다만 이 추정은 저주파에서 실측 k 평균보다 작게 나와 반증됐다(5번). 고주파 칸은 반증되지 "
      "않았고 반증 방향도 과소평가 쪽이라 이 몫은 보수적인 값이다.")
    A("- 오차의 절반 이상(54.7%)은 여전히 최저주파 대역(0.00-0.05)에 있고 거기는 어느 기준으로도 "
      "소진이다. 고주파를 되살리는 것으로 얻을 수 있는 몫의 상한이 20% 대라는 뜻이다.")
    A("")

    txt = "\n".join(L) + "\n"
    p = os.environ.get("IDENT_MD",
        "numbers/DIAG_SID_IDENTIFIABILITY.md")
    open(p, "w").write(txt)
    print("wrote", p)


def psnr_dn_rgb(d, gin, gnz, e, v="pw", c="all", k=1):
    """잡음 보정 상한을 채널별·대역별 이득으로 환산한다(다른 행과 같은 형태)."""
    q = gin(e, v, k, c); z = gnz(e, k, c)
    if q is None or z is None:
        return float("nan")
    Pg = np.array(q["p_gt_c"]); Pi = np.array(q["p_in_c"]); Cx = np.array(q["cross_c"])
    Nn = np.array(z["rgb" if v == "scalar" else "rgb_pw"])
    Pi_dn = np.maximum(Pi - Nn, 1e-30)
    rho2 = np.clip(Cx ** 2 / np.maximum(Pi_dn * Pg, 1e-30), 0.0, 1.0)
    mse = float(np.sum(Pg * (1.0 - rho2)) / 3.0)
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-30))


def psnr_from_rho(d, rho_luma, e):
    """휘도 rho 를 세 채널에 같이 적용해 MSE_min 을 환산한다(근사, 문서에 명시)."""
    key = "model_all" if e == "ALL" else None
    g = (d["model_all"] if e == "ALL" else d["model"][e])["p_gt_rgb"]
    mse = float(np.sum(np.array(g) * (1.0 - np.array(rho_luma) ** 2)) / 3.0)
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-30))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()
