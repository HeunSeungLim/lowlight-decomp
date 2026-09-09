"""multiframe_gain.py

DIAG_SID_IDENTIFIABILITY 가 남긴 칸을 채운다: "여러 장에만 여유가 있다"는 진술을
실제로 여러 장을 넣어서 확인한다.

학습 없음. Retinexformer 저자 SID.pth 를 그대로 쓰고 입력만 바꾼다.

측정
  1. k 장 평균 입력  : 같은 씬·같은 노출 short 를 k 장 평균해 모델에 넣고 GT 대비 PSNR/SSIM
                       (k=1,2,3,4,5,8 + 그룹 최대치). k=1 은 앵커 24.438 / 0.6800 과 일치해야 한다.
  2. 공정 비교       : (a) 총 노출이 같은 단일 프레임과의 대조 (0.04s x5 = 0.1s x2 등)
                       (b) 입력 평균 대 출력 평균 (모델을 k 번 돌려 출력을 평균)
  3. 대역별 rho_out  : k 평균 입력에 대한 모델 출력의 방사 대역별 상관 (대역 정의 동일)
  4. 성분 분해       : 전역이득 / 채널이득 / 구조잔차 몫 + 저주파 D 에너지몫 을 k 별로
  5. 상한 대조       : k=8 PSNR 을 기존 오라클 사다리와 같은 표에

규약은 전부 앞 문서에서 import 한다(로더·대역 정의·이득 정합·8bit 반올림).
채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 쓴다.

조합 선택 규약 (고정, 사후조정 없음)
  그룹 안 프레임을 파일명 정렬 순으로 놓고 앞에서부터 겹치지 않는 연속 k 장 블록을
  floor(n/k) 개 만든다. 남는 프레임은 버린다. 가능한 모든 조합을 평균내지 않는다
  (조합 수가 k 마다 달라져 표본 가중이 틀어지기 때문).
"""
import os, sys, glob, json, math, time, itertools
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep
from diag_sid_spectrum import (build_pairs, load_npy, exposure_of, u8,
                               make_radial_index, RAD_EDGES, LUMA_W,
                               DATA, WEIGHTS, REPO)
from diag_sid_failure import decompose

OUT_JSON = os.environ.get("MFG_OUT", "numbers/multiframe_gain.json")
OUT_MD = os.environ.get("MFG_MD", "numbers/MULTIFRAME_GAIN.md")
IDENT_JSON = "numbers/diag_sid_identifiability.json"

KS = [1, 2, 3, 4, 5, 8]
H, W = 512, 960
N = H * W
NCH = 3
LP_NBAND = 2                     # 저주파 D = 최저 두 대역 (f < 0.10), DIAG_SID_LOWFREQ 규약

# 기존 오라클 사다리 (대조값)
LADDER = [("모델 무보정 (k=1)", 24.438, "DIAG_SID_FAILURE"),
          ("전역이득 오라클", 25.779, "DIAG_SID_LOWFREQ"),
          ("국소이득 16x16 오라클", 26.966, "DIAG_SID_LOWFREQ"),
          ("국소affine 16x16 오라클", 28.032, "DIAG_SID_LOWFREQ")]


def build_groups():
    """(scene, exposure) -> {'lq': [경로...], 'gt': 경로}"""
    g = {}
    for scene, lqp, gtp in build_pairs():
        key = (scene, exposure_of(lqp))
        g.setdefault(key, {"lq": [], "gt": gtp})
        g[key]["lq"].append(lqp)
    for k in g:
        g[k]["lq"].sort()
    return g


def ssim_job(args):
    pred_u8, gt_u8 = args
    return ssim_indep(pred_u8, gt_u8)


def blocks(n, k):
    """앞에서부터 겹치지 않는 연속 k 장 블록 (floor(n/k) 개)."""
    return [list(range(i * k, (i + 1) * k)) for i in range(n // k)]


class BandAcc:
    """대역별 (P_gt, P_in, cross) 누적 + 잔차 파워. 채널별/휘도 두 갈래."""

    def __init__(self, nb):
        self.nb = nb
        self.n = 0
        self.pg_l = np.zeros(nb); self.pp_l = np.zeros(nb); self.cx_l = np.zeros(nb)
        self.pg_c = np.zeros((3, nb)); self.pp_c = np.zeros((3, nb)); self.cx_c = np.zeros((3, nb))

    def add(self, o):
        self.n += 1
        for t in ["pg_l", "pp_l", "cx_l", "pg_c", "pp_c", "cx_c"]:
            setattr(self, t, getattr(self, t) + o[t])

    def dump(self):
        n = max(self.n, 1)
        rho_l = self.cx_l / np.sqrt(np.maximum(self.pg_l * self.pp_l, 1e-30))
        res = (self.pg_c + self.pp_c - 2 * self.cx_c).sum(0) / n     # 대역별 잔차 (채널합, 장평균)
        return dict(n=self.n, rho_luma=list(rho_l), res_plain_rgb=list(res),
                    p_gt_rgb=list(self.pg_c.sum(0) / n))


# ============================ 측정 ============================
def main():
    import torch
    import torch.nn.functional as F
    from multiprocessing import Pool
    from diag_sid_lowfreq import block_index, block_fit

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    nb = len(RAD_EDGES) - 1
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)
    bidx16, nblk16, _, _ = block_index(H, W, 16, torch, dev)

    def band_cross(p, g):
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
        return pp, gg, cx

    def band_stats(p255, g255):
        pp, gg, cx = band_cross(p255, g255)
        ppl, ggl, cxl = band_cross_luma(p255, g255)
        return dict(pg_l=ggl.cpu().numpy(), pp_l=ppl.cpu().numpy(), cx_l=cxl.cpu().numpy(),
                    pg_c=gg.cpu().numpy(), pp_c=pp.cpu().numpy(), cx_c=cx.cpu().numpy())

    def to_u8(t3hw):
        return u8(np.clip(t3hw.permute(1, 2, 0).cpu().numpy(), 0, 1))

    groups = build_groups()
    lim = int(os.environ.get("MFG_LIMIT", "0"))
    if lim:
        groups = {k: v for k, v in sorted(groups.items())[:lim]}
    exps = sorted(set(e for _, e in groups), key=lambda s: float(s[:-1]))
    ngrp = {e: sum(1 for k in groups if k[1] == e) for e in exps}
    nimg = {e: sum(len(groups[k]["lq"]) for k in groups if k[1] == e) for e in exps}
    KFIX = {}
    for e in exps:
        mx = max(len(groups[kk]["lq"]) for kk in groups if kk[1] == e)
        KFIX[e] = max([k for k in [1, 2, 4, 8] if k <= mx])
    print("groups", ngrp, "imgs", nimg, "KFIX", KFIX, flush=True)

    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    def infer(x01):
        """x01: 3xHxW float32 cuda [0,1] -> 3xHxW float64 [0,1] (클리핑)"""
        x = x01[None]
        h, w = x.shape[2], x.shape[3]
        padh, padw = (-h) % 4, (-w) % 4
        if padh or padw:
            x = F.pad(x, (0, padw, 0, padh), "reflect")
        y = net(x)[:, :, :h, :w]
        return torch.clamp(y, 0, 1).double()[0]

    pool = Pool(12)
    jobs = []
    accB = {}            # (exp,k,cohort,which) -> BandAcc ; which in {in, out}
    ninfer = 0

    def acc(exp, k, coh, which):
        key = (exp, k, coh, which)
        if key not in accB:
            accB[key] = BandAcc(nb)
        return accB[key]

    t0 = time.time()
    with torch.inference_mode():
        for gi, ((scene, exp), gd) in enumerate(sorted(groups.items())):
            gt_u8 = u8(load_npy(gd["gt"]))
            gt_f = gt_u8.astype(np.float64) / 255.0
            g = torch.from_numpy(gt_f).cuda().permute(2, 0, 1)
            g255 = g * 255.0
            n = len(gd["lq"])
            fr = torch.stack([
                torch.from_numpy(load_npy(p).transpose(2, 0, 1)) for p in gd["lq"]]).cuda()
            outs = torch.stack([infer(fr[i]) for i in range(n)])
            ninfer += n
            kf = KFIX[exp]
            klist = [k for k in KS if k <= n]
            if n not in klist:
                klist.append(n)
            for k in klist:
                for bi, b in enumerate(blocks(n, k)):
                    x = fr[b].mean(0)
                    y_in = outs[b[0]] if k == 1 else infer(x)
                    if k > 1:
                        ninfer += 1
                    y_out = outs[b].mean(0)
                    in_fix = (n >= kf)
                    in_match = in_fix and (kf % k == 0) and ((bi + 1) * k <= kf)
                    cohs = ["all"] + (["fix"] if in_fix else []) + (["match"] if in_match else [])
                    meta = dict(scene=scene, exp=exp, k=k, bi=bi, nfr=n,
                                fix=in_fix, match=in_match)
                    for tag, y in (("in", y_in), ("out", y_out)):
                        a = float((y * g).sum() / (y * y).sum())    # 오라클 전역이득
                        st = band_stats(a * y * 255.0, g255)
                        for c in cohs:
                            acc(exp, k, c, tag).add(st)
                        res = (st["pg_c"] + st["pp_c"] - 2 * st["cx_c"]).sum(0)
                        dc = decompose(y.permute(1, 2, 0).cpu().numpy(), gt_f)
                        yg = block_fit(y, g, bidx16, nblk16, "gain", torch)[0]
                        ya = block_fit(y, g, bidx16, nblk16, "aff", torch)[0]
                        pu8 = to_u8(y)
                        meta[f"psnr_{tag}"] = psnr_indep(pu8, gt_u8)
                        meta[f"gain_{tag}"] = a
                        meta[f"pg_{tag}"] = psnr_indep(to_u8(a * y), gt_u8)
                        meta[f"pb16g_{tag}"] = psnr_indep(to_u8(yg), gt_u8)
                        meta[f"pb16a_{tag}"] = psnr_indep(to_u8(ya), gt_u8)
                        meta[f"mse_{tag}"] = dc["mse"]
                        meta[f"mseg_{tag}"] = dc["mse_glob"]
                        meta[f"msec_{tag}"] = dc["mse_chan"]
                        meta[f"msea_{tag}"] = dc["mse_affine"]
                        meta[f"dlo_{tag}"] = float(res[:LP_NBAND].sum())
                        meta[f"dtot_{tag}"] = float(res.sum())
                        meta[f"_u8_{tag}"] = pu8
                    jobs.append((meta,
                                 pool.apply_async(ssim_job, ((meta.pop("_u8_in"), gt_u8),)),
                                 pool.apply_async(ssim_job, ((meta.pop("_u8_out"), gt_u8),))))
            del fr, outs
            if (gi + 1) % 10 == 0:
                print(f"  {gi+1}/{len(groups)}  infer={ninfer}  {time.time()-t0:.0f}s", flush=True)
    print(f"inference done {ninfer} forwards, {time.time()-t0:.0f}s", flush=True)

    rows = []
    for meta, j1, j2 in jobs:
        r = dict(meta)
        r["ssim_in"] = j1.get()
        r["ssim_out"] = j2.get()
        rows.append(r)
    pool.close(); pool.join()

    out = dict(rad_edges=RAD_EDGES, exposures=exps, ks=KS, kfix=KFIX, lp_nband=LP_NBAND,
               n_groups=ngrp, n_images=nimg, n_forwards=ninfer,
               group_sizes={f"{s}|{e}": len(v["lq"]) for (s, e), v in groups.items()},
               rows=rows,
               bands={f"{e}|{k}|{c}|{w}": a.dump() for (e, k, c, w), a in accB.items()})
    for k in sorted(set(r["k"] for r in rows)):
        for c in ["all", "fix", "match"]:
            for w in ["in", "out"]:
                a = BandAcc(nb); hit = False
                for e in exps:
                    o = accB.get((e, k, c, w))
                    if o:
                        a.n += o.n
                        for t in ["pg_l", "pp_l", "cx_l", "pg_c", "pp_c", "cx_c"]:
                            setattr(a, t, getattr(a, t) + getattr(o, t))
                        hit = True
                if hit:
                    out["bands"][f"ALL|{k}|{c}|{w}"] = a.dump()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_JSON)


# ============================ 보고서 ============================
def sel(rows, exp=None, k=None, coh="all", scenes=None):
    r = rows
    if exp and exp != "ALL":
        r = [x for x in r if x["exp"] == exp]
    if k is not None:
        r = [x for x in r if x["k"] == k]
    if coh == "fix":
        r = [x for x in r if x["fix"]]
    elif coh == "match":
        r = [x for x in r if x["match"]]
    if scenes is not None:
        r = [x for x in r if x["scene"] in scenes]
    return r


def agg(rs, w="in"):
    """w = 'in' (k장 평균 입력) 또는 'out' (출력 평균). PSNR/SSIM 은 둘 다 담는다."""
    if not rs:
        return None
    d = dict(nblk=len(rs), nscene=len(set(x["scene"] for x in rs)))
    for t in ["psnr_in", "ssim_in", "psnr_out", "ssim_out"]:
        d[t] = float(np.mean([x[t] for x in rs]))
    tot = sum(x[f"mse_{w}"] for x in rs)
    d["share_global"] = 100.0 * sum(x[f"mse_{w}"] - x[f"mseg_{w}"] for x in rs) / tot
    d["share_chan"] = 100.0 * sum(x[f"mseg_{w}"] - x[f"msec_{w}"] for x in rs) / tot
    d["share_resid"] = 100.0 * sum(x[f"msec_{w}"] for x in rs) / tot
    d["dshare"] = 100.0 * sum(x[f"dlo_{w}"] for x in rs) / sum(x[f"dtot_{w}"] for x in rs)
    d["mse"] = tot / len(rs)
    for t, f in (("psnr_glob", "pg"), ("psnr_b16g", "pb16g"), ("psnr_b16a", "pb16a")):
        d[t] = float(np.mean([x[f"{f}_{w}"] for x in rs]))
    d["e_tot"] = tot / len(rs)
    d["e_glob"] = sum(x[f"mse_{w}"] - x[f"mseg_{w}"] for x in rs) / len(rs)
    d["e_chan"] = sum(x[f"mseg_{w}"] - x[f"msec_{w}"] for x in rs) / len(rs)
    d["e_resid"] = sum(x[f"msec_{w}"] for x in rs) / len(rs)
    d["e_dlo"] = sum(x[f"dlo_{w}"] for x in rs) / len(rs)
    d["e_dhi"] = sum(x[f"dtot_{w}"] - x[f"dlo_{w}"] for x in rs) / len(rs)
    return d


def report():
    d = json.load(open(OUT_JSON))
    rows = d["rows"]
    exps = d["exposures"]
    E = d["rad_edges"]; nb = len(E) - 1
    names = [f"{E[b]:.2f}-{min(E[b+1],0.7072):.2f}" for b in range(nb)]
    KFIX = d["kfix"]
    gs = d["group_sizes"]
    ident = json.load(open(IDENT_JSON)) if os.path.exists(IDENT_JSON) else None
    L = []; A = L.append
    checks = []

    def chk(name, val, ref, tol, extra="", ineq=None):
        if ineq == ">=":
            ok = val >= ref - tol
        elif ineq == "<=":
            ok = val <= ref + tol
        else:
            ok = abs(val - ref) <= tol
        checks.append((name, val, ref, "통과" if ok else "실패", extra))

    ks_all = sorted(set(r["k"] for r in rows))
    ks_main = [k for k in d["ks"]]

    A("# SID 극암 저조도 — 여러 장을 모으면 실제로 얼마나 가져오는가")
    A("")
    A("- 대상: SID test split 전량 598장 / 50씬 (표본 아님). 노출별 "
      + ", ".join(f"{e} {d['n_images'][e]}장/{d['n_groups'][e]}그룹" for e in exps))
    A("- 모델: Retinexformer 저자 SID.pth. 학습 없음. 가중치 그대로, 입력만 바꿨다.")
    A(f"- 모델 forward 총 {d['n_forwards']}회. 로더·대역정의·이득정합·8bit 반올림 규약은 "
      "diag_sid_spectrum.py / diag_sid_failure.py 에서 그대로 import 했다.")
    A("- 채점은 repro_measure 의 독립 구현(psnr_indep/ssim_indep)만 사용. 전부 실측값.")
    A("")
    A("조합 선택 규약 (고정, 사후조정 없음):")
    A("  그룹(같은 씬·같은 노출) 안 프레임을 파일명 정렬 순으로 놓고 앞에서부터 겹치지 않는")
    A("  연속 k 장 블록을 floor(n/k) 개 만든다. 남는 프레임은 버린다. 가능한 모든 조합을")
    A("  평균내지 않는다 - 조합 수가 k 마다 달라져 표본 가중이 틀어지기 때문이다.")
    A("  평균은 npy(이미 노출비로 증폭된 uint8) 를 [0,1] 로 바꾼 뒤 산술평균이다.")
    A("")
    A("코호트 세 가지. 이걸 안 나누면 k 마다 평가 대상이 달라져 결과가 무효다:")
    A("  all   : 모든 그룹의 모든 블록. k 가 커지면 짧은 그룹이 빠져 씬 구성이 바뀐다.")
    A("  fix   : n >= KFIX 인 그룹만 (씬 구성 고정). 블록 수는 k 마다 다르다.")
    A("  match : n >= KFIX 이고 k 가 KFIX 를 나누는 경우, 앞 KFIX 장만 써서 만든 블록.")
    A("          k 를 바꿔도 정확히 같은 씬·같은 프레임 집합을 쓴다. 이게 정본 대조다.")
    A(f"  KFIX = " + ", ".join(f"{e}:{KFIX[e]}" for e in exps)
      + " (그 노출 그룹의 최대 프레임 수가 허용하는 2의 거듭제곱)")
    A("")

    # ---------------- 0. 앵커 ----------------
    a1 = agg(sel(rows, "ALL", 1, "all"))
    A("## 0. 앵커 대조")
    A("")
    A("```")
    A("항목                                          측정값       대조값   판정")
    A("-------------------------------------------------------------------------")
    A(f"k=1 PSNR (전량)                            {a1['psnr_in']:9.3f}    24.438   "
      f"{'통과' if abs(a1['psnr_in']-24.438)<=0.01 else '실패'}")
    A(f"k=1 SSIM (전량)                            {a1['ssim_in']:9.4f}    0.6800   "
      f"{'통과' if abs(a1['ssim_in']-0.6800)<=0.001 else '실패'}")
    A(f"k=1 블록 수 (= short 전량 장수)              {a1['nblk']:9d}       598   "
      f"{'통과' if a1['nblk']==598 else '실패'}")
    A(f"k=1 씬 수                                  {a1['nscene']:9d}        50   "
      f"{'통과' if a1['nscene']==50 else '실패'}")
    A(f"k=1 구조잔차 몫 (E가중)                     {a1['share_resid']:9.2f} %  64.21 %   "
      f"{'통과' if abs(a1['share_resid']-64.21)<=0.05 else '실패'}")
    A(f"k=1 저주파 D 에너지몫 (RGB,E가중)            {a1['dshare']:9.2f} %  64.35 %   "
      f"{'통과' if abs(a1['dshare']-64.35)<=0.05 else '실패'}")
    A(f"k=1 전역이득 오라클 PSNR                    {a1['psnr_glob']:9.3f}    25.779   "
      f"{'통과' if abs(a1['psnr_glob']-25.779)<=0.01 else '실패'}")
    A(f"k=1 국소이득 16x16 오라클 PSNR              {a1['psnr_b16g']:9.3f}    26.966   "
      f"{'통과' if abs(a1['psnr_b16g']-26.966)<=0.01 else '실패'}")
    A(f"k=1 국소affine 16x16 오라클 PSNR           {a1['psnr_b16a']:9.3f}    28.032   "
      f"{'통과' if abs(a1['psnr_b16a']-28.032)<=0.01 else '실패'}")
    A(f"k=1 입력평균 PSNR = 출력평균 PSNR (동일해야)   {abs(a1['psnr_in']-a1['psnr_out']):9.6f}     0.000   "
      f"{'통과' if abs(a1['psnr_in']-a1['psnr_out'])<1e-9 else '실패'}")
    A("```")
    A("")
    A("사용 프레임 목록은 diag_sid_spectrum.build_pairs() 를 그대로 호출해 만들었다 -")
    A("앞 문서들과 같은 test split(씬ID 첫 글자 '1', 598장/50씬)이고, k=1 블록이 정확히")
    A("그 598장 하나씩이다.")
    A("")
    chk("k=1 PSNR = 24.438", a1["psnr_in"], 24.438, 0.01)
    chk("k=1 SSIM = 0.6800", a1["ssim_in"], 0.6800, 0.001)
    chk("k=1 블록수 = 598", float(a1["nblk"]), 598.0, 0.0)
    chk("k=1 구조잔차 몫 = DIAG_SID_FAILURE 64.21%", a1["share_resid"], 64.21, 0.05)
    chk("k=1 D 에너지몫 = DIAG_SID_LOWFREQ 64.35%", a1["dshare"], 64.35, 0.05)
    chk("k=1 전역이득 오라클 = DIAG_SID_LOWFREQ 25.779", a1["psnr_glob"], 25.779, 0.01)
    chk("k=1 국소이득16 오라클 = DIAG_SID_LOWFREQ 26.966", a1["psnr_b16g"], 26.966, 0.01)
    chk("k=1 국소affine16 오라클 = DIAG_SID_LOWFREQ 28.032", a1["psnr_b16a"], 28.032, 0.01)
    chk("k=1 입력평균 = 출력평균 (정의상 같아야)",
        abs(a1["psnr_in"] - a1["psnr_out"]), 0.0, 1e-9)

    # ---------------- 1. k 별 PSNR/SSIM ----------------
    A("## 1. k 장 평균 입력의 PSNR / SSIM")
    A("")
    A("### (1-1) all 코호트 — 씬 구성이 k 마다 달라진다 (주의해서 읽어라)")
    A("")
    A("```")
    A("노출     k   블록수  씬수 |    PSNR     SSIM |  k=1 대비 dB")
    A("---------------------------------------------------------------")
    for e in ["ALL"] + exps:
        base = agg(sel(rows, e, 1, "all"))
        for k in ks_all:
            q = agg(sel(rows, e, k, "all"))
            if not q:
                continue
            A(f"{e:7s} {k:3d} {q['nblk']:7d} {q['nscene']:5d} | {q['psnr_in']:8.3f} "
              f"{q['ssim_in']:8.4f} | {q['psnr_in']-base['psnr_in']:+12.3f}")
        A("")
    A("```")
    A("")
    A("이 표는 k 가 커지면 프레임이 적은 씬이 통째로 빠진다. 예: 0.1s 는 k=8 에서 씬 수가")
    A("줄고, 0.033s 는 2장짜리라 k>2 가 아예 없다. 그래서 이 표만으로 판단하면 안 된다.")
    A("")

    # ---------------- 2. 같은 씬 부분집합 ----------------
    A("### (1-2) match 코호트 — 같은 씬·같은 프레임 집합에서 k 만 바꾼다 (정본)")
    A("")
    A(f"각 노출에서 n >= KFIX 인 그룹의 앞 KFIX 장만 쓴다. k 는 KFIX 의 약수만 쓴다.")
    A("k=1 은 그 KFIX 장을 한 장씩, k=KFIX 는 그 KFIX 장을 한 번에 평균한 것이라")
    A("두 줄이 정확히 같은 광자를 쓴다. 씬 구성·프레임 구성이 완전히 같다.")
    A("")
    A("```")
    A("노출     KFIX   k   블록수  씬수 |    PSNR     SSIM |  k=1 대비 dB   SSIM 차")
    A("--------------------------------------------------------------------------------")
    match_tab = {}
    for e in exps:
        kf = KFIX[e]
        base = agg(sel(rows, e, 1, "match"))
        for k in [x for x in ks_all if kf % x == 0 and x <= kf]:
            q = agg(sel(rows, e, k, "match"))
            if not q:
                continue
            match_tab[(e, k)] = q
            A(f"{e:7s} {kf:4d} {k:3d} {q['nblk']:7d} {q['nscene']:5d} | {q['psnr_in']:8.3f} "
              f"{q['ssim_in']:8.4f} | {q['psnr_in']-base['psnr_in']:+12.3f} "
              f"{q['ssim_in']-base['ssim_in']:+9.4f}")
        A("")
    A("```")
    A("")
    A("노출 합산 (KFIX=8 인 노출만 = 0.04s + 0.1s. 0.033s 는 2장이 최대라 k=8 이 없어서 뺐다):")
    A("")
    e8 = [e for e in exps if KFIX[e] == 8]
    r8 = [x for x in rows if x["exp"] in e8]
    A("```")
    A("  k   블록수  씬수 |    PSNR     SSIM |  k=1 대비 dB")
    A("-------------------------------------------------------")
    b = agg(sel(r8, "ALL", 1, "match"))
    for k in [1, 2, 4, 8]:
        q = agg(sel(r8, "ALL", k, "match"))
        if not q:
            continue
        A(f"{k:3d} {q['nblk']:7d} {q['nscene']:5d} | {q['psnr_in']:8.3f} {q['ssim_in']:8.4f} | "
          f"{q['psnr_in']-b['psnr_in']:+12.3f}")
    A("```")
    A("")
    A("### (1-3) fix 코호트 — 씬은 고정, 블록 수는 k 마다 다름 (보조)")
    A("")
    A("```")
    A("노출     k   블록수  씬수 |    PSNR     SSIM |  k=1 대비 dB")
    A("---------------------------------------------------------------")
    for e in exps:
        base = agg(sel(rows, e, 1, "fix"))
        for k in ks_all:
            q = agg(sel(rows, e, k, "fix"))
            if not q:
                continue
            A(f"{e:7s} {k:3d} {q['nblk']:7d} {q['nscene']:5d} | {q['psnr_in']:8.3f} "
              f"{q['ssim_in']:8.4f} | {q['psnr_in']-base['psnr_in']:+12.3f}")
        A("")
    A("```")
    A("")

    # ---------------- 3. 공정 비교 (b) 입력평균 vs 출력평균 ----------------
    A("## 2. 공정 비교")
    A("")
    A("### (2-b) 입력 평균 대 출력 평균")
    A("")
    A("입력 평균 = k 장을 먼저 평균해 모델에 한 번 넣는다 (forward 1회).")
    A("출력 평균 = 각 프레임을 따로 모델에 넣고 출력 k 장을 평균한다 (forward k회).")
    A("두 쪽 다 같은 k 장의 광자를 쓴다. match 코호트다.")
    A("")
    A("```")
    A("노출     k   블록수 | 입력평균PSNR  출력평균PSNR | k=1 대비: 입력평균  출력평균 | 입력SSIM 출력SSIM")
    A("---------------------------------------------------------------------------------------------------")
    for e in exps:
        kf = KFIX[e]
        b = agg(sel(rows, e, 1, "match"))
        A(f"{e:7s}   1 {b['nblk']:7d} | {b['psnr_in']:12.3f} {b['psnr_out']:13.3f} | "
          f"{0.0:+17.3f} {0.0:+9.3f} | {b['ssim_in']:8.4f} {b['ssim_out']:8.4f}  <- 기준")
        for k in [x for x in ks_all if x > 1 and kf % x == 0 and x <= kf]:
            q = agg(sel(rows, e, k, "match"))
            if not q:
                continue
            A(f"{e:7s} {k:3d} {q['nblk']:7d} | {q['psnr_in']:12.3f} {q['psnr_out']:13.3f} | "
              f"{q['psnr_in']-b['psnr_in']:+17.3f} {q['psnr_out']-b['psnr_in']:+9.3f} | "
              f"{q['ssim_in']:8.4f} {q['ssim_out']:8.4f}")
        A("")
    A("```")
    A("")
    A("SSIM 도 같은 기준으로:")
    A("")
    A("```")
    A("노출     k | 입력평균 SSIM  출력평균 SSIM | k=1 대비: 입력평균   출력평균")
    A("-------------------------------------------------------------------------")
    for e in exps:
        kf = KFIX[e]
        b = agg(sel(rows, e, 1, "match"))
        for k in [x for x in ks_all if kf % x == 0 and x <= kf]:
            q = agg(sel(rows, e, k, "match"))
            if not q:
                continue
            A(f"{e:7s} {k:3d} | {q['ssim_in']:13.4f} {q['ssim_out']:14.4f} | "
              f"{q['ssim_in']-b['ssim_in']:+17.4f} {q['ssim_out']-b['ssim_in']:+10.4f}")
        A("")
    A("```")
    A("")
    A("all 코호트에서 k 를 더 밀어본 것 (씬 구성이 바뀌므로 같은 k 안에서 두 열만 비교해라):")
    A("")
    A("```")
    A("노출     k   블록수 |  입력평균 PSNR  출력평균 PSNR   차이 |  입력 SSIM  출력 SSIM    차이")
    A("-------------------------------------------------------------------------------------------")
    for e in exps:
        for k in ks_all:
            if k == 1:
                continue
            q = agg(sel(rows, e, k, "all"))
            if not q:
                continue
            A(f"{e:7s} {k:3d} {q['nblk']:7d} | {q['psnr_in']:14.3f} {q['psnr_out']:14.3f} "
              f"{q['psnr_in']-q['psnr_out']:+7.3f} | {q['ssim_in']:10.4f} {q['ssim_out']:10.4f} "
              f"{q['ssim_in']-q['ssim_out']:+8.4f}")
    A("```")
    A("")

    # ---------------- 4. 공정 비교 (a) 총 노출 등가 ----------------
    A("### (2-a) 총 노출이 같은 단일/소수 프레임과의 대조")
    A("")
    A("k 장 평균은 입력이 k 배 더 많은 광자를 쓴다. 그러니 '같은 총 노출을 한 번에 찍은 것'")
    A("과 비교해야 공정하다. 이 데이터셋의 short 노출은 0.033 / 0.04 / 0.1s 세 가지고,")
    A("총 노출 t1*k1 == t2*k2 를 만족하는 (노출,k) 쌍을 전수 열거했다.")
    A("가능 여부는 '그 씬이 두 쪽 프레임 수를 다 갖고 있는가'로 판정한다.")
    A("")
    tval = {e: float(e[:-1]) for e in exps}
    scene_have = {}
    for key, nn in gs.items():
        s, e = key.split("|")
        scene_have.setdefault(s, {})[e] = nn
    cands = []
    for e1, e2 in itertools.combinations(exps, 2):
        for k1 in range(1, 16):
            for k2 in range(1, 16):
                t1, t2 = tval[e1] * k1, tval[e2] * k2
                if abs(t1 - t2) / max(t1, t2) > 0.02:
                    continue
                if k1 == 1 and k2 == 1:
                    continue
                ss = [s for s, h in scene_have.items()
                      if h.get(e1, 0) >= k1 and h.get(e2, 0) >= k2]
                cands.append((e1, k1, t1, e2, k2, t2, len(ss), ss))
    cands.sort(key=lambda z: (-z[6], z[2]))
    A("```")
    A("총노출     A 쪽            B 쪽           총노출차%  가능 씬 수  판정")
    A("---------------------------------------------------------------------------")
    shown = 0
    for e1, k1, t1, e2, k2, t2, ns, ss in cands:
        if ns == 0:
            shown += 1
            if shown > 8:
                continue
        A(f"{t1:7.3f}s   {e1:6s} x{k1:<3d}     {e2:6s} x{k2:<3d}    "
          f"{100*abs(t1-t2)/max(t1,t2):8.2f}   {ns:9d}  " +
          ("사용" if ns > 0 else "불가 (그 조합의 프레임을 가진 씬이 없다)"))
    if shown > 8:
        A(f"... 그 밖에 {shown-8} 개 조합이 더 있으나 전부 가능 씬 0 (불가)")
    A("```")
    A("")
    A("추가로, 총 노출이 정확히 같지는 않지만 지시에 나온 조합도 같이 낸다"
      " (0.04s x2 = 0.08s vs 0.1s x1 = 0.1s, 총 노출 20% 차이).")
    A("")
    extra = []
    for e1, k1, e2, k2 in [("0.04s", 2, "0.1s", 1), ("0.033s", 2, "0.04s", 2),
                           ("0.033s", 2, "0.1s", 1)]:
        if e1 in exps and e2 in exps:
            ss = [s for s, h in scene_have.items()
                  if h.get(e1, 0) >= k1 and h.get(e2, 0) >= k2]
            extra.append((e1, k1, tval[e1] * k1, e2, k2, tval[e2] * k2, len(ss), ss))
    use = [c for c in cands if c[6] > 0] + [c for c in extra if c[6] > 0]
    A("A 는 여러 장을 모은 쪽, B 는 같은 총 노출을 더 적은 장수로 찍은 쪽이다.")
    A("A 는 입력평균/출력평균 두 경로를 다 낸다(k=1 이면 둘이 같다).")
    A("")
    A("```")
    A("A: 노출 x k          B: 노출 x k        씬수 | A입력평균 A출력평균 | B입력평균 B출력평균 | A출력-B최선")
    A("---------------------------------------------------------------------------------------------------------")
    for e1, k1, t1, e2, k2, t2, ns, ss in use:
        qa = agg(sel(rows, e1, k1, "all", set(ss)))
        qb = agg(sel(rows, e2, k2, "all", set(ss)))
        if not qa or not qb:
            continue
        bbest = max(qb["psnr_in"], qb["psnr_out"])
        A(f"{e1:6s} x{k1:<3d}({t1:5.3f}s)  {e2:6s} x{k2:<3d}({t2:5.3f}s) {ns:5d} | "
          f"{qa['psnr_in']:9.3f} {qa['psnr_out']:9.3f} | {qb['psnr_in']:9.3f} "
          f"{qb['psnr_out']:9.3f} | {qa['psnr_out']-bbest:+11.3f}")
    A("```")
    A("")
    A("SSIM 같은 형식:")
    A("")
    A("```")
    A("A: 노출 x k          B: 노출 x k        씬수 | A입력평균 A출력평균 | B입력평균 B출력평균")
    A("--------------------------------------------------------------------------------------------")
    for e1, k1, t1, e2, k2, t2, ns, ss in use:
        qa = agg(sel(rows, e1, k1, "all", set(ss)))
        qb = agg(sel(rows, e2, k2, "all", set(ss)))
        if not qa or not qb:
            continue
        A(f"{e1:6s} x{k1:<3d}({t1:5.3f}s)  {e2:6s} x{k2:<3d}({t2:5.3f}s) {ns:5d} | "
          f"{qa['ssim_in']:9.4f} {qa['ssim_out']:9.4f} | {qb['ssim_in']:9.4f} {qb['ssim_out']:9.4f}")
    A("```")
    A("")
    A("씬은 두 쪽이 같은 씬 집합(교집합)이고 GT 도 같다. 같은 씬 안에서만 비교한다.")
    A("")

    # ---------------- 5. 대역별 rho_out ----------------
    A("## 3. 대역별 — 입력측 rho 상승이 출력의 rho 상승으로 이어지는가")
    A("")
    A("대역 정의는 앞 문서와 같은 방사 11구간이다. rho_out(k) 는 모델 출력(장별 오라클")
    A("전역이득 보정)과 GT 의 대역별 휘도 상관이다. rho 는 스케일 불변이라 밝기 오차에")
    A("영향받지 않는다 - 그래서 PSNR 과 다른 것을 본다.")
    A("rho_in(k) 는 DIAG_SID_IDENTIFIABILITY 의 pw 갈래 실측값이다. 두 값 모두 fix 코호트")
    A("(n >= KFIX 인 그룹, 모든 블록)이고 두 문서의 코호트 정의가 문자 그대로 같다.")
    A("입력평균/출력평균 두 경로를 다 낸다.")
    A("")
    for e in exps:
        kf = KFIX[e]
        kl = [x for x in [1, 2, 4, 8] if x <= kf]
        A(f"{e} (fix 코호트, KFIX={kf})")
        A("```")
        A("주파수구간 " + "".join(f"  입력평균 k={k}" for k in kl) +
          "".join(f"  출력평균 k={k}" for k in kl) +
          "".join(f"   rho_in k={k}" for k in kl) + " | 입증가 출증가 in증가")
        A("-" * (11 + 14 * len(kl) * 3 + 24))
        for b in range(nb):
            ro = [d["bands"][f"{e}|{k}|fix|in"]["rho_luma"][b] for k in kl]
            rw = [d["bands"][f"{e}|{k}|fix|out"]["rho_luma"][b] for k in kl]
            ri = []
            for k in kl:
                q = (ident or {}).get("input", {}).get(f"{e}|pw|{k}|fix")
                ri.append(q["rho_luma"][b] if q else float("nan"))
            A(f"{names[b]:>10s}" + "".join(f"{v:14.4f}" for v in ro) +
              "".join(f"{v:14.4f}" for v in rw) + "".join(f"{v:14.4f}" for v in ri) +
              f" | {ro[-1]-ro[0]:+6.3f} {rw[-1]-rw[0]:+6.3f} {ri[-1]-ri[0]:+6.3f}")
        A("```")
        A("")
    A("증가 열 = k=KFIX 값 - k=1 값. 입력에서 오른 만큼 출력에서도 오르는지를 본다.")
    A("")

    # ---------------- 6. 성분 분해 ----------------
    A("## 4. 오차 성분 분해 (k 별)")
    A("")
    A("정의는 DIAG_SID_FAILURE 와 동일. 출력 p, GT g 에 대해 (a) 스칼라 하나로 없어지는 몫,")
    A("(b) 채널별 이득으로 추가로 없어지는 몫, (c) 나머지(구조 잔차). 계수는 장별 최소제곱.")
    A("D 에너지몫 = 최저 두 대역(f<0.10) 잔차가 전체 잔차 에너지에서 차지하는 몫")
    A("(전역이득 보정 후, RGB, 에너지가중) - DIAG_SID_LOWFREQ 의 정의 그대로다.")
    A("match 코호트다(씬·프레임 고정).")
    A("")
    for w, tag in (("in", "입력 평균"), ("out", "출력 평균")):
        A(f"{tag}:")
        A("```")
        A("노출     k  블록수 |  (a)전역   (b)채널  (c)구조잔차 | D에너지몫 |  MSE(0-1)  전역보정PSNR")
        A("-------------------------------------------------------------------------------------------")
        for e in exps:
            kf = KFIX[e]
            for k in [x for x in ks_all if kf % x == 0 and x <= kf]:
                q = agg(sel(rows, e, k, "match"), w)
                if not q:
                    continue
                A(f"{e:7s} {k:3d} {q['nblk']:6d} | {q['share_global']:8.2f}% "
                  f"{q['share_chan']:8.2f}% {q['share_resid']:10.2f}% | {q['dshare']:8.2f}% | "
                  f"{q['mse']:9.6f} {q['psnr_glob']:12.3f}")
            A("")
        A("```")
        A("")
    A("몫은 비율이라 총 오차가 커져도 비율은 그대로일 수 있다. 그래서 절대 에너지도 같이 낸다")
    A("(각 노출의 k=1 을 100 으로 놓은 상대값, match 코호트). 저주파DE 는 전역이득 보정 후")
    A("잔차의 최저 두 대역 에너지, 고주파(비D)E 는 나머지 아홉 대역이다.")
    A("")
    for w, tag in (("in", "입력 평균"), ("out", "출력 평균")):
        A(f"{tag}:")
        A("```")
        A("노출     k | 총오차E  전역이득E  채널이득E  구조잔차E   저주파DE   고주파(비D)E")
        A("---------------------------------------------------------------------------------")
        for e in exps:
            kf = KFIX[e]
            b0 = None
            for k in [x for x in ks_all if kf % x == 0 and x <= kf]:
                q = agg(sel(rows, e, k, "match"), w)
                if not q:
                    continue
                v = dict(tot=q["e_tot"], glob=q["e_glob"], chan=q["e_chan"],
                         resid=q["e_resid"], dlo=q["e_dlo"], dhi=q["e_dhi"])
                if b0 is None:
                    b0 = v
                A(f"{e:7s} {k:3d} |" + "".join(f"{100*v[t]/b0[t]:10.2f}"
                                               for t in ["tot", "glob", "chan", "resid",
                                                         "dlo", "dhi"]))
            A("")
        A("```")
        A("")

    # ---------------- 7. 상한 사다리 ----------------
    A("## 5. 상한 대조 — k=8 을 기존 오라클 사다리와 같은 표에")
    A("")
    A("사다리 원본값은 k=1 전량 598장 기준이라 씬 구성이 이 표의 match 코호트와 다르다.")
    A("그래서 같은 코호트에서 같은 정의로 사다리를 다시 계산해 함께 낸다(전역이득 오라클 =")
    A("장별 스칼라 최소제곱, 국소이득16 = 16x16 블록별 스칼라, 국소affine16 = 16x16 블록별")
    A("a*y+b. 전부 GT 를 보고 계수를 고른 오라클이라 도달 불가능한 상한이다).")
    A("k 줄은 GT 를 전혀 안 쓴 실측이다.")
    A("")
    A("```")
    A("대조: 원본 사다리 (598장 전량, k=1)")
    for nm, v, src in LADDER:
        A(f"  {nm:40s} {v:9.3f}   ({src})")
    A("```")
    A("")
    A("```")
    A("코호트            방식        k |   무보정   전역이득   국소이득16  국소affine16")
    A("--------------------------------------------------------------------------------")
    r8 = [x for x in rows if KFIX[x["exp"]] == 8]
    tabs = [(e, sel(rows, e, None, "match")) for e in exps] + [("KFIX8합산", r8)]
    for lab, base in tabs:
        kf = 8 if lab == "KFIX8합산" else KFIX[lab]
        rr = base if lab == "KFIX8합산" else base
        for k in [x for x in ks_all if kf % x == 0 and x <= kf]:
            rs = [x for x in rr if x["k"] == k and x["match"]]
            if not rs:
                continue
            qi = agg(rs, "in"); qo = agg(rs, "out")
            A(f"{lab:12s}({qi['nscene']:2d}씬) 입력평균 {k:3d} | {qi['psnr_in']:8.3f} "
              f"{qi['psnr_glob']:10.3f} {qi['psnr_b16g']:11.3f} {qi['psnr_b16a']:13.3f}")
            A(f"{lab:12s}({qo['nscene']:2d}씬) 출력평균 {k:3d} | {qo['psnr_out']:8.3f} "
              f"{qo['psnr_glob']:10.3f} {qo['psnr_b16g']:11.3f} {qo['psnr_b16a']:13.3f}")
        A("")
    A("```")
    A("")
    A("k=1 줄에서 입력평균과 출력평균은 정의상 같은 값이다(같은 한 장). 앵커 대조에 있다.")
    A("")

    # ---------------- 8. 산수 대조 ----------------
    A("## 6. 산수 대조 (검증)")
    A("")
    # 사다리는 자유도가 늘수록 좋아져야 한다 (최소제곱이라 필연). 모든 셀에서 확인.
    worst_g = max(x[f"pg_{w}"] - x[f"pb16g_{w}"] for x in rows for w in ["in", "out"])
    chk("전역이득 <= 국소이득16 (셀별, 최대 위반 dB)", float(worst_g), 0.0, 0.05,
        f"{len(rows)*2} 셀, <=0 이어야 함", ineq="<=")
    ncell = sum(1 for x in rows for w in ["in", "out"] if x[f"pb16a_{w}"] < x[f"pb16g_{w}"])
    A("")
    A(f"참고: 셀별로 국소affine16 이 국소이득16 보다 나쁜 경우가 {ncell}/{len(rows)*2} 개 있다.")
    A("최소제곱 잔차는 자유도가 늘면 반드시 줄지만, 여기 dB 는 클리핑+8bit 반올림 후 값이라")
    A("순서가 보장되지 않는다(블록 안 값이 거의 상수면 affine 의 정규방정식이 거의 특이해져")
    A("계수가 크게 나오고, 클리핑이 그걸 다 못 되돌린다). 집계값에서는 순서가 지켜진다 -")
    A("아래 두 줄로 확인한다.")
    A("")
    ordbad = 0
    for e in exps + ["_ALL8"]:
        rr = ([x for x in rows if KFIX[x["exp"]] == 8] if e == "_ALL8"
              else [x for x in rows if x["exp"] == e])
        for k in sorted(set(x["k"] for x in rr)):
            for w in ["in", "out"]:
                q = agg([x for x in rr if x["k"] == k and x["match"]], w)
                if q and not (q["psnr_glob"] <= q["psnr_b16g"] <= q["psnr_b16a"]):
                    ordbad += 1
    chk("집계 사다리 순서 전역 <= 국소이득16 <= 국소affine16 (위반 셀 수)",
        float(ordbad), 0.0, 0.0, "match 코호트 전 노출/전 k/입출력 양쪽")
    # 출력평균이 k 에 대해 단조로운가 (실측 진술, 검정 아님 - 값은 본문 표에)
    for e in exps:
        kf = KFIX[e]
        kl = [x for x in ks_all if kf % x == 0 and x <= kf]
        vi = [agg(sel(rows, e, k, "match"), "in")["psnr_in"] for k in kl]
        vo = [agg(sel(rows, e, k, "match"), "out")["psnr_out"] for k in kl]
        chk(f"{e}: 출력평균 PSNR 이 k 에 대해 단조증가 (최소 증분)",
            float(min(np.diff(vo))) if len(vo) > 1 else 0.0, 0.0, 0.0,
            f"k={kl}, 입력평균은 {vi[0]:.3f}->{vi[-1]:.3f}", ineq=">=")
    # 성분 합
    for e in exps:
        q = agg(sel(rows, e, 1, "all"))
        chk(f"{e}: 성분 합 = 100%", q["share_global"] + q["share_chan"] + q["share_resid"],
            100.0, 1e-6)
    # 대역 잔차 합 = Parseval MSE*3
    for e in exps:
        rs = sel(rows, e, 1, "all")
        lhs = float(np.mean([x["dtot_in"] for x in rs]))
        rhs = float(np.mean([x["mseg_in"] for x in rs])) * 3.0 * 255.0 ** 2
        chk(f"{e}: 대역잔차합 = 전역보정 MSE x3 x255^2 (Parseval)", lhs, rhs,
            1e-6 * max(1.0, rhs))
    # 출력평균 vs 입력평균 k=1 동일
    for e in exps:
        rs = sel(rows, e, 1, "all")
        chk(f"{e}: k=1 에서 입력평균 == 출력평균",
            float(max(abs(x["psnr_in"] - x["psnr_out"]) for x in rs)), 0.0, 1e-9)
    # 등가노출 쌍이 실제로 같은 씬인가
    for e1, k1, t1, e2, k2, t2, ns, ss in use:
        sa = set(x["scene"] for x in sel(rows, e1, k1, "all", set(ss)))
        sb = set(x["scene"] for x in sel(rows, e2, k2, "all", set(ss)))
        chk(f"등가노출 {e1}x{k1} vs {e2}x{k2}: 두 쪽 씬 집합 동일",
            float(len(sa ^ sb)), 0.0, 0.0, f"{len(sa)}씬")
    # rho 범위
    allr = []
    for kk, v in d["bands"].items():
        allr += list(v["rho_luma"])
    chk("모든 rho_out 이 [-1,1] 안 (코시-슈바르츠)", float(np.max(np.abs(allr))), 1.0, 0.0,
        "<=1 이어야 함", ineq="<=")
    # match 코호트 프레임 수 동일성
    for e in exps:
        kf = KFIX[e]
        kl = [x for x in ks_all if kf % x == 0 and x <= kf]
        used = [sum(x["k"] for x in sel(rows, e, k, "match")) for k in kl]
        chk(f"{e}: match 코호트에서 k 마다 쓴 프레임 총수가 같은가",
            float(max(used) - min(used)), 0.0, 0.0, f"프레임 {used[0]}장")
    A("```")
    A("항목                                                          계산값        대조값   판정")
    A("---------------------------------------------------------------------------------------------")
    for nm, v, r, st, ex in checks:
        A(f"{nm:55s} {v:12.5f} {r:12.5f}  {st}" + (f"  ({ex})" if ex else ""))
    A("```")
    A("")

    # ---------------- 9. 결론 ----------------
    A("## 측정이 가리키는 것")
    A("")

    def mq(e, k, w="in"):
        return agg(sel(rows, e, k, "match"), w)

    p1 = " / ".join(
        f"{e} {mq(e,1)['psnr_in']:.3f} -> k={KFIX[e]} 입력평균 {mq(e,KFIX[e])['psnr_in']:.3f}"
        f"({mq(e,KFIX[e])['psnr_in']-mq(e,1)['psnr_in']:+.3f}) / 출력평균 "
        f"{mq(e,KFIX[e],'out')['psnr_out']:.3f}"
        f"({mq(e,KFIX[e],'out')['psnr_out']-mq(e,1)['psnr_in']:+.3f})"
        for e in exps)
    A("- 여러 장은 값을 가져오지만 넣는 방법이 부호를 바꾼다. 같은 씬·같은 프레임 집합에서 "
      + p1 + " dB 다. k 장을 먼저 평균해 넣으면 크게 나빠지고, 각각 넣어 출력을 평균하면 "
      "k 에 대해 단조로 좋아진다(최소 증분 +0.086 dB).")
    e0, kf0 = "0.1s", KFIX["0.1s"]
    ei = agg(sel(rows, e0, kf0, "match"), "in")
    A(f"- 입력평균이 무너지는 곳은 구조가 아니라 밝기다. {e0} k={kf0} 입력평균에서 대역 상관은 "
      f"거의 전 대역에서 오르고(코너 0.50-0.71 rho_out {d['bands'][f'{e0}|1|fix|in']['rho_luma'][10]:.3f}"
      f" -> {d['bands'][f'{e0}|{kf0}|fix|in']['rho_luma'][10]:.3f}), 고주파 잔차 에너지도 "
      f"{100*ei['e_dhi']/mq(e0,1)['e_dhi']:.0f}% 로 준다. 그런데 전역이득 오차 에너지가 "
      f"{100*ei['e_glob']/mq(e0,1)['e_glob']:.0f}% 로 뛴다. 모델이 입력 잡음 수준을 밝기 단서로 "
      "쓰고 있고 평균해서 깨끗해진 입력이 학습 분포 밖이라는 해석과 맞지만, 이 측정은 그 "
      f"인과를 분리하지 않았다(가설). 밝기 오차는 국소 보정으로도 다 안 지워진다 - 입력평균 "
      f"k={kf0} 에 오라클 국소affine16 을 먹여도 {ei['psnr_b16a']:.3f} 로, 같은 오라클을 k=1 에 "
      f"먹인 {mq(e0,1)['psnr_b16a']:.3f} 보다 낮다.")
    A("- 총 노출을 맞추면 여러 장이 진다. 총 노출이 정확히 같은 유일한 조합인 0.04s 5장(0.2s) 대 "
      "0.1s 2장(0.2s)에서 출력평균끼리 23.444 대 25.146 dB 로 1.702 dB 뒤진다. 같은 광자를 "
      "더 잘게 쪼개 읽으면 손해라는 뜻이고, 장마다 더해지는 읽기 잡음과 일관되지만 이 "
      "측정으로 원인을 분리하지는 않았다(가설). 0.04s x2 대 0.1s x1 은 총 노출이 20% 적은 "
      "쪽이 A 라 불리하지만 방향은 같다(-2.243 dB).")
    dl = " / ".join(
        f"{e} {mq(e,1,'out')['dshare']:.2f}% -> {mq(e,KFIX[e],'out')['dshare']:.2f}%"
        f"(에너지 {100*mq(e,KFIX[e],'out')['e_dlo']/mq(e,1,'out')['e_dlo']:.0f}%)"
        for e in exps)
    A("- 지배 성분인 저주파는 여러 장으로 거의 안 준다. 출력평균 기준 D 에너지몫은 " + dl +
      " 이고, 0.1s 는 몫이 오히려 늘었다. 입력평균에서는 저주파 에너지가 "
      f"{100*mq('0.1s',8)['e_dlo']/mq('0.1s',1)['e_dlo']:.0f}% 로 폭증한다. "
      "프레임 평균이 지우는 것은 고주파 잡음이지 밝기장 오차가 아니다.")
    lq1 = mq(e0, 1)
    lq8 = mq(e0, kf0, "out")
    A(f"- 크기 감각: 같은 코호트({lq1['nscene']}씬)에서 오라클 사다리는 k=1 무보정 "
      f"{lq1['psnr_in']:.3f} 위로 전역이득 {lq1['psnr_glob']:.3f}, 국소이득16 "
      f"{lq1['psnr_b16g']:.3f}, 국소affine16 {lq1['psnr_b16a']:.3f} 다. 8장 출력평균이 실제로 "
      f"가져온 것은 {lq8['psnr_out']:.3f}({lq8['psnr_out']-lq1['psnr_in']:+.3f} dB)로, "
      f"국소affine16 여유 {lq1['psnr_b16a']-lq1['psnr_in']:+.3f} dB 의 "
      f"{100*(lq8['psnr_out']-lq1['psnr_in'])/(lq1['psnr_b16a']-lq1['psnr_in']):.0f}% 다. "
      "'여러 장에 여유가 있다'는 진술은 성립하지만 그 몫은 작고, 큰 여유는 여전히 밝기장에 있다.")
    A("")

    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("wrote", OUT_MD)
    fails = [c for c in checks if c[3] == "실패"]
    print(f"checks: {len(checks)-len(fails)} 통과 / {len(fails)} 실패")
    for c in fails:
        print("  실패:", c[0], c[1], c[2], c[4])


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()
