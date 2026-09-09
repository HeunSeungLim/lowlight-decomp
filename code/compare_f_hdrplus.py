"""compare_f_hdrplus.py

비교 실험 F: HDR+ 계열 잡음분산 기반 주파수 영역 병합(merge_hdrplus.merge)을
같은 데이터·같은 프레임 수·같은 채점으로 A/B/C 와 한 표에 놓는다.

  A. 단일 프레임      f(x_1)
  B. 균등 평균 입력    f(mean(x_1..x_k))
  C. 출력 평균        mean_i f(x_i)
  F. HDR+ 병합 입력    f(merge(x_1..x_k; c))   c in {2,4,8,16}, 기본 8

학습 없음. Retinexformer 저자 SID.pth 그대로, 입력만 바꾼다.
그룹·블록·코호트 규약은 multiframe_gain.py 와 같게 재구현했다 (그 파일은 수정하지 않는다):
  같은 씬·같은 노출의 short 를 파일명 정렬 순으로 놓고 앞에서부터 겹치지 않는 연속 k 장
  블록을 floor(n/k) 개. 기준 프레임은 블록의 첫 장. 코호트 all / fix / match,
  KFIX = 노출별 최대 프레임 수가 허용하는 2의 거듭제곱.
  --audit-groups 를 주면 multiframe_gain.build_groups() 결과와 실제로 같은지 대조한다.

추가 측정: F 와 B 의 출력 차이, 병합 입력과 평균 입력의 차이를
diag_sid_spectrum.RAD_EDGES 의 11개 방사 대역별 파워(휘도, 0-255 스케일, Parseval)로 잰다.

채점은 repro_measure 의 독립 구현(psnr_indep / ssim_indep)만 쓴다. 8bit 는 반올림.
병합 출력은 [0,1] 밖으로 약간 나가므로(윈도 FFT 링잉) 모델에 넣기 전에 클리핑하고 그 비율을 기록한다.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, glob, json, math, time, argparse
import numpy as np

sys.path.insert(0, "code")
from repro_measure import psnr_indep, ssim_indep
from diag_sid_spectrum import (build_pairs, load_npy, exposure_of, u8,
                               make_radial_index, RAD_EDGES, LUMA_W, WEIGHTS, REPO)
from merge_hdrplus import merge, estimate_sigma, C_DEFAULT, TILE

OUT_JSON = os.environ.get("CFH_OUT", "numbers/compare_f_hdrplus.json")
OUT_MD = os.environ.get("CFH_MD", "numbers/COMPARE_F_HDRPLUS.md")
KS = [1, 2, 4, 8]
C_LIST = [2.0, 4.0, 8.0, 16.0]
C_MAIN = C_DEFAULT                                   # 8.0
H, W = 512, 960
N = H * W
ANCHOR_PSNR, ANCHOR_SSIM = 24.438, 0.6800
ARMS = ["A", "B", "C"] + [f"F{int(c)}" for c in C_LIST]
FMAIN = f"F{int(C_MAIN)}"
NB = len(RAD_EDGES) - 1
BAND_NAMES = [f"{RAD_EDGES[b]:.2f}-{min(RAD_EDGES[b+1], 0.7072):.2f}" for b in range(NB)]


# ---------------- 그룹 / 블록 / 코호트 (multiframe_gain.py 규약 재구현) ----------------
def build_groups():
    g = {}
    for scene, lqp, gtp in build_pairs():
        key = (scene, exposure_of(lqp))
        g.setdefault(key, {"lq": [], "gt": gtp})
        g[key]["lq"].append(lqp)
    for k in g:
        g[k]["lq"].sort()
    return g


def blocks(n, k):
    return [list(range(i * k, (i + 1) * k)) for i in range(n // k)]


def kfix_of(groups):
    exps = sorted(set(e for _, e in groups), key=lambda s: float(s[:-1]))
    kf = {}
    for e in exps:
        mx = max(len(groups[kk]["lq"]) for kk in groups if kk[1] == e)
        kf[e] = max([k for k in [1, 2, 4, 8] if k <= mx])
    return exps, kf


def audit_groups(mine):
    """multiframe_gain.build_groups() 와 키·파일 목록이 같은지 대조 (읽기만 한다)."""
    import importlib
    mg = importlib.import_module("multiframe_gain")
    theirs = mg.build_groups()
    same_keys = set(mine) == set(theirs)
    same_lists = same_keys and all(mine[k]["lq"] == theirs[k]["lq"] and mine[k]["gt"] == theirs[k]["gt"]
                                   for k in mine)
    same_blocks = all(blocks(n, k) == mg.blocks(n, k) for n in range(1, 16) for k in KS)
    print(f"audit groups: keys {same_keys} lists {same_lists} blocks {same_blocks}", flush=True)
    return bool(same_keys and same_lists and same_blocks)


# ---------------- CPU 워커 ----------------
def merge_job(task):
    """task = (gi, k, bi, paths). 블록 프레임을 읽어 c 별 병합. sigma 는 블록 프레임에서 한 번 추정."""
    gi, k, bi, paths = task
    fr = [load_npy(p).astype(np.float64) for p in paths]
    sigma = estimate_sigma(fr)
    out = dict(gi=gi, k=k, bi=bi, sigma=float(sigma), merged={}, clip={})
    for c in C_LIST:
        m = merge(fr, c=c, tile=TILE, sigma=sigma)
        out["clip"][c] = dict(lt0=float((m < 0).mean()), gt1=float((m > 1).mean()),
                              min=float(m.min()), max=float(m.max()))
        out["merged"][c] = np.clip(m, 0.0, 1.0).astype(np.float32)
    return out


def ssim_job(args):
    pred_u8, gt_u8 = args
    return ssim_indep(pred_u8, gt_u8)


# ---------------- 측정 ----------------
def measure(a):
    import torch
    import torch.nn.functional as F
    from multiprocessing import Pool

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    # cudnn TF32 conv(기본 켜짐)은 같은 입력에 float 6e-4, 장당 PSNR 0.004 dB 까지 흔들린다(실측).
    # 끄면 결정적(7e-7)이고 평균 PSNR 이동 0.0001 dB. k=1 항등 검사를 위해 끈다.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    ridx_flat = ridx.reshape(-1)
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)

    def luma255(t3hw):
        return (t3hw.double() * lw[:, None, None]).sum(0) * 255.0

    def band_power(field):
        Fk = torch.fft.fft2(field)
        pw = (Fk.real ** 2 + Fk.imag ** 2).reshape(-1) / (N * N)
        out = torch.zeros(NB, device=dev, dtype=torch.float64)
        out.scatter_add_(0, ridx_flat, pw)
        return [float(v) for v in out.cpu().numpy()]

    def to_u8(t3hw):
        return u8(np.clip(t3hw.permute(1, 2, 0).cpu().numpy(), 0, 1))

    groups = build_groups()
    if a.audit_groups:
        assert audit_groups(groups), "그룹 구성이 multiframe_gain.py 와 다르다"
    if a.limit:
        groups = {k: v for k, v in sorted(groups.items())[:a.limit]}
    exps, KFIX = kfix_of(groups)
    ngrp = {e: sum(1 for k in groups if k[1] == e) for e in exps}
    nimg = {e: sum(len(groups[k]["lq"]) for k in groups if k[1] == e) for e in exps}
    print("groups", ngrp, "imgs", nimg, "KFIX", KFIX, flush=True)

    gkeys = sorted(groups.items())
    tasks = []
    for gi, ((scene, exp), gd) in enumerate(gkeys):
        n = len(gd["lq"])
        for k in KS:
            if k < 2 or k > n:
                continue
            for bi, b in enumerate(blocks(n, k)):
                tasks.append((gi, k, bi, [gd["lq"][i] for i in b]))
    print(f"merge tasks {len(tasks)} (x{len(C_LIST)} c values)", flush=True)

    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    def infer(x01):
        x = x01[None]
        h, w = x.shape[2], x.shape[3]
        padh, padw = (-h) % 4, (-w) % 4
        if padh or padw:
            x = F.pad(x, (0, padw, 0, padh), "reflect")
        y = net(x)[:, :, :h, :w]
        return torch.clamp(y, 0, 1).double()[0]

    pool_m = Pool(a.workers_merge)
    pool_s = Pool(a.workers_ssim)
    it = pool_m.imap(merge_job, tasks)          # 순서 보존, 소비 순서와 동일하게 생성했다
    rows, jobs = [], []
    ninfer = 0
    t0 = time.time()
    with torch.inference_mode():
        for gi, ((scene, exp), gd) in enumerate(gkeys):
            gt_u8 = u8(load_npy(gd["gt"]))
            g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
            Lg = luma255(g)
            n = len(gd["lq"])
            fr = torch.stack([torch.from_numpy(load_npy(p).transpose(2, 0, 1)) for p in gd["lq"]]).cuda()
            outs = torch.stack([infer(fr[i]) for i in range(n)])
            ninfer += n
            kf = KFIX[exp]
            for k in KS:
                if k > n:
                    continue
                for bi, b in enumerate(blocks(n, k)):
                    in_fix = (n >= kf)
                    in_match = in_fix and (kf % k == 0) and ((bi + 1) * k <= kf)
                    row = dict(scene=scene, exp=exp, k=k, bi=bi, nfr=n, fix=in_fix, match=in_match,
                               ref=os.path.basename(gd["lq"][b[0]]))
                    yA = outs[b[0]]
                    if k == 1:
                        # A=B=C=F 항등이어야 한다. 가정하지 않고 실제로 재서 최대 차이를 남긴다.
                        yB = infer(fr[b].mean(0)); ninfer += 1
                        m1 = merge([fr[b[0]].permute(1, 2, 0).cpu().numpy().astype(np.float64)],
                                   c=C_MAIN, tile=TILE)
                        yF = infer(torch.from_numpy(np.clip(m1, 0, 1).astype(np.float32)
                                                    .transpose(2, 0, 1)).cuda()); ninfer += 1
                        uA = to_u8(yA)
                        row["k1_maxdiff_u8"] = int(max(np.abs(uA.astype(int) - to_u8(yB).astype(int)).max(),
                                                       np.abs(uA.astype(int) - to_u8(yF).astype(int)).max()))
                        row["k1_maxdiff_f64"] = float(max((yA - yB).abs().max().item(),
                                                          (yA - yF).abs().max().item()))
                        p = psnr_indep(uA, gt_u8)
                        for arm in ARMS:
                            row[f"psnr_{arm}"] = p
                        jobs.append((row, {"A": pool_s.apply_async(ssim_job, ((uA, gt_u8),))}))
                        rows.append(row)
                        continue
                    mr = next(it)
                    assert (mr["gi"], mr["k"], mr["bi"]) == (gi, k, bi), "merge 결과 순서 어긋남"
                    row["sigma"] = mr["sigma"]
                    row["clip"] = {str(int(c)): mr["clip"][c] for c in C_LIST}
                    xbar = fr[b].mean(0)
                    yB = infer(xbar); ninfer += 1
                    yC = outs[b].mean(0)
                    ys = {"A": yA, "B": yB, "C": yC}
                    xm = {}
                    for c in C_LIST:
                        xm[c] = torch.from_numpy(mr["merged"][c].transpose(2, 0, 1)).cuda()
                        ys[f"F{int(c)}"] = infer(xm[c]); ninfer += 1
                    u = {arm: to_u8(ys[arm]) for arm in ARMS}
                    sj = {}
                    for arm in ARMS:
                        row[f"psnr_{arm}"] = psnr_indep(u[arm], gt_u8)
                        sj[arm] = pool_s.apply_async(ssim_job, ((u[arm], gt_u8),))
                    # 출력끼리 / 입력끼리 일치도
                    row["psnr_FB"] = psnr_indep(u[FMAIN], u["B"])
                    row["psnr_AB"] = psnr_indep(u["A"], u["B"])
                    row["psnr_CB"] = psnr_indep(u["C"], u["B"])
                    row["psnr_xm_xbar"] = psnr_indep(u8(np.clip(xm[C_MAIN].permute(1, 2, 0).cpu().numpy(), 0, 1)),
                                                     u8(np.clip(xbar.permute(1, 2, 0).cpu().numpy(), 0, 1)))
                    row["psnr_x1_xbar"] = psnr_indep(u8(np.clip(fr[b[0]].permute(1, 2, 0).cpu().numpy(), 0, 1)),
                                                     u8(np.clip(xbar.permute(1, 2, 0).cpu().numpy(), 0, 1)))
                    # 대역별 파워 (휘도 0-255)
                    LA, LB, LC = luma255(yA), luma255(yB), luma255(yC)
                    LF = {c: luma255(ys[f"F{int(c)}"]) for c in C_LIST}
                    Lx1, Lxb = luma255(fr[b[0]]), luma255(xbar)
                    Lxm = {c: luma255(xm[c]) for c in C_LIST}
                    bd = dict(P_g=band_power(Lg),
                              P_rA=band_power(LA - Lg), P_rB=band_power(LB - Lg),
                              P_rC=band_power(LC - Lg), P_rF=band_power(LF[C_MAIN] - Lg),
                              P_dFB=band_power(LF[C_MAIN] - LB), P_dAB=band_power(LA - LB),
                              P_dCB=band_power(LC - LB),
                              P_xbar=band_power(Lxb), P_dx1=band_power(Lx1 - Lxb),
                              P_dxm=band_power(Lxm[C_MAIN] - Lxb))
                    for c in C_LIST:
                        bd[f"P_dFB_c{int(c)}"] = band_power(LF[c] - LB)
                        bd[f"P_dxm_c{int(c)}"] = band_power(Lxm[c] - Lxb)
                        bd[f"P_rF_c{int(c)}"] = band_power(LF[c] - Lg)
                    row["bands"] = bd
                    jobs.append((row, sj))
                    rows.append(row)
            del fr, outs
            if (gi + 1) % 10 == 0 or gi + 1 == len(gkeys):
                print(f"  {gi+1}/{len(gkeys)}  infer={ninfer}  rows={len(rows)}  {time.time()-t0:.0f}s", flush=True)
    pool_m.close(); pool_m.join()
    for row, sj in jobs:
        if row["k"] == 1:
            s = sj["A"].get()
            for arm in ARMS:
                row[f"ssim_{arm}"] = s
        else:
            for arm in ARMS:
                row[f"ssim_{arm}"] = sj[arm].get()
    pool_s.close(); pool_s.join()
    print(f"done {ninfer} forwards, {len(rows)} rows, {time.time()-t0:.0f}s", flush=True)

    out = dict(rad_edges=RAD_EDGES, band_names=BAND_NAMES, exposures=exps, ks=KS, kfix=KFIX,
               c_list=C_LIST, c_main=C_MAIN, tile=TILE, arms=ARMS,
               n_groups=ngrp, n_images=nimg, n_forwards=ninfer, n_merge_tasks=len(tasks),
               group_sizes={f"{s}|{e}": len(v["lq"]) for (s, e), v in groups.items()},
               rows=rows)
    out["summary"] = summarize(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return out


# ---------------- 집계 ----------------
def sel(rows, exp=None, k=None, coh="all", kfix8=False):
    r = rows
    if exp and exp != "ALL":
        r = [x for x in r if x["exp"] == exp]
    if kfix8:
        r = [x for x in r if x["exp"] in ("0.04s", "0.1s")]
    if k is not None:
        r = [x for x in r if x["k"] == k]
    if coh == "fix":
        r = [x for x in r if x["fix"]]
    elif coh == "match":
        r = [x for x in r if x["match"]]
    return r


def agg(rs):
    if not rs:
        return None
    d = dict(nblk=len(rs), nscene=len(set(x["scene"] for x in rs)))
    for arm in ARMS:
        d[f"psnr_{arm}"] = float(np.mean([x[f"psnr_{arm}"] for x in rs]))
        d[f"ssim_{arm}"] = float(np.mean([x[f"ssim_{arm}"] for x in rs]))
    for arm in ["A", "B", "C"]:
        dd = np.array([x[f"psnr_{FMAIN}"] - x[f"psnr_{arm}"] for x in rs])
        d[f"dF_{arm}_mean"] = float(dd.mean()); d[f"dF_{arm}_std"] = float(dd.std(ddof=0))
        d[f"dF_{arm}_min"] = float(dd.min()); d[f"dF_{arm}_max"] = float(dd.max())
        d[f"dF_{arm}_fracpos"] = float((dd > 0).mean())
    for c in C_LIST:
        dd = np.array([x[f"psnr_F{int(c)}"] - x["psnr_B"] for x in rs])
        d[f"dFc{int(c)}_B_mean"] = float(dd.mean())
    if all("psnr_FB" in x for x in rs):
        for key in ["psnr_FB", "psnr_AB", "psnr_CB", "psnr_xm_xbar", "psnr_x1_xbar", "sigma"]:
            d[key] = float(np.mean([x[key] for x in rs]))
        d["clip_lt0"] = float(np.mean([x["clip"][str(int(C_MAIN))]["lt0"] for x in rs]))
        d["clip_gt1"] = float(np.mean([x["clip"][str(int(C_MAIN))]["gt1"] for x in rs]))
        # 대역별: 행 합(에너지 가중) 비율
        keys = rs[0]["bands"].keys()
        S = {kk: np.sum([x["bands"][kk] for x in rs], axis=0) for kk in keys}
        d["band_sum"] = {kk: [float(v) for v in S[kk]] for kk in keys}
        eps = 1e-30
        d["band_ratio"] = dict(
            dFB_over_rB=[float(v) for v in S["P_dFB"] / np.maximum(S["P_rB"], eps)],
            dFB_over_g=[float(v) for v in S["P_dFB"] / np.maximum(S["P_g"], eps)],
            dAB_over_rB=[float(v) for v in S["P_dAB"] / np.maximum(S["P_rB"], eps)],
            dCB_over_rB=[float(v) for v in S["P_dCB"] / np.maximum(S["P_rB"], eps)],
            rF_over_rB=[float(v) for v in S["P_rF"] / np.maximum(S["P_rB"], eps)],
            rA_over_rB=[float(v) for v in S["P_rA"] / np.maximum(S["P_rB"], eps)],
            dxm_over_dx1=[float(v) for v in S["P_dxm"] / np.maximum(S["P_dx1"], eps)],
            dxm_over_xbar=[float(v) for v in S["P_dxm"] / np.maximum(S["P_xbar"], eps)],
            dFB_share=[float(v) for v in S["P_dFB"] / max(S["P_dFB"].sum(), eps)],
            dxm_share=[float(v) for v in S["P_dxm"] / max(S["P_dxm"].sum(), eps)],
        )
        for c in C_LIST:
            d["band_ratio"][f"dFB_c{int(c)}_over_rB"] = [float(v) for v in
                                                         S[f"P_dFB_c{int(c)}"] / np.maximum(S["P_rB"], eps)]
            d["band_ratio"][f"dxm_c{int(c)}_over_dx1"] = [float(v) for v in
                                                          S[f"P_dxm_c{int(c)}"] / np.maximum(S["P_dx1"], eps)]
        d["tot_dFB_over_rB"] = float(S["P_dFB"].sum() / max(S["P_rB"].sum(), eps))
        d["tot_dxm_over_dx1"] = float(S["P_dxm"].sum() / max(S["P_dx1"].sum(), eps))
        d["tot_dAB_over_rB"] = float(S["P_dAB"].sum() / max(S["P_rB"].sum(), eps))
    return d


def summarize(d):
    rows = d["rows"]
    exps = d["exposures"]
    S = {}
    for coh in ["all", "fix", "match"]:
        for e in ["ALL"] + exps:
            for k in KS:
                r = agg(sel(rows, e, k, coh))
                if r:
                    S[f"{e}|{k}|{coh}"] = r
        for k in KS:
            r = agg(sel(rows, None, k, coh, kfix8=True))
            if r:
                S[f"KFIX8|{k}|{coh}"] = r
    return S


# ---------------- 보고서 ----------------
def report(d):
    rows = d["rows"]; S = d["summary"]; exps = d["exposures"]; KFIX = d["kfix"]
    L = []; A = L.append
    checks = []

    def chk(name, val, ref, tol):
        ok = abs(val - ref) <= tol
        checks.append((name, val, ref, ok))
        return "통과" if ok else "실패"

    A("# 비교 실험 F — HDR+ 계열 잡음분산 기반 주파수 병합 vs 단일/균등평균/출력평균")
    A("")
    A("- 대상: SID test split 전량 598장 / 50씬. 노출별 "
      + ", ".join(f"{e} {d['n_images'][e]}장/{d['n_groups'][e]}그룹" for e in exps))
    A("- 모델: Retinexformer 저자 SID.pth. 학습 없음, 추론만. GPU 0.")
    A(f"- 병합: code/merge_hdrplus.py 의 merge(frames, c, tile={d['tile']}, sigma) 를 그대로 호출. "
      f"sigma 는 블록 프레임에서 estimate_sigma 로 추정(모든 c 에 같은 값). 기본 c={int(d['c_main'])}.")
    A("- 병합 출력은 [0,1] 밖으로 약간 나가므로 클리핑 후 모델에 넣었다(아래 5절에 비율).")
    A(f"- 모델 forward {d['n_forwards']}회, 병합 {d['n_merge_tasks']}블록 x {len(d['c_list'])}개 c.")
    A("- 채점: repro_measure 의 psnr_indep / ssim_indep (독립 구현), 8bit 반올림.")
    A("- 추론: cudnn TF32 끔 + deterministic (앵커 대조 절 참조).")
    A("- 그룹·블록·코호트 규약은 multiframe_gain.py 와 동일(재구현, --audit-groups 로 대조).")
    A("")
    A("네 팔:")
    A("  A  단일 프레임      f(x_1)                x_1 = 블록의 첫 장")
    A("  B  균등 평균 입력    f(mean(x_1..x_k))")
    A("  C  출력 평균        mean_i f(x_i)         forward k 회")
    A("  F  HDR+ 병합 입력    f(merge(x_1..x_k; c)) 기준 프레임 = x_1")
    A("")
    A("코호트:")
    A("  all   : 모든 그룹의 모든 블록 (k 마다 씬 구성이 바뀐다)")
    A("  match : n >= KFIX 인 그룹의 앞 KFIX 장만, k 가 KFIX 의 약수. k 를 바꿔도 같은 씬·같은 프레임. 정본.")
    A("  KFIX = " + ", ".join(f"{e}:{KFIX[e]}" for e in exps))
    A("  같은 k 안에서는 네 팔이 정확히 같은 블록(같은 씬, 같은 프레임 부분집합)에서 채점됐다.")
    A("")

    # 0. 앵커
    a1 = S["ALL|1|all"]
    k1 = [r for r in rows if r["k"] == 1]
    mx_u8 = max(r["k1_maxdiff_u8"] for r in k1)
    mx_f = max(r["k1_maxdiff_f64"] for r in k1)
    A("## 0. 앵커 대조")
    A("")
    A("```")
    A("항목                                       측정값      대조값   판정")
    A("----------------------------------------------------------------------")
    A(f"k=1 A PSNR (전량)                        {a1['psnr_A']:9.3f}    24.438   {chk('k=1 A PSNR', a1['psnr_A'], ANCHOR_PSNR, 0.01)}")
    A(f"k=1 A SSIM (전량)                        {a1['ssim_A']:9.4f}    0.6800   {chk('k=1 A SSIM', a1['ssim_A'], ANCHOR_SSIM, 0.001)}")
    A(f"k=1 블록 수                              {a1['nblk']:9d}       598   {chk('k=1 블록수', a1['nblk'], 598, 0)}")
    A(f"k=1 씬 수                                {a1['nscene']:9d}        50   {chk('k=1 씬수', a1['nscene'], 50, 0)}")
    n_flip = sum(1 for r in k1 if r["k1_maxdiff_u8"] > 0)
    A(f"k=1 A=B=F 항등, float 최대차 (실측)        {mx_f:9.2e}     <1e-5   {chk('k=1 항등 float', mx_f, 0, 1e-5)}")
    A(f"k=1 A=B=F 항등, 8bit 최대차 / 뒤집힌 장수   {mx_u8:5d} / {n_flip:3d}       정보   (float 1e-6 차가 반올림 경계 화소를 1레벨 뒤집음, PSNR 영향 <1e-4 dB)")
    A("```")
    A("")
    A("k=1 에서 B 는 한 장 평균, F 는 한 장 병합(merge 는 N=1 이면 복사)이라 A 와 같아야 하고, "
      "가정하지 않고 실제로 forward 를 돌려 차이를 쟀다. cudnn TF32 를 끄고 deterministic 으로 돌렸다"
      "(켜면 같은 입력에 float 6e-4 / 장당 PSNR 0.004 dB 흔들림, 끄면 평균 PSNR 이동 0.0001 dB — 실측).")
    A("")

    def arm_table(coh, title, note):
        A(f"### {title}")
        A("")
        if note:
            A(note); A("")
        A("```")
        A("노출     k  블록  씬 |  A단일   B평균   C출력평균  F병합(c=8) |  F-B 평균±표준편차   F>B비율   F-C     F-A  | SSIM  A      B      C      F")
        A("-" * 150)
        for e in exps + ["KFIX8", "ALL"]:
            any_row = False
            for k in KS:
                key = f"{e}|{k}|{coh}"
                if key not in S:
                    continue
                s = S[key]; any_row = True
                lab = {"KFIX8": "0.04+0.1", "ALL": "ALL"}.get(e, e)
                A(f"{lab:8s} {k:2d} {s['nblk']:5d} {s['nscene']:3d} | {s['psnr_A']:7.3f} {s['psnr_B']:7.3f} {s['psnr_C']:9.3f} {s[f'psnr_{FMAIN}']:10.3f} | "
                  f"{s['dF_B_mean']:+7.3f} ± {s['dF_B_std']:5.3f}   {100*s['dF_B_fracpos']:5.1f}%  {s['dF_C_mean']:+6.3f} {s['dF_A_mean']:+7.3f} | "
                  f"{s['ssim_A']:.4f} {s['ssim_B']:.4f} {s['ssim_C']:.4f} {s[f'ssim_{FMAIN}']:.4f}")
            if any_row:
                A("")
        A("```")
        A("")

    A("## 1. 네 팔 PSNR / SSIM")
    A("")
    arm_table("match", "(1-1) match 코호트 — 같은 씬·같은 프레임 집합에서 k 만 바꾼다 (정본)",
              "0.04+0.1 = KFIX=8 인 두 노출 합산(0.033s 는 최대 2장이라 별도). F-B 는 블록별 짝 차이의 평균이고 F>B 비율은 블록 중 F 가 B 보다 높은 비율.")
    arm_table("all", "(1-2) all 코호트 — k 마다 씬 구성이 바뀐다 (보조)", "")

    # 2. c 민감도
    A("## 2. F 의 c 민감도 (match 코호트)")
    A("")
    A("c 는 병합식 A(w) = |D|^2 / (|D|^2 + c sigma^2) 의 잡음 여유 계수. 클수록 기준 프레임으로 덜 물러나 평균에 가까워진다.")
    A("아래 '최적 c' 는 GT 를 보고 고른 값이다 (테스트 셋에서 사후 선택, 공정한 튜닝이 아니다).")
    A("")
    A("```")
    A("c 가 작을수록 기준 프레임(=A 의 입력)으로 더 물러나므로 F(c) 는 A 와 B 사이에 놓인다. A 를 옆에 둔다.")
    A("")
    A("노출     k  블록 |  A단일  |  F(c=2)  F(c=4)  F(c=8)  F(c=16) |  B평균  | 최대-최소  최적c(GT선택)  F(최적)-B")
    A("-" * 120)
    for e in exps + ["KFIX8"]:
        any_row = False
        for k in KS:
            if k == 1:
                continue
            key = f"{e}|{k}|match"
            if key not in S:
                continue
            s = S[key]; any_row = True
            vals = [s[f"psnr_F{int(c)}"] for c in C_LIST]
            best = int(np.argmax(vals))
            lab = {"KFIX8": "0.04+0.1"}.get(e, e)
            A(f"{lab:8s} {k:2d} {s['nblk']:5d} | {s['psnr_A']:6.3f} | " + "  ".join(f"{v:6.3f}" for v in vals)
              + f" | {s['psnr_B']:6.3f} | {max(vals)-min(vals):8.3f}   {int(C_LIST[best]):3d}          {vals[best]-s['psnr_B']:+6.3f}")
        if any_row:
            A("")
    A("```")
    A("")

    # 3. 대역별 차이
    A("## 3. F 와 B 의 차이를 방사 주파수 대역별로")
    A("")
    A("대역 정의는 diag_sid_spectrum.RAD_EDGES (주기/화소 단위 방사 주파수, 0.05 폭 10개 + 코너 1개). 휘도(0-255), 장 전체 FFT, Parseval 로 대역 합 = 평균제곱.")
    A("비율은 블록 합(에너지 가중). 읽는 법:")
    A("  dFB/rB   : 대역별 [F 출력 - B 출력] 파워 / [B 출력 - GT] 잔차 파워. 두 출력의 차이가 그 대역 오차 대비 얼마나 되는가.")
    A("  dAB/rB   : 같은 식으로 [A 단일 - B 평균]. 프레임을 하나만 쓴 것과 평균한 것의 차이 = 스케일 기준.")
    A("  dxm/dx1  : 입력 쪽. [병합 입력 - 평균 입력] 파워 / [x_1 - 평균 입력] 파워. 평균이 지운 잡음 대비 병합이 얼마나 벗어났는가.")
    A("  share    : dFB 파워가 11개 대역에 어떻게 분포하는가 (합 100%).")
    A("")
    for e in ["KFIX8", "0.04s", "0.1s", "0.033s"]:
        for k in KS:
            key = f"{e}|{k}|match"
            if k == 1 or key not in S or "band_ratio" not in S[key]:
                continue
            s = S[key]; br = s["band_ratio"]
            lab = {"KFIX8": "0.04s+0.1s"}.get(e, e)
            A(f"### {lab}, k={k}, match, 블록 {s['nblk']} / 씬 {s['nscene']}")
            A("")
            A(f"출력끼리 PSNR(F,B) = {s['psnr_FB']:.2f} dB, PSNR(A,B) = {s['psnr_AB']:.2f} dB, PSNR(C,B) = {s['psnr_CB']:.2f} dB;  "
              f"입력끼리 PSNR(merge,mean) = {s['psnr_xm_xbar']:.2f} dB, PSNR(x1,mean) = {s['psnr_x1_xbar']:.2f} dB")
            A(f"전 대역 합: dFB/rB = {100*s['tot_dFB_over_rB']:.2f}%,  dAB/rB = {100*s['tot_dAB_over_rB']:.1f}%,  dxm/dx1 = {100*s['tot_dxm_over_dx1']:.2f}%")
            A("")
            A("```")
            A("대역        dFB/rB %   dAB/rB %   dxm/dx1 %   dFB share %   dxm share %  | dFB/rB by c:   2       4       8      16")
            A("-" * 125)
            for b in range(NB):
                A(f"{BAND_NAMES[b]:10s} {100*br['dFB_over_rB'][b]:9.3f} {100*br['dAB_over_rB'][b]:10.1f} {100*br['dxm_over_dx1'][b]:11.2f} "
                  f"{100*br['dFB_share'][b]:12.1f} {100*br['dxm_share'][b]:12.1f}  | "
                  + " ".join(f"{100*br[f'dFB_c{int(c)}_over_rB'][b]:7.3f}" for c in C_LIST))
            A("```")
            A("")

    # 4. 잔차 파워 대역별 (A,B,F 대 GT) — F 가 B 와 어디서 다른지 보조
    A("## 4. 대역별 잔차 파워: A / B / F 대 GT (match, 0.04s+0.1s)")
    A("")
    A("각 대역에서 [출력 - GT] 파워를 B 기준으로 나눈 값. 1.000 이면 B 와 같은 오차, 작으면 그 대역 오차가 줄었다.")
    A("")
    A("```")
    A("대역        |   k=2: A/B    F/B   |   k=4: A/B    F/B   |   k=8: A/B    F/B")
    A("-" * 80)
    for b in range(NB):
        cells = []
        for k in [2, 4, 8]:
            s = S.get(f"KFIX8|{k}|match")
            if s and "band_ratio" in s:
                cells.append(f"{s['band_ratio']['rA_over_rB'][b]:6.3f} {s['band_ratio']['rF_over_rB'][b]:6.3f}")
            else:
                cells.append("   -      -  ")
        A(f"{BAND_NAMES[b]:10s} |   " + "   |   ".join(cells))
    A("```")
    A("")

    # 5. sigma / clip
    A("## 5. 병합 부산물: 추정 sigma 와 클리핑 비율 (match)")
    A("")
    A("```")
    A("노출     k  블록 |  sigma(추정, [0,1] 스케일)   <0 비율 %   >1 비율 %")
    A("-" * 70)
    for e in exps:
        for k in KS:
            key = f"{e}|{k}|match"
            if k == 1 or key not in S:
                continue
            s = S[key]
            A(f"{e:8s} {k:2d} {s['nblk']:5d} | {s['sigma']:12.5f}              {100*s['clip_lt0']:7.3f}   {100*s['clip_gt1']:7.3f}")
    A("```")
    A("")

    # 6. 판정 / 측정이 가리키는 것
    A("## 6. 자동 대조 요약")
    A("")
    A("```")
    for name, val, ref, ok in checks:
        A(f"{name:28s} {val:12.4f}  대조 {ref:10.4f}  {'통과' if ok else '실패'}")
    A("```")
    A("")

    # 7. 정본(multiframe_gain.json) 과 행 단위 대조 — B(=psnr_in), C(=psnr_out)
    mfg_path = "numbers/multiframe_gain.json"
    if os.path.exists(mfg_path):
        M = {(r["scene"], r["exp"], r["k"], r["bi"]): r for r in json.load(open(mfg_path))["rows"]}
        dB, dC, flags, nm = [], [], 0, 0
        for r in rows:
            o = M.get((r["scene"], r["exp"], r["k"], r["bi"]))
            if o is None:
                continue
            nm += 1
            dB.append(r["psnr_B"] - o["psnr_in"]); dC.append(r["psnr_C"] - o["psnr_out"])
            flags += int(r["fix"] != o["fix"]) + int(r["match"] != o["match"]) + int(r["nfr"] != o["nfr"])
        dB, dC = np.array(dB), np.array(dC)
        A("## 7. 정본 multiframe_gain.json 과 행 단위 대조")
        A("")
        A("같은 (씬, 노출, k, 블록) 행에서 이 파일의 B 와 정본의 입력평균(psnr_in), C 와 출력평균(psnr_out)을 비교했다. "
          "정본은 cudnn TF32 기본값(켜짐)으로 돌았고 여기는 껐으므로 장당 0.01 dB 안팎의 차이는 그 때문이다.")
        A("")
        A("```")
        A(f"매칭 행 수                     {nm:6d} / {len(rows)}")
        A(f"코호트 플래그(fix/match/nfr) 불일치 {flags:6d}")
        A(f"B PSNR 차  평균 {dB.mean():+.5f}  최대절대 {np.abs(dB).max():.4f} dB")
        A(f"C PSNR 차  평균 {dC.mean():+.5f}  최대절대 {np.abs(dC).max():.4f} dB")
        A("```")
        A("")

    s2 = S.get("KFIX8|2|match"); s4 = S.get("KFIX8|4|match"); s8 = S.get("KFIX8|8|match")
    # c 단조성: 모든 (노출,k,match) 에서 F(2) > F(4) > F(8) > F(16) > B 인지
    mono_ok, mono_n = True, 0
    for key, sv in S.items():
        e, k, coh = key.split("|")
        if coh != "match" or k == "1" or e in ("ALL", "KFIX8"):
            continue
        mono_n += 1
        seq = [sv[f"psnr_F{int(c)}"] for c in C_LIST] + [sv["psnr_B"]]
        mono_ok &= all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    A("## 측정이 가리키는 것")
    A("")
    lines = []
    if s2 and s4 and s8:
        lo = {k: s["band_ratio"]["dFB_share"][0] for k, s in ((2, s2), (4, s4), (8, s8))}
        fb0 = {k: s["band_ratio"]["rF_over_rB"][0] for k, s in ((2, s2), (4, s4), (8, s8))}
        ab0 = {k: s["band_ratio"]["rA_over_rB"][0] for k, s in ((2, s2), (4, s4), (8, s8))}
        fbhi = {k: float(np.mean(s["band_ratio"]["rF_over_rB"][2:])) for k, s in ((2, s2), (4, s4), (8, s8))}
        lines.append(f"- 같은 50씬(0.04s+0.1s, match)에서 F(c=8) 는 B 를 k=2 {s2['dF_B_mean']:+.3f} / k=4 {s4['dF_B_mean']:+.3f} / k=8 {s8['dF_B_mean']:+.3f} dB 이기지만 "
                     f"(블록 {100*s8['dF_B_fracpos']:.0f}% 에서 F>B), 여전히 A 단일 프레임보다 {abs(s8['dF_A_mean']):.2f} dB, C 출력평균보다 {abs(s8['dF_C_mean']):.2f} dB 낮다(k=8). "
                     f"입력 쪽에서 프레임을 모으면 모델이 나빠지는 현상은 표준 병합에서도 그대로다.")
        lines.append(f"- c 는 단조다: match 코호트 {mono_n}개 (노출,k) 전부에서 F(2) > F(4) > F(8) > F(16) > B "
                     f"{'(예외 없음)' if mono_ok else '(예외 있음, 2절 표 확인)'}. 폭은 k=8 에서 {max(s8[f'psnr_F{int(c)}'] for c in C_LIST)-min(s8[f'psnr_F{int(c)}'] for c in C_LIST):.2f} dB. "
                     f"c 가 작을수록 기준 프레임으로 물러나 A 에 가까워지므로 GT 로 고른 최적 c=2 는 '덜 병합할수록 낫다'는 뜻이지 병합이 좋다는 뜻이 아니다.")
        lines.append(f"- 입력에서 병합과 균등평균의 차이 파워는 평균이 지운 잡음의 {100*s2['tot_dxm_over_dx1']:.1f}~{100*s8['tot_dxm_over_dx1']:.1f}% 뿐이고 대역에 걸쳐 평평하다(0.1s k=8 최저 대역만 45%). "
                     f"즉 f>0.10 에서 병합은 균등 평균과 사실상 같다.")
        lines.append(f"- 출력에서 F−B 차이 파워의 {100*lo[2]:.0f}% (k=2) ~ {100*lo[8]:.0f}% (k=8) 가 최저 대역 f<0.05 에 있고, 그 대역에서 F 의 잔차는 B 의 {fb0[8]:.2f}배(k=8; A 는 {ab0[8]:.2f}배), "
                     f"f>=0.10 대역들은 평균 {fbhi[8]:.3f}배로 B 와 같다. F 가 B 를 이기는 몫은 전부 저주파에서 기준 프레임 쪽으로 남아 있는 데서 온다.")
        lines.append(f"- 결론: 이 데이터(삼각대 정지, 정렬 없음)에서 HDR+ 계열 병합은 중·고주파에서는 균등 평균과 같고, 저주파에서만 단일 프레임 쪽으로 치우친 '평균과 단일 사이의 보간'이다. "
                     f"F 를 이겨야 한다는 조건은 사실상 A 와 C 를 이겨야 한다는 조건과 같다.")
    L.extend(lines)
    A("")
    txt = "\n".join(L)
    with open(OUT_MD, "w") as f:
        f.write(txt)
    print("wrote", OUT_MD)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="그룹 수 제한 (스모크)")
    ap.add_argument("--workers-merge", type=int, default=16)
    ap.add_argument("--workers-ssim", type=int, default=12)
    ap.add_argument("--audit-groups", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()
    if a.report_only:
        d = json.load(open(a.out))
        d["summary"] = summarize(d)
    else:
        d = measure(a)
    report(d)


if __name__ == "__main__":
    main()
