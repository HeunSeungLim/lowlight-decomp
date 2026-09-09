"""compare_methods.py

기존 방법들을 같은 벤치마크·같은 채점으로 나란히 놓는다 (정량 표 1개 + 정성 그림 2장).
학습 없음. 추론 + 채점 + 분해만. GPU 0, 추론만.

벤치마크
  LOL  : LOLv1 eval15 전량 15장 (400x600), GT = high/
  Sony : Retinexformer 배포 SID 처리본 test 598장 (512x960, 씬 50), GT = 씬의 long 1장
         (로더 규약은 diag_sid_failure.build_pairs/load_npy 를 import — repro 와 동일)

비교 대상 (전부 저자 배포 가중치 또는 무학습, 우리 파이프라인으로 추론)
  Input          : 저조도 입력 그대로 (기준선)
  Gamma 0.5      : x^0.5 (enhancers.arm_gamma 와 같은 감마, 8bit 만 LUT 버림 대신 반올림)
  CLAHE          : enhancers.arm_clahe (LAB L채널, clip 2.0, 8x8) — cv2 가 u8 로만 동작
  SCI            : 공식 medium.pt, enhancers.SCIArm 의 모델을 float 출력으로 사용
  AdaptiveRetinex: 자체 학습본(adaptive_retinex_trained.pth), enhancers.ARArm — ★자체 방법
  CIDNet         : HVI-CIDNet-LOLv1-woperc / -wperc, LOL 만. gated=True (repro_infer.py 규약)
                   캐시(repro/LOLv1_{woperc,wperc}/*.npy, float) 재사용
  Retinexformer  : LOL_v1.pth (LOL), SID.pth (Sony)
  CIDNet 의 Sony 가중치는 쓰지 않는다: CIDNet+ 의 Sony-Total-Dark 는 우리 데이터(Retinexformer
  SID 처리본)와 화소 규약이 다른 별개 변종이라 같은 데이터에서 PSNR 5.5 밖에 안 나온다
  (repro/SID_BENCHMARK_ID.md). 같은 표에 넣을 수 없다.

채점 (전부 같은 조건)
  8bit           : 반올림(rint) 이 주 값. 버림(floor) 은 CIDNet 앵커 대조용으로만 병기.
  PSNR/SSIM      : repro_measure.psnr_indep / ssim_indep (독립 구현). GT평균 미적용이 주 값,
                   GT평균 적용값(gt_mean_rectify)은 부 값으로 저장만.
  성분 분해      : diag_sid_failure.decompose (u8 반올림 출력 기준, 최소제곱, 클리핑 없음)
                   전역이득 / 채널이득 / 구조잔차 몫 = E가중 (장별 MSE 합 기준), 합 100.
  Δ16            : diag_sid_lowfreq.block_fit 의 gain 사다리 (float 출력, 오라클 전역이득 앵커)
                   psnrf_gain_16 - psnrf_gain_global - (white 대조군 같은 차). SEED 20260905.
  LF             : 전역이득 보정 후 f<0.10 이상적 저역통과 오차 에너지 몫 (RGB, E가중) — 참고열.

산출물
  repro/compare_methods.json        최상위 "rows" = 원고 표 생성기용
  repro/compare_methods_figframes.npz 그림용 프레임 출력 (재추론 없이 그림 재생성)
  repro/COMPARE_METHODS.md          정리
  paper/fig_cmp_lol.pdf, fig_cmp_sony.pdf, fig_cmp_numbers.json

사용: python compare_methods.py run   (측정 → json/npz)
      python compare_methods.py figs  (npz → pdf + numbers json)
      python compare_methods.py report(json → md)
      python compare_methods.py all
"""
import os, sys, json, math, time, re
import numpy as np

CODE = "./code"
sys.path.insert(0, CODE)
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify
from diag_sid_failure import decompose, p2, build_pairs as sid_pairs, load_npy as sid_load, \
    exposure_of
from diag_sid_spectrum import RAD_EDGES, LUMA_W, make_radial_index, u8
from diag_sid_lowfreq import BLOCKS, BLOCK_NAMES, LP_NBAND, block_index, block_fit
import diag_lolv1 as DL

REPRO = "numbers"
PAPER = "paper_tables"
OUT_JSON = os.path.join(REPRO, "compare_methods.json")
OUT_NPZ = os.path.join(REPRO, "compare_methods_figframes.npz")
OUT_MD = os.path.join(REPRO, "COMPARE_METHODS.md")
FIG_NUM = os.path.join(PAPER, "fig_cmp_numbers.json")
RF_SID_W = (os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth")
CID_CACHE = {"cidnet_woperc": os.path.join(REPRO, "LOLv1_woperc"),
             "cidnet_wperc": os.path.join(REPRO, "LOLv1_wperc")}
SEED = 20260905
SONY_FIG_SCENE, SONY_FIG_EXP = "10198", "0.033s"      # fig_qual 과 같은 실패 사례
LADDER_BLOCKS = BLOCKS                                # gain 사다리 전 칸 (표는 16 만 씀)

NAME = {"input": "Input", "gamma": "Gamma 0.5", "clahe": "CLAHE", "sci": "SCI",
        "ar": "AdaptiveRetinex (ours)", "cidnet_woperc": "CIDNet",
        "cidnet_wperc": "CIDNet (perc.)", "retinexformer": "Retinexformer",
        "zerodcepp": "Zero-DCE++", "uretinex": "URetinex-Net", "snrnet": "SNR-Net",
        "llformer": "LLFormer", "gsad": "GSAD", "lightendiff": "LightenDiffusion"}
# 추가 베이스라인: 저자 공개 가중치로 newbase_*.py 가 만든 출력 캐시 (float32 CHW RGB [0,1])
NEW_CACHE = {k: os.path.join(REPRO, f"cache_{k}") for k in
             ("zerodcepp", "uretinex", "snrnet", "llformer", "gsad", "lightendiff")}
def _has_cache(key, bench):
    d = os.path.join(NEW_CACHE.get(key, "/nonexistent"), bench)
    return os.path.isdir(d) and len(os.listdir(d)) > 0
LEARNED = {"sci": "SCI medium.pt (공식)", "ar": "자체 학습본",
           "zerodcepp": "Zero-DCE++ Epoch99.pth (저자, 무참조)", "uretinex": "URetinex-Net ckpt (저자, LOL)",
           "snrnet": "SNR-Net LOL/SID .pth (저자)", "llformer": "LLFormer LOL-v1 (저자)",
           "gsad": "GSAD LOL-v1 (저자)", "lightendiff": "LightenDiffusion (저자, 비지도)",
           "cidnet_woperc": "HVI-CIDNet-LOLv1-woperc (저자)",
           "cidnet_wperc": "HVI-CIDNet-LOLv1-wperc (저자)",
           "retinexformer": "LOL_v1.pth / SID.pth (저자)"}
METHODS = {"LOL": ["input", "gamma", "clahe", "zerodcepp", "sci", "ar", "uretinex", "snrnet",
                   "llformer", "gsad", "lightendiff", "cidnet_woperc", "cidnet_wperc", "retinexformer"],
           "Sony": ["input", "gamma", "clahe", "zerodcepp", "sci", "ar", "lightendiff", "snrnet",
                    "retinexformer"]}
# 캐시가 아직 없는 추가 베이스라인은 건너뛴다 (부분 실행 허용)
METHODS = {b: [k for k in ks if k not in NEW_CACHE or _has_cache(k, b)] for b, ks in METHODS.items()}
FIG_METHODS = {"LOL": [k for k in ["input", "llformer", "gsad", "cidnet_woperc", "retinexformer"] if k in METHODS["LOL"]],      # + GT
               "Sony": [k for k in ["input", "zerodcepp", "snrnet", "lightendiff", "retinexformer"] if k in METHODS["Sony"]]}
ANCHORS = [  # (bench, key, field, value, source)
    ("Sony", "retinexformer", "psnr", 24.438, "measure_SID_retinexformer.json psnr_round"),
    ("LOL", "cidnet_woperc", "psnr_trunc", 23.498, "AUDIT_LOLv1 / DIAG_LOLV1 trunc"),
    ("LOL", "cidnet_woperc", "psnr", 23.609, "DIAG_LOLV1 round"),
    ("LOL", "retinexformer", "psnr", 25.152, "DIAG_LOLV1 / measure_LOLv1_retinexformer round"),
]


# ---------------- 데이터 ----------------
def lol_frames():
    for f, lq, gt in DL.build_pairs():
        yield dict(bench="LOL", id=f, scene=f, exp="", lq01=lq.astype(np.float32) / 255.0,
                   gt_u8=gt)


def sony_frames(limit=0):
    pairs = sid_pairs()
    if limit:
        pairs = pairs[:limit]
    gt_cache = {}
    for scene, lqp, gtp in pairs:
        if gtp not in gt_cache:
            gt_cache[gtp] = u8(sid_load(gtp))
        yield dict(bench="Sony", id=os.path.basename(lqp), scene=scene, exp=exposure_of(lqp),
                   lq01=sid_load(lqp), gt_u8=gt_cache[gtp], lq_path=lqp)


# ---------------- 방법 (전부 float64 3xHxW [0,1] 텐서를 돌려준다) ----------------
class Methods:
    def __init__(self, bench, torch):
        self.t = torch
        self.bench = bench
        self._sci = self._ar = self._rf = None

    def _rf_net(self):
        if self._rf is None:
            if self.bench == "LOL":
                self._rf = DL.make_retinexformer()
            else:
                import torch.nn.functional as F
                torch = self.t
                if DL.RF_REPO not in sys.path:
                    sys.path.insert(0, DL.RF_REPO)
                from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
                net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                                    num_blocks=[1, 2, 2])
                net.load_state_dict(torch.load(RF_SID_W, map_location="cpu")["params"])
                net = net.cuda().eval()

                def run(lq01):
                    x = torch.from_numpy(lq01.transpose(2, 0, 1))[None].cuda()
                    h, w = x.shape[2], x.shape[3]
                    ph, pw = (-h) % 4, (-w) % 4
                    if ph or pw:
                        x = F.pad(x, (0, pw, 0, ph), "reflect")
                    return torch.clamp(net(x)[:, :, :h, :w], 0, 1)[0].double()
                self._rf = run
        return self._rf

    def __call__(self, key, fr):
        torch = self.t
        x01 = fr["lq01"]
        if key == "input":
            return torch.from_numpy(x01.astype(np.float64)).cuda().permute(2, 0, 1)
        if key == "gamma":
            return torch.from_numpy(np.clip(x01, 0, 1).astype(np.float64) ** 0.5
                                    ).cuda().permute(2, 0, 1)
        if key == "clahe":
            from enhancers import arm_clahe
            bgr = u8(x01)[:, :, ::-1].copy()
            out = arm_clahe(bgr)[:, :, ::-1].astype(np.float64) / 255.0
            return torch.from_numpy(np.ascontiguousarray(out)).cuda().permute(2, 0, 1)
        if key == "sci":
            if self._sci is None:
                from enhancers import SCIArm
                self._sci = SCIArm().m
            x = torch.from_numpy(x01.transpose(2, 0, 1))[None].float().cuda()
            _, r = self._sci(x)
            return torch.clamp(r, 0, 1)[0].double()
        if key == "ar":
            if self._ar is None:
                from enhancers import ARArm
                self._ar = ARArm().m
            x = torch.from_numpy(x01.transpose(2, 0, 1))[None].float().cuda()
            return torch.clamp(self._ar(x)[0], 0, 1)[0].double()
        if key.startswith("cidnet"):
            p = os.path.join(CID_CACHE[key], fr["id"] + ".npy")
            if os.path.exists(p):
                y = np.load(p).astype(np.float64)                 # CHW float32 [0,1] (repro_infer)
                return torch.from_numpy(np.clip(y, 0, 1)).cuda()
            run = DL.make_cidnet(DL.CID_W[key])
            return run(x01)
        if key == "retinexformer":
            return self._rf_net()(x01)
        if key in NEW_CACHE:
            p = os.path.join(NEW_CACHE[key], self.bench, fr["id"] + ".npy")
            y = np.load(p).astype(np.float64)                     # CHW float32 RGB [0,1] (newbase_*.py)
            assert y.shape == (3,) + x01.shape[:2], (key, fr["id"], y.shape)
            return torch.from_numpy(np.clip(y, 0, 1)).cuda()
        raise ValueError(key)


# ---------------- 측정 ----------------
def measure_bench(bench, torch, limit=0, keep=None):
    """keep(fr) -> True 면 그 프레임의 모든 방법 출력을 float16 으로 보관 (그림용)."""
    import torch.nn.functional as F
    dev = "cuda"
    frames = list(lol_frames()) if bench == "LOL" else list(sony_frames(limit))
    H, W = frames[0]["gt_u8"].shape[:2]
    N = H * W
    ridx, _ = make_radial_index(H, W, RAD_EDGES, dev, torch)
    lp_mask = (ridx.reshape(-1) < LP_NBAND).reshape(H, W)
    bidx = {B: block_index(H, W, B, torch, dev) for B in LADDER_BLOCKS}
    M = Methods(bench, torch)
    rows = {k: [] for k in METHODS[bench]}
    kept = {}
    t0 = time.time()
    print(f"=== {bench}: {len(frames)} frames {H}x{W}", flush=True)
    for key in METHODS[bench]:
        gen = torch.Generator(device=dev); gen.manual_seed(SEED)
        tk = time.time()
        with torch.inference_mode():
            for i, fr in enumerate(frames):
                gt_u8 = fr["gt_u8"]
                g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                y = M(key, fr)                                          # 3xHxW float64 [0,1]
                y_np = y.permute(1, 2, 0).cpu().numpy()
                pred_u8 = u8(y_np)
                pred_tr = np.clip(np.floor(y_np * 255.0), 0, 255).astype(np.uint8)
                r = dict(id=fr["id"], scene=fr["scene"], exp=fr["exp"])
                # --- 채점 (반올림 주 값, 버림·GT평균 부 값) ---
                r["psnr"] = psnr_indep(pred_u8, gt_u8)
                r["ssim"] = ssim_indep(pred_u8, gt_u8)
                r["psnr_trunc"] = psnr_indep(pred_tr, gt_u8)
                rec = gt_mean_rectify(pred_u8, gt_u8)
                r["psnr_gtmean"] = psnr_indep(rec, gt_u8)
                r["ssim_gtmean"] = ssim_indep(np.rint(rec).astype(np.uint8), gt_u8)
                # --- 성분 분해 (u8 반올림 출력 기준, diag_sid_failure 정의) ---
                r.update(decompose(pred_u8.astype(np.float64) / 255.0,
                                   gt_u8.astype(np.float64) / 255.0))
                # --- float 출력 + 오라클 전역이득 ---
                a = float((y * g).sum() / (y * y).sum())
                p = a * y
                r["gain_global_float"] = a
                r["psnr_gainfix"] = psnr_indep(u8(np.clip(p.permute(1, 2, 0).cpu().numpy(), 0, 1)),
                                               gt_u8)
                # --- LF 오차 몫 (RGB) ---
                E = (p - g) * 255.0
                D3 = torch.fft.ifft2(torch.fft.fft2(E) * lp_mask[None]).real
                r["lf_d_en_rgb"] = float((D3 ** 2).mean()); r["lf_e_en_rgb"] = float((E ** 2).mean())
                # --- gain 사다리 (실측 + white 자유도 대조군, lowfreq/diag_lolv1 와 같은 절차) ---
                e_true = p - g
                sd = e_true.std(dim=(1, 2), unbiased=False, keepdim=True)
                wn = torch.randn(3, H, W, device=dev, dtype=torch.float64, generator=gen)
                wn = wn / torch.clamp(wn.std(dim=(1, 2), unbiased=False, keepdim=True),
                                      min=1e-30) * sd
                tgt_w = p - wn
                for B, nm in zip(LADDER_BLOCKS, BLOCK_NAMES):
                    idx, nblk, _, _ = bidx[B]
                    out, _, _ = block_fit(y, g, idx, nblk, "gain", torch)
                    r[f"mse_gain_{nm}"] = float(((out - g) ** 2).mean()) * 255.0 ** 2
                    out, _, _ = block_fit(y, tgt_w, idx, nblk, "gain", torch)
                    r[f"mse_white_gain_{nm}"] = float(((out - tgt_w) ** 2).mean()) * 255.0 ** 2
                rows[key].append(r)
                if keep is not None and keep(fr):
                    kept.setdefault(fr["id"], {})[key] = y.to(torch.float16).cpu().numpy()
                    kept[fr["id"]]["_gt"] = gt_u8
                    kept[fr["id"]]["_lq01"] = fr["lq01"]
                    kept[fr["id"]]["_meta"] = dict(scene=fr["scene"], exp=fr["exp"])
                if (i + 1) % 100 == 0:
                    print(f"  {key} {i+1}/{len(frames)} {time.time()-tk:.0f}s", flush=True)
        print(f"  {key} done {time.time()-tk:.0f}s  psnr {np.mean([r['psnr'] for r in rows[key]]):.4f}",
              flush=True)
        torch.cuda.empty_cache()
    print(f"=== {bench} total {time.time()-t0:.0f}s", flush=True)
    return rows, kept, dict(H=H, W=W, n=len(frames))


def aggregate(bench, key, rs):
    n = len(rs)

    def m(k):
        return float(np.mean([r[k] for r in rs]))

    def s(k):
        return float(np.std([r[k] for r in rs], ddof=1)) if n > 1 else 0.0

    tot = sum(r["mse"] for r in rs)
    o = dict(bench=bench, key=key, method=NAME[key], n=n,
             psnr=m("psnr"), psnr_sd=s("psnr"), ssim=m("ssim"), ssim_sd=s("ssim"),
             psnr_trunc=m("psnr_trunc"), psnr_gtmean=m("psnr_gtmean"), ssim_gtmean=m("ssim_gtmean"),
             psnr_gainfix=m("psnr_gainfix"),
             glob=100.0 * sum(r["mse"] - r["mse_glob"] for r in rs) / tot,
             chan=100.0 * sum(r["mse_glob"] - r["mse_chan"] for r in rs) / tot,
             resid=100.0 * sum(r["mse_chan"] for r in rs) / tot,
             glob_perimg=float(np.mean([100 * (r["mse"] - r["mse_glob"]) / r["mse"] for r in rs])),
             resid_perimg=float(np.mean([100 * r["mse_chan"] / r["mse"] for r in rs])),
             gain_global=m("gain_global"), gain_global_sd=s("gain_global"),
             lf=100.0 * sum(r["lf_d_en_rgb"] for r in rs) / sum(r["lf_e_en_rgb"] for r in rs))
    pf = lambda k: float(np.mean([10 * np.log10(255.0 ** 2 / r[k]) for r in rs]))
    o["psnrf_gain_global"] = pf("mse_gain_global(1x1)")
    o["psnrf_gain_16"] = pf("mse_gain_16")
    o["white16"] = pf("mse_white_gain_16") - pf("mse_white_gain_global(1x1)")
    o["d16_raw"] = o["psnrf_gain_16"] - o["psnrf_gain_global"]
    o["d16"] = o["d16_raw"] - o["white16"]
    o["ladder"] = {nm: pf(f"mse_gain_{nm}") - o["psnrf_gain_global"] for nm in BLOCK_NAMES}
    if bench == "Sony":
        o["by_exp"] = {}
        for e in sorted(set(r["exp"] for r in rs), key=lambda t: float(t[:-1])):
            q = [r for r in rs if r["exp"] == e]
            o["by_exp"][e] = dict(n=len(q), psnr=float(np.mean([r["psnr"] for r in q])),
                                  ssim=float(np.mean([r["ssim"] for r in q])))
    return o


def pick_lol_frame(rows):
    """LOL frame where methods differ most: the largest PSNR range (max minus min) over the three learned methods SCI, CIDNet (w/o perceptual loss) and Retinexformer. The displayed set (FIG_METHODS) is chosen separately."""
    ids = [r["id"] for r in rows["sci"]]
    cand = []
    for i, f in enumerate(ids):
        v = {k: rows[k][i]["psnr"] for k in ["sci", "cidnet_woperc", "retinexformer"]}
        cand.append((max(v.values()) - min(v.values()), f, v))
    cand.sort(reverse=True)
    return cand


def run(limit=0):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False          # diag_lolv1 와 동일
    out = dict(seed=SEED, rows=[], agg={}, per_frame={}, meta={}, anchors=[], checks=[],
               tf32=dict(matmul=False, cudnn=bool(torch.backends.cudnn.allow_tf32)),
               methods=METHODS, learned_weights=LEARNED,
               cidnet_sony_excluded="CIDNet+ Sony-Total-Dark 는 화소 규약이 다른 별개 변종 "
                                    "(같은 데이터에서 PSNR 5.5; repro/SID_BENCHMARK_ID.md)")
    kept_all = {}
    # ---- LOL: 모든 프레임 보관 (프레임 선택은 측정 후) ----
    rows, kept, meta = measure_bench("LOL", torch, keep=lambda fr: True)
    out["meta"]["LOL"] = meta
    for k in METHODS["LOL"]:
        out["agg"][f"LOL/{k}"] = aggregate("LOL", k, rows[k])
    out["per_frame"]["LOL"] = rows
    cand = pick_lol_frame(rows)
    ci = [r["id"] for r in rows["sci"]].index(cand[0][1])
    out["fig_lol"] = dict(criterion="LOL frame with the largest PSNR range (max minus min) over the three learned methods SCI, CIDNet (w/o perceptual loss) and Retinexformer",
                          chosen=cand[0][1], range_db=cand[0][0], psnr_learned=cand[0][2],
                          psnr={k: rows[k][ci]["psnr"] for k in METHODS["LOL"]},
                          ranking=[dict(id=c[1], range_db=c[0]) for c in cand[:5]])
    kept_all["LOL/" + cand[0][1]] = kept[cand[0][1]]
    del kept
    # ---- Sony ----
    rows, kept, meta = measure_bench(
        "Sony", torch, limit=limit,
        keep=lambda fr: fr["scene"] == SONY_FIG_SCENE and fr["exp"] == SONY_FIG_EXP)
    out["meta"]["Sony"] = meta
    for k in METHODS["Sony"]:
        out["agg"][f"Sony/{k}"] = aggregate("Sony", k, rows[k])
    out["per_frame"]["Sony"] = rows
    if kept:
        fid = sorted(kept)[0]                    # fig_qual 규칙: 정렬 첫 0.033s 파일
        rf = rows["retinexformer"]
        worst = min((r for r in rf if r["exp"] == SONY_FIG_EXP), key=lambda r: r["psnr"])
        out["fig_sony"] = dict(chosen=fid, scene=SONY_FIG_SCENE, exp=SONY_FIG_EXP,
                               criterion="Sony frame 10198 at 0.033 s, the frame on which Retinexformer scores lowest (same failure case as fig_qual)",
                               rf_worst_0033=dict(id=worst["id"], psnr=worst["psnr"]),
                               is_rf_worst_0033=(worst["id"] == fid),
                               psnr={k: next(r["psnr"] for r in rows[k] if r["id"] == fid)
                                     for k in METHODS["Sony"]})
        kept_all["Sony/" + fid] = kept[fid]
    build_checks(out)
    # ---- 표 생성기용 rows ----
    for bench in ["LOL", "Sony"]:
        for k in METHODS[bench]:
            a = out["agg"][f"{bench}/{k}"]
            out["rows"].append({f: a[f] for f in ["bench", "method", "key", "psnr", "psnr_sd",
                                                    "ssim", "ssim_sd", "glob", "chan", "resid",
                                                    "d16", "n", "lf", "psnr_trunc", "psnr_gtmean"]})
    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1)
    np.savez_compressed(OUT_NPZ, **{f"{fid}|{k}": v for fid, dd in kept_all.items()
                                     for k, v in dd.items() if not k.startswith("_meta")},
                        **{f"{fid}|_meta": json.dumps(dd["_meta"]) for fid, dd in kept_all.items()})
    print("wrote", OUT_JSON, OUT_NPZ)
    for a in out["anchors"]:
        print(f"anchor {a['bench']}/{a['key']} {a['field']}: {a['got']:.4f} vs {a['expect']} "
              f"{'통과' if a['ok'] else '실패'}")


def build_checks(out):
    """agg 만으로 앵커·산수 대조를 다시 만든다 (재측정 없이 recheck 가능)."""
    out["anchors"], out["checks"] = [], []
    for bench, key, field, val, src in ANCHORS:
        a = out["agg"].get(f"{bench}/{key}")
        if a is None:
            continue
        got = a[field]
        out["anchors"].append(dict(bench=bench, key=key, field=field, expect=val, got=got,
                                   ok=bool(abs(got - val) < 5e-3), source=src))
    for name, a in out["agg"].items():
        ssum = a["glob"] + a["chan"] + a["resid"]
        out["checks"].append(dict(item=f"{name} 성분 합", value=ssum, ok=bool(abs(ssum - 100) < 1e-6)))
        out["checks"].append(dict(item=f"{name} white16 자유도", value=a["white16"],
                                  ok=bool(abs(a["white16"]) < 0.05)))
        out["checks"].append(dict(item=f"{name} 반올림-버림 (dB)", value=a["psnr"] - a["psnr_trunc"],
                                  ok=bool(abs(a["psnr"] - a["psnr_trunc"]) < 0.2)))   # 부호는 출력 편향에 따라 갈린다
    # 캐시 진단값과의 교차 대조 (같은 정의로 다시 잰 값이 같아야 한다)
    try:
        dl = json.load(open(os.path.join(REPRO, "diag_lolv1.json")))["models"]
        for k in ["cidnet_woperc", "cidnet_wperc", "retinexformer"]:
            a = out["agg"][f"LOL/{k}"]; m = dl[k]
            d16_ref = (m["psnrf_gain_16"] - m["psnrf_gain_global(1x1)"]) - \
                      (m["psnrf_white_gain_16"] - m["psnrf_white_gain_global(1x1)"])
            for fld, ref in [("psnr", m["psnr_model"]), ("ssim", m["ssim_model"]),
                             ("glob", m["share_global_pct"]), ("resid", m["share_resid_pct"]),
                             ("d16", d16_ref), ("lf", 100 * m["lf_energy_share_rgb_ew"])]:
                out["checks"].append(dict(item=f"LOL/{k} {fld} vs diag_lolv1", value=a[fld],
                                          ref=ref, ok=bool(abs(a[fld] - ref) < 0.02)))
        sf = json.load(open(os.path.join(REPRO, "diag_sid_failure.json")))["overall"]
        lf = json.load(open(os.path.join(REPRO, "diag_sid_lowfreq.json")))["overall"]
        a = out["agg"].get("Sony/retinexformer")
        if a and a["n"] == 598:
            d16_ref = (lf["psnrf_gain_16"] - lf["psnrf_gain_global(1x1)"]) - \
                      (lf["psnrf_white_gain_16"] - lf["psnrf_white_gain_global(1x1)"])
            for fld, ref in [("psnr", sf["psnr_model"]), ("ssim", sf["ssim_model"]),
                             ("glob", sf["share_global_pct"]), ("chan", sf["share_chan_pct"]),
                             ("resid", sf["share_resid_pct"]), ("d16", d16_ref),
                             ("lf", 100 * lf["lf_energy_share_rgb_ew"])]:
                out["checks"].append(dict(item=f"Sony/retinexformer {fld} vs diag_sid_*", value=a[fld],
                                          ref=ref, ok=bool(abs(a[fld] - ref) < 0.02)))
    except Exception as e:                       # 대조 실패는 기록만
        out["checks"].append(dict(item="cache cross-check", error=str(e), ok=False))
    return out


# ---------------- 그림 ----------------
def lf_residual(y01, gt_u8):
    """전역이득(오라클) 보정 후 휘도 잔차의 f<0.10 이상적 저역통과 (0-255 단위). numpy."""
    y = y01.astype(np.float64); g = gt_u8.astype(np.float64) / 255.0
    a = float((y * g).sum() / (y * y).sum())
    e = (a * y - g) * 255.0
    el = (e * LUMA_W[None, None, :]).sum(2)
    H, W = el.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    rr = np.sqrt(fy ** 2 + fx ** 2)
    mask = rr < RAD_EDGES[LP_NBAND]
    D = np.real(np.fft.ifft2(np.fft.fft2(el) * mask))
    return D, a


def figs():
    import matplotlib; matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    Z = np.load(OUT_NPZ)
    J = json.load(open(OUT_JSON))
    numbers = {}
    for bench, fig_key, pdf in [("LOL", "fig_lol", "fig_cmp_lol.pdf"),
                                ("Sony", "fig_sony", "fig_cmp_sony.pdf")]:
        if fig_key not in J:
            print("skip", pdf, "(no frame kept)"); continue
        info = J[fig_key]; fid = info["chosen"]
        pre = f"{bench}/{fid}|"
        gt = Z[pre + "_gt"]; lq = Z[pre + "_lq01"]
        H, W = gt.shape[:2]
        panels, resid, rec = [], [], {}
        for k in FIG_METHODS[bench]:
            y = Z[pre + k].astype(np.float64).transpose(1, 2, 0)       # HxWx3 [0,1] (float16 저장)
            y_u8 = u8(y)
            psnr = psnr_indep(y_u8, gt)
            D, a = lf_residual(y, gt)
            if k == "input":
                show = np.clip(a * y, 0, 1)
                title = f"Input ($\\times${a:.1f})\n{psnr:.1f} dB"
            else:
                show = np.clip(y, 0, 1)
                nm = dict(NAME)[k]
                title = f"{nm}\n{psnr:.1f} dB"
            panels.append((show, title)); resid.append(D)
            rec[k] = dict(psnr=float(psnr), psnr_shown=round(float(psnr), 1), gain=float(a),
                          lf_rms=float(np.sqrt((D ** 2).mean())),
                          psnr_json=float(info["psnr"][k]))
        panels.append((gt.astype(np.float64) / 255.0, "Reference"))
        V = float(np.percentile(np.abs(np.concatenate([d.ravel() for d in resid])), 99.0))
        ncol = len(panels)
        pw = 7.008 / ncol
        ph = pw * H / W
        fig = plt.figure(figsize=(7.008, 2 * ph + 0.46), dpi=300)
        gs = gridspec.GridSpec(2, ncol, figure=fig, wspace=0.03, hspace=0.04,
                               left=0.003, right=0.997, top=1 - 0.40 / (2 * ph + 0.46), bottom=0.005)
        for c, (im, ti) in enumerate(panels):
            ax = fig.add_subplot(gs[0, c]); ax.imshow(im, interpolation="nearest"); ax.axis("off")
            ax.set_title(ti, fontsize=9.2, pad=2)
        for c, D in enumerate(resid):
            ax = fig.add_subplot(gs[1, c])
            h = ax.imshow(D, cmap="coolwarm", vmin=-V, vmax=V, interpolation="nearest"); ax.axis("off")
        axc = fig.add_subplot(gs[1, ncol - 1]); axc.axis("off")
        cax = axc.inset_axes([0.42, 0.15, 0.08, 0.7])
        cb = fig.colorbar(h, cax=cax); cb.ax.tick_params(labelsize=9.2, length=1.5, pad=1)
        cb.set_label("LF residual", fontsize=9.2)
        fig.savefig(os.path.join(PAPER, pdf))
        plt.close(fig)
        numbers[bench] = dict(frame=fid, scene=json.loads(str(Z[pre + "_meta"])),
                              criterion=info["criterion"], displayed_set=FIG_METHODS[bench], selection_note="the frame is selected by the criterion above; the displayed methods are FIG_METHODS and differ from the selection set", methods=rec, color_range=V,
                              color_range_rule="pooled |D| 99 percentile, symmetric",
                              lowpass="f<0.10 (RAD_EDGES[LP_NBAND]), luma 0.299/0.587/0.114",
                              size=[H, W], figsize_in=[float(x) for x in fig.get_size_inches()])
        if bench == "LOL":
            numbers[bench]["range_db"] = info["range_db"]; numbers[bench]["ranking"] = info["ranking"]
        else:
            numbers[bench]["rf_worst_0033"] = info["rf_worst_0033"]
            numbers[bench]["is_rf_worst_0033"] = info["is_rf_worst_0033"]
        print(pdf, fid, {k: v["psnr_shown"] for k, v in rec.items()}, "V", round(V, 2))
    json.dump(numbers, open(FIG_NUM, "w"), indent=2, ensure_ascii=False)
    print("wrote", FIG_NUM)


# ---------------- 보고서 ----------------
def report():
    J = json.load(open(OUT_JSON))
    A = J["agg"]
    L = []; P = L.append
    P("# 기존 방법 나란히 비교 — 같은 벤치마크·같은 채점 (260905)")
    P("")
    P("코드 code/compare_methods.py, 수치 repro/compare_methods.json, 그림 paper/fig_cmp_{lol,sony}.pdf.")
    P("추론만(GPU 0). 채점은 repro_measure 독립 구현. 8bit 반올림, GT평균 미적용이 주 값.")
    P("성분 분해 = diag_sid_failure.decompose (E가중 몫, 합 100). Δ16 = gain 사다리 16화소 블록")
    P("오라클 이득에서 white 자유도 대조군을 뺀 값(diag_sid_lowfreq/diag_lolv1 와 같은 정의, seed 20260905).")
    P(f"TF32: matmul {J['tf32']['matmul']}, cudnn {J['tf32']['cudnn']}.")
    P("")
    P("## 0. 비교 대상")
    P("")
    P("```")
    P("방법                      학습    가중치                                  LOL  Sony")
    P("-----------------------------------------------------------------------------------")
    desc = {"input": ("무",   "-"), "gamma": ("무", "x^0.5 (enhancers.arm_gamma 감마, 8bit 반올림)"),
            "clahe": ("무", "cv2 CLAHE clip2.0 8x8, LAB L (enhancers.arm_clahe)"),
            "sci": ("유", J["learned_weights"]["sci"]), "ar": ("유", J["learned_weights"]["ar"] + " ★자체 방법"),
            "cidnet_woperc": ("유", J["learned_weights"]["cidnet_woperc"] + ", gated, 캐시 재사용"),
            "cidnet_wperc": ("유", J["learned_weights"]["cidnet_wperc"] + ", gated, 캐시 재사용"),
            "retinexformer": ("유", J["learned_weights"]["retinexformer"])}
    for k in NEW_CACHE:
        desc.setdefault(k, ("유" if k != "zerodcepp" else "무참조", J["learned_weights"].get(k, LEARNED.get(k, "")) + ", 캐시(newbase)"))
    for k in METHODS["LOL"]:
        P(f"{NAME[k]:25s} {desc[k][0]:6s} {desc[k][1]:40s} {'o' if k in METHODS['LOL'] else '-':>3s}  "
          f"{'o' if k in METHODS['Sony'] else '-':>3s}")
    P("```")
    P("CIDNet Sony 제외 사유: " + J["cidnet_sony_excluded"])
    P("")
    P("## 1. 정량 비교표 (주 값: 8bit 반올림, GT평균 미적용)")
    P("")
    for bench in ["LOL", "Sony"]:
        n = J["meta"][bench]["n"]
        P(f"### {bench}  (n={n}장, {J['meta'][bench]['H']}x{J['meta'][bench]['W']}; sd 는 장별 표준편차 ddof=1)")
        P("")
        P("```")
        P(f"{'방법':25s} {'PSNR':>7s} {'sd':>6s} {'SSIM':>7s} {'sd':>6s} | {'전역%':>6s} {'채널%':>6s} {'구조%':>6s} | "
          f"{'Δ16 dB':>7s} {'white':>6s} | {'LF%':>5s} | {'버림':>7s} {'GT평균':>7s}")
        P("-" * 128)
        order = sorted(METHODS[bench], key=lambda k: -A[f"{bench}/{k}"]["psnr"])
        for k in order:
            a = A[f"{bench}/{k}"]
            P(f"{a['method']:25s} {a['psnr']:7.3f} {a['psnr_sd']:6.3f} {a['ssim']:7.4f} {a['ssim_sd']:6.4f} | "
              f"{a['glob']:6.1f} {a['chan']:6.1f} {a['resid']:6.1f} | {a['d16']:7.3f} {a['white16']:6.3f} | "
              f"{a['lf']:5.1f} | {a['psnr_trunc']:7.3f} {a['psnr_gtmean']:7.3f}")
        P("```")
        P("(PSNR 내림차순. 전역/채널/구조 = 오차 성분 몫(E가중). Δ16 = 국소이득 오라클(16px) 자유도 보정 후 이득,")
        P(" white = 뺀 자유도 대조군 값. LF = f<0.10 오차 에너지 몫(RGB, E가중). 버림/GT평균 은 참고열.)")
        P("")
        if bench == "Sony":
            P("노출시간별 PSNR (참고):")
            P("```")
            exps = list(A["Sony/retinexformer"]["by_exp"].keys())
            P(f"{'방법':25s} " + " ".join(f"{e+' (n='+str(A['Sony/retinexformer']['by_exp'][e]['n'])+')':>16s}" for e in exps))
            P("-" * 80)
            for k in order:
                a = A[f"Sony/{k}"]
                P(f"{a['method']:25s} " + " ".join(f"{a['by_exp'][e]['psnr']:16.3f}" for e in exps))
            P("```")
            P("")
    P("## 2. 앵커 대조 (3개 + 캐시 진단값 교차)")
    P("")
    P("```")
    P(f"{'항목':45s} {'기대':>10s} {'실측':>10s}  판정")
    P("-" * 75)
    for a in J["anchors"]:
        P(f"{a['bench']+'/'+a['key']+' '+a['field']:45s} {a['expect']:10.3f} {a['got']:10.4f}  {'통과' if a['ok'] else '실패'}")
    P("```")
    P("")
    P("산수 대조:")
    P("```")
    P(f"{'항목':50s} {'값':>10s} {'기준':>10s}  판정")
    P("-" * 80)
    for c in J["checks"]:
        if "error" in c:
            P(f"{c['item']:50s} ERROR {c['error']}")
            continue
        ref = f"{c['ref']:10.4f}" if "ref" in c else "          "
        P(f"{c['item']:50s} {c['value']:10.4f} {ref}  {'통과' if c['ok'] else '실패'}")
    P("```")
    nfail = sum(1 for c in J["checks"] if not c["ok"]) + sum(1 for a in J["anchors"] if not a["ok"])
    P(f"실패 {nfail}건 / 전체 {len(J['checks']) + len(J['anchors'])}건.")
    P("")
    P("## 3. 정성 그림")
    P("")
    fl = J["fig_lol"]
    P(f"- fig_cmp_lol.pdf: LOLv1 {fl['chosen']}. 선택 기준 = {fl['criterion']}, 범위 {fl['range_db']:.2f} dB "
      f"(SCI {fl['psnr']['sci']:.2f} / CIDNet {fl['psnr']['cidnet_woperc']:.2f} / Retinexformer {fl['psnr']['retinexformer']:.2f}).")
    P("  " + ", ".join(f"{NAME[k]} {v:.2f}" for k, v in fl["psnr"].items()))
    P("  차점: " + ", ".join(f"{c['id']} {c['range_db']:.2f}" for c in fl["ranking"][1:4]))
    if "fig_sony" in J:
        fs = J["fig_sony"]
        P(f"- fig_cmp_sony.pdf: Sony {fs['chosen']} ({fs['criterion']}). Retinexformer 0.033s 최악 장 = "
          f"{fs['rf_worst_0033']['id']} {fs['rf_worst_0033']['psnr']:.2f} dB, 이 프레임과 동일: {fs['is_rf_worst_0033']}.")
        P("  " + ", ".join(f"{NAME[k]} {v:.2f}" for k, v in fs["psnr"].items()))
    P("- 구성: 윗줄 입력(밝기정합 표시) | Gamma | SCI | CIDNet | Retinexformer | GT, 아랫줄 각 방법의 전역이득 보정 후")
    P("  (Sony 는 CIDNet 자리를 CLAHE 로 대체 — CIDNet Sony 가중치는 위 0절 사유로 같은 데이터에 못 쓴다)")
    P("  f<0.10 저역통과 휘도 잔차(coolwarm, 그림 안 공통 색범위 = 합동 |D| 99 백분위). 그림 안 수치는 paper/fig_cmp_numbers.json.")
    P("- 캔버스는 전폭(7.16in) 기준이고 높이는 6열 2행 실제 프레임 비율로 정해진다(LOL 3:2, Sony 15:8 프레임을")
    P("  자르지 않았다). 16:9 를 강제하면 여백만 생기므로 프레임 원비율을 유지했다.")
    P("")
    P("## 4. 측정이 가리키는 것")
    P("")
    SOTA = {"cidnet_woperc", "cidnet_wperc", "retinexformer"}
    lo = sorted(METHODS["LOL"], key=lambda k: -A[f"LOL/{k}"]["psnr"])
    so = sorted(METHODS["Sony"], key=lambda k: -A[f"Sony/{k}"]["psnr"])
    bl = next(k for k in lo if k not in SOTA); bs = next(k for k in so if k not in SOTA)
    g = lambda b, k, f: A[f"{b}/{k}"][f]
    P(f"- 순위: LOL {' > '.join(NAME[k] for k in lo)}; Sony {' > '.join(NAME[k] for k in so)}. "
      f"Retinexformer 가 SOTA 외 최선을 LOL 에서 {g('LOL','retinexformer','psnr')-g('LOL',bl,'psnr'):.1f} dB({NAME[bl]} {g('LOL',bl,'psnr'):.2f}), "
      f"Sony 에서 {g('Sony','retinexformer','psnr')-g('Sony',bs,'psnr'):.1f} dB({NAME[bs]} {g('Sony',bs,'psnr'):.2f}) 앞선다. "
      f"SCI·AdaptiveRetinex(자체)·CLAHE·Gamma 는 두 벤치마크 모두 15 dB 아래 — 극암 입력을 못 다룬다(실패로 기록). "
      f"LOL SSIM 최고는 CIDNet {g('LOL','cidnet_woperc','ssim'):.3f}(PSNR 은 RF 가 앞섬).")
    P(f"- 성분 몫은 방법·벤치마크마다 다르다: 전역이득 몫 LOL CIDNet {g('LOL','cidnet_woperc','glob'):.0f} / RF {g('LOL','retinexformer','glob'):.0f} / "
      f"SCI {g('LOL','sci','glob'):.0f} / Gamma {g('LOL','gamma','glob'):.0f} %, Sony RF {g('Sony','retinexformer','glob'):.0f} / Gamma {g('Sony','gamma','glob'):.0f} / "
      f"SCI {g('Sony','sci','glob'):.0f} %. 구조 잔차가 과반인 것은 Sony 전 방법({min(g('Sony',k,'resid') for k in METHODS['Sony']):.0f}~"
      f"{max(g('Sony',k,'resid') for k in METHODS['Sony']):.0f}%)이고 LOL 에선 RF({g('LOL','retinexformer','resid'):.0f}%)뿐이다.")
    P(f"- Δ16 은 모든 방법·두 벤치마크에서 white 대조군(0.006)보다 두 자릿수 크다: SOTA 가 LOL {min(g('LOL',k,'d16') for k in SOTA):.2f}~"
      f"{max(g('LOL',k,'d16') for k in SOTA):.2f}, Sony {g('Sony','retinexformer','d16'):.2f} dB 이고 무학습 출력도 같은 자릿수다"
      f"(Input {g('LOL','input','d16'):.2f}/{g('Sony','input','d16'):.2f}, CLAHE {g('LOL','clahe','d16'):.2f}/{g('Sony','clahe','d16'):.2f}, "
      f"Gamma {g('LOL','gamma','d16'):.2f}/{g('Sony','gamma','d16'):.2f}; LOL/Sony). 국소 이득 오라클 여지는 특정 모델의 결함이 아니라 "
      f"출력 전반에 남는 항이고, SOTA 도 그것을 닫지 못한다.")
    P("")
    P("## 5. 특이사항")
    P("")
    P(f"- Sony SCI 는 반올림({g('Sony','sci','psnr'):.3f}) 이 버림({g('Sony','sci','psnr_trunc'):.3f}) 보다 낮다(-0.008 dB). 출력이 GT 보다 밝은 쪽으로")
    P("  치우쳐 있어 1/255 깎는 버림이 유리한 것. 그래서 반올림-버림 판정은 부호 없이 |차| < 0.2 dB 로 둔다. CLAHE·Input 은 u8 원본이라 차 0.")
    P("- 기존 fig_qual.pdf 의 Sony 10198 0.033s 수치 15.1 dB 는 make_figure2.py 가 npy 를 채널 뒤집기 없이 넣어 나온 값이다")
    P("  (실측 재현: 같은 프레임·같은 SID.pth 로 무뒤집기 15.119 dB, 뒤집기+u8 반올림 17.659 dB — 차이 전부가 뒤집기).")
    P("  저자 로더 규약(BGR->RGB, repro/diag 공통 sid_load)으로는 같은 프레임이 17.66 dB(diag_sid_failure.json 장별 값과 일치)이고")
    P("  여전히 0.033s 최악 장이다. 이 그림·캡션은 sid_load 로 다시 만들어야 한다(이번 작업 범위 밖, 손대지 않음).")
    P("- Sony SCI/AdaptiveRetinex 는 Input 대비 +0.3/+1.4 dB 뿐이다. 0.033s 에서는 Gamma 10.7, CLAHE 10.8 dB 로 사실상 잡음이다(그림 참조).")
    with open(OUT_MD, "w") as fp:
        fp.write("\n".join(L) + "\n")
    print("wrote", OUT_MD)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    limit = int(os.environ.get("CMP_LIMIT", "0"))
    if cmd in ("run", "all"):
        run(limit)
    if cmd == "recheck":
        J = json.load(open(OUT_JSON)); build_checks(J)
        json.dump(J, open(OUT_JSON, "w"), indent=1); print("rechecked", OUT_JSON)
    if cmd in ("figs", "all"):
        figs()
    if cmd in ("report", "all"):
        report()
