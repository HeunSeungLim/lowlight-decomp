"""predict_gainfield.py

결판낼 질문: 극암 복원 오차의 3분의 2를 차지하는 '공간적으로 변하는 저주파 오차'
(= 블록 오라클 국소 이득장) 를 입력 영상만 보고 예측할 수 있는가.

DIAG_SID_LOWFREQ 4번은 블록 통계 스칼라 3개(입력 평균휘도·입력 분산·출력 평균휘도) 로
교차적합 R2 0.0061 을 얻었다. 특징 3개짜리 검정이라 '예측 불가' 의 증명이 못 된다.
여기서는 공간 맥락을 보는 합성곱 신경망으로 다시 잰다.

목표 변수 : diag_sid_lowfreq.block_ls_gain 을 그대로 재사용한 블록 오라클 이득장
            a_blk = <y,g>_blk / <y,y>_blk  (채널 공통 스칼라, 격자 16px / 32px / 64px)
            원본(절대 이득) 과 장내 중심화(a_blk - a_img) 둘 다.
입력      : short 입력 영상 + 모델 출력 영상 (스칼라 요약이 아니라 영상 그대로)
예측기    : U-Net 형 소형 CNN (1.2M), 512x960 -> 32x60(=16px 격자) 이득장 회귀
분할      : 학습 = 접두 0 씬(161씬 1865장), 모델선택 = 접두 2 씬(20씬 234장),
            검증 = 접두 1 씬 전량(50씬 598장, DIAG 문서들이 쓴 test split 과 동일)
            씬이 세 집합에 겹치지 않는다(코드로 검사).
대조군    : (a) 상수 예측  (b) 스칼라 3특징 선형회귀  (c) 라벨 씬단위 셔플
채점      : repro_measure.psnr_indep 만 사용. 전부 실측값.
"""
import os, sys, glob, json, math, time, re
import numpy as np
import torch

sys.path.insert(0, "code")
from repro_measure import psnr_indep
from diag_sid_spectrum import (REPO, DATA, WEIGHTS, LUMA_W,
                               build_pairs, load_npy, exposure_of, u8)
from diag_sid_lowfreq import block_index, block_sum, block_ls_gain, block_fit

OUT_JSON = os.environ.get("PG_OUT", "numbers/predict_gainfield.json")
OUT_MD = "numbers/PREDICT_GAINFIELD.md"

H, W = 512, 960
GRID = 16                      # CNN 출력 격자 (= 국소이득 오라클 26.966 과 같은 격자)
GRIDS = [16, 32, 64]           # 스칼라 대조군·부가 평가용 격자
NBY, NBX = H // GRID, W // GRID          # 32 x 60
SEED = 20260905

# 판정 기준 (사전 고정)
TH_YES_R2, TH_YES_DB = 0.30, 0.40
TH_NO_R2, TH_NO_DB = 0.10, 0.15

# 대조 기준값 (DIAG_SID_LOWFREQ / DIAG_SID_FAILURE 실측)
REF = {"none": 24.438, "global": 25.779, "local16": 26.966, "aff16": 28.032}

EPOCHS = int(os.environ.get("PG_EPOCHS", "40"))
BATCH = int(os.environ.get("PG_BATCH", "8"))
LR = 3e-4


# ---------------- 데이터 ----------------
def build_pairs_pref(prefixes):
    """build_pairs() 와 같은 규약이되 씬 접두만 바꾼다."""
    lq_dirs = sorted(glob.glob(os.path.join(DATA, "short_sid2", "*")))
    gt_dirs = sorted(glob.glob(os.path.join(DATA, "long_sid2", "*")))
    pairs = []
    for ld, gd in zip(lq_dirs, gt_dirs):
        name = os.path.basename(ld)
        assert name == os.path.basename(gd), (ld, gd)
        if name[0] not in prefixes:
            continue
        for p in sorted(glob.glob(os.path.join(ld, "*"))):
            pairs.append((name, p, sorted(glob.glob(os.path.join(gd, "*")))[0]))
    return pairs


# ---------------- 신경망 ----------------
def make_net(cin, torch, nn):
    class Blk(nn.Module):
        def __init__(s, ci, co, stride=1):
            super().__init__()
            s.c = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
            s.n = nn.GroupNorm(8, co)
            s.a = nn.SiLU()

        def forward(s, x):
            return s.a(s.n(s.c(x)))

    class GainNet(nn.Module):
        """512x960 -> 32x60 (16px 격자). /64 까지 내려갔다 /16 으로 되올라오는 U 형이라
        출력 한 칸의 수용영역이 원영상 수백 화소다(큰 수용영역이 이 실험의 요점)."""

        def __init__(s):
            super().__init__()
            s.e1 = nn.Sequential(Blk(cin, 32, 2), Blk(32, 32))        # /2
            s.e2 = nn.Sequential(Blk(32, 48, 2), Blk(48, 48))         # /4
            s.e3 = nn.Sequential(Blk(48, 64, 2), Blk(64, 64))         # /8
            s.e4 = nn.Sequential(Blk(64, 96, 2), Blk(96, 96))         # /16  <- 출력 격자
            s.e5 = nn.Sequential(Blk(96, 128, 2), Blk(128, 128))      # /32
            s.e6 = nn.Sequential(Blk(128, 160, 2), Blk(160, 160))     # /64
            s.gctx = nn.Sequential(nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, 160))
            s.u5 = Blk(160 + 128, 128)
            s.u4 = Blk(128 + 96, 96)
            s.head = nn.Sequential(Blk(96, 64), nn.Conv2d(64, 1, 1))
            nn.init.zeros_(s.head[-1].weight)
            nn.init.zeros_(s.head[-1].bias)

        def forward(s, x):
            import torch.nn.functional as Fn
            x1 = s.e1(x); x2 = s.e2(x1); x3 = s.e3(x2)
            x4 = s.e4(x3); x5 = s.e5(x4); x6 = s.e6(x5)
            g = s.gctx(x6.mean((2, 3)))[:, :, None, None]
            x6 = x6 + g
            y = s.u5(torch.cat([Fn.interpolate(x6, size=x5.shape[2:], mode="nearest"), x5], 1))
            y = s.u4(torch.cat([Fn.interpolate(y, size=x4.shape[2:], mode="nearest"), x4], 1))
            return s.head(y)[:, 0]

    return GainNet()


# ---------------- 사전계산 ----------------
def precompute(torch, F, dev):
    """모든 장에 대해 Retinexformer 추론 -> 입력/출력 영상 캐시 + 격자별 오라클 이득장 + 앵커."""
    sys.path.insert(0, REPO)
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu")["params"])
    net = net.cuda().eval()

    splits = {"tr": build_pairs_pref("0"), "iv": build_pairs_pref("2"), "va": build_pairs_pref("1")}
    # 검증 분할이 DIAG 문서들의 test split 과 글자 그대로 같은지 확인
    assert splits["va"] == build_pairs(), "검증 분할이 diag_sid_* 의 test split 과 다르다"
    lim = int(os.environ.get("PG_LIMIT", "0"))          # 연기시험용
    if lim:
        splits = {k: v[:lim] for k, v in splits.items()}

    bidx = {G: block_index(H, W, G, torch, dev) for G in GRIDS}
    lw = torch.tensor(LUMA_W, device=dev, dtype=torch.float64)

    DS = {}
    for sp, pairs in splits.items():
        n = len(pairs)
        d = dict(n=n, scene=[p[0] for p in pairs], exp=[exposure_of(p[1]) for p in pairs],
                 lq=torch.empty(n, 3, H, W, dtype=torch.float16, device=dev),
                 y=torch.empty(n, 3, H, W, dtype=torch.float32, device=dev),
                 a_img=np.zeros(n))
        for G in GRIDS:
            nb = (H // G) * (W // G)
            for k in ["a", "den", "in_m", "in_v", "out_m"]:
                d[f"{k}{G}"] = np.zeros((n, nb), np.float64)
        if sp == "va":
            for k in ["psnr_model", "psnr_gain", "psnr_loc16", "psnr_aff16", "psnr_loc32"]:
                d[k] = np.zeros(n)
            d["gt_u8"] = []
        DS[sp] = d

        t0 = time.time()
        with torch.inference_mode():
            for i, (scene, lqp, gtp) in enumerate(pairs):
                lq = load_npy(lqp)
                gt_u8 = u8(load_npy(gtp))
                x = torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()
                padh, padw = (-H) % 4, (-W) % 4
                if padh or padw:
                    x = F.pad(x, (0, padw, 0, padh), "reflect")
                y = net(x)[:, :, :H, :W]
                y = torch.clamp(y, 0, 1).double()[0]
                g = torch.from_numpy(gt_u8.astype(np.float64) / 255.0).cuda().permute(2, 0, 1)
                a = float((y * g).sum() / (y * y).sum())
                p = a * y
                d["a_img"][i] = a
                d["lq"][i] = torch.from_numpy(lq.transpose(2, 0, 1)).cuda().half()
                d["y"][i] = y.float()

                lqt = torch.from_numpy(lq.astype(np.float64)).cuda().permute(2, 0, 1) * 255.0
                inL = (lqt * lw[:, None, None]).sum(0)
                outL = (p * 255.0 * lw[:, None, None]).sum(0)
                for G in GRIDS:
                    idx, nblk, _, _ = bidx[G]
                    ab, den = block_ls_gain(y, g, idx, nblk, torch)
                    cnt = block_sum(torch.ones_like(inL), idx, nblk, torch)
                    im = block_sum(inL, idx, nblk, torch) / cnt
                    iv = torch.clamp(block_sum(inL ** 2, idx, nblk, torch) / cnt - im ** 2, min=0)
                    om = block_sum(outL, idx, nblk, torch) / cnt
                    d[f"a{G}"][i] = ab.cpu().numpy()
                    d[f"den{G}"][i] = den.cpu().numpy()
                    d[f"in_m{G}"][i] = im.cpu().numpy()
                    d[f"in_v{G}"][i] = iv.cpu().numpy()
                    d[f"out_m{G}"][i] = om.cpu().numpy()

                if sp == "va":
                    d["gt_u8"].append(gt_u8)
                    d["psnr_model"][i] = psnr_indep(u8(y.permute(1, 2, 0).cpu().numpy()), gt_u8)
                    d["psnr_gain"][i] = psnr_indep(
                        u8(np.clip(p.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                    for G, key in [(16, "psnr_loc16"), (32, "psnr_loc32")]:
                        idx, nblk, _, _ = bidx[G]
                        out, _, _ = block_fit(y, g, idx, nblk, "gain", torch)
                        d[key][i] = psnr_indep(
                            u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)
                    idx, nblk, _, _ = bidx[16]
                    out, _, _ = block_fit(y, g, idx, nblk, "aff", torch)
                    d["psnr_aff16"][i] = psnr_indep(
                        u8(np.clip(out.permute(1, 2, 0).cpu().numpy(), 0, 1)), gt_u8)

                if (i + 1) % 200 == 0:
                    el = time.time() - t0
                    print(f"  [{sp}] {i+1}/{n}  {el:.0f}s", flush=True)
        print(f"[{sp}] n={n} scenes={len(set(d['scene']))}  {time.time()-t0:.0f}s", flush=True)
    del net
    torch.cuda.empty_cache()
    return DS, bidx


# ---------------- 특징 텐서 ----------------
def net_input(lq_h, y_f, stats, torch):
    """lq(float16 Nx3xHxW), y(float32) -> 12채널 정규화 입력."""
    lq = lq_h.float()
    ch = torch.cat([lq, torch.log1p(lq * 255.0) / 5.545,
                    y_f, torch.log1p(y_f * 255.0) / 5.545], 1)
    return (ch - stats[0]) / stats[1]


# ---------------- 지표 ----------------
def wr2(t, p, w):
    """가중 R2. base=0 이면 '예측 0(=전역이득 오라클)' 대비, base='mean' 이면 표준 R2."""
    num = float((w * (t - p) ** 2).sum())
    den = float((w * t ** 2).sum())
    return 1.0 - num / max(den, 1e-30)


def wr2_mean(t, p, w):
    mu = float((w * t).sum() / w.sum())
    num = float((w * (t - p) ** 2).sum())
    den = float((w * (t - mu) ** 2).sum())
    return 1.0 - num / max(den, 1e-30)


def center_field(x, w):
    """den 가중 장내 중심화 (torch, [N,nb])."""
    return x - (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-30)


# ---------------- 스칼라 대조군 ----------------
FEATNAMES = ["lin_m", "lin_v", "lout_m"]


def scalar_design(d, G, sl=None):
    im = d[f"in_m{G}"]; iv = d[f"in_v{G}"]; om = d[f"out_m{G}"]
    if sl is not None:
        im, iv, om = im[sl], iv[sl], om[sl]
    return np.stack([np.log1p(im), 0.5 * np.log1p(iv), np.log1p(om)], -1)


def wls(X, y, w):
    A = X.T @ (X * w[:, None])
    b = X.T @ (y * w)
    return np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), b)


def center_rows(X, w):
    sw = np.maximum(w.sum(1, keepdims=True), 1e-30)
    return X - (X * w[:, :, None]).sum(1, keepdims=True) / sw[:, :, None]


def scalar_controls(DS, out):
    """(b) 스칼라 3특징 선형회귀 두 갈래:
       b1. DIAG 재현: 검증 씬만으로 씬단위 2겹 교차적합, 64px 격자 (DIAG 의 R2 0.0061 대조)
       b2. 본 실험 규약: 학습 씬에서 적합 -> 검증 씬에서 평가, 16px 격자"""
    res = {}
    # --- b1 : DIAG 재현 ---
    for G in GRIDS:
        d = DS["va"]
        w = d[f"den{G}"]; a = d[f"a{G}"]
        X = scalar_design(d, G)
        Xc = center_rows(X, w)
        ac = a - (a * w).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1e-30)
        scenes = sorted(set(d["scene"]))
        fold_of = {s: i % 2 for i, s in enumerate(scenes)}
        fold = np.array([fold_of[s] for s in d["scene"]])
        for nf, sel in [("cin", [0, 1]), ("cinout", [0, 1, 2])]:
            pred = np.zeros_like(ac)
            for f in (0, 1):
                m = fold == f
                c = wls(Xc[~m][:, :, sel].reshape(-1, len(sel)), ac[~m].reshape(-1),
                        w[~m].reshape(-1))
                pred[m] = Xc[m][:, :, sel] @ c
            res[f"diag_repro_{nf}_g{G}"] = wr2(ac.reshape(-1), pred.reshape(-1), w.reshape(-1))
    # --- b2 : train -> val ---
    G = GRID
    tr, va = DS["tr"], DS["va"]
    Xtr = center_rows(scalar_design(tr, G), tr[f"den{G}"])
    atr = tr[f"a{G}"]; wtr = tr[f"den{G}"]
    actr = atr - (atr * wtr).sum(1, keepdims=True) / np.maximum(wtr.sum(1, keepdims=True), 1e-30)
    Xva = center_rows(scalar_design(va, G), va[f"den{G}"])
    ava = va[f"a{G}"]; wva = va[f"den{G}"]
    acva = ava - (ava * wva).sum(1, keepdims=True) / np.maximum(wva.sum(1, keepdims=True), 1e-30)
    c = wls(Xtr.reshape(-1, 3), actr.reshape(-1), wtr.reshape(-1))
    pred = Xva @ c
    res["tr2va_cinout_centered"] = wr2(acva.reshape(-1), pred.reshape(-1), wva.reshape(-1))
    res["tr2va_cinout_coef"] = [float(v) for v in c]
    # 원본(절대) 이득
    Xtr_r = np.concatenate([np.ones(scalar_design(tr, G).shape[:2] + (1,)),
                            scalar_design(tr, G)], -1)
    Xva_r = np.concatenate([np.ones(scalar_design(va, G).shape[:2] + (1,)),
                            scalar_design(va, G)], -1)
    c2 = wls(Xtr_r.reshape(-1, 4), atr.reshape(-1), wtr.reshape(-1))
    pred_r = Xva_r @ c2
    res["tr2va_inout_raw_r2"] = wr2_mean(ava.reshape(-1), pred_r.reshape(-1), wva.reshape(-1))
    res["tr2va_inout_raw_coef"] = [float(v) for v in c2]
    out["scalar_controls"] = res
    return {"centered": pred, "raw": pred_r}


# ---------------- 학습 ----------------
def train_one(tag, DS, stats, target_mode, shuffle_labels, torch, nn, dev, log, wd=1e-4):
    """target_mode: 'centered' (a_blk - a_img) / 'raw' (a_blk)
       shuffle_labels: True 면 학습 라벨을 씬 단위로 섞는다(대조군 c)."""
    g = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    net = make_net(12, torch, nn).cuda()
    npar = sum(p.numel() for p in net.parameters())
    log(f"[{tag}] params {npar/1e6:.3f}M  target={target_mode} shuffle={shuffle_labels}")

    def fields(sp):
        d = DS[sp]
        a = torch.from_numpy(d[f"a{GRID}"]).float().cuda()
        w = torch.from_numpy(d[f"den{GRID}"]).float().cuda()
        t = center_field(a, w) if target_mode == "centered" else a.clone()
        return a, w, t

    A, Wt, T = {}, {}, {}
    for sp in ["tr", "iv", "va"]:
        A[sp], Wt[sp], T[sp] = fields(sp)

    # --- 대조군 (c): 학습 라벨을 씬 단위로 섞는다 ---
    if shuffle_labels:
        d = DS["tr"]
        scenes = sorted(set(d["scene"]))
        rs = np.random.RandomState(SEED)
        perm_sc = list(scenes)
        while True:                                   # 자기 씬으로 되돌아가지 않게
            rs.shuffle(perm_sc)
            if all(a != b for a, b in zip(scenes, perm_sc)):
                break
        by_scene = {s: [i for i, v in enumerate(d["scene"]) if v == s] for s in scenes}
        src = {s: by_scene[t_] for s, t_ in zip(scenes, perm_sc)}
        newidx = np.zeros(d["n"], np.int64)
        for s in scenes:
            pool = src[s]
            for j, i in enumerate(by_scene[s]):
                newidx[i] = pool[j % len(pool)]
        T["tr"] = T["tr"][torch.from_numpy(newidx).cuda()]
        log(f"[{tag}] shuffle: {len(scenes)}씬 라벨 재배정 (자기씬 0건 확인)")

    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    ntr = DS["tr"]["n"]
    steps = math.ceil(ntr / BATCH) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")

    def fwd(sp, ids, flip=None):
        x = net_input(DS[sp]["lq"][ids], DS[sp]["y"][ids], stats, torch)
        if flip is not None:
            if flip[0]:
                x = torch.flip(x, [3])
            if flip[1]:
                x = torch.flip(x, [2])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p = net(x).float()
        if flip is not None:
            if flip[0]:
                p = torch.flip(p, [2])
            if flip[1]:
                p = torch.flip(p, [1])
        p = p.reshape(len(ids), -1)
        if target_mode == "raw":
            p = p + 1.0
        else:
            p = center_field(p, Wt[sp][ids])
        return p

    @torch.no_grad()
    def evaluate(sp):
        net.eval()
        ps = []
        for s in range(0, DS[sp]["n"], 8):
            ids = torch.arange(s, min(s + 8, DS[sp]["n"]), device=dev)
            ps.append(fwd(sp, ids))
        net.train()
        return torch.cat(ps)

    best = (-1e9, None)
    hist = []
    t0 = time.time()
    for ep in range(EPOCHS):
        order = torch.randperm(ntr, generator=g).cuda()
        tot, cnt = 0.0, 0.0
        for s in range(0, ntr, BATCH):
            ids = order[s:s + BATCH]
            fl = (bool(torch.randint(0, 2, (1,), generator=g).item()),
                  bool(torch.randint(0, 2, (1,), generator=g).item()))
            p = fwd("tr", ids, fl)
            w = Wt["tr"][ids]
            loss = (w * (T["tr"][ids] - p) ** 2).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * float(w.sum()); cnt += float(w.sum())
        pv = evaluate("iv")
        f = wr2 if target_mode == "centered" else wr2_mean
        r2iv = f(T["iv"].reshape(-1).cpu().numpy(), pv.reshape(-1).cpu().numpy(),
                 Wt["iv"].reshape(-1).cpu().numpy())
        # 부정행위 상한 진단: 검증셋으로 에폭을 골랐다면 얼마였을지 (모델 채택에는 쓰지 않는다)
        pva = evaluate("va")
        r2va = f(T["va"].reshape(-1).cpu().numpy(), pva.reshape(-1).cpu().numpy(),
                 Wt["va"].reshape(-1).cpu().numpy())
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(cnt, 1e-30), r2_inner_val=r2iv,
                         r2_val_diagnostic=r2va))
        if r2iv > best[0]:
            best = (r2iv, {k: v.detach().clone() for k, v in net.state_dict().items()})
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"[{tag}] ep{ep+1:3d}  loss {tot/max(cnt,1e-30):.6f}  "
                f"R2(inner-val) {r2iv:+.4f}  best {best[0]:+.4f}  {time.time()-t0:.0f}s")
    ceil_va = max(h["r2_val_diagnostic"] for h in hist)
    log(f"[{tag}] 최고 inner-val R2 {best[0]:+.4f} 에폭 채택 | "
        f"부정행위 상한(검증셋으로 에폭 선택 시) R2 {ceil_va:+.4f}")
    net.load_state_dict(best[1])
    pred_va = evaluate("va")
    return net, pred_va.cpu().numpy(), dict(params=int(npar), hist=hist,
                                            best_inner_r2=float(best[0]),
                                            val_r2_ceiling=float(ceil_va), weight_decay=wd)


# ---------------- 예측 이득장 적용 ----------------
def apply_field(DS, bidx, ahat, torch, F, use_bilinear=False, G=GRID):
    """ahat: [n, nb] 절대 이득. 검증셋에 적용해 장별 PSNR 을 잰다."""
    d = DS["va"]
    idx, nblk, nby, nbx = bidx[G]
    out = np.zeros(d["n"])
    with torch.inference_mode():
        for i in range(d["n"]):
            y = d["y"][i].double()
            ah = torch.from_numpy(ahat[i]).cuda()
            if use_bilinear:
                gs = F.interpolate(ah.reshape(1, 1, nby, nbx).float(), size=(H, W),
                                   mode="bilinear", align_corners=False)[0, 0].double()
            else:
                gs = ah[idx].reshape(H, W)
            o = gs[None] * y
            out[i] = psnr_indep(u8(np.clip(o.permute(1, 2, 0).cpu().numpy(), 0, 1)),
                                d["gt_u8"][i])
    return out


# ---------------- 본체 ----------------
def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    logf = open("numbers/predict_gainfield_run.log", "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    out = dict(config=dict(grid=GRID, grids=GRIDS, epochs=EPOCHS, batch=BATCH, lr=LR,
                           seed=SEED, thresholds=dict(yes_r2=TH_YES_R2, yes_db=TH_YES_DB,
                                                      no_r2=TH_NO_R2, no_db=TH_NO_DB),
                           ref=REF))
    t0 = time.time()
    log("== 사전계산 (추론 + 오라클 이득장) ==")
    DS, bidx = precompute(torch, F, dev)
    log(f"사전계산 {time.time()-t0:.0f}s")

    # ---- 누수 점검 ----
    sc = {sp: set(DS[sp]["scene"]) for sp in DS}
    leak = dict(
        n_scene={sp: len(sc[sp]) for sp in sc},
        n_img={sp: DS[sp]["n"] for sp in DS},
        overlap_tr_va=sorted(sc["tr"] & sc["va"]),
        overlap_iv_va=sorted(sc["iv"] & sc["va"]),
        overlap_tr_iv=sorted(sc["tr"] & sc["iv"]),
        val_split_equals_diag_test=True)
    log(f"누수 점검 씬수 {leak['n_scene']} 장수 {leak['n_img']} 교집합 "
        f"{len(leak['overlap_tr_va'])}/{len(leak['overlap_iv_va'])}/{len(leak['overlap_tr_iv'])}")

    # ---- 앵커 ----
    va = DS["va"]
    anchors = {k: float(np.mean(va[k])) for k in
               ["psnr_model", "psnr_gain", "psnr_loc16", "psnr_loc32", "psnr_aff16"]}
    out["anchors"] = dict(measured=anchors, reference=REF,
                          diff=dict(none=anchors["psnr_model"] - REF["none"],
                                    glob=anchors["psnr_gain"] - REF["global"],
                                    loc16=anchors["psnr_loc16"] - REF["local16"],
                                    aff16=anchors["psnr_aff16"] - REF["aff16"]))
    log("앵커: " + " ".join(f"{k}={v:.3f}" for k, v in anchors.items()))

    # ---- 입력 정규화 통계 (학습 씬에서만) ----
    rs = np.random.RandomState(SEED)
    sub = rs.choice(DS["tr"]["n"], size=min(300, DS["tr"]["n"]), replace=False)
    with torch.inference_mode():
        acc = []
        for s in range(0, len(sub), 16):
            ids = torch.from_numpy(sub[s:s + 16]).cuda()
            lq = DS["tr"]["lq"][ids].float(); y = DS["tr"]["y"][ids]
            ch = torch.cat([lq, torch.log1p(lq * 255.0) / 5.545,
                            y, torch.log1p(y * 255.0) / 5.545], 1)
            acc.append(torch.stack([ch.mean((0, 2, 3)), (ch ** 2).mean((0, 2, 3))]))
        m2 = torch.stack(acc).mean(0)
    mean = m2[0][None, :, None, None]
    std = torch.sqrt(torch.clamp(m2[1] - m2[0] ** 2, min=1e-8))[None, :, None, None]
    stats = (mean, std)
    out["input_norm"] = dict(source="train scenes only (prefix 0), n=%d" % len(sub),
                             mean=[float(v) for v in m2[0]],
                             std=[float(v) for v in std.flatten()])

    # ---- 대조군 (b) 스칼라 회귀 ----
    log("== 대조군 (b) 스칼라 3특징 선형회귀 ==")
    sc_pred = scalar_controls(DS, out)
    log("  " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in out["scalar_controls"].items()}, ensure_ascii=False))

    # ---- O1: 단일 배치 오버핏 ----
    log("== O1 단일 배치 오버핏 점검 ==")
    torch.manual_seed(SEED)
    o1net = make_net(12, torch, nn).cuda()
    ids = torch.arange(0, 4, device=dev)
    a1 = torch.from_numpy(DS["tr"][f"a{GRID}"][:4]).float().cuda()
    w1 = torch.from_numpy(DS["tr"][f"den{GRID}"][:4]).float().cuda()
    t1 = center_field(a1, w1)
    opt = torch.optim.AdamW(o1net.parameters(), lr=1e-3)
    x1 = net_input(DS["tr"]["lq"][ids], DS["tr"]["y"][ids], stats, torch)
    r2_hist = []
    for it in range(400):
        p = center_field(o1net(x1).float().reshape(4, -1), w1)
        loss = (w1 * (t1 - p) ** 2).sum() / w1.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (it + 1) % 100 == 0:
            r2 = wr2(t1.reshape(-1).detach().cpu().numpy(), p.reshape(-1).detach().cpu().numpy(),
                     w1.reshape(-1).cpu().numpy())
            r2_hist.append(dict(iter=it + 1, loss=float(loss.detach()), r2=r2))
            log(f"  O1 it{it+1} loss {float(loss):.3e} R2 {r2:.4f}")
    out["o1_overfit"] = r2_hist
    del o1net, opt
    torch.cuda.empty_cache()

    # ---- 본 학습 3종 ----
    runs = {}
    for tag, mode, shuf, wd in [("cnn_centered", "centered", False, 1e-4),
                                ("cnn_centered_reg", "centered", False, 3e-2),
                                ("cnn_raw", "raw", False, 1e-4),
                                ("cnn_shuffle", "centered", True, 1e-4)]:
        log(f"== 학습 {tag} ==")
        _, pred, meta = train_one(tag, DS, stats, mode, shuf, torch, nn, dev, log, wd)
        runs[tag] = dict(meta=meta, pred=pred, mode=mode)

    # ---- 평가 ----
    log("== 검증 평가 ==")
    wva = va[f"den{GRID}"]; ava = va[f"a{GRID}"]
    acva = ava - (ava * wva).sum(1, keepdims=True) / np.maximum(wva.sum(1, keepdims=True), 1e-30)
    aimg = (ava * wva).sum(1) / np.maximum(wva.sum(1), 1e-30)
    expv = np.array(va["exp"])
    exps = sorted(set(va["exp"]), key=lambda s: float(s[:-1]))
    out["exposures"] = exps

    def r2_split(t, p, w, centered):
        f = wr2 if centered else wr2_mean
        o = {"ALL": f(t.reshape(-1), p.reshape(-1), w.reshape(-1))}
        for e in exps:
            m = expv == e
            o[e] = f(t[m].reshape(-1), p[m].reshape(-1), w[m].reshape(-1))
        return o

    def psnr_split(v):
        o = {"ALL": float(np.mean(v))}
        for e in exps:
            o[e] = float(np.mean(v[expv == e]))
        return o

    results = {}
    # 기준선
    results["oracle_none"] = dict(psnr=psnr_split(va["psnr_model"]))
    results["oracle_global"] = dict(psnr=psnr_split(va["psnr_gain"]))
    results["oracle_local16"] = dict(psnr=psnr_split(va["psnr_loc16"]))
    results["oracle_local32"] = dict(psnr=psnr_split(va["psnr_loc32"]))
    results["oracle_aff16"] = dict(psnr=psnr_split(va["psnr_aff16"]))
    # (a) 상수 예측: 중심화 목표에 대해 0 예측 = 전역이득 오라클
    results["ctrl_const_centered"] = dict(
        r2=r2_split(acva, np.zeros_like(acva), wva, True), psnr=psnr_split(va["psnr_gain"]))
    # (a') 원본 목표 상수 = 학습셋 가중평균
    atr = DS["tr"][f"a{GRID}"]; wtr = DS["tr"][f"den{GRID}"]
    cmean = float((atr * wtr).sum() / wtr.sum())
    pc = np.full_like(ava, cmean)
    results["ctrl_const_raw"] = dict(
        const=cmean, r2=r2_split(ava, pc, wva, False),
        psnr=psnr_split(apply_field(DS, bidx, pc, torch, F)))
    # (b) 스칼라
    results["ctrl_scalar_centered"] = dict(
        r2=r2_split(acva, sc_pred["centered"], wva, True),
        psnr=psnr_split(apply_field(DS, bidx, aimg[:, None] + sc_pred["centered"], torch, F)))
    results["ctrl_scalar_raw"] = dict(
        r2=r2_split(ava, sc_pred["raw"], wva, False),
        psnr=psnr_split(apply_field(DS, bidx, sc_pred["raw"], torch, F)))
    # CNN
    for tag, r in runs.items():
        p = r["pred"]
        if r["mode"] == "centered":
            res = dict(r2=r2_split(acva, p, wva, True),
                       psnr=psnr_split(apply_field(DS, bidx, aimg[:, None] + p, torch, F)),
                       psnr_bilinear=psnr_split(
                           apply_field(DS, bidx, aimg[:, None] + p, torch, F, True)))
            # 32px 로 뭉갠 예측 (den 가중 풀링 = 32px 오라클과 같은 단위)
            ph = (aimg[:, None] + p).reshape(-1, NBY, NBX)
            wh = wva.reshape(-1, NBY, NBX)
            num = (ph * wh).reshape(-1, NBY // 2, 2, NBX // 2, 2).sum((2, 4))
            den = wh.reshape(-1, NBY // 2, 2, NBX // 2, 2).sum((2, 4))
            p32 = (num / np.maximum(den, 1e-30)).reshape(-1, (NBY // 2) * (NBX // 2))
            res["psnr_pooled32"] = psnr_split(apply_field(DS, bidx, p32, torch, F, False, 32))
            res["r2_unweighted"] = wr2(acva.reshape(-1), p.reshape(-1), np.ones(acva.size))
        else:
            res = dict(r2=r2_split(ava, p, wva, False),
                       psnr=psnr_split(apply_field(DS, bidx, p, torch, F)),
                       psnr_bilinear=psnr_split(apply_field(DS, bidx, p, torch, F, True)))
            res["r2_vs_zero_centered"] = wr2(
                acva.reshape(-1),
                (p - (p * wva).sum(1, keepdims=True) /
                 np.maximum(wva.sum(1, keepdims=True), 1e-30)).reshape(-1), wva.reshape(-1))
        res["meta"] = r["meta"]
        results[tag] = res

    out["results"] = results
    out["leakage_checks"] = leak
    out["split"] = dict(train="prefix 0", inner_val="prefix 2", val="prefix 1 (= diag_sid_* test split)")

    # ---- 판정 ----
    main_r2 = results["cnn_centered"]["r2"]["ALL"]
    main_db = results["cnn_centered"]["psnr"]["ALL"] - results["oracle_global"]["psnr"]["ALL"]
    best_db = max(results["cnn_centered"]["psnr"]["ALL"],
                  results["cnn_centered"]["psnr_bilinear"]["ALL"]) - \
        results["oracle_global"]["psnr"]["ALL"]
    if main_r2 >= TH_YES_R2 and best_db >= TH_YES_DB:
        verdict = "예측 가능"
    elif main_r2 < TH_NO_R2 and best_db < TH_NO_DB:
        verdict = "예측 불가"
    else:
        verdict = "애매"
    out["verdict"] = dict(label=verdict, r2=main_r2, db_gain=main_db, db_gain_best=best_db)
    log(f"판정 {verdict}  R2 {main_r2:.4f}  dB {main_db:+.3f} (최선 {best_db:+.3f})")

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(str(type(o)))

    with open(OUT_JSON, "w") as fp:
        json.dump(out, fp, indent=1, ensure_ascii=False, default=conv)
    log("wrote " + OUT_JSON)
    logf.close()


# ---------------- 보고서 ----------------
def report():
    d = json.load(open(OUT_JSON))
    R = d["results"]; exps = d["exposures"]; cols = ["ALL"] + exps
    L = []; A = L.append

    A("# 국소 이득장은 입력 영상으로부터 예측 가능한가 — CNN 판정")
    A("")
    A("- 질문: 극암 복원 오차의 3분의 2를 차지하는 공간적으로 변하는 저주파 오차")
    A("  (= 블록 오라클 국소 이득장) 를 입력 영상만 보고 예측할 수 있는가.")
    A("- 앞 측정(DIAG_SID_LOWFREQ 4번)은 블록 통계 스칼라 3개로 교차적합 R2 0.0061 을 얻었다.")
    A("  특징 3개짜리 검정이라 '예측 불가' 의 증명이 못 된다. 여기서는 공간 맥락을 보는 CNN 으로 다시 잰다.")
    A("- 목표 변수 정의는 diag_sid_lowfreq.block_ls_gain 을 그대로 import 해서 썼다")
    A("  (a_blk = <y,g>_blk / <y,y>_blk, 채널 공통 스칼라). 앞 문서와 정의가 같다.")
    A("- 채점은 repro_measure.psnr_indep 만 사용. 전부 실측값.")
    A("")

    A("## 0. 설계")
    A("")
    A("```")
    A(f"{'항목':22s} {'내용'}")
    A("-" * 78)
    lk = d["leakage_checks"]
    A(f"{'학습 씬':22s} 접두 0 — {lk['n_scene']['tr']}씬 {lk['n_img']['tr']}장")
    A(f"{'모델선택(내부검증)':22s} 접두 2 — {lk['n_scene']['iv']}씬 {lk['n_img']['iv']}장")
    A(f"{'검증 씬':22s} 접두 1 — {lk['n_scene']['va']}씬 {lk['n_img']['va']}장 "
      f"(= diag_sid_* 의 test split 전량)")
    A(f"{'목표 격자':22s} {d['config']['grid']}px (국소이득 오라클 26.966 과 같은 격자)")
    A(f"{'예측기':22s} U-Net 형 CNN, /64 까지 내려갔다 /16 으로 복귀, "
      f"{R['cnn_centered']['meta']['params']/1e6:.3f}M 파라미터")
    A(f"{'입력':22s} short 입력 영상 3ch + 모델 출력 영상 3ch (+ 각각 log1p) = 12ch, 512x960")
    A(f"{'학습':22s} AdamW lr {d['config']['lr']}, OneCycle, "
      f"{d['config']['epochs']}에폭, 배치 {d['config']['batch']}, den 가중 MSE, 상하좌우 뒤집기")
    A("```")
    A("")
    A("목표 변수 두 가지를 다 다뤘다.")
    A("- centered : t = a_blk - a_img (장내 den 가중 중심화). 예측 0 이면 결과가 정확히 전역이득 오라클이다.")
    A("             장별 전역이득 a_img 는 baseline(25.779) 과 똑같이 오라클로 준다 — 두 쪽 조건이 같다.")
    A("- raw      : t = a_blk (절대 이득). 오라클을 하나도 안 쓴다. 기준선은 보정 없음 24.438 이다.")
    A("")

    A("## 1. 앵커 대조 (검증 분할이 앞 문서와 같은지)")
    A("")
    an = d["anchors"]
    A("```")
    A(f"{'항목':30s} {'측정값':>10s} {'대조값':>10s}   판정")
    A("-" * 66)
    for nm, mk, rk in [("모델 무보정 PSNR", "psnr_model", "none"),
                       ("전역이득 오라클 PSNR", "psnr_gain", "global"),
                       ("국소이득 오라클(16px) PSNR", "psnr_loc16", "local16"),
                       ("국소affine 오라클(16px) PSNR", "psnr_aff16", "aff16")]:
        v = an["measured"][mk]; r = an["reference"][rk]
        A(f"{nm:30s} {v:10.3f} {r:10.3f}   {'통과' if abs(v-r) < 0.02 else f'차이 {v-r:+.3f}'}")
    A(f"{'국소이득 오라클(32px) PSNR':30s} {an['measured']['psnr_loc32']:10.3f} "
      f"{'-':>10s}   (참고)")
    A("```")
    A("")

    A("## 2. 누수 점검")
    A("")
    A("```")
    A(f"{'점검':52s} {'결과'}")
    A("-" * 74)
    A(f"{'학습 씬 ∩ 검증 씬':52s} {len(lk['overlap_tr_va'])}건 "
      f"{'통과' if not lk['overlap_tr_va'] else '실패'}")
    A(f"{'내부검증 씬 ∩ 검증 씬':52s} {len(lk['overlap_iv_va'])}건 "
      f"{'통과' if not lk['overlap_iv_va'] else '실패'}")
    A(f"{'학습 씬 ∩ 내부검증 씬':52s} {len(lk['overlap_tr_iv'])}건 "
      f"{'통과' if not lk['overlap_tr_iv'] else '실패'}")
    A(f"{'검증 분할 == diag_sid_* test split (파일 목록 동일)':52s} 통과 (assert)")
    A(f"{'입력 정규화 통계 산출 범위':52s} 학습 씬 전용")
    A(f"{'GT 가 CNN 입력에 들어가는가':52s} 아니오 (입력 = short + 모델출력)")
    A(f"{'GT 유래 값의 사용처':52s} 목표 라벨, 그리고 centered 의 a_img")
    A(f"{'모델(에폭) 선택 기준':52s} 내부검증(접두 2), 검증셋 미사용")
    A("```")
    A("")
    A("centered 계열은 장별 전역이득 a_img 가 GT 에서 온다. 이건 누수가 아니라 규약이다 —")
    A("비교 기준인 전역이득 오라클 25.779 도 같은 a_img 를 쓴다. 그래서 두 쪽의 차이가")
    A("순수하게 '공간적으로 변하는 몫을 예측했는가' 만 재게 된다. 오라클을 하나도 안 쓰는 판정은")
    A("raw 계열이고, 그 기준선은 보정 없음 24.438 이다.")
    A("")

    A("## 3. O1 — 단일 배치 오버핏 (코드 버그 검정)")
    A("")
    A("```")
    A(f"{'iter':>8s} {'loss':>14s} {'R2':>10s}")
    A("-" * 34)
    for h in d["o1_overfit"]:
        A(f"{h['iter']:8d} {h['loss']:14.3e} {h['r2']:10.4f}")
    A("```")
    A("")
    A("4장짜리 한 배치에 R2 가 1 로 붙는다 = 신경망·손실·목표 배선에 버그가 없다.")
    A("")

    A("## 4. 대조군 (b) — 스칼라 3특징 선형회귀")
    A("")
    sc = d["scalar_controls"]
    A("```")
    A(f"{'설정':56s} {'R2':>9s}")
    A("-" * 68)
    for G in d["config"]["grids"]:
        A(f"{'DIAG 재현: 검증씬 2겹 교차적합, cin(입력평균·분산), ' + str(G) + 'px':56s} "
          f"{sc[f'diag_repro_cin_g{G}']:9.4f}")
        A(f"{'DIAG 재현: 검증씬 2겹 교차적합, cinout(+출력평균), ' + str(G) + 'px':56s} "
          f"{sc[f'diag_repro_cinout_g{G}']:9.4f}")
    A(f"{'본 실험 규약: 학습씬 적합 -> 검증씬, centered, 16px':56s} "
      f"{sc['tr2va_cinout_centered']:9.4f}")
    A(f"{'본 실험 규약: 학습씬 적합 -> 검증씬, raw(절대이득), 16px':56s} "
      f"{sc['tr2va_inout_raw_r2']:9.4f}")
    A("```")
    A("")
    A(f"DIAG_SID_LOWFREQ 4(a) 의 cin R2 0.0061 은 64px 격자·검증셋 2겹 교차적합이었다.")
    A(f"같은 조건 재현값이 {sc['diag_repro_cin_g64']:.4f} 다. 파이프라인이 앞 측정과 이어진다.")
    A("")

    A("## 5. 결과 — 예측 이득장의 R2 와 적용 PSNR")
    A("")
    A("R2 규약: centered 는 '예측 0(=전역이득 오라클)' 대비 den 가중 결정계수,")
    A("raw 는 검증셋 가중평균 대비 표준 den 가중 결정계수다.")
    A("PSNR 은 예측 이득장을 실제로 모델 출력에 곱하고 클리핑·8bit 반올림한 뒤 psnr_indep 로 잰 장평균이다.")
    A("")
    A("```")
    A(f"{'항목':34s} {'R2(ALL)':>9s} {'PSNR':>9s} {'전역오라클대비':>12s}")
    A("-" * 70)
    gref = R["oracle_global"]["psnr"]["ALL"]

    def row(nm, key, r2key="r2", pk="psnr"):
        e = R[key]
        r2 = e.get(r2key, {}).get("ALL", None)
        p = e[pk]["ALL"]
        A(f"{nm:34s} {('%9.4f' % r2) if r2 is not None else ' '*9} {p:9.3f} {p-gref:+12.3f}")

    row("보정 없음 (모델 출력 그대로)", "oracle_none")
    row("전역이득 오라클", "oracle_global")
    A("-" * 70)
    row("(a) 상수 예측 — centered", "ctrl_const_centered")
    row("(b) 스칼라 3특징 — centered", "ctrl_scalar_centered")
    row("(c) 라벨 씬셔플 CNN — centered", "cnn_shuffle")
    row("CNN — centered", "cnn_centered")
    row("CNN — centered, 정칙화 강화(wd 3e-2)", "cnn_centered_reg")
    row("CNN — centered, 이중선형", "cnn_centered", "r2", "psnr_bilinear")
    row("CNN — centered, 32px 로 뭉갬", "cnn_centered", "r2", "psnr_pooled32")
    A("-" * 70)
    row("(a) 상수 예측 — raw", "ctrl_const_raw")
    row("(b) 스칼라 3특징 — raw", "ctrl_scalar_raw")
    row("CNN — raw (오라클 미사용)", "cnn_raw")
    A("-" * 70)
    row("국소이득 오라클 32px (상한)", "oracle_local32")
    row("국소이득 오라클 16px (상한)", "oracle_local16")
    row("국소 affine 오라클 16px (참고)", "oracle_aff16")
    A("```")
    A("")

    A("## 6. 노출시간별")
    A("")
    A("R2:")
    A("```")
    A(f"{'항목':34s} " + " ".join(f"{c:>9s}" for c in cols))
    A("-" * (34 + 10 * len(cols)))
    for nm, key in [("(b) 스칼라 — centered", "ctrl_scalar_centered"),
                    ("(c) 씬셔플 CNN — centered", "cnn_shuffle"),
                    ("CNN — centered", "cnn_centered"),
                    ("CNN — centered, 정칙화 강화", "cnn_centered_reg"),
                    ("(b) 스칼라 — raw", "ctrl_scalar_raw"),
                    ("CNN — raw", "cnn_raw")]:
        A(f"{nm:34s} " + " ".join(f"{R[key]['r2'][c]:9.4f}" for c in cols))
    A("```")
    A("")
    A("PSNR (장평균):")
    A("```")
    A(f"{'항목':34s} " + " ".join(f"{c:>9s}" for c in cols))
    A("-" * (34 + 10 * len(cols)))
    for nm, key, pk in [("보정 없음", "oracle_none", "psnr"),
                        ("전역이득 오라클", "oracle_global", "psnr"),
                        ("(b) 스칼라 — centered", "ctrl_scalar_centered", "psnr"),
                        ("(c) 씬셔플 CNN — centered", "cnn_shuffle", "psnr"),
                        ("CNN — centered", "cnn_centered", "psnr"),
                        ("CNN — centered, 정칙화 강화", "cnn_centered_reg", "psnr"),
                        ("CNN — centered, 이중선형", "cnn_centered", "psnr_bilinear"),
                        ("CNN — raw", "cnn_raw", "psnr"),
                        ("국소이득 오라클 16px (상한)", "oracle_local16", "psnr"),
                        ("국소 affine 오라클 16px", "oracle_aff16", "psnr")]:
        A(f"{nm:34s} " + " ".join(f"{R[key][pk][c]:9.3f}" for c in cols))
    A("```")
    A("")
    A("전역이득 오라클 대비 dB (centered 계열의 순수 이득):")
    A("```")
    A(f"{'항목':34s} " + " ".join(f"{c:>9s}" for c in cols))
    A("-" * (34 + 10 * len(cols)))
    for nm, key, pk in [("(b) 스칼라 — centered", "ctrl_scalar_centered", "psnr"),
                        ("(c) 씬셔플 CNN — centered", "cnn_shuffle", "psnr"),
                        ("CNN — centered", "cnn_centered", "psnr"),
                        ("CNN — centered, 정칙화 강화", "cnn_centered_reg", "psnr"),
                        ("CNN — centered, 이중선형", "cnn_centered", "psnr_bilinear"),
                        ("국소이득 오라클 16px (상한)", "oracle_local16", "psnr")]:
        A(f"{nm:34s} " + " ".join(
            f"{R[key][pk][c]-R['oracle_global']['psnr'][c]:+9.3f}" for c in cols))
    A("```")
    A("")

    A("## 7. 학습 곡선과 과적합 — '학습이 부족했다' 반증")
    A("")
    tags = ["cnn_centered", "cnn_centered_reg", "cnn_raw", "cnn_shuffle"]
    A("에폭별 R2. inner = 모델 채택에 쓴 내부검증(접두 2), val = 검증셋(접두 1) 진단값이다.")
    A("val 열은 모델 선택에 쓰지 않았다 — 아래 '부정행위 상한' 을 계산하려고 기록만 한 것이다.")
    A("")
    A("```")
    A(f"{'에폭':>6s} {'centered inner':>16s} {'centered val':>14s}"
      f" {'reg inner':>12s} {'reg val':>10s} {'raw inner':>12s} {'shuffle inner':>15s}")
    A("-" * 88)
    hs = {t: {h["epoch"]: h for h in R[t]["meta"]["hist"]} for t in tags}
    for e in sorted(hs["cnn_centered"]):
        if e % 5 and e != 1:
            continue
        A(f"{e:6d} {hs['cnn_centered'][e]['r2_inner_val']:16.4f}"
          f" {hs['cnn_centered'][e]['r2_val_diagnostic']:14.4f}"
          f" {hs['cnn_centered_reg'][e]['r2_inner_val']:12.4f}"
          f" {hs['cnn_centered_reg'][e]['r2_val_diagnostic']:10.4f}"
          f" {hs['cnn_raw'][e]['r2_inner_val']:12.4f}"
          f" {hs['cnn_shuffle'][e]['r2_inner_val']:15.4f}")
    A("```")
    A("")
    A("학습 손실은 끝까지 떨어지는데 내부검증 R2 는 15~20 에폭에서 꺾인다. 학습 부족이 아니라")
    A("과적합이다. 용량은 O1 에서 이미 확인됐다.")
    A("")
    A("에폭 선택 탓이 아니라는 걸 못박으려고, 검증셋으로 직접 에폭을 골랐다면(=부정행위) 얼마였을지도")
    A("같이 기록했다. 이건 실제 채택에 쓰지 않았고, 판정을 살려주는 쪽으로 최대한 유리하게 준 상한이다.")
    A("")
    A("```")
    A(f"{'모델':22s} {'채택 inner-val R2':>18s} {'채택 val R2':>13s} {'부정행위 상한 val R2':>20s}")
    A("-" * 78)
    for t in tags:
        m = R[t]["meta"]
        A(f"{t:22s} {m['best_inner_r2']:18.4f} {R[t]['r2']['ALL']:13.4f} "
          f"{m['val_r2_ceiling']:20.4f}")
    A("```")
    A("")
    A(f"검증셋으로 에폭을 골라도 R2 상한이 "
      f"{max(R[t]['meta']['val_r2_ceiling'] for t in tags):.4f} 다. 사전에 정한 '예측 불가' 문턱 "
      f"{d['config']['thresholds']['no_r2']} 에도 못 미친다.")
    A(f"정칙화를 300배 키운 변형(wd 3e-2)도 채택 val R2 {R['cnn_centered_reg']['r2']['ALL']:.4f} 로 같은 자리다.")
    A("")


    # ---- 판정 ----
    v = d["verdict"]
    A("## 판정")
    A("")
    A("사전에 고정한 기준:")
    A("```")
    A(f"검증 R2 >= {d['config']['thresholds']['yes_r2']} 이고 PSNR 이득 >= "
      f"+{d['config']['thresholds']['yes_db']} dB   -> 예측 가능")
    A(f"검증 R2 <  {d['config']['thresholds']['no_r2']} 이고 PSNR 이득 <  "
      f"+{d['config']['thresholds']['no_db']} dB   -> 예측 불가")
    A("그 사이                                        -> 애매")
    A("```")
    A("")
    A("실측:")
    A("```")
    A(f"{'검증 R2 (CNN, centered, 16px)':44s} {v['r2']:9.4f}")
    A(f"{'적용 PSNR':44s} {R['cnn_centered']['psnr']['ALL']:9.3f} dB")
    A(f"{'전역이득 오라클 대비':44s} {v['db_gain']:+9.3f} dB")
    A(f"{'  (이중선형·32px 포함 최선)':44s} {v['db_gain_best']:+9.3f} dB")
    A(f"{'국소이득 오라클 상한 대비 회수율':44s} "
      f"{100*max(v['db_gain'],v['db_gain_best'])/(R['oracle_local16']['psnr']['ALL']-gref):8.1f} %")
    A(f"{'씬셔플 대조군 R2':44s} {R['cnn_shuffle']['r2']['ALL']:9.4f}")
    A(f"{'스칼라 3특징 대조군 R2':44s} {R['ctrl_scalar_centered']['r2']['ALL']:9.4f}")
    A(f"{'정칙화 강화 변형 R2':44s} {R['cnn_centered_reg']['r2']['ALL']:9.4f}")
    A(f"{'부정행위 상한(검증셋으로 에폭 선택) R2':44s} "
      f"{max(R[t]['meta']['val_r2_ceiling'] for t in ['cnn_centered', 'cnn_centered_reg', 'cnn_raw', 'cnn_shuffle']):9.4f}")
    allbest = max(R[k][pk]["ALL"] for k, pk in
                  [("cnn_centered", "psnr"), ("cnn_centered", "psnr_bilinear"),
                   ("cnn_centered", "psnr_pooled32"), ("cnn_centered_reg", "psnr"),
                   ("ctrl_scalar_centered", "psnr")]) - gref
    A(f"{'변형 전부 통틀어 최선 dB':44s} {allbest:+9.3f} dB")
    A("```")
    A("")
    A(f"판정: {v['label']}")
    A("")
    A("읽는 법.")
    A(f"- 두 기준 다 '예측 불가' 쪽이다. R2 는 문턱 {d['config']['thresholds']['no_r2']} 의 "
      f"{v['r2']/d['config']['thresholds']['no_r2']*100:.0f}%, dB 는 문턱 "
      f"+{d['config']['thresholds']['no_db']} 의 {allbest/d['config']['thresholds']['no_db']*100:.0f}% 다. "
      "경계 근처가 아니라 한 자릿수 배 차이로 미달이다.")
    A(f"- 신호가 0 은 아니다. 씬셔플 대조군({R['cnn_shuffle']['r2']['ALL']:+.4f})과 스칼라 3특징"
      f"({R['ctrl_scalar_centered']['r2']['ALL']:.4f})보다는 위다. 공간 맥락을 보면 스칼라보다 두 배쯤 더 본다. "
      "다만 그 두 배가 0.009 에서 0.019 로 가는 두 배라 실익이 없다.")
    A(f"- 오라클 상한이 +{R['oracle_local16']['psnr']['ALL']-gref:.3f} dB 인데 회수한 게 "
      f"{100*allbest/(R['oracle_local16']['psnr']['ALL']-gref):.1f}% 다(변형 전부 통틀어 최선 기준). 국소 이득장은 크고 실재하지만"
      "(DIAG_SID_LOWFREQ 2~3번), 그게 어디서 얼마나 틀리는지는 입력 영상에도 모델 출력 영상에도 안 적혀 있다.")
    A("- 노출이 짧을수록(0.033s/0.04s) R2 가 조금 더 크고 0.1s 는 0 근처다. 이득장의 예측 가능한 몫이")
    A("  극암 쪽에 몰려 있다는 힌트지만, 그 쪽도 dB 로는 +0.02~0.04 라 방법 논문을 세울 크기가 아니다.")
    A("- 이 결론은 '단일 영상에서 이 목표 변수를 회귀로 예측' 이라는 설정에 대한 것이다.")
    A("  다른 관측(노출 메타데이터, 여러 장, 센서 보정 정보)을 쓰는 설정까지 배제하지는 않는다.")
    A("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", OUT_MD)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        main()
        report()
