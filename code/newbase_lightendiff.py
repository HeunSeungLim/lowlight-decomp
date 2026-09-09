"""newbase_lightendiff.py

LightenDiffusion (ECCV 2024, Latent-Retinex Diffusion) 저자 배포 가중치로
LOLv1 eval15 15장 + Sony(SID sRGB 처리본) test 598장을 추론해 출력 캐시를 만든다.

비지도 방법이고 저자 배포 가중치는 폴더에 stage1/stage2 한 벌뿐이라
두 벤치마크에 같은 stage2_weight.pth.tar 를 그대로 쓴다.

규약
  입력   LOL  : diag_lolv1.build_pairs() (PIL RGB uint8 HWC 400x600) → /255
         Sony : diag_sid_failure.build_pairs()/load_npy() (HWC float32 [0,1] RGB 512x960,
                저자 로더의 BGR→RGB 뒤집기 포함) — compare_methods.sid_pairs/sid_load 와 동일 객체
  전처리 : 저자 evaluate.py → models/restoration.py 의 DiffusiveRestoration.restore 재현.
           x_cond = to_tensor(low) (CHW [0,1]), 우/하단 reflect 패딩으로 H,W 를 64 배수로,
           모델 입력은 cat([x_cond, x_cond], dim=1) (6채널), 출력은 [:, :, :h, :w] 로 크롭.
           저자 데이터로더(datasets/*)는 import 하지 않고 위 두 줄로 독립 재구현했다.
           (PairToTensor = torchvision F.to_tensor = uint8/255, CHW, RGB — 대조 확인함)
  출력   : [0,1] 클리핑 후 float32 CHW RGB .npy
           (저자는 tvu.save_image 로 저장하는데 그 안에서 mul(255).add(0.5).clamp(0,255)
            → 같은 클리핑 + 반올림이다. 우리는 float 를 저장하고 채점에서 rint 한다.)
  시드   : DDIM 초기 잡음 x=torch.randn 이 유일한 난수원. 장마다 SEED 로 재시드해서
           부분 재실행해도 같은 값이 나오게 한다. cudnn deterministic.

산출  numbers/cache_lightendiff/{LOL,Sony}/<id>.npy
      (compare_methods.py 의 NEW_CACHE 규약: CHW float32 RGB [0,1], id = LOL 파일명 / Sony basename)
      numbers/newbase_lightendiff_progress.json  진행 상황

사용  python newbase_lightendiff.py LOL
      python newbase_lightendiff.py Sony
      python newbase_lightendiff.py score          (캐시 → PSNR)
"""
import os, sys, json, time, argparse
import numpy as np

CODE = "code"
sys.path.insert(0, CODE)
from repro_measure import psnr_indep, ssim_indep, gt_mean_rectify        # noqa: E402
import diag_lolv1 as DL                                                  # noqa: E402
from diag_sid_failure import build_pairs as sid_pairs, load_npy as sid_load  # noqa: E402
from diag_sid_spectrum import u8                                         # noqa: E402

REPO = "third_party/LightenDiffusion"
CKPT = "third_party/dl/lightendiff/stage2/stage2_weight.pth.tar"
CFG = os.path.join(REPO, "configs", "unsupervised.yml")
OUT_ROOT = "numbers/cache_lightendiff"
PROG = "numbers/newbase_lightendiff_progress.json"
SEED = 20260905


# ---------------- 모델 (저자 코드의 모델 부분만 사용, 데이터로더는 안 씀) ----------------
def build_model():
    import torch, yaml
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from models.ddm import Net

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    def d2n(d):
        ns = argparse.Namespace()
        for k, v in d.items():
            setattr(ns, k, d2n(v) if isinstance(v, dict) else v)
        return ns

    config = d2n(cfg)
    config.device = torch.device("cuda")
    net_args = argparse.Namespace(mode="evaluation")

    net = Net(net_args, config)
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    # 저자는 DataParallel 로 감싼 뒤 load 한다(단일 GPU 에서 DataParallel 은 항등).
    # 여기서는 접두사만 떼고 같은 텐서를 그대로 넣는다 — strict=True 로 누락/여분을 잡는다.
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing = net.load_state_dict(sd, strict=True)
    print("load_state_dict:", missing, flush=True)
    net = net.cuda().eval()
    return net, config


def run_one(net, torch, F, lq01):
    """lq01: HWC float [0,1] RGB → CHW float32 [0,1] RGB (입력 해상도)."""
    x = torch.from_numpy(np.ascontiguousarray(lq01.transpose(2, 0, 1)))[None].float().cuda()
    h, w = x.shape[2], x.shape[3]
    hh = int(64 * np.ceil(h / 64.0))
    ww = int(64 * np.ceil(w / 64.0))
    if (hh, ww) != (h, w):
        x = F.pad(x, (0, ww - w, 0, hh - h), "reflect")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.no_grad():
        y = net(torch.cat((x, x), dim=1))["pred_x"][:, :, :h, :w]
    y = torch.clamp(y, 0.0, 1.0)[0].cpu().numpy().astype(np.float32)
    return y


# ---------------- 프레임 소스 ----------------
def lol_frames():
    for f, lq, gt in DL.build_pairs():
        yield f, lq.astype(np.float32) / 255.0, gt


def sony_frames():
    gt_cache = {}
    for scene, lqp, gtp in sid_pairs():
        if gtp not in gt_cache:
            gt_cache[gtp] = u8(sid_load(gtp))
        yield os.path.basename(lqp), sid_load(lqp), gt_cache[gtp]


FRAMES = {"LOL": lol_frames, "Sony": sony_frames}


def progress(bench, **kw):
    d = {}
    if os.path.exists(PROG):
        try:
            d = json.load(open(PROG))
        except Exception:
            d = {}
    d.setdefault(bench, {}).update(kw)
    d[bench]["stamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(d, open(PROG, "w"), indent=1, ensure_ascii=False)


def infer(bench, limit=0):
    import torch
    import torch.nn.functional as F
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_dir = os.path.join(OUT_ROOT, bench)
    os.makedirs(out_dir, exist_ok=True)
    net, _ = build_model()

    frames = list(FRAMES[bench]())
    if limit:
        frames = frames[:limit]
    n = len(frames)
    print(f"[{bench}] {n} frames", flush=True)
    progress(bench, total=n, done=0, state="running")

    t0 = time.time()
    psnrs, done = [], 0
    for i, (fid, lq01, gt_u8) in enumerate(frames):
        p = os.path.join(out_dir, fid + ".npy")
        if os.path.exists(p):
            y = np.load(p)
        else:
            y = run_one(net, torch, F, lq01)
            assert y.shape == (3,) + lq01.shape[:2], (fid, y.shape, lq01.shape)
            np.save(p, y)
        done += 1
        psnrs.append(psnr_indep(u8(y.transpose(1, 2, 0)), gt_u8))
        if (i + 1) % 25 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"  {i+1}/{n}  psnr_mean={np.mean(psnrs):.3f}  "
                  f"{el:.1f}s  ({el/(i+1):.2f}s/img)", flush=True)
            progress(bench, done=done, psnr_running=float(np.mean(psnrs)),
                     elapsed_s=round(el, 1), state="running")
    progress(bench, done=done, psnr_running=float(np.mean(psnrs)),
             elapsed_s=round(time.time() - t0, 1), state="done")
    print(f"[{bench}] done, psnr_round mean = {np.mean(psnrs):.4f}", flush=True)


# ---------------- 채점 (캐시 → PSNR) ----------------
def score(bench):
    out_dir = os.path.join(OUT_ROOT, bench)
    rows = []
    for fid, lq01, gt_u8 in FRAMES[bench]():
        p = os.path.join(out_dir, fid + ".npy")
        if not os.path.exists(p):
            continue
        y = np.load(p).astype(np.float64).transpose(1, 2, 0)
        yr = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)      # 반올림
        yt = np.clip(np.floor(y * 255.0), 0, 255).astype(np.uint8)     # 버림
        rows.append(dict(id=fid,
                         psnr_round=psnr_indep(yr, gt_u8),
                         psnr_trunc=psnr_indep(yt, gt_u8),
                         psnr_round_gtmean=psnr_indep(gt_mean_rectify(yr, gt_u8).astype(np.float64),
                                                      gt_u8.astype(np.float64)),
                         ssim_round=ssim_indep(yr, gt_u8)))
    agg = {k: float(np.mean([r[k] for r in rows]))
           for k in ("psnr_round", "psnr_trunc", "psnr_round_gtmean", "ssim_round")}
    agg["n"] = len(rows)
    agg["psnr_round_sd"] = float(np.std([r["psnr_round"] for r in rows]))
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["LOL", "Sony", "score"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd in ("LOL", "Sony"):
        infer(a.cmd, a.limit)
    else:
        out = {}
        for b in ("LOL", "Sony"):
            if os.path.isdir(os.path.join(OUT_ROOT, b)):
                agg, rows = score(b)
                out[b] = dict(agg=agg, per_image=rows)
                print(b, json.dumps(agg, indent=1))
        json.dump(out, open("numbers/newbase_lightendiff.json", "w"),
                  indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
