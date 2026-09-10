"""목적함수 3팔(l1 대조 / gainfac / lowfreq) 독립 비교.

생산자 RESULT.json 의 수치를 믿지 않고 저장된 예측 PNG 와 GT 에서 지표를 다시 계산한 뒤
생산자 값과 대조한다. 부트스트랩 단위는 GT 파일 해시 그룹이다.
"""
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gauss11():
    # cv2.getGaussianKernel(11,1.5) 와 같은 정의를 직접 구성
    x = np.arange(11) - 5.0
    k = np.exp(-(x**2) / (2 * 1.5**2))
    return (k / k.sum()).astype(np.float64)

K = gauss11()

def blur_valid(a):
    """11탭 분리 가우시안, valid 영역만 남김 (경계 패딩 불필요)."""
    from scipy.ndimage import correlate1d
    b = correlate1d(a, K, axis=0, mode='constant')
    b = correlate1d(b, K, axis=1, mode='constant')
    return b[5:-5, 5:-5, :]

def metrics(pred, gt):
    p = pred.astype(np.float64) / 255.0
    g = gt.astype(np.float64) / 255.0
    mse = float(np.sum((p - g) ** 2, dtype=np.float64) / p.size)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float('inf')
    mp, mg = blur_valid(p), blur_valid(g)
    vp = blur_valid(p * p) - mp * mp
    vg = blur_valid(g * g) - mg * mg
    cov = blur_valid(p * g) - mp * mg
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mp * mg + c1) * (2 * cov + c2)) / ((mp * mp + mg * mg + c1) * (vp + vg + c2))
    return psnr, float(s.mean()), mse

def load_arm(run_dir, step, data, gt_root):
    rd = Path(run_dir) / f'dev_step{step:06d}'
    res = json.loads((rd / 'RESULT.json').read_text())
    assert res['step'] == step
    rows = {}
    for r in res['rows']:
        png = rd / 'predictions' / (r['id'] + '.png')
        if not png.exists():
            png = rd / (r['id'] + '.png')
        assert png.exists(), f'예측 PNG 없음: {png}'
        assert sha(png) == r['prediction_sha256'], f'예측 해시 불일치: {r["id"]}'
        gt = np.asarray(Image.open(gt_root / data[r['id']]['gt_rel']).convert('RGB'))
        assert sha(gt_root / data[r['id']]['gt_rel']) == r['gt_sha256']
        pr = np.asarray(Image.open(png).convert('RGB'))
        psnr, ssim, mse = metrics(pr, gt)
        rows[r['id']] = dict(psnr=psnr, ssim=ssim, mse=mse, gt_sha=r['gt_sha256'],
                             producer=dict(psnr=r['psnr'], ssim=r['ssim'], mse=r['mse']))
    return rows, res['means']

def boot(groups, deltas, counts, repeats=20000, seed=260909):
    rng = np.random.default_rng(seed)
    g = np.array(groups)
    out = np.empty(repeats)
    idx = np.arange(len(g))
    for i in range(repeats):
        s = rng.choice(idx, len(idx), replace=True)
        out[i] = deltas[s].sum() / counts[s].sum()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

def paired(a, b):
    d = a - b
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float('inf')
    from math import erf
    # 정규근사가 아니라 t 분포 양측 p (scipy 없이)
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
        pos = int((d > 0).sum())
        sign = float(stats.binomtest(pos, n, 0.5).pvalue)
    except Exception:
        p = float('nan'); pos = int((d > 0).sum()); sign = float('nan')
    return dict(mean=float(m), t=float(t), p=p, positive=pos, n=n, sign_p=sign)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/work/HSL/lowlight_gainfac_260909')
    ap.add_argument('--step', type=int, default=250000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = Path(a.root)
    D = json.loads((R / 'study/DATA.json').read_text())
    gt_root = Path(D['data_root'])
    if not gt_root.exists():
        gt_root = R / 'data'
    index = {}
    for k in ('fit', 'dev'):
        for r in D.get(k, []):
            index[r['id']] = dict(gt_rel=r['gt'], input_rel=r['input'])
    arms = {'l1': R / 'runs_l1/plain', 'gainfac': R / 'runs/gainfac', 'lowfreq': R / 'runs_lf/lowfreq'}
    loaded, means, maxerr = {}, {}, 0.0
    for name, d in arms.items():
        rows, mn = load_arm(d, a.step, index, gt_root)
        loaded[name] = rows; means[name] = mn
        for r in rows.values():
            for k in ('psnr', 'ssim', 'mse'):
                maxerr = max(maxerr, abs(r[k] - r['producer'][k]))
    ids = sorted(set.intersection(*[set(v) for v in loaded.values()]))
    report = dict(step=a.step, frames=len(ids), producer_means=means,
                  max_abs_error_vs_producer=maxerr, comparisons={})
    ctrl = loaded['l1']
    for cand in ('gainfac', 'lowfreq'):
        cur = loaded[cand]
        dp = np.array([cur[i]['psnr'] - ctrl[i]['psnr'] for i in ids])
        ds = np.array([cur[i]['ssim'] - ctrl[i]['ssim'] for i in ids])
        gkeys = sorted({ctrl[i]['gt_sha'] for i in ids})
        gidx = {k: j for j, k in enumerate(gkeys)}
        sums = np.zeros(len(gkeys)); cnts = np.zeros(len(gkeys))
        for i, v in zip(ids, dp):
            sums[gidx[ctrl[i]['gt_sha']]] += v; cnts[gidx[ctrl[i]['gt_sha']]] += 1
        lo, hi = boot(gkeys, sums, cnts)
        gm = sums / np.maximum(cnts, 1)
        report['comparisons'][cand] = dict(
            psnr_mean_delta=float(dp.mean()), ssim_mean_delta=float(ds.mean()),
            improved_psnr=int((dp > 0).sum()), improved_ssim=int((ds > 0).sum()),
            group_bootstrap_CI95=[lo, hi], groups=len(gkeys),
            frame_level=paired(np.array([cur[i]['psnr'] for i in ids]), np.array([ctrl[i]['psnr'] for i in ids])),
            group_level=paired(gm + np.zeros(len(gm)), np.zeros(len(gm))),
            candidate_psnr=float(np.mean([cur[i]['psnr'] for i in ids])),
            control_psnr=float(np.mean([ctrl[i]['psnr'] for i in ids])),
            candidate_ssim=float(np.mean([cur[i]['ssim'] for i in ids])),
            control_ssim=float(np.mean([ctrl[i]['ssim'] for i in ids])))
    ps = [(k, v['group_level']['p']) for k, v in report['comparisons'].items()]
    ps.sort(key=lambda x: x[1])
    m = len(ps)
    holm = {}
    prev = 0.0
    for j, (k, p) in enumerate(ps):
        adj = max(prev, min(1.0, (m - j) * p)); holm[k] = adj; prev = adj
    report['holm_adjusted_group_p'] = holm
    gate = json.loads((R / 'PREDECLARATION.json').read_text())['gate']
    report['gate'] = gate
    report['gate_result'] = {k: dict(
        psnr_gain_ok=v['psnr_mean_delta'] >= gate['psnr_gain_min_db'],
        ssim_ok=v['ssim_mean_delta'] >= 0,
        ci_above_zero=v['group_bootstrap_CI95'][0] > 0,
        holm_ok=holm[k] < 0.05,
        psnr_gain_ok_strict_020=v['psnr_mean_delta'] >= 0.20,
        passed=(v['psnr_mean_delta'] >= gate['psnr_gain_min_db'] and v['ssim_mean_delta'] >= 0
                and v['group_bootstrap_CI95'][0] > 0 and holm[k] < 0.05))
        for k, v in report['comparisons'].items()}
    # 정성 패널: 후보-대조 PSNR 차 기준 최악/중앙/최고 3장, 실제 픽셀만 사용
    figdir = Path(a.out).parent / 'qualitative'
    figdir.mkdir(parents=True, exist_ok=True)
    panels = []
    for cand in ('gainfac', 'lowfreq'):
        cur = loaded[cand]
        order = sorted(ids, key=lambda i: (cur[i]['psnr'] - ctrl[i]['psnr'], i))
        picks = [('worst', order[0]), ('median', order[len(order) // 2]), ('best', order[-1])]
        for label, fid in picks:
            tiles = []
            for src, tag in ((gt_root / index[fid]['input_rel'], 'input'),
                             (arms['l1'] / f'dev_step{a.step:06d}' / (fid + '.png'), 'l1'),
                             (arms[cand] / f'dev_step{a.step:06d}' / (fid + '.png'), cand),
                             (gt_root / index[fid]['gt_rel'], 'reference')):
                tiles.append((np.asarray(Image.open(src).convert('RGB')), tag, sha(src)))
            h = max(t[0].shape[0] for t in tiles)
            canvas = np.zeros((h, sum(t[0].shape[1] for t in tiles) + 3 * 8, 3), np.uint8)
            x = 0
            for arr, tag, _ in tiles:
                canvas[:arr.shape[0], x:x + arr.shape[1]] = arr
                x += arr.shape[1] + 8
            out_png = figdir / f'{cand}_{label}_{fid}.png'
            Image.fromarray(canvas).save(out_png)
            reopened = np.asarray(Image.open(out_png).convert('RGB'))
            x = 0
            for arr, tag, _ in tiles:
                assert np.array_equal(reopened[:arr.shape[0], x:x + arr.shape[1]], arr), '패널 픽셀 검증 실패'
                x += arr.shape[1] + 8
            panels.append(dict(path=str(out_png), candidate=cand, selection=label, id=fid,
                               delta_psnr=float(cur[fid]['psnr'] - ctrl[fid]['psnr']),
                               order='input | l1 control | candidate | reference',
                               sources=[dict(tag=t[1], sha256=t[2]) for t in tiles],
                               pixels_exact_after_reopen=True))
    report['qualitative'] = panels
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    for k, v in report['comparisons'].items():
        g = report['gate_result'][k]
        print(f"{k:8s} PSNR {v['control_psnr']:.4f} -> {v['candidate_psnr']:.4f} "
              f"({v['psnr_mean_delta']:+.4f} dB, CI [{v['group_bootstrap_CI95'][0]:+.3f}, {v['group_bootstrap_CI95'][1]:+.3f}], "
              f"Holm {holm[k]:.4f}) SSIM {v['ssim_mean_delta']:+.5f} | 게이트(0.15) {'통과' if g['passed'] else '탈락'}"
              f" | 0.20 기준 {'충족' if g['psnr_gain_ok_strict_020'] else '미달'}")
    print(f"생산자 값과 최대 절대오차 {maxerr:.3e}")

main()
