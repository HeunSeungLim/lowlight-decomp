"""우리 방법(하이라이트 정합 장면 보정) 수치를 method_260909 의 실측 json 에서 매크로로 굽는다.
numbers_method.tex 와 tab_cal_rows.tex 를 만들고, 감사가 대조할 method_evidence.json 을 함께 남긴다."""
import json, os
P = os.path.dirname(os.path.abspath(__file__))
def _method_dir_common(P):
    for c in (os.environ.get("LLMETHOD"),
              os.path.join(P, "..", "method_260909"),
              os.path.join(P, "..", "numbers", "method_260909"),
              "/home/user/lowlight_paper/method_260909"):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
M = _method_dir_common(P)
S = json.load(open(f"{M}/safe_calib.json")); C = json.load(open(f"{M}/control_stats.json"))
F = json.load(open(f"{M}/final_calib.json")); B = json.load(open(f"{M}/stack_burst.json"))
Q = json.load(open(f"{M}/qualitative.json")); T = json.load(open(f"{M}/transfer.json"))
NAME = {"retinexformer": "Retinexformer", "snrnet": "SNR-Net", "lightendiff": "LightenDiffusion", "zerodcepp": "Zero-DCE++"}
mac, ev = {}, {}
for k, n in NAME.items():
    s, c = S[k], C[k]
    tag = {"retinexformer": "Ret", "snrnet": "Snr", "lightendiff": "Ld", "zerodcepp": "Zdce"}[k]
    mac[f"nCAL{tag}base"] = f"{s['base']:.2f}"; mac[f"nCAL{tag}new"] = f"{s['base']+s['gain']:.2f}"
    mac[f"nCAL{tag}gain"] = f"{s['gain']:+.2f}".replace("+", "")
    mac[f"nCAL{tag}lo"] = f"{s['ci'][0]:+.2f}"; mac[f"nCAL{tag}hi"] = f"{s['ci'][1]:+.2f}"
    mac[f"nCAL{tag}imp"] = f"{s['improved']:d}"
    mac[f"nCAL{tag}ssold"] = f"{s['ssim0']:.3f}"; mac[f"nCAL{tag}ss"] = f"{s['ssim']:.3f}"
    mac[f"nCAL{tag}ctl"] = f"{c['control_const']:+.2f}"; mac[f"nCAL{tag}dif"] = f"{c['diff']:+.2f}"
    mac[f"nCAL{tag}p"] = f"{c['p_raw']:.3f}"; mac[f"nCAL{tag}ph"] = f"{c['p_holm']:.3f}"
    ev[k] = dict(base=s["base"], new=s["base"]+s["gain"], gain=s["gain"], ci=s["ci"], ssim0=s["ssim0"], ssim=s["ssim"],
                 control=c["control_const"], diff=c["diff"], p=c["p_raw"], p_holm=c["p_holm"], improved=s["improved"],
                 unsafe_gain=T[k]["보정"]["gain"])
for n, key in ((10, "10_장면단위"), (20, "20_장면단위"), (25, "25_장면단위"), (40, "40_장면단위")):
    mac["nDOSE" + {10:"ten",20:"twenty",25:"twentyfive",40:"forty"}[n]] = f"{F[key]['gain']:+.2f}".replace("+", ""); ev[f"dose{n}"] = F[key]["gain"]
mac["nBURSTsingle"] = f"{B['single']:.2f}"; mac["nBURSTavg"] = f"{B['burst']:.2f}"; mac["nBURSTcal"] = f"{B['burst_calib']:.2f}"
mac["nBURSTgain"] = f"{B['calib_gain']:+.2f}".replace("+", ""); mac["nBURSTlo"] = f"{B['ci'][0]:+.2f}"; mac["nBURSThi"] = f"{B['ci'][1]:+.2f}"
mac["nBURSTn"] = f"{B['n_groups']:d}"
mac["nCALbest"] = f"{Q['scene_gain'][Q['picked'][0]]:+.2f}"; mac["nCALworst"] = f"{Q['scene_gain'][Q['picked'][2]]:+.2f}"
ev["burst"] = B; ev["qual"] = {k: Q["scene_gain"][k] for k in Q["picked"]}
mac["nCALncal"] = "40"; mac["nCALnfold"] = "5"
open(os.path.join(P, "numbers_method.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
with open(os.path.join(P, "tab_cal_rows.tex"), "w") as t:
    for k, n in NAME.items():
        s, c = S[k], C[k]
        t.write(f"{n} & {s['base']:.2f} & {s['base']+s['gain']:.2f} & {s['gain']:+.2f} & [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] & "
                f"{c['control_const']:+.2f} \\\\\n")
print(f"매크로 {len(mac)}개, 표 4행 생성")
# ---- 진단 격차·제거실험·그림 수치까지 한 곳에서 굽는다 (재생성 시 유실 방지)
import numpy as np, collections
MM = M
tz = np.load(f"{MM}/train_feats.npz", allow_pickle=True); zz = np.load(f"{MM}/calib2_cache.npz", allow_pickle=True)
A_tr, A_te, QG, SS = tz["A"], zz["A"], zz["QG"], zz["S"]; QL = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
_rows = json.load(open("/home/user/lowlight_paper/repro/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
_g = collections.defaultdict(list)
for _r, _a in zip(_rows, A_te): _g[(_r["scene"], _r["exp"])].append(_a)
within = float(np.median([np.std(v, ddof=1) for v in _g.values() if len(v) >= 3])); cvv = QG.std(0) / QG.mean(0)
SM = json.load(open(f"{MM}/stat_match.json")); EF = json.load(open(f"{MM}/est_family.json"))
LR = json.load(open(f"{MM}/lol_replicate.json")); HC = json.load(open(f"{MM}/highlight_calib.json"))
QJ = json.load(open(f"{MM}/qualitative.json")); sg = np.array([QJ["scene_gain"][k] for k in sorted(QJ["scene_gain"])])
fig_gain = float(QJ["scene_gain_pred"]["10198"]); fig_delta = float(json.load(open(os.path.join(P, "fig_cmp2_numbers.json")))["calibrated_psnr"]
                                                                   - json.load(open(os.path.join(P, "fig_cmp2_numbers.json")))["retinexformer_psnr"])
mac2 = {"nAGENtrain": f"{A_tr.std(ddof=1):.3f}", "nAGENtest": f"{A_te.std(ddof=1):.3f}", "nAGENwithin": f"{within:.4f}",
        "nAGENcv": f"{cvv[QL.index(99)]:.3f}", "nAGENcvmean": f"{QG[:, QL.index(50)].std() / QG[:, QL.index(50)].mean():.3f}",
        "nDOSEn": "10, 20, 25 and 40", "nCALSnrunsafe": f"{abs(T['snrnet']['보정']['gain']):.2f}",
        "nCALSnrimp": f"{S['snrnet']['improved']}", "nCALLdimp": f"{S['lightendiff']['improved']}", "nCALZdceimp": f"{S['zerodcepp']['improved']}",
        "nCALmedian": f"{np.median(sg):+.2f}", "nCALworse": f"{int((sg < 0).sum())}", "nCALworseHalf": f"{int((sg < -0.5).sum())}",
        "nCALtopfive": f"{np.sort(sg)[-5:].sum() / sg.sum() * 100:.0f}",
        "nABLplain": f"{SM['p99']['gain']:+.2f}", "nABLplainLo": f"{SM['p99']['ci'][0]:+.2f}", "nABLplainHi": f"{SM['p99']['ci'][1]:+.2f}",
        "nABLinvar": f"{HC['res']['전역 하이라이트 정합']['gain']:+.2f}", "nABLinvarLo": f"{HC['res']['전역 하이라이트 정합']['ci'][0]:+.2f}",
        "nABLinvarHi": f"{HC['res']['전역 하이라이트 정합']['ci'][1]:+.2f}",
        "nABLwide": f"{EF['retinexformer']['method']:+.2f}", "nABLwideP": f"{EF['retinexformer']['p_holm']:.2f}",
        "nLOLharm": f"{LR['gsad']['gain']:+.2f}", "nCALfiggain": f"{fig_gain:.3f}", "nCALfigdelta": f"{fig_delta:+.1f}"}
with open(os.path.join(P, "numbers_method.tex"), "a") as t:
    t.write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac2.items()))
ev["diag"] = dict(agen_train=float(A_tr.std(ddof=1)), agen_test=float(A_te.std(ddof=1)), within=within,
                  cv99=float(cvv[QL.index(99)]), cv50=float(QG[:, QL.index(50)].std() / QG[:, QL.index(50)].mean()))
ev["ablation"] = dict(plain=SM["p99"], invariance=HC["res"]["전역 하이라이트 정합"], wide=EF["retinexformer"], lol_harm=LR["gsad"],
                      median=float(np.median(sg)), worse=int((sg < 0).sum()), worse_half=int((sg < -0.5).sum()),
                      top5_share=float(np.sort(sg)[-5:].sum() / sg.sum() * 100), fig_gain=fig_gain, fig_delta=fig_delta)
json.dump(ev, open(os.path.join(P, "method_evidence.json"), "w"), indent=1)
print(f"보강 매크로 {len(mac2)}개")

