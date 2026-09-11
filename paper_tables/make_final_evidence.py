"""최종 규칙(q=99.9, K=1)의 모든 인쇄값을 한 파일에서 굽는다. 중복 정의 없이 numbers_final.tex 만 만든다."""
import re, json, os, numpy as np
P=os.path.dirname(os.path.abspath(__file__))
def _method_dir_common(P):
    for c in (os.environ.get("LLMETHOD"),
              os.path.join(P, "..", "method_260909"),
              os.path.join(P, "..", "numbers", "method_260909"),
              ):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
M=_method_dir_common(P)
F=json.load(open(f"{M}/final_rule.json")); QS=json.load(open(f"{M}/qsweep.json"))
A=json.load(open(f"{M}/anchor.json")); S49=json.load(open(f"{M}/safe_calib.json"))
tz=np.load(f"{M}/train_feats.npz",allow_pickle=True); QLt=[50,75,90,95,98,99,99.5,99.9]
TAG={"retinexformer":"Ret","snrnet":"Snr","lightendiff":"Ld","zerodcepp":"Zdce"}
NAME={"retinexformer":"Retinexformer","snrnet":"SNR-Net","lightendiff":"LightenDiffusion","zerodcepp":"Zero-DCE++"}
mac={"nFQ":"99.9","nFK":"1.00","nFsatS":f"{F['sat_train_sony']:.0f}","nFsatL":f"{F['sat_train_lol']:.0f}","nFKL":f"{F['K_lol']:.2f}"}
rows=[]
for k,n in NAME.items():
    v=F["sony"][k]; t=TAG[k]
    mac.update({f"nF{t}base":f"{v['base']:.2f}", f"nF{t}new":f"{v['base']+v['gain']:.2f}", f"nF{t}gain":f"{v['gain']:+.3f}",
                f"nF{t}lo":f"{v['ci'][0]:+.3f}", f"nF{t}hi":f"{v['ci'][1]:+.3f}", f"nF{t}p":f"{v['p_raw']:.4f}",
                f"nF{t}imp":f"{v['improved']}", f"nF{t}worst":f"{v['worst']:+.2f}", f"nF{t}med":f"{v['median']:+.3f}",
                f"nF{t}ssa":f"{v['ssim0']:.3f}", f"nF{t}ssb":f"{v['ssim']:.3f}"})
    rows.append(f"{n} & {v['base']:.2f} & {v['base']+v['gain']:.2f} & {v['gain']:+.3f} & [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] & {v['p_raw']:.4f} & {v['improved']}/50 \\\\")
r=F["sony"]["retinexformer"]
mac.update({"nFsatlo":f"{r['sat_lo']:+.2f}","nFsathi":f"{r['sat_hi']:+.2f}","nFtopfive":f"{r['top5']:.0f}"})
b=F["burst"]; mac.update({"nFbavg":f"{b['avg']:.2f}","nFbanc":f"{b['avg_anc']:.2f}","nFbgain":f"{b['gain']:+.2f}","nFbn":f"{b['n']}"})
h=F["holm"]; mac["nFholmM"]=f"{h['m']}"
pos=[x for x in h["table"] if x["p_holm"]<0.05 and x["gain"]>0]; neg=[x for x in h["table"] if x["p_holm"]<0.05 and x["gain"]<0]
mac["nFholmPos"]=f"{len(pos)}"; mac["nFholmNeg"]=f"{len(neg)}"
mac["nFholmRet"]=f"{[x for x in h['table'] if x['pair']=='Sony/retinexformer'][0]['p_holm']:.3f}"
qq=QS["models"]["retinexformer"]["q"]
QR=json.load(open(f"{M}/qsweep_rule.json"))["rule"]
for q,tag in ((95,"a"),(98,"b"),(99,"c"),(99.5,"d"),(99.95,"e"),(99.99,"f")):
    mac[f"nFq{tag}"]=f"{QR[str(q)]['gain']:+.2f}"; mac[f"nFq{tag}K"]=f"{q}"
for q,tag in ((95,"a"),(98,"b"),(99,"c"),(99.5,"d")):
    pass   # 옛 161장면 포화율은 쓰지 않는다. 아래에서 qsat_train181 로 굽는다.
FX=json.load(open(f"{M}/fix_v52.json"))
mac["nFqdGain"]=f"{FX['regression_variant']['gain']:+.2f}"; mac["nFqdWorst"]=f"{FX['regression_variant']['worst']:+.2f}"
for q,t in ((99.95,"e"),(99.99,"f")):
    mac["nFptt"]=f"{FX['q']['99.9']['p_t']:.3f}"; mac["nFpwil"]=f"{FX['q']['99.9']['p_wilcoxon']:.3f}"
mac["nFtopone"]=f"{FX['q']['99.9']['top1']:.0f}"; mac["nFtopfiveB"]=f"{FX['q']['99.9']['top5']:.0f}"
mac["nFsatLall"]=f"{FX['lol_sat_all']:.0f}"; mac["nFlolNimg"]=f"{FX['lol_n']}"
lolrows=[]
for k,v in F["lol"].items():
    lolrows.append((NAME.get(k,k), v["gain"], v["p_raw"]))
mac["nFlolBad"]=f"{sum(1 for _,g,_ in lolrows if g<0)}"; mac["nFlolN"]=f"{len(lolrows)}"
mac["nFlolWorst"]=f"{min(g for _,g,_ in lolrows):+.2f}"; mac["nFlolBest"]=f"{max(g for _,g,_ in lolrows):+.2f}"
NP=json.load(open(f"{M}/noop.json"))
for k,t in TAG.items():
    mac[f"nF{t}act"]=f"{NP[k]['n_act']}"; mac[f"nF{t}actpct"]=f"{NP[k]['act_frames']:.0f}"; mac[f"nF{t}gact"]=f"{NP[k]['gain_acting']:+.2f}"
ev_np=NP
RA=json.load(open(f"{M}/robust_acting.json"))
a=RA["acting_frames"]
mac["nFactSc"]=f"{a['n']}"; mac["nFactGain"]=f"{a['mean']:+.2f}"; mac["nFactLo"]=f"{a['ci'][0]:+.2f}"; mac["nFactHi"]=f"{a['ci'][1]:+.2f}"
mac["nFactT"]=f"{a['t']:.3f}"; mac["nFactSign"]=f"{a['sign_two']:.3f}"; mac["nFactPos"]=f"{a['pos']}"
mac["nFdropTop"]=f"{RA['drop_top3']['mean']:+.2f}"; mac["nFdropT"]=f"{RA['drop_top3']['t']:.3f}"
mac["nFjkLo"]=f"{RA['jackknife'][0]:+.2f}"; mac["nFjkHi"]=f"{RA['jackknife'][1]:+.2f}"
mac["nFactAll"]=f"{RA['acting_scenes_allframes']['mean']:+.2f}"
mac["nFactAllFr"]=f"{RA['acting_scenes_allframes']['frame_mean']:+.2f}"   # 같은 프레임 집합의 프레임평균
mac["nFtopfiveB"]=f"{F['sony']['retinexformer']['top5']:.0f}"
mac["nFwScene"]=f"{RA['worst']['scene_all50']:+.2f}"; mac["nFwAct"]=f"{RA['worst']['scene_acting']:+.2f}"; mac["nFwFrame"]=f"{RA['worst']['frame']:+.2f}"
for e,t in (("0.033s","s"),("0.04s","m"),("0.1s","l")):
    k=f"exp_{e}"; mac[f"nFex{t}"]=f"{RA[k]['mean']:+.2f}"; mac[f"nFex{t}t"]=f"{RA[k]['t']:.3f}"
mac["nFcircRho"]=f"{RA['noncircular_spearman']['rho']:+.2f}"; mac["nFcircP"]=f"{RA['noncircular_spearman']['p']:.3f}"
SH=json.load(open(f"{M}/shares.json"))
mac["nFbestAct"]=f"{SH['best_share_acting']:.0f}"; mac["nFtopthreeAct"]=f"{SH['top3_share_acting']:.0f}"
mac["nFbestPool"]=f"{SH['best_share_pooled']:.0f}"
mac["nFfrW"]=f"{SH['frame_weighted']:+.3f}"; mac["nFscW"]=f"{SH['scene_weighted']:+.3f}"
mac["nFnoopSc"]=f"{SH['unchanged_scenes']}"
for _e,_t in (("0.033s","s"),("0.04s","m"),("0.1s","l")):
    mac[f"nFexn{_t}"]=f"{SH['exp_frames'][_e]}"; mac[f"nFexsc{_t}"]=f"{SH['exp_scenes'][_e]}"

CT=json.load(open(f"{M}/controls_v58.json"))
mac["nCconstPool"]=f"{CT['oracle_constant_pooled']['frame_mean']:+.2f}"
mac["nCconstAct"]=f"{CT['oracle_constant_acting']['frame_mean']:+.2f}"
mac["nCconstActSc"]=f"{CT['oracle_constant_acting']['scene_mean']:+.2f}"
mac["nCconstActLo"]=f"{CT['oracle_constant_acting']['scene_ci'][0]:+.2f}"
mac["nCconstActHi"]=f"{CT['oracle_constant_acting']['scene_ci'][1]:+.2f}"
mac["nCanchActFr"]=f"{CT['anchor_acting']['frame_mean']:+.2f}"
mac["nCperm"]=f"{CT['permutation_acting']['mean']:+.2f}"
mac["nCpermN"]=f"{CT['permutation_acting']['repeats']}"
mac["nCrecPool"]=f"{CT['anchor_recovery_pooled']:.0f}"
mac["nCrecAct"]=f"{CT['anchor_recovery_acting']:.0f}"
mac["nCoraPool"]=f"{CT['oracle_perframe_pooled']['frame_mean']:.2f}"
mac["nCoraAct"]=f"{CT['oracle_perframe_acting']['frame_mean']:.2f}"
_pd=CT["paired_anchor_minus_constant_acting"]
mac["nCpair"]=f"{_pd['scene_mean']:+.2f}"; mac["nCpairLo"]=f"{_pd['scene_ci'][0]:+.2f}"
mac["nCpairHi"]=f"{_pd['scene_ci'][1]:+.2f}"; mac["nCpairPos"]=f"{_pd['positive']}"

EF=json.load(open(f"{M}/exposure_frame_means.json"))
for _e,_t in (("0.033s","s"),("0.04s","m"),("0.1s","l")):
    mac[f"nFexf{_t}"]=f"{EF['exposures'][_e]['frame_mean']:+.3f}"

AF=json.load(open(f"{M}/affine_oracle.json"))
mac["nAFshare"]=f"{AF['share_affine_pct']:.1f}"; mac["nAFover"]=f"{AF['affine_minus_channel_db']:+.2f}"
mac["nAFchan"]=f"{AF['psnr_channel']:.3f}"
CN=json.load(open(f"{M}/concentration.json"))
mac["nCONone"]=f"{CN['drop_top1']['scene_mean']:+.3f}"; mac["nCONfive"]=f"{CN['drop_top5']['scene_mean']:+.3f}"
mac["nCONtopfive"]=f"{CN['top5_share_of_scene_pooled_pct']:.0f}"
# 헤드라인(프레임가중)의 집중도도 같이 인쇄한다
mac["nCONtopfiveFr"]=f"{CN['top5_share_of_frame_pooled_pct']:.0f}"   # 장면 쪽과 같은 양(직접 몫)이어야 한다
mac["nCONoneFr"]=f"{CN['drop_top1']['frame_mean']:+.3f}"; mac["nCONfiveFr"]=f"{CN['drop_top5']['frame_mean']:+.3f}"
TO=json.load(open(f"{M}/tost_equivalence.json"))
mac["nCTOST"]=f"{TO['tost_margin']:.2f}"
mac["nFZdcegain"]=f"{F['sony']['zerodcepp']['gain']:+.3f}"
mac["nFZdcep"]=f"{F['sony']['zerodcepp']['p_raw']:.4f}"
MFD=json.load(open(f"{M}/multiframe_drop.json"))
BS=json.load(open(f"{M}/bootstrap_config.json"))
mac["nMFdropRe"]=f"{MFD['drop_db']:.2f}"
mac["nBootB"]=f"{BS['resamples']}"; mac["nBootSeed"]=f"{BS['seed']}"
mac["nBootSpecB"]=f"{BS['spectral_bootstrap']['resamples']}"; mac["nBootSpecSeed"]=f"{BS['spectral_bootstrap']['seed']}"
SW=json.load(open(f"{M}/protocol_sweep.json"))
mac["nSWn"]=f"{SW['rectification_sweep']['pairs']}"
mac["nSWlo"]=f"{SW['rectification_sweep']['range_db'][0]:+.2f}"
mac["nSWhi"]=f"{SW['rectification_sweep']['range_db'][1]:+.2f}"
mac["nAGlo"]=f"{SW['aggregation_convention']['range_db'][0]:.2f}"
mac["nAGhi"]=f"{SW['aggregation_convention']['range_db'][1]:.2f}"
_bf=SW["rectification_sweep"]["by_family"]
mac["nSWtrHi"]=f"{_bf['paired_supervised']['range_db'][1]:+.2f}"
mac["nSWtrLo"]=f"{_bf['paired_supervised']['range_db'][0]:+.2f}"
mac["nSWtrN"]=f"{_bf['paired_supervised']['pairs']}"
LOLRF=json.load(open(f"{M}/anchor_lol_retinexformer.json"))
mac["nLOLRFloss"]=f"{LOLRF['raw']['psnr']-LOLRF['anchored']['psnr']:.1f}"   # 영수증 권고: 표의 반올림 두 행 차가 2.68 이라 한 자리로 인쇄한다
QS181=json.load(open(f"{M}/qsat_train181.json"))
for _q,_t in ((95,"a"),(98,"b"),(99,"c"),(99.5,"d")):
    mac[f"nFq{_t}sat"]=f"{QS181['saturated_pct'][str(_q)]:.0f}"
for _e,_t in (("0.033s","s"),("0.04s","m"),("0.1s","l")):
    mac[f"nFext{_t}"]=f"{RA[f'exp_{_e}']['t']:.3f}"
mac["nCTOSTpct"]=f"{TO['margin_as_pct_of_effect']:.0f}"
mac["nQSATn"]=f"{QS181['scenes']}"

mac = {k: ("$-$"+v[1:] if isinstance(v, str) and v.startswith("-") else v) for k, v in mac.items()}   # 줄바꿈 분리 방지
open(os.path.join(P,"numbers_final.tex"),"w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in mac.items()))
rows = [re.sub(r"(?<![A-Za-z])-(?=\d)", "$-$", r) for r in rows]   # 숫자 앞 음수만 수식으로
open(os.path.join(P,"tab_fin_rows.tex"),"w").write("\n".join(rows)+"\n")
json.dump(dict(final=F, qsweep=QS, qrule=QR, noop=ev_np, robust=RA,
               measured_printed={"anchor_after": {k: round(v["base"]+v["gain"], 2) for k, v in F["sony"].items()}},
               top1=RA.get("top1"), q_labels=[95, 98, 99, 99.5, 99.9]), open(os.path.join(P,"final_evidence.json"),"w"), indent=1)
assert len(set(mac)) == len(mac)
print(f"매크로 {len(mac)}개(중복 0), 표 {len(rows)}행")
