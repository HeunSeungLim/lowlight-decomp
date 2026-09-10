"""Single source of truth for every number the paper prints.

Pulls from the four measurement dumps in repro/ and writes paper/evidence.json plus the
LaTeX row files the manuscript reads via input. Nothing in the .tex is typed by hand, so a
re-measurement propagates into the paper instead of silently drifting from it.
"""
import json, os
import numpy as np
P = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("LOWLIGHT_EVIDENCE_ROOT", os.path.join(P, "..", "numbers") if os.path.isdir(os.path.join(P, "..", "numbers")) else os.path.join(P, "..", "repro"))
L = lambda n: json.load(open(os.path.join(R, n + ".json")))
fail, spec, ident, low = (L("diag_sid_failure"), L("diag_sid_spectrum"),
                          L("diag_sid_identifiability"), L("diag_sid_lowfreq"))
E, o, lo = {}, fail["overall"], low["overall"]
EXPS = ["0.033s", "0.04s", "0.1s"]

# ---------------- anchors ----------------
E["n_images"], E["n_scenes"] = o["n"], fail["n_scenes"]
E["anchors"] = dict(model=o["psnr_model"], ssim=o["ssim_model"],
                    gain=o["psnr_model_gainfix"], chan=o["psnr_model_chanfix"])
E["share"] = dict(glob=o["share_global_pct"], chan=o["share_chan_pct"], resid=o["share_resid_pct"])
E["by_exposure"] = {k: dict(n=v["n"], psnr=v["psnr_model"], glob=v["share_global_pct"],
                            chan=v["share_chan_pct"], resid=v["share_resid_pct"])
                    for k, v in fail["by_exposure"].items()}
E["trivial"] = dict(gain=o["psnr_base_gain"], gain_ssim=o["ssim_base_gain"])

# ---------------- spectrum ----------------
E["rad_edges"] = spec["rad_edges"]
E["power_ratio"] = spec["overall"]["spec"]["ratio_energy"]

# ---------------- band-wise correlation: model vs input ----------------
E["ident_edges"] = ident["rad_edges"]
E["rho"] = {}
for e in EXPS:
    row = {}
    if e in ident["model"]:
        row["model"] = ident["model"][e]["rho_luma"]
    for tag, k in (("in_k1", f"{e}|pw|1|all"), ("in_k2", f"{e}|pw|2|all"),
                   ("in_k4", f"{e}|pw|4|all"), ("in_k8", f"{e}|pw|8|all")):
        if k in ident["input"]:
            row[tag] = ident["input"][k]["rho_luma"]
    E["rho"][e] = row

# ---------------- spatially varying low-frequency component ----------------
E["lf"] = dict(energy_share=lo["lf_energy_share_rgb_ew"], const_share=lo["lf_const_share"],
               std_over_meanabs=lo["lf_std_over_meanabs"], ac_h=lo["blk_ac_h"],
               ac_v=lo["blk_ac_v"], sign_h=lo["blk_signagree_h"], sign_v=lo["blk_signagree_v"])
# Each family is anchored on its own global(1x1) fit, and the gain a family buys is
# reported after subtracting the same-family white-target control, which absorbs the
# free improvement that simply comes from having more blocks to fit.
E["ladder"] = []
for b in low["block_names"]:
    r = dict(block=b)
    for fam, tag in (("gain", "gain"), ("aff", "affine")):
        a0 = lo[f"psnrf_{fam}_global(1x1)"]; w0 = lo[f"psnrf_white_{fam}_global(1x1)"]
        a = lo[f"psnrf_{fam}_{b}"]; w = lo[f"psnrf_white_{fam}_{b}"]
        r[tag] = a
        r[tag + "_raw"] = a - a0
        r[tag + "_dof"] = (a - a0) - (w - w0)
    E["ladder"].append(r)
E["r2"] = {k: v for k, v in low["regression"]["r2_centered"].items() if "@" not in k}
E["r2_by_exp"] = {k: v for k, v in low["regression"]["r2_centered"].items() if "@" in k}

json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)

# ---------------- LaTeX rows ----------------
f = lambda x, d=2: ("--" if x is None else f"{x:.{d}f}")

# Table 1: where the residual is, and what an oracle correction can buy
with open(os.path.join(P, "tab_decomp_rows.tex"), "w") as t:
    for e in EXPS:
        v = E["by_exposure"][e]
        t.write(f"{e} & {v['n']} & {f(v['psnr'])} & {f(v['glob'],1)} & {f(v['chan'],1)} "
                f"& {f(v['resid'],1)} \\\\\n")
    s = E["share"]
    t.write(f"\\midrule\nall & {E['n_images']} & {f(E['anchors']['model'])} & {f(s['glob'],1)} "
            f"& {f(s['chan'],1)} & {f(s['resid'],1)} \\\\\n")

# Table 2: the oracle ladder, degrees-of-freedom controlled
with open(os.path.join(P, "tab_ladder_rows.tex"), "w") as t:
    for r in E["ladder"]:
        if r["block"] in ("256", "128"):
            continue                              # trend is visible from 64/32/16; page budget
        name = r["block"].replace("global(1x1)", "global")
        t.write(f"{name} & {f(r['gain'])} & {f(r['gain_dof'],3)} & {f(r['affine'])} "
                f"& {f(r['affine_dof'],3)} \\\\\n")

# Table 3: is the high-frequency information there at all
with open(os.path.join(P, "tab_rho_rows.tex"), "w") as t:
    ed = E["ident_edges"]
    for i in range(len(ed) - 1):
        cells = []
        for e in EXPS:
            r = E["rho"][e]
            cells += [f(r["model"][i], 3) if "model" in r else "--",
                      f(r["in_k1"][i], 3) if "in_k1" in r else "--"]
        t.write(f"{ed[i]:.2f}--{ed[i+1]:.2f} & " + " & ".join(cells) + " \\\\\n")

print("wrote evidence.json + 3 row files")
print("anchors:", {k: round(v, 4) for k, v in E["anchors"].items()})
print("ladder :", [(r["block"], f(r["gain"]), f(r["gain_dof"],3), f(r["affine_dof"],3)) for r in E["ladder"]])
print("r2     :", {k: round(v, 4) for k, v in E["r2"].items()})
print("rho k available:", [k for k in E["rho"]["0.1s"] if k != "model"])

# ---------------- prose macros: every number in the text comes from here ----------------
so2 = spec["overall"]
M = {
    "NIMG": f"{E['n_images']}", "NSCENE": f"{E['n_scenes']}",
    "PSNRmodel": f"{E['anchors']['model']:.3f}", "SSIMmodel": f"{E['anchors']['ssim']:.3f}",
    "PSNRgain": f"{E['anchors']['gain']:.3f}", "PSNRchan": f"{E['anchors']['chan']:.3f}",
    "SHAREglob": f"{E['share']['glob']:.1f}", "SHAREchan": f"{E['share']['chan']:.1f}",
    "SHAREresid": f"{E['share']['resid']:.1f}",
    "TRIVgain": f"{E['trivial']['gain']:.2f}",
    "BLURsigma": f"{so2['best_sigma_gainfix']}",
    "BLURpsnr": f"{so2['best_psnr_gainfix']:.3f}",
    "BLURdelta": f"{so2['best_psnr_gainfix'] - so2['psnr_model_gainfix']:.3f}",
    "AMPpsnr": f"{so2['psnr_ampmatch_64']:.3f}",
    "AMPdelta": f"{so2['psnr_ampmatch_64'] - so2['psnr_model_gainfix']:.3f}",
    "BANDLSdelta": f"{so2['psnr_bandls_64'] - so2['psnr_model_gainfix']:.3f}",
    "POWlow": f"{E['power_ratio'][0]:.3f}", "POWhigh": f"{E['power_ratio'][-1]:.3f}",
    "LFshare": f"{100*E['lf']['energy_share']:.1f}",
    "LFconst": f"{100*E['lf']['const_share']:.1f}",
    "LFstdratio": f"{E['lf']['std_over_meanabs']:.2f}",
    "LFach": f"{E['lf']['ac_h'][0]:.3f}", "LFacv": f"{E['lf']['ac_v'][0]:.3f}",
    "LFsignh": f"{E['lf']['sign_h'][0]:.2f}", "LFsignv": f"{E['lf']['sign_v'][0]:.2f}",
    "LADgain": f"{E['ladder'][-1]['gain_dof']:.3f}",
    "LADaff": f"{E['ladder'][-1]['affine_dof']:.3f}",
    "RTWOin": f"{E['r2']['cinout']:.3f}", "RTWOall": f"{E['r2']['call']:.3f}",
    "RHOshortcorner": f"{E['rho']['0.033s']['model'][-1]:.3f}",
    "RHOlongcornerK": f"{E['rho']['0.1s']['in_k1'][-1]:.3f}",
    "RHOlongcornerKe": f"{E['rho']['0.1s']['in_k8'][-1]:.3f}",
    "RHOlongcornerM": f"{E['rho']['0.1s']['model'][-1]:.3f}",
}
with open(os.path.join(P, "numbers.tex"), "w") as t:
    for k, v in M.items():
        t.write("\\newcommand{\\n%s}{%s}\n" % (k, v))
print("wrote numbers.tex:", len(M), "macros")
print({k: M[k] for k in ("PSNRmodel", "BLURdelta", "AMPdelta", "LFshare", "LADgain", "LADaff",
                         "RTWOin", "RTWOall", "RHOshortcorner", "RHOlongcornerK", "RHOlongcornerKe")})

# ---------------- dual identifiability criterion + denoise bound ----------------
# The raw-input criterion and the noise-corrected ceiling criterion disagree, so the
# paper must carry both. Percentages are shares of squared error per exposure, read
# from the identifiability dump's verdict tables.
import re as _re
_ID = open(os.path.join(R, "DIAG_SID_IDENTIFIABILITY.md")).read()
_m = _re.search(r"노출\s+천장: 회복가능%.*?\n-+\n(.*?)\n```", _ID, _re.S)
DUAL = {}
if _m:
    for ln in _m.group(1).strip().split("\n"):
        p = ln.split()
        if len(p) >= 8 and p[0].endswith("s"):
            DUAL[p[0]] = dict(ceil_rec=float(p[1]), ceil_exh=float(p[2]), ceil_un=float(p[3]),
                              raw_rec=float(p[5]), raw_exh=float(p[6]), raw_un=float(p[7]))
E["dual"] = DUAL
_n = _re.search(r"전체\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*%", open(os.path.join(R, "DIAG_SID_FAILURE.md")).read())
E["noise_share_pct"] = float(_n.group(3)) if _n else None
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)

with open(os.path.join(P, "numbers.tex"), "a") as t:
    for e, tag in (("0.033s", "short"), ("0.04s", "mid"), ("0.1s", "long")):
        if e in DUAL:
            d = DUAL[e]
            t.write("\\newcommand{\\nCEILrec%s}{%.1f}\n" % (tag, d["ceil_rec"]))
            t.write("\\newcommand{\\nCEILun%s}{%.1f}\n" % (tag, d["ceil_un"]))
            t.write("\\newcommand{\\nRAWun%s}{%.1f}\n" % (tag, d["raw_un"]))
    if E["noise_share_pct"] is not None:
        t.write("\\newcommand{\\nNOISEshare}{%.1f}\n" % E["noise_share_pct"])
print("dual criterion:", json.dumps(DUAL))
print("noise share pct:", E["noise_share_pct"])

# ---------------- learned predictor for the local term ----------------
pg = L("predict_gainfield")
E["cnn_train_scenes"] = int(pg["leakage_checks"]["n_scene"]["tr"])
# per-frame spread on LOL (15 frames): sd of the three shares (points) and of the control-corrected 16-px ceiling (dB), max over the Table 1 methods
_cmj = json.load(open(os.path.join(R, "compare_methods.json")))
_pfl = _cmj["per_frame"]["LOL"]; _sdS, _sdD = {}, {}
for _k, _rows in _pfl.items():
    if _k in ("gamma", "cidnet_wperc"): continue
    _g = np.array([100 * (r["mse"] - r["mse_glob"]) / r["mse"] for r in _rows]); _c = np.array([100 * (r["mse_glob"] - r["mse_chan"]) / r["mse"] for r in _rows]); _r = np.array([100 * r["mse_chan"] / r["mse"] for r in _rows])
    _pf = lambda r, k: 10 * np.log10(255.0 ** 2 / r[k])
    _d = np.array([(_pf(r, "mse_gain_16") - _pf(r, "mse_gain_global(1x1)")) - (_pf(r, "mse_white_gain_16") - _pf(r, "mse_white_gain_global(1x1)")) for r in _rows])
    assert len(_rows) == 15, (_k, len(_rows))
    _sdS[_k] = float(max(_g.std(ddof=1), _c.std(ddof=1), _r.std(ddof=1))); _sdD[_k] = float(_d.std(ddof=1))
E["lol_frame_sd"] = dict(share_points=_sdS, d16_db=_sdD, share_max=max(_sdS.values()), d16_max=max(_sdD.values()))
V, TH = pg["verdict"], pg["config"]["thresholds"]
E["predict"] = dict(r2=V["r2"], db=V["db_gain"], db_best=V["db_gain_best"],
                    thr_r2=TH["no_r2"], thr_db=TH["no_db"],
                    oracle=pg["anchors"]["measured"]["psnr_loc16"],
                    anchor=pg["anchors"]["measured"]["psnr_gain"])
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
pr = E["predict"]
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nCNNrtwo}{%.3f}\n" % pr["r2"])
    t.write("\\newcommand{\\nCNNdb}{%.3f}\n" % pr["db"])
    t.write("\\newcommand{\\nCNNfrac}{%.1f}\n" % (100 * pr["db"] / (pr["oracle"] - pr["anchor"])))
    t.write("\\newcommand{\\nTHRrtwo}{%.1f}\n" % pr["thr_r2"])
    t.write("\\newcommand{\\nTHRdb}{%.2f}\n" % pr["thr_db"])
print("predictor:", {k: round(v, 4) for k, v in pr.items()})

# ---------------- second benchmark: LOLv1, two architectures ----------------
lol = L("diag_lolv1")
E["sid_sd"] = lol["sid_scatter"]["overall"]
NICE = {"cidnet_woperc": "CIDNet", "cidnet_wperc": "CIDNet (perc.)",
        "retinexformer": "Retinexformer"}
E["lol"] = {}
for key, m in lol["models"].items():
    g0, w0 = m["psnrf_gain_global(1x1)"], m["psnrf_white_gain_global(1x1)"]
    g16, w16 = m["psnrf_gain_16"], m["psnrf_white_gain_16"]
    lab = lol["labels"][key]
    E["lol"][key] = dict(
        name=NICE[key], psnr=m["psnr_model"], psnr_sd=m["psnr_model_sd"],
        glob=m["share_global_pct"], chan=m["share_chan_pct"], resid=m["share_resid_pct"],
        ladder16=(g16 - g0) - (w16 - w0),
        lf=100 * m["lf_energy_share_rgb_ew"],
        nband=len(lab.get("share_pct", [])))
sid_lad = E["ladder"][-1]["gain_dof"]
E["sony_row"] = dict(name="Retinexformer", psnr=E["anchors"]["model"],
                     psnr_sd=E["sid_sd"]["psnr_model"]["sd"],
                     glob=E["share"]["glob"], chan=E["share"]["chan"],
                     resid=E["share"]["resid"], ladder16=sid_lad,
                     lf=100 * E["lf"]["energy_share"])
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)

with open(os.path.join(P, "tab_cross_rows.tex"), "w") as t:
    r = E["sony_row"]
    t.write(f"Sony & {r['name']} & {r['psnr']:.2f}$\\pm${r['psnr_sd']:.2f} & {r['glob']:.1f} "
            f"& {r['chan']:.1f} & {r['resid']:.1f} & {r['ladder16']:.2f} & {r['lf']:.1f} \\\\\n")
    t.write("\\midrule\n")
    for key in ("cidnet_woperc", "cidnet_wperc", "retinexformer"):
        r = E["lol"][key]
        t.write(f"LOL & {r['name']} & {r['psnr']:.2f}$\\pm${r['psnr_sd']:.2f} & {r['glob']:.1f} "
                f"& {r['chan']:.1f} & {r['resid']:.1f} & {r['ladder16']:.2f} & {r['lf']:.1f} \\\\\n")

with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nSONYsd}{%.2f}\n" % E["sid_sd"]["psnr_model"]["sd"])
    t.write("\\newcommand{\\nLOLglobCID}{%.1f}\n" % E["lol"]["cidnet_woperc"]["glob"])
    t.write("\\newcommand{\\nLOLglobRET}{%.1f}\n" % E["lol"]["retinexformer"]["glob"])
    t.write("\\newcommand{\\nLOLladCID}{%.2f}\n" % E["lol"]["cidnet_woperc"]["ladder16"])
    t.write("\\newcommand{\\nLOLladRET}{%.2f}\n" % E["lol"]["retinexformer"]["ladder16"])
print("cross-benchmark rows written")
for k, v in E["lol"].items():
    print(" ", v["name"], f"psnr {v['psnr']:.2f}±{v['psnr_sd']:.2f}",
          f"glob {v['glob']:.1f} chan {v['chan']:.1f} resid {v['resid']:.1f}",
          f"lad16 {v['ladder16']:.3f} lf {v['lf']:.1f}")

# ---------------- multi-frame averaging does not reach the dominant term ----------------
bv = L("audit_lf_bias_vs_variance"); nc = L("refute_noise_cue")
E["mf"] = dict(n=bv["n_groups"], rows={r["k"]: r for r in bv["rows"]}, verdict=bv["verdict"],
               noise=nc)
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nMFn}{%d}\n" % bv["n_groups"])
    for r in bv["rows"]:
        t.write("\\newcommand{\\nMFlf%s}{%.3f}\n" % ("abcdefgh"[r["k"]-1], r["lf_ratio"]))
        t.write("\\newcommand{\\nMFhf%s}{%.3f}\n" % ("abcdefgh"[r["k"]-1], r["hf_ratio"]))
    t.write("\\newcommand{\\nNCmatch}{%.3f}\n" % nc["k8+noise_match"]["lf_ratio"])
    t.write("\\newcommand{\\nNCextra}{%.3f}\n" % nc["k1+extra_noise"]["lf_ratio"])
print("multi-frame macros:", {r["k"]: round(r["lf_ratio"], 3) for r in bv["rows"]},
      "noise-match", round(nc["k8+noise_match"]["lf_ratio"], 3))

# ---------------- independent multi-frame run (other worker), aggregated here from rows ----------------
mg = L("multiframe_gain")
import collections as _c
agg = _c.defaultdict(list)
for r in mg["rows"]:
    if r.get("match"):
        agg[(r["exp"], r["k"])].append((r["psnr_in"], r["psnr_out"]))
MF2 = {}
for (e, k), v in agg.items():
    MF2[f"{e}|{k}"] = dict(n=len(v), psnr_in=float(np.mean([a for a, _ in v])),
                         psnr_out=float(np.mean([b for _, b in v])))

rho_in_k1 = mg["bands"]["0.1s|1|match|in"]["rho_luma"][-1]
rho_in_k8 = mg["bands"]["0.1s|8|match|in"]["rho_luma"][-1]
E["mf2"] = dict(cohort=MF2, rho_corner_in_k1=rho_in_k1, rho_corner_in_k8=rho_in_k8)
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    a1, a8 = MF2["0.1s|1"], MF2["0.1s|8"]
    t.write("\\newcommand{\\nMFinDrop}{%.2f}\n" % (a1["psnr_in"] - a8["psnr_in"]))
    t.write("\\newcommand{\\nMFoutGain}{%.3f}\n" % (a8["psnr_out"] - a1["psnr_out"]))
    t.write("\\newcommand{\\nMFlongN}{%d}\n" % a8["n"])
    t.write("\\newcommand{\\nMFrhoA}{%.3f}\n" % rho_in_k1)
    t.write("\\newcommand{\\nMFrhoB}{%.3f}\n" % rho_in_k8)
print("mf2 0.1s: k1 in/out %.3f/%.3f  k8 in/out %.3f/%.3f  n=%d | rho corner in k1 %.3f k8 %.3f" %
      (MF2["0.1s|1"]["psnr_in"], MF2["0.1s|1"]["psnr_out"], MF2["0.1s|8"]["psnr_in"],
       MF2["0.1s|8"]["psnr_out"], MF2["0.1s|8"]["n"], rho_in_k1, rho_in_k8))

# ---------------- reproduction table, generated from the measurement dumps ----------------
def _avg(name, key):
    """mean of a per-image metric, tolerant to the two dump layouts in repro/"""
    d = json.load(open(os.path.join(R, name + ".json")))
    if "avg" in d and key in d["avg"]:
        return d["avg"][key]
    for lst in ("per_image", "rows", "images"):
        if lst in d and isinstance(d[lst], list) and d[lst] and key in d[lst][0]:
            return float(np.mean([r[key] for r in d[lst]]))
    for k2 in (key, "psnr_mean", "psnr", "mean_psnr"):
        if k2 in d and isinstance(d[k2], (int, float)):
            return float(d[k2])
    raise KeyError(f"{name}: cannot find {key}; top keys {list(d)[:8]}")
REPRO = [
    ("LOL",  "CIDNet, no rect.",         _avg("measure_LOLv1_woperc", "psnr_saved_png"), 23.5000),
    ("LOL",  "CIDNet (perc.), no rect.", _avg("measure_LOLv1_wperc",  "psnr_saved_png"), 23.8091),
    ("LOL",  "Retinexformer, rect.",     _avg("measure_LOLv1_retinexformer", "psnr_round_gtmean"), 27.18),
    ("Sony", "Retinexformer, no rect.",  E["anchors"]["model"], 24.44),
]
lolv2 = os.path.join(R, "measure_LOLv2.json")
lolv2_done = os.path.exists(os.path.join(R, "AUDIT_LOLv2_260905.md"))   # rows enter only after the audit is written
# one headline row per (benchmark, method): the setting the authors' own headline uses
KEEP = ("(w_perc, alpha 0.84), rect.", "(wo_perc), rect.", "Retinexformer, rect.")
if os.path.exists(lolv2) and lolv2_done:
    for row in json.load(open(lolv2)).get("table4_rows", []):
        if any(k in row["setting"] for k in KEEP):
            short = "Retinexformer, rect." if "Retinexformer" in row["setting"] else "CIDNet, rect."
            REPRO.append((row["bench"], short, row["ours"], row["reported"]))
E["repro_rows"] = [dict(bench=b, setting=s, ours=o, reported=r) for b, s, o, r in REPRO]
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "tab_repro_rows.tex"), "w") as t:
    for b, s_, o, r in REPRO:
        # print the reported value with the precision its authors used, ours always at 3
        rs = f"{r:.3f}" if abs(r * 1000 - round(r * 1000)) < 1e-6 and abs(r * 100 - round(r * 100)) > 1e-6 else (f"{r:.4f}" if abs(r * 1000 - round(r * 1000)) > 1e-6 else f"{r:.2f}")
        s_ = s_.replace("_", "\\_")          # underscores in setting names would open math mode
        t.write(f"{b} & {s_} & {o:.3f} & {rs} \\\\\n")
print("repro rows:", [(b, round(o, 3), r) for b, _, o, r in REPRO])

# ---------------- comparison F: noise-variance frequency-domain merge (HDR+ style) ----------------
cf = L("compare_f_hdrplus")["summary"]["ALL|8|match"]
E["hdrplus"] = dict(n=cf["nscene"], A=cf["psnr_A"], B=cf["psnr_B"], C=cf["psnr_C"],
                    F8=cf["psnr_F8"], F2=cf["psnr_F2"], F16=cf["psnr_F16"],
                    dF_B=cf["dF_B_mean"], dF_A=cf["dF_A_mean"])
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
h = E["hdrplus"]
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nHPn}{%d}\n" % h["n"])
    for k, name in (("A", "A"), ("B", "B"), ("C", "C"), ("F8", "Fe"), ("F2", "Ft")):
        t.write("\\newcommand{\\nHP%s}{%.2f}\n" % (name, h[k]))   # letters only: TeX control words end at a digit
    t.write("\\newcommand{\\nHPdFB}{%.2f}\n" % h["dF_B"])
    t.write("\\newcommand{\\nHPdFA}{%.2f}\n" % (-h["dF_A"]))
print("hdrplus:", {k: round(v, 3) for k, v in h.items()})

# ---------------- comparison of existing methods (rows arrive from compare_methods.json) ----------------
cmp_path = os.path.join(R, "compare_methods.json")
if os.path.exists(cmp_path):
    rows = json.load(open(cmp_path)).get("rows", [])
    E["cmp_rows"] = rows
    json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
    with open(os.path.join(P, "tab_cmp_rows.tex"), "w") as t:
        last = None
        SKIP_ROWS = {"Gamma 0.5", "CIDNet (perc.)"}
        ORDER = ["Input", "CLAHE", "Zero-DCE++", "SCI", "AdaptiveRetinex (ours)", "LightenDiffusion", "URetinex-Net", "SNR-Net", "LLFormer", "GSAD", "Retinexformer", "CIDNet"]
        rows = sorted(rows, key=lambda r: (r["bench"] != "LOL", ORDER.index(r["method"]) if r["method"] in ORDER else 99))
        REF = {"URetinex-Net": 2022, "SNR-Net": 2022, "LLFormer": 2023, "GSAD": 2023, "Retinexformer": 2023, "CIDNet": 2025}
        for b in ("LOL", "Sony"):
            seq = [r["method"] for r in rows if r["bench"] == b and r["method"] not in SKIP_ROWS]
            k = [i for i, mth in enumerate(seq) if mth in REF]
            assert k == list(range(len(seq) - len(k), len(seq))), ("reference-trained rows must come last", b, seq)
            yrs = [REF[seq[i]] for i in k]; assert yrs == sorted(yrs), ("reference-trained rows must be in publication order", b, yrs)          # 지면: 감마·perceptual 변형 행은 본문 표에서 제외 (release 의 COMPARE_METHODS.md 에 있음)
        for r in rows:
            if r["method"] in SKIP_ROWS: continue
            if r["bench"] != last:
                if last is not None: t.write("\\midrule\n")
                t.write("\\multicolumn{7}{l}{\\textit{%s}} \\\\\n" % r["bench"])
            last = r["bench"]
            m = r["method"].replace("_", "\\_").replace("(ours)", "(in-house)").replace("AdaptiveRetinex", "AR")
            t.write(f"{m} & {r['psnr']:.2f}$\\pm${r['psnr_sd']:.2f} & {r['ssim']:.3f} "
                    f"& {r['glob']:.1f} & {r['chan']:.1f} & {r['resid']:.1f} & {r['d16']:.2f} \\\\\n")
    print("comparison rows:", len(rows))
else:
    print("comparison rows: pending")

# ---------------- review fixes: ceiling shares by input path, perm control, noise-share in dB ----------------
import re as _re2
_md = open(os.path.join(R, "DIAG_SID_IDENTIFIABILITY.md")).read()
def _verdict(rin, rout, tz=0.20, tg=0.05):
    return "un" if rin < tz else ("rec" if rin - rout > tg else "exh")
CEIL = {}
for exp in ("0.033s", "0.04s", "0.1s"):
    i = _md.find(f"\n{exp} ("); blk = _md[i:i + 3000]
    rows = _re2.findall(r"\n\s*(\d\.\d\d-\d\.\d\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|", blk)[:11]
    for name, col in (("scalar_dn", 3), ("pw_dn", 4)):
        agg = {"rec": 0.0, "exh": 0.0, "un": 0.0}
        for r in rows:
            agg[_verdict(float(r[col]), float(r[5]))] += float(r[8])
        CEIL[f"{exp}|{name}"] = agg
E["ceiling_shares"] = CEIL
lo2 = L("diag_sid_lowfreq")["overall"]
E["perm16"] = dict(gain=lo2["psnrf_gain_16"] - lo2["psnrf_gain_global(1x1)"],
                   perm=lo2["psnrf_perm_gain_16"] - lo2["psnrf_perm_gain_global(1x1)"],
                   white=lo2["psnrf_white_gain_16"] - lo2["psnrf_white_gain_global(1x1)"])
E["noise_share_db"] = float(10 * np.log10(1 / (1 - E["noise_share_pct"] / 100)))
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    for exp, tag in (("0.033s", "short"), ("0.04s", "mid"), ("0.1s", "long")):
        t.write("\\newcommand{\\nSCALrec%s}{%.1f}\n" % (tag, CEIL[f"{exp}|scalar_dn"]["rec"]))
        t.write("\\newcommand{\\nPWrec%s}{%.1f}\n" % (tag, CEIL[f"{exp}|pw_dn"]["rec"]))
    t.write("\\newcommand{\\nPERMgain}{%.3f}\n" % E["perm16"]["perm"])
    t.write("\\newcommand{\\nPERMpct}{%.0f}\n" % (100 * E["perm16"]["perm"] / E["perm16"]["gain"]))
    t.write("\\newcommand{\\nNOISEdb}{%.2f}\n" % E["noise_share_db"])
print("ceiling shares:", {k: round(v["rec"], 1) for k, v in CEIL.items()}, "| perm", round(E["perm16"]["perm"], 3), "| noise dB", round(E["noise_share_db"], 3))

# ---------------- re-review fixes: scalar-path Table 2, margin sensitivity, LOL perm, headroom deltas, k8 band check ----------------
for e in EXPS:                                   # Table 2 / Fig.1 input column = scalar brightness match, as the text says
    k = f"{e}|scalar|1|all"
    if k in ident["input"]:
        E["rho"][e]["in_k1"] = ident["input"][k]["rho_luma"]
    for kk, tag in ((2, "in_k2"), (4, "in_k4"), (8, "in_k8")):
        k2 = f"{e}|scalar|{kk}|all"
        if k2 in ident["input"]: E["rho"][e][tag] = ident["input"][k2]["rho_luma"]
_npf = {}
for e in EXPS:
    _Pp = np.array(ident["input"][f"{e}|scalar|1|all"]["p_in_luma"]); _Nn = np.array(ident["noise"][f"{e}|1|all"]["luma"])
    _npf[e] = np.clip(_Nn / np.maximum(_Pp, 1e-30), 0.0, 0.999)
with open(os.path.join(P, "tab_rho_rows.tex"), "w") as t:      # regenerate with scalar values; dagger = unstable under the noise correction (N/P > 0.9)
    ed = E["ident_edges"]
    for i in range(len(ed) - 1):
        cells = []
        for e in EXPS:
            r = E["rho"][e]; cells += [f(r["model"][i], 3), f(r["in_k1"][i], 3) + ("\\dag" if _npf[e][i] > 0.9 else "")]
        t.write(f"{ed[i]:.2f}--{ed[i+1]:.2f} & " + " & ".join(cells) + " \\\\\n")
# LOL perm controls per model (16 px gain family)
LP = {}
for key, mm in lol["models"].items():
    g = mm["psnrf_gain_16"] - mm["psnrf_gain_global(1x1)"]; p = mm["psnrf_perm_gain_16"] - mm["psnrf_perm_gain_global(1x1)"]
    LP[key] = dict(gain=g, perm=p)
E["lol_perm"] = LP
# headroom in one unit
E["headroom"] = dict(glob=E["anchors"]["gain"] - E["anchors"]["model"], chan=E["anchors"]["chan"] - E["anchors"]["gain"],
                     blk=E["ladder"][-1]["gain_dof"], aff=E["ladder"][-1]["affine_dof"],
                     bandls=so2["psnr_bandls_64"] - so2["psnr_model_gainfix"], noise=E["noise_share_db"])
# does 8-frame agreement rise in every band at 0.1 s?
r1, r8 = E["rho"]["0.1s"]["in_k1"], E["rho"]["0.1s"]["in_k8"]
E["k8_rise_bands"] = [i for i in range(len(r1)) if r8[i] > r1[i]]
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nLPcidGain}{%.2f}\n\\newcommand{\\nLPcidPerm}{%.3f}\n" % (LP["cidnet_woperc"]["gain"], LP["cidnet_woperc"]["perm"]))
    t.write("\\newcommand{\\nLPretGain}{%.2f}\n\\newcommand{\\nLPretPerm}{%.3f}\n" % (LP["retinexformer"]["gain"], LP["retinexformer"]["perm"]))
    h = E["headroom"]
    for k, v in h.items(): t.write("\\newcommand{\\nHR%s}{%.2f}\n" % (k, v))
print("LOL perm:", {k: (round(v["gain"], 2), round(v["perm"], 2)) for k, v in LP.items()})
print("headroom:", {k: round(v, 2) for k, v in E["headroom"].items()}, "| k8 rises in bands:", E["k8_rise_bands"], "of", len(r1))


# ---------------- v14: identifiability quantities from json only (no markdown parsing) ----------------
_idj = ident
def _np_scalar(exp):
    P = np.array(_idj["input"][f"{exp}|scalar|1|all"]["p_in_luma"]); Nn = np.array(_idj["noise"][f"{exp}|1|all"]["luma"])
    return np.clip(Nn / np.maximum(P, 1e-30), 0.0, 0.999)
def _rows(exp):
    i = _md.find(f"\n{exp} ("); blk = _md[i:i + 3000]
    return _re2.findall(r"\n\s*(\d\.\d\d)-(\d\.\d\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|\s*(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\|", blk)[:11]
NPU, MZU, GAPJ, MZJ, RECJ = {}, {}, {}, {}, {}
for exp in ("0.033s", "0.04s", "0.1s"):
    fr = _np_scalar(exp); rows = _rows(exp)
    rin = np.array(_idj["input"][f"{exp}|scalar|1|all"]["rho_luma"]) / np.sqrt(1.0 - fr)
    rout = np.array(_idj["model"][exp]["rho_luma"])
    rp = np.array(_idj["model"][exp]["res_plain_rgb"]); share = 100.0 * rp / rp.sum()          # band share of model residual energy
    share_md = np.array([float(r[9]) for r in rows]); assert np.abs(share - share_md).max() < 0.011, (exp, "share json vs md")
    uns = [j for j in range(11) if fr[j] > 0.9]
    NPU[exp] = dict(first_band_start=float(rows[uns[0]][0]) if uns else None, n_unstable=len(uns), corner_np=float(fr[-1]))
    rec5 = [j for j in range(11) if rin[j] >= 0.20 and rin[j] - rout[j] > 0.05]
    rec0 = [j for j in range(11) if rin[j] >= 0.20 and rin[j] - rout[j] > 0.0]
    RECJ[exp] = float(share[rec5].sum()); MZJ[exp] = float(share[rec0].sum()); MZU[exp] = float(sum(share[j] for j in rec0 if fr[j] > 0.9))
    GAPJ[exp] = max((float(rin[j] - rout[j]) for j in rec5), default=None)
    assert abs(RECJ[exp] - CEIL[f"{exp}|scalar_dn"]["rec"]) < 0.11, (exp, "rec share json vs md")
E["np_unstable"] = NPU; E["ceiling_margin0"] = MZJ; E["ceiling_margin0_unstable"] = MZU; E["gap_cross_json"] = GAPJ; E["ceiling_rec_json"] = RECJ
# pointwise (pw) input path from the json dump: N_j/P_j, corrected rho_in and the recoverable share, cross-checked against the md table
PWJ, NPPW = {}, {}
for exp in ("0.033s", "0.04s", "0.1s"):
    _Ppw = np.array(_idj["input"][f"{exp}|pw|1|all"]["p_in_luma"]); _Npw = np.array(_idj["noise"][f"{exp}|1|all"]["luma_pw"])
    frp = np.clip(_Npw / np.maximum(_Ppw, 1e-30), 0.0, 0.999)
    rinp = np.array(_idj["input"][f"{exp}|pw|1|all"]["rho_luma"]) / np.sqrt(1.0 - frp)
    rout = np.array(_idj["model"][exp]["rho_luma"]); rp = np.array(_idj["model"][exp]["res_plain_rgb"]); share = 100.0 * rp / rp.sum()
    rec5p = [j for j in range(11) if rinp[j] >= 0.20 and rinp[j] - rout[j] > 0.05]
    PWJ[exp] = float(share[rec5p].sum()); NPPW[exp] = [float(v) for v in frp]
    assert abs(PWJ[exp] - CEIL[f"{exp}|pw_dn"]["rec"]) < 0.11, (exp, "pw rec share json vs md", PWJ[exp], CEIL[f"{exp}|pw_dn"]["rec"])
    E["rho"][exp]["in_pw"] = _idj["input"][f"{exp}|pw|1|all"]["rho_luma"]
E["np_pw"] = NPPW; E["ceiling_rec_pw_json"] = PWJ
fx1 = _idj["input"]["0.1s|scalar|1|fix"]["rho_luma"]; fx8 = _idj["input"]["0.1s|scalar|8|fix"]["rho_luma"]
E["mf_input_fix"] = dict(n1=_idj["input"]["0.1s|scalar|1|fix"]["n"], n8=_idj["input"]["0.1s|scalar|8|fix"]["n"], corner1=fx1[-1], corner8=fx8[-1], rise=sum(b > a for a, b in zip(fx1, fx8)))
E["rho"]["0.1s"]["in_k1_fix"] = fx1
# Sony offset-only ladder at 16 px (control-corrected), 8-bit rule deltas on the three LOL checkpoints
_lf = json.load(open(os.path.join(R, "diag_sid_lowfreq.json")))["overall"]
E["ladder_off16"] = (_lf["psnrf_off_16"] - _lf["psnrf_off_global(1x1)"]) - (_lf["psnrf_white_off_16"] - _lf["psnrf_white_off_global(1x1)"])
BIT = {}
for tag in ("woperc", "wperc", "retinexformer"):
    av = json.load(open(os.path.join(R, f"measure_LOLv1_{tag}.json")))["avg"]
    kr = [k for k in av if "round" in k and "psnr" in k and "gtmean" not in k and "rect" not in k]; kt = [k for k in av if "trunc" in k and "psnr" in k and "gtmean" not in k and "rect" not in k]
    BIT[tag] = av[kr[0]] - av[kt[0]]
E["bit_rule_delta"] = BIT
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    for exp, tag in (("0.033s", "short"), ("0.04s", "mid"), ("0.1s", "long")):
        t.write("\\newcommand{\\nMZrec%s}{%.1f}\n" % (tag, MZJ[exp]))
    t.write("\\newcommand{\\nLPcidPct}{%.0f}\n\\newcommand{\\nLPretPct}{%.0f}\n" % (100 * LP["cidnet_woperc"]["perm"] / E["lol"]["cidnet_woperc"]["ladder16"], 100 * LP["retinexformer"]["perm"] / E["lol"]["retinexformer"]["ladder16"]))
    t.write("\\newcommand{\\nLPcidSpec}{%.0f}\n\\newcommand{\\nLPretSpec}{%.0f}\n" % (100 - 100 * LP["cidnet_woperc"]["perm"] / E["lol"]["cidnet_woperc"]["ladder16"], 100 - 100 * LP["retinexformer"]["perm"] / E["lol"]["retinexformer"]["ladder16"]))
    t.write("\\newcommand{\\nPERMpctC}{%.0f}\n" % (100 * E["perm16"]["perm"] / (E["perm16"]["gain"] - E["perm16"]["white"])))
    t.write("\\newcommand{\\nNPshortFrom}{%.2f}\n\\newcommand{\\nNPmidFrom}{%.2f}\n\\newcommand{\\nNPmidCorner}{%.3f}\n" % (NPU["0.033s"]["first_band_start"], NPU["0.04s"]["first_band_start"], NPU["0.04s"]["corner_np"]))
    t.write("\\newcommand{\\nMZunshort}{%.1f}\n\\newcommand{\\nMZunmid}{%.1f}\n" % (MZU["0.033s"], MZU["0.04s"]))
    t.write("\\newcommand{\\nEXmid}{%.3f}\n\\newcommand{\\nEXlong}{%.3f}\n" % (GAPJ["0.04s"] - 0.05, GAPJ["0.1s"] - 0.05))
    t.write("\\newcommand{\\nMFinFixA}{%.3f}\n\\newcommand{\\nMFinFixB}{%.3f}\n" % (fx1[-1], fx8[-1]))
    t.write("\\newcommand{\\nLADoff}{%.2f}\n\\newcommand{\\nBITlo}{%.3f}\n\\newcommand{\\nBIThi}{%.3f}\n" % (E["ladder_off16"], min(BIT.values()), max(BIT.values())))
print("np_unstable:", NPU, "| MZ:", MZJ, "| MZ unstable:", MZU, "| rec:", RECJ, "| gaps:", GAPJ, "| off16:", round(E["ladder_off16"],3), "| bit:", {k: round(v,3) for k,v in BIT.items()})

# ---------------- v14: cross-variant score from the full CIDNet run (n=598) ----------------
_cv = json.load(open(os.path.join(R, "measure_SID.json")))
E["cross_variant"] = dict(n=_cv["n"], psnr_round=_cv["avg"]["psnr_round"], psnr_trunc=_cv["avg"]["psnr_trunc"])
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nCROSSpsnr}{%.2f}\n" % _cv["avg"]["psnr_round"])
print("cross-variant:", E["cross_variant"])

# ---------- v15: override md-derived macros with json values (scalar-dn recoverable share, raw-input unidentifiable share) ----------
def _setmacro(name, value):
    p = os.path.join(P, "numbers.tex"); s = open(p).read()
    s2 = re.sub(r"\\newcommand\{\\%s\}\{[^}]*\}" % name, "\\\\newcommand{\\\\%s}{%s}" % (name, value), s)
    assert s2 != s or ("{\\%s}{%s}" % (name, value)) in s, name
    open(p, "w").write(s2)
import re
RAWJ = {}
for exp, tag in (("0.033s", "short"), ("0.04s", "mid"), ("0.1s", "long")):
    rp = np.array(_idj["model"][exp]["res_plain_rgb"]); share = 100.0 * rp / rp.sum()
    rin_raw = np.array(_idj["input"][f"{exp}|scalar|1|all"]["rho_luma"])
    RAWJ[exp] = float(share[rin_raw < 0.20].sum())
    _setmacro("nSCALrec" + tag, "%.1f" % RECJ[exp]); _setmacro("nRAWun" + tag, "%.1f" % RAWJ[exp]); _setmacro("nPWrec" + tag, "%.1f" % PWJ[exp])
E["raw_unident_json"] = RAWJ
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
print("json overrides: rec", {k: round(v, 1) for k, v in RECJ.items()}, "raw-unident", {k: round(v, 1) for k, v in RAWJ.items()})

# ---------------- v17: comparison-table macros for the text (from compare_methods.json) ----------------
if os.path.exists(cmp_path):
    _cj = json.load(open(cmp_path)); _rows = _cj["rows"] if isinstance(_cj, dict) and "rows" in _cj else _cj
    SKIP = {"Gamma 0.5", "CIDNet (perc.)"}
    def _sel(b): return [r for r in _rows if r["bench"] == b and r["method"] not in SKIP]
    def _meth(b): return [r for r in _sel(b) if r["method"] != "Input"]
    L, S = _meth("LOL"), _meth("Sony")
    clear = [r["method"] for r in L if r["psnr"] >= 20.0]; rest = [r for r in L if r["psnr"] < 20.0]
    with open(os.path.join(P, "numbers.tex"), "a") as t:
        t.write("\\newcommand{\\nCMPnLOL}{%d}\n\\newcommand{\\nCMPnSony}{%d}\n\\newcommand{\\nCMPnAll}{%s}\n" % (len(L), len(S), {18: "eighteen", 17: "seventeen", 19: "nineteen", 20: "twenty"}.get(len(L) + len(S), str(len(L) + len(S)))))
        t.write("\\newcommand{\\nCMPclearLOL}{%s}\n" % {5: "five", 6: "six", 7: "seven", 4: "four"}.get(len(clear), str(len(clear))))
        t.write("\\newcommand{\\nCMPrestLOL}{%.0f}\n" % (max(r["psnr"] for r in rest) + 0.5))
        for tag, rr in (("LOL", L), ("Sony", S)):
            import math as _m
            t.write("\\newcommand{\\nCMPglob%slo}{%d}\n\\newcommand{\\nCMPglob%shi}{%d}\n" % (tag, _m.floor(min(r["glob"] for r in rr)), tag, _m.ceil(max(r["glob"] for r in rr))))
            t.write("\\newcommand{\\nCMPd%slo}{%.1f}\n\\newcommand{\\nCMPd%shi}{%.1f}\n" % (tag, _m.floor(10 * min(r["d16"] for r in rr)) / 10, tag, _m.ceil(10 * max(r["d16"] for r in rr)) / 10))
    _ld = [r for r in L if r["method"] == "LightenDiffusion"][0]; _ldS = [r for r in S if r["method"] == "LightenDiffusion"]
    with open(os.path.join(P, "numbers.tex"), "a") as t:
        t.write("\\newcommand{\\nLDlol}{%.2f}\n" % _ld["psnr"])
        if _ldS: t.write("\\newcommand{\\nLDsonyChan}{%.0f}\n" % _ldS[0]["chan"])
    E["cmp_text"] = dict(clear_LOL=clear, rest_LOL=[(r["method"], round(r["psnr"], 2)) for r in rest], sony_over20=[r["method"] for r in S if r["psnr"] >= 20.0])
    json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
    print("cmp text:", E["cmp_text"])

# ---------------- v18: GSAD rectified reproduction (newbase_gsad.json) ----------------
_gs = json.load(open(os.path.join(R, "newbase_gsad.json")))
def _find(d, keys):
    for k in keys:
        if k in d: return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = _find(v, keys)
            if r is not None: return r
    return None
_gr = _find(_gs, ["psnr_round_gtmean", "psnr_gtmean_round", "psnr_rect_round"]); _gu = _find(_gs, ["psnr_round"])
assert _gr is not None and _gu is not None, ("gsad keys", list(_gs.keys())[:10])
E["gsad"] = dict(round=_gu, round_gtmean=_gr)
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nGSADrect}{%.3f}\n\\newcommand{\\nGSADrectGain}{%.2f}\n\\newcommand{\\nGSADpub}{27.84}\n" % (_gr, _gr - _gu))
print("gsad:", E["gsad"])

# ---------------- v22: URetinex ratio gap (json fixed-ratio value + md reference-ratio value), LightenDiffusion seed spread (md sweep) ----------------
_ur = json.load(open(os.path.join(R, "newbase_uretinex.json")))
_urmd = open(os.path.join(R, "NEWBASE_uretinex.md")).read()
assert _ur["avg"]["ratio"] == 5.0, "URetinex cache ratio changed"; _urfix = _ur["avg"].get("psnr_round_ratio5", _ur["avg"]["psnr_round"])
_urgt = float(re.search(r"evaluate\.py GT비율 \(진단\)\s+반올림\s+([\d.]+)", _urmd).group(1))
_ldj = json.load(open(os.path.join(R, "newbase_lightendiff.json")))["seed_sweep_psnr_round"]; _lo, _hi = _ldj["min"], _ldj["max"]
E["uretinex"] = dict(fixed=_urfix, gtratio=_urgt); E["ld_sweep"] = [_lo, _hi]
json.dump(E, open(os.path.join(P, "evidence.json"), "w"), indent=2)
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nURfixed}{%.2f}\n\\newcommand{\\nURpub}{%.2f}\n\\newcommand{\\nURratioGain}{%.2f}\n\\newcommand{\\nLDspread}{%.2f}\n" % (_urfix, _urgt, _urgt - _urfix, _hi - _lo))
print("uretinex:", E["uretinex"], "| ld sweep:", E["ld_sweep"])
_np_ = os.path.join(P, "numbers.tex"); _t = open(_np_).read()
_t = "\n".join(l for l in _t.split("\n") if "nLPcidSpec" not in l and "nLPretSpec" not in l)
open(_np_, "w").write(_t)

import glob
# ---------------- v24: exposure-wise residual share macros (diag_sid_failure by_exposure), prune unused macros ----------------
_bx = E.get("by_exposure") or json.load(open(os.path.join(R, "diag_sid_failure.json"))).get("by_exposure")
def _resid_share(entry):
    for k in ("share_resid_pct", "resid", "share_residual_pct"):
        if k in entry: return entry[k]
    return None
if isinstance(_bx, dict):
    _short = _resid_share(_bx.get("0.033s", {})); _long = _resid_share(_bx.get("0.1s", {}))
    if _short is not None and _long is not None:
        with open(os.path.join(P, "numbers.tex"), "a") as t:
            t.write("\\newcommand{\\nSHAREresidShort}{%.1f}\n\\newcommand{\\nSHAREresidLong}{%.1f}\n" % (_short, _long))
        print("resid share by exposure:", round(_short, 3), round(_long, 3))
    else:
        print("WARN: by_exposure resid share keys not found:", list(_bx.get("0.1s", {}).keys())[:8])
_np_ = os.path.join(P, "numbers.tex"); _t = open(_np_).read()
_used = set(re.findall(r"\\(n[A-Za-z]+)", "".join(open(f).read() for f in glob.glob(os.path.join(P, "*.tex")) + glob.glob(os.path.join(P, "..", "paper", "*.tex")) if not f.endswith("numbers.tex"))))
if not (glob.glob(os.path.join(P, "sec_*.tex")) or glob.glob(os.path.join(P, "..", "paper", "sec_*.tex"))):
    print("macro prune skipped: no manuscript .tex found next to %s (found %d macro uses)" % (P, len(_used))); _kept = _t.split("\n")
else:
    _kept = [l for l in _t.split("\n") if not l.startswith("\\newcommand{\\n") or re.match(r"\\newcommand\{\\(n[A-Za-z]+)\}", l).group(1) in _used]
    _dropped = [re.match(r"\\newcommand\{\\(n[A-Za-z]+)\}", l).group(1) for l in _t.split("\n") if l.startswith("\\newcommand{\\n") and l not in _kept]
    print("pruned macros:", _dropped)
open(_np_, "w").write("\n".join(_kept)); print("numbers.tex macros:", sum(1 for l in _t.split("\n") if l.startswith("\\newcommand")), "->", sum(1 for l in _kept if l.startswith("\\newcommand")))

import math as _mm
# ---------------- v31: n per exposure, blur anchor, LOL LF-share range ----------------
with open(os.path.join(P, "numbers.tex"), "a") as t:
    t.write("\\newcommand{\\nCNNtrainScenes}{%d}\n" % E["cnn_train_scenes"])
    t.write("\\newcommand{\\nLOLshareSDmax}{%.0f}\n\\newcommand{\\nLOLdSDmax}{%.1f}\n" % (E["lol_frame_sd"]["share_max"], E["lol_frame_sd"]["d16_max"]))
    _ni = ident["n_images"]; t.write("\\newcommand{\\nNshort}{%d}\n\\newcommand{\\nNmid}{%d}\n\\newcommand{\\nNlong}{%d}\n" % (_ni["0.033s"], _ni["0.04s"], _ni["0.1s"]))
    t.write("\\newcommand{\\nBLURbase}{%.3f}\n" % so2["psnr_model_gainfix"])
    _lf = [v for k, v in lol["models"].items() if k in ("cidnet_woperc", "cidnet_wperc", "retinexformer")]
    _key = "lf_energy_share_rgb_ew" if "lf_energy_share_rgb_ew" in _lf[0] else [k for k in _lf[0].keys() if "lf" in k.lower() and "share" in k.lower()][0]
    t.write("\\newcommand{\\nLOLlfLo}{%d}\n\\newcommand{\\nLOLlfHi}{%d}\n" % (_mm.floor(100 * min(v[_key] for v in _lf)), _mm.ceil(100 * max(v[_key] for v in _lf))))
    print("v31 macros: n", _ni, "blur base", round(so2["psnr_model_gainfix"], 3), "lf key", _key, [round(v[_key], 1) for v in _lf])

from make_diagnostic_evidence import augment
augment(P)
