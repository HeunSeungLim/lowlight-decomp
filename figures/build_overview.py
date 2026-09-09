"""Figure 1, drawn entirely in code from the real panels at 2x and downsampled once, so
every stroke is crisp and uniform (same recipe as the depth paper's adopted figure).
Five stages left->right on one teal path: dark frame -> frozen network (hexagon + padlock)
with its output below -> reference + residual symbol -> decomposition panel (diamond, grid,
arcs) with the low-frequency residual map below -> two verdict symbols. Times labels above,
three measurement plots in a second row, one sentence below.
"""
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "..", "paper_tables", "evidence.json")))
TG = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
FONT, FONTB, FONTI = TG + "texgyretermes-regular.otf", TG + "texgyretermes-bold.otf", TG + "texgyretermes-italic.otf"
FREE = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
NAVY, TEAL, GREEN, ORANGE, DARK = (28, 43, 94), (23, 128, 128), (46, 125, 50), (214, 118, 20), (40, 40, 40)
SS = 2; W1 = 3800; W = W1 * SS

# ---------------------------------------------------------------- plots, restyled for print
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"], "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8})
def fig_panel(fn, name, size=(3.6, 1.55)):
    fig, ax = plt.subplots(figsize=size, dpi=300); fn(ax)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4); fig.savefig(f"{HERE}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.06); plt.close(fig)
ed = E["ident_edges"]; cc = [(ed[i] + ed[i + 1]) / 2 for i in range(len(ed) - 1)]
def p1(ax):
    r33, r1 = E["rho"]["0.033s"], E["rho"]["0.1s"]
    ax.plot(cc, r1["model"], "-", color=NAVY and (0.11, 0.17, 0.37), lw=1.8, label="model, 0.1 s")
    ax.plot(cc, r1["in_k1_fix"], ":", color=(0.11, 0.17, 0.37), lw=1.8, label="input, 0.1 s")
    ax.plot(cc, r1["in_k8"], "--", color=(0.09, 0.5, 0.5), lw=1.8, label="input, 0.1 s, 8 frames")
    ax.plot(cc, r33["model"], "-", color=(0.84, 0.46, 0.08), lw=1.8, label="model, 0.033 s")
    ax.plot(cc, r33["in_k1"], ":", color=(0.84, 0.46, 0.08), lw=1.8, label="input, 0.033 s")
    ax.set_xlabel("radial frequency"); ax.set_ylabel("correlation"); ax.set_ylim(0, 2.75); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, handlelength=2.0, columnspacing=1.0)
def p2(ax):
    L = E["ladder"]; names = [r["block"].replace("global(1x1)", "frame") for r in L]; xs = range(len(L))
    ax.plot(xs, [r["affine_dof"] for r in L], "s--", color=(0.84, 0.46, 0.08), ms=4, lw=1.8, label="affine per block")
    ax.plot(xs, [r["gain_dof"] for r in L], "o-", color=(0.11, 0.17, 0.37), ms=4, lw=1.8, label="gain per block")
    ax.set_xticks(list(xs)); ax.set_xticklabels(names); ax.set_xlabel("block side (px)"); ax.set_ylabel("dB over frame")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
def p3(ax):
    rows = E["mf"]["rows"]; ks = sorted(int(k) for k in rows)
    ax.plot(ks, [rows[str(k)]["lf_ratio"] for k in ks], "o-", color=(0.84, 0.46, 0.08), ms=4, lw=1.8, label="low-freq. error, k-frame mean")
    ax.plot(ks, [rows[str(k)]["hf_ratio"] for k in ks], "s-", color=(0.11, 0.17, 0.37), ms=4, lw=1.8, label="high-freq. error")
    ax.plot(ks, [1 / k for k in ks], ":", color="0.4", lw=1.6, label="1/k, pure variance")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("frames averaged k"); ax.set_ylabel("energy / k=1"); ax.set_ylim(0, 4.3); ax.set_yticks([0, 1, 2]); ax.legend(frameon=False, fontsize=10, loc="upper left")
fig_panel(p1, "F1_rho"); fig_panel(p2, "F2_ladder"); fig_panel(p3, "F3_kavg")

# ---------------------------------------------------------------- real panels (no abstract glyphs)
import os, sys, torch
sys.path.insert(0, os.environ.get("RETINEXFORMER_REPO", "third_party/Retinexformer"))
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
R = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID"); SC, EXPO = "10198", "0.033s"
net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
net.load_state_dict(torch.load((os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth"), map_location="cpu")["params"]); net = net.cuda().eval()
sf = [f for f in sorted(os.listdir(f"{R}/short_sid2/{SC}")) if EXPO in f][0]; gf = sorted(os.listdir(f"{R}/long_sid2/{SC}"))[0]
lq = np.load(f"{R}/short_sid2/{SC}/{sf}")[:, :, ::-1].astype(np.float32) / 255.      # BGR -> RGB, authors' convention
gt = np.load(f"{R}/long_sid2/{SC}/{gf}")[:, :, ::-1].astype(np.float32) / 255.
with torch.inference_mode():
    y = torch.clamp(net(torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()), 0, 1)[0].cpu().numpy().transpose(1, 2, 0)
a = float((y * gt).sum() / (y * y).sum()); res = a * y - gt                            # residual after the oracle global gain
def to_img(arr): return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
def diverging(m, lim):
    return Image.fromarray((plt.get_cmap("coolwarm")((np.clip(m / lim, -1, 1) + 1) / 2)[:, :, :3] * 255).astype(np.uint8))
# low- / high-frequency split of the residual (luminance), cutoff 0.10 as in the paper
Fr = np.fft.fftshift(np.fft.fft2(res.mean(2))); Hh, Ww = Fr.shape
u = np.fft.fftshift(np.fft.fftfreq(Hh))[:, None]; v = np.fft.fftshift(np.fft.fftfreq(Ww))[None, :]
lowm = np.sqrt((2 * u) ** 2 + (2 * v) ** 2) <= 0.10
lf = np.real(np.fft.ifft2(np.fft.ifftshift(np.where(lowm, Fr, 0)))); hf = res.mean(2) - lf
# block gain field: least-squares scalar per 16 px block on the gain-corrected output
B = 16; yb = (a * y).mean(2); gb = gt.mean(2); Hb, Wb = Hh // B, Ww // B
blk = np.ones((Hb, Wb))
for r_ in range(Hb):
    for c_ in range(Wb):
        yy = yb[r_ * B:(r_ + 1) * B, c_ * B:(c_ + 1) * B]; gg = gb[r_ * B:(r_ + 1) * B, c_ * B:(c_ + 1) * B]
        blk[r_, c_] = (yy * gg).sum() / max((yy * yy).sum(), 1e-9)
blk_img = diverging(np.kron(blk - 1.0, np.ones((B, B))), 0.5)                         # deviation of the block gain from 1
lim = float(np.percentile(np.abs(res), 99))
P1, P2, P3 = to_img(lq * 20), to_img(y), to_img(gt)
P_res = diverging(res.mean(2), lim); P_lf = diverging(lf, lim); P_hf = diverging(hf, float(np.percentile(np.abs(hf), 99)))
# band verdict strip from the paper's rule (gap 0.05, zero 0.20) on the raw-input criterion
# the verdict rule is the measurement code's own (TH_ZERO = 0.20 on the input, TH_GAP = 0.05), not a re-implementation
sys.path.insert(0, os.path.join(HERE, "..", "code"))
from diag_sid_identifiability import verdict as _verdict
def verdicts(exp):
    m, i_ = E["rho"][exp]["model"], E["rho"][exp]["in_k1"]
    return [{"불가": "un", "회복가능": "rec", "소진": "exh"}[_verdict(ii, mm)] for mm, ii in zip(m, i_)]
COL = {"rec": (46, 125, 50), "exh": (150, 150, 150), "un": (200, 60, 40)}
def strip(exp, w, h):
    vs = verdicts(exp); im_ = Image.new("RGB", (w, h), "white"); dd = ImageDraw.Draw(im_); cw = w / len(vs)
    for k, vv in enumerate(vs): dd.rectangle([k * cw + 2, 0, (k + 1) * cw - 2, h - 1], fill=COL[vv])
    return im_
def load(name): return Image.open(f"{HERE}/{name}.png").convert("RGB")
F1, F2, F3 = load("F1_rho"), load("F2_ladder"), load("F3_kavg")
PH = 280 * SS
def fit_h(im, h): return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
P1, P2, P3, P_res = fit_h(P1, PH), fit_h(P2, PH), fit_h(P3, PH), fit_h(P_res, PH); PW = P1.width
SMALLH = (PH - 2 * 18 * SS) // 3                                                       # three stacked decomposition panels
P_lf, P_hf, blk_img = fit_h(P_lf, SMALLH), fit_h(P_hf, SMALLH), fit_h(blk_img, SMALLH); SW4 = P_lf.width
# assets for the vector figure (fig1_mpl.py)
for _nm, _im in (("P_res_raw", P_res), ("P_lf_raw", P_lf), ("P_hf_raw", P_hf), ("P_blk_raw", blk_img)): _im.save(f"{HERE}/{_nm}.png")
json.dump({e: verdicts(e) for e in ("0.033s", "0.1s")}, open(f"{HERE}/verdicts.json", "w"))

# ---------------------------------------------------------------- layout (2x): labels sit directly above the panel row
MARG = 60 * SS; GAP = 130 * SS
Y_LAB = 110 * SS; Y_PAN = 150 * SS
Y_ROW2 = Y_PAN + PH + 120 * SS
PWD0 = (W - 2 * MARG - 2 * GAP) // 3
PLOT_H = round(F1.height * PWD0 / F1.width)
H = Y_ROW2 + PLOT_H + 70 * SS
im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
LW = 10 * SS
def arrow(p0, p1, col, lw=LW, head=40 * SS):
    (x0, y0), (x1, y1) = p0, p1; v = np.array([x1 - x0, y1 - y0], float); L = np.hypot(*v); u = v / L; n = np.array([-u[1], u[0]])
    tip = np.array([x1, y1]); base = tip - u * head; d.line([x0, y0, *base], fill=col, width=lw)
    d.polygon([tuple(tip), tuple(base + n * head * 0.55), tuple(base - n * head * 0.55)], fill=col)
def frame(x, y, w, h, col): d.rectangle([x, y, x + w - 1, y + h - 1], outline=col, width=6 * SS)
SW5 = 470 * SS
SW = [PW, PW, PW, PW, SW4 + 250 * SS, SW5 + 40 * SS]
total = sum(SW) + 5 * GAP; x0 = (W - total) // 2
xs = []; x = x0
for w in SW: xs.append(x); x += w + GAP
cx = [x + w // 2 for x, w in zip(xs, SW)]; Y_MID = Y_PAN + PH // 2
# 1 input, 2 output, 3 reference, 4 residual: real images
for k, (P, col) in enumerate(((P1, NAVY), (P2, TEAL), (P3, NAVY), (P_res, NAVY))):
    im.paste(P, (xs[k], Y_PAN)); frame(xs[k], Y_PAN, PW, PH, col)
arrow((xs[0] + PW + 12 * SS, Y_MID), (xs[1] - 12 * SS, Y_MID), TEAL)
d.text(((xs[0] + PW + xs[1]) // 2, Y_MID - 22 * SS), "f", font=ImageFont.truetype(FREE, 60 * SS), fill=TEAL, anchor="ms")
arrow((xs[1] + PW + 12 * SS, Y_MID), (xs[2] - 12 * SS, Y_MID), TEAL)
arrow((xs[2] + PW + 12 * SS, Y_MID), (xs[3] - 12 * SS, Y_MID), TEAL)
# 5 decomposition: three stacked real maps
for k, (P, col) in enumerate(((P_lf, NAVY), (blk_img, GREEN), (P_hf, ORANGE))):
    yy = Y_PAN + k * (SMALLH + 18 * SS); im.paste(P, (xs[4], yy)); frame(xs[4], yy, SW4, SMALLH, col)
arrow((xs[3] + PW + 12 * SS, Y_MID), (xs[4] - 12 * SS, Y_MID), TEAL)
# 6 verdicts: real band strips per exposure + measured ceilings (compact column)
vx = xs[5] + 40 * SS; vy = Y_PAN
fS2 = ImageFont.truetype(FONT, 38 * SS); fS3 = ImageFont.truetype(FONTB, 38 * SS)
d.text((vx, vy), "0.033 s", font=fS2, fill=DARK, anchor="la"); im.paste(strip("0.033s", SW5, 46 * SS), (vx, vy + 44 * SS))
d.text((vx, vy + 104 * SS), "0.1 s", font=fS2, fill=DARK, anchor="la"); im.paste(strip("0.1s", SW5, 46 * SS), (vx, vy + 148 * SS))
xk = vx
for name, col in (("recoverable", COL["rec"]), ("exhausted", COL["exh"]), ("unidentif.", COL["un"])):
    yk = vy + 214 * SS; d.rectangle([xk, yk, xk + 24 * SS, yk + 24 * SS], fill=col)
    d.text((xk + 32 * SS, yk - 6 * SS), name, font=ImageFont.truetype(FONT, 30 * SS), fill=DARK, anchor="la"); xk += 185 * SS
L16 = E["ladder"][-1]
d.text((vx, vy + PH - 46 * SS), f"16 px ceiling: +{L16['gain_dof']:.2f} / +{L16['affine_dof']:.2f} dB", font=fS3, fill=ORANGE, anchor="la")
arrow((xs[4] + SW4 + 250 * SS - 40 * SS, Y_MID), (xs[5] - 12 * SS, Y_MID), TEAL)
# labels directly above each panel
fT = ImageFont.truetype(FONTB, 54 * SS); fS = ImageFont.truetype(FREE, 46 * SS)
labels = ["dark frame x", "output y = f (x)", "reference g", "residual a·y − g", "decompose", "band verdicts"]
for c, t1 in zip(cx, labels): d.text((c, Y_LAB), t1, font=fT, fill="black", anchor="ms")
# a is the least-squares global gain; say so under the residual panel
d.text((cx[3], Y_PAN + PH + 46 * SS), "a = \u27e8y,g\u27e9 / \u27e8y,y\u27e9,  the best single gain", font=ImageFont.truetype(FREE, 40 * SS), fill=DARK, anchor="ms")
fq = ImageFont.truetype(FONT, 34 * SS)
for k, t in enumerate(("low-frequency", "block gain", "high-frequency")):
    d.text((xs[4] + SW4 + 10 * SS, Y_PAN + k * (SMALLH + 18 * SS) + SMALLH // 2), t, font=fq, fill=DARK, anchor="lm")
# second row: three plots
PWD = (W - 2 * MARG - 2 * GAP) // 3
for k, P in enumerate((F1, F2, F3)):
    p = P.resize((PWD, round(P.height * PWD / P.width)), Image.LANCZOS)
    x = MARG + k * (PWD + GAP); im.paste(p, (x, Y_ROW2))
    frame(x, Y_ROW2, PWD, p.height, (150, 150, 150))
out = im.resize((W1, H // SS), Image.LANCZOS); out.save(f"{HERE}/fig_workflow_v2.png")
print("saved", out.size, "aspect", round(out.width / out.height, 2))
