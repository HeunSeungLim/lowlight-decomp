"""Vector version of the overview figure: every label >= 9 pt, output fig_workflow.pdf (7.008 in = \\textwidth)."""
import os, json, re, numpy as np, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(HERE, "..", "paper_tables", "evidence.json")))
V = json.load(open(os.path.join(HERE, "verdicts.json")))
plt.rcParams.update({"mathtext.fontset": "dejavuserif", "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9.6,
                     "axes.labelsize": 9.6, "xtick.labelsize": 9.4, "ytick.labelsize": 9.4, "legend.fontsize": 9.4,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
NAVY, TEAL, ORANGE, DARK = (0.11, 0.17, 0.37), (0.09, 0.5, 0.5), (0.84, 0.46, 0.08), (0.15, 0.15, 0.15)
COL = {"rec": (46/255, 125/255, 50/255), "exh": (150/255, 150/255, 150/255), "un": (200/255, 60/255, 40/255)}
src = open(os.path.join(HERE, "build_overview.py")).read()
plot_src = src[src.index("ed = E[\"ident_edges\"]"):src.index("fig_panel(p1, \"F1_rho\")")]
plot_src = plot_src.replace('fontsize=9.4, loc="upper center", ncol=2, handlelength=2.0, columnspacing=1.0', 'fontsize=9.4, loc="upper center", ncol=2, handlelength=1.6, columnspacing=0.8, labelspacing=0.25').replace('fontsize=10, loc="upper left")', 'fontsize=9.4, loc="upper left", labelspacing=0.25)').replace('fontsize=11, loc="upper left")', 'fontsize=9.4, loc="upper left", labelspacing=0.25)').replace('ax.set_ylim(0, 2.75); ax.set_yticks([0, 0.5, 1.0])', 'ax.set_ylim(0, 3.6); ax.set_yticks([0, 1])').replace('ax.set_ylim(0, 4.3)', 'ax.set_ylim(0, 5.6)')
plot_src = (plot_src.replace('label="model, 0.1 s"', 'label="model 0.1 s"').replace('label="input, 0.1 s"', 'label="input 0.1 s"').replace('label="input, 0.1 s, 8 frames"', 'label="input 0.1 s, 8 fr."').replace('label="model, 0.033 s"', 'label="model 0.033 s"').replace('label="input, 0.033 s"', 'label="input 0.033 s"')
            .replace('label="low-freq. error, k-frame mean"', 'label="LF error, $k$-mean"').replace('label="high-freq. error"', 'label="HF error"').replace('label="1/k, pure variance"', 'label="$1/k$"')
            .replace('ncol=2, handlelength=1.6, columnspacing=0.8, labelspacing=0.25', 'ncol=2, handlelength=1.2, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.1'))
plot_src = plot_src.replace('ax.legend(frameon=False, fontsize=9.4, loc="upper left", labelspacing=0.25)\ndef p3', 'ax.legend(frameon=False, fontsize=9.4, loc="lower right", labelspacing=0.25)\ndef p3')
plot_src = plot_src.replace('ax.set_yticks([0, 1, 2]); ax.legend(frameon=False, fontsize=9.4, loc="upper left", labelspacing=0.25)', 'ax.set_yticks([0, 1, 2]); ax.legend(frameon=False, fontsize=9.4, loc="upper right", labelspacing=0.25)')
exec(plot_src)                                                     # defines p1, p2, p3 (same data, same styling)
img = lambda n: np.asarray(Image.open(os.path.join(HERE, n)).convert("RGB"))
P1, P2, P3, PR = img("P1_input.png"), img("P2_output.png"), img("P3_gt.png"), img("P_res_raw.png")
PL, PB, PH = img("P_lf_raw.png"), img("P_blk_raw.png"), img("P_hf_raw.png")
W, H = 7.008, 2.0
fig = plt.figure(figsize=(W, H), dpi=300)
# ---- row 1: panels
top, bot, gap = 0.91, 0.715, 0.012
pw = 0.128; gp = 0.036; xs = [0.005 + k * (pw + gp) for k in range(4)]
titles = ["dark frame $x$", "output $y=f(x)$", "reference $g$", "residual $a\\,y-g$"]
axes1 = []
for x, im, ti, col in zip(xs, (P1, P2, P3, PR), titles, (NAVY, TEAL, NAVY, NAVY)):
    ax = fig.add_axes([x, bot, pw, top - bot]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.2)
    ax.set_title(ti, fontsize=9.6, fontweight="bold", pad=4); axes1.append(ax)
def arrow(x0, x1, y, label=None):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=12, lw=2.0, color=TEAL))
    if label: fig.text((x0 + x1) / 2, y + 0.035, label, ha="center", va="bottom", fontsize=10, style="italic", color=TEAL)
ym = (top + bot) / 2
arrow(xs[0] + pw + 0.004, xs[1] - 0.004, ym, "f"); arrow(xs[1] + pw + 0.004, xs[2] - 0.004, ym); arrow(xs[2] + pw + 0.004, xs[3] - 0.004, ym)
# decomposition stack
x4 = xs[3] + pw + 0.038; sw = 0.058; sh = (top - bot - 2 * 0.02) / 3
for k, (im, lab, col) in enumerate(((PL, "low-freq.", NAVY), (PB, "block gain", (0.2, 0.55, 0.25)), (PH, "high-freq.", ORANGE))):
    ax = fig.add_axes([x4, top - (k + 1) * sh - k * 0.02, sw, sh]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); axes1.append(ax)
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(1.0)
    fig.text(x4 + sw + 0.006, top - (k + 0.5) * sh - k * 0.02, lab, ha="left", va="center", fontsize=9.6)
fig.text(x4 + sw / 2 + 0.035, top + 0.014, "decompose", ha="center", va="bottom", fontsize=9.6, fontweight="bold")
arrow(xs[3] + pw + 0.004, x4 - 0.004, ym)
fig.text(x4, bot - 0.1, "$a=\\langle y,g\\rangle/\\langle y,y\\rangle$, the best single gain", ha="left", va="top", fontsize=9.6, style="italic", color=DARK)
# verdict strips
x5 = x4 + sw + 0.105; vw = 0.992 - x5
fig.text(x5 + vw / 2, top + 0.014, "band verdicts", ha="center", va="bottom", fontsize=9.6, fontweight="bold")
STRIPS = []
def strip(y, exp):
    vs = V[exp]; cw = vw / len(vs)
    fig.text(x5, y + 0.076, exp.replace("s", " s"), ha="left", va="bottom", fontsize=9.4)
    for k, vv in enumerate(vs): fig.add_artist(Rectangle((x5 + k * cw + 0.002, y), cw - 0.004, 0.058, transform=fig.transFigure, color=COL[vv])); STRIPS.append((x5 + k * cw + 0.002, y, x5 + (k + 1) * cw - 0.002, y + 0.058))
strip(top - 0.12, "0.033s"); strip(top - 0.275, "0.1s")
arrow(x4 + sw + 0.066, x5 + 0.012, top + 0.054)
# ---- row 2: plots
pb, pt = 0.205, 0.455
def p1b(ax):
    p1(ax); lg = ax.get_legend(); lg and lg.remove(); ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.5, 1])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.005, pt + 0.012), ncol=2, frameon=False, fontsize=9.2, handlelength=1.3, columnspacing=0.6, labelspacing=0.15, borderpad=0.2, borderaxespad=0.0)
def p2b(ax):
    p2(ax); lg = ax.get_legend(); lg and lg.remove(); ax.set_ylim(0, 2.1); ax.set_yticks([0, 1, 2])
    h, l = ax.get_legend_handles_labels()                          # legend above the axes, like the first plot
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.075 + 0.335, pt + 0.012), ncol=1, frameon=False, fontsize=9.2, handlelength=1.3, labelspacing=0.15, borderpad=0.2, borderaxespad=0.0)
def p3b(ax):
    p3(ax); lg = ax.get_legend(); lg and lg.remove(); ax.set_ylim(0, 2.3); ax.set_yticks([0, 1, 2]); ax.set_xlim(0.92, 8 * 2.05); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
    short = {"LF error, $k$-mean": ("LF", 0.06), "HF error": ("HF", -0.06), "$1/k$": ("$1/k$", 0.0)}
    for ln in ax.get_lines():
        lab = ln.get_label()
        if lab in short:
            yy = float(np.asarray(ln.get_ydata())[-1]); ax.text(8.7, yy + short[lab][1], short[lab][0], ha="left", va="center", fontsize=9.4, color=ln.get_color())
for k, fn in enumerate((p1b, p2b, p3b)):
    ax = fig.add_axes([0.075 + k * 0.335, pb, 0.255, pt - pb]); fn(ax)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
OUT = os.path.join(HERE, "fig_workflow.pdf")
fig.savefig(OUT); print("saved fig_workflow.pdf", W, H)

# ---- self-check on the saved PDF: no foreign coloured ink inside any text box, no black-text / coloured-curve overlap
import subprocess, xml.etree.ElementTree as ET
from scipy import ndimage
DPI = 600; png = os.path.join(HERE, "fig_workflow_check")
subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", OUT, png], check=True)
im = np.asarray(Image.open(png + ".png").convert("RGB")).astype(int); Hp, Wp = im.shape[:2]
mx, mn = im.max(2), im.min(2); sat = mx - mn
coloured = (sat > 60) & (mx < 252); black = (mx < 110) & (sat < 40)
RECTS = list(STRIPS)
for ax in axes1:                                                   # photographs and residual maps are coloured by nature: mask their interior (spines stay)
    bb = ax.get_position(); RECTS.append((bb.x0, bb.y0, bb.x1, bb.y1))
    x0, x1 = int(bb.x0 * Wp) + 2, int(bb.x1 * Wp) - 2; y0, y1 = int((1 - bb.y1) * Hp) + 2, int((1 - bb.y0) * Hp) - 2
    coloured[y0:y1, x0:x1] = False
root = ET.fromstring(subprocess.run(["mutool", "draw", "-F", "stext", OUT], capture_output=True, text=True).stdout)
sc = DPI / 72.0; log = []; worst_gap = 1e9; foreign_total = 0
for ln in root.iter("line"):
    txt = "".join(ch.get("c", "") for ch in ln.iter("char")).strip()
    if not txt: continue
    bx0, by0, bx1, by1 = (float(v) for v in ln.get("bbox").split())
    px0, px1 = int(bx0 * sc), int(np.ceil(bx1 * sc)); py0, py1 = int(by0 * sc), int(np.ceil(by1 * sc))
    fx0, fx1, fy0, fy1 = bx0 / (W * 72), bx1 / (W * 72), 1 - by1 / (H * 72), 1 - by0 / (H * 72)   # box in figure fraction
    for (rx0, ry0, rx1, ry1) in RECTS:                             # geometric: no text box may enter a photograph, residual map or verdict cell
        assert fx1 <= rx0 or fx0 >= rx1 or fy1 <= ry0 or fy0 >= ry1, (txt, "text box enters an image or verdict cell", (round(fx0, 3), round(fy0, 3), round(fx1, 3), round(fy1, 3)), tuple(round(v, 3) for v in (rx0, ry0, rx1, ry1)))
    box_c = coloured[py0:py1, px0:px1]; box_b = black[py0:py1, px0:px1]
    nb, nc = int(box_b.sum()), int(box_c.sum())
    if nb >= nc: foreign = nc                                      # black text: any coloured pixel inside the box is foreign
    else:                                                          # coloured label (LF/HF/1/k, f): its own colour is the median of its pixels
        pix = im[py0:py1, px0:px1]; own = np.median(pix[box_c], axis=0)          # anti-aliased edges of the own colour lie on the white-to-own line
        v = 255 - pix; o = 255 - own; cos = (v * o).sum(2) / (np.sqrt((v * v).sum(2)) * np.sqrt((o * o).sum()) + 1e-9)
        foreign = int((box_c & (cos < 0.97)).sum())
    h = py1 - py0; band0, band1 = max(py0 - h, 0), min(py1 + h, Hp); gap = None
    if nb >= nc and nb > 0:                                        # column-wise clearance between the black glyph ink and coloured ink nearby
        for x in range(px0, px1):
            rb = np.flatnonzero(black[py0:py1, x]); rc = np.flatnonzero(coloured[band0:band1, x])
            if len(rb) and len(rc):
                dmin = np.abs((rb[:, None] + py0) - (rc[None, :] + band0)).min() - 1
                gap = dmin if gap is None else min(gap, dmin)
    gap_bp = None if gap is None else gap / sc
    if gap_bp is not None: worst_gap = min(worst_gap, gap_bp)
    foreign_total += foreign
    log.append(f"{txt!r:32s} foreign_coloured_px={foreign:4d} min_gap_bp={'-' if gap_bp is None else f'{gap_bp:.2f}'}")
# black-black clearance: text boxes that overlap horizontally must keep their glyph ink apart vertically
TB = []
for ln in root.iter("line"):
    txt = "".join(ch.get("c", "") for ch in ln.iter("char")).strip()
    if txt: TB.append((txt, tuple(float(v) for v in ln.get("bbox").split())))
bb_worst = 1e9
for i in range(len(TB)):
    for j in range(len(TB)):
        (ta, (ax0, ay0, ax1, ay1)), (tb, (bx0, by0, bx1, by1)) = TB[i], TB[j]
        if i == j or (by0 + by1) <= (ay0 + ay1) or by0 - ay1 > 4.0 or min(ax1, bx1) - max(ax0, bx0) <= 0: continue   # b centred below a (font boxes may overlap), within 4 bp, overlapping columns
        cx0, cx1 = int(max(ax0, bx0) * sc), int(np.ceil(min(ax1, bx1) * sc)); y0, y1 = int(min(ay0, by0) * sc), int(np.ceil(max(ay1, by1) * sc))
        sub = black[y0:y1, cx0:cx1]; lab, ncomp = ndimage.label(sub)
        if ncomp == 0: continue
        ma, mb = np.zeros_like(sub), np.zeros_like(sub); ca, cb = (ay0 + ay1) / 2 * sc - y0, (by0 + by1) / 2 * sc - y0
        for k in range(1, ncomp + 1):                                # each glyph component belongs to the box whose centre is nearer
            comp = lab == k; cy = np.nonzero(comp)[0].mean()
            (ma if abs(cy - ca) <= abs(cy - cb) else mb)[comp] = True
        g = None                                                    # column-wise: only ink in the same column can touch
        for x in range(sub.shape[1]):
            c1, c2 = np.flatnonzero(ma[:, x]), np.flatnonzero(mb[:, x])
            if len(c1) and len(c2):
                d = (c2.min() - c1.max() - 1) / sc; g = d if g is None else min(g, d)
        if g is None: continue
        bb_worst = min(bb_worst, g); log.append(f"black-black {ta!r} / {tb!r} ink_gap_bp={g:.2f}")
GAP_MIN = 0.5                                                      # bp: clearance a black glyph must keep from coloured ink
BB_MIN = 1.0                                                       # bp: clearance black text must keep from neighbouring black text (typographic, not just non-touching)
log.append(f"boxes={len(TB)} foreign_total={foreign_total} worst_gap_bp={'-' if worst_gap == 1e9 else f'{worst_gap:.2f}'} worst_black_black_bp={'-' if bb_worst == 1e9 else f'{bb_worst:.2f}'} (min {GAP_MIN}/{BB_MIN}) rects_checked={len(RECTS)} -> {'PASS' if foreign_total == 0 and worst_gap >= GAP_MIN and bb_worst >= BB_MIN else 'FAIL'}")
open(os.path.join(HERE, "fig1_check.log"), "w").write("\n".join(log) + "\n"); print(log[-1])
os.remove(png + ".png")
assert foreign_total == 0 and worst_gap >= GAP_MIN and bb_worst >= BB_MIN, "fig_workflow.pdf: text overlaps or nearly touches other ink, see fig1_check.log"
