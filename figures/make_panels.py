"""Step 1 of the figure recipe: every photo slot and fact panel rendered from real data,
plus a wireframe that fixes the layout, the boxes, the arrows and the label positions.
Photo slots are filled with a flat magenta (#FFCCFF) so they can be detected and the real
panels composited in afterwards.
"""
import os, sys, json, numpy as np, torch
CLEAN = "--clean" in sys.argv
sys.path.insert(0, "code")
sys.path.insert(0, "third_party/Retinexformer")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

OUT = "figures"
E = json.load(open("paper_tables/evidence.json"))
R = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
SC, EXPO = "10198", "0.033s"
net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
net.load_state_dict(torch.load((os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth"), map_location="cpu")["params"])
net = net.cuda().eval()

# ---- real photo slots P1..P4
sf = [f for f in sorted(os.listdir(f"{R}/short_sid2/{SC}")) if EXPO in f][0]
gf = sorted(os.listdir(f"{R}/long_sid2/{SC}"))[0]
# the benchmark npy files are stored BGR; the authors' loader flips to RGB before the model,
# and the reproduction (24.438 dB) used that convention -- so must every figure
lq = np.load(f"{R}/short_sid2/{SC}/{sf}")[:, :, ::-1].astype(np.float32) / 255.
gt = np.load(f"{R}/long_sid2/{SC}/{gf}")[:, :, ::-1].astype(np.float32) / 255.
with torch.inference_mode():
    y = torch.clamp(net(torch.from_numpy(lq.transpose(2, 0, 1))[None].cuda()), 0, 1)[0].cpu().numpy().transpose(1, 2, 0)
a = float((y * gt).sum() / (y * y).sum()); res = a * y - gt
F = np.fft.fftshift(np.fft.fft2(res.mean(2))); H, W = F.shape
u = np.fft.fftshift(np.fft.fftfreq(H))[:, None]; v = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
F[np.sqrt((2 * u) ** 2 + (2 * v) ** 2) > 0.10] = 0
lf = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
def save_img(arr, name):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(f"{OUT}/{name}.png")
save_img(lq * 20, "P1_input"); save_img(y, "P2_output"); save_img(gt, "P3_gt")
print("figure frame PSNR (canonical convention): %.2f dB" % (10 * np.log10(1.0 / np.mean((np.clip(y, 0, 1) - gt) ** 2))))
m = np.abs(lf).max(); cm = plt.get_cmap("coolwarm")((lf / m + 1) / 2)[:, :, :3]; save_img(cm, "P4_lfres")

# ---- fact panels F1..F3 from evidence.json only
ed = E["ident_edges"]; c = [(ed[i] + ed[i + 1]) / 2 for i in range(len(ed) - 1)]
def panel(fn, name):
    fig, ax = plt.subplots(figsize=(2.6, 1.55), dpi=300); fn(ax); fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)
def f1(ax):
    for e, col in (("0.033s", "k"), ("0.1s", "0.5")):
        r = E["rho"][e]; ax.plot(c, r["model"], "-", color=col, lw=1, label=f"model {e}")
        ax.plot(c, r["in_k1"], ":", color=col, lw=1, label=f"input {e}")
    ax.plot(c, E["rho"]["0.1s"]["in_k8"], "--", color="0.5", lw=1, label="input 0.1s k=8")
    ax.set_ylim(0, 1.02); ax.set_xlabel("radial frequency", fontsize=6); ax.set_ylabel(r"$\rho$", fontsize=6)
    ax.tick_params(labelsize=5); ax.legend(fontsize=4, frameon=False, loc="lower left")
def f2(ax):
    L = E["ladder"]; names = [r["block"].replace("global(1x1)", "frame") for r in L]; xs = range(len(L))
    ax.plot(xs, [r["gain_dof"] for r in L], "o-", color="k", ms=2, lw=1, label="gain")
    ax.plot(xs, [r["affine_dof"] for r in L], "s--", color="0.45", ms=2, lw=1, label="affine")
    ax.set_xticks(list(xs)); ax.set_xticklabels(names, fontsize=5, rotation=30); ax.set_ylabel("dB over frame fit", fontsize=6)
    ax.tick_params(labelsize=5); ax.legend(fontsize=5, frameon=False)
def f3(ax):
    rows = E["mf"]["rows"]; ks = sorted(int(k) for k in rows); lf_ = [rows[str(k)]["lf_ratio"] for k in ks]
    ax.plot(ks, lf_, "o-", color="k", ms=2.5, lw=1, label="LF error, input mean")
    ax.plot(ks, [1 / k for k in ks], ":", color="0.5", lw=1, label="1/k (pure variance)")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=5)
    ax.set_xlabel("frames averaged k", fontsize=6); ax.set_ylabel("rel. to k=1", fontsize=6)
    ax.tick_params(labelsize=5); ax.legend(fontsize=4.5, frameon=False)
panel(f1, "F1_rho"); panel(f2, "F2_ladder"); panel(f3, "F3_kavg")

# ---- wireframe: wide canvas, tall top row so labels stay legible at column width
Wc, Hc = 3360, 1180
im = Image.new("RGB", (Wc, Hc), "white"); d = ImageDraw.Draw(im)
MAG, NAVY, TEAL, ORANGE = (255, 204, 255), (20, 40, 90), (0, 110, 110), (200, 90, 0)
G = {}                                                    # geometry shared with the label pass
def box(x0, y0, x1, y1, key, col=NAVY):
    d.rectangle([x0, y0, x1, y1], outline=col, width=6); G[key] = [x0, y0, x1, y1]
def slot(x0, y0, x1, y1, key):
    d.rectangle([x0, y0, x1, y1], fill=MAG, outline=(120, 120, 120), width=3); G[key] = [x0, y0, x1, y1]
def arrow(x0, y0, x1, y1, col=TEAL, w=7):
    import math
    d.line([x0, y0, x1, y1], fill=col, width=w); ang = math.atan2(y1 - y0, x1 - x0)
    for s_ in (0.5, -0.5):
        d.line([x1, y1, x1 - 30 * math.cos(ang - s_), y1 - 30 * math.sin(ang - s_)], fill=col, width=w)
ty, sh, sw, bw = 70, 330, 400, 330; cy = ty + sh // 2
x = 30
slot(x, ty, x + sw, ty + sh, "P1"); x += sw
arrow(x + 8, cy, x + 62, cy); x += 72
box(x, ty, x + bw, ty + sh, "model"); x += bw
arrow(x + 8, cy, x + 62, cy); x += 72
slot(x, ty, x + sw, ty + sh, "P2"); ox1 = x + sw; x += sw + 24
slot(x, ty, x + sw, ty + sh, "P3"); x += sw
arrow(x + 8, cy, x + 62, cy); x += 72
box(x, ty, x + bw, ty + sh, "residual"); rx0 = x; x += bw
d.line([ox1 + 12, cy, ox1 + 12, ty - 34, rx0 + 50, ty - 34], fill=TEAL, width=7, joint="curve"); arrow(rx0 + 50, ty - 34, rx0 + 50, ty - 6)
fx0 = x + 10; bx = fx0 + 60; bwd = 300
for key, by_ in (("gain", ty - 30), ("block", ty + 110), ("bands", ty + 250)):
    arrow(fx0, cy, bx - 6, by_ + 50); box(bx, by_, bx + bwd, by_ + 100, key)
gx = bx + bwd + 70; gw = Wc - gx - 30
box(gx, ty - 30, gx + gw, ty + 130, "ceiling", col=ORANGE)
box(gx, ty + 170, gx + gw, ty + 350, "ident", col=ORANGE)
for by_ in (ty - 30, ty + 110, ty + 250):
    arrow(bx + bwd + 6, by_ + 50, gx - 6, ty + 50); arrow(bx + bwd + 6, by_ + 50, gx - 6, ty + 260)
slot(bx, ty + 370, bx + 300, ty + 370 + 168, "P4")
py, ph = 640, 470
for i, key in enumerate(("F1", "F2", "F3")):
    x0 = 30 + i * 1110; slot(x0, py, x0 + 1080, py + ph, key)
im.save(f"{OUT}/" + ("wire_clean.png" if CLEAN else "wireframe.png"))
json.dump(G, open(f"{OUT}/geometry.json", "w"), indent=1)
print("panels + wireframe written to", OUT, "| canvas", Wc, Hc)
