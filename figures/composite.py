"""Step 3 of the figure recipe: detect the magenta windows in the wireframe
background, paste the real panels into them in reading order, and overlay Times labels.
Usage: python composite.py <background.png> <out_prefix>
"""
import sys, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = "paper_tables/figures/workflow"
bg = Image.open(sys.argv[1]).convert("RGB"); out = sys.argv[2]
a = np.array(bg).astype(int)
# pale magenta #FFCCFF with tolerance; also accept the wireframe's exact fill
mask = (abs(a[:, :, 0] - 255) < 25) & (abs(a[:, :, 1] - 204) < 40) & (abs(a[:, :, 2] - 255) < 25)
lab, n = ndimage.label(mask)
boxes = []
for i in range(1, n + 1):
    ys, xs = np.where(lab == i)
    if len(ys) < 0.002 * mask.size: continue                    # ignore specks
    boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
print("magenta windows found:", len(boxes), [(b[2]-b[0], b[3]-b[1]) for b in boxes])
order = ["P1_input", "P2_output", "P3_gt", "P4_lfres", "F1_rho", "F2_ladder", "F3_kavg"]
KEY = {"P1_input": "P1", "P2_output": "P2", "P3_gt": "P3", "P4_lfres": "P4", "F1_rho": "F1", "F2_ladder": "F2", "F3_kavg": "F3"}
import os
if os.path.exists(f"{HERE}/geometry.json") and bg.size == (3360, 1180):
    # code-drawn surface: assign each detected window to the wireframe slot whose centre is nearest
    G = json.load(open(f"{HERE}/geometry.json"))
    def centre(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    assigned = []
    for name in order:
        gx0, gy0, gx1, gy1 = G[KEY[name]]; gc = ((gx0 + gx1) / 2, (gy0 + gy1) / 2)
        b = min(boxes, key=lambda bb: (centre(bb)[0] - gc[0]) ** 2 + (centre(bb)[1] - gc[1]) ** 2)
        assigned.append(b)
    boxes = assigned; print("placement: by geometry.json")
else:
    boxes.sort(key=lambda b: (round(b[1] / (bg.height * 0.25)), b[0])); print("placement: reading order")
labels = ["input frame (0.033 s, shown ×20)", "model output", "reference", "low-frequency residual",
          "band-wise agreement", "local correction ceiling", "frame averaging"]
if len(boxes) != len(order):
    print("WARNING: window count mismatch; compositing the first", min(len(boxes), len(order)))
im = bg.copy(); d = ImageDraw.Draw(im)
TG = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf"
try: font = ImageFont.truetype(TG, max(26, bg.height // 36))
except Exception: font = ImageFont.load_default()
placed = []
for (x0, y0, x1, y1), name, lab_ in zip(boxes, order, labels):
    p = Image.open(f"{HERE}/{name}.png").convert("RGB")
    w, h = x1 - x0 + 1, y1 - y0 + 1
    # fit inside the window preserving aspect, centred, white letterbox
    s = min(w / p.width, h / p.height); pw, ph = int(p.width * s), int(p.height * s)
    im.paste(Image.new("RGB", (w, h), "white"), (x0, y0))
    im.paste(p.resize((pw, ph), Image.LANCZOS), (x0 + (w - pw) // 2, y0 + (h - ph) // 2))
    d.text((x0 + (w // 2), y1 + 6), lab_, fill="black", font=font, anchor="ma")
    placed.append(dict(name=name, box=[int(x0), int(y0), int(x1), int(y1)]))
im.save(f"{out}.png"); json.dump(placed, open(f"{out}_placement.json", "w"), indent=1)
print("saved", f"{out}.png", im.size)
