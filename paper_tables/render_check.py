"""Render every page at 300 dpi and report the vertical ink layout per column: blank gaps above 130 px
(empty areas) and gaps under 5 px between separate ink runs (blocks touching). Writes render_<tag>/ink_check.log."""
import sys, os, glob, subprocess, numpy as np
from PIL import Image
P = os.path.dirname(os.path.abspath(__file__)); tag = sys.argv[1] if len(sys.argv) > 1 else "check"
out = os.path.join(P, f"render_{tag}"); os.makedirs(out, exist_ok=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", os.path.join(P, "main.pdf"), os.path.join(out, "p")], check=True)
lines = []
for f in sorted(glob.glob(os.path.join(out, "p-*.png"))):
    im = np.asarray(Image.open(f).convert("L")); H, W = im.shape; ink = im < 200
    assert (W, H) == (2550, 3300), (f, W, H)
    for name, (x0, x1) in (("L", (200, 1260)), ("R", (1290, 2350)), ("full", (200, 2350))):
        rows = ink[:, x0:x1].any(1); runs = []; s = None
        for y, v in enumerate(rows):
            if v and s is None: s = y
            if not v and s is not None: runs.append((s, y)); s = None
        if s is not None: runs.append((s, H))
        gaps = [(runs[i + 1][0] - runs[i][1], runs[i][1]) for i in range(len(runs) - 1)]
        big = [(g, y) for g, y in gaps if g > 130]; tiny = [(g, y) for g, y in gaps if 0 < g < 5]
        lines.append(f"{os.path.basename(f)} {name:4s} ink {runs[0][0]}-{runs[-1][1]} gaps>130px={big} gaps<5px={tiny}")
open(os.path.join(out, "ink_check.log"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
