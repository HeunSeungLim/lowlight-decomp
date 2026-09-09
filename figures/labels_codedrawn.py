"""Times labels for the code-drawn surface, placed from geometry.json written by make_panels."""
import json
from PIL import Image, ImageDraw, ImageFont
TG = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
fB = ImageFont.truetype(TG + "texgyretermes-bold.otf", 44)
fBs = ImageFont.truetype(TG + "texgyretermes-bold.otf", 38)
fM = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf", 40)
fR = ImageFont.truetype(TG + "texgyretermes-regular.otf", 34)
NAVY, ORANGE = (20, 40, 90), (200, 90, 0)
G = json.load(open("geometry.json"))
im = Image.open("codedrawn.png").convert("RGB"); d = ImageDraw.Draw(im)
def title(key, t, col, sub=None, sub2=None, f=fB):
    x0, y0, x1, y1 = G[key]; d.text((x0 + 16, y0 + 12), t, font=f, fill=col)
    if sub: d.text((x0 + 16, y0 + 12 + f.size + 12), sub, font=fM, fill="black")
    if sub2: d.text((x0 + 16, y0 + 12 + f.size + 12 + fM.size + 10), sub2, font=fM if key != "ident" else fR, fill="black")
title("model", "frozen model f", NAVY, "y = f(x)")
title("residual", "residual", NAVY, "r = a y − g", "a = ⟨y,g⟩ / ⟨y,y⟩")
title("gain", "global / channel gain", NAVY, "a, a_c", f=fBs)
title("block", "block field (16 px)", NAVY, "a_blk", f=fBs)
title("bands", "radial bands", NAVY, "ρ_j", f=fBs)
title("ceiling", "ceiling with DoF control", ORANGE, "Δ = PSNR(oracle) − PSNR(frame) − Δ_white")
title("ident", "identifiability", ORANGE, "ρ_out vs ρ_in(k)", "recoverable / exhausted / unidentifiable")
im.save("fig_workflow_codedrawn.png"); print("saved fig_workflow_codedrawn.png", im.size)
