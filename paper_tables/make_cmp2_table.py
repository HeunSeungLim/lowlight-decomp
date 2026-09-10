"""표 1(두 벤치마크 비교)의 행을 영수증에서 굽는다. 손으로 친 숫자를 남기지 않기 위한 것이고,
강조(최고 굵게 / 차상위 밑줄)도 값에서 계산한다."""
import json, os, re

P = os.path.dirname(os.path.abspath(__file__))
def _method_dir_common(P):
    for c in (os.environ.get("LLMETHOD"),
              os.path.join(P, "..", "method_260909"),
              os.path.join(P, "..", "numbers", "method_260909"),
              ):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
M = _method_dir_common(P)
def _root():
    for c in (os.environ.get("LOWLIGHT_EVIDENCE_ROOT"),
              os.path.join(P, "..", "numbers"), os.path.join(P, "..", "repro"),
              ):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "repro")
R = _root()

def _load(*cands):
    for c in cands:
        if os.path.exists(c): return json.load(open(c))
    raise FileNotFoundError(cands[0])

CMP = _load(os.path.join(P, "compare_methods.json"), os.path.join(R, "compare_methods.json"))
AL = _load(os.path.join(P, "anchor_lol_retinexformer.json"), os.path.join(M, "anchor_lol_retinexformer.json"))
AS = _load(os.path.join(P, "anchored_sony_sd.json"), os.path.join(M, "anchored_sony_sd.json"))
DA = _load(os.path.join(P, "decomp_anchored.json"), os.path.join(M, "decomp_anchored.json"))
FR = _load(os.path.join(P, "final_rule.json"), os.path.join(M, "final_rule.json"))

AGG = CMP["agg"]

# 행 순서는 원고가 정한 것이고, 숫자는 전부 영수증에서 온다
LOL = ["input", "clahe", "zerodcepp", "sci", "ar", "lightendiff", "uretinex", "snrnet",
       "llformer", "gsad", "retinexformer", "cidnet_woperc"]
SONY = ["input", "clahe", "zerodcepp", "sci", "ar", "lightendiff", "snrnet", "retinexformer"]
LABEL = {"input": "Input", "clahe": "CLAHE", "zerodcepp": "Zero-DCE++", "sci": "SCI", "ar": "AR (in-house)",
         "lightendiff": "LightenDiffusion", "uretinex": "URetinex-Net", "snrnet": "SNR-Net",
         "llformer": "LLFormer", "gsad": "GSAD", "retinexformer": "Retinexformer", "cidnet_woperc": "CIDNet"}

def row(bench, key):
    a = AGG[f"{bench}/{key}"]
    return dict(label=LABEL[key], psnr=a["psnr"], sd=a["psnr_sd"], ssim=a["ssim"],
                gl=a["glob"], ch=a["chan"], res=a["resid"], d16=a["d16"])

lol = [row("LOL", k) for k in LOL]
lol.append(dict(label="+anc.\\ (ours)", psnr=AL["anchored"]["psnr"], sd=AL["anchored"]["psnr_sd"],
                ssim=AL["anchored"]["ssim"], gl=AL["anchored"]["share_global_pct"],
                ch=AL["anchored"]["share_channel_pct"], res=AL["anchored"]["share_residual_pct"],
                d16=AL["anchored"]["d16"]))
sony = [row("Sony", k) for k in SONY]
sony.append(dict(label="+anc.\\ (ours)", psnr=AS["anchored_mean"], sd=AS["anchored_sd"],
                 ssim=FR["sony"]["retinexformer"]["ssim"], gl=DA["anchored"]["share_global_pct"],
                 ch=DA["anchored"]["share_channel_pct"], res=DA["anchored"]["share_residual_pct"],
                 d16=DA["anchored"]["d16"]))

def emph(rows):
    """최고는 굵게, 차상위는 밑줄. 캡션이 선언한 규칙 그대로 값에서 정한다."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["psnr"])
    for rank, i in enumerate(order):
        rows[i]["mark"] = ("bf" if rank == 0 else "ul" if rank == 1 else "")
    return rows

def cell(r):
    p = f"{r['psnr']:.2f}$\\pm${r['sd']:.2f}"
    if r["mark"] == "bf": p = "\\textbf{" + p + "}"
    elif r["mark"] == "ul": p = p + "$^{\\ddagger}$"   # 밑줄은 다음 행 숫자에 닿고, 단검은 표 2 가 불안정 셀에 쓴다. 겹치지 않는 기호를 쓴다.
    return f"{r['label']} & {p} & {r['ssim']:.3f} & {r['gl']:.1f} & {r['ch']:.1f} & {r['res']:.1f} & {r['d16']:.2f}"

lol, sony = emph(lol), emph(sony)
lines = []
for i in range(max(len(lol), len(sony))):
    left = cell(lol[i]) if i < len(lol) else " &  &  &  &  &  & "
    right = cell(sony[i]) if i < len(sony) else " &  &  &  &  &  & "
    lines.append(f"{left} & {right} \\\\")
out = "\n".join(lines) + "\n"
open(os.path.join(P, "tab_cmp2_rows.tex"), "w").write(out)
print(f"tab_cmp2_rows.tex: LOL {len(lol)}행, Sony {len(sony)}행")
