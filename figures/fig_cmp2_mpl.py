"""기존 방법 정성 비교에 우리 보정 칸을 추가한 판. 위: 입력·방법 출력·보정·기준, 아래: 각 잔차의 저주파 성분.
프레임과 색 범위·저주파 규약은 기존 그림과 동일하게 유지한다."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"]=42
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); P=os.path.dirname(HERE); R=os.environ.get("LLROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
M=f"{R}/method_260909"; CODEX=os.environ.get("LLCACHE", "cache")
GTD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/long_sid2"; SHD=(os.environ.get("LLDATA", "data") + "/lowlight_model") + "/data/SID_raw/SID/short_sid2"
plt.rcParams.update({"font.family":"serif","font.serif":["Nimbus Roman","TeX Gyre Termes","Liberation Serif"],"font.size":9.2,"mathtext.fontset":"stix"})
def _find(name):
    """번들·공개 레이아웃 어느 쪽에서든 생성된 수치 파일을 찾는다."""
    for d in (P, os.path.join(P, "paper_tables"), os.path.join(HERE, "..", "..")):
        c = os.path.join(d, name)
        if os.path.exists(c):
            return c
    return os.path.join(P, name)
J=json.load(open(_find("fig_cmp_numbers.json"))); Sy=J.get("Sony",J); FR=Sy["frame"]; SC=FR.split("_")[0]; V=Sy["color_range"]
rows=json.load(open(f"{R}/numbers/compare_methods.json"))["per_frame"]["Sony"]["retinexformer"]
IDX={r["id"]: i for i,r in enumerate(rows)}
K=1.0; QANC=99.9                                     # 8비트 상한 앵커: 데이터 없이 정해진다
gt=np.load(f"{GTD}/{SC}/"+[x for x in sorted(os.listdir(f'{GTD}/{SC}')) if x.endswith('.npy')][0])[:,:,::-1].astype(np.float32)/255.
x=np.load(f"{SHD}/{SC}/{FR}")[:,:,::-1].astype(np.float32)/255.
def out(m):
    if m=="retinexformer": return np.load(f"{CODEX}/{IDX[FR]:04d}.npy").transpose(1,2,0).astype(np.float32)
    a=np.load(f"{R}/numbers/cache_{m}/Sony/{FR}.npy"); return (a.transpose(1,2,0) if a.shape[0]==3 else a).astype(np.float32)
p8=lambda a,b:10*np.log10(255.0**2/np.mean((a.astype(np.float64)-b.astype(np.float64))**2)); g8=lambda v:np.rint(np.clip(v,0,1)*255).astype(np.uint8)
def lf(y, g):                                          # 오라클 전역이득 후 잔차의 저주파 성분 (luma, f<0.10)
    a=float((y*g).sum()/(y*y).sum()); d=(a*y-g)*255.0
    L=0.299*d[:,:,0]+0.587*d[:,:,1]+0.114*d[:,:,2]
    F=np.fft.fftshift(np.fft.fft2(L)); H,W=L.shape
    v,u=np.meshgrid(np.arange(H)-H/2, np.arange(W)-W/2, indexing="ij")
    m=(np.sqrt((v/(H/2))**2+(u/(W/2))**2)<0.10); return np.real(np.fft.ifft2(np.fft.ifftshift(F*m)))
cols=[("input", np.clip(x*Sy["methods"]["input"]["gain"],0,1), f"input ($\\times${Sy['methods']['input']['gain']:.1f})\n{Sy['methods']['input']['psnr_shown']:.1f} dB", x),
      ("zerodcepp", None, None, None), ("snrnet", None, None, None), ("lightendiff", None, None, None), ("retinexformer", None, None, None)]
panels=[]
panels.append((np.clip(x*Sy["methods"]["input"]["gain"],0,1), f"input ($\\times${Sy['methods']['input']['gain']:.1f})\n{Sy['methods']['input']['psnr_shown']:.1f} dB", lf(x,gt)))
NAME={"zerodcepp":"Zero-DCE++","snrnet":"SNR-Net","lightendiff":"LightenDiff.","retinexformer":"Retinexformer"}
for m in ("zerodcepp","snrnet","lightendiff","retinexformer"):
    y=out(m); panels.append((np.clip(y,0,1), f"{NAME[m]}\n{p8(g8(y),g8(gt)):.1f} dB", lf(y,gt)))
yr=out("retinexformer"); gcal=float(np.clip(K/max(float(np.percentile(yr,QANC)),1e-6),0.5,2.0)); yc=np.clip(yr*gcal,0,1)
panels.append((yc, f"+anc. (ours)\n{p8(g8(yc),g8(gt)):.1f} dB", lf(yc,gt)))
panels.append((gt, "reference", None))
W_,H_=7.008,1.38; fig=plt.figure(figsize=(W_,H_),dpi=300)
n=len(panels); RIGHT=0.104; pw=(1.0-RIGHT-0.004*(n-1))/n; ph=0.355
for k,(im,t,d) in enumerate(panels):
    xx=0.004+k*(pw+0.004)
    ax=fig.add_axes([xx,0.432,pw,ph]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.5)
    ax.set_title(t, fontsize=9.2, pad=2.6, linespacing=1.25)
    if d is not None:
        ax2=fig.add_axes([xx,0.068,pw,ph]); im2=ax2.imshow(d, cmap="coolwarm", vmin=-V, vmax=V); ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values(): sp.set_linewidth(0.5)
cax=fig.add_axes([1.0-RIGHT+0.010,0.068,0.009,ph]); cb=fig.colorbar(im2, cax=cax); cb.set_ticks([-100,0,100]); cb.ax.set_yticklabels(["$-$100","0","100"]); cb.ax.tick_params(labelsize=9.2, length=1.5, pad=1)
cb.set_label("LF residual", fontsize=9.2, labelpad=1)
rec=dict(frame=FR, scene=SC, calibration_gain=gcal, panels={t.split("\n")[0]: float(t.split("\n")[-1].replace(" dB","")) for _,t,_ in panels if t and "dB" in t},
         calibrated_psnr=p8(g8(yc),g8(gt)), retinexformer_psnr=p8(g8(yr),g8(gt)), color_range=V)
json.dump(rec, open(f"{P}/fig_cmp2_numbers.json","w"), indent=1)
fig.savefig(f"{P}/fig_cmp_sony2.pdf")
import subprocess as _sp, numpy as _np                        # 픽셀 기반: 잉크가 경계에 닿지 않아야 한다
from PIL import Image as _Im
_png=f"{P}/_cbchk"
_sp.run(["pdftoppm","-r","600","-png","-singlefile",f"{P}/fig_cmp_sony2.pdf",_png],check=True)
_a=_np.asarray(_Im.open(_png+".png").convert("L")); _ink=_a<250
_cols=_np.flatnonzero(_ink.any(0)); _rows=_np.flatnonzero(_ink.any(1))
_mx=_a.shape[1]-1-_cols.max(); _my=_a.shape[0]-1-_rows.max()
import os as _os; _os.remove(_png+".png")
assert _cols.min()>=1 and _mx>=2 and _rows.min()>=1 and _my>=1, ("잉크가 캔버스 경계에 닿음: 좌 %d 우 %d 상 %d 하 %d"%(_cols.min(),_mx,_rows.min(),_my))
print("여백 검사 통과: 좌 %d / 우 %d / 상 %d / 하 %d 화소"%(_cols.min(),_mx,_rows.min(),_my))
