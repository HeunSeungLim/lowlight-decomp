"""SID 공식 학습 장면(161개)에 Retinexformer 를 돌려 예측기 학습자료를 만든다.
저장하는 건 프레임별 백분위 특징과 최적 전역이득뿐이고 영상은 저장하지 않는다."""
import os, sys, glob, json, numpy as np, torch
R = os.environ.get("LLROOT", "."); sys.path.insert(0, f"{R}/code/third_party/Retinexformer")
DATA = os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID"; W = os.environ.get("LLDATA", "data") + "/lowlight_model/weights/retinexformer/pretrain_model/SID.pth"
OUT = os.path.dirname(os.path.abspath(__file__)); QL = [50, 75, 90, 95, 98, 99, 99.5, 99.9]
PER_SCENE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1,2,2])
net.load_state_dict(torch.load(W, map_location="cpu")["params"]); net = net.cuda().eval()
def load_npy(p):
    import cv2
    a = np.load(p)
    if (a.shape[1], a.shape[0]) != (960, 512): a = cv2.resize(a, (960, 512))
    return np.ascontiguousarray(a.astype(np.float32)[:, :, ::-1] / 255.0)
pairs = []
for ld in sorted(glob.glob(f"{DATA}/short_sid2/*")):
    n = os.path.basename(ld)
    if n[0] != "0": continue                      # 학습 장면만
    gts = sorted(glob.glob(f"{DATA}/long_sid2/{n}/*"))
    if not gts: continue
    lqs = sorted(glob.glob(f"{ld}/*"))[:PER_SCENE]
    pairs += [(n, p, gts[0]) for p in lqs]
print(f"학습 프레임 {len(pairs)}개 / 장면 {len(set(p[0] for p in pairs))}개", flush=True)
QY, QG, A, S = [], [], [], []
with torch.no_grad():
    for i, (n, lp, gp) in enumerate(pairs):
        x = torch.from_numpy(load_npy(lp)).permute(2,0,1)[None].cuda()
        y = net(x).clamp(0,1)[0].permute(1,2,0).cpu().numpy(); g = load_npy(gp)
        QY.append(np.percentile(y, QL)); QG.append(np.percentile(g, QL))
        A.append(float((y*g).sum()/((y*y).sum()+1e-12))); S.append(n)
        if i % 100 == 0: print(f"  {i}/{len(pairs)}", flush=True)
np.savez(f"{OUT}/train_feats.npz", QY=np.array(QY), QG=np.array(QG), A=np.array(A), S=np.array(S), QL=np.array(QL))
print("저장 완료", np.array(A).mean(), np.array(A).std())
