"""Enhancement arms for the power check.

All arms take a uint8 BGR image (as read by cv2) and return uint8 BGR.
No arm is trained on ExDark (leakage guard), except where explicitly named _ft.
"""
import sys, os
import numpy as np, cv2, torch

SCI_DIR = (os.environ.get("LLDATA", "data") + "/enhance_data/lowlight_bench/SCI_official/CVPR")
AR_DIR = (os.environ.get("ADAPTIVE_RETINEX_DIR", "weights"))
DEV = "cuda"


# ---------- classical ----------
def arm_raw(img):
    return img


def arm_gamma(img, g=0.5):
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def arm_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------- learned ----------
class SCIArm:
    def __init__(self, weights=os.path.join(SCI_DIR, "weights/medium.pt")):
        sys.path.insert(0, SCI_DIR)
        from model import Finetunemodel
        self.m = Finetunemodel(weights).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, img):
        t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.
        i, r = self.m(t)
        out = r.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (out[:, :, ::-1] * 255).astype(np.uint8)


class ARArm:
    def __init__(self, ckpt=os.path.join(AR_DIR, "adaptive_retinex_trained.pth")):
        sys.path.insert(0, AR_DIR)
        from adaptive_retinex import AdaptiveRetinex
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and not any(k.startswith("e1") for k in sd):
            sd = sd["model"]
        self.m = AdaptiveRetinex()
        self.m.load_state_dict(sd)
        self.m = self.m.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, img):
        t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(DEV) / 255.
        out = self.m(t)[0].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (out[:, :, ::-1] * 255).astype(np.uint8)


def build(name):
    if name == "raw":   return arm_raw
    if name == "gamma": return arm_gamma
    if name == "clahe": return arm_clahe
    if name == "sci":   return SCIArm()
    if name == "ar":    return ARArm()
    raise ValueError(name)
