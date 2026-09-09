import os, json, csv, math
import numpy as np

EXDARK_ROOT = (os.environ.get("LLDATA", "data") + "/enhance_data/exdark/extracted/ExDark")
META = os.path.join(EXDARK_ROOT, "metadata.csv")

# ExDark 12 classes -> open-vocabulary prompt used by YOLO-World
CLASSES = ["Dog","Motorbike","People","Cat","Chair","Table",
           "Car","Bicycle","Bottle","Bus","Cup","Boat"]
PROMPTS = ["dog","motorcycle","person","cat","chair","dining table",
           "car","bicycle","bottle","bus","cup","boat"]
CID = {c:i for i,c in enumerate(CLASSES)}


def load_gt():
    """returns dict: file_name -> {'boxes': (N,4) xyxy float, 'cls': (N,) int, 'wh': (w,h)}"""
    gt = {}
    with open(META) as f:
        for r in csv.DictReader(f):
            fn = r["file_name"]
            x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
            iw, ih = int(r["img_w"]), int(r["img_h"])
            g = gt.setdefault(fn, {"boxes": [], "cls": [], "wh": (iw, ih)})
            g["boxes"].append([x, y, x + w, y + h])
            g["cls"].append(CID[r["class"]])
    for fn, g in gt.items():
        g["boxes"] = np.array(g["boxes"], np.float32)
        g["cls"] = np.array(g["cls"], np.int32)
    return gt


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    iy = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = ix * iy
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return (inter / np.maximum(aa + bb - inter, 1e-9)).astype(np.float32)
