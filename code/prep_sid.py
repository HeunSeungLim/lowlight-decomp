#!/usr/bin/env python3
"""
Retinexformer 저자 배포본 sid_processed(short_sid2/long_sid2, .npy) 를
HVI-CIDNet 계열 코드가 기대하는 datasets/Sony_total_dark 배치(PNG)로 변환한다.

출처
  입력  : $LLDATA/lowlight_model/data/SID_raw/SID/{short_sid2,long_sid2}/<씬ID>/*.npy
  출력  : $LLDATA/lowlight_model/data/Sony_total_dark/{train,test,eval}/{short,long}/<씬ID>/*.png

스플릿 규칙 (근거: Retinexformer basicsr/data/SID_image_dataset.py)
  - 씬ID 첫 글자 '1' -> 테스트  (코드: phase!=train 일 때 '1' in name[0])
  - 씬ID 첫 글자 '0' 또는 '2' -> 학습 (코드: '0' in name[0] or '2' in name[0])
  CIDNet Readme 트리는 train(0*)/test(1*)/eval 세 칸을 두므로,
  여기서는 0* -> train, 1* -> test, 2* -> eval 로 배치한다.
  (2* 를 학습에 쓰려면 train/ 과 eval/ 을 합쳐서 쓰면 된다.
   CIDNet 배포 코드에는 SID 학습 코드가 없어 2* 의 원 용도는 확정 불가.)

짝짓기
  Retinexformer 로더는 LQ 씬 폴더의 각 프레임에 대해 GT 로 같은 이름의 GT 씬 폴더의
  첫 번째 파일(img_paths_GT[0]) 하나를 쓴다. long_sid2 는 씬당 파일이 1개이므로
  short N장 <-> long 1장 이다.
  CIDNet measure_SID.py 도 label_dir 안의 listdir()[0] 을 GT 로 쓴다.
  따라서 GT 파일명 대응 규칙은 없고 "폴더 안 유일 파일"이 GT 다.
  입력 파일명은 그대로 출력 파일명이 된다(eval_SID.py 가 name[0] 으로 저장).

비트깊이
  원본 .npy 는 dtype=uint8, shape=(512,960,3), 값범위 0..255 이다.
  즉 8비트 변환(스케일링/반올림/버림) 이 전혀 필요 없다.
  round 도 trunc 도 쓰지 않고 uint8 배열을 그대로 PNG(무손실)로 저장한다.
  만약 uint8 이 아닌 파일이 하나라도 발견되면 즉시 예외를 던지고 중단한다
  (조용히 스케일링해서 값이 바뀌는 사고를 막기 위함).
"""

import os
import sys
import glob
import json
import random
import argparse

import numpy as np
from PIL import Image

SRC = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/SID_raw/SID")
DST = (os.environ.get("LLDATA", "data") + "/lowlight_model/data/Sony_total_dark")

SPLIT_BY_PREFIX = {"0": "train", "1": "test", "2": "eval"}


def scan(src):
    """씬ID -> {short:[npy...], long:[npy...]} 수집 + 통계."""
    scenes = {}
    for kind, sub in (("short", "short_sid2"), ("long", "long_sid2")):
        for d in sorted(os.listdir(os.path.join(src, sub))):
            p = os.path.join(src, sub, d)
            if not os.path.isdir(p):
                continue
            scenes.setdefault(d, {"short": [], "long": []})
            scenes[d][kind] = sorted(glob.glob(os.path.join(p, "*.npy")))
    return scenes


def convert(src, dst, dry=False):
    scenes = scan(src)
    stats = {"n_scene": len(scenes), "per_split": {}, "dtype_seen": set(),
             "shape_seen": set(), "vmin": 255, "vmax": 0}
    n_written = 0
    for sid in sorted(scenes):
        split = SPLIT_BY_PREFIX.get(sid[0])
        if split is None:
            raise RuntimeError(f"알 수 없는 씬ID 접두: {sid}")
        s = stats["per_split"].setdefault(
            split, {"scenes": 0, "short": 0, "long": 0})
        s["scenes"] += 1
        for kind in ("short", "long"):
            files = scenes[sid][kind]
            if not files:
                raise RuntimeError(f"{sid} 의 {kind} 가 비어 있다")
            s[kind] += len(files)
            outdir = os.path.join(dst, split, kind, sid)
            if not dry:
                os.makedirs(outdir, exist_ok=True)
            for f in files:
                a = np.load(f)
                stats["dtype_seen"].add(str(a.dtype))
                stats["shape_seen"].add(tuple(a.shape))
                if a.dtype != np.uint8:
                    raise RuntimeError(
                        f"uint8 이 아님: {f} dtype={a.dtype} — 변환 중단")
                if a.ndim != 3 or a.shape[2] != 3:
                    raise RuntimeError(f"HxWx3 아님: {f} shape={a.shape}")
                stats["vmin"] = min(stats["vmin"], int(a.min()))
                stats["vmax"] = max(stats["vmax"], int(a.max()))
                out = os.path.join(
                    outdir,
                    os.path.basename(f)[:-len(".npy")] + ".png")
                if not dry:
                    # 값 변형 없음: uint8 -> PNG 무손실 저장 (반올림/버림 미사용)
                    Image.fromarray(a, mode="RGB").save(out, format="PNG",
                                                        compress_level=6)
                n_written += 1
    stats["dtype_seen"] = sorted(stats["dtype_seen"])
    stats["shape_seen"] = sorted(str(x) for x in stats["shape_seen"])
    stats["n_files"] = n_written
    return stats


def verify(src, dst, n=20, seed=0, full=False):
    """변환본을 다시 읽어 원본 npy 와 화소 단위로 완전히 같은지 대조."""
    npys = sorted(glob.glob(os.path.join(src, "short_sid2", "*", "*.npy"))) + \
        sorted(glob.glob(os.path.join(src, "long_sid2", "*", "*.npy")))
    if not full:
        random.seed(seed)
        npys = random.sample(npys, n)
    ok, bad = 0, []
    for f in npys:
        sid = os.path.basename(os.path.dirname(f))
        kind = "short" if "short_sid2" in f else "long"
        split = SPLIT_BY_PREFIX[sid[0]]
        png = os.path.join(dst, split, kind, sid,
                           os.path.basename(f)[:-len(".npy")] + ".png")
        a = np.load(f)
        b = np.array(Image.open(png).convert("RGB"))
        if a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b):
            ok += 1
        else:
            bad.append((png, a.shape, b.shape,
                        int(np.abs(a.astype(int) - b.astype(int)).max())
                        if a.shape == b.shape else -1))
    return ok, bad, len(npys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--verify-n", type=int, default=20)
    ap.add_argument("--verify-full", action="store_true")
    a = ap.parse_args()

    st = convert(a.src, a.dst, dry=a.dry)
    print(json.dumps(st, indent=2, ensure_ascii=False))
    if not a.dry:
        ok, bad, tot = verify(a.src, a.dst, n=a.verify_n,
                              full=a.verify_full)
        print(f"무손실 검증: {ok}/{tot} 완전 일치")
        for b in bad:
            print("  불일치:", b)
        sys.exit(0 if not bad else 1)
