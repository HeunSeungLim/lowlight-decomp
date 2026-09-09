# NEWBASE — LLFormer (LOL-v1 재현)

## 방법

- 이름: LLFormer
- 논문: "Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and
  Transformer-Based Method", AAAI 2023 (Oral). arXiv:2212.11548
- 저장소: `third_party/LLFormer`
  (GitHub TaoWangzj/LLFormer)
- 파라미터: 실측 24.549 M (저자 표기 24.55 M와 일치)

## 가중치

- 출처: README "Pretrained Model" 의 **LLFormer trained on LOL** Google Drive 폴더
  https://drive.google.com/drive/folders/1J7NvvPsCtT0j8Rd9ombJ6sVIC6v0Xweb
- 받은 파일: `models/model_bestPSNR.pth` (file id `1nTLY_gChn2_Va2BfS_R_W-bdjsqsLHJB`)
- 저장 위치: `.../newbase/dl/llformer/model_bestPSNR.pth`
  (296,109,482 B, md5 `086c642db41bca3497e8b3b7e8ee5b2a`, 체크포인트 `epoch=2186`)
- 지시받은 URL `.../folders/1c33pXjeqX-Fwxc_yAozGQ0UlsSZ4IOqn` 은 가중치가 아니라
  **MIT-Adobe FiveK 데이터셋** 폴더였다(목록에 `test/high/a45xx.png` 등 이미지만 존재).
  README 의 LOL 가중치 폴더로 바꿔 받았다.

## 학습 데이터

- LOL(=LOL-v1) 학습 분할 `our485`. 저자 학습 설정 `configs/LOL/train/training_LOL.yaml`
  (TRAIN_DIR `./datasets/LOL/train`, TRAIN_PS 128, BATCH 8, EPOCHS 4000).
- 평가에 쓴 eval15 15장은 학습 분할과 겹치지 않는다.

## 전처리 (저자 `test.py` 재현, 저자 데이터로더는 import 안 함)

입력은 우리 파이프라인 규약 `diag_lolv1.build_pairs()` 하나만 쓴다
(lq/gt = PIL RGB uint8 HWC 400x600x3).

1. `lq/255` → float32 CHW  (저자 `TF.to_tensor(PIL)` 와 동일)
2. `mul=16` 배수로 우/하단 `F.pad(..., 'reflect')`
   `H=((h+16)//16)*16`, `padh=H-h`(단 `h%16==0` 이면 0) → 400x600 은 padh=0, padw=8 → 400x608
3. `model(x)` → `clamp(0,1)` → `[:, :, :400, :600]`
4. 저장: float32 CHW RGB [0,1] `.npy`
   (저자는 여기서 `skimage.img_as_ubyte` = 반올림으로 8bit PNG 저장)

## 실행

```
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate edge
python code/newbase_llformer.py
```

- GPU: RTX PRO 6000 Blackwell, torch 2.11.0+cu128, float32
- 15장 총 4.4 s (첫 장 0.50 s, 이후 0.26 s/장)

## 결과 (LOL-v1 eval15, n=15)

채점은 `repro_measure` 독립 구현(`psnr_indep`/`ssim_indep`)만 사용.
GT-mean 정합은 `gt_mean_rectify` (그레이 평균 `0.299R+0.587G+0.114B` 를 GT에 맞춤).

| 8bit 규칙 | 정합 없음 PSNR | 정합 있음 PSNR |
|---|---|---|
| 반올림 (저자와 동일) | **23.6485** | 26.1057 |
| 버림 | 23.6373 | 26.0976 |

- 장별 PSNR 표준편차(반올림, 정합 없음) 4.635 dB — 15장이라 산포가 크다.
- SSIM(반올림, 정합 없음) 0.8185 / 정합 있음 0.8294

## 저자 보고값과 비교

- 저자 보고(논문 Table 2, 저장소 `figures/LOL-5K.png`): LOL PSNR **23.6491**,
  SSIM 0.8163, LPIPS 0.1692, MAE 0.0635.
- 저자 `evaluation.py` 에는 GT-mean 정합 절차가 없다 → 23.6491 은 **정합 없음** 값이다.
  우리 정합 없음/반올림 값 23.6485 와 대조해야 맞다.
- PSNR 차이 **0.0006 dB**.
- SSIM 은 0.8185 vs 0.8163 (+0.0022). 저자는 skimage 기본값(7x7 균일창)을,
  우리는 Wang et al. 관례(11x11 가우시안 sigma 1.5)를 쓴다. 규약 차이로 설명된다.
- 참고: GSAD 논문 표는 LLFormer LOLv1 을 25.758 로 적는데 그건 GT-mean 정합을 적용한
  값이다. 우리 정합 값은 26.106 으로, 정합 구현 차이(정합 후 재양자화 여부 등) 범위다.

## 재현 판정

**재현 성공.** 저자 보고 지표(정합 없음 PSNR)와 0.001 dB 이내로 일치.

## 산출물

- 스크립트: `code/newbase_llformer.py`
- 캐시: `numbers/cache_llformer/LOL/<id>.npy`
  (15개, float32, CHW, RGB, [0,1], 3x400x600)
- 수치 원본: `numbers/newbase_llformer.json` (장별 포함)
