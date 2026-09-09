# NEWBASE — LightenDiffusion

## 방법

- 이름: LightenDiffusion (Latent-Retinex Diffusion Model)
- 논문: Hai Jiang, Ao Luo, Xiaohong Liu, Songchen Han, Shuaicheng Liu,
  "LightenDiffusion: Unsupervised Low-Light Image Enhancement with Latent-Retinex
  Diffusion Models", ECCV 2024, pp. 161-179. arXiv:2407.08939
- 분류: 비지도(unpaired). 잠재공간 Retinex 분해(CTDN) + 조건부 DDIM(20 step) + 재구성 디코더.
- 코드: 저자 공개 저장소 (newbase/LightenDiffusion), `evaluate.py` / `models/{ddm,decom,unet}.py`

## 가중치

- 출처: README 의 Google Drive 폴더
  `https://drive.google.com/drive/folders/1m3t15rWw76IDDWJ0exLOe5P0uEnjk3zl`
- 받은 파일 (134MB): `newbase/dl/lightendiff/stage1/stage1_weight.pth.tar`,
  `newbase/dl/lightendiff/stage2/stage2_weight.pth.tar`
- 실제로 쓴 것: `stage2/stage2_weight.pth.tar` 하나.
  저자 `evaluate.py` 의 기본값 `--resume ckpt/stage2/stage2_weight.pth.tar` 와 같다.
  평가 경로에서는 `Net.decom` 도 stage2 체크포인트 안에 들어 있어(480개 텐서 전량 strict 일치)
  stage1 은 학습 전용이고 추론에 안 쓴다.
- README 의 파일 링크 `18bs_mAREhLipaM2qvhxs7u7ff2VSHet2` 는 가중치가 아니라 LOL **데이터셋** 이다
  (README 의 "Download the raw training and evaluation datasets" 항목). 받지 않았다.
- 벤치마크별 가중치는 없다. 배포본이 한 벌뿐이라 LOL 과 Sony 에 **같은 stage2 가중치**를 그대로 썼다.
  비지도 방법이라 이게 저자 의도와 맞는다.

## 학습 데이터 (저자 기준)

- 비지도(unpaired) 학습. 저자 설정 `configs/unsupervised.yml` 의 `train_dataset: "unpaired"`,
  `data_dir: /data/Image_restoration/LSRW_dataset`.
  논문 4.1 은 LSRW 학습셋에서 짝이 안 맞는 저조도/정상광 영상으로 학습했다고 쓴다.
- 우리 벤치마크(LOL-v1 eval15, SID sRGB test) 는 학습에 쓰이지 않았다.
  LOL-v1 test 는 이 방법에 대해 완전 미학습 데이터다.

## 전처리 (저자 `models/restoration.py` 재현, 데이터로더는 독립 재구현)

1. 저조도 입력을 CHW RGB float [0,1] 로 (= `PairToTensor` = `torchvision.F.to_tensor` = uint8/255).
2. 우/하단에 `reflect` 패딩으로 H,W 를 64 배수로 맞춘다
   (`img_h_64 = 64*ceil(h/64)`, LOL 400x600 → 448x640, Sony 512x960 은 이미 64 배수라 패딩 없음).
3. 모델 입력은 6채널 `cat([x_cond, x_cond], dim=1)`.
4. 출력 `pred_x` 를 `[:, :, :h, :w]` 로 원 해상도 크롭, [0,1] 클리핑.
   (저자는 `tvu.save_image` 로 저장하는데 그 안에서 `mul(255).add_(0.5).clamp_(0,255)` 를 하므로
    같은 클리핑 + 반올림이다. 우리는 float 를 캐시하고 채점에서 rint 한다.)
5. 시드: 저자 `evaluate.py` 는 시드를 안 박는다. DDIM 초기 잡음 `torch.randn` 이 유일한 난수원이라
   장마다 `SEED=20260905` 로 재시드했고 cudnn deterministic 을 켰다. 부분 재실행에도 값이 같다.

입력 규약은 우리 파이프라인과 동일:
LOL 은 `diag_lolv1.build_pairs()`, Sony 는 `diag_sid_failure.build_pairs()/load_npy()`
(= `compare_methods.sid_pairs/sid_load`, 저자 로더의 BGR→RGB 뒤집기 포함, 512x960).

## 감사 (독립 재구현 대조)

저자 데이터로더(`datasets/`)를 import 하지 않고 위 1~4 를 직접 짠 뒤, 저자 경로와 수치로 대조했다.

| 대조 항목 | 결과 |
|---|---|
| 저자 `datasets.LLdataset` val_loader 의 `x[:, :3]` 대 우리 `lq/255` (15장) | id 순서 동일, `max abs diff = 0.000e+00` |
| 저자 `DenoisingDiffusion`(DataParallel, `load_ddm_ckpt`) 출력 대 우리 캐시 (5장, 같은 시드) | `max abs diff = 0.000e+00` |

즉 우리 로더 = 저자 로더, 우리 `module.` 접두사 제거 로딩 = 저자 DataParallel 로딩이 비트 단위로 같다.

## 명령

```
# 가중치
conda activate edge
gdown --folder "https://drive.google.com/drive/folders/1m3t15rWw76IDDWJ0exLOe5P0uEnjk3zl" \
      -O newbase/dl/lightendiff

# 추론 (GPU 136, RTX PRO 6000)
python code/newbase_lightendiff.py LOL
python code/newbase_lightendiff.py Sony
python code/newbase_lightendiff.py score
```

소요: LOL 15장 3.0s, Sony 598장 117.2s (둘 다 0.20 s/장).

## 캐시

```
numbers/cache_lightendiff/LOL/<파일명>.npy     15개  42MB
numbers/cache_lightendiff/Sony/<basename>.npy 598개 3.3GB
```
float32, CHW, RGB, [0,1], 입력 해상도. `compare_methods.py` 의 `NEW_CACHE` 규약과 같다.

## PSNR

채점은 `repro_measure.psnr_indep` / `ssim_indep` (독립 구현). 8bit 변환만 다르게 두 가지.

| 벤치마크 | n | PSNR 반올림 | PSNR 버림 | PSNR 반올림+GT평균정합 | SSIM 반올림 | 장별 표준편차(반올림) |
|---|---|---|---|---|---|---|
| LOL-v1 eval15 | 15 | 20.116 | 20.068 | 23.720 | 0.8129 | 2.983 |
| Sony SID sRGB test | 598 | 17.093 | 17.076 | 17.986 | 0.4607 | 3.538 |

## 저자 보고값과 대조

- 논문 Table 1, LOL: PSNR **20.453** / SSIM **0.803** / LPIPS 0.192.
- GT 평균 정합 여부: **쓰지 않았다.** 논문 4.1 은 "two distortion metrics PSNR and SSIM,
  and a full-reference perceptual metric LPIPS" 라고만 쓰고 GT-mean 보정을 언급하지 않는다.
  수치로도 확인된다 — GT 평균 정합을 적용하면 23.720 으로 저자값보다 3.27 dB 높다.
  따라서 저자값과 비교할 값은 **정합 없는 반올림 20.116** 이다.
- 저자는 확산 표본추출의 무작위성 때문에 "the reported performance of GDP and our method are
  the mean values for five times evaluation" 이라고 밝힌다. 같은 프로토콜로 재현:

| 시드 | 0 | 1 | 2 | 3 | 4 | 5회 평균 |
|---|---|---|---|---|---|---|
| PSNR | 20.034 | 19.973 | 20.514 | 20.007 | 20.136 | **20.133** (sd 0.198) |
| SSIM | 0.8128 | 0.8126 | 0.8134 | 0.8131 | 0.8125 | **0.8129** |

  시드 6개(20260905/0/1/42/230/2024) 를 더 넓게 훑으면 19.973 ~ 20.545 로, 저자 보고 20.453 이
  이 범위 안에 들어온다.
- Sony(SID): 저자 논문에 SID/Sony 결과가 **없다** (LOL, LSRW, DICM, NPE, VV 만 평가).
  17.093 은 대조할 저자 보고값이 없는 우리 측정값이다.

## 재현 판정

**재현 성공.**

- LOL PSNR: 우리 20.116 (캐시 시드) / 20.133 (저자 5회 평균 프로토콜) 대 저자 20.453.
  차이 −0.32 dB 는 시드 간 표준편차 0.198 의 약 1.6배이고, 시드 산포 구간(19.97~20.54)이
  저자값을 덮는다. 계통 오차가 아니라 확산 표본추출의 무작위성으로 설명된다.
- LOL SSIM: 우리 0.8129 대 저자 0.803 으로 오히려 +0.010 높다. 밝기/구조 어느 쪽도 깎이지 않았다.
- 로더·가중치 로딩 경로는 저자 코드와 비트 단위로 같다(위 감사표). 남은 차이는 시드뿐이다.

## 문제점 / 주의

- 저자 `evaluate.py` 에 시드가 없어서 저자 수치를 결정론적으로 똑같이 재생성할 수 없다.
  우리 캐시는 `SEED=20260905` 한 벌이고, 이 방법만 다른 결정론적 방법들과 달리
  장당 ±0.2 dB 수준의 실행 간 산포를 갖는다. 표에 쓸 때 이 사실을 각주로 남겨야 한다.
- Sony 는 저자가 평가한 적 없는 벤치마크다. 저자 보고값 대조가 불가능하므로
  "저자 배포 가중치를 우리 규약으로 돌린 값" 이상으로 해석하면 안 된다.
- Sony 캐시가 3.3GB 다 (598 x 512x960x3 float32).

## 스크립트

`code/newbase_lightendiff.py`
(장별 수치는 `numbers/newbase_lightendiff.json`,
 진행 로그는 `newbase_lightendiff_progress.json` / `newbase_lightendiff_sony.log`)
