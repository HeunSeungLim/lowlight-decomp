# NEWBASE — GSAD (LOL-v1 재현)

## 방법

- 이름: GSAD (GlobalDiff)
- 논문: "Global Structure-Aware Diffusion Process for Low-Light Image
  Enhancement", NeurIPS 2023. arXiv:2310.17577
- 저장소: `third_party/GSAD`
  (GitHub jinnh/GSAD)
- 파라미터: 실측 17.434 M (저자 표기 17.36 M — 확산 스케줄 버퍼 포함 여부 차이)

## 가중치

- 출처: README "pretrained model" Google Drive 폴더
  https://drive.google.com/drive/folders/1KLPm2oOg2Fx4WlbnOXMjN2rbyzzG8Hd-
- 받은 파일: `lolv1_gen.pth` (file id `1Qx4y04tPsKwF_WJypY9HZc7K4BVm4f8H`)
- 저장 위치: `.../newbase/dl/gsad/lolv1_gen.pth`
  (69,954,591 B, md5 `313a52c9fe21e7bfa7346c0129914f9e`)
- 키에 `module.` 접두어가 붙어 있다(DataParallel 저장). 벗겨서 `strict=True` 로 적재.
- 대조용으로 README 의 시각결과 폴더
  https://drive.google.com/drive/folders/1UIBn5Wle8FySag5Fby6PBm3zcxmN3qmY 에서
  `LOLv1_results.zip` (id `1OJZVneAImiH4fFivhsfZzb6Ed9wi4s8o`) 도 받았다.

## 학습 데이터

- LOL-v1 학습 분할 `our485`. 저자 설정 `config/lolv1.yml`
  (root `./dataset/LOLv1`, train=our485 / val=eval15, patch 96, batch 8).
- 평가에 쓴 eval15 15장은 학습 분할과 겹치지 않는다.

## 전처리·샘플링 (저자 `test.py` + `config/lolv1_test.json` 재현, 저자 데이터로더는 import 안 함)

입력은 우리 파이프라인 규약 `diag_lolv1.build_pairs()` 하나만 쓴다.

1. `cv2.copyMakeBorder(lq, 8,8,4,4, cv2.BORDER_REFLECT)` → 416x608
   (저자 `LOLv1_Dataset.__getitem__`, split=='val')
2. `/255` → float32 CHW → `x*2-1` (저자 `transform_augment(min_max=(-1,1))`)
3. 확산 역과정 `GaussianDiffusion.super_resolution` (저자 모델 코드 그대로 사용)
   - val 스케줄: `linear`, **n_timestep=20**, `linear_start=6e-4`, `linear_end=8.8e-1`
     (저자 기본값 그대로)
   - 조건부 샘플링, `clip_denoised=True`, 초기 `torch.randn` + 매 스텝 `randn_like`
   - 버퍼 등록 순서도 저자와 동일: train 스케줄(500스텝)로 등록 → state_dict 적재 →
     val 스케줄로 교체
4. 출력 `clamp(-1,1)` → `(x+1)/2` → `[:, 8:408, 4:604]` 로 패딩 제거 → 400x600
5. 저장: float32 CHW RGB [0,1] `.npy`
   (저자 `tensor2img` 는 여기서 반올림 8bit)
- 시드: 저자 `config/lolv1_test.json` 의 **4088** 을 루프 직전에 고정.

## 실행

```
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate edge
pip install kmeans_pytorch      # 저자 diffusion.py 의 import 의존성
python code/newbase_gsad.py --seed-sweep 1,7,42,2026,20260905
```

- GPU: RTX PRO 6000 Blackwell, torch 2.11.0+cu128, float32
- 15장 0.45 s/장 (첫 장은 cudnn 알고리즘 탐색으로 24 s), 시드 6회 전체 71.6 s

## 결과 (LOL-v1 eval15, n=15, 시드 4088)

채점은 `repro_measure` 독립 구현(`psnr_indep`/`ssim_indep`)만 사용.

| 8bit 규칙 | 정합 없음 PSNR | 정합 있음 PSNR |
|---|---|---|
| 반올림 (저자와 동일) | 22.8083 | **27.7065** |
| 버림 | 22.7294 | 27.7016 |

- SSIM(반올림): 정합 없음 0.8517 / 정합 있음 0.8749
- 장별 PSNR 표준편차(반올림, 정합 있음) 2.546 dB
- **정합 있음이 정본이다.** 저자 README 와 `test.py` 가 LLFlow·KinD 관례대로
  출력 밝기를 GT 그레이 평균에 맞춘 뒤 PSNR 을 잰다고 명시하고, 보고값도 그 값이다.

### 샘플링 확률성 (같은 가중치, 시드/런만 다름)

| 시드 | 4088 | 1 | 7 | 42 | 2026 | 20260905 |
|---|---|---|---|---|---|---|
| PSNR(정합) | 27.7065 | 27.7199 | 27.6104 | 27.6646 | 27.6495 | 27.7856 |

- 6회 평균 27.6894, 표준편차 0.062, 범위 27.610–27.786
- 같은 시드 재실행에서도 ±0.015 dB 흔들린다(`cudnn.benchmark` 알고리즘 선택).
  확산 샘플링이라 단일 실행값을 소수점 둘째 자리까지 신뢰하면 안 된다.

## 저자 보고값과 비교

- 저자 보고(저장소 `images/quantitative results.png` 표): LOLv1 PSNR **27.839**,
  SSIM 0.877, LPIPS 0.091, Param 17.36 M.
- 독립 대조: 저자가 공개한 `LOLv1_results.zip` 의 `output/*_normal.png` 15장을
  우리 GT 와 우리 채점기로 직접 재면 **PSNR 27.8396 / SSIM 0.8774** 다
  (프레임 대응은 `gt/*_gt.png` 를 우리 eval15 GT 와 화소 완전일치로 매칭해 확인,
  채널 순서는 RGB 그대로). 즉 27.839 는 공개 출력물 그 자체의 값이다.
- 우리 재현 27.7065(시드 4088) / 6회 평균 27.6894 → 차이 **0.13–0.15 dB**.
- 우리 출력 vs 저자 공개 출력의 프레임별 PSNR 평균 **39.34 dB**
  (최저 31.4, 최고 43.0) — 같은 모델·같은 가중치이고 차이는 샘플링 잡음뿐이라는 증거.
- 남는 0.15 dB 는 시드 산포(sd 0.062)의 약 2.4배다. 저자 평가 루프는 `os.listdir`
  순서로 15장을 돌아 잡음 추출 순서가 우리(정렬 순서)와 다르므로 장별 잡음이 다르고,
  공개 출력물은 그 분포의 상단에 위치한다. 계통 오차로 볼 만한 전처리 불일치는
  찾지 못했다.

## 재현 판정

**재현 성공(확률적 방법 허용 오차 내).** 저자 보고 27.839 대비 0.13 dB 낮은 27.71
(6회 평균 27.69, sd 0.06). 저자 공개 출력물과 프레임별 39.3 dB 일치.
다만 저자 보고값은 우리 관측 분포의 상단이며, 단일 실행으로는 27.839 를
재현하지 못했다. 표에 인용할 때는 실행 평균과 산포를 함께 적는다.

## 산출물

- 스크립트: `code/newbase_gsad.py`
- 캐시: `numbers/cache_gsad/LOL/<id>.npy`
  (15개, float32, CHW, RGB, [0,1], 3x400x600 — 시드 4088 실행, GT-mean 정합 **전** 값)
- 수치 원본: `numbers/newbase_gsad.json` (장별·시드별 포함)
