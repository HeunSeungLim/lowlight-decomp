# URetinex-Net 재현 기록 (저자 공개 가중치)

작성 260905. 학습 없음, 추론 + 측정만. 스크립트 `code/newbase_uretinex.py`.

## 0. 대상

| 항목 | 값 |
|---|---|
| 방법 | URetinex-Net |
| 논문 | Wu, Weng, Zhang, Wang, Yang, Jiang, "URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement", CVPR 2022, pp.5901-5910 |
| 코드 | github.com/AndersonYong/URetinex-Net, 커밋 `7ee2196e093b6a9ae5c20242e48f9d61021b6364` (2025-01-22). 프로젝트 안 사본 `third_party/URetinex-Net` (network/, ckpt/, test.py, evaluate.py, utils.py, LICENSE, GIT_COMMIT.txt) |
| 가중치 | 저장소 동봉 `ckpt/init_low.pth`, `ckpt/init_high.pth`, `ckpt/unfolding.pth`, `ckpt/L_adjust.pth` (저자 배포본, 별도 내려받기 없음) |
| 학습 데이터 | LOL-v1 train (`our485`). `unfolding.pth` 안에 저자 학습 설정이 그대로 들어있고 `patch_low=.../LOLdataset/our485/low`, `eval_low=.../LOLdataset/eval15/low` 로 확인된다. 즉 우리가 평가하는 eval15 는 저자의 검증 분할이다. |
| 재학습·미세조정 | 없음 |
| 벤치마크 | LOL-v1 test(eval15) 15장, 400x600. Sony(SID)는 돌리지 않았다 — 이 가중치는 LOL 지도학습본이라 SID 는 규약 밖 |
| 실행 환경 | 서버136, RTX PRO 6000 Blackwell, conda `edge` (torch 2.11.0+cu128) |

`unfolding.pth` 의 저자 설정(그대로 사용): `round=3, gamma=0.1, Roffset=0.05, lamda=0.5, Loffset=0.05, concat_L=True, R_model=HalfDnCNNSE, L_model=Illumination_Alone`.

## 1. 전처리 / 입력 규약

우리 파이프라인 규약을 쓴다.

```
sys.path.insert(0, "code")
import diag_lolv1 as DL
for f, lq, gt in DL.build_pairs():      # PIL RGB uint8 HWC 400x600x3
    x = torch.from_numpy(lq.astype(np.float32)/255.0).permute(2,0,1)[None].cuda()
```

저자 데이터로더(`transforms.ToTensor()(Image.open(path))`)는 import 하지 않고 독립 재구현했다.
같은 값인지 15장 전량 화소 단위로 대조한다(`audit_loader`):

```
[audit] 우리 입력 규약 vs 저자 ToTensor(PIL): 최대 절대차 0.000e+00
```

리사이즈·패딩·크롭 없음. 출력도 400x600 그대로이고 [0,1] 클리핑 후 float32 CHW RGB 로 저장한다.

## 2. 조명 조정비(ratio) — 두 설정이 서로 다른 수를 낸다

저자 저장소에는 추론 진입점이 둘이고, 둘의 ratio 결정 방식이 다르다.

- `test.py` : `--ratio` 기본값 **5** 고정. GT를 쓰지 않는다. → **이번 캐시는 이 설정**
- `evaluate.py` : 정상광(GT) 영상을 `init_high.pth` 로 분해해 얻은 조도로 프레임마다 ratio 를 만든다
  (`ratio = (low_L/(high_L+1e-4)).mean()`, `low_ratio = 1/(ratio+1e-4)`).
  저자 주석은 "공정 비교를 위해 KinD/KinD++ 처럼 정상광 영상의 조도로 ratio 를 만든다"고 적고 있다.

논문 본문도 조정 모듈이 "user-defined ratio" 로 동작한다고 쓴다(Fig.2 설명). 즉 논문 표의 값은
GT를 쓰는 `evaluate.py` 설정에서 나온 것이다. 캐시에는 지시대로 `test.py` 기본값(ratio=5)만 넣고,
`evaluate.py` 설정은 재현 판정용 진단으로만 같이 쟀다(캐시에 저장하지 않음).

## 3. 명령

```
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate edge
cd code
python newbase_uretinex.py            # ratio=5 (test.py 기본값), 15장 4.4초
```
로그 `numbers/newbase_uretinex_run.log`, 수치 `newbase_uretinex.json`.

## 4. 결과 (LOL-v1 eval15, n=15, sd 는 장별 표준편차 ddof=1)

측정은 `repro_measure.psnr_indep` / `ssim_indep` (우리 독립 구현)만 썼다. GT평균 정합은
`repro_measure.gt_mean_rectify`.

```
설정                          8bit    PSNR     sd     SSIM   | GT평균 PSNR  GT평균 SSIM
test.py ratio=5 (캐시 값)     반올림  19.860  3.968  0.8255  |    24.619      0.8493
test.py ratio=5 (캐시 값)     버림    19.842  3.938  0.8259  |    24.675      0.8498
evaluate.py GT비율 (진단)     반올림  21.329  3.927  0.8334  |      -           -
evaluate.py GT비율 (진단)     버림    21.329  3.892  0.8341  |      -           -
```

고정 ratio 훑기(반올림 PSNR): ratio 3 = 20.104, 4 = 19.699, 5 = 19.860.
`evaluate.py` 가 GT에서 만든 ratio 의 평균은 7.27 (장별 5.16~9.60) 으로 기본값 5보다 크다.

저자 저장 규칙은 `np_save_TensorImg` 의 `np.clip(img*255,0,255).astype('uint8')` = **버림**이다.
버림/반올림 차이는 0.02 dB 미만이라 판정에 영향이 없다.

## 5. 저자 보고값과 대조

CVPR 2022 본문 Table 1 ("Quantitative comparison on LOL and SICE datasets"), LOL 행:

```
URetinex-Net    MAE 0.0832   PSNR 21.3282   SSIM 0.8348   LPIPS 1.2234
```

같은 값이 Table 2(제거실험)의 기본 설정 행 `Ours* (IM+UOM+IG, T=3)` 에도 나온다.
출처 PDF 는 `numbers/refs/uretinex_cvpr2022.pdf` 에 받아 두었고
위 수치는 그 PDF 본문에서 직접 확인했다.

| | 저자 보고 | 우리 재현 | 차이 |
|---|---|---|---|
| PSNR (evaluate.py 설정) | 21.3282 | 21.3288 | +0.0006 dB |
| SSIM (evaluate.py 설정) | 0.8348 | 0.8334 | -0.0014 |
| PSNR (test.py ratio=5) | (없음) | 19.8600 | - |

제3자 재보고: CUE(ICCV 2023, arXiv 2309.01958) Table 1 은 LOL 에서 URetinexNet 을
PSNR 21.33 / SSIM 0.834 로 옮겨 적는다 — 저자 표 값을 그대로 인용한 것이고 우리 값과도 맞는다.

## 6. 재현 판정

**재현 성공.**

- 저자 `evaluate.py` 설정(GT로 ratio 생성)에서 PSNR 21.329 dB — 논문 21.3282 와 0.0006 dB 차이.
  SSIM 도 0.8334 대 0.8348 로 0.0014 차이(우리는 SSIM 을 저자 코드가 아니라 독립 구현으로 잰다).
  가중치·모델 구조·전개 설정이 논문 값을 그대로 낸다는 뜻이다.
- 다만 **논문 값 21.33 은 GT를 입력으로 쓰는 설정의 값**이다. GT 없이 쓸 수 있는
  `test.py` 기본값(ratio=5)에서는 19.860 dB 로 1.47 dB 낮다.
  캐시에 저장한 것은 이 GT 비의존 값이므로, 비교표에 넣을 때 논문 표의 21.33 과 같은 칸에
  올려두면 안 된다. 표에는 19.86(무-GT, ratio=5)으로 쓰고 21.33 은 GT 사용 설정임을 각주로 남긴다.

## 7. 산출물

```
캐시  numbers/cache_uretinex/LOL/<id>.npy   15개
      float32, CHW, RGB, [0,1], 400x600  (id = "1.png" 등 → 파일명 "1.png.npy")
수치  numbers/newbase_uretinex.json
로그  numbers/newbase_uretinex_run.log
코드  code/newbase_uretinex.py
참고  numbers/refs/uretinex_cvpr2022.pdf
저자코드/가중치  third_party/URetinex-Net
```

`compare_methods.py` 의 `NEW_CACHE["uretinex"]` 규약(`<캐시>/<벤치>/<id>.npy`, `(3,H,W)`)과 맞춘다.

## 8. 캐시 재검증

저장된 `.npy` 를 디스크에서 다시 읽어 PSNR 을 다시 잰 결과가 추론 중 값과 일치한다.

```
cache_uretinex  LOL  n=15  반올림 19.8600  버림 19.8420
```

파일 점검: 15개 전부 `dtype=float32`, `shape=(3,400,600)`, 값 범위 [0.0000, 1.0000].
