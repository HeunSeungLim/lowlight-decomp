# Zero-DCE++ 재현 기록 (저자 공개 가중치)

작성 260905. 학습 없음, 추론 + 측정만. 스크립트 `code/newbase_zerodcepp.py`.

## 0. 대상

| 항목 | 값 |
|---|---|
| 방법 | Zero-DCE++ |
| 논문 | Li, Guo, Loy, "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation", IEEE TPAMI 44(8), 2022 (arXiv 2103.00860) |
| 코드 | github.com/Li-Chongyi/Zero-DCE_extension, 하위 폴더 `Zero-DCE++`, 커밋 `09f202b690f82da939b8e6ec8535960ae97ad8bd` (2022-06-19). 프로젝트 안 사본 `third_party/Zero-DCE++` (model.py, lowlight_test.py, snapshots_Zero_DCE++/, GIT_COMMIT.txt) |
| 가중치 | 저장소 동봉 `snapshots_Zero_DCE++/Epoch99.pth` (저자 배포본). 파라미터 10,561개 |
| 학습 데이터 | 저자 학습본. **무참조(zero-reference)** 학습이라 정답쌍을 쓰지 않는다. 논문 3절: SICE 데이터셋 Part1 의 다중노출 360 시퀀스, 3,022장 중 2,422장 학습 / 나머지 검증, 학습·시험 영상은 512x512 로 리사이즈 |
| 재학습·미세조정 | 없음 |
| 벤치마크 | LOL-v1 test(eval15) 15장 400x600 + Sony(SID sRGB) test 598장 512x960. 무참조 방법이라 두 벤치마크 모두 같은 가중치 하나로 돌린다 |
| 실행 환경 | 서버136, RTX PRO 6000 Blackwell, conda `edge` (torch 2.11.0+cu128) |

## 1. 전처리 / 입력 규약

우리 파이프라인 규약을 쓴다. 저자 데이터로더(`dataloader.py`, `lowlight_test.py` 의 파일 읽기)는
import 하지 않고 독립 재구현했다.

```
sys.path.insert(0, "code")
import diag_lolv1 as DL, compare_methods as CM
# LOL : PIL RGB uint8 HWC 400x600 → /255
for f, lq, gt in DL.build_pairs(): ...
# Sony: CM.sid_load(lq_path) → HWC float [0,1] RGB 512x960 (저자 로더의 BGR→RGB 뒤집기 포함)
for scene, lqp, gtp in CM.sid_pairs(): ...
```

저자 `lowlight_test.py` 의 전처리는 `np.asarray(PIL)/255 → float → HWC 자르기 → permute(2,0,1)` 이다.
LOL 은 PNG 를 그대로 읽으므로 우리 규약과 같은 값이고, Sony 는 우리 파이프라인이 이미 저자 로더의
채널 규약을 따르고 있어 `CM.sid_load` 를 그대로 쓴다.

`scale_factor` 는 저자 기본값 **12** 를 그대로 썼다. 논문도 12배 다운샘플이 기본이라고 적는다
("12× downsampling operation as default", 4.5절).

논문의 자체 평가(SICE Part2)는 영상을 512x512 로 리사이즈하지만, 저자가 배포한 추론 진입점
`lowlight_test.py` 는 리사이즈하지 않고 scale_factor 배수로 자르기만 한다. 지시대로 배포된
`lowlight_test.py` 기본 동작을 따랐다(리사이즈 없음).

### 해상도 문제와 처리 (우리가 저자 코드에서 벗어난 유일한 지점)

저자 `lowlight_test.py` 는 입력을 `scale_factor` 의 배수로 자른다:

```
h = (H//12)*12 ;  w = (W//12)*12
LOL  400x600 → 396x600  (아래 4행 버림)
Sony 512x960 → 504x960  (아래 8행 버림)
```

캐시 규약은 입력 해상도 `(3,H,W)` 이므로 잘린 출력을 그대로 쓸 수 없다. 다음처럼 처리했다.

1. 저자와 똑같이 자른 입력으로 곡선맵 `x_r` 을 얻는다 (여기까지 저자 `forward` 그대로 호출).
2. `x_r` 을 아래쪽으로 가장자리 복제(`F.pad(..., mode="replicate")`)해 입력 해상도로 만든다.
3. 저자 `enhance()` 를 원본 해상도 입력에 적용한다.

곡선 적용은 화소별 연산이라, 잘린 영역 안에서는 저자 출력과 **비트 단위로 같다**. 매 프레임 확인했다:

```
크롭 영역 저자 출력과의 최대 불일치: 0     (LOL 15장, Sony 598장 전량)
```

버려지는 4/8행의 영향을 따로 보려고 "저자 크롭 영역만" PSNR 도 같이 냈다. 차이는
LOL -0.001 dB, Sony +0.010 dB 로 판정에 영향이 없다.

## 2. 명령

```
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate edge
cd code
python newbase_zerodcepp.py            # LOL 2.6초 + Sony 197초
```
로그 `numbers/newbase_zerodcepp_run.log`, 수치 `newbase_zerodcepp.json`.

## 3. 결과 (sd 는 장별 표준편차 ddof=1)

측정은 `repro_measure.psnr_indep` / `ssim_indep` (우리 독립 구현)만 썼다. GT평균 정합은
`repro_measure.gt_mean_rectify`.

```
벤치마크   n     8bit     PSNR     sd     SSIM   | GT평균 PSNR  GT평균 SSIM  | 크롭만 PSNR
LOL       15   반올림   15.344  4.659  0.5675  |    20.076      0.5346     |   15.343
LOL       15   버림     15.278  4.649  0.5659  |    20.011      0.5330     |
Sony     598   반올림   12.737  3.323  0.1376  |    12.746      0.1343     |   12.747
Sony     598   버림     12.741  3.322  0.1376  |    12.724      0.1340     |
```

저자 저장 규칙은 `torchvision.utils.save_image` = `mul(255).add_(0.5).clamp_(0,255)` 이라
**반올림**이다(URetinex-Net 의 버림과 반대). 두 규칙 차이는 0.07 dB 미만이다.

Sony 는 반올림이 버림보다 0.004 dB 낮다. 기존 SCI 에서 본 것과 같은 방향이고, 출력이 GT 보다
밝은 쪽으로 치우쳐 있을 때 나타나는 부호다.

## 4. 저자 보고값과 대조

### 4-1. LOL

Zero-DCE++ 논문은 **LOL 에 대한 자기 PSNR/SSIM 을 싣지 않는다.** 논문의 정량 표
(Table 1/2/4/6)는 전부 SICE Part2 testing set 이고, Table 6 의 Zero-DCE++ 행은
PSNR 16.42 / SSIM 0.58 / MAE 102.87 이다(SICE Part2, LOL 아님). 본문에서 LOL 은
다른 방법의 일반화 논의에 언급될 뿐이다. 출처 PDF `refs/zerodcepp_arxiv_2103.00860.pdf`.

그래서 타 논문의 재보고값과 비교한다.

| 출처 | 설정 | PSNR | SSIM |
|---|---|---|---|
| CUE, ICCV 2023 (arXiv 2309.01958) Table 1 | LOL, 원해상도 | 15.53 | 0.567 |
| ClassLIE (arXiv 2312.13265) Table | LOL, **256x256 로 리사이즈** | 16.11 | 0.53 |
| 우리 재현 | LOL-v1 eval15, 원해상도 400x600 | **15.344** | **0.5675** |

CUE 와는 PSNR 0.19 dB, SSIM 0.0005 차이다. ClassLIE 는 학습·평가 영상을 256x256 으로
리사이즈하는 다른 규약이라 직접 비교 대상이 아니다(그 논문 4-A절에 명시).
두 재보고값 모두 지시에 적힌 14.7~16 범위 안이다.

### 4-2. Sony (SID sRGB)

이 데이터 분할(Retinexformer 배포 SID sRGB, 598장)에 대한 Zero-DCE++ 저자 보고값은 없다.
무참조 학습본을 극암 영역에 그대로 옮긴 것이라 대조할 공표값 자체가 존재하지 않는다.
대신 같은 분할·같은 측정으로 이미 확보한 우리 값들과 나란히 둔다
(`COMPARE_METHODS.md` 1절, 전부 반올림 PSNR):

```
Retinexformer 24.438 > Gamma 0.5 14.228 > CLAHE 13.321 > AdaptiveRetinex 12.929
              > Zero-DCE++ 12.737 > SCI 11.805 > Input 11.504
```

Zero-DCE++ 는 무보정 입력보다 1.23 dB 높고 감마 보정보다 1.49 dB 낮다. 무참조 곡선
추정이 극암에서는 사실상 전역 밝기 보정 수준에 머문다는 뜻이고, 기존 SID 진단과 방향이 같다.

## 5. 재현 판정

- **LOL: 재현 성공(대조군은 제3자 재보고값).** 15.344 dB / SSIM 0.5675 는 CUE 가 옮겨 적은
  15.53 / 0.567 과 PSNR 0.19 dB, SSIM 0.0005 차이다. 저자 논문에 LOL 직접 수치가 없으므로
  "논문 값 재현"이 아니라 "가중치·전처리가 통용 재보고값을 재현한다"로 읽어야 한다.
  0.19 dB 차이는 재보고 논문의 8bit 규칙·크롭·SSIM 구현 차이로 설명 가능한 크기다.
- **Sony: 대조 불가, 실행은 성립.** 공표값이 없어 재현 여부를 판정할 대상이 없다.
  캐시는 우리 규약대로 만들어졌고, 저자 크롭 영역 안에서 저자 forward 와 비트 단위로
  일치함을 598장 전량에서 확인했다.
- 두 벤치마크 모두 같은 가중치 하나(무참조 학습본)를 썼다. 벤치마크별 미세조정 없음.

## 6. 산출물

```
캐시  numbers/cache_zerodcepp/LOL/<id>.npy    15개  (3,400,600)
      numbers/cache_zerodcepp/Sony/<id>.npy  598개 (3,512,960)
      float32, CHW, RGB, [0,1]
      id 는 LOL = "1.png" 등, Sony = os.path.basename(lq_path) = "10003_00_0.033s.npy"
      → 파일명은 각각 "1.png.npy", "10003_00_0.033s.npy.npy"
수치  numbers/newbase_zerodcepp.json
로그  numbers/newbase_zerodcepp_run.log
코드  code/newbase_zerodcepp.py
참고  numbers/refs/zerodcepp_arxiv_2103.00860.pdf
      numbers/refs/arxiv_2309.01958.pdf (CUE)
      numbers/refs/arxiv_2312.13265.pdf (ClassLIE)
저자코드/가중치  third_party/Zero-DCE++
```

`compare_methods.py` 의 `NEW_CACHE["zerodcepp"]` 규약(`<캐시>/<벤치>/<id>.npy`, `(3,H,W)`)과 맞춘다.

## 7. 캐시 재검증

저장된 `.npy` 를 디스크에서 다시 읽어 PSNR 을 다시 잰 결과가 추론 중 값과 일치한다.

```
cache_zerodcepp  LOL   n= 15  반올림 15.3441  버림 15.2780
cache_zerodcepp  Sony  n=598  반올림 12.7369  버림 12.7411
cache_uretinex   LOL   n= 15  반올림 19.8600  버림 19.8420
```
