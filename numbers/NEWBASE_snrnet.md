# NEWBASE — SNR-Net

저자 공개 가중치로 우리 벤치마크 두 개(LOLv1 eval15 15장, Sony SID sRGB test 598장)를
추론해 출력 캐시를 만들고, 저자 보고값이 재현되는지 확인한 기록. 실측값만 적는다.

## 1. 방법

- 이름: SNR-Net (표기 키 `snrnet`)
- 논문: "SNR-aware Low-Light Image Enhancement", Xiaogang Xu, Ruixing Wang, Chi-Wing Fu,
  Jiaya Jia. CVPR 2022, pp. 17714–17724. (refs.bib 의 `snrnet2022` 와 같은 항목)
- 구조: 3단 인코더(stride 2 두 번, 1/4 해상도) + SNR 맵으로 게이팅되는
  장거리(transformer, 4x4 패치 unfold, d_model 1024, 6층)/단거리(잔차블록) 두 경로 융합.
  파라미터 실측 39,124,099 (39.12M, 체크포인트 텐서 합계와 정확히 일치, 파일 156,523,164 B).
  Retinexformer 비교표는 SNR-Net Params 를 4.01M 으로 적어 두었는데 공개 체크포인트와는
  10배 가까이 어긋난다. 이 항목은 그 표를 인용하지 말고 실측값을 쓴다.
- 저장소: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
  (clone 커밋 1113144c82adc8bcc4a9ec27749ed75f196a4e4d, 2022-11-21)

## 2. 가중치 출처

- URL: https://drive.google.com/file/d/1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb/view
  (저장소 README 의 "Pre-trained Model" 링크). 파일명 `pretrain_model_cvpr_snr.zip`,
  1,017,280,510 B, md5 3e32ab2225f90a585e63a60aa64b7664.
  압축을 풀면 `pretrain_model/` 아래 7개 .pth (LOLv1, LOLv2_real, LOLv2_synthetic,
  SID, SMID, indoor_G, outdoor_G).
- 사용한 두 개
  - LOL : `pretrain_model/LOLv1.pth`  md5 3865acbaf9f9104512f752dd0d1a8d07
  - Sony: `pretrain_model/SID.pth`    md5 39f460347a990d2f9641a513a5f70324
- 로컬 경로: `<scratch>/newbase/dl/snr/pretrain_model/`
- 주의: README 가 SID 데이터셋 링크로 적어 둔 드라이브 폴더
  `1eQ-5Z303sbASEvsgCBSDbhijzLTWQJtR` 에는 `sid_processed.zip`(3.1GB, 데이터) 한 개뿐이고
  가중치는 없다. 가중치는 위의 단일 파일 링크에만 있다. 데이터 zip 은 받지 않았다.

## 3. 학습 데이터 (저자 기준)

- LOLv1.pth : LOL-v1 `our485` 485쌍 (options/train/LOLv1.yml 의 dataroot 확인).
  평가 분할은 `eval15` 15장.
- SID.pth   : SID Sony 를 rawpy 기본 ISP 로 RAW→RGB 변환한 sRGB 처리본
  (`short_sid2` / `long_sid2`, options/train/SID.yml 의 dataroot 확인).
  저자 로더가 씬 폴더명 첫 글자로 분할하며
  '0','2' 로 시작하는 씬이 학습, '1' 로 시작하는 씬이 테스트다.
  우리 Sony 벤치마크(598장)와 정확히 같은 데이터·같은 분할이다.

## 4. 전처리 요약 (저자 코드 해독 → 독립 재구현)

저자 저장소에서 가져온 것은 네트워크 정의(`models/archs/low_light_transformer.py`)뿐이다.
데이터로더(`data/dataset_LOLv1.py`, `data/dataset_SID.py`, `data/util.py`)와
모델 래퍼(`models/Video_base_model4_m.py`)는 읽고 아래대로 다시 짰다.

| 항목 | 내용 | 저자 코드 위치 |
|---|---|---|
| 입력 범위·채널 | float32 [0,1], RGB, CHW. 파일은 BGR 로 읽고 `[2,1,0]` 으로 뒤집는다 | data/util.py `read_img_seq`/`read_img_seq2` |
| LOL 입력 | PNG 원본 400x600, 리사이즈 없음 (yml 의 train_size 는 test phase 에서 안 쓰임) | data/dataset_LOLv1.py, phase=='test' 분기 |
| Sony 입력 | .npy(uint8 512x960 BGR) → /255 → 채널 뒤집기. train_size [960,512] 는 원본과 같아 실질 무변화 | data/dataset_SID.py + read_img2 |
| SNR 맵용 블러 | `nf = cv2.blur(LQ*255, (5,5))/255` (정규화 박스필터, 경계 reflect_101) | 두 dataset 파일 공통 |
| SNR 맵 | dark=0.299R+0.587G+0.114B(LQ), light=같은 식(nf), noise=abs(dark-light), mask=light/(noise+1e-4), mask/=(max+1e-4), clamp[0,1] | models/Video_base_model4_m.py `test()`/`test4()` |
| LOL 추론 | 입력·마스크를 400x608 로 bilinear 리사이즈 → forward → 출력을 400x600 으로 되돌림 | test_LOLv1_v2_real.py → `test4()` |
| Sony 추론 | 리사이즈 없이 512x960 그대로 forward | test.py → `test()` |
| 출력 | [0,1] 클리핑 | (저자는 tensor2img 에서 clamp) |

LOL 만 리사이즈하는 이유: 특징맵이 입력의 1/4 이고 거기서 4x4 unfold 를 하므로 입력 변이
16 의 배수여야 한다. 600 은 아니고 608 은 맞다. 512x960 은 둘 다 16 의 배수라 그대로 간다.

네트워크 설정은 두 벤치마크 동일(options/test/LOLv1.yml, SID.yml):
nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1, predeblur=true, HR_in=true, w_TSA=true.
state_dict 는 `module.` 접두 제거 후 strict=True 로 적재(누락·잉여 키 없음).

## 5. 실행

```
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate edge
cd code
python newbase_snrnet.py --bench both
```

- 스크립트: `code/newbase_snrnet.py`
- 로그: `numbers/newbase_snrnet_run.log`
- 수치 json: `numbers/newbase_snrnet.json` (장별 + 평균)
- 장비: 136 RTX PRO 6000 Blackwell, conda env `edge`, torch 2.11.0+cu128
- 속도: LOL 0.43 s/장, Sony 0.28 s/장 (채점 포함, 전체 약 3분)

출력 캐시 (float32, CHW, RGB, [0,1], 입력과 같은 해상도):

```
numbers/cache_snrnet/LOL /<프레임id>.npy   15개   43.2 MB
numbers/cache_snrnet/Sony/<프레임id>.npy  598개  3527.2 MB
```

프레임 id 규약은 compare_methods.py 와 같다. LOL 은 `1.png` → `1.png.npy`,
Sony 는 `10003_00_0.04s.npy` → `10003_00_0.04s.npy.npy`.
`compare_methods.METHODS` 의 두 벤치마크 목록에 `snrnet` 이 실제로 잡히는 것까지 확인했다.

## 6. 결과

채점은 `repro_measure.psnr_indep` / `ssim_indep` (독립 구현) 로만 했다.
평균은 프레임 평평 평균(LOL 15장, Sony 598장).

```
                     LOL (15장)        Sony (598장)
  ------------------------------------------------
  PSNR round            24.6095            22.8672
  PSNR trunc            24.5981            22.8568
  PSNR round +GT-mean   26.8118            24.1094
  PSNR trunc +GT-mean   26.8310            24.1180
  SSIM round             0.8419             0.6251
  SSIM round +GT-mean    0.8520             0.6358
  장별 표준편차(round)     4.1115             3.5214
  최저 / 최고 (round)   15.89 (179.png)    13.54 (10069_03_0.1s)
                        29.21 (111.png)    31.16 (10030_09_0.1s)
```

Sony 를 씬 단위로 먼저 평균낸 뒤 50 씬을 평균하면 22.9105 (평평 평균 22.8672 와 0.04 dB 차).
저자 test.py 는 두 가지를 다 찍는데, 보고값과 일치하는 쪽은 평평 평균이다.

## 7. 저자 보고값과 비교

저자 보고값 출처: Retinexformer 저장소 README 의 7개 벤치마크 비교표
(`third_party/Retinexformer/figure/seven_results.png`) 의 SNR-Net [57] 행 —
LOL-v1 24.61 / 0.842, SID 22.87 / 0.625. 이 표는 우리 SID_DATA_NOTE.md 에서 이미
Retinexformer 행(24.44 / 0.680) 검증에 쓴 것과 같은 표다.
CVF 원문 PDF 는 이번에 직접 열지 못했다(openaccess.thecvf.com 이 403 반환).
따라서 위 두 수치는 "비교표 기재값"으로 표시하고, SNR-Net 원문 표와의 대조는 미확인으로 남긴다.

GT-mean 정합 여부: 저자 평가 코드(test.py / test_LOLv1_v2_real.py → utils/util.py
`calculate_psnr`)는 uint8 출력과 uint8 GT 를 그대로 비교하고 평균 정합을 하지 않는다.
8bit 변환도 `(x*255).round()` 로 반올림이다. 저장소 전체에서 GT 평균으로 출력을 재조정하는
코드는 없다. 즉 보고값은 "반올림 + GT-mean 정합 없음" 규약이다.

```
             저자 보고    우리 재현(round, 정합 없음)   차이
  ----------------------------------------------------------
  LOL  PSNR    24.61            24.6095              -0.0005
  LOL  SSIM     0.842            0.8419              -0.0001
  Sony PSNR    22.87            22.8672              -0.0028
  Sony SSIM     0.625            0.6251              +0.0001
```

## 8. 재현 판정

재현 성공. 두 벤치마크 모두 보고 자릿수 안에서 일치한다(PSNR 차이 0.003 dB 이하,
SSIM 차이 0.0001 이하). 채널 순서·정규화·패딩을 손볼 일은 없었다.

부수적으로 확인된 것:
- 보고값 규약이 "반올림 + 정합 없음"이라는 판단이 수치로 뒷받침된다. GT-mean 을 적용하면
  LOL 26.81, Sony 24.11 로 보고값에서 2.2 / 1.2 dB 멀어진다. 버림으로 바꾸면 0.01 dB 수준만
  움직여 규약을 가르지 못한다.
- Sony 는 씬 단위 macro 평균(22.91)과 프레임 평평 평균(22.87)이 갈리는데 보고값은 후자다.
- 두 벤치마크 모두 장별 산포가 크다(표준편차 LOL 4.11 dB, Sony 3.52 dB). 평균 비교만으로
  방법 간 우열을 말하기 어려운 크기다.

## 9. 적대적 재확인

전처리가 우연히 맞은 게 아닌지 LOLv1 15장 전량으로 대조군을 돌렸다(PSNR round 평균).

```
  정상 (RGB, 실제 SNR 맵)                24.6095
  채널 뒤집어 입력(BGR)                  20.3768   -4.23 dB
  mask=1 강제 (전부 단거리 conv 경로)     16.7590   -7.85 dB
  mask=0 강제 (전부 transformer 경로)    24.6100   +0.0005 dB
```

채널 순서와 SNR 맵은 둘 다 결과를 크게 바꾸므로, 보고값과 일치했다는 사실이
"둔감해서 맞은 것"이 아니다.

한편 mask=0 을 강제해도 결과가 사실상 그대로다. 실제 SNR 맵을 재보니 두 벤치마크 모두
거의 0 이기 때문이다.

```
              mask 평균    mask>0.5 화소비율
  LOL          0.0240          0.0023          (15장 전량)
  Sony         0.0030          0.0003          (598장 중 50장 표본)
```

즉 이 두 벤치마크에서 SNR-Net 은 사실상 장거리(transformer) 경로만으로 동작하고
SNR 게이팅은 거의 켜지지 않는다. 방법 이름이 내세우는 공간적응 분기가 우리 두 벤치마크에서는
실질적으로 비활성이라는 뜻이므로, 이 방법을 "SNR 인지 공간적응"의 대표로 인용할 때는
이 측정을 같이 밝히는 편이 맞다.
