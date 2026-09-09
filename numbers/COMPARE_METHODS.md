# 기존 방법 나란히 비교 — 같은 벤치마크·같은 채점 (260905)

코드 code/compare_methods.py, 수치 repro/compare_methods.json, 그림 paper/fig_cmp_{lol,sony}.pdf.
추론만(GPU 0). 채점은 repro_measure 독립 구현. 8bit 반올림, GT평균 미적용이 주 값.
성분 분해 = diag_sid_failure.decompose (E가중 몫, 합 100). Δ16 = gain 사다리 16화소 블록
오라클 이득에서 white 자유도 대조군을 뺀 값(diag_sid_lowfreq/diag_lolv1 와 같은 정의, seed 20260905).
TF32: matmul False, cudnn True.

## 0. 비교 대상

```
방법                      학습    가중치                                  LOL  Sony
-----------------------------------------------------------------------------------
Input                     무      -                                          o    o
Gamma 0.5                 무      x^0.5 (enhancers.arm_gamma 감마, 8bit 반올림)   o    o
CLAHE                     무      cv2 CLAHE clip2.0 8x8, LAB L (enhancers.arm_clahe)   o    o
Zero-DCE++                무참조    Zero-DCE++ Epoch99.pth (저자, 무참조), 캐시(newbase)   o    o
SCI                       유      SCI medium.pt (공식)                         o    o
AdaptiveRetinex (ours)    유      자체 학습본 ★자체 방법                              o    o
URetinex-Net              유      URetinex-Net ckpt (저자, LOL), 캐시(newbase)   o    -
SNR-Net                   유      SNR-Net LOL/SID .pth (저자), 캐시(newbase)     o    o
LLFormer                  유      LLFormer LOL-v1 (저자), 캐시(newbase)          o    -
GSAD                      유      GSAD LOL-v1 (저자), 캐시(newbase)              o    -
LightenDiffusion          유      LightenDiffusion (저자, 비지도), 캐시(newbase)    o    o
CIDNet                    유      HVI-CIDNet-LOLv1-woperc (저자), gated, 캐시 재사용   o    -
CIDNet (perc.)            유      HVI-CIDNet-LOLv1-wperc (저자), gated, 캐시 재사용   o    -
Retinexformer             유      LOL_v1.pth / SID.pth (저자)                  o    o
```
CIDNet Sony 제외 사유: CIDNet+ Sony-Total-Dark 는 화소 규약이 다른 별개 변종 (같은 데이터에서 PSNR 5.5; repro/SID_BENCHMARK_ID.md)

## 1. 정량 비교표 (주 값: 8bit 반올림, GT평균 미적용)

### LOL  (n=15장, 400x600; sd 는 장별 표준편차 ddof=1)

```
방법                           PSNR     sd    SSIM     sd |    전역%    채널%    구조% |  Δ16 dB  white |   LF% |      버림    GT평균
--------------------------------------------------------------------------------------------------------------------------------
Retinexformer              25.152  2.872  0.8454 0.0584 |   41.4    4.1   54.5 |   1.171  0.006 |  68.1 |  25.070  27.169
SNR-Net                    24.610  4.256  0.8419 0.0718 |   53.7    4.3   42.0 |   1.149  0.006 |  72.0 |  24.598  26.812
CIDNet (perc.)             23.840  5.620  0.8574 0.0737 |   74.9    1.1   23.9 |   2.111  0.006 |  67.2 |  23.808  27.726
LLFormer                   23.648  4.635  0.8185 0.0807 |   63.9    3.0   33.2 |   1.127  0.006 |  70.1 |  23.637  26.106
CIDNet                     23.609  5.443  0.8703 0.0615 |   75.7    1.1   23.3 |   2.289  0.006 |  69.6 |  23.498  28.229
GSAD                       22.808  5.420  0.8517 0.0948 |   80.9    1.3   17.8 |   1.369  0.006 |  66.4 |  22.729  27.706
LightenDiffusion           20.116  3.087  0.8129 0.0883 |   62.7    3.6   33.7 |   1.867  0.006 |  81.8 |  20.068  23.720
URetinex-Net               19.860  3.968  0.8255 0.0882 |   73.5    3.2   23.2 |   1.726  0.006 |  82.2 |  19.842  24.619
Zero-DCE++                 15.344  4.659  0.5675 0.1269 |   75.7    3.4   20.9 |   0.881  0.006 |  71.9 |  15.278  20.076
SCI                        14.854  4.425  0.5239 0.1398 |   70.1    2.1   27.8 |   1.740  0.006 |  78.8 |  14.784  19.053
Gamma 0.5                  12.329  4.054  0.5626 0.1345 |   92.3    0.7    7.0 |   0.521  0.006 |  55.1 |  12.260  23.247
AdaptiveRetinex (ours)     11.041  3.717  0.4487 0.1515 |   86.5    0.8   12.7 |   0.957  0.006 |  67.6 |  10.985  19.939
CLAHE                       9.132  2.959  0.3617 0.1133 |   88.2    0.4   11.4 |   2.991  0.006 |  81.2 |   9.132  18.942
Input                       7.773  2.617  0.1952 0.0957 |   88.2    0.7   11.2 |   3.151  0.006 |  81.7 |   7.773  18.786
```
(PSNR 내림차순. 전역/채널/구조 = 오차 성분 몫(E가중). Δ16 = 국소이득 오라클(16px) 자유도 보정 후 이득,
 white = 뺀 자유도 대조군 값. LF = f<0.10 오차 에너지 몫(RGB, E가중). 버림/GT평균 은 참고열.)

### Sony  (n=598장, 512x960; sd 는 장별 표준편차 ddof=1)

```
방법                           PSNR     sd    SSIM     sd |    전역%    채널%    구조% |  Δ16 dB  white |   LF% |      버림    GT평균
--------------------------------------------------------------------------------------------------------------------------------
Retinexformer              24.438  3.513  0.6800 0.1692 |   27.0    8.8   64.2 |   1.194  0.006 |  64.4 |  24.427  25.602
SNR-Net                    22.867  3.524  0.6251 0.1619 |   33.6    9.9   56.5 |   1.353  0.006 |  69.3 |  22.857  24.109
LightenDiffusion           17.093  3.541  0.4607 0.1887 |   27.3   23.9   48.8 |   1.711  0.006 |  86.2 |  17.076  17.986
Gamma 0.5                  14.228  3.675  0.1727 0.1331 |   19.0    6.8   74.2 |   1.124  0.006 |  53.9 |  14.206  14.362
CLAHE                      13.321  3.080  0.2231 0.1563 |   26.2    4.1   69.7 |   1.223  0.006 |  55.0 |  13.321  14.748
AdaptiveRetinex (ours)     12.929  3.333  0.1477 0.1144 |   15.0    3.5   81.5 |   0.749  0.006 |  58.4 |  12.902  12.666
Zero-DCE++                 12.737  3.323  0.1376 0.1108 |   28.4    6.3   65.3 |   1.268  0.006 |  57.6 |  12.741  12.746
SCI                        11.805  3.217  0.1383 0.1186 |   41.4    4.4   54.3 |   1.017  0.006 |  57.4 |  11.813  12.731
Input                      11.504  2.821  0.1504 0.1075 |   21.7    4.6   73.7 |   1.139  0.006 |  68.7 |  11.504  12.716
```
(PSNR 내림차순. 전역/채널/구조 = 오차 성분 몫(E가중). Δ16 = 국소이득 오라클(16px) 자유도 보정 후 이득,
 white = 뺀 자유도 대조군 값. LF = f<0.10 오차 에너지 몫(RGB, E가중). 버림/GT평균 은 참고열.)

노출시간별 PSNR (참고):
```
방법                           0.033s (n=28)    0.04s (n=180)     0.1s (n=390)
--------------------------------------------------------------------------------
Retinexformer                       22.491           22.909           25.284
SNR-Net                             21.584           22.164           23.284
LightenDiffusion                    13.778           15.232           18.189
Gamma 0.5                           10.695           11.558           15.713
CLAHE                               10.775           11.283           14.444
AdaptiveRetinex (ours)               9.869           10.555           14.244
Zero-DCE++                           9.161           10.199           14.165
SCI                                  8.515            9.513           13.098
Input                                9.600            9.929           12.367
```

## 2. 앵커 대조 (3개 + 캐시 진단값 교차)

```
항목                                                    기대         실측  판정
---------------------------------------------------------------------------
Sony/retinexformer psnr                           24.438    24.4383  통과
LOL/cidnet_woperc psnr_trunc                      23.498    23.4984  통과
LOL/cidnet_woperc psnr                            23.609    23.6087  통과
LOL/retinexformer psnr                            25.152    25.1523  통과
```

산수 대조:
```
항목                                                          값         기준  판정
--------------------------------------------------------------------------------
LOL/input 성분 합                                       100.0000             통과
LOL/input white16 자유도                                  0.0059             통과
LOL/input 반올림-버림 (dB)                                  0.0000             통과
LOL/gamma 성분 합                                       100.0000             통과
LOL/gamma white16 자유도                                  0.0057             통과
LOL/gamma 반올림-버림 (dB)                                  0.0689             통과
LOL/clahe 성분 합                                       100.0000             통과
LOL/clahe white16 자유도                                  0.0059             통과
LOL/clahe 반올림-버림 (dB)                                  0.0000             통과
LOL/zerodcepp 성분 합                                   100.0000             통과
LOL/zerodcepp white16 자유도                              0.0059             통과
LOL/zerodcepp 반올림-버림 (dB)                              0.0661             통과
LOL/sci 성분 합                                         100.0000             통과
LOL/sci white16 자유도                                    0.0059             통과
LOL/sci 반올림-버림 (dB)                                    0.0698             통과
LOL/ar 성분 합                                          100.0000             통과
LOL/ar white16 자유도                                     0.0060             통과
LOL/ar 반올림-버림 (dB)                                     0.0560             통과
LOL/uretinex 성분 합                                    100.0000             통과
LOL/uretinex white16 자유도                               0.0059             통과
LOL/uretinex 반올림-버림 (dB)                               0.0181             통과
LOL/snrnet 성분 합                                      100.0000             통과
LOL/snrnet white16 자유도                                 0.0059             통과
LOL/snrnet 반올림-버림 (dB)                                 0.0114             통과
LOL/llformer 성분 합                                    100.0000             통과
LOL/llformer white16 자유도                               0.0059             통과
LOL/llformer 반올림-버림 (dB)                               0.0112             통과
LOL/gsad 성분 합                                        100.0000             통과
LOL/gsad white16 자유도                                   0.0059             통과
LOL/gsad 반올림-버림 (dB)                                   0.0788             통과
LOL/lightendiff 성분 합                                 100.0000             통과
LOL/lightendiff white16 자유도                            0.0058             통과
LOL/lightendiff 반올림-버림 (dB)                            0.0481             통과
LOL/cidnet_woperc 성분 합                               100.0000             통과
LOL/cidnet_woperc white16 자유도                          0.0061             통과
LOL/cidnet_woperc 반올림-버림 (dB)                          0.1103             통과
LOL/cidnet_wperc 성분 합                                100.0000             통과
LOL/cidnet_wperc white16 자유도                           0.0060             통과
LOL/cidnet_wperc 반올림-버림 (dB)                           0.0315             통과
LOL/retinexformer 성분 합                               100.0000             통과
LOL/retinexformer white16 자유도                          0.0059             통과
LOL/retinexformer 반올림-버림 (dB)                          0.0818             통과
Sony/input 성분 합                                      100.0000             통과
Sony/input white16 자유도                                 0.0064             통과
Sony/input 반올림-버림 (dB)                                 0.0000             통과
Sony/gamma 성분 합                                      100.0000             통과
Sony/gamma white16 자유도                                 0.0059             통과
Sony/gamma 반올림-버림 (dB)                                 0.0219             통과
Sony/clahe 성분 합                                      100.0000             통과
Sony/clahe white16 자유도                                 0.0060             통과
Sony/clahe 반올림-버림 (dB)                                 0.0000             통과
Sony/zerodcepp 성분 합                                  100.0000             통과
Sony/zerodcepp white16 자유도                             0.0060             통과
Sony/zerodcepp 반올림-버림 (dB)                            -0.0043             통과
Sony/sci 성분 합                                        100.0000             통과
Sony/sci white16 자유도                                   0.0060             통과
Sony/sci 반올림-버림 (dB)                                  -0.0081             통과
Sony/ar 성분 합                                         100.0000             통과
Sony/ar white16 자유도                                    0.0060             통과
Sony/ar 반올림-버림 (dB)                                    0.0271             통과
Sony/lightendiff 성분 합                                100.0000             통과
Sony/lightendiff white16 자유도                           0.0057             통과
Sony/lightendiff 반올림-버림 (dB)                           0.0170             통과
Sony/snrnet 성분 합                                     100.0000             통과
Sony/snrnet white16 자유도                                0.0057             통과
Sony/snrnet 반올림-버림 (dB)                                0.0104             통과
Sony/retinexformer 성분 합                              100.0000             통과
Sony/retinexformer white16 자유도                         0.0057             통과
Sony/retinexformer 반올림-버림 (dB)                         0.0115             통과
LOL/cidnet_woperc psnr vs diag_lolv1                  23.6087    23.6091  통과
LOL/cidnet_woperc ssim vs diag_lolv1                   0.8703     0.8703  통과
LOL/cidnet_woperc glob vs diag_lolv1                  75.6797    75.6800  통과
LOL/cidnet_woperc resid vs diag_lolv1                 23.2563    23.2560  통과
LOL/cidnet_woperc d16 vs diag_lolv1                    2.2889     2.2889  통과
LOL/cidnet_woperc lf vs diag_lolv1                    69.5701    69.5692  통과
LOL/cidnet_wperc psnr vs diag_lolv1                   23.8398    23.8396  통과
LOL/cidnet_wperc ssim vs diag_lolv1                    0.8574     0.8574  통과
LOL/cidnet_wperc glob vs diag_lolv1                   74.9102    74.9107  통과
LOL/cidnet_wperc resid vs diag_lolv1                  23.9419    23.9414  통과
LOL/cidnet_wperc d16 vs diag_lolv1                     2.1110     2.1110  통과
LOL/cidnet_wperc lf vs diag_lolv1                     67.1505    67.1514  통과
LOL/retinexformer psnr vs diag_lolv1                  25.1523    25.1528  통과
LOL/retinexformer ssim vs diag_lolv1                   0.8454     0.8455  통과
LOL/retinexformer glob vs diag_lolv1                  41.3769    41.3784  통과
LOL/retinexformer resid vs diag_lolv1                 54.5128    54.5102  통과
LOL/retinexformer d16 vs diag_lolv1                    1.1710     1.1707  통과
LOL/retinexformer lf vs diag_lolv1                    68.0645    68.0607  통과
Sony/retinexformer psnr vs diag_sid_*                 24.4383    24.4383  통과
Sony/retinexformer ssim vs diag_sid_*                  0.6800     0.6800  통과
Sony/retinexformer glob vs diag_sid_*                 27.0341    27.0302  통과
Sony/retinexformer chan vs diag_sid_*                  8.7548     8.7555  통과
Sony/retinexformer resid vs diag_sid_*                64.2111    64.2143  통과
Sony/retinexformer d16 vs diag_sid_*                   1.1937     1.1937  통과
Sony/retinexformer lf vs diag_sid_*                   64.3505    64.3506  통과
```
실패 0건 / 전체 98건.

## 3. 정성 그림

- fig_cmp_lol.pdf: LOLv1 665.png. 선택 기준 = LOL frame with the largest PSNR range (max minus min) over the three learned methods SCI, CIDNet (w/o perceptual loss) and Retinexformer, 범위 20.07 dB (SCI 9.75 / CIDNet 29.82 / Retinexformer 27.26).
  Input 6.82, Gamma 0.5 9.81, CLAHE 7.71, Zero-DCE++ 11.14, SCI 9.75, AdaptiveRetinex (ours) 8.76, URetinex-Net 20.88, SNR-Net 27.53, LLFormer 25.52, GSAD 28.55, LightenDiffusion 20.11, CIDNet 29.82, CIDNet (perc.) 27.22, Retinexformer 27.26
  차점: 23.png 17.85, 22.png 14.35, 79.png 14.33
- fig_cmp_sony.pdf: Sony 10198_00_0.033s.npy (Sony frame 10198 at 0.033 s, the frame on which Retinexformer scores lowest (same failure case as fig_qual)). Retinexformer 0.033s 최악 장 = 10198_00_0.033s.npy 17.66 dB, 이 프레임과 동일: True.
  Input 7.62, Gamma 0.5 9.33, CLAHE 9.15, Zero-DCE++ 8.52, SCI 7.96, AdaptiveRetinex (ours) 8.44, LightenDiffusion 11.55, SNR-Net 16.11, Retinexformer 17.66
- 구성: 윗줄 입력(밝기정합 표시) | Gamma | SCI | CIDNet | Retinexformer | GT, 아랫줄 각 방법의 전역이득 보정 후
  (Sony 는 CIDNet 자리를 CLAHE 로 대체 — CIDNet Sony 가중치는 위 0절 사유로 같은 데이터에 못 쓴다)
  f<0.10 저역통과 휘도 잔차(coolwarm, 그림 안 공통 색범위 = 합동 |D| 99 백분위). 그림 안 수치는 paper/fig_cmp_numbers.json.
- 캔버스는 전폭(7.16in) 기준이고 높이는 6열 2행 실제 프레임 비율로 정해진다(LOL 3:2, Sony 15:8 프레임을
  자르지 않았다). 16:9 를 강제하면 여백만 생기므로 프레임 원비율을 유지했다.

## 4. 측정이 가리키는 것

- 순위: LOL Retinexformer > SNR-Net > CIDNet (perc.) > LLFormer > CIDNet > GSAD > LightenDiffusion > URetinex-Net > Zero-DCE++ > SCI > Gamma 0.5 > AdaptiveRetinex (ours) > CLAHE > Input; Sony Retinexformer > SNR-Net > LightenDiffusion > Gamma 0.5 > CLAHE > AdaptiveRetinex (ours) > Zero-DCE++ > SCI > Input. Retinexformer 가 SOTA 외 최선을 LOL 에서 0.5 dB(SNR-Net 24.61), Sony 에서 1.6 dB(SNR-Net 22.87) 앞선다. SCI·AdaptiveRetinex(자체)·CLAHE·Gamma 는 두 벤치마크 모두 15 dB 아래 — 극암 입력을 못 다룬다(실패로 기록). LOL SSIM 최고는 CIDNet 0.870(PSNR 은 RF 가 앞섬).
- 성분 몫은 방법·벤치마크마다 다르다: 전역이득 몫 LOL CIDNet 76 / RF 41 / SCI 70 / Gamma 92 %, Sony RF 27 / Gamma 19 / SCI 41 %. 구조 잔차가 과반인 것은 Sony 전 방법(49~81%)이고 LOL 에선 RF(55%)뿐이다.
- Δ16 은 모든 방법·두 벤치마크에서 white 대조군(0.006)보다 두 자릿수 크다: SOTA 가 LOL 1.17~2.29, Sony 1.19 dB 이고 무학습 출력도 같은 자릿수다(Input 3.15/1.14, CLAHE 2.99/1.22, Gamma 0.52/1.12; LOL/Sony). 국소 이득 오라클 여지는 특정 모델의 결함이 아니라 출력 전반에 남는 항이고, SOTA 도 그것을 닫지 못한다.

## 5. 특이사항

- Sony SCI 는 반올림(11.805) 이 버림(11.813) 보다 낮다(-0.008 dB). 출력이 GT 보다 밝은 쪽으로
  치우쳐 있어 1/255 깎는 버림이 유리한 것. 그래서 반올림-버림 판정은 부호 없이 |차| < 0.2 dB 로 둔다. CLAHE·Input 은 u8 원본이라 차 0.
- 기존 fig_qual.pdf 의 Sony 10198 0.033s 수치 15.1 dB 는 make_figure2.py 가 npy 를 채널 뒤집기 없이 넣어 나온 값이다
  (실측 재현: 같은 프레임·같은 SID.pth 로 무뒤집기 15.119 dB, 뒤집기+u8 반올림 17.659 dB — 차이 전부가 뒤집기).
  저자 로더 규약(BGR->RGB, repro/diag 공통 sid_load)으로는 같은 프레임이 17.66 dB(diag_sid_failure.json 장별 값과 일치)이고
  여전히 0.033s 최악 장이다. 이 그림·캡션은 sid_load 로 다시 만들어야 한다(이번 작업 범위 밖, 손대지 않음).
- Sony SCI/AdaptiveRetinex 는 Input 대비 +0.3/+1.4 dB 뿐이다. 0.033s 에서는 Gamma 10.7, CLAHE 10.8 dB 로 사실상 잡음이다(그림 참조).
