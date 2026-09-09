# 재현 감사 — LOLv2-real / LOLv2-synthetic, CIDNet + Retinexformer 저자 가중치 (260905)

목적: 원고 재현 표(Table 4)에 세 번째 벤치마크를 붙인다. LOLv1 감사(AUDIT_LOLv1_260905.md)와
같은 방식이다. 추론은 저자 코드의 망 정의와 평가 설정을 그대로 쓰고, 채점은 저자 코드를 쓰지 않고
우리 독립 구현(repro_measure.py 의 psnr_indep / ssim_indep)으로 한다. 대조용으로 저자의 cv2 식과
scikit-image 를 같이 돌린다. 8비트 변환은 반올림/버림 둘 다, GT평균 보정은 적용/미적용 둘 다 잰다.

## 데이터

HF 배포본(okhater/lolv2-real, okhater/lolv2-synthetic)이라 저자 코드가 기대하는
Real_captured/Test/{Low,Normal}, Synthetic/Test/{Low,Normal} 과 폴더 이름이 다르다. 실제 경로:

    벤치마크    입력                                              GT                                            장수   크기
    real       $LLDATA/lowlight_model/data/LOLv2_real/Test/Input  $LLDATA/lowlight_model/data/LOLv2_real/Test/GT  100   600x400
    syn        $LLDATA/lowlight_model/data/LOLv2_syn/Test/Input   $LLDATA/lowlight_model/data/LOLv2_syn/Test/GT   100   384x384

- real: 00690.png ~ 00789.png, 입력과 GT 파일명 동일.
- syn: GT 는 00816405t.png ~ 197770adt.png, 입력은 같은 이름에 접두사 r 이 붙어 있다(r00816405t.png).
  r 을 떼면 100장 전부 일대일로 맞는다. 저자 Retinexformer 코드는 두 폴더를 각각 정렬해 zip 하므로
  같은 짝이 된다.
- 장수·크기는 LOLv2 원본 규격(real 100장 600x400, syn 100장 384x384)과 같다. 원본 압축본과의
  화소 단위 동일성은 확인 수단이 없다(데이터 변종 가능성은 아래 판정에서 언급).

## 가중치와 저자 설정

CIDNet: HF 저장소 Fediory/HVI-CIDNet 의 weights/ 에서 받아 $LLDATA/lowlight_model/weights/author_all/ 에 두었다.

    파일                          원본 경로                          크기(B)   sha256 앞 16자리
    LOLv2_real_best_PSNR.pth     weights/LOLv2_real/best_PSNR.pth   7971269   44a7f92cf02a1da3
    LOLv2_real_best_SSIM.pth     weights/LOLv2_real/best_SSIM.pth   7971269   3d7ad3e93d65effa
    LOLv2_real_w_perc.pth        weights/LOLv2_real/w_perc.pth      7967682   1e0affc8ce89ccee
    LOLv2_syn_wo_perc.pth        weights/LOLv2_syn/wo_perc.pth      7974353   519cc5fc55491cf3
    LOLv2_syn_w_perc.pth         weights/LOLv2_syn/w_perc.pth       7967682   460a9956018935ca

저자 eval.py 의 LOLv2 설정(전사, 임포트 아님):
- --lol_v2_real: model.trans.gated2 = True, model.trans.alpha = 0.84 (w_perc, README 의 "best gt mean"),
  0.80 (best_PSNR), 0.82 (best_SSIM). gated2 는 HVI 역변환 끝에서 최종 RGB 에 alpha 를 곱한다.
  즉 출력을 전역으로 어둡게 만드는 시험 시 스케일이며, 가중치마다 다르게 박혀 있다.
- --lol_v2_syn: gated / gated2 모두 False. 스케일 없음.
- CLI 의 alpha 기본값은 1.0 인데 lol_v2_real 분기에서는 무시되고 위 값이 하드코딩된다.
  지시대로 real 세 가중치는 README 값과 1.0 둘 다 쟀다(1.0 은 gated2 무효와 동치).
- 저장은 ToPILImage 이므로 버림. 저장 PNG 와 float 출력의 버림이 열 개 런 전부에서 화소 단위로 일치했다(최대 차이 0 단계).

README 표(Readme.md 111~127줄)의 보고값과 GT평균 적용 여부:

    가중치                    GT평균   PSNR      SSIM
    LOLv2_real/best_PSNR      없음    23.9040   0.8656
    LOLv2_real/best_SSIM      없음    23.8975   0.8705
    LOLv2_real/best_SSIM      적용    28.3926   0.8873
    LOLv2_real/w_perc         적용    28.1387   0.8920   (표에는 w_prec.pth 로 오타)
    LOLv2_syn/wo_perc         없음    25.7048   0.9419
    LOLv2_syn/wo_perc         적용    29.5663   0.9497
    LOLv2_syn/w_perc          없음    25.1294   0.9388
    LOLv2_syn/w_perc          적용    29.3666   0.9500

  real 의 "wo perc loss" 23.4269/27.7619 와 "best Normal" 24.1106 은 가중치가 (lost) 로 표시되어
  있어 재현 대상이 아니다. w_perc 의 GT평균 미적용값과 best_PSNR 의 GT평균 적용값은 README 에 없다.

Retinexformer: $LLDATA/lowlight_model/weights/retinexformer/pretrain_model/LOL_v2_real.pth,
LOL_v2_synthetic.pth (n_feat 40, stage 1, num_blocks [1,2,2], 1,605,701 파라미터, LOL_v1 과 동일 구조).
보고값은 README 72~76줄 표에 GT평균 적용 설정으로 LOL-v2-real 27.71/0.856, LOL-v2-synthetic 29.04/0.939.
GT평균 미적용값은 README 이미지(seven_results.png) 안에만 있어 텍스트로 확인하지 못했다.
저자 test_from_dataset.py 는 4의 배수로 reflect 패딩(여기서는 두 크기 모두 배수라 패딩 없음),
self-ensemble 없음, PSNR 은 양자화 전 float 출력으로, SSIM 은 img_as_ubyte(반올림)로 계산한다.
GT평균 보정은 RGB 배열에 cv2 BGR2GRAY 를 적용하므로 가중치가 뒤집혀(0.114R+0.587G+0.299B) 들어간다.

## 결과 — CIDNet / LOLv2-real (n=100, 8비트 버림 = 저자 방식)

    가중치      alpha   PSNR     sd     SSIM    | PSNR(GT평균)  sd     SSIM(GT평균)  | 보고값                    차이
    best_PSNR   0.80   23.9045  4.17   0.8656  |   27.9749    2.88     0.8829     | 23.9040/0.8656 (미적용)  +0.0005
    best_PSNR   1.00   20.6523  4.68   0.8408  |   27.9726    2.89     0.8831     | -
    best_SSIM   0.82   23.8976  4.11   0.8705  |   28.3928    2.70     0.8873     | 23.8975/0.8705 (미적용)  +0.0001
                                                                                   28.3926/0.8873 (적용)    +0.0002
    best_SSIM   1.00   20.8643  4.47   0.8454  |   28.3941    2.70     0.8875     | -
    w_perc      0.84   23.2261  5.24   0.8637  |   28.1396    3.20     0.8920     | 28.1387/0.8920 (적용)    +0.0009
    w_perc      1.00   20.3722  4.98   0.8463  |   28.1423    3.20     0.8922     | -

판정: 보고값이 있는 네 칸 전부 0.001 dB 이내로 재현된다. SSIM 은 소수 넷째 자리까지 같다.

## 결과 — CIDNet / LOLv2-synthetic (n=100, 8비트 버림)

    가중치     PSNR     sd     SSIM    | PSNR(GT평균)  sd     SSIM(GT평균)  | 보고값                   차이
    wo_perc   25.7045  4.77   0.9419  |   29.6469    3.74     0.9503     | 25.7048/0.9419 (미적용)  -0.0003
                                                                           29.5663/0.9497 (적용)    +0.0806  벗어남
    w_perc    25.1305  5.07   0.9388  |   29.3651    3.68     0.9500     | 25.1294/0.9388 (미적용)  +0.0011
                                                                           29.3666/0.9500 (적용)    -0.0015

판정: 네 칸 중 셋은 0.002 dB 이내로 재현. wo_perc 의 GT평균 적용 칸만 +0.081 dB 벗어난다(아래).

## 결과 — Retinexformer / LOLv2 (n=100)

    벤치마크  8비트    PSNR     sd     SSIM    | PSNR(GT평균)  sd     SSIM(GT평균)  | 보고값(적용)    차이
    real     반올림   22.7937  3.73   0.8397  |   27.6891    2.58     0.8566     | 27.71/0.856   -0.0209
    real     버림     22.7898  3.72   0.8394  |   27.6511    2.62     0.8562     |
    real     저자식   22.7951  (float PSNR)   |   27.7070             0.8564     |               -0.0030
    syn      반올림   25.6676  4.97   0.9295  |   29.0224    3.34     0.9386     | 29.04/0.939   -0.0176
    syn      버림     25.6467  4.95   0.9299  |   29.0394    3.36     0.9390     |
    syn      저자식   25.6711  (float PSNR)   |   29.0370             0.9385     |               -0.0030

저자식 = 양자화 없는 float 출력에 뒤집힌 회색 가중치로 보정한 뒤 float PSNR (저자 코드의 절차를 전사).
판정: 반올림 규칙 기준 0.021 / 0.018 dB 차이로 재현. 저자 절차대로 계산하면 0.003 dB 로 줄어들어
남은 차이는 PSNR 을 양자화 전에 재는 저자 관례로 설명된다. 보고값이 소수 둘째 자리라 그 이상은 판별 불가.

## 세 구현의 SSIM 대조

같은 저장 PNG 에 대해 독립 재구현 대 저자 cv2 식, scikit-image 의 평균 차이:

    런                        독립 재구현    cv2 식 차이     scikit-image 차이
    real best_PSNR a0.80     0.865611     +8.88e-16      +1.89e-15
    real best_SSIM a0.82     0.870455     +0.00e+00      +1.11e-15
    real w_perc a0.84        0.863742     +8.88e-16      +2.00e-15
    syn wo_perc              0.941924     -9.99e-16      -6.66e-16
    syn w_perc               0.938788     -2.22e-16      -5.55e-16
    real Retinexformer       0.839671     +1.11e-16      +1.22e-15
    syn Retinexformer        0.929540     +5.55e-16      +4.44e-16

열 개 런 전부 1e-15 수준. LOLv1 과 같다. SSIM 계산에는 이견이 없다.
GT평균 보정도 저자식(cv2 uint8 회색 → 평균)으로 따로 계산해 봤고 우리 float 휘도식과 0.0003 dB 이내였다
(cv2 5.0 의 uint8 회색 변환 전사는 화소 0.27% 에서 1단계 다르지만 평균 비율에는 1e-5 수준).

## 벗어난 한 칸 — syn wo_perc 의 GT평균 적용 PSNR

우리 29.6469 대 보고 29.5663, +0.081 dB. SSIM 도 0.9503 대 0.9497.
같은 가중치의 GT평균 미적용 칸은 -0.0003 으로 정확히 맞으므로 출력 자체는 같다. 차이는 보정 단계나
표의 출처에 있다. LOLv1 에서도 같은 자리(wo_perc 의 GT평균 적용 칸)만 +0.060 벗어났다.
보정식 변종으로 보고값이 나오는지 확인했다(맞추려는 게 아니라 후보 배제용):

    보정 변종                              LOLv1 wo_perc   LOLv2syn wo_perc   LOLv2syn w_perc
    보고값                                 28.1405         29.5663            29.3666
    회색비율 (우리)                        28.2008         29.6469            29.3651
    회색비율 후 uint8 반올림               28.1938         29.6408            29.3601
    회색비율 후 uint8 버림                 28.1707         29.5743            29.2983
    RGB 전체평균 비율                      28.2125         29.6635            29.3585
    채널별 비율                            28.4661         29.7615            29.5279
    float 출력에 회색비율                  28.2332         29.6575            29.4002
    float 출력에 회색비율 후 uint8 버림    28.2041         29.5864            29.3381
    가산 오프셋                            28.5483         30.4144            30.3974

어느 변종도 두 wo_perc 칸을 동시에 맞추면서 이미 맞는 w_perc 칸을 깨지 않는 것이 없다.
보정식 차이로는 설명되지 않는다. 남는 후보는 (1) 그 칸이 배포 가중치와 다른 학습 회차 산출물,
(2) 표 작성 시점의 measure.py 가 지금과 다름, (3) 데이터 변종. (3)은 미적용 칸이 정확히 맞으므로
가능성이 낮다. 확인 수단이 없어 미해결로 둔다. 차이가 우리 쪽이 높은 방향이라 우리에게 유리한
실수는 아니다. 원고에서는 저자 보고값을 쓰고 우리 재측정값을 병기한다.

## 부수 확인 1 — alpha 가 LOLv2-real 미적용 수치에 미치는 영향

    가중치      alpha=README   alpha=1.0   차이(dB)   | GT평균 적용 차이
    best_PSNR   23.9045        20.6523     +3.25      |  +0.002
    best_SSIM   23.8976        20.8643     +3.03      |  -0.001
    w_perc      23.2261        20.3722     +2.85      |  -0.003

출력에 0.80~0.84 를 곱하는 것만으로 GT평균 미적용 PSNR 이 3 dB 오른다. GT평균 적용값은 전역
스케일이 보정에서 상쇄되므로 변하지 않는다. 즉 CIDNet 의 LOLv2-real "GT평균 미적용" 수치는 가중치별로
따로 정한 시험 시 밝기 스케일을 포함한 값이다. 이 스케일은 GT 를 보고 고른 상수이므로 성격상 GT평균
보정의 전역 한 점 근사에 가깝다. 원고에서 LOLv2-real 미적용 열을 인용할 때 이 사실을 각주로 적는다.
우리 비교표에서는 alpha 없이(1.0) 같은 규칙으로 다시 돌린 값을 쓴다.

## 부수 확인 2 — 8비트 변환 규칙

    런                    버림(저자)   반올림     차이
    real best_PSNR 0.80   23.9045     23.8884   -0.016
    real best_SSIM 0.82   23.8976     23.8881   -0.010
    real w_perc 0.84      23.2261     23.2218   -0.004
    syn wo_perc           25.7045     25.7164   +0.012
    syn w_perc            25.1305     25.1616   +0.031
    real Retinexformer    22.7898     22.7937   +0.004
    syn Retinexformer     25.6467     25.6676   +0.021

LOLv1 의 0.03~0.11 보다 작고 방향도 일정하지 않다(alpha 로 어둡게 만든 real 출력은 버림이 오히려 높다).
규칙을 통일해야 한다는 결론은 같다. 우리 표는 반올림으로 통일한다.

## 재현 방법

    conda activate edge
    CUDA_VISIBLE_DEVICES=0 python code/repro_lolv2.py --workers 20
    # 영상별 PNG/npy: $LLDATA/lowlight_model/repro_out/LOLv2_<런태그>/ (2.7 GB, 홈 디스크 96% 라 /data 에 둠)
    # 수치: repro/measure_LOLv2.json (runs.<태그>.avg/std/per_image, table4_rows), 로그: repro/repro_lolv2_run.log

## 원고 Table 4 행 (벤치마크 / 방법·설정 / 우리 / 보고값)

8비트 규칙은 저자 파이프라인과 같게(CIDNet 버림, Retinexformer 반올림). 우리 값은 PSNR/SSIM.

    벤치마크     방법·설정                              우리              보고값           차이
    LOLv2-real  CIDNet (best PSNR, alpha 0.80), 미적용  23.9045/0.8656   23.9040/0.8656   +0.0005
    LOLv2-real  CIDNet (best SSIM, alpha 0.82), 미적용  23.8976/0.8705   23.8975/0.8705   +0.0001
    LOLv2-real  CIDNet (best SSIM, alpha 0.82), 적용    28.3928/0.8873   28.3926/0.8873   +0.0002
    LOLv2-real  CIDNet (w_perc, alpha 0.84), 적용       28.1396/0.8920   28.1387/0.8920   +0.0009
    LOLv2-real  Retinexformer, 적용                     27.6891/0.8566   27.71/0.856      -0.0209
    LOLv2-syn   CIDNet (wo_perc), 미적용                25.7045/0.9419   25.7048/0.9419   -0.0003
    LOLv2-syn   CIDNet (wo_perc), 적용                  29.6469/0.9503   29.5663/0.9497   +0.0806 (벗어남, 미해결)
    LOLv2-syn   CIDNet (w_perc), 미적용                 25.1305/0.9388   25.1294/0.9388   +0.0011
    LOLv2-syn   CIDNet (w_perc), 적용                   29.3651/0.9500   29.3666/0.9500   -0.0015
    LOLv2-syn   Retinexformer, 적용                     29.0224/0.9386   29.04/0.939      -0.0176

같은 행이 repro/measure_LOLv2.json 의 table4_rows 에 들어 있다(키: bench, setting, ours, reported,
ours_ssim, reported_ssim, ours_psnr_std, diff, reproduced, rule8bit, run).

## 현재 검증된 기준선 목록 (LOLv1 문서의 표에 추가)

    벤치마크     방법                       설정          우리 재측정        저자 보고        차이
    LOLv2-real  CIDNet best_PSNR (a0.80)   GT평균 없음   23.9045/0.8656    23.9040/0.8656  +0.0005
    LOLv2-real  CIDNet best_SSIM (a0.82)   GT평균 없음   23.8976/0.8705    23.8975/0.8705  +0.0001
    LOLv2-real  CIDNet best_SSIM (a0.82)   GT평균 적용   28.3928/0.8873    28.3926/0.8873  +0.0002
    LOLv2-real  CIDNet w_perc (a0.84)      GT평균 적용   28.1396/0.8920    28.1387/0.8920  +0.0009
    LOLv2-real  Retinexformer              GT평균 적용   27.6891/0.8566    27.71/0.856     -0.021
    LOLv2-syn   CIDNet wo_perc             GT평균 없음   25.7045/0.9419    25.7048/0.9419  -0.0003
    LOLv2-syn   CIDNet w_perc              GT평균 없음   25.1305/0.9388    25.1294/0.9388  +0.0011
    LOLv2-syn   CIDNet w_perc              GT평균 적용   29.3651/0.9500    29.3666/0.9500  -0.0015
    LOLv2-syn   Retinexformer              GT평균 적용   29.0224/0.9386    29.04/0.939     -0.018
    (LOLv2-syn  CIDNet wo_perc             GT평균 적용   29.6469/0.9503    29.5663/0.9497  +0.081  미해결)
