# SID (Sony_total_dark) 데이터 정비 기록

작성 260905. 작업 스크립트: code/prep_sid.py
실행 로그: numbers/prep_sid_run.log

## 1. 확인된 사실 (숫자)

### 원자료
경로 $LLDATA/lowlight_model/data/SID_raw/SID/{short_sid2,long_sid2}
출처 Retinexformer 저자 배포 sid_processed.zip

- 씬 폴더 수: short_sid2 231개, long_sid2 231개 (씬ID 집합 동일)
- 씬ID 접두: `0` 시작 161개, `1` 시작 50개, `2` 시작 20개
- 파일 총수: short 2697장, long 231장 (합 2928장)
- long 은 씬당 정확히 1장. short 는 씬당 8~22장 (분포: 10장 98씬, 12장 62씬,
  14장 42씬, 15장 9씬, 13장 7씬, 9장 6씬, 16장 3씬, 8장 2씬, 19장 1씬, 22장 1씬)
- 모든 .npy 파일 크기 1474688 바이트로 동일
- 배열: shape (512, 960, 3), dtype uint8, 값범위 0..255 (전수 2928장 확인)
- 노출시간 종류
  - short: 0.1s 1891장, 0.04s 699장, 0.033s 107장
  - long: 10s 125장, 30s 106장
- 접두별 노출 분포
  - 접두 0 (161씬): short 0.1s 1340 / 0.04s 457 / 0.033s 68, long 10s 85 / 30s 76
  - 접두 1 (50씬): short 0.1s 390 / 0.04s 180 / 0.033s 28, long 10s 29 / 30s 21
  - 접두 2 (20씬): short 0.1s 161 / 0.04s 62 / 0.033s 11, long 10s 11 / 30s 9

### Retinexformer 쪽 규약
파일: Retinexformer/basicsr/data/SID_image_dataset.py, Options/RetinexFormer_SID.yml

- dataroot_lq = data/SID/short_sid2, dataroot_gt = data/SID/long_sid2
- 스플릿은 폴더명 첫 글자로 결정한다.
  - phase == train: `'0' in name[0] or '2' in name[0]` → 접두 0 과 2 를 학습에 사용
  - 그 외(val/test): `'1' in name[0]` → 접두 1 만 테스트
- 짝짓기: LQ 씬 폴더의 각 프레임 index i 에 대해 GT 는 항상 같은 이름 GT 폴더의
  `img_paths_GT[0]` (정렬 후 첫 파일) 하나. long 폴더에 파일이 1장뿐이므로
  short N장 대 long 1장 대응이다. 파일명 사이 대응 규칙은 없다.
- train_size [960, 512] 로 원본 해상도 그대로 사용
- 테스트 장수 = 접두 1 씬의 short 파일 수 = 598장 (직접 세어 확인)
- Retinexformer 가 README 표(figure/seven_results.png)에서 보고한 SID 성능:
  PSNR 24.44, SSIM 0.680 (Retinexformer 행). 비교표의 SNR-Net 은 22.87 / 0.625

### CIDNet 계열 쪽 규약
파일: $LLDATA/lowlight_model/CIDNet_extension/{eval_SID.py, measure_SID.py,
data/data.py, data/eval_sets.py, data/util.py, Readme.md}

- eval_SID.py: `../datasets/Sony_total_dark/test/short/` 아래 `1` + zfill(4) 로
  index 1..229 를 훑는다 (10001~10229). 폴더가 존재하는 것만 처리.
- 입력 로더 DatasetFromFolderEval: 폴더 안 파일을 `is_image_file` 로 거른 뒤 정렬,
  PIL 로 열어 ToTensor. 허용 확장자는 .png .jpg .bmp .JPG .jpeg
- 출력은 `output/SID/<씬ID>/<입력파일명>` 으로 저장한다. 즉 입력 파일명이 곧 출력 파일명.
- measure_SID.py: index 1..256 을 훑고 GT 는
  `./datasets/Sony_total_dark/test/long/<씬ID>/` 안의 `listdir()[0]` 한 장.
  즉 GT 는 "폴더 안 유일 파일"이고 파일명 대응 규칙이 없다.
- 이 배포본에는 SID 학습 코드가 없다. Sony_total_dark 를 참조하는 코드는
  eval_SID.py 와 measure_SID.py 두 개뿐이고 둘 다 test/ 만 쓴다.

## 2. 같은 벤치마크인지 판정

판정: 같은 데이터다. 근거는 다음 네 가지가 모두 맞아떨어진다.

1. 원자료 자체가 동일하다. Retinexformer README 의 디렉터리 트리가
   `SID/short_sid2/00001/00001_00_0.04s.npy` 형식을 그대로 명시하고, 우리가 받은
   파일이 그 형식이다. 우리가 받은 것이 Retinexformer 저자 배포 sid_processed 다.
2. 씬ID 접두 규칙이 두 코드에서 일치한다. Retinexformer 로더는 테스트를
   접두 `1` 로 고르고, CIDNet eval_SID.py 는 `'1' + zfill(4)` 로 폴더를 찾는다.
   실제 접두 1 씬은 10003~10228 의 50개이고, eval_SID 의 index 범위 1..229
   (10001~10229) 안에 50개가 전부 들어간다.
3. 테스트 장수가 일치한다. 접두 1 short 파일 수 598장. SID Sony 테스트셋의
   통상 보고치와 같은 값이다.
4. 짝짓기 규칙이 일치한다. 양쪽 모두 "씬 폴더 안 long 1장을 그 씬 short 전부의 GT 로
   쓴다"이다 (Retinexformer `img_paths_GT[0]`, CIDNet `listdir(label_dir)[0]`).
   long 폴더에 파일이 1장뿐이라 두 규칙의 결과가 같다.
5. 해상도가 일치한다. 원본 npy 가 (512, 960, 3) 이고 Retinexformer yml 의
   train_size 가 [960, 512] 다. CIDNet 은 리사이즈 없이 원본 그대로 쓴다.

## 3. 변환 결과

출력: $LLDATA/lowlight_model/data/Sony_total_dark/

```
Sony_total_dark/
  train/{short,long}/<0XXXX>/   161씬, short 1865장, long 161장
  test/ {short,long}/<1XXXX>/    50씬, short  598장, long  50장
  eval/ {short,long}/<2XXXX>/    20씬, short  234장, long  20장
```

- 총 2928장 PNG, 디스크 3.0G. 변환 전 /data 여유 1.9T (300GB 기준 통과)
- 파일명은 확장자만 .npy → .png 로 바꿨다. 예: `10003_00_0.04s.npy` → `10003_00_0.04s.png`
- 비트깊이 처리: 원본이 이미 uint8 (0..255) 이므로 스케일링이 필요 없다.
  반올림(round)도 버림(trunc)도 사용하지 않았다. uint8 배열을 그대로
  PIL 로 PNG 무손실 저장했다. 스크립트는 uint8 이 아닌 배열을 만나면
  예외를 던지고 중단하도록 되어 있다 (조용한 값 변형 방지).

### 무손실 검증
- 1차: prep_sid.py 내장 검증, 무작위 20장 (seed 0), PIL 로 재독 → 20/20 완전 일치
- 2차 독립 재검증: prep_sid.py 코드를 재사용하지 않고 PIL 대신 cv2.imread 로
  PNG 를 읽어 BGR→RGB 변환 후 원본 npy 와 전수 대조
  → 2928/2928 완전 일치, 불일치 0, 누락 0. shape, dtype, 화소값 전부 동일.

## 4. 확인하지 못한 것 (불확실성)

- 접두 2 씬 20개의 원 용도. Retinexformer 로더는 접두 2 를 학습에 포함시킨다
  (`'0' in name[0] or '2' in name[0]`). CIDNet Readme 의 트리는 train 예시로
  0XXXX 만 보여주고 eval/ 칸은 하위 씬 폴더 예시가 없어 무엇이 들어가는지 명시가 없다.
  여기서는 2XXXX 를 eval/ 에 넣었다. Retinexformer 방식으로 학습하려면
  train/ 과 eval/ 을 합쳐 써야 한다. CIDNet 배포본에 SID 학습 코드가 없어
  원저자가 eval/ 에 무엇을 넣었는지는 확인 불가다.
- CIDNet(HVI-CIDNet) 논문이 보고한 Sony-Total-Dark PSNR/SSIM 수치는 이 저장소
  Readme 에 없어 확인하지 못했다. Retinexformer 수치(24.44 / 0.680)만 확인했다.
- CIDNet 배포 Sony_total_dark 압축본(Baidu/OneDrive)을 직접 받아 파일 단위로
  대조하지는 않았다. 위 판정은 두 저장소의 코드 규약과 우리 원자료의 실제 통계를
  대조한 결과이고, 원저자 PNG 파일과 바이트 단위로 비교한 것은 아니다.
  다만 원자료가 uint8 이라 PNG 인코딩 경로가 달라도 화소값은 같아야 한다.
- CIDNet 이 배포한 sid.pth 가 이 스플릿(접두 0+2 학습)으로 학습됐는지는
  가중치만으로는 확인 불가다. 성능 재현치가 논문값과 크게 어긋나면 이 지점을 의심할 것.
- eval_SID.py 는 `../datasets/...`, measure_SID.py 는 `./datasets/...` 로 기준
  디렉터리가 다르다. 실행 시 심볼릭 링크 위치를 맞춰야 한다.

---

## 정정 (260905, 화소 단위 검증 결과)

위에서 "Retinexformer 의 SID 와 CIDNet 의 Sony_total_dark 는 같은 벤치마크"라고 판정했는데,
그 판정은 구조/파일명/씬수/분할규칙 대조까지만 한 것이고 화소 규약은 대조하지 않았다.
그 단서가 맞았다. 화소 규약은 다르다.

증거: CIDNet 저자 배포 SID 가중치(HF Fediory/HVI-CIDNet weights/SID.pth)를 이 데이터에 돌리면
PSNR 5.29(반올림)/5.32(버림) / SSIM 0.18 이 나온다 (전량 598장, repro/measure_SID.json, 260905 21:02 재실행; 아래 표의 5.496 은 초기 부분집합 탐침값이라 정본이 아니다). 이 체크포인트(CIDNet)의 저자 보고값은 22.904 / 0.676 (arXiv 2502.20272 Table 2)이고, 23.482 / 0.691 은 별도 모델 CIDNet+ 의 값이다.
입력 규약을 의심해서 훑어봤지만 어느 것도 근처에 못 간다.

    입력 변환        PSNR    SSIM
    그대로            5.496   0.193
    x^1.5            11.509   0.317
    x^2.2            14.681   0.421
    x^3.0            14.271   0.409
    x*0.5             8.183   0.301
    x*0.3            11.379   0.412
    x*0.2            13.723   0.465
    x*0.1            13.681   0.483
    x*0.05           13.534   0.513
    x^(1/2.2)         3.764   0.164
    그대로+gated      5.676   0.176
    그대로+gated2     5.496   0.193

한 장 진단: 입력 평균 8.4, GT 평균 63.8 인데 모델 출력 평균이 192.2 다. 과도하게 밝힌다.
저자 학습 입력이 우리 것보다 어두웠다는 뜻이다.

부수 확인: 노출시간별 평균 밝기가 0.04s 20.08 / 0.1s 35.42 로 1.76배 차이난다.
노출비로 증폭된 데이터가 아니다(증폭됐다면 두 값이 같아야 한다).

## 그래서 지금 상태

- 우리 데이터가 Retinexformer 의 SID 벤치마크인지는 Retinexformer 자신의 가중치로
  그들의 보고값 24.44/0.680 이 재현되는지 확인해야 확정된다. 확인 진행 중.
- CIDNet 의 Sony_total_dark 원본은 저자 OneDrive/Baidu 에만 있고 접근이 막혀 있다
  (OneDrive 공유 링크가 비밀번호 보호라 api.onedrive.com 직접 접근이 401).
- 따라서 CIDNet 의 Sony-Total-Dark 22.904/0.676 (및 CIDNet+ 의 23.482/0.691) 은 현재 우리가 재현할 수 없다.
  논문에서 이 수치를 비교표에 쓰려면 저자 데이터를 구하거나, 벤치마크를 우리 것으로
  통일하고 모든 비교 대상을 우리가 직접 돌려야 한다.

## 최종 결론 (260905)

우리 데이터는 Retinexformer 의 SID 벤치마크 정본이다. 근거는 그들의 공개 SID 가중치로
보고값이 재현된다는 것이다.

    측정 방식                       PSNR        SSIM
    저자 공식 코드                  24.440479   0.680034
    독립 추론 + 독립 채점            24.438298   0.680033
    Retinexformer 보고값            24.44       0.680

차이 0.002 dB. 평평한 평균 598장 기준이며(씬 단위 평균은 24.587 로 어긋난다),
가중치 파라미터 수 1,605,701 도 논문 1.61M 과 일치한다.
GT평균 보정을 걸면 25.602 이므로 보고값은 GT평균 미적용이다.

CIDNet 의 Sony-Total-Dark 는 별개 변종으로 확정한다. 채널 순서까지 확인했다.

    CIDNet SID.pth 를 이 데이터에 적용   PSNR    SSIM
    npy 그대로                          5.496   0.1933   (초기 부분집합 탐침값; 전량 598장은 5.29/5.32)
    입력 채널 뒤집기                     5.498   0.1903
    입출력 둘 다 뒤집기                  5.558   0.1942

## 논문 작성 시 지켜야 할 것

CIDNet 이 보고한 Sony-Total-Dark 22.904/0.676 (CIDNet+ 23.482/0.691) 과 Retinexformer 가 보고한 SID 24.44/0.680 은
서로 다른 화소 데이터에서 나온 값이다. 두 수치를 같은 비교표에 나란히 올리면 안 된다.
우리 논문에서는 벤치마크를 Retinexformer 변종으로 통일하고, 비교 대상 방법은 우리가 직접
같은 데이터에 돌려서 수치를 만든다. 저자 보고값을 그대로 옮겨 적지 않는다.
