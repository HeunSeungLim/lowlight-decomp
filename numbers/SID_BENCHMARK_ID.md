# SID 벤치마크 정체 확정 (260905)

우리가 가진 SID 데이터가 어느 저자의 전처리본인지 확정하기 위해, Retinexformer 저자의
공개 SID 가중치를 우리 데이터에 그대로 돌려 저자 보고값이 재현되는지 확인했다.

## 결론

우리 데이터는 Retinexformer 의 SID 벤치마크 정본이 맞다. 재현 오차 0.002 dB.

| 항목 | PSNR | SSIM |
|---|---|---|
| Retinexformer 논문 보고값 (ICCV 2023, SID) | 24.44 | 0.680 |
| 저자 공식 코드 `Enhancement/test_from_dataset.py` | 24.440479 | 0.680034 |
| 독립 재구현 추론 + 독립 채점 (반올림 8bit) | 24.438298 | 0.680033 |
| 독립 재구현 추론 + 독립 채점 (버림 8bit) | 24.426810 | 0.680149 |

CIDNet 저자 가중치가 같은 데이터에서 PSNR 5.29(전량 598장 반올림; 입력 감마·게인 보정을 걸어도 최고 14.7)
밖에 안 나온 것은 우리 데이터 문제가 아니다. CIDNet 의 Sony-Total-Dark(22.904 / 0.676; CIDNet+ 23.482 / 0.691)
는 화소 규약이 다른 별개 변종이다.

## 확인된 것 (실측)

### 가중치

- Retinexformer README 3장의 Google Drive 폴더
  `https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV` 에서
  `gdown --folder` 로 10개 가중치 전부 확보.
- 저장 위치: `$LLDATA/lowlight_model/weights/retinexformer/pretrain_model/`
  (`SID.pth` 6,475,085 B, 2023-09-21)
- `SID.pth` 를 `RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
  num_blocks=[1,2,2])` 에 로드 → `All keys matched successfully`, 파라미터 1,605,701 개
  (논문 표기 1.61 M 과 일치).

### 데이터

`$LLDATA/lowlight_model/data/SID_raw/SID/{short_sid2,long_sid2}/` 원본 npy 를 사용했다.
PNG 변환본(`Sony_total_dark/`)이 아니라 npy 를 쓴 이유는 저자 로더가 `np.load` 로
npy 를 직접 읽기 때문이다(`basicsr/data/util.py:read_img2`).

- npy: `(512, 960, 3)` uint8. `train_size: [960, 512]` 이라 `cv2.resize` 는 무작동.
- 저자 테스트 분할 규칙(`SID_image_dataset.py`): 씬 폴더명 첫 글자가 `1` 인 것만 테스트.
  실측 50 씬 / short 598 장 — 저자 코드가 찍은 `test dataset length: 598` 과 일치.
- GT 대응: 씬 폴더의 long npy 중 정렬 0번째 1장을 그 씬의 short 전부에 대응
  (`img_GT_path = self.imgs_GT[folder][0]`). 씬당 long 은 실제로 1장뿐이었다.
- 평균은 short 598 장에 대한 평평한 평균. (참고: 씬 단위로 먼저 평균 낸 뒤
  50 씬 평균을 내면 24.587 로 0.15 dB 높게 나온다. 보고값 24.44 와 맞는 쪽은 평평한 평균이다.)

노출별 분해(반올림, 독립 채점):

| 노출 | 장수 | PSNR |
|---|---|---|
| 0.033s | 28 | 22.491 |
| 0.04s | 180 | 22.909 |
| 0.1s | 390 | 25.284 |
| 전체 | 598 | 24.438 |

장당 PSNR 범위 15.93 ~ 31.04.

### 채점

`code/repro_measure.py` 의 `psnr_indep` / `ssim_indep`
(순수 numpy 재구현)로 독립 채점했고, 비교용으로 저자 공식(`Enhancement/utils.py`)
전사본과 저자 코드 자체 실행값을 같이 냈다. 세 값이 소수 셋째 자리에서 일치한다.

8bit 변환 규칙별 차이는 PSNR 0.011 dB, SSIM 0.0001 로 결론에 영향이 없다.

GT 평균 보정(GT_mean)을 걸면 25.602 / 0.6875 로 올라간다. 보고값 24.44 는
GT_mean 없는 값이다 — 즉 저자는 SID 에 GT_mean 을 쓰지 않았다.

## 실행 방법 (재현)

```
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate edge

# 독립 재현 (권장)
cd code
CUDA_VISIBLE_DEVICES=0 python repro_sid_retinexformer.py --workers 16 \
  --out numbers/measure_SID_retinexformer.json

# 저자 공식 코드 교차 확인
cd <Retinexformer repo>
ln -sfn $LLDATA/lowlight_model/data/SID_raw/SID data/SID
PYTHONPATH=$PWD python Enhancement/test_from_dataset.py \
  --opt Options/RetinexFormer_SID.yml \
  --weights $LLDATA/lowlight_model/weights/retinexformer/pretrain_model/SID.pth \
  --dataset SID --result_dir $LLDATA/lowlight_model/results/retinexformer_SID_authorcode/
```

산출물:
- `numbers/measure_SID_retinexformer.json` (598 장 장별 수치 + 요약)
- `logs/repro_sid_rf.log`
- `$LLDATA/lowlight_model/results/retinexformer_SID_authorcode/` (저자 코드 출력 png)

## 환경 손질 내역

edge 콘다 환경(torch 2.11 + cu128)에서 `pip install lmdb einops addict yapf natsort` 만
추가했다. `basicsr` 은 `python setup.py develop` 하지 않고 `PYTHONPATH` 로 레포를 얹어
그대로 돌렸다. 저자 코드는 `torch.cuda.amp.GradScaler` deprecation 경고만 내고
수정 없이 완주했다 — `torchvision.transforms.functional_tensor` 류 충돌은 없었다.
데이터와 채점 로직은 손대지 않았다.

## 확인 못 한 것

- CIDNet 의 Sony-Total-Dark 가 정확히 어떤 화소 규약인지는 확인하지 못했다.
  여기서 확인한 것은 "우리 데이터 = Retinexformer SID" 뿐이고, CIDNet 것이
  무엇인지는 별도 조사가 필요하다(CIDNet 레포의 SID 데이터 로더와 전처리 스크립트를
  읽어야 한다).
- CIDNet 가중치를 이번의 npy + 저자 로더 규약(BGR→RGB 뒤집기, 씬 0번 GT)으로
  다시 돌려보지는 않았다. 앞선 시도는 PNG 변환본 기준이었다.
- Retinexformer 의 SID PNG 변환본(`Sony_total_dark/`)이 npy 와 화소 단위로 동일한지는
  이번에 대조하지 않았다. 이번 재현은 npy 경로만 사용했다.
