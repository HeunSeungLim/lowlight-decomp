# 워크플로 그림 고정 명세 (260905) — 이 명세가 정본.

비율 21:9 전폭 figure*. 글 최소. 사진 칸은 실제 데이터로 코드가 채운다(생성 이미지 금지).

## 상단 한 줄: 왼쪽 → 오른쪽 5단

[P1] 사진 칸: 실제 short 입력 프레임 (Sony 10198, 0.033 s, x20 표시). 라벨: "input x"
   →
[B1] 블록 "frozen model f"  수식 한 줄: y = f(x)     (오른쪽에 사진 칸 [P2]: 실제 출력 y)
   →
[B2] 블록 "residual"  수식: r = a*y - g,  a = <y,g>/<y,y>     (사진 칸 [P3]: 실제 GT g)
   →
[B3] 블록 "decompose"  세 갈래 화살표로 나뉨:
      - "global / channel gain"      기호: 스칼라 a, a_c
      - "block field"                 기호: a_blk (16 px)   (사진 칸 [P4]: 실제 저주파 잔차 맵, coolwarm)
      - "radial bands"                기호: rho_j
   →
[B4] 블록 "ceiling with DoF control"  수식: Delta = PSNR(oracle) - PSNR(frame) - Delta_white
   →
[B5] 블록 "identifiability"  수식: rho_j^out vs rho_j^in(k)   판정 세 글자: recoverable / exhausted / unidentifiable

## 하단 사실 패널 3개 (실제 데이터 그래프, 코드 생성)

[F1] band-wise rho: model vs input, 0.033 s and 0.1 s, plus input k=8   (= 기존 Fig.1a)
[F2] local correction ceiling vs block side, gain/affine, DoF-corrected  (= 기존 Fig.1b)
[F3] frame averaging: LF error energy vs k (1.00 / 1.09 / 1.29 / 1.52) with 1/k reference line

## 화살표 방향: 전부 왼→오. B3 에서 세 갈래는 위/중/아래로 갈라져 B4 로 다시 모임.
## 금지: 추가 모듈, 추가 수식, 가짜 그래프, 장식 박스, 그라데이션, 둥근 카드, 그림자, 텍스트 렌더(라벨은 코드가 얹음).
## 색: navy(블록 테두리) · teal(측정 경로) · orange(판정) · 나머지 흑백.

## 레이아웃 결정 (지면)
- 이 그림이 기존 Fig.1 (a)(b) 를 하단 패널로 흡수한다 → Fig.1 삭제.
- 기존 Table 3(사다리) 는 F2 패널이 대체 → 삭제.
- 기존 Fig.2(정성) 는 기존 방법 비교 그림(fig_cmp_*)이 대체.
