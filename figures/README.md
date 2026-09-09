# 워크플로 그림 제작 기록 (260905)

규칙: 구조(노드·화살표·라벨·사진 칸)를 코드로 먼저 확정(SPEC.md, make_panels.py → wireframe.png, geometry.json)하고 실제 사진·그래프 패널을 코드로 합성한다(composite.py).


## 정본 생성 순서

    python make_panels.py --clean      # 실제 패널 7장 + 라벨 없는 면(wire_clean.png) + geometry.json
    python composite.py wire_clean.png codedrawn   # 마젠타 칸에 실제 패널 합성(geometry.json 기준 배치)
    python labels_codedrawn.py          # Times 라벨·수식 오버레이 → fig_workflow_codedrawn.png
    cp fig_workflow_codedrawn.png ../../fig_workflow.png

## 정정 (260905 밤)

그림 스크립트가 npy 를 채널 뒤집기 없이(저장은 BGR) 모델에 넣고 있었다. 저자 로더와 재현(24.438 dB)은
BGR→RGB 로 뒤집어 넣는다. make_panels.py / make_figure2.py(릴리즈에는 포함하지 않는 생성기) 를 같은 규약으로 고쳤고, 그림 프레임(10198,
0.033 s)의 정본 점수는 17.66 dB 다(이전 그림의 15.1 dB 는 잘못된 규약의 값이라 폐기). 발견: 기존 방법 비교
작업자의 대조.

## v3 (사용자 지적 반영, 260905 밤)

추상 기호(육각형·사각형·다이아·격자·호·막대·물결)를 전부 뺐다. 그 자리에 실제 데이터: 잔차 영상,
저주파 잔차, 16px 블록 이득장, 고주파 잔차, 대역 판정 띠(노출 2줄, 규칙은 diag_sid_identifiability.verdict 를
그대로 import), 16px 상한 실측치. 기호 띠를 없애 라벨이 패널 바로 위에 붙고 빈 공간이 사라졌다.
frozen model 은 입력→출력 화살표 위 작은 자물쇠로만 표시.


The manuscript uses the vector overview `fig_workflow.pdf`, produced by `fig1_mpl.py` from the panel PNGs, `verdicts.json` and `../paper_tables/evidence.json` (run from this directory). The raster composite path above (`build_overview.py` -> `fig_workflow_codedrawn.png`) is the earlier version and is no longer used by the manuscript; its output is not shipped.

fig_cmp_lol.pdf: the LOL frame is selected by the largest PSNR range over the three learned methods SCI, CIDNet (w/o perceptual loss) and Retinexformer (`pick_lol_frame` in code/compare_methods.py); the four methods displayed are `FIG_METHODS` and differ from that selection set. Both are recorded in fig_cmp_numbers.json (`criterion`, `displayed_set`).
