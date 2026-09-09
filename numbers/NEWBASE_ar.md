# AR baseline (Table 1 rows "AR", in-house)

- Model: `AdaptiveRetinex` (`weights/adaptive_retinex.py`, ch=12, 691 parameters; checkpoint `weights/adaptive_retinex_trained.pth`,
  18 tensors, 691 parameters, both counted from the files).
- Training: zero-reference (self-supervised) on COCO val2017 (`image_dir` default `.../val2017`, dataset repeat 4,
  script defaults: image size 192, batch 64, 30 epochs, AdamW lr 1e-3 -> 1e-5 cosine, weight decay 1e-4);
  losses: one-sided exposure target 0.55, brightness identity on already-bright samples, chroma and local-contrast preservation
  (see `train_adaptiveRetinex.py` in the same in-house repository; not part of this release).
- No paired low-light data was used, so it is listed with the reference-free methods in Table 1.
- Code path: `code/enhancers.py::ARArm` (input BGR uint8 -> RGB float [0,1], output clamped); `code/compare_methods.py` scores the
  float output with the same scorer as every other row (8-bit by rounding).
- No rectification, full frames on both benchmarks.
