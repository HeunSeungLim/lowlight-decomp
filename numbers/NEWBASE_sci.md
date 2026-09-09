# SCI baseline (Table 1 rows "SCI")

- Method: SCI, Ma et al., CVPR 2022 (refs.bib key `sci2022`); official repository, checkpoint `weights/medium.pt`
  (one of the three released checkpoints easy/medium/difficult; the released weights are trained on LOL and LSRW).
- Code path: `code/enhancers.py::SCIArm` (`Finetunemodel(weights)`, input BGR uint8 -> RGB float [0,1], output `r.clamp(0,1)`);
  `code/compare_methods.py` takes the float output directly and scores it with the same scorer as every other row
  (8-bit by rounding; the truncation variant is also stored per frame).
- Location: `$LLDATA/enhance_data/lowlight_bench/SCI_official/CVPR` (`SCI_DIR` in `enhancers.py`); weights are not redistributed here.
- No rectification, no reference-derived ratio, full frames on both benchmarks.
