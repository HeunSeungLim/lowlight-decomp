"""audit_*: check that every number the compiled PDF prints exists in the measurement dumps.

The macros in numbers.tex are generated, so the risk is not a typo in a macro but a number
typed straight into the prose or a table during editing. This walks the PDF text, pulls
every decimal figure, and flags any that cannot be matched to a value in evidence.json,
the generated row files, or a short list of declared literals (thresholds, published
values we quote on purpose).
"""
import json, re, subprocess, sys, os

P = os.path.dirname(os.path.abspath(__file__))
def _strip_printed(o):
    """영수증이 담고 있는 인쇄값 사본은 대조 출처가 될 수 없다."""
    if isinstance(o, dict):
        return {k: _strip_printed(v) for k, v in o.items()
                if k not in ("printed", "table_reference", "measured_printed", "reported_source",
                             "reported_on_its_own", "value", "previous_text", "correction", "printed_after",
                             "psnr_sd_ddof0", "sat_train_lol_truncated120",
                             "previous_scope_prefix0_only")}
    if isinstance(o, list): return [_strip_printed(v) for v in o]
    return o

_POOL_BLOCK = ("sat_superseded_161scenes", "rad_edges", "ident_edges", "band_edges", "edges", "bins",
               "cnn_train_scenes", "printed_bounds")

def _strip_pool(o):
    """축·빈 경계와 다른 분모의 장면수는 측정 결과가 아니라 설정값이다. 대조 근거로 쓰지 않는다."""
    if isinstance(o, dict):
        return {k: _strip_pool(v) for k, v in o.items() if k not in _POOL_BLOCK}
    if isinstance(o, list):
        return [_strip_pool(v) for v in o]
    return o

_MISSING_RECEIPTS = []

def _src(path):
    """영수증을 찾는 순서. 고정본(원고 옆)과 공개본(numbers/ 아래) 어느 배치에서도 돌게 한다.
    못 찾으면 저자 트리로 넘어가지 않는다 — 그러면 남의 기계에서만 조용히 실패하기 때문이다."""
    b = os.path.basename(path)
    for cand in (os.path.join(P, b),
                 os.path.join(P, "..", "numbers", "method_260909", b),
                 os.path.join(P, "..", "numbers", b),
                 os.path.join(P, "..", "method_260909", b),
                 os.path.join(P, "..", "repro", b)):
        if os.path.exists(cand):
            return cand
    _MISSING_RECEIPTS.append(b)
    return os.path.join(P, b)

E = json.load(open(os.path.join(P, "evidence.json")))

def leaves(o, acc):
    if isinstance(o, dict):
        for v in o.values(): leaves(v, acc)
    elif isinstance(o, list):
        for v in o: leaves(v, acc)
    elif isinstance(o, (int, float)):
        acc.append(float(o))
    return acc

vals = leaves(_strip_pool(E), [])
# values a table can legitimately show that are derived from the dumps
derived = []
for v in list(vals):
    derived += [v, v * 100, v / 100]
for r in E["ladder"]:
    for k in ("gain", "gain_dof", "affine", "affine_dof"):
        if r.get(k) is not None: derived.append(r[k])
for _f in ("diag_sid_identifiability.json",
           "diag_sid_lowfreq.json"):
    try: _d = json.load(open(_src(_f)))
    except Exception: continue
    def _scal(o, acc):                       # 배열은 제외: 프레임별 값을 다 넣으면 검사가 무의미해진다
        if isinstance(o, dict):
            for v in o.values():
                if not isinstance(v, list): _scal(v, acc)
        elif isinstance(o, (int, float)) and not isinstance(o, bool): acc.append(float(o))
        return acc
    for _v in _scal(_d, []):
        derived += [_v, round(_v, 2), round(_v, 3)]

def _leaves_all(o, acc):
    if isinstance(o, dict): [_leaves_all(v, acc) for v in o.values()]
    elif isinstance(o, list): [_leaves_all(v, acc) for v in o]
    elif isinstance(o, (int, float)) and not isinstance(o, bool): acc.append(float(o))
    return acc
for _v in _leaves_all(_strip_pool(E), []):   # 진단 측정 덤프 전량 (설정값은 _strip_pool 이 걸러낸다)
    derived += [_v, round(_v, 1), round(_v, 2), round(_v, 3)]

declared_tex = []   # 생성된 tex: 인쇄값의 사본이므로 독립 출처가 아니다
for f in ("tab_rho_rows.tex", "tab_cmp_rows.tex"):
    p = os.path.join(P, f)
    if os.path.exists(p):
        declared_tex += [float(x) for x in re.findall(r"\d+\.\d+", open(p).read())]
declared_tex += [float(x) for x in re.findall(r"\{([\d.]+)\}\s*$", open(os.path.join(P, "numbers.tex")).read(), re.M)]

# numbers we quote deliberately from other people's papers or fix by declaration
DECLARED = {
    27.84: "GSAD published LOL-v1 PSNR, rectified (repo results table)",
    0.003: "SNR-Net/LLFormer/URetinex reproduction gap (NEWBASE_*.md)", 0.13: "GSAD gap to published, rectified (NEWBASE_gsad.md)", 0.34: "LightenDiffusion gap to published (NEWBASE_lightendiff.md)",
    22.904: "CIDNet reported Sony-Total-Dark (arXiv 2502.20272 Table 2)",
    0.9: "N/P instability threshold for the noise correction (Sec. 2, code)",
    23.500: "CIDNet reported LOLv1 wo perc",
    27.18: "Retinexformer reported LOLv1 with rectification",
    24.44: "Retinexformer reported SID",
    23.498: "our CIDNet wo perc",
    23.808: "our CIDNet w perc",
    27.169: "our Retinexformer LOLv1",
    24.438: "our Retinexformer SID",
    1.16: "rectification worth on Sony",
    28.14: "CIDNet published rectified LOLv1",
    0.021: "reproduction tolerance (LOLv2)",
    0.081: "reproduction gap, LOLv2-syn wo_perc rect.",
    0.002: "reproduction tolerance",
    0.10: "low-pass cutoff",
    0.87: "single-batch overfit control",
    0.033: "exposure",
    0.04: "exposure",
    0.1: "exposure",
    0.0: "axis tick",
    0.2: "axis tick",
    0.4: "axis tick",
    0.5: "axis tick",
    0.6: "axis tick",
    0.8: "axis tick",
    15.0: "LOL frames",
}

for _extra in ("fix_v52.json",
               "robust_acting.json",
               "qsweep_rule.json",
               "final_rule.json",
               "shares.json",
               "controls_v58.json",
               "precond_transfer.json",
               "affine_oracle.json",
               "concentration.json",
               "tost_equivalence.json",
               "decomp_anchored.json",
               "anchor_lol_retinexformer.json",
               "protocol_pricing.json",
               "unresolved_cells.json",
               "anchored_sony_sd.json",
               "tab3_anchored_column.json",
               "qsat_train181.json",
               "protocol_sweep.json",
               "multiframe_drop.json",
               "bootstrap_config.json"):
    if os.path.exists(_src(_extra)):
        def _lvx(o, acc):
            if isinstance(o, dict): [_lvx(v, acc) for v in o.values()]
            elif isinstance(o, list): [_lvx(v, acc) for v in o]
            elif isinstance(o, (int, float)): acc.append(float(o))
            return acc
        for v in _lvx(_strip_pool(_strip_printed(json.load(open(_src(_extra))))), []):
            derived += [v, round(v, 1), round(v, 2), round(v, 3), float(round(v))]
_fe = os.path.join(P, "final_evidence.json")                  # 최종 규칙 실측값·백분위 라벨
if os.path.exists(_src(_fe)):
    def _lv4(o, acc):
        if isinstance(o, dict): [_lv4(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv4(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        elif isinstance(o, str):
            try: acc.append(float(o))
            except ValueError: pass
        return acc
    _fej = _strip_pool(_strip_printed(json.load(open(_src(_fe))))); _fej.pop("printed", None)      # 인쇄값이 스스로를 검증하지 않도록 제외
    for v in _lv4(_fej, []):
        derived += [v, round(v, 2), round(v, 3), round(v, 4)]
        if abs(v - round(v)) < 1e-9: ints_extra = int(round(v))
_an = os.path.join(P, "anchor_evidence.json")                 # 앵커판 실측값
if os.path.exists(_src(_an)):
    def _lv3(o, acc):
        if isinstance(o, dict): [_lv3(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv3(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv3(_strip_pool(_strip_printed(json.load(open(_src(_an))))), []):
        derived += [v, round(v, 2), round(v, 3), v * 100]
_f2 = os.path.join(P, "fig_cmp2_numbers.json")
if os.path.exists(_src(_f2)):
    def _lv2(o, acc):
        if isinstance(o, dict): [_lv2(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv2(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv2(json.load(open(_src(_f2))), []):
        derived += [v, round(v, 1), round(v, 2), round(v, 3)]
_me = os.path.join(P, "method_evidence.json")            # 우리 방법 실측값
if os.path.exists(_src(_me)):
    def _lv(o, acc):
        if isinstance(o, dict): [_lv(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv(json.load(open(_src(_me))), []):
        derived += [v, round(v, 2), round(v, 3), v * 100, v / 100]
_dg = os.path.join(P, "diag_gap.json")
if os.path.exists(_src(_dg)):
    for v in json.load(open(_src(_dg))).values():
        derived += [float(v), round(float(v), 3), round(float(v), 4)]
_fc = os.path.join(P, "fig_cmp_numbers.json")
if os.path.exists(_src(_fc)):
    def _leaves(o, acc):
        if isinstance(o, dict): [ _leaves(v, acc) for v in o.values() ]
        elif isinstance(o, list): [ _leaves(v, acc) for v in o ]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _leaves(json.load(open(_src(_fc))), []):
        derived += [v, round(v, 1), round(v, 2)]

txt = subprocess.run(["pdftotext", os.path.join(P, "main.pdf"), "-"],
                     capture_output=True, text=True).stdout
body = txt.split("REFERENCES")[0]
# 부호까지 읽는다. 인쇄된 뺄셈 기호(U+2212)와 하이픈을 모두 음수로 본다.
_NUMPAT = re.compile("(?<![\\w.])([\u2212\\-])?(\\d{1,3}\\.\\d{1,4})(?![\\w])")
found = [(-1.0 if m.group(1) else 1.0) * float(m.group(2)) for m in _NUMPAT.finditer(body)]

def known(x):
    for v in derived:
        if abs(v - x) < max(5e-4, abs(x) * 1e-4): return True
    for v in DECLARED:
        if abs(v - x) < 5e-4: return True
    return False

def only_declared_tex(x):
    """측정 덤프로는 못 맞고 생성된 tex 사본으로만 맞는 값 = 독립 검증이 아니다."""
    if known(x): return False
    return any(abs(v - x) < max(5e-4, abs(x) * 1e-4) for v in declared_tex)

unmatched = sorted({x for x in found if not known(x) and not only_declared_tex(x)})
weak = sorted({x for x in found if only_declared_tex(x)})
_meas = {x for x in found if any(abs(v - x) < max(5e-4, abs(x) * 1e-4) for v in derived)}
_decl = {x for x in found if x not in _meas and any(abs(v - x) < 5e-4 for v in DECLARED)}
print(f"독립 검증: 측정 덤프로 대조된 값 {len(_meas)}개, 선언된 인용값 {len(_decl)}개, "
      f"생성된 tex 사본으로만 대조된 값 {len(weak)}개 (독립 아님)")
if weak:
    print("  tex 사본에만 의존하는 값:", weak[:24], "..." if len(weak) > 24 else "")
    _WEAK_FATAL = True
else:
    _WEAK_FATAL = False

# integers (2-4 digits) printed in the prose, outside citations, years and section numbers
INT_DECLARED = {95: "percentile label", 98: "percentile label", 99: "percentile label", 16: "block side (px)", 32: "ladder block side", 64: "ladder block side", 128: "ladder block side", 256: "ladder block side",
                12: "Zero-DCE++ size multiple", 11: "number of radial bands", 20: "20 dB level in the prose", 10: "10 pixels / percent scale",
                15: "LOL test frames", 50: "Sony scenes", 100: "percent scale", 255: "8-bit range", 512: "Sony frame height", 960: "Sony frame width"}
ints_found = [int(x) for x in re.findall(r"(?<![\w.\[\-\u2013/:])(\d{2,4})(?![\w\]%\u2013\-/:]|\.\d)", body)]   # a sentence-final dot is not a decimal point
ints_found = [v for v in ints_found if not (1900 <= v <= 2100)]
import math as _math
ints_known = {int(round(v)) for v in derived if abs(v - round(v)) < 1e-9}
# 본문이 "바깥쪽 반올림" 이라고 밝힌 경계만 유도한다: 제곱오차 몫과 저주파 몫 계열
_bounds = []
for _row in E.get("cmp_rows", []):
    for _k in ("glob", "chan", "resid", "lf"):
        if isinstance(_row, dict) and isinstance(_row.get(_k), (int, float)): _bounds.append(float(_row[_k]))
for _m, _v in (E.get("lol") or {}).items():
    if isinstance(_v, dict) and isinstance(_v.get("lf"), (int, float)): _bounds.append(float(_v["lf"]))
_printed_bounds = {41, 89, 14, 42, 67, 70, 3, 60, 13, 18}   # 본문이 바깥쪽 반올림이라 밝힌 경계만
ints_known |= {int(_math.floor(v)) for v in _bounds if int(_math.floor(v)) in _printed_bounds}
ints_known |= {int(_math.ceil(v)) for v in _bounds if int(_math.ceil(v)) in _printed_bounds}
ints_declared_tex = {int(x) for x in re.findall(r"\{(\d+)\}\s*$", open(os.path.join(P, "numbers.tex")).read(), re.M)}
for f in ("tab_rho_rows.tex", "tab_cmp_rows.tex"):
    p = os.path.join(P, f)
    if os.path.exists(p): ints_declared_tex |= {int(x) for x in re.findall(r"(?<![\d.])(\d+)(?![\d.])", open(p).read())}
ints_unmatched = sorted({v for v in ints_found if v not in ints_known and v not in INT_DECLARED})

# self-report of the matching width: how much of [0, 100) the tolerance windows cover, and how many prose values are ambiguous
_w = sorted({float(v) for v in derived if 0 <= v < 100})
_cov, _cur = 0.0, None
for v in _w:
    lo, hi = v - max(5e-4, v * 1e-4), v + max(5e-4, v * 1e-4)
    if _cur is not None and lo <= _cur[1]: _cur[1] = max(_cur[1], hi)
    else:
        if _cur is not None: _cov += _cur[1] - _cur[0]
        _cur = [lo, hi]
if _cur is not None: _cov += _cur[1] - _cur[0]
def _hits(x): return sorted({round(v, 6) for v in derived if abs(v - x) < max(5e-4, abs(x) * 1e-4)})
_amb = [x for x in set(found) if len(_hits(x)) > 1 and max(_hits(x)) - min(_hits(x)) > 5e-4]
assert len(found) > 100, "nothing scanned: PDF text not found or empty"
print(f"PDF 본문 숫자 {len(found)}개, 고유 {len(set(found))}개; 정수 토큰 {len(ints_found)}개, 고유 {len(set(ints_found))}개")
print(f"자기보고: 대조값 {len(_w)}개의 허용폭이 [0,100) 중 {_cov:.3f} 을 덮음 (무작위 값이 우연히 대조될 확률 {_cov / 100:.4f}); 서로 다른 대조값 둘 이상에 맞는 본문 값 {len(_amb)}개 {sorted(_amb)[:8]}")
_acc = {v for v in ints_known | set(INT_DECLARED) if 10 <= v <= 9999}
print(f"자기보고(정수): 수용 집합이 두 자리 {len([v for v in _acc if v < 100])}/90, 세 자리 {len([v for v in _acc if 100 <= v < 1000])}/900, [10,9999] 전체 {len(_acc)}/9990 ({100 * len(_acc) / 9990:.2f} %) 을 덮음")
if ints_unmatched:
    print(f"대조 안 되는 정수 {len(ints_unmatched)}개: {ints_unmatched}"); sys.exit(1)
if unmatched:
    print(f"측정 덤프와 대조 안 되는 값 {len(unmatched)}개:")
    for x in unmatched: print("   ", x)
    sys.exit(1)
# 1:1 배선: 생성 파일은 영수증의 함수여야 한다. 사본에서 생성기를 다시 돌려 바이트 비교한다.
# 허용폭이 없으므로 생성 파일 안의 인쇄값 하나를 고치면 반드시 잡힌다.
_GEN = (("make_evidence.py", ("numbers.tex", "tab_cmp_rows.tex",
                              "tab_rho_rows.tex")),
        ("make_final_evidence.py", ("numbers_final.tex", "tab_fin_rows.tex")),
        ("make_method_evidence.py", ("numbers_method.tex",)),
        ("make_cmp2_table.py", ("tab_cmp2_rows.tex",)),
        ("make_declared_evidence.py", ("numbers_declared.tex",)),
        ("make_diagnostic_evidence.py", ("evidence.json", "tab_rho_rows.tex")))
_USED_MACROS = set()
for _bt in ("main.tex", "sec_intro.tex", "sec_method.tex", "sec_results.tex", "sec_discussion.tex"):
    _bp = os.path.join(P, _bt)
    if os.path.exists(_bp):
        _USED_MACROS |= set(re.findall(r"\\(n[A-Za-z]+)", open(_bp).read()))
_regen_ran, _regen_bad, _regen_skip = [], [], []
for _g, _outs in _GEN:
    if not os.path.exists(os.path.join(P, _g)):
        # 생성기를 지우면 검사가 사라지는 구멍을 막는다: 산출물이 있는데 생성기가 없으면 실패다
        if any(os.path.exists(os.path.join(P, _o)) for _o in _outs):
            _regen_bad.append((_g, "생성기가 없는데 산출물은 있다")); continue
        _regen_skip.append((_g, "생성기와 산출물 모두 없음")); continue
    import shutil as _sh, subprocess as _sp, tempfile as _tf, filecmp as _fc
    _base = _tf.mkdtemp(prefix="regen_")
    _tmp = os.path.join(_base, "paper"); os.makedirs(_tmp, exist_ok=True)
    try:
        for _n in os.listdir(P):
            _s0 = os.path.join(P, _n)
            if os.path.isfile(_s0): _sh.copy2(_s0, os.path.join(_tmp, _n))
        # 생성기가 기대하는 이웃 관계를 사본 안에 그대로 만든다.
        # 이걸 안 하면 생성기들이 폴백 사슬 끝의 저자 트리로 떨어지고, 사본만으로 도는지 알 수 없게 된다.
        _numdir = os.path.join(_base, "numbers"); os.makedirs(_numdir, exist_ok=True)
        _mdst = os.path.join(_numdir, "method_260909"); os.makedirs(_mdst, exist_ok=True)
        # 번들 배치: 영수증이 원고 옆에 있다. 이 경우 형제 폴더를 들여다보지 않는다.
        for _n in os.listdir(_tmp):
            if _n.endswith((".json", ".npz", ".npy", ".md")):
                _sh.copy2(os.path.join(_tmp, _n), os.path.join(_mdst, _n))
                _sh.copy2(os.path.join(_tmp, _n), os.path.join(_numdir, _n))
        # 번들이면 영수증이 원고 옆에 다 있다. 그때만 형제 폴더를 안 본다.
        if not os.path.exists(os.path.join(_tmp, "final_rule.json")):   # 공개·정본 배치
            for _msrc in (os.path.join(P, "..", "method_260909"),
                          os.path.join(P, "..", "numbers", "method_260909")):
                if os.path.isdir(_msrc):
                    for _n in os.listdir(_msrc):
                        _f0 = os.path.join(_msrc, _n)
                        if os.path.isfile(_f0) and not os.path.exists(os.path.join(_mdst, _n)):
                            _sh.copy2(_f0, os.path.join(_mdst, _n))
            for _rsrc in (os.path.join(P, "..", "numbers"), os.path.join(P, "..", "repro")):
                if os.path.isdir(_rsrc):
                    for _n in os.listdir(_rsrc):
                        _f0 = os.path.join(_rsrc, _n)
                        if os.path.isfile(_f0) and not os.path.exists(os.path.join(_numdir, _n)):
                            _sh.copy2(_f0, os.path.join(_numdir, _n))
        for _ed in (os.path.join(P, "evidence"), os.path.join(P, "..", "evidence_diag"),
                    os.path.join(P, "..", "evidence")):
            if os.path.isdir(_ed):
                _sh.copytree(_ed, os.path.join(_base, "evidence")); break
        _env = dict(os.environ)
        _env["LLMETHOD"] = _mdst
        _env["LOWLIGHT_EVIDENCE_ROOT"] = _numdir
        _env["LOWLIGHT_REANALYSIS_ROOT"] = os.path.join(_base, "evidence")
        # 입력이면서 산출물인 파일(제자리 보강)은 지우면 생성기가 못 돈다
        _INPLACE = {"evidence.json"}
        for _o in _outs:
            _e0 = os.path.join(_tmp, _o)
            if _o not in _INPLACE and os.path.exists(_e0):
                os.remove(_e0)      # 생성기가 다시 쓰지 않으면 없는 채로 남아 실패한다
        _r = _sp.run([sys.executable, _g], cwd=_tmp, capture_output=True, text=True, env=_env)
        if _r.returncode != 0:
            # 영수증이 없어서 생성기가 죽는 것을 "건너뜀" 으로 넘기면, 영수증을 지워도 감사가 통과한다
            _regen_bad.append((_g, "재실행 실패: " + (_r.stderr.strip().splitlines() or ["실패"])[-1][:90])); continue
        _diff = []
        for _n in _outs:
            _a, _b = os.path.join(P, _n), os.path.join(_tmp, _n)
            if not os.path.exists(_a):
                _diff.append(_n + " (생성기가 만드는 파일이 원고 폴더에 없다)"); continue
            if not os.path.exists(_b):
                _diff.append(_n + " (재생성이 이 파일을 만들지 않았다)"); continue
            if _n.startswith("numbers"):
                # 미사용 매크로는 정리 단계가 지우므로, 원고가 실제로 쓰는 매크로만 값으로 비교한다
                _mv = lambda f: dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(f).read()))
                _ca, _cb = _mv(_a), _mv(_b)
                _bad = sorted(k for k in _ca if k in _USED_MACROS and k in _cb and _ca[k] != _cb[k])
                _lost = sorted(k for k in _ca if k in _USED_MACROS and k not in _cb)
                if _lost:  # 빈 재생성이 "불일치 없음" 으로 읽히면 검사가 헛돈다
                    _diff.append(_n + " (재생성이 원고가 쓰는 매크로 " + str(len(_lost)) + "개를 만들지 못했다: " + ", ".join(_lost[:5]) + ")")
                if _bad: _diff.append(_n + " (" + ", ".join(f"{k}: {_ca[k]} vs {_cb[k]}" for k in _bad[:6]) + ")")
            elif not _fc.cmp(_a, _b, shallow=False):
                _diff.append(_n)
        _regen_ran.append(_g)
        if _diff: _regen_bad += [(_g, _n) for _n in _diff]
    finally:
        _sh.rmtree(_base, ignore_errors=True)
assert len(_USED_MACROS) > 30, "원고에서 매크로를 못 읽었다 — 비교가 헛돈다"
# 생성된 매크로를 본문이 다시 정의하면 생성 파일은 멀쩡한 채 인쇄값만 바뀐다.
# 재생성 대조가 보증하는 범위 밖이라 여기서 따로 막는다.
_GENMAC = set()
for _gf in ("numbers.tex", "numbers_method.tex", "numbers_final.tex", "numbers_declared.tex"):
    _gp = os.path.join(P, _gf)
    if os.path.exists(_gp):
        _GENMAC |= set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", open(_gp).read()))
_redef = []
for _bt in ("main.tex", "sec_intro.tex", "sec_method.tex", "sec_results.tex", "sec_discussion.tex"):
    _bp = os.path.join(P, _bt)
    if not os.path.exists(_bp): continue
    for _m in re.finditer(r"\\(?:re)?newcommand\{\\(n[A-Za-z]+)\}", open(_bp).read()):
        if _m.group(1) in _GENMAC: _redef.append((_bt, _m.group(1)))
print(f"매크로 재정의: 생성 매크로 {len(_GENMAC)}개 중 본문이 다시 정의한 것 {len(_redef)}개")
if _redef:
    for _f0, _m0 in _redef: print(f"  {_f0} 가 {_m0} 를 다시 정의한다 — 생성값이 인쇄되지 않는다")
    sys.exit(1)

# 영수증이 기록해 둔 원본 덤프의 해시를 실제 파일과 대조한다.
# 영수증과 인쇄값을 같이 바꾸는 변조는 이 대조에서만 걸린다.
import hashlib as _hl
_hash_checked, _hash_bad = 0, []
for _rc in ("protocol_sweep.json", "protocol_pricing.json"):
    _rp = _src(_rc)
    if not os.path.exists(_rp): continue
    try: _rj = json.load(open(_rp))
    except Exception: continue
    _dp = (_rj.get("dump") or {})
    _want, _name = _dp.get("sha256"), os.path.basename(_dp.get("path", ""))
    if not (_want and _name): continue
    _target = _src(_name)
    if not os.path.exists(_target):
        _hash_bad.append((_name, "파일 없음")); continue
    _got = _hl.sha256(open(_target, "rb").read()).hexdigest()
    _hash_checked += 1
    if _got != _want: _hash_bad.append((_name, f"{_got[:12]} != {_want[:12]}"))
print(f"덤프 해시: {_hash_checked}건 대조, 불일치 {len(_hash_bad)}건")
for _n, _w in _hash_bad: print(f"  {_n}: {_w}")
if _hash_bad: sys.exit(1)

print(f"생성 재현: 원고가 쓰는 매크로 {len(_USED_MACROS)}개, 생성기 {len(_regen_ran)}개 재실행, 불일치 {len(_regen_bad)}건" +
      (f", 건너뜀 {len(_regen_skip)}개 {[g for g, _ in _regen_skip]}" if _regen_skip else ""))
for _g, _r in _regen_skip: print(f"  건너뜀 {_g}: {_r}")
if _regen_bad:
    for _g, _n in _regen_bad: print(f"  {_n} 이 {_g} 의 재생성 결과와 다르다 — 인쇄값이 영수증의 함수가 아니다")
    sys.exit(1)

if _WEAK_FATAL:
    print(f"생성된 tex 사본으로만 맞는 값 {len(weak)}개 — 측정 덤프로 독립 대조되지 않는다"); sys.exit(1)
if _MISSING_RECEIPTS:
    print(f"영수증 {len(set(_MISSING_RECEIPTS))}개를 찾지 못했다: {sorted(set(_MISSING_RECEIPTS))}"); sys.exit(1)
print("전부 대조됨 (측정 덤프 / 생성된 표 / 선언된 인용값)")
