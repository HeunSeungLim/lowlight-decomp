"""audit_*: check that every number the compiled PDF prints exists in the measurement dumps.

The macros in numbers.tex are generated, so the risk is not a typo in a macro but a number
typed straight into the prose or a table during editing. This walks the PDF text, pulls
every decimal figure, and flags any that cannot be matched to a value in evidence.json,
the generated row files, or a short list of declared literals (thresholds, published
values we quote on purpose).
"""
import json, re, subprocess, sys, os

P = os.path.dirname(os.path.abspath(__file__))
def _src(path):
    """번들 안에 같은 이름이 있으면 그것을 쓴다. 고정본이 자기 안에서 돌게 하는 배선."""
    local = os.path.join(P, os.path.basename(path))
    return local if os.path.exists(local) else path

E = json.load(open(os.path.join(P, "evidence.json")))

def leaves(o, acc):
    if isinstance(o, dict):
        for v in o.values(): leaves(v, acc)
    elif isinstance(o, list):
        for v in o: leaves(v, acc)
    elif isinstance(o, (int, float)):
        acc.append(float(o))
    return acc

vals = leaves(E, [])
# values a table can legitimately show that are derived from the dumps
derived = []
for v in list(vals):
    derived += [v, v * 100, v / 100]
for r in E["ladder"]:
    for k in ("gain", "gain_dof", "affine", "affine_dof"):
        if r.get(k) is not None: derived.append(r[k])
for _f in (os.environ.get("LLROOT", ".") + "/numbers/diag_sid_identifiability.json",
           os.environ.get("LLROOT", ".") + "/numbers/diag_sid_lowfreq.json"):
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
for _v in _leaves_all(E, []):          # 진단 측정 덤프 전량 (인쇄값 사본이 아니라 측정 결과)
    derived += [_v, abs(_v), round(_v, 1), round(_v, 2), round(_v, 3)]

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

# numbers printed on the qualitative figure are real per-frame scores; take them from the
# figure script's own record rather than from the prose
_fq = os.path.join(P, "fig_qual_numbers.json")
if os.path.exists(_fq):
    for v in json.load(open(_src(_fq))).values():
        derived += [v["psnr"], v["psnr_shown"], v["gain"]]

for _extra in (os.environ.get("LLROOT", ".") + "/method_260909/fix_v52.json",
               os.environ.get("LLROOT", ".") + "/method_260909/robust_acting.json",
               os.environ.get("LLROOT", ".") + "/method_260909/qsweep_rule.json",
               os.environ.get("LLROOT", ".") + "/method_260909/final_rule.json",
               os.environ.get("LLROOT", ".") + "/method_260909/shares.json",
               os.environ.get("LLROOT", ".") + "/method_260909/controls_v58.json",
               os.environ.get("LLROOT", ".") + "/method_260909/precond_transfer.json",
               os.environ.get("LLROOT", ".") + "/method_260909/affine_oracle.json",
               os.environ.get("LLROOT", ".") + "/method_260909/concentration.json",
               os.environ.get("LLROOT", ".") + "/method_260909/tost_equivalence.json",
               os.environ.get("LLROOT", ".") + "/method_260909/decomp_anchored.json",
               os.environ.get("LLROOT", ".") + "/method_260909/anchor_lol_retinexformer.json",
               os.environ.get("LLROOT", ".") + "/method_260909/protocol_pricing.json",
               os.environ.get("LLROOT", ".") + "/method_260909/unresolved_cells.json",
               os.environ.get("LLROOT", ".") + "/method_260909/anchored_sony_sd.json",
               os.environ.get("LLROOT", ".") + "/method_260909/tab3_anchored_column.json",
               os.environ.get("LLROOT", ".") + "/method_260909/qsat_train181.json"):
    if os.path.exists(_extra):
        def _strip_printed(o):
            """영수증이 담고 있는 인쇄값 사본은 대조 출처가 될 수 없다."""
            if isinstance(o, dict):
                return {k: _strip_printed(v) for k, v in o.items()
                        if k not in ("printed", "table_reference", "measured_printed", "reported_source",
                                     "reported_on_its_own", "value", "previous_text", "correction",
                                     "psnr_sd_ddof0", "sat_train_lol_truncated120",
                                     "previous_scope_prefix0_only")}
            if isinstance(o, list): return [_strip_printed(v) for v in o]
            return o
        def _lvx(o, acc):
            if isinstance(o, dict): [_lvx(v, acc) for v in o.values()]
            elif isinstance(o, list): [_lvx(v, acc) for v in o]
            elif isinstance(o, (int, float)): acc.append(float(o))
            return acc
        for v in _lvx(_strip_printed(json.load(open(_src(_extra)))), []):
            derived += [v, abs(v), round(v, 1), round(v, 2), round(abs(v), 2), round(v, 3), float(round(v)), float(round(abs(v)))]
_fe = os.path.join(P, "final_evidence.json")                  # 최종 규칙 실측값·백분위 라벨
if os.path.exists(_fe):
    def _lv4(o, acc):
        if isinstance(o, dict): [_lv4(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv4(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        elif isinstance(o, str):
            try: acc.append(float(o))
            except ValueError: pass
        return acc
    _fej = _strip_printed(json.load(open(_src(_fe)))); _fej.pop("printed", None)      # 인쇄값이 스스로를 검증하지 않도록 제외
    for v in _lv4(_fej, []):
        derived += [v, abs(v), round(v, 2), round(abs(v), 2), round(v, 3), round(v, 4)]
        if abs(v - round(v)) < 1e-9: ints_extra = int(round(v))
_an = os.path.join(P, "anchor_evidence.json")                 # 앵커판 실측값
if os.path.exists(_an):
    def _lv3(o, acc):
        if isinstance(o, dict): [_lv3(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv3(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv3(_strip_printed(json.load(open(_src(_an)))), []):
        derived += [v, abs(v), round(v, 2), round(abs(v), 2), round(v, 3), v * 100]
_f2 = os.path.join(P, "fig_cmp2_numbers.json")
if os.path.exists(_f2):
    def _lv2(o, acc):
        if isinstance(o, dict): [_lv2(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv2(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv2(json.load(open(_src(_f2))), []):
        derived += [v, round(v, 1), round(v, 2), round(v, 3)]
_me = os.path.join(P, "method_evidence.json")            # 우리 방법 실측값
if os.path.exists(_me):
    def _lv(o, acc):
        if isinstance(o, dict): [_lv(v, acc) for v in o.values()]
        elif isinstance(o, list): [_lv(v, acc) for v in o]
        elif isinstance(o, (int, float)): acc.append(float(o))
        return acc
    for v in _lv(json.load(open(_src(_me))), []):
        derived += [v, abs(v), round(v, 2), round(abs(v), 2), round(v, 3), v * 100, v / 100]
_dg = os.path.join(P, "diag_gap.json")
if os.path.exists(_dg):
    for v in json.load(open(_src(_dg))).values():
        derived += [float(v), round(float(v), 3), round(float(v), 4)]
_fc = os.path.join(P, "fig_cmp_numbers.json")
if os.path.exists(_fc):
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
found = [float(x) for x in re.findall(r"(?<![\w.])\d{1,3}\.\d{1,4}(?![\w])", body)]

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
ints_known |= {int(_math.floor(v)) for v in _bounds}
ints_known |= {int(_math.ceil(v)) for v in _bounds}
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
print("전부 대조됨 (측정 덤프 / 생성된 표 / 선언된 인용값)")
