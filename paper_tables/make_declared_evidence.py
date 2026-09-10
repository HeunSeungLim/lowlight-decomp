"""본문이 인쇄하는 선언 상수와 인용값을 영수증에서 굽는다. 손으로 친 숫자를 남기지 않기 위한 것이다."""
import json, os

P = os.path.dirname(os.path.abspath(__file__))
def _method_dir():
    for c in (os.environ.get("LLMETHOD"), os.path.join(P, "..", "method_260909"),
              ):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
def _method_dir_common(P):
    for c in (os.environ.get("LLMETHOD"),
              os.path.join(P, "..", "method_260909"),
              os.path.join(P, "..", "numbers", "method_260909"),
              ):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
M = _method_dir_common(P)

def _load(name):
    _roots = [P, M, os.environ.get("LOWLIGHT_EVIDENCE_ROOT", os.path.join(P, "..", "numbers")),
              os.path.join(P, "..", "repro")]
    for c in [os.path.join(r, name) for r in _roots if r]:
        if os.path.exists(c): return json.load(open(c))
    return None

AC = _load("analysis_constants.json")
MG = _load("multiframe_gain.json")
FC = _load("fig_cmp2_numbers.json")
CR = _load("cross_rendition_source.json")
UC = _load("unresolved_cells.json")

mac = {}
if AC:
    mac["nNPthresh"] = f"{AC['np_instability_threshold']['value']}"
    mac["nCIlevel"] = f"{AC['confidence_level_pct']['value']}"
if CR and "value" in CR:
    mac["nCIDreported"] = f"{CR['value']:.3f}"
if MG:
    from collections import Counter as _C
    _c = _C()
    for _k, _v in MG["group_sizes"].items():
        _n = _v if isinstance(_v, int) else len(_v)
        if _n >= 8: _c[_k.split("|")[1]] += 1
    mac["nMFlongGroups"] = f"{_c['0.1s']}"
    mac["nMFmidGroups"] = f"{_c['0.04s']}"
if FC:
    mac["nCALIBgain"] = f"{FC['calibration_gain']:.3f}"
if UC:
    cells = sorted(c["delta"] for c in UC["cells"])
    mac["nUnresLo"] = f"{cells[0]:.3f}"
    mac["nUnresHi"] = f"{cells[-1]:.3f}"

assert len(mac) >= 8, "선언 상수를 못 읽었다 — 빈 파일을 쓰지 않는다"
out = "".join("\\newcommand{\\%s}{%s}\n" % (k, v) for k, v in sorted(mac.items()))
open(os.path.join(P, "numbers_declared.tex"), "w").write(out)
print("numbers_declared.tex: %d개 (%s)" % (len(mac), ", ".join(sorted(mac))))
