"""본문이 인쇄하는 선언 상수와 인용값을 영수증에서 굽는다. 손으로 친 숫자를 남기지 않기 위한 것이다."""
import json, os

P = os.path.dirname(os.path.abspath(__file__))
def _method_dir():
    for c in (os.environ.get("LLMETHOD"), os.path.join(P, "..", "method_260909"),
              "/home/user/lowlight_paper/method_260909"):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
def _method_dir_common(P):
    for c in (os.environ.get("LLMETHOD"),
              os.path.join(P, "..", "method_260909"),
              os.path.join(P, "..", "numbers", "method_260909"),
              "/home/user/lowlight_paper/method_260909"):
        if c and os.path.isdir(c): return c
    return os.path.join(P, "..", "method_260909")
M = _method_dir_common(P)

def _load(name):
    for c in (os.path.join(P, name), os.path.join(M, name)):
        if os.path.exists(c): return json.load(open(c))
    return None

AC = _load("analysis_constants.json")
CR = _load("cross_rendition_source.json")
UC = _load("unresolved_cells.json")

mac = {}
if AC:
    mac["nNPthresh"] = f"{AC['np_instability_threshold']['value']}"
    mac["nCIlevel"] = f"{AC['confidence_level_pct']['value']}"
if CR and "value" in CR:
    mac["nCIDreported"] = f"{CR['value']:.3f}"
if UC:
    cells = sorted(c["delta"] for c in UC["cells"])
    mac["nUnresLo"] = f"{cells[0]:.3f}"
    mac["nUnresHi"] = f"{cells[-1]:.3f}"

assert len(mac) >= 5, "선언 상수를 못 읽었다 — 빈 파일을 쓰지 않는다"
out = "".join("\\newcommand{\\%s}{%s}\n" % (k, v) for k, v in sorted(mac.items()))
open(os.path.join(P, "numbers_declared.tex"), "w").write(out)
print("numbers_declared.tex: %d개 (%s)" % (len(mac), ", ".join(sorted(mac))))
