"""번들 자립 시험. 번들 디렉터리 밖의 어떤 경로도 열거나 존재 확인조차 못 하게 막고 감사기를 돌린다.

이전 판본은 builtins.open 만 후킹했는데, 실제 버그는 os.path.exists 채널에 있었다.
그래서 버그가 있어도 "차단 0건" 이 찍혀 시험이 통과처럼 보였다. 이제 세 채널을 다 막고
비교는 realpath 로 한다.
"""
import builtins, os, os.path, runpy, sys, tempfile

HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
OUTSIDE = []
OWN_TMP = []          # 감사기가 스스로 만든 작업 폴더. 자기가 만든 것만 허용한다.
_mkdtemp = tempfile.mkdtemp
def _mkdtemp_tracked(*a, **k):
    d = _mkdtemp(*a, **k)
    OWN_TMP.append(os.path.realpath(d))
    return d
tempfile.mkdtemp = _mkdtemp_tracked

def _outside(p):
    try:
        rp = os.path.realpath(os.path.abspath(str(p)))
    except Exception:
        return False
    if rp == HERE or rp.startswith(HERE + os.sep):
        return False
    if any(rp == o or rp.startswith(o + os.sep) for o in OWN_TMP):
        return False
    if rp == os.path.realpath(tempfile.gettempdir()):
        return False   # 임시 폴더를 만들려면 그 부모를 볼 수 있어야 한다
    # 표준 라이브러리와 인터프리터는 허용한다. 저자 트리만 막는다.
    for allow in (sys.prefix, sys.base_prefix, '/usr', '/lib', '/proc', '/dev', '/etc/localtime'):
        if rp.startswith(os.path.realpath(allow)):
            return False
    return True

_open, _exists, _isdir, _isfile, _stat = (
    builtins.open, os.path.exists, os.path.isdir, os.path.isfile, os.stat)

def guard_open(p, *a, **k):
    if _outside(p):
        OUTSIDE.append(('open', str(p))); raise FileNotFoundError('isolation: outside bundle ' + str(p))
    return _open(p, *a, **k)

def guard_pred(fn, name):
    def g(p, *a, **k):
        if _outside(p):
            OUTSIDE.append((name, str(p))); return False
        return fn(p, *a, **k)
    return g

builtins.open = guard_open
os.path.exists = guard_pred(_exists, 'exists')
os.path.isdir = guard_pred(_isdir, 'isdir')
os.path.isfile = guard_pred(_isfile, 'isfile')

# 감사기는 생성기를 자식 프로세스로 돌린다. 부모만 후킹하면 그 접근이 통째로 안 보인다.
# 자식에도 같은 차단을 설치하고, 위반을 파일로 받아 합친다.
GUARD = tempfile.mkdtemp(prefix="isoguard_"); OWN_TMP.append(os.path.realpath(GUARD))
VIOL = os.path.join(GUARD, "violations.txt")
open(os.path.join(GUARD, "sitecustomize.py"), "w").write("""
import builtins, os, os.path
ROOTS = [r for r in os.environ.get("ISOLATION_ROOTS", "").split(os.pathsep) if r]
LOG = os.environ.get("ISOLATION_LOG", "")
ALLOW = ("/usr", "/lib", "/proc", "/dev", "/etc/localtime")
import sys as _s
ALLOW = ALLOW + (_s.prefix, _s.base_prefix)
def _out(p):
    try: rp = os.path.realpath(os.path.abspath(str(p)))
    except Exception: return False
    for r in ROOTS:
        if rp == r or rp.startswith(r + os.sep): return False
    for a in ALLOW:
        if rp.startswith(os.path.realpath(a)): return False
    return True
def _note(ch, p):
    try:
        with open(LOG, "a") as f: f.write(ch + " " + str(p) + chr(10))
    except Exception: pass
_o, _e, _d, _f = builtins.open, os.path.exists, os.path.isdir, os.path.isfile
def _go(p, *a, **k):
    if _out(p): _note("open", p); raise FileNotFoundError("isolation: outside " + str(p))
    return _o(p, *a, **k)
def _mk(fn, ch):
    def g(p, *a, **k):
        if _out(p): _note(ch, p); return False
        return fn(p, *a, **k)
    return g
builtins.open = _go
os.path.exists = _mk(_e, "exists"); os.path.isdir = _mk(_d, "isdir"); os.path.isfile = _mk(_f, "isfile")
""")

import subprocess as _subprocess
_run = _subprocess.run
def _run_guarded(*a, **k):
    env = dict(k.get("env") or os.environ)
    env["PYTHONPATH"] = GUARD + os.pathsep + env.get("PYTHONPATH", "")
    env["ISOLATION_ROOTS"] = os.pathsep.join([HERE] + OWN_TMP)
    env["ISOLATION_LOG"] = VIOL
    k["env"] = env
    return _run(*a, **k)
_subprocess.run = _run_guarded

code = 0
try:
    runpy.run_path('audit_paper_numbers.py', run_name='__main__')
except SystemExit as e:
    code = e.code
    print('exit code', code)
except Exception as e:
    code = 'EXCEPTION'
    print('ISOLATION FAILED:', type(e).__name__, e)
if os.path.exists(VIOL):
    for _line in open(VIOL).read().splitlines():
        _ch, _, _pp = _line.partition(' ')
        OUTSIDE.append((_ch + '(child)', _pp))
print('outside-bundle accesses attempted:', len(OUTSIDE))
for ch, p in OUTSIDE[:8]:
    print('   ', ch, p)
print('VERDICT:', 'self-contained' if (code == 0 or code is None) and not OUTSIDE else 'NOT self-contained')
