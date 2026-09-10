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

def guard_raise(fn, name):
    def g(p, *a, **k):
        if _outside(p):
            OUTSIDE.append((name, str(p))); raise FileNotFoundError('isolation: outside bundle ' + str(p))
        return fn(p, *a, **k)
    return g

builtins.open = guard_open
os.path.exists = guard_pred(_exists, 'exists')
os.path.isdir = guard_pred(_isdir, 'isdir')
os.path.isfile = guard_pred(_isfile, 'isfile')
# open 만 막으면 os.open / pathlib / glob 로 새어 나간다.
# os.stat/listdir/scandir 과 os.open 은 인터프리터·subprocess 가 내부적으로 써서 후킹하면 멈춘다.
# 미차단 채널로 남기고 여기 적어 둔다: os.open, os.stat, os.listdir, os.scandir, 비파이썬 자식 프로세스.
import io as _io
_io.open = guard_open
import glob as _glob
_glob_orig = _glob.glob
def _glob_guard(pat, *a, **k):
    if _outside(os.path.dirname(str(pat)) or '.'):
        OUTSIDE.append(('glob', str(pat))); return []
    return _glob_orig(pat, *a, **k)
_glob.glob = _glob_guard
import pathlib as _pl
_path_open_orig = _pl.Path.open
def _path_open(self, *a, **k):
    if _outside(self):
        OUTSIDE.append(('pathlib.open', str(self))); raise FileNotFoundError('isolation: outside bundle ' + str(self))
    return _path_open_orig(self, *a, **k)
_pl.Path.open = _path_open

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
def _mr(fn, ch):
    def g(p, *a, **k):
        if _out(p): _note(ch, p); raise FileNotFoundError("isolation: outside " + str(p))
        return fn(p, *a, **k)
    return g
os.open = _mr(os.open, "os.open")
import io as _io2
_io2.open = _go
import glob as _g2
_gg2 = _g2.glob
def _glob_g(pat, *a, **k):
    if _out(os.path.dirname(str(pat)) or "."): _note("glob", pat); return []
    return _gg2(pat, *a, **k)
_g2.glob = _glob_g
import pathlib as _pl2
_po2 = _pl2.Path.open
def _path_open2(self, *a, **k):
    if _out(self): _note("pathlib.open", self); raise FileNotFoundError("isolation: outside " + str(self))
    return _po2(self, *a, **k)
_pl2.Path.open = _path_open2
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
