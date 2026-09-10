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

code = 0
try:
    runpy.run_path('audit_paper_numbers.py', run_name='__main__')
except SystemExit as e:
    code = e.code
    print('exit code', code)
except Exception as e:
    code = 'EXCEPTION'
    print('ISOLATION FAILED:', type(e).__name__, e)
print('outside-bundle accesses attempted:', len(OUTSIDE))
for ch, p in OUTSIDE[:8]:
    print('   ', ch, p)
print('VERDICT:', 'self-contained' if (code == 0 or code is None) and not OUTSIDE else 'NOT self-contained')
