"""원고가 쓰는 모든 생성 매크로가 실제로 정의돼 있는지 검사한다. 하나라도 없으면 실패."""
import re, sys, os, glob
P=os.path.dirname(os.path.abspath(__file__))
defined=set()
for f in ("numbers.tex","numbers_method.tex","numbers_final.tex"):
    p=os.path.join(P,f)
    if os.path.exists(p): defined |= set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", open(p).read()))
used=set()
for f in glob.glob(os.path.join(P,"sec_*.tex"))+[os.path.join(P,"main.tex")]+glob.glob(os.path.join(P,"tab_*_rows.tex")):
    used |= set(re.findall(r"\\(n[A-Za-z]+)\{\}", open(f).read()))
missing=sorted(used-defined); unused=sorted(defined-used)
print(f"사용 {len(used)}개 / 정의 {len(defined)}개 / 미정의 {len(missing)}개 / 미사용 {len(unused)}개")
if missing: print("미정의:", missing); sys.exit(1)
