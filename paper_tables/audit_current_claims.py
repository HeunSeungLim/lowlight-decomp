"""Independent semantic checks for corrected manuscript quantities."""
from pathlib import Path
import json
import math
import re
import statistics

P = Path(__file__).resolve().parent
R = P.parent / 'evidence'
E = json.loads((P/'evidence.json').read_text())
N = dict(re.findall(r'\\newcommand\{\\(n[A-Za-z]+)\}\{([^}]*)\}', (P/'numbers.tex').read_text()))
F = json.loads((R/'h200/frame_stats.json').read_text())['rows']
C = json.loads((R/'h200/control_rows.json').read_text())
rows = C['rows']
checks = {}

def near(name, observed, expected, tolerance):
    assert abs(observed-expected) <= tolerance, (name, observed, expected)
    checks[name] = dict(observed=observed, expected=expected, tolerance=tolerance)

near('three_variable_r2', float(N['nRTWOin']), E['r2']['cinout'], .0005)
for exp, tag in [('0.033s','short'),('0.04s','mid'),('0.1s','long')]:
    rr = [r for r in F if r['exp']==exp]
    sr = E['spectral_reanalysis']['spectral'][exp]
    for b in range(11):
        cross = math.fsum(r['model']['cx'][b] for r in rr)
        power = math.fsum(r['model']['pp'][b] for r in rr)
        gt = math.fsum(r['model']['pg'][b] for r in rr)
        independent = cross/math.sqrt(power*gt)
        near(f'pooled_{exp}_{b}', E['rho'][exp]['model'][b], independent, 1e-12)
    b = 10
    cross = math.fsum(r['input']['cx'][b] for r in rr)
    power = math.fsum(r['input']['pp'][b] for r in rr)
    gt = math.fsum(r['input']['pg'][b] for r in rr)
    noise = math.fsum(r['input_gain']**2*r['noise_raw'][b] for r in rr)
    assert 0 <= noise < power
    corrected = cross/math.sqrt((power-noise)*gt)
    near('gain_weighted_gap_'+exp, float(N['nGap'+tag]), corrected-E['rho'][exp]['model'][-1], .0005)
    labelled = 100*math.fsum(w for w,l in zip(sr['residual_band_share'],sr['labels']) if l=='input_advantage')
    near('labelled_share_'+exp, float(N['nSCALrec'+tag]), labelled, .05)

def delta(mse):
    return 10*math.log10(mse[0]/mse[1])

for family, macro in [('same','nDonorSame'),('cross','nDonorCross')]:
    vals = [statistics.fmean(delta(d[family+'_mse']) for d in r['draws']) for r in rows]
    near(family+'_donor_mean', float(N[macro]), statistics.fmean(vals), .0005)
violations = []
for r in rows:
    i = r['index']
    for d in r['draws']:
        same, other = rows[d['same_donor']], rows[d['cross_donor']]
        if same['scene']!=r['scene'] or other['scene']==r['scene'] or same['index']==i:
            violations.append(i)
        assert same['exp']==other['exp']==r['exp']
assert not violations
checks['donor_selection_violations'] = 0
assert E['spectral_reanalysis']['correction_policy'].startswith('N>=P')
assert min(min(s['bootstrap_valid_fraction']) for s in E['spectral_reanalysis']['spectral'].values())==1
assert all(Path(p).read_bytes() for p in [P/'main.pdf',P/'fig_workflow.pdf'])
body = '\n'.join((P/f).read_text() for f in ['main.tex','sec_intro.tex','sec_method.tex','sec_results.tex','sec_discussion.tex'])
for prohibited in ['information the rendition never carried','numbers are released','would require training on burst statistics','on Sony almost none']:
    assert prohibited not in body, prohibited
abstract = (P/'main.tex').read_text().split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
assert 100 <= len(abstract.split()) <= 150
checks['abstract_words'] = len(abstract.split())
checks['status'] = 'PASS'
(R/'audit_current_claims.json').write_text(json.dumps(checks,indent=2))
print('PASS:',len(checks),'claim-linked checks')
