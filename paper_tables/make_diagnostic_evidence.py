"""Add independently measured diagnostic controls and corrected spectral summaries."""
from pathlib import Path
import json
import math
import os
import re


def augment(paper):
    paper = Path(paper)
    root = Path(os.environ.get('LOWLIGHT_REANALYSIS_ROOT', paper.parent / 'evidence'))
    spectral = json.loads((root / 'h200/summary_validity.json').read_text())
    controls_path = root / 'h200/donor_summary.json'
    if not controls_path.exists():
        controls_path = root / 'h200/summary.json'
    controls = json.loads(controls_path.read_text())['controls']
    validation = json.loads((root / 'diagnostic_controls.json').read_text())
    evidence = json.loads((paper / 'evidence.json').read_text())
    evidence['spectral_reanalysis'] = spectral
    evidence['donor_controls'] = controls
    evidence['diagnostic_controls'] = {
        'protocol': validation['protocol'], 'checks': validation['checks'],
        'summaries': validation['summaries'], 'provenance': validation['provenance'],
    }
    values = {'nRTWOin': f"{evidence['r2']['cinout']:.3f}"}
    exposures = [('0.033s', 'short'), ('0.04s', 'mid'), ('0.1s', 'long')]
    for exp, tag in exposures:
        s = spectral['spectral'][exp]
        assert all(s['correction_valid']), (exp, 'nominal correction invalid')
        evidence['rho'][exp]['model'] = s['model_pooled']
        evidence['rho'][exp]['in_k1'] = s['input_pooled']
        share = s['residual_band_share']
        selected = [i for i, label in enumerate(s['labels']) if label == 'input_advantage']
        fraction = 100 * sum(share[i] for i in selected)
        evidence.setdefault('ceiling_rec_json', {})[exp] = fraction
        evidence.setdefault('raw_unident_json', {})[exp] = 100*sum(w for w,r in zip(share,s['input_pooled']) if r<.2)
        evidence.setdefault('ceiling_margin0', {})[exp] = 100*sum(w for w,r,g in zip(share,s['corrected_input_pooled'],s['corrected_gap']) if r>=.2 and g>0)
        evidence.setdefault('ceiling_margin0_unstable', {})[exp] = 100*sum(w for w,r,g,n in zip(share,s['corrected_input_pooled'],s['corrected_gap'],s['noise_over_power']) if r>=.2 and g>0 and n>.9)
        evidence.setdefault('gap_cross_json', {})[exp] = max((s['corrected_gap'][i] for i in selected), default=None)
        values['nSCALrec' + tag] = f'{fraction:.1f}'
        values['nRAWun' + tag] = f"{100*sum(w for w,r in zip(share,s['input_pooled']) if r<.2):.1f}"
        values['nMZrec' + tag] = f"{100*sum(w for w,r,g in zip(share,s['corrected_input_pooled'],s['corrected_gap']) if r>=.2 and g>0):.1f}"
        values['nMZun' + tag] = f"{100*sum(w for w,r,g,n in zip(share,s['corrected_input_pooled'],s['corrected_gap'],s['noise_over_power']) if r>=.2 and g>0 and n>.9):.1f}"
        values['nGap' + tag] = f"{s['corrected_gap'][-1]:.3f}"
        ci = s['corrected_gap_ci95_conditional_valid'][-1]
        values['nGap' + tag + 'Lo'] = f'{ci[0]:.3f}'
        values['nGap' + tag + 'Hi'] = f'{ci[1]:.3f}'
        values['nLabel' + tag + 'Pct'] = f"{100*s['nominal_label_bootstrap_stability_unconditional'][-1]:.1f}"
        if selected:
            values['nEX' + tag] = f"{max(s['corrected_gap'][i] for i in selected)-.05:.3f}"
        if tag in ('short', 'mid'):
            flagged = [i for i, v in enumerate(s['noise_over_power']) if v > .9]
            assert flagged
            values['nNP' + tag + 'From'] = f'{flagged[0]*.05:.2f}'
        flagged = [i for i, v in enumerate(s['noise_over_power']) if v > .9]
        evidence.setdefault('np_unstable', {})[exp] = dict(first_band_start=flagged[0]*.05 if flagged else None,
                                                         n_unstable=len(flagged),corner_np=s['noise_over_power'][-1])
    values['nNPmidCorner'] = f"{spectral['spectral']['0.04s']['noise_over_power'][-1]:.3f}"
    values['nRHOshortcorner'] = f"{spectral['spectral']['0.033s']['model_pooled'][-1]:.3f}"
    values['nBootN'] = str(spectral['bootstrap']['replicates'])
    values['nBootLevel'] = '95'
    values['nBootValidPct'] = f"{100*min(min(s['bootstrap_valid_fraction']) for s in spectral['spectral'].values()):.0f}"
    donor = controls['ALL']
    for name, key in [('nDonorSame','same_d16_mean'),('nDonorCross','cross_d16_mean'),
                      ('nDonorDiff','same_minus_cross_d16_mean')]:
        values[name] = f'{donor[key]:.3f}'
    for name, key in [('nDonorCross','cross_d16_ci95'),('nDonorDiff','same_minus_cross_d16_ci95')]:
        values[name+'Lo'], values[name+'Hi'] = [f'{v:.3f}' for v in donor[key]]
    values['nDonorCrossPct'] = f"{100*donor['cross_retention_white_corrected']:.1f}"
    values['nDonorCrossPctLo'], values['nDonorCrossPctHi'] = [f'{100*v:.1f}' for v in donor['cross_retention_ci95']]
    values['nDonorDraws'] = str(len(donor['draws']))
    values['nDonorOldSame'] = str(donor['legacy_same_scene_pairs'])
    ctrl = validation['summaries']['float_exact_control']
    values['nCTRLn'] = str(validation['checks']['distinct_test_scenes'])
    values['nCTRLpsnr'] = f"{ctrl['global_gain']['mean_frame_psnr']:.2f}"
    values['nCTRLglob'] = f"{ctrl['global_gain']['global_removed_pct']:.0f}"
    values['nCTRLspglob'] = f"{ctrl['smooth_spatial_gain']['global_removed_pct']:.1f}"
    values['nCTRLspblk'] = f"{ctrl['smooth_spatial_gain']['block64_additional_vs_global_removed_pct']:.1f}"
    values['nCTRLblock'] = str(validation['protocol']['block_side'])
    values['nCTRLgain'] = f"{validation['protocol']['global_gain']:.2f}"
    values['nCTRLwave'] = '0.8'
    values['nCTRLfreq'] = '0.30'
    values['nCTRLshiftY'], values['nCTRLshiftX'] = map(str, validation['protocol']['circular_shift_pixels'])
    values['nCTRLshift'] = f"{ctrl['invertible_circular_shift']['mean_high_rho']:.4f}"
    evidence['diagnostic_macro_values'] = values
    text = (paper / 'numbers.tex').read_text()
    for name, value in values.items():
        pattern = r'\\newcommand\{\\' + re.escape(name) + r'\}\{[^}]*\}'
        _inmath = name in ('nRTWOin', 'nGaplongLo', 'nGapmidLo')   # 수식 안에서 쓰이면 이미 수식 마이너스다
        value = ('$-$' + value[1:]) if isinstance(value, str) and value.startswith('-') and not _inmath else value   # 줄바꿈 분리 방지
        replacement = '\\newcommand{\\' + name + '}{' + value + '}'
        if re.search(pattern, text):
            text = re.sub(pattern, lambda _: replacement, text)
        else:
            text += '\n' + replacement
    (paper / 'numbers.tex').write_text(text + '\n')
    rows = []
    for band in range(11):
        end = .05*(band+1) if band<10 else math.sqrt(.5)
        cells = [f'{.05*band:.2f}--{end:.2f}']
        for exp, _ in exposures:
            s = spectral['spectral'][exp]
            mark = '\\dag' if s['noise_over_power'][band]>.9 else ''
            cells += [f"{s['model_pooled'][band]:.3f}", f"{s['input_pooled'][band]:.3f}"+mark]
        rows.append(' & '.join(cells) + ' \\\\')
    (paper / 'tab_rho_rows.tex').write_text('\n'.join(rows)+'\n')
    (paper / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    print('Diagnostic evidence integrated:', len(values), 'macros; scene-excluded donor and valid spectral summaries')


if __name__ == '__main__':
    augment(Path(__file__).resolve().parent)
