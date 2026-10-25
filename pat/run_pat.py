#!/usr/bin/env python3
"""Command-line front end for the Proliferation-Adjusted Test (PAT).

Usage
-----
python run_pat.py --expr expr.tsv --clin clin.csv --gene TRIP13 \
                  --endpoint efs --covariates age18 stage4 mycn --out results/

Inputs
------
--expr   genes (rows) x samples (columns), log2 scale, tab- or comma-separated.
         Row names must be HGNC symbols; column names must match the clinical IDs.
--clin   one row per sample, first column the sample ID, and columns
         efs_t / efs_e (days, 0-1) and/or os_t / os_e, plus the clinical
         covariates named in --covariates.
--gene   candidate gene symbol.
Outputs
-------
<out>/PAT_<gene>_<endpoint>.csv   statistics for each proliferation covariate
<out>/PAT_<gene>_<endpoint>.png   four-panel standard figure
and the verdict printed to stdout.
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pat_core import pat, verdict, build_proliferation, z
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

def read_matrix(path):
    sep = '\t' if path.endswith(('.tsv', '.txt', '.gz')) else ','
    return pd.read_csv(path, sep=sep, index_col=0)

def figure(expr, clin, gene, endpoint, covariates, rows, path):
    t, e = f'{endpoint}_t', f'{endpoint}_e'
    d = clin.copy()
    d['x'] = z(expr.loc[gene, d.index]).values
    d['prolif'] = build_proliferation(expr, d.index, 'score', exclude=gene).values
    cov = [c for c in covariates if c in d and d[c].nunique() > 1]
    d = d.dropna(subset=['x', 'prolif', t, e] + cov)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.1))
    # A: nested hazard ratios
    steps, hrs, los, his = [], [], [], []
    for i in range(len(cov) + 2):
        c = ['x'] + (cov + ['prolif'])[:i]
        m = CoxPHFitter(penalizer=1e-4).fit(d[c + [t, e]], t, e); s = m.summary.loc['x']
        steps.append('gene alone' if i == 0 else '+ ' + (cov + ['prolif'])[i-1])
        hrs.append(np.exp(s['coef'])); los.append(np.exp(s['coef lower 95%'])); his.append(np.exp(s['coef upper 95%']))
    y = np.arange(len(steps))[::-1]
    ax[0].errorbar(hrs, y, xerr=[np.array(hrs)-los, np.array(his)-hrs], fmt='o', color='#2c5f8a', capsize=3)
    ax[0].axvline(1, ls='--', color='k', lw=.8); ax[0].set_yticks(y); ax[0].set_yticklabels(steps, fontsize=7)
    ax[0].set_xlabel('HR per SD'); ax[0].set_title('A  Nested models', loc='left', fontweight='bold', fontsize=9)
    # B: gene vs proliferation
    ax[1].scatter(d.prolif, d.x, s=5, c='#888888', lw=0)
    r = rows[0].get('rho_prolif', np.nan)
    ax[1].set_xlabel('proliferation covariate (z)'); ax[1].set_ylabel(f'{gene} (z)')
    ax[1].set_title(f'B  Collinearity (rho = {r:.2f})', loc='left', fontweight='bold', fontsize=9)
    # C: Kaplan-Meier, median split
    d['grp'] = np.where(d.x > d.x.median(), 'high', 'low'); k = KaplanMeierFitter()
    for g, c in [('high', '#c0504d'), ('low', '#2c5f8a')]:
        s = d[d.grp == g]; k.fit(s[t]/365.25, s[e], label=f'{g} (n={len(s)})'); k.plot_survival_function(ax=ax[2], color=c, lw=1.4)
    lr = logrank_test(d[d.grp=='high'][t], d[d.grp=='low'][t], d[d.grp=='high'][e], d[d.grp=='low'][e])
    ax[2].text(.04, .08, f'log-rank p = {lr.p_value:.2g}', transform=ax[2].transAxes, fontsize=7)
    ax[2].set_xlabel('years'); ax[2].set_ylabel(endpoint.upper()); ax[2].set_ylim(0, 1)
    ax[2].legend(frameon=False, fontsize=7); ax[2].set_title('C  Unadjusted split', loc='left', fontweight='bold', fontsize=9)
    # D: added value
    labs, vals = [], []
    for r_ in rows:
        labs.append(r_['prolif_covariate']); vals.append(r_['delta_c'])
    ax[3].bar(range(len(vals)), vals, color=['#4f8a3d' if v >= .005 else '#c0504d' for v in vals])
    ax[3].axhline(0.005, ls=':', color='k', lw=.8); ax[3].axhline(0, color='k', lw=.8)
    ax[3].set_xticks(range(len(labs))); ax[3].set_xticklabels(labs, fontsize=7)
    ax[3].set_ylabel('change in Harrell C'); ax[3].set_title('D  Added discrimination', loc='left', fontweight='bold', fontsize=9)
    ax[3].text(.97, .93, 'dotted line =\npass threshold', transform=ax[3].transAxes, ha='right', va='top', fontsize=6)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def main():
    p = argparse.ArgumentParser(description='Proliferation-Adjusted Test for a candidate prognostic gene')
    p.add_argument('--expr', required=True); p.add_argument('--clin', required=True)
    p.add_argument('--gene', required=True); p.add_argument('--endpoint', default='efs', choices=['efs', 'os'])
    p.add_argument('--covariates', nargs='*', default=['age18', 'stage4', 'mycn'])
    p.add_argument('--out', default='.')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    expr = read_matrix(a.expr)
    clin = pd.read_csv(a.clin, index_col=0)
    clin = clin.loc[[s for s in clin.index if s in expr.columns]]
    rows = [pat(expr, clin, a.gene, a.endpoint, m, a.covariates) for m in ['CCNB2', 'MKI67', 'score']]
    rows = [r for r in rows if r.get('status') == 'ok'] or rows
    df = pd.DataFrame(rows); out_csv = os.path.join(a.out, f'PAT_{a.gene}_{a.endpoint}.csv')
    df.to_csv(out_csv, index=False)
    ok = [r for r in rows if r.get('status') == 'ok']
    passes = [r for r in ok if r['p_lrt'] < 0.05 and r['delta_c'] >= 0.005 and r['delta_aic'] <= -2]
    print(df.to_string(index=False)); print()
    if not ok:
        print(f'PROVISIONAL RESULT ({a.gene}, {a.endpoint.upper()}): not evaluable -- ' + rows[0].get('status', 'unknown'))
    else:
        med_rho = float(np.median([abs(r['rho_prolif']) for r in ok]))
        under = all(r['events'] < 50 for r in ok)
        label = ('UNDERPOWERED' if under else
                 'PROVISIONAL PASS' if len(passes) == len(ok) else
                 'PROVISIONAL FAIL' if not passes else 'MIXED ACROSS PROLIFERATION COVARIATES')
        print(f'PROVISIONAL RESULT ({a.gene}, {a.endpoint.upper()}, one cohort): {label}')
        print(f'  passes {len(passes)} of {len(ok)} proliferation covariates; '
              f'median |rho| with proliferation {med_rho:.2f}; events = {ok[0]["events"]}')
        if not under and not passes and med_rho >= 0.70:
            print('  pattern consistent with a PROLIFERATION SURROGATE')
    print('  NOTE: one cohort cannot return a final verdict. The full rule requires a pass in at')
    print('        least two independent cohorts with the same direction of effect.')
    if any(r.get('status') == 'ok' for r in rows):
        figure(expr, clin, a.gene, a.endpoint, a.covariates, [r for r in rows if r.get('status') == 'ok'],
               os.path.join(a.out, f'PAT_{a.gene}_{a.endpoint}.png'))
    print(f'  written: {out_csv}')

if __name__ == '__main__':
    main()
