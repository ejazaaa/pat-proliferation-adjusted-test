"""Proliferation-Adjusted Test (PAT) for candidate prognostic genes.
Core routine: nested Cox model comparison of a baseline clinical model against
the same model plus a candidate gene, with a proliferation covariate in the baseline.
"""
import numpy as np, pandas as pd, warnings
from lifelines import CoxPHFitter
from scipy import stats
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
warnings.filterwarnings('ignore')

PROLIF_SCORE_GENES = ['CCNB1','CCNB2','CDK1','CDC20','BUB1','AURKA','TOP2A','CENPF','BIRC5','RRM2']

def z(v):
    v = pd.Series(v).astype(float)
    return (v - v.mean()) / v.std()

def build_proliferation(expr, samples, mode='CCNB2', exclude=None):
    """mode: 'CCNB2' (primary), 'MKI67', or 'score' (multi-gene, leave-one-out)."""
    if mode in ('CCNB2','MKI67'):
        if mode not in expr.index: return None
        if exclude == mode: return None
        return z(expr.loc[mode, samples])
    genes = [g for g in PROLIF_SCORE_GENES if g in expr.index and g != exclude]
    if len(genes) < 5: return None
    return z(expr.loc[genes, samples].apply(z, axis=1).mean())

def pat(expr, clin, gene, endpoint='efs', prolif_mode='CCNB2', covariates=None, horizons=(3,5)):
    """Run the proliferation-adjusted test for one gene in one cohort.
    expr : genes x samples DataFrame (log2). clin : indexed by sample, with
    <endpoint>_t, <endpoint>_e and clinical covariates. Returns a dict of statistics."""
    t, e = f'{endpoint}_t', f'{endpoint}_e'
    if gene not in expr.index: return {'gene': gene, 'status': 'gene not measured'}
    d = clin.copy()
    cov = [c for c in (covariates or ['age18','stage4','mycn']) if c in d and d[c].notna().sum() > 0 and d[c].nunique() > 1]
    p = build_proliferation(expr, d.index, prolif_mode, exclude=gene)
    if p is None: return {'gene': gene, 'status': f'proliferation covariate {prolif_mode} unavailable'}
    d['prolif'] = p.values
    d['x'] = z(expr.loc[gene, d.index]).values
    d = d.dropna(subset=['x','prolif',t,e] + cov)
    if d[e].sum() < 10: return {'gene': gene, 'status': 'too few events'}
    base_cov = cov + ['prolif']
    base = CoxPHFitter(penalizer=1e-4).fit(d[base_cov + [t,e]], t, e)
    full = CoxPHFitter(penalizer=1e-4).fit(d[['x'] + base_cov + [t,e]], t, e)
    uni  = CoxPHFitter(penalizer=1e-4).fit(d[['x',t,e]], t, e)
    lrt = 2 * (full.log_likelihood_ - base.log_likelihood_)
    p_lrt = stats.chi2.sf(max(lrt, 0), 1)
    aic = lambda m, k: -2*m.log_likelihood_ + 2*k
    d_aic = aic(full, len(base_cov)+1) - aic(base, len(base_cov))
    s = full.summary.loc['x']
    out = {'gene': gene, 'endpoint': endpoint.upper(), 'prolif_covariate': prolif_mode,
           'n': len(d), 'events': int(d[e].sum()), 'covariates': '+'.join(base_cov),
           'hr_univariable': float(np.exp(uni.params_['x'])), 'p_univariable': float(uni.summary.p['x']),
           'hr_adjusted': float(np.exp(s['coef'])),
           'ci_low': float(np.exp(s['coef lower 95%'])), 'ci_high': float(np.exp(s['coef upper 95%'])),
           'p_adjusted': float(s['p']), 'lrt_chi2': float(lrt), 'p_lrt': float(p_lrt),
           'c_base': float(base.concordance_index_), 'c_full': float(full.concordance_index_),
           'delta_c': float(full.concordance_index_ - base.concordance_index_),
           'delta_aic': float(d_aic),
           'rho_prolif': float(stats.spearmanr(d.x, d.prolif).correlation),
           'status': 'ok'}
    # time-dependent AUC and calibration
    try:
        y = Surv.from_arrays(d[e].astype(bool), d[t].astype(float))
        hz = [h*365.25 for h in horizons if h*365.25 < d[t].max()*0.95]
        if hz:
            a_b,_ = cumulative_dynamic_auc(y, y, base.predict_partial_hazard(d).values, hz)
            a_f,_ = cumulative_dynamic_auc(y, y, full.predict_partial_hazard(d).values, hz)
            for h,ab,af in zip(horizons, a_b, a_f):
                out[f'auc{h}y_base'] = float(ab); out[f'auc{h}y_full'] = float(af); out[f'delta_auc{h}y'] = float(af-ab)
    except Exception as ex:
        out['auc_note'] = str(ex)[:60]
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ph = full.check_assumptions(d[['x']+base_cov+[t,e]], show_plots=False)
        out['ph_violation'] = bool(len(ph) > 0)
    except Exception:
        out['ph_violation'] = None
    return out

def verdict(rows, n_required=2, min_events=50, dc=0.005, daic=-2.0, rho_hi=0.70):
    """Apply the PAT decision rule across cohorts and endpoints for one gene.

    Returns (verdict, rationale). A single analysis 'passes' only if all three
    hold: likelihood-ratio p < 0.05, change in Harrell's C >= +0.005, and
    change in AIC <= -2. Analyses with fewer than min_events events are not
    counted either way."""
    ok = [r for r in rows if r.get('status') == 'ok']
    if not ok:
        return 'INDETERMINATE', 'no cohort returned a usable model'
    powered = [r for r in ok if r['events'] >= min_events]
    if len(powered) < n_required:
        return 'INDETERMINATE', f'only {len(powered)} analyses with >= {min_events} events'
    passes = [r for r in powered if r['p_lrt'] < 0.05 and r['delta_c'] >= dc and r['delta_aic'] <= daic]
    nominal = [r for r in powered if r['p_lrt'] < 0.05]
    med_rho = float(np.median([abs(r['rho_prolif']) for r in powered]))
    if len(passes) >= n_required:
        dirs = {np.sign(np.log(r['hr_adjusted'])) for r in passes}
        if len(dirs) == 1:
            return 'ADDS INFORMATION', (f'{len(passes)} of {len(powered)} powered analyses pass '
                                        f'with consistent direction')
        return 'INDETERMINATE', f'{len(passes)} analyses pass but the direction is inconsistent'
    if med_rho >= rho_hi:
        extra = f'; nominal LRT in {len(nominal)} analysis(es) without meeting the effect-size criteria' if nominal else ''
        return 'PROLIFERATION SURROGATE', (f'{len(passes)} of {len(powered)} powered analyses pass; '
                                           f'median |rho| with the proliferation covariate {med_rho:.2f}{extra}')
    return 'NOT INFORMATIVE', (f'{len(passes)} of {len(powered)} powered analyses pass; '
                               f'collinearity with proliferation is modest (median |rho| {med_rho:.2f})')
