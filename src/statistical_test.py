"""Paired inference, cluster-robust logistic regression, Bayesian binomial GLMM, figures."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, extract_answer, normalize_type

def read_results(path):
    df = pd.read_csv(path, dtype={'riddle_id': str}, keep_default_na=False)
    required = {'riddle_id', 'group', 'correct', 'status', 'answer', 'response', 'predicted_answer',
                'type', 'component_count', 'radical_count', 'run_id', 'question', 'model_requested'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError('Missing required columns or empty CSV')
    if df.run_id.nunique() != 1 or df.model_requested.nunique() != 1:
        raise ValueError('Analyze one model/configuration at a time')
    if df.duplicated(['riddle_id', 'group']).any() or not set(df.group) <= {'A', 'B'}:
        raise ValueError('Duplicate pairs or invalid groups')
    for col in ['question', 'answer', 'type', 'component_count', 'radical_count']:
        if df.groupby('riddle_id')[col].nunique().gt(1).any():
            raise ValueError(f'Within-pair metadata mismatch: {col}')
    df['type'] = df.type.map(normalize_type)
    valid = df.status.isin(['ok', 'no_answer'])
    for _, r in df.loc[valid].iterrows():
        pred = extract_answer(r.response)
        if pred != r.predicted_answer or int(r.correct) != int(pred == r.answer):
            raise ValueError('Stored prediction/correctness disagrees with response; do not edit raw CSV')
    df['correct'] = pd.to_numeric(df.correct, errors='coerce')
    if not df.loc[valid, 'correct'].isin([0, 1]).all():
        raise ValueError('Correctness must be binary')
    good = df.loc[valid].groupby('riddle_id').group.size()
    ids = good[good == 2].index
    paired = df.loc[df.riddle_id.isin(ids)].copy()
    if len(ids) < 2:
        raise ValueError('Need at least two complete valid A/B pairs')
    return df, paired

def coefficient_table(names, means, ses, method, pvalues=None):
    table = pd.DataFrame({'term': names, 'coefficient': means, 'std_error_or_posterior_sd': ses})
    table['lower'] = table.coefficient - 1.96 * table.std_error_or_posterior_sd
    table['upper'] = table.coefficient + 1.96 * table.std_error_or_posterior_sd
    table['odds_ratio'] = np.exp(np.clip(table.coefficient, -700, 700))
    table['method'] = method
    if pvalues is not None:
        table['p_value'] = np.asarray(pvalues)
    return table

def save_figures(paired, pivot, coefs, folder):
    folder.mkdir(parents=True, exist_ok=True)
    categories = ['Overall'] + sorted(paired.type.unique().tolist())
    fig, ax = plt.subplots(figsize=(max(6, len(categories) * 1.5), 4))
    x = np.arange(len(categories))
    for offset, group, color in [(-.18, 'A', '#577590'), (.18, 'B', '#43aa8b')]:
        means, lows, highs = [], [], []
        for category in categories:
            sub = paired[paired.group.eq(group)]
            if category != 'Overall':
                sub = sub[sub.type.eq(category)]
            n, k = len(sub), int(sub.correct.sum())
            lo, hi = proportion_confint(k, n, method='wilson')
            means.append(k / n)
            lows.append(k / n - lo)
            highs.append(hi - k / n)
        ax.bar(x + offset, means, .36, label=group, color=color,
               yerr=[lows, highs], capsize=3)
    ax.set(xticks=x, xticklabels=categories, ylim=(0, 1.08), ylabel='Accuracy',
           title='Paired sample accuracy (marginal 95% Wilson intervals)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'accuracy_comparison.png', dpi=180)
    plt.close(fig)
    table = pd.crosstab(pivot.A, pivot.B).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(table, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(table.iloc[i, j]), ha='center', va='center', fontsize=18)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['B wrong', 'B correct'],
           yticklabels=['A wrong', 'A correct'], title='Paired correctness transition matrix')
    fig.tight_layout()
    fig.savefig(folder / 'paired_confusion_matrix.png', dpi=180)
    plt.close(fig)
    # Full multiclass character confusion matrices are also exported without collapsing labels.
    for group in ['A', 'B']:
        sub = paired[paired.group.eq(group)]
        pd.crosstab(sub.answer, sub.predicted_answer.replace('', '<NO_ANSWER>')).to_csv(
            folder / f'character_confusion_{group}.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, max([len(t) for t in coefs.values()] + [3]) * .4)))
    for ax, key, title in zip(axes, ['logistic', 'glmm'],
                            ['Logistic: cluster-robust 95% CI', 'Binomial GLMM: approximate 95% credible interval']):
        table = coefs.get(key)
        if table is None:
            ax.text(.5, .5, 'Model unavailable; see stats_summary.txt', ha='center', va='center', wrap=True)
            ax.set_axis_off()
            continue
        y = np.arange(len(table))
        ax.errorbar(table.coefficient, y, xerr=1.96 * table.std_error_or_posterior_sd, fmt='o', capsize=3)
        ax.axvline(0, color='gray', linestyle='--')
        ax.set(yticks=y, yticklabels=table.term, xlabel='Log-odds coefficient', title=title)
    fig.tight_layout()
    fig.savefig(folder / 'regression_coefficients.png', dpi=180)
    plt.close(fig)

def analyze(path, output, allow_incomplete=False, annotations=None):
    raw, paired = read_results(path)
    annotation_hash = None
    if annotations is not None:
        import hashlib
        labels = pd.read_csv(annotations, dtype=str, keep_default_na=False)
        if not {'riddle_id', 'type'}.issubset(labels.columns) or labels.riddle_id.duplicated().any():
            raise ValueError('Annotations need unique riddle_id and type columns')
        mapping = dict(zip(labels.riddle_id, labels.type.map(normalize_type)))
        if not set(raw.riddle_id) <= set(mapping):
            raise ValueError('Annotations do not cover all observed riddles')
        for frame in [raw, paired]:
            frame['type'] = frame.riddle_id.map(mapping)
        annotation_hash = hashlib.sha256(Path(annotations).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    pivot = paired.pivot(index='riddle_id', columns='group', values='correct').astype(int)
    table = pd.crosstab(pivot.A, pivot.B).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    result = mcnemar(table.to_numpy(), exact=True)
    diff = (pivot.B - pivot.A).to_numpy()
    rng = np.random.default_rng(20261009)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
    ci = np.quantile(boot, [.025, .975])
    summary = {'n_observed_riddles': int(raw.riddle_id.nunique()), 'n_complete_pairs': len(pivot),
        'n_excluded_riddles': int(raw.riddle_id.nunique() - len(pivot)),
        'accuracy_A': float(pivot.A.mean()), 'accuracy_B': float(pivot.B.mean()),
        'difference_B_minus_A': float(diff.mean()), 'paired_bootstrap_95_CI': ci.tolist(),
        'both_wrong': int(table.loc[0, 0]), 'A_wrong_B_correct': int(table.loc[0, 1]),
        'A_correct_B_wrong': int(table.loc[1, 0]), 'both_correct': int(table.loc[1, 1]),
        'mcnemar_exact_two_sided_p': float(result.pvalue), 'status_counts': raw.status.value_counts().to_dict()}
    notes = ['Primary endpoint: exact two-sided paired McNemar test, alpha=0.05.',
             'A positive difference alone is not evidence of a statistically significant improvement.',
             'API errors / truncated outputs exclude the entire pair; successful no-Han responses count as wrong.',
             'Bootstrap resamples riddle pairs (10000 draws, seed=20261009). Degenerate intervals may occur.',
             'Character confusion matrices are CSVs; plotted 2x2 matrix is paired correctness, not binary classification.',
             'Random intercept is riddle_id; binary outcome uses a binomial GLMM, not a Gaussian mixed model.']
    if annotation_hash:
        summary['analysis_annotations_sha256'] = annotation_hash
        notes.append('Types joined at analysis from a separate annotation file; raw responses are unchanged. Disclose if annotations were made after observing outcomes.')
    manifest = path.with_suffix('.manifest.json')
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding='utf-8'))
        expected = {str(r['riddle_id']) for r in m['config']['data']}
        if not set(raw.riddle_id) <= expected or m['run_id'] != raw.run_id.iloc[0]:
            raise ValueError('Manifest and CSV mismatch')
        summary['n_planned_riddles'] = len(expected)
        summary['n_unobserved_riddles'] = len(expected - set(raw.riddle_id))
    else:
        notes.append('No manifest: cannot verify planned sample size or completely absent pairs.')
    raw.groupby(['group', 'status']).size().to_csv(output / 'api_status_counts.csv', encoding='utf-8-sig')
    paired['treatment'] = paired.group.eq('B').astype(int)
    terms = ['treatment']
    unknown = paired.type.eq('unknown').any()
    missing_structure = False
    for column in ['component_count', 'radical_count']:
        paired[column] = pd.to_numeric(paired[column], errors='coerce')
        if column == 'component_count':
            missing_structure |= paired[column].isna().any()
        if paired[column].dropna().lt(1).any():
            raise ValueError('Component counts must be positive')
    coefs = {}
    blocked = (unknown or missing_structure) and not allow_incomplete
    if blocked:
        notes.append('CONTROLLED MODELS NOT FIT: missing type/structure. Complete annotations first; --allow-incomplete-controls permits explicitly exploratory reduced models.')
    else:
        for label in ['glyph', 'meaning', 'pronunciation']:
            col = 'has_' + label
            paired[col] = paired.type.map(lambda s: int(label in s.split('|')))
            if paired[col].nunique() > 1:
                import patsy
                design = patsy.dmatrix(' + '.join(terms + [col]), paired)
                if np.linalg.matrix_rank(design) == design.shape[1]:
                    terms.append(col)
                else:
                    notes.append(f'{col} omitted as collinear (reference indicator).')
            else:
                notes.append(f'{col} constant: separate coefficient not identifiable.')
        if 'source' in paired and paired.source.nunique() > 1:
            terms.append('C(source)')
        # A single prespecified structural measure avoids redundant component-count controls.
        values = paired.component_count
        if values.isna().any():
            notes.append('Exploratory model: component_count missing; median imputation plus missingness indicator.')
            if values.notna().any():
                paired['component_missing'] = values.isna().astype(int)
                terms.append('component_missing')
                values = values.fillna(values.median())
        if values.notna().all() and values.nunique() > 1:
            paired['component_z'] = (values - values.mean()) / values.std(ddof=0)
            terms.append('component_z')
        else:
            notes.append('component_count constant or wholly missing: omitted (not identifiable).')
        if unknown:
            notes.append('Exploratory model: unknown has all type indicators zero; this is not complete type adjustment.')
        formula = 'correct ~ ' + ' + '.join(terms)
        summary['model_formula'] = formula
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit = smf.glm(formula, data=paired, family=sm.families.Binomial()).fit(
                    cov_type='cluster', cov_kwds={'groups': paired.riddle_id})
            notes.extend('Logistic warning: ' + str(w.message) for w in caught)
            if not fit.converged or not np.isfinite(fit.bse).all() or np.max(np.abs(fit.params)) > 20 or any('separation' in str(w.message).lower() for w in caught):
                raise ValueError('Nonconvergence, separation, or unstable coefficients; no reliable Wald inference')
            coefs['logistic'] = coefficient_table(fit.params.index, fit.params.to_numpy(), fit.bse.to_numpy(),
                'frequentist_cluster_robust_95_CI', fit.pvalues)
            (output / 'logistic_summary.txt').write_text(str(fit.summary()), encoding='utf-8')
        except Exception as exc:
            notes.append(f'Logistic unavailable: {type(exc).__name__}: {exc}')
        try:
            np.random.seed(20261009)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = BinomialBayesMixedGLM.from_formula(formula, {'riddle': '0 + C(riddle_id)'},
                    paired, vcp_p=.5, fe_p=2)
                fit = model.fit_vb()
            notes.extend('GLMM warning: ' + str(w.message) for w in caught)
            if not fit.optim_retvals.success or not np.isfinite(fit.fe_sd).all():
                raise ValueError('Variational Bayes optimization did not converge')
            coefs['glmm'] = coefficient_table(model.exog_names, fit.fe_mean, fit.fe_sd,
                'Bayesian_variational_approximate_95_credible_interval')
            summary['glmm_random_intercept_sd_posterior_median'] = float(np.exp(fit.vcp_mean[0]))
            notes.append('GLMM priors: fixed effects Normal(0,2^2); log random-effect SD Normal(0,0.5^2). VB intervals are approximate credible intervals, not frequentist p-values.')
            notes.append('Only two observations per riddle: random-effect variance can be weakly identified and prior-sensitive.')
            (output / 'glmm_summary.txt').write_text(str(fit.summary()), encoding='utf-8')
        except Exception as exc:
            notes.append(f'GLMM unavailable: {type(exc).__name__}: {exc}')
    for name, coef in coefs.items():
        coef.to_csv(output / f'{name}_coefficients.csv', index=False, encoding='utf-8-sig')
    summary['notes'] = notes
    (output / 'stats_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'stats_summary.txt').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save_figures(paired, pivot, coefs, output / 'figures')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'output/experiment_results.csv')
    p.add_argument('--output', type=Path, default=ROOT / 'output/analysis')
    p.add_argument('--allow-incomplete-controls', action='store_true')
    p.add_argument('--annotations', type=Path, default=ROOT / 'data/type_annotations.csv')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Analysis output directory is nonempty; choose a new directory to avoid stale results')
    if 'C' in set(pd.read_csv(args.input, usecols=['group']).group):
        from statistical_abc import analyze_abc
        analyze_abc(args.input, args.output, args.annotations)
    else:
        analyze(args.input, args.output, args.allow_incomplete_controls)

if __name__ == '__main__':
    main()
