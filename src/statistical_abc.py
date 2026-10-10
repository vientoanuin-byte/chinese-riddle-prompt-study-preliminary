"""Prespecified three-condition paired analysis; B-C is the primary contrast."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import GROUPS, extract_answer, normalize_type, digest
from statistical_test import coefficient_table

VALID = {'ok', 'no_answer'}
CONTRASTS = [('B', 'C'), ('B', 'A'), ('C', 'A')]

def read_abc(path, annotations):
    df = pd.read_csv(path, dtype={'riddle_id': str}, keep_default_na=False)
    required = {'riddle_id', 'group', 'correct', 'status', 'answer', 'response', 'predicted_answer',
                'component_count', 'source', 'run_id', 'question', 'model_requested', 'position', 'group_order'}
    if df.empty or not required <= set(df):
        raise ValueError('Empty results or missing ABC columns')
    if df.run_id.nunique() != 1 or df.model_requested.nunique() != 1:
        raise ValueError('Do not mix models/configurations')
    if df.duplicated(['riddle_id', 'group']).any() or set(df.group) != set(GROUPS):
        raise ValueError('Duplicate observations or missing/invalid conditions')
    for column in ['answer', 'question', 'source', 'component_count', 'group_order']:
        if df.groupby('riddle_id')[column].nunique().gt(1).any():
            raise ValueError(f'Within-riddle inconsistency: {column}')
    for _, row in df.iterrows():
        if ''.join(sorted(row.group_order)) != 'ABC' or row.group_order[int(row.position) - 1] != row.group:
            raise ValueError('Invalid condition order/position')
        if row.status in VALID:
            pred = extract_answer(row.response)
            if pred != row.predicted_answer or float(row.correct) != int(pred == row.answer):
                raise ValueError('Prediction/score does not match raw response')
    df['correct'] = pd.to_numeric(df.correct, errors='coerce')
    df.loc[~df.status.isin(VALID), 'correct'] = np.nan
    labels = pd.read_csv(annotations, dtype=str, keep_default_na=False)
    if labels.riddle_id.duplicated().any() or not set(df.riddle_id) <= set(labels.riddle_id):
        raise ValueError('Annotation IDs are missing or duplicated')
    labels['type'] = labels.type.map(normalize_type)
    labels = labels.set_index('riddle_id')
    # Annotation joins may not silently attach explanations to a different riddle.
    for col in ['question', 'answer']:
        if col in labels and not (df[col] == df.riddle_id.map(labels[col])).all():
            raise ValueError(f'Annotation {col} mismatch')
    df['type'] = df.riddle_id.map(labels.type)
    df['confidence'] = df.riddle_id.map(labels.confidence)
    if not set(df.confidence) <= {'high', 'medium', 'low'}:
        raise ValueError('Invalid annotation confidence')
    manifest_path = Path(path).with_suffix('.manifest.json')
    if not manifest_path.exists():
        raise ValueError('Missing run manifest; results must be traceable')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if digest(manifest['config']) != df.run_id.iloc[0] or manifest['run_id'] != df.run_id.iloc[0]:
        raise ValueError('Run manifest fingerprint mismatch')
    expected = {r['riddle_id'] for r in manifest['config']['data']}
    if set(zip(df.riddle_id, df.group)) != {(i, g) for i in expected for g in GROUPS}:
        raise ValueError('Incomplete scheduled run; resume collection before inference')
    return df

def pair_result(df, left, right):
    pivot = df.pivot(index='riddle_id', columns='group', values='correct')[[left, right]].dropna().astype(int)
    n = len(pivot)
    if n < 2:
        return {'contrast': f'{left}-{right}', 'n_pairs': n, 'available': False}, pivot
    table = pd.crosstab(pivot[right], pivot[left]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    difference = (pivot[left] - pivot[right]).to_numpy()
    rng = np.random.default_rng(20261009)
    boot = rng.choice(difference, size=(5000, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return {'contrast': f'{left}-{right}', 'n_pairs': n, 'available': True,
            'accuracy_left': float(pivot[left].mean()), 'accuracy_right': float(pivot[right].mean()),
            'difference': float(difference.mean()), 'ci_lower': float(lo), 'ci_upper': float(hi),
            'p_exact': float(mcnemar(table.to_numpy(), exact=True).pvalue),
            'right_wrong_left_correct': int(table.loc[0, 1]),
            'right_correct_left_wrong': int(table.loc[1, 0]),
            'table_right_rows_left_columns': table.to_numpy().tolist()}, pivot

def fit_models(data, output, notes):
    coef = {}
    if len(data) < 12 or data.correct.nunique() < 2:
        notes.append('回归未估计：可用样本过少或因变量没有变异。')
        return coef
    d = data.copy()
    d['is_B'] = d.group.eq('B').astype(int)
    d['is_A'] = d.group.eq('A').astype(int)  # C is reference, is_B is the primary contrast.
    terms = ['is_B', 'is_A']
    candidates = []
    for label in ['glyph', 'meaning', 'pronunciation']:
        key = 'has_' + label
        d[key] = d.type.map(lambda s: int(label in s.split('|')))
        candidates.append(key)
    d['type_unknown'] = d.type.eq('unknown').astype(int)
    candidates += ['type_unknown', 'C(source)', 'C(position)']
    values = pd.to_numeric(d.component_count, errors='coerce')
    if values.isna().any() or values.le(0).any():
        notes.append('回归未估计：结构变量缺失或非法，不自动补造。')
        return coef
    if values.nunique() > 1:
        d['component_z'] = (values - values.mean()) / values.std(ddof=0)
        candidates += ['component_z']
    for term in candidates:
        old = patsy.dmatrix(' + '.join(terms), d)
        new = patsy.dmatrix(' + '.join(terms + [term]), d)
        if new.shape[1] > old.shape[1] and np.linalg.matrix_rank(new) == new.shape[1]:
            terms.append(term)
        else:
            notes.append(f'控制项 {term} 为常数或共线，未单独估计。')
    formula = 'correct ~ ' + ' + '.join(terms)
    notes.append('回归公式（C为参照）: ' + formula)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': d.riddle_id})
        if not fit.converged or not np.isfinite(fit.bse).all() or abs(fit.params).max() > 20 or any('separation' in str(w.message).lower() for w in caught):
            raise ValueError('nonconvergence, separation, or unstable coefficients')
        coef['logistic'] = coefficient_table(fit.params.index, fit.params.to_numpy(), fit.bse.to_numpy(),
            'cluster_robust_95_CI', fit.pvalues)
        (output / 'logistic_summary.txt').write_text(str(fit.summary()), encoding='utf-8')
        # Shared answers can induce dependence beyond riddle IDs: report a sensitivity fit.
        if d.answer.nunique() > 1:
            fit2 = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': d.answer})
            coefficient_table(fit2.params.index, fit2.params.to_numpy(), fit2.bse.to_numpy(),
                'answer_cluster_sensitivity_95_CI', fit2.pvalues).to_csv(
                    output / 'logistic_answer_cluster_sensitivity.csv', index=False, encoding='utf-8-sig')
    except Exception as exc:
        notes.append(f'Logistic 未得到可靠估计：{type(exc).__name__}: {exc}')
    try:
        np.random.seed(20261009)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = BinomialBayesMixedGLM.from_formula(formula, {'riddle': '0 + C(riddle_id)'}, d, vcp_p=.5, fe_p=2)
            fit = model.fit_vb()
        if not fit.optim_retvals.success or not np.isfinite(fit.fe_sd).all():
            raise ValueError('VB did not converge')
        coef['glmm'] = coefficient_table(model.exog_names, fit.fe_mean, fit.fe_sd,
            'VB_approximate_95_credible_interval')
        (output / 'glmm_summary.txt').write_text(str(fit.summary()), encoding='utf-8')
        notes.extend('GLMM warning: ' + str(w.message) for w in caught)
    except Exception as exc:
        notes.append(f'GLMM 未得到可靠估计：{type(exc).__name__}: {exc}')
    for name, table in coef.items():
        table.to_csv(output / f'{name}_coefficients.csv', index=False, encoding='utf-8-sig')
    return coef

def figures(df, contrasts, coefficients, folder):
    folder.mkdir(parents=True, exist_ok=True)
    triples = df.pivot(index='riddle_id', columns='group', values='correct').dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (g, color) in enumerate(zip(GROUPS, ['#577590', '#43aa8b', '#f9c74f'])):
        if len(triples):
            k, n = int(triples[g].sum()), len(triples)
            lo, hi = proportion_confint(k, n, method='wilson')
            ax.bar(i, k/n, color=color, yerr=[[k/n-lo], [hi-k/n]], capsize=4)
    ax.set(xticks=[0, 1, 2], xticklabels=GROUPS, ylim=(0, 1.05), ylabel='Accuracy',
           title=f'Complete triples (n={len(triples)}); marginal 95% Wilson CI')
    fig.tight_layout(); fig.savefig(folder / 'accuracy_ABC.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, result in zip(axes, contrasts):
        if not result['available']:
            ax.text(.5, .5, 'Insufficient valid pairs', ha='center'); continue
        left, right = result['contrast'].split('-')
        table = np.array(result['table_right_rows_left_columns'])
        ax.imshow(table, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(table[i, j]), ha='center', va='center')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['wrong', 'correct'], yticklabels=['wrong', 'correct'],
               xlabel=left, ylabel=right, title=f'{left}-{right}, n={result["n_pairs"]}')
    fig.tight_layout(); fig.savefig(folder / 'paired_matrices.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name in zip(axes, ['logistic', 'glmm']):
        if name not in coefficients:
            ax.text(.5, .5, 'Unavailable: see stats_summary.txt', ha='center', va='center'); ax.set_axis_off(); continue
        t = coefficients[name]; y = np.arange(len(t))
        ax.errorbar(t.coefficient, y, xerr=1.96*t.std_error_or_posterior_sd, fmt='o', capsize=3)
        ax.axvline(0, color='gray', ls='--')
        ax.set(yticks=y, yticklabels=t.term, xlabel='Log odds (C reference)',
               title='Logistic: cluster 95% CI' if name == 'logistic' else 'GLMM: approximate 95% credible interval')
    fig.tight_layout(); fig.savefig(folder / 'regression_coefficients.png', dpi=180); plt.close(fig)
    for group in GROUPS:
        sub = df[df.group.eq(group) & df.status.isin(VALID)]
        pd.crosstab(sub.answer, sub.predicted_answer.replace('', '<NO_ANSWER>')).to_csv(
            folder / f'character_confusion_{group}.csv', encoding='utf-8-sig')

def analyze_abc(path, output, annotations):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty analysis folder to avoid stale artifacts')
    df = read_abc(path, annotations)
    output.mkdir(parents=True, exist_ok=True)
    contrasts = [pair_result(df, a, b)[0] for a, b in CONTRASTS]
    pivot = df.pivot(index='riddle_id', columns='group', values='correct')
    complete = pivot.dropna().index
    notes = ['主比较 B−C；B−A 与 C−A 为辅助比较。精确双侧 McNemar，alpha=0.05。',
        '每个比较采用该两组均有效的配对题；三组回归及准确率柱图使用完整三元组，分母可能不同。',
        '无汉字但正常结束的输出记错；截断、拒绝服务和接口失败不计能力错误，排除对应配对。',
        '区间按题配对 bootstrap 5000次；不同题共享谜底可能相关，提供按谜底聚类的Logistic敏感性结果。',
        '形音义为AI辅助初标，不是官方题型或独立人工金标准；低置信度与未知保留。类型控制回归属于探索性补充。',
        '主检验不依赖题型标签。语义与读音题很少，不据此宣称稳定的题型异质性。',
        '贝叶斯GLMM：固定项Normal(0,2²)，随机截距log(SD)~Normal(0,0.5²)，变分近似可信区间不是频率学派p值。',
        'B−C检验两套具体提示的差异，不能证明内部推理机制；汉字数近似匹配不保证token数相同。']
    coefficients = fit_models(df[df.riddle_id.isin(complete)], output, notes)
    reliable = df[df.confidence.isin(['high', 'medium']) & df.type.ne('unknown')]
    sensitivity, _ = pair_result(reliable, 'B', 'C')
    sensitivity['interpretation'] = 'Exploratory subset excluding low-confidence/unknown annotations; selection changes target population.'
    sens_ids = reliable.pivot(index='riddle_id', columns='group', values='correct').dropna().index
    sens_dir = output / 'annotation_sensitivity'; sens_dir.mkdir()
    sens_notes = []
    fit_models(reliable[reliable.riddle_id.isin(sens_ids)], sens_dir, sens_notes)
    sensitivity['model_notes'] = sens_notes
    per_group = df.groupby('group').agg(n_valid=('correct', 'count'), accuracy=('correct', 'mean'))
    unique = df.drop_duplicates('riddle_id')
    summary = {'n_riddles': len(unique), 'n_complete_triples': len(complete),
        'contrasts': contrasts, 'per_group_available': per_group.to_dict(orient='index'),
        'annotation_type_counts': unique.type.value_counts().to_dict(),
        'annotation_confidence_counts': unique.confidence.value_counts().to_dict(),
        'status_counts': df.groupby(['group', 'status']).size().to_dict(),
        'annotation_sensitivity': sensitivity, 'notes': notes,
        'model_requested': df.model_requested.iloc[0], 'models_returned': sorted(set(df.model_returned) - {''}) if 'model_returned' in df else [],
        'controlled_models_available': sorted(coefficients)}
    summary['status_counts'] = {f'{g}/{s}': int(v) for (g, s), v in summary['status_counts'].items()}
    import hashlib
    summary['annotations_sha256'] = hashlib.sha256(Path(annotations).read_bytes()).hexdigest()
    for suffix in ['json', 'txt']:
        (output / f'stats_summary.{suffix}').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(contrasts).to_csv(output / 'paired_tests.csv', index=False, encoding='utf-8-sig')
    df.groupby(['group', 'status']).size().to_csv(output / 'api_status_counts.csv', encoding='utf-8-sig')
    figures(df, contrasts, coefficients, output / 'figures')
    return summary
