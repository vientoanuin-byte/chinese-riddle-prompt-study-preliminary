"""Notebook helpers. Credentials remain in memory and are never serialized."""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import pandas as pd
from common import ROOT, load_data, digest, GROUPS
from run_experiment import ChatClient, main as run_main, make_payload
from statistical_abc import analyze_abc
from statsmodels.stats.multitest import multipletests

PROVIDERS = {
    'deepseek': {'model': 'deepseek-flash', 'base_url': 'https://api.deepseek.com',
                 'extra_body': {'thinking': {'type': 'disabled'}}, 'token_parameter': 'max_tokens'},
    'doubao': {'model': 'doubao-seed-2-0-lite-260428', 'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
               'extra_body': {'thinking': {'type': 'disabled'}}, 'token_parameter': 'max_tokens'},
    'qwen': {'model': 'qwen-plus', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
             'extra_body': {'enable_thinking': False}, 'token_parameter': 'max_tokens'},
}

def config_fingerprint(config):
    return digest(config)

def smoke_test(config, key, rpm=20):
    """One non-dataset request. Tests transport and output, not riddle accuracy."""
    client = ChatClient(config['base_url'], key, rpm, retries=2, timeout=90)
    try:
        result = client.call(make_payload(config['model'], [
            {'role': 'user', 'content': '请只输出一个汉字：好'}],
            token_parameter=config['token_parameter'], extra_body=config['extra_body']))
    finally:
        client.session.close()
    return {**result, 'config_fingerprint': config_fingerprint(config)}

def run_provider(name, config, key, output_root, rpm=20):
    output = Path(output_root) / name / 'experiment_results.csv'
    run_main(['--data', str(ROOT / 'data/test.json'), '--structure', str(ROOT / 'data/structure.json'),
              '--annotations', str(ROOT / 'data/type_annotations.csv'),
              '--output', str(output), '--base-url', config['base_url'], '--model', config['model'],
              '--rpm', str(rpm), '--token-parameter', config['token_parameter'],
              '--extra-body', json.dumps(config['extra_body'])], api_key=key)
    return output

def analyze_providers(names, output_root, annotations):
    """Separate models, then Holm adjustment across the prespecified model family."""
    output_root = Path(output_root)
    import hashlib
    plan_path = output_root / 'study_plan.json'
    if not plan_path.exists():
        raise ValueError('Missing preregistered study_plan.json')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['providers'] != list(names) or plan['annotations_sha256'] != hashlib.sha256(Path(annotations).read_bytes()).hexdigest():
        raise ValueError('Analysis model family/annotations differs from the frozen study plan')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    table = []
    expected = {r['riddle_id'] for r in load_data(ROOT / 'data/test.json')}
    for name in names:
        path = output_root / name / 'experiment_results.csv'
        if not path.exists():
            print(f'{name}: 没有结果文件，跳过。')
            continue
        data = pd.read_csv(path, dtype={'riddle_id': str}, keep_default_na=False)
        actual = set(zip(data.riddle_id, data.group))
        if actual != {(rid, group) for rid in expected for group in GROUPS}:
            print(f'{name}: 尚未完成全部计划题组，请先续跑；本次不生成中途显著性结果。')
            continue
        folder = output_root / name / ('analysis_' + stamp)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = analyze_abc(path, folder, annotations=annotations)
        for result in summary['contrasts']:
            table.append({**{k: v for k, v in result.items() if k != 'table_right_rows_left_columns'},
                'provider': name, 'model': data.model_requested.iloc[0], 'analysis_dir': str(folder),
                'controlled_models_available': len(summary['controlled_models_available']) == 2})
    result = pd.DataFrame(table)
    if len(result) == 3 * len(names) and result.available.all():
        # Primary family: B-C across models. Secondary family: both auxiliary contrasts across models.
        result['p_holm'] = float('nan')
        for mask in [result.contrast.eq('B-C'), ~result.contrast.eq('B-C')]:
            result.loc[mask, 'p_holm'] = multipletests(result.loc[mask, 'p_exact'], method='holm')[1]
        result['significant_positive_after_holm'] = (result.p_holm < .05) & (result.difference > 0)
    elif len(result):
        print('部分预设模型未完成，暂不计算全家族多重检验校正。')
    if len(result):
        result.to_csv(output_root / ('model_comparison_' + stamp + '.csv'), index=False, encoding='utf-8-sig')
        from generate_report import write_section
        write_section(result, output_root / ('实证分析_' + stamp + '.md'), names)
    return result
