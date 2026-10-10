"""Generate the deliverable notebook without execution output or credentials."""
import json
import textwrap
from common import ROOT

cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})
def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                  'source': textwrap.dedent(text).strip() + '\n'})

md('''
# 大模型汉字结构理解缺陷与形音意提示的实证检验

**ABC运行版 · 300 道固定样本 · DeepSeek / 豆包 / Qwen**

请将本 Notebook 保留在 `Project_Chinese_NLP` 目录，与 `src/`、`data/` 同级。
用 VS Code 或 Jupyter 打开，选择本项目 `.venv` 的 Python 内核，按顺序运行。
首次使用可双击 `启动实验Notebook.cmd`。每家默认900次正式调用，三家共2700次；
每家另有1次非实验连通性测试，失败重试另计。20 RPM下三家顺序运行至少约135分钟，实际含生成时间可能更久。

**密钥通过隐藏输入框填写，不要粘贴进代码单元格。** 输入后存在当前内核内存中，重启内核需重新输入。
本文件未执行真实 API，离线验证不能替代账号权限、余额和模型可用性检查。

300题已逐题完成AI辅助初标（非独立人工金标准），每题带依据和置信度；不能可靠判断的保留unknown。
主比较B−C不依赖标签。自动分析同时运行控制回归与剔除低置信度标注的敏感性分析。
运行全部单元格后自动生成结果CSV、图表、统计摘要和可插入报告的“实证分析.md”章节。
原始论文初稿不覆盖；统计或回归失败会如实说明，不制造显著结果。不要将模拟测试数据作为实证。
''')
md('## 1. 环境与数据检查（不联网）')
code('''
from pathlib import Path
import sys, json, copy, getpass, warnings, hashlib
import pandas as pd
from IPython.display import display, Image

candidates = [Path.cwd(), Path.cwd() / 'Project_Chinese_NLP']
PROJECT = next((p.resolve() for p in candidates if (p / 'src/notebook_workflow.py').exists()), None)
if PROJECT is None:
    raise RuntimeError('请在 Project_Chinese_NLP 目录打开本 Notebook，并保留 src 和 data 文件夹。')
sys.path.insert(0, str(PROJECT / 'src'))
from common import load_data, normalize_type, digest, GUIDANCE, CONTROL, balanced_schedule
from notebook_workflow import PROVIDERS, smoke_test, run_provider, analyze_providers, config_fingerprint

DATA = load_data(PROJECT / 'data/test.json')
ANNOTATIONS = PROJECT / 'data/type_annotations.csv'
labels = pd.read_csv(ANNOTATIONS, dtype=str, keep_default_na=False)
assert len(DATA) == 300 and len({r['riddle_id'] for r in DATA}) == 300
assert not labels.riddle_id.duplicated().any()
assert set(labels.riddle_id) == {r['riddle_id'] for r in DATA}
labels['type'] = labels.type.map(normalize_type)
print('Python:', sys.executable)
print('数据:', len(DATA), '题；来源:', pd.Series([r['source'] for r in DATA]).value_counts().to_dict())
print('保留未知题数:', int(labels.type.eq('unknown').sum()))
print('置信度:', labels.confidence.value_counts().to_dict())
print('B/C提示字符数:', len(GUIDANCE), len(CONTROL), '（不等于不同模型的token数）')
print('六种顺序分配:', pd.Series([order for _, _, pos, order in balanced_schedule(DATA) if pos == 1]).value_counts().to_dict())
print('B：', GUIDANCE)
print('C：', CONTROL)
''')
md('''
## 2. 模型配置

下面预填官方按量付费接口。Qwen 默认中国内地（北京）百炼；若你的密钥来自国际区或其他平台，
必须换成该账户对应的地址与模型 ID，不能把 Coding Plan 的密钥当普通按量接口密钥使用。
豆包默认 Lite；如果只开通了其他模型或推理接入点，修改 `model` 为控制台显示的标识。
DeepSeek/Qwen 的别名可能更新，保留请求与服务端返回型号及调用时间；尽量在短时间内跑完同一模型。

每款模型A/B/C都使用同样的温度、思考模式、输出上限。没有联网搜索、历史对话或工具。
正式实验开始后不修改模型、提示、样本；更改配置请使用新批次目录。
''')
code('''
CONFIGS = copy.deepcopy(PROVIDERS)
ENABLED = ['deepseek', 'doubao', 'qwen']
RPM = 20
OUTPUT_ROOT = PROJECT / 'output/three_models_ABC_v1'  # 续跑保持不变；更换模型/方案需新目录
assert ENABLED and len(ENABLED) == len(set(ENABLED)) and all(n in CONFIGS for n in ENABLED)
# ===== 豆包预置模型：Doubao-Seed-2.0-lite（260428） =====
# 这是火山方舟模型列表中显示的预置 Model ID；本 Notebook 不要求填写 ep-接入点。
DOUBAO_MODEL_ID = 'doubao-seed-2-0-lite-260428'
CONFIGS['doubao']['model'] = DOUBAO_MODEL_ID
display(pd.DataFrame([{'provider': n, **CONFIGS[n]} for n in ENABLED]))
print('正式调用计划:', len(ENABLED) * 3 * len(DATA), '次（不含试调用与重试）')
''')
md('## 3. 输入密钥（不发起请求；密钥不回显）')
code('''
KEYS = {}
with warnings.catch_warnings():
    warnings.simplefilter('error', getpass.GetPassWarning)
    for name in ENABLED:
        key = getpass.getpass(f'{name} API Key（隐藏输入）: ').strip()
        if not key:
            raise ValueError(f'{name} 密钥为空。若不使用该服务，请在 ENABLED 中删除并重新运行配置。')
        KEYS[name] = key
key = None
print('已在内存中录入密钥。')
''')
md('''
## 4. 连通性测试（每家 1 次基础付费请求）

只请求输出“好”，不使用测试集题目，不评估模型好坏，也不据此筛选提示。
401/403 请核对密钥与权限；404 核对模型及区域；400/422 核对模型参数；429 核对额度与速率。
出现失败时修正配置后重跑本格；正式实验要求所有启用模型通过测试。
''')
code('''
PREFLIGHT = {}
for name in ENABLED:
    result = smoke_test(CONFIGS[name], KEYS[name], RPM)
    PREFLIGHT[name] = result
    print(name, 'status=', result['status'], 'HTTP=', result.get('http_status'),
          'requested_model=', CONFIGS[name].get('model'),
          'returned_model=', result.get('model_returned', ''),
          'error=', result.get('error_body', ''),
          '输出检查=', result.get('response', '').strip() == '好')
print('所有模型均 status=ok 且输出检查=True 后再运行下一步。')
''')
md('''
## 5. 正式实验（运行本格即开始批量调用）

每家逐题运行A/B/C；六种顺序各50题，题序随机且固定。结果逐条落盘；中断后重新运行可跳过已有记录。
已有失败记录保留，不因答错而重新采样。请求配置指纹阻止混用不同模型或提示的旧结果。
题型与标准答案结构均不会发给模型；标注文件在此记录校验值，分析时防止事后静默修改。
''')
code('''
for name in ENABLED:
    check = PREFLIGHT.get(name, {})
    if (check.get('status') != 'ok' or check.get('response', '').strip() != '好'
            or check.get('config_fingerprint') != config_fingerprint(CONFIGS[name])):
        raise RuntimeError(f'{name} 未通过当前配置的试调用，请先完成第4步。')

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
plan_path = OUTPUT_ROOT / 'study_plan.json'
plan = {'providers': ENABLED, 'configs': {n: CONFIGS[n] for n in ENABLED},
        'sample_sha256': digest(DATA), 'n_riddles': len(DATA),
        'annotations_sha256': hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
        'primary_contrast': 'B-C', 'groups': ['A', 'B', 'C'],
        'prompts': {'B': GUIDANCE, 'C': CONTROL},
        'multiple_testing': 'Primary B-C family: Holm across models; secondary B-A/C-A: separate Holm family'}
if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
    raise RuntimeError('实验计划与已有批次不符，请恢复配置或指定新 OUTPUT_ROOT。')
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')

for name in ENABLED:
    print(f'开始/继续 {name} ...')
    try:
        run_provider(name, CONFIGS[name], KEYS[name], OUTPUT_ROOT, RPM)
    except SystemExit:
        print(f'{name} 已停止；查看上方错误，已有记录已保存。继续其他模型。')
print('实验循环结束，下一格检查各家完整性与失败记录。')
''')
md('## 6. 检查采集完整性')
code('''
for name in ENABLED:
    path = OUTPUT_ROOT / name / 'experiment_results.csv'
    if path.exists():
        frame = pd.read_csv(path, dtype={'riddle_id': str}, keep_default_na=False)
        print(name, f'{len(frame)}/{3 * len(DATA)} 条', frame.status.value_counts().to_dict())
        failed = frame[~frame.status.isin(['ok', 'no_answer'])]
        if len(failed):
            display(failed[['riddle_id', 'group', 'status', 'http_status']])
    else:
        print(name, '未生成结果')
''')
md('''
## 7. 统计与绘图（不调用 API）

每个模型单独做精确双侧McNemar；B−C为主要比较，B−A、C−A为辅助比较。
同一300题在多模型重复，不视为900道独立题。分别对主、辅比较家族给出Holm校正p值。
失败题成对排除并报告，缺失机制可能影响结论。未完成采集的模型不生成中途显著性结果。
自动生成以C为参照的聚类稳健Logistic与题目随机截距二项GLMM；GLMM为近似贝叶斯可信区间。
回归作为探索性补充，未知标签单列，低置信度剔除另作敏感性分析，按谜底聚类也作敏感性检查。
若存在完全分离或非收敛，保留主检验并报告回归不可估计，不把缺失结果写成零效果。
新分析写入时间戳目录，保留旧输出与原始答案；分析记录标注文件 SHA256。
''')
code('''
COMPARISON = analyze_providers(ENABLED, OUTPUT_ROOT, ANNOTATIONS)
if not COMPARISON.empty:
    display(COMPARISON.drop(columns=['analysis_dir']))
    for _, row in COMPARISON.drop_duplicates('provider').iterrows():
        print(row['provider'], row['model'], '完整控制模型可用:', row['controlled_models_available'])
        folder = Path(row['analysis_dir']) / 'figures'
        for filename in ['accuracy_ABC.png', 'paired_matrices.png', 'regression_coefficients.png']:
            display(Image(filename=str(folder / filename)))
else:
    print('尚无完整实验可分析。')
reports = sorted(OUTPUT_ROOT.glob('实证分析_*.md'))
if reports:
    print('本次可插入论文的结果章节：', reports[-1])
    print('保留章节及相对路径下的图表；合并进论文时需一并调整图片路径。')
''')
md('''
## 8. 下一步

保留整个 `output/three_models_ABC_v1/`，包括各家的CSV、manifest、统计摘要、图表与实证分析章节。
自动生成的章节依据实际数据判断方向与显著性；提交前与理论初稿合并审阅。尚未产生数据时不生成最终报告。
关闭内核可清除内存中的密钥。不要打印 `KEYS`，也不要把密钥改写到代码或输出中。

接口核对来源（2026-10-09；账户实际可用型号以试调用为准）：
- [DeepSeek Chat API](https://api-docs.deepseek.com/api/create-chat-completion/)
- [豆包模型列表](https://docs.volcengine.com/docs/82379/1330310)
- [豆包深度思考](https://www.volcengine.com/docs/82379/1449737)
- [Qwen 深度思考参数](https://help.aliyun.com/zh/model-studio/deep-thinking)
- [Qwen 区域与调用](https://help.aliyun.com/en/model-studio/model-calling-in-sub-workspace)
''')

for i, cell in enumerate(cells):
    cell['id'] = f'cell-{i:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3 (ipykernel)',
    'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}},
    'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / '汉字字谜_ABC实验.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print(path)
legacy = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'id': 'deprecated',
    'source': '# 旧版已停用\n\n请打开同目录的 [汉字字谜_ABC实验.ipynb](汉字字谜_ABC实验.ipynb)。\n本页不调用API；三模型三组方案已替代原四模型A/B方案。'}],
    'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5}
(ROOT / '汉字字谜_AB实验.ipynb').write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding='utf-8')
