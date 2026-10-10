"""Generate a standalone pilot notebook, without running any API requests."""
import json
from common import ROOT
from pilot31 import prepare

prepare()
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s,execution_count=None,outputs=[]))
md('''# 31题 ABC 探索性复测

选择项目 .venv 内核，依次运行，只需隐藏输入 DeepSeek、豆包、Qwen 密钥。
正式请求共279次，另有3次预检，重试另计。20 RPM下限约14分钟，实际可能更久。
输出单独保存在 output/pilot31_v1，不覆盖300题实验。

这不是大样本或独立验证：31题从已经分析过的300题中固定种子20261010抽取，不按旧结果挑题。
每道题做ABC配对，主要比较B-C；保留A以检查通用指令的影响。难度标签不发送给模型。
难度标准：低=少量常见部件且线索明确；中=局部提取、替换或一个语义中间步骤；高=多步操作、隐喻/典故、较不熟悉答案或显著歧义。是AI辅助初标，不是经验证的客观难度；低置信度项目需人工复核。
不会根据结果重新抽样、挑选显著模型或中途修改提示。若要优化提示，应在此开发，之后另取未见题验证。
''')
code('''from pathlib import Path
import sys, json, copy, getpass, warnings
import pandas as pd
from IPython.display import display, Image
PROJECT=next(p.resolve() for p in [Path.cwd(),Path.cwd()/'Project_Chinese_NLP'] if (p/'src/pilot31.py').exists())
sys.path.insert(0,str(PROJECT/'src'))
from pilot31 import PROMPTS, schedule, run, analyze, messages
from notebook_workflow import PROVIDERS, smoke_test, config_fingerprint
DATA=json.loads((PROJECT/'data/pilot31.json').read_text(encoding='utf-8'))
CONFIGS=copy.deepcopy(PROVIDERS)
OUTPUT=PROJECT/'output/pilot31_v1'
RPM=20
assert len(DATA)==31 and len({r['riddle_id'] for r in DATA})==31
display(pd.DataFrame(DATA)[['riddle_id','question','answer','difficulty','difficulty_confidence','difficulty_reason']])
print('模型配置：',CONFIGS)
for g,p in PROMPTS.items(): print(g,len(p),'字符：',p)
print('B/C长度近似匹配，但字符数不等于各模型token数。')
for i,name in enumerate(CONFIGS):
    print(name,pd.Series([o for r,g,p,o in schedule(DATA,i) if p==1]).value_counts().to_dict())
''')
md('''## 控制变量与局限

每次请求独立，仅包含同一system和当前题目的条件提示，没有历史回答或工具。
三组相同模型、温度0、关闭深度思考、512 token上限、只输出一字；不提供答案、难度或题型。
B优先字形且按线索选择音义；C为相近篇幅、相同输出要求的通用检查。文字内容是有意改变的变量，不能声称所有其他认知因素完全相同。
六种组顺序各出现5或6次，题序固定随机化，额外一次顺序跨模型轮换。
31题不能六等分；服务器负载与模型别名版本变化也无法完全控制。耗时包含限流，不做思考时间回归。
512 token上限减少截断风险，但与旧实验不同；不要将新旧轮变化直接视为提示的因果效应。
''')
md('## 输入密钥（不联网、不回显）')
code('''KEYS={}
with warnings.catch_warnings():
    warnings.simplefilter('error',getpass.GetPassWarning)
    for name in CONFIGS:
        value=getpass.getpass(name+' API Key：').strip()
        if not value: raise ValueError(name+' 密钥为空')
        KEYS[name]=value
value=None
print('密钥仅保存在内核内存。')
''')
md('## 预检（3次非实验题请求）')
code('''PREFLIGHT={}
for name,config in CONFIGS.items():
    result=smoke_test(config,KEYS[name],RPM)
    PREFLIGHT[name]=result
    print(name,config['model'],result.get('status'),result.get('http_status'),result.get('response',''),result.get('error_body',''))
''')
md('## 正式复测（运行此格开始279次调用）\n\n结果逐条落盘。中断后保持同一配置、重新输入密钥和预检，可续跑；已有失败记录不自动重跑。')
code('''for name,config in CONFIGS.items():
    check=PREFLIGHT.get(name,{})
    if check.get('status')!='ok' or check.get('response','').strip()!='好' or check.get('config_fingerprint')!=config_fingerprint(config):
        raise RuntimeError(name+' 尚未通过当前配置预检')
run(DATA,CONFIGS,KEYS,OUTPUT,RPM)
''')
md('''## 完成后分析

精确双侧McNemar检验，B-C为主比较，跨三模型Holm校正；B-A/C-A辅助比较独立校正。
主口径保留ABC均有效的配对，敏感性口径将失败视为未完成任务；两种口径都报告。
输出配对差异及bootstrap区间、正确率柱图、难度描述表和Markdown分析。难度分层不作显著性筛选。
31题且正确事件可能稀少，不强行做多控制变量Logistic/混合模型。A组是必要的直接作答基线，保留。
''')
code('''rates,tests=analyze(DATA,CONFIGS,OUTPUT)
display(rates)
display(tests)
display(Image(filename=str(OUTPUT/'accuracy.png')))
print('报告：',OUTPUT/'探索性分析.md')
''')
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells):cell['id']=f'pilot31-{i:02d}'
path=ROOT/'31题_ABC探索性小测试.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print(path)

if __name__=='__main__':
    for cell in cells:
        if cell['cell_type']=='code':compile(cell['source'],'pilot31-cell','exec')
