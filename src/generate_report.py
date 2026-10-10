"""Create a Markdown empirical section only from completed real-result analysis artifacts."""
import json
import math
import os
from pathlib import Path
import pandas as pd

def write_section(comparison, destination, planned_models):
    destination = Path(destination)
    lines = ['## 形音义提示的实证检验', '',
        '### 实验设计', '',
        '本研究从 CC-Riddle 官方测试集固定抽取300道题，随机种子为20261009；按来源近似等比例分层，'
        '人写谜语212道、模型生成谜语88道。抽样规模受API费用、标注核验工作量与课程期限约束，'
        '不意味着更大样本没有统计价值，也未声称已达到预设统计功效。', '',
        '每款模型分别在A（直接作答）、B（形音义策略提示）、C（一般审慎提示）下回答相同题目。'
        'ABC、ACB、BAC、BCA、CAB、CBA六种顺序各分配50题；每次请求不传历史回答，'
        '不使用联网或检索工具，显式关闭可配置的深度思考。B/C尽量匹配长度、三步结构及输出要求，'
        '但不声称不同模型的token长度完全相同。标准谜底、题型标签和答案部件资料均不提供给模型。', '',
        '主要比较为B−C，检验形音义指引相对于一般审慎指引的额外效果；B−A、C−A为辅助比较。'
        '采用精确双侧McNemar检验；主比较在预设模型之间作Holm校正，辅助比较作为另一检验家族校正。'
        '准确率差区间通过按题配对bootstrap（5000次）获得。', '',
        '形音义标签由AI在本次模型输出产生前依据谜面与标准答案逐题初标、自检，并保留多标签、置信度与说明；'
        '不属于官方标签或独立人工金标准。低置信度和未知题不从主检验中删除，另作剔除后的敏感性分析。', '',
        '### 配对比较结果', '',
        '| 模型 | 比较 | 有效配对数 | 左组准确率 | 右组准确率 | 差值（百分点） | 95%配对区间（百分点） | 原始p | Holm p |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|']
    def number(x):
        return f'{float(x):.4g}' if pd.notna(x) else '未计算'
    for _, r in comparison.iterrows():
        if not r.available:
            lines.append(f'| {r.provider} | {r.contrast} | {r.n_pairs} | — | — | — | — | — | — |'); continue
        lines.append(f'| {r.provider} | {r.contrast} | {int(r.n_pairs)} | {r.accuracy_left:.1%} | {r.accuracy_right:.1%} | '
            f'{100*r.difference:.2f} | [{100*r.ci_lower:.2f}, {100*r.ci_upper:.2f}] | {number(r.p_exact)} | {number(r.get("p_holm"))} |')
    lines += ['', '主比较的解释如下：', '']
    for _, r in comparison[comparison.contrast.eq('B-C')].iterrows():
        p = r.get('p_holm', float('nan'))
        if not r.available or pd.isna(p):
            text = '计划结果未全部具备或有效配对不足，暂不作主检验显著性结论。'
        elif p < .05 and r.difference > 0:
            text = '在本样本与指定提示下，B组优于C组，校正后差异达到0.05显著性水平。'
        elif p < .05 and r.difference < 0:
            text = '在本样本与指定提示下，B组低于C组，校正后差异达到0.05显著性水平；结果不支持提升假设。'
        else:
            text = '未获得B、C存在差异的充分统计证据；这不等于证明两组等效或提示无效。'
        lines.append(f'- **{r.provider}**：{text}')
    lines += ['', '### 回归与稳健性分析', '',
        '补充二项Logistic回归以C组为参照，控制形音义指示变量、未知标签、来源、一级部件数量与调用位置，'
        '标准误按题目聚类；另按谜底聚类作敏感性检查。二项GLMM设置riddle_id随机截距，采用变分贝叶斯估计。'
        'GLMM区间为近似后验可信区间，不是频率学派置信区间；两类回归均为探索性补充。', '']
    for _, r in comparison.drop_duplicates('provider').iterrows():
        folder = Path(r.analysis_dir)
        summary = json.loads((folder / 'stats_summary.json').read_text(encoding='utf-8'))
        lines += [f'**{r.provider}（请求型号：`{r.model}`）**', '',
            f'完整三组题数为{summary["n_complete_triples"]}；各对比的有效样本以表中配对数为准。'
            f'运行状态：`{summary["status_counts"]}`。服务端返回型号：`{summary["models_returned"]}`。', '']
        for model in ['logistic', 'glmm']:
            path = folder / f'{model}_coefficients.csv'
            if not path.exists():
                lines.append(f'- {model} 未得到可报告估计，详见统计摘要中的非收敛、分离或样本不足说明。'); continue
            coef = pd.read_csv(path).set_index('term')
            if 'is_B' in coef.index:
                b = coef.loc['is_B']
                kind = '聚类稳健95%置信区间' if model == 'logistic' else '近似95%后验可信区间'
                lines.append(f'- {model} 的B相对C系数为{b.coefficient:.3f}，优势比OR={b.odds_ratio:.3f}，'
                    f'系数的{kind}为[{b.lower:.3f}, {b.upper:.3f}]。OR不是准确率提升倍数。')
        sen = summary['annotation_sensitivity']
        if sen['available']:
            lines.append(f'- 去掉低置信度及未知标签后，B−C保留{sen["n_pairs"]}对，差值为{100*sen["difference"]:.2f}个百分点，'
                f'未校正p={sen["p_exact"]:.4g}；该子集分析改变样本构成，仅作探索性敏感性检查。')
        lines += ['']
        for filename, caption in [('accuracy_ABC.png', '三组准确率（完整三元组）'),
                ('paired_matrices.png', '配对正误转移矩阵'), ('regression_coefficients.png', '回归系数')]:
            relative = os.path.relpath(folder / 'figures' / filename, destination.parent).replace('\\', '/')
            lines += [f'![{r.provider}：{caption}]({relative})', '']
    lines += ['### 解释边界', '',
        '本实验属于提示干预的实证分析，不采用断点回归。结果只能评价这些具体提示、模型和样本，'
        '不能单独证明模型内部真正进行了形音义推理，或证明tokenization是错误的因果来源。'
        '题型分布高度偏向字形，音义标签样本稀少且存在标注误差，不能据此作稳定的细分类效果结论。'
        '每条件每题仅一次生成，无法充分估计服务端随机性；公开数据可能存在训练污染。'
        '失败记录成对排除可能引入选择偏差，需结合失败比例判断。按题bootstrap未消除同谜底相关性，'
        '应结合谜底聚类的敏感性结果。该章节由实际CSV自动生成，提交前应结合原文论证审阅。', '',
        '数据来源：[CC-Riddle](https://github.com/pku0xff/CC-Riddle)，'
        '[论文](https://arxiv.org/abs/2206.13778)。']
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return destination
