"""Separate exploratory pilot; never modifies the original experiment."""
import csv
import itertools
import json
import os
import random
from pathlib import Path

from common import ROOT, SYSTEM, digest, extract_answer
from run_experiment import ChatClient, make_payload

SEED = 20261010
PROMPTS = {
    'A': '请解答以下单字字谜。',
    'B': '先辨认谜面的明确线索；优先检查部件增删、移动、组合及位置关系；仅在有明确线索时考虑谐音或字义，不强求兼用形音义；最后核对候选是否满足全部条件。只输出一个汉字，不加解释。',
    'C': '先辨认谜面的明确线索；优先检查条件要求、关联、限制及前后关系；仅在有明确依据时考虑其他可能，不强求兼用多方案；最后核对候选是否满足全部条件。只输出一个汉字，不加解释。',
}

# Difficulty is an AI-assisted, pre-pilot judgement, not an empirical difficulty score.
# Do not interpret uncertain explanations as verified solutions.
LABELS = [
 ('4648','低','高','两地提示两个土，相思可能借相的木；部件少。'),
 ('3269','高','低','皇字相关拆分加托孤的隐喻，操作对象不明确，需人工复核。'),
 ('15035','高','中','雪后、堂前与枝的部分提取，多处位置线索合并。'),
 ('25217','低','高','谜面直接出现月与善两个组成部件。'),
 ('15624','高','低','灯谜有格涉及谜格知识及语义转换，拆解不透明。'),
 ('16223','高','中','人名、上下位置及曰形线索并存，需要多步提取。'),
 ('6461','中','中','情的局部提取并与人组合，涉及拆分及替换。'),
 ('17791','中','低','荷叶与包初的取部件线索较隐晦，确切拆解需复核。'),
 ('23429','中','高','先生及个个需要先后位置和竹头变形识别。'),
 ('22722','高','中','远山与雁影涉及象形变形，答案较不熟悉。'),
 ('1954','高','中','芳去草与雁的斜影组合，含隐喻及多步拆分。'),
 ('108','中','高','旧的局部位置改变得到旦，操作对象清楚但变换隐含。'),
 ('3972','高','低','星月及道路转折为象形描述，映射不明确。'),
 ('21602','高','低','枝头、柳条、客心等多处提取，答案较不熟悉。'),
 ('8892','高','低','语义表层强，如何从心满意足映射到音不透明，需复核。'),
 ('3655','中','中','安去女换丁，需要识别裙钗与男装的部件借代。'),
 ('13723','中','高','山、夹的图形关系及两点组合，需要多步位置判断。'),
 ('2585','高','中','叉与头部为象形线索，需确定穿插笔画关系。'),
 ('17982','高','低','山、水、方的象形部件及位置转换，答案较不熟悉。'),
 ('5750','中','低','大人走来存在语义或图形歧义，不能仅按字面拼接。'),
 ('26276','高','低','良与泊船语义及去人线索交织，答案较不熟悉，需复核。'),
 ('17450','中','高','十日对应旬，谈提示言旁，需一个词义中间步骤。'),
 ('15132','中','中','一家五口与衣涉及数量、部件及偏旁辨别。'),
 ('22029','高','中','自与多人部件的位置组合，答案较不熟悉。'),
 ('3323','高','低','旧梦及相会含部件残留与去除隐喻，需复核。'),
 ('112','高','低','五音和女主可能涉及文化典故或戏曲语义，需复核。'),
 ('16750','高','中','雄心抽取及今加点组合，需要多步部件变换。'),
 ('10254','低','高','加去口剩力，再与工组合，部件常见且操作明确。'),
 ('14165','中','中','军功语义引向人持戈，含字义与构形联系。'),
 ('12209','低','高','明确排除右，石头上方增加部件可指向岩。'),
 ('20515','中','中','甚与根基对应的足旁组合，答案较不熟悉。'),
]

def prepare():
    source = json.loads((ROOT / 'data/test.json').read_text(encoding='utf-8-sig'))
    sampled = random.Random(SEED).sample(source, 31)
    assert [r['riddle_id'] for r in sampled] == [x[0] for x in LABELS]
    annotations = {rid: dict(difficulty=d, difficulty_confidence=c, difficulty_reason=why,
                            annotation_status='AI辅助初标，非人工金标准')
                   for rid,d,c,why in LABELS}
    sampled = [{**r, **annotations[r['riddle_id']]} for r in sampled]
    path = ROOT / 'data/pilot31.json'
    path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding='utf-8')
    with (ROOT / 'data/pilot31_annotations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sampled[0]));w.writeheader();w.writerows(sampled)
    return sampled

def messages(question, group):
    return [{'role':'system','content':SYSTEM},
            {'role':'user','content':PROMPTS[group]+'\n谜面：'+json.dumps(question,ensure_ascii=False)}]

def schedule(data, provider_index):
    rng=random.Random(SEED)
    rows=list(data);rng.shuffle(rows)
    orders=list(itertools.permutations('ABC'))
    # 31 cannot split evenly across six orders: counts are 6/5/5/5/5/5.
    assigned=[orders[(i+provider_index*2)%6] for i in range(len(rows))]
    rng.shuffle(assigned)
    return [(r,g,p+1,''.join(order)) for r,order in zip(rows,assigned) for p,g in enumerate(order)]

def run(data, configs, keys, output, rpm=20):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan={'seed':SEED,'data':data,'configs':configs,'provider_order':list(configs),
          'prompts':PROMPTS,'system':SYSTEM,'max_tokens':512,'temperature':0,'rpm':rpm,
          'primary':'B-C; Holm across three providers','design':'exploratory reuse of previously observed 300-item sample'}
    fingerprint=digest(plan)
    manifest=output/'plan.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=plan:
            raise ValueError('本批次方案已冻结且与当前设置不同；请恢复设置或使用新的输出目录。')
    else:manifest.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    fields=['fingerprint','provider','riddle_id','group','position','order','difficulty','question','answer',
            'response','prediction','correct','status','http_status','finish_reason','model_requested',
            'model_returned','attempts','latency_seconds','prompt_tokens','completion_tokens']
    for index,(name,config) in enumerate(configs.items()):
        path=output/(name+'.csv');done=set()
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    pair=(r['riddle_id'],r['group'])
                    if r['fingerprint']!=fingerprint or pair in done:raise ValueError('续跑记录不一致')
                    done.add(pair)
        new=not path.exists()
        client=ChatClient(config['base_url'],keys[name],rpm,retries=2,timeout=90)
        try:
            with path.open('a',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields)
                if new:w.writeheader()
                for row,g,pos,order in schedule(data,index):
                    if (row['riddle_id'],g) in done:continue
                    result=client.call(make_payload(config['model'],messages(row['question'],g),
                         temperature=0,max_tokens=512,token_parameter=config['token_parameter'],extra_body=config['extra_body']))
                    pred=extract_answer(result.get('response',''))
                    record={k:result.get(k,'') for k in fields}
                    record.update(fingerprint=fingerprint,provider=name,riddle_id=row['riddle_id'],group=g,
                        position=pos,order=order,difficulty=row['difficulty'],question=row['question'],answer=row['answer'],
                        prediction=pred,correct=int(pred==row['answer']) if result['status']=='ok' else '',model_requested=config['model'])
                    w.writerow(record);f.flush();os.fsync(f.fileno())
                    print(name,row['riddle_id'],g,result['status'],flush=True)
                    if result.get('http_status') in {400,401,403,404,422}:
                        raise RuntimeError('模型配置或权限错误，已停止；请保留已写记录。')
        finally:client.session.close()

def analyze(data, configs, output):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
    from statsmodels.stats.multitest import multipletests
    import matplotlib.pyplot as plt
    output=Path(output);tables=[];accuracy=[];difficulty=[]
    expected={(r['riddle_id'],g) for r in data for g in 'ABC'}
    for name in configs:
        path=output/(name+'.csv')
        d=pd.read_csv(path,dtype={'riddle_id':str})
        if len(d)!=93 or set(zip(d.riddle_id,d.group))!=expected:
            raise ValueError(name+' 尚未跑齐31题ABC，本次不生成中途显著性结果。')
        d['valid']=d.status.eq('ok') & d.correct.notna()
        wide=d.pivot(index='riddle_id',columns='group',values='correct')
        complete=wide.dropna()
        for g in 'ABC':
            s=d[d.group==g]
            accuracy.append(dict(provider=name,group=g,n=31,valid=int(s.valid.sum()),
                 correct=int(s.correct.fillna(0).sum()),task_success_rate=float(s.correct.fillna(0).mean())))
        for level,part in d.groupby('difficulty'):
            for g,s in part.groupby('group'):
                difficulty.append(dict(provider=name,difficulty=level,group=g,n=len(s),correct=int(s.correct.fillna(0).sum())))
        for policy,w in [('complete_ABC',complete),('failure_as_unsolved',wide.fillna(0))]:
            for a,b in [('B','C'),('B','A'),('C','A')]:
                left=int(((w[a]==1)&(w[b]==0)).sum());right=int(((w[a]==0)&(w[b]==1)).sum())
                delta=(w[a]-w[b]).to_numpy()
                rng=np.random.default_rng(SEED)
                ci=np.quantile(rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1),[.025,.975]) if len(delta) else [np.nan,np.nan]
                tables.append(dict(provider=name,policy=policy,contrast=a+'-'+b,n=len(w),left_only=left,right_only=right,
                    difference=float(delta.mean()) if len(delta) else np.nan,ci_low=ci[0],ci_high=ci[1],
                    p=binomtest(left,left+right).pvalue if left+right else 1.0))
    result=pd.DataFrame(tables)
    for policy in result.policy.unique():
        for primary in [True,False]:
            mask=result.policy.eq(policy)&(result.contrast.eq('B-C') if primary else result.contrast.ne('B-C'))
            result.loc[mask,'p_holm']=multipletests(result.loc[mask,'p'],method='holm')[1]
    rates=pd.DataFrame(accuracy);rates.to_csv(output/'accuracy.csv',index=False,encoding='utf-8-sig')
    result.to_csv(output/'paired_tests.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(difficulty).to_csv(output/'difficulty_descriptive.csv',index=False,encoding='utf-8-sig')
    rates.pivot(index='provider',columns='group',values='task_success_rate').plot.bar(ylim=(0,1),ylabel='Task success rate',rot=0)
    plt.tight_layout();plt.savefig(output/'accuracy.png',dpi=160);plt.close()
    body='# 31题探索性复测\n\n固定种子20261010，从已观察过的300题中随机抽样。不是独立验证集或大样本。\n\n'
    body+='每模型31题，每题ABC共93次；三个模型279次。重复调用不等于279个独立谜题。\n\n'
    body+='主比较B-C，跨模型Holm校正；辅助比较B-A/C-A另作一组校正。两种失败处理口径分别列出。\n\n'
    body+='区间为按题配对bootstrap百分位95%区间，小样本可能不稳定；无差异对时退化区间不代表确定等效。\n\n'
    body+='难度是AI辅助初标；难度分层仅描述，不进行稀疏分层显著性筛选。耗时含限流等待，不代表思考时间。\n\n'
    body+='本轮token上限统一512，旧轮为256；因此新旧轮差异不能全部归因于prompt改变。\n\n'
    body+='```text\n'+rates.to_string(index=False)+'\n```\n\n```text\n'+result.to_string(index=False)+'\n```\n'
    (output/'探索性分析.md').write_text(body,encoding='utf-8')
    return rates,result
