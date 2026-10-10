"""Publication figures from real CSVs only. No API requests."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from common import ROOT

OUT=ROOT/'paper/figures'
OUT.mkdir(parents=True,exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(str(font))
    plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#FAF9F5',
 'axes.facecolor':'#FAF9F5','text.color':'#22313E','axes.labelcolor':'#22313E','xtick.color':'#61717B',
 'ytick.color':'#61717B','savefig.facecolor':'#FAF9F5','svg.fonttype':'path'})
colors=['#8599AA','#087F8C','#D0A16A']
names=['deepseek','doubao','qwen'];labels=['DeepSeek Flash','豆包 Seed 2.0 Lite','Qwen Plus']

def load(round_name,name):
    p=ROOT/'output'/round_name/(name+'.csv') if round_name=='pilot31_v1' else ROOT/'output'/round_name/name/'experiment_results.csv'
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    assert not any(r.get('model_returned')=='offline_mock' for r in rows)
    assert len(rows)==(93 if round_name=='pilot31_v1' else 900)
    by={}
    for r in rows:by.setdefault(r['riddle_id'],{})[r['group']]=r
    return [v for v in by.values() if all(g in v and v[g]['correct'] in ('0','1') for g in 'ABC')]

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=210)
    plt.close(fig)

def overview(round_name,title,subtitle,name,footer):
    fig,axes=plt.subplots(1,3,figsize=(13.5,5.6))
    fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.70,wspace=.25)
    fig.text(.06,.91,title,fontsize=23,weight='bold')
    fig.text(.06,.83,subtitle,fontsize=11,color='#61717B')
    for i,(provider,ax) in enumerate(zip(names,axes)):
        data=load(round_name,provider);n=len(data)
        counts=[sum(v[g]['correct']=='1' for v in data) for g in 'ABC']
        rates=np.array(counts)/n*100
        ax.bar(range(3),rates,color=colors,width=.56,zorder=3)
        for x,y,k in zip(range(3),rates,counts):
            ax.text(x,y+1.3,f'{y:.1f}%',ha='center',fontsize=16,weight='bold')
            ax.text(x,y+5.3,f'{k}/{n}',ha='center',fontsize=10,color='#61717B')
        ax.set_title(labels[i],loc='left',pad=16,fontsize=13,weight='bold')
        ax.set_xticks(range(3),['A 直接作答','B 结构策略','C 通用检查'],fontsize=10)
        ax.set_ylim(0,55);ax.set_yticks([0,10,20,30,40,50],[f'{x}%' for x in [0,10,20,30,40,50]])
        ax.grid(axis='y',color='#E1E4E4',linewidth=.7,zorder=0)
        ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.text(.06,.055,footer,fontsize=10,color='#61717B')
    save(fig,name)

overview('three_models_ABC_v1','01  综合提示没有带来稳定优势',
 '第一轮 · 每模型300题、同题ABC独立调用 · B要求综合考虑形、音、义',
 '01_initial_results','口径：ABC完整有效配对；DeepSeek / 豆包 n=300，Qwen n=295（5题含截断）。各面板使用相同纵轴。')
overview('pilot31_v1','02  选择性提示出现改善迹象',
 '探索性复测 · 每模型31题 · B改为优先字形，仅在有线索时考虑音义',
 '02_pilot_results','豆包 B−C：+9.68个百分点，精确双侧 p=0.25；Qwen：+6.45个百分点，p=0.50。均未达显著，不能视为独立验证。')

data=load('pilot31_v1','doubao');lookup={v['A']['riddle_id']:v for v in data}
fig=plt.figure(figsize=(13.5,6.7));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
ax.text(.055,.925,'03  从具体字谜看见变化，也看见边界',fontsize=23,weight='bold')
ax.text(.055,.865,'豆包 · 同题独立调用 · 青色表示与标准答案一致，灰色表示不一致',fontsize=11,color='#61717B')
for x,t in zip([.07,.56,.71,.86],['谜面 / 标准答案','A 直接作答','B 结构策略','C 通用检查']):ax.text(x,.775,t,fontsize=11)
for i,rid in enumerate(['25217','15035','5750','12209']):
    v=lookup[rid];y=.65-i*.155
    ax.text(.07,y+.015,v['A']['question'],fontsize=14,weight='bold')
    ax.text(.07,y-.035,'标准答案：'+v['A']['answer'],fontsize=11,color='#61717B')
    for x,g in zip([.55,.70,.85],'ABC'):
        correct=v[g]['correct']=='1'
        ax.add_patch(FancyBboxPatch((x-.015,y-.045),.12,.11,boxstyle='round,pad=0.008,rounding_size=0.012',
                                  linewidth=0,facecolor='#DCEFEA' if correct else '#EAEDEC'))
        ax.text(x+.045,y+.004,v[g]['response'],ha='center',va='center',fontsize=24,color='#087F8C' if correct else '#7A858D')
ax.text(.055,.045,'案例为说明而选取，非随机样本；完整正确数和全部31题记录见正文及附录。答案改善不等于已观察到推理机制。',fontsize=10,color='#61717B')
save(fig,'03_riddle_cases')
print(OUT)
