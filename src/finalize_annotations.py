"""Build auditable AI annotations from the individually reviewed 300-row ledger."""
import csv
import hashlib
import json
from collections import Counter
from common import ROOT, load_data, normalize_type

data = load_data(ROOT / 'data/test.json')
ledger = ROOT / 'data/annotation_review.tsv'
with ledger.open(encoding='utf-8', newline='') as f:
    reviewed = list(csv.DictReader(f, delimiter='\t'))
assert len(reviewed) == len(data) == 300
corrections = {14: ('g', 'medium', '军借兵，宵去小留宀与月；月配宀兵组成膑，不把膑误写为月加兵。'),
 25: ('g', 'medium', '一点、大与口为构形提示；狗叫未能验证为答案读音，故不标音。'),
 33: ('g', 'low', '木与尊为樽；村上取木，鸡鸣如何引出尊未核实，不强加音标签。'),
 35: ('g', 'high', '第二次借又，握手相见借见，组成观；词义用于取部件，不是答案字义线索。'),
 80: ('g', 'high', '公与讠组成讼；称颂取言旁，不直接描述讼的本义。'),
 82: ('g', 'medium', '古借昔，币借金，钅昔构错；语义替换服务于部件，不单独标义。'),
 120: ('u', 'low', '二度梅与馥的构形或读音关系未能可靠解释，保留未知。'),
 125: ('g', 'low', '怕借畏、火为偏旁，火畏构煨；并非直接解释煨的字义，墙上生待核。'),
 145: ('g', 'high', '绿色借青，道借言说，讠青成请；以词义取部件，不直接标答案字义。'),
 180: ('g', 'high', '妇借女，病借疾，女疾构嫉；疾病不是嫉的本义。'),
 189: ('g', 'high', '想借心旁，想不开借困，忄困构悃；不直接描述悃的字义。'),
 266: ('g', 'high', '说话借云，配力成动；云为说义的部件替换，不需要读音线索。'),
 248: ('g|p', 'high', '莫加饣成馍，同时利用莫mò与馍mó声韵相同但声调不同的近音关系，不称完全同音。')}
mapping = {'g': 'glyph', 'm': 'meaning', 'p': 'pronunciation', 'u': 'unknown'}
rows = []
for i, (r, a) in enumerate(zip(data, reviewed)):
    assert int(a['index']) == i
    labels, confidence, rationale = corrections.get(i, (a['labels'], a['confidence'], a['rationale']))
    if confidence == 'high' and any(t in rationale for t in ['待核', '需核', '需复核', '未完全', '未完整', '歧义']):
        confidence = 'medium'
    rows.append({**r, 'type': normalize_type([mapping[t] for t in labels.split('|')]) if labels != 'u' else 'unknown',
        'confidence': confidence, 'rationale': rationale, 'annotator': 'AI assistant; not independently human-verified',
        'review_status': 'AI initial annotation; uncertainty retained', 'annotation_date': '2026-10-09'})
out = ROOT / 'data/type_annotations.csv'
with out.open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with (ROOT / 'data/annotations_needing_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(r for r in rows if r['confidence'] == 'low' or r['type'] == 'unknown')
manifest = {'n': len(rows), 'type_counts': dict(Counter(r['type'] for r in rows)),
    'confidence_counts': dict(Counter(r['confidence'] for r in rows)),
    'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'annotation_method': 'AI per-item review of question and gold answer, followed by a self-review of semantic-vs-component labels; no experimental model outputs used',
    'human_verified': False, 'unknown_policy': 'Retain unknown; never impute a confidently asserted mechanism',
    'source_ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest()}
(ROOT / 'data/annotation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
