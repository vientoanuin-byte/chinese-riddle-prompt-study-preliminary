"""Freeze a reproducible sample and derive measured component counts."""
import argparse
import csv
import json
import random
from pathlib import Path
from common import ROOT, load_data, digest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT.parent / 'data/test_random.jsonl')
    p.add_argument('--decomposition', type=Path, default=ROOT.parent / 'data/decomposition.jsonl')
    p.add_argument('--sampling', choices=['source', 'simple'], default='source')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=20261009)
    args = p.parse_args()
    folder = ROOT / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'test.json').exists():
        raise SystemExit('data/test.json already exists; preserve the frozen sample or move it explicitly.')
    source = [r for r in load_data(args.source) if r['split'] == 'test']
    if not 0 < args.n <= len(source):
        raise ValueError('Invalid sample size')
    rng = random.Random(args.seed)
    allocation = {}
    if args.sampling == 'source':
        strata = {s: [r for r in source if r['source'] == s] for s in sorted({r['source'] for r in source})}
        quotas = {s: args.n * len(rows) / len(source) for s, rows in strata.items()}
        allocation = {s: int(q) for s, q in quotas.items()}
        for s in sorted(strata, key=lambda s: (-(quotas[s] - allocation[s]), s))[:args.n - sum(allocation.values())]:
            allocation[s] += 1
        sample = [r for s, rows in strata.items() for r in rng.sample(rows, allocation[s])]
        rng.shuffle(sample)
    else:
        sample = rng.sample(source, args.n)
    decomps = {}
    if args.decomposition.exists():
        for line in args.decomposition.read_text(encoding='utf-8-sig').splitlines():
            row = json.loads(line)
            decomps[row['character']] = row
    structure = {}
    for row in sample:
        d = decomps.get(row['answer'], {})
        structure[row['answer']] = {
            'component_count': len(d['once']) if d.get('once') else None,
            'radical_count': len(d['radical']) if d.get('radical') else None,
            'structure_source': 'local decomposition.jsonl: once/radical' if d else 'missing'}
    for name, value in [('test.json', sample), ('structure.json', structure), ('sample_manifest.json', {
        'source': str(args.source.resolve()), 'source_sha256': digest(source), 'population_n': len(source),
        'sample_n': args.n, 'seed': args.seed, 'sampling': args.sampling,
        'source_allocation': allocation,
        'sample_sha256': digest(sample), 'unknown_types': sum(r['type'] == 'unknown' for r in sample)} )]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    with (folder / 'type_annotations.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['riddle_id', 'question', 'answer', 'type', 'source', 'split'])
        w.writeheader()
        w.writerows(sample)
    print(f'Prepared {len(sample)} records. Type annotation remains a human task.')

if __name__ == '__main__':
    main()
