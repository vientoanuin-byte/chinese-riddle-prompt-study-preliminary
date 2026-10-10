"""Offline verification. All generated responses here are synthetic, never empirical evidence."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from common import ROOT, build_messages, extract_answer, normalize_type, balanced_schedule, GUIDANCE, CONTROL
from run_experiment import ChatClient, FIELDS
from statistical_test import analyze, read_results

class PipelineTests(unittest.TestCase):
    def test_balanced_three_conditions(self):
        from collections import Counter
        data = [{'riddle_id': str(i)} for i in range(300)]
        schedule = balanced_schedule(data)
        self.assertEqual(len(schedule), 900)
        self.assertEqual(set(Counter(o for _, _, pos, o in schedule if pos == 1).values()), {50})
        self.assertEqual(len({(r['riddle_id'], g) for r, g, _, _ in schedule}), 900)
        self.assertEqual(schedule, balanced_schedule(data))
        self.assertEqual(len(GUIDANCE), len(CONTROL))

    def test_primary_pair_ignores_missing_A(self):
        import pandas as pd
        from statistical_abc import pair_result
        records = []
        for i in range(8):
            b, c = ((1, 0) if i < 5 else (0, 1) if i == 5 else (1, 1))
            for group, value in [('A', np.nan), ('B', b), ('C', c)]:
                records.append({'riddle_id': str(i), 'group': group, 'correct': value})
        result, _ = pair_result(pd.DataFrame(records), 'B', 'C')
        self.assertEqual(result['n_pairs'], 8)
        self.assertAlmostEqual(result['difference'], .5)
        self.assertAlmostEqual(result['p_exact'], .21875)

    def test_extraction_and_multilabel(self):
        self.assertEqual(extract_answer('解释：日加一。答案：旦。'), '旦')
        self.assertEqual(extract_answer('abc'), '')
        self.assertEqual(extract_answer('𠮷'), '𠮷')
        self.assertEqual(normalize_type(['meaning', 'glyph']), 'glyph|meaning')

    def test_prompt_has_no_metadata(self):
        text = json.dumps(build_messages('天上明月光', 'B'), ensure_ascii=False)
        self.assertNotIn('旦', text)
        self.assertNotIn('component_count', text)
        self.assertNotEqual(build_messages('谜面', 'A'), build_messages('谜面', 'B'))

    @patch('run_experiment.time.sleep')
    def test_retry_429(self, sleep):
        client = ChatClient('https://example.invalid/v1', 'TEST_ONLY', 20, 2, 10)
        bad = Mock(status_code=429, headers={'Retry-After': '5'})
        good = Mock(status_code=200)
        good.json.return_value = {'choices': [{'message': {'content': '旦'}, 'finish_reason': 'stop'}]}
        client.session.post = Mock(side_effect=[bad, good])
        result = client.call({})
        self.assertEqual((result['status'], result['attempts']), ('ok', 2))
        self.assertTrue(any(call.args[0] >= 5 for call in sleep.call_args_list))

    @patch('run_experiment.time.sleep')
    def test_no_retry_auth_error(self, sleep):
        client = ChatClient('https://example.invalid/v1', 'TEST_ONLY', 20, 4, 10)
        client.session.post = Mock(return_value=Mock(status_code=401))
        self.assertEqual(client.call({})['attempts'], 1)

    def test_paired_analysis_and_models(self):
        rng = np.random.default_rng(19)
        rows = []
        for i in range(120):
            kind = ['glyph', 'meaning', 'pronunciation', 'glyph|meaning'][i % 4]
            latent = rng.normal(0, .6)
            for group in ['A', 'B']:
                correct = int(rng.random() < 1 / (1 + np.exp(-(-.7 + .5 * (group == 'B') + latent))))
                response = '旦' if correct else '日'
                rows.append(dict(run_id='SYNTHETIC_TEST_ONLY', riddle_id=str(i), group=group,
                    question=f'SYNTHETIC {i}', answer='旦', type=kind, source='human' if i % 3 else 'gen',
                    component_count=2 + i % 3, radical_count=3 + i % 3, response=response,
                    predicted_answer=response, correct=correct, status='ok', model_requested='FAKE'))
        # Failed B response must remove both observations of that riddle.
        rows[-1].update(status='api_error', response='', predicted_answer='', correct='')
        with tempfile.TemporaryDirectory(prefix='cc_riddle_OFFLINE_', dir=ROOT) as td:
            path = Path(td) / 'SYNTHETIC.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(path, Path(td) / 'analysis')
            self.assertEqual(result['n_complete_pairs'], 119)
            self.assertEqual(result['n_excluded_riddles'], 1)
            self.assertTrue((Path(td) / 'analysis/logistic_coefficients.csv').exists())
            self.assertTrue((Path(td) / 'analysis/glmm_coefficients.csv').exists())
            self.assertEqual(len(list((Path(td) / 'analysis/figures').glob('*.png'))), 3)
            rows.append(rows[0])
            with path.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            with self.assertRaises(ValueError):
                read_results(path)

if __name__ == '__main__':
    unittest.main()
