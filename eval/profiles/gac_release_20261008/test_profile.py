"""Synthetic parser/safety/aggregation tests; no data download, GPU or code exec."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import bbh_choice_protocol
import final_code_protocol
from code_protocol import normalize_candidate
from protocol import DATA_FINGERPRINT, GENERATION, SIZES, aggregate, fingerprint
from reasoning import extract_letter, object_answer
from science_answer import score_science


class ProfileTests(unittest.TestCase):
    def test_final_code_cases(self):
        final_code_protocol.self_test()

    def test_bbh_choice_cases(self):
        bbh_choice_protocol.self_test()

    def test_mbpp_uses_tested_entrypoint_not_first_helper(self):
        reference = 'def helper(x):\n    return 987654\ndef solve(x):\n    return helper(x)\n'
        program, mode = normalize_candidate('mbpp', 'return x + 1', {
            'code': reference, 'test_list': ['assert solve(2) == 3']})
        self.assertEqual(mode, 'body_wrapped')
        self.assertIn('def solve(x):', program)
        self.assertNotIn('987654', program)
        self.assertNotIn('def helper', program)

    def test_humaneval_preserves_prompt_imports_not_reference_solution(self):
        program, mode = normalize_candidate('humaneval', 'def solve(x):\n    return x', {
            'raw_prompt': 'from typing import List\ndef solve(x: List[int]):\n    """Identity."""\n',
            'entry_point': 'solve', 'canonical_solution': 'return 987654'})
        self.assertEqual(mode, 'full_function_with_context')
        self.assertIn('from typing import List', program)
        self.assertNotIn('987654', program)

    def test_mcq_and_object_markers(self):
        self.assertEqual(extract_letter(r'Reasoning. \boxed{C}'), 'C')
        self.assertEqual(object_answer('The answer is 3.\n\\boxed{4}'), '4')
        self.assertEqual(object_answer('\\boxed{3}\nThe answer is 4.'), '4')

    def test_science_unit_scaling_and_fail_closed(self):
        self.assertTrue(score_science(r'\boxed{1.0\text{ km}}', '1000', 'm').correct)
        self.assertFalse(score_science(r'\boxed{1000\text{ s}}', '1000', 'm').correct)
        self.assertFalse(score_science(r"\boxed{__import__('os').system('echo bad')}", '1').correct)
        self.assertFalse(score_science('No final numeric answer', '1').correct)

    def test_fingerprint_ignores_identifier_not_question_or_order(self):
        item = {'benchmark': 'amc', 'id': 'local_id', 'eval_item': {'prompt': 'p', 'gold': '2'}}
        clone = {**item, 'id': 'portable_id'}
        self.assertEqual(fingerprint([item]), fingerprint([clone]))
        other = {**item, 'eval_item': {'prompt': 'different', 'gold': '2'}}
        self.assertNotEqual(fingerprint([item]), fingerprint([other]))
        self.assertNotEqual(fingerprint([item, other]), fingerprint([other, item]))

    def test_counts_and_macro_match_release_metadata(self):
        root = Path(__file__).resolve().parents[3]
        release = json.loads((root / 'model_release/gac_qwen35_4b/EVAL_RESULTS.json').read_text())
        expected = {m['key']: m for m in release['metrics']}
        rows = []
        for task, n in SIZES.items():
            for seed, correct in enumerate(expected[task]['correct_counts']):
                rows.extend({'benchmark': task, 'id': str(i), 'mode': f'seed_{seed}',
                             'correct': i < correct} for i in range(n))
        result = aggregate(rows)['metrics']
        for task, metric in result.items():
            self.assertAlmostEqual(metric['mean_percentage'], expected[task]['mean_percentage'])
            self.assertEqual(metric['display_percentage'], expected[task]['display_percentage'])
        self.assertAlmostEqual(result['bbh_macro']['mean_percentage'], 92.6074074074074)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        with self.assertRaises(ValueError):
            aggregate(rows + [rows[-1]])
        self.assertEqual(sum(SIZES.values()), 4447)
        self.assertEqual(release['evaluation_protocol']['portable_data_fingerprint'], DATA_FINGERPRINT)
        self.assertEqual(release['evaluation_protocol']['max_new_tokens'], GENERATION['max_new_tokens'])


if __name__ == '__main__':
    unittest.main()
