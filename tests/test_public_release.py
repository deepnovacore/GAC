"""Keep all current public surfaces on the same measured release snapshot."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_tables_are_synchronized(self):
        release = json.loads((ROOT / 'model_release/gac_qwen35_4b/EVAL_RESULTS.json').read_text())
        documents = [(ROOT / 'README.md').read_text(),
                     (ROOT / 'model_release/gac_qwen35_4b/README.md').read_text()]
        html = (ROOT / 'docs/index.html').read_text()
        for metric in release['metrics']:
            value = f"{metric['display_percentage']:.1f}%"
            for text in documents:
                self.assertRegex(text, re.escape('| ' + metric['benchmark'] + ' |') + r'[^\n]*' + re.escape(value))
            match = re.search(r'<tr data-benchmark="' + re.escape(metric['key']) + r'">(.*?)</tr>', html)
            self.assertIsNotNone(match, metric['key'])
            self.assertIn(value, match[1])
        for text in documents:
            self.assertNotIn('Qwen3.5-4B Base |', text)
            self.assertNotIn('higher on 9 of 11', text)
            self.assertIn('general', text.lower())
        self.assertIn('<summary>Original paper experiments', html)
        self.assertNotIn('HF checkpoints coming soon', html)

    def test_exact_means_are_from_counts(self):
        release = json.loads((ROOT / 'model_release/gac_qwen35_4b/EVAL_RESULTS.json').read_text())
        metrics = {m['key']: m for m in release['metrics']}
        for key, item in metrics.items():
            if key == 'bbh_macro':
                continue
            self.assertEqual(len(item['correct_counts']), 3)
            mean = 100 * sum(item['correct_counts']) / (3 * item['n_examples'])
            self.assertAlmostEqual(mean, item['mean_percentage'])
            self.assertEqual(round(mean, 1), item['display_percentage'])
        macro = sum(metrics[key]['mean_percentage'] for key in
                    ['logical_deduction', 'object_counting', 'tracking_shuffled_objects']) / 3
        self.assertAlmostEqual(macro, metrics['bbh_macro']['mean_percentage'])


if __name__ == '__main__':
    unittest.main()
