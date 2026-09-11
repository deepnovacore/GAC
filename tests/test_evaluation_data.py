"""CPU-only regression checks for fixed subsets and gated dataset handling.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_evaluation_data.py' -v

Requires datasets; no network access, model weights, or vLLM are used.
All question/answer content below is synthetic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import knowledge_bench as kb


class MMLUManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(kb.MMLU_PRO_MANIFEST.read_text())
        self.entries = self.manifest["entries"]
        self.rows = [
            {
                "question_id": entry["question_id"],
                "category": entry["category"],
                "question": "Synthetic test question",
                "options": [f"choice {j}" for j in range(entry["n_options"])],
                "answer_index": entry["n_options"] - 1,
            }
            for entry in self.entries
        ]
        # Non-selected rows test ID-based resolution rather than row positions.
        self.rows.extend(
            {"question_id": -i - 1}
            for i in range(self.manifest["dataset_size"] - len(self.rows))
        )

    def test_variable_options_and_stable_ids_after_row_reordering(self):
        random.Random(9).shuffle(self.rows)
        with patch.object(kb, "load_dataset", return_value=self.rows) as loader:
            items = kb._load_mmlu_pro()
        loader.assert_called_once_with(
            "TIGER-Lab/MMLU-Pro", "default", split="test",
            revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
        )
        self.assertEqual(len(items), 1000)
        self.assertEqual(sum(len(item["options"]) != 10 for item in items), 173)
        self.assertEqual(
            [item["question_id"] for item in items],
            [entry["question_id"] for entry in self.entries],
        )
        for item in items:
            self.assertIn(item["gold"], item["options"])
            self.assertTrue(kb._score_mcq("\\boxed{" + item["gold"] + "}", item["gold"]))

    def test_missing_id_rejected_even_with_correct_split_size(self):
        self.rows[0]["question_id"] = -99999
        with patch.object(kb, "load_dataset", return_value=self.rows):
            with self.assertRaisesRegex(RuntimeError, "missing manifest question_id"):
                kb._load_mmlu_pro()

    def test_duplicate_id_rejected(self):
        self.rows[-1]["question_id"] = self.rows[0]["question_id"]
        with patch.object(kb, "load_dataset", return_value=self.rows):
            with self.assertRaisesRegex(RuntimeError, "duplicate question_id"):
                kb._load_mmlu_pro()

    def test_dataset_drift_and_invalid_gold_rejected(self):
        for update, message in [
            ({"category": "wrong category"}, "differs from manifest"),
            ({"options": ["only one"]}, "expected 3-10 options"),
            ({"answer_index": 10}, "invalid answer_index"),
        ]:
            with self.subTest(update=update):
                changed = [dict(self.rows[0], **update)] + self.rows[1:]
                with patch.object(kb, "load_dataset", return_value=changed):
                    with self.assertRaisesRegex(RuntimeError, message):
                        kb._load_mmlu_pro()

    def test_subset_parameters_cannot_silently_change_protocol(self):
        with patch.object(kb, "load_dataset") as loader:
            with self.assertRaises(ValueError):
                kb._load_mmlu_pro(seed=0)
            with self.assertRaises(ValueError):
                kb._load_mmlu_pro(subset_size=999)
            loader.assert_not_called()

    def test_summary_carries_revision_and_manifest_hash(self):
        cfg = kb.GenerationConfig(model_path="synthetic-test-model", seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(kb, "load_dataset", return_value=self.rows):
                with patch.object(kb, "generate", return_value=[["no answer"]] * 1000):
                    result = kb.run_benchmark("mmlu-pro", cfg.model_path, Path(tmp), cfg)
            saved = json.loads((Path(tmp) / "mmlu-pro" / "summary.json").read_text())
        self.assertEqual(result, saved)
        self.assertEqual(result["dataset_revision"], kb.MMLU_PRO_REVISION)
        self.assertEqual(result["subset_seed"], 42)
        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["subset_manifest_sha256"], hashlib.sha256(kb.MMLU_PRO_MANIFEST.read_bytes()).hexdigest())


class GPQAAccessTests(unittest.TestCase):
    def test_gated_failure_explains_both_supported_access_paths(self):
        with patch.dict("os.environ", {"GAC_GPQA_CSV": ""}):
            with patch.object(kb, "load_dataset", side_effect=OSError("gated access required")):
                with self.assertRaisesRegex(RuntimeError, "hf auth login.*--gpqa_csv"):
                    kb._load_gpqa_diamond()

    def test_complete_csv_uses_no_hf_access_and_records_hash(self):
        columns = ["Question", "Correct Answer"] + [f"Incorrect Answer {j}" for j in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_gpqa.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for i in range(198):
                    writer.writerow(dict(zip(columns, [f"synthetic {i}", "correct", "wrong1", "wrong2", "wrong3"])))
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(kb, "load_dataset") as loader:
                items, source = kb._load_gpqa_diamond(str(path))
                loader.assert_not_called()
            self.assertEqual(len(items), 198)
            self.assertEqual(source, "csv:provided_file:sha256=" + expected_hash)
            for item in items:
                self.assertEqual(item["options"][item["gold"]], "correct")

    def test_partial_csv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_gpqa.csv"
            path.write_text("Question,Correct Answer\nsynthetic,correct\n")
            with self.assertRaisesRegex(RuntimeError, "expected 198 examples"):
                kb._load_gpqa_diamond(str(path))


if __name__ == "__main__":
    unittest.main()
