"""Compatibility coverage for the EER threshold search; runs with unittest or pytest."""
import tempfile
import unittest
import csv
from pathlib import Path
import tracemalloc

import numpy as np

from src.benchmark.voice_identity_evaluator import VoiceIdentityEvaluator


def reference_eer(same, different):
    """Original exhaustive threshold scan, retained as a numerical oracle."""
    best_difference, eer, threshold = 1.0, 0.5, 0.5
    for candidate in np.linspace(-0.2, 1.0, 500):
        far = np.mean(different >= candidate) if len(different) else 0.0
        frr = np.mean(same < candidate) if len(same) else 0.0
        difference = abs(far - frr)
        if difference < best_difference:
            best_difference = difference
            eer = float(0.5 * (far + frr))
            threshold = float(candidate)
    return eer, threshold


class EERCompatibilityTests(unittest.TestCase):
    def test_pair_order_quoting_and_subclass_override(self):
        class ConstantEvaluator(VoiceIdentityEvaluator):
            @staticmethod
            def cosine_similarity(v1, v2):
                return 0.25

        corpus = {'A,"': {"one": [0.0], "two": [1.0]},
                  "B": {"three": [-1.0]}}
        with tempfile.TemporaryDirectory() as output:
            metrics = ConstantEvaluator.evaluate_corpus(corpus, output)
            with open(Path(output) / "pairwise_similarities.csv", newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual([r["type"] for r in rows],
                         ["same_singer", "diff_singer", "diff_singer"])
        self.assertEqual([r["song_1"] for r in rows], ["one", "one", "two"])
        self.assertTrue(all(r["singer_1"] == 'A,"' for r in rows))
        self.assertTrue(all(r["similarity"] == "0.25" for r in rows))
        self.assertEqual(metrics["same_singer_similarity_mean"], 0.25)

    def test_failed_scoring_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as output:
            path = Path(output) / "pairwise_similarities.csv"
            path.write_text("existing result")
            with self.assertRaises(ValueError):
                VoiceIdentityEvaluator.evaluate_corpus(
                    {"A": {"1": [1.0], "2": [1.0, 2.0]}}, output)
            self.assertEqual(path.read_text(), "existing result")

    def test_corpus_peak_allocation_is_bounded(self):
        corpus = {str(i): {"one": [1.0], "two": [0.5]} for i in range(100)}
        with tempfile.TemporaryDirectory() as output:
            tracemalloc.start()
            try:
                VoiceIdentityEvaluator.evaluate_corpus(corpus, output)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        # 19,900 pairs: catches retaining dictionaries and boxed floats per row.
        self.assertLess(peak, 2_000_000)

    def test_threshold_boundaries_ties_and_nonfinite_scores(self):
        thresholds = np.linspace(-0.2, 1.0, 500)
        cases = [
            ([], []), ([0.0], [0.0]), ([1.0], [-1.0]),
            ([-1.0], [1.0]), ([np.nan], [np.nan]),
            ([np.inf, -np.inf, np.nan, 0.0], [0.0, np.nan, np.inf]),
            (thresholds, thresholds[::-1]),
            (np.nextafter(thresholds, -np.inf),
             np.nextafter(thresholds, np.inf)),
        ]
        for same, different in cases:
            same, different = np.asarray(same), np.asarray(different)
            with self.subTest(same=same.size, different=different.size):
                np.testing.assert_equal(
                    VoiceIdentityEvaluator._equal_error_rate(same, different),
                    reference_eer(same, different),
                )

    def test_random_scores_match_exhaustive_scan_without_mutation(self):
        rng = np.random.default_rng(42)
        for count in (1, 36, 630, 10000, 65537):
            same = rng.uniform(-1.0, 1.0, count)
            different = rng.uniform(-1.0, 1.0, count * 3)
            original = same.copy(), different.copy()
            self.assertEqual(
                VoiceIdentityEvaluator._equal_error_rate(same, different),
                reference_eer(same, different),
            )
            np.testing.assert_array_equal(same, original[0])
            np.testing.assert_array_equal(different, original[1])

    def test_existing_corpus_contract(self):
        corpus = {"A": {"1": [1.0, 0.0], "2": [1.0, 0.0]},
                  "B": {"1": [0.0, 1.0], "2": [0.0, 1.0]}}
        with tempfile.TemporaryDirectory() as output:
            metrics = VoiceIdentityEvaluator.evaluate_corpus(corpus, output)
        self.assertEqual(metrics["same_singer_pairs"], 2)
        self.assertEqual(metrics["different_singer_pairs"], 4)
        self.assertEqual(metrics["equal_error_rate_eer"], 0.0)
        self.assertEqual(metrics["optimal_threshold"], 0.002)


if __name__ == "__main__":
    unittest.main()
