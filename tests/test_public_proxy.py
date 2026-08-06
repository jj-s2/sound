import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_public_presence_proxy import (
    gated_text,
    parse_args as evaluate_parse_args,
)
from scripts.train_public_presence_proxy import (
    cache_metadata,
    parse_args,
    train,
)
from xh202615.public_proxy import (
    aggregate_window_evidence,
    brier_score,
    bucketed_presence_proxy_metrics,
    build_public_proxy_features,
    expected_calibration_error,
    GlobalPresenceCalibrator,
    presence_proxy_metrics,
    select_public_validation_threshold,
)
from xh202615.training_data import TrainingManifestRow


class TestAggregateWindowEvidence(unittest.TestCase):
    def test_aggregate_window_evidence(self):
        similarity = [0.1, 0.9, 0.8, 0.2]
        log_energy = [0.0, 0.2, 0.4, 0.1]
        result = aggregate_window_evidence(similarity, log_energy)

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (10,))

        np.testing.assert_allclose(result[0], 0.5, atol=1e-5)
        np.testing.assert_allclose(result[1], 0.9, atol=1e-5)
        np.testing.assert_allclose(result[2], 0.1, atol=1e-5)
        np.testing.assert_allclose(result[4], 0.85, atol=1e-5)
        np.testing.assert_allclose(result[5], 0.5, atol=1e-5)
        np.testing.assert_allclose(result[6], 2, atol=1e-5)
        np.testing.assert_allclose(result[7], 0.175, atol=1e-5)
        np.testing.assert_allclose(result[8], 0.4, atol=1e-5)


class TestCalibrationScores(unittest.TestCase):
    def test_brier_and_ece(self):
        self.assertEqual(brier_score([1, 0], [0.75, 0.25]), 0.0625)

        ece = expected_calibration_error([1, 0], [0.75, 0.25])
        self.assertTrue(np.isfinite(ece))
        self.assertGreaterEqual(ece, 0)
        self.assertAlmostEqual(ece, 0.25, places=7)

        with self.assertRaises(ValueError):
            brier_score([1, 0], [1.2, 0.25])


class TestPresenceProxyPolicy(unittest.TestCase):
    def test_presence_proxy_metrics(self):
        metrics = presence_proxy_metrics([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], 0.85)
        self.assertAlmostEqual(metrics["false_reject_rate"], 0.5, places=7)
        self.assertAlmostEqual(metrics["reject_accuracy"], 1.0, places=7)
        self.assertAlmostEqual(metrics["false_accept_rate"], 0.0, places=7)
        self.assertAlmostEqual(metrics["target_accept_rate"], 0.5, places=7)
        self.assertAlmostEqual(metrics["presence_proxy_utility"], 0.75, places=7)

    def test_bucketed_presence_proxy_metrics(self):
        bucketed = bucketed_presence_proxy_metrics(
            [1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2], ["easy", "hard", "easy", "hard"], 0.85
        )
        self.assertEqual(set(bucketed), {"easy", "hard"})
        for value in bucketed.values():
            self.assertTrue(np.isfinite(value["presence_proxy_utility"]))

    def test_select_public_validation_threshold(self):
        selector = select_public_validation_threshold(
            [1, 1, 0, 0], [0.95, 0.85, 0.2, 0.1], ["easy", "hard", "easy", "hard"]
        )
        self.assertEqual(selector["threshold_source"], "public_validation")
        self.assertAlmostEqual(selector["threshold"], 0.85, places=7)
        self.assertLessEqual(selector["metrics"]["false_reject_rate"], 0.10)
        self.assertGreaterEqual(selector["metrics"]["reject_accuracy"], 0.85)
        self.assertTrue(np.isfinite(selector["worst_bucket_utility"]))

    def test_select_public_validation_threshold_raises(self):
        with self.assertRaises(ValueError):
            select_public_validation_threshold(
                [1, 1, 0, 0],
                [0.8, 0.7, 0.9, 0.85],
                ["easy", "hard", "easy", "hard"],
                max_false_reject_rate=0.10,
                min_reject_accuracy=0.85,
            )


class TestThresholdValidation(unittest.TestCase):
    def test_presence_proxy_metrics_nan_threshold_raises(self):
        with self.assertRaises(ValueError):
            presence_proxy_metrics([1, 0], [0.9, 0.1], float("nan"))

    def test_presence_proxy_metrics_out_of_range_threshold_raises(self):
        with self.assertRaises(ValueError):
            presence_proxy_metrics([1, 0], [0.9, 0.1], 1.1)

    def test_select_public_validation_threshold_nan_max_frr_raises(self):
        with self.assertRaisesRegex(ValueError, "max_false_reject_rate"):
            select_public_validation_threshold(
                [1, 1, 0, 0],
                [0.95, 0.85, 0.2, 0.1],
                ["easy", "hard", "easy", "hard"],
                max_false_reject_rate=float("nan"),
            )


class PublicProxyFeatureTest(unittest.TestCase):
    def test_build_public_proxy_features(self):
        global_features, frame_features = build_public_proxy_features(
            [1.0, 0.0], [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [0.1, 0.2]
        )

        self.assertEqual(tuple(global_features.shape), (10,))
        self.assertEqual(tuple(frame_features.shape), (2, 2))
        np.testing.assert_allclose(frame_features[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(frame_features[1, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(frame_features[:, 1], [0.1, 0.2], atol=1e-6)

    def test_build_public_proxy_features_rejects_zero_norm_enrollment(self):
        with self.assertRaises(ValueError):
            build_public_proxy_features([0.0, 0.0], [1.0, 0.0], [[1.0, 0.0]], [0.1])

    def test_global_presence_calibrator(self):
        calibrator = GlobalPresenceCalibrator(input_dim=10)
        logits = calibrator(torch.zeros(3, 10))
        self.assertEqual(tuple(logits.shape), (3, 1))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_global_presence_calibrator_rejects_zero_input_dim(self):
        with self.assertRaises(ValueError):
            GlobalPresenceCalibrator(input_dim=0)


class PublicProxyTrainingBoundaryTest(unittest.TestCase):
    def test_train_rejects_manifest_audio_under_dataset_a(self):
        with tempfile.TemporaryDirectory() as temp_root_str:
            temp_root = Path(temp_root_str)
            dataset_a_root = temp_root / "datasetA"
            dataset_a_root.mkdir()
            manifest_path = temp_root / "manifest.jsonl"
            output_dir = temp_root / "out"

            row = TrainingManifestRow(
                row_id="row-1",
                split="train",
                source="datasetA",
                enrollment_audio=dataset_a_root / "enrollment.wav",
                target_audio=dataset_a_root / "target.wav",
                mixture_audio=dataset_a_root / "mixture.wav",
                target_speaker_id="spk-target",
                interferer_speaker_id="spk-interferer",
                target_present=True,
                overlap_ratio=0.5,
                snr_db=10.0,
                sir_db=5.0,
                text="hello world",
                seed=42,
            )
            with manifest_path.open("w") as manifest_file:
                manifest_file.write(json.dumps(row.to_dict()) + "\n")

            args = parse_args(
                [
                    "--manifest", str(manifest_path),
                    "--output-dir", str(output_dir),
                    "--dataset-a-root", str(dataset_a_root),
                ]
            )
            with self.assertRaisesRegex(ValueError, "forbidden_path"):
                train(args)

    def test_cache_metadata_includes_all_feature_identity_inputs(self):
        rows = [
            SimpleNamespace(row_id="row-1", mixture_audio="audio/mixture_1.wav"),
            SimpleNamespace(row_id="row-2", mixture_audio="audio/mixture_2.wav"),
        ]
        metadata = cache_metadata(
            rows,
            model_name="chinese",
            audio_field="mixture_audio",
            window_seconds=1.0,
            hop_seconds=0.5,
        )
        self.assertEqual(metadata["model_name"], "chinese")
        self.assertEqual(metadata["audio_field"], "mixture_audio")
        self.assertEqual(metadata["window_seconds"], 1.0)
        self.assertEqual(metadata["hop_seconds"], 0.5)
        self.assertTrue(metadata["manifest_id"])

    def test_train_global_fails_when_no_eligible_threshold(self):
        import scripts.train_public_presence_proxy
        from unittest.mock import patch

        rows = [
            SimpleNamespace(
                row_id=f"r{i}",
                split=split,
                target_present=target_present,
                snr_db=10.0,
                sir_db=5.0,
                overlap_ratio=0.5,
            )
            for i, (split, target_present) in enumerate(
                [
                    ("train", True),
                    ("train", False),
                    ("val", True),
                    ("val", False),
                    ("test", True),
                    ("test", False),
                ]
            )
        ]
        global_features = torch.zeros(6, 10)
        targets = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32
        ).unsqueeze(1)
        args = SimpleNamespace(
            lr=1e-3,
            weight_decay=1e-4,
            epochs=1,
            max_false_reject_rate=0.10,
            min_reject_accuracy=0.85,
        )
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(
                scripts.train_public_presence_proxy,
                "_select_on_val",
                side_effect=ValueError("no eligible public validation threshold"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "no eligible public validation threshold"
                ):
                    scripts.train_public_presence_proxy._train_global(
                        global_features,
                        targets,
                        train_idx=[0, 1],
                        val_idx=[2, 3],
                        test_idx=[4, 5],
                        rows=rows,
                        args=args,
                        device=device,
                        output_dir=Path(output_dir),
                    )

    def test_bucket_key_is_shared_snr_tier(self):
        import scripts.train_public_presence_proxy

        present_row = SimpleNamespace(
            row_id="present",
            split="val",
            target_present=True,
            snr_db=10.0,
            sir_db=5.0,
            overlap_ratio=0.5,
        )
        absent_row = SimpleNamespace(
            row_id="absent",
            split="val",
            target_present=False,
            snr_db=10.0,
            sir_db=None,
            overlap_ratio=0.0,
        )
        bucket_key = scripts.train_public_presence_proxy._bucket_key
        self.assertEqual(bucket_key(present_row), "snr_med")
        self.assertEqual(bucket_key(absent_row), "snr_med")

    def test_train_continues_to_frame_when_global_has_no_eligible_threshold(self):
        import scripts.train_public_presence_proxy as tppp
        from contextlib import ExitStack
        from unittest.mock import patch

        rows = [
            SimpleNamespace(row_id="r0", split="train", snr_db=10.0),
            SimpleNamespace(row_id="r1", split="train", snr_db=10.0),
            SimpleNamespace(row_id="r2", split="val", snr_db=10.0),
            SimpleNamespace(row_id="r3", split="val", snr_db=10.0),
            SimpleNamespace(row_id="r4", split="test", snr_db=10.0),
            SimpleNamespace(row_id="r5", split="test", snr_db=10.0),
        ]
        global_features = torch.zeros(6, 10)
        frame_features = [torch.zeros(3, 2) for _ in range(6)]
        targets = torch.tensor(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32
        ).unsqueeze(1)
        row_ids = ("r0", "r1", "r2", "r3", "r4", "r5")
        frame_report = {
            "selected_threshold": 0.7,
            "test": {
                "false_reject_rate": 0.0,
                "reject_accuracy": 1.0,
                "false_accept_rate": 0.0,
                "target_accept_rate": 1.0,
                "presence_proxy_utility": 0.8,
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            args = parse_args(
                [
                    "--manifest", str(Path(output_dir) / "manifest.jsonl"),
                    "--output-dir", output_dir,
                    "--dataset-a-root", str(Path(output_dir) / "datasetA"),
                    "--candidate", "all",
                ]
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(tppp, "read_training_manifest", return_value=rows)
                )
                stack.enter_context(
                    patch.object(
                        tppp, "assert_valid_training_manifest", return_value=rows
                    )
                )
                stack.enter_context(
                    patch.object(
                        tppp,
                        "_load_or_build_feature_cache",
                        return_value=(global_features, frame_features, targets, row_ids),
                    )
                )
                stack.enter_context(
                    patch.object(
                        tppp,
                        "_train_global",
                        side_effect=ValueError(
                            "no eligible public validation threshold"
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(tppp, "_train_frame", return_value=frame_report)
                )
                summary = tppp.train(args)

            self.assertEqual(
                summary["candidate_reports"]["global"]["status"],
                "no_eligible_public_validation_threshold",
            )
            self.assertIn("frame", summary["candidate_reports"])
            self.assertEqual(
                summary["candidate_reports"]["frame"]["selected_threshold"], 0.7
            )
            self.assertEqual(summary["selected_thresholds"]["frame"], 0.7)


class PublicPresenceProxyEvaluationTest(unittest.TestCase):
    def test_gated_text_branches(self):
        self.assertEqual(gated_text(0.9, "hello world", threshold=0.85), "hello world")
        self.assertEqual(gated_text(0.85, "hello world", threshold=0.85), "hello world")
        self.assertEqual(gated_text(0.5, "hello world", threshold=0.85), "")

    def test_parse_args_defaults(self):
        args = evaluate_parse_args(
            [
                "--checkpoint", "ckpt.pt",
                "--asr-predictions", "asr.jsonl",
                "--dataset-a-root", "datasetA",
                "--output", "out",
            ]
        )
        self.assertIsNone(args.device)
        self.assertEqual(args.model, "chinese")


if __name__ == "__main__":
    unittest.main()
