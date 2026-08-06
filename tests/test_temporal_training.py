import unittest

import numpy as np
import torch

from scripts.train_temporal_head import _embedding, prepare_frozen_encoder
from xh202615.temporal_training import (
    binary_metrics,
    build_pair_features,
    select_presence_threshold,
)


class TemporalTrainingHelpersTest(unittest.TestCase):
    def test_encoder_without_eval_method_is_still_accepted(self):
        encoder = object()
        self.assertIs(prepare_frozen_encoder(encoder), encoder)

    def test_encoder_moves_to_requested_device_when_supported(self):
        class DeviceAwareEncoder:
            def __init__(self):
                self.requested_device = None

            def set_device(self, device):
                self.requested_device = device

        encoder = DeviceAwareEncoder()
        self.assertIs(prepare_frozen_encoder(encoder, device="cuda"), encoder)
        self.assertEqual(encoder.requested_device, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this integration unit")
    def test_gpu_encoder_extracts_features_on_its_model_device(self):
        class FakeSpeaker:
            def __init__(self):
                self.device = torch.device("cuda")
                self.model = torch.nn.Identity().to(self.device)

            def compute_features(self, pcm, sample_rate, cmn):
                self.assert_pcm_device = pcm.device.type
                self.assert_sample_rate = sample_rate
                self.assert_cmn = cmn
                return torch.zeros(1, 256)

            def extract_embedding_from_pcm(self, pcm, sample_rate):
                raise AssertionError("GPU path must not call the CPU-only wrapper")

        embedding = _embedding(FakeSpeaker(), np.zeros(3200, dtype=np.float32), 16_000)
        self.assertEqual(embedding.shape, (256,))

    def test_build_pair_features_contains_embedding_delta_similarity_and_energy(self):
        enrollment = np.array([1.0, 0.0], dtype=np.float32)
        windows = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        energy = np.array([0.25, 0.75], dtype=np.float32)
        features = build_pair_features(enrollment, windows, energy)
        self.assertEqual(features.shape, (2, 8))
        self.assertAlmostEqual(float(features[0, 6]), 1.0)
        self.assertAlmostEqual(float(features[1, 6]), 0.0)
        self.assertTrue(np.isfinite(features).all())

    def test_binary_metrics_reports_accuracy_precision_recall_and_f1(self):
        result = binary_metrics(
            np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.8]), threshold=0.5
        )
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_threshold_selection_maximizes_specificity_with_required_recall(self):
        threshold = select_presence_threshold(
            np.array([1, 1, 0, 0]),
            np.array([0.2, 0.9, 0.1, 0.8]),
            min_recall=0.5,
        )
        self.assertAlmostEqual(threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
