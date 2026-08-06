import unittest

import torch

from xh202615.temporal_head import TemporalSpeakerHead


class TemporalSpeakerHeadTest(unittest.TestCase):
    def test_gru_head_returns_presence_and_overlap_logits(self):
        model = TemporalSpeakerHead(input_dim=8, hidden_dim=4, mode="gru")
        features = torch.randn(3, 5, 8)
        presence, overlap = model(features)
        self.assertEqual(tuple(presence.shape), (3, 1))
        self.assertEqual(tuple(overlap.shape), (3, 1))

    def test_mlp_head_accepts_single_step_sequences(self):
        model = TemporalSpeakerHead(input_dim=8, hidden_dim=4, mode="mlp")
        features = torch.randn(2, 1, 8)
        presence, overlap = model(features)
        self.assertEqual(tuple(presence.shape), (2, 1))
        self.assertEqual(tuple(overlap.shape), (2, 1))

    def test_fused_head_combines_temporal_and_summary_features(self):
        model = TemporalSpeakerHead(input_dim=8, hidden_dim=4, mode="fused")
        features = torch.randn(3, 5, 8)
        presence, overlap = model(features)
        self.assertEqual(tuple(presence.shape), (3, 1))
        self.assertEqual(tuple(overlap.shape), (3, 1))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            TemporalSpeakerHead(input_dim=8, hidden_dim=4, mode="bad")


if __name__ == "__main__":
    unittest.main()
