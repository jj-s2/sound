import unittest

import numpy as np

from xh202615.snr import estimate_snr_db, mix_at_snr


class SnrTest(unittest.TestCase):
    def test_mix_at_snr_close_to_target(self):
        rng = np.random.default_rng(1234)
        clean = rng.standard_normal(16000).astype(np.float32) * 0.1
        noise = rng.standard_normal(16000).astype(np.float32)
        mixed = mix_at_snr(clean, noise, 5.0, rng)
        estimated = estimate_snr_db(clean, mixed)
        self.assertAlmostEqual(estimated, 5.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
