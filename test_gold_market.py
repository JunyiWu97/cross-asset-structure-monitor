import unittest

import numpy as np
import pandas as pd

from gold_market import build_gold_comparison


class GoldMarketTest(unittest.TestCase):
    def test_comparison_calculates_basis_and_direction_confirmation(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        london = np.linspace(100, 120, len(dates))
        comex = london + 2
        futures = pd.DataFrame({"date": dates, "close": comex})
        spot = pd.DataFrame({"date": dates, "close": london})

        comparison, summary = build_gold_comparison(futures, spot)

        self.assertEqual(len(comparison), 80)
        self.assertAlmostEqual(summary["basis"], 2.0)
        self.assertGreater(summary["correlation_20d"], 0.99)
        self.assertEqual(summary["confirmation"], "同向确认")


if __name__ == "__main__":
    unittest.main()
