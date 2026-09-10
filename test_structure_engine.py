import unittest

import numpy as np
import pandas as pd

from signal_engine import calculate_signal
from structure_engine import analyze_market_structure


class StructureEngineTest(unittest.TestCase):
    def test_rising_price_volume_and_open_interest_support_long_side(self):
        size = 140
        close = np.linspace(100, 130, size)
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=size, freq="h"),
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": [100] * (size - 1) + [220],
            "open_interest": np.arange(1000, 1000 + size),
        })
        model = calculate_signal(frame)

        result = analyze_market_structure(frame, model)
        components = result["components"].set_index("维度")

        self.assertGreater(components.loc["趋势", "分值"], 0)
        self.assertGreater(components.loc["成交量确认", "分值"], 0)
        self.assertGreater(components.loc["期货持仓", "分值"], 0)
        self.assertGreater(result["total_score"], 0)

    def test_unreliable_continuous_volume_is_not_scored(self):
        size = 140
        close = np.linspace(100, 130, size)
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=size, freq="h"),
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": [100] * (size - 1) + [10_000],
        })
        model = calculate_signal(frame)

        result = analyze_market_structure(frame, model, volume_reliable=False)
        volume_component = result["components"].set_index("维度").loc["成交量确认"]

        self.assertEqual(volume_component["分值"], 0)
        self.assertIn("未计分", volume_component["证据"])


if __name__ == "__main__":
    unittest.main()
