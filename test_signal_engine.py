import unittest

import numpy as np
import pandas as pd

from signal_engine import calculate_signal, prepare_history


class SignalEngineTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=180)
        close = 100 + np.linspace(0, 28, len(dates)) + rng.normal(0, 0.5, len(dates))
        self.history = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.2,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100_000,
            }
        )

    def test_indicators_do_not_use_current_high_for_trigger(self):
        prepared = prepare_history(self.history)
        expected = self.history["high"].iloc[-21:-1].max()
        self.assertAlmostEqual(prepared["bull_trigger"].iloc[-1], expected)

    def test_targets_are_equally_spaced(self):
        signal = calculate_signal(self.history)
        sign = 1 if signal["direction"] == "long" else -1
        self.assertAlmostEqual(sign * (signal["t1"] - signal["b"]), signal["delta"])
        self.assertAlmostEqual(sign * (signal["t2"] - signal["t1"]), signal["delta"])
        self.assertAlmostEqual(sign * (signal["t3"] - signal["t2"]), signal["delta"])

    def test_short_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "历史数据不足"):
            calculate_signal(self.history.head(50))


if __name__ == "__main__":
    unittest.main()
