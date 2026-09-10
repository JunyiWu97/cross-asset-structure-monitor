import unittest

import pandas as pd

from intraday_data import resample_ohlcv


class IntradayDataTest(unittest.TestCase):
    def test_four_hour_ohlcv_aggregation(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=8, freq="h"),
            "open": [10, 11, 12, 13, 14, 15, 16, 17],
            "high": [12, 13, 14, 15, 16, 17, 18, 19],
            "low": [9, 10, 11, 12, 13, 14, 15, 16],
            "close": [11, 12, 13, 14, 15, 16, 17, 18],
            "volume": [1, 2, 3, 4, 5, 6, 7, 8],
        })

        result = resample_ohlcv(frame, "4h")

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["open"], 10)
        self.assertEqual(result.iloc[0]["high"], 15)
        self.assertEqual(result.iloc[0]["low"], 9)
        self.assertEqual(result.iloc[0]["close"], 14)
        self.assertEqual(result.iloc[0]["volume"], 10)


if __name__ == "__main__":
    unittest.main()
