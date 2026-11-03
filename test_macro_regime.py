import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from macro_regime import (
    SERIES_META,
    _fetch_fred_series,
    analyze_treasury_drivers,
    build_asset_impact,
    fetch_macro_series,
    macro_freshness,
    parse_acm_term_premium,
)


class MacroRegimeTest(unittest.TestCase):
    def test_fred_fetch_uses_browser_impersonation_when_available(self):
        response = Mock()
        response.content = b"observation_date,DGS10\n2026-10-30,4.94\n"
        response.raise_for_status.return_value = None
        browser = Mock()
        browser.get.return_value = response
        with patch("macro_regime.browser_requests", browser):
            result = _fetch_fred_series("DGS10", retries=1)
        self.assertEqual(result.iloc[-1]["value"], 4.94)
        browser.get.assert_called_once_with(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
            timeout=20,
            impersonate="chrome",
        )

    def test_macro_freshness_distinguishes_release_lag_from_stale_cache(self):
        data = {
            series_id: pd.DataFrame({"date": ["2026-10-30"], "value": [1.0]})
            for series_id in SERIES_META
        }
        data["PCEPILFE"] = pd.DataFrame({"date": ["2026-09-01"], "value": [1.0]})
        fresh = macro_freshness(data, as_of="2026-11-03").set_index("series_id")
        self.assertEqual(fresh.loc["DGS10", "数据状态"], "正常")
        self.assertEqual(fresh.loc["PCEPILFE", "数据状态"], "正常")

        data["PCEPILFE"] = pd.DataFrame({"date": ["2026-07-01"], "value": [1.0]})
        stale = macro_freshness(data, as_of="2026-11-03").set_index("series_id")
        self.assertEqual(stale.loc["PCEPILFE", "数据状态"], "滞后")

    def test_prefer_cache_avoids_remote_macro_request(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            for series_id in SERIES_META:
                pd.DataFrame({"date": ["2026-09-01"], "value": [1.0]}).to_csv(
                    cache_dir / f"{series_id}.csv", index=False
                )
            with patch("macro_regime.MACRO_CACHE_DIR", cache_dir), patch(
                "macro_regime.requests.Session.get",
                side_effect=AssertionError("cache-first mode must not use the network"),
            ):
                result = fetch_macro_series(prefer_cache=True)
            self.assertEqual(set(result), set(SERIES_META))

    def test_acm_parser_separates_expected_short_rate_and_term_premium(self):
        parsed = parse_acm_term_premium(
            "RunDates,TERMYld,ACMFITYld,GSWYld\n"
            "31-Jul-2026,0.80,4.80,4.82\n"
            "31-Aug-2026,0.70,4.70,4.72\n"
        )
        self.assertAlmostEqual(parsed["ACMTP10"].iloc[-1]["value"], 0.70)
        self.assertAlmostEqual(parsed["ACMEXP10"].iloc[-1]["value"], 4.00)

    def test_treasury_driver_decomposition_identifies_real_yield_move(self):
        dates = pd.to_datetime(["2026-07-01", "2026-08-01"])

        def frame(old, new):
            return pd.DataFrame({"date": dates, "value": [old, new]})

        data = {
            "DGS10": frame(4.00, 4.30),
            "DGS2": frame(3.80, 3.90),
            "DFII10": frame(1.60, 1.85),
            "T10YIE": frame(2.40, 2.45),
            "ACMTP10": frame(0.40, 0.45),
            "ACMEXP10": frame(3.60, 3.85),
        }
        result = analyze_treasury_drivers(data, days=30)
        self.assertEqual(result["dominant_driver"], "实际利率主导")
        self.assertEqual(result["curve_move"], "熊市陡峭化")
        self.assertAlmostEqual(result["changes_bp"]["分解残差"], 0.0)

    def test_liquidity_and_low_stress_support_bitcoin(self):
        factors = {
            "增长": 20.0,
            "通胀压力": 0.0,
            "利率紧缩": 0.0,
            "流动性": 70.0,
            "金融压力": -60.0,
            "中国周期": 0.0,
            "美元动量": -40.0,
        }
        impact = build_asset_impact(factors).set_index("资产")
        self.assertGreater(impact.loc["比特币", "宏观分数"], 20)
        self.assertEqual(impact.loc["比特币", "方向先验"], "偏多")

    def test_hot_inflation_offsets_weak_growth_support_for_long_bonds(self):
        factors = {
            "增长": -30.0,
            "通胀压力": 70.0,
            "利率紧缩": 50.0,
            "流动性": -20.0,
            "金融压力": 10.0,
            "中国周期": 0.0,
            "美元动量": 20.0,
        }
        impact = build_asset_impact(factors).set_index("资产")
        self.assertLess(impact.loc["长期国债", "宏观分数"], 0)

    def test_gold_separates_inflation_support_from_rate_headwind(self):
        neutral = {
            "增长": 0.0,
            "通胀压力": 0.0,
            "利率紧缩": 0.0,
            "流动性": 0.0,
            "金融压力": 0.0,
            "中国周期": 0.0,
            "美元动量": 0.0,
        }
        inflation = build_asset_impact({**neutral, "通胀压力": 60.0}).set_index("资产")
        tightening = build_asset_impact({**neutral, "利率紧缩": 60.0}).set_index("资产")

        self.assertGreater(inflation.loc["黄金", "宏观分数"], 0)
        self.assertLess(tightening.loc["黄金", "宏观分数"], 0)
        self.assertIn("通胀压力", inflation.loc["黄金", "贡献拆解"])


if __name__ == "__main__":
    unittest.main()
