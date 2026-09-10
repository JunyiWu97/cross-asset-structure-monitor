import unittest

import pandas as pd

from macro_regime import analyze_treasury_drivers, build_asset_impact, parse_acm_term_premium


class MacroRegimeTest(unittest.TestCase):
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
