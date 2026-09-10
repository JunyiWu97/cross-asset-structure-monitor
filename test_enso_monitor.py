import unittest

from enso_monitor import build_enso_snapshot, parse_roni, parse_weekly_sst


WEEKLY_SAMPLE = """ Weekly SST data starts week centered on 2Sept1981

                Nino1+2      Nino3        Nino34        Nino4
 Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA
 01JUL2026     25.7 3.3     28.2 2.0     29.2 1.7     29.9 1.1
 29JUL2026     25.2 3.8     28.2 2.8     29.4 2.4     29.7 1.0
"""


class EnsoMonitorTests(unittest.TestCase):
    def test_parse_weekly_sst_handles_joined_negative_values(self):
        text = WEEKLY_SAMPLE + " 05AUG1981     20.6-0.1     24.8-0.1     26.5-0.2     28.3-0.3\n"
        frame = parse_weekly_sst(text)
        old = frame.iloc[0]
        self.assertAlmostEqual(old["nino12_sst"], 20.6)
        self.assertAlmostEqual(old["nino12_anom"], -0.1)

    def test_snapshot_reports_warming_and_eastward_pattern(self):
        snapshot = build_enso_snapshot(parse_weekly_sst(WEEKLY_SAMPLE))
        self.assertEqual(snapshot["phase"], "暖异常强化")
        self.assertEqual(snapshot["pattern"], "暖异常偏东")
        self.assertEqual(snapshot["breadth"], "4/4区偏暖")

    def test_parse_roni(self):
        frame = parse_roni("SEAS YR ANOM\nAMJ 2026 0.49\nMJJ 2026 0.98\n")
        self.assertEqual(frame.iloc[-1]["season"], "MJJ")
        self.assertAlmostEqual(frame.iloc[-1]["roni"], 0.98)


if __name__ == "__main__":
    unittest.main()
