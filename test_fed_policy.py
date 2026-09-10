import unittest

import pandas as pd

from fed_policy import (
    build_probability_tree,
    expectation_changes,
    fed_funds_symbol,
    infer_meeting_moves,
    parse_cme_fedwatch_page,
    parse_fomc_calendar,
)


class FedPolicyTest(unittest.TestCase):
    def test_contract_symbol_uses_cme_month_code(self):
        self.assertEqual(fed_funds_symbol(pd.Period("2026-09", freq="M")), "ZQU26.CBT")
        self.assertEqual(fed_funds_symbol(pd.Period("2027-12", freq="M")), "ZQZ27.CBT")

    def test_calendar_parser_uses_policy_decision_day(self):
        html = """
        <div class="panel panel-default">
          <div class="panel-heading"><h4>2027 FOMC Meetings</h4></div>
          <div class="row fomc-meeting">
            <div class="fomc-meeting__month">March</div>
            <div class="fomc-meeting__date">16-17*</div>
          </div>
        </div>
        """
        dates = parse_fomc_calendar(html, {2027})
        self.assertEqual(dates.tolist(), [pd.Timestamp("2027-03-17")])

    def test_meeting_rate_is_backsolved_from_nonmeeting_anchor(self):
        curve = pd.DataFrame(
            {
                "contract_month": pd.period_range("2026-09", periods=3, freq="M"),
                "implied_rate": [3.70, 3.79, 3.85],
            }
        )
        meetings = pd.to_datetime(["2026-09-16", "2026-10-28"])
        moves = infer_meeting_moves(curve, meetings)
        self.assertEqual(len(moves), 2)
        self.assertEqual(round(moves.iloc[0]["start_effr"], 3), 3.627)
        self.assertEqual(round(moves.iloc[0]["end_effr"], 3), 3.784)
        self.assertEqual(round(moves.iloc[0]["expected_steps"], 3), 0.627)

    def test_probability_tree_preserves_expected_value(self):
        moves = pd.DataFrame(
            {
                "meeting_date": pd.to_datetime(["2026-09-16", "2026-10-28"]),
                "expected_move_bp": [15.0, -5.0],
                "expected_steps": [0.6, -0.2],
            }
        )
        probabilities, summary = build_probability_tree(moves, 3.50, 3.75)
        for _, group in probabilities.groupby("meeting_date"):
            self.assertEqual(round(group["probability"].sum(), 8), 100)
        self.assertEqual(round(summary.iloc[-1]["expected_cumulative_bp"], 8), 10)

    def test_expectation_changes_reports_probability_points_and_basis_points(self):
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-08-03", "2026-08-04", "2026-08-27", "2026-09-02", "2026-09-03"]
                ),
                "cut_probability": [20.0, 18.0, 15.0, 10.0, 8.0],
                "hold_probability": [60.0, 58.0, 55.0, 50.0, 47.0],
                "hike_probability": [20.0, 24.0, 30.0, 40.0, 45.0],
                "next_expected_move_bp": [0.0, 1.0, 3.75, 7.5, 9.25],
                "year_end_midpoint": [3.50, 3.55, 3.60, 3.70, 3.75],
            }
        )
        changes = expectation_changes(history)
        self.assertEqual(changes["1日"]["hike_probability_pp"], 5.0)
        self.assertEqual(changes["1周"]["next_expected_move_bp"], 5.5)
        self.assertEqual(changes["1月"]["year_end_midpoint_bp"], 25.0)

    def test_cme_page_parser_reads_official_probability_history(self):
        html = """
        <table>
          <tr><td>Meeting Information</td></tr>
          <tr><th>Meeting Date</th><th>Contract</th><th>Expires</th><th>Mid Price</th><th>Prior Volume</th><th>Prior OI</th></tr>
          <tr><td>16 Sep 2026</td><td>ZQU6</td><td>30 Sep 2026</td><td>96.3013</td><td>50,046</td><td>234,140</td></tr>
        </table>
        <table>
          <tr><th>Probabilities</th></tr>
          <tr><th>Ease</th><th>No Change</th><th>Hike</th></tr>
          <tr><td>0.0 %</td><td>39.8 %</td><td>60.2 %</td></tr>
        </table>
        <table>
          <tr><th>Target Rate (bps)</th><th>Probability(%)</th></tr>
          <tr><th>Now *</th><th>1 Day 2 Sep 2026</th><th>1 Week 27 Aug 2026</th><th>1 Month 3 Aug 2026</th></tr>
          <tr><td>350-375 (Current)</td><td>39.8%</td><td>36.8%</td><td>64.6%</td><td>32.8%</td></tr>
          <tr><td>375-400</td><td>60.2%</td><td>63.2%</td><td>35.4%</td><td>67.2%</td></tr>
          <tr><td>* Data as of 3 Sep 2026 03:40:25 CT</td></tr>
        </table>
        """
        info, probabilities = parse_cme_fedwatch_page(html)
        self.assertEqual(info["meeting_date"], pd.Timestamp("2026-09-16"))
        self.assertEqual(info["mid_price"], 96.3013)
        self.assertEqual(info["hike_probability"], 60.2)
        current_hike = probabilities[
            (probabilities["horizon"] == "now") & (probabilities["target_lower"] == 3.75)
        ]
        self.assertEqual(float(current_hike.iloc[0]["probability"]), 60.2)
        self.assertEqual(
            probabilities[probabilities["horizon"] == "1m"]["comparison_date"].iloc[0],
            pd.Timestamp("2026-08-03"),
        )


if __name__ == "__main__":
    unittest.main()
