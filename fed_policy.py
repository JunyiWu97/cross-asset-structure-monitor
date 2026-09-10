from __future__ import annotations

import calendar
import math
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


FRED_POLICY_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF,DFEDTARU,DFEDTARL"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
CME_FEDWATCH_URL = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
CME_METHOD_URL = "https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html"
CME_EMBED_URL = "https://cmegroup-tools.quikstrike.net/User/QuikStrikeTools.aspx"
CME_VIEW_URL = "https://cmegroup-tools.quikstrike.net/User/QuikStrikeView.aspx"

MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
MONTH_NUMBERS = {
    name: number
    for number, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}

# Used only if the official calendar page is temporarily unavailable.
FALLBACK_FOMC_DATES = pd.to_datetime(
    [
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
        "2026-07-29",
        "2026-09-16",
        "2026-10-28",
        "2026-12-09",
        "2027-01-27",
        "2027-03-17",
        "2027-04-28",
        "2027-06-09",
        "2027-07-28",
        "2027-09-15",
        "2027-10-27",
        "2027-12-08",
    ]
)


def fed_funds_symbol(month: pd.Period) -> str:
    return f"ZQ{MONTH_CODES[month.month]}{month.year % 100:02d}.CBT"


def _month_range(as_of: pd.Timestamp, count: int = 18) -> list[pd.Period]:
    first = pd.Period(as_of, freq="M")
    return [first + offset for offset in range(count)]


def parse_fomc_calendar(html: str, years: set[int]) -> pd.DatetimeIndex:
    soup = BeautifulSoup(html, "html.parser")
    dates: list[pd.Timestamp] = []
    for panel in soup.select("div.panel.panel-default"):
        heading = panel.select_one("div.panel-heading")
        if heading is None:
            continue
        match = re.search(r"(20\d{2}) FOMC Meetings", heading.get_text(" ", strip=True))
        if not match:
            continue
        year = int(match.group(1))
        if year not in years:
            continue
        for meeting in panel.select("div.fomc-meeting"):
            month_element = meeting.select_one(".fomc-meeting__month")
            date_element = meeting.select_one(".fomc-meeting__date")
            if month_element is None or date_element is None:
                continue
            month_name = month_element.get_text(" ", strip=True).split("/")[-1]
            month = MONTH_NUMBERS.get(month_name)
            day_matches = re.findall(r"\d+", date_element.get_text(" ", strip=True))
            if month is None or not day_matches:
                continue
            dates.append(pd.Timestamp(year=year, month=month, day=int(day_matches[-1])))
    return pd.DatetimeIndex(sorted(set(dates)))


def fetch_fomc_calendar(as_of: pd.Timestamp) -> tuple[pd.DatetimeIndex, str]:
    years = {as_of.year, as_of.year + 1}
    try:
        response = requests.get(
            FOMC_CALENDAR_URL,
            headers={"User-Agent": "cross-asset-monitor/0.9"},
            timeout=30,
        )
        response.raise_for_status()
        dates = parse_fomc_calendar(response.text, years)
        if len(dates) >= 8:
            return dates, "美联储官网"
    except (requests.RequestException, ValueError):
        pass
    fallback = FALLBACK_FOMC_DATES[FALLBACK_FOMC_DATES.year.isin(years)]
    return fallback, "内置日历兜底"


def _price_series(frame: pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol not in frame.columns.get_level_values(0):
            return pd.Series(dtype=float)
        block = frame[symbol]
    else:
        block = frame
    column = "Close" if "Close" in block else "Adj Close"
    return pd.to_numeric(block[column], errors="coerce").dropna()


def _value_at_or_before(series: pd.Series, target: pd.Timestamp) -> float:
    sample = series[series.index <= target]
    return float(sample.iloc[-1]) if not sample.empty else np.nan


def fetch_fed_funds_curve(
    as_of: pd.Timestamp | None = None,
    count: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of = pd.Timestamp(as_of or pd.Timestamp.now()).tz_localize(None).normalize()
    months = _month_range(as_of, count)
    symbols = [fed_funds_symbol(month) for month in months]
    prices = yf.download(
        symbols,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    rows: list[dict] = []
    histories: dict[str, pd.Series] = {}
    for month, symbol in zip(months, symbols):
        series = _price_series(prices, symbol)
        if series.empty:
            retry = yf.download(
                symbol,
                period="3mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=30,
            )
            series = _price_series(retry, symbol)
        if series.empty:
            continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        histories[symbol] = series
        quote_date = series.index[-1]
        latest = float(series.iloc[-1])
        prior = _value_at_or_before(series, quote_date - pd.Timedelta(days=1))
        week = _value_at_or_before(series, quote_date - pd.Timedelta(days=7))
        month_ago = _value_at_or_before(series, quote_date - pd.Timedelta(days=30))
        rows.append(
            {
                "contract_month": month,
                "month": month.to_timestamp("M"),
                "month_label": month.strftime("%Y-%m"),
                "symbol": symbol,
                "quote_date": quote_date,
                "price": latest,
                "implied_rate": 100 - latest,
                "prior_rate": 100 - prior if np.isfinite(prior) else np.nan,
                "week_rate": 100 - week if np.isfinite(week) else np.nan,
                "month_rate": 100 - month_ago if np.isfinite(month_ago) else np.nan,
                "one_day_bp": (prior - latest) * 100 if np.isfinite(prior) else np.nan,
                "one_week_bp": (week - latest) * 100 if np.isfinite(week) else np.nan,
                "one_month_bp": (month_ago - latest) * 100 if np.isfinite(month_ago) else np.nan,
            }
        )
    curve = pd.DataFrame(rows)
    if len(curve) < min(10, count):
        raise RuntimeError("联邦基金期货分月合约缺失过多，暂不生成概率")
    history = pd.DataFrame(histories).sort_index().ffill(limit=5)
    return curve.sort_values("contract_month").reset_index(drop=True), history


def fetch_policy_rates() -> dict:
    response = requests.get(
        FRED_POLICY_URL,
        headers={"User-Agent": "cross-asset-monitor/0.9"},
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content), parse_dates=["observation_date"])
    result: dict[str, float | pd.Timestamp] = {}
    for series_id in ["DFF", "DFEDTARU", "DFEDTARL"]:
        values = pd.to_numeric(frame[series_id], errors="coerce")
        valid = frame.loc[values.notna(), ["observation_date"]].copy()
        valid["value"] = values[values.notna()].to_numpy()
        if valid.empty:
            raise RuntimeError(f"FRED缺少{series_id}数据")
        result[series_id] = float(valid["value"].iloc[-1])
        result[f"{series_id}_date"] = pd.Timestamp(valid["observation_date"].iloc[-1])
    return result


def _parse_cme_date(value: str) -> pd.Timestamp:
    match = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+20\d{2})", value)
    if not match:
        raise ValueError(f"无法解析CME日期：{value}")
    return pd.to_datetime(match.group(1), format="%d %b %Y")


def parse_cme_fedwatch_page(html: str) -> tuple[dict, pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    meeting_info: dict = {}
    for table in soup.find_all("table"):
        text = " ".join(table.get_text(" ", strip=True).split())
        if "Meeting Information" not in text or "Mid Price" not in text:
            continue
        for row in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in row.find_all(["th", "td"], recursive=False)]
            if len(cells) == 6 and cells[1].startswith("ZQ"):
                meeting_info = {
                    "meeting_date": _parse_cme_date(cells[0]),
                    "contract": cells[1],
                    "mid_price": float(cells[3]),
                    "prior_volume": int(cells[4].replace(",", "")),
                    "prior_oi": int(cells[5].replace(",", "")),
                }
                break
        if meeting_info:
            break
    if not meeting_info:
        raise RuntimeError("CME页面缺少会议报价")

    for table in soup.find_all("table"):
        text = " ".join(table.get_text(" ", strip=True).split())
        if not text.startswith("Probabilities Ease No Change Hike"):
            continue
        values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", text)]
        if len(values) >= 3:
            meeting_info["ease_probability"] = values[0]
            meeting_info["hold_probability"] = values[1]
            meeting_info["hike_probability"] = values[2]
            break

    probability_table = None
    for table in soup.find_all("table"):
        text = " ".join(table.get_text(" ", strip=True).split())
        if text.startswith("Target Rate (bps) Probability(%)") and "1 Week" in text:
            probability_table = table
            break
    if probability_table is None:
        raise RuntimeError("CME页面缺少目标利率概率表")

    rows = probability_table.find_all("tr")
    horizon_cells = [
        " ".join(cell.get_text(" ", strip=True).split())
        for cell in rows[1].find_all(["th", "td"], recursive=False)
    ]
    page_text = " ".join(probability_table.get_text(" ", strip=True).split())
    as_of_match = re.search(r"Data as of (\d{1,2}\s+[A-Za-z]{3}\s+20\d{2})\s+([0-9:]+)\s+CT", page_text)
    if not as_of_match:
        raise RuntimeError("CME页面缺少数据时间")
    as_of_date = _parse_cme_date(as_of_match.group(1))
    horizon_names = ["now", "1d", "1w", "1m"]
    horizon_dates = {"now": as_of_date}
    for name, label in zip(horizon_names[1:], horizon_cells[1:]):
        horizon_dates[name] = _parse_cme_date(label)

    probability_rows: list[dict] = []
    for row in rows[2:]:
        cells = [
            " ".join(cell.get_text(" ", strip=True).split())
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        if len(cells) != 5:
            continue
        range_match = re.match(r"(\d+)-(\d+)", cells[0])
        if not range_match:
            continue
        lower = int(range_match.group(1)) / 100
        upper = int(range_match.group(2)) / 100
        for horizon, value in zip(horizon_names, cells[1:]):
            probability = float(value.replace("%", "").strip())
            if probability <= 0:
                continue
            probability_rows.append(
                {
                    "meeting_date": meeting_info["meeting_date"],
                    "horizon": horizon,
                    "comparison_date": horizon_dates[horizon],
                    "target_lower": lower,
                    "target_upper": upper,
                    "target_range": f"{lower:.2f}-{upper:.2f}%",
                    "probability": probability,
                }
            )
    if not probability_rows:
        raise RuntimeError("CME目标利率概率为空")
    meeting_info["as_of_ct"] = f"{as_of_match.group(1)} {as_of_match.group(2)} CT"
    return meeting_info, pd.DataFrame(probability_rows)


def fetch_cme_fedwatch(meeting_count: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    params = {
        "viewitemid": "IntegratedFedWatchTool",
        "userId": "lwolf",
        "jobRole": "",
        "company": "",
        "companyType": "",
    }
    headers = {
        "Referer": CME_FEDWATCH_URL,
        "User-Agent": "Mozilla/5.0 (compatible; cross-asset-monitor/0.9)",
    }
    session = requests.Session()
    wrapper = session.get(CME_EMBED_URL, params=params, headers=headers, timeout=30)
    wrapper.raise_for_status()
    wrapper_soup = BeautifulSoup(wrapper.text, "html.parser")
    instance = wrapper_soup.select_one("#global_instanceCache")
    if instance is None:
        raise RuntimeError("CME FedWatch会话初始化失败")
    view_params = {**params}
    for pair in instance["value"].split("&"):
        key, value = pair.split("=", 1)
        view_params[key] = value
    response = session.get(
        CME_VIEW_URL,
        params=view_params,
        headers={**headers, "Referer": wrapper.url},
        timeout=30,
    )
    response.raise_for_status()

    infos: list[dict] = []
    probability_frames: list[pd.DataFrame] = []
    as_of_ct = ""
    for index in range(meeting_count):
        soup = BeautifulSoup(response.text, "html.parser")
        if index:
            links = [link for link in soup.find_all("a", href=True) if "lbMeeting" in link["href"]]
            if index >= len(links):
                break
            form_data = {
                field.get("name"): field.get("value", "")
                for field in soup.select("input[type=hidden][name]")
            }
            form_data["__EVENTTARGET"] = links[index]["href"].split("'")[1]
            form_data["__EVENTARGUMENT"] = ""
            response = session.post(
                response.url,
                data=form_data,
                headers={**headers, "Referer": response.url},
                timeout=30,
            )
            response.raise_for_status()
        info, frame = parse_cme_fedwatch_page(response.text)
        infos.append(info)
        probability_frames.append(frame)
        as_of_ct = info["as_of_ct"]
    if len(infos) < 3:
        raise RuntimeError("CME FedWatch返回的会议数量不足")
    return pd.DataFrame(infos), pd.concat(probability_frames, ignore_index=True), as_of_ct


def infer_meeting_moves(curve: pd.DataFrame, meetings: pd.DatetimeIndex) -> pd.DataFrame:
    work = curve.copy().reset_index(drop=True)
    work["meeting_date"] = pd.NaT
    future_meetings = pd.DatetimeIndex(meetings)
    for meeting in future_meetings:
        mask = work["contract_month"] == pd.Period(meeting, freq="M")
        if mask.any():
            work.loc[mask, "meeting_date"] = meeting

    start_rates: dict[int, float] = {}
    end_rates: dict[int, float] = {}
    for anchor in work.index[work["meeting_date"].isna()]:
        anchor = int(anchor)
        anchor_rate = float(work.at[anchor, "implied_rate"])
        start_rates[anchor] = anchor_rate
        end_rates[anchor] = anchor_rate
        index = anchor - 1
        while index >= 0 and pd.notna(work.at[index, "meeting_date"]) and index not in start_rates:
            end_rate = start_rates[index + 1]
            meeting_date = pd.Timestamp(work.at[index, "meeting_date"])
            days = calendar.monthrange(meeting_date.year, meeting_date.month)[1]
            days_before = meeting_date.day
            days_after = days - days_before
            average_rate = float(work.at[index, "implied_rate"])
            start_rate = (average_rate * days - days_after * end_rate) / days_before
            start_rates[index] = start_rate
            end_rates[index] = end_rate
            index -= 1

    rows: list[dict] = []
    for index, row in work.iterrows():
        if pd.isna(row["meeting_date"]) or index not in start_rates:
            continue
        start_rate = start_rates[index]
        end_rate = end_rates[index]
        move_bp = (end_rate - start_rate) * 100
        rows.append(
            {
                "meeting_date": pd.Timestamp(row["meeting_date"]),
                "contract_month": row["contract_month"],
                "start_effr": start_rate,
                "end_effr": end_rate,
                "expected_move_bp": move_bp,
                "expected_steps": move_bp / 25,
            }
        )
    return pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True)


def build_probability_tree(
    meeting_moves: pd.DataFrame,
    target_lower: float,
    target_upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution: dict[int, float] = {0: 1.0}
    probability_rows: list[dict] = []
    summary_rows: list[dict] = []
    expected_cumulative_bp = 0.0

    for meeting in meeting_moves.itertuples():
        expected_steps = float(meeting.expected_steps)
        lower_steps = math.floor(expected_steps)
        upper_steps = math.ceil(expected_steps)
        if lower_steps == upper_steps:
            node = {lower_steps: 1.0}
        else:
            node = {
                lower_steps: upper_steps - expected_steps,
                upper_steps: expected_steps - lower_steps,
            }
        next_distribution: dict[int, float] = {}
        for existing_steps, existing_probability in distribution.items():
            for move_steps, move_probability in node.items():
                total_steps = existing_steps + move_steps
                next_distribution[total_steps] = (
                    next_distribution.get(total_steps, 0.0) + existing_probability * move_probability
                )
        distribution = next_distribution
        expected_cumulative_bp += float(meeting.expected_move_bp)

        meeting_rows: list[dict] = []
        for steps, probability in sorted(distribution.items()):
            lower = target_lower + steps * 0.25
            upper = target_upper + steps * 0.25
            item = {
                "meeting_date": pd.Timestamp(meeting.meeting_date),
                "target_lower": lower,
                "target_upper": upper,
                "target_range": f"{lower:.2f}-{upper:.2f}%",
                "probability": probability * 100,
                "steps": steps,
            }
            probability_rows.append(item)
            meeting_rows.append(item)

        most_likely = max(meeting_rows, key=lambda item: item["probability"])
        cut_probability = sum(item["probability"] for item in meeting_rows if item["steps"] < 0)
        hold_probability = sum(item["probability"] for item in meeting_rows if item["steps"] == 0)
        hike_probability = sum(item["probability"] for item in meeting_rows if item["steps"] > 0)
        summary_rows.append(
            {
                "meeting_date": pd.Timestamp(meeting.meeting_date),
                "expected_move_bp": float(meeting.expected_move_bp),
                "expected_cumulative_bp": expected_cumulative_bp,
                "most_likely_range": most_likely["target_range"],
                "top_probability": most_likely["probability"],
                "cut_probability": cut_probability,
                "hold_probability": hold_probability,
                "hike_probability": hike_probability,
            }
        )

    return pd.DataFrame(probability_rows), pd.DataFrame(summary_rows)


def summarize_official_probabilities(
    probabilities: pd.DataFrame,
    target_lower: float,
    target_upper: float,
    horizon: str = "now",
) -> pd.DataFrame:
    rows: list[dict] = []
    previous_midpoint = (target_lower + target_upper) / 2
    current_midpoint = previous_midpoint
    selected = probabilities[probabilities["horizon"] == horizon]
    for meeting_date, group in selected.groupby("meeting_date", sort=True):
        expected_midpoint = float(
            (((group["target_lower"] + group["target_upper"]) / 2) * group["probability"]).sum() / 100
        )
        top = group.loc[group["probability"].idxmax()]
        cut_probability = float(group.loc[group["target_lower"] < target_lower, "probability"].sum())
        hold_probability = float(group.loc[group["target_lower"] == target_lower, "probability"].sum())
        hike_probability = float(group.loc[group["target_lower"] > target_lower, "probability"].sum())
        rows.append(
            {
                "meeting_date": pd.Timestamp(meeting_date),
                "expected_move_bp": (expected_midpoint - previous_midpoint) * 100,
                "expected_cumulative_bp": (expected_midpoint - current_midpoint) * 100,
                "expected_midpoint": expected_midpoint,
                "most_likely_range": top["target_range"],
                "top_probability": float(top["probability"]),
                "cut_probability": cut_probability,
                "hold_probability": hold_probability,
                "hike_probability": hike_probability,
            }
        )
        previous_midpoint = expected_midpoint
    return pd.DataFrame(rows)


def build_official_changes(
    probabilities: pd.DataFrame,
    target_lower: float,
    target_upper: float,
) -> tuple[dict[str, dict], pd.DataFrame]:
    summaries = {
        horizon: summarize_official_probabilities(probabilities, target_lower, target_upper, horizon)
        for horizon in ["now", "1d", "1w", "1m"]
    }
    current = summaries["now"]
    if current.empty:
        raise RuntimeError("CME当前概率为空")
    first_meeting = current.iloc[0]["meeting_date"]
    year_end_date = current[current["meeting_date"].dt.year == first_meeting.year].iloc[-1]["meeting_date"]

    changes: dict[str, dict] = {}
    history_rows: list[dict] = []
    labels = {"1d": "1日", "1w": "1周", "1m": "1月"}
    now_first = current.iloc[0]
    now_year_end = current[current["meeting_date"] == year_end_date].iloc[0]
    for horizon in ["1d", "1w", "1m", "now"]:
        summary = summaries[horizon]
        first = summary.iloc[0]
        source_rows = probabilities[
            (probabilities["horizon"] == horizon)
            & (probabilities["meeting_date"] == first_meeting)
        ]
        history_rows.append(
            {
                "date": pd.Timestamp(source_rows["comparison_date"].iloc[0]),
                "cut_probability": float(first["cut_probability"]),
                "hold_probability": float(first["hold_probability"]),
                "hike_probability": float(first["hike_probability"]),
                "next_expected_move_bp": float(first["expected_cumulative_bp"]),
            }
        )
        if horizon == "now":
            continue
        year_end = summary[summary["meeting_date"] == year_end_date].iloc[0]
        changes[labels[horizon]] = {
            "base_date": pd.Timestamp(source_rows["comparison_date"].iloc[0]),
            "cut_probability_pp": float(now_first["cut_probability"] - first["cut_probability"]),
            "hold_probability_pp": float(now_first["hold_probability"] - first["hold_probability"]),
            "hike_probability_pp": float(now_first["hike_probability"] - first["hike_probability"]),
            "next_expected_move_bp": float(
                now_first["expected_cumulative_bp"] - first["expected_cumulative_bp"]
            ),
            "year_end_midpoint_bp": float(
                (now_year_end["expected_midpoint"] - year_end["expected_midpoint"]) * 100
            ),
        }
    history = pd.DataFrame(history_rows).drop_duplicates("date", keep="last").sort_values("date")
    return changes, history.reset_index(drop=True)


def build_expectation_history(
    price_history: pd.DataFrame,
    curve: pd.DataFrame,
    meetings: pd.DatetimeIndex,
    target_lower: float,
    target_upper: float,
    calendar_dates: pd.DatetimeIndex,
    quote_date: pd.Timestamp,
) -> pd.DataFrame:
    contract_map = curve.set_index("symbol")["contract_month"].to_dict()
    prior_meetings = calendar_dates[calendar_dates < quote_date]
    tracking_start = quote_date - pd.Timedelta(days=60)
    if len(prior_meetings):
        tracking_start = max(tracking_start, pd.Timestamp(prior_meetings[-1]) + pd.Timedelta(days=1))

    current_midpoint = (target_lower + target_upper) / 2
    rows: list[dict] = []
    sample = price_history[(price_history.index >= tracking_start) & (price_history.index <= quote_date)]
    for date, prices in sample.iterrows():
        snapshot_rows = [
            {
                "contract_month": contract_map[symbol],
                "implied_rate": 100 - float(price),
            }
            for symbol, price in prices.items()
            if symbol in contract_map and np.isfinite(price)
        ]
        snapshot = pd.DataFrame(snapshot_rows).sort_values("contract_month").reset_index(drop=True)
        if len(snapshot) < 10:
            continue
        moves = infer_meeting_moves(snapshot, meetings)
        if moves.empty:
            continue
        _, summary = build_probability_tree(moves, target_lower, target_upper)
        if summary.empty:
            continue
        next_meeting = summary.iloc[0]
        year_end = summary[summary["meeting_date"].dt.year == quote_date.year]
        year_end_row = year_end.iloc[-1] if not year_end.empty else summary.iloc[-1]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "cut_probability": float(next_meeting["cut_probability"]),
                "hold_probability": float(next_meeting["hold_probability"]),
                "hike_probability": float(next_meeting["hike_probability"]),
                "next_expected_move_bp": float(next_meeting["expected_move_bp"]),
                "year_end_midpoint": current_midpoint
                + float(year_end_row["expected_cumulative_bp"]) / 100,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def expectation_changes(history: pd.DataFrame) -> dict[str, dict]:
    if history.empty:
        return {}
    latest = history.iloc[-1]
    result: dict[str, dict] = {}
    offsets = {"1日": pd.Timedelta(days=1), "1周": pd.Timedelta(days=7), "1月": pd.DateOffset(months=1)}
    for label, offset in offsets.items():
        candidates = history[history["date"] <= latest["date"] - offset]
        if candidates.empty:
            continue
        base = candidates.iloc[-1]
        result[label] = {
            "base_date": pd.Timestamp(base["date"]),
            "cut_probability_pp": float(latest["cut_probability"] - base["cut_probability"]),
            "hold_probability_pp": float(latest["hold_probability"] - base["hold_probability"]),
            "hike_probability_pp": float(latest["hike_probability"] - base["hike_probability"]),
            "next_expected_move_bp": float(
                latest["next_expected_move_bp"] - base["next_expected_move_bp"]
            ),
            "year_end_midpoint_bp": float(
                (latest["year_end_midpoint"] - base["year_end_midpoint"]) * 100
            ),
        }
    return result


def analyze_fed_policy(as_of: pd.Timestamp | None = None) -> dict:
    as_of = pd.Timestamp(as_of or pd.Timestamp.now()).tz_localize(None).normalize()
    cme_result = None
    cme_error = None
    with ThreadPoolExecutor(max_workers=4) as executor:
        curve_future = executor.submit(fetch_fed_funds_curve, as_of)
        policy_future = executor.submit(fetch_policy_rates)
        calendar_future = executor.submit(fetch_fomc_calendar, as_of)
        cme_future = executor.submit(fetch_cme_fedwatch, 8)
        curve, price_history = curve_future.result()
        policy = policy_future.result()
        calendar_dates, calendar_source = calendar_future.result()
        try:
            cme_result = cme_future.result()
        except (requests.RequestException, RuntimeError, ValueError, IndexError) as exc:
            cme_error = str(exc)

    future_meetings = calendar_dates[calendar_dates > as_of][:8]
    meeting_moves = infer_meeting_moves(curve, future_meetings)
    expected_meetings = {
        pd.Timestamp(meeting)
        for meeting in future_meetings
        if pd.Period(meeting, freq="M") in set(curve["contract_month"])
    }
    computed_meetings = set(pd.to_datetime(meeting_moves["meeting_date"]))
    missing_meetings = sorted(expected_meetings - computed_meetings)
    if missing_meetings:
        missing_text = ", ".join(date.strftime("%Y-%m-%d") for date in missing_meetings)
        raise RuntimeError(f"以下会议的合约链不完整：{missing_text}")
    local_probabilities, local_meeting_summary = build_probability_tree(
        meeting_moves,
        float(policy["DFEDTARL"]),
        float(policy["DFEDTARU"]),
    )
    if local_meeting_summary.empty:
        raise RuntimeError("未来会议缺少可计算的期货合约")

    quote_date = pd.Timestamp(curve["quote_date"].max())
    local_history = build_expectation_history(
        price_history,
        curve,
        future_meetings,
        float(policy["DFEDTARL"]),
        float(policy["DFEDTARU"]),
        calendar_dates,
        quote_date,
    )
    probabilities = local_probabilities
    meeting_summary = local_meeting_summary
    history = local_history
    changes = expectation_changes(local_history)
    probability_source = "开源复算降级"
    cme_as_of_ct = None
    if cme_result is not None:
        cme_info, cme_probabilities, cme_as_of_ct = cme_result
        official_summary = summarize_official_probabilities(
            cme_probabilities,
            float(policy["DFEDTARL"]),
            float(policy["DFEDTARU"]),
        )
        aggregate_columns = ["meeting_date", "ease_probability", "hold_probability", "hike_probability"]
        if set(aggregate_columns).issubset(cme_info.columns):
            aggregates = cme_info[aggregate_columns].rename(columns={"ease_probability": "cut_probability"})
            official_summary = official_summary.drop(
                columns=["cut_probability", "hold_probability", "hike_probability"]
            ).merge(aggregates, on="meeting_date", how="left", validate="one_to_one")
        official_changes, official_history = build_official_changes(
            cme_probabilities,
            float(policy["DFEDTARL"]),
            float(policy["DFEDTARU"]),
        )
        if not official_summary.empty and official_changes:
            probabilities = cme_probabilities[cme_probabilities["horizon"] == "now"].copy()
            meeting_summary = official_summary
            history = official_history
            changes = official_changes
            probability_source = "CME FedWatch官网"

    next_meeting = meeting_summary.iloc[0]
    dominant = max(
        [
            ("降息", next_meeting["cut_probability"]),
            ("维持", next_meeting["hold_probability"]),
            ("加息", next_meeting["hike_probability"]),
        ],
        key=lambda item: item[1],
    )
    year_end = meeting_summary[meeting_summary["meeting_date"].dt.year == as_of.year]
    year_end_row = year_end.iloc[-1] if not year_end.empty else meeting_summary.iloc[-1]
    current_midpoint = (float(policy["DFEDTARL"]) + float(policy["DFEDTARU"])) / 2
    year_end_midpoint = current_midpoint + float(year_end_row["expected_cumulative_bp"]) / 100

    return {
        "as_of": as_of,
        "quote_date": quote_date,
        "effr": float(policy["DFF"]),
        "effr_date": pd.Timestamp(policy["DFF_date"]),
        "target_lower": float(policy["DFEDTARL"]),
        "target_upper": float(policy["DFEDTARU"]),
        "target_date": max(pd.Timestamp(policy["DFEDTARL_date"]), pd.Timestamp(policy["DFEDTARU_date"])),
        "current_midpoint": current_midpoint,
        "year_end_midpoint": year_end_midpoint,
        "next_meeting": pd.Timestamp(next_meeting["meeting_date"]),
        "next_cut_probability": float(next_meeting["cut_probability"]),
        "next_hold_probability": float(next_meeting["hold_probability"]),
        "next_hike_probability": float(next_meeting["hike_probability"]),
        "dominant_next_action": dominant[0],
        "dominant_next_probability": float(dominant[1]),
        "probability_source": probability_source,
        "cme_as_of_ct": cme_as_of_ct,
        "cme_error": cme_error,
        "calendar_source": calendar_source,
        "curve": curve,
        "meeting_moves": meeting_moves,
        "probabilities": probabilities,
        "meeting_summary": meeting_summary,
        "expectation_history": history,
        "expectation_changes": changes,
    }
