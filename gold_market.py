from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf


NBP_GOLD_URL = "https://api.nbp.pl/api/cenyzlota/{start}/{end}/?format=json"
NBP_USD_URL = "https://api.nbp.pl/api/exchangerates/rates/a/usd/{start}/{end}/?format=json"
TROY_OUNCE_GRAMS = 31.1034768
GOLD_MONTHS = (2, 4, 6, 8, 10, 12)
MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def _date_chunks(start: date, end: date, days: int = 90) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def fetch_london_gold_proxy(lookback_days: int = 550) -> pd.DataFrame:
    """Return an official daily London-fixing proxy in USD per troy ounce.

    NBP publishes a PLN/gram gold price derived from the London fixing and FX.
    Converting it with NBP's same-day USD/PLN rate gives a transparent daily
    reference series. It is not a live executable XAU/USD quote.
    """
    end = date.today()
    start = end - timedelta(days=lookback_days)
    gold_rows: list[dict] = []
    fx_rows: list[dict] = []
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "cross-asset-monitor/0.6"})

    for chunk_start, chunk_end in _date_chunks(start, end):
        dates = {"start": chunk_start.isoformat(), "end": chunk_end.isoformat()}
        gold_response = session.get(NBP_GOLD_URL.format(**dates), timeout=30)
        if gold_response.status_code != 404:
            gold_response.raise_for_status()
            gold_rows.extend(gold_response.json())

        fx_response = session.get(NBP_USD_URL.format(**dates), timeout=30)
        if fx_response.status_code != 404:
            fx_response.raise_for_status()
            fx_rows.extend(fx_response.json().get("rates", []))

    gold = pd.DataFrame(gold_rows).rename(columns={"data": "date", "cena": "gold_pln_g"})
    fx = pd.DataFrame(fx_rows).rename(columns={"effectiveDate": "date", "mid": "usd_pln"})
    if gold.empty or fx.empty:
        raise RuntimeError("NBP未返回可用的黄金或美元汇率数据")

    gold["date"] = pd.to_datetime(gold["date"])
    fx["date"] = pd.to_datetime(fx["date"])
    gold["gold_pln_g"] = pd.to_numeric(gold["gold_pln_g"], errors="coerce")
    fx["usd_pln"] = pd.to_numeric(fx["usd_pln"], errors="coerce")
    result = gold[["date", "gold_pln_g"]].merge(
        fx[["date", "usd_pln"]], on="date", how="inner", validate="one_to_one"
    )
    result["close"] = result["gold_pln_g"] / result["usd_pln"] * TROY_OUNCE_GRAMS
    return result.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def build_gold_comparison(
    futures_history: pd.DataFrame,
    london_history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | str | pd.Timestamp]]:
    futures = futures_history[["date", "close"]].copy()
    futures["date"] = pd.to_datetime(futures["date"]).dt.normalize()
    futures["comex"] = pd.to_numeric(futures["close"], errors="coerce")
    london = london_history[["date", "close"]].copy()
    london["date"] = pd.to_datetime(london["date"]).dt.normalize()
    london["london"] = pd.to_numeric(london["close"], errors="coerce")
    comparison = futures[["date", "comex"]].merge(
        london[["date", "london"]], on="date", how="inner", validate="one_to_one"
    ).dropna()
    if len(comparison) < 60:
        raise RuntimeError("COMEX与伦敦代理的重叠历史不足60个交易日")

    comparison["basis"] = comparison["comex"] - comparison["london"]
    comparison["basis_pct"] = comparison["basis"] / comparison["london"]
    comparison["comex_return"] = comparison["comex"].pct_change()
    comparison["london_return"] = comparison["london"].pct_change()
    comparison["comex_index"] = comparison["comex"] / comparison["comex"].iloc[0] * 100
    comparison["london_index"] = comparison["london"] / comparison["london"].iloc[0] * 100

    latest = comparison.iloc[-1]
    five_day = comparison.tail(6)
    comex_5d = float(five_day["comex"].iloc[-1] / five_day["comex"].iloc[0] - 1)
    london_5d = float(five_day["london"].iloc[-1] / five_day["london"].iloc[0] - 1)
    correlation = float(comparison[["comex_return", "london_return"]].tail(20).corr().iloc[0, 1])
    same_direction = np.sign(comex_5d) == np.sign(london_5d)
    summary: dict[str, float | str | pd.Timestamp] = {
        "date": pd.Timestamp(latest["date"]),
        "comex": float(latest["comex"]),
        "london": float(latest["london"]),
        "basis": float(latest["basis"]),
        "basis_pct": float(latest["basis_pct"]),
        "comex_5d": comex_5d,
        "london_5d": london_5d,
        "correlation_20d": correlation,
        "confirmation": "同向确认" if same_direction else "走势背离",
    }
    return comparison.reset_index(drop=True), summary


def _gold_contract_candidates(horizon_months: int = 15) -> list[str]:
    start = pd.Timestamp(date.today().replace(day=1))
    symbols: list[str] = []
    for offset in range(horizon_months + 1):
        contract_month = start + pd.DateOffset(months=offset)
        if contract_month.month in GOLD_MONTHS:
            symbols.append(f"GC{MONTH_CODES[contract_month.month]}{contract_month.year % 100:02d}.CMX")
    return symbols


def _download_field(raw: pd.DataFrame, field: str, symbols: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field not in raw.columns.get_level_values(0):
            return pd.DataFrame(index=raw.index)
        result = raw[field].copy()
    elif field in raw:
        result = raw[[field]].copy()
        result.columns = symbols[:1]
    else:
        return pd.DataFrame(index=raw.index)
    if isinstance(result, pd.Series):
        result = result.to_frame(symbols[0])
    return result.reindex(columns=[symbol for symbol in symbols if symbol in result.columns])


def fetch_gold_contract_activity(
    period: str = "6mo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the available listed GC contract volumes from Yahoo.

    Yahoo does not expose exchange open interest here, so this dataset is used
    only for roll-aware volume monitoring. CFTC OI remains the position source.
    """
    symbols = _gold_contract_candidates()
    raw = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if raw.empty:
        raise RuntimeError("Yahoo未返回COMEX黄金分月合约")
    volumes = _download_field(raw, "Volume", symbols)
    closes = _download_field(raw, "Close", symbols)
    if volumes.empty:
        raise RuntimeError("分月合约缺少成交量")

    volume_long = volumes.rename_axis("date").stack(future_stack=True).rename("volume").reset_index()
    close_long = closes.rename_axis("date").stack(future_stack=True).rename("close").reset_index()
    volume_long.columns = ["date", "symbol", "volume"]
    close_long.columns = ["date", "symbol", "close"]
    activity = volume_long.merge(close_long, on=["date", "symbol"], how="left")
    activity["date"] = pd.to_datetime(activity["date"]).dt.tz_localize(None)
    activity["volume"] = pd.to_numeric(activity["volume"], errors="coerce")
    activity = activity.dropna(subset=["volume"])
    activity = activity[activity["volume"] > 0].sort_values(["date", "volume"])
    if activity.empty:
        raise RuntimeError("分月合约没有有效成交量")

    totals = activity.groupby("date", as_index=False)["volume"].sum().rename(columns={"volume": "total_volume"})
    dominant = activity.loc[activity.groupby("date")["volume"].idxmax(), ["date", "symbol", "volume", "close"]]
    dominant = dominant.rename(columns={"symbol": "dominant_contract", "volume": "dominant_volume", "close": "dominant_price"})
    timeline = totals.merge(dominant, on="date", how="left").sort_values("date").reset_index(drop=True)
    timeline["dominant_share"] = timeline["dominant_volume"] / timeline["total_volume"]
    previous_contract = timeline["dominant_contract"].shift(1)
    timeline["roll_flag"] = previous_contract.notna() & timeline["dominant_contract"].ne(previous_contract)
    median_volume = timeline["total_volume"].shift(1).rolling(20, min_periods=10).median()
    timeline["volume_ratio"] = timeline["total_volume"] / median_volume

    latest_date = timeline.iloc[-1]["date"]
    snapshot = activity[activity["date"] == latest_date].copy()
    snapshot["volume_share"] = snapshot["volume"] / snapshot["volume"].sum()
    snapshot = snapshot.sort_values("volume", ascending=False).reset_index(drop=True)
    return timeline, snapshot
