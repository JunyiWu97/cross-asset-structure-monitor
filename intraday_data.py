from __future__ import annotations

import pandas as pd
import yfinance as yf
import akshare as ak


YAHOO_INTERVALS = {
    "1m": ("1m", "7d"),
    "5m": ("5m", "60d"),
    "1h": ("60m", "730d"),
    "4h": ("60m", "730d"),
}
SINA_INTERVALS = {"1m": "1", "5m": "5", "1h": "60", "4h": "60"}


def _standardize(frame: pd.DataFrame, timezone: str | None = None) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    normalized_columns = [str(column).strip().lower() for column in data.columns]
    if "date" in normalized_columns or "datetime" in normalized_columns:
        data = data.reset_index(drop=True)
    else:
        data = data.reset_index()
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"datetime": "date", "index": "date", "hold": "open_interest"})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"分钟行情缺少字段：{missing}")

    dates = pd.to_datetime(data["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        if timezone:
            dates = dates.dt.tz_convert(timezone)
        dates = dates.dt.tz_localize(None)
    data["date"] = dates
    for column in ["open", "high", "low", "close", "volume", "open_interest"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = [column for column in required + ["open_interest"] if column in data]
    return (
        data[keep]
        .dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def resample_ohlcv(frame: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "open_interest" in frame:
        aggregations["open_interest"] = "last"
    result = (
        frame.set_index("date")
        .resample(rule, origin="start_day")
        .agg(aggregations)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return result


def fetch_intraday_bars(
    symbol: str,
    source: str,
    interval: str,
    timezone: str | None = None,
) -> pd.DataFrame:
    if interval not in YAHOO_INTERVALS:
        raise ValueError(f"不支持的周期：{interval}")

    if source == "yahoo":
        vendor_interval, period = YAHOO_INTERVALS[interval]
        raw = yf.download(
            symbol,
            period=period,
            interval=vendor_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30,
        )
        if raw.empty:
            raise RuntimeError("Yahoo未返回该标的的分钟行情")
        result = _standardize(raw, timezone)
    elif source == "akshare_sina":
        raw = ak.futures_zh_minute_sina(symbol=symbol, period=SINA_INTERVALS[interval])
        if raw.empty:
            raise RuntimeError("AKShare/新浪未返回该标的的分钟行情")
        result = _standardize(raw)
    else:
        raise RuntimeError(f"数据源{source}暂不支持分钟行情")

    if interval == "4h":
        result = resample_ohlcv(result, "4h")
    if len(result) < 80:
        raise RuntimeError(f"该周期仅取得{len(result)}根K线，无法计算模型所需的80根")
    return result


def fetch_daily_bars(symbol: str, source: str, timezone: str | None = None) -> pd.DataFrame:
    if source == "yahoo":
        raw = yf.download(
            symbol,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=25,
        )
        if raw.empty:
            raise RuntimeError("Yahoo未返回该标的的日线行情")
        result = _standardize(raw, timezone)
    elif source == "akshare_sina":
        raw = ak.futures_zh_daily_sina(symbol=symbol)
        if raw.empty:
            raise RuntimeError("AKShare/新浪未返回该标的的日线行情")
        result = _standardize(raw)
    else:
        raise RuntimeError(f"数据源{source}暂不支持日线行情")
    if len(result) < 80:
        raise RuntimeError(f"日线仅取得{len(result)}根K线，无法计算模型所需的80根")
    return result
