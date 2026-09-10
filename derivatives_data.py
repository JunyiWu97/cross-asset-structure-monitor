from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf


CFTC_DISAGG_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
CFTC_CODES = {
    "gc": "088691",
    "si": "084691",
    "hg": "085692",
    "cl": "067651",
    "ng": "023651",
    "cc": "073732",
    "kc": "083731",
    "zs": "005602",
}

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CONTRACT_SPECS = {
    "es": ("ES", "CME", [3, 6, 9, 12]),
    "nq": ("NQ", "CME", [3, 6, 9, 12]),
    "zn": ("ZN", "CBT", [3, 6, 9, 12]),
    "gc": ("GC", "CMX", [2, 4, 6, 8, 10, 12]),
    "si": ("SI", "CMX", [3, 5, 7, 9, 12]),
    "hg": ("HG", "CMX", [3, 5, 7, 9, 12]),
    "cl": ("CL", "NYM", list(range(1, 13))),
    "ng": ("NG", "NYM", list(range(1, 13))),
    "dx": ("DX", "NYB", [3, 6, 9, 12]),
    "zs": ("ZS", "CBT", [1, 3, 5, 7, 8, 9, 11]),
}


def fetch_cot_history(asset_id: str, limit: int = 160) -> pd.DataFrame:
    code = CFTC_CODES.get(asset_id)
    if not code:
        return pd.DataFrame()
    fields = [
        "report_date_as_yyyy_mm_dd",
        "market_and_exchange_names",
        "open_interest_all",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
        "change_in_m_money_long_all",
        "change_in_m_money_short_all",
        "prod_merc_positions_long",
        "prod_merc_positions_short",
    ]
    params = {
        "$select": ",".join(fields),
        "$where": f"cftc_contract_market_code='{code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": limit,
    }
    response = requests.get(CFTC_DISAGG_URL, params=params, timeout=30)
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    if frame.empty:
        return frame
    frame = frame.rename(columns={
        "report_date_as_yyyy_mm_dd": "date",
        "market_and_exchange_names": "market",
        "open_interest_all": "open_interest",
        "m_money_positions_long_all": "managed_long",
        "m_money_positions_short_all": "managed_short",
        "change_in_m_money_long_all": "managed_long_change",
        "change_in_m_money_short_all": "managed_short_change",
        "prod_merc_positions_long": "producer_long",
        "prod_merc_positions_short": "producer_short",
    })
    frame["date"] = pd.to_datetime(frame["date"])
    numeric = [column for column in frame.columns if column not in ["date", "market"]]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["managed_net"] = frame["managed_long"] - frame["managed_short"]
    frame["managed_net_change"] = frame["managed_long_change"] - frame["managed_short_change"]
    frame["producer_net"] = frame["producer_long"] - frame["producer_short"]
    frame = frame.sort_values("date").reset_index(drop=True)
    rolling_mean = frame["managed_net"].rolling(52, min_periods=26).mean()
    rolling_std = frame["managed_net"].rolling(52, min_periods=26).std()
    frame["managed_net_z"] = (frame["managed_net"] - rolling_mean) / rolling_std.replace(0, np.nan)
    return frame


def _contract_candidates(asset_id: str, horizon_months: int = 18) -> list[tuple[str, pd.Timestamp]]:
    root, suffix, listed_months = CONTRACT_SPECS[asset_id]
    start = pd.Timestamp(date.today().replace(day=1))
    result: list[tuple[str, pd.Timestamp]] = []
    for offset in range(horizon_months + 1):
        contract_month = start + pd.DateOffset(months=offset)
        if contract_month.month not in listed_months:
            continue
        symbol = f"{root}{MONTH_CODES[contract_month.month]}{contract_month.year % 100:02d}.{suffix}"
        result.append((symbol, contract_month))
    return result


def fetch_term_structure(asset_id: str) -> pd.DataFrame:
    if asset_id not in CONTRACT_SPECS:
        return pd.DataFrame()
    contracts = _contract_candidates(asset_id)
    symbols = [symbol for symbol, _ in contracts]
    raw = yf.download(
        symbols,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if raw.empty or "Close" not in raw:
        return pd.DataFrame()
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    rows: list[dict] = []
    for symbol, contract_month in contracts:
        if symbol not in closes:
            continue
        valid = closes[symbol].dropna()
        if valid.empty:
            continue
        rows.append({
            "symbol": symbol,
            "contract_month": contract_month,
            "date": pd.Timestamp(valid.index[-1]).tz_localize(None),
            "price": float(valid.iloc[-1]),
        })
    frame = pd.DataFrame(rows).sort_values("contract_month").reset_index(drop=True)
    if len(frame) < 2:
        return frame
    front_price = frame.iloc[0]["price"]
    front_month = frame.iloc[0]["contract_month"]
    frame["days_from_front"] = (frame["contract_month"] - front_month).dt.days
    frame["carry_from_front_pct"] = frame["price"] / front_price - 1
    frame["annualized_carry_pct"] = np.where(
        frame["days_from_front"] > 0,
        frame["carry_from_front_pct"] * 365 / frame["days_from_front"],
        np.nan,
    )
    return frame


def fetch_crypto_options(asset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    currency = {"btc": "BTC", "eth": "ETH"}.get(asset_id)
    if not currency:
        return pd.DataFrame(), pd.DataFrame()
    response = requests.get(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        params={"currency": currency, "kind": "option"},
        timeout=30,
    )
    response.raise_for_status()
    frame = pd.DataFrame(response.json().get("result", []))
    if frame.empty:
        return frame, frame
    parts = frame["instrument_name"].str.split("-", expand=True)
    frame["expiry"] = pd.to_datetime(parts[1], format="%d%b%y", errors="coerce")
    frame["strike"] = pd.to_numeric(parts[2], errors="coerce")
    frame["option_type"] = parts[3].map({"C": "Call", "P": "Put"})
    for column in ["open_interest", "volume", "mark_iv", "underlying_price"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["expiry", "strike", "option_type"])
    frame = frame[frame["expiry"] >= pd.Timestamp(date.today())].copy()
    summary = (
        frame.groupby(["expiry", "option_type"], as_index=False)
        .agg(open_interest=("open_interest", "sum"), volume=("volume", "sum"))
    )
    return frame, summary
