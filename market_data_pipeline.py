from __future__ import annotations

import time
from datetime import date
from io import StringIO
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
import yfinance as yf

from climate_pipeline import build_climate_dataset
from signal_engine import calculate_signal


ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "config" / "asset_universe_v2.csv"
HISTORY_DIR = ROOT / "data" / "history_public"
SIGNALS_PATH = ROOT / "data" / "signals_latest.csv"
SNAPSHOT_PATH = ROOT / "data" / "market_snapshot_latest.csv"
ENVIRONMENT_DIR = ROOT / "data" / "environment"
ENVIRONMENT_PATH = ROOT / "data" / "environment_latest.csv"

VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
HY_OAS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "date" in [str(column).strip().lower() for column in data.columns]:
        data = data.reset_index(drop=True)
    else:
        data = data.reset_index()
    data.columns = [str(column).strip().lower().replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"datetime": "date", "index": "date", "hold": "open_interest"})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"缺少行情字段：{missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.tz_localize(None)
    for column in ["open", "high", "low", "close", "volume", "open_interest"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    keep = [column for column in ["date", "open", "high", "low", "close", "volume", "open_interest"] if column in data]
    data = (
        data[keep]
        .dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    # 平台只在收盘后运行，避免把当天未完成的日线送进模型。
    return data[data["date"].dt.date < date.today()].reset_index(drop=True)


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    result = yf.download(
        symbol,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if result.empty:
        raise RuntimeError("Yahoo返回空数据")
    return _standardize(result)


def fetch_akshare_sina(symbol: str) -> pd.DataFrame:
    result = ak.futures_zh_daily_sina(symbol=symbol)
    if result.empty:
        raise RuntimeError("AKShare/新浪返回空数据")
    return _standardize(result)


FETCHERS = {"yahoo": fetch_yahoo, "akshare_sina": fetch_akshare_sina}


def fetch_with_retry(source: str, symbol: str, retries: int = 3) -> pd.DataFrame:
    fetcher = FETCHERS[source]
    error: Exception | None = None
    for attempt in range(retries):
        try:
            return fetcher(symbol)
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(error))


def data_quality(history: pd.DataFrame) -> tuple[str, str]:
    if len(history) < 252:
        return "failed", "有效历史不足252日"
    recent = history.tail(252)
    inverted_range = (recent["high"] < recent["low"]).sum()
    if inverted_range:
        return "failed", f"近252日有{inverted_range}条高低价倒置"
    outside_range = (
        (recent["high"] < recent[["open", "close"]].max(axis=1))
        | (recent["low"] > recent[["open", "close"]].min(axis=1))
    ).sum()
    returns = recent["close"].pct_change().abs()
    median_move = returns.rolling(60, min_periods=20).median()
    jump_count = ((returns > 0.20) & (returns > median_move * 12)).sum()
    notes: list[str] = []
    if outside_range:
        notes.append(f"{outside_range}日结算价在成交区间外")
    if jump_count:
        notes.append(f"{jump_count}次异常跳变")
    if notes:
        return "warning", "；".join(notes)
    return "verified", "结构完整"


def fetch_environment() -> pd.DataFrame:
    ENVIRONMENT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    vix_response = requests.get(VIX_URL, timeout=30)
    vix_response.raise_for_status()
    vix = pd.read_csv(StringIO(vix_response.text))
    vix.columns = [column.lower() for column in vix.columns]
    vix["date"] = pd.to_datetime(vix["date"], format="%m/%d/%Y")
    vix.to_csv(ENVIRONMENT_DIR / "vix.csv", index=False, encoding="utf-8-sig")
    vix_latest = vix.iloc[-1]
    vix_state = "压力" if vix_latest["close"] >= 25 else "偏高" if vix_latest["close"] >= 20 else "常态"
    rows.append({
        "indicator_id": "vix", "name_cn": "VIX", "as_of": vix_latest["date"],
        "value": vix_latest["close"], "unit": "指数点", "state": vix_state,
        "source": "Cboe官方CSV", "source_url": VIX_URL,
        "note": "衡量美股近端隐含波动，不直接决定方向",
    })

    oni_response = requests.get(ONI_URL, timeout=30)
    oni_response.raise_for_status()
    oni = pd.read_csv(StringIO(oni_response.text), sep=r"\s+")
    oni.columns = [column.lower() for column in oni.columns]
    oni = oni.rename(columns={"seas": "season", "yr": "year", "total": "sst", "anom": "anomaly"})
    season_month = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    oni["date"] = pd.to_datetime({"year": oni["year"], "month": oni["season"].map(season_month), "day": 1})
    oni.to_csv(ENVIRONMENT_DIR / "oni.csv", index=False, encoding="utf-8-sig")
    oni_latest = oni.iloc[-1]
    warm_streak = int((oni["anomaly"].tail(5) >= 0.5).sum())
    cold_streak = int((oni["anomaly"].tail(5) <= -0.5).sum())
    if oni_latest["anomaly"] >= 0.5:
        oni_state = "暖位相，已确认" if warm_streak == 5 else "暖位相发展中"
    elif oni_latest["anomaly"] <= -0.5:
        oni_state = "冷位相，已确认" if cold_streak == 5 else "冷位相发展中"
    else:
        oni_state = "中性"
    rows.append({
        "indicator_id": "oni", "name_cn": "Oceanic Niño Index", "as_of": oni_latest["date"],
        "value": oni_latest["anomaly"], "unit": "°C异常", "state": oni_state,
        "source": "NOAA/CPC文本序列", "source_url": ONI_URL,
        "note": f"最新季节{oni_latest['season']}；近5个重叠季中{warm_streak}个≥+0.5°C",
    })

    hy_response = requests.get(HY_OAS_URL, timeout=30)
    hy_response.raise_for_status()
    hy = pd.read_csv(StringIO(hy_response.text))
    hy.columns = [column.lower() for column in hy.columns]
    hy = hy.rename(columns={"observation_date": "date", "bamlh0a0hym2": "value"})
    hy["date"] = pd.to_datetime(hy["date"])
    hy["value"] = pd.to_numeric(hy["value"], errors="coerce")
    hy = hy.dropna(subset=["value"])
    hy.to_csv(ENVIRONMENT_DIR / "hy_oas.csv", index=False, encoding="utf-8-sig")
    hy_latest = hy.iloc[-1]
    hy_state = "压力" if hy_latest["value"] >= 5 else "偏高" if hy_latest["value"] >= 4 else "常态"
    rows.append({
        "indicator_id": "hy_oas", "name_cn": "美国高收益债OAS", "as_of": hy_latest["date"],
        "value": hy_latest["value"], "unit": "百分点", "state": hy_state,
        "source": "FRED/ICE BofA", "source_url": HY_OAS_URL,
        "note": "信用市场压力过滤器；仅用于内部研究",
    })

    result = pd.DataFrame(rows)
    result.to_csv(ENVIRONMENT_PATH, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    universe = pd.read_csv(UNIVERSE_PATH)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    signals: list[dict] = []
    snapshots: list[dict] = []
    failures: list[str] = []

    for item in universe.itertuples(index=False):
        try:
            cache_path = HISTORY_DIR / f"{item.asset_id}.csv"
            used_cache = False
            try:
                history = fetch_with_retry(item.price_source, item.public_symbol)
                history.insert(0, "asset_id", item.asset_id)
                history.to_csv(cache_path, index=False, encoding="utf-8-sig")
            except Exception:
                if not cache_path.exists():
                    raise
                history = pd.read_csv(cache_path, parse_dates=["date"])
                used_cache = True
            status, note = data_quality(history)
            if used_cache:
                status = "warning"
                note = f"本次刷新失败，使用{history.iloc[-1]['date']:%Y-%m-%d}缓存"
            signal = calculate_signal(history.drop(columns=["asset_id"]))
            latest = history.iloc[-1]
            signals.append({
                "asset_id": item.asset_id,
                **signal,
                "history_rows": len(history),
                "data_status": status,
                "data_note": note,
                "price_source": item.price_source,
            })
            snapshots.append({
                "asset_id": item.asset_id,
                "trade_date": latest["date"],
                "close": latest["close"],
                "volume": latest["volume"],
                "open_interest": latest.get("open_interest"),
                "data_source": item.price_source,
                "data_status": status,
            })
            mode = "CACHE" if used_cache else "OK"
            print(f"{mode:<5}{item.asset_id:>4}  {len(history):>4} rows  {status:<8}  {signal['status']}", flush=True)
        except Exception as exc:
            failures.append(f"{item.asset_id}: {exc}")
            print(f"ERR {item.asset_id:>4}  {exc}", flush=True)
        time.sleep(0.4)

    pd.DataFrame(signals).to_csv(SIGNALS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(snapshots).to_csv(SNAPSHOT_PATH, index=False, encoding="utf-8-sig")
    environment = fetch_environment()
    climate = build_climate_dataset()
    print(f"\nSaved {len(signals)}/{len(universe)} signals to {SIGNALS_PATH}", flush=True)
    print(f"Saved {len(environment)} environment indicators to {ENVIRONMENT_PATH}", flush=True)
    print(f"Saved {len(climate)} climate regions to {ROOT / 'data' / 'climate_latest.csv'}", flush=True)
    if failures:
        print("Failures:\n" + "\n".join(failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
