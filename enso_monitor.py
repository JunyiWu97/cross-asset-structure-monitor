from __future__ import annotations

import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "enso"
WEEKLY_SST_URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"

NOAA_IMAGE_URLS = {
    "sst_map": "https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso_update/sstweek_c.gif",
    "sst_hovmoller": "https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso_update/ssttlon5_c.gif",
    "heat_content": "https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso_update/heat-last-year-hr.png",
    "subsurface": "https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso_update/zlon_last-hr.png",
}

_SESSION_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CrossAssetMonitor/1.0; research dashboard)",
}


def parse_weekly_sst(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    regions = ("nino12", "nino3", "nino34", "nino4")
    for line in text.splitlines():
        if not re.match(r"^\s\d{2}[A-Z]{3}\d{4}", line):
            continue
        date = pd.to_datetime(line[1:10], format="%d%b%Y", errors="coerce")
        blocks = [line[start:end] for start, end in ((14, 23), (27, 36), (40, 49), (53, 62))]
        values: list[float] = []
        for block in blocks:
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", block)
            if len(numbers) != 2:
                values = []
                break
            values.extend(float(number) for number in numbers)
        if pd.isna(date) or len(values) != 8:
            continue
        row: dict[str, object] = {"date": date}
        for index, region in enumerate(regions):
            row[f"{region}_sst"] = values[index * 2]
            row[f"{region}_anom"] = values[index * 2 + 1]
        rows.append(row)
    if not rows:
        raise ValueError("NOAA周度SST文本中没有可解析的观测")
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def parse_roni(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), sep=r"\s+")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"seas", "yr", "anom"}
    if not required.issubset(frame.columns):
        raise ValueError("NOAA RONI文本缺少必要字段")
    season_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    frame = frame.rename(columns={"seas": "season", "yr": "year", "anom": "roni"})
    frame["date"] = pd.to_datetime(
        {"year": frame["year"], "month": frame["season"].map(season_month), "day": 1},
        errors="coerce",
    )
    frame["roni"] = pd.to_numeric(frame["roni"], errors="coerce")
    return frame[["date", "season", "year", "roni"]].dropna().reset_index(drop=True)


def _request_text(url: str, cache_name: str, prefer_cache: bool = False) -> tuple[str, bool, datetime]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_name
    if prefer_cache and cache_path.exists():
        timestamp = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return cache_path.read_text(encoding="utf-8"), True, timestamp
    try:
        response = requests.get(url, headers=_SESSION_HEADERS, timeout=30)
        response.raise_for_status()
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text, False, datetime.now(timezone.utc)
    except Exception:
        if not cache_path.exists():
            raise
        timestamp = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return cache_path.read_text(encoding="utf-8"), True, timestamp


def fetch_weekly_sst(prefer_cache: bool = False) -> tuple[pd.DataFrame, bool, datetime]:
    text, cached, fetched_at = _request_text(WEEKLY_SST_URL, "weekly_sst.txt", prefer_cache)
    return parse_weekly_sst(text), cached, fetched_at


def fetch_roni(prefer_cache: bool = False) -> tuple[pd.DataFrame, bool, datetime]:
    text, cached, fetched_at = _request_text(RONI_URL, "roni.txt", prefer_cache)
    return parse_roni(text), cached, fetched_at


def fetch_noaa_image(image_id: str, prefer_cache: bool = False) -> tuple[bytes, bool, datetime]:
    if image_id not in NOAA_IMAGE_URLS:
        raise KeyError(f"未知NOAA图层：{image_id}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = NOAA_IMAGE_URLS[image_id]
    suffix = Path(url).suffix or ".img"
    cache_path = CACHE_DIR / f"{image_id}{suffix}"
    if prefer_cache and cache_path.exists():
        timestamp = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return cache_path.read_bytes(), True, timestamp
    try:
        response = requests.get(url, headers=_SESSION_HEADERS, timeout=45)
        response.raise_for_status()
        if not response.content or not response.headers.get("content-type", "").startswith("image/"):
            raise ValueError("NOAA图层返回的不是图像")
        cache_path.write_bytes(response.content)
        return response.content, False, datetime.now(timezone.utc)
    except Exception:
        if not cache_path.exists():
            raise
        timestamp = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return cache_path.read_bytes(), True, timestamp


def build_enso_snapshot(weekly: pd.DataFrame) -> dict:
    if weekly.empty:
        raise ValueError("周度SST序列为空")
    latest = weekly.iloc[-1]
    base = weekly.iloc[-5] if len(weekly) >= 5 else weekly.iloc[0]
    anomalies = {region: float(latest[f"{region}_anom"]) for region in ("nino12", "nino3", "nino34", "nino4")}
    changes = {region: anomalies[region] - float(base[f"{region}_anom"]) for region in anomalies}

    nino34 = anomalies["nino34"]
    nino34_change = changes["nino34"]
    if nino34 >= 0.5:
        phase = "暖异常强化" if nino34_change >= 0.2 else "暖异常"
    elif nino34 <= -0.5:
        phase = "冷异常强化" if nino34_change <= -0.2 else "冷异常"
    else:
        phase = "中性波动"

    east_west_spread = anomalies["nino12"] - anomalies["nino4"]
    if east_west_spread >= 0.7:
        pattern = "暖异常偏东"
    elif east_west_spread <= -0.7:
        pattern = "暖异常偏西" if nino34 >= 0 else "冷异常偏东"
    else:
        pattern = "东西分布较均衡"

    warm_regions = sum(value >= 0.5 for value in anomalies.values())
    cold_regions = sum(value <= -0.5 for value in anomalies.values())
    breadth = f"{warm_regions}/4区偏暖" if warm_regions else f"{cold_regions}/4区偏冷" if cold_regions else "各区中性"
    return {
        "as_of": pd.Timestamp(latest["date"]),
        "phase": phase,
        "pattern": pattern,
        "breadth": breadth,
        "anomalies": anomalies,
        "changes_4w": changes,
        "east_west_spread": east_west_spread,
    }
