from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
POINTS_PATH = ROOT / "config" / "climate_regions_v1.csv"
CLIMATE_DIR = ROOT / "data" / "climate"
POINT_DIR = CLIMATE_DIR / "points"
DAILY_PATH = CLIMATE_DIR / "region_daily.csv"
LATEST_PATH = ROOT / "data" / "climate_latest.csv"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("PRECTOTCORR", "GWETTOP", "GWETROOT")


def fetch_power_point(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    response = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "LST",
        },
        timeout=90,
    )
    response.raise_for_status()
    parameters = response.json()["properties"]["parameter"]
    frame = pd.DataFrame({name: pd.Series(values) for name, values in parameters.items()})
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    frame = frame.reset_index().rename(columns={
        "PRECTOTCORR": "precip_mm",
        "GWETTOP": "surface_wetness",
        "GWETROOT": "root_wetness",
    })
    for column in ["precip_mm", "surface_wetness", "root_wetness"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").mask(lambda values: values <= -900)
    return frame.dropna(subset=["precip_mm", "surface_wetness", "root_wetness"])


def _fetch_with_cache(point, start: str, end: str) -> tuple[pd.DataFrame, bool]:
    cache_path = POINT_DIR / f"{point.point_id}.csv"
    error: Exception | None = None
    for attempt in range(3):
        try:
            frame = fetch_power_point(point.latitude, point.longitude, start, end)
            frame.to_csv(cache_path, index=False, encoding="utf-8-sig")
            return frame, False
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"]), True
    raise RuntimeError(f"{point.point_id}无法更新且没有缓存：{error}")


def _circular_day_distance(day: pd.Series, target: int) -> pd.Series:
    direct = (day - target).abs()
    return np.minimum(direct, 366 - direct)


def _region_snapshot(region: pd.DataFrame, metadata: pd.Series, used_cache: bool) -> dict:
    data = region.sort_values("date").copy()
    data["rain_30d_mm"] = data["precip_mm"].rolling(30, min_periods=25).sum()
    data["root_14d"] = data["root_wetness"].rolling(14, min_periods=10).mean()
    latest = data.dropna(subset=["rain_30d_mm", "root_14d"]).iloc[-1]
    target_day = int(latest["date"].dayofyear)
    historical = data[
        (data["date"].dt.year < latest["date"].year)
        & (_circular_day_distance(data["date"].dt.dayofyear, target_day) <= 15)
    ]
    rain_reference = historical["rain_30d_mm"].dropna()
    rain_median = rain_reference.median()
    rain_vs_season = 100 * (latest["rain_30d_mm"] / rain_median - 1) if rain_median > 0 else np.nan
    earlier = data[data["date"] <= latest["date"] - pd.Timedelta(days=30)].dropna(subset=["root_14d"]).iloc[-1]
    root_change = latest["root_14d"] - earlier["root_14d"]

    if (rain_vs_season >= 25 and root_change <= -0.05) or (rain_vs_season <= -25 and root_change >= 0.05):
        state = "水分分化"
    elif rain_vs_season <= -25 and root_change <= -0.03:
        state = "偏干"
    elif rain_vs_season >= 25 and root_change >= 0.03:
        state = "偏湿"
    elif rain_vs_season <= -25 or root_change <= -0.05:
        state = "干燥关注"
    elif rain_vs_season >= 25 or root_change >= 0.05:
        state = "湿润关注"
    else:
        state = "季节常态"
    if used_cache:
        state += "（缓存）"

    return {
        "region_id": metadata["region_id"],
        "crop": metadata["crop"],
        "region_cn": metadata["region_cn"],
        "related_assets": metadata["related_assets"],
        "as_of": latest["date"],
        "rain_30d_mm": latest["rain_30d_mm"],
        "rain_vs_season_pct": rain_vs_season,
        "root_wetness": latest["root_14d"],
        "root_change_30d": root_change,
        "state": state,
        "point_count": int(data["point_count"].max()),
        "source": "NASA POWER Daily API",
        "baseline": "2016年至上一完整年份的同季节窗口",
    }


def build_climate_dataset() -> pd.DataFrame:
    points = pd.read_csv(POINTS_PATH)
    POINT_DIR.mkdir(parents=True, exist_ok=True)
    end = date.today().strftime("%Y%m%d")
    point_frames: list[pd.DataFrame] = []
    cache_regions: set[str] = set()

    for point in points.itertuples(index=False):
        frame, used_cache = _fetch_with_cache(point, "20160101", end)
        frame["point_id"] = point.point_id
        frame["crop"] = point.crop
        frame["region_id"] = point.region_id
        frame["region_cn"] = point.region_cn
        frame["related_assets"] = point.related_assets
        frame["weight"] = point.weight
        point_frames.append(frame)
        if used_cache:
            cache_regions.add(point.region_id)
        print(f"CLIMATE {point.point_id:<22} {len(frame):>4} rows", flush=True)
        time.sleep(0.25)

    point_data = pd.concat(point_frames, ignore_index=True)
    value_columns = ["precip_mm", "surface_wetness", "root_wetness"]
    weighted = point_data.copy()
    for column in value_columns:
        weighted[column] *= weighted["weight"]
    region_daily = weighted.groupby(
        ["region_id", "region_cn", "crop", "related_assets", "date"], as_index=False
    ).agg(
        precip_sum=("precip_mm", "sum"),
        surface_sum=("surface_wetness", "sum"),
        root_sum=("root_wetness", "sum"),
        weight_sum=("weight", "sum"),
        point_count=("point_id", "nunique"),
    )
    region_daily["precip_mm"] = region_daily["precip_sum"] / region_daily["weight_sum"]
    region_daily["surface_wetness"] = region_daily["surface_sum"] / region_daily["weight_sum"]
    region_daily["root_wetness"] = region_daily["root_sum"] / region_daily["weight_sum"]
    region_daily = region_daily.drop(columns=["precip_sum", "surface_sum", "root_sum"])
    region_daily.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")

    latest_rows = []
    for region_id, frame in region_daily.groupby("region_id"):
        metadata = points[points["region_id"] == region_id].iloc[0]
        latest_rows.append(_region_snapshot(frame, metadata, region_id in cache_regions))
    latest = pd.DataFrame(latest_rows).sort_values(["crop", "region_cn"])
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    return latest


if __name__ == "__main__":
    result = build_climate_dataset()
    print(result.to_string(index=False))
