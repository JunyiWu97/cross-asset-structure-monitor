from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineConfig:
    breakout_window: int = 20
    invalidation_window: int = 10
    fast_ema: int = 20
    slow_ema: int = 60
    atr_window: int = 14
    pivot_span: int = 3
    fallback_delta_atr: float = 5.0


def _rma(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def prepare_history(history: pd.DataFrame, config: EngineConfig | None = None) -> pd.DataFrame:
    cfg = config or EngineConfig()
    frame = history.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"])
    frame = (
        frame.dropna(subset=["open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr"] = _rma(true_range, cfg.atr_window)
    frame["ema_fast"] = frame["close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=cfg.slow_ema, adjust=False).mean()
    frame["bull_trigger"] = (
        frame["high"].shift(1).rolling(cfg.breakout_window).max()
    )
    frame["bear_trigger"] = (
        frame["low"].shift(1).rolling(cfg.breakout_window).min()
    )
    frame["long_invalidation"] = (
        frame["low"].shift(1).rolling(cfg.invalidation_window).min()
    )
    frame["short_invalidation"] = (
        frame["high"].shift(1).rolling(cfg.invalidation_window).max()
    )
    return frame


def _pivot_levels(frame: pd.DataFrame, direction: str, span: int) -> np.ndarray:
    column = "high" if direction == "long" else "low"
    values = frame[column]
    rolling = values.rolling(span * 2 + 1, center=True)
    if direction == "long":
        mask = values.eq(rolling.max())
    else:
        mask = values.eq(rolling.min())
    return values[mask].dropna().to_numpy(dtype=float)


def _structure_delta(
    frame: pd.DataFrame,
    base: float,
    atr: float,
    direction: str,
    config: EngineConfig,
) -> tuple[float, str, float | None]:
    history = frame.tail(504)
    levels = _pivot_levels(history, direction, config.pivot_span)
    signed_distance = levels - base if direction == "long" else base - levels
    multiples = signed_distance / atr
    eligible = (multiples >= 3.0) & (multiples <= 8.0)
    if eligible.any():
        candidates = signed_distance[eligible]
        candidate_multiples = multiples[eligible]
        index = int(np.argmin(np.abs(candidate_multiples - 5.0)))
        return float(candidates[index]), "历史结构位", float(candidate_multiples[index])
    return config.fallback_delta_atr * atr, "5×ATR回退", None


def calculate_signal(
    history: pd.DataFrame,
    config: EngineConfig | None = None,
) -> dict[str, float | str | pd.Timestamp]:
    cfg = config or EngineConfig()
    frame = prepare_history(history, cfg)
    if len(frame) < cfg.slow_ema + cfg.breakout_window:
        raise ValueError("历史数据不足，至少需要80个有效交易日")

    latest = frame.iloc[-1]
    direction = "long" if latest["ema_fast"] >= latest["ema_slow"] else "short"
    sign = 1.0 if direction == "long" else -1.0
    base = float(latest["bull_trigger"] if direction == "long" else latest["bear_trigger"])
    invalidation = float(
        latest["long_invalidation"] if direction == "long" else latest["short_invalidation"]
    )
    close = float(latest["close"])
    atr = float(latest["atr"])
    if not all(np.isfinite(value) for value in (base, invalidation, close, atr)) or atr <= 0:
        raise ValueError("最新交易日缺少有效的B、失效位或ATR")

    distance_to_base_atr = sign * (close - base) / atr
    if -0.5 <= distance_to_base_atr < 0:
        status = "接近触发"
    elif 0 <= distance_to_base_atr <= 0.75:
        status = "进入候选区"
    elif distance_to_base_atr > 0.75:
        status = "趋势运行"
    else:
        status = "等待"

    delta, delta_source, structure_multiple = _structure_delta(
        frame.iloc[:-1], base, atr, direction, cfg
    )
    if direction == "long":
        zone_low, zone_high = base - 0.25 * atr, base + 0.50 * atr
    else:
        zone_low, zone_high = base - 0.50 * atr, base + 0.25 * atr

    return {
        "trade_date": latest["date"],
        "close": close,
        "direction": direction,
        "status": status,
        "b": base,
        "atr14": atr,
        "delta": delta,
        "delta_source": delta_source,
        "structure_multiple": structure_multiple,
        "zone_low": float(zone_low),
        "zone_high": float(zone_high),
        "invalidation": invalidation,
        "distance_to_b_atr": float(abs(close - base) / atr),
        "signed_distance_to_b_atr": float(distance_to_base_atr),
        "t1": float(base + sign * delta),
        "t2": float(base + sign * 2 * delta),
        "t3": float(base + sign * 3 * delta),
        "ema_fast": float(latest["ema_fast"]),
        "ema_slow": float(latest["ema_slow"]),
    }

