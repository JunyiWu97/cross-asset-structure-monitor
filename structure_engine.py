from __future__ import annotations

import numpy as np
import pandas as pd

from signal_engine import prepare_history


def analyze_market_structure(
    history: pd.DataFrame,
    model: dict,
    cot_history: pd.DataFrame | None = None,
    volume_reliable: bool = True,
) -> dict:
    frame = prepare_history(history)
    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    atr = float(latest["atr"])
    price_change = float(latest["close"] - previous["close"])
    direction_sign = 1 if model["direction"] == "long" else -1
    components: list[dict] = []

    trend_gap_atr = float((latest["ema_fast"] - latest["ema_slow"]) / atr)
    trend_score = float(np.clip(trend_gap_atr / 1.5 * 30, -30, 30))
    components.append({
        "维度": "趋势",
        "证据": f"EMA20−EMA60 = {trend_gap_atr:+.2f} ATR",
        "分值": trend_score,
    })

    status = str(model["status"])
    trigger_points = {"接近触发": 15, "进入候选区": 25, "趋势运行": 10, "等待": 0}[status]
    trigger_score = float(direction_sign * trigger_points)
    components.append({
        "维度": "触发位置",
        "证据": f"{status}，距B {float(model['signed_distance_to_b_atr']):+.2f} ATR",
        "分值": trigger_score,
    })

    ema_slope_atr = float((latest["ema_fast"] - frame.iloc[-6]["ema_fast"]) / atr)
    momentum_score = float(np.clip(ema_slope_atr / 1.5 * 10, -10, 10))
    components.append({
        "维度": "短期动量",
        "证据": f"EMA20五根变化 {ema_slope_atr:+.2f} ATR",
        "分值": momentum_score,
    })

    prior_volume = pd.to_numeric(frame["volume"], errors="coerce").shift(1).tail(20).mean()
    volume_ratio = float(latest["volume"] / prior_volume) if prior_volume and np.isfinite(prior_volume) else np.nan
    volume_score = 0.0
    if volume_reliable and np.isfinite(volume_ratio) and volume_ratio >= 1.1 and price_change != 0:
        magnitude = min(15.0, 7.5 + (volume_ratio - 1.1) * 7.5)
        volume_score = float(np.sign(price_change) * magnitude)
    if not volume_reliable:
        volume_evidence = "连续合约成交量受换月影响，未计分"
        if np.isfinite(volume_ratio):
            volume_evidence = f"连续合约量比 {volume_ratio:.2f}×，受换月影响，未计分"
    else:
        volume_evidence = "无有效成交量" if not np.isfinite(volume_ratio) else f"量比 {volume_ratio:.2f}×，本根价格{'上涨' if price_change >= 0 else '下跌'}"
    components.append({
        "维度": "成交量确认",
        "证据": volume_evidence,
        "分值": volume_score,
    })

    oi_change = np.nan
    oi_change_pct = np.nan
    oi_state = "无持仓量数据"
    oi_score = 0.0
    if "open_interest" in frame and pd.notna(latest.get("open_interest")) and pd.notna(previous.get("open_interest")):
        oi_change = float(latest["open_interest"] - previous["open_interest"])
        oi_change_pct = float(oi_change / previous["open_interest"]) if previous["open_interest"] else np.nan
        if price_change > 0 and oi_change > 0:
            oi_state, oi_score = "价格涨、OI增：新增多仓倾向", 10.0
        elif price_change < 0 and oi_change > 0:
            oi_state, oi_score = "价格跌、OI增：新增空仓倾向", -10.0
        elif price_change > 0 and oi_change < 0:
            oi_state, oi_score = "价格涨、OI降：空头回补倾向", 4.0
        elif price_change < 0 and oi_change < 0:
            oi_state, oi_score = "价格跌、OI降：多头平仓倾向", -4.0
        else:
            oi_state = "价格或OI变化不显著"
    components.append({"维度": "期货持仓", "证据": oi_state, "分值": oi_score})

    cot_score = 0.0
    cot_latest = None
    if cot_history is not None and not cot_history.empty:
        cot_latest = cot_history.iloc[-1]
        change_share = float(cot_latest["managed_net_change"] / cot_latest["open_interest"])
        cot_score = float(np.clip(change_share / 0.01 * 10, -10, 10))
        z_value = cot_latest.get("managed_net_z", np.nan)
        z_text = "无52周Z值" if pd.isna(z_value) else f"52周Z值 {z_value:+.2f}"
        cot_evidence = f"管理基金净仓周变 {cot_latest['managed_net_change']:+,.0f}张，{z_text}"
    else:
        cot_evidence = "该标的暂无可映射的CFTC分类"
    components.append({"维度": "COT持仓背景", "证据": cot_evidence, "分值": cot_score})

    total_score = float(np.clip(sum(item["分值"] for item in components), -100, 100))
    score_direction = "偏多" if total_score >= 20 else "偏空" if total_score <= -20 else "中性"
    agrees = total_score * direction_sign >= 20
    if status in ["接近触发", "进入候选区"] and agrees:
        decision = "多头候选" if direction_sign > 0 else "空头候选"
    elif status == "趋势运行" and agrees:
        decision = "趋势已运行，不追价"
    else:
        decision = "等待"

    return {
        "total_score": total_score,
        "score_direction": score_direction,
        "decision": decision,
        "volume_ratio": volume_ratio,
        "oi_change": oi_change,
        "oi_change_pct": oi_change_pct,
        "oi_state": oi_state,
        "cot_latest": cot_latest,
        "components": pd.DataFrame(components),
    }
