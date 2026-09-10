from __future__ import annotations

from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_ids}"
ACM_TERM_PREMIUM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/acmPlot_data.csv"

SERIES_META = {
    "INDPRO": ("增长", "美国工业生产", "指数"),
    "PAYEMS": ("增长", "美国非农就业", "千人"),
    "UNRATE": ("增长", "美国失业率", "%"),
    "CHNLOLITOAASTSAM": ("中国周期", "中国OECD领先指标", "指数"),
    "CPIAUCSL": ("通胀", "美国CPI", "指数"),
    "PCEPILFE": ("通胀", "美国核心PCE", "指数"),
    "T5YIE": ("市场定价", "5年盈亏平衡通胀", "%"),
    "DGS2": ("市场定价", "2年期美国国债收益率", "%"),
    "DGS10": ("市场定价", "10年期美国国债收益率", "%"),
    "DFII10": ("市场定价", "10年实际利率", "%"),
    "T10YIE": ("市场定价", "10年盈亏平衡通胀", "%"),
    "T10Y2Y": ("市场定价", "10年-2年期限利差", "%"),
    "WALCL": ("流动性", "美联储总资产", "百万美元"),
    "WTREGEN": ("流动性", "美国财政部TGA", "百万美元"),
    "RRPONTSYD": ("流动性", "隔夜逆回购RRP", "十亿美元"),
    "BAMLH0A0HYM2": ("金融条件", "美国高收益债OAS", "%"),
    "NFCI": ("金融条件", "芝加哥联储金融条件", "指数"),
    "DTWEXBGS": ("金融条件", "广义贸易加权美元", "指数"),
    "VIXCLS": ("金融条件", "VIX", "指数"),
}

ACM_SERIES_META = {
    "ACMTP10": ("利率分解", "10年ACM期限溢价", "%"),
    "ACMEXP10": ("利率分解", "10年预期短端利率成分", "%"),
}


def _read_fred_response(content: bytes) -> dict[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if content[:2] == b"PK":
        with ZipFile(BytesIO(content)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    frames.append(pd.read_csv(archive.open(name)))
    else:
        frames.append(pd.read_csv(BytesIO(content)))

    result: dict[str, pd.DataFrame] = {}
    for frame in frames:
        date_column = "observation_date"
        if date_column not in frame:
            continue
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        for column in frame.columns:
            if column == date_column or column not in SERIES_META:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            clean = pd.DataFrame({"date": dates, "value": values}).dropna()
            result[column] = clean.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return result


def fetch_macro_series() -> dict[str, pd.DataFrame]:
    series_ids = list(SERIES_META)
    result: dict[str, pd.DataFrame] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "cross-asset-monitor/0.7"})
    for start in range(0, len(series_ids), 8):
        batch = series_ids[start:start + 8]
        response = session.get(FRED_GRAPH_URL.format(series_ids=",".join(batch)), timeout=60)
        response.raise_for_status()
        result.update(_read_fred_response(response.content))
    missing = sorted(set(series_ids) - set(result))
    if missing:
        raise RuntimeError(f"FRED缺少宏观序列：{', '.join(missing)}")
    try:
        response = session.get(ACM_TERM_PREMIUM_URL, timeout=60)
        response.raise_for_status()
        result.update(parse_acm_term_premium(response.text))
    except Exception:
        # ACM is an explanatory model input; FRED state monitoring remains usable without it.
        pass
    return result


def parse_acm_term_premium(text: str) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(StringIO(text))
    required = {"RunDates", "TERMYld", "ACMFITYld"}
    if not required.issubset(frame.columns):
        raise ValueError("纽约联储ACM数据缺少必要字段")
    dates = pd.to_datetime(frame["RunDates"], format="%d-%b-%Y", errors="coerce")
    term_premium = pd.to_numeric(frame["TERMYld"], errors="coerce")
    fitted_yield = pd.to_numeric(frame["ACMFITYld"], errors="coerce")

    def clean(values: pd.Series) -> pd.DataFrame:
        return (
            pd.DataFrame({"date": dates, "value": values})
            .dropna()
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )

    return {
        "ACMTP10": clean(term_premium),
        "ACMEXP10": clean(fitted_yield - term_premium),
    }


def _series(data: dict[str, pd.DataFrame], series_id: str) -> pd.Series:
    frame = data[series_id]
    return frame.set_index("date")["value"].astype(float)


def _zscore_latest(series: pd.Series, years: int = 10) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    cutoff = clean.index.max() - pd.DateOffset(years=years)
    sample = clean[clean.index >= cutoff]
    std = float(sample.std())
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float((sample.iloc[-1] - sample.mean()) / std)


def _factor_score(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    if not clean:
        return 0.0
    return float(np.clip(np.mean(clean) / 2 * 100, -100, 100))


def _change_at(series: pd.Series, days: int, percent: bool = False) -> float:
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    target = clean.index[-1] - pd.Timedelta(days=days)
    prior = clean[clean.index <= target]
    if prior.empty:
        return np.nan
    old = float(prior.iloc[-1])
    latest = float(clean.iloc[-1])
    if percent:
        return (latest / old - 1) * 100 if old else np.nan
    return latest - old


def _state(score: float, positive: str, negative: str, neutral: str = "中性") -> str:
    if score >= 15:
        return positive
    if score <= -15:
        return negative
    return neutral


def analyze_treasury_drivers(data: dict[str, pd.DataFrame], days: int = 31) -> dict:
    required = {"DGS10", "DGS2", "DFII10", "T10YIE"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"美债驱动分解缺少序列：{', '.join(sorted(missing))}")

    ten_year = _series(data, "DGS10").dropna()
    two_year = _series(data, "DGS2").dropna()
    real_yield = _series(data, "DFII10").dropna()
    breakeven = _series(data, "T10YIE").dropna()
    changes_bp = {
        "10年名义收益率": _change_at(ten_year, days) * 100,
        "2年名义收益率": _change_at(two_year, days) * 100,
        "10年实际利率": _change_at(real_yield, days) * 100,
        "10年盈亏平衡通胀": _change_at(breakeven, days) * 100,
    }
    changes_bp["分解残差"] = (
        changes_bp["10年名义收益率"]
        - changes_bp["10年实际利率"]
        - changes_bp["10年盈亏平衡通胀"]
    )

    real_abs = abs(changes_bp["10年实际利率"])
    inflation_abs = abs(changes_bp["10年盈亏平衡通胀"])
    if real_abs >= inflation_abs + 2:
        dominant_driver = "实际利率主导"
    elif inflation_abs >= real_abs + 2:
        dominant_driver = "通胀补偿主导"
    else:
        dominant_driver = "实际利率与通胀共同驱动"

    ten_change = changes_bp["10年名义收益率"]
    two_change = changes_bp["2年名义收益率"]
    spread_change = ten_change - two_change
    if ten_change > 1 and two_change > 1:
        curve_move = "熊市陡峭化" if spread_change > 1 else "熊市平坦化"
    elif ten_change < -1 and two_change < -1:
        curve_move = "牛市陡峭化" if spread_change > 1 else "牛市平坦化"
    elif spread_change > 1:
        curve_move = "曲线陡峭化"
    elif spread_change < -1:
        curve_move = "曲线平坦化"
    else:
        curve_move = "曲线变化有限"

    term_premium = None
    if "ACMTP10" in data and "ACMEXP10" in data:
        tp = _series(data, "ACMTP10").dropna()
        expected_short = _series(data, "ACMEXP10").dropna()
        term_premium = {
            "as_of": min(tp.index[-1], expected_short.index[-1]),
            "term_premium": float(tp.iloc[-1]),
            "expected_short": float(expected_short.iloc[-1]),
            "term_premium_change_bp": _change_at(tp, max(days, 31)) * 100,
            "expected_short_change_bp": _change_at(expected_short, max(days, 31)) * 100,
            "history": pd.concat(
                [tp.rename("期限溢价"), expected_short.rename("预期短端利率路径")],
                axis=1,
            ).dropna(),
        }

    if ten_change > 1 and "实际利率" in dominant_driver:
        implication = "实际贴现率上升：通常压制长久期国债价格、黄金和高估值股票，并对美元形成支持。"
    elif ten_change > 1:
        implication = "通胀补偿上升：名义国债价格承压；黄金方向取决于实际利率是否同步上升。"
    elif ten_change < -1 and "实际利率" in dominant_driver:
        implication = "实际贴现率下降：通常支持长久期国债、黄金和高估值股票。"
    elif ten_change < -1:
        implication = "通胀补偿下降：支持名义国债，但若源于增长骤降，风险资产未必受益。"
    else:
        implication = "10年收益率变化有限，暂不构成独立的跨资产驱动。"

    return {
        "as_of": ten_year.index[-1],
        "levels": {
            "10年名义收益率": float(ten_year.iloc[-1]),
            "2年名义收益率": float(two_year.iloc[-1]),
            "10年实际利率": float(real_yield.iloc[-1]),
            "10年盈亏平衡通胀": float(breakeven.iloc[-1]),
        },
        "changes_bp": changes_bp,
        "dominant_driver": dominant_driver,
        "curve_move": curve_move,
        "spread_change_bp": spread_change,
        "implication": implication,
        "term_premium": term_premium,
    }


def analyze_macro_regime(data: dict[str, pd.DataFrame]) -> dict:
    indpro = _series(data, "INDPRO")
    payrolls = _series(data, "PAYEMS")
    unemployment = _series(data, "UNRATE")
    china_cli = _series(data, "CHNLOLITOAASTSAM")
    cpi = _series(data, "CPIAUCSL")
    core_pce = _series(data, "PCEPILFE")
    breakeven = _series(data, "T5YIE")
    two_year_yield = _series(data, "DGS2")
    ten_year_yield = _series(data, "DGS10")
    real_yield = _series(data, "DFII10")
    ten_year_breakeven = _series(data, "T10YIE")
    curve = _series(data, "T10Y2Y")
    fed_assets = _series(data, "WALCL")
    tga = _series(data, "WTREGEN")
    rrp = _series(data, "RRPONTSYD")
    hy_oas = _series(data, "BAMLH0A0HYM2")
    nfci = _series(data, "NFCI")
    dollar = _series(data, "DTWEXBGS")
    vix = _series(data, "VIXCLS")

    indpro_yoy = indpro.pct_change(12) * 100
    payroll_yoy = payrolls.pct_change(12) * 100
    cpi_yoy = cpi.pct_change(12) * 100
    core_pce_yoy = core_pce.pct_change(12) * 100
    china_momentum = china_cli.diff(3)
    fed_assets_13w = fed_assets.pct_change(13) * 100
    tga_4w = tga.pct_change(4) * 100
    rrp_1m = rrp.diff(21)
    two_year_yield_3m = two_year_yield.diff(63)
    real_yield_3m = real_yield.diff(63)
    dollar_3m = dollar.pct_change(63) * 100

    growth_score = _factor_score([
        _zscore_latest(indpro_yoy),
        _zscore_latest(payroll_yoy),
        -_zscore_latest(unemployment),
        _zscore_latest(indpro_yoy.diff(3)),
    ])
    inflation_score = _factor_score([
        _zscore_latest(cpi_yoy),
        _zscore_latest(core_pce_yoy),
        _zscore_latest(breakeven),
    ])
    liquidity_score = _factor_score([
        _zscore_latest(fed_assets_13w),
        -_zscore_latest(tga_4w),
        -_zscore_latest(rrp_1m),
    ])
    rate_tightening_score = _factor_score([
        _zscore_latest(two_year_yield_3m),
        _zscore_latest(real_yield_3m),
        _zscore_latest(real_yield),
    ])
    stress_score = _factor_score([
        _zscore_latest(hy_oas),
        _zscore_latest(nfci),
        _zscore_latest(vix),
        _zscore_latest(dollar_3m),
    ])
    china_score = _factor_score([
        _zscore_latest(china_cli),
        _zscore_latest(china_momentum),
    ])
    usd_score = _factor_score([
        _zscore_latest(dollar_3m),
    ])

    if abs(growth_score) < 15 or abs(inflation_score) < 15:
        regime = f"增长{_state(growth_score, '偏强', '偏弱')} · 通胀{_state(inflation_score, '偏热', '降温')}"
    elif growth_score >= 15 and inflation_score <= -15:
        regime = "金发姑娘"
    elif growth_score >= 15 and inflation_score >= 15:
        regime = "再通胀"
    elif growth_score <= -15 and inflation_score >= 15:
        regime = "滞胀压力"
    else:
        regime = "衰退/通缩"

    factor_scores = {
        "增长": growth_score,
        "通胀压力": inflation_score,
        "利率紧缩": rate_tightening_score,
        "流动性": liquidity_score,
        "金融压力": stress_score,
        "中国周期": china_score,
        "美元动量": usd_score,
    }
    states = {
        "增长": _state(growth_score, "偏强", "偏弱"),
        "通胀压力": _state(inflation_score, "偏热", "降温"),
        "利率紧缩": _state(rate_tightening_score, "增强", "缓和"),
        "流动性": _state(liquidity_score, "宽松", "收紧"),
        "金融压力": _state(stress_score, "升高", "偏低"),
        "中国周期": _state(china_score, "改善", "走弱"),
        "美元动量": _state(usd_score, "走强", "走弱"),
    }

    derived = {
        "INDPRO": indpro_yoy,
        "PAYEMS": payroll_yoy,
        "UNRATE": unemployment,
        "CHNLOLITOAASTSAM": china_cli,
        "CPIAUCSL": cpi_yoy,
        "PCEPILFE": core_pce_yoy,
        "T5YIE": breakeven,
        "DGS2": two_year_yield,
        "DGS10": ten_year_yield,
        "DFII10": real_yield,
        "T10YIE": ten_year_breakeven,
        "T10Y2Y": curve,
        "WALCL": fed_assets,
        "WTREGEN": tga,
        "RRPONTSYD": rrp,
        "BAMLH0A0HYM2": hy_oas,
        "NFCI": nfci,
        "DTWEXBGS": dollar,
        "VIXCLS": vix,
    }
    rows: list[dict] = []
    for series_id, series in derived.items():
        group, name, unit = SERIES_META[series_id]
        clean = series.dropna()
        rows.append({
            "series_id": series_id,
            "模块": group,
            "指标": name,
            "最新值": float(clean.iloc[-1]),
            "单位": "%同比" if series_id in {"INDPRO", "PAYEMS", "CPIAUCSL", "PCEPILFE"} else unit,
            "截至": clean.index[-1],
            "1个月变化": _change_at(clean, 31),
            "3个月变化": _change_at(clean, 93),
            "10年Z值": _zscore_latest(clean),
        })
    metrics = pd.DataFrame(rows)
    latest_market_date = max(
        _series(data, series_id).index.max()
        for series_id in ["T5YIE", "DGS2", "DGS10", "DFII10", "T10YIE", "T10Y2Y", "BAMLH0A0HYM2", "DTWEXBGS", "VIXCLS"]
    )
    return {
        "regime": regime,
        "factor_scores": factor_scores,
        "states": states,
        "confidence": {
            "增长": "中（月频）",
            "通胀压力": "中（月频+市场）",
            "利率紧缩": "高（日频）",
            "流动性": "高（日/周频）",
            "金融压力": "高（日/周频）",
            "中国周期": "低（月频代理）",
            "美元动量": "高（日频）",
        },
        "metrics": metrics,
        "series": derived,
        "treasury_drivers": {
            "1周": analyze_treasury_drivers(data, days=7),
            "1月": analyze_treasury_drivers(data, days=31),
            "3月": analyze_treasury_drivers(data, days=93),
        },
        "latest_market_date": pd.Timestamp(latest_market_date),
        "summary": (
            f"增长{states['增长']}、通胀{states['通胀压力']}、利率紧缩{states['利率紧缩']}、流动性{states['流动性']}、"
            f"金融压力{states['金融压力']}；中国周期{states['中国周期']}，美元{states['美元动量']}。"
        ),
    }


ASSET_WEIGHTS = {
    "美国宽基股指": {"增长": 0.9, "通胀压力": -0.3, "利率紧缩": -0.6, "流动性": 0.8, "金融压力": -1.0, "美元动量": -0.2},
    "纳斯达克": {"增长": 0.5, "通胀压力": -0.5, "利率紧缩": -1.1, "流动性": 1.1, "金融压力": -1.0, "美元动量": -0.3},
    "中国股指": {"增长": 0.3, "利率紧缩": -0.2, "流动性": 0.4, "金融压力": -0.6, "中国周期": 1.3, "美元动量": -0.5},
    "长期国债": {"增长": -0.9, "通胀压力": -1.1, "利率紧缩": -1.0, "流动性": 0.3, "金融压力": 0.5},
    "黄金": {"通胀压力": 0.5, "利率紧缩": -1.2, "流动性": 0.7, "金融压力": 0.5, "美元动量": -0.9},
    "白银": {"增长": 0.4, "通胀压力": 0.3, "利率紧缩": -0.7, "流动性": 0.8, "金融压力": -0.2, "中国周期": 0.3, "美元动量": -0.8},
    "铜与铝": {"增长": 0.8, "通胀压力": 0.2, "利率紧缩": -0.3, "流动性": 0.2, "金融压力": -0.6, "中国周期": 1.1, "美元动量": -0.6},
    "铁矿石": {"增长": 0.3, "利率紧缩": -0.1, "金融压力": -0.3, "中国周期": 1.4, "美元动量": -0.3},
    "碳酸锂": {"增长": 0.3, "利率紧缩": -0.2, "流动性": 0.2, "金融压力": -0.3, "中国周期": 0.8, "美元动量": -0.2},
    "原油": {"增长": 0.8, "通胀压力": 0.5, "利率紧缩": -0.2, "金融压力": -0.2, "美元动量": -0.5},
    "天然气": {"增长": 0.3, "通胀压力": 0.2, "利率紧缩": -0.1, "美元动量": -0.2},
    "农产品": {"通胀压力": 0.4, "利率紧缩": -0.1, "金融压力": 0.1, "美元动量": -0.4},
    "比特币": {"增长": 0.3, "利率紧缩": -0.9, "流动性": 1.3, "金融压力": -1.1, "美元动量": -0.7},
    "美元": {"增长": 0.2, "通胀压力": 0.2, "利率紧缩": 0.6, "流动性": -0.9, "金融压力": 0.8, "美元动量": 1.0},
    "信用债": {"增长": 0.7, "通胀压力": -0.2, "利率紧缩": -0.5, "流动性": 0.6, "金融压力": -1.2},
}

MISSING_DRIVERS = {
    "铁矿石": "仍需中国地产、钢厂利润与港口库存确认",
    "碳酸锂": "仍需现货库存、产能和排产确认",
    "原油": "仍需OPEC、库存与地缘供给确认",
    "天然气": "仍需天气、库存和LNG流量确认",
    "农产品": "仍需产区天气、库存和出口政策确认",
}


def build_asset_impact(factor_scores: dict[str, float]) -> pd.DataFrame:
    rows: list[dict] = []
    for asset, weights in ASSET_WEIGHTS.items():
        contributions = {
            factor: factor_scores.get(factor, 0.0) * weight
            for factor, weight in weights.items()
        }
        denominator = sum(abs(weight) for weight in weights.values())
        normalized_contributions = {
            factor: value / denominator for factor, value in contributions.items()
        }
        score = float(np.clip(sum(normalized_contributions.values()), -100, 100))
        if score >= 20:
            bias = "偏多"
        elif score <= -20:
            bias = "偏空"
        else:
            bias = "中性"
        leading = sorted(normalized_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:2]
        driver_text = "；".join(f"{name}{value:+.0f}" for name, value in leading)
        contribution_text = "；".join(f"{name}{value:+.1f}" for name, value in normalized_contributions.items())
        missing = MISSING_DRIVERS.get(asset, "-")
        confidence = "中" if missing == "-" else "低"
        rows.append({
            "资产": asset,
            "宏观分数": score,
            "方向先验": bias,
            "通胀贡献": normalized_contributions.get("通胀压力", 0.0),
            "利率贡献": normalized_contributions.get("利率紧缩", 0.0),
            "主要贡献": driver_text,
            "贡献拆解": contribution_text,
            "置信度": confidence,
            "尚缺确认": missing,
        })
    return pd.DataFrame(rows).sort_values("宏观分数", ascending=False).reset_index(drop=True)
