from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from intraday_data import fetch_intraday_bars
from derivatives_data import fetch_cot_history, fetch_crypto_options, fetch_term_structure
from enso_monitor import build_enso_snapshot, fetch_noaa_image, fetch_roni, fetch_weekly_sst
from fed_policy import CME_FEDWATCH_URL, CME_METHOD_URL, analyze_fed_policy
from gold_market import build_gold_comparison, fetch_gold_contract_activity, fetch_london_gold_proxy
from macro_regime import analyze_macro_regime, build_asset_impact, fetch_macro_series
from signal_engine import calculate_signal
from structure_engine import analyze_market_structure


ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "config" / "asset_universe_v2.csv"
SIGNALS_PATH = ROOT / "data" / "signals_latest.csv"
SOURCES_PATH = ROOT / "config" / "data_sources_v2.csv"
HISTORY_DIR = ROOT / "data" / "history_public"
ENVIRONMENT_PATH = ROOT / "data" / "environment_latest.csv"
ENVIRONMENT_DIR = ROOT / "data" / "environment"
CLIMATE_LATEST_PATH = ROOT / "data" / "climate_latest.csv"
CLIMATE_DAILY_PATH = ROOT / "data" / "climate" / "region_daily.csv"
MACRO_ENGINE_PATH = ROOT / "macro_regime.py"
FED_POLICY_ENGINE_PATH = ROOT / "fed_policy.py"

CATEGORY_LABELS = {
    "equity_index": "股指",
    "rates": "债券",
    "precious_metals": "贵金属",
    "industrial_metals": "工业金属",
    "industrial_materials": "工业原料",
    "battery_materials": "电池材料",
    "enso_agriculture": "ENSO农产品",
    "broad_agriculture": "农产品",
    "energy": "能源",
    "fx": "外汇",
    "crypto": "加密货币",
}
SOURCE_LABELS = {"yahoo": "Yahoo公共行情", "akshare_sina": "AKShare/新浪"}
QUALITY_LABELS = {"verified": "通过", "warning": "注意", "failed": "暂停"}
DIRECTION_LABELS = {"long": "多", "short": "空"}
TIMEFRAME_LABELS = {"1m": "1分钟", "5m": "5分钟", "1h": "1小时", "4h": "4小时", "1d": "日线"}


@st.cache_data
def load_data(version: tuple[int, int, int, int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(UNIVERSE_PATH)
    signals = pd.read_csv(SIGNALS_PATH, parse_dates=["trade_date"])
    sources = pd.read_csv(SOURCES_PATH)
    environment = pd.read_csv(ENVIRONMENT_PATH, parse_dates=["as_of"])
    climate = pd.read_csv(CLIMATE_LATEST_PATH, parse_dates=["as_of"])
    data = universe.merge(signals, on="asset_id", how="left", validate="one_to_one", suffixes=("", "_signal"))
    data["类别"] = data["category"].map(CATEGORY_LABELS)
    data["方向"] = data["direction"].map(DIRECTION_LABELS)
    data["数据源"] = data["price_source"].map(SOURCE_LABELS)
    data["质量"] = data["data_status"].map(QUALITY_LABELS)
    data["候选"] = data["status"].isin(["接近触发", "进入候选区"])
    climate_context = {
        asset_id: " / ".join(f"{row.region_cn}:{row.state}" for row in group.itertuples())
        for asset_id, group in climate.groupby("related_assets")
    }
    data["气候背景"] = data["asset_id"].map(climate_context).fillna("-")
    return data, sources, environment, climate


@st.cache_data
def load_history(asset_id: str, version: int) -> pd.DataFrame:
    return pd.read_csv(HISTORY_DIR / f"{asset_id}.csv", parse_dates=["date"])


@st.cache_data(ttl=60, show_spinner=False)
def load_intraday_history(symbol: str, source: str, interval: str, timezone: str) -> pd.DataFrame:
    return fetch_intraday_bars(symbol, source, interval, timezone)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cot_history(asset_id: str) -> pd.DataFrame:
    return fetch_cot_history(asset_id)


@st.cache_data(ttl=900, show_spinner=False)
def load_term_structure(asset_id: str) -> pd.DataFrame:
    return fetch_term_structure(asset_id)


@st.cache_data(ttl=300, show_spinner=False)
def load_crypto_options(asset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_crypto_options(asset_id)


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def load_london_gold_proxy() -> pd.DataFrame:
    return fetch_london_gold_proxy()


@st.cache_data(ttl=900, max_entries=2, show_spinner=False)
def load_gold_contract_activity() -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_gold_contract_activity()


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def load_macro_regime(version: int) -> dict:
    return analyze_macro_regime(fetch_macro_series())


@st.cache_data(ttl=900, max_entries=2, show_spinner=False)
def load_fed_policy(version: int) -> dict:
    return analyze_fed_policy()


@st.cache_data(ttl=21600, max_entries=2, show_spinner=False)
def load_enso_monitor(version: int) -> dict:
    weekly, weekly_cached, weekly_fetched_at = fetch_weekly_sst()
    roni, roni_cached, roni_fetched_at = fetch_roni()
    return {
        "weekly": weekly,
        "roni": roni,
        "snapshot": build_enso_snapshot(weekly),
        "cached": weekly_cached or roni_cached,
        "fetched_at": min(weekly_fetched_at, roni_fetched_at),
    }


@st.cache_data(ttl=21600, max_entries=8, show_spinner=False)
def load_enso_image(image_id: str, version: int) -> tuple[bytes, bool, object]:
    return fetch_noaa_image(image_id)


@st.cache_data
def load_environment_history(indicator_id: str, version: int) -> pd.DataFrame:
    frame = pd.read_csv(ENVIRONMENT_DIR / f"{indicator_id}.csv", parse_dates=["date"])
    value_column = {"vix": "close", "oni": "anomaly", "hy_oas": "value"}[indicator_id]
    return frame[["date", value_column]].rename(columns={value_column: "value"})


@st.cache_data
def load_climate_history(region_id: str, version: int) -> pd.DataFrame:
    frame = pd.read_csv(CLIMATE_DAILY_PATH, parse_dates=["date"])
    frame = frame[frame["region_id"] == region_id].sort_values("date").copy()
    frame["rain_30d_mm"] = frame["precip_mm"].rolling(30, min_periods=25).sum()
    frame["root_14d"] = frame["root_wetness"].rolling(14, min_periods=10).mean()
    return frame


def price(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    return f"{value:,.3f}"


st.set_page_config(page_title="跨资产结构监测", page_icon=":material/query_stats:", layout="wide")
data_version = (
    SIGNALS_PATH.stat().st_mtime_ns,
    UNIVERSE_PATH.stat().st_mtime_ns,
    SOURCES_PATH.stat().st_mtime_ns,
    ENVIRONMENT_PATH.stat().st_mtime_ns,
    CLIMATE_LATEST_PATH.stat().st_mtime_ns,
)
data, sources, environment, climate = load_data(data_version)

with st.sidebar:
    st.subheader("筛选")
    selected_status = st.pills(
        "信号状态",
        ["接近触发", "进入候选区", "趋势运行", "等待"],
        default=["接近触发", "进入候选区", "趋势运行", "等待"],
        selection_mode="multi",
    )
    selected_categories = st.multiselect(
        "资产类别",
        sorted(data["类别"].dropna().unique()),
        default=[],
        placeholder="全部类别",
    )
    selected_sources = st.multiselect(
        "数据源",
        sorted(data["数据源"].dropna().unique()),
        default=[],
        placeholder="全部公开源",
    )
    include_warnings = st.toggle("显示质量提醒", value=True)
    st.caption("免费公开数据 · 宏观优先版 v1.0")

filtered = data[data["status"].isin(selected_status or [])].copy()
if selected_categories:
    filtered = filtered[filtered["类别"].isin(selected_categories)]
if selected_sources:
    filtered = filtered[filtered["数据源"].isin(selected_sources)]
if not include_warnings:
    filtered = filtered[filtered["data_status"] == "verified"]
filtered = filtered.reset_index(drop=True)

st.title("跨资产结构监测")
latest_date = data["trade_date"].max()
st.caption(f"宏观状态优先 · 日线雷达截至 {latest_date:%Y-%m-%d} · 详情页支持1分钟至日线")

macro_tab, radar_tab, detail_tab, environment_tab, source_tab = st.tabs(
    ["宏观局势", "信号雷达", "结构详情", "环境状态", "数据源"]
)

with macro_tab:
    try:
        macro = load_macro_regime(MACRO_ENGINE_PATH.stat().st_mtime_ns)
    except Exception as exc:
        macro = None
        st.error(f"宏观数据暂时读取失败：{exc}")
    if macro is not None:
        scores = macro["factor_scores"]
        states = macro["states"]
        confidence = macro["confidence"]
        st.subheader("宏观状态")
        with st.container(horizontal=True):
            st.metric("当前象限", macro["regime"], border=True)
            st.metric("增长", f"{scores['增长']:+.0f}", states["增长"], border=True)
            st.metric("通胀压力", f"{scores['通胀压力']:+.0f}", states["通胀压力"], border=True)
            st.metric("利率紧缩", f"{scores['利率紧缩']:+.0f}", states["利率紧缩"], border=True)
            st.metric("流动性", f"{scores['流动性']:+.0f}", states["流动性"], border=True)
            st.metric("金融压力", f"{scores['金融压力']:+.0f}", states["金融压力"], border=True)
        st.info(macro["summary"] + " 宏观状态只提供方向先验，资产价格结构仍决定是否交易。")

        macro_view = st.segmented_control(
            "宏观模块",
            options=["宏观状态", "流动性", "市场定价", "驱动拆解", "美联储预期", "资产影响"],
            default="宏观状态",
        ) or "宏观状态"
        factor_frame = pd.DataFrame({
            "因子": list(scores),
            "分数": list(scores.values()),
            "状态": [states[name] for name in scores],
            "置信度": [confidence[name] for name in scores],
        })
        factor_frame["方向"] = np.where(factor_frame["分数"] >= 0, "正", "负")

        if macro_view == "宏观状态":
            factor_chart = alt.Chart(factor_frame).mark_bar(size=24).encode(
                x=alt.X("分数:Q", title="标准化宏观分数", scale=alt.Scale(domain=[-100, 100])),
                y=alt.Y("因子:N", title=None, sort=None),
                color=alt.Color(
                    "方向:N",
                    legend=None,
                    scale=alt.Scale(domain=["正", "负"], range=["#2F6B5F", "#C44F4A"]),
                ),
                tooltip=["因子", alt.Tooltip("分数:Q", format="+.1f"), "状态", "置信度"],
            ).properties(height=260, width="container")
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8B9390").encode(x="x:Q")
            st.altair_chart(factor_chart + zero, width="stretch")
            overview = macro["metrics"][
                macro["metrics"]["模块"].isin(["增长", "通胀", "中国周期"])
            ].copy()
            st.dataframe(
                overview[["模块", "指标", "最新值", "单位", "截至", "1个月变化", "3个月变化", "10年Z值"]],
                hide_index=True,
                column_config={
                    "截至": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "最新值": st.column_config.NumberColumn(format="%.2f"),
                    "1个月变化": st.column_config.NumberColumn(format="%+.2f"),
                    "3个月变化": st.column_config.NumberColumn(format="%+.2f"),
                    "10年Z值": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.caption("分数是相对过去10年的标准化状态，不是未来收益预测；月度数据按各自发布日期更新。")
        elif macro_view == "流动性":
            liquidity = macro["metrics"][macro["metrics"]["模块"].isin(["流动性", "金融条件"])].copy()
            st.dataframe(
                liquidity[["模块", "指标", "最新值", "单位", "截至", "1个月变化", "3个月变化", "10年Z值"]],
                hide_index=True,
                column_config={
                    "截至": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "最新值": st.column_config.NumberColumn(format="%.2f"),
                    "1个月变化": st.column_config.NumberColumn(format="%+.2f"),
                    "3个月变化": st.column_config.NumberColumn(format="%+.2f"),
                    "10年Z值": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.caption("流动性分数综合美联储资产负债表、TGA、RRP和实际利率变化；金融压力综合信用利差、NFCI、VIX和美元。")
        elif macro_view == "市场定价":
            pricing_ids = ["T5YIE", "DGS2", "DGS10", "DFII10", "T10YIE", "T10Y2Y", "BAMLH0A0HYM2", "DTWEXBGS", "VIXCLS"]
            labels = macro["metrics"].set_index("series_id")["指标"].to_dict()
            pricing_id = st.selectbox(
                "市场指标",
                pricing_ids,
                format_func=lambda item: labels[item],
            )
            pricing_series = macro["series"][pricing_id].dropna().tail(756).rename("value").reset_index()
            pricing_chart = alt.Chart(pricing_series).mark_line(color="#3D6380", strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("value:Q", title=labels[pricing_id], scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("value:Q", title="数值", format=".3f")],
            ).properties(height=320, width="container")
            st.altair_chart(pricing_chart, width="stretch")
            pricing_table = macro["metrics"][macro["metrics"]["series_id"].isin(pricing_ids)].copy()
            st.dataframe(
                pricing_table[["指标", "最新值", "单位", "截至", "1个月变化", "3个月变化", "10年Z值"]],
                hide_index=True,
                column_config={
                    "截至": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "最新值": st.column_config.NumberColumn(format="%.3f"),
                    "1个月变化": st.column_config.NumberColumn(format="%+.3f"),
                    "3个月变化": st.column_config.NumberColumn(format="%+.3f"),
                    "10年Z值": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.caption("市场价格比月度宏观数据更快，但反映的是已被交易的预期，需要与经济事实分开阅读。")
        elif macro_view == "驱动拆解":
            st.subheader("10年期美国国债收益率")
            driver_window = st.segmented_control(
                "观察窗口",
                options=["1周", "1月", "3月"],
                default="1月",
            ) or "1月"
            treasury = macro["treasury_drivers"][driver_window]
            levels = treasury["levels"]
            changes_bp = treasury["changes_bp"]
            term = treasury["term_premium"]

            with st.container(horizontal=True):
                st.metric(
                    "10年收益率",
                    f"{levels['10年名义收益率']:.2f}%",
                    f"{driver_window} {changes_bp['10年名义收益率']:+.0f}bp",
                    delta_color="off",
                    border=True,
                )
                st.metric(
                    f"实际利率贡献 · 当前{levels['10年实际利率']:.2f}%",
                    f"{changes_bp['10年实际利率']:+.0f}bp",
                    border=True,
                )
                st.metric(
                    f"通胀补偿贡献 · 当前{levels['10年盈亏平衡通胀']:.2f}%",
                    f"{changes_bp['10年盈亏平衡通胀']:+.0f}bp",
                    border=True,
                )
                st.metric("主导项", treasury["dominant_driver"], border=True)
                st.metric(
                    "曲线形态",
                    treasury["curve_move"],
                    f"2s10s变化 {treasury['spread_change_bp']:+.0f}bp",
                    delta_color="off",
                    border=True,
                )

            st.info(treasury["implication"])

            contribution = pd.DataFrame({
                "分解项": ["实际利率", "盈亏平衡通胀", "残差"],
                "变化bp": [
                    changes_bp["10年实际利率"],
                    changes_bp["10年盈亏平衡通胀"],
                    changes_bp["分解残差"],
                ],
            })
            contribution["方向"] = np.where(contribution["变化bp"] >= 0, "推高收益率", "压低收益率")
            contribution_chart = alt.Chart(contribution).mark_bar(size=28).encode(
                x=alt.X("变化bp:Q", title=f"{driver_window}变化（bp）"),
                y=alt.Y("分解项:N", title=None, sort=None),
                color=alt.Color(
                    "方向:N",
                    legend=None,
                    scale=alt.Scale(domain=["推高收益率", "压低收益率"], range=["#C44536", "#1769AA"]),
                ),
                tooltip=["分解项", alt.Tooltip("变化bp:Q", format="+.1f")],
            ).properties(height=190, width="container")
            contribution_zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#69747C").encode(x="x:Q")
            st.altair_chart(contribution_chart + contribution_zero, width="stretch")
            st.caption("市场恒等式近似：10年名义收益率 = 10年TIPS实际利率 + 10年盈亏平衡通胀；残差来自报价时点、流动性和口径差异。")

            rate_history = pd.concat(
                [
                    macro["series"]["DGS10"].rename("10年名义收益率"),
                    macro["series"]["DFII10"].rename("10年实际利率"),
                    macro["series"]["T10YIE"].rename("10年盈亏平衡通胀"),
                ],
                axis=1,
            ).dropna().tail(504).reset_index().melt(
                id_vars=["date"], var_name="分解项", value_name="利率"
            )
            rate_chart = alt.Chart(rate_history).mark_line(strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("利率:Q", title="%", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "分解项:N",
                    scale=alt.Scale(
                        domain=["10年名义收益率", "10年实际利率", "10年盈亏平衡通胀"],
                        range=["#202A32", "#1769AA", "#D17A22"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="日期"),
                    alt.Tooltip("分解项:N"),
                    alt.Tooltip("利率:Q", title="利率", format=".2f"),
                ],
            ).properties(height=300, width="container")
            st.altair_chart(rate_chart, width="stretch")

            if term is not None:
                st.subheader("政策路径与期限溢价")
                with st.container(horizontal=True):
                    st.metric(
                        "ACM期限溢价",
                        f"{term['term_premium']:.2f}%",
                        f"月度可比 {term['term_premium_change_bp']:+.0f}bp",
                        delta_color="off",
                        border=True,
                    )
                    st.metric(
                        "预期短端利率成分",
                        f"{term['expected_short']:.2f}%",
                        f"月度可比 {term['expected_short_change_bp']:+.0f}bp",
                        delta_color="off",
                        border=True,
                    )
                    st.metric("模型截至", f"{term['as_of']:%Y-%m-%d}", border=True)

                if term["expected_short_change_bp"] > 2 and term["term_premium_change_bp"] < -2:
                    st.info("预期短端利率路径上修、期限溢价下降：政策预期在推高收益率，而久期风险补偿正在抵消部分上行。")
                elif term["expected_short_change_bp"] < -2 and term["term_premium_change_bp"] > 2:
                    st.info("预期短端利率路径下修、期限溢价上升：降息预期在压低收益率，但长期供给或不确定性要求更高补偿。")
                elif abs(term["term_premium_change_bp"]) > abs(term["expected_short_change_bp"]):
                    st.info("期限溢价变化占主导：更应检查财政供给、拍卖结果、波动率和长期债券需求，而不是只看美联储。")
                else:
                    st.info("预期短端利率路径变化占主导：更应检查通胀、就业数据和FedWatch重定价。")

                term_history = term["history"].tail(36).reset_index().melt(
                    id_vars=["date"], var_name="成分", value_name="利率"
                )
                term_chart = alt.Chart(term_history).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("利率:Q", title="ACM估算 %", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "成分:N",
                        scale=alt.Scale(
                            domain=["期限溢价", "预期短端利率路径"],
                            range=["#9C4F78", "#087F8C"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="月份"),
                        alt.Tooltip("成分:N"),
                        alt.Tooltip("利率:Q", title="估算", format=".2f"),
                    ],
                ).properties(height=280, width="container")
                st.altair_chart(term_chart, width="stretch")
                st.caption("纽约联储ACM月度模型估计。期限溢价不可直接观测，适合解释中期变化，不适合分钟级择时。")
            else:
                st.warning("纽约联储ACM期限溢价暂时不可用；实际利率/通胀分解仍可正常使用。")

            st.subheader("驱动检查表")
            driver_map = pd.DataFrame({
                "驱动层": ["增长与就业", "通胀", "美联储路径", "财政与供给", "需求与风险偏好", "市场技术面"],
                "重点观察": [
                    "非农、失业率、ISM、零售销售",
                    "CPI、核心PCE、工资、油价",
                    "FedWatch概率及其变化、2年收益率",
                    "季度融资计划、长端拍卖规模、财政赤字",
                    "拍卖尾差、间接投标比例、海外与银行需求",
                    "波动率、交易商持仓、回购与杠杆压力",
                ],
                "主要传导": [
                    "增长上修通常推高预期短端利率和中性利率",
                    "通胀上修推高盈亏平衡通胀，也可能推高实际利率",
                    "更鹰的政策路径首先影响2年，再传至10年",
                    "净久期供给增加通常抬高期限溢价",
                    "避险买盘压低收益率；需求疲弱抬高期限溢价",
                    "去杠杆或流动性恶化可造成短期收益率超调",
                ],
                "当前状态": ["部分量化", "已量化", "已接入", "待接入事件层", "待接入拍卖层", "部分量化"],
            })
            st.dataframe(driver_map, hide_index=True)
            st.caption("驱动表表达的是条件关系，不是永恒相关性。同一数据在不同通胀制度、政策反应函数和持仓环境下，价格反应可能相反。")
        elif macro_view == "美联储预期":
            try:
                policy = load_fed_policy(FED_POLICY_ENGINE_PATH.stat().st_mtime_ns)
            except Exception as exc:
                policy = None
                st.error(f"联邦基金期货暂时读取失败：{exc}")
            if policy is not None:
                available_windows = list(policy["expectation_changes"])
                change_window = st.segmented_control(
                    "变化窗口",
                    options=available_windows,
                    default="1周" if "1周" in available_windows else available_windows[0],
                )
                change_window = change_window or available_windows[0]
                changes = policy["expectation_changes"][change_window]
                base_label = f"{changes['base_date']:%m-%d}至{policy['quote_date']:%m-%d}"

                front_change = changes["next_expected_move_bp"]
                year_end_change = changes["year_end_midpoint_bp"]
                if front_change >= 2 and year_end_change >= 2:
                    repricing_state = "全面转鹰"
                elif front_change <= -2 and year_end_change <= -2:
                    repricing_state = "全面转鸽"
                elif front_change >= 1.5 and year_end_change <= -1.5:
                    repricing_state = "近端鹰、远端鸽"
                elif front_change <= -1.5 and year_end_change >= 1.5:
                    repricing_state = "近端鸽、远端鹰"
                elif year_end_change > 1:
                    repricing_state = "远端小幅转鹰"
                elif year_end_change < -1:
                    repricing_state = "远端小幅转鸽"
                else:
                    repricing_state = "变化有限"

                st.subheader("政策预期变化")
                with st.container(horizontal=True):
                    st.metric("观察区间", base_label, border=True)
                    st.metric("加息概率 Δ", f"{changes['hike_probability_pp']:+.1f}pp", border=True)
                    st.metric("维持概率 Δ", f"{changes['hold_probability_pp']:+.1f}pp", border=True)
                    st.metric("下次会议隐含 Δ", f"{front_change:+.1f}bp", border=True)
                    st.metric("年底利率 Δ", f"{year_end_change:+.1f}bp", border=True)
                    st.metric("重定价状态", repricing_state, border=True)

                if "鹰" in repricing_state and "鸽" not in repricing_state:
                    st.warning(
                        f"{change_window}窗口呈{repricing_state}：下次会议隐含变动 {front_change:+.1f}bp，"
                        f"年底利率 {year_end_change:+.1f}bp。整体边际上偏空黄金、长久期国债和高估值股，偏多美元。"
                    )
                elif "鸽" in repricing_state and "鹰" not in repricing_state:
                    st.info(
                        f"{change_window}窗口呈{repricing_state}：下次会议隐含变动 {front_change:+.1f}bp，"
                        f"年底利率 {year_end_change:+.1f}bp。整体边际上偏多黄金和长久期资产、偏空美元；"
                        "仍需区分通胀降温与衰退冲击。"
                    )
                elif "鹰" in repricing_state and "鸽" in repricing_state:
                    st.info(
                        f"{change_window}窗口出现期限分化：{repricing_state}。"
                        "这类曲线扭转不宜直接翻译成单一资产方向，应分别观察美元、实际利率和风险资产确认。"
                    )
                else:
                    st.info(
                        f"{change_window}窗口政策预期变化有限，暂不构成独立的宏观交易驱动。"
                    )

                expectation_history = policy["expectation_history"].melt(
                    id_vars=["date"],
                    value_vars=["cut_probability", "hold_probability", "hike_probability"],
                    var_name="action",
                    value_name="probability",
                )
                expectation_history["政策结果"] = expectation_history["action"].map(
                    {
                        "cut_probability": "降息",
                        "hold_probability": "维持",
                        "hike_probability": "加息",
                    }
                )
                probability_history_chart = alt.Chart(expectation_history).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("probability:Q", title="下次会议概率 (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "政策结果:N",
                        scale=alt.Scale(
                            domain=["降息", "维持", "加息"],
                            range=["#3D6380", "#7A8582", "#B24A3A"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("date:T", title="日期"),
                        alt.Tooltip("政策结果:N"),
                        alt.Tooltip("probability:Q", title="概率", format=".1f"),
                    ],
                ).properties(height=300, width="container")
                st.altair_chart(probability_history_chart, width="stretch")
                st.caption(
                    f"CME官方对比点：当前、1日前、1周前和1月前，对 {policy['next_meeting']:%Y-%m-%d} 会议结果的重新定价。"
                )

                st.subheader("ZQ期限曲线变化（辅助）")
                curve = policy["curve"].copy()
                window_fields = {
                    "1日": ("prior_rate", "one_day_bp"),
                    "1周": ("week_rate", "one_week_bp"),
                    "1月": ("month_rate", "one_month_bp"),
                }
                base_rate_field, shift_field = window_fields[change_window]
                curve_long = curve.melt(
                    id_vars=["month_label"],
                    value_vars=["implied_rate", base_rate_field],
                    var_name="series",
                    value_name="rate",
                ).dropna()
                curve_long["路径"] = curve_long["series"].map(
                    {"implied_rate": "当前", base_rate_field: f"{change_window}前"}
                )
                curve_chart = alt.Chart(curve_long).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("month_label:N", title="合约月份", sort=curve["month_label"].tolist()),
                    y=alt.Y("rate:Q", title="隐含月均EFFR (%)", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "路径:N",
                        scale=alt.Scale(domain=["当前", f"{change_window}前"], range=["#B24A3A", "#7A8582"]),
                    ),
                    strokeDash=alt.StrokeDash(
                        "路径:N",
                        scale=alt.Scale(domain=["当前", f"{change_window}前"], range=[[1, 0], [5, 4]]),
                    ),
                    tooltip=[
                        alt.Tooltip("month_label:N", title="合约月"),
                        alt.Tooltip("路径:N"),
                        alt.Tooltip("rate:Q", title="隐含利率", format=".3f"),
                    ],
                ).properties(height=320, width="container")
                st.altair_chart(curve_chart, width="stretch")

                curve_shift = curve[["month_label", shift_field]].rename(columns={shift_field: "shift_bp"})
                curve_shift["方向"] = np.where(curve_shift["shift_bp"] >= 0, "转鹰", "转鸽")
                shift_chart = alt.Chart(curve_shift).mark_bar().encode(
                    x=alt.X("month_label:N", title="合约月份", sort=curve["month_label"].tolist()),
                    y=alt.Y("shift_bp:Q", title=f"{change_window}隐含利率变化 (bp)"),
                    color=alt.Color(
                        "方向:N",
                        scale=alt.Scale(domain=["转鹰", "转鸽"], range=["#B24A3A", "#3D6380"]),
                    ),
                    tooltip=[
                        alt.Tooltip("month_label:N", title="合约月"),
                        alt.Tooltip("shift_bp:Q", title="变化bp", format="+.1f"),
                        alt.Tooltip("方向:N"),
                    ],
                ).properties(height=230, width="container")
                shift_zero = alt.Chart(pd.DataFrame({"shift_bp": [0]})).mark_rule(color="#7A8582").encode(
                    y="shift_bp:Q"
                )
                st.altair_chart(shift_chart + shift_zero, width="stretch")
                st.caption(
                    "Yahoo逐月ZQ延迟行情仅用于观察变化分布：柱形高于零表示该期限隐含利率上修，低于零表示下修。"
                )

                st.subheader("当前概率与路径")
                st.caption(
                    f"目标区间 {policy['target_lower']:.2f}-{policy['target_upper']:.2f}% · "
                    f"EFFR {policy['effr']:.2f}% · 下次会议 {policy['next_meeting']:%Y-%m-%d} · "
                    f"年底隐含中点 {policy['year_end_midpoint']:.2f}%"
                )
                meeting_dates = policy["meeting_summary"]["meeting_date"].tolist()
                selected_meeting = st.selectbox(
                    "会议概率分布",
                    meeting_dates,
                    format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
                )
                meeting_probability = policy["probabilities"][
                    policy["probabilities"]["meeting_date"] == pd.Timestamp(selected_meeting)
                ].copy()
                probability_chart = alt.Chart(meeting_probability).mark_bar(size=30).encode(
                    x=alt.X("target_range:N", title="会后目标区间", sort=None),
                    y=alt.Y("probability:Q", title="概率 (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.condition(
                        alt.datum.probability == meeting_probability["probability"].max(),
                        alt.value("#2F6B5F"),
                        alt.value("#9DB3AD"),
                    ),
                    tooltip=[
                        alt.Tooltip("target_range:N", title="目标区间"),
                        alt.Tooltip("probability:Q", title="概率", format=".1f"),
                    ],
                ).properties(height=280, width="container")
                st.altair_chart(probability_chart, width="stretch")

                meeting_table = policy["meeting_summary"].rename(
                    columns={
                        "meeting_date": "会议",
                        "expected_move_bp": "本次隐含变动bp",
                        "expected_cumulative_bp": "累计隐含变动bp",
                        "most_likely_range": "最高概率目标区间",
                        "top_probability": "最高单档概率",
                        "cut_probability": "累计降息概率",
                        "hold_probability": "累计不变概率",
                        "hike_probability": "累计加息概率",
                    }
                )
                st.dataframe(
                    meeting_table,
                    hide_index=True,
                    column_config={
                        "会议": st.column_config.DateColumn(format="YYYY-MM-DD"),
                        "本次隐含变动bp": st.column_config.NumberColumn(format="%+.1f"),
                        "累计隐含变动bp": st.column_config.NumberColumn(format="%+.1f"),
                        "最高单档概率": st.column_config.NumberColumn(format="%.1f%%"),
                        "累计降息概率": st.column_config.NumberColumn(format="%.1f%%"),
                        "累计不变概率": st.column_config.NumberColumn(format="%.1f%%"),
                        "累计加息概率": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )
                with st.container(horizontal=True):
                    st.link_button("CME FedWatch 官方值", CME_FEDWATCH_URL, icon=":material/open_in_new:")
                    st.link_button("CME 计算方法", CME_METHOD_URL, icon=":material/function:")
                if policy["probability_source"] == "CME FedWatch官网":
                    st.caption(
                        f"概率来源：CME FedWatch官网，数据截至 {policy['cme_as_of_ct']} · "
                        f"Yahoo ZQ曲线截至 {policy['quote_date']:%Y-%m-%d} · "
                        f"EFFR截至 {policy['effr_date']:%Y-%m-%d} · 会议日历：{policy['calendar_source']}。"
                    )
                else:
                    st.warning(
                        f"CME页面暂不可用，当前概率为开源复算降级值：{policy['cme_error'] or '未知原因'}。"
                        "下单前请在CME官方页面核验。"
                    )
        else:
            impact = build_asset_impact(scores)
            impact["方向"] = np.where(impact["宏观分数"] >= 0, "正", "负")
            gold_impact = impact[impact["资产"] == "黄金"].iloc[0]
            st.info(
                f"黄金当前宏观先验 {gold_impact['宏观分数']:+.1f}："
                f"通胀贡献 {gold_impact['通胀贡献']:+.1f}，利率紧缩贡献 {gold_impact['利率贡献']:+.1f}。"
                "两条路径分别计算，最终再与美元、流动性和避险需求合并。"
            )
            impact_chart = alt.Chart(impact).mark_bar(size=18).encode(
                x=alt.X("宏观分数:Q", title="方向先验", scale=alt.Scale(domain=[-100, 100])),
                y=alt.Y("资产:N", title=None, sort="-x"),
                color=alt.Color(
                    "方向:N",
                    legend=None,
                    scale=alt.Scale(domain=["正", "负"], range=["#2F6B5F", "#C44F4A"]),
                ),
                tooltip=["资产", alt.Tooltip("宏观分数:Q", format="+.1f"), "方向先验", "主要贡献", "置信度"],
            ).properties(height=430, width="container")
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8B9390").encode(x="x:Q")
            st.altair_chart(impact_chart + zero, width="stretch")
            st.dataframe(
                impact[["资产", "宏观分数", "方向先验", "通胀贡献", "利率贡献", "主要贡献", "贡献拆解", "置信度", "尚缺确认"]],
                hide_index=True,
                column_config={
                    "宏观分数": st.column_config.NumberColumn(format="%+.1f"),
                    "通胀贡献": st.column_config.NumberColumn(format="%+.1f"),
                    "利率贡献": st.column_config.NumberColumn(format="%+.1f"),
                    "贡献拆解": st.column_config.TextColumn(width="large"),
                },
            )
            st.caption("资产影响是条件映射。得分接近零表示宏观证据相互抵消；低置信度资产必须等待库存、供给或天气数据确认。")
        st.caption(f"市场数据最新截至 {macro['latest_market_date']:%Y-%m-%d} · FRED与OECD公开序列 · 缓存1小时")

with radar_tab:
    with st.container(horizontal=True):
        st.metric("监测标的", f"{len(filtered)} / {len(data)}", border=True)
        st.metric("候选区", f"{filtered['候选'].sum()}", border=True)
        st.metric("多 / 空", f"{(filtered['direction'] == 'long').sum()} / {(filtered['direction'] == 'short').sum()}", border=True)
        st.metric("质量提醒", f"{(filtered['data_status'] == 'warning').sum()}", border=True)
    radar = filtered[
        [
            "name_cn", "类别", "方向", "status", "气候背景", "close", "b", "zone_low", "zone_high",
            "distance_to_b_atr", "delta_source", "质量", "数据源",
        ]
    ].rename(columns={
        "name_cn": "标的", "status": "状态", "close": "收盘", "b": "B",
        "zone_low": "候选区下沿", "zone_high": "候选区上沿",
        "distance_to_b_atr": "距B/ATR", "delta_source": "Delta来源",
    })
    event = st.dataframe(
        radar,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "标的": st.column_config.TextColumn(pinned=True),
            "收盘": st.column_config.NumberColumn(format="%.3f"),
            "B": st.column_config.NumberColumn(format="%.3f"),
            "候选区下沿": st.column_config.NumberColumn(format="%.3f"),
            "候选区上沿": st.column_config.NumberColumn(format="%.3f"),
            "距B/ATR": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if event.selection.rows:
        st.session_state["selected_asset_id"] = filtered.iloc[event.selection.rows[0]]["asset_id"]

with detail_tab:
    default_id = st.session_state.get("selected_asset_id", filtered.iloc[0]["asset_id"] if not filtered.empty else data.iloc[0]["asset_id"])
    options = data["asset_id"].tolist()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        selected_id = st.selectbox(
            "标的",
            options,
            index=options.index(default_id) if default_id in options else 0,
            format_func=lambda asset_id: data.set_index("asset_id").loc[asset_id, "name_cn"],
        )
        timeframe = st.segmented_control(
            "K线周期",
            options=list(TIMEFRAME_LABELS),
            default="1d",
            format_func=lambda item: TIMEFRAME_LABELS[item],
        ) or "1d"
        target_display = st.segmented_control(
            "目标线",
            options=["核心位", "含推演目标"],
            default="核心位",
        )
        refresh_clicked = st.button("刷新", icon=":material/refresh:", disabled=timeframe == "1d")
    selected = data.set_index("asset_id").loc[selected_id]
    history_path = HISTORY_DIR / f"{selected_id}.csv"
    if refresh_clicked:
        load_intraday_history.clear()
    try:
        if timeframe == "1d":
            full_history = load_history(selected_id, history_path.stat().st_mtime_ns)
        else:
            with st.spinner(f"读取{TIMEFRAME_LABELS[timeframe]}行情..."):
                full_history = load_intraday_history(
                    str(selected["public_symbol"]),
                    str(selected["price_source"]),
                    timeframe,
                    str(selected["timezone"]),
                )
        model = calculate_signal(full_history)
    except Exception as exc:
        st.error(f"{selected['name_cn']}的{TIMEFRAME_LABELS[timeframe]}数据暂不可用：{exc}")
        st.caption("免费分钟数据存在回溯长度和交易所覆盖限制；页面不会用日线替代分钟线。")
        st.stop()

    is_long = model["direction"] == "long"
    sign = 1.0 if is_long else -1.0
    point_prefix = f"point_{selected_id}_{timeframe}"
    point_keys = {
        "b": f"{point_prefix}_b",
        "delta": f"{point_prefix}_delta",
        "invalidation": f"{point_prefix}_invalidation",
    }
    reset_points = st.button("恢复模型点位", icon=":material/restart_alt:")
    for field, key in point_keys.items():
        if reset_points or key not in st.session_state:
            st.session_state[key] = float(model[field])

    value_format = "%.4f" if abs(float(model["close"])) < 100 else "%.2f"
    point_step = max(float(model["atr14"]) / 20, 0.0001)
    with st.container(horizontal=True):
        custom_b = st.number_input("B", key=point_keys["b"], step=point_step, format=value_format)
        custom_delta = st.number_input(
            "Delta",
            key=point_keys["delta"],
            min_value=point_step / 100,
            step=point_step,
            format=value_format,
            help="Delta = |T1 − B|。先沿交易方向选择第一个主要结构位T1；优先检查其距离是否约为3–8个ATR，并以接近5个ATR作为候选尺度，而不是固定使用5×ATR。",
        )
        custom_invalidation = st.number_input(
            "失效位", key=point_keys["invalidation"], step=point_step, format=value_format
        )

    custom_t1 = float(custom_b + sign * custom_delta)
    custom_t2 = float(custom_b + sign * 2 * custom_delta)
    custom_t3 = float(custom_b + sign * 3 * custom_delta)
    atr = float(model["atr14"])
    close = float(full_history.iloc[-1]["close"])
    if is_long:
        zone_low, zone_high = custom_b - 0.25 * atr, custom_b + 0.50 * atr
    else:
        zone_low, zone_high = custom_b - 0.50 * atr, custom_b + 0.25 * atr
    signed_distance = sign * (close - custom_b) / atr
    if -0.5 <= signed_distance < 0:
        display_status = "接近触发"
    elif 0 <= signed_distance <= 0.75:
        display_status = "进入候选区"
    elif signed_distance > 0.75:
        display_status = "趋势运行"
    else:
        display_status = "等待"

    cot_history = pd.DataFrame()
    cot_error = None
    try:
        cot_history = load_cot_history(selected_id)
    except Exception as exc:
        cot_error = str(exc)
    scoring_model = dict(model)
    scoring_model["status"] = display_status
    scoring_model["signed_distance_to_b_atr"] = signed_distance
    scoring_cot = cot_history if timeframe in ["4h", "1d"] else pd.DataFrame()
    structure = analyze_market_structure(
        full_history,
        scoring_model,
        scoring_cot,
        volume_reliable=str(selected["volume_reliability"]) == "usable",
    )

    observed_extreme = float(full_history["high"].max() if is_long else full_history["low"].min())
    projected_targets = {"T2": custom_t2, "T3": custom_t3}
    beyond_history = [
        name for name, value in projected_targets.items()
        if (is_long and value > observed_extreme) or (not is_long and value < observed_extreme)
    ]
    st.subheader(
        f"{selected['name_cn']} · {TIMEFRAME_LABELS[timeframe]} · "
        f"{DIRECTION_LABELS[model['direction']]} · {display_status}"
    )
    with st.container(horizontal=True):
        st.metric("收盘", price(close), border=True)
        st.metric("B", price(custom_b), border=True)
        st.metric("ATR(14)", price(atr), border=True)
        st.metric("Delta", price(custom_delta), border=True)
    with st.container(horizontal=True):
        st.metric("候选区", f"{price(zone_low)} – {price(zone_high)}", border=True)
        st.metric("失效位", price(custom_invalidation), border=True)
        st.metric("结构 T1", price(custom_t1), border=True)
        st.metric("推演 T2 / T3", f"{price(custom_t2)} / {price(custom_t3)}", border=True)

    if beyond_history:
        boundary_name = "历史最高价" if is_long else "历史最低价"
        st.warning(
            f"{', '.join(beyond_history)} 已超过当前周期样本的{boundary_name} {price(observed_extreme)}，"
            "属于等距价格发现推演，不是已验证的历史支撑/阻力。"
        )

    st.subheader("多空判断")
    with st.container(horizontal=True):
        st.metric("结构方向", DIRECTION_LABELS[model["direction"]], border=True)
        st.metric("证据合计", f"{structure['total_score']:+.1f}", border=True)
        st.metric("综合倾向", structure["score_direction"], border=True)
        st.metric("执行结论", structure["decision"], border=True)
    st.dataframe(
        structure["components"],
        hide_index=True,
        column_config={
            "维度": st.column_config.TextColumn(pinned=True),
            "分值": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    st.caption(
        "正分支持多头，负分支持空头；价格结构决定方向与触发，量价、OI和COT只做确认。"
        "只有处于接近触发/候选区且证据与结构同向时，才输出多头或空头候选。"
    )

    max_display_bars = min(len(full_history), 1000)
    min_display_bars = min(50, max_display_bars)
    default_display_bars = min(240, max_display_bars)
    bars_key = f"bars_{selected_id}_{timeframe}"
    if bars_key not in st.session_state or st.session_state[bars_key] > max_display_bars:
        st.session_state[bars_key] = default_display_bars
    display_bars = st.slider(
        "横轴显示K线根数",
        min_value=min_display_bars,
        max_value=max_display_bars,
        step=10 if max_display_bars >= 100 else 1,
        key=bars_key,
    )
    max_offset = max(0, len(full_history) - display_bars)
    offset_key = f"offset_{selected_id}_{timeframe}"
    if offset_key not in st.session_state or st.session_state[offset_key] > max_offset:
        st.session_state[offset_key] = 0
    history_offset = st.slider(
        "横轴向前回看",
        min_value=0,
        max_value=max_offset,
        step=10 if max_offset >= 100 else 1,
        key=offset_key,
        help="0表示显示最新K线；增大数值可查看更早的同周期历史。",
    ) if max_offset else 0
    end_index = len(full_history) - history_offset
    start_index = max(0, end_index - display_bars)
    history = full_history.iloc[start_index:end_index].copy()
    history["涨跌"] = np.where(history["close"] >= history["open"], "上涨", "下跌")
    axis_format = "%Y-%m-%d" if timeframe == "1d" else "%Y-%m-%d %H:%M"
    history["交易时间"] = history["date"].dt.strftime(axis_format)
    tick_indexes = np.unique(np.linspace(0, len(history) - 1, min(8, len(history)), dtype=int))
    tick_values = history.iloc[tick_indexes]["交易时间"].tolist()
    has_oi = "open_interest" in history and history["open_interest"].notna().any()
    hidden_x = alt.X(
        "交易时间:O",
        title=None,
        sort=None,
        axis=alt.Axis(labels=False, ticks=False),
    )
    visible_x = alt.X(
        "交易时间:O",
        title=None,
        sort=None,
        axis=alt.Axis(values=tick_values, labelAngle=0, labelLimit=130),
    )

    shown_levels = [custom_b, custom_t1, custom_invalidation]
    if target_display == "含推演目标":
        shown_levels.extend([custom_t2, custom_t3])
    auto_low = min(float(history["low"].min()), *shown_levels)
    auto_high = max(float(history["high"].max()), *shown_levels)
    padding = max((auto_high - auto_low) * 0.04, atr * 0.25)
    auto_low -= padding
    auto_high += padding
    manual_y = st.toggle("手动纵轴", value=False, key=f"manual_y_{selected_id}_{timeframe}")
    if manual_y:
        y_low_key = f"y_low_{selected_id}_{timeframe}"
        y_high_key = f"y_high_{selected_id}_{timeframe}"
        if y_low_key not in st.session_state:
            st.session_state[y_low_key] = auto_low
        if y_high_key not in st.session_state:
            st.session_state[y_high_key] = auto_high
        with st.container(horizontal=True):
            y_low = st.number_input("纵轴下限", key=y_low_key, step=point_step, format=value_format)
            y_high = st.number_input("纵轴上限", key=y_high_key, step=point_step, format=value_format)
        if y_low >= y_high:
            st.error("纵轴下限必须小于上限。")
            y_low, y_high = auto_low, auto_high
    else:
        y_low, y_high = auto_low, auto_high

    candle_colors = alt.Scale(domain=["上涨", "下跌"], range=["#2F7D5B", "#C44F4A"])
    y_scale = alt.Scale(domain=[y_low, y_high], zero=False)
    base_chart = alt.Chart(history).encode(x=hidden_x)
    wicks = base_chart.mark_rule().encode(
        y=alt.Y("low:Q", title="价格", scale=y_scale),
        y2="high:Q",
        color=alt.Color("涨跌:N", scale=candle_colors, legend=None),
    )
    candle_size = max(1, min(7, int(900 / max(display_bars, 1))))
    bodies = base_chart.mark_bar(size=candle_size).encode(
        y=alt.Y("open:Q", title="价格", scale=y_scale),
        y2="close:Q",
        color=alt.Color("涨跌:N", scale=candle_colors, legend=None),
        tooltip=[
            alt.Tooltip("date:T", title="日期"),
            alt.Tooltip("open:Q", title="开盘", format=",.3f"),
            alt.Tooltip("high:Q", title="最高", format=",.3f"),
            alt.Tooltip("low:Q", title="最低", format=",.3f"),
            alt.Tooltip("close:Q", title="收盘", format=",.3f"),
            alt.Tooltip("volume:Q", title="成交量", format=","),
        ],
    )
    zone = pd.DataFrame({"下沿": [zone_low], "上沿": [zone_high]})
    zone_band = alt.Chart(zone).mark_rect(color="#D8A64B", opacity=0.10).encode(
        y=alt.Y("下沿:Q", scale=y_scale), y2="上沿:Q"
    )
    core_levels = pd.DataFrame({
        "水平": ["B · 20根突破", "T1 · 结构参考", "失效位 · 10根结构"],
        "价格": [custom_b, custom_t1, custom_invalidation],
    })
    core_lines = alt.Chart(core_levels).mark_rule(strokeDash=[5, 4]).encode(
        y=alt.Y("价格:Q", scale=y_scale),
        color=alt.Color(
            "水平:N",
            scale=alt.Scale(
                domain=["B · 20根突破", "T1 · 结构参考", "失效位 · 10根结构"],
                range=["#202725", "#2F6B5F", "#B24A3A"],
            ),
        ),
        tooltip=["水平", alt.Tooltip("价格:Q", format=",.3f")],
    )
    chart = zone_band + wicks + bodies + core_lines
    if target_display == "含推演目标":
        projection_levels = pd.DataFrame({
            "水平": ["T2 · 等距推演", "T3 · 等距推演"],
            "价格": [custom_t2, custom_t3],
        })
        projection_lines = alt.Chart(projection_levels).mark_rule(
            color="#7A8581", strokeDash=[2, 5], opacity=0.65
        ).encode(
            y=alt.Y("价格:Q", scale=y_scale),
            tooltip=["水平", alt.Tooltip("价格:Q", format=",.3f")],
        )
        chart += projection_lines
    volume_chart = alt.Chart(history).mark_bar().encode(
        x=hidden_x if has_oi else visible_x,
        y=alt.Y("volume:Q", title="成交量"),
        color=alt.Color("涨跌:N", scale=candle_colors, legend=None),
        tooltip=[
            alt.Tooltip("date:T", title="日期"),
            alt.Tooltip("volume:Q", title="成交量", format=","),
        ],
    ).properties(height=95, width="container")
    chart_panels = [chart.properties(height=430, width="container"), volume_chart]
    if has_oi:
        oi_chart = alt.Chart(history.dropna(subset=["open_interest"])).mark_line(
            color="#3D6380", strokeWidth=1.5
        ).encode(
            x=visible_x,
            y=alt.Y("open_interest:Q", title="持仓量", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="日期"),
                alt.Tooltip("open_interest:Q", title="持仓量", format=","),
            ],
        ).properties(height=95, width="container")
        chart_panels.append(oi_chart)
    st.altair_chart(alt.vconcat(*chart_panels).resolve_scale(x="shared"), width="stretch")
    point_source = "用户修改" if any(
        not np.isclose(value, float(model[field]))
        for field, value in [("b", custom_b), ("delta", custom_delta), ("invalidation", custom_invalidation)]
    ) else str(model["delta_source"])
    st.caption(
        f"{len(full_history):,}根{TIMEFRAME_LABELS[timeframe]}K线 · 截至{full_history.iloc[-1]['date']:%Y-%m-%d %H:%M} · "
        f"点位来源：{point_source} · T2、T3为等距推演"
    )

    derivative_options = ["成交与持仓", "期限结构", "期权结构"]
    if selected_id == "gc":
        derivative_options.append("黄金跨市场")
    derivative_view = st.segmented_control(
        "衍生品结构",
        options=derivative_options,
        default="成交与持仓",
    )
    if derivative_view == "成交与持仓":
        with st.container(horizontal=True):
            volume_text = "-" if pd.isna(structure["volume_ratio"]) else f"{structure['volume_ratio']:.2f}×"
            oi_change_text = "-" if pd.isna(structure["oi_change"]) else f"{structure['oi_change']:+,.0f}"
            volume_label = "连续合约量比" if str(selected["volume_reliability"]) != "usable" else "最新量比"
            st.metric(volume_label, volume_text, border=True)
            st.metric("持仓量变化", oi_change_text, border=True)
            st.metric("价格/OI结构", structure["oi_state"], border=True)
        if selected_id == "gc":
            try:
                gold_activity, gold_contracts = load_gold_contract_activity()
            except Exception as exc:
                gold_activity, gold_contracts = pd.DataFrame(), pd.DataFrame()
                st.warning(f"COMEX分月成交量暂时读取失败：{exc}")
            if not gold_activity.empty:
                activity_latest = gold_activity.iloc[-1]
                recent_roll = bool(gold_activity.tail(5)["roll_flag"].any())
                ratio_text = "-" if pd.isna(activity_latest["volume_ratio"]) else f"{activity_latest['volume_ratio']:.2f}×"
                with st.container(horizontal=True):
                    st.metric("分月合约汇总量", f"{activity_latest['total_volume']:,.0f}", border=True)
                    st.metric("活跃合约", str(activity_latest["dominant_contract"]), border=True)
                    st.metric("活跃合约占比", f"{activity_latest['dominant_share']:.1%}", border=True)
                    st.metric("汇总量/20日中位数", ratio_text, border=True)
                activity_chart_data = gold_activity.tail(90).copy()
                activity_chart_data["交易日"] = activity_chart_data["date"].dt.strftime("%Y-%m-%d")
                activity_chart = alt.Chart(activity_chart_data).mark_bar(color="#3D6380").encode(
                    x=alt.X("交易日:O", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=100)),
                    y=alt.Y("total_volume:Q", title="分月合约汇总成交量"),
                    tooltip=[
                        alt.Tooltip("date:T", title="日期"),
                        alt.Tooltip("total_volume:Q", title="汇总成交量", format=","),
                        alt.Tooltip("dominant_contract:N", title="活跃合约"),
                        alt.Tooltip("dominant_share:Q", title="占比", format=".1%"),
                    ],
                ).properties(height=190, width="container")
                roll_points = alt.Chart(activity_chart_data[activity_chart_data["roll_flag"]]).mark_point(
                    color="#C44F4A", filled=True, size=65
                ).encode(
                    x=alt.X("交易日:O", sort=None),
                    y="total_volume:Q",
                    tooltip=[alt.Tooltip("date:T", title="主力切换日"), "dominant_contract:N"],
                )
                st.altair_chart(activity_chart + roll_points, width="stretch")
                st.dataframe(
                    gold_contracts[["symbol", "close", "volume", "volume_share"]],
                    hide_index=True,
                    column_config={
                        "symbol": "合约",
                        "close": st.column_config.NumberColumn("价格", format="%.2f"),
                        "volume": st.column_config.NumberColumn("成交量", format=","),
                        "volume_share": st.column_config.ProgressColumn("成交占比", format="percent", min_value=0, max_value=1),
                    },
                )
                if recent_roll:
                    st.warning("最近5个交易日发生过活跃合约切换，单一连续合约成交量不参与方向评分。")
                else:
                    st.caption("成交量为Yahoo可取得的COMEX黄金分月合约汇总；持仓量仍以CFTC周报为准。")
        if not cot_history.empty:
            cot_latest = cot_history.iloc[-1]
            with st.container(horizontal=True):
                st.metric("管理基金净仓", f"{cot_latest['managed_net']:+,.0f}", border=True)
                st.metric("净仓周变化", f"{cot_latest['managed_net_change']:+,.0f}", border=True)
                z_text = "-" if pd.isna(cot_latest["managed_net_z"]) else f"{cot_latest['managed_net_z']:+.2f}"
                st.metric("52周Z值", z_text, border=True)
            cot_chart = alt.Chart(cot_history.tail(104)).mark_line(color="#3D6380").encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("managed_net:Q", title="管理基金净仓"),
                tooltip=[alt.Tooltip("date:T", title="报告日"), alt.Tooltip("managed_net:Q", format=",")],
            ).properties(height=210)
            st.altair_chart(cot_chart)
            st.caption("CFTC周频数据；仅在4小时和日线评分中计分，分钟周期只作背景参考。")
        elif cot_error:
            st.warning(f"CFTC持仓暂时读取失败：{cot_error}")
        else:
            st.caption("该标的暂无可直接映射的CFTC持仓分类。")
    elif derivative_view == "期限结构":
        try:
            term_structure = load_term_structure(selected_id)
        except Exception as exc:
            term_structure = pd.DataFrame()
            st.warning(f"期限结构读取失败：{exc}")
        if len(term_structure) >= 2:
            front, second = term_structure.iloc[0], term_structure.iloc[1]
            spread = float(second["price"] - front["price"])
            curve_state = "Contango（远月升水）" if spread > 0 else "Backwardation（近月升水）" if spread < 0 else "平坦"
            with st.container(horizontal=True):
                st.metric("近月", f"{front['symbol']} · {price(front['price'])}", border=True)
                st.metric("次月价差", f"{spread:+,.3f}", border=True)
                st.metric("曲线形态", curve_state, border=True)
            curve_chart = alt.Chart(term_structure).mark_line(point=True, color="#2F6B5F").encode(
                x=alt.X("contract_month:T", title="合约月份", axis=alt.Axis(format="%Y-%m")),
                y=alt.Y("price:Q", title="结算/收盘价", scale=alt.Scale(zero=False)),
                tooltip=["symbol", alt.Tooltip("contract_month:T", format="%Y-%m"), alt.Tooltip("price:Q", format=",.3f")],
            ).properties(height=260)
            st.altair_chart(curve_chart)
            st.dataframe(
                term_structure[["symbol", "contract_month", "price", "carry_from_front_pct", "annualized_carry_pct"]],
                hide_index=True,
                column_config={
                    "symbol": "合约",
                    "contract_month": st.column_config.DateColumn("合约月份", format="YYYY-MM"),
                    "price": st.column_config.NumberColumn("价格", format="%.3f"),
                    "carry_from_front_pct": st.column_config.NumberColumn("较近月", format="percent"),
                    "annualized_carry_pct": st.column_config.NumberColumn("年化升贴水", format="percent"),
                },
            )
            st.caption("期限结构用于识别展期和供需背景，不直接跨品种计入多空分数。")
        else:
            st.info("该标的暂未取得至少两个可用的分月合约报价。")
    elif derivative_view == "黄金跨市场":
        try:
            london_gold = load_london_gold_proxy()
            gold_activity, _ = load_gold_contract_activity()
            active_gc = gold_activity[["date", "dominant_price"]].rename(columns={"dominant_price": "close"})
            gold_comparison, gold_summary = build_gold_comparison(active_gc, london_gold)
        except Exception as exc:
            gold_comparison, gold_summary = pd.DataFrame(), {}
            st.warning(f"黄金跨市场数据暂时读取失败：{exc}")
        if not gold_comparison.empty:
            with st.container(horizontal=True):
                st.metric("COMEX活跃合约", price(float(gold_summary["comex"])), border=True)
                st.metric("伦敦定盘代理", price(float(gold_summary["london"])), border=True)
                st.metric(
                    "参考基差",
                    f"{float(gold_summary['basis']):+,.2f}",
                    f"{float(gold_summary['basis_pct']):+.2%}",
                    border=True,
                )
                st.metric("20日相关性", f"{float(gold_summary['correlation_20d']):.2f}", border=True)
            with st.container(horizontal=True):
                st.metric("5日COMEX", f"{float(gold_summary['comex_5d']):+.2%}", border=True)
                st.metric("5日伦敦代理", f"{float(gold_summary['london_5d']):+.2%}", border=True)
                st.metric("价格确认", str(gold_summary["confirmation"]), border=True)

            comparison_chart_data = gold_comparison.tail(120).copy()
            comparison_chart_data["交易日"] = comparison_chart_data["date"].dt.strftime("%Y-%m-%d")
            first_comex = float(comparison_chart_data.iloc[0]["comex"])
            first_london = float(comparison_chart_data.iloc[0]["london"])
            comparison_chart_data["COMEX黄金"] = comparison_chart_data["comex"] / first_comex * 100
            comparison_chart_data["伦敦定盘代理"] = comparison_chart_data["london"] / first_london * 100
            indexed = comparison_chart_data.melt(
                id_vars=["date", "交易日"],
                value_vars=["COMEX黄金", "伦敦定盘代理"],
                var_name="市场",
                value_name="归一化价格",
            )
            price_comparison_chart = alt.Chart(indexed).mark_line(strokeWidth=2).encode(
                x=alt.X("交易日:O", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=100)),
                y=alt.Y("归一化价格:Q", title="起点=100", scale=alt.Scale(zero=False)),
                color=alt.Color("市场:N", title=None, scale=alt.Scale(range=["#2F6B5F", "#C28B2C"])),
                tooltip=[
                    alt.Tooltip("date:T", title="日期"),
                    "市场:N",
                    alt.Tooltip("归一化价格:Q", format=".2f"),
                ],
            ).properties(height=260, width="container")
            basis_chart = alt.Chart(comparison_chart_data).mark_bar().encode(
                x=alt.X("交易日:O", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=100)),
                y=alt.Y("basis:Q", title="COMEX - 伦敦代理"),
                color=alt.condition("datum.basis >= 0", alt.value("#3D6380"), alt.value("#C44F4A")),
                tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("basis:Q", title="参考基差", format="+,.2f")],
            ).properties(height=150, width="container")
            st.altair_chart(alt.vconcat(price_comparison_chart, basis_chart).resolve_scale(x="shared"), width="stretch")
            st.caption(
                f"截至 {pd.Timestamp(gold_summary['date']):%Y-%m-%d}。伦敦代理由NBP公布的PLN/克黄金价与同日USD/PLN反推；"
                "NBP价格基于伦敦定盘价。两地发布时间不同，基差仅作方向和异常监测，不是可交易套利报价。"
            )
            st.link_button("查看NBP黄金数据说明", "https://api.nbp.pl/en.html", icon=":material/open_in_new:")
    else:
        if selected_id in ["btc", "eth"]:
            try:
                option_detail, option_summary = load_crypto_options(selected_id)
            except Exception as exc:
                option_detail, option_summary = pd.DataFrame(), pd.DataFrame()
                st.warning(f"期权结构读取失败：{exc}")
            if not option_summary.empty:
                totals = option_summary.groupby("option_type")["open_interest"].sum()
                call_oi = float(totals.get("Call", 0))
                put_oi = float(totals.get("Put", 0))
                put_call = put_oi / call_oi if call_oi else np.nan
                with st.container(horizontal=True):
                    st.metric("Call OI", f"{call_oi:,.1f}", border=True)
                    st.metric("Put OI", f"{put_oi:,.1f}", border=True)
                    st.metric("Put/Call OI", "-" if pd.isna(put_call) else f"{put_call:.2f}", border=True)
                option_chart = alt.Chart(option_summary).mark_bar().encode(
                    x=alt.X("expiry:T", title="到期日", axis=alt.Axis(format="%m-%d")),
                    y=alt.Y("open_interest:Q", title="持仓量"),
                    color=alt.Color("option_type:N", title=None, scale=alt.Scale(range=["#2F7D5B", "#C44F4A"])),
                    xOffset="option_type:N",
                    tooltip=[alt.Tooltip("expiry:T", format="%Y-%m-%d"), "option_type", alt.Tooltip("open_interest:Q", format=",.1f")],
                ).properties(height=280)
                st.altair_chart(option_chart)
                st.caption("Deribit公开期权快照；Put/Call只能表示持仓结构，不能单独视为方向信号。")
            else:
                st.info("当前没有可用的公开期权快照。")
        elif selected_id in ["gc", "si", "hg", "es", "nq", "zn"]:
            st.info("该品种的交易所期权数据为日终公告格式，当前版本先保留官方入口，尚不纳入自动评分。")
            st.link_button(
                "查看CME Daily Bulletin",
                "https://www.cmegroup.com/market-data/daily-bulletin.html",
                icon=":material/open_in_new:",
            )
        else:
            st.info("该标的暂未接入稳定的免费期权链。")
    if selected["气候背景"] != "-":
        st.info(f"气候背景：{selected['气候背景']}。价格结构决定触发，气候只用于确认或否决。")

with environment_tab:
    env = environment.set_index("indicator_id")
    with st.container(horizontal=True):
        st.metric("VIX", f"{env.loc['vix', 'value']:.2f}", env.loc["vix", "state"], border=True)
        st.metric("ONI", f"{env.loc['oni', 'value']:+.2f}°C", env.loc["oni", "state"], border=True)
        st.metric("高收益债OAS", f"{env.loc['hy_oas', 'value']:.2f}%", env.loc["hy_oas", "state"], border=True)

    env_id = st.segmented_control(
        "历史轨迹",
        options=["vix", "oni", "hy_oas"],
        default="oni",
        format_func=lambda item: {"vix": "VIX", "oni": "ONI", "hy_oas": "高收益债OAS"}[item],
    )
    env_id = env_id or "oni"
    env_path = ENVIRONMENT_DIR / f"{env_id}.csv"
    env_history = load_environment_history(env_id, env_path.stat().st_mtime_ns)
    env_history = env_history.tail(120 if env_id == "oni" else 756)
    env_chart = alt.Chart(env_history).mark_line(color="#2F6B5F", strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=env.loc[env_id, "unit"], scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("value:Q", title="数值", format=".2f")],
    ).properties(height=330)
    zero_line = alt.Chart(pd.DataFrame({"value": [0]})).mark_rule(color="#8B9390", strokeDash=[4, 4]).encode(y="value:Q")
    st.altair_chart(env_chart + zero_line)
    st.caption(f"截至 {env.loc[env_id, 'as_of']:%Y-%m-%d} · {env.loc[env_id, 'source']} · {env.loc[env_id, 'note']}")

    st.subheader("ENSO海洋监控")
    try:
        enso = load_enso_monitor((ROOT / "enso_monitor.py").stat().st_mtime_ns)
    except Exception as exc:
        enso = None
        st.error(f"NOAA ENSO监控数据暂时读取失败：{exc}")
    if enso is not None:
        snapshot = enso["snapshot"]
        roni_latest = enso["roni"].iloc[-1]
        oni_latest = env.loc["oni"]
        same_season = pd.Timestamp(oni_latest["as_of"]).to_period("M") == pd.Timestamp(roni_latest["date"]).to_period("M")
        background_gap = float(oni_latest["value"] - roni_latest["roni"]) if same_season else np.nan

        with st.container(horizontal=True):
            st.metric(
                "周度 Niño 3.4",
                f"{snapshot['anomalies']['nino34']:+.1f}°C",
                f"4周 {snapshot['changes_4w']['nino34']:+.1f}°C",
                border=True,
            )
            st.metric("周度阶段", snapshot["phase"], f"截至 {snapshot['as_of']:%m-%d}", border=True)
            st.metric("空间形态", snapshot["pattern"], snapshot["breadth"], border=True)
            st.metric(
                "RONI",
                f"{roni_latest['roni']:+.2f}°C",
                f"ONI-RONI {background_gap:+.2f}°C" if same_season else f"{roni_latest['season']} {int(roni_latest['year'])}",
                border=True,
            )

        if snapshot["anomalies"]["nino34"] >= 0.5 and float(roni_latest["roni"]) >= 0.5:
            st.info(
                "海洋端的周度Niño 3.4与背景调整后的RONI均为暖信号。"
                "这提高了厄尔尼诺发展概率，但交易确认仍需观察SOI、信风/OLR以及具体产区降雨。"
            )
        elif snapshot["anomalies"]["nino34"] <= -0.5 and float(roni_latest["roni"]) <= -0.5:
            st.info(
                "海洋端的周度Niño 3.4与背景调整后的RONI均为冷信号。"
                "是否形成拉尼娜仍需大气耦合与持续性确认。"
            )
        else:
            st.warning(
                "周度海温与三个月背景调整指标尚未同向确认，当前更接近过渡或快速重定价阶段。"
                "不要仅凭单周海温生成农产品方向信号。"
            )

        region_labels = {
            "nino12_anom": "Niño 1+2",
            "nino3_anom": "Niño 3",
            "nino34_anom": "Niño 3.4",
            "nino4_anom": "Niño 4",
        }
        weekly_long = enso["weekly"].tail(52).melt(
            id_vars=["date"],
            value_vars=list(region_labels),
            var_name="region",
            value_name="anomaly",
        )
        weekly_long["区域"] = weekly_long["region"].map(region_labels)
        weekly_chart = alt.Chart(weekly_long).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("anomaly:Q", title="海温异常 °C", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "区域:N",
                scale=alt.Scale(
                    domain=["Niño 1+2", "Niño 3", "Niño 3.4", "Niño 4"],
                    range=["#C44536", "#D17A22", "#087F8C", "#1769AA"],
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="周中心日"),
                alt.Tooltip("区域:N"),
                alt.Tooltip("anomaly:Q", title="异常", format="+.1f"),
            ],
        ).properties(height=300, width="container")
        thresholds = alt.Chart(pd.DataFrame({"threshold": [-0.5, 0.0, 0.5]})).mark_rule(
            color="#8B9390", strokeDash=[4, 4]
        ).encode(y="threshold:Q")
        st.altair_chart(weekly_chart + thresholds, width="stretch")
        st.caption(
            "周度值用于发现转折；ONI/RONI是三个月滑动值，用于判断持续性。"
            "ONI-RONI可近似理解为热带平均增暖背景对传统ONI的抬升，不应被当作独立交易因子。"
        )

        st.subheader("海温与次表层图")
        try:
            image_results = {
                image_id: load_enso_image(image_id, (ROOT / "enso_monitor.py").stat().st_mtime_ns)
                for image_id in ("sst_map", "sst_hovmoller", "heat_content", "subsurface")
            }
            map_col, evolution_col = st.columns(2)
            with map_col:
                st.image(image_results["sst_map"][0], caption="热带太平洋周度海表温度异常：确认暖/冷水的空间位置", width="stretch")
            with evolution_col:
                st.image(image_results["sst_hovmoller"][0], caption="赤道太平洋海温异常时经图：观察异常是否向东传播", width="stretch")
            heat_col, depth_col = st.columns(2)
            with heat_col:
                st.image(image_results["heat_content"][0], caption="上层海洋热含量：观察水下热量领先变化", width="stretch")
            with depth_col:
                st.image(image_results["subsurface"][0], caption="赤道太平洋次表层温度异常：识别暖水体位置与深度", width="stretch")
            cached_images = [image_id for image_id, result in image_results.items() if result[1]]
            if cached_images:
                st.caption(f"NOAA实时图层暂时不可达，正在显示最近缓存：{', '.join(cached_images)}。")
            else:
                st.caption("图层来自NOAA/CPC，页面每6小时检查更新；图内日期为观测有效期。")
        except Exception as exc:
            st.warning(f"NOAA诊断图暂时不可用：{exc}")

    enso_uses = pd.DataFrame({
        "标的": ["可可", "咖啡", "白糖"],
        "主要产区": ["西非", "巴西与中美洲", "巴西、印度、泰国"],
        "平台用法": ["ONI + 西非降雨确认", "ONI + 巴西降雨/霜冻确认", "ONI + 产区降雨及政策确认"],
    })
    st.dataframe(enso_uses, hide_index=True)
    st.info("历史序列用于计算当前阈值、识别周期状态和做无前视回测；当前判断只读取最新已发布值。ONI不能单独生成农产品买卖信号。")

    st.subheader("产区水分")
    climate_table = climate[[
        "crop", "region_cn", "state", "rain_30d_mm", "rain_vs_season_pct",
        "root_wetness", "root_change_30d", "point_count", "as_of",
    ]].rename(columns={
        "crop": "作物", "region_cn": "产区", "state": "状态",
        "rain_30d_mm": "30日降雨mm", "rain_vs_season_pct": "较季节中位数%",
        "root_wetness": "根区湿度", "root_change_30d": "根区30日变化",
        "point_count": "采样点", "as_of": "截至",
    })
    st.dataframe(
        climate_table,
        hide_index=True,
        column_config={
            "30日降雨mm": st.column_config.NumberColumn(format="%.1f"),
            "较季节中位数%": st.column_config.NumberColumn(format="%+.1f%%"),
            "根区湿度": st.column_config.NumberColumn(format="%.3f"),
            "根区30日变化": st.column_config.NumberColumn(format="%+.3f"),
            "截至": st.column_config.DateColumn(format="YYYY-MM-DD"),
        },
    )
    region_ids = climate["region_id"].tolist()
    region_id = st.selectbox(
        "产区轨迹",
        region_ids,
        format_func=lambda item: climate.set_index("region_id").loc[item, "region_cn"],
    )
    climate_history = load_climate_history(region_id, CLIMATE_DAILY_PATH.stat().st_mtime_ns).tail(180)
    rain_chart = alt.Chart(climate_history).mark_bar(color="#7AA6A1", opacity=0.55).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("precip_mm:Q", title="日降雨 mm"),
        tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("precip_mm:Q", title="降雨", format=".1f")],
    )
    soil_chart = alt.Chart(climate_history).mark_line(color="#B24A3A", strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("root_14d:Q", title="14日根区湿度", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="日期"), alt.Tooltip("root_14d:Q", title="根区湿度", format=".3f")],
    )
    st.altair_chart(alt.layer(rain_chart, soil_chart).resolve_scale(y="independent").properties(height=330))
    st.caption("NASA POWER多点等权篮子 · 30日降雨相对2016年以来同季节中位数 · 土壤湿度显示同一近实时段内30日变化")

with source_tab:
    display_sources = sources.rename(columns={
        "name": "来源", "cost": "费用", "coverage": "覆盖", "role": "用途", "caveat": "限制", "endpoint_or_docs": "地址",
    })[["来源", "费用", "覆盖", "用途", "限制", "地址"]]
    st.dataframe(
        display_sources,
        hide_index=True,
        column_config={"地址": st.column_config.LinkColumn(display_text="打开")},
    )
    quality = data[["name_cn", "public_symbol", "数据源", "trade_date", "history_rows", "质量", "data_note"]].rename(columns={
        "name_cn": "标的", "public_symbol": "公开代码", "trade_date": "最新交易日",
        "history_rows": "历史行数", "data_note": "质量说明",
    })
    st.dataframe(quality, hide_index=True, column_config={"最新交易日": st.column_config.DateColumn(format="YYYY-MM-DD")})
    st.warning("免费公共接口没有SLA。Yahoo连续期货成交量仅作参考，不用于仓位容量或换月判断；真实交易前需核对交易所当前合约。", icon=":material/warning:")
