# 跨资产结构监测

一个以宏观状态、价格结构和公开数据为核心的多资产交易观察工具。项目用于研究和监测，不构成投资建议。

## 当前功能

- 宏观状态：增长、通胀、利率、流动性、金融压力、中国周期与美元动量。
- 政策预期：美联储会议路径及预期变化。
- 利率拆解：实际利率、通胀补偿、预期短端利率和期限溢价。
- 信号雷达：跨资产趋势、候选区和风险提示。
- 结构详情：K线、成交量、持仓、期限结构和期权结构。
- 环境状态：ENSO、海温与相关农产品气候监测。
- 黄金观察：COMEX黄金与伦敦金代理价格的交叉验证。

## 数据原则

项目优先使用免费公开数据，包括 FRED、OECD、NOAA、CFTC、CME、广州期货交易所、Yahoo Finance、AKShare/新浪以及 Deribit 公共接口。仓库中的 `data/` 是应用启动所需的数据快照；部分页面会在运行时刷新公开数据。

公开接口可能改版、延迟或暂时不可用。页面会尽量显示数据日期和质量提醒，任何信号都应结合原始来源复核。

## 本地运行

推荐使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501/`。

## 测试

```powershell
python -m unittest discover -v
```

GitHub Actions 会在每次推送和拉取请求时自动执行同一套测试。

## 部署到 Streamlit Community Cloud

1. 将仓库推送到 GitHub。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)，连接 GitHub 账户。
3. 创建应用并选择本仓库、`main` 分支和入口文件 `streamlit_app.py`。
4. 在高级设置中选择 Python 3.12，然后部署。

部署后，推送到 GitHub 的代码通常会自动更新在线应用。修改 `requirements.txt` 会触发完整重建。

## 更新流程

```powershell
git add .
git commit -m "描述本次修改"
git push
```

建议将功能修改、数据管道修改和数据快照更新分开提交，便于定位问题和回退。

## 目录说明

- `streamlit_app.py`：应用入口。
- `signal_engine.py`、`structure_engine.py`：信号和结构计算。
- `macro_regime.py`、`fed_policy.py`：宏观状态和政策预期。
- `market_data_pipeline.py`：公开行情更新管道。
- `climate_pipeline.py`、`enso_monitor.py`：气候数据处理。
- `config/`：资产池与数据源配置。
- `data/`：应用所需的公开数据快照。
- `test_*.py`：核心计算测试。

## 免责声明

本项目仅用于教育、研究和信息展示，不提供投资建议、收益承诺或自动交易服务。公开数据可能存在延迟、缺失、修订和口径差异；使用者应自行核验并承担决策风险。
