"""冻结的 GEM 规则。不要在这里搜参数。"""
US = "SPY"
# 国际腿：用 50% 日本（EWJ）+ 50% 中国 A 股（ASHR）替代原来的 EFA。
INTL_LEGS = (("EWJ", 0.5), ("ASHR", 0.5))
INTL = "EWJ/ASHR"
CASH = "BIL"
SAFE = "AGG"
GEM_TICKERS = (US, CASH, SAFE) + tuple(ticker for ticker, _weight in INTL_LEGS)
LOOKBACK_MONTHS = 12
COST_BPS = (0.0, 5.0)
CORE_WEIGHT = 0.70
SATELLITE_WEIGHT = 0.30

# 组合再拆：70% 给上面的核心/卫星策略，30% 给 IEF 的 CTA 切换仓（顶底背离都做，没信号走动量）。买入持有 IEF 只作对照。
BOND = "IEF"
STRATEGY_WEIGHT = 0.70
BOND_WEIGHT = 0.30
ALLOC_START = "2017-01-01"
