"""Antonacci GEM：12 个月相对动量 + 绝对动量，月频，规则冻结。"""
import pandas as pd

from dual_mom.config import CASH, INTL, INTL_LEGS, LOOKBACK_MONTHS, SAFE, US


def month_end_closes(close_frame):
    """每个月最后一个交易日的收盘价，索引保留实际交易日。未走完的月份丢掉。"""
    last_days = close_frame.index.to_series().groupby(close_frame.index.to_period("M")).last()
    monthly = close_frame.loc[last_days]
    last_bar = pd.Timestamp(close_frame.index[-1]).normalize()
    month_cal_end = pd.Timestamp(last_bar.to_period("M").end_time).normalize()
    if (month_cal_end - last_bar).days > 3:
        monthly = monthly.iloc[:-1]
    return monthly


def trailing_return(monthly, months=LOOKBACK_MONTHS):
    return monthly / monthly.shift(months) - 1.0


def monthly_holdings(monthly_returns):
    """月底用已实现的 12 个月收益决定下一段持仓。"""
    needed = [US, INTL, CASH, SAFE]
    missing = [name for name in needed if name not in monthly_returns.columns]
    if missing:
        raise ValueError(f"缺少收益列 {missing}")
    rows = []
    for stamp, row in monthly_returns.iterrows():
        if row[needed].isna().any():
            continue
        if row[US] > row[CASH]:
            holding = US if row[US] >= row[INTL] else INTL
            regime = "风险开"
        else:
            holding = SAFE
            regime = "风险关"
        rows.append({
            "日期": pd.Timestamp(stamp).date(),
            "体制": regime,
            "持仓": holding,
            f"{US}_12m": float(row[US]),
            f"{INTL}_12m": float(row[INTL]),
            f"{CASH}_12m": float(row[CASH]),
            f"{SAFE}_12m": float(row[SAFE]),
        })
    return pd.DataFrame(rows)


def daily_weights(close_frame, decisions):
    """月底收盘写入新持仓，现有 daily_pnl 会滞后一天兑现。"""
    if decisions.empty:
        raise RuntimeError("没有形成任何 GEM 月度信号")
    choice = pd.Series(
        decisions["持仓"].to_numpy(),
        index=pd.to_datetime(decisions["日期"]),
        dtype="object",
    )
    daily = choice.reindex(close_frame.index)
    daily = daily.ffill()
    weights = pd.DataFrame(0.0, index=close_frame.index, columns=[US, INTL, SAFE])
    for name in weights.columns:
        weights[name] = (daily == name).astype(float)
    weights = weights.where(daily.notna())
    return weights, daily


def portfolio_from_weights(weights, close_frame, cost_bps, daily_pnl, assert_lagged):
    pnls = {}
    for ticker in weights.columns:
        asset_return = close_frame[ticker].pct_change()
        pnl = daily_pnl(weights[ticker], asset_return, cost_bps)
        if cost_bps == 0:
            assert_lagged(weights[ticker], asset_return, pnl)
        pnls[ticker] = pnl
    frame = pd.DataFrame(pnls).sort_index()
    return frame.sum(axis=1, min_count=1)



def basket_close(closes, legs=INTL_LEGS):
    """按给定权重、只在各腿都有收盘价的交易日，做每日再平衡后的价格指数。"""
    rets = pd.DataFrame({ticker: closes[ticker].pct_change() for ticker, _weight in legs})
    weights = pd.Series({ticker: weight for ticker, weight in legs}, dtype=float)
    if abs(float(weights.sum()) - 1.0) > 1e-12:
        raise RuntimeError("篮子权重之和应为 1")
    basket_ret = rets.mul(weights, axis=1).sum(axis=1, min_count=len(legs))
    realized = basket_ret.dropna()
    if realized.empty:
        raise RuntimeError("篮子没有可用的日收益")
    index = (1.0 + realized).cumprod()
    index.name = INTL
    return index
