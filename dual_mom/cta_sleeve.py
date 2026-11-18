"""只读计算现有 CTA 切换账，不刷新、不写入共享缓存。"""
from dual_mom.paths import add_cta_to_path


def load_switch_book():
    add_cta_to_path()
    from cta.backtest import daily_pnl
    from cta.config import COST_BPS
    from cta.data import load_market
    from cta.mix_book import compute_sleeves, portfolio_daily_pnl, portfolio_weights, presets_for
    from cta.universe import listed_members

    members = listed_members()
    if not members:
        raise RuntimeError("现有 CTA 名单是空的")
    presets, groups = presets_for(members)
    markets = {}
    metas = []
    for ticker, _group in members:
        daily, _weekly, meta = load_market(ticker)
        markets[ticker] = daily
        metas.append(meta)
    weight_frame, _source, _close = compute_sleeves(markets, presets, groups)
    port_weights = portfolio_weights(weight_frame)
    pnls = {}
    for cost in COST_BPS:
        pnls[cost] = portfolio_daily_pnl(weight_frame, markets, cost)
        if cost == 0.0:
            # 合成盈亏应等于各品种滞后仓位的等权平均，这里只做非空检查。
            if pnls[cost].dropna().empty:
                raise RuntimeError("CTA 切换账没有可用盈亏")
    equity = [ticker for ticker, group in members if group == "股票"]
    commodity = [ticker for ticker, group in members if group == "商品"]
    fx = [ticker for ticker, group in members if group == "外汇"]
    return {
        "members": members,
        "groups": groups,
        "markets": markets,
        "metas": metas,
        "weight_frame": weight_frame,
        "port_weights": port_weights,
        "pnls": pnls,
        "equity": equity,
        "commodity": commodity,
        "fx": fx,
        "daily_pnl": daily_pnl,
    }
