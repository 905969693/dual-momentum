"""IEF 腿的 CTA 切换仓：顶底背离都做，没有背离时走动量。不改 CTA 等权名单。"""
import pandas as pd

from dual_mom.config import BOND, COST_BPS
from dual_mom.data import load_daily
from dual_mom.paths import add_cta_to_path


def _source_frame(weight, side):
    source = pd.Series(pd.NA, index=weight.index, dtype="object")
    live = weight.notna()
    source = source.mask(live, "动量")
    source = source.mask(live & (side > 0), "底背离")
    source = source.mask(live & (side < 0), "顶背离")
    return source


def bond_cta_preset(ticker):
    add_cta_to_path()
    from run_fx import default_preset

    preset = dict(default_preset(ticker))
    preset["ticker"] = ticker
    preset["long_only"] = False
    text = str(preset.get("summary") or "").replace("、只做多", "").replace("、多空都做", "")
    preset["summary"] = (text + "、多空都做") if text else "多空都做"
    return preset


def load_bond_cta(ticker=BOND):
    add_cta_to_path()
    from cta.backtest import daily_pnl
    from cta.divergence_book import divergence_side, switch_position
    from cta.forecast import tsmom_forecast

    daily, meta = load_daily(ticker, allow_download=True)
    preset = bond_cta_preset(ticker)
    forecast = tsmom_forecast(daily.Close).clip(-1, 1)
    weight = switch_position(daily, preset, forecast)
    side, _events = divergence_side(daily, preset, both_sides=True)
    source = _source_frame(weight, side.reindex(weight.index).fillna(0.0))
    asset_return = daily.Close.pct_change()
    pnls = {cost: daily_pnl(weight, asset_return, cost) for cost in COST_BPS}
    return {
        "ticker": ticker,
        "daily": daily,
        "meta": meta,
        "preset": preset,
        "weight": weight,
        "source": source,
        "pnls": pnls,
        "asset_return": asset_return,
    }
