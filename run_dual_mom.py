"""GEM 双动量回测，以及和现有 CTA 切换账的核心/卫星接法。

本项目独立存放，只读取现有 CTA / TradingView 的代码和缓存，不改写它们。
新标的（EWJ、ASHR、BIL、AGG、IEF）的日线只写进本项目 cache/。
策略70/IEF CTA30 里，30% 的 IEF 腿按 CTA 切换规则交易：顶底背离都做，没有背离时走动量。
原来的买入持有 IEF，以及策略70/买入持有IEF30，留作对照。
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_mom.config import (  # noqa: E402
    ALLOC_START,
    BOND,
    BOND_WEIGHT,
    CASH,
    CORE_WEIGHT,
    COST_BPS,
    GEM_TICKERS,
    INTL,
    INTL_LEGS,
    LOOKBACK_MONTHS,
    SAFE,
    SATELLITE_WEIGHT,
    STRATEGY_WEIGHT,
    US,
)
from dual_mom.bond_sleeve import load_bond_cta  # noqa: E402
from dual_mom.cta_sleeve import load_switch_book  # noqa: E402
from dual_mom.data import load_daily  # noqa: E402
from dual_mom.gem import (  # noqa: E402
    basket_close,
    daily_weights,
    month_end_closes,
    monthly_holdings,
    portfolio_from_weights,
    trailing_return,
)
from dual_mom.paths import OUTPUT_ROOT, add_cta_to_path, cta_root  # noqa: E402

GEM_NAME = "GEM双动量"
CTA_NAME = "CTA切换"
BLEND_NAME = f"核心{int(CORE_WEIGHT * 100)}/卫星{int(SATELLITE_WEIGHT * 100)}"
ALLOC_NAME = f"策略{int(STRATEGY_WEIGHT * 100)}/{BOND} CTA{int(BOND_WEIGHT * 100)}"
ALLOC_BH_NAME = f"策略{int(STRATEGY_WEIGHT * 100)}/买入持有{BOND}{int(BOND_WEIGHT * 100)}"
SPY_HOLD = "买入持有SPY"
BOND_HOLD = f"买入持有{BOND}"
BOND_CTA_NAME = f"{BOND} CTA"
ROLLING_SHARPE_WINDOW = 252


def _cta_helpers():
    add_cta_to_path()
    from cta.backtest import assert_lagged, daily_pnl, format_summary, performance_row, wealth
    from cta.config import ANNUALIZATION

    return assert_lagged, daily_pnl, format_summary, performance_row, wealth, ANNUALIZATION


def _group_exposure(port_weights, names, index):
    if not names:
        return pd.Series(0.0, index=index)
    have = [name for name in names if name in port_weights.columns]
    if not have:
        return pd.Series(0.0, index=index)
    return port_weights[have].sum(axis=1).reindex(index).fillna(0.0)


def exposure_table(gem_weights, gem_choice, cta_port, equity, commodity, index, core, satellite):
    spy = gem_weights[US].reindex(index).fillna(0.0)
    efa = gem_weights[INTL].reindex(index).fillna(0.0)
    agg = gem_weights[SAFE].reindex(index).fillna(0.0)
    cta_eq = _group_exposure(cta_port, equity, index)
    cta_cmdty = _group_exposure(cta_port, commodity, index)
    cta_spy = cta_port[US].reindex(index).fillna(0.0) if US in cta_port.columns else 0.0
    rows = []
    for name, gem_frac, cta_frac in (
        (GEM_NAME, 1.0, 0.0),
        (CTA_NAME, 0.0, 1.0),
        (BLEND_NAME, core, satellite),
    ):
        eq = gem_frac * (spy + efa) + cta_frac * cta_eq
        bond = gem_frac * agg
        cmdty = cta_frac * cta_cmdty
        overlap_spy = gem_frac * spy + cta_frac * cta_spy
        gross = gem_frac * (spy + efa + agg) + cta_frac * cta_port.reindex(index).abs().sum(axis=1).fillna(0.0)
        rows.append({
            "组合": name,
            "股票平均敞口": float(eq.mean()),
            "股票最大敞口": float(eq.max()),
            "债券平均敞口": float(bond.mean()),
            "债券最大敞口": float(bond.max()),
            "商品平均敞口": float(cmdty.mean()),
            "商品最大敞口": float(cmdty.max()),
            "SPY重叠平均": float(overlap_spy.mean()),
            "SPY重叠最大": float(overlap_spy.max()),
            "毛敞口平均": float(gross.mean()),
            "毛敞口最大": float(gross.max()),
        })
    return pd.DataFrame(rows)


def latest_status(decisions, gem_choice, cta_port, equity, commodity, bond_cta=None):
    if decisions.empty:
        return "还没有 GEM 信号。"
    last = decisions.iloc[-1]
    last_day = pd.Timestamp(gem_choice.dropna().index[-1]).date()
    holding = gem_choice.dropna().iloc[-1]
    lines = [
        f"GEM 最近一次月度决定：{last['日期']}  {last['体制']}，持有 {last['持仓']}。",
        (
            f"12 个月收益  {US} {last[f'{US}_12m'] * 100:.2f}%  "
            f"{INTL} {last[f'{INTL}_12m'] * 100:.2f}%  "
            f"{CASH} {last[f'{CASH}_12m'] * 100:.2f}%  "
            f"{SAFE} {last[f'{SAFE}_12m'] * 100:.2f}%。"
        ),
        f"回测最后一天 {last_day} 的 GEM 持仓仍是 {holding}。",
    ]
    if cta_port is not None and not cta_port.empty:
        day = cta_port.dropna(how="all").index[-1]
        row = cta_port.loc[day].dropna()
        held = row[row.abs() > 1e-9]
        if held.empty:
            cta_text = "空仓"
        else:
            parts = [f"{name} {value * 100:+.1f}%" for name, value in held.items()]
            cta_text = "，".join(parts)
        lines.append(
            f"同一天 CTA 切换账（等权后）净敞口 {float(row.sum()) * 100:.1f}%：{cta_text}。"
        )
        eq = float(_group_exposure(cta_port, equity, [day]).iloc[0])
        cmdty = float(_group_exposure(cta_port, commodity, [day]).iloc[0])
        gem_eq = 1.0 if holding in (US, INTL) else 0.0
        gem_bond = 1.0 if holding == SAFE else 0.0
        lines.append(
            f"若按 {BLEND_NAME}：股票约 {(CORE_WEIGHT * gem_eq + SATELLITE_WEIGHT * eq) * 100:.1f}%，"
            f"债券约 {CORE_WEIGHT * gem_bond * 100:.1f}%，"
            f"商品约 {SATELLITE_WEIGHT * cmdty * 100:.1f}%。"
        )
        held_parts = []
        for name, value in held.items():
            sleeve = STRATEGY_WEIGHT * SATELLITE_WEIGHT * float(value)
            if abs(sleeve) > 1e-9:
                held_parts.append(f"{name} {sleeve * 100:+.2f}%")
        cta_text_alloc = "，".join(held_parts) if held_parts else "卫星部分空仓"
        core_parts = []
        if holding == INTL:
            for ticker, weight in INTL_LEGS:
                core_parts.append(f"{ticker} {STRATEGY_WEIGHT * CORE_WEIGHT * weight * 100:.2f}%")
        elif holding in (US, SAFE):
            core_parts.append(f"{holding} {STRATEGY_WEIGHT * CORE_WEIGHT * 100:.2f}%")
        core_text = "，".join(core_parts) if core_parts else "核心空仓"
        if bond_cta is not None:
            bw = bond_cta["weight"].reindex([day])
            w = float(bw.iloc[0]) if len(bw) and pd.notna(bw.iloc[0]) else 0.0
            src_row = bond_cta["source"].reindex([day])
            src = src_row.iloc[0] if len(src_row) and pd.notna(src_row.iloc[0]) else "—"
            if w > 1e-9:
                direction = "多"
            elif w < -1e-9:
                direction = "空"
            else:
                direction = "空仓"
            lines.append(
                f"若再按 {ALLOC_NAME}（70% 给上述策略，30% 给 {BOND} CTA：顶底背离都做，没信号走动量）："
                f"{core_text}，{BOND} CTA {direction} 品种仓位 {w * 100:+.1f}%（{src}），"
                f"组合权重 {BOND_WEIGHT * w * 100:+.2f}%，"
                f"{cta_text_alloc}。"
            )
        else:
            lines.append(
                f"若再按 {ALLOC_NAME}（70% 给上述策略，30% 给 {BOND}）："
                f"{core_text}，{BOND} {BOND_WEIGHT * 100:.2f}%，"
                f"{cta_text_alloc}。"
            )
    return "\n".join(lines)


def holding_breakdown(decisions):
    if decisions.empty:
        return "还没有月度持仓。"
    lines = ["月度持仓分布："]
    counts = decisions["持仓"].value_counts()
    for name, count in counts.items():
        lines.append(f"  {name}  {int(count)} 个月（{count / len(decisions) * 100:.1f}%）")
    runs = []
    held = decisions["持仓"].tolist()
    start = 0
    while start < len(held):
        end = start + 1
        while end < len(held) and held[end] == held[start]:
            end += 1
        runs.append(end - start)
        start = end
    lines.append(
        f"平均连续持有 {sum(runs) / len(runs):.1f} 个月，最长 {max(runs)} 个月，共 {len(decisions)} 个有效月底。"
    )
    return chr(10).join(lines)


def _use_cjk_font():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    )
    for path in candidates:
        if Path(path).is_file():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def rolling_sharpe(pnl, window=ROLLING_SHARPE_WINDOW, annualization=252):
    realized = pnl.dropna()
    if realized.empty:
        return realized
    mean = realized.rolling(window, min_periods=window).mean()
    std = realized.rolling(window, min_periods=window).std()
    return mean / std.replace(0, np.nan) * np.sqrt(annualization)


def plot_equity_and_rolling_sharpe(
    path,
    title,
    series,
    wealth,
    annualization=252,
    window=ROLLING_SHARPE_WINDOW,
    start=None,
    rebase=False,
):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, (ax_eq, ax_sh) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    last_date = None
    first_date = pd.Timestamp(start) if start is not None else None
    for name, pnl, style in series:
        curve = wealth(pnl)
        if curve.empty:
            continue
        sharpe = rolling_sharpe(pnl, window=window, annualization=annualization)
        if start is not None:
            curve = curve.loc[curve.index >= start]
            sharpe = sharpe.loc[sharpe.index >= start]
        if curve.empty:
            continue
        if rebase:
            curve = curve / curve.iloc[0]
        ax_eq.plot(curve.index, curve.to_numpy(), style, lw=1.4, label=name)
        ax_sh.plot(sharpe.index, sharpe.to_numpy(), style, lw=1.2, label=name)
        if first_date is None or curve.index[0] < first_date:
            first_date = pd.Timestamp(curve.index[0])
        if last_date is None or curve.index[-1] > last_date:
            last_date = pd.Timestamp(curve.index[-1])
    ax_eq.set_title(title)
    ax_eq.set_ylabel("净值（从窗口起点重定为 1）" if rebase else "净值")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(frameon=False)
    ax_sh.axhline(0.0, color="0.5", lw=0.8)
    ax_sh.set_ylabel("一年滚动夏普")
    ax_sh.grid(True, alpha=0.3)
    if first_date is not None and last_date is not None:
        span_days = (last_date - first_date).days
        if span_days <= 400:
            ax_sh.xaxis.set_major_locator(mdates.MonthLocator())
            ax_sh.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        elif span_days <= 900:
            ax_sh.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_sh.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax_sh.xaxis.set_major_locator(mdates.YearLocator())
            ax_sh.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run():
    assert_lagged, daily_pnl, format_summary, performance_row, wealth, annualization = _cta_helpers()
    _use_cjk_font()
    out_dir = OUTPUT_ROOT / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"只读 CTA 项目：{cta_root()}")
    closes = {}
    sources = []
    for ticker in GEM_TICKERS:
        daily, meta = load_daily(ticker, allow_download=True)
        closes[ticker] = daily.Close
        sources.append(meta)
        print(f"{ticker}  {meta['来源']}  {meta['日线起始']} → {meta['日线结束']}  {meta['日线根数']} 根")
    close_frame = pd.DataFrame({
        US: closes[US],
        INTL: basket_close(closes, INTL_LEGS),
        CASH: closes[CASH],
        SAFE: closes[SAFE],
    }).sort_index()
    monthly = month_end_closes(close_frame)
    decisions = monthly_holdings(trailing_return(monthly, LOOKBACK_MONTHS))
    gem_weights, gem_choice = daily_weights(close_frame, decisions)
    live_weight = gem_weights.sum(axis=1, min_count=1)
    if (live_weight.dropna() - 1.0).abs().max() > 1e-9:
        raise RuntimeError("GEM 日度权重之和应为 1")

    cta = load_switch_book()
    print("CTA 切换账品种：" + "、".join(f"{ticker}（{group}）" for ticker, group in cta["members"]))

    rows = []
    gem_pnls = {}
    for cost in COST_BPS:
        gem_pnls[cost] = portfolio_from_weights(
            gem_weights, close_frame, cost, cta["daily_pnl"], assert_lagged
        )
        rows.append(performance_row("组合", GEM_NAME, cost, gem_pnls[cost], live_weight))
        rows.append(performance_row("组合", CTA_NAME, cost, cta["pnls"][cost]))

    spy_return = close_frame[US].pct_change()
    common0 = gem_pnls[0.0].dropna().index.intersection(cta["pnls"][0.0].dropna().index)
    if common0.empty:
        raise RuntimeError("GEM 与 CTA 没有重叠样本")

    blend_pnls = {}
    for cost in COST_BPS:
        idx = gem_pnls[cost].dropna().index.intersection(cta["pnls"][cost].dropna().index)
        blend_pnls[cost] = CORE_WEIGHT * gem_pnls[cost].reindex(idx) + SATELLITE_WEIGHT * cta["pnls"][cost].reindex(idx)
        rows.append(performance_row("组合", BLEND_NAME, cost, blend_pnls[cost]))

    bond_daily, bond_meta = load_daily(BOND, allow_download=True)
    sources.append(bond_meta)
    print(f"{BOND}  {bond_meta['来源']}  {bond_meta['日线起始']} → {bond_meta['日线结束']}  {bond_meta['日线根数']} 根")
    bond_cta = load_bond_cta(BOND)
    sources.append(bond_cta["meta"])
    print(
        f"{BOND} CTA  {bond_cta['preset']['summary']}  "
        f"最后仓位 {float(bond_cta['weight'].dropna().iloc[-1]) * 100:+.1f}% "
        f"（{bond_cta['source'].dropna().iloc[-1]}）"
    )
    if bond_cta["pnls"][0.0].dropna().empty:
        raise RuntimeError("IEF CTA 没有可用盈亏")
    bond_return = bond_daily.Close.pct_change()
    bond_weight = pd.Series(1.0, index=bond_daily.index)
    bond_pnls = {}
    alloc_pnls = {}
    alloc_bh_pnls = {}
    alloc_start = pd.Timestamp(ALLOC_START)
    for cost in COST_BPS:
        bond_pnls[cost] = daily_pnl(bond_weight, bond_return, cost)
        if cost == 0.0:
            assert_lagged(bond_weight, bond_return, bond_pnls[cost])
            assert_lagged(bond_cta["weight"], bond_cta["asset_return"], bond_cta["pnls"][cost])
        alloc_idx = (
            blend_pnls[cost].dropna().index
            .intersection(bond_cta["pnls"][cost].dropna().index)
            .intersection(bond_pnls[cost].dropna().index)
        )
        alloc_idx = alloc_idx[alloc_idx >= alloc_start]
        if alloc_idx.empty:
            raise RuntimeError(f"{ALLOC_NAME} 在 {ALLOC_START} 之后没有样本")
        alloc_pnls[cost] = (
            STRATEGY_WEIGHT * blend_pnls[cost].reindex(alloc_idx)
            + BOND_WEIGHT * bond_cta["pnls"][cost].reindex(alloc_idx)
        )
        alloc_bh_pnls[cost] = (
            STRATEGY_WEIGHT * blend_pnls[cost].reindex(alloc_idx)
            + BOND_WEIGHT * bond_pnls[cost].reindex(alloc_idx)
        )
        rows.append(performance_row("组合", ALLOC_NAME, cost, alloc_pnls[cost]))
        rows.append(performance_row("组合", ALLOC_BH_NAME, cost, alloc_bh_pnls[cost]))
        rows.append(performance_row("组合", f"{BLEND_NAME}（2017年起）", cost, blend_pnls[cost].reindex(alloc_idx)))
        if cost == 0.0:
            rows.append(performance_row("组合", f"{BOND_CTA_NAME}（2017年起）", 0.0, bond_cta["pnls"][0.0].reindex(alloc_idx)))
            rows.append(performance_row("组合", f"{BOND_HOLD}（2017年起）", 0.0, bond_pnls[0.0].reindex(alloc_idx)))
            rows.append(performance_row("组合", f"{SPY_HOLD}（2017年起）", 0.0, spy_return.reindex(alloc_idx)))

    spy_hold = spy_return.reindex(common0)
    spy_hold_full = spy_return.reindex(gem_pnls[0.0].dropna().index)
    if len(gem_pnls[0.0].dropna()) > len(common0):
        rows.append(performance_row("组合", f"{GEM_NAME}（重叠样本）", 0.0, gem_pnls[0.0].reindex(common0)))
        rows.append(performance_row("组合", f"{SPY_HOLD}（GEM全程）", 0.0, spy_hold_full))
    rows.append(performance_row("组合", SPY_HOLD, 0.0, spy_hold))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    decisions.to_csv(out_dir / "gem_monthly_holdings.csv", index=False)
    pd.DataFrame(sources).to_csv(out_dir / "gem_sources.csv", index=False)
    pd.DataFrame(cta["metas"]).to_csv(out_dir / "cta_sources.csv", index=False)

    daily_out = pd.DataFrame({
        "GEM持仓": gem_choice,
        "GEM_SPY": gem_weights[US],
        "GEM_INTL": gem_weights[INTL],
        "GEM_EWJ": gem_weights[INTL] * dict(INTL_LEGS)["EWJ"],
        "GEM_ASHR": gem_weights[INTL] * dict(INTL_LEGS)["ASHR"],
        "GEM_AGG": gem_weights[SAFE],
        "GEM盈亏_0bps": gem_pnls[0.0],
        "CTA盈亏_0bps": cta["pnls"][0.0],
        f"{BLEND_NAME}_0bps": blend_pnls[0.0],
        f"{BOND_HOLD}_0bps": bond_pnls[0.0],
        f"{BOND_CTA_NAME}_仓位": bond_cta["weight"],
        f"{BOND_CTA_NAME}_来源": bond_cta["source"],
        f"{BOND_CTA_NAME}_0bps": bond_cta["pnls"][0.0],
        f"{ALLOC_BH_NAME}_0bps": alloc_bh_pnls[0.0],
        f"{ALLOC_NAME}_0bps": alloc_pnls[0.0],
    })
    daily_out.to_csv(out_dir / "daily.csv")

    exposures = exposure_table(
        gem_weights,
        gem_choice,
        cta["port_weights"],
        cta["equity"],
        cta["commodity"],
        common0,
        CORE_WEIGHT,
        SATELLITE_WEIGHT,
    )
    exposures.to_csv(out_dir / "exposures.csv", index=False)

    chart = [
        (GEM_NAME, gem_pnls[0.0].reindex(common0), "-"),
        (CTA_NAME, cta["pnls"][0.0].reindex(common0), "-"),
        (BLEND_NAME, blend_pnls[0.0], "-"),
        (SPY_HOLD, spy_hold, "k--"),
    ]
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare.png",
        "GEM 双动量（国际腿 50% EWJ + 50% ASHR）与 CTA 切换账，0 bps，重叠样本",
        chart,
        wealth,
        annualization=annualization,
    )
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare_2026.png",
        "GEM 双动量（国际腿 50% EWJ + 50% ASHR）与 CTA 切换账，0 bps，2026 年",
        chart,
        wealth,
        annualization=annualization,
        start=pd.Timestamp("2026-01-01"),
        rebase=True,
    )
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare_2017.png",
        "GEM 双动量（国际腿 50% EWJ + 50% ASHR）与 CTA 切换账，0 bps，2017 年起",
        chart,
        wealth,
        annualization=annualization,
        start=pd.Timestamp("2017-01-01"),
        rebase=True,
    )
    alloc_idx0 = alloc_pnls[0.0].dropna().index
    alloc_chart = [
        (ALLOC_NAME, alloc_pnls[0.0], "-"),
        (ALLOC_BH_NAME, alloc_bh_pnls[0.0], "-"),
        (f"{BLEND_NAME}（2017年起）", blend_pnls[0.0].reindex(alloc_idx0), "-"),
        (BOND_CTA_NAME, bond_cta["pnls"][0.0].reindex(alloc_idx0), "-"),
        (BOND_HOLD, bond_pnls[0.0].reindex(alloc_idx0), "-"),
        (SPY_HOLD, spy_return.reindex(alloc_idx0), "k--"),
    ]
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare_alloc_2017.png",
        "策略70/IEF CTA30（顶底背离都做，没信号走动量），对照买入持有IEF，0 bps，2017 年起",
        alloc_chart,
        wealth,
        annualization=annualization,
        start=alloc_start,
        rebase=True,
    )
    end_bar = max(common0.max(), alloc_idx0.max())
    start_24m = end_bar - pd.DateOffset(months=24)
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare_24m.png",
        "GEM 双动量（国际腿 50% EWJ + 50% ASHR）与 CTA 切换账，0 bps，过去 24 个月",
        chart,
        wealth,
        annualization=annualization,
        start=start_24m,
        rebase=True,
    )
    plot_equity_and_rolling_sharpe(
        out_dir / "equity_compare_alloc_24m.png",
        "策略70/IEF CTA30（顶底背离都做，没信号走动量），对照买入持有IEF，0 bps，过去 24 个月",
        alloc_chart,
        wealth,
        annualization=annualization,
        start=start_24m,
        rebase=True,
    )
    print()
    print(f"过去24个月窗口：{start_24m.date()} 起，至 {end_bar.date()}")
    zoom_rows = []
    seen = set()
    for name, pnl, _style in (*alloc_chart, *chart):
        label = name.replace("（2017年起）", "")
        if label in seen:
            continue
        seen.add(label)
        window_pnl = pnl.loc[(pnl.index >= start_24m) & (pnl.index <= end_bar)].dropna()
        if window_pnl.empty:
            continue
        zoom_rows.append(performance_row("组合", label, 0.0, window_pnl))
    zoom_summary = pd.DataFrame(zoom_rows)
    zoom_summary.to_csv(out_dir / "summary_24m.csv", index=False)
    print(format_summary(zoom_summary).to_string(index=False))

    shown = format_summary(summary)
    combo = shown[shown["标的"] == "组合"]
    print()
    print(combo[combo["成本bps"].isin([0, 0.0])].to_string(index=False))
    print()
    print(combo[combo["成本bps"].isin([5, 5.0])].to_string(index=False))
    print()
    shown_exp = exposures.copy()
    for col in shown_exp.columns:
        if col == "组合":
            continue
        shown_exp[col] = shown_exp[col].map(lambda value: f"{value * 100:.2f}%")
    print(shown_exp.to_string(index=False))
    print()
    print(holding_breakdown(decisions))
    print()
    status = latest_status(decisions, gem_choice, cta["port_weights"], cta["equity"], cta["commodity"], bond_cta)
    print(status)
    (out_dir / "latest_status.txt").write_text(status + "\n", encoding="utf-8")
    print()
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
