"""读取日线。共享缓存只读；缺失的标的下载到本项目 cache/。"""
import re

import pandas as pd

from dual_mom.paths import CACHE_DIR, add_cta_to_path

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def safe_chart_name(ticker):
    return re.sub(r"[^\w.\-]+", "_", ticker)


def _normalize(df):
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(col[0]).strip() for col in frame.columns]
    rename = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = "Close" if key == "close" else key.capitalize()
    frame = frame.rename(columns=rename)
    missing = [col for col in OHLCV if col not in frame.columns]
    if missing:
        raise ValueError(f"缺少列 {missing}，实际列为 {list(frame.columns)}")
    out = frame.loc[:, OHLCV].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("没有可用的日线")
    return out


def _read_csv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalize(df)


def _shared_cache_dir():
    add_cta_to_path()
    from cta.config import CACHE_DIR as shared
    return shared


def _period_candidates():
    add_cta_to_path()
    from cta.config import PERIOD_CANDIDATES
    return PERIOD_CANDIDATES


def _best_cached(ticker, cache_dir):
    name = safe_chart_name(ticker)
    best = None
    periods = list(_period_candidates()) + ["max"]
    seen = set()
    for period in periods:
        if period in seen:
            continue
        seen.add(period)
        path = cache_dir / f"{name}_{period}.csv"
        if not path.is_file():
            continue
        df = _read_csv(path)
        rank = (df.index[0], -len(df))
        if best is None or rank < best[0]:
            best = (rank, period, path, df)
    return best


def _download(ticker, period="max"):
    import yfinance as yf

    raw = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"{ticker} 没有从 Yahoo 下载到日线")
    out = _normalize(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{safe_chart_name(ticker)}_{period}.csv"
    out.to_csv(path)
    return out, period, path


def load_daily(ticker, allow_download=True):
    """先读共享缓存（只读），再读本项目缓存；都没有才下载到本项目 cache/。"""
    searches = []
    shared = _shared_cache_dir()
    if shared.is_dir():
        searches.append(("共享缓存只读", shared))
    searches.append(("本项目缓存", CACHE_DIR))

    best = None
    source = None
    for label, folder in searches:
        found = _best_cached(ticker, folder)
        if found is None:
            continue
        if best is None or found[0] < best[0]:
            best = found
            source = label

    if best is not None:
        _rank, period, path, df = best
        meta = {
            "标的": ticker,
            "区间": period,
            "缓存文件": str(path),
            "来源": source,
            "日线起始": df.index[0].date().isoformat(),
            "日线结束": df.index[-1].date().isoformat(),
            "日线根数": len(df),
        }
        return df, meta

    if not allow_download:
        raise FileNotFoundError(f"{ticker} 没有日线缓存，且不允许下载")
    df, period, path = _download(ticker)
    meta = {
        "标的": ticker,
        "区间": period,
        "缓存文件": str(path),
        "来源": "下载到本项目缓存",
        "日线起始": df.index[0].date().isoformat(),
        "日线结束": df.index[-1].date().isoformat(),
        "日线根数": len(df),
    }
    return df, meta
