# backtest/runner.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("runner")

# ====== Jouw bestaande modules ======
from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
from backtest.breakout_exec import BTExecCfg, backtest_breakout

# Nieuwe ORB-variant
from strategies.premarket_orb import ORBParams, premarket_orb_entries
from backtest.orb_exec import ORBExecCfg, backtest_orb_bidirectional


# ---------- helpers ----------
def _coerce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Zorg voor tz-aware (UTC) DatetimeIndex; accepteert kolommen time/datetime/Date/date."""
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        return df

    for col in ("time", "datetime", "Date", "date", "timestamp"):
        if col in df.columns:
            idx = pd.to_datetime(df[col], utc=True, errors="coerce")
            if not idx.isna().all():
                df = df.set_index(idx).drop(columns=[col])
                return df

    # laatste redmiddel: probeer eerste kolom
    first = df.columns[0]
    idx = pd.to_datetime(df[first], utc=True, errors="coerce")
    if not idx.isna().all():
        df = df.set_index(idx).drop(columns=[first])
        return df

    raise ValueError("Could not find/convert a datetime index in CSV.")

def _load_csv(csv: Path) -> pd.DataFrame:
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    df = _coerce_dt_index(df)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"CSV missing columns: {sorted(missing)}")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[pd.Timestamp(start, tz="UTC") :]
    if end:
        df = df.loc[: pd.Timestamp(end, tz="UTC")]
    return df

def _resolve_specs(symbol: str, specs: Optional[Dict[str, SymbolSpec]]) -> Dict[str, SymbolSpec]:
    base_specs = dict(DEFAULT_SPECS)
    if specs:
        base_specs.update(specs)
    if symbol not in base_specs:
        base = symbol.split(".")[0]
        if base in base_specs:
            base_specs[symbol] = base_specs[base]
        else:
            base_specs[symbol] = SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)
    return base_specs


# ---------- public API ----------
def run_backtest(
    *,
    strategy_name: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    csv: Optional[str] = None,
    engine: str = "native",
    outdir: Optional[str] = "output",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Centrale runner. Retourneert (equity_df, trades_df, metrics_dict).

    strategy_name:
      - "breakout"       (jouw prev-day/opening-breakout)
      - "premarket_orb"  (Trader Tom premarket-ORB)
    engine:
      - "native" (hier), "vbt" is elders (runner_vbt).
    """
    if engine != "native":
        raise NotImplementedError("Only engine='native' is implemented in this runner.")

    if not csv:
        raise ValueError("CSV path is required. Provide --csv pointing to OHLC data.")

    df = _load_csv(Path(csv))
    df = _slice(df, start, end)
    logger.info("Loaded %s rows for %s %s", len(df), symbol, timeframe)

    specs = _resolve_specs(symbol, None)

    if strategy_name == "breakout":
        # === bestaande opening-breakout (prev-day based) ===
        p = BreakoutParams(
            window=int(params.get("window", 20)),
            atr_mult_sl=float(params.get("atr_sl", 1.0)),
            atr_mult_tp=float(params.get("atr_tp", 1.8)),
        )
        cfg = BTExecCfg(
            equity0=float(params.get("equity0", 20_000.0)),
            fee_rate=float(params.get("fee_bps", 2.0)) / 10_000.0,
            entry_slip_pts=float(params.get("entry_slip_pts", 0.1)),
            sl_slip_pts=float(params.get("sl_slip_pts", 0.5)),
            tp_slip_pts=float(params.get("tp_slip_pts", 0.0)),
            specs=specs,
            risk_frac=float(params.get("risk_pct", 1.0)) / 100.0,
            atr_n=int(params.get("atr_period", 14)),
        )
        open_window_bars = int(params.get("open_window_bars", 4))
        confirm = str(params.get("confirm", "close"))  # "close"|"wick"

        eq, trades, metrics = backtest_breakout(
            df=df,
            symbol=symbol,
            params=p,
            cfg=cfg,
            open_window_bars=open_window_bars,
            confirm=confirm,
        )

    elif strategy_name == "premarket_orb":
        # === nieuwe premarket ORB ===
        if "GER40" in symbol or "DAX" in symbol:
            orb_p = ORBParams(session_open_local="09:00", session_tz="Europe/Berlin",
                              premarket_minutes=int(params.get("premarket_minutes", 60)),
                              confirm=str(params.get("confirm", "close")))
        elif "US30" in symbol or "Dow" in symbol:
            orb_p = ORBParams(session_open_local="09:30", session_tz="America/New_York",
                              premarket_minutes=int(params.get("premarket_minutes", 60)),
                              confirm=str(params.get("confirm", "close")))
        else:
            orb_p = ORBParams(
                session_open_local=str(params.get("session_open_local", "09:00")),
                session_tz=str(params.get("session_tz", "Europe/Berlin")),
                premarket_minutes=int(params.get("premarket_minutes", 60)),
                confirm=str(params.get("confirm", "close")),
            )

        e_long, e_short = premarket_orb_entries(df, orb_p)

        p = BreakoutParams(
            window=int(params.get("window", 20)),
            atr_mult_sl=float(params.get("atr_sl", 1.0)),
            atr_mult_tp=float(params.get("atr_tp", 1.8)),
        )
        cfg = ORBExecCfg(
            equity0=float(params.get("equity0", 20_000.0)),
            fee_rate=float(params.get("fee_bps", 2.0)) / 10_000.0,
            entry_slip_pts=float(params.get("entry_slip_pts", 0.1)),
            sl_slip_pts=float(params.get("sl_slip_pts", 0.5)),
            tp_slip_pts=float(params.get("tp_slip_pts", 0.0)),
            specs=specs,
            risk_frac=float(params.get("risk_pct", 1.0)) / 100.0,
            atr_n=int(params.get("atr_period", 14)),
        )

        eq, trades, metrics = backtest_orb_bidirectional(
            df=df,
            symbol=symbol,
            params=p,
            cfg=cfg,
            entries_long=e_long,
            entries_short=e_short,
        )
    else:
        raise ValueError(f"Unknown strategy_name: {strategy_name}")

    # outputs
    if outdir:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"equity_{strategy_name}_{symbol}.csv").write_text(eq.to_csv())
        trades.to_csv(out / f"trades_{strategy_name}_{symbol}.csv", index=False)
        with open(out / f"metrics_{strategy_name}_{symbol}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("[%s|%s] n_trades=%s ret=%.2f%% sharpe=%.2f maxDD=%.2f%%",
                    strategy_name, symbol, metrics.get("n_trades"),
                    metrics.get("return_total_pct", 0.0),
                    metrics.get("sharpe", 0.0),
                    metrics.get("max_drawdown_pct", 0.0))

    return eq.to_frame("equity"), trades, metrics
