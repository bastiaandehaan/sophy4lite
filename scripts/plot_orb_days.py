# scripts/plot_orb_days.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import textwrap


@dataclass(frozen=True)
class ORConfig:
    session_open_local: str = "09:00"          # DAX cash open (lokale beurs-tijd)
    session_tz: str = "Europe/Berlin"
    premarket_minutes: int = 60
    minutes_after_open: int = 60               # hoeveel minuten na de open in de plot
    only_days_with_trades: bool = True         # alleen dagen plotten met een trade in trades_csv
    outdir: Path = Path("output/plots/GER40.cash")


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Zorg dat timestamps tz-aware UTC zijn."""
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df = df.set_index("time")
    idx = pd.to_datetime(df.index, utc=True)
    # Als het al tz-aware maar niet UTC is, convert:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    return df.sort_index()


def _parse_time_hhmm(hhmm: str) -> tuple[int, int]:
    try:
        hh, mm = hhmm.split(":")
        return int(hh), int(mm)
    except Exception:
        raise ValueError(f"Invalid time '{hhmm}', expected HH:MM")


def _opening_ts_utc(day: pd.Timestamp, open_local: str, tz: str) -> pd.Timestamp:
    """Return de cash open als UTC timestamp voor een gegeven kalenderdag (YYYY-MM-DD in die TZ)."""
    h, m = _parse_time_hhmm(open_local)
    local = pd.Timestamp(day.date()).replace(hour=h, minute=m, tzinfo=ZoneInfo(tz))
    return local.tz_convert("UTC")


def _compute_or(df_utc: pd.DataFrame, open_utc: pd.Timestamp, premarket_min: int) -> tuple[float, float, pd.Timestamp, pd.Timestamp]:
    """Bereken OR uit premarket venster: [open - premarket, open)."""
    start = open_utc - pd.Timedelta(minutes=premarket_min)
    prem = df_utc.loc[start:open_utc - pd.Timedelta(microseconds=1)]
    if prem.empty:
        raise ValueError("Geen premarket data in venster. Check je CSV en tijden.")
    return float(prem["high"].max()), float(prem["low"].min()), start, open_utc


def _candles(ax, df: pd.DataFrame):
    """Eenvoudige M1 candlesticks (zonder extra libs)."""
    # Verwacht index=UTC time, kolommen: open, high, low, close
    times = mdates.date2num(df.index.to_pydatetime())
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    width = 1 / (24 * 60) * 0.8  # 0.8 minuut breed
    up = c >= o
    down = ~up

    # wicks
    ax.vlines(times, l, h, linewidth=1)

    # bodies
    ax.bar(times[up], c[up] - o[up], width, bottom=o[up], align="center", edgecolor="black")
    ax.bar(times[down], o[down] - c[down], width, bottom=c[down], align="center", edgecolor="black")


def _load_trades(trades_csv: Path) -> pd.DataFrame:
    t = pd.read_csv(trades_csv)
    cols = {c.lower(): c for c in t.columns}
    # Verwachte kolommen (probeer flexibel te mappen)
    required = ["symbol", "entry_time", "entry_px", "side"]
    for r in required:
        if r not in [c.lower() for c in t.columns]:
            raise ValueError(f"Trades CSV mist kolom '{r}' (gevonden: {list(t.columns)})")
    # Normaliseer kolomnamen
    t.columns = [c.lower() for c in t.columns]
    # Timestamps -> UTC
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    if "exit_time" in t.columns:
        t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    return t


def _plot_one_day(df_utc: pd.DataFrame, trades_day: pd.DataFrame, day_local: pd.Timestamp, cfg: ORConfig, symbol: str, outdir: Path):
    # Bereken open & OR
    open_utc = _opening_ts_utc(day_local, cfg.session_open_local, cfg.session_tz)
    or_high, or_low, pre_start, pre_end = _compute_or(df_utc, open_utc, cfg.premarket_minutes)

    # Plot venster: premarket + N min na open
    end_utc = open_utc + pd.Timedelta(minutes=cfg.minutes_after_open)
    win = df_utc.loc[pre_start:end_utc]
    if win.empty:
        print(f"[WARN] Geen bars voor {day_local.date()} — skip")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    _candles(ax, win)

    # Premarket shading
    ax.axvspan(mdates.date2num(pre_start.to_pydatetime()),
               mdates.date2num(pre_end.to_pydatetime()),
               alpha=0.15)

    # OR lijnen
    ax.axhline(or_high, linestyle="--")
    ax.axhline(or_low, linestyle="--")

    # Entries/Exits/SL/TP markers voor deze dag
    for _, r in trades_day.iterrows():
        et: pd.Timestamp = r["entry_time"]
        if not (open_utc <= et <= end_utc):
            continue
        ep = float(r["entry_px"])
        side = str(r["side"]).lower()

        marker = "^" if side == "long" else "v"
        ax.plot(mdates.date2num(et.to_pydatetime()), ep, marker, markersize=9)

        # Exit
        if "exit_time" in trades_day.columns and pd.notna(r.get("exit_time")):
            xt = pd.to_datetime(r["exit_time"], utc=True)
            if open_utc <= xt <= end_utc:
                xp = float(r.get("exit_px", np.nan)) if "exit_px" in trades_day.columns else np.nan
                if np.isfinite(xp):
                    ax.plot(mdates.date2num(xt.to_pydatetime()), xp, "x", markersize=9)

        # SL/TP (indien aanwezig)
        for key, ls in [("sl", ":"), ("tp", "-.")]:
            if key in trades_day.columns and pd.notna(r.get(key)):
                lvl = float(r[key])
                ax.axhline(lvl, linestyle=ls, linewidth=1)

    # As formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo("UTC")))
    ax.set_title(f"{symbol} — {day_local.date()}  (premarket={cfg.premarket_minutes}m, OR=[{or_low:.1f},{or_high:.1f}])")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()

    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{symbol}_{day_local.date()}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {fp}")


def main():
    p = argparse.ArgumentParser(
        description="Plot ORB-dagcharts voor DAX met OR-zone, premarket en trades.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Voorbeeld:
          python -m scripts.plot_orb_days --symbol GER40.cash --csv data/GER40.cash_M1.csv --trades output/trades_premarket_orb_native.csv \\
                 --start 2025-05-15 --end 2025-08-29 --session-open-local 09:00 --session-tz Europe/Berlin --premarket-minutes 60
        """)
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--csv", required=True, help="M1 CSV met kolommen time,open,high,low,close[,volume] in UTC")
    p.add_argument("--trades", required=True, help="Trades CSV uit je backtest")
    p.add_argument("--start", required=False)
    p.add_argument("--end", required=False)
    p.add_argument("--session-open-local", default="09:00")
    p.add_argument("--session-tz", default="Europe/Berlin")
    p.add_argument("--premarket-minutes", type=int, default=60)
    p.add_argument("--minutes-after-open", type=int, default=60)
    p.add_argument("--only-days-with-trades", action="store_true", default=False)
    p.add_argument("--outdir", default="output/plots/GER40.cash")

    args = p.parse_args()

    cfg = ORConfig(
        session_open_local=args.session_open_local,
        session_tz=args.session_tz,
        premarket_minutes=args.premarket_minutes,
        minutes_after_open=args.minutes_after_open,
        only_days_with_trades=args.only_days_with_trades,
        outdir=Path(args.outdir),
    )

    # Load data
    df = pd.read_csv(args.csv, parse_dates=["time"])
    df = _ensure_utc_index(df[["time", "open", "high", "low", "close"]])

    trades = _load_trades(Path(args.trades))
    trades = trades[trades["symbol"].astype(str).str.lower() == args.symbol.lower()].copy()

    # Maak lijst van dagen
    if args.start:
        start_day = pd.Timestamp(args.start).tz_localize(ZoneInfo(cfg.session_tz))
    else:
        start_day = df.index[0].tz_convert(ZoneInfo(cfg.session_tz))
    if args.end:
        end_day = pd.Timestamp(args.end).tz_localize(ZoneInfo(cfg.session_tz))
    else:
        end_day = df.index[-1].tz_convert(ZoneInfo(cfg.session_tz))

    # Dagen in lokale sessietijd
    days = pd.date_range(start=start_day.normalize(), end=end_day.normalize(), freq="D", tz=ZoneInfo(cfg.session_tz))

    if cfg.only_days_with_trades and not trades.empty:
        tr_days = trades["entry_time"].dt.tz_convert(cfg.session_tz).dt.normalize().unique()
        days = pd.DatetimeIndex(tr_days).sort_values()

    # Plot elke dag
    for day_local in days:
        # Subset aan trades van deze dag
        mask = trades["entry_time"].dt.tz_convert(cfg.session_tz).dt.normalize() == day_local.normalize()
        trades_day = trades.loc[mask]
        try:
            _plot_one_day(df, trades_day, pd.Timestamp(day_local), cfg, args.symbol, cfg.outdir)
        except Exception as e:
            print(f"[WARN] Dag {day_local.date()} overgeslagen: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
