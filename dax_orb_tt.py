#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAX TraderTom ORB — optimized SL/TP parameter sweep with progress/ETA
- Pre-market: 08:00-09:00 (server-time)
- Session:     09:00-18:30 (server-time)
- Entry:       first break of pre-market high/low
- Exit:        TP/SL only (no time-exit), trades can span days
- Max 1 trade per day
"""
from __future__ import annotations
import argparse, glob, math, time, multiprocessing as mp
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo
import numpy as np

# ------------------------------- CLI -----------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DAX TraderTom ORB - Optimized SL/TP sweep + ETA")
    p.add_argument("paths", nargs="*", default=None, help="Path or glob to MT5 CSV files (time,open,high,low,close,volume). If none, searches current and 'data' directory.")
    p.add_argument("--tz", default="Europe/Athens", help="Server timezone, e.g., Europe/Athens")
    p.add_argument("--pre", default="08:00-09:00", help="Pre-market window (HH:MM-HH:MM)")
    p.add_argument("--sess", default="09:00-18:30", help="Session window (HH:MM-HH:MM)")
    p.add_argument("--confirm", choices=["touch", "close"], default="touch", help="Entry: wick ('touch') or close ('close')")
    p.add_argument("--samebar", choices=["skip", "worst", "best"], default="worst",
                   help="If both sides break in same bar: skip/worst/best")
    p.add_argument("--date-from", default=None, help="Filter from (YYYY-MM-DD) in server-tz")
    p.add_argument("--date-to", default=None, help="Filter to (YYYY-MM-DD) in server-tz")
    p.add_argument("--sl-min", type=float, default=2.0)
    p.add_argument("--sl-max", type=float, default=15.0)
    p.add_argument("--sl-step", type=float, default=1.0)
    p.add_argument("--tp-min", type=float, default=2.0)
    p.add_argument("--tp-max", type=float, default=15.0)
    p.add_argument("--tp-step", type=float, default=1.0)
    p.add_argument("--min-trades", type=int, default=120, help="Min. #trades for ranking")
    p.add_argument("--out", default="dax_orb_grid_results.csv", help="Output CSV with all combinations")
    p.add_argument("--save-best-trades", action="store_true", help="Save trades of best combo")
    p.add_argument("--best-trades-out", default="dax_orb_best_trades.csv", help="Best-trades CSV path")
    p.add_argument("--progress-sec", type=float, default=5.0, help="Progress/ETA update interval in seconds")
    return p.parse_args()

# ------------------------------ Helpers --------------------------------------
def read_merge_csv(paths: list[str] | None) -> pd.DataFrame:
    if not paths:
        current_dir = Path.cwd()
        data_dir = current_dir / "data"
        paths = [str(current_dir / "*.csv"), str(data_dir / "*.csv")]
    files = [f for pat in paths for f in glob.glob(pat)]
    if not files:
        raise FileNotFoundError("No CSV files found. Provide paths or place CSV files in current or 'data' directory.")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp).rename(columns={c: c.lower() for c in pd.read_csv(fp).columns})
        if not all(c in df.columns for c in ["time", "open", "high", "low", "close"]):
            raise ValueError(f"Missing required column in {fp}")
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)[["time", "open", "high", "low", "close"]]
    big["time"] = pd.to_datetime(big["time"], utc=True, errors="coerce")
    return big.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")

def hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)

def build_orb_levels(df_local: pd.DataFrame, pre_start: int, pre_end: int) -> dict:
    orb = df_local.groupby(df_local.index.date).apply(
        lambda g: (float(g[(g["minute"] >= pre_start) & (g["minute"] <= pre_end)]["low"].min()),
                   float(g[(g["minute"] >= pre_start) & (g["minute"] <= pre_end)]["high"].max()))
    ).dropna()
    return orb.to_dict()

def fmt_secs(x: float) -> str:
    x = max(0, int(x))
    h, s = divmod(x, 3600)
    m, s = divmod(s, 60)
    return f"{h}u{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"

# ---------------------------- Optimized Backtest -----------------------------
def run_backtest(df: pd.DataFrame, tz: str, pre_s: str, sess_s: str,
                 sl: float, tp: float, confirm: str, samebar: str,
                 date_from: str | None, date_to: str | None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["local"] = df["time"].dt.tz_convert(ZoneInfo(tz))
    df = df.set_index("local").sort_index()
    if date_from:
        df = df[df.index.date >= pd.to_datetime(date_from).date()]
    if date_to:
        df = df[df.index.date <= pd.to_datetime(date_to).date()]
    if df.empty:
        return pd.DataFrame(), dict(trades=0, wins=0, losses=0, pnl=0.0, pf=float("nan"), winrate=float("nan"))

    # Filter on trading hours (08:00-22:00 GMT+2/3)
    df = df[(df.index.hour >= 8) & (df.index.hour < 22)]
    df["minute"] = df.index.hour * 60 + df.index.minute
    pre_start, pre_end = [hhmm_to_minutes(x) for x in pre_s.split("-")]
    sess_start, sess_end = [hhmm_to_minutes(x) for x in sess_s.split("-")]

    orb_levels = build_orb_levels(df, pre_start, pre_end)

    trades = []
    in_pos = False
    for day, group in df.groupby(df.index.date):
        if day not in orb_levels or (in_pos and day == last_trade_day):
            continue
        orb_lo, orb_hi = orb_levels[day]
        if not (math.isfinite(orb_hi) and math.isfinite(orb_lo) and orb_hi > orb_lo):
            continue

        # Vectorized breakout detection
        mask_long_touch = (group["high"] >= orb_hi) & (group["low"] < orb_hi)
        mask_short_touch = (group["low"] <= orb_lo) & (group["high"] > orb_lo)
        mask_long_close = (group["close"] > orb_hi)
        mask_short_close = (group["close"] < orb_lo)
        trigger_long = mask_long_touch if confirm == "touch" else mask_long_close
        trigger_short = mask_short_touch if confirm == "touch" else mask_short_close

        if trigger_long.any() and not trigger_short.any():
            pos_side = "long"
            entry_idx = group[trigger_long].index[0]
            entry = orb_hi
        elif trigger_short.any() and not trigger_long.any():
            pos_side = "short"
            entry_idx = group[trigger_short].index[0]
            entry = orb_lo
        elif trigger_long.any() and trigger_short.any():
            if samebar == "skip":
                continue
            elif samebar == "worst":
                pos_side = "long" if sl > tp else "short"
                entry = orb_hi if pos_side == "long" else orb_lo
                pnl = -sl if pos_side == "long" else -tp
                trades.append({"day": str(day), "entry_time": entry_idx, "exit_time": entry_idx,
                               "side": pos_side, "entry": entry, "exit": entry - sl if pos_side == "long" else entry + sl,
                               "reason": "SL", "pnl": pnl})
                continue
            elif samebar == "best":
                pos_side = "long" if tp > sl else "short"
                entry = orb_hi if pos_side == "long" else orb_lo
                pnl = tp if pos_side == "long" else tp
                trades.append({"day": str(day), "entry_time": entry_idx, "exit_time": entry_idx,
                               "side": pos_side, "entry": entry, "exit": entry + tp if pos_side == "long" else entry - tp,
                               "reason": "TP", "pnl": pnl})
                continue
            continue
        else:
            continue

        if pos_side:
            stop = entry - sl if pos_side == "long" else entry + sl
            target = entry + tp if pos_side == "long" else entry - tp
            entry_time = entry_idx
            in_pos = True
            last_trade_day = day

            # Vectorized exit detection
            if pos_side == "long":
                hit_sl = (group.loc[entry_idx:]["low"] <= stop).any()
                hit_tp = (group.loc[entry_idx:]["high"] >= target).any()
                if hit_sl and hit_tp:
                    exit_idx = group.loc[entry_idx:][(group.loc[entry_idx:]["low"] <= stop) | (group.loc[entry_idx:]["high"] >= target)].index[0]
                    exit_px = stop if group.loc[exit_idx]["low"] <= stop else target
                    reason = "SL" if exit_px == stop else "TP"
                    pnl = -sl if reason == "SL" else tp
                elif hit_sl:
                    exit_idx = group.loc[entry_idx:][group.loc[entry_idx:]["low"] <= stop].index[0]
                    exit_px = stop
                    reason = "SL"
                    pnl = -sl
                elif hit_tp:
                    exit_idx = group.loc[entry_idx:][group.loc[entry_idx:]["high"] >= target].index[0]
                    exit_px = target
                    reason = "TP"
                    pnl = tp
                else:
                    continue
            else:
                hit_sl = (group.loc[entry_idx:]["high"] >= stop).any()
                hit_tp = (group.loc[entry_idx:]["low"] <= target).any()
                if hit_sl and hit_tp:
                    exit_idx = group.loc[entry_idx:][(group.loc[entry_idx:]["high"] >= stop) | (group.loc[entry_idx:]["low"] <= target)].index[0]
                    exit_px = stop if group.loc[exit_idx]["high"] >= stop else target
                    reason = "SL" if exit_px == stop else "TP"
                    pnl = -sl if reason == "SL" else tp
                elif hit_sl:
                    exit_idx = group.loc[entry_idx:][group.loc[entry_idx:]["high"] >= stop].index[0]
                    exit_px = stop
                    reason = "SL"
                    pnl = -sl
                elif hit_tp:
                    exit_idx = group.loc[entry_idx:][group.loc[entry_idx:]["low"] <= target].index[0]
                    exit_px = target
                    reason = "TP"
                    pnl = tp
                else:
                    continue

            trades.append({"day": str(day), "entry_time": entry_time, "exit_time": exit_idx,
                           "side": pos_side, "entry": entry, "exit": exit_px, "reason": reason, "pnl": pnl})
            in_pos = False

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, dict(trades=0, wins=0, losses=0, pnl=0.0, pf=float("nan"), winrate=float("nan"),
                               gross_profit=0.0, gross_loss=0.0, rr=(tp/sl if sl>0 else float("nan")))

    wins = int((trades_df["reason"] == "TP").sum())
    losses = int((trades_df["reason"] == "SL").sum())
    pnl = float(trades_df["pnl"].sum())
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    winrate = 100.0 * wins / len(trades_df)
    rr = (tp / sl) if sl > 0 else float("nan")
    return trades_df, dict(trades=len(trades_df), wins=wins, losses=losses, pnl=pnl,
                           pf=pf, winrate=winrate, gross_profit=gross_profit, gross_loss=gross_loss, rr=rr)

# ----------------------------- Optimized Main --------------------------------
def process_combo(args):
    df, tz, pre_s, sess_s, sl, tp, confirm, samebar, date_from, date_to = args
    return run_backtest(df, tz, pre_s, sess_s, sl, tp, confirm, samebar, date_from, date_to)

def main():
    a = parse_args()
    df_utc = read_merge_csv(a.paths)

    print("=== TraderTom DAX ORB — Optimized SL/TP Sweep ===")
    print(f"Timezone(server): {a.tz} | Premarket: {a.pre} | Session: {a.sess}")
    print(f"Entry: confirm={a.confirm}, samebar={a.samebar}")
    print(f"Grid SL: {a.sl_min}..{a.sl_max} step {a.sl_step} | TP: {a.tp_min}..{a.tp_max} step {a.tp_step}")
    if a.date_from or a.date_to:
        print(f"Period filter: {a.date_from or '-'} to {a.date_to or '-'}")
    print("-" * 72)

    sl_vals = list(np.arange(a.sl_min, a.sl_max + a.sl_step, a.sl_step))
    tp_vals = list(np.arange(a.tp_min, a.tp_max + a.tp_step, a.tp_step))
    total = len(sl_vals) * len(tp_vals)
    done = 0
    t0 = time.perf_counter()
    t_last = t0

    def maybe_progress(force=False):
        nonlocal done, t_last
        now = time.perf_counter()
        if force or (now - t_last) >= a.progress_sec:
            elapsed = now - t0
            pct = done / total if total else 1.0
            eta = (elapsed / pct - elapsed) if done else 0.0
            print(f"\rProgress: {done}/{total} ({pct:.2%}) | Elapsed: {fmt_secs(elapsed)} | ETA: {fmt_secs(eta)}",
                  end="", flush=True)
            t_last = now

    # Parallel execution
    with mp.Pool() as pool:
        combos = [(df_utc, a.tz, a.pre, a.sess, sl, tp, a.confirm, a.samebar, a.date_from, a.date_to)
                  for sl in sl_vals for tp in tp_vals]
        results = pool.map(process_combo, combos)

    # Process results
    rows = []
    best = None
    for i, (trades_df, s) in enumerate(results):
        done += 1
        maybe_progress(force=(done == 1 or done == total))
        if s["trades"] < a.min_trades:
            continue
        be_winrate = 100.0 * (sl / (sl + tp))
        row = dict(sl=sl, tp=tp, rr=(tp / sl), trades=s["trades"], winrate=s["winrate"],
                   pnl=s["pnl"], gross_profit=s["gross_profit"], gross_loss=s["gross_loss"],
                   pf=s["pf"], be_winrate=be_winrate)
        rows.append(row)
        score = (float("inf") if s["pf"] == float("inf") else s["pf"], s["pnl"], s["winrate"], s["trades"])
        if best is None or score > best[0]:
            best = (score, dict(row), trades_df.copy())

    print()  # newline after \r
    total_elapsed = time.perf_counter() - t0
    print(f"Completed in: {fmt_secs(total_elapsed)}")

    if not rows:
        print("No combinations with sufficient trades found. Lower --min-trades or extend period.")
        return

    res = pd.DataFrame(rows).sort_values(["pf", "pnl", "winrate", "trades"], ascending=[False, False, False, False])
    res.to_csv(a.out, index=False)

    print("\nTop 10 combinations (filtered by min-trades):")
    print(res.head(10).to_string(index=False, formatters={
        "sl": "{:.2f}".format, "tp": "{:.2f}".format, "rr": "{:.2f}".format,
        "winrate": "{:.2f}%".format, "pf": "{:.3f}".format, "pnl": "{:.2f}".format,
        "gross_profit": "{:.2f}".format, "gross_loss": "{:.2f}".format,
        "be_winrate": "{:.2f}%".format
    }))

    best_row = best[1]
    print("\n=== Best combo (by PF, then PnL, winrate, trades) ===")
    print(f"SL={best_row['sl']:.2f} | TP={best_row['tp']:.2f} | RR={best_row['rr']:.2f}")
    print(f"Trades={best_row['trades']} | Winrate={best_row['winrate']:.2f}% | BE-winrate={best_row['be_winrate']:.2f}%")
    print(f"PF={best_row['pf']:.3f} | PnL={best_row['pnl']:.2f} (GP={best_row['gross_profit']:.2f}, GL={best_row['gross_loss']:.2f})")
    print(f"\nAll results saved in: {a.out}")

    if a.save_best_trades and best[2] is not None and not best[2].empty:
        best[2][["day", "entry_time", "exit_time", "side", "entry", "exit", "reason", "pnl"]].to_csv(a.best_trades_out, index=False)
        print(f"Best trades saved in: {a.best_trades_out}")

if __name__ == "__main__":
    main()