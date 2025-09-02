This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where line numbers have been added, security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
backtest/
  breakout_exec.py
  data_loader.py
  orb_exec.py
  runner_vbt.py
  runner.py
cli/
  main.py
live/
  live_trading.py
risk/
  ftmo_guard.py
scripts/
  diagnostics.py
  export_mt5_data.py
  export_mt5_orb.py
  plot_orb_days.py
  run_backtest_demo.py
strategies/
  __init__.py
  breakout_params.py
  breakout_signals.py
  order_block.py
  premarket_orb.py
test/
  test_metrics.py
utils/
  data_health.py
  days.py
  metrics.py
  plot.py
  position.py
.gitignore
config.py
dax_orb_tt.py
pyproject.toml
quick_smoke_opening_breakout.py
README.md
requirements.txt
validate_framework.py
```

# Files

## File: backtest/breakout_exec.py
````python
  1: # backtest/breakout_exec.py
  2: from __future__ import annotations
  3: 
  4: from dataclasses import dataclass
  5: from typing import Dict, Tuple
  6: 
  7: import math
  8: import numpy as np
  9: import pandas as pd
 10: 
 11: from config import logger
 12: from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
 13: from strategies.breakout_signals import opening_breakout_long
 14: from utils.position import _size  # risk-based position sizing (float)
 15: 
 16: 
 17: # ========= Config =========
 18: 
 19: @dataclass(frozen=True)
 20: class BTExecCfg:
 21:     equity0: float = 20_000.0           # startkapitaal
 22:     fee_rate: float = 0.0002            # 2 bps per LEG (entry én exit)
 23:     entry_slip_pts: float = 0.1         # adverse slip bij ENTRY (nieuw)
 24:     sl_slip_pts: float = 0.5            # adverse slip bij SL
 25:     tp_slip_pts: float = 0.0            # gunstig of neutraal bij TP
 26:     specs: Dict[str, SymbolSpec] = None
 27:     risk_frac: float = 0.01             # 1% per trade
 28:     atr_n: int = 14                      # ATR lookback
 29:     atr_floor: float = 1e-6              # minimale ATR om sizing te doen
 30: 
 31:     def get_spec(self, symbol: str) -> SymbolSpec:
 32:         specs = self.specs or DEFAULT_SPECS
 33:         if symbol in specs:
 34:             return specs[symbol]
 35:         base = symbol.split(".")[0]
 36:         if base in specs:
 37:             return specs[base]
 38:         return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)
 39: 
 40: 
 41: # ========= Helpers =========
 42: 
 43: def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
 44:     h, l, c = df["high"], df["low"], df["close"]
 45:     prev_c = c.shift(1)
 46:     tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
 47:     atr = tr.ewm(alpha=1 / n, adjust=False).mean()
 48:     return atr.rename(f"ATR({n})")
 49: 
 50: 
 51: def _entry_fee(notional: float, fee_rate: float) -> float:
 52:     return abs(notional) * fee_rate  # enkele leg
 53: 
 54: 
 55: def _exit_fee(notional: float, fee_rate: float) -> float:
 56:     return abs(notional) * fee_rate  # enkele leg
 57: 
 58: 
 59: def _sl_tp_on_entry(entry_px: float, atr_val: float, p: BreakoutParams) -> Tuple[float, float]:
 60:     sl = entry_px - p.atr_mult_sl * atr_val
 61:     tp = entry_px + p.atr_mult_tp * atr_val
 62:     return sl, tp
 63: 
 64: 
 65: # ========= Kern backtester =========
 66: 
 67: def backtest_breakout(
 68:     df: pd.DataFrame,
 69:     symbol: str,
 70:     params: BreakoutParams,
 71:     cfg: BTExecCfg,
 72:     open_window_bars: int = 4,
 73:     confirm: str = "close",
 74: ):
 75:     """
 76:     1 trade per dag: opening-breakout op previous-day high.
 77:     Entry: eerste trigger in openingsvenster.
 78:     Exit: SL of TP (ATR-multiples bij entry), intra-bar, SL prioriteit.
 79:     Fees per leg (entry+exit), slippage (ENTRY/SL/TP apart), risk-based sizing.
 80:     """
 81:     req = {"open", "high", "low", "close"}
 82:     if not isinstance(df.index, pd.DatetimeIndex):
 83:         raise TypeError("Index must be DatetimeIndex")
 84:     if not req.issubset(df.columns):
 85:         miss = req.difference(df.columns)
 86:         raise KeyError(f"df mist kolommen: {sorted(miss)}")
 87: 
 88:     df = df.sort_index().copy()
 89:     spec = cfg.get_spec(symbol)
 90:     atr = _atr(df, n=cfg.atr_n)
 91: 
 92:     # Max 1 entry per dag in openingsvenster
 93:     entries = opening_breakout_long(
 94:         df["close"], df["high"], open_window_bars=open_window_bars, confirm=confirm
 95:     )
 96: 
 97:     # Warmup: geen entries toestaan vóór een geldige ATR
 98:     entries = entries & atr.notna()
 99: 
100:     # Containers
101:     eq = pd.Series(index=df.index, dtype="float64")
102:     equity = float(cfg.equity0)
103:     trades = []
104: 
105:     in_pos = False
106:     entry_px = sl_px = tp_px = np.nan
107:     size = 0.0
108:     entry_time = None
109:     entry_leg_fee = 0.0  # fee op entry-notional
110: 
111:     for ts, row in df.iterrows():
112:         h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
113:         a_val = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else None
114: 
115:         if not in_pos:
116:             # Entry: signaal + bruikbare ATR
117:             if entries.loc[ts] and (a_val is not None) and (a_val > cfg.atr_floor):
118:                 sl_raw, tp_raw = _sl_tp_on_entry(c, a_val, params)
119:                 if sl_raw >= c:  # guard tegen pathologische data
120:                     eq.loc[ts] = equity
121:                     continue
122: 
123:                 this_size = _size(equity, c, sl_raw, spec.point_value, cfg.risk_frac)
124:                 if this_size <= 0:
125:                     eq.loc[ts] = equity
126:                     continue
127: 
128:                 # Slippage per type: ENTRY
129:                 effective_entry = c + cfg.entry_slip_pts  # adverse richting voor long entry
130:                 entry_px = effective_entry
131:                 sl_px, tp_px = sl_raw, tp_raw
132:                 size = this_size
133:                 entry_time = ts
134:                 in_pos = True
135: 
136:                 notional = entry_px * size * spec.point_value
137:                 entry_leg_fee = _entry_fee(notional, cfg.fee_rate)
138: 
139:                 eq.loc[ts] = equity
140:                 continue
141: 
142:             eq.loc[ts] = equity
143:             continue
144: 
145:         # In positie: SL -> TP
146:         exit_reason = None
147:         exit_px = None
148: 
149:         if l <= sl_px:
150:             exit_reason = "SL"
151:             exit_px = sl_px - cfg.sl_slip_pts  # adverse slip bij stop
152:         elif h >= tp_px:
153:             exit_reason = "TP"
154:             exit_px = tp_px + cfg.tp_slip_pts  # niet-nadelig (0 of klein positief)
155: 
156:         if exit_reason is None:
157:             eq.loc[ts] = equity
158:             continue
159: 
160:         # Realiseer P&L + exit fee
161:         notional_exit = exit_px * size * spec.point_value
162:         pnl_per_contract = (exit_px - entry_px) * spec.point_value
163:         gross = pnl_per_contract * size
164:         net = gross - entry_leg_fee - _exit_fee(notional_exit, cfg.fee_rate)
165: 
166:         equity += net
167: 
168:         trades.append({
169:             "symbol": symbol,
170:             "entry_time": entry_time,
171:             "entry_px": entry_px,
172:             "exit_time": ts,
173:             "exit_px": exit_px,
174:             "reason": exit_reason,
175:             "size": size,
176:             "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
177:             "sl_px": sl_px,
178:             "tp_px": tp_px,
179:             "pnl": net,
180:             "gross": gross,
181:             "fees": entry_leg_fee + _exit_fee(notional_exit, cfg.fee_rate),
182:         })
183: 
184:         # Reset
185:         in_pos = False
186:         entry_px = sl_px = tp_px = np.nan
187:         size = 0.0
188:         entry_time = None
189:         entry_leg_fee = 0.0
190:         eq.loc[ts] = equity
191: 
192:     # Sluit open positie op laatste bar
193:     if in_pos and entry_time is not None:
194:         last_ts = df.index[-1]
195:         last_close = float(df["close"].iloc[-1])
196:         # conservatief: sluit tegen last_close met entry-slip voor adverse richting
197:         exit_px = last_close - cfg.sl_slip_pts
198: 
199:         notional_exit = exit_px * size * spec.point_value
200:         pnl_per_contract = (exit_px - entry_px) * spec.point_value
201:         gross = pnl_per_contract * size
202:         net = gross - entry_leg_fee - _exit_fee(notional_exit, cfg.fee_rate)
203: 
204:         equity += net
205:         trades.append({
206:             "symbol": symbol,
207:             "entry_time": entry_time,
208:             "entry_px": entry_px,
209:             "exit_time": last_ts,
210:             "exit_px": exit_px,
211:             "reason": "EOD",
212:             "size": size,
213:             "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
214:             "sl_px": sl_px,
215:             "tp_px": tp_px,
216:             "pnl": net,
217:             "gross": gross,
218:             "fees": entry_leg_fee + _exit_fee(notional_exit, cfg.fee_rate),
219:         })
220:         eq.iloc[-1] = equity
221: 
222:     # Equity carry
223:     eq = eq.ffill().fillna(cfg.equity0).rename("equity")
224: 
225:     trades_df = (
226:         pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
227:         if trades else pd.DataFrame(columns=[
228:             "symbol","entry_time","entry_px","exit_time","exit_px","reason",
229:             "size","atr_at_entry","sl_px","tp_px","pnl","gross","fees"
230:         ])
231:     )
232: 
233:     # Metrics
234:     def _max_dd(series: pd.Series):
235:         if len(series) == 0:
236:             return 0.0, 0
237:         roll_max = series.cummax()
238:         dd = series / roll_max - 1.0
239:         min_dd = float(dd.min())
240:         under = dd < 0
241:         dur = int(pd.Series(np.where(under, 1, 0), index=series.index)
242:                   .groupby((~under).cumsum()).sum().max() or 0)
243:         return min_dd * 100.0, dur
244: 
245:     total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) else 0.0
246:     max_dd_pct, dd_dur = _max_dd(eq)
247: 
248:     bar_rets = eq.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
249:     sd = float(bar_rets.std(ddof=1))
250:     if sd > 0:
251:         bars_per_day = df.index.normalize().value_counts().mean() if len(df) else 60.0
252:         sharpe = float(bar_rets.mean() / sd * math.sqrt(bars_per_day * 252.0))
253:     else:
254:         sharpe = 0.0
255: 
256:     metrics = {
257:         "n_trades": int(len(trades_df)),
258:         "final_equity": float(eq.iloc[-1]) if len(eq) else cfg.equity0,
259:         "return_total_pct": total_ret,
260:         "max_drawdown_pct": max_dd_pct,
261:         "dd_duration_bars": dd_dur,
262:         "sharpe": sharpe,
263:     }
264: 
265:     return eq, trades_df, metrics
````

## File: backtest/data_loader.py
````python
 1: from __future__ import annotations
 2: import numpy as np
 3: import pandas as pd
 4: from pathlib import Path
 5: from typing import Optional
 6: 
 7: 
 8: def fetch_data(
 9:     csv_path: Optional[str | Path] = None,
10:     symbol: Optional[str] = None,
11:     timeframe: Optional[str] = None,
12:     start: Optional[str] = None,
13:     end: Optional[str] = None,
14: ) -> pd.DataFrame:
15:     """
16:     Laadt OHLC(V) data van CSV of (placeholder) MT5.
17:     Vereist kolommen: open, high, low, close. Volume optioneel.
18:     """
19:     if csv_path:
20:         csv_path = Path(csv_path)
21:         df = pd.read_csv(csv_path)
22: 
23:         # tijdkolom detecteren
24:         for col in ("date", "datetime", "timestamp", "time"):
25:             if col in df.columns:
26:                 df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
27:                 df = df.dropna(subset=[col]).set_index(col)
28:                 break
29:         else:
30:             raise KeyError("CSV mist tijdkolom: date/datetime/timestamp/time")
31:     else:
32:         # Placeholder voor MT5 fetch (implementeer als nodig)
33:         raise NotImplementedError("MT5 data fetch nog niet geïmplementeerd; gebruik CSV.")
34: 
35:     # kolommen normaliseren
36:     df = df.rename(columns={c: c.lower() for c in df.columns})
37:     required = ["open", "high", "low", "close"]
38:     missing = [c for c in required if c not in df.columns]
39:     if missing:
40:         raise KeyError(f"Vereiste kolommen ontbreken: {missing}")
41: 
42:     if "volume" not in df.columns:
43:         df["volume"] = np.nan
44: 
45:     df = df.sort_index()
46:     df = df[["open", "high", "low", "close", "volume"]].astype(float)
47:     df = df[~df.index.duplicated(keep="last")]
48: 
49:     # Zorg dat start/end dezelfde tijdzone krijgen als de index
50:     tz = df.index.tz
51: 
52:     if start:
53:         ts = pd.Timestamp(start)
54:         if tz is not None and ts.tzinfo is None:
55:             ts = ts.tz_localize(tz)  # align naar index-tz
56:         df = df.loc[ts:]
57: 
58:     if end:
59:         ts = pd.Timestamp(end)
60:         if tz is not None and ts.tzinfo is None:
61:             ts = ts.tz_localize(tz)
62:         df = df.loc[:ts]
63: 
64:     return df
````

## File: backtest/orb_exec.py
````python
  1: # backtest/orb_exec.py
  2: from __future__ import annotations
  3: from dataclasses import dataclass
  4: from typing import Dict, Tuple
  5: import math
  6: import numpy as np
  7: import pandas as pd
  8: 
  9: from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
 10: from utils.position import _size
 11: 
 12: @dataclass(frozen=True)
 13: class ORBExecCfg:
 14:     equity0: float = 20_000.0
 15:     fee_rate: float = 0.0002      # per leg
 16:     entry_slip_pts: float = 0.1
 17:     sl_slip_pts: float = 0.5
 18:     tp_slip_pts: float = 0.0
 19:     specs: Dict[str, SymbolSpec] = None
 20:     risk_frac: float = 0.01
 21:     atr_n: int = 14
 22:     atr_floor: float = 1e-6
 23: 
 24:     def get_spec(self, symbol: str) -> SymbolSpec:
 25:         specs = self.specs or DEFAULT_SPECS
 26:         if symbol in specs:
 27:             return specs[symbol]
 28:         base = symbol.split(".")[0]
 29:         if base in specs:
 30:             return specs[base]
 31:         return SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)
 32: 
 33: def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
 34:     h, l, c = df["high"], df["low"], df["close"]
 35:     prev_c = c.shift(1)
 36:     tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
 37:     return tr.ewm(alpha=1 / n, adjust=False).mean().rename(f"ATR({n})")
 38: 
 39: def backtest_orb_bidirectional(
 40:     df: pd.DataFrame,
 41:     symbol: str,
 42:     params: BreakoutParams,
 43:     cfg: ORBExecCfg,
 44:     entries_long: pd.Series,
 45:     entries_short: pd.Series,
 46: ) -> Tuple[pd.Series, pd.DataFrame, dict]:
 47:     """
 48:     Backtest pre-market ORB met long Ã©n short entries.
 49:     Er wordt max 1 trade per dag gezet (eerste trigger wint).
 50:     Intraâ€‘bar SLâ€‘prioriteit, ATRâ€‘based SL/TP, fees en slippage zoals in breakout_exec.
 51:     """
 52:     req = {"open", "high", "low", "close"}
 53:     if not isinstance(df.index, pd.DatetimeIndex):
 54:         raise TypeError("Index must be DatetimeIndex")
 55:     if not req.issubset(df.columns):
 56:         miss = req.difference(df.columns)
 57:         raise KeyError(f"df mist kolommen: {sorted(miss)}")
 58:     if not entries_long.index.equals(df.index) or not entries_short.index.equals(df.index):
 59:         raise ValueError("entries index mismatch met df.index")
 60: 
 61:     df = df.sort_index().copy()
 62:     spec = cfg.get_spec(symbol)
 63:     atr = _atr(df, n=cfg.atr_n)
 64: 
 65:     # ATRâ€‘warmup: geen entries als atr NaN of < floor
 66:     entries_long = entries_long & atr.notna() & (atr > cfg.atr_floor)
 67:     entries_short = entries_short & atr.notna() & (atr > cfg.atr_floor)
 68: 
 69:     eq = pd.Series(index=df.index, dtype="float64")
 70:     equity = float(cfg.equity0)
 71:     trades = []
 72: 
 73:     in_pos = False
 74:     side = None  # "long" of "short"
 75:     entry_px = sl_px = tp_px = np.nan
 76:     size = 0.0
 77:     entry_time = None
 78:     entry_leg_fee = 0.0
 79: 
 80:     for ts, row in df.iterrows():
 81:         h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
 82:         a_val = float(atr.loc[ts]) if pd.notna(atr.loc[ts]) else None
 83: 
 84:         if not in_pos:
 85:             go_long = entries_long.loc[ts]
 86:             go_short = entries_short.loc[ts]
 87: 
 88:             if (go_long or go_short) and (a_val is not None):
 89:                 side = "long" if go_long and not go_short else "short" if go_short and not go_long else "long"
 90:                 effective_entry = c + cfg.entry_slip_pts if side == "long" else c - cfg.entry_slip_pts
 91:                 if side == "long":
 92:                     sl_raw = effective_entry - params.atr_mult_sl * a_val
 93:                     tp_raw = effective_entry + params.atr_mult_tp * a_val
 94:                 else:
 95:                     sl_raw = effective_entry + params.atr_mult_sl * a_val
 96:                     tp_raw = effective_entry - params.atr_mult_tp * a_val
 97:                 if (side == "long" and sl_raw >= effective_entry) or (side == "short" and sl_raw <= effective_entry):
 98:                     eq.loc[ts] = equity
 99:                     continue
100:                 this_size = _size(equity, effective_entry, sl_raw, spec.point_value, cfg.risk_frac)
101:                 if this_size <= 0:
102:                     eq.loc[ts] = equity
103:                     continue
104: 
105:                 entry_px = effective_entry
106:                 sl_px, tp_px = sl_raw, tp_raw
107:                 size = this_size
108:                 entry_time = ts
109:                 in_pos = True
110: 
111:                 notional = entry_px * size * spec.point_value
112:                 entry_leg_fee = abs(notional) * cfg.fee_rate
113: 
114:                 eq.loc[ts] = equity
115:                 continue
116: 
117:             eq.loc[ts] = equity
118:             continue
119: 
120:         # in positie â†’ SL first, dan TP
121:         exit_reason = None
122:         exit_px = None
123: 
124:         if side == "long":
125:             if l <= sl_px:
126:                 exit_reason, exit_px = "SL", sl_px - cfg.sl_slip_pts
127:             elif h >= tp_px:
128:                 exit_reason, exit_px = "TP", tp_px + cfg.tp_slip_pts
129:         else:
130:             if h >= sl_px:
131:                 exit_reason, exit_px = "SL", sl_px + cfg.sl_slip_pts
132:             elif l <= tp_px:
133:                 exit_reason, exit_px = "TP", tp_px - cfg.tp_slip_pts
134: 
135:         if exit_reason is None:
136:             eq.loc[ts] = equity
137:             continue
138: 
139:         notional_exit = exit_px * size * spec.point_value
140:         pnl_per_contract = (exit_px - entry_px) * spec.point_value if side == "long" else (entry_px - exit_px) * spec.point_value
141:         gross = pnl_per_contract * size
142:         net = gross - entry_leg_fee - abs(notional_exit) * cfg.fee_rate
143: 
144:         equity += net
145:         trades.append({
146:             "symbol": symbol, "side": side, "entry_time": entry_time,
147:             "entry_px": entry_px, "exit_time": ts, "exit_px": exit_px,
148:             "reason": exit_reason, "size": size, "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
149:             "sl_px": sl_px, "tp_px": tp_px, "pnl": net, "gross": gross,
150:             "fees": entry_leg_fee + abs(notional_exit) * cfg.fee_rate
151:         })
152: 
153:         in_pos = False
154:         side = None
155:         entry_px = sl_px = tp_px = np.nan
156:         size = 0.0
157:         entry_time = None
158:         entry_leg_fee = 0.0
159:         eq.loc[ts] = equity
160: 
161:     # eindbar â€“ conservatief sluiten
162:     if in_pos and entry_time is not None:
163:         last_ts = df.index[-1]
164:         last_close = float(df["close"].iloc[-1])
165:         exit_px = (last_close - cfg.sl_slip_pts) if side == "long" else (last_close + cfg.sl_slip_pts)
166:         notional_exit = exit_px * size * spec.point_value
167:         pnl_per_contract = (exit_px - entry_px) * spec.point_value if side == "long" else (entry_px - exit_px) * spec.point_value
168:         gross = pnl_per_contract * size
169:         net = gross - entry_leg_fee - abs(notional_exit) * cfg.fee_rate
170:         equity += net
171: 
172:         trades.append({
173:             "symbol": symbol, "side": side or "NA", "entry_time": entry_time,
174:             "entry_px": entry_px, "exit_time": last_ts, "exit_px": exit_px, "reason": "EOD",
175:             "size": size, "atr_at_entry": float(atr.loc[entry_time]) if entry_time in atr.index else np.nan,
176:             "sl_px": sl_px, "tp_px": tp_px, "pnl": net, "gross": gross,
177:             "fees": entry_leg_fee + abs(notional_exit) * cfg.fee_rate
178:         })
179:         eq.iloc[-1] = equity
180: 
181:     eq = eq.ffill().fillna(cfg.equity0).rename("equity")
182:     trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
183: 
184:     def _max_dd(series: pd.Series):
185:         if len(series) == 0:
186:             return 0.0, 0
187:         roll_max = series.cummax()
188:         dd = series / roll_max - 1.0
189:         min_dd = float(dd.min())
190:         under = dd < 0
191:         dur = int(pd.Series(np.where(under, 1, 0), index=series.index).groupby((~under).cumsum()).sum().max() or 0)
192:         return min_dd * 100.0, dur
193: 
194:     total_ret = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) else 0.0
195:     max_dd_pct, dd_dur = _max_dd(eq)
196:     bar_rets = eq.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
197:     if bar_rets.std(ddof=1) > 0:
198:         bars_per_day = df.index.normalize().value_counts().mean() if len(df) else 60.0
199:         sharpe = float(bar_rets.mean() / bar_rets.std(ddof=1) * math.sqrt(bars_per_day * 252.0))
200:     else:
201:         sharpe = 0.0
202: 
203:     metrics = {
204:         "n_trades": int(len(trades_df)),
205:         "final_equity": float(eq.iloc[-1]),
206:         "return_total_pct": total_ret,
207:         "max_drawdown_pct": max_dd_pct,
208:         "dd_duration_bars": dd_dur,
209:         "sharpe": sharpe,
210:     }
211:     return eq, trades_df, metrics
````

## File: backtest/runner_vbt.py
````python
 1: import pandas as pd
 2: import vectorbt as vbt
 3: from typing import Dict, Any, Tuple
 4: from backtest.data_loader import fetch_data
 5: from strategies.order_block import order_block_signals
 6: from utils.metrics import summarize_equity_metrics
 7: 
 8: def run_backtest_vbt(
 9:     strategy_name: str,
10:     params: Dict[str, Any],
11:     symbol: str,
12:     timeframe: str,
13:     start: str,
14:     end: str,
15:     csv_path: str | None = None,
16: ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
17:     df = fetch_data(csv_path=csv_path, symbol=symbol, timeframe=timeframe, start=start, end=end)
18: 
19:     if strategy_name == "order_block":
20:         signals = order_block_signals(df, swing_w=params.get("lookback_bos", 3))
21:         entries = signals["long"]
22:         exits = ~signals["long"]  # Simpele exit op ~long
23:     else:
24:         raise ValueError(f"Strategy not implemented in VBT runner: {strategy_name}")
25: 
26:     pf = vbt.Portfolio.from_signals(
27:         df["close"],
28:         entries,
29:         exits,
30:         init_cash=20_000,
31:         fees=0.0002,
32:         slippage=0.0001,
33:     )
34: 
35:     equity = pf.value().to_frame("equity")
36:     trades = pf.trades.records_readable
37:     # Voeg pnl_cash toe voor metrics compatibiliteit als nodig
38:     if not trades.empty:
39:         trades["pnl_cash"] = trades["PnL"]
40:     metrics = summarize_equity_metrics(equity, trades)
41: 
42:     return equity, trades, metrics
````

## File: backtest/runner.py
````python
  1: # backtest/runner.py
  2: from __future__ import annotations
  3: 
  4: import json
  5: from pathlib import Path
  6: from typing import Any, Dict, Optional, Tuple
  7: 
  8: import pandas as pd
  9: 
 10: try:
 11:     from config import logger  # type: ignore
 12: except Exception:  # pragma: no cover
 13:     import logging
 14:     logging.basicConfig(level=logging.INFO)
 15:     logger = logging.getLogger("runner")
 16: 
 17: # ====== Jouw bestaande modules ======
 18: from strategies.breakout_params import BreakoutParams, SymbolSpec, DEFAULT_SPECS
 19: from backtest.breakout_exec import BTExecCfg, backtest_breakout
 20: 
 21: # Nieuwe ORB-variant
 22: from strategies.premarket_orb import ORBParams, premarket_orb_entries
 23: from backtest.orb_exec import ORBExecCfg, backtest_orb_bidirectional
 24: 
 25: 
 26: # ---------- helpers ----------
 27: def _coerce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
 28:     """Zorg voor tz-aware (UTC) DatetimeIndex; accepteert kolommen time/datetime/Date/date."""
 29:     if isinstance(df.index, pd.DatetimeIndex):
 30:         df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
 31:         return df
 32: 
 33:     for col in ("time", "datetime", "Date", "date", "timestamp"):
 34:         if col in df.columns:
 35:             idx = pd.to_datetime(df[col], utc=True, errors="coerce")
 36:             if not idx.isna().all():
 37:                 df = df.set_index(idx).drop(columns=[col])
 38:                 return df
 39: 
 40:     # laatste redmiddel: probeer eerste kolom
 41:     first = df.columns[0]
 42:     idx = pd.to_datetime(df[first], utc=True, errors="coerce")
 43:     if not idx.isna().all():
 44:         df = df.set_index(idx).drop(columns=[first])
 45:         return df
 46: 
 47:     raise ValueError("Could not find/convert a datetime index in CSV.")
 48: 
 49: def _load_csv(csv: Path) -> pd.DataFrame:
 50:     if not csv.exists():
 51:         raise FileNotFoundError(csv)
 52:     df = pd.read_csv(csv)
 53:     df = _coerce_dt_index(df)
 54:     df.columns = [c.strip().lower() for c in df.columns]
 55:     required = {"open", "high", "low", "close"}
 56:     missing = required.difference(df.columns)
 57:     if missing:
 58:         raise KeyError(f"CSV missing columns: {sorted(missing)}")
 59:     df = df.sort_index()
 60:     df = df[~df.index.duplicated(keep="first")]
 61:     return df
 62: 
 63: def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
 64:     if start:
 65:         df = df.loc[pd.Timestamp(start, tz="UTC") :]
 66:     if end:
 67:         df = df.loc[: pd.Timestamp(end, tz="UTC")]
 68:     return df
 69: 
 70: def _resolve_specs(symbol: str, specs: Optional[Dict[str, SymbolSpec]]) -> Dict[str, SymbolSpec]:
 71:     base_specs = dict(DEFAULT_SPECS)
 72:     if specs:
 73:         base_specs.update(specs)
 74:     if symbol not in base_specs:
 75:         base = symbol.split(".")[0]
 76:         if base in base_specs:
 77:             base_specs[symbol] = base_specs[base]
 78:         else:
 79:             base_specs[symbol] = SymbolSpec(name=symbol, point_value=1.0, min_step=0.1)
 80:     return base_specs
 81: 
 82: 
 83: # ---------- public API ----------
 84: def run_backtest(
 85:     *,
 86:     strategy_name: str,
 87:     params: Dict[str, Any],
 88:     symbol: str,
 89:     timeframe: str,
 90:     start: Optional[str] = None,
 91:     end: Optional[str] = None,
 92:     csv: Optional[str] = None,
 93:     engine: str = "native",
 94:     outdir: Optional[str] = "output",
 95: ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
 96:     """
 97:     Centrale runner. Retourneert (equity_df, trades_df, metrics_dict).
 98: 
 99:     strategy_name:
100:       - "breakout"       (jouw prev-day/opening-breakout)
101:       - "premarket_orb"  (Trader Tom premarket-ORB)
102:     engine:
103:       - "native" (hier), "vbt" is elders (runner_vbt).
104:     """
105:     if engine != "native":
106:         raise NotImplementedError("Only engine='native' is implemented in this runner.")
107: 
108:     if not csv:
109:         raise ValueError("CSV path is required. Provide --csv pointing to OHLC data.")
110: 
111:     df = _load_csv(Path(csv))
112:     df = _slice(df, start, end)
113:     logger.info("Loaded %s rows for %s %s", len(df), symbol, timeframe)
114: 
115:     specs = _resolve_specs(symbol, None)
116: 
117:     if strategy_name == "breakout":
118:         # === bestaande opening-breakout (prev-day based) ===
119:         p = BreakoutParams(
120:             window=int(params.get("window", 20)),
121:             atr_mult_sl=float(params.get("atr_sl", 1.0)),
122:             atr_mult_tp=float(params.get("atr_tp", 1.8)),
123:         )
124:         cfg = BTExecCfg(
125:             equity0=float(params.get("equity0", 20_000.0)),
126:             fee_rate=float(params.get("fee_bps", 2.0)) / 10_000.0,
127:             entry_slip_pts=float(params.get("entry_slip_pts", 0.1)),
128:             sl_slip_pts=float(params.get("sl_slip_pts", 0.5)),
129:             tp_slip_pts=float(params.get("tp_slip_pts", 0.0)),
130:             specs=specs,
131:             risk_frac=float(params.get("risk_pct", 1.0)) / 100.0,
132:             atr_n=int(params.get("atr_period", 14)),
133:         )
134:         open_window_bars = int(params.get("open_window_bars", 4))
135:         confirm = str(params.get("confirm", "close"))  # "close"|"wick"
136: 
137:         eq, trades, metrics = backtest_breakout(
138:             df=df,
139:             symbol=symbol,
140:             params=p,
141:             cfg=cfg,
142:             open_window_bars=open_window_bars,
143:             confirm=confirm,
144:         )
145: 
146:     elif strategy_name == "premarket_orb":
147:         # === nieuwe premarket ORB ===
148:         if "GER40" in symbol or "DAX" in symbol:
149:             orb_p = ORBParams(session_open_local="09:00", session_tz="Europe/Berlin",
150:                               premarket_minutes=int(params.get("premarket_minutes", 60)),
151:                               confirm=str(params.get("confirm", "close")))
152:         elif "US30" in symbol or "Dow" in symbol:
153:             orb_p = ORBParams(session_open_local="09:30", session_tz="America/New_York",
154:                               premarket_minutes=int(params.get("premarket_minutes", 60)),
155:                               confirm=str(params.get("confirm", "close")))
156:         else:
157:             orb_p = ORBParams(
158:                 session_open_local=str(params.get("session_open_local", "09:00")),
159:                 session_tz=str(params.get("session_tz", "Europe/Berlin")),
160:                 premarket_minutes=int(params.get("premarket_minutes", 60)),
161:                 confirm=str(params.get("confirm", "close")),
162:             )
163: 
164:         e_long, e_short = premarket_orb_entries(df, orb_p)
165: 
166:         p = BreakoutParams(
167:             window=int(params.get("window", 20)),
168:             atr_mult_sl=float(params.get("atr_sl", 1.0)),
169:             atr_mult_tp=float(params.get("atr_tp", 1.8)),
170:         )
171:         cfg = ORBExecCfg(
172:             equity0=float(params.get("equity0", 20_000.0)),
173:             fee_rate=float(params.get("fee_bps", 2.0)) / 10_000.0,
174:             entry_slip_pts=float(params.get("entry_slip_pts", 0.1)),
175:             sl_slip_pts=float(params.get("sl_slip_pts", 0.5)),
176:             tp_slip_pts=float(params.get("tp_slip_pts", 0.0)),
177:             specs=specs,
178:             risk_frac=float(params.get("risk_pct", 1.0)) / 100.0,
179:             atr_n=int(params.get("atr_period", 14)),
180:         )
181: 
182:         eq, trades, metrics = backtest_orb_bidirectional(
183:             df=df,
184:             symbol=symbol,
185:             params=p,
186:             cfg=cfg,
187:             entries_long=e_long,
188:             entries_short=e_short,
189:         )
190:     else:
191:         raise ValueError(f"Unknown strategy_name: {strategy_name}")
192: 
193:     # outputs
194:     if outdir:
195:         out = Path(outdir)
196:         out.mkdir(parents=True, exist_ok=True)
197:         (out / f"equity_{strategy_name}_{symbol}.csv").write_text(eq.to_csv())
198:         trades.to_csv(out / f"trades_{strategy_name}_{symbol}.csv", index=False)
199:         with open(out / f"metrics_{strategy_name}_{symbol}.json", "w", encoding="utf-8") as f:
200:             json.dump(metrics, f, indent=2)
201:         logger.info("[%s|%s] n_trades=%s ret=%.2f%% sharpe=%.2f maxDD=%.2f%%",
202:                     strategy_name, symbol, metrics.get("n_trades"),
203:                     metrics.get("return_total_pct", 0.0),
204:                     metrics.get("sharpe", 0.0),
205:                     metrics.get("max_drawdown_pct", 0.0))
206: 
207:     return eq.to_frame("equity"), trades, metrics
````

## File: cli/main.py
````python
  1: # cli/main.py
  2: from __future__ import annotations
  3: 
  4: from pathlib import Path
  5: from typing import Optional
  6: 
  7: import typer
  8: import pandas as pd
  9: from rich.table import Table
 10: from rich.console import Console
 11: 
 12: try:
 13:     from config import logger  # type: ignore
 14: except Exception:  # pragma: no cover
 15:     import logging
 16:     logging.basicConfig(level=logging.INFO)
 17:     logger = logging.getLogger("cli")
 18: 
 19: from backtest.runner import run_backtest
 20: from backtest.runner_vbt import run_backtest_vbt  # blijft beschikbaar
 21: 
 22: app = typer.Typer(
 23:     help="Sophy4Lite CLI for trading backtests",
 24:     no_args_is_help=True,
 25:     add_completion=False,
 26: )
 27: console = Console()
 28: 
 29: 
 30: @app.command()
 31: def backtest(
 32:     symbol: str = typer.Option("GER40.cash", help="Trading symbol, e.g. GER40.cash or US30.cash"),
 33:     timeframe: str = typer.Option("M1", help="Timeframe label (CSV moet M1 zijn voor ORB)"),
 34:     start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
 35:     end: str = typer.Option(..., help="End date YYYY-MM-DD"),
 36:     csv: Optional[Path] = typer.Option(None, help="Path to CSV data file (UTC OHLC)"),
 37:     engine: str = typer.Option("native", help="Backtest engine: native or vbt"),
 38:     strategy: str = typer.Option("breakout", help="Strategy: breakout | premarket_orb | order_block"),
 39:     # risk/fees/ATR
 40:     atr_period: int = typer.Option(14, help="ATR period"),
 41:     atr_sl: float = typer.Option(1.0, help="ATR SL multiplier"),
 42:     atr_tp: float = typer.Option(1.8, help="ATR TP multiplier"),
 43:     fee_bps: float = typer.Option(2.0, help="Fee per leg in bps"),
 44:     risk_pct: float = typer.Option(1.0, help="Risk per trade in % of equity"),
 45:     equity0: float = typer.Option(20_000.0, help="Starting equity"),
 46:     entry_slip_pts: float = typer.Option(0.1, help="Entry slippage (pts)"),
 47:     sl_slip_pts: float = typer.Option(0.5, help="SL slippage (pts)"),
 48:     tp_slip_pts: float = typer.Option(0.0, help="TP slippage (pts)"),
 49:     # opening-breakout specifics
 50:     open_window_bars: int = typer.Option(4, help="Bars after open (breakout only)"),
 51:     confirm: str = typer.Option("close", help="Confirmation: close|wick"),
 52:     # ORB specifics
 53:     session_open_local: Optional[str] = typer.Option(None, help="(ORB) cash-open, e.g. 09:00 or 09:30"),
 54:     session_tz: Optional[str] = typer.Option(None, help="(ORB) timezone, e.g. Europe/Berlin"),
 55:     premarket_minutes: int = typer.Option(60, help="(ORB) minutes for premarket range"),
 56:     # output
 57:     outdir: Optional[Path] = typer.Option(Path("output"), help="Output directory"),
 58: ):
 59:     """Run a backtest with either the native or vectorbt engine."""
 60:     logger.info(f"Starting backtest: strategy={strategy}, engine={engine}, symbol={symbol}, timeframe={timeframe}, csv={csv}")
 61:     if engine == "native" and csv is None:
 62:         console.print("[red]Error:[/red] CSV is required for native engine")
 63:         raise typer.Exit(code=2)
 64: 
 65:     params = {
 66:         "atr_period": atr_period,
 67:         "atr_sl": atr_sl,
 68:         "atr_tp": atr_tp,
 69:         "fee_bps": fee_bps,
 70:         "risk_pct": risk_pct,
 71:         "equity0": equity0,
 72:         "entry_slip_pts": entry_slip_pts,
 73:         "sl_slip_pts": sl_slip_pts,
 74:         "tp_slip_pts": tp_slip_pts,
 75:         "open_window_bars": open_window_bars,
 76:         "confirm": confirm,
 77:         "premarket_minutes": premarket_minutes,
 78:     }
 79:     if session_open_local:
 80:         params["session_open_local"] = session_open_local
 81:     if session_tz:
 82:         params["session_tz"] = session_tz
 83: 
 84:     try:
 85:         if engine == "native":
 86:             df_eq, trades, metrics = run_backtest(
 87:                 strategy_name=strategy,
 88:                 params=params,
 89:                 symbol=symbol,
 90:                 timeframe=timeframe,
 91:                 start=start,
 92:                 end=end,
 93:                 csv=str(csv) if csv else None,
 94:                 engine=engine,
 95:                 outdir=str(outdir) if outdir else None,
 96:             )
 97:         elif engine == "vbt":
 98:             # behoud jouw bestaande VBT-flow (bv. order_block)
 99:             if strategy == "breakout":
100:                 raise ValueError("Breakout strategy not supported in VBT engine; use native engine.")
101:             df_eq, trades, metrics = run_backtest_vbt(
102:                 strategy_name=strategy,
103:                 params=params,
104:                 symbol=symbol,
105:                 timeframe=timeframe,
106:                 start=start,
107:                 end=end,
108:                 csv_path=csv,
109:             )
110:         else:
111:             raise ValueError("engine must be 'native' or 'vbt'")
112: 
113:         # Tabel-summary (metrics die ontbreken worden 0/NA getoond)
114:         tbl = Table(title=f"Backtest Summary ({strategy}, {engine})")
115:         tbl.add_column("Metric"); tbl.add_column("Value", justify="right")
116:         order = ["final_equity","return_total_pct","n_trades","winrate_pct","avg_rr","sharpe","max_drawdown_pct"]
117:         for k in order:
118:             v = metrics.get(k, None)
119:             if isinstance(v, (int, float)):
120:                 tbl.add_row(k, f"{v:,.4f}")
121:             else:
122:                 tbl.add_row(k, "-" if v is None else str(v))
123:         console.print(tbl)
124: 
125:         # Opslaan
126:         out = outdir or Path("output")
127:         Path(out).mkdir(exist_ok=True)
128:         df_eq.to_csv(Path(out) / f"equity_{strategy}_{engine}.csv")
129:         trades.to_csv(Path(out) / f"trades_{strategy}_{engine}.csv", index=False)
130:         console.print(f"[green]Saved[/green] {Path(out) / f'equity_{strategy}_{engine}.csv'} and {Path(out) / f'trades_{strategy}_{engine}.csv'}")
131:         logger.info("Backtest completed successfully")
132: 
133:     except Exception as e:
134:         logger.error(f"Backtest failed: {str(e)}")
135:         console.print(f"[red]Error:[/red] {str(e)}")
136:         raise typer.Exit(code=1)
137: 
138: 
139: if __name__ == "__main__":
140:     app()
````

## File: live/live_trading.py
````python
  1: # Write the updated version of the file with the requested changes.
  2: from pathlib import Path
  3: 
  4: updated_code = """from typing import Optional, Dict, Any
  5: import MetaTrader5 as mt5
  6: import pandas as pd
  7: from datetime import datetime
  8: from config import logger, RISK_PER_TRADE, MAX_DAILY_LOSS, MAX_TOTAL_LOSS, STOP_AFTER_LOSSES
  9: from utils.position import _size
 10: # # # # from risk.ftmo import fixed_percent_sizing
 11: # TODO: Create risk/ftmo.py or remove this import
 12: # TODO: Create risk/ftmo.py or remove this import
 13: # TODO: Create risk/ftmo.py or remove this import
 14: # TODO: Create risk/ftmo.py or remove this import
 15: 
 16: class DailyState:
 17:     def __init__(self):
 18:         self.date = None
 19:         self.loss_streak = 0
 20:         self.start_equity = None
 21:         self.blocked = False
 22: 
 23: _state = DailyState()
 24: 
 25: def _reset_if_new_day():
 26:     today = datetime.utcnow().date()
 27:     if _state.date != today:
 28:         _state.date = today
 29:         _state.loss_streak = 0
 30:         _state.blocked = False
 31:         _state.start_equity = None
 32: 
 33: def _get_equity() -> float:
 34:     info = mt5.account_info()
 35:     return float(info.equity) if info else 0.0
 36: 
 37: def _guardrails_pass() -> bool:
 38:     if _state.blocked:
 39:         logger.warning(\"Daily trading blocked due to guardrails\")
 40:         return False
 41:     eq = _get_equity()
 42:     if _state.start_equity is None:
 43:         _state.start_equity = eq
 44:     day_dd = max(0.0, (_state.start_equity - eq) / max(_state.start_equity, 1e-9))
 45:     if day_dd >= MAX_DAILY_LOSS:
 46:         logger.error(\"Max daily loss reached -> blocking trading for today\")
 47:         _state.blocked = True
 48:         return False
 49:     return True
 50: 
 51: def place_trade(symbol: str, entry_price: float, sl_price: float, tp_price: float) -> Dict[str, Any]:
 52:     if not mt5.initialize():
 53:         return {\"success\": False, \"message\": f\"MT5 init failed: {mt5.last_error()}\"}
 54:     _reset_if_new_day()
 55:     if not _guardrails_pass():
 56:         return {\"success\": False, \"message\": \"Guardrails: daily blocked\"}
 57:     info = mt5.symbol_info(symbol)
 58:     if info is None:
 59:         return {\"success\": False, \"message\": f\"Symbol {symbol} not found\"}
 60:     if not info.visible:
 61:         mt5.symbol_select(symbol, True)
 62: 
 63:     eq = _get_equity()
 64: 
 65:     # --- New sizing logic using utils.position._size ---
 66:     size = _size(eq, entry_price, sl_price, info.trade_contract_size, RISK_PER_TRADE)
 67:     if size <= 0:
 68:         return {\"success\": False, \"message\": \"Position size calculation failed\"}
 69: 
 70:     request = {
 71:         \"action\": mt5.TRADE_ACTION_DEAL,
 72:         \"symbol\": symbol,
 73:         \"volume\": size,
 74:         \"type\": mt5.ORDER_TYPE_BUY if entry_price <= tp_price else mt5.ORDER_TYPE_SELL,
 75:         \"price\": entry_price,
 76:         \"sl\": sl_price,
 77:         \"tp\": tp_price,
 78:         \"magic\": 424242,
 79:         \"type_time\": mt5.ORDER_TIME_GTC,
 80:         \"type_filling\": mt5.ORDER_FILLING_IOC,
 81:         \"comment\": \"Sophy4Lite\",
 82:     }
 83:     result = mt5.order_send(request)
 84:     if result.retcode != mt5.TRADE_RETCODE_DONE:
 85:         logger.error(f\"Order failed: {result.comment} code={result.retcode}\")
 86:         return {\"success\": False, \"message\": result.comment, \"retcode\": result.retcode}
 87:     logger.info(f\"Trade executed {symbol} vol={size} @ {entry_price} SL={sl_price} TP={tp_price}\")
 88:     return {\"success\": True, \"ticket\": result.order, \"volume\": size}
 89: 
 90: def register_trade_result(pnl: float):
 91:     _reset_if_new_day()
 92:     if pnl < 0:
 93:         _state.loss_streak += 1
 94:         if _state.loss_streak >= STOP_AFTER_LOSSES:
 95:             _state.blocked = True
 96:             logger.warning(\"Stop after consecutive losses reached -> block for today\")
 97:     else:
 98:         _state.loss_streak = 0
 99: """
100: 
101: # Overwrite the original file location with the updated code
102: target_path = Path("/mnt/data/612d9232-7190-4390-82b9-be4a5e98ed7a.py")
103: target_path.write_text(updated_code, encoding="utf-8")
104: 
105: # Also save a copy with a clearer filename for download history
106: copy_path = Path("/mnt/data/place_trade_updated.py")
107: copy_path.write_text(updated_code, encoding="utf-8")
108: 
109: str(target_path), str(copy_path)
````

## File: risk/ftmo_guard.py
````python
 1: from __future__ import annotations
 2: from datetime import date
 3: from dataclasses import dataclass
 4: 
 5: @dataclass
 6: class FtmoRules:
 7:     max_daily_loss_pct: float = 0.05  # 5%
 8:     max_total_loss_pct: float = 0.10  # 10%
 9:     stop_after_losses: int = 2  # consecutive losing trades per day
10: 
11: class FtmoGuard:
12:     def __init__(self, initial_equity: float, rules: FtmoRules):
13:         self.initial_equity = initial_equity
14:         self.current_equity = initial_equity
15:         self.peak_equity = initial_equity
16:         self.daily_start_equity = initial_equity
17:         self.daily_loss = 0.0
18:         self.loss_streak = 0
19:         self.blocked = False
20:         self.current_day = None
21:         self.rules = rules
22: 
23:     def new_day(self, day: date, equity: float):
24:         if self.current_day != day:
25:             self.current_day = day
26:             self.daily_start_equity = equity
27:             self.daily_loss = 0.0
28:             self.loss_streak = 0
29:             self.blocked = False
30: 
31:     def update_equity(self, equity: float):
32:         self.current_equity = equity
33:         if equity > self.peak_equity:
34:             self.peak_equity = equity
35:         self.daily_loss = self.daily_start_equity - equity
36: 
37:     def pretrade_ok(self, worst_loss: float) -> bool:
38:         if self.blocked:
39:             return False
40:         projected_equity = self.current_equity - worst_loss
41:         if (self.daily_start_equity - projected_equity) / self.daily_start_equity > self.rules.max_daily_loss_pct:
42:             return False
43:         if (self.initial_equity - projected_equity) / self.initial_equity > self.rules.max_total_loss_pct:
44:             return False
45:         return True
46: 
47:     def allowed_now(self) -> bool:
48:         return not self.blocked
````

## File: scripts/diagnostics.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Sophy4Lite - data diagnostics & quick visuals.
  4: 
  5: Usage examples:
  6:   python -m scripts.diagnostics --csv data/GER40.cash_M15.csv --start 2023-01-01 --end 2023-03-31
  7:   python -m scripts.diagnostics --csv data/XAUUSD_H1.csv
  8: 
  9: Outputs:
 10:   - output/plots/df_overview.txt         (shape, date range, tz, columns, NaN stats)
 11:   - output/plots/close.png               (Close price over time)
 12:   - output/plots/volume.png              (Volume over time, if available)
 13:   - output/plots/atr14.png               (Rolling ATR(14))
 14:   - output/plots/returns_hist.png        (Histogram of 1-bar returns)
 15:   - output/plots/missing_heatmap.png     (Simple missingness visualization)
 16: """
 17: from __future__ import annotations
 18: import argparse
 19: from pathlib import Path
 20: import numpy as np
 21: import pandas as pd
 22: import matplotlib.pyplot as plt
 23: 
 24: from backtest.data_loader import fetch_data
 25: 
 26: # ---------- Helpers ----------
 27: 
 28: def ensure_outdir() -> Path:
 29:     outdir = Path("output/plots")
 30:     outdir.mkdir(parents=True, exist_ok=True)
 31:     return outdir
 32: 
 33: def save_txt(outpath: Path, text: str) -> None:
 34:     outpath.write_text(text, encoding="utf-8")
 35: 
 36: def atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, period: int = 14) -> pd.Series:
 37:     h, l, c = series_high, series_low, series_close
 38:     tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
 39:     return tr.rolling(period).mean()
 40: 
 41: def plot_series(y: pd.Series, title: str, filepath: Path) -> None:
 42:     fig = plt.figure(figsize=(10, 4))
 43:     y.plot()
 44:     plt.title(title)
 45:     plt.xlabel("time")
 46:     plt.ylabel(y.name or "")
 47:     plt.tight_layout()
 48:     fig.savefig(filepath, dpi=120)
 49:     plt.close(fig)
 50: 
 51: def plot_hist(values: pd.Series, bins: int, title: str, filepath: Path) -> None:
 52:     fig = plt.figure(figsize=(6, 4))
 53:     plt.hist(values.dropna().values, bins=bins)
 54:     plt.title(title)
 55:     plt.tight_layout()
 56:     fig.savefig(filepath, dpi=120)
 57:     plt.close(fig)
 58: 
 59: def plot_missing_heatmap(df: pd.DataFrame, filepath: Path) -> None:
 60:     """Simple missingness plot: rows vs columns (no seaborn dependency)."""
 61:     miss = df.isna().astype(int)
 62:     fig = plt.figure(figsize=(8, 4))
 63:     plt.imshow(miss.T, aspect="auto", interpolation="nearest")
 64:     plt.yticks(range(len(df.columns)), df.columns)
 65:     plt.xticks([])
 66:     plt.title("Missingness (1=NaN, 0=valid)")
 67:     plt.tight_layout()
 68:     fig.savefig(filepath, dpi=120)
 69:     plt.close(fig)
 70: 
 71: # ---------- Main diagnostics ----------
 72: 
 73: def summarize_df(df: pd.DataFrame) -> str:
 74:     lines = []
 75:     lines.append(f"shape: {df.shape}")
 76:     if isinstance(df.index, pd.DatetimeIndex):
 77:         lines.append(f"index: DatetimeIndex tz={df.index.tz}")
 78:         if len(df.index) > 0:
 79:             lines.append(f"range: {df.index.min()} -> {df.index.max()}")
 80:         # crude frequency guess
 81:         if len(df.index) >= 3:
 82:             deltas = df.index.to_series().diff().dropna().value_counts().head(3)
 83:             lines.append("top time deltas:")
 84:             for delta, cnt in deltas.items():
 85:                 lines.append(f"  {delta}  ({cnt} steps)")
 86:     else:
 87:         lines.append("index: (not DatetimeIndex)")
 88:     lines.append(f"columns: {list(df.columns)}")
 89:     miss = df.isna().mean() * 100.0
 90:     for c, p in miss.items():
 91:         lines.append(f"NaN % in {c:>6}: {p:5.2f}%")
 92:     dups = df.index.duplicated().sum() if isinstance(df.index, pd.DatetimeIndex) else 0
 93:     lines.append(f"duplicate index entries: {dups}")
 94:     return "\n".join(lines)
 95: 
 96: def main():
 97:     parser = argparse.ArgumentParser(description="Sophy4Lite data diagnostics & visuals")
 98:     parser.add_argument("--csv", type=str, required=True, help="Path to CSV with time,open,high,low,close[,volume]")
 99:     parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
100:     parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
101:     args = parser.parse_args()
102: 
103:     # Load
104:     df = fetch_data(csv_path=args.csv, start=args.start, end=args.end)
105: 
106:     # Ensure required columns
107:     cols = ["open", "high", "low", "close"]
108:     for c in cols:
109:         if c not in df.columns:
110:             raise KeyError(f"Required column missing: {c}")
111:     if "volume" not in df.columns:
112:         df["volume"] = np.nan
113: 
114:     outdir = ensure_outdir()
115: 
116:     # Text overview
117:     overview = summarize_df(df)
118:     save_txt(outdir / "df_overview.txt", overview)
119: 
120:     # Basic plots
121:     plot_series(df["close"], "Close", outdir / "close.png")
122:     if df["volume"].notna().any():
123:         plot_series(df["volume"], "Volume", outdir / "volume.png")
124: 
125:     # ATR(14)
126:     atr14 = atr(df["high"], df["low"], df["close"], period=14).rename("ATR(14)")
127:     plot_series(atr14, "ATR(14)", outdir / "atr14.png")
128: 
129:     # Returns histogram
130:     rets = df["close"].pct_change()
131:     plot_hist(rets, bins=50, title="1-bar Returns", filepath=outdir / "returns_hist.png")
132: 
133:     # Missingness
134:     plot_missing_heatmap(df[["open","high","low","close","volume"]], outdir / "missing_heatmap.png")
135: 
136:     print("[OK] Wrote diagnostics to", outdir.resolve())
137:     print(overview)
138: 
139: if __name__ == "__main__":
140:     main()
````

## File: scripts/export_mt5_data.py
````python
 1: import MetaTrader5 as mt5
 2: import pandas as pd
 3: from pathlib import Path
 4: import typer
 5: 
 6: def export(
 7:     symbol: str = typer.Option(..., help="Instrument, bv. GER40.cash of XAUUSD"),
 8:     tf: str = typer.Option("M15", help="Timeframe: M1, M5, M15, H1, D1"),
 9:     start: str = typer.Option(..., help="Startdatum (YYYY-MM-DD)"),
10:     end: str = typer.Option(..., help="Einddatum (YYYY-MM-DD)")
11: ):
12:     """ Exporteer MT5 rates naar CSV in de project-root/data/ map """
13: 
14:     # Mapping van string → MT5 constant
15:     tf_map = {
16:         "M1": mt5.TIMEFRAME_M1,
17:         "M5": mt5.TIMEFRAME_M5,
18:         "M15": mt5.TIMEFRAME_M15,
19:         "H1": mt5.TIMEFRAME_H1,
20:         "D1": mt5.TIMEFRAME_D1,
21:     }
22:     if tf not in tf_map:
23:         raise ValueError(f"Unsupported timeframe {tf}. Kies uit {list(tf_map.keys())}")
24: 
25:     timeframe = tf_map[tf]
26:     start_ts = pd.Timestamp(start).to_pydatetime()
27:     end_ts = pd.Timestamp(end).to_pydatetime()
28: 
29:     # Init MT5
30:     if not mt5.initialize():
31:         raise RuntimeError(f"MT5 init failed, error code: {mt5.last_error()}")
32: 
33:     rates = mt5.copy_rates_range(symbol, timeframe, start_ts, end_ts)
34:     if rates is None or len(rates) == 0:
35:         raise RuntimeError(f"Geen data ontvangen voor {symbol} {tf} {start}–{end}")
36: 
37:     df = pd.DataFrame(rates)
38:     df = df.rename(columns={
39:         "time": "time",
40:         "open": "open",
41:         "high": "high",
42:         "low": "low",
43:         "close": "close",
44:         "tick_volume": "volume"
45:     })
46:     df["time"] = pd.to_datetime(df["time"], unit="s")
47: 
48:     # Pad altijd relatief aan project-root
49:     ROOT = Path(__file__).resolve().parent.parent
50:     outpath = ROOT / "data" / f"{symbol}_{tf}.csv"
51:     outpath.parent.mkdir(exist_ok=True)
52:     df.to_csv(outpath, index=False)
53: 
54:     print(f"[OK] Wrote {len(df)} rows to {outpath}")
55: 
56:     mt5.shutdown()
57: 
58: 
59: if __name__ == "__main__":
60:     typer.run(export)
````

## File: scripts/export_mt5_orb.py
````python
 1: # scripts/export_mt5_orb.py
 2: from __future__ import annotations
 3: import sys
 4: from pathlib import Path
 5: from datetime import datetime, timedelta, timezone
 6: import pandas as pd
 7: import MetaTrader5 as mt5
 8: 
 9: TF_MAP = {
10:     "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
11:     "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1,
12: }
13: 
14: def _backfill_until(symbol: str, tf: int, start: datetime, end: datetime,
15:                     chunk: int = 100_000, max_iters: int = 200) -> pd.DataFrame:
16:     """
17:     Trek data achteruit vanaf 'end' in blokken totdat 'start' is bereikt of geen data meer komt.
18:     """
19:     # "wakker maken"
20:     mt5.symbol_select(symbol, True)
21:     mt5.copy_rates_from_pos(symbol, tf, 0, 1)
22: 
23:     # eerste blok rond 'end'
24:     r = mt5.copy_rates_from(symbol, tf, end, chunk)
25:     if r is None or len(r) == 0:
26:         return pd.DataFrame(columns=["time","open","high","low","close","volume"])
27: 
28:     df = pd.DataFrame(r)
29:     df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
30: 
31:     # achteruit blijven trekken
32:     for _ in range(max_iters):
33:         tmin = df["time"].min().to_pydatetime().replace(tzinfo=timezone.utc)
34:         if tmin <= start:  # genoeg historie
35:             break
36:         r = mt5.copy_rates_from(symbol, tf, tmin - timedelta(seconds=1), chunk)
37:         if r is None or len(r) == 0:
38:             # kleine nudge en nog één poging
39:             mt5.copy_rates_from_pos(symbol, tf, 0, 1)
40:             r = mt5.copy_rates_from(symbol, tf, tmin - timedelta(seconds=1), chunk)
41:             if r is None or len(r) == 0:
42:                 break
43:         tmp = pd.DataFrame(r)
44:         tmp["time"] = pd.to_datetime(tmp["time"], unit="s", utc=True)
45:         df = pd.concat([df, tmp], ignore_index=True).drop_duplicates("time").sort_values("time")
46: 
47:     # trimmen op [start, end] + volume kiezen
48:     df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))].copy()
49:     if "real_volume" in df.columns and df["real_volume"].notna().any():
50:         vol = df["real_volume"]
51:     else:
52:         vol = df["tick_volume"] if "tick_volume" in df.columns else 0
53: 
54:     out = pd.DataFrame({
55:         "time": df["time"],
56:         "open": df["open"],
57:         "high": df["high"],
58:         "low": df["low"],
59:         "close": df["close"],
60:         "volume": vol,
61:     }).drop_duplicates("time").sort_values("time")
62:     return out
63: 
64: def main():
65:     if len(sys.argv) != 6:
66:         print("Usage: python scripts/export_mt5_orb.py SYMBOL TF START END OUTCSV", file=sys.stderr)
67:         sys.exit(2)
68:     symbol, tf_label, start_s, end_s, out = sys.argv[1:]
69:     tf = TF_MAP.get(tf_label)
70:     if tf is None:
71:         sys.exit(f"Unknown TF: {tf_label}")
72: 
73:     start = datetime.fromisoformat(start_s).replace(tzinfo=timezone.utc)
74:     end   = datetime.fromisoformat(end_s).replace(tzinfo=timezone.utc)
75: 
76:     if not mt5.initialize():
77:         sys.exit("mt5.initialize() failed")
78:     try:
79:         df = _backfill_until(symbol, tf, start, end)
80:     finally:
81:         mt5.shutdown()
82: 
83:     if df.empty:
84:         sys.exit(f"No data received for {symbol} {tf_label}")
85: 
86:     outp = Path(out); outp.parent.mkdir(parents=True, exist_ok=True)
87:     df.to_csv(outp, index=False)
88:     print(f"[OK] {symbol} {tf_label} -> {outp} rows={len(df)} range={df['time'].min()} .. {df['time'].max()}")
89: 
90: if __name__ == "__main__":
91:     main()
````

## File: scripts/plot_orb_days.py
````python
  1: # scripts/plot_orb_days.py
  2: from __future__ import annotations
  3: 
  4: import argparse
  5: from dataclasses import dataclass
  6: from datetime import datetime, timedelta
  7: from zoneinfo import ZoneInfo
  8: 
  9: import matplotlib.dates as mdates
 10: import matplotlib.pyplot as plt
 11: import numpy as np
 12: import pandas as pd
 13: from pathlib import Path
 14: import sys
 15: import textwrap
 16: 
 17: 
 18: @dataclass(frozen=True)
 19: class ORConfig:
 20:     session_open_local: str = "09:00"          # DAX cash open (lokale beurs-tijd)
 21:     session_tz: str = "Europe/Berlin"
 22:     premarket_minutes: int = 60
 23:     minutes_after_open: int = 60               # hoeveel minuten na de open in de plot
 24:     only_days_with_trades: bool = True         # alleen dagen plotten met een trade in trades_csv
 25:     outdir: Path = Path("output/plots/GER40.cash")
 26: 
 27: 
 28: def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
 29:     """Zorg dat timestamps tz-aware UTC zijn."""
 30:     if ts.tzinfo is None:
 31:         return ts.tz_localize("UTC")
 32:     return ts.tz_convert("UTC")
 33: 
 34: 
 35: def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
 36:     if "time" in df.columns:
 37:         df = df.set_index("time")
 38:     idx = pd.to_datetime(df.index, utc=True)
 39:     # Als het al tz-aware maar niet UTC is, convert:
 40:     if idx.tz is None:
 41:         idx = idx.tz_localize("UTC")
 42:     else:
 43:         idx = idx.tz_convert("UTC")
 44:     df = df.copy()
 45:     df.index = idx
 46:     return df.sort_index()
 47: 
 48: 
 49: def _parse_time_hhmm(hhmm: str) -> tuple[int, int]:
 50:     try:
 51:         hh, mm = hhmm.split(":")
 52:         return int(hh), int(mm)
 53:     except Exception:
 54:         raise ValueError(f"Invalid time '{hhmm}', expected HH:MM")
 55: 
 56: 
 57: def _opening_ts_utc(day: pd.Timestamp, open_local: str, tz: str) -> pd.Timestamp:
 58:     """Return de cash open als UTC timestamp voor een gegeven kalenderdag (YYYY-MM-DD in die TZ)."""
 59:     h, m = _parse_time_hhmm(open_local)
 60:     local = pd.Timestamp(day.date()).replace(hour=h, minute=m, tzinfo=ZoneInfo(tz))
 61:     return local.tz_convert("UTC")
 62: 
 63: 
 64: def _compute_or(df_utc: pd.DataFrame, open_utc: pd.Timestamp, premarket_min: int) -> tuple[float, float, pd.Timestamp, pd.Timestamp]:
 65:     """Bereken OR uit premarket venster: [open - premarket, open)."""
 66:     start = open_utc - pd.Timedelta(minutes=premarket_min)
 67:     prem = df_utc.loc[start:open_utc - pd.Timedelta(microseconds=1)]
 68:     if prem.empty:
 69:         raise ValueError("Geen premarket data in venster. Check je CSV en tijden.")
 70:     return float(prem["high"].max()), float(prem["low"].min()), start, open_utc
 71: 
 72: 
 73: def _candles(ax, df: pd.DataFrame):
 74:     """Eenvoudige M1 candlesticks (zonder extra libs)."""
 75:     # Verwacht index=UTC time, kolommen: open, high, low, close
 76:     times = mdates.date2num(df.index.to_pydatetime())
 77:     o = df["open"].values
 78:     h = df["high"].values
 79:     l = df["low"].values
 80:     c = df["close"].values
 81: 
 82:     width = 1 / (24 * 60) * 0.8  # 0.8 minuut breed
 83:     up = c >= o
 84:     down = ~up
 85: 
 86:     # wicks
 87:     ax.vlines(times, l, h, linewidth=1)
 88: 
 89:     # bodies
 90:     ax.bar(times[up], c[up] - o[up], width, bottom=o[up], align="center", edgecolor="black")
 91:     ax.bar(times[down], o[down] - c[down], width, bottom=c[down], align="center", edgecolor="black")
 92: 
 93: 
 94: def _load_trades(trades_csv: Path) -> pd.DataFrame:
 95:     t = pd.read_csv(trades_csv)
 96:     cols = {c.lower(): c for c in t.columns}
 97:     # Verwachte kolommen (probeer flexibel te mappen)
 98:     required = ["symbol", "entry_time", "entry_px", "side"]
 99:     for r in required:
100:         if r not in [c.lower() for c in t.columns]:
101:             raise ValueError(f"Trades CSV mist kolom '{r}' (gevonden: {list(t.columns)})")
102:     # Normaliseer kolomnamen
103:     t.columns = [c.lower() for c in t.columns]
104:     # Timestamps -> UTC
105:     t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
106:     if "exit_time" in t.columns:
107:         t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
108:     return t
109: 
110: 
111: def _plot_one_day(df_utc: pd.DataFrame, trades_day: pd.DataFrame, day_local: pd.Timestamp, cfg: ORConfig, symbol: str, outdir: Path):
112:     # Bereken open & OR
113:     open_utc = _opening_ts_utc(day_local, cfg.session_open_local, cfg.session_tz)
114:     or_high, or_low, pre_start, pre_end = _compute_or(df_utc, open_utc, cfg.premarket_minutes)
115: 
116:     # Plot venster: premarket + N min na open
117:     end_utc = open_utc + pd.Timedelta(minutes=cfg.minutes_after_open)
118:     win = df_utc.loc[pre_start:end_utc]
119:     if win.empty:
120:         print(f"[WARN] Geen bars voor {day_local.date()} — skip")
121:         return
122: 
123:     fig, ax = plt.subplots(figsize=(12, 6))
124:     _candles(ax, win)
125: 
126:     # Premarket shading
127:     ax.axvspan(mdates.date2num(pre_start.to_pydatetime()),
128:                mdates.date2num(pre_end.to_pydatetime()),
129:                alpha=0.15)
130: 
131:     # OR lijnen
132:     ax.axhline(or_high, linestyle="--")
133:     ax.axhline(or_low, linestyle="--")
134: 
135:     # Entries/Exits/SL/TP markers voor deze dag
136:     for _, r in trades_day.iterrows():
137:         et: pd.Timestamp = r["entry_time"]
138:         if not (open_utc <= et <= end_utc):
139:             continue
140:         ep = float(r["entry_px"])
141:         side = str(r["side"]).lower()
142: 
143:         marker = "^" if side == "long" else "v"
144:         ax.plot(mdates.date2num(et.to_pydatetime()), ep, marker, markersize=9)
145: 
146:         # Exit
147:         if "exit_time" in trades_day.columns and pd.notna(r.get("exit_time")):
148:             xt = pd.to_datetime(r["exit_time"], utc=True)
149:             if open_utc <= xt <= end_utc:
150:                 xp = float(r.get("exit_px", np.nan)) if "exit_px" in trades_day.columns else np.nan
151:                 if np.isfinite(xp):
152:                     ax.plot(mdates.date2num(xt.to_pydatetime()), xp, "x", markersize=9)
153: 
154:         # SL/TP (indien aanwezig)
155:         for key, ls in [("sl", ":"), ("tp", "-.")]:
156:             if key in trades_day.columns and pd.notna(r.get(key)):
157:                 lvl = float(r[key])
158:                 ax.axhline(lvl, linestyle=ls, linewidth=1)
159: 
160:     # As formatting
161:     ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo("UTC")))
162:     ax.set_title(f"{symbol} — {day_local.date()}  (premarket={cfg.premarket_minutes}m, OR=[{or_low:.1f},{or_high:.1f}])")
163:     ax.set_ylabel("Price")
164:     ax.grid(True, alpha=0.25)
165:     fig.autofmt_xdate()
166: 
167:     outdir.mkdir(parents=True, exist_ok=True)
168:     fp = outdir / f"{symbol}_{day_local.date()}.png"
169:     fig.savefig(fp, dpi=150, bbox_inches="tight")
170:     plt.close(fig)
171:     print(f"[OK] Saved {fp}")
172: 
173: 
174: def main():
175:     p = argparse.ArgumentParser(
176:         description="Plot ORB-dagcharts voor DAX met OR-zone, premarket en trades.",
177:         formatter_class=argparse.RawDescriptionHelpFormatter,
178:         epilog=textwrap.dedent("""\
179:         Voorbeeld:
180:           python -m scripts.plot_orb_days --symbol GER40.cash --csv data/GER40.cash_M1.csv --trades output/trades_premarket_orb_native.csv \\
181:                  --start 2025-05-15 --end 2025-08-29 --session-open-local 09:00 --session-tz Europe/Berlin --premarket-minutes 60
182:         """)
183:     )
184:     p.add_argument("--symbol", required=True)
185:     p.add_argument("--csv", required=True, help="M1 CSV met kolommen time,open,high,low,close[,volume] in UTC")
186:     p.add_argument("--trades", required=True, help="Trades CSV uit je backtest")
187:     p.add_argument("--start", required=False)
188:     p.add_argument("--end", required=False)
189:     p.add_argument("--session-open-local", default="09:00")
190:     p.add_argument("--session-tz", default="Europe/Berlin")
191:     p.add_argument("--premarket-minutes", type=int, default=60)
192:     p.add_argument("--minutes-after-open", type=int, default=60)
193:     p.add_argument("--only-days-with-trades", action="store_true", default=False)
194:     p.add_argument("--outdir", default="output/plots/GER40.cash")
195: 
196:     args = p.parse_args()
197: 
198:     cfg = ORConfig(
199:         session_open_local=args.session_open_local,
200:         session_tz=args.session_tz,
201:         premarket_minutes=args.premarket_minutes,
202:         minutes_after_open=args.minutes_after_open,
203:         only_days_with_trades=args.only_days_with_trades,
204:         outdir=Path(args.outdir),
205:     )
206: 
207:     # Load data
208:     df = pd.read_csv(args.csv, parse_dates=["time"])
209:     df = _ensure_utc_index(df[["time", "open", "high", "low", "close"]])
210: 
211:     trades = _load_trades(Path(args.trades))
212:     trades = trades[trades["symbol"].astype(str).str.lower() == args.symbol.lower()].copy()
213: 
214:     # Maak lijst van dagen
215:     if args.start:
216:         start_day = pd.Timestamp(args.start).tz_localize(ZoneInfo(cfg.session_tz))
217:     else:
218:         start_day = df.index[0].tz_convert(ZoneInfo(cfg.session_tz))
219:     if args.end:
220:         end_day = pd.Timestamp(args.end).tz_localize(ZoneInfo(cfg.session_tz))
221:     else:
222:         end_day = df.index[-1].tz_convert(ZoneInfo(cfg.session_tz))
223: 
224:     # Dagen in lokale sessietijd
225:     days = pd.date_range(start=start_day.normalize(), end=end_day.normalize(), freq="D", tz=ZoneInfo(cfg.session_tz))
226: 
227:     if cfg.only_days_with_trades and not trades.empty:
228:         tr_days = trades["entry_time"].dt.tz_convert(cfg.session_tz).dt.normalize().unique()
229:         days = pd.DatetimeIndex(tr_days).sort_values()
230: 
231:     # Plot elke dag
232:     for day_local in days:
233:         # Subset aan trades van deze dag
234:         mask = trades["entry_time"].dt.tz_convert(cfg.session_tz).dt.normalize() == day_local.normalize()
235:         trades_day = trades.loc[mask]
236:         try:
237:             _plot_one_day(df, trades_day, pd.Timestamp(day_local), cfg, args.symbol, cfg.outdir)
238:         except Exception as e:
239:             print(f"[WARN] Dag {day_local.date()} overgeslagen: {e}", file=sys.stderr)
240: 
241: 
242: if __name__ == "__main__":
243:     main()
````

## File: scripts/run_backtest_demo.py
````python
  1: # scripts/run_backtest_demo.py
  2: from __future__ import annotations
  3: import json
  4: import math
  5: import os
  6: from pathlib import Path
  7: from typing import Optional, Dict, Any
  8: 
  9: import numpy as np
 10: import pandas as pd
 11: 
 12: from config import logger
 13: from backtest.data_loader import fetch_data
 14: from strategies.breakout_signals import breakout_long
 15: from utils.data_health import health_line
 16: from utils.days import summarize_day_health
 17: from utils.plot import save_equity_and_dd  # module heet 'plot'
 18: 
 19: # -------- metrics helpers (lite, zonder extra deps) --------
 20: def _max_drawdown(series: pd.Series) -> Dict[str, Any]:
 21:     """Return max drawdown pct en duur (bars)."""
 22:     roll_max = series.cummax()
 23:     dd = series / roll_max - 1.0
 24:     min_dd = float(dd.min())  # negatief getal
 25:     # duur: langste aaneengesloten periode onder vorige top
 26:     under = dd < 0
 27:     # tel aaneengesloten 'under' streaks
 28:     max_dur = int(
 29:         pd.Series(np.where(under, 1, 0), index=series.index)
 30:         .groupby((~under).cumsum())  # blokken tussen toppen
 31:         .sum()
 32:         .max()
 33:         or 0
 34:     )
 35:     return {"max_drawdown_pct": min_dd * 100.0, "dd_duration_bars": max_dur}
 36: 
 37: def _sharpe_from_bar_returns(bar_rets: pd.Series, bars_per_year: float) -> float:
 38:     """0% rf, annualized Sharpe op basis van bar-returns."""
 39:     if len(bar_rets) < 2:
 40:         return 0.0
 41:     mu = float(bar_rets.mean())
 42:     sigma = float(bar_rets.std(ddof=1))
 43:     if sigma == 0:
 44:         return 0.0
 45:     return (mu / sigma) * math.sqrt(bars_per_year)
 46: 
 47: def _cagr_from_equity(eq: pd.Series, trading_days: int) -> float:
 48:     """CAGR op basis van trading-dagen (≈252/jaar)."""
 49:     if len(eq) == 0 or eq.iloc[0] <= 0:
 50:         return 0.0
 51:     years = max(trading_days, 1) / 252.0
 52:     return float(eq.iloc[-1]) ** (1.0 / years) - 1.0
 53: 
 54: # -----------------------------------------------------------
 55: 
 56: def run(csv: str, start: Optional[str] = None, end: Optional[str] = None, window: int = 20) -> Path:
 57:     # 1) Data laden
 58:     df = fetch_data(csv_path=csv, start=start, end=end)
 59:     if not isinstance(df.index, pd.DatetimeIndex):
 60:         raise TypeError("Data index must be DatetimeIndex")
 61:     df = df.sort_index()
 62: 
 63:     # 2) Health + dag-samenvatting (alleen loggen)
 64:     logger.info(health_line(df, expected_freq="15min"))
 65:     days, min_bars, mean_bars = summarize_day_health(df)
 66:     logger.info(f"DAYS {{'count': {days}, 'min_bars': {min_bars}, 'mean_bars': {mean_bars:.1f}}}")
 67: 
 68:     # 3) Simpel breakout-signaal (gap-safe, bar-based)  -> boolean Series
 69:     sig = breakout_long(df["close"], df["high"], window=window).astype("boolean")
 70: 
 71:     # 4) Naïeve equity (1x notional; geen fees/slippage)
 72:     pos = sig.fillna(False).astype(int)                 # 0/1
 73:     bar_rets = df["close"].pct_change().fillna(0.0) * pos  # bar-PnL
 74:     eq = (1.0 + bar_rets).cumprod().rename("Equity")
 75: 
 76:     # 5) Metrics (lite maar nuttig)
 77:     bars_per_year = max(mean_bars, 1.0) * 252.0 if days > 0 else 252.0 * 56  # 56≈bars/dag @15m voor +/− 14h sessie
 78:     md = _max_drawdown(eq)
 79:     metrics = {
 80:         "entries": int((sig & (~sig.shift(1).fillna(False))).sum()),
 81:         "bars": int(len(df)),
 82:         "final_equity": float(eq.iloc[-1]) if len(eq) else 1.0,
 83:         "return_total_pct": (float(eq.iloc[-1]) - 1.0) * 100.0 if len(eq) else 0.0,
 84:         "cagr_pct": _cagr_from_equity(eq, days) * 100.0 if days else 0.0,
 85:         "sharpe": _sharpe_from_bar_returns(bar_rets.replace([np.inf, -np.inf], 0.0), bars_per_year),
 86:         **md,
 87:     }
 88: 
 89:     # 6) Output pad + visuals
 90:     outdir = Path("output/plots") / pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
 91:     outdir.mkdir(parents=True, exist_ok=True)
 92: 
 93:     if os.getenv("SOPHY_VIZ") == "1":
 94:         save_equity_and_dd(eq, outdir)
 95: 
 96:     # 7) Bewaar equity + metrics
 97:     eq.to_csv(outdir / "equity.csv", index=True)
 98:     with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
 99:         json.dump(metrics, f, indent=2)
100: 
101:     # 8) Log human-friendly samenvatting
102:     logger.info(
103:         "METRICS "
104:         f"entries={metrics['entries']} total_return={metrics['return_total_pct']:.2f}% "
105:         f"CAGR={metrics['cagr_pct']:.2f}% sharpe={metrics['sharpe']:.2f} "
106:         f"maxDD={metrics['max_drawdown_pct']:.2f}% dur={metrics['dd_duration_bars']} bars"
107:     )
108: 
109:     return outdir
110: 
111: if __name__ == "__main__":
112:     import argparse
113:     p = argparse.ArgumentParser(description="Sophy4Lite breakout demo (data-health + viz + metrics)")
114:     p.add_argument("--csv", required=True, help="Path to CSV (time,open,high,low,close[,volume])")
115:     p.add_argument("--start", default=None)
116:     p.add_argument("--end", default=None)
117:     p.add_argument("--window", type=int, default=20)
118:     args = p.parse_args()
119:     run(args.csv, args.start, args.end, window=args.window)
````

## File: strategies/__init__.py
````python
 1: # strategies/__init__.py
 2: """
 3: Sophy4Lite strategies package.
 4: 
 5: Exports:
 6: - Breakout signals (long, opening breakout)
 7: - Breakout parameters & helpers (ATR, levels, specs)
 8: """
 9: 
10: from .breakout_signals import (
11:     breakout_long,
12:     opening_breakout_long,
13: )
14: from .breakout_params import (
15:     BreakoutParams,
16:     SymbolSpec,
17:     DEFAULT_SPECS,
18:     daily_levels,
19: )
20: 
21: __all__ = [
22:     "breakout_long",
23:     "opening_breakout_long",
24:     "BreakoutParams",
25:     "SymbolSpec",
26:     "DEFAULT_SPECS",
27:     "daily_levels",
28: ]
````

## File: strategies/breakout_params.py
````python
 1: # strategies/breakout_params.py
 2: from __future__ import annotations
 3: from dataclasses import dataclass
 4: from typing import Dict
 5: import pandas as pd
 6: 
 7: @dataclass(frozen=True)
 8: class SymbolSpec:
 9:     """Contract-/symboolspecificaties voor sizing en validatie."""
10:     name: str
11:     point_value: float = 1.0     # contract point value / pip value
12:     min_step: float = 0.01       # minimale prijsstap
13: 
14: # Voeg hier eventuele extra symbolen toe wanneer nodig.
15: DEFAULT_SPECS: Dict[str, SymbolSpec] = {
16:     "GER40.cash": SymbolSpec("GER40.cash", point_value=1.0, min_step=0.5),
17: }
18: 
19: @dataclass
20: class BreakoutParams:
21:     """Parameters voor een eenvoudige breakout-strategie."""
22:     window: int = 20
23:     atr_mult_sl: float = 2.0
24:     atr_mult_tp: float = 3.0
25: 
26: def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
27:     """
28:     Per dag simpele high/low-levels teruggeven.
29:     Vereist kolommen: ['high', 'low'] en DatetimeIndex (tz mag, bijv. UTC).
30:     Output: DataFrame met index=dag en kolommen=['day_high','day_low'].
31:     """
32:     if not isinstance(df.index, pd.DatetimeIndex):
33:         raise TypeError("Index must be DatetimeIndex")
34:     if not {"high", "low"}.issubset(df.columns):
35:         raise KeyError("DataFrame must contain 'high' and 'low' columns")
36: 
37:     # Groepeer per kalenderdag op basis van de index (tz-aware OK).
38:     groups = df.groupby(df.index.normalize())
39:     levels = groups.agg(day_high=("high", "max"), day_low=("low", "min"))
40: 
41:     # Zorg voor stabiele dtypes (handig voor tests/downstream)
42:     return levels.astype({"day_high": "float64", "day_low": "float64"})
43: 
44: __all__ = [
45:     "SymbolSpec",
46:     "DEFAULT_SPECS",
47:     "BreakoutParams",
48:     "daily_levels",
49: ]
````

## File: strategies/breakout_signals.py
````python
 1: # strategies/breakout_signals.py
 2: from __future__ import annotations
 3: import pandas as pd
 4: 
 5: 
 6: def breakout_long(close: pd.Series, high: pd.Series, window: int = 20) -> pd.Series:
 7:     """
 8:     Klassiek bar-based breakout: True als close > rolling max van vorige highs.
 9:     Voor exploratie; kan vaker dan 1x per dag vuren.
10:     """
11:     if not isinstance(close.index, pd.DatetimeIndex):
12:         raise TypeError("Index must be DatetimeIndex")
13:     if not close.index.equals(high.index):
14:         raise ValueError("close/high index mismatch")
15: 
16:     prev_high = high.shift(1)
17:     lvl = prev_high.rolling(window, min_periods=window).max()
18:     sig = (close > lvl) & lvl.notna()
19:     return sig.astype(bool).rename("breakout_long")
20: 
21: 
22: def _prev_day_high_series(high: pd.Series) -> pd.Series:
23:     """
24:     Voor elke rij: de high van de VORIGE kalenderdag.
25:     Eenvoudig, vectorized en tz-safe (werkt op de index zelf).
26:     """
27:     if not isinstance(high.index, pd.DatetimeIndex):
28:         raise TypeError("Index must be DatetimeIndex")
29: 
30:     day = high.index.normalize()              # één label per dag
31:     daily_highs = high.groupby(day).max()     # index: unieke dagen
32:     prev_daily_highs = daily_highs.shift(1)   # vorige dag
33: 
34:     # Map de vorige-dag-high terug naar elke rij:
35:     # .loc met 'day' (duplicate indexlabels) dupliceert de juiste waarde per rij.
36:     prev_per_row = pd.Series(prev_daily_highs.loc[day].values, index=high.index, name="prev_day_high")
37:     return prev_per_row
38: 
39: 
40: def opening_breakout_long(
41:     close: pd.Series,
42:     high: pd.Series,
43:     open_window_bars: int = 4,
44:     confirm: str = "close",  # "close" of "wick"
45: ) -> pd.Series:
46:     """
47:     Maximaal 1 entry per dag in het openingsvenster (eerste N bars).
48:     Breakout-niveau = previous-day high.
49: 
50:     - confirm="close":  close > prev_day_high  (strict)
51:     - confirm="wick" :  high  > prev_day_high  (strict)
52: 
53:     Retourneert een boolean Series met ten hoogste 1 True per dag.
54:     """
55:     if not isinstance(close.index, pd.DatetimeIndex):
56:         raise TypeError("Index must be DatetimeIndex")
57:     if not close.index.equals(high.index):
58:         raise ValueError("close/high index mismatch")
59: 
60:     day = close.index.normalize()
61:     bar_idx = day.groupby(day).cumcount()   # 0,1,2,... per dag
62:     in_open = bar_idx < int(open_window_bars)
63: 
64:     prev_day_high = _prev_day_high_series(high)
65: 
66:     if confirm == "wick":
67:         raw = (high > prev_day_high)
68:     else:
69:         raw = (close > prev_day_high)
70: 
71:     raw = raw & prev_day_high.notna() & in_open
72: 
73:     # Alleen de EERSTE True per dag behouden
74:     first_of_day = raw & ~raw.groupby(day).cummax().shift(fill_value=False)
75:     return first_of_day.astype(bool).rename("opening_breakout_long")
````

## File: strategies/order_block.py
````python
 1: import pandas as pd
 2: 
 3: 
 4: def order_block_signals(df: pd.DataFrame, swing_w: int = 3) -> pd.DataFrame:
 5:     """
 6:     Minimale placeholder:
 7:       LONG = close breekt boven vorige swing-high (BOS).
 8:     Geen echte OB-detectie, maar genoeg om de pipeline te testen.
 9:     """
10:     hi = df["high"].rolling(swing_w, center=True).max()
11:     swing_high = (df["high"] == hi) & df["high"].notna()
12:     prev_swing_high = df["high"].where(swing_high).ffill().shift(1)
13:     long = df["close"] > prev_swing_high
14:     return pd.DataFrame({"long": long.fillna(False)}, index=df.index)
````

## File: strategies/premarket_orb.py
````python
 1: # strategies/premarket_orb.py
 2: from __future__ import annotations
 3: from dataclasses import dataclass
 4: from typing import Tuple, Optional
 5: import pandas as pd
 6: from zoneinfo import ZoneInfo
 7: 
 8: @dataclass(frozen=True)
 9: class ORBParams:
10:     # Lokale cash‑open (HH:MM) en bijbehorende tijdzone
11:     session_open_local: str = "09:00"       # DAX: "09:00"; Dow: "09:30"
12:     session_tz: str = "Europe/Berlin"       # DAX; voor Dow "America/New_York"
13:     premarket_minutes: int = 60             # Trader Tom range = 60 minuten
14:     confirm: str = "close"                  # "close" of "wick"
15:     one_trade_per_day: bool = True          # max 1 trade per dag
16:     allow_both_sides: bool = True           # zowel long als short
17: 
18: def _first_idx(s: pd.Series) -> Optional[pd.Timestamp]:
19:     """Returneer de eerste True‑index (of None)."""
20:     if not s.any():
21:         return None
22:     return s[s].index[0]
23: 
24: def premarket_orb_entries(
25:     df: pd.DataFrame,
26:     p: ORBParams,
27: ) -> Tuple[pd.Series, pd.Series]:
28:     """
29:     Bepaal pre‑market high/low en geef twee boolean Series terug:
30:     entries_long en entries_short.
31:     """
32:     req = {"open", "high", "low", "close"}
33:     if not isinstance(df.index, pd.DatetimeIndex):
34:         raise TypeError("Index must be DatetimeIndex")
35:     if not req.issubset(df.columns):
36:         miss = req.difference(df.columns)
37:         raise KeyError(f"df mist kolommen: {sorted(miss)}")
38: 
39:     df = df.sort_index()
40:     loc_idx = df.index.tz_convert(ZoneInfo(p.session_tz))
41:     loc_day = loc_idx.normalize()
42: 
43:     el = pd.Series(False, index=df.index, name="entries_long")
44:     es = pd.Series(False, index=df.index, name="entries_short")
45: 
46:     hh, mm = map(int, p.session_open_local.split(":"))
47:     for d in pd.unique(loc_day):
48:         open_loc = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=ZoneInfo(p.session_tz))
49:         pre_start = open_loc - pd.Timedelta(minutes=p.premarket_minutes)
50: 
51:         pre_mask = (loc_idx >= pre_start) & (loc_idx < open_loc)
52:         ses_mask = (loc_idx >= open_loc) & (loc_day == d)
53: 
54:         pre = df.loc[pre_mask]
55:         ses = df.loc[ses_mask]
56:         if pre.empty or ses.empty:
57:             continue
58: 
59:         hi = float(pre["high"].max())
60:         lo = float(pre["low"].min())
61: 
62:         if p.confirm == "wick":
63:             cond_long = ses["high"] > hi
64:             cond_short = ses["low"] < lo
65:         else:
66:             cond_long = ses["close"] > hi
67:             cond_short = ses["close"] < lo
68: 
69:         t_long = _first_idx(cond_long)
70:         t_short = _first_idx(cond_short)
71: 
72:         if p.one_trade_per_day:
73:             if t_long is not None and (t_short is None or t_long <= t_short):
74:                 el.loc[t_long] = True
75:             elif t_short is not None:
76:                 es.loc[t_short] = True
77:         else:
78:             if t_long is not None:
79:                 el.loc[t_long] = True
80:             if t_short is not None:
81:                 es.loc[t_short] = True
82: 
83:     return el.astype(bool), es.astype(bool)
````

## File: test/test_metrics.py
````python
 1: # test/test_metrics.py
 2: import pandas as pd
 3: from utils.metrics import summarize_equity_metrics
 4: 
 5: def test_metrics_basic():
 6:     idx = pd.date_range("2024-01-01", periods=10, freq="h")
 7:     df_eq = pd.DataFrame(index=idx, data={"equity": [10000 + i*10 for i in range(10)]})
 8:     trades = pd.DataFrame({"pnl": [1, -2, 3]})
 9:     m = summarize_equity_metrics(df_eq, trades)
10:     # verwachtte kern-keys moeten aanwezig zijn (er mogen extra keys zijn):
11:     expected = {"sharpe", "max_dd", "dd_duration", "total_return", "n_trades"}
12:     assert expected <= set(m.keys())
````

## File: utils/data_health.py
````python
 1: # utils/data_health.py
 2: from __future__ import annotations
 3: import pandas as pd
 4: 
 5: def _normalize_freq(freq: str) -> str:
 6:     """
 7:     Pandas deprecates 'T' for minutes. Map common shorthands to new forms.
 8:     Examples: '15T' -> '15min', '5T' -> '5min'.
 9:     """
10:     if isinstance(freq, str) and freq.endswith("T"):
11:         return freq[:-1] + "min"
12:     return freq
13: 
14: def health_stats(df: pd.DataFrame, expected_freq: str = "15min") -> dict:
15:     """
16:     Kale feiten over index-regulariteit (geen thresholds/policy):
17:       - gaps: aantal index-deltas ≠ expected
18:       - top: top-3 afwijkende deltas met counts
19:       - dups: aantal duplicate timestamps
20:     """
21:     if not isinstance(df.index, pd.DatetimeIndex):
22:         return {"error": "index!=DatetimeIndex"}
23: 
24:     exp = pd.to_timedelta(_normalize_freq(expected_freq))
25:     diffs = df.index.to_series().diff().dropna()
26:     off = diffs[diffs != exp]
27:     top = off.value_counts().head(3)
28:     return {
29:         "gaps": int(len(off)),
30:         "dups": int(df.index.duplicated().sum()),
31:         "top": {str(delta): int(cnt) for delta, cnt in top.items()},
32:     }
33: 
34: def health_line(df: pd.DataFrame, expected_freq: str = "15min") -> str:
35:     """Compacte 1-regel logstring op basis van health_stats."""
36:     st = health_stats(df, expected_freq=expected_freq)
37:     if "error" in st:
38:         return f"DATA {{error:{st['error']}}}"
39:     top = " ".join([f"{k}x{v}" for k, v in st["top"].items()])
40:     return f"DATA {{gaps:{st['gaps']} dups:{st['dups']} top:[{top}]}}"
````

## File: utils/days.py
````python
 1: # utils/days.py
 2: from __future__ import annotations
 3: import pandas as pd
 4: from typing import Tuple
 5: 
 6: def day_counts(df: pd.DataFrame) -> pd.Series:
 7:     """Aantal bars per kalenderdag die in de data voorkomen."""
 8:     if not isinstance(df.index, pd.DatetimeIndex):
 9:         raise TypeError("Index must be DatetimeIndex")
10:     return df.index.normalize().value_counts().sort_index()
11: 
12: def trading_days(df: pd.DataFrame) -> pd.Index:
13:     """Unieke dagen die daadwerkelijk bars hebben (geen lege/weekenddagen)."""
14:     return df.index.normalize().unique()
15: 
16: def summarize_day_health(df: pd.DataFrame) -> Tuple[int, int, float]:
17:     """
18:     Geeft (days, min_bars, mean_bars) terug.
19:     Alleen feiten loggen; geen auto-skip hier.
20:     """
21:     cnts = day_counts(df)
22:     if cnts.empty:
23:         return (0, 0, 0.0)
24:     return (int(len(cnts)), int(cnts.min()), float(cnts.mean()))
````

## File: utils/metrics.py
````python
 1: from __future__ import annotations
 2: import numpy as np
 3: import pandas as pd
 4: 
 5: TRADING_DAYS = 252
 6: 
 7: def _equity_series(df_eq: pd.DataFrame) -> pd.Series:
 8:     if "equity" not in df_eq.columns:
 9:         raise KeyError("Column 'equity' ontbreekt in equity DataFrame")
10:     s = pd.Series(df_eq["equity"], dtype="float64").dropna()
11:     if s.empty:
12:         raise ValueError("Lege equity-reeks ontvangen")
13:     return s
14: 
15: def equity_sharpe(
16:     df_eq: pd.DataFrame,
17:     risk_free: float = 0.0,
18:     periods_per_year: int = TRADING_DAYS,
19: ) -> float:
20:     s = _equity_series(df_eq)
21:     rets = s.pct_change().dropna()
22:     if rets.size < 2:
23:         return 0.0
24:     ex = rets - (risk_free / periods_per_year)
25:     std = ex.std(ddof=1)
26:     if std <= 1e-12:
27:         return float("inf") if ex.mean() > 0 else 0.0
28:     return float(ex.mean() / std * np.sqrt(periods_per_year))
29: 
30: def equity_max_dd_and_duration(df_eq: pd.DataFrame) -> tuple[float, int]:
31:     s = _equity_series(df_eq)
32:     v = s.values.astype("float64")
33:     run_max = np.maximum.accumulate(v)
34:     dd = v / run_max - 1.0
35:     max_dd_signed = float(dd.min())
36: 
37:     duration = 0
38:     max_duration = 0
39:     peak = v[0]
40:     for x in v:
41:         if x >= peak:
42:             peak = x
43:             duration = 0
44:         else:
45:             duration += 1
46:             max_duration = max(max_duration, duration)
47: 
48:     return abs(max_dd_signed), int(max_duration)
49: 
50: def summarize_equity_metrics(df_eq: pd.DataFrame, trades: pd.DataFrame | None = None) -> dict:
51:     s = _equity_series(df_eq)
52:     sharpe = equity_sharpe(df_eq)
53:     max_dd, dd_dur = equity_max_dd_and_duration(df_eq)
54:     total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) >= 2 else 0.0
55:     n_trades = int(len(trades)) if trades is not None else 0
56:     winrate = float((trades["pnl_cash"] > 0).mean() * 100.0) if trades is not None and "pnl_cash" in trades.columns and not trades.empty else 0.0
57:     avg_rr = float((trades["tp_pts"] / trades["sl_pts"]).mean()) if trades is not None and "tp_pts" in trades.columns and not trades.empty else 0.0
58:     final_equity = float(s.iloc[-1]) if not s.empty else 0.0
59:     return_total_pct = total_ret * 100.0
60:     max_drawdown_pct = max_dd * 100.0
61:     return {
62:         "sharpe": sharpe,
63:         "max_dd": max_dd,
64:         "dd_duration": dd_dur,
65:         "total_return": total_ret,
66:         "n_trades": n_trades,
67:         "winrate_pct": winrate,
68:         "avg_rr": avg_rr,
69:         "final_equity": final_equity,
70:         "return_total_pct": return_total_pct,
71:         "max_drawdown_pct": max_drawdown_pct,
72:     }
````

## File: utils/plot.py
````python
 1: # utils/plot.py
 2: from __future__ import annotations
 3: from pathlib import Path
 4: import pandas as pd
 5: import matplotlib.pyplot as plt
 6: 
 7: def save_equity_and_dd(equity: pd.Series, outdir: Path) -> None:
 8:     outdir.mkdir(parents=True, exist_ok=True)
 9:     # Equity
10:     fig = plt.figure(figsize=(10, 4))
11:     equity.rename("Equity").plot()
12:     plt.title("Equity")
13:     plt.tight_layout()
14:     fig.savefig(outdir / "equity.png", dpi=120)
15:     plt.close(fig)
16:     # Drawdown
17:     dd = (equity / equity.cummax() - 1.0).rename("Drawdown")
18:     fig = plt.figure(figsize=(10, 3))
19:     dd.plot()
20:     plt.title("Drawdown")
21:     plt.tight_layout()
22:     fig.savefig(outdir / "drawdown.png", dpi=120)
23:     plt.close(fig)
````

## File: utils/position.py
````python
 1: """
 2: Position sizing and trade utilities for Sophy4Lite.
 3: """
 4: from __future__ import annotations
 5: from enum import Enum
 6: from typing import NamedTuple
 7: 
 8: 
 9: class Side(Enum):
10:     """Trade side enumeration."""
11:     BUY = 1
12:     SELL = -1
13: 
14: 
15: def side_factor(side: Side) -> float:
16:     """Return 1.0 for BUY, -1.0 for SELL."""
17:     return float(side.value)
18: 
19: 
20: def side_name(side: Side) -> str:
21:     """Return 'BUY' or 'SELL' string."""
22:     return side.name
23: 
24: 
25: class Position(NamedTuple):
26:     """Position information."""
27:     symbol: str
28:     side: Side
29:     size: float
30:     entry_price: float
31:     sl_price: float
32:     tp_price: float
33: 
34:     @property
35:     def risk_points(self) -> float:
36:         """Distance to stop loss in points."""
37:         return abs(self.entry_price - self.sl_price)
38: 
39:     @property
40:     def reward_points(self) -> float:
41:         """Distance to take profit in points."""
42:         return abs(self.tp_price - self.entry_price)
43: 
44:     @property
45:     def risk_reward_ratio(self) -> float:
46:         """R:R ratio of the position."""
47:         if self.risk_points == 0:
48:             return 0.0
49:         return self.reward_points / self.risk_points
50: 
51: 
52: def _size(equity: float, entry: float, stop: float, vpp: float,
53:           risk_frac: float) -> float:
54:     """
55:     Calculate position size based on fixed percentage risk.
56: 
57:     Args:
58:         equity: Current account equity
59:         entry: Entry price
60:         stop: Stop loss price
61:         vpp: Value per point
62:         risk_frac: Risk fraction (e.g., 0.01 for 1%)
63: 
64:     Returns:
65:         Position size in lots/contracts
66:     """
67:     risk_cash = equity * risk_frac
68:     pts = abs(entry - stop)
69:     if pts <= 0 or vpp <= 0:
70:         return 0.0
71:     return float(risk_cash / (pts * vpp))
````

## File: .gitignore
````
 1: ---
 2: 
 3: ## ✅ `.gitignore`
 4: ```gitignore
 5: # -------------------------
 6: # Python build artefacts
 7: # -------------------------
 8: __pycache__/
 9: *.py[cod]
10: *.pyo
11: *.pyd
12: *.so
13: *.egg-info/
14: *.egg
15: .eggs/
16: build/
17: dist/
18: 
19: # -------------------------
20: # Virtual environments
21: # -------------------------
22: .venv/
23: venv/
24: .env
25: .envrc
26: 
27: # -------------------------
28: # Logs & debug
29: # -------------------------
30: *.log
31: 
32: # -------------------------
33: # IDE / Editor
34: # -------------------------
35: .idea/
36: .vscode/
37: *.swp
38: *.swo
39: 
40: # -------------------------
41: # Jupyter notebooks
42: # -------------------------
43: .ipynb_checkpoints/
44: *.ipynb
45: 
46: # -------------------------
47: # Data / outputs
48: # -------------------------
49: # Local run outputs (ignore generated reports & artifacts)
50: output/
51: reports/
52: 
53: # Ignore bulk data files
54: *.csv
55: *.png
56: 
57: # If you want to keep example plots/CSVs, place them in docs/ or examples/
58: #!docs/*.png
59: #!examples/*.csv
60: 
61: # -------------------------
62: # Config / profiles
63: # -------------------------
64: # Keep live.yaml under version control, ignore backups or generated variants
65: profiles/*.bak
66: profiles/*.old
67: profiles/*.tmp
68: # Important: keep this file tracked
69: !profiles/live.yaml
70: 
71: # -------------------------
72: # Type checkers / caches
73: # -------------------------
74: .mypy_cache/
75: 
76: # -------------------------
77: # OS junk
78: # -------------------------
79: .DS_Store
80: Thumbs.db
````

## File: config.py
````python
 1: from dataclasses import dataclass
 2: import logging
 3: import os
 4: 
 5: # --- Core settings ---
 6: INITIAL_CAPITAL: float = 20000.0
 7: FEES: float = 0.0002  # 2 bps
 8: DEFAULT_TIMEFRAME: str = "H1"
 9: OUTPUT_DIR: str = "output"
10: 
11: # --- FTMO guardrails (sync met FtmoRules) ---
12: MAX_DAILY_LOSS: float = 0.05   # 5%
13: MAX_TOTAL_LOSS: float = 0.10   # 10%
14: RISK_PER_TRADE: float = 0.01   # 1%
15: STOP_AFTER_LOSSES: int = 2     # consecutive losing trades per day
16: 
17: # --- Logging ---
18: LOG_LEVEL: str = "INFO"
19: 
20: # Maak logger en configureer handlers
21: logger = logging.getLogger(__name__)
22: logger.setLevel(LOG_LEVEL)
23: 
24: # File handler voor logbestand
25: file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "sophy4lite.log"))
26: file_handler.setLevel(LOG_LEVEL)
27: file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
28: file_handler.setFormatter(file_formatter)
29: 
30: # Stream handler voor console
31: stream_handler = logging.StreamHandler()
32: stream_handler.setLevel(LOG_LEVEL)
33: stream_handler.setFormatter(file_formatter)
34: 
35: # Voeg handlers toe aan logger
36: logger.addHandler(file_handler)
37: logger.addHandler(stream_handler)
````

## File: dax_orb_tt.py
````python
  1: #!/usr/bin/env python3
  2: # -*- coding: utf-8 -*-
  3: """
  4: DAX ORB Realistic - MT5 data fetch + realistic spread/slippage simulation
  5: CRITICAL: This version includes transaction costs for realistic results
  6: """
  7: from __future__ import annotations
  8: import argparse, glob, math, time, multiprocessing as mp
  9: from pathlib import Path
 10: from datetime import datetime, timedelta
 11: import pandas as pd
 12: import numpy as np
 13: from zoneinfo import ZoneInfo
 14: import matplotlib.pyplot as plt
 15: import matplotlib.dates as mdates
 16: import warnings
 17: 
 18: warnings.filterwarnings('ignore')
 19: 
 20: # Optional MT5 integration
 21: try:
 22:     import MetaTrader5 as mt5
 23: 
 24:     MT5_AVAILABLE = True
 25: except ImportError:
 26:     MT5_AVAILABLE = False
 27:     print("⚠️ MetaTrader5 package not installed. Install with: pip install MetaTrader5")
 28:     print("   Continuing with CSV file support only.\n")
 29: 
 30: 
 31: # ============================= REALISTIC COSTS =====================================
 32: class TradingCosts:
 33:     """
 34:     Realistic trading costs for DAX CFD/Futures
 35:     Based on typical retail broker conditions
 36:     """
 37:     # Spread (bid-ask) - varies by time of day
 38:     SPREAD_OPENING = 2.0  # Points during first 30 min (high volatility)
 39:     SPREAD_NORMAL = 1.0  # Points during normal hours
 40:     SPREAD_CLOSING = 1.5  # Points during last hour
 41: 
 42:     # Slippage on stop orders (worse during volatility)
 43:     SLIPPAGE_STOP_OPENING = 1.5  # Extra points on stops during opening
 44:     SLIPPAGE_STOP_NORMAL = 0.5  # Normal slippage on stops
 45: 
 46:     # Commission per side (points equivalent)
 47:     COMMISSION = 0.5  # Per trade side (entry + exit = 1.0 total)
 48: 
 49:     @staticmethod
 50:     def get_entry_cost(hour: int, minute: int, is_opening_window: bool) -> float:
 51:         """Get realistic entry cost based on time of day"""
 52:         total_minutes = hour * 60 + minute
 53: 
 54:         # Opening window (09:00-09:30) - highest costs
 55:         if is_opening_window and 540 <= total_minutes <= 570:
 56:             return TradingCosts.SPREAD_OPENING + TradingCosts.COMMISSION
 57:         # Closing hour (17:30-18:30) - elevated costs
 58:         elif 1050 <= total_minutes <= 1110:
 59:             return TradingCosts.SPREAD_CLOSING + TradingCosts.COMMISSION
 60:         # Normal hours
 61:         else:
 62:             return TradingCosts.SPREAD_NORMAL + TradingCosts.COMMISSION
 63: 
 64:     @staticmethod
 65:     def get_stop_slippage(hour: int, minute: int) -> float:
 66:         """Get realistic slippage for stop orders"""
 67:         total_minutes = hour * 60 + minute
 68: 
 69:         # Opening 30 minutes - high slippage
 70:         if 540 <= total_minutes <= 570:
 71:             return TradingCosts.SLIPPAGE_STOP_OPENING
 72:         else:
 73:             return TradingCosts.SLIPPAGE_STOP_NORMAL
 74: 
 75: 
 76: # ============================= MT5 Integration ======================================
 77: def fetch_mt5_data(symbol: str = "DE40", days_back: int = 60, server: str = None,
 78:                    login: int = None, password: str = None) -> pd.DataFrame:
 79:     """
 80:     Fetch M1 data directly from MT5
 81: 
 82:     Args:
 83:         symbol: Symbol to fetch (DE40, GER40, etc.)
 84:         days_back: Number of days of history
 85:         server: MT5 server (optional, uses current if None)
 86:         login: MT5 account (optional)
 87:         password: MT5 password (optional)
 88:     """
 89:     if not MT5_AVAILABLE:
 90:         raise ImportError("MetaTrader5 package not installed")
 91: 
 92:     # Initialize MT5
 93:     if login and password and server:
 94:         if not mt5.initialize(login=login, password=password, server=server):
 95:             raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
 96:     else:
 97:         if not mt5.initialize():
 98:             raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
 99: 
100:     # Check symbol exists
101:     symbol_info = mt5.symbol_info(symbol)
102:     if symbol_info is None:
103:         mt5.shutdown()
104:         raise ValueError(
105:             f"Symbol {symbol} not found. Available symbols: {[s.name for s in mt5.symbols_get()][:10]}...")
106: 
107:     # Calculate date range
108:     utc_to = datetime.now()
109:     utc_from = utc_to - timedelta(days=days_back)
110: 
111:     # Fetch M1 bars
112:     rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)
113: 
114:     if rates is None or len(rates) == 0:
115:         mt5.shutdown()
116:         raise ValueError(f"No data received for {symbol}")
117: 
118:     # Convert to DataFrame
119:     df = pd.DataFrame(rates)
120:     df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
121:     df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].rename(
122:         columns={'tick_volume': 'volume'})
123: 
124:     mt5.shutdown()
125: 
126:     print(f"✓ Fetched {len(df):,} M1 bars from MT5 for {symbol}")
127:     print(f"  Period: {df['time'].min()} to {df['time'].max()}")
128: 
129:     return df
130: 
131: 
132: # ============================= CLI ==================================================
133: def parse_args():
134:     p = argparse.ArgumentParser(description="DAX ORB Realistic - MT5 + True Costs")
135: 
136:     # Data source
137:     p.add_argument("paths", nargs="*", default=None,
138:                    help="CSV files (if none, tries MT5 or searches for CSVs)")
139:     p.add_argument("--mt5", action="store_true", help="Fetch data from MT5")
140:     p.add_argument("--symbol", default="DE40", help="MT5 symbol (DE40, GER40, etc.)")
141:     p.add_argument("--days", type=int, default=60,
142:                    help="Days of history to fetch from MT5")
143: 
144:     # Trading parameters
145:     p.add_argument("--tz", default="Europe/Athens", help="Server timezone")
146:     p.add_argument("--pre", default="08:00-09:00", help="Pre-market window")
147:     p.add_argument("--sess", default="09:00-18:30", help="Session window")
148: 
149:     # Strategy modes
150:     p.add_argument("--tradertom", action="store_true",
151:                    help="TraderTom fixed params (SL=9, TP=6)")
152:     p.add_argument("--realistic", action="store_true", default=True,
153:                    help="Apply realistic costs (default: ON)")
154:     p.add_argument("--no-costs", dest="realistic", action="store_false",
155:                    help="Disable transaction costs")
156:     p.add_argument("--opening-window", type=int, default=30,
157:                    help="Minutes after open for entry")
158: 
159:     # Parameter sweep (if not --tradertom)
160:     p.add_argument("--sl-min", type=float, default=5.0)
161:     p.add_argument("--sl-max", type=float, default=15.0)
162:     p.add_argument("--sl-step", type=float, default=1.0)
163:     p.add_argument("--tp-min", type=float, default=3.0)
164:     p.add_argument("--tp-max", type=float, default=12.0)
165:     p.add_argument("--tp-step", type=float, default=1.0)
166: 
167:     # Analysis
168:     p.add_argument("--min-trades", type=int, default=30, help="Min trades for validity")
169:     p.add_argument("--train-pct", type=float, default=0.7,
170:                    help="Training set percentage (0.7 = 70%)")
171: 
172:     # Output
173:     p.add_argument("--plot", action="store_true", help="Generate plots")
174:     p.add_argument("--save-trades", action="store_true", help="Save trade list")
175:     p.add_argument("--verbose", action="store_true", help="Detailed output")
176: 
177:     return p.parse_args()
178: 
179: 
180: # ============================= Data Loading =========================================
181: def load_data(args) -> pd.DataFrame:
182:     """Load data from MT5 or CSV files"""
183: 
184:     # Try MT5 first if requested
185:     if args.mt5:
186:         if not MT5_AVAILABLE:
187:             print("❌ MT5 requested but MetaTrader5 package not installed")
188:             print("   Install with: pip install MetaTrader5")
189:             print("   Falling back to CSV files...")
190:         else:
191:             try:
192:                 return fetch_mt5_data(args.symbol, args.days)
193:             except Exception as e:
194:                 print(f"❌ MT5 fetch failed: {e}")
195:                 print("   Falling back to CSV files...")
196: 
197:     # Load from CSV files
198:     if not args.paths:
199:         # Search for CSV files
200:         current_dir = Path.cwd()
201:         data_dir = current_dir / "data"
202:         patterns = ["*.csv", "*_M1.csv", "GER*.csv", "DAX*.csv", "DE40*.csv",
203:             "data/*.csv", "data/*_M1.csv"]
204:         files = []
205:         for pattern in patterns:
206:             files.extend(glob.glob(str(current_dir / pattern)))
207:             if data_dir.exists():
208:                 files.extend(glob.glob(str(data_dir / pattern)))
209:         files = list(set(files))  # Remove duplicates
210:     else:
211:         files = [f for pat in args.paths for f in glob.glob(pat)]
212: 
213:     if not files:
214:         raise FileNotFoundError("No data source available!\n"
215:                                 "Options:\n"
216:                                 "1. Use --mt5 to fetch from MetaTrader5\n"
217:                                 "2. Place CSV files in current directory\n"
218:                                 "3. Specify CSV path as argument")
219: 
220:     # Read and merge CSV files
221:     dfs = []
222:     for fp in files:
223:         df = pd.read_csv(fp)
224:         # Standardize column names
225:         df.columns = df.columns.str.lower()
226:         if 'time' not in df.columns and 'date' in df.columns:
227:             df['time'] = df['date']
228:         required = ['time', 'open', 'high', 'low', 'close']
229:         if not all(c in df.columns for c in required):
230:             print(f"⚠️ Skipping {fp}: missing required columns")
231:             continue
232:         dfs.append(df[required + (['volume'] if 'volume' in df.columns else [])])
233: 
234:     if not dfs:
235:         raise ValueError("No valid CSV files found")
236: 
237:     # Combine all data
238:     df = pd.concat(dfs, ignore_index=True)
239:     df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
240:     df = df.dropna(subset=['time']).drop_duplicates(subset=['time']).sort_values('time')
241: 
242:     print(f"✓ Loaded {len(df):,} M1 bars from {len(files)} file(s)")
243:     print(f"  Period: {df['time'].min()} to {df['time'].max()}")
244: 
245:     return df
246: 
247: 
248: # ============================= Realistic Backtest ===================================
249: def run_realistic_backtest(df: pd.DataFrame, tz: str, pre_s: str, sess_s: str,
250:                            sl: float, tp: float, use_costs: bool = True,
251:                            tradertom_mode: bool = False, opening_window: int = 30,
252:                            verbose: bool = False) -> tuple[pd.DataFrame, dict]:
253:     """
254:     Realistic backtest with proper costs and execution simulation
255: 
256:     CRITICAL IMPROVEMENTS:
257:     1. Spread costs on entry
258:     2. Slippage on stop losses
259:     3. Commission on both sides
260:     4. No lookahead bias on pre-market levels
261:     5. Proper position reset between tests
262:     """
263: 
264:     # Convert to local timezone
265:     df = df.copy()
266:     df['local'] = df['time'].dt.tz_convert(ZoneInfo(tz))
267:     df = df.set_index('local').sort_index()
268: 
269:     # Add time components for cost calculation
270:     df['hour'] = df.index.hour
271:     df['minute_of_day'] = df.index.hour * 60 + df.index.minute
272: 
273:     # Parse time windows
274:     pre_start, pre_end = [int(h) * 60 + int(m) for h, m in
275:                           [t.split(':') for t in pre_s.split('-')]]
276:     sess_start, sess_end = [int(h) * 60 + int(m) for h, m in
277:                             [t.split(':') for t in sess_s.split('-')]]
278: 
279:     # Build pre-market levels (with proper timing)
280:     orb_levels = {}
281:     for day, day_data in df.groupby(df.index.date):
282:         # Only use data up to pre-market end (no lookahead)
283:         pre_data = day_data[(day_data['minute_of_day'] >= pre_start) & (
284:                     day_data['minute_of_day'] < pre_end)]
285:         if not pre_data.empty:
286:             orb_levels[day] = {'low': float(pre_data['low'].min()),
287:                 'high': float(pre_data['high'].max()),
288:                 'pre_close': float(pre_data['close'].iloc[-1])  # Where pre-market ended
289:             }
290: 
291:     trades = []
292:     position = None  # Proper position tracking
293: 
294:     for day, day_data in df.groupby(df.index.date):
295:         # Skip if no pre-market data or already in position from previous day
296:         if day not in orb_levels or position is not None:
297:             continue
298: 
299:         orb = orb_levels[day]
300:         if not (math.isfinite(orb['high']) and math.isfinite(orb['low']) and orb[
301:             'high'] > orb['low']):
302:             continue
303: 
304:         # Determine entry window
305:         if tradertom_mode:
306:             entry_end = sess_start + opening_window
307:         else:
308:             entry_end = sess_end
309: 
310:         entry_window = day_data[(day_data['minute_of_day'] >= sess_start) & (
311:                     day_data['minute_of_day'] <= entry_end)]
312: 
313:         if entry_window.empty:
314:             continue
315: 
316:         # Check for breakout (pending order simulation)
317:         for idx, bar in entry_window.iterrows():
318:             if position is not None:  # Already entered
319:                 break
320: 
321:             # Calculate costs for this specific time
322:             if use_costs:
323:                 entry_cost = TradingCosts.get_entry_cost(bar['hour'],
324:                     bar['minute_of_day'] % 60, tradertom_mode)
325:             else:
326:                 entry_cost = 0
327: 
328:             # Buy stop order simulation (triggers if high touches level)
329:             if bar['high'] >= orb['high']:
330:                 position = {'side': 'long', 'entry_time': idx, 'entry_raw': orb['high'],
331:                     'entry': orb['high'] + entry_cost,  # Pay spread + commission
332:                     'stop': orb['high'] - sl, 'target': orb['high'] + tp, 'day': day}
333:                 if verbose:
334:                     print(
335:                         f"LONG entry: {idx} @ {position['entry']:.1f} (raw: {position['entry_raw']:.1f}, cost: {entry_cost:.1f})")
336:                 break
337: 
338:             # Sell stop order simulation
339:             elif bar['low'] <= orb['low']:
340:                 position = {'side': 'short', 'entry_time': idx, 'entry_raw': orb['low'],
341:                     'entry': orb['low'] - entry_cost,  # Pay spread + commission
342:                     'stop': orb['low'] + sl, 'target': orb['low'] - tp, 'day': day}
343:                 if verbose:
344:                     print(
345:                         f"SHORT entry: {idx} @ {position['entry']:.1f} (raw: {position['entry_raw']:.1f}, cost: {entry_cost:.1f})")
346:                 break
347: 
348:         # Check for exit if position was opened
349:         if position is not None:
350:             # Use all remaining data for exit (not limited to entry window)
351:             exit_data = day_data.loc[position['entry_time']:]
352: 
353:             for idx, bar in exit_data.iterrows():
354:                 exit_triggered = False
355: 
356:                 if position['side'] == 'long':
357:                     # Check stop loss
358:                     if bar['low'] <= position['stop']:
359:                         if use_costs:
360:                             slippage = TradingCosts.get_stop_slippage(bar['hour'], bar[
361:                                 'minute_of_day'] % 60)
362:                             exit_price = position['stop'] - slippage
363:                         else:
364:                             exit_price = position['stop']
365: 
366:                         pnl = exit_price - position['entry']
367:                         trades.append({'day': str(position['day']),
368:                             'entry_time': position['entry_time'], 'exit_time': idx,
369:                             'side': position['side'], 'entry': position['entry_raw'],
370:                             # Show raw entry in results
371:                             'exit': exit_price, 'reason': 'SL', 'pnl': pnl,
372:                             'costs': position['entry'] - position[
373:                                 'entry_raw'] + slippage if use_costs else 0})
374:                         exit_triggered = True
375: 
376:                     # Check take profit
377:                     elif bar['high'] >= position['target']:
378:                         if use_costs:
379:                             exit_price = position['target'] - TradingCosts.COMMISSION
380:                         else:
381:                             exit_price = position['target']
382: 
383:                         pnl = exit_price - position['entry']
384:                         trades.append({'day': str(position['day']),
385:                             'entry_time': position['entry_time'], 'exit_time': idx,
386:                             'side': position['side'], 'entry': position['entry_raw'],
387:                             'exit': exit_price, 'reason': 'TP', 'pnl': pnl,
388:                             'costs': position['entry'] - position[
389:                                 'entry_raw'] + TradingCosts.COMMISSION if use_costs else 0})
390:                         exit_triggered = True
391: 
392:                 else:  # Short position
393:                     # Check stop loss
394:                     if bar['high'] >= position['stop']:
395:                         if use_costs:
396:                             slippage = TradingCosts.get_stop_slippage(bar['hour'], bar[
397:                                 'minute_of_day'] % 60)
398:                             exit_price = position['stop'] + slippage
399:                         else:
400:                             exit_price = position['stop']
401: 
402:                         pnl = position['entry'] - exit_price
403:                         trades.append({'day': str(position['day']),
404:                             'entry_time': position['entry_time'], 'exit_time': idx,
405:                             'side': position['side'], 'entry': position['entry_raw'],
406:                             'exit': exit_price, 'reason': 'SL', 'pnl': pnl,
407:                             'costs': position['entry_raw'] - position[
408:                                 'entry'] + slippage if use_costs else 0})
409:                         exit_triggered = True
410: 
411:                     # Check take profit
412:                     elif bar['low'] <= position['target']:
413:                         if use_costs:
414:                             exit_price = position['target'] + TradingCosts.COMMISSION
415:                         else:
416:                             exit_price = position['target']
417: 
418:                         pnl = position['entry'] - exit_price
419:                         trades.append({'day': str(position['day']),
420:                             'entry_time': position['entry_time'], 'exit_time': idx,
421:                             'side': position['side'], 'entry': position['entry_raw'],
422:                             'exit': exit_price, 'reason': 'TP', 'pnl': pnl,
423:                             'costs': position['entry_raw'] - position[
424:                                 'entry'] + TradingCosts.COMMISSION if use_costs else 0})
425:                         exit_triggered = True
426: 
427:                 if exit_triggered:
428:                     position = None
429:                     break
430: 
431:     # Convert to DataFrame
432:     trades_df = pd.DataFrame(trades)
433: 
434:     if trades_df.empty:
435:         return trades_df, {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'winrate': 0,
436:             'pf': 0, 'avg_winner': 0, 'avg_loser': 0, 'total_costs': 0, 'gross_pnl': 0}
437: 
438:     # Calculate statistics
439:     wins = (trades_df['reason'] == 'TP').sum()
440:     losses = (trades_df['reason'] == 'SL').sum()
441:     total_pnl = trades_df['pnl'].sum()
442:     total_costs = trades_df['costs'].sum() if use_costs else 0
443:     gross_pnl = total_pnl + total_costs  # PnL before costs
444: 
445:     winning_trades = trades_df[trades_df['pnl'] > 0]
446:     losing_trades = trades_df[trades_df['pnl'] < 0]
447: 
448:     stats = {'trades': len(trades_df), 'wins': wins, 'losses': losses, 'pnl': total_pnl,
449:         'gross_pnl': gross_pnl, 'total_costs': total_costs,
450:         'winrate': 100 * wins / len(trades_df) if len(trades_df) > 0 else 0,
451:         'pf': winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) if len(
452:             losing_trades) > 0 else float('inf'),
453:         'avg_winner': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
454:         'avg_loser': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
455:         'max_dd': calculate_max_drawdown(trades_df['pnl']),
456:         'sharpe': calculate_sharpe_ratio(trades_df['pnl'])}
457: 
458:     return trades_df, stats
459: 
460: 
461: def calculate_max_drawdown(pnl_series):
462:     """Calculate maximum drawdown"""
463:     cumsum = pnl_series.cumsum()
464:     running_max = cumsum.expanding().max()
465:     drawdown = cumsum - running_max
466:     return float(drawdown.min())
467: 
468: 
469: def calculate_sharpe_ratio(pnl_series, periods_per_year=252 * 390):
470:     """Calculate Sharpe ratio (assuming minute data)"""
471:     if len(pnl_series) < 2:
472:         return 0
473:     returns = pnl_series
474:     if returns.std() == 0:
475:         return 0
476:     return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())
477: 
478: 
479: # ============================= Visualization ========================================
480: def create_comprehensive_plots(df, trades_df, stats, args):
481:     """Create comprehensive analysis plots"""
482:     import matplotlib.pyplot as plt
483:     from matplotlib.gridspec import GridSpec
484: 
485:     fig = plt.figure(figsize=(20, 12))
486:     gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
487: 
488:     # 1. Equity Curve with Drawdown
489:     ax1 = fig.add_subplot(gs[0, :2])
490:     if not trades_df.empty:
491:         trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
492:         trades_df['running_max'] = trades_df['cumulative_pnl'].expanding().max()
493:         trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['running_max']
494: 
495:         ax1.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 'b-',
496:                  label='Equity', linewidth=2)
497:         ax1.fill_between(range(len(trades_df)), trades_df['cumulative_pnl'],
498:                          trades_df['running_max'], alpha=0.3, color='red',
499:                          label='Drawdown')
500:         ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
501:         ax1.set_title(f'Equity Curve ({"With" if args.realistic else "Without"} Costs)')
502:         ax1.set_xlabel('Trade Number')
503:         ax1.set_ylabel('Cumulative PnL (points)')
504:         ax1.legend()
505:         ax1.grid(True, alpha=0.3)
506: 
507:     # 2. Win/Loss Distribution
508:     ax2 = fig.add_subplot(gs[0, 2])
509:     if stats['trades'] > 0:
510:         sizes = [stats['wins'], stats['losses']]
511:         colors = ['green', 'red']
512:         ax2.pie(sizes, labels=['Wins', 'Losses'], colors=colors, autopct='%1.1f%%')
513:         ax2.set_title(f"Win Rate: {stats['winrate']:.1f}%")
514: 
515:     # 3. PnL Distribution
516:     ax3 = fig.add_subplot(gs[1, 0])
517:     if not trades_df.empty:
518:         ax3.hist(trades_df['pnl'], bins=30, alpha=0.7, color='blue', edgecolor='black')
519:         ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
520:         ax3.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', alpha=0.8,
521:                     label=f'Mean: {trades_df["pnl"].mean():.2f}')
522:         ax3.set_title('PnL Distribution')
523:         ax3.set_xlabel('PnL (points)')
524:         ax3.set_ylabel('Frequency')
525:         ax3.legend()
526: 
527:     # 4. Time of Day Analysis
528:     ax4 = fig.add_subplot(gs[1, 1])
529:     if not trades_df.empty:
530:         trades_df['entry_hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
531:         hourly_pnl = trades_df.groupby('entry_hour')['pnl'].sum()
532:         colors = ['green' if x > 0 else 'red' for x in hourly_pnl.values]
533:         ax4.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.7,
534:                 edgecolor='black')
535:         ax4.set_title('PnL by Entry Hour')
536:         ax4.set_xlabel('Hour')
537:         ax4.set_ylabel('Total PnL')
538:         ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
539: 
540:     # 5. Recent Trades Performance
541:     ax5 = fig.add_subplot(gs[1, 2])
542:     if not trades_df.empty:
543:         recent = trades_df.tail(20)
544:         colors = ['green' if x > 0 else 'red' for x in recent['pnl']]
545:         ax5.bar(range(len(recent)), recent['pnl'], color=colors, alpha=0.7,
546:                 edgecolor='black')
547:         ax5.set_title('Last 20 Trades')
548:         ax5.set_xlabel('Trade')
549:         ax5.set_ylabel('PnL')
550:         ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
551: 
552:     # 6. Statistics Table
553:     ax6 = fig.add_subplot(gs[2, :])
554:     ax6.axis('tight')
555:     ax6.axis('off')
556: 
557:     # Create statistics table
558:     table_data = [['Metric', 'Value', 'Metric', 'Value'],
559:         ['Total Trades', f"{stats['trades']}", 'Win Rate', f"{stats['winrate']:.1f}%"],
560:         ['Wins', f"{stats['wins']}", 'Losses', f"{stats['losses']}"],
561:         ['Net PnL', f"{stats['pnl']:.1f}", 'Gross PnL', f"{stats['gross_pnl']:.1f}"],
562:         ['Total Costs', f"{stats['total_costs']:.1f}", 'Profit Factor',
563:          f"{stats['pf']:.2f}"],
564:         ['Avg Winner', f"{stats['avg_winner']:.1f}", 'Avg Loser',
565:          f"{stats['avg_loser']:.1f}"],
566:         ['Max Drawdown', f"{stats['max_dd']:.1f}", 'Sharpe Ratio',
567:          f"{stats['sharpe']:.3f}"], ['Required WR (BE)',
568:                                      f"{100 * args.sl_min / (args.sl_min + args.tp_min):.1f}%" if not args.tradertom else "60.0%",
569:                                      'Actual vs Required',
570:                                      f"{stats['winrate'] - 100 * args.sl_min / (args.sl_min + args.tp_min):.1f}%" if not args.tradertom else f"{stats['winrate'] - 60:.1f}%"]]
571: 
572:     table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
573:                       colWidths=[0.2, 0.2, 0.2, 0.2])
574:     table.auto_set_font_size(False)
575:     table.set_fontsize(10)
576:     table.scale(1, 1.5)
577: 
578:     # Color header
579:     for i in range(4):
580:         table[(0, i)].set_facecolor('#40466e')
581:         table[(0, i)].set_text_props(weight='bold', color='white')
582: 
583:     # Color cells based on values
584:     if stats['pnl'] > 0:
585:         table[(3, 1)].set_facecolor('#90EE90')
586:     else:
587:         table[(3, 1)].set_facecolor('#FFB6C1')
588: 
589:     fig.suptitle(
590:         f'DAX ORB Analysis - {"TraderTom Mode" if args.tradertom else "Parameter Sweep"} - {"With Realistic Costs" if args.realistic else "Without Costs"}',
591:         fontsize=16, fontweight='bold')
592: 
593:     plt.savefig('dax_orb_analysis.png', dpi=150, bbox_inches='tight')
594:     plt.show()
595:     print("\n✓ Analysis saved to dax_orb_analysis.png")
596: 
597: 
598: # ============================= Main =================================================
599: def main():
600:     args = parse_args()
601: 
602:     print("=" * 80)
603:     print("DAX ORB REALISTIC BACKTEST - With Transaction Costs")
604:     print("=" * 80)
605: 
606:     # Load data
607:     df = load_data(args)
608: 
609:     # Split into train/test if not in TraderTom mode
610:     if not args.tradertom and args.train_pct < 1.0:
611:         split_idx = int(len(df) * args.train_pct)
612:         df_train = df.iloc[:split_idx].copy()
613:         df_test = df.iloc[split_idx:].copy()
614:         print(
615:             f"\n✓ Data split: {len(df_train):,} train bars, {len(df_test):,} test bars")
616:     else:
617:         df_train = df.copy()
618:         df_test = None
619: 
620:     # Run backtest
621:     if args.tradertom:
622:         # TraderTom fixed parameters
623:         print(
624:             f"\n🎯 TRADERTOM MODE: SL=9, TP=6, Opening window={args.opening_window} min")
625:         print(f"   Realistic costs: {'ON' if args.realistic else 'OFF'}")
626: 
627:         # Run with costs
628:         trades_with_costs, stats_with_costs = run_realistic_backtest(df_train, args.tz,
629:             args.pre, args.sess, sl=9.0, tp=6.0, use_costs=True, tradertom_mode=True,
630:             opening_window=args.opening_window, verbose=args.verbose)
631: 
632:         # Run without costs for comparison
633:         trades_no_costs, stats_no_costs = run_realistic_backtest(df_train, args.tz,
634:             args.pre, args.sess, sl=9.0, tp=6.0, use_costs=False, tradertom_mode=True,
635:             opening_window=args.opening_window, verbose=False)
636: 
637:         # Display results
638:         print("\n" + "=" * 60)
639:         print("RESULTS COMPARISON:")
640:         print("=" * 60)
641: 
642:         print("\n📊 WITHOUT COSTS (Unrealistic):")
643:         print(f"   Trades: {stats_no_costs['trades']}")
644:         print(
645:             f"   Win Rate: {stats_no_costs['winrate']:.1f}% (Need 60.0% for break-even)")
646:         print(f"   Net PnL: {stats_no_costs['pnl']:.1f} points")
647:         print(f"   Profit Factor: {stats_no_costs['pf']:.2f}")
648: 
649:         print("\n💰 WITH REALISTIC COSTS:")
650:         print(f"   Trades: {stats_with_costs['trades']}")
651:         print(f"   Win Rate: {stats_with_costs['winrate']:.1f}% (Need ~70% with costs)")
652:         print(f"   Net PnL: {stats_with_costs['pnl']:.1f} points")
653:         print(f"   Total Costs Paid: {stats_with_costs['total_costs']:.1f} points")
654:         print(f"   Profit Factor: {stats_with_costs['pf']:.2f}")
655:         print(f"   Max Drawdown: {stats_with_costs['max_dd']:.1f} points")
656: 
657:         impact = stats_no_costs['pnl'] - stats_with_costs['pnl']
658:         print(
659:             f"\n⚠️  COST IMPACT: -{impact:.1f} points ({100 * impact / max(abs(stats_no_costs['pnl']), 1):.1f}% of gross PnL)")
660: 
661:         # Critical assessment
662:         print("\n" + "=" * 60)
663:         print("🔍 CRITICAL ASSESSMENT:")
664:         print("=" * 60)
665: 
666:         if stats_with_costs['winrate'] < 60:
667:             print("❌ Win rate below break-even threshold (60% without costs)")
668:         if stats_with_costs['winrate'] < 70:
669:             print("❌ Win rate below realistic break-even (~70% with costs)")
670:         if stats_with_costs['pnl'] < 0:
671:             print("❌ Strategy is UNPROFITABLE with realistic costs")
672:         if stats_with_costs['trades'] < 100:
673:             print("⚠️  Sample size too small for statistical significance")
674:         if abs(stats_with_costs['max_dd']) > 50:
675:             print("⚠️  Large drawdown risk")
676: 
677:         # Save trades if requested
678:         if args.save_trades:
679:             trades_with_costs.to_csv('tradertom_trades_realistic.csv', index=False)
680:             print(f"\n✓ Trades saved to tradertom_trades_realistic.csv")
681: 
682:         # Create plots
683:         if args.plot:
684:             create_comprehensive_plots(df_train, trades_with_costs, stats_with_costs,
685:                                        args)
686: 
687:     else:
688:         # Parameter sweep mode
689:         print(f"\n🔍 PARAMETER SWEEP MODE")
690:         print(f"   SL range: {args.sl_min}-{args.sl_max} (step {args.sl_step})")
691:         print(f"   TP range: {args.tp_min}-{args.tp_max} (step {args.tp_step})")
692:         print(f"   Realistic costs: {'ON' if args.realistic else 'OFF'}")
693: 
694:         # Create parameter combinations
695:         sl_values = np.arange(args.sl_min, args.sl_max + args.sl_step, args.sl_step)
696:         tp_values = np.arange(args.tp_min, args.tp_max + args.tp_step, args.tp_step)
697: 
698:         results = []
699:         best_score = -float('inf')
700:         best_params = None
701:         best_trades = None
702: 
703:         total_combos = len(sl_values) * len(tp_values)
704:         completed = 0
705: 
706:         print(f"\n   Testing {total_combos} combinations...")
707: 
708:         # Test each combination
709:         with mp.Pool() as pool:
710:             tasks = [
711:                 (df_train, args.tz, args.pre, args.sess, sl, tp, args.realistic, False,
712:                  args.opening_window, False) for sl in sl_values for tp in tp_values]
713: 
714:             async_results = [pool.apply_async(run_realistic_backtest, task) for task in
715:                 tasks]
716: 
717:             for i, (sl, tp) in enumerate(
718:                     [(sl, tp) for sl in sl_values for tp in tp_values]):
719:                 trades_df, stats = async_results[i].get()
720:                 completed += 1
721: 
722:                 if completed % 10 == 0 or completed == total_combos:
723:                     print(
724:                         f"\r   Progress: {completed}/{total_combos} ({100 * completed / total_combos:.1f}%)",
725:                         end='')
726: 
727:                 if stats['trades'] >= args.min_trades:
728:                     score = stats['pnl'] / max(abs(stats['max_dd']),
729:                                                1)  # Risk-adjusted score
730: 
731:                     results.append(
732:                         {'sl': sl, 'tp': tp, 'rr': tp / sl, 'trades': stats['trades'],
733:                             'winrate': stats['winrate'], 'pnl': stats['pnl'],
734:                             'pf': stats['pf'], 'max_dd': stats['max_dd'],
735:                             'sharpe': stats['sharpe'], 'score': score})
736: 
737:                     if score > best_score:
738:                         best_score = score
739:                         best_params = (sl, tp)
740:                         best_trades = trades_df.copy()
741: 
742:         print()  # New line after progress
743: 
744:         if results:
745:             # Sort and display results
746:             results_df = pd.DataFrame(results).sort_values('score', ascending=False)
747: 
748:             print("\n" + "=" * 60)
749:             print("TOP 10 PARAMETER COMBINATIONS:")
750:             print("=" * 60)
751:             print(results_df.head(10).to_string(index=False))
752: 
753:             # Test on out-of-sample data if available
754:             if df_test is not None and best_params is not None:
755:                 print("\n" + "=" * 60)
756:                 print("OUT-OF-SAMPLE TEST:")
757:                 print("=" * 60)
758: 
759:                 test_trades, test_stats = run_realistic_backtest(df_test, args.tz,
760:                     args.pre, args.sess, sl=best_params[0], tp=best_params[1],
761:                     use_costs=args.realistic, tradertom_mode=False,
762:                     opening_window=args.opening_window)
763: 
764:                 print(f"Best params: SL={best_params[0]:.1f}, TP={best_params[1]:.1f}")
765:                 print(f"In-sample PnL: {results_df.iloc[0]['pnl']:.1f}")
766:                 print(f"Out-of-sample PnL: {test_stats['pnl']:.1f}")
767:                 print(f"Out-of-sample Win Rate: {test_stats['winrate']:.1f}%")
768: 
769:                 if test_stats['pnl'] < 0:
770:                     print("❌ Strategy FAILED on out-of-sample data!")
771: 
772:             # Save results
773:             results_df.to_csv('parameter_sweep_results.csv', index=False)
774:             print(f"\n✓ Results saved to parameter_sweep_results.csv")
775: 
776:             # Create plots for best combination
777:             if args.plot and best_trades is not None:
778:                 create_comprehensive_plots(df_train, best_trades,
779:                                            results_df.iloc[0].to_dict(), args)
780:         else:
781:             print("\n❌ No parameter combinations met minimum trade requirement")
782: 
783: 
784: if __name__ == "__main__":
785:     main()
````

## File: pyproject.toml
````toml
 1: [build-system]
 2: requires = ["setuptools>=69", "wheel"]
 3: build-backend = "setuptools.build_meta"
 4: 
 5: [project]
 6: name = "sophy4lite"
 7: version = "0.1.0"
 8: description = "Lean trading research framework with ATR breakout backtesting + FTMO guards"
 9: readme = "README.md"
10: requires-python = ">=3.11"
11: license = {text = "MIT"}
12: authors = [{name="Bastiaan de Haan"}]
13: 
14: dependencies = [
15:   "pandas>=2.2",
16:   "numpy>=1.26",
17:   "typer>=0.12",
18:   "rich>=13.7",
19: ]
20: 
21: [project.optional-dependencies]
22: live = ["MetaTrader5>=5.0.45; platform_system == 'Windows'"]
23: dev  = ["pytest>=8.0", "pytest-cov>=4.1", "black>=24.4", "flake8>=7.0"]
24: 
25: [tool.setuptools]
26: packages = ["backtest", "cli", "risk", "strategies", "utils"]
27: 
28: [tool.setuptools.package-data]
29: # Bijvoorbeeld: templates of statische assets
30: "cli" = []
31: 
32: [tool.black]
33: line-length = 100
34: target-version = ["py311"]
35: 
36: [tool.flake8]
37: max-line-length = 100
38: extend-ignore = ["E203"]
````

## File: quick_smoke_opening_breakout.py
````python
 1: # quick_smoke_opening_breakout.py
 2: import pandas as pd
 3: from backtest.breakout_exec import backtest_breakout, BTExecCfg
 4: from strategies.breakout_params import BreakoutParams
 5: 
 6: # Data: 15m, 200 bars vanaf 2024-01-01 00:00
 7: dates = pd.date_range('2024-01-01', periods=200, freq='15min')
 8: df = pd.DataFrame({
 9:     'open': 100.0,
10:     'high': 100.2,
11:     'low' :  99.8,
12:     'close': 100.0,
13: }, index=dates)
14: 
15: # Forceer breakout op dag 2 09:00 (bar ~36)
16: df.loc['2024-01-02 09:00', 'close'] = 101.0   # > prev day high (100.2)
17: df.loc['2024-01-02 09:00', 'high']  = 101.5
18: 
19: cfg = BTExecCfg(equity0=10_000, fee_rate=0.0002)
20: params = BreakoutParams(atr_mult_sl=1.0, atr_mult_tp=2.0)
21: 
22: # Belangrijk: open_window_bars groot genoeg zodat 09:00 in-venster valt
23: eq, trades, metrics = backtest_breakout(df, "TEST", params, cfg,
24:                                         open_window_bars=40,  # <--
25:                                         confirm="close")
26: 
27: print(f"Trades: {metrics['n_trades']}")
28: print(f"Final equity: {metrics['final_equity']:.2f}")
29: assert metrics['n_trades'] >= 1, "Er is geen opening-breakout trade geplaatst"
````

## File: README.md
````markdown
 1: # Sophy4Lite
 2: 
 3: Lean trading-research framework met:
 4: - **ATR-gebaseerde breakout** (sessie-range → ‘close-confirm’ of ‘pending-stop’)
 5: - **Realistische backtest**: intra-bar SL/TP op high/low, slippage & fees, %-risico position sizing
 6: - **FTMO-guards**: daily loss / total loss **voor** entry (pre-trade worst-case check)
 7: - Eenvoudige **CLI**: `python -m cli.main breakout ...`
 8: 
 9: > Doel: **snel valideren of er een edge is** (falsifieerbaar, geen hype), zonder over-engineering.
10: 
11: ---
12: 
13: ## Installatie
14: 
15: ```bash
16: # Windows / Python 3.11
17: python -m venv .venv
18: .venv\Scripts\activate
19: pip install -r requirements.txt
````

## File: requirements.txt
````
 1: # Core
 2: pandas>=2.2
 3: numpy>=1.26
 4: 
 5: # CLI & UX
 6: typer>=0.12
 7: rich>=13.7
 8: 
 9: # (Optioneel) Live met MT5 in Windows-omgevingen
10: MetaTrader5>=5.0.45 ; platform_system == "Windows"
11: 
12: # (Optioneel) Onderzoek/optimalisatie (indien je later een grid/optimizer aankoppelt)
13: # scikit-learn>=1.4
14: # hyperopt>=0.2.7
15: 
16: # (Optioneel) Tests & tooling
17: # pytest>=8.0
18: # pytest-cov>=4.1
19: # black>=24.4
20: # flake8>=7.0
21: # vectorbt==0.24.7
````

## File: validate_framework.py
````python
  1: #!/usr/bin/env python3
  2: """
  3: Sophy4Lite Repository Validator
  4: Controleert het hele framework op fouten en biedt fixes.
  5: """
  6: 
  7: import ast
  8: import sys
  9: import importlib.util
 10: from pathlib import Path
 11: from typing import List, Dict, Tuple
 12: import subprocess
 13: 
 14: 
 15: class FrameworkValidator:
 16:     def __init__(self, root_path: Path = Path.cwd()):
 17:         self.root = root_path
 18:         self.errors: List[Dict] = []
 19:         self.warnings: List[Dict] = []
 20: 
 21:     def validate_all(self) -> bool:
 22:         """Hoofdfunctie die alle checks uitvoert."""
 23:         print("🔍 Starting Sophy4Lite Framework Validation...\n")
 24: 
 25:         # 1. Python syntax check
 26:         self._check_syntax()
 27: 
 28:         # 2. Import validation
 29:         self._check_imports()
 30: 
 31:         # 3. Missing files check
 32:         self._check_required_files()
 33: 
 34:         # 4. Type hints check (optioneel met mypy)
 35:         self._check_types()
 36: 
 37:         # 5. Framework-specifieke checks
 38:         self._check_framework_specific()
 39: 
 40:         # Report
 41:         self._print_report()
 42: 
 43:         return len(self.errors) == 0
 44: 
 45:     def _check_syntax(self):
 46:         """Controleert Python syntax in alle .py files."""
 47:         print("1️⃣ Checking Python syntax...")
 48: 
 49:         for py_file in self.root.rglob("*.py"):
 50:             if ".venv" in str(py_file) or "__pycache__" in str(py_file):
 51:                 continue
 52: 
 53:             try:
 54:                 with open(py_file, 'r', encoding='utf-8') as f:
 55:                     ast.parse(f.read())
 56:             except SyntaxError as e:
 57:                 self.errors.append(
 58:                     {'file': str(py_file.relative_to(self.root)), 'line': e.lineno,
 59:                         'type': 'SyntaxError', 'msg': str(e.msg)})
 60: 
 61:         print(f"   ✓ Checked {len(list(self.root.rglob('*.py')))} files\n")
 62: 
 63:     def _check_imports(self):
 64:         """Valideert alle imports."""
 65:         print("2️⃣ Checking imports...")
 66: 
 67:         # KRITIEKE FIX voor live_trading.py
 68:         live_trading = self.root / "live" / "live_trading.py"
 69:         if live_trading.exists():
 70:             with open(live_trading, 'r') as f:
 71:                 content = f.read()
 72:                 if "from risk.ftmo import" in content:
 73:                     self.errors.append({'file': 'live/live_trading.py', 'line': 6,
 74:                         'type': 'ImportError',
 75:                         'msg': "Module 'risk.ftmo' does not exist! Use 'risk.ftmo_guard' instead",
 76:                         'fix': "Remove line 6 or create risk/ftmo.py with fixed_percent_sizing function"})
 77: 
 78:         # Check alle andere imports
 79:         for py_file in self.root.rglob("*.py"):
 80:             if ".venv" in str(py_file) or "__pycache__" in str(py_file):
 81:                 continue
 82: 
 83:             self._validate_file_imports(py_file)
 84: 
 85:         print(f"   ✓ Import validation complete\n")
 86: 
 87:     def _validate_file_imports(self, file_path: Path):
 88:         """Valideert imports in een specifiek bestand."""
 89:         try:
 90:             with open(file_path, 'r', encoding='utf-8') as f:
 91:                 tree = ast.parse(f.read())
 92: 
 93:             for node in ast.walk(tree):
 94:                 if isinstance(node, ast.Import):
 95:                     for alias in node.names:
 96:                         self._check_module_exists(alias.name, file_path, node.lineno)
 97: 
 98:                 elif isinstance(node, ast.ImportFrom):
 99:                     if node.module:
100:                         # Check relatieve imports binnen project
101:                         if not node.level and not node.module.startswith('.'):
102:                             if node.module.startswith(
103:                                     ('backtest', 'cli', 'risk', 'strategies', 'utils',
104:                                      'live')):
105:                                 module_path = self.root / node.module.replace('.', '/')
106:                                 if not module_path.exists() and not (
107:                                 module_path.with_suffix('.py')).exists():
108:                                     self.errors.append(
109:                                         {'file': str(file_path.relative_to(self.root)),
110:                                             'line': node.lineno, 'type': 'ImportError',
111:                                             'msg': f"Local module '{node.module}' not found"})
112:         except Exception as e:
113:             self.warnings.append({'file': str(file_path.relative_to(self.root)),
114:                 'msg': f"Could not parse imports: {e}"})
115: 
116:     def _check_module_exists(self, module_name: str, file_path: Path, line: int):
117:         """Controleert of een module bestaat."""
118:         try:
119:             if module_name in ['backtest', 'cli', 'risk', 'strategies', 'utils',
120:                                'live']:
121:                 # Lokale modules
122:                 module_path = self.root / module_name
123:                 if not module_path.exists():
124:                     self.errors.append(
125:                         {'file': str(file_path.relative_to(self.root)), 'line': line,
126:                             'type': 'ImportError',
127:                             'msg': f"Local module '{module_name}' directory missing"})
128:             else:
129:                 # External modules
130:                 spec = importlib.util.find_spec(module_name.split('.')[0])
131:                 if spec is None and module_name not in ['MetaTrader5', 'vectorbt']:
132:                     self.warnings.append(
133:                         {'file': str(file_path.relative_to(self.root)), 'line': line,
134:                             'msg': f"External module '{module_name}' not installed"})
135:         except (ImportError, ModuleNotFoundError, ValueError):
136:             pass  # Optionele dependencies
137: 
138:     def _check_required_files(self):
139:         """Controleert of alle vereiste bestanden aanwezig zijn."""
140:         print("3️⃣ Checking required files...")
141: 
142:         required = ['backtest/__init__.py', 'cli/__init__.py', 'risk/__init__.py',
143:             'strategies/__init__.py', 'utils/__init__.py', 'config.py',
144:             'requirements.txt']
145: 
146:         for req_file in required:
147:             if not (self.root / req_file).exists():
148:                 self.errors.append({'file': req_file, 'type': 'MissingFile',
149:                     'msg': f"Required file '{req_file}' is missing"})
150: 
151:         # Check of output directory bestaat
152:         output_dir = self.root / 'output'
153:         if not output_dir.exists():
154:             self.warnings.append(
155:                 {'msg': "Output directory missing - will cause crash in config.py",
156:                     'fix': "Run: mkdir output"})
157: 
158:         print(f"   ✓ File structure validated\n")
159: 
160:     def _check_types(self):
161:         """Optioneel: run mypy voor type checking."""
162:         print("4️⃣ Checking types (if mypy available)...")
163: 
164:         try:
165:             result = subprocess.run(
166:                 ['mypy', '--ignore-missing-imports', '--no-error-summary',
167:                  str(self.root)], capture_output=True, text=True, timeout=10)
168:             if result.returncode != 0 and result.stdout:
169:                 # Parse alleen de belangrijkste type errors
170:                 for line in result.stdout.split('\n')[:10]:  # Max 10 errors
171:                     if 'error:' in line:
172:                         self.warnings.append({'msg': line.strip()})
173:         except (subprocess.TimeoutExpired, FileNotFoundError):
174:             print("   ⚠️ mypy not available, skipping type checks")
175: 
176:         print()
177: 
178:     def _check_framework_specific(self):
179:         """Sophy4Lite specifieke validaties."""
180:         print("5️⃣ Checking Sophy4Lite specific requirements...")
181: 
182:         # Check timezone handling
183:         breakout_exec = self.root / "backtest" / "breakout_exec.py"
184:         if breakout_exec.exists():
185:             with open(breakout_exec, 'r') as f:
186:                 content = f.read()
187:                 if 'tz_localize(tz)' in content and 'df.index.tz is None' in content:
188:                     self.warnings.append(
189:                         {'file': 'backtest/breakout_exec.py', 'line': 24,
190:                             'msg': "Dangerous timezone handling - blindly localizes to UTC",
191:                             'fix': "Check original timezone before localizing"})
192: 
193:         # Check ATR calculation for look-ahead bias
194:         breakout_signals = self.root / "strategies" / "breakout_signals.py"
195:         if breakout_signals.exists():
196:             with open(breakout_signals, 'r') as f:
197:                 content = f.read()
198:                 if 'df.loc[:w.index.max()]' in content:
199:                     self.warnings.append(
200:                         {'file': 'strategies/breakout_signals.py', 'line': 91,
201:                             'msg': "Potential look-ahead bias in ATR calculation",
202:                             'fix': "Verify that w.index.max() doesn't include future data"})
203: 
204:         print(f"   ✓ Framework validation complete\n")
205: 
206:     def _print_report(self):
207:         """Print het eindrapport."""
208:         print("=" * 60)
209:         print("VALIDATION REPORT")
210:         print("=" * 60)
211: 
212:         if self.errors:
213:             print(f"\n❌ CRITICAL ERRORS ({len(self.errors)}):\n")
214:             for i, err in enumerate(self.errors, 1):
215:                 print(f"  {i}. [{err['type']}] {err['file']}")
216:                 if 'line' in err:
217:                     print(f"     Line {err['line']}: {err['msg']}")
218:                 else:
219:                     print(f"     {err['msg']}")
220:                 if 'fix' in err:
221:                     print(f"     💡 FIX: {err['fix']}")
222:                 print()
223: 
224:         if self.warnings:
225:             print(f"\n⚠️ WARNINGS ({len(self.warnings)}):\n")
226:             for i, warn in enumerate(self.warnings, 1):
227:                 if 'file' in warn:
228:                     print(f"  {i}. {warn['file']}")
229:                     if 'line' in warn:
230:                         print(f"     Line {warn['line']}")
231:                 print(f"     {warn['msg']}")
232:                 if 'fix' in warn:
233:                     print(f"     💡 FIX: {warn['fix']}")
234:                 print()
235: 
236:         if not self.errors and not self.warnings:
237:             print("\n✅ All checks passed! Framework is ready to run.\n")
238:         elif self.errors:
239:             print("\n🛑 Fix critical errors before running the framework!\n")
240:         else:
241:             print("\n✅ No critical errors, but review warnings.\n")
242: 
243:         print("=" * 60)
244: 
245:         # Quick fixes sectie
246:         if self.errors:
247:             print("\n🔧 QUICK FIXES TO APPLY:")
248:             print("-" * 40)
249:             print("1. Fix import in live/live_trading.py:")
250:             print("   DELETE line 6: from risk.ftmo import fixed_percent_sizing")
251:             print("\n2. Create output directory:")
252:             print("   RUN: mkdir output")
253:             print("\n3. Install missing dependencies:")
254:             print("   RUN: pip install -r requirements.txt")
255:             print("-" * 40)
256: 
257: 
258: def auto_fix_critical_issues():
259:     """Automatisch fixen van de meest kritieke issues."""
260:     print("\n🔧 Attempting auto-fixes...\n")
261: 
262:     # Fix 1: Maak output directory
263:     output_dir = Path.cwd() / 'output'
264:     if not output_dir.exists():
265:         output_dir.mkdir(exist_ok=True)
266:         print("✅ Created output/ directory")
267: 
268:     # Fix 2: Comment out broken import
269:     live_trading = Path.cwd() / "live" / "live_trading.py"
270:     if live_trading.exists():
271:         with open(live_trading, 'r') as f:
272:             lines = f.readlines()
273: 
274:         for i, line in enumerate(lines):
275:             if "from risk.ftmo import fixed_percent_sizing" in line:
276:                 lines[
277:                     i] = "# " + line + "# TODO: Create risk/ftmo.py or remove this import\n"
278:                 with open(live_trading, 'w') as f:
279:                     f.writelines(lines)
280:                 print("✅ Commented out broken import in live/live_trading.py")
281:                 break
282: 
283:     # Fix 3: Create missing __init__.py files
284:     for module in ['backtest', 'cli', 'risk', 'strategies', 'utils', 'live', 'test',
285:                    'scripts']:
286:         init_file = Path.cwd() / module / '__init__.py'
287:         if (Path.cwd() / module).exists() and not init_file.exists():
288:             init_file.touch()
289:             print(f"✅ Created {module}/__init__.py")
290: 
291:     print("\n✨ Auto-fixes applied!\n")
292: 
293: 
294: if __name__ == "__main__":
295:     validator = FrameworkValidator()
296: 
297:     # Vraag om auto-fix
298:     print("🚀 Sophy4Lite Framework Validator\n")
299:     response = input(
300:         "Apply automatic fixes for critical issues? (y/n): ").strip().lower()
301: 
302:     if response == 'y':
303:         auto_fix_critical_issues()
304: 
305:     # Run validation
306:     is_valid = validator.validate_all()
307: 
308:     sys.exit(0 if is_valid else 1)
````
