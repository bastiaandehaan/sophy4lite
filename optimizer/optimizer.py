from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional
import itertools
import pandas as pd

from backtest.runner import run_backtest
from utils.metrics import summarize_equity_metrics


@dataclass
class Optimizer:
    strategy_name: str
    param_grid: Dict[str, List[Any]]
    symbol: str
    timeframe: str
    start: str
    end: str
    csv_path: Optional[Path] = None

    def _grid(self):
        keys, values = zip(*self.param_grid.items()) if self.param_grid else ([], [])
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def run(self) -> pd.DataFrame:
        rows = []
        for params in self._grid():
            df_eq, trades = run_backtest(
                strategy_name=self.strategy_name,
                params=params,
                symbol=self.symbol,
                timeframe=self.timeframe,
                start=self.start,
                end=self.end,
                csv_path=self.csv_path,
            )
            metrics = summarize_equity_metrics(df_eq, trades)
            rows.append({"params": params, **metrics})
        return pd.DataFrame(rows)

    def select_best(self, min_trades: int = 50) -> dict:
        df = self.run() if not hasattr(self, "_last_df") else self._last_df
        if not hasattr(self, "_last_df"):
            self._last_df = df
        df = df[df["n_trades"] >= min_trades]
        if df.empty:
            raise ValueError("No configs meet min_trades criterion")
        df = df.sort_values(
            by=["sharpe", "max_dd", "dd_duration", "total_return"],
            ascending=[False, True, True, False],
        )
        return df.iloc[0].to_dict()

    def export_best_config(self, profiles_dir: Path):
        best = self.select_best()
        profiles_dir.mkdir(parents=True, exist_ok=True)
        with open(profiles_dir / "live.yaml", "w", encoding="utf-8") as f:
            import yaml
            yaml.safe_dump(
                {
                    "version": 1,
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "strategy": {"name": self.strategy_name, "params": best["params"]},
                    "risk": {
                        "per_trade_risk_pct": 0.3,
                        "max_daily_loss_pct": 4.5,
                        "max_total_loss_pct": 9.0,
                    },
                    "execution": {"broker": "mt5", "lot_min": 0.1, "lot_step": 0.1, "slippage": 0.0},
                },
                f,
                sort_keys=False,
            )