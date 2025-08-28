from __future__ import annotations
from datetime import date
from dataclasses import dataclass

@dataclass
class FtmoRules:
    max_daily_loss_pct: float = 0.05  # 5%
    max_total_loss_pct: float = 0.10  # 10%
    stop_after_losses: int = 2  # consecutive losing trades per day

class FtmoGuard:
    def __init__(self, initial_equity: float, rules: FtmoRules):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.daily_loss = 0.0
        self.loss_streak = 0
        self.blocked = False
        self.current_day = None
        self.rules = rules

    def new_day(self, day: date, equity: float):
        if self.current_day != day:
            self.current_day = day
            self.daily_start_equity = equity
            self.daily_loss = 0.0
            self.loss_streak = 0
            self.blocked = False

    def update_equity(self, equity: float):
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.daily_loss = self.daily_start_equity - equity

    def pretrade_ok(self, worst_loss: float) -> bool:
        if self.blocked:
            return False
        projected_equity = self.current_equity - worst_loss
        if (self.daily_start_equity - projected_equity) / self.daily_start_equity > self.rules.max_daily_loss_pct:
            return False
        if (self.initial_equity - projected_equity) / self.initial_equity > self.rules.max_total_loss_pct:
            return False
        return True

    def allowed_now(self) -> bool:
        return not self.blocked