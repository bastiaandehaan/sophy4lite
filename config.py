from dataclasses import dataclass

# --- Core settings ---
INITIAL_CAPITAL: float = 20000.0
FEES: float = 0.0002  # 2 bps
DEFAULT_TIMEFRAME: str = "H1"
OUTPUT_DIR: str = "output"

# --- FTMO guardrails ---
MAX_DAILY_LOSS: float = 0.05   # 5%
MAX_TOTAL_LOSS: float = 0.10   # 10%
RISK_PER_TRADE: float = 0.01   # 1%
STOP_AFTER_LOSSES: int = 2     # consecutive losing trades per day

# --- Logging ---
LOG_LEVEL: str = "INFO"

import logging, os, sys
from utils.metrics import setup_logging
logger = setup_logging(level=LOG_LEVEL, logfile=os.path.join(OUTPUT_DIR, "sophy4lite.log"))
