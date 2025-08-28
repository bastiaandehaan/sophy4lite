from dataclasses import dataclass
import logging
import os

# --- Core settings ---
INITIAL_CAPITAL: float = 20000.0
FEES: float = 0.0002  # 2 bps
DEFAULT_TIMEFRAME: str = "H1"
OUTPUT_DIR: str = "output"

# --- FTMO guardrails (sync met FtmoRules) ---
MAX_DAILY_LOSS: float = 0.05   # 5%
MAX_TOTAL_LOSS: float = 0.10   # 10%
RISK_PER_TRADE: float = 0.01   # 1%
STOP_AFTER_LOSSES: int = 2     # consecutive losing trades per day

# --- Logging ---
LOG_LEVEL: str = "INFO"

# Maak logger en configureer handlers
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

# File handler voor logbestand
file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "sophy4lite.log"))
file_handler.setLevel(LOG_LEVEL)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Stream handler voor console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(file_formatter)

# Voeg handlers toe aan logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)