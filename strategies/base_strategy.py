from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return entries(bool series), sl_stop(%), tp_stop(%) aligned to df.index"""
        ...
