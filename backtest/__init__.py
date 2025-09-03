# backtest/__init__.py
from .mtf_exec_fast import backtest_mtf_confluence_fast, MTFExecCfg
# Backwards-compatibele alias verwacht door tests/tools:
backtest_mtf_confluence = backtest_mtf_confluence_fast

__all__ = ["backtest_mtf_confluence", "MTFExecCfg"]
