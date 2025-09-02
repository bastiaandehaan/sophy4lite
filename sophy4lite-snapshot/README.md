# Sophy4Lite

Lean trading-research framework met:
- **ATR-gebaseerde breakout** (sessie-range → ‘close-confirm’ of ‘pending-stop’)
- **Realistische backtest**: intra-bar SL/TP op high/low, slippage & fees, %-risico position sizing
- **FTMO-guards**: daily loss / total loss **voor** entry (pre-trade worst-case check)
- Eenvoudige **CLI**: `python -m cli.main breakout ...`

> Doel: **snel valideren of er een edge is** (falsifieerbaar, geen hype), zonder over-engineering.

---

## Installatie

```bash
# Windows / Python 3.11
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
