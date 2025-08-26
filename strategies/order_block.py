import pandas as pd


def order_block_signals(df: pd.DataFrame, swing_w: int = 3) -> pd.DataFrame:
    """
    Minimale placeholder:
      LONG = close breekt boven vorige swing-high (BOS).
    Geen echte OB-detectie, maar genoeg om de pipeline te testen.
    """
    hi = df["high"].rolling(swing_w, center=True).max()
    swing_high = (df["high"] == hi) & df["high"].notna()
    prev_swing_high = df["high"].where(swing_high).ffill().shift(1)
    long = df["close"] > prev_swing_high
    return pd.DataFrame({"long": long.fillna(False)}, index=df.index)
