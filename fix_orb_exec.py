import re
from pathlib import Path

p = Path(r"backtest/orb_exec.py")
s = p.read_text(encoding="utf-8", errors="replace")

# 1) elke niet-ASCII na 'e' in wetenschappelijke notatie (bv. 1eâ€‘6) -> e-<digit>
s = re.sub(r"(?<=\d)e[^\x00-\x7F](\d)", r"e-\1", s)

# 2) specifieke mis-decoded variant
s = s.replace("eâ€‘", "e-")

# 3) alle unicode hyphens (en minus) -> ASCII '-'
s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\uFE63\uFF0D]", "-", s)

p.write_text(s, encoding="utf-8", newline="\n")
print("sanitized", p)
