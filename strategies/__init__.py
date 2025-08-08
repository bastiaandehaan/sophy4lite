from .bollong import BollongStrategy
from .simple_order_block import SimpleOrderBlockStrategy
STRATEGIES = {
    "bollong": BollongStrategy,
    "simple_ob": SimpleOrderBlockStrategy
}
def get_strategy(name: str, **kwargs):
    key = name.lower()
    if key not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name}")
    return STRATEGIES[key](**kwargs)
