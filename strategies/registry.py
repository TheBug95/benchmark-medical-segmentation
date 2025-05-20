from typing import Dict, Type

STRATEGY_REGISTRY: Dict[str,"BaseStrategy"] = {}

def register(name:str):
    def decorator(cls):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator
