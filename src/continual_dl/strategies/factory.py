"""Strategy factory used by the CLI runner."""

from __future__ import annotations

from .ewc import EWCStrategy
from .joint import JointStrategy
from .lwf import LwFStrategy
from .naive import NaiveStrategy
from .ncm import NCMStrategy
from .replay import ReplayStrategy


STRATEGIES = {
    "naive": NaiveStrategy,
    "joint": JointStrategy,
    "ncm": NCMStrategy,
    "replay": ReplayStrategy,
    "lwf": LwFStrategy,
    "ewc": EWCStrategy,
}


def build_strategy(model, device, config: dict):
    strategy_config = config["strategy"]
    name = str(strategy_config["name"]).lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choices: {sorted(STRATEGIES)}")
    training_config = dict(config["training"])
    training_config.setdefault("seed", int(config.get("seed", 42)))
    return STRATEGIES[name](
        model=model,
        device=device,
        training_config=training_config,
        strategy_config=strategy_config,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
