import LabelBench.strategy.strategy_impl
from LabelBench.skeleton.active_learning_skeleton import strategies


def get_strategy(name, strategy_config, trainer):
    return strategies[name](strategy_config, trainer)
